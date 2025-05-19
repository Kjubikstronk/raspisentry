import os
import time
import logging
import requests
import numpy as np
import cv2
import smtplib
import sqlite3
from picamera2 import Picamera2
from pantilthat import pan, tilt
from threading import Thread, Lock, Event
from concurrent.futures import ThreadPoolExecutor
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from typing import Any, List, Tuple
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from flask import Flask, Response, jsonify
from email_notify import EmailNotifier

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    CAMERA_WIDTH        = 320
    CAMERA_HEIGHT       = 200
    PAN_SENSITIVITY     = 15
    TILT_SENSITIVITY    = 15
    PAN_LIMIT           = 70
    TILT_LIMIT          = 90
    MOVE_THRESHOLD      = 0.5
    MAX_FACE_LOSS_FRAMES= 30
    SENTRY_TILT_OFFSET  = -50
    SENTRY_SWEEP_STEP   = 3
    SENTRY_WAIT_TIME    = 0.05
    EMAIL_INTERVAL      = 60
    DEVICE_ID           = "RaspiSentry-001"
    DB_PATH             = "face_logs.db"
    COMMAND_PORT        = 5001  # Port for receiving commands

# --- Command Server ---
class CommandHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/video':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    frame = tracker.stream.read()
                    if frame is not None:
                        ret, buffer = cv2.imencode('.jpg', frame)
                        if ret:
                            self.wfile.write(b'--frame\r\n')
                            self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                            self.wfile.write(buffer.tobytes())
                            self.wfile.write(b'\r\n')
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Video stream error: {e}")
        else:
            self.send_error(404)

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data.decode('utf-8'))
            command = data.get('command')
            value = data.get('value')
            
            if command == 'pan':
                tracker.manual_control = True
                tracker.sentry_evt.clear()
                pan(int(value))
            elif command == 'tilt':
                tracker.manual_control = True
                tracker.sentry_evt.clear()
                tilt(int(value))
            elif command == 'sentry_mode':
                logger.info(f"Received sentry_mode command: {value}")
                if value:
                    tracker.manual_control = False
                    if not tracker.sentry_evt.is_set():
                        tracker.sentry_evt.set()
                        if not tracker.sentry_thread or not tracker.sentry_thread.is_alive():
                            tracker.sentry_thread = Thread(target=tracker._sentry_mode, daemon=True)
                            tracker.sentry_thread.start()
                else:
                    tracker.manual_control = False
                    tracker.sentry_evt.clear()
                    logger.info("Sentry mode deactivated (event cleared)")
                    if tracker.sentry_thread and tracker.sentry_thread.is_alive():
                        tracker.sentry_thread.join(timeout=1)
                        logger.info("Sentry thread joined after standby")
            elif command == 'reset':
                tracker.manual_control = False
                tracker.sentry_evt.clear()
                pan(0)
                tilt(Config.SENTRY_TILT_OFFSET)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'success'}).encode())
        except Exception as e:
            logger.error(f"Command error: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'error', 'message': str(e)}).encode())

def start_command_server():
    server = HTTPServer(('0.0.0.0', Config.COMMAND_PORT), CommandHandler)
    Thread(target=server.serve_forever, daemon=True).start()
    logger.info(f"Command server started on port {Config.COMMAND_PORT}")

# --- Geolocation ---
def get_location() -> str:
    try:
        resp = requests.get("http://ip-api.com/json/", timeout=5).json()
        city, region, country = resp.get("city"), resp.get("regionName"), resp.get("country")
        if city and region and country:
            return f"{city}, {region}, {country}"
        lat, lon = resp.get("lat"), resp.get("lon")
        if lat is not None and lon is not None:
            return f"Lat: {lat}, Lon: {lon}"
    except Exception as e:
        logger.error("Geolocation error: %s", e)
    return "Location unavailable"

# --- Database Logger ---
class FaceDatabase:
    def __init__(self, path: str = Config.DB_PATH) -> None:
        self.path = path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp    TEXT,
                    confidence   REAL,
                    face_area    INTEGER,
                    location     TEXT,
                    ip           TEXT,
                    device_id    TEXT,
                    face_image   BLOB
                )
            ''')
            conn.commit()

    def log_detection(self, confidence: float, area: int, location: str, ip: str, device_id: str, image_bytes: bytes) -> None:
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                'INSERT INTO detections (timestamp,confidence,face_area,location,ip,device_id,face_image) VALUES (?,?,?,?,?,?,?)',
                (ts, confidence, area, location, ip, device_id, image_bytes)
            )
            conn.commit()

# --- Face Detection ---
class FaceDetector:
    def __init__(self) -> None:
        model, cfg = 'res10_300x300_ssd_iter_140000.caffemodel', 'deploy.prototxt'
        if not (os.path.exists(model) and os.path.exists(cfg)):
            raise FileNotFoundError('Model files missing')
        self.net = cv2.dnn.readNetFromCaffe(cfg, model)

    def detect(self, frame: Any, threshold: float = 0.5) -> List[Tuple[int,int,int,int,float,int]]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (160,120)), 1.0, (300,300), (104,177,123))
        self.net.setInput(blob)
        dets = self.net.forward()
        results = []
        for i in range(dets.shape[2]):
            conf = float(dets[0,0,i,2])
            if conf > threshold:
                x1,y1,x2,y2 = (dets[0,0,i,3:7] * np.array([w,h,w,h])).astype(int)
                area = (x2-x1)*(y2-y1)
                results.append((x1,y1,x2,y2,conf,area))
        return results

# --- Video Stream ---
class PiVideoStream:
    def __init__(self, resolution: Tuple[int,int] = (Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT)) -> None:
        self.camera = Picamera2()
        cfg = self.camera.create_video_configuration(
            main={"size": resolution, "format": "RGB888"},
            controls={"FrameDurationLimits": (20000,20000)}
        )
        self.camera.configure(cfg)
        self.camera.start()
        self.frame = None
        self.stopped = False
        self.lock = Lock()

    def start(self) -> 'PiVideoStream':
        Thread(target=self._update, daemon=True).start()
        return self

    def _update(self) -> None:
        while not self.stopped:
            f = self.camera.capture_array()
            with self.lock:
                self.frame = f

    def read(self) -> Any:
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self) -> None:
        self.stopped = True
        self.camera.stop()

# --- Face Tracker ---
class FaceTracker:
    def __init__(self) -> None:
        self.stream    = PiVideoStream()
        self.detector  = FaceDetector()
        self.db        = FaceDatabase()
        self.emailer   = EmailNotifier()
        self.last_time = 0.0
        self.face_loss = 0
        self.consec    = 0
        self.sentry_evt= Event()
        self.manual_control = False
        self.pan_x     = 0
        self.pan_y     = Config.SENTRY_TILT_OFFSET
        self.lock      = Lock()
        self.sentry_thread = None

    def _reset_motion(self) -> None:
        self.pan_x = 0
        self.pan_y = Config.SENTRY_TILT_OFFSET
        pan(self.pan_x)
        tilt(self.pan_y)

    def _sentry_mode(self) -> None:
        logger.info("Sentry mode activated.")
        tilt(Config.SENTRY_TILT_OFFSET)
        direction = 1
        pan_angle = -Config.PAN_LIMIT
        while self.sentry_evt.is_set() and not self.manual_control:
            pan_angle += direction * Config.SENTRY_SWEEP_STEP
            if abs(pan_angle) > Config.PAN_LIMIT:
                direction *= -1
                pan_angle += direction * Config.SENTRY_SWEEP_STEP
            pan(pan_angle)
            time.sleep(Config.SENTRY_WAIT_TIME)
        logger.info("Sentry mode finished.")
        self.sentry_evt.clear()

    def track(self) -> None:
        self.stream.start()
        time.sleep(1)
        cv2.namedWindow("Face Tracking", cv2.WINDOW_AUTOSIZE)
        self._reset_motion()
        logger.info("Tracking started...")

        while True:
            frame = self.stream.read()
            if frame is None:
                continue

            # Only process face detection if not in manual control
            if not self.manual_control:
                dets = self.detector.detect(frame)
                if dets:
                    self.consec += 1
                    self.face_loss = 0
                    x1,y1,x2,y2,conf,area = max(dets, key=lambda b: b[5])
                    if self.consec > 3 and time.time() - self.last_time > Config.EMAIL_INTERVAL:
                        roi = frame[y1:y2, x1:x2]
                        self.emailer.send(roi, conf, area, self.db)
                        self.last_time = time.time()
                    if self.sentry_evt.is_set():
                        self.sentry_evt.clear()
                    cx,cy = (x1+x2)//2, (y1+y2)//2
                    dx = ((Config.CAMERA_WIDTH/2)-cx)/Config.PAN_SENSITIVITY
                    dy = -((Config.CAMERA_HEIGHT/2)-cy)/Config.TILT_SENSITIVITY
                    if abs(dx) >= Config.MOVE_THRESHOLD:
                        self.pan_x = max(min(self.pan_x + dx, Config.PAN_LIMIT), -Config.PAN_LIMIT)                
                    if abs(dy) >= Config.MOVE_THRESHOLD:
                        self.pan_y = max(min(self.pan_y + dy, Config.TILT_LIMIT), -Config.TILT_LIMIT)
                    with self.lock:
                        pan(self.pan_x)
                        tilt(self.pan_y)
                else:
                    self.face_loss += 1
                    logger.info(f"No face detected. Count: {self.face_loss}")
                    if self.face_loss > Config.MAX_FACE_LOSS_FRAMES and not self.sentry_evt.is_set() and not self.manual_control:
                        self.sentry_evt.set()
                        Thread(target=self._sentry_mode, daemon=True).start()

            cv2.imshow("Face Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stream.stop()
        cv2.destroyAllWindows()

app = Flask(__name__)

@app.route('/snapshot')
def snapshot():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "Failed to capture image", 500
    # Optionally resize for faster transfer
    # frame = cv2.resize(frame, (320, 240))
    _, jpeg = cv2.imencode('.jpg', frame)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

@app.route('/detections')
def get_detections():
    try:
        with sqlite3.connect(Config.DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute('''
                SELECT id, timestamp, confidence, face_area, location, ip, device_id
                FROM detections
                WHERE face_image IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 50
            ''')
            rows = cur.fetchall()
            return jsonify([dict(row) for row in rows])
    except Exception as e:
        logger.error(f"Error fetching detections: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/detection_image/<int:det_id>')
def get_detection_image(det_id):
    try:
        with sqlite3.connect(Config.DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute('SELECT face_image FROM detections WHERE id = ?', (det_id,))
            result = cur.fetchone()
            if result and result[0]:
                return Response(result[0], mimetype='image/jpeg')
            return "Image not found", 404
    except Exception as e:
        logger.error(f"Error fetching detection image: {e}")
        return str(e), 500

if __name__ == "__main__":
    try:
        # Start the command server before initializing the tracker
        start_command_server()
        # Initialize and start the tracker
        tracker = FaceTracker()
        tracker.track()
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"Program terminated due to error: {e}") 
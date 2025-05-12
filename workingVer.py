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

# --- Logging ---
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

# --- Email Notification ---
class EmailNotifier:
    def __init__(self) -> None:
        self.sender_email    = "&&&"
        self.sender_password = "&&&"
        self.receiver_email  = "&&&"
        self.executor        = ThreadPoolExecutor(max_workers=2)

    def _get_public_ip(self) -> str:
        try:
            return requests.get("https://api.ipify.org", timeout=5).text
        except:
            return "Unavailable"

    def send(self, frame: Any, confidence: float, area: int, db: FaceDatabase) -> None:
        self.executor.submit(self._send_email, frame, confidence, area, db)

    def _send_email(self, frame: Any, confidence: float, area: int, db: FaceDatabase) -> None:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        public_ip = self._get_public_ip()
        location  = get_location()

        ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        image_bytes = jpg.tobytes() if ok else None

        db.log_detection(confidence, area, location, public_ip, Config.DEVICE_ID, image_bytes)

        body = (
            f"Timestamp: {timestamp}\n"
            f"Device ID: {Config.DEVICE_ID}\n"
            f"Public IP: {public_ip}\n"
            f"Location: {location}\n"
            f"Confidence: {confidence:.2f}\n"
            f"Face Area: {area}\n"
        )
        msg = MIMEMultipart()
        msg['Subject'] = 'Face Detection Alert'
        msg['From']    = self.sender_email
        msg['To']      = self.receiver_email
        msg.attach(MIMEText(body, 'plain'))

        if image_bytes:
            img = MIMEImage(image_bytes, _subtype='jpeg')
            img.add_header('Content-Disposition', 'attachment', filename='face.jpg')
            msg.attach(img)

        try:
            with smtplib.SMTP('smtp.gmail.com', 587, timeout=10) as s:
                s.starttls()
                s.login(self.sender_email, self.sender_password)
                s.send_message(msg)
                logger.info("Email sent successfully.")
        except Exception as e:
            logger.error("Email error: %s", e)

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
        self.pan_x     = 0
        self.pan_y     = Config.SENTRY_TILT_OFFSET
        self.lock      = Lock()

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
        while self.sentry_evt.is_set():
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
                if self.face_loss > Config.MAX_FACE_LOSS_FRAMES and not self.sentry_evt.is_set():
                    self.sentry_evt.set()
                    Thread(target=self._sentry_mode, daemon=True).start()

            cv2.imshow("Face Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stream.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    FaceTracker().track()

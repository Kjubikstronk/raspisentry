import os
import time
import math
import socket
import numpy as np
import cv2
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from picamera2 import Picamera2
from pantilthat import pan, tilt
from threading import Thread, Lock, Event
from concurrent.futures import ThreadPoolExecutor
import logging

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

# --- Configuration (KISS, YAGNI) ---
class Config:
    CAMERA_WIDTH = 320
    CAMERA_HEIGHT = 200
    PAN_SENSITIVITY = 15
    TILT_SENSITIVITY = 15
    PAN_LIMIT = 70
    TILT_LIMIT = 90
    MOVE_THRESHOLD = 0.5
    MAX_FACE_LOSS_FRAMES = 30
    SENTRY_TILT_OFFSET = -50
    SENTRY_SWEEP_STEP = 3
    SENTRY_WAIT_TIME = 0.05
    EMAIL_INTERVAL = 60
    DEVICE_ID = "RaspiSentry-001"  # Simple device identifier

# --- GPS Location (SOLID: Single Responsibility) ---
class GPSLocator:
    def __init__(self):
        try:
            import gps  # Requires python-gps package and gpsd running
            self.gps = gps
            self.session = self.gps.gps(mode=self.gps.WATCH_ENABLE)
            self.use_dummy = False
        except ImportError:
            logger.warning("gps module not available; using dummy GPS location.")
            self.use_dummy = True

    def get_location(self):
        if self.use_dummy:
            return "Lat: 0.0, Lon: 0.0"
        try:
            report = self.session.next()
            if report['class'] == 'TPV':
                lat = getattr(report, 'lat', None)
                lon = getattr(report, 'lon', None)
                if lat is not None and lon is not None:
                    return f"Lat: {lat}, Lon: {lon}"
        except Exception as e:
            logger.error("GPS error: " + str(e))
        return "Location unavailable"

# --- Email Notification (DRY, SOLID) ---
class EmailNotifier:
    def __init__(self):
        self.sender_email = "&&&"
        self.sender_password = "&&&"  # Replace with your actual or app-specific password
        self.receiver_email = "&&&"
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.gps_locator = GPSLocator()

    def get_additional_info(self):
        # Gather extra info as a dictionary (easy to later store in SQLite)
        timestamp = time.ctime()
        device_id = Config.DEVICE_ID
        try:
            local_ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            local_ip = "Unavailable"
        gps_info = self.gps_locator.get_location()
        return {
            "Timestamp": timestamp,
            "Device ID": device_id,
            "Local IP": local_ip,
            "GPS Location": gps_info
        }

    def send_notification(self, frame):
        self.executor.submit(self._send_email, frame)

    def _send_email(self, frame):
        info = self.get_additional_info()
        # Create a simple list of key-value pairs for the email body.
        info_lines = [f"{key}: {value}" for key, value in info.items()]
        extra_info = "\n".join(info_lines)
        subject = "Face Detection Notification"
        body = f"A face was detected by your camera.\n\n{extra_info}\n\nSee the attached image."

        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = self.sender_email
        msg["To"] = self.receiver_email
        msg.attach(MIMEText(body, "plain"))

        # Compress the frame to JPEG before attaching
        success, encoded_image = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        if not success:
            logger.error("Failed to encode image for email.")
            return

        img = MIMEImage(encoded_image.tobytes(), _subtype="jpeg")
        img.add_header("Content-Disposition", "attachment", filename="detected_face.jpg")
        msg.attach(img)

        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.ehlo()
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)
            server.quit()
            logger.info("Email with image and additional info sent successfully!")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")

# --- Face Detection (Single Responsibility) ---
class FaceDetector:
    def __init__(self):
        self.model_file = "res10_300x300_ssd_iter_140000.caffemodel"
        self.config_file = "deploy.prototxt"
        if not (os.path.exists(self.model_file) and os.path.exists(self.config_file)):
            raise FileNotFoundError("DNN model files are missing. Check your paths!")
        self.net = cv2.dnn.readNetFromCaffe(self.config_file, self.model_file)
    
    def detect(self, frame, conf_threshold=0.5, min_box_area=1000):
        original_h, original_w = frame.shape[:2]
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (160, 120))
        small_h, small_w = small_frame.shape[:2]
        blob = cv2.dnn.blobFromImage(small_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([small_w, small_h, small_w, small_h])
                x1, y1, x2, y2 = box.astype("int")
                if (x2 - x1) * (y2 - y1) > min_box_area:
                    scale_x = original_w / small_w
                    scale_y = original_h / small_h
                    boxes.append((int(x1 * scale_x), int(y1 * scale_y),
                                  int(x2 * scale_x), int(y2 * scale_y)))
        return boxes

# --- PiCamera Video Streaming (Single Responsibility) ---
class PiVideoStream:
    def __init__(self, resolution=(Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT)):
        self.camera = Picamera2()
        self.camera_config = self.camera.create_video_configuration(
            main={"size": resolution, "format": "RGB888"},
            controls={"FrameDurationLimits": (20000, 20000)}
        )
        self.camera.configure(self.camera_config)
        self.camera.start()
        self.frame = None
        self.stopped = False
        self.frame_lock = Lock()
    
    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self
    
    def update(self):
        while not self.stopped:
            frame = self.camera.capture_array()
            with self.frame_lock:
                self.frame = frame
    
    def read(self):
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        self.stopped = True
        self.camera.stop()

# --- Face Tracking (High-Level Logic) ---
class FaceTracker:
    def __init__(self):
        self.config = Config
        self.video_stream = PiVideoStream()
        self.detector = FaceDetector()
        self.notifier = EmailNotifier()
        self.last_email_time = 0
        self.face_loss_counter = 0
        self.consecutive_valid_faces = 0
        self.sentry_active_event = Event()
        self.motion_lock = Lock()
        # Initial motor positions
        self.pan_cx = 0
        self.pan_cy = self.config.SENTRY_TILT_OFFSET

    def _reset_motion(self):
        self.pan_cx = 0
        self.pan_cy = self.config.SENTRY_TILT_OFFSET
        pan(self.pan_cx)
        tilt(self.pan_cy)
    
    def sentry_mode(self):
        logger.info("Sentry mode activated.")
        tilt_angle = self.config.SENTRY_TILT_OFFSET
        pan_direction = 1
        pan_angle = -self.config.PAN_LIMIT
        while self.sentry_active_event.is_set():
            pan_angle += pan_direction * self.config.SENTRY_SWEEP_STEP
            if pan_angle > self.config.PAN_LIMIT or pan_angle < -self.config.PAN_LIMIT:
                pan_direction *= -1
                pan_angle += pan_direction * self.config.SENTRY_SWEEP_STEP
            pan(pan_angle)
            tilt(tilt_angle)
            time.sleep(self.config.SENTRY_WAIT_TIME)
        logger.info("Sentry mode finished.")
        self.sentry_active_event.clear()

    def track(self):
        self.video_stream.start()
        time.sleep(1)
        self._reset_motion()
        logger.info("Tracking started...")
        
        while True:
            img_frame = self.video_stream.read()
            if img_frame is None:
                continue
            
            boxes = self.detector.detect(img_frame)
            if boxes:
                self.consecutive_valid_faces += 1
                self.face_loss_counter = 0

                with self.motion_lock:
                    if self.consecutive_valid_faces > 3 and time.time() - self.last_email_time > self.config.EMAIL_INTERVAL:
                        self.notifier.send_notification(img_frame.copy())
                        self.last_email_time = time.time()

                if self.sentry_active_event.is_set():
                    self.sentry_active_event.clear()

                # Choose the largest detected face
                x, y, x2, y2 = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                cx, cy = (x + x2) // 2, (y + y2) // 2
                offset_x = ((self.config.CAMERA_WIDTH / 2) - cx) / self.config.PAN_SENSITIVITY
                offset_y = -((self.config.CAMERA_HEIGHT / 2) - cy) / self.config.TILT_SENSITIVITY

                if abs(offset_x) >= self.config.MOVE_THRESHOLD:
                    self.pan_cx = max(min(self.pan_cx + offset_x, self.config.PAN_LIMIT), -self.config.PAN_LIMIT)
                if abs(offset_y) >= self.config.MOVE_THRESHOLD:
                    self.pan_cy = max(min(self.pan_cy + offset_y, self.config.TILT_LIMIT), -self.config.TILT_LIMIT)

                pan(self.pan_cx)
                tilt(self.pan_cy)
            else:
                self.face_loss_counter += 1
                logger.info(f"No face detected. Count: {self.face_loss_counter}")
                if self.face_loss_counter > self.config.MAX_FACE_LOSS_FRAMES and not self.sentry_active_event.is_set():
                    self.sentry_active_event.set()
                    Thread(target=self.sentry_mode, daemon=True).start()
            
            cv2.imshow("Face Tracking", img_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        self.video_stream.stop()
        cv2.destroyAllWindows()

def main():
    tracker = FaceTracker()
    try:
        tracker.track()
    except KeyboardInterrupt:
        logger.info("Exiting...")

if __name__ == "__main__":
    main()

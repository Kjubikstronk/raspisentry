import time
import logging
from threading import Thread, Lock, Event
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
import smtplib
import mediapipe as mp
from picamera2 import Picamera2
from pantilthat import pan, tilt
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from typing import List, Tuple

# --- Basic Logging ---
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

# --- Configuration ---
class Config:
    CAMERA_WIDTH               = 320
    CAMERA_HEIGHT              = 200
    PAN_SENSITIVITY            = 15
    TILT_SENSITIVITY           = 15
    PAN_LIMIT                  = 70
    TILT_LIMIT                 = 90
    MOVE_THRESHOLD             = 2.0    # only move if offset ? 2 px
    SENTRY_TILT_OFFSET         = -50
    SENTRY_SWEEP_STEP          = 3
    SENTRY_WAIT_TIME           = 0.05
    EMAIL_INTERVAL             = 60
    MIN_BOX_AREA               = 5000   # only accept boxes ?
    DEVICE_ID                  = "RaspiSentry-001"
    DETECTION_CONFIDENCE       = 0.8    # only accept ?80% confidence
    MIN_CONSECUTIVE_DETECTIONS = 5      # require 5 frames before moving
    SMOOTHING_ALPHA            = 0.7    # pan/tilt smoothing factor (0?1)
    IDLE_TIME                  = 2.0    # seconds of ?no face? before sweeping

# --- GPS Locator ---
class GPSLocator:
    def __init__(self):
        try:
            import gps
            self.session   = gps.gps(mode=gps.WATCH_ENABLE)
            self.use_dummy = False
        except ImportError:
            logger.warning("gps module not available; using IP-based location.")
            self.use_dummy = True

    def get_location(self) -> str:
        if not self.use_dummy:
            try:
                report = self.session.next()
                if report['class'] == 'TPV':
                    lat = getattr(report, 'lat', None)
                    lon = getattr(report, 'lon', None)
                    if lat is not None and lon is not None:
                        return f"Lat: {lat}, Lon: {lon}"
            except Exception as e:
                logger.error("GPS error: %s", e)

        try:
            r = requests.get("http://ip-api.com/json/")
            r.raise_for_status()
            data = r.json()
            city, region, country = data.get("city"), data.get("regionName"), data.get("country")
            if city and region and country:
                return f"{city}, {region}, {country}"
            lat, lon = data.get("lat"), data.get("lon")
            if lat is not None and lon is not None:
                return f"Lat: {lat}, Lon: {lon}"
        except Exception as e:
            logger.error("IP geolocation error: %s", e)

        return "Location unavailable"

# --- Email Notifier ---
class EmailNotifier:
    def __init__(self):
        self.sender_email    = "!!!"
        self.sender_password = "!!!"
        self.receiver_email  = "!!!"
        self.executor        = ThreadPoolExecutor(max_workers=2)
        self.gps_locator     = GPSLocator()

    def _get_public_ip(self) -> str:
        try:
            return requests.get("https://api.ipify.org").text
        except Exception as e:
            logger.error("Error getting public IP: %s", e)
            return "Unavailable"

    def _get_info_str(self) -> str:
        info = {
            "Timestamp": time.ctime(),
            "Device ID": Config.DEVICE_ID,
            "Public IP": self._get_public_ip(),
            "Location":  self.gps_locator.get_location()
        }
        return "\n".join(f"{k}: {v}" for k, v in info.items())

    def send_notification(self, frame: np.ndarray):
        self.executor.submit(self._send_email, frame)

    def _send_email(self, frame: np.ndarray):
        body = "A face was detected.\n\n" + self._get_info_str()
        msg = MIMEMultipart()
        msg["Subject"] = "Face Detection Notification"
        msg["From"]    = self.sender_email
        msg["To"]      = self.receiver_email
        msg.attach(MIMEText(body, "plain"))

        success, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        if not success:
            logger.error("Failed to encode frame for email.")
            return

        img = MIMEImage(jpg.tobytes(), _subtype="jpeg")
        img.add_header("Content-Disposition", "attachment", filename="detected_face.jpg")
        msg.attach(img)

        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as s:
                s.ehlo()
                s.starttls()
                s.login(self.sender_email, self.sender_password)
                s.send_message(msg)
            logger.info("Email sent.")
        except Exception as e:
            logger.error("Failed to send email: %s", e)

# --- Face Detector (MediaPipe) ---
class FaceDetector:
    def __init__(self):
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=Config.DETECTION_CONFIDENCE
        )

    def detect(self, frame: np.ndarray) -> List[Tuple[int,int,int,int]]:
        h, w    = frame.shape[:2]
        result  = self.detector.process(frame)
        boxes   = []
        if result.detections:
            for det in result.detections:
                if det.score[0] < Config.DETECTION_CONFIDENCE:
                    continue
                bb = det.location_data.relative_bounding_box
                x1 = max(int(bb.xmin * w), 0)
                y1 = max(int(bb.ymin * h), 0)
                x2 = min(x1 + int(bb.width  * w), w)
                y2 = min(y1 + int(bb.height * h), h)
                if (x2-x1)*(y2-y1) > Config.MIN_BOX_AREA:
                    boxes.append((x1, y1, x2, y2))
        return boxes

# --- PiCamera Stream ---
class PiVideoStream:
    def __init__(self, res=(Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT)):
        self.camera      = Picamera2()
        cfg = self.camera.create_video_configuration(
            main={"size":res, "format":"RGB888"},
            controls={"FrameDurationLimits":(20000,20000)}
        )
        self.camera.configure(cfg)
        self.camera.start()
        self.frame      = None
        self.stopped    = False
        self.frame_lock = Lock()

    def start(self):
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            img = self.camera.capture_array()
            with self.frame_lock:
                self.frame = img

    def read(self) -> np.ndarray:
        with self.frame_lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.camera.stop()

# --- Face Tracker ---
class FaceTracker:
    def __init__(self):
        self.stream            = PiVideoStream()
        self.detector          = FaceDetector()
        self.notifier          = EmailNotifier()
        self.last_email_time   = time.monotonic()
        self.consecutive_valid = 0
        self.last_seen         = time.monotonic()
        self.sentry_event      = Event()
        self.motion_lock       = Lock()
        self._stop_event       = Event()
        self.pan_cx            = 0.0
        self.pan_cy            = float(Config.SENTRY_TILT_OFFSET)

    def _reset(self):
        self.pan_cx = 0.0
        self.pan_cy = float(Config.SENTRY_TILT_OFFSET)
        pan(self.pan_cx); tilt(self.pan_cy)

    def sentry(self):
        logger.info("Entering sentry sweep")
        tilt_angle = Config.SENTRY_TILT_OFFSET
        direction  = 1
        angle      = -Config.PAN_LIMIT
        while self.sentry_event.is_set():
            angle += direction * Config.SENTRY_SWEEP_STEP
            if abs(angle) > Config.PAN_LIMIT:
                direction *= -1
                angle += direction * Config.SENTRY_SWEEP_STEP
            pan(angle); tilt(tilt_angle)
            time.sleep(Config.SENTRY_WAIT_TIME)
        logger.info("Exiting sentry sweep")

    def track(self):
        self.stream.start()
        time.sleep(1)
        self._reset()
        logger.info("Tracking started")

        try:
            while not self._stop_event.is_set():
                frame = self.stream.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                now   = time.monotonic()
                boxes = self.detector.detect(frame)

                if boxes:
                    self.last_seen = now
                    self.consecutive_valid += 1

                    if self.consecutive_valid >= Config.MIN_CONSECUTIVE_DETECTIONS:
                        if now - self.last_email_time > Config.EMAIL_INTERVAL:
                            self.notifier.send_notification(frame.copy())
                            self.last_email_time = now

                        if self.sentry_event.is_set():
                            self.sentry_event.clear()

                        x1,y1,x2,y2 = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                        cx,cy      = (x1+x2)/2.0, (y1+y2)/2.0
                        off_x      = (Config.CAMERA_WIDTH/2 - cx)/Config.PAN_SENSITIVITY
                        off_y      = -(Config.CAMERA_HEIGHT/2 - cy)/Config.TILT_SENSITIVITY

                        if abs(off_x) >= Config.MOVE_THRESHOLD:
                            tgt = np.clip(self.pan_cx+off_x, -Config.PAN_LIMIT, Config.PAN_LIMIT)
                            self.pan_cx = (Config.SMOOTHING_ALPHA*self.pan_cx +
                                           (1-Config.SMOOTHING_ALPHA)*tgt)
                        if abs(off_y) >= Config.MOVE_THRESHOLD:
                            tgt = np.clip(self.pan_cy+off_y, -Config.TILT_LIMIT, Config.TILT_LIMIT)
                            self.pan_cy = (Config.SMOOTHING_ALPHA*self.pan_cy +
                                           (1-Config.SMOOTHING_ALPHA)*tgt)

                        pan(self.pan_cx); tilt(self.pan_cy)
                else:
                    self.consecutive_valid = 0
                    # if idle too long, start sentry
                    if (now - self.last_seen) > Config.IDLE_TIME and not self.sentry_event.is_set():
                        self.sentry_event.set()
                        Thread(target=self.sentry, daemon=True).start()

                # display (RGB?BGR for OpenCV)
                cv2.imshow("Face Tracking", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except Exception:
            logger.exception("Error in tracking loop")
        finally:
            self.stream.stop()
            cv2.destroyAllWindows()

    def stop(self):
        self._stop_event.set()

if __name__ == "__main__":
    try:
        FaceTracker().track()
    except KeyboardInterrupt:
        logger.info("Shutting down")

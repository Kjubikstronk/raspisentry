import os
import time
import math
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

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

DEBUG = False  # Toggle for detailed debug output

class TrackerConfig:
    def __init__(self):
        # Camera settings
        self.CAMERA_WIDTH = 320
        self.CAMERA_HEIGHT = 200

        # Sensitivity for pan/tilt adjustments
        self.PAN_SENSITIVITY = 15
        self.TILT_SENSITIVITY = 15

        # Movement limits for pan/tilt
        self.PAN_LIMIT = 70
        self.TILT_LIMIT = 90

        # Minimum movement threshold
        self.MOVE_THRESHOLD = 0.5

        # Sentry mode settings
        self.MAX_FACE_LOSS_FRAMES = 30
        self.SENTRY_TILT_OFFSET = -50  # Default tilt for scanning
        self.SENTRY_SWEEP_STEP = 3     # Sweep step size
        self.SENTRY_WAIT_TIME = 0.05   # Pause between sentry movements

        # Email throttling (seconds between emails)
        self.EMAIL_INTERVAL = 60

config = TrackerConfig()

# Paths to the DNN model files
MODEL_FILE = "res10_300x300_ssd_iter_140000.caffemodel"
CONFIG_FILE = "deploy.prototxt"

if not (os.path.exists(MODEL_FILE) and os.path.exists(CONFIG_FILE)):
    raise FileNotFoundError("DNN model files are missing. Check your paths!")

# Load the pre-trained DNN model
net = cv2.dnn.readNetFromCaffe(CONFIG_FILE, MODEL_FILE)

# Shared resources and thread control
sentry_active_event = Event()  # Controls whether sentry mode is active
last_email_time = 0            # Timestamp for throttling emails
lock = Lock()                  # Protect shared variables

# Thread pool for asynchronous email sending
email_executor = ThreadPoolExecutor(max_workers=2)

def send_notification_email_with_image(frame):
    """
    Sends an email with an attached image.
    Email credentials are now hard-coded in plain text.
    """
    SENDER_EMAIL = " "
    SENDER_PASSWORD = " "  # Replace with your actual password or app-specific password
    RECEIVER_EMAIL = " "
    
    subject = "Face Detection Notification"
    body = "A face was detected by your camera. See the attached image."

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg.attach(MIMEText(body, "plain"))

    # Compress the frame to JPEG before sending
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
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        logger.info("Email with image sent successfully!")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

def detect_faces_dnn(frame, conf_threshold=0.5, min_box_area=1000):
    """
    Runs the DNN model to detect faces in a frame.
    Uses dynamic scaling based on the original and resized frame sizes.
    """
    original_h, original_w = frame.shape[:2]
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (160, 120))
    small_h, small_w = small_frame.shape[:2]

    blob = cv2.dnn.blobFromImage(small_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([small_w, small_h, small_w, small_h])
            x1, y1, x2, y2 = box.astype("int")
            box_area = (x2 - x1) * (y2 - y1)
            if box_area > min_box_area:
                scale_x = original_w / small_w
                scale_y = original_h / small_h
                boxes.append((int(x1 * scale_x), int(y1 * scale_y),
                              int(x2 * scale_x), int(y2 * scale_y)))
    return boxes

def sentry_mode():
    """
    Moves the camera in a sweeping pattern (sentry mode) when no faces are detected.
    Uses an Event to control activation.
    """
    logger.info("Sentry mode activated.")
    tilt_angle = config.SENTRY_TILT_OFFSET
    pan_direction = 1  # Initial direction: right
    pan_angle = -config.PAN_LIMIT  # Start at leftmost position

    try:
        while sentry_active_event.is_set():
            pan_angle += pan_direction * config.SENTRY_SWEEP_STEP
            if pan_angle > config.PAN_LIMIT or pan_angle < -config.PAN_LIMIT:
                pan_direction *= -1
                pan_angle += pan_direction * config.SENTRY_SWEEP_STEP

            pan(pan_angle)
            tilt(tilt_angle)
            time.sleep(config.SENTRY_WAIT_TIME)
    finally:
        sentry_active_event.clear()
        logger.info("Sentry mode finished.")

def track_face():
    """
    Main loop for tracking faces in real time.
    Switches to sentry mode after prolonged absence of faces.
    """
    global last_email_time

    vs = PiVideoStream().start()
    time.sleep(1)  # Allow the camera to initialize

    pan_cx, pan_cy = 0, config.SENTRY_TILT_OFFSET
    pan(pan_cx)
    tilt(pan_cy)

    face_loss_counter = 0
    consecutive_valid_faces = 0
    logger.info("Tracking started...")

    try:
        while True:
            img_frame = vs.read()
            if img_frame is None:
                continue

            faces = detect_faces_dnn(img_frame)

            if len(faces) > 0:
                consecutive_valid_faces += 1
                face_loss_counter = 0

                # Send email with image attachment if conditions are met
                with lock:
                    if consecutive_valid_faces > 3 and time.time() - last_email_time > config.EMAIL_INTERVAL:
                        email_executor.submit(send_notification_email_with_image, img_frame.copy())
                        last_email_time = time.time()

                if sentry_active_event.is_set():
                    sentry_active_event.clear()

                # Choose the largest detected face
                x, y, x2, y2 = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                cx, cy = (x + x2) // 2, (y + y2) // 2

                offset_x = ((config.CAMERA_WIDTH / 2) - cx) / config.PAN_SENSITIVITY
                offset_y = -((config.CAMERA_HEIGHT / 2) - cy) / config.TILT_SENSITIVITY

                if abs(offset_x) >= config.MOVE_THRESHOLD:
                    pan_cx = max(min(pan_cx + offset_x, config.PAN_LIMIT), -config.PAN_LIMIT)
                if abs(offset_y) >= config.MOVE_THRESHOLD:
                    pan_cy = max(min(pan_cy + offset_y, config.TILT_LIMIT), -config.TILT_LIMIT)

                pan(pan_cx)
                tilt(pan_cy)
            else:
                face_loss_counter += 1
                if face_loss_counter > config.MAX_FACE_LOSS_FRAMES and not sentry_active_event.is_set():
                    sentry_active_event.set()
                    Thread(target=sentry_mode, daemon=True).start()

            cv2.imshow("Face Tracking", img_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        vs.stop()
        cv2.destroyAllWindows()

class PiVideoStream:
    """
    Handles video streaming from the PiCamera with thread-safe frame access.
    """
    def __init__(self, resolution=(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)):
        self.camera = Picamera2()
        self.camera_config = self.camera.create_video_configuration(
            main={"size": resolution, "format": "RGB888"},
            controls={"FrameDurationLimits": (20000, 20000)},
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

if __name__ == "__main__":
    try:
        track_face()
    except KeyboardInterrupt:
        logger.info("Exiting...")

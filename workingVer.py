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
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor
import logging

# Set up logging to keep track of what's happening
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()
DEBUG = False  # Toggle this to True if you need detailed debug output

class TrackerConfig:
    def __init__(self):
        # Camera settings
        self.CAMERA_WIDTH = 320  # Smaller width for better performance
        self.CAMERA_HEIGHT = 200  # Corresponding height for the resolution

        # Sensitivity for adjusting the pan/tilt
        self.PAN_SENSITIVITY = 15
        self.TILT_SENSITIVITY = 15

        # Movement limits to prevent the camera from going out of range
        self.PAN_LIMIT = 70
        self.TILT_LIMIT = 90

        # How much movement is required before adjusting position
        self.MOVE_THRESHOLD = 0.5

        # Settings for face loss and sentry mode
        self.MAX_FACE_LOSS_FRAMES = 30
        self.SENTRY_TILT_OFFSET = -50  # Default tilt when scanning
        self.SENTRY_SWEEP_STEP = 3  # Step size for sweeping motion
        self.SENTRY_WAIT_TIME = 0.05  # Pause between each movement

        # Email throttling to prevent spam
        self.EMAIL_INTERVAL = 60  # Minimum time (in seconds) between emails

config = TrackerConfig()

# Paths to the DNN model files
MODEL_FILE = "res10_300x300_ssd_iter_140000.caffemodel"
CONFIG_FILE = "deploy.prototxt"

# Check if the DNN files exist, fail early if not
if not (os.path.exists(MODEL_FILE) and os.path.exists(CONFIG_FILE)):
    raise FileNotFoundError("DNN model files are missing. Check your paths!")

# Load the pre-trained DNN model
net = cv2.dnn.readNetFromCaffe(CONFIG_FILE, MODEL_FILE)

# Flags and shared variables
sentry_active = False  # Indicates whether sentry mode is running
last_email_time = 0  # Tracks the last time an email was sent
lock = Lock()  # Protect shared resources like email timestamps

# Thread pool to handle email sending without blocking the main process
email_executor = ThreadPoolExecutor(max_workers=2)

def send_notification_email_with_image(frame):
    """
    Handles email notifications with an attached image. Offloads the task to a thread pool.
    """
    email_executor.submit(_send_email_task, frame)

def _send_email_task(frame):
    """
    The core logic for sending an email with the image attached.
    """
    SENDER_EMAIL = " "
    SENDER_PASSWORD = " "  # Replace with your app-specific password if 2FA is enabled
    RECEIVER_EMAIL = " "

    subject = "Face Detection Notification"
    body = "A face was detected by your camera. See the attached image."

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg.attach(MIMEText(body, "plain"))

    # Compress the frame to a smaller JPEG before sending
    success, encoded_image = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
    if not success:
        logger.error("Failed to encode image for email.")
        return

    img = MIMEImage(encoded_image.tobytes(), _subtype="jpeg")
    img.add_header("Content-Disposition", "attachment", filename="detected_face.jpg")
    msg.attach(img)

    try:
        # Set up the email server and send the email
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
    Runs the DNN model to detect faces in a frame. Performs some preprocessing to improve detection speed.
    """
    # Scale down the frame to speed up processing
    small_frame = cv2.resize(frame, (160, 120))
    gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    enhanced_frame = cv2.equalizeHist(gray_frame)

    h, w = small_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(small_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []

    # Loop through detections and filter by confidence and size
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            box_area = (x2 - x1) * (y2 - y1)
            if box_area > min_box_area:
                # Scale the box back to the original resolution
                boxes.append((x1 * 2, y1 * 2, x2 * 2, y2 * 2))

    return boxes

def sentry_mode():
    """
    Moves the camera in a sweeping pattern when no faces are detected.
    """
    global sentry_active
    logger.info("Sentry mode activated.")

    tilt_angle = config.SENTRY_TILT_OFFSET
    pan_direction = 1  # Start by moving right
    pan_angle = -config.PAN_LIMIT  # Start from the leftmost position

    try:
        while sentry_active:
            pan_angle += pan_direction * config.SENTRY_SWEEP_STEP
            if pan_angle > config.PAN_LIMIT or pan_angle < -config.PAN_LIMIT:
                pan_direction *= -1
                pan_angle += pan_direction * config.SENTRY_SWEEP_STEP

            pan(pan_angle)
            tilt(tilt_angle)
            time.sleep(config.SENTRY_WAIT_TIME)
    finally:
        sentry_active = False
        logger.info("Sentry mode finished.")

def track_face():
    """
    Main loop to track faces in real time and handle sentry mode.
    """
    global sentry_active, last_email_time

    vs = PiVideoStream().start()
    time.sleep(1)

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

                if consecutive_valid_faces > 3 and time.time() - last_email_time > config.EMAIL_INTERVAL:
                    send_notification_email_with_image(img_frame.copy())
                    last_email_time = time.time()

                if sentry_active:
                    sentry_active = False

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
                if face_loss_counter > config.MAX_FACE_LOSS_FRAMES and not sentry_active:
                    sentry_active = True
                    Thread(target=sentry_mode, daemon=True).start()

            cv2.imshow("Face Tracking", img_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        vs.stop()
        cv2.destroyAllWindows()

class PiVideoStream:
    """
    Handles video streaming from the PiCamera.
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

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.frame = self.camera.capture_array()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.camera.stop()

if __name__ == "__main__":
    try:
        track_face()
    except KeyboardInterrupt:
        logger.info("Exiting...")

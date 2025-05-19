import json
import os
import time
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv, set_key

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'notification_config.json')
ENV_PATH = os.path.join(os.path.dirname(__file__), '.env')

# --- Notification Config Management ---
def is_notification_enabled():
    if not os.path.exists(CONFIG_PATH):
        return False
    try:
        with open(CONFIG_PATH, 'r') as f:
            data = json.load(f)
            return data.get('enabled', False)
    except Exception:
        return False

def set_notification_enabled(enabled: bool):
    with open(CONFIG_PATH, 'w') as f:
        json.dump({'enabled': enabled}, f)

# --- Email Config Management ---
def get_email_config():
    load_dotenv(ENV_PATH, override=True)
    return {
        'sender': os.getenv('EMAIL_SENDER', ''),
        'receiver': os.getenv('EMAIL_RECEIVER', ''),
        'password': os.getenv('EMAIL_PASSWORD', ''),
        'api_key': os.getenv('EMAIL_API_KEY', '')
    }

def set_email_config(sender, receiver, password, api_key):
    set_key(ENV_PATH, 'EMAIL_SENDER', sender)
    set_key(ENV_PATH, 'EMAIL_RECEIVER', receiver)
    set_key(ENV_PATH, 'EMAIL_PASSWORD', password)
    set_key(ENV_PATH, 'EMAIL_API_KEY', api_key)

# --- Email Notifier ---
class EmailNotifier:
    def __init__(self) -> None:
        self.sender_email    = os.getenv('EMAIL_SENDER', 'your@email.com')
        self.sender_password = os.getenv('EMAIL_PASSWORD', 'yourpassword')
        self.receiver_email  = os.getenv('EMAIL_RECEIVER', 'receiver@email.com')
        self.executor        = ThreadPoolExecutor(max_workers=2)

    def _get_public_ip(self) -> str:
        try:
            return requests.get("https://api.ipify.org", timeout=5).text
        except:
            return "Unavailable"

    def send(self, frame, confidence, area, db):
        if not is_notification_enabled():
            return
        self.executor.submit(self._send_email, frame, confidence, area, db)

    def _send_email(self, frame, confidence, area, db):
        from sentry_core import get_location  # avoid circular import
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        public_ip = self._get_public_ip()
        location  = get_location()

        import cv2
        ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        image_bytes = jpg.tobytes() if ok else None

        db.log_detection(confidence, area, location, public_ip, "RaspiSentry-001", image_bytes)

        body = (
            f"Timestamp: {timestamp}\n"
            f"Device ID: RaspiSentry-001\n"
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
                print("Email sent successfully.")
        except Exception as e:
            print(f"Email error: {e}") 
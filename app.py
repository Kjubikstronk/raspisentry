from flask import Flask, render_template, send_file, abort, url_for, request, redirect, flash, jsonify, Response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import sqlite3
import io
import os
import cv2
import numpy as np
import time
import json
import requests
from flask_socketio import SocketIO, emit
try:
    import psutil
except ImportError:
    psutil = None
from pantilthat import pan, tilt  # Add this import
from email_notify import is_notification_enabled, set_notification_enabled, get_email_config, set_email_config

# --- Configuration ---
DB_PATH = os.path.join(os.path.dirname(__file__), 'face_logs.db')
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')  # Change this to a secure secret key in production

# Raspberry Pi Configuration
# Set these environment variables to match your Raspberry Pi's settings
RASPI_HOST = '192.168.244.46'  # Raspberry Pi's IP address
RASPI_PORT = 5001  # Must match the port your Raspberry Pi server is running on

# Flask Server Configuration
FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')  # Set to 0.0.0.0 to allow external access
FLASK_PORT = int(os.getenv('FLASK_PORT', '5000'))  # Port for the Flask server

# Initialize Flask app
app = Flask(__name__)
app.secret_key = SECRET_KEY

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for tracking system
tracking_state = {
    'sentry_mode': False,
    'pan': 0,
    'tilt': 0,
    'is_tracking': False
}

# Create a single session for all Raspberry Pi communication
raspi_session = requests.Session()
raspi_session.headers.update({
    'Connection': 'keep-alive',
    'Keep-Alive': 'timeout=5, max=1000'
})

# --- Flask-Login Setup ---
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# --- Database Migration / Initialization ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # Create detections table if missing, including face_image
    cur.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            confidence REAL,
            face_area INTEGER,
            location TEXT,
            ip TEXT,
            device_id TEXT,
            face_image BLOB
        )
    ''')
    # Ensure face_image column exists (for older DBs)
    cur.execute("PRAGMA table_info(detections)")
    cols = [row[1] for row in cur.fetchall()]
    if 'face_image' not in cols:
        cur.execute('ALTER TABLE detections ADD COLUMN face_image BLOB')
    conn.commit()
    conn.close()

def add_sample_data():
    """Add sample detection data for testing"""
    import random
    from datetime import datetime, timedelta
    import numpy as np
    
    # Create a sample face image (just a colored rectangle for testing)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    _, img_encoded = cv2.imencode('.jpg', img)
    img_blob = img_encoded.tobytes()
    
    locations = ['Front Door', 'Back Yard', 'Garage', 'Side Gate', 'Living Room']
    device_ids = ['CAM001', 'CAM002', 'CAM003']
    ips = ['192.168.1.100', '192.168.1.101', '192.168.1.102']
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Add 20 sample detections
    for i in range(20):
        timestamp = (datetime.now() - timedelta(minutes=i*30)).strftime('%Y-%m-%d %H:%M:%S')
        confidence = round(random.uniform(0.75, 0.99), 2)
        face_area = random.randint(1000, 5000)
        location = random.choice(locations)
        ip = random.choice(ips)
        device_id = random.choice(device_ids)
        
        cur.execute('''
            INSERT INTO detections (timestamp, confidence, face_area, location, ip, device_id, face_image)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, confidence, face_area, location, ip, device_id, img_blob))
    
    conn.commit()
    conn.close()

# Run migrations and add sample data on startup
init_db()
add_sample_data()  # Comment this line out after first run if you don't want to keep adding sample data

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# --- Data Access ---
def get_detections(limit=50):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute('''
            SELECT id, timestamp, confidence, face_area, location, ip, device_id
            FROM detections
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        detections = []
        for row in cur.fetchall():
            detections.append({
                'id': row[0],
                'timestamp': row[1],
                'confidence': row[2],
                'face_area': row[3],
                'location': row[4],
                'ip': row[5],
                'device_id': row[6]
            })
        conn.close()
        return detections
    except Exception as e:
        logger.error(f"Error fetching detections from local database: {e}")
        return []

def get_image_blob(det_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute('SELECT face_image FROM detections WHERE id = ?', (det_id,))
        result = cur.fetchone()
        conn.close()
        if result and result[0]:
            return result[0]
        logger.error(f"No image found for detection ID: {det_id}")
        return None
    except Exception as e:
        logger.error(f"Error fetching image from local database: {e}")
        return None

def get_cpu_temp():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = int(f.read()) / 1000.0
        return f"{temp:.1f}°C"
    except Exception:
        return "N/A"

def get_system_status():
    status = {}
    if psutil:
        status['cpu_percent'] = psutil.cpu_percent(interval=0.2)
        status['ram_percent'] = psutil.virtual_memory().percent
        status['disk_percent'] = psutil.disk_usage('/').percent
        uptime_sec = time.time() - psutil.boot_time()
        hours, rem = divmod(uptime_sec, 3600)
        minutes, seconds = divmod(rem, 60)
        status['uptime'] = f"{int(hours)}h {int(minutes)}m"
    else:
        status['cpu_percent'] = status['ram_percent'] = status['disk_percent'] = 'N/A'
        status['uptime'] = 'N/A'
    status['cpu_temp'] = get_cpu_temp()
    return status

# --- Routes ---
@app.route('/')
def splash():
    return render_template('splash.html')

@app.route('/dashboard')
@login_required
def dashboard():
    detections = get_detections()
    return render_template('index.html', detections=detections)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Simple hardcoded admin/admin credentials
        if username == 'admin' and password == 'admin':
            user = User(username)
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/image/<int:det_id>')
@login_required
def image(det_id):
    blob = get_image_blob(det_id)
    if blob is None:
        abort(404)
    # Decode image from blob
    arr = np.frombuffer(blob, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        abort(500)
    # Overlay text at bottom of ROI
    text = "INTRUDER!"
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_h, img_w = img.shape[:2]
    # scale text to fit width
    scale = max(0.5, min(img_w / 300.0, img_h / 50.0))
    thickness = int(max(1, img_w / 200))
    color = (0, 0, 255)  # red
    # position text at bottom center
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(10, (img_w - text_w) // 2)
    y = img_h - 10
    cv2.putText(img, text, (x, y), font, scale, color, thickness)
    # Re-encode annotated image
    ok, encoded = cv2.imencode('.jpg', img)
    if not ok:
        abort(500)
    return send_file(
        io.BytesIO(encoded.tobytes()),
        mimetype='image/jpeg',
        as_attachment=False,
        download_name=f"face_{det_id}.jpg"
    )

@app.route('/system_status')
@login_required
def system_status():
    return jsonify(get_system_status())

@app.route('/video_feed')
@login_required
def video_feed():
    """Video streaming route"""
    try:
        url = f'http://{RASPI_HOST}:{RASPI_PORT}/video'
        logger.info(f"Attempting to connect to video feed at {url}")
        response = raspi_session.get(
            url,
            stream=True,
            timeout=10
        )
        if response.status_code != 200:
            logger.error(f"Video feed server returned status code {response.status_code}")
            return f"Video feed server error: {response.status_code}", 500
        logger.info("Successfully connected to video feed")
        def generate():
            try:
                for chunk in response.iter_content(chunk_size=10*1024):
                    if chunk:
                        yield chunk
            except Exception as e:
                logger.error(f"Error in video stream: {e}")
                time.sleep(1)
                return video_feed()
        return Response(
            generate(),
            content_type=response.headers.get('Content-Type', 'multipart/x-mixed-replace; boundary=frame'),
            headers={
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error to video feed: {e}")
        return f"Could not connect to video feed at {RASPI_HOST}:{RASPI_PORT}", 500
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout error to video feed: {e}")
        return "Video feed request timed out", 500
    except Exception as e:
        logger.error(f"Error accessing video feed: {e}")
        return str(e), 500

@app.route('/liveview')
@login_required
def liveview():
    return render_template('liveview.html')

@app.route('/api/notifications/enabled', methods=['GET'])
@login_required
def get_notifications_enabled():
    return jsonify({'enabled': is_notification_enabled()})

@app.route('/api/notifications/enabled', methods=['POST'])
@login_required
def set_notifications_enabled():
    data = request.get_json()
    enabled = bool(data.get('enabled', False))
    set_notification_enabled(enabled)
    return jsonify({'enabled': enabled})

@app.route('/api/sentry/status', methods=['GET'])
@login_required
def get_sentry_status():
    # Example: return sentry mode status from tracking_state
    return jsonify({'active': tracking_state.get('sentry_mode', False)})

@app.route('/api/email/config', methods=['GET'])
@login_required
def get_email_config_api():
    config = get_email_config()
    # Do not send password/api_key in response for security
    config.pop('password', None)
    config.pop('api_key', None)
    return jsonify(config)

@app.route('/api/email/config', methods=['POST'])
@login_required
def set_email_config_api():
    data = request.get_json()
    sender = data.get('sender', '')
    receiver = data.get('receiver', '')
    password = data.get('password', '')
    api_key = data.get('api_key', '')
    set_email_config(sender, receiver, password, api_key)
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True)

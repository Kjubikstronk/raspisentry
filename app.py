from flask import Flask, render_template, send_file, abort, url_for
import sqlite3
import io
import os
import cv2
import numpy as np

# --- Configuration ---
DB_PATH = os.path.join(os.path.dirname(__file__), 'face_logs.db')

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

# Run migrations on startup
init_db()

app = Flask(__name__)

# --- Data Access ---
def get_detections(limit=50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        '''
        SELECT id, timestamp, confidence, face_area, location, ip, device_id
          FROM detections
         WHERE face_image IS NOT NULL
         ORDER BY timestamp DESC
         LIMIT ?
        ''', (limit,)
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def get_image_blob(det_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('SELECT face_image FROM detections WHERE id = ?', (det_id,))
    result = cur.fetchone()
    conn.close()
    return result[0] if result and result[0] else None

# --- Routes ---
@app.route('/')
def index():
    detections = get_detections()
    return render_template('index.html', detections=detections)

@app.route('/image/<int:det_id>')
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

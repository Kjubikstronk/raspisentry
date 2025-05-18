# RaspiSentry

A Raspberry Pi-based face detection and tracking system with pan-tilt camera control.

## Features

- Real-time face detection using OpenCV
- Pan-tilt camera control with automatic tracking
- Sentry mode for automatic area scanning
- Email notifications for detected faces
- Web interface for viewing detections
- System status monitoring
- Database logging of detections

## Requirements

- Raspberry Pi (tested on Raspberry Pi 4)
- Raspberry Pi Camera Module
- Pimoroni Pan-Tilt HAT
- Python 3.7+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Kjubikstronk/raspisentry.git
cd raspisentry
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the face detection model files:
   - `res10_300x300_ssd_iter_140000.caffemodel`
   - `deploy.prototxt`
   Place these files in the project root directory.

4. Configure email settings in `workingVer.py`:
```python
self.sender_email    = "your-email@gmail.com"
self.sender_password = "your-app-password"
self.receiver_email  = "recipient-email@example.com"
```

## Usage

1. Run the application:
```bash
python workingVer.py
```

2. Access the web interface:
   - Open a web browser and navigate to `http://[raspberry-pi-ip]:5000`
   - Default login credentials:
     - Username: admin
     - Password: admin

3. Control the camera:
   - Use the web interface for manual control
   - Enable sentry mode for automatic scanning
   - View detected faces and system status

## Project Structure

- `workingVer.py` - Main application file
- `templates/` - HTML templates for web interface
- `static/` - Static files (CSS, JS, images)
- `face_logs.db` - SQLite database for face detections
- `requirements.txt` - Python dependencies

## Security Notes

- Change the default admin password after first login
- Use a secure email password (app password for Gmail)
- Consider using HTTPS in production
- Keep your Raspberry Pi updated with security patches

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
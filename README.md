# RaspiSentry

A Raspberry Pi turns its camera to follow your face, and emails you a photo when it
sees one. Built for a Pi 4 with a Pimoroni Pan-Tilt HAT.

> **Status: paused.** The Pi and the pan-tilt rig are currently disassembled, so
> nothing here is being actively worked on. The code is the last known-good state.

Two processes, and it matters which is which:

- **`sentry_core.py` runs on the Pi.** It owns the camera and the servos: detects
  faces with OpenCV's SSD model, steers the HAT to keep the face centred, sweeps the
  area when it loses one, and serves the video stream and a command API on port 5001.
- **`app.py` is the dashboard**, a Flask app on port 5000. It talks to the Pi over
  that API, shows the live view, and browses past detections. It can run on the Pi or
  on another machine on the same network.

## Features

- Face detection and continuous pan-tilt tracking
- Sentry mode: sweeps the area when no face is in view
- Live MJPEG stream and manual camera control from the browser
- Email alert with the captured frame attached, rate-limited to one a minute
- Detections logged to SQLite with confidence, timestamp and image

## Hardware

- Raspberry Pi (developed on a Pi 4)
- Raspberry Pi Camera Module
- Pimoroni Pan-Tilt HAT
- Python 3.7+

## Setup

Install dependencies on the Pi:

```bash
pip install -r requirements.txt
```

The face detector needs two model files in the project root, which are not checked in:

- `res10_300x300_ssd_iter_140000.caffemodel`
- `deploy.prototxt`

Both ship with OpenCV's face detector and are widely mirrored.

## Running it

On the Pi:

```bash
python sentry_core.py
```

Then the dashboard, pointed at the Pi:

```bash
RASPI_HOST=192.168.1.42 python app.py
```

Open `http://<host>:5000`. The username is `admin`. If you have not set a password,
one is generated at startup and printed to the console:

```
[RaspiSentry] Generated admin password: qN7pKx2vLm4T
```

That password changes every restart. To pin it, set `ADMIN_PASSWORD`.

Email alerts are configured in the dashboard, not in a file. What you enter is
written to a local `.env`, which is gitignored.

## Configuration

All optional, all environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `RASPI_HOST` | `127.0.0.1` | Address of the Pi running `sentry_core.py` |
| `ADMIN_USERNAME` | `admin` | Dashboard login |
| `ADMIN_PASSWORD` | generated at boot | Dashboard password |
| `SECRET_KEY` | generated at boot | Flask session key |
| `FLASK_HOST` | `0.0.0.0` | Dashboard bind address |
| `FLASK_PORT` | `5000` | Dashboard port |
| `FLASK_DEBUG` | off | Set to `1` for the Werkzeug debugger |

Leaving `SECRET_KEY` and `ADMIN_PASSWORD` unset is safe; it just means everyone gets
logged out when the dashboard restarts.

## Security

This is a camera on your network with a web interface, so a few things are worth
saying plainly:

- **Do not set `FLASK_DEBUG=1` on a reachable interface.** The Werkzeug debugger
  executes arbitrary code by design.
- There is no HTTPS. Keep it on your LAN, or put it behind a reverse proxy.
- The command server in `sentry_core.py` (port 5001) is unauthenticated. Anything on
  your network that can reach it can move the camera and read the stream.

## Files

- `sentry_core.py` — camera, tracking, sweep and command server; runs on the Pi
- `app.py` — Flask dashboard and its API
- `email_notify.py` — alert delivery and credential storage
- `templates/`, `static/` — dashboard front end
- `face_logs.db` — SQLite detection log, created on first run

## License

MIT — see [LICENSE](LICENSE).

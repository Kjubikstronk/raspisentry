@echo off
set RASPI_HOST=192.168.244.46
set RASPI_PORT=5001
set FLASK_HOST=0.0.0.0
set FLASK_PORT=5000
set SECRET_KEY=your-secret-key-here

echo Starting Flask server...
echo Raspberry Pi IP: %RASPI_HOST%
echo Raspberry Pi Port: %RASPI_PORT%
echo Flask Host: %FLASK_HOST%
echo Flask Port: %FLASK_PORT%

python app.py 
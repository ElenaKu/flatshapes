@echo off
REM Start the flatshapes test bench: the browser opens by itself; choose the image folder there.
cd /d "%~dp0"
python tools\webui\server.py %*
pause

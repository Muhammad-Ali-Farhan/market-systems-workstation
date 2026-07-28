@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Run the setup commands in README.md first.
  pause
  exit /b 1
)
"%PYTHON%" l2_capture.py --symbols BTCUSDT ETHUSDT --output-dir recordings\l2
pause

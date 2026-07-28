@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage: VERIFY_L2_RECORDING.bat recording.l2bin
  pause
  exit /b 1
)
.venv\Scripts\python.exe verify_l2_replay.py "%~1" --speeds 0 10 1
pause

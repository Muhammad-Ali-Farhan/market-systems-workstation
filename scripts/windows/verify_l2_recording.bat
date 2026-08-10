@echo off
REM Verify one L2 recording at deterministic replay speeds.
setlocal
cd /d "%~dp0\..\.."
if "%~1"=="" (
  echo Usage: verify_l2_recording.bat recording.l2bin
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m market_engine.cli.verify_l2_replay "%~1" --speeds 0 10 1
pause

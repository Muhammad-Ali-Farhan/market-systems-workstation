@echo off
REM Start a default BTCUSDT/ETHUSDT L2 capture from the repository root.
setlocal
cd /d "%~dp0\..\.."
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Run the setup commands in README.md first.
  pause
  exit /b 1
)
"%PYTHON%" -m market_engine.market_data.l2_capture --symbols BTCUSDT ETHUSDT --output-dir recordings\l2
pause

@echo off
setlocal
cd /d "%~dp0"
PowerShell -NoProfile -ExecutionPolicy Bypass -File "%~dp0BUILD_NATIVE.ps1"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" echo Build failed. Read the error above.
if "%EXIT_CODE%"=="0" echo Build completed successfully.
pause
exit /b %EXIT_CODE%

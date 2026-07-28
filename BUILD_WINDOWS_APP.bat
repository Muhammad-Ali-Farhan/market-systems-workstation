@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Run the setup commands in README.md first.
    goto :error
)
"%PYTHON%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
if errorlevel 1 (
    echo ERROR: The build environment must use CPython 3.12.
    goto :error
)
if not exist "quant_engine.cp312-win_amd64.pyd" (
    echo ERROR: Required build input quant_engine.cp312-win_amd64.pyd is missing.
    goto :error
)

"%PYTHON%" -m pip install -r requirements-build.txt
if errorlevel 1 goto :error

rmdir /s /q build-app 2>nul
rmdir /s /q dist\MarketSystemsWorkstation 2>nul

"%PYTHON%" -m PyInstaller ^
  --noconfirm --clean --windowed ^
  --name MarketSystemsWorkstation ^
  --distpath dist --workpath build-app\gui --specpath build-app\specs ^
  --paths . ^
  --add-binary "quant_engine.cp312-win_amd64.pyd;." ^
  --collect-submodules market_ui ^
  market_workstation.py
if errorlevel 1 goto :error

call :build_console MarketL2Capture l2_capture.py
if errorlevel 1 goto :error
call :build_console MarketL2Verify verify_l2_replay.py
if errorlevel 1 goto :error
call :build_console MarketL2Benchmark l2_benchmark.py
if errorlevel 1 goto :error
call :build_console MarketL2Research l2_research.py
if errorlevel 1 goto :error
call :build_console MarketL2Execute l2_execution_sensitivity.py
if errorlevel 1 goto :error

for %%F in (
  MarketSystemsWorkstation.exe
  MarketL2Capture.exe
  MarketL2Verify.exe
  MarketL2Benchmark.exe
  MarketL2Research.exe
  MarketL2Execute.exe
) do (
  if not exist "dist\MarketSystemsWorkstation\%%F" (
    echo ERROR: Missing packaged tool %%F.
    goto :error
  )
)

echo.
echo Build complete:
echo dist\MarketSystemsWorkstation\MarketSystemsWorkstation.exe
pause
exit /b 0

:build_console
set "TOOL_NAME=%~1"
set "TOOL_SCRIPT=%~2"
"%PYTHON%" -m PyInstaller ^
  --noconfirm --clean --onefile --console ^
  --name "%TOOL_NAME%" ^
  --distpath dist\MarketSystemsWorkstation ^
  --workpath "build-app\%TOOL_NAME%" ^
  --specpath build-app\specs ^
  --paths . ^
  "%TOOL_SCRIPT%"
exit /b %errorlevel%

:error
echo.
echo Build failed.
pause
exit /b 1

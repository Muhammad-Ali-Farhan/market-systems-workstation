$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
# When downloaded separately, the script should be placed in the repository root.
if (-not (Test-Path (Join-Path $Root 'Bindings.cpp'))) {
    Write-Host 'ERROR: BUILD_NATIVE.ps1 must be placed in the repository root.' -ForegroundColor Red
    exit 1
}

Set-Location $Root
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$CMake = Join-Path $Root '.venv\Scripts\cmake.exe'

if (-not (Test-Path $Python)) {
    throw 'Missing .venv\Scripts\python.exe. Create the Python 3.12 environment described in README.md first.'
}

& $Python -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version"
Write-Host '[PASS] CPython 3.12 environment found.' -ForegroundColor Green

Write-Host 'Installing/checking native build requirements...'
& $Python -m pip install -r requirements-build.txt

if (-not (Test-Path $CMake)) {
    throw 'CMake was not installed into .venv.'
}

$Candidates = @()
if ($env:VCPKG_ROOT) { $Candidates += $env:VCPKG_ROOT }
$Candidates += @(
    (Join-Path $Root 'vcpkg'),
    (Join-Path (Split-Path $Root -Parent) 'vcpkg'),
    (Join-Path $HOME 'vcpkg'),
    'C:\vcpkg'
)

$VcpkgRoot = $null
foreach ($Candidate in $Candidates | Select-Object -Unique) {
    if ($Candidate -and (Test-Path (Join-Path $Candidate 'scripts\buildsystems\vcpkg.cmake'))) {
        $VcpkgRoot = (Resolve-Path $Candidate).Path
        break
    }
}

if (-not $VcpkgRoot) {
    Write-Host 'ERROR: vcpkg could not be located.' -ForegroundColor Red
    Write-Host 'Set VCPKG_ROOT to your existing vcpkg folder, for example:' -ForegroundColor Yellow
    Write-Host '  $env:VCPKG_ROOT = "C:\vcpkg"' -ForegroundColor Yellow
    Write-Host 'Then run this script again.' -ForegroundColor Yellow
    exit 1
}

Write-Host "[PASS] vcpkg: $VcpkgRoot" -ForegroundColor Green
$Toolchain = Join-Path $VcpkgRoot 'scripts\buildsystems\vcpkg.cmake'
$Build = Join-Path $Root 'build-native'

if (Test-Path $Build) {
    Remove-Item $Build -Recurse -Force
}

Write-Host 'Configuring the native extension...'
& $CMake -S . -B $Build -A x64 `
    "-DCMAKE_TOOLCHAIN_FILE=$Toolchain" `
    '-DVCPKG_TARGET_TRIPLET=x64-windows-static-md' `
    '-DMARKET_BUILD_TESTS=ON' `
    '-DMARKET_BUILD_BENCHMARKS=ON'

Write-Host 'Building Release...'
& $CMake --build $Build --config Release --parallel

Write-Host 'Running native tests...'
& (Join-Path (Split-Path $CMake -Parent) 'ctest.exe') --test-dir $Build -C Release --output-on-failure

$BuiltPyd = Get-ChildItem (Join-Path $Build 'Release') -Filter 'quant_engine*.pyd' -File |
    Select-Object -First 1
if (-not $BuiltPyd) {
    throw 'Build completed but quant_engine*.pyd was not found in build-native\Release.'
}

$Destination = Join-Path $Root 'quant_engine.cp312-win_amd64.pyd'
if (Test-Path $Destination) {
    $Timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $BackupDirectory = Join-Path $Root '.native-backups'
    New-Item -ItemType Directory -Path $BackupDirectory -Force | Out-Null
    $Backup = Join-Path $BackupDirectory "quant_engine.cp312-win_amd64-$Timestamp.pyd"
    Copy-Item $Destination $Backup -Force
    Write-Host "Backed up old native module outside the repository surface: $Backup"
}
Copy-Item $BuiltPyd.FullName $Destination -Force
Write-Host "Installed new native module: $Destination" -ForegroundColor Green

Write-Host 'Checking the L2 binding...'
& $Python -c "import quant_engine; print('Loaded:', quant_engine.__file__); print('L2Synchronizer:', hasattr(quant_engine, 'L2Synchronizer')); assert quant_engine.L2Synchronizer().state == 'awaiting_snapshot'"

Write-Host 'Running full workstation verification...'
& $Python verify_workstation.py

Write-Host ''
Write-Host 'SUCCESS: The native module is built and installed.' -ForegroundColor Green
Write-Host 'You can now run: .\.venv\Scripts\python.exe market_workstation.py' -ForegroundColor Green

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path (Join-Path $Root 'Bindings.cpp'))) {
    Write-Host 'ERROR: BUILD_NATIVE.ps1 must be placed in the repository root.' -ForegroundColor Red
    exit 1
}

Set-Location $Root
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$CMake = Join-Path $Root '.venv\Scripts\cmake.exe'
$CTest = Join-Path $Root '.venv\Scripts\ctest.exe'
$VenvScripts = Split-Path $Python -Parent
$env:Path = "$VenvScripts;$env:Path"

if (-not (Test-Path $Python)) {
    throw 'Missing .venv\Scripts\python.exe. Create the Python 3.12 environment described in README.md first.'
}

& $Python -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version"
Write-Host '[PASS] CPython 3.12 environment found.' -ForegroundColor Green

Write-Host 'Installing/checking native build requirements...'
& $Python -m pip install -r requirements-build.txt
if ($LASTEXITCODE -ne 0) {
    throw 'Installing native build requirements failed.'
}

foreach ($Tool in @($CMake, $CTest)) {
    if (-not (Test-Path $Tool)) {
        throw "Required build tool is missing from .venv: $Tool"
    }
}

$Candidates = @()
if ($env:VCPKG_ROOT) {
    $Candidates += $env:VCPKG_ROOT
}
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

$VsWhereCandidates = @()
if (${env:ProgramFiles(x86)}) {
    $VsWhereCandidates += (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe')
}
if ($env:ProgramFiles) {
    $VsWhereCandidates += (Join-Path $env:ProgramFiles 'Microsoft Visual Studio\Installer\vswhere.exe')
}

$VsWhere = $VsWhereCandidates |
    Where-Object { Test-Path $_ } |
    Select-Object -First 1
if (-not $VsWhere) {
    throw 'Visual Studio Installer\vswhere.exe was not found. Install Visual Studio Build Tools with Desktop development with C++.'
}

$VsInstall = & $VsWhere `
    -latest `
    -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath |
    Select-Object -First 1
if ($VsInstall) {
    $VsInstall = $VsInstall.Trim()
}
if (-not $VsInstall) {
    throw 'A Visual Studio installation with the x64 C++ toolchain was not found.'
}

$VsDevCmd = Join-Path $VsInstall 'Common7\Tools\VsDevCmd.bat'
if (-not (Test-Path $VsDevCmd)) {
    throw "Visual Studio developer environment script was not found: $VsDevCmd"
}

Write-Host "Activating MSVC x64 environment: $VsInstall"
$EnvironmentCommand = "call `"$VsDevCmd`" -no_logo -arch=x64 -host_arch=x64 >nul && set"
$EnvironmentLines = & $env:ComSpec /d /s /c $EnvironmentCommand
if ($LASTEXITCODE -ne 0) {
    throw 'Visual Studio x64 developer environment initialization failed.'
}
foreach ($Line in $EnvironmentLines) {
    $Separator = $Line.IndexOf('=')
    if ($Separator -gt 0) {
        $Name = $Line.Substring(0, $Separator)
        $Value = $Line.Substring($Separator + 1)
        [Environment]::SetEnvironmentVariable($Name, $Value, 'Process')
    }
}

$Compiler = (Get-Command cl.exe -ErrorAction Stop).Source
$Linker = (Get-Command link.exe -ErrorAction Stop).Source
if ($Compiler -notmatch 'Microsoft Visual Studio' -or $Linker -notmatch 'Microsoft Visual Studio') {
    throw "MSVC activation did not select the Visual Studio compiler and linker. Compiler=$Compiler Linker=$Linker"
}
Write-Host "[PASS] MSVC compiler: $Compiler" -ForegroundColor Green
Write-Host "[PASS] MSVC linker:   $Linker" -ForegroundColor Green

$Toolchain = Join-Path $VcpkgRoot 'scripts\buildsystems\vcpkg.cmake'
$Build = Join-Path $Root 'build-native'
if (Test-Path $Build) {
    Remove-Item $Build -Recurse -Force
}

Write-Host 'Configuring the native extension with Ninja and MSVC...'
& $CMake -S . -B $Build -G Ninja `
    '-DCMAKE_BUILD_TYPE=Release' `
    "-DCMAKE_C_COMPILER=$Compiler" `
    "-DCMAKE_CXX_COMPILER=$Compiler" `
    "-DCMAKE_TOOLCHAIN_FILE=$Toolchain" `
    '-DVCPKG_TARGET_TRIPLET=x64-windows-static-md' `
    '-DMARKET_BUILD_TESTS=ON' `
    '-DMARKET_BUILD_BENCHMARKS=ON'
if ($LASTEXITCODE -ne 0) {
    throw 'CMake configuration failed.'
}

Write-Host 'Building Release...'
& $CMake --build $Build --parallel
if ($LASTEXITCODE -ne 0) {
    throw 'Native build failed.'
}

Write-Host 'Running native tests...'
& $CTest --test-dir $Build --output-on-failure
if ($LASTEXITCODE -ne 0) {
    throw 'Native tests failed.'
}

$BuiltPyd = Get-ChildItem $Build -Filter 'quant_engine*.pyd' -File -Recurse |
    Where-Object { $_.FullName -notlike '*\vcpkg_installed\*' } |
    Select-Object -First 1
if (-not $BuiltPyd) {
    throw 'Build completed but quant_engine*.pyd was not found in build-native.'
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
if ($LASTEXITCODE -ne 0) {
    throw 'Native binding verification failed.'
}

Write-Host 'Running full workstation verification...'
& $Python verify_workstation.py
if ($LASTEXITCODE -ne 0) {
    throw 'Full workstation verification failed.'
}

Write-Host ''
Write-Host 'SUCCESS: The native module is built and installed.' -ForegroundColor Green
Write-Host 'You can now run: .\.venv\Scripts\python.exe market_workstation.py' -ForegroundColor Green

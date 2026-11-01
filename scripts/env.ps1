# Build environment for Aegean TGE
#
# Dot-source this from a PowerShell session before configuring or building:
#   . .\scripts\env.ps1
#
# The toolchain is installed per user by winget, so it lives under
# %LOCALAPPDATA%\Microsoft\WinGet\Packages rather than on PATH.

$packages = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages'

$mingwRoot = Get-ChildItem $packages -Directory -Filter 'BrechtSanders.WinLibs.POSIX.UCRT_*' -ErrorAction SilentlyContinue |
  Select-Object -First 1
$cmakeRoot = Get-ChildItem $packages -Directory -Filter 'Kitware.CMake_*' -ErrorAction SilentlyContinue |
  Select-Object -First 1

if (-not $mingwRoot) {
  throw 'WinLibs GCC not found. Install it with: winget install BrechtSanders.WinLibs.POSIX.UCRT'
}

$mingwBin = Join-Path $mingwRoot.FullName 'mingw64\bin'
if (-not (Test-Path $mingwBin)) {
  throw "WinLibs bin directory missing: $mingwBin"
}

$paths = @($mingwBin)

if ($cmakeRoot) {
  $cmakeBin = Get-ChildItem $cmakeRoot.FullName -Directory -Filter 'cmake-*' -ErrorAction SilentlyContinue |
    Select-Object -First 1
  if ($cmakeBin) {
    $paths += (Join-Path $cmakeBin.FullName 'bin')
  }
}

$env:PATH = ($paths -join ';') + ';' + $env:PATH

Write-Host "aegean-tge toolchain:" -ForegroundColor Cyan
g++ --version | Select-Object -First 1
cmake --version | Select-Object -First 1
ninja --version

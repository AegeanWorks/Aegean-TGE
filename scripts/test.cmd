@echo off
rem Canonical regression gate for Aegean TGE.
rem Configures, builds and runs the test suite with the project toolchain.
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  ". '%~dp0env.ps1'; cmake --preset release; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; cmake --build --preset release; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; ctest --preset release; exit $LASTEXITCODE"

exit /b %ERRORLEVEL%

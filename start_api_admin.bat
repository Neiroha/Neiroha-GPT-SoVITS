@echo off
cd /d "%~dp0"
set PY=.pixi\envs\default\python.exe

where pixi >nul 2>nul
if errorlevel 1 (
  echo Pixi is required. Install Pixi, then rerun this script.
  exit /b 1
)

if not exist "%PY%" (
  pixi install
  if errorlevel 1 exit /b 1
)

pixi run serve %*

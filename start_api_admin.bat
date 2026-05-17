@echo off
cd /d "%~dp0"
set PY=.pixi\envs\default\python.exe

if not exist "%PY%" (
  pixi install
)

"%PY%" -B scripts\launch_gpt_sovits.py --mode api-admin-preload %*

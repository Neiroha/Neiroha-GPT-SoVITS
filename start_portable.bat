@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "ENV_DIR=%ROOT%.pixi\envs\default"
set "PY=%ENV_DIR%\python.exe"
if not exist "%PY%" (
  echo [ERROR] Bundled Python not found:
  echo   %PY%
  echo This portable package is incomplete.
  pause
  exit /b 1
)

if not exist "%ROOT%runtime\cache" mkdir "%ROOT%runtime\cache"
if not exist "%ROOT%runtime\logs" mkdir "%ROOT%runtime\logs"
if not exist "%ROOT%runtime\outputs" mkdir "%ROOT%runtime\outputs"
if not exist "%ROOT%runtime\temp" mkdir "%ROOT%runtime\temp"

set "PATH=%ENV_DIR%;%ENV_DIR%\Scripts;%ENV_DIR%\Library\bin;%ENV_DIR%\Library\usr\bin;%ROOT%GPT-SoVITS;%PATH%"
set "PYTHONPATH=%ROOT%;%ROOT%GPT-SoVITS"
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "TOKENIZERS_PARALLELISM=false"
set "TEMP=%ROOT%runtime\temp"
set "TMP=%ROOT%runtime\temp"
set "TMPDIR=%ROOT%runtime\temp"
set "GRADIO_TEMP_DIR=%ROOT%runtime\temp\gradio"
set "MODELSCOPE_CACHE=%ROOT%runtime\cache\modelscope"
set "HF_HOME=%ROOT%runtime\cache\huggingface"

if /I "%~1"=="serve" goto serve
if /I "%~1"=="api-admin" goto serve
if /I "%~1"=="api" goto api
if /I "%~1"=="admin" goto admin
if /I "%~1"=="webui" goto webui
if /I "%~1"=="help" goto help
if /I "%~1"=="--help" goto help

:menu
echo.
echo Neiroha GPT-SoVITS Portable
echo ===========================
echo 1. API + Admin
echo 2. API only
echo 3. Admin only
echo 4. Official GPT-SoVITS WebUI
echo 5. Help
echo.
set /p "choice=Select [1-5]: "
if "%choice%"=="1" goto serve
if "%choice%"=="2" goto api
if "%choice%"=="3" goto admin
if "%choice%"=="4" goto webui
if "%choice%"=="5" goto help
echo Invalid choice.
goto menu

:serve
"%PY%" "%ROOT%scripts\launch_engine.py" --surface both
exit /b %ERRORLEVEL%

:api
"%PY%" "%ROOT%scripts\launch_engine.py" --surface api
exit /b %ERRORLEVEL%

:admin
"%PY%" "%ROOT%scripts\launch_engine.py" --surface admin
exit /b %ERRORLEVEL%

:webui
"%PY%" "%ROOT%scripts\launch_gpt_sovits.py" --mode webui
exit /b %ERRORLEVEL%

:help
echo.
echo Usage:
echo   start_portable.bat              Show menu
echo   start_portable.bat serve        Start API + Admin
echo   start_portable.bat api          Start API only
echo   start_portable.bat admin        Start Admin only
echo   start_portable.bat webui        Start official GPT-SoVITS WebUI
echo.
echo The launcher uses only files under this unpacked directory.
exit /b 0

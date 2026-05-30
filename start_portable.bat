@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "ENV_DIR=%ROOT%.pixi\envs\default"
set "PY=%ENV_DIR%\python.exe"
if exist "%PY%" goto python_ok
echo [ERROR] Bundled Python not found:
echo   %PY%
echo This portable package is incomplete. Please unpack all archive parts again.
pause
exit /b 1

:python_ok

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
call :say title
echo =========================
call :say menu_serve
call :say menu_api
call :say menu_admin
call :say menu_webui
call :say menu_help
echo.
call :say prompt nonewline
set /p "choice="
if "%choice%"=="1" goto serve
if "%choice%"=="2" goto api
if "%choice%"=="3" goto admin
if "%choice%"=="4" goto webui
if "%choice%"=="5" goto help
call :say invalid
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
call :say usage
call :say help_menu
call :say help_serve
call :say help_api
call :say help_admin
call :say help_webui
echo.
call :say note
exit /b 0

:say
"%PY%" -X utf8 -c "import sys; messages={'title':'Neiroha GPT-SoVITS \u4fbf\u643a\u7248','menu_serve':'1. \u542f\u52a8 API + Admin','menu_api':'2. \u4ec5\u542f\u52a8 API','menu_admin':'3. \u4ec5\u542f\u52a8 Admin','menu_webui':'4. \u542f\u52a8 GPT-SoVITS \u5b98\u65b9 WebUI','menu_help':'5. \u5e2e\u52a9','prompt':'\u8bf7\u9009\u62e9 [1-5]\uff1a','invalid':'\u65e0\u6548\u9009\u62e9\uff0c\u8bf7\u91cd\u65b0\u8f93\u5165\u3002','usage':'\u7528\u6cd5\uff1a','help_menu':'  start_portable.bat              \u663e\u793a\u83dc\u5355','help_serve':'  start_portable.bat serve        \u542f\u52a8 API + Admin','help_api':'  start_portable.bat api          \u4ec5\u542f\u52a8 API','help_admin':'  start_portable.bat admin        \u4ec5\u542f\u52a8 Admin','help_webui':'  start_portable.bat webui        \u542f\u52a8 GPT-SoVITS \u5b98\u65b9 WebUI','note':'\u542f\u52a8\u5668\u53ea\u4f7f\u7528\u5f53\u524d\u89e3\u538b\u76ee\u5f55\u5185\u7684\u6587\u4ef6\u3002'}; text=messages[sys.argv[1]]; end='' if len(sys.argv)>2 and sys.argv[2]=='nonewline' else '\n'; sys.stdout.write(text+end); sys.stdout.flush()" "%~1" "%~2"
exit /b %ERRORLEVEL%

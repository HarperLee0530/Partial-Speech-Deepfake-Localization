@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if exist ".venv\Scripts\python.exe" goto install
py -3.10 -m venv ".venv"
if errorlevel 1 (
  echo Install Python 3.10 with the Windows Python launcher, then retry.
  exit /b 1
)
:install
".venv\Scripts\python.exe" -m ensurepip --upgrade
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r "requirements-lock.txt" --extra-index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" "tools/check_environment.py" --environment-only
exit /b %errorlevel%

@echo off
setlocal
cd /d "%~dp0"
py -3.10 -m venv .venv
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m unittest discover -s tests -v
exit /b %errorlevel%

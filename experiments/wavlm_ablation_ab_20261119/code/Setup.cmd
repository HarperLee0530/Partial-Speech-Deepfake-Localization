@echo off
setlocal
cd /d "%~dp0"
rem Reuse a working project environment, including one created from Conda.
rem The py launcher may not know that interpreter, and recreating it is unnecessary.
if exist ".venv\Scripts\python.exe" goto install
py -3.10 -m venv .venv
if errorlevel 1 exit /b 1
:install
.venv\Scripts\python.exe -m pip install -r requirements-wavlm.txt
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m unittest discover -s tests -v
exit /b %errorlevel%

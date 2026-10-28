@echo off
setlocal
cd /d "%~dp0"
python "%~dp0scripts\abrir_dashboard.py" --stop
if errorlevel 1 pause

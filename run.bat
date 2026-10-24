@echo off
rem Starts the Local Voice Transcriber. Extra arguments are passed on, for example: run.bat --latency 160
cd /d "%~dp0"
".\.venv\Scripts\python.exe" run.py %*
if errorlevel 1 pause

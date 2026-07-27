@echo off
set BACKEND=%1
if "%BACKEND%"=="" set BACKEND=cpu
powershell -ExecutionPolicy Bypass -File "%~dp0setup_venv.ps1" -TorchBackend %BACKEND%

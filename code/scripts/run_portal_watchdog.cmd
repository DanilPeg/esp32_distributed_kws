@echo off
set ROOT=%~dp0..\..
pushd "%ROOT%" >nul
python code\scripts\portal_watchdog.py
popd >nul

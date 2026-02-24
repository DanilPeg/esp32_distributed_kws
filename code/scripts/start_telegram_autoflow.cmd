@echo off
set REPO=C:\Users\Danil\diploma_esp32_distributed_nn

start "telegram-autoflow" /min powershell -NoProfile -STA -ExecutionPolicy Bypass -File "%REPO%\code\scripts\telegram_autoflow.ps1"

# Hardware Ops Playbook (ESP32)

Purpose: a repeatable, low-risk procedure for autonomous hardware work.

## Scope & Limits
- Safe actions: read serial logs, build sketches, flash known examples, collect metrics.
- Unsafe actions (require explicit approval): erase flash, change fuses, mass-update firmware, install drivers.

## Pre-flight Checklist
1) Identify board type (ESP32-S3 vs ESP32-C3) and the connected COM port.
2) Ensure Windows session is active (no lock screen) if serial monitoring is needed.
3) Confirm that flashing is allowed for this step.

## Detect COM Port (Windows)
- Registry: HKLM\HARDWARE\DEVICEMAP\SERIALCOMM
- Typical USB-serial looks like \Device\USBSER000 -> COMx.

## Read Boot Log (no flashing)
Use PowerShell to open the port at 115200 and capture output.
- Save raw log to: 
otes/Research/esp32_<date>_bootlog_raw.txt
- Write summary to: 
otes/Research/esp32_<date>_bootlog_summary.md
- Add a Journal entry with: board, port, IDF version, flash size mismatch, project name, warnings.

## Chip Info (optional)
If esptool is installed:
- python -m esptool --port COMx chip_id
- python -m esptool --port COMx flash_id

If not installed: request permission to install esptool.

## Baseline micro_speech (without mic)
Use the repo copy: code/firmware/micro_speech_sim/
- This version has USE_FAKE_MIC=1 in udio_provider.cpp.
- It simulates audio (zeros) and increments timestamp so inference runs.
- Result: serial log shows inference loop without physical microphone.

## Baseline micro_speech (with mic)
- Set USE_FAKE_MIC=0 in udio_provider.cpp.
- Ensure I2S pins match wiring (default example uses BCK=26, WS=32, DATA_IN=33).
- Flash with Arduino IDE and capture serial logs.

## Logging Protocol
Every hardware interaction must produce:
1) Raw logs in 
otes/Research/
2) A concise summary file in 
otes/Research/
3) A Journal entry (
otes/Journal/*.yaml) with:
   - date, board, port, firmware, actions, results, warnings, next steps.

## Troubleshooting
- No serial output: check correct COM port, baud rate 115200, board powered.
- window_not_found or missing logs: ensure the correct window title and relay is running.
- Flash mismatch warning: note it in summary; verify actual flash size later.

## Safety Rules
- Do not erase flash without explicit permission.
- Do not change fuses.
- Always capture logs before and after flashing.
**Summary**
Reviewed `code/scripts/` for automation, safety, idempotency, and logging. Main risks are hardcoded paths (repo + python) and non-idempotent startup scripts that can spawn duplicate processes or wipe logs. Applied a small fix to keep Telegram bridge errors readable in English.

**Findings**
- Hardcoded absolute paths for repo and Python reduce portability and break if the repo is moved or Python is upgraded. Affects `code/scripts/run_agent_loop.cmd`, `code/scripts/run_agent_loop.ps1`, `code/scripts/start_codex_session.ps1`, `code/scripts/start_telegram_autoflow.cmd`, `code/scripts/start_telegram_autoflow.ps1`, `code/scripts/stop_telegram_autoflow.ps1`, `code/scripts/telegram_autoflow.ps1`, `code/scripts/telegram_autosend.ps1`, `code/scripts/telegram_simple_relay.ps1`.
- `code/scripts/start_telegram_autoflow.ps1` clears existing logs and starts new bridge/relay/autosend processes without checking for existing instances. Re-running can spawn duplicates and lose previous logs (non-idempotent behavior).
- `code/scripts/portal_watchdog.py` spawns a new uvicorn whenever `http_ok` fails and does not track/guard existing processes. If the port is blocked or the process is hung, repeated spawns can pile up.
- `code/scripts/portal_watchdog.py` uses `taskkill /F ngrok.exe`, which can terminate unrelated ngrok tunnels on the machine.
- `code/scripts/telegram_bridge.py` had a non-English error message for repo-bound file path checks, which can confuse automation logs and violates the “English-only” guidance. This was fixed.

**Actions**
- Updated `code/scripts/telegram_bridge.py` to use an English error message for out-of-repo file path validation.

**Tests**
- Not run (review + small edit only).

**Next**
1. Replace hardcoded repo paths with script-relative resolution (`$PSScriptRoot` + `Resolve-Path ..\..`) and add a `PYTHON` env override / `Get-Command python` fallback in the PowerShell and CMD launchers.
2. Make Telegram autoflow startup idempotent by checking `daemon_pids.json` (and `Get-Process`) before spawning; add a `--force` flag or separate cleanup command instead of clearing logs by default.
3. Track uvicorn/ngrok PIDs in `portal_watchdog.py` and only restart when the existing process is absent or unresponsive; avoid global `taskkill` by storing PID or using ngrok’s API for targeted shutdown.
4. Consider basic log rotation or size caps for long-running telemetry logs under `notes/Journal/telegram/`.

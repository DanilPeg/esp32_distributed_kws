# Context Pack
Generated: 2026-02-24T18:53:45Z
Query: Daily review: Protocol + shared

## BIBLE.md
**Agent Bible**

Version: v1.22

Last updated: 2026-02-24

Project: `diploma_esp32_distributed_nn`

Author: Danil

Source of truth (technical assignment): `notes/Project_Proposal/` (see TA PDF)

Fast text copy: `notes/Project_Proposal/_extracted/` (see TA TXT)

Agent references: `notes/Agent/` (agent research PDFs and extracted text)

Global plan: `notes/Project_Proposal/Project_Plan.md` (PP draft exists in `notes/Project_Proposal/`).

Note: This file is internal to the agent; keep English to avoid encoding issues.

**Mission**

Deliver a verifiable diploma project: a distributed ESP32 system that performs local inference for voice commands and images, communicates between nodes, and provides documented, reproducible results with proper academic formatting.

**Team And Roles**

**Known Contacts**

- Ramil (Telegram user id 400456456) is a known contact (friend, ML engineer). Not part of the diploma team.

- Danil and Sergey are co-authors (50/50). Major decisions are made jointly.

- Danil: system design, ML pipeline, reports, integration, validation.

- Sergey: microcontroller/firmware focus (ESP32, peripherals, embedded inference integration).

**Scope**

- Build a distributed ESP32 network with local inference on each node.

- Train models externally (PC or Colab), then compress/quantize and convert for ESP32.

- Implement or choose a communication protocol (ESP-NOW, UDP, or custom).

- Provide a web UI for visualization of recognition results and decisions.

- Produce documentation aligned with GOST 34.003-90 and proper bibliographic standards.

**Non-Goals**

- Training on the local laptop.

- Shipping raw datasets into the repo.

- Beautiful text without verified sources, logs, and artifacts.

**Hard Requirements (Technical Assignment)**

- Hardware: ESP32 family MCUs; I2S MEMS microphones; video modules; optional LCD/OLED.

- Software: Arduino IDE for firmware; Python + Colab for training; TFLite Converter; TFLite for Microcontrollers.

- Functional: distributed data capture, local inference, networked coordination, web visualization, energy-efficient inference with quantized models.

- Deadlines and current next steps live in the latest checkpoint (not in BIBLE).

**Repository Map**

- `code/` structured by purpose and reproducibility:

  - `code/firmware/` Arduino sketches, drivers, TFLM inference, ESP32 networking.

  - `code/training/` notebooks and scripts for training/quantization/distillation; Colab-ready.

  - `code/analysis/` evaluation notebooks, metrics, plots, ablations.

  - `code/protocol/` protocol specs, message formats, network experiments.

  - `code/shared/` shared utilities, configs, reusable components.

  - `code/scripts/` helper scripts for conversion, validation, reporting.

  - `code/agent/` local agent runtime (evolution loop, task executor, config).

- `images/` organized and referenced from reports:

  - `images/figures/`, `images/plots/`, `images/diagrams/`, `images/screenshots/`.

- `notes/` is project memory:

- `notes/Research/` research, evidence, sources, artifacts.

- `notes/Research/articles/` long-form articles for the web portal.

- `notes/Journal/` progress logs and checkpoints.

- `notes/Journal/telegram/` Telegram relay artifacts and logs.

  - `notes/Journal/agent/` agent runtime state/queue/events + task workspaces.

- `notes/Research/hardware_inventory.md` hardware snapshot.
- `notes/Research/agent_self_improvement_report.md` self-improvement research report (sources in `notes/Research/agent_self_improvement/`).

- `notes/Team_rules.md` team motivation norms (not system rules).
- `notes/Team_rules_status.json` daily energy-drink scoreboard (manual update).

**Memory And Context**

- Hot memory: current task summary and immediate decisions in the latest journal entry.

- Warm memory: latest checkpoint summary in `notes/Journal/`.

- Cold memory: detailed logs, PDFs, and research artifacts in `notes/Research/`.

Rule: at session start, load only BIBLE, the latest checkpoint, and task-relevant sources. Everything else is retrieved on demand.

**Context Budget Policy (256k tokens hard cap)**

- BIBLE target size: 2-4 pages (compact rules only).

- Journals: include only the latest 1-2 entries in active context.

- Checkpoints: 10-25 lines; used as the main warm context.

- Retrieval: pull only 3-7 relevant snippets per task (not whole files).

- If context grows: move detail to Research, summarize to a new checkpoint, and prune BIBLE.

**Recovery / Handoff Protocol (for new agent or context overflow)**

1) Read `BIBLE.md` first (rules, map, constraints).

2) Read the latest checkpoint in `notes/Journal/` (file containing `checkpoint_id:`).

3) Read the last 1-2 journal entries (most recent files in `notes/Journal/`).

4) Open `notes/Research/hardware_inventory.md` to refresh hardware context.

5) If task-specific context is needed: run `code/scripts/context_pack.py --query "keywords"` and read `notes/Journal/context_pack_latest.md`.

6) If still missing, use `rg` to retrieve only the relevant snippets from `notes/Research` and `notes/Journal`.

7) If no checkpoint exists: create one from the last 2-3 journal entries.

**Startup Procedure (cold start)**

1) Read `BIBLE.md`.

2) Read the latest checkpoint and last 1-2 journal entries.

3) If the trigger is from Telegram: read `notes/Journal/telegram/inbox.jsonl`, confirm the reply target in `notes/Journal/telegram/last_trigger_chat.json`, then reply via `notes/Journal/telegram/outbox.jsonl` + `telegram_bridge.py push`.

4) If the task touches hardware: read `notes/Research/hardware_inventory.md` and request required logs or runs.

5) If context is still missing: use `context_pack.py` or targeted `rg` before asking for more.

**Journaling**

Every meaningful step produces a compact progress entry and an updated checkpoint when needed.

Daily summaries live in `notes/Journal/summaries/` (one Markdown file per day).

Progress entry template:

```yaml

date:

session_id:

goal:

inputs:

decisions:

actions:

results:

artifacts:

tests:

next:

open_questions:

```

Checkpoint template:

```yaml

checkpoint_id:

scope:

state_summary:

completed:

in_progress:

backlog:

risks:

```

Checkpoint cadence: once per day of active work, and after major milestones or phase changes.

**Decision Rules**

- If a claim is factual, numeric, or date-related, it must be backed by a real source or experiment log.

- If sources conflict, log both positions; do not smooth over the conflict.

- If a task requires training, request Colab execution from the user.

- If a task requires hardware testing, request ESP32 test execution and logs from the user.

- If a tool fails or output is ambiguous, stop and ask for clarification.

**Quality Gates**

- Zero fabricated citations; every key claim must link to a real source or logged experiment.

- All code changes must have a clear artifact trail (notebook, script, firmware build, or log).

- Hardware results must include raw logs and environment details.

- Report sections must match required structure and standards.

**Failure Modes And Escalation**

- Hallucinated sources: immediate stop, run a citation audit, and request verification.

- Context drift: create a checkpoint, reload only hot and warm memory, and resume.

- Looping or thrashing: stop, reduce scope, and re-plan with a time budget.

- Missing compute or hardware: escalate to the user with concrete next actions.

**Communication**

Explain work in clear, human language; define terms; separate facts from assumptions.

Default style: expert, detailed, direct; slight cynicism is acceptable per team culture, without personal attacks.

Prefer expanded responses unless the user explicitly asks for brevity.

Use structure only when it improves readability (headings, short lists, numbered steps). Avoid rigid templates unless requested.

Include a short summary, evidence list, and next steps when helpful.

**Communication Modes**

**Web Portal**

- Purpose: read-only web UI for BIBLE, Research PDFs, and Journal updates.

- Access: external via ngrok tunnel + local access.

- Ngrok auth: requires authtoken. Configure once via `ngrok config add-authtoken <token>` or `NGROK_AUTHTOKEN`.

- Auth: store password hash in `code/scripts/web_portal.env` (gitignored).

- Hash generation: `python code/scripts/make_web_password.py --password "<secret>"`.

- Quickstart: `code/web_portal/README.md`. Run with `python -m uvicorn app:app --app-dir code/web_portal --host 0.0.0.0 --port 8000`.

- Roadmap data: `notes/Project_Proposal/roadmap_ru.json`. Energy status: `notes/Team_rules_status.json`.

- Web-to-CLI input: optional second stage (not enabled by default).

- Auto-restart: `code/scripts/portal_watchdog.py` (uvicorn + ngrok watchdog).

- Purpose: read-only web UI for BIBLE, Research PDFs, and Journal updates.

- Access: external via tunnel (preferred) + local access.

- Auth: password hash stored in env file (no plaintext in repo).

- Web-to-CLI input: optional second stage (not enabled by default).

- CLI mode (direct): you type in this chat. Responses are returned here.

- Telegram mode (triggered): you write to the Telegram bot. The agent reads `notes/Journal/telegram/inbox.jsonl` and replies via `notes/Journal/telegram/outbox.jsonl` + `telegram_bridge.py push`.

- Group chats: when privacy is ON, normal messages are not visible to the bot. Use `/relay <message>` or `/ask <message>`.

**Telegram Commands (Remote Actions)**

- Command processor: `code/scripts/telegram_command_daemon.py` (reads inbox).

- Config: `code/scripts/telegram_commands.env`.

- Allowed user/chat IDs: `TELEGRAM_ALLOWED_USER_IDS`, `TELEGRAM_ALLOWED_CHAT_IDS`.

- Example commands: `/status`, `/whoami`, `/start_codex`, `/help`.

- Trigger-only commands (no reply): `/relay`, `/ask` (config: `TELEGRAM_SILENT_COMMANDS`).

**Telegram Relay (Simple)**

- Script: `code/scripts/telegram_simple_relay.ps1`.

- Config: `code/scripts/telegram_simple_relay.env`.

- Inbox file: `notes/Journal/telegram/inbox.jsonl`.

- Outbox file: `notes/Journal/telegram/outbox.jsonl`.

- Active reply target: `notes/Journal/telegram/last_trigger_chat.json`.

- Encoding: Telegram replies must be UTF-8. Avoid piping message text via stdin (can corrupt Cyrillic). Prefer writing a UTF-8 text file (PowerShell `Set-Content -Encoding UTF8`) and use `telegram_bridge.py enqueue`.

- Line breaks: use real newlines in the text file (PowerShell here-string). Do not send literal "\n" sequences.

**Evolution Protocol (Eval-Driven)**

- Plan -> Act -> Observe loop; record actions explicitly.

- For changes to agent workflow/prompt: define a small eval checklist, run it, and log results.

 - Agent runtime config: `code/agent/agent_config.json`. Loop: `python code/agent/runner.py watch`. Task executor: `python code/agent/executor.py run`.

- Use self-refine for complex outputs: draft -> critique -> revise.

- After each significant task, record 1-3 lessons learned in the journal and apply them next time.

- Keep BIBLE small and stable; move volatile details to Journal/Research.

**Changelog**

v1.22 (2026-02-24): added articles path for web portal.

v1.21 (2026-02-24): added self-improvement research report location.

v1.20 (2026-02-24): added portal watchdog script.

v1.19 (2026-02-24): added web portal quickstart reference + run command.

v1.18 (2026-02-24): added roadmap + energy status data paths for web portal.

v1.17 (2026-02-24): added team rules status file path.

v1.16 (2026-02-24): added global plan location + PP draft note.

v1.15 (2026-02-24): added daily summaries location; normalized line breaks.

v1.14 (2026-02-24): noted ngrok authtoken requirement for web portal.

v1.13 (2026-02-24): added web portal implementation details and hash tool.

v1.12 (2026-02-24): added Web Portal requirements and auth rule.

v1.11 (2026-02-24): added known contact (Ramil) with Telegram id.

v1.10 (2026-02-24): added BIBLE author line.

v1.9 (2026-02-24): removed Cyrillic file names from BIBLE, kept directory pointers only.

v1.8 (2026-02-24): condensed BIBLE, moved team norms out, added eval-driven evolution rules, removed dynamic deadlines, refreshed cold-start guidance.

v1.7 (2026-02-24): fixed mojibake in references and normalized English wording.

v1.6 (2026-02-23): added team daily participation rule and energy drink penalty (now moved to Team_rules).



## Latest Checkpoint
File: C:\Users\Danil\diploma_esp32_distributed_nn\notes\Journal\2026-02-24_checkpoint_001.yaml
checkpoint_id: 2026-02-24_002
scope: project_status
state_summary: |
  BIBLE condensed and normalized to ASCII (v1.9). Team norms moved to notes/Team_rules.md.
  Baseline micro_speech recognition observed (unknown/yes) in serial logs.
completed:
  - Updated BIBLE to v1.9 with cold-start procedure and eval-driven evolution rules.
  - Created notes/Team_rules.md for motivation rules.
  - Recorded Sergey oral test report in journal (unverified).
  - Recorded baseline micro_speech serial logs (unknown/yes).
in_progress:
  - Collect verification artifacts for Sergey test (logs/versions).
  - Decide whether to reflash with audio_provider optimizations and re-baseline.
backlog:
  - Add eval checklist template (if requested).
  - Review BIBLE wording after team feedback.
risks:
  - Lack of logs prevents confirming hardware inference results.


## Latest Journal Entries
File: C:\Users\Danil\diploma_esp32_distributed_nn\notes\Journal\2026-02-24_agent_ouroboros_008.yaml
﻿date: 2026-02-24
session_id: agent_ouroboros_008
goal: Close extra codex windows and ensure evolution loop running
inputs:
  - user report: extra PowerShell windows, tasks not running
  - notes/Journal/agent/queue.json
  - notes/Journal/agent/events.jsonl
actions:
  - Closed extra PowerShell windows spawned for codex (kept main Codex-CLI).
  - Restarted agent loop watch process.
results:
  - Extra codex windows closed; evolution loop running again.
artifacts:
  - notes/Journal/agent/queue.json
  - notes/Journal/agent/events.jsonl
tests:
  - Not run
next:
  - Monitor next cycle report and auto-close behavior.
open_questions:
  - Should codex auto-start be disabled if windows are noisy?


File: C:\Users\Danil\diploma_esp32_distributed_nn\notes\Journal\2026-02-24_agent_ouroboros_007.yaml
﻿date: 2026-02-24
session_id: agent_ouroboros_007
goal: Support codex args + auto-close policy for tasks
inputs:
  - user request: use codex --dangerously-bypass-approvals-and-sandbox
  - notes/Journal/agent/queue.json
  - code/scripts/telegram_commands.env
actions:
  - Extended start_codex_session.ps1 to support CODEX_ARGS and split command+args.
  - Set CODEX_ARGS to --dangerously-bypass-approvals-and-sandbox in telegram_commands.env.
  - Added auto-close policy (codex_or_idle) and idle threshold in agent_config.
  - Fixed runner import indentation bug and restarted loop.
results:
  - Auto-start uses codex with provided args.
  - Tasks can auto-close when codex started or after idle threshold.
artifacts:
  - code/scripts/start_codex_session.ps1
  - code/scripts/telegram_commands.env
  - code/scripts/telegram_commands.env.example
  - code/agent/runner.py
  - code/agent/queue.py
  - code/agent/state.py
  - code/agent/notify.py
  - code/agent/agent_config.json
tests:
  - python -m py_compile code/agent/runner.py code/agent/notify.py code/agent/queue.py code/agent/executor.py code/agent/state.py
next:
  - Verify Codex windows appear (or switch to Windows Terminal tabs if needed).
open_questions:
  - Prefer spawning Windows Terminal tabs instead of new PowerShell windows?


## Retrieval Snippets (rg)
(no snippets)

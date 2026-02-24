# Agent Ops (local)

Local files used by the agent runtime.

Files:
- state.json: current runtime state (enabled flags, timestamps, counters).
- queue.json: task queue (pending/running/completed).
- events.jsonl: event log (append-only).
- health.json: last health-check snapshot.
- tasks/: per-task workspace (task.md, prompt.txt, context_pack.md, report.md, codex logs).

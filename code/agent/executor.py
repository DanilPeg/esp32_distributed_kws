from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

try:
    from .events import append_event
    from .paths import REPO_ROOT, TASKS_DIR
    from .queue import load_queue, mark_done, pop_next_task
    from .state import load_state, save_state
    from .utils import atomic_write_text, save_json, utc_now_iso
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from agent.events import append_event
    from agent.paths import REPO_ROOT, TASKS_DIR
    from agent.queue import load_queue, mark_done, pop_next_task
    from agent.state import load_state, save_state
    from agent.utils import atomic_write_text, save_json, utc_now_iso


def _sanitize_query(text: str) -> str:
    return " ".join((text or "").split())


def _run_git(args: List[str]) -> tuple[int, str]:
    proc = subprocess.run(
        ["git"] + args,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.strip()


def prepare_git_branch(task_id: str, config: Dict[str, Any]) -> str:
    mode = str(config.get("git_branch_mode", "off")).strip().lower()
    if mode in ("", "off", "false", "0", "none"):
        return ""
    if not (REPO_ROOT / ".git").exists():
        append_event("git_branch_skipped", {"task_id": task_id, "reason": "no_git"})
        return ""
    prefix = str(config.get("git_branch_prefix", "agent/task")).strip() or "agent/task"
    branch = f"{prefix}-{task_id}"
    code, out = _run_git(["rev-parse", "--verify", branch])
    if code != 0:
        code, out = _run_git(["checkout", "-b", branch])
        if code != 0:
            append_event("git_branch_error", {"task_id": task_id, "detail": out})
            return ""
        append_event("git_branch_created", {"task_id": task_id, "branch": branch})
    else:
        code, out = _run_git(["checkout", branch])
        if code != 0:
            append_event("git_branch_error", {"task_id": task_id, "detail": out})
            return ""
        append_event("git_branch_checked_out", {"task_id": task_id, "branch": branch})
    return branch


def task_dir(task_id: str) -> Path:
    return TASKS_DIR / task_id


def build_prompt(task: Dict[str, Any], prep: Dict[str, Any]) -> str:
    workspace = prep.get("workspace", "")
    context_pack = prep.get("context_pack_path", "")
    report_path = str(Path(workspace) / "report.md") if workspace else ""
    branch = prep.get("branch", "")
    lines = [
        "You are running a non-interactive Codex task.",
        f"Repo: {REPO_ROOT}",
        f"Task id: {task.get('id')}",
        f"Title: {task.get('title')}",
        "",
        "Description:",
        (task.get("description") or "").strip() or "(none)",
        "",
        "Workspace files:",
        f"- task.md: {workspace}\\task.md" if workspace else "- task.md: (unknown)",
    ]
    if context_pack:
        lines.append(f"- context_pack.md: {context_pack}")
    else:
        lines.append("- context_pack.md: (missing)")
    if branch:
        lines.append(f"- git branch: {branch} (already checked out)")
    lines.extend(
        [
            "",
            "Instructions:",
            "1) Read task.md and context_pack.md (if present).",
            "2) Focus on the paths described in the task.",
            "3) Produce a report in report.md with sections: Summary, Findings, Actions, Tests, Next.",
            "4) Code edits are allowed; keep them minimal and list them in the report.",
            "5) Do not merge branches. Do not edit BIBLE.md unless necessary for the task.",
            "6) Use UTF-8 markdown.",
            f"Report path: {report_path}",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def write_task_markdown(task: Dict[str, Any]) -> str:
    lines = [
        f"# Task {task.get('id')}",
        "",
        f"Created: {task.get('created_at')}",
        f"Started: {task.get('started_at')}",
        f"Type: {task.get('type')}",
        f"Title: {task.get('title')}",
        "",
        "## Description",
        task.get("description", "").strip() or "(none)",
        "",
        "## Notes",
        "- Add progress updates in the journal.",
        "- Attach artifacts/paths here as you work.",
    ]
    return "\n".join(lines).strip() + "\n"


def run_context_pack(query: str, out_path: Path, config: Dict[str, Any]) -> tuple[int, str]:
    script = REPO_ROOT / "code" / "scripts" / "context_pack.py"
    if not script.exists():
        return 1, f"missing script: {script}"
    cmd = [
        "python",
        str(script),
        "--query",
        query,
        "--out",
        str(out_path),
        "--journal-count",
        str(config.get("context_pack_journal_count", 2)),
        "--max-snippets",
        str(config.get("context_pack_max_snippets", 7)),
        "--context-lines",
        str(config.get("context_pack_context_lines", 2)),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.strip()


def start_codex_session(
    config: Dict[str, Any],
    prompt_file: str | None = None,
    task_id: str | None = None,
    out_file: str | None = None,
    log_file: str | None = None,
) -> tuple[int, str]:
    script = config.get("start_codex_script")
    if not script:
        script = str(REPO_ROOT / "code" / "scripts" / "start_codex_session.ps1")
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        script,
    ]
    if prompt_file:
        cmd += ["-PromptFile", prompt_file]
    if task_id:
        cmd += ["-TaskId", task_id]
    if out_file:
        cmd += ["-OutFile", out_file]
    if log_file:
        cmd += ["-LogFile", log_file]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.strip()


def prepare_task(task: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    tdir = task_dir(task["id"])
    tdir.mkdir(parents=True, exist_ok=True)
    branch = prepare_git_branch(task["id"], config)
    save_json(tdir / "task.json", task)
    atomic_write_text(tdir / "task.md", write_task_markdown(task))

    query = config.get("context_pack_query")
    if not query:
        query = task.get("title") or task.get("type") or "agent task"
    query = _sanitize_query(query)

    context_pack_ok = False
    context_pack_path = ""
    if config.get("context_pack_enabled", True):
        out_path = tdir / "context_pack.md"
        code, out = run_context_pack(query, out_path, config)
        if code == 0:
            append_event("context_pack_written", {"task_id": task["id"], "path": str(out_path)})
            context_pack_ok = True
            context_pack_path = str(out_path)
        else:
            append_event("context_pack_failed", {"task_id": task["id"], "detail": out})

    prompt_text = build_prompt(task, {
        "workspace": str(tdir),
        "context_pack_path": context_pack_path,
        "branch": branch,
    })
    prompt_path = tdir / "prompt.txt"
    atomic_write_text(prompt_path, prompt_text)

    append_event("task_prepared", {"task_id": task["id"], "path": str(tdir)})
    if branch:
        task["branch"] = branch
    save_json(tdir / "task.json", task)
    return {
        "workspace": str(tdir),
        "task_md": str(tdir / "task.md"),
        "prompt_path": str(prompt_path),
        "context_pack_ok": context_pack_ok,
        "context_pack_path": context_pack_path,
        "branch": branch,
    }


def process_next_task(config: Dict[str, Any]) -> Dict[str, Any] | None:
    task = pop_next_task()
    if not task:
        return None
    prep = prepare_task(task, config)
    actions = ["workspace", "task.md"]
    if prep.get("context_pack_ok"):
        actions.append("context_pack")
        task["context_pack_path"] = prep.get("context_pack_path", "")
    task["prompt_path"] = prep.get("prompt_path", "")
    report_path = str(Path(prep.get("workspace", task_dir(task["id"]))) / "report.md")
    task["report_path"] = report_path
    if config.get("auto_start_codex", False) and task.get("needs_agent", False):
        out_file = str(Path(prep.get("workspace", task_dir(task["id"]))) / "codex_last.md")
        log_file = str(Path(prep.get("workspace", task_dir(task["id"]))) / "codex_exec.log")
        task["codex_out_path"] = out_file
        task["codex_log_path"] = log_file
        code, out = start_codex_session(
            config,
            prompt_file=prep.get("prompt_path"),
            task_id=task.get("id"),
            out_file=out_file,
            log_file=log_file,
        )
        status = "started" if code == 0 else "error"
        if "limit_reached" in (out or ""):
            status = "limit_reached"
        task["codex_start"] = {"status": status, "detail": out}
        actions.append(f"codex_start={status}")
        append_event("codex_start", {"task_id": task["id"], "code": code, "detail": out})
    task["actions"] = actions
    if config.get("auto_mark_done", False):
        finished = mark_done(task["id"], status="prepared")
        if finished:
            task["finished_at"] = finished.get("finished_at")
        state = load_state()
        state["completed_tasks"] = int(state.get("completed_tasks", 0)) + 1
        save_state(state)
    return task


def process_queue(max_tasks: int, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    processed: List[Dict[str, Any]] = []
    for _ in range(max_tasks):
        task = process_next_task(config)
        if not task:
            break
        processed.append(task)
    return processed


def complete_task(task_id: str, status: str = "done") -> None:
    mark_done(task_id, status=status)
    state = load_state()
    state["completed_tasks"] = int(state.get("completed_tasks", 0)) + 1
    save_state(state)
    append_event("task_completed", {"task_id": task_id, "status": status})


def main() -> int:
    parser = argparse.ArgumentParser(description="Process agent tasks from queue.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Prepare tasks and optionally start a session")
    run.add_argument("--max-tasks", type=int, default=1)
    done = sub.add_parser("complete", help="Mark a running task as complete")
    done.add_argument("task_id")
    done.add_argument("--status", default="done")
    args = parser.parse_args()

    config_path = REPO_ROOT / "code" / "agent" / "agent_config.json"
    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            config = {}

    if args.cmd == "run":
        processed = process_queue(max_tasks=int(args.max_tasks), config=config)
        return 0 if processed else 1
    complete_task(args.task_id, status=args.status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

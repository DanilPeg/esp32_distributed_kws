# Summary
Reviewed `code/web_portal/app.py`, templates, and CSS with focus on rendering, performance, search, and materials.

# Findings
- Search runs a full synchronous `os.walk` over the repo and reads file contents on every `/search` request, which will block the event loop and slow the portal as the repo grows.
- Home page status reads the entire `notes/Journal/agent/events.jsonl` into memory each request to find the last line; this will degrade as the log grows.
- Materials browser links are built without URL encoding (`browse.html` concatenates `current_path`/`f`), so filenames with spaces or non-ASCII may break navigation or resolve incorrectly.
- Top nav has no responsive handling; on mobile widths the link list will overflow the header and can become unreadable/unclickable.

# Actions
- Added a short journal entry for this review in `notes/Journal/2026-02-24_agent_codex_d62e4428.yaml`.

# Tests
- Not run (review only).

# Next
1. Move search to a background thread (`asyncio.to_thread`/`run_in_threadpool`) or pre-index with `rg` and cache results; consider scoping to `notes/` + `code/` for faster queries.
2. Tail `events.jsonl` instead of reading the full file (seek from end, keep last N lines) before rendering the home page.
3. Encode browse/view links with URL-encoded paths (precompute `url_path` in Python or add a template filter).
4. Add responsive nav behavior (wrap, horizontal scroll, or a collapsed menu) for smaller viewports.

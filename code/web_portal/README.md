# Web Portal (read-only)

## What it shows
- BIBLE (rendered Markdown)
- Documents from `notes/Research/`, `notes/Project_Proposal/`, `notes/Agent/`
- Journal updates (latest checkpoint + recent journal files)
- Daily summaries (`notes/Journal/summaries/*.md`)
  - View at `/summary`
- Materials browser for repo files
- Roadmap (RU) from `notes/Project_Proposal/roadmap_ru.json`
- Search across repo materials
- Articles from `notes/Research/articles/`

## Requirements
- Python 3.10+
- Packages: fastapi, uvicorn, jinja2, markdown, python-multipart

Install:
```
pip install fastapi uvicorn jinja2 markdown python-multipart
```

## Configure password
Generate a hash:
```
python code/scripts/make_web_password.py --password "<secret-word>"
```
Save it to `code/scripts/web_portal.env`:
```
WEB_PORTAL_PASSWORD_HASH=pbkdf2_sha256$...
```

## Run locally
```
python -m uvicorn app:app --app-dir code/web_portal --host 0.0.0.0 --port 8000
```

## External access (ngrok)
```
ngrok http 8000
```
Use the HTTPS URL provided by ngrok.

## PDF inline preview
PDFs are opened via `/view?path=<file>` and rendered in an inline iframe.

## Materials browser
`/browse` provides a read-only file browser and `/view?path=...` shows images/PDFs/code with syntax highlighting.

## Roadmap + Search
- `/roadmap` renders `notes/Project_Proposal/roadmap_ru.json`
- `/search` searches file names and (small) text files

## Auto-restart (watchdog)
Use the watchdog to restart uvicorn/ngrok after network drops:
```
python code/scripts/portal_watchdog.py
```
Optional (Windows): schedule `code/scripts/run_portal_watchdog.cmd` at logon.

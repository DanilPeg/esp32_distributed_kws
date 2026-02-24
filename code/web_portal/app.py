import os
import time
import hmac
import hashlib
import secrets
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import quote

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

try:
    import markdown as md
except Exception:
    md = None

ROOT = Path(__file__).resolve().parents[2]
NOTES_DIR = ROOT / 'notes'
BIBLE_PATH = ROOT / 'BIBLE.md'
RESEARCH_DIR = NOTES_DIR / 'Research'
PROJECT_PROPOSAL_DIR = NOTES_DIR / 'Project_Proposal'
AGENT_DIR = NOTES_DIR / 'Agent'
JOURNAL_DIR = NOTES_DIR / 'Journal'
SUMMARIES_DIR = JOURNAL_DIR / 'summaries'
PLAN_PATH = PROJECT_PROPOSAL_DIR / 'Project_Plan.md'
ROADMAP_PATH = PROJECT_PROPOSAL_DIR / 'roadmap_ru.json'
TEAM_RULES_STATUS_PATH = NOTES_DIR / 'Team_rules_status.json'
ARTICLES_DIR = NOTES_DIR / 'Research' / 'articles'
ARTICLES_INDEX = ARTICLES_DIR / 'index.json'
ENV_PATH = ROOT / 'code' / 'scripts' / 'web_portal.env'
AGENT_RUNTIME_DIR = JOURNAL_DIR / 'agent'
AGENT_STATE_PATH = AGENT_RUNTIME_DIR / 'state.json'
AGENT_QUEUE_PATH = AGENT_RUNTIME_DIR / 'queue.json'
AGENT_EVENTS_PATH = AGENT_RUNTIME_DIR / 'events.jsonl'
AGENT_CONFIG_PATH = ROOT / 'code' / 'agent' / 'agent_config.json'

TEMPLATES_DIR = Path(__file__).resolve().parent / 'templates'
STATIC_DIR = Path(__file__).resolve().parent / 'static'

jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(['html'])
)

SESSION_TTL_SEC = 24 * 3600
_sessions: Dict[str, float] = {}
ALLOWED_ROOTS = [ROOT]
IGNORED_NAMES = {
    '.git', '.codex', '.idea', '.vscode', '__pycache__',
    '.pytest_cache', '.mypy_cache', 'node_modules',
}
DENY_SUFFIXES = {'.env'}
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp'}
TEXT_EXTS = {
    '.py', '.cpp', '.c', '.cc', '.h', '.hpp', '.ino', '.puml',
    '.md', '.txt', '.yaml', '.yml', '.json', '.toml', '.ini',
    '.ps1', '.bat', '.sh', '.html', '.css', '.js', '.ts',
    '.csv', '.log',
}
LANG_MAP = {
    '.py': 'python',
    '.cpp': 'cpp',
    '.c': 'c',
    '.cc': 'cpp',
    '.h': 'c',
    '.hpp': 'cpp',
    '.ino': 'cpp',
    '.md': 'markdown',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.json': 'json',
    '.toml': 'toml',
    '.ini': 'ini',
    '.ps1': 'powershell',
    '.sh': 'bash',
    '.bat': 'bat',
    '.html': 'html',
    '.css': 'css',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.puml': 'plaintext',
    '.txt': 'plaintext',
    '.csv': 'plaintext',
    '.log': 'plaintext',
}
MAX_TEXT_BYTES = 400_000
MAX_SEARCH_TEXT_BYTES = 200_000


def load_env() -> dict:
    cfg = {}
    if ENV_PATH.exists():
        for raw in ENV_PATH.read_text(encoding='utf-8').splitlines():
            line = raw.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, val = line.split('=', 1)
            key = key.strip().lstrip('\ufeff')
            cfg[key] = val.strip().strip('"').strip("'")
    for k, v in os.environ.items():
        if v is not None:
            cfg[k] = v
    return cfg


def parse_hash(s: str):
    # format: pbkdf2_sha256$<iters>$<salt_hex>$<hash_hex>
    parts = s.split('$')
    if len(parts) != 4:
        return None
    algo, iters, salt_hex, hash_hex = parts
    if algo != 'pbkdf2_sha256':
        return None
    try:
        iters = int(iters)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except Exception:
        return None
    return iters, salt, expected


def verify_password(pw: str, stored: str) -> bool:
    parsed = parse_hash(stored)
    if not parsed:
        return False
    iters, salt, expected = parsed
    derived = hashlib.pbkdf2_hmac('sha256', pw.encode('utf-8'), salt, iters)
    return hmac.compare_digest(derived, expected)


def create_session() -> str:
    token = secrets.token_urlsafe(32)
    _sessions[token] = time.time() + SESSION_TTL_SEC
    return token


def is_session_valid(token: str) -> bool:
    exp = _sessions.get(token)
    if not exp:
        return False
    if time.time() > exp:
        _sessions.pop(token, None)
        return False
    return True


def render_markdown(text: str) -> str:
    if md is None:
        return '<pre>' + text + '</pre>'
    return md.markdown(text, extensions=['fenced_code', 'tables'])


def file_excerpt(path: Path, max_lines: int = 20) -> str:
    try:
        lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
    except Exception:
        return ''
    return '\n'.join(lines[:max_lines])


def list_journals(limit: int = 8):
    files = [p for p in JOURNAL_DIR.glob('*.yaml') if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    checkpoints = [p for p in files if 'checkpoint' in p.name]
    latest_checkpoint = checkpoints[0] if checkpoints else None
    journals = [p for p in files if 'checkpoint' not in p.name][:limit]
    return latest_checkpoint, journals


def list_summaries(limit: int = 7):
    if not SUMMARIES_DIR.exists():
        return []
    files = [p for p in SUMMARIES_DIR.glob('*.md') if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    entries = []
    for p in files[:limit]:
        text = p.read_text(encoding='utf-8', errors='ignore')
        entries.append({
            'name': p.name,
            'mtime': datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
            'content_html': render_markdown(text),
        })
    return entries


def resolve_safe_path(rel: str) -> Optional[Path]:
    if not rel:
        return None
    rel = rel.lstrip('/').replace('\\', '/')
    if ':' in rel:
        return None
    target = (ROOT / rel).resolve()
    if not target.exists():
        return None
    if target.name in IGNORED_NAMES or target.suffix.lower() in DENY_SUFFIXES:
        return None
    for part in target.parts:
        if part in IGNORED_NAMES:
            return None
    for root in ALLOWED_ROOTS:
        if target == root or root in target.parents:
            return target
    return None


def list_dir(path: Path):
    dirs = []
    files = []
    for p in path.iterdir():
        name = p.name
        if name in IGNORED_NAMES or name.startswith('.'):
            continue
        try:
            if p.is_dir():
                dirs.append(name)
            elif p.is_file():
                if p.suffix.lower() in DENY_SUFFIXES:
                    continue
                files.append(name)
        except Exception:
            continue
    dirs.sort(key=str.lower)
    files.sort(key=str.lower)
    return dirs, files


def load_roadmap():
    if not ROADMAP_PATH.exists():
        return None
    try:
        data = json.loads(ROADMAP_PATH.read_text(encoding='utf-8'))
    except Exception:
        return {'error': 'Failed to parse roadmap JSON.'}
    return data


def load_team_rules_status():
    default = {
        'updated': '',
        'status': 'pending',
        'status_line': 'Оценка не задана.',
        'message': 'Ждем решения судьи. Энергосы в тени.',
        'score': {'danil': 0, 'sergey': 0},
    }
    if not TEAM_RULES_STATUS_PATH.exists():
        return default
    try:
        data = json.loads(TEAM_RULES_STATUS_PATH.read_text(encoding='utf-8'))
    except Exception:
        return default
    status = data.get('status', default['status'])
    status_map = {
        'danil_owes': 'Даня должен Серёже.',
        'sergey_owes': 'Серёжа должен Дане.',
        'both_owe': 'Оба провинились. Энергосы летят.',
        'clean': 'Долгов нет. Пока.',
        'pending': 'Оценка не задана.',
    }
    score = data.get('score', default['score'])
    if not isinstance(score, dict):
        score = default['score']
    data['status_line'] = status_map.get(status, default['status_line'])
    data['score_line'] = f"Даня {score.get('danil', 0)} : {score.get('sergey', 0)} Серёжа"
    return {**default, **data, 'score': score}


def load_agent_status():
    status = {
        'enabled': False,
        'pending': 0,
        'running': 0,
        'last_evolution_at': '',
        'last_health_check_at': '',
        'last_event': '',
    }
    config_enabled = None
    if AGENT_CONFIG_PATH.exists():
        try:
            cfg = json.loads(AGENT_CONFIG_PATH.read_text(encoding='utf-8'))
            config_enabled = bool(cfg.get('evolution_enabled'))
        except Exception:
            config_enabled = None
    if AGENT_STATE_PATH.exists():
        try:
            data = json.loads(AGENT_STATE_PATH.read_text(encoding='utf-8'))
            status['enabled'] = bool(data.get('evolution_enabled'))
            status['last_evolution_at'] = data.get('last_evolution_at', '')
            status['last_health_check_at'] = data.get('last_health_check_at', '')
        except Exception:
            pass
    if config_enabled is not None:
        status['enabled'] = status['enabled'] or config_enabled
    if AGENT_QUEUE_PATH.exists():
        try:
            q = json.loads(AGENT_QUEUE_PATH.read_text(encoding='utf-8'))
            status['pending'] = len(q.get('pending', []))
            status['running'] = len(q.get('running', []))
        except Exception:
            pass
    if AGENT_EVENTS_PATH.exists():
        try:
            lines = AGENT_EVENTS_PATH.read_text(encoding='utf-8').splitlines()
            for raw in reversed(lines):
                if raw.strip():
                    status['last_event'] = raw.strip()
                    break
        except Exception:
            pass
    return status


def search_materials(query: str, limit: int = 200):
    q = query.strip().lower()
    if not q:
        return []
    results = []
    for root in ALLOWED_ROOTS:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in IGNORED_NAMES and not d.startswith('.')]
            for fn in filenames:
                if fn in IGNORED_NAMES or fn.startswith('.'):
                    continue
                p = Path(dirpath) / fn
                if p.suffix.lower() in DENY_SUFFIXES:
                    continue
                rel = p.relative_to(ROOT).as_posix()
                url_path = quote(rel)
                hay = (fn + ' ' + rel).lower()
                matched = False
                reason = ''
                snippet = ''
                if q in hay:
                    matched = True
                    reason = 'name'
                elif p.suffix.lower() in TEXT_EXTS:
                    try:
                        data = p.read_bytes()
                        if len(data) > MAX_SEARCH_TEXT_BYTES:
                            data = data[:MAX_SEARCH_TEXT_BYTES]
                        text = data.decode('utf-8', errors='ignore')
                        idx = text.lower().find(q)
                        if idx >= 0:
                            matched = True
                            reason = 'content'
                            start = max(0, idx - 60)
                            end = min(len(text), idx + 120)
                            snippet = text[start:end].replace('\n', ' ')
                    except Exception:
                        matched = False
                if matched:
                    results.append({
                        'name': fn,
                        'rel_path': rel,
                        'url_path': url_path,
                        'reason': reason,
                        'snippet': snippet,
                    })
                    if len(results) >= limit:
                        return results
    return results


def content_disposition_inline(name: str) -> str:
    try:
        name.encode('ascii')
        return f'inline; filename=\"{name}\"'
    except UnicodeEncodeError:
        return "inline; filename*=UTF-8''" + quote(name)


def load_articles_index():
    if not ARTICLES_INDEX.exists():
        return []
    try:
        raw = ARTICLES_INDEX.read_text(encoding='utf-8').lstrip('\ufeff')
        data = json.loads(raw)
    except Exception:
        return []
    if isinstance(data, dict):
        return data.get('articles', [])
    if isinstance(data, list):
        return data
    return []


def get_article(slug: str):
    path = ARTICLES_DIR / f"{slug}.md"
    if not path.exists():
        return None
    text = path.read_text(encoding='utf-8', errors='ignore')
    return render_markdown(text)


def collect_docs():
    sources = [
        ('Research', RESEARCH_DIR),
        ('Project Proposal', PROJECT_PROPOSAL_DIR),
        ('Agent', AGENT_DIR),
    ]
    allowed = {'.pdf', '.docx'}
    entries = []
    for label, base in sources:
        if not base.exists():
            continue
        for p in base.rglob('*'):
            if not p.is_file():
                continue
            if p.suffix.lower() not in allowed:
                continue
            stat = p.stat()
            rel = p.relative_to(ROOT).as_posix()
            url_path = quote(rel)
            entries.append({
                'name': p.name,
                'rel_path': rel,
                'url_path': url_path,
                'category': label,
                'ext': p.suffix.lower(),
                'size_kb': int(stat.st_size / 1024),
                'mtime': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
            })
    entries.sort(key=lambda e: (e['category'], e['name'].lower()))
    return entries


def latest_project_proposal():
    if not PROJECT_PROPOSAL_DIR.exists():
        return None
    docs = [p for p in PROJECT_PROPOSAL_DIR.glob('*.docx') if p.is_file()]
    if not docs:
        return None
    latest = max(docs, key=lambda p: p.stat().st_mtime)
    rel = latest.relative_to(ROOT).as_posix()
    return {
        'name': latest.name,
        'mtime': datetime.fromtimestamp(latest.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
        'url_path': quote(rel),
    }


app = FastAPI()
app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')


@app.middleware('http')
async def auth_middleware(request: Request, call_next):
    cfg = load_env()
    password_hash = cfg.get('WEB_PORTAL_PASSWORD_HASH', '').strip()
    auth_enabled = bool(password_hash)

    if request.url.path.startswith('/static') or request.url.path.startswith('/login'):
        return await call_next(request)
    if not auth_enabled:
        return await call_next(request)

    token = request.cookies.get('web_portal_session', '')
    if not token or not is_session_valid(token):
        return RedirectResponse('/login', status_code=303)

    return await call_next(request)


@app.get('/login', response_class=HTMLResponse)
async def login_get(request: Request):
    tpl = jinja_env.get_template('login.html')
    return tpl.render(error=None)


@app.post('/login')
async def login_post(request: Request, password: str = Form(...)):
    cfg = load_env()
    password_hash = cfg.get('WEB_PORTAL_PASSWORD_HASH', '').strip()
    if not password_hash:
        tpl = jinja_env.get_template('login.html')
        return HTMLResponse(tpl.render(error='Password hash not configured.'), status_code=500)

    if not verify_password(password, password_hash):
        tpl = jinja_env.get_template('login.html')
        return HTMLResponse(tpl.render(error='Invalid password.'), status_code=401)

    token = create_session()
    resp = RedirectResponse('/', status_code=303)
    resp.set_cookie('web_portal_session', token, httponly=True, samesite='lax')
    return resp


@app.get('/', response_class=HTMLResponse)
async def index():
    pdfs = sorted([p for p in RESEARCH_DIR.glob('*.pdf') if p.is_file()])
    latest_checkpoint, journals = list_journals(limit=6)
    plan_excerpt = file_excerpt(PLAN_PATH, max_lines=12) if PLAN_PATH.exists() else ''
    latest_pp = latest_project_proposal()
    roadmap = load_roadmap()
    team_rules = load_team_rules_status()
    articles = load_articles_index()
    agent_status = load_agent_status()
    tpl = jinja_env.get_template('index.html')
    return tpl.render(
        bible_exists=BIBLE_PATH.exists(),
        pdf_count=len(pdfs),
        journal_count=len(journals),
        latest_checkpoint=latest_checkpoint.name if latest_checkpoint else None,
        plan_excerpt=plan_excerpt,
        latest_pp=latest_pp,
        roadmap=roadmap,
        team_rules=team_rules,
        articles=articles[:3],
        agent_status=agent_status,
    )


@app.get('/bible', response_class=HTMLResponse)
async def bible():
    text = BIBLE_PATH.read_text(encoding='utf-8', errors='ignore') if BIBLE_PATH.exists() else ''
    html = render_markdown(text)
    tpl = jinja_env.get_template('bible.html')
    return tpl.render(content=html)


@app.get('/research', response_class=HTMLResponse)
async def research():
    entries = collect_docs()
    tpl = jinja_env.get_template('research.html')
    return tpl.render(entries=entries)


@app.get('/browse', response_class=HTMLResponse)
async def browse(path: str = ''):
    if not path:
        if len(ALLOWED_ROOTS) == 1 and ALLOWED_ROOTS[0] == ROOT:
            dirs, files = list_dir(ROOT)
            tpl = jinja_env.get_template('browse.html')
            return tpl.render(
                current_path='',
                roots=[],
                dirs=dirs,
                files=files,
            )
        roots = []
        for r in ALLOWED_ROOTS:
            if r.exists():
                rel = r.relative_to(ROOT).as_posix()
                roots.append(rel)
        tpl = jinja_env.get_template('browse.html')
        return tpl.render(
            current_path='',
            roots=roots,
            dirs=[],
            files=[],
        )
    safe = resolve_safe_path(path)
    if not safe or not safe.is_dir():
        return RedirectResponse('/browse', status_code=303)
    rel = safe.relative_to(ROOT).as_posix()
    dirs, files = list_dir(safe)
    tpl = jinja_env.get_template('browse.html')
    return tpl.render(
        current_path=rel,
        roots=[],
        dirs=dirs,
        files=files,
    )


@app.get('/view', response_class=HTMLResponse)
async def view(path: str):
    safe = resolve_safe_path(path)
    if not safe or not safe.is_file():
        return RedirectResponse('/browse', status_code=303)
    suffix = safe.suffix.lower()
    kind = 'file'
    content = ''
    language = 'plaintext'
    if suffix in IMAGE_EXTS:
        kind = 'image'
    elif suffix == '.pdf':
        kind = 'pdf'
    elif suffix in TEXT_EXTS:
        kind = 'text'
        data = safe.read_bytes()
        if len(data) > MAX_TEXT_BYTES:
            data = data[:MAX_TEXT_BYTES]
            content = data.decode('utf-8', errors='ignore') + '\n\n[truncated]'
        else:
            content = data.decode('utf-8', errors='ignore')
        language = LANG_MAP.get(suffix, suffix.lstrip('.'))
    tpl = jinja_env.get_template('view.html')
    rel = safe.relative_to(ROOT).as_posix()
    url_path = quote(rel)
    return tpl.render(
        name=safe.name,
        rel_path=rel,
        url_path=url_path,
        kind=kind,
        content=content,
        language=language,
    )


@app.get('/file')
async def file(path: str):
    safe = resolve_safe_path(path)
    if not safe or not safe.is_file():
        return RedirectResponse('/browse', status_code=303)
    headers = {'Content-Disposition': content_disposition_inline(safe.name)}
    media_type = 'application/octet-stream'
    suffix = safe.suffix.lower()
    if suffix == '.pdf':
        media_type = 'application/pdf'
    elif suffix in IMAGE_EXTS:
        if suffix == '.svg':
            media_type = 'image/svg+xml'
        elif suffix in ('.jpg', '.jpeg'):
            media_type = 'image/jpeg'
        else:
            media_type = f"image/{suffix.lstrip('.')}"
    return FileResponse(str(safe), media_type=media_type, headers=headers)


@app.get('/research/file/{name}')
async def research_file(name: str):
    safe = (RESEARCH_DIR / name).resolve()
    if RESEARCH_DIR not in safe.parents or safe.suffix.lower() != '.pdf' or not safe.exists():
        return RedirectResponse('/research', status_code=303)
    headers = {'Content-Disposition': content_disposition_inline(safe.name)}
    return FileResponse(str(safe), media_type='application/pdf', headers=headers)


@app.get('/research/view/{name}', response_class=HTMLResponse)
async def research_view(name: str):
    safe = (RESEARCH_DIR / name).resolve()
    if RESEARCH_DIR not in safe.parents or safe.suffix.lower() != '.pdf' or not safe.exists():
        return RedirectResponse('/research', status_code=303)
    tpl = jinja_env.get_template('research_view.html')
    return tpl.render(name=safe.name)


@app.get('/journal', response_class=HTMLResponse)
async def journal():
    latest_checkpoint, journals = list_journals(limit=8)
    checkpoint_data = None
    if latest_checkpoint:
        checkpoint_data = {
            'name': latest_checkpoint.name,
            'mtime': datetime.fromtimestamp(latest_checkpoint.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
            'excerpt': file_excerpt(latest_checkpoint, max_lines=30)
        }
    entries = []
    for p in journals:
        entries.append({
            'name': p.name,
            'mtime': datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
            'excerpt': file_excerpt(p, max_lines=20)
        })
    tpl = jinja_env.get_template('journal.html')
    return tpl.render(checkpoint=checkpoint_data, entries=entries)


@app.get('/summary', response_class=HTMLResponse)
async def summary():
    summaries = list_summaries(limit=30)
    tpl = jinja_env.get_template('summary.html')
    return tpl.render(summaries=summaries)


@app.get('/roadmap', response_class=HTMLResponse)
async def roadmap():
    data = load_roadmap()
    tpl = jinja_env.get_template('roadmap.html')
    return tpl.render(data=data)


@app.get('/search', response_class=HTMLResponse)
async def search(q: str = ''):
    results = search_materials(q) if q else []
    tpl = jinja_env.get_template('search.html')
    return tpl.render(query=q, results=results, count=len(results))


@app.get('/articles', response_class=HTMLResponse)
async def articles():
    items = load_articles_index()
    tpl = jinja_env.get_template('articles.html')
    return tpl.render(items=items)


@app.get('/articles/{slug}', response_class=HTMLResponse)
async def article(slug: str):
    items = load_articles_index()
    article_meta = next((a for a in items if a.get('slug') == slug), None)
    content = get_article(slug)
    if content is None:
        return RedirectResponse('/articles', status_code=303)
    tpl = jinja_env.get_template('article.html')
    return tpl.render(meta=article_meta, content=content)

**Telegram Bridge (File-Based, Low-Real-Time)**

**What It Does**
This bridge uses Telegram long-polling to fetch messages into a local inbox file and send responses from a local outbox file. It is intentionally simple and not real-time.

**Files**
- Inbox: `notes/Journal/telegram/inbox.jsonl` (append-only)
- Outbox: `notes/Journal/telegram/outbox.jsonl` (append-only)
- State: `notes/Journal/telegram/state.json` (last update id and last sent line)

**Install / Setup**
1. Create a Telegram bot with BotFather:
   1. Open Telegram and message `@BotFather`
   2. Send `/newbot`
   3. Set a bot name and username
   4. Copy the bot token
2. Create `code/scripts/telegram.env` (or set env vars):
```env
TELEGRAM_BOT_TOKEN=123456:ABCDEF...
TELEGRAM_DEFAULT_CHAT_ID=123456789
TELEGRAM_DEFAULT_SESSION_ID=2026-02-23T17-35-01Z
```

**How To Get `TELEGRAM_DEFAULT_CHAT_ID`**
1. Send any message to your bot (e.g. "hi")
2. Run:
```powershell
python code/scripts/telegram_bridge.py pull
```
3. Open `notes/Journal/telegram/inbox.jsonl` and read `chat_id`.

**Usage**
Fetch new messages:
```powershell
python code/scripts/telegram_bridge.py pull
```

Send queued replies:
```powershell
python code/scripts/telegram_bridge.py push
```

Do both:
```powershell
python code/scripts/telegram_bridge.py sync
```

**Outbox Format**
Append one JSON per line to `notes/Journal/telegram/outbox.jsonl`:
```json
{"chat_id": 123456789, "text": "Сделано: ...", "session_id": "2026-02-23T17-35-01Z"}
```
If `chat_id` is omitted, the bridge uses `TELEGRAM_DEFAULT_CHAT_ID`.

**Session ID**
If you want messages to map to a CLI session, include `session_id` in your Telegram message:
`session_id: 2026-02-23T17-35-01Z`

**Notes**
- The bridge stores state locally to avoid double-processing.
- It does not require any external Python packages.

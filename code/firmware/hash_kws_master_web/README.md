# hash_kws_master_web — Master ESP32 with embedded dashboard

Dedicated 4th ESP32 board (no microphone).
Receives ESP-NOW packets from 3 inference nodes, aggregates logits,
and serves a live web dashboard at **http://micro_network.local** (mDNS)
and at the raw IP printed on Serial after boot.

---

## Files in this sketch folder

| File | Role |
|---|---|
| `hash_kws_master_web.ino` | Main sketch |
| `web_page.h` | Embedded dashboard HTML/CSS/JS (PROGMEM) |
| `hash_ensemble_aggregator.h` | C++ aggregator — **copy from** `code/firmware/hash_kws_aggregator/` |
| `hash_ensemble_aggregator.cpp` | C++ aggregator — **copy from** `code/firmware/hash_kws_aggregator/` |
| `aggregator_params.h` | Calibrated T and weights — **copy from** `code/deploy/hash_ensemble/reports/` |

**Before opening Arduino IDE**, copy the three files into this folder:

```powershell
$src = "..\..\hash_kws_aggregator"
$par = "..\..\..\deploy\hash_ensemble\reports"
Copy-Item "$src\hash_ensemble_aggregator.h"   .
Copy-Item "$src\hash_ensemble_aggregator.cpp"  .
Copy-Item "$par\aggregator_params.h"           .
```

---

## Required Arduino libraries

Install all three via **Tools → Manage Libraries**:

| Library | Author | Note |
|---|---|---|
| **ESP Async WebServer** | mathieucarbou | Must be the mathieucarbou fork — the `me-no-dev` version does NOT compile with ESP32 core 3.x |
| **AsyncTCP** | mathieucarbou | Companion to the above |
| **ArduinoJson** | bblanchon | Version ≥ 7.0 |

> If the Library Manager shows two "ESP Async WebServer" entries, pick the one
> with "mathieucarbou" as author.

---

## Board settings (Tools menu)

| Setting | Value |
|---|---|
| Board | **ESP32 Dev Module** (ESP32 WROOM-32) |
| CPU Frequency | 240 MHz |
| Flash Size | 4MB (standard WROOM-32) |
| Partition Scheme | Default 4MB with spiffs |
| Upload Speed | 921600 |

> Serial output is via UART0 through the onboard USB-serial chip (CP2102 / CH340).
> No USB CDC setting required — open the port at **115200 baud** in any terminal.
>
> The built-in blue LED (GPIO2) blinks briefly on each fusion event.

---

## WiFi configuration

Edit the two `#define` lines near the top of `hash_kws_master_web.ino`:

```cpp
#define WIFI_SSID     "YourSSID"
#define WIFI_PASSWORD "YourPassword"
```

**STA mode** (default): master connects to your 2.4 GHz router.
The ESP-NOW channel is locked to the router's channel.
Find your router's channel with:

```
netsh wlan show interfaces
```

Then recompile **all 4 boards** with the same channel:

```cpp
-DHASH_KWS_ESPNOW_CHANNEL=<channel>
```

**AP fallback**: if STA connect fails after 10 seconds, master creates its
own access point (`KWS-Master` / `kwsmaster1`). Inference nodes must be
compiled with `HASH_KWS_ESPNOW_CHANNEL=1` (the AP default). Connect your
laptop/phone to `KWS-Master` to open the dashboard.

---

## Aggregator mode

Set at compile time via `HASH_KWS_AGG_MODE` in the sketch (or as a build flag):

| Value | Mode | Notes |
|---|---|---|
| `0` | `mean_logits` | Default, recommended (no `exp()` on MCU) |
| `1` | `temperature_scaled` | Uses T from `aggregator_params.h` |
| `2` | `learned_weights` | Uses weights from `aggregator_params.h` |

All three give < 0.1 pp accuracy difference on the test set.

---

## What the dashboard shows

Open `http://micro_network.local` in any browser on the same network:

- **Connection badge** — green `● live` / red `○ disconnected` with 1-second auto-reconnect.
- **Counters strip** — `fusion`, `packets`, `rejected`, `aggregator`, `uptime`.
- **Node tiles (3×)** — per-node last label (large, coloured), score/margin/packet count.
  - Green left border = packet received within last 4 s.
  - Amber border = stale (> 4 s since last packet).
  - Dark border = never seen.
- **Inference latency** — per-node median / p95 / max `invoke_ms` from last 30 packets,
  with a colour-coded bar (green < 200 ms, amber 200–500 ms, red > 500 ms).
- **Fusion decisions list** — newest first, max 50 entries.
  Shows label, score, margin, voter count, and wall-clock time.

Multiple browser tabs/devices work simultaneously (WebSocket broadcast).

---

## Serial log

Master continues to print `hash_evt kind=fusion ...` lines as before,
so the host-side bridge (`hash_ensemble_master_bridge.py`) still works for
JSONL recording without any changes.

---

## Acceptance checklist

| Check | Expected |
|---|---|
| `hash_evt kind=wifi phase=sta_ok` in Serial | IP + channel printed |
| `hash_evt kind=mdns hostname=micro_network.local` in Serial | mDNS started |
| `http://micro_network.local` opens in browser | Dashboard loads |
| Speak a command into any inference node | Tile turns green, label appears, fusion list updates |
| 30 s of silence | Tiles turn amber |
| Reconnect WiFi (unplug/replug router) | Badge returns to `● live`, counters unchanged |
| Open on phone + laptop simultaneously | Both show identical state |

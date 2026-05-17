# ESP32 Distributed Hash-KWS — Run Guide

Полностью рабочая версия системы для ВКР. Три аудио-ноды + master-агрегатор + опциональная камера, два дашборда на выбор.

---

## 0. Что в репо

```
code/firmware/
  hash_kws_master_web/         master: ESP-NOW recv + fusion + WS/HTTP dashboard + camera trigger
  camera_classifier/           visual node: MCUNet VWW INT8, ESP-NOW triggered
  micro_speech_sim/micro_speech/   audio inference sketch (один и тот же для всех трёх нод)
  hash_kws_runtime/            «активный» bundle модели; перезаписывается select-скриптом
  hash_kws_runtime_ens_{a,b,c} три варианта обученных моделей ансамбля

code/scripts/
  select_hash_kws_variant.ps1            переключение варианта перед прошивкой
  hash_kws_master_demux_bridge.py        USB-мост master → JSONL
  reset_dashboard_streams.py             очистка JSONL перед демо (опционально)

code/dashboard/                          FastAPI-дашборд (USB-bridge)
run_dashboard.py                         запуск FastAPI
```

---

## 1. Железо

- **3× ESP32-S3 DevKitC (N16R8)** — аудио-ноды с INMP441 (I2S MEMS).
- **1× ESP32-S3 (N16R8)** — master-агрегатор. Без микрофона.
- **1× ESP32-S3 + OV5640** — визуальная нода (опционально).
- USB-кабель к master (для USB-bridge дашборда) или просто питание (для on-board дашборда).

---

## 2. Прошивка аудио-нод (×3)

Для каждой ноды:

```powershell
pwsh -File code\scripts\select_hash_kws_variant.ps1 -Variant ens_a
```

Меняй `ens_a` → `ens_b` → `ens_c` для второй и третьей ноды. Скрипт копирует модель варианта в `code/firmware/hash_kws_runtime/`, откуда её включает скетч.

Открой `code/firmware/micro_speech_sim/micro_speech/micro_speech.ino` в Arduino IDE.

**Tools-опции (всё обязательно):**

| Поле | Значение |
|---|---|
| Board | `ESP32S3 Dev Module` |
| PSRAM | `OPI PSRAM` |
| Flash Size | `16MB (128Mb)` |
| Flash Mode | `QIO 80MHz` |
| Partition Scheme | `Huge APP (3MB No OTA/1MB SPIFFS)` |
| USB CDC On Boot | `Enabled` |
| USB Mode | `Hardware CDC and JTAG` |
| Erase All Flash Before Sketch Upload | `Enabled` (рекомендуется при первой прошивке) |

Загрузить. На каждой ноде в Serial Monitor должны идти:
```
hash_evt kind=infer node=N t=... invoke_ms=~218 top1=... top1_score=... ...
```

> **Гoтча PSRAM cold-boot race на S3**: при холодном включении OPI-PSRAM может не успеть подняться (`octal_psram: chip is not connected`). 5–10 раз нажми RST. Если не помогает — конденсатор 10–22 µF на 3V3/GND.

---

## 3. Прошивка master

Открой `code/firmware/hash_kws_master_web/hash_kws_master_web.ino`. Те же Tools-опции что у аудио-нод.

**Дополнительно потребуются библиотеки** (Arduino IDE → Library Manager):
- `ESP Async WebServer` (mathieucarbou fork)
- `Async TCP` (mathieucarbou fork)
- `ArduinoJson` ≥ 7.0

WiFi-конфиг — в `.ino` через `#define`:
```cpp
#define HASH_KWS_WIFI_MODE      0     // 0 = AP fallback (рекомендуется), 1 = STA с роутером
#define HASH_KWS_ESPNOW_CHANNEL 1
#define HASH_KWS_AGG_MODE       0     // 0=mean_logits, 1=temperature_scaled, 2=learned_weights
```

В AP-mode master поднимает SoftAP `esp32-hash-master` / `12345678` на канале 1.

После прошивки в Serial master'а:
```
hash_evt kind=wifi phase=ap_fallback ssid=esp32-hash-master ip=192.168.4.1 channel=1 ok=1
hash_evt kind=espnow phase=init status=ok node=master mac=... broadcast_peer=1 cam_trigger_label=yes
hash_evt kind=infer node=N ...              # ретрансляция от аудио-нод
hash_evt kind=fusion node=master label=... voters=N ...
```

---

## 4. Прошивка камеры (опционально)

`code/firmware/camera_classifier/camera_classifier.ino`. Те же Tools-опции.

Камера автоматически подключится к мастерскому AP (`HASH_KWS_AP_SSID` / `HASH_KWS_AP_PASS` в коде — должны совпадать с мастером, дефолтно совпадают). Это **форсит правильный канал** через ассоциацию, никакой ручной настройки канала не нужно.

После прошивки в Serial камеры:
```
kind=boot status=start app=camera_classifier
kind=wifi phase=joined ssid=esp32-hash-master rssi=-XX channel=1 ip=192.168.4.X
kind=espnow phase=init status=ok ... channel_actual=1 joined=1 bcast=1
kind=heartbeat uptime_s=... triggers=N replies_ok=N ...
```

Trigger-слово — `yes` по умолчанию. Поменять: `#define HASH_KWS_CAMERA_TRIGGER_LABEL_IDX N` в master_web.ino (0=yes, 1=no, 2=up, 3=down, …) — реflash **только мастера**.

---

## 5. Дашборд — два варианта

### Вариант A: на самом мастере (HTTP/WS)

Уже работает, делать ничего не надо. Подключись к WiFi `esp32-hash-master` / `12345678` и открой:
- `http://192.168.4.1/`
- или `http://kws_master.local/` (mDNS)

Содержит: hero-карточку ансамблевого решения, 3 плитки нод с лейтенси-баром, fusion-таблицу, камера-карточку.

### Вариант B: на хосте через USB-мост (FastAPI)

Нужен Python и USB-кабель к мастеру.

```powershell
pip install fastapi uvicorn jinja2 pyserial
```

В двух окнах из корня репо:

```powershell
# окно 1 — мост, перенаправляет Serial мастера в JSONL
python code\scripts\hash_kws_master_demux_bridge.py --port COM7

# окно 2 — дашборд
python run_dashboard.py
```

(`COM7` поменяй на свой порт мастера.)

Открыть `http://127.0.0.1:8765/`. UI визуально совпадает с on-board дашбордом (через `/api/wire_stream` SSE с теми же типами `node`/`fusion`/`camera`/`counters`).

Перед демо опционально очистить JSONL-стримы:
```powershell
python code\scripts\reset_dashboard_streams.py
```

---

## 6. Sanity-чек

В Serial мастера (или в `--echo` мосту) во время речи должны быть:
1. `hash_evt kind=infer node=1/2/3 ... top1=<word>` — это значит ноды реально слышат и инферят.
2. `hash_evt kind=fusion node=master label=<word> voters=N ...` — fusion отработал.
3. (Если камера в сети) `hash_evt kind=cam_status ...` каждые 2с + `hash_evt kind=cam_trigger/cam_reply ...` при срабатывании trigger-слова.

Если видишь только `Heard X (Y) @Zms` — это устаревший вывод от стороннего callback'а. Прошёл не тот скетч / стэйл-кэш Arduino IDE. Делай Erase All Flash → перепрошей.

---

## 7. Ключевые параметры (master)

| Define | Default | Что делает |
|---|---|---|
| `HASH_KWS_AGG_NUM_NODES` | 3 | Сколько нод ждать |
| `HASH_KWS_AGG_WINDOW_MS` | 1200 | Окно усреднения логитов (мс) |
| `HASH_KWS_AGG_MODE` | 0 | 0=mean_logits, 1=temperature_scaled, 2=learned_weights |
| `HASH_KWS_AGG_NOISE_BOOST` | 24.0 | Сдвиг логитов на noise-классы (подавляет ложные срабатывания) |
| `HASH_KWS_POST_COMMAND_QUIET_MS` | 1300 | Quiet после публикации команды (мс) |
| `HASH_KWS_CAMERA_TRIGGER_LABEL_IDX` | 0 (yes) | KWS-индекс, по которому стреляет камера |
| `HASH_KWS_CAMERA_TRIGGER_QUIET_MS` | 1500 | Quiet между триггерами камеры |

Аггрегатор держит кольцо 5 слотов на ноду, усредняет **все** в-windows слоты всех нод (~15 logit-векторов steady-state). Решение: argmax после noise-boost; публикуется только если ≥2 нод проголосовали в окне.

---

## 8. Известные особенности

- **vww1 на камере** — модель смещена к `no_person` на близкой дистанции (объясняется небольшим разрешением 80×80 и узким VWW-датасетом). На дальнем кадре с человеком в полный рост работает. Для демо целься объективом сознательно. Перейти на vww2 (144×144, 91.8% vs 88.9%) — пересобрать через `python code/training/image_classifier/import_mcunet_vww.py --variant vww2` + reflash (+250 мс инференса).
- **USB CDC On Boot = Disabled** скрывает ROM-bootloader логи (включая ошибки PSRAM) от USB-CDC. Включай `Enabled` для прозрачности.
- **Камерный Serial** использует framed-протокол с `\r\n` после CRC — читается в обычном Serial Monitor.
- **Master в AP-mode** использует `WIFI_AP_STA` (а не `WIFI_AP`), чтобы ESP-NOW-broadcast мог уйти через STA-iface. Не трогай эту строку.

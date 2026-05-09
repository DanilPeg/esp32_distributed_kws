# Audio distributed KWS demo — quickstart for an LLM operator

Этот документ — пошаговая инструкция для **LLM-агента**, который помогает коллеге запустить уже протестированный аудио-вариант распределённого KWS-кластера на двух платах ESP32-S3 с локальным дашбордом.

Образ работы: операторы (человек + LLM) выполняют шаги по порядку, после каждого блока — короткая проверка с явным критерием прохождения. Если критерий не выполнен — переходим в раздел `## Troubleshooting`, не идём дальше.

Всё, что относится к **картинкам/камере/изображениям, в этом гайде НЕ запускается**. Ветка камеры (`code/firmware/camera_stream*`, `code/scripts/camera_stream_server.py`) — отдельный трек, его игнорировать.

---

## 1. Что демонстрируется

Распределённый Keyword-Spotting (10 команд + silence + unknown) на двух MCU ESP32-S3 с микрофонами I2S MEMS. Каждая плата делает локальный инференс собственной TFLite Micro моделью и обменивается результатами по ESP-NOW. Хост-машина:

- читает Serial одной из плат (`hash_kws_serial_bridge.py`),
- симулирует виртуальные ноды и фьюжен-агрегатор (`hash_kws_cluster_sim.py`),
- крутит FastAPI-дашборд `code/dashboard/` на `http://127.0.0.1:8765/`.

Дашборд показывает: статус каждой ноды, последние события `top1`/score, лента инференса, фьюжен-решения, ESP-NOW счётчики (`tx_ok`/`tx_fail`).

Один лаунчер `code/scripts/run_distributed_demo.py` поднимает все три процесса разом и кладёт всё под `notes/Journal/hash_kws_telemetry/` и `notes/Journal/hash_kws_fusion/`.

---

## 2. Что должно быть у коллеги (hardware + software)

### Hardware (минимум для демо со звуком)

- 2 × ESP32-S3 (рекомендуется N16R8: 16 MB flash, 8 MB OPI PSRAM). Любая S3-плата с I2S микрофоном подойдёт, если пины совпадают с проектом.
- 2 × I2S MEMS микрофон (INMP441 / SPH0645 — то, что было в проекте).
- USB-кабели для прошивки. Один из кабелей должен оставаться подключённым к хосту во время демо (с него хост читает Serial).
- Wi-Fi не нужен для демо. ESP-NOW работает на L2 без роутера.

### Software на хосте

- Python 3.10+ (3.11 / 3.12 проверены).
- Arduino IDE 2.x **или** `arduino-cli` с установленным core `esp32` от Espressif (версии Arduino-ESP32 2.0.x или 3.x — оба пути обработаны в `micro_speech.ino`).
  - В Arduino IDE поставить board package: `esp32` by Espressif Systems.
  - Версия пакета — последняя стабильная на момент демо (2.0.14+ или 3.0.x).
- pip-зависимости хоста:
  ```text
  pyserial
  fastapi
  uvicorn
  jinja2
  ```

LLM-checklist (выполнить и убедиться):
1. `python --version` → `>= 3.10`.
2. `python -c "import serial, fastapi, uvicorn, jinja2"` не падает. Если падает — `pip install pyserial fastapi uvicorn jinja2`.
3. В Arduino IDE → Tools → Board → доступен `ESP32S3 Dev Module` (или эквивалент).

---

## 3. Подготовка репозитория

```powershell
git clone https://github.com/DanilPeg/esp32_distributed_kws.git
cd esp32_distributed_kws
git switch docs/audio-demo-quickstart   # ветка, где лежит этот гайд
```

Готовые artefacts для прошивки уже лежат в репо: `dist/esp32_two_mcu_hash_kws_models/`. **Не нужно** ничего пересобирать или пере-обучать ради демо.

```
dist/esp32_two_mcu_hash_kws_models/
├── README_FOR_COLLEAGUE.md
├── board1_model_9128_node1/        # → флэшится на ПЛАТУ #1
│   └── code/firmware/micro_speech_sim/micro_speech/micro_speech.ino
├── board2_model_9261_node2/        # → флэшится на ПЛАТУ #2
│   └── code/firmware/micro_speech_sim/micro_speech/micro_speech.ino
└── raw_model_artifacts/             # сырые TFLite/PyTorch — для отчёта, не для прошивки
```

LLM-checklist:
1. `dir dist\esp32_two_mcu_hash_kws_models` (Win) или `ls dist/esp32_two_mcu_hash_kws_models` (POSIX) → видны обе папки `board1_*` и `board2_*`.
2. В каждой папке есть `code/firmware/micro_speech_sim/micro_speech/micro_speech.ino`.

---

## 4. Прошивка плат (Arduino IDE — рекомендованный путь)

Локальный `arduino-cli` на машине лида падает по 300 s timeout (см. `notes/Journal/2026-04-24_checkpoint_001.yaml` → `risks`). Рабочий путь — Arduino IDE 2.x.

### 4.1 Board options (одинаково для обеих плат)

В Arduino IDE → Tools:

| Setting              | Value                                |
|----------------------|--------------------------------------|
| Board                | `ESP32S3 Dev Module`                 |
| USB CDC On Boot      | `Enabled`                            |
| USB Mode             | `Hardware CDC and JTAG`              |
| CPU Frequency        | `240MHz (WiFi)`                      |
| Flash Mode           | `QIO 80MHz`                          |
| Flash Size           | `16MB (128Mb)`                       |
| Partition Scheme     | `Huge APP (3MB No OTA/1MB SPIFFS)`   |
| PSRAM                | `OPI PSRAM`                          |
| Upload Speed         | `921600`                             |
| Core Debug Level     | `None`                               |

### 4.2 Board #1 (модель 91.28%, node id = 1)

1. File → Open → выбрать
   `dist/esp32_two_mcu_hash_kws_models/board1_model_9128_node1/code/firmware/micro_speech_sim/micro_speech/micro_speech.ino`.
2. Tools → Port → COM-порт первой платы.
3. Sketch → Verify (≈ 1–3 минуты на холодную).
4. Sketch → Upload.

### 4.3 Board #2 (модель 92.61%, node id = 2)

То же самое, но для папки `board2_model_9261_node2/...`. Tools → Port — другой COM-порт.

### 4.4 Что должно быть в Serial Monitor (115200 baud) после ребута

На каждой плате — события в формате `key=value`:

```
hash_evt kind=boot node=1 ...
hash_evt kind=espnow phase=init status=ok ...
hash_evt kind=heartbeat ...
```

Когда обе платы включены:

```
hash_evt kind=peer ...           # увидели соседа
hash_evt kind=fusion ...         # хотя бы одно фьюжен-решение от своего PoV
```

LLM-checklist (после прошивки обеих плат):
1. На плате #1 в Serial — есть строки `kind=boot node=1`.
2. На плате #2 в Serial — есть строки `kind=boot node=2`.
3. Хотя бы у одной из плат есть `kind=peer` или `kind=espnow` с активностью (TX/RX).
4. Если 3 не выполняется — [Troubleshooting → ESP-NOW silent](#tr-espnow).

> ВАЖНО: Arduino IDE Serial Monitor блокирует COM-порт. Перед запуском хост-стека (шаг 5) **закрыть Serial Monitor** для платы, к которой прицепится бридж.

---

## 5. Запуск host-стека (бридж + sim + дашборд)

Подключить к хосту **одну** из плат (любую — рекомендуется плата #1, чтобы ID совпадал с `--node-id 1`). Вторая плата питается отдельно (USB-зарядник или второй USB-порт без открытого Serial Monitor).

### 5.1 Установить зависимости (если ещё не сделано)

```powershell
pip install pyserial fastapi uvicorn jinja2
```

### 5.2 Команда запуска

```powershell
python code\scripts\run_distributed_demo.py --port COM5
```

Заменить `COM5` на реальный порт первой платы (Linux: `/dev/ttyUSB0` или `/dev/ttyACM0`).

Что лаунчер делает:
1. Стартует `hash_kws_serial_bridge.py` — пишет события платы #1 в `notes/Journal/hash_kws_telemetry/node1/events.jsonl`.
2. Стартует `hash_kws_cluster_sim.py` — на основании реального потока ноды #1 имитирует ноду #2 и мастер-аггрегатор; кладёт фьюжен-решения в `notes/Journal/hash_kws_fusion/decisions.jsonl`.
3. Стартует `run_dashboard.py` — поднимает FastAPI на `http://127.0.0.1:8765/`.

В консоли должно быть примерно:

```
[demo] starting bridge: ...
[demo] starting cluster_sim: ...
[demo] starting dashboard: ...
[demo] dashboard → http://127.0.0.1:8765/
[demo] press Ctrl+C once to stop everything.
```

### 5.3 Полезные флаги

- `--single-node` — отключить виртуального fusion-симулятора. В дашборде остаётся одна реальная нода #1.
- `--no-dashboard` — не поднимать UI (если уже запущен отдельно).
- `--baud 115200` (default), `--node-id 1` (default), `--node-label real_esp32`.
- `--dashboard-port 8765` (default).

LLM-checklist:
1. `run_distributed_demo.py` стартовал без traceback в первые 10 с.
2. Все три строки `[demo] starting ...` в логе.
3. `curl http://127.0.0.1:8765/health` отдаёт 200 (или браузер открывает страницу).

---

## 6. Что должно быть видно в дашборде (критерий «демо работает»)

Открыть `http://127.0.0.1:8765/` в браузере на хосте.

| Панель                | Должно быть видно                                                |
|-----------------------|------------------------------------------------------------------|
| Cluster overview      | Карточка `node1` (real) — online, last kind свежий, top1+score   |
| Cluster overview      | Карточка `node2` (emulated) — online, обновляется                |
| Cluster overview      | Карточка `node3` / master — online, фьюжен-счётчики растут       |
| Live inference feed   | Лента событий, новые строки появляются раз в несколько секунд    |
| Fusion decisions      | Появляются записи `agree` / `single-node` / `waiting`            |
| Counters (tail)       | Per-label tag cloud формируется, когда говоришь команду          |

Speech-тест: чётко произнести в микрофон одно из ключевых слов (`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`). В пределах 1–2 с в Live inference feed должна появиться строка с подходящим `top1` и score выше шумового порога.

LLM-checklist (полный успех демо):
1. UI рендерится без 500/404.
2. Серверный лог содержит `GET /api/stream` — соединение SSE открыто.
3. После произнесения 5 разных команд видим как минимум 3 разных `top1`-лейбла в ленте.
4. Появляется хотя бы одно `kind=fusion` событие с участием обеих нод.

Если все 4 пункта зелёные — демо **со звуком и дашбордом** запущено корректно. Это та самая конфигурация, которая отчётно тестировалась 2026-04-24/25.

---

## 7. Завершение работы

`Ctrl+C` в окне с `run_distributed_demo.py` — гасит все три дочерних процесса (бридж, sim, dashboard). Платы продолжают работать автономно (ESP-NOW не зависит от хоста).

Артефакты прогона остаются:
- `notes/Journal/hash_kws_telemetry/node1/events.jsonl` (сырой поток платы #1)
- `notes/Journal/hash_kws_telemetry/events.jsonl` (нормализованный merged stream)
- `notes/Journal/hash_kws_fusion/decisions.jsonl` (фьюжен-решения)
- `notes/Journal/hash_kws_telemetry/state.json`, `hash_kws_fusion/state.json`, `hash_kws_cluster/state.json`

Если коллеге нужно обнулить состояние перед повторным прогоном — просто удалить эти файлы; продьюсеры пересоздадут их.

---

## 8. Troubleshooting

### 8.1 Compile fails в Arduino IDE
- Убедиться, что выбран **именно** `ESP32S3 Dev Module`, а не S2 / C3 / классический ESP32.
- PSRAM **обязательно** `OPI PSRAM`. На «Disabled» прошивка зальётся, но модель не поместится в arena.
- Partition Scheme должен быть `Huge APP` — иначе бинарь не влезет.
- Если ошибка `'esp_nn_*' was not declared` — обновить Arduino-ESP32 core до последней стабильной (`Boards Manager → esp32 → Update`).

### 8.2 Upload fails (`Failed to connect`)
- Зажать `BOOT`, кратко `RESET`, отпустить `BOOT` → повторить Upload.
- Скорость загрузки `921600` иногда нестабильна на дешёвых USB-кабелях — упасть до `460800` или `230400`.

### 8.3 <a id="tr-espnow"></a>В Serial есть `boot`, но нет `espnow`/`peer`/`fusion`
- Обе платы должны быть на одном Wi-Fi канале. В прошивках забит канал `1` — `dist/.../README_FOR_COLLEAGUE.md` подтверждает. Если в проекте поменян — обе платы должны быть пересобраны с одним и тем же `HASH_KWS_ESPNOW_CHANNEL`.
- ESP-NOW бродкастит — MAC-адреса прописывать не нужно.
- Если рядом сильное Wi-Fi помеховое поле (2.4 GHz роутер на канале 1) — переместить платы или переключить канал в обеих прошивках.

### 8.4 `pyserial` не находит порт / `PermissionError`
- Закрыть Arduino IDE Serial Monitor — он держит COM эксклюзивно.
- На Linux: добавить пользователя в группу `dialout` (`sudo usermod -aG dialout $USER`, перелогин).

### 8.5 Дашборд открывается, но карточки нод серые / `offline`
- Бридж не получает данных. Проверить:
  - корректный `--port` в `run_distributed_demo.py`,
  - в `notes/Journal/hash_kws_telemetry/node1/raw.log` есть свежие байты,
  - плата не в reset-loop (по Serial раз в N секунд должны идти `heartbeat`).

### 8.6 Высокая задержка инференса (`invoke_ms` > 2.5 s) / срывы аудио
- Это известное поведение 128-channel модели на ESP32-S3 (см. checkpoint `2026-04-24_001` → `risks` → 128-channel inflated student edge of realtime).
- Демо всё равно работает: scheduler в `micro_speech.ino` использует one-invoke-per-episode и ladder ring-buffer fallback (131072 → 32768 байт). Это компромисс на случай PSRAM-fragmentation.
- Если совсем плохо — `--single-node` режим даёт менее впечатляющую, но более стабильную картинку.

### 8.7 Дашборд не запустился: `ImportError: dashboard.app`
- Запускать `run_dashboard.py` или `run_distributed_demo.py` из корня репозитория, либо через абсолютный путь — лаунчер сам подгружает `code/` в `sys.path`.
- Проверить, что не активирован конфликтующий venv с другим Python.

---

## 9. Reference: ключевые файлы

| Файл | Роль |
|------|------|
| `dist/esp32_two_mcu_hash_kws_models/board1_model_9128_node1/.../micro_speech.ino` | Прошивка платы #1 (модель 91.28%, node 1) |
| `dist/esp32_two_mcu_hash_kws_models/board2_model_9261_node2/.../micro_speech.ino` | Прошивка платы #2 (модель 92.61%, node 2) |
| `dist/esp32_two_mcu_hash_kws_models/README_FOR_COLLEAGUE.md` | Краткая записка от автора об артефактах прошивки |
| `code/scripts/run_distributed_demo.py` | Лаунчер всего хост-стека одной командой |
| `code/scripts/hash_kws_serial_bridge.py` | Парсер Serial → JSONL |
| `code/scripts/hash_kws_cluster_sim.py` | Виртуальная нода #2 + мастер + фьюжен |
| `run_dashboard.py` | Запуск FastAPI-дашборда (можно вызывать отдельно) |
| `code/dashboard/README.md` | Документация дашборда (endpoints, layout) |
| `notes/Journal/2026-04-24_checkpoint_001.yaml` | Чекпоинт спринта, в котором эта связка собиралась |

---

## 10. Что НЕ делает этот гайд (out of scope)

- Не запускает камеру / image-classifier (`code/firmware/camera_stream*`, `code/scripts/camera_stream_server.py`). Image-track ещё в P0/P1 (см. `notes/Journal/2026-05-03_image_training_plan_004.yaml`).
- Не тренирует модели локально. Тренировка — только в Colab (`code/training/`), артефакты лежат в `dist/`.
- Не работает со старым deprecated `code/web_portal/` — только новый `code/dashboard/`.
- Не настраивает Telegram-runtime (`code/scripts/telegram_*`) — это отдельная подсистема, не нужна для демо.

---

## 11. Если что-то пошло не так — что показать обратно

LLM-агенту, который ведёт коллегу: при любом застрявании собрать и приложить:

1. Вывод первых 20 строк Serial каждой платы после ребута.
2. Файл `notes/Journal/hash_kws_telemetry/node1/raw.log` (последние 100 строк).
3. Файл `notes/Journal/hash_kws_telemetry/state.json` целиком.
4. Скриншот вкладки браузера с дашбордом + DevTools → Network для запросов `/api/snapshot` и `/api/stream`.
5. Версии: `python --version`, версия Arduino-ESP32 core (Boards Manager).

С этим набором почти любой блок диагностируется без дополнительных запросов.

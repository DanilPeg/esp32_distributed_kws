# Master Dashboard — спецификация для реализации

Документ для Сергея. Описывает, что должен делать дашборд, который
живёт **на самой master‑плате ESP32** (4‑й МК, без микрофона) и заменяет
наш предыдущий хост‑side FastAPI dashboard.

Архитектура такая же, как у Сергея в координаторе из ветки
`docs/audio-demo-sergey`: AsyncWebServer + AsyncWebSocket, single‑page
HTML внутри прошивки, push обновлений по WS. Отличия — в данных и UX.

---

## 1. Что master уже делает

В ветке `feat/hash-ensemble-deploy`, в `code/firmware/hash_kws_master_web/`,
лежит рабочий **прототип** с минимальным дашбордом — можно использовать
как стартовую точку или переписать с нуля. Ключевые файлы:

| Файл | Роль |
|---|---|
| `hash_kws_master_web.ino` | главный sketch (504 LOC) |
| `web_page.h` | embedded HTML/CSS/JS (~150 LOC PROGMEM) |
| `hash_ensemble_aggregator.{h,cpp}` | C++ агрегатор логитов (готовый, трогать не надо) |
| `aggregator_params.h` | калиброванные T и веса с обучения (auto‑gen) |
| `README.md` | инструкция по сборке |

Прототип работает; задача — **довести UX до уровня нашего FastAPI‑дашборда** (см. раздел 4).

Master:
1. Поднимает WiFi (STA по дефолту, AP fallback).
2. Принимает ESP‑NOW пакеты от 3 inference‑узлов (`ens_a` / `ens_b` / `ens_c`).
3. Гонит логиты в `hash_kws_ensemble::Aggregator::submit(...)`.
4. Каждый цикл `loop()` зовёт `aggregator.resolve(...)` — если ≥ 2 голосов
   в окне 1.2 сек, получает решение `{label, score, margin, voters, mode}`.
5. Обновляет внутренние состояния (per‑node + ring fusion) и пушит через
   WebSocket клиентам.

---

## 2. Контракт ESP‑NOW пакета

Inference‑узлы шлют пакет фиксированного layout. Это **тот же** пакет,
который master уже декодирует — никаких изменений со стороны inference
не нужно. Источник истины — `code/firmware/micro_speech_sim/micro_speech/micro_speech.ino`,
константы:

```cpp
constexpr uint32_t kHashKwsEspNowMagic   = 0x4B485731UL;  // "KHW1"
constexpr uint8_t  kHashKwsEspNowVersion = 1;

struct __attribute__((packed)) HashKwsEspNowPacket {
  uint32_t magic;
  uint8_t  version;
  uint8_t  node;          // 1..3, идентификатор inference-узла
  uint16_t seq;
  uint32_t t_ms;          // отправитель millis()
  uint16_t invoke_ms;     // время инференса
  uint8_t  kind;          // 0=infer, 1=episode, 2=emit
  uint8_t  label;         // top1 индекс класса (0..11)
  uint8_t  score;         // top1 «уверенность» (0..255)
  uint8_t  margin;        // top1 - top2
  uint8_t  recent_max;    // максимум амплитуды аудио за окно
  uint8_t  flags;         // зарезервировано
  int8_t   logits[12];    // ← главное, для агрегации
  uint16_t crc16;         // CRC-16/IBM, по всем байтам кроме самого crc16
};
```

Канал: задаётся compile‑time через `HASH_KWS_ESPNOW_CHANNEL`. Должен
совпадать на всех 4 платах. В STA‑режиме master автоматически использует
канал роутера — inference‑узлы должны быть скомпилированы под этот канал.

---

## 3. Лейблы и режимы агрегации

Двенадцать классов в фиксированном порядке (индекс = поле `label`):

```
0: yes   1: no    2: up    3: down
4: left  5: right 6: on    7: off
8: stop  9: go   10: unknown 11: silence
```

Три режима агрегатора (compile‑time через `HASH_KWS_AGG_MODE`):

| Mode | Имя | Что делает | Когда выбирать |
|---|---|---|---|
| `0` | `mean_logits` | среднее int8 логитов поэлементно | по умолчанию (headline исследования) |
| `1` | `temperature_scaled` | mean(softmax(logits_k / T_k)) с per‑model `T` из `aggregator_params.h` | если хочется откалиброванных вероятностей |
| `2` | `learned_weights` | `Σ w_k · logits_k` с весами из `aggregator_params.h` | альтернатива, на наших данных ≈ `mean_logits` |

Numeric‑различие на test < 0.1 п.п., поэтому в production можно держать
`mean_logits`. Параметр `mode` мастер уже передаёт в WebSocket как
число — на UI показывать строкой через таблицу выше.

---

## 4. Что должен показывать дашборд

Цель — повторить и слегка упростить наш FastAPI‑дашборд из
`code/dashboard/templates/index.html`. Минимально‑необходимые блоки
сверху вниз:

### 4.1 Заголовок + статус соединения

```
Hash KWS Distributed Master
3 hashed students — ensemble aggregation on this ESP32
[ live | disconnected ]
```

`live` зелёным когда WS подключён, `disconnected` красным с авто‑reconnect
каждую секунду.

### 4.2 Counters strip

Однострочный блок с цифрами, обновляются по push:

| Что | Откуда |
|---|---|
| `fusion: <N>` | счётчик принятых решений с момента boot |
| `packets: <N>` | сколько валидных ESP‑NOW пакетов master принял |
| `rejected: <N>` | сколько пакетов отбросил (CRC / magic / version mismatch) |
| `aggregator: <name>` | текущий режим (`mean_logits` / `temperature_scaled` / `learned_weights`) |
| `uptime: <s>` | `millis() / 1000` мастера |

### 4.3 Per‑node tiles (3 штуки)

Сетка 3×1 (на узких экранах — 1×3). Каждый тайл показывает:

```
┌──────────────────────┐
│ NODE 1 — ens_a       │   <- маленький subdued title
│   yes                │   <- крупная label (28-30 px), цвет по классу
│ score=200 margin=42  │   <- мелкий мета‑текст
│ packets=142          │
└──────────────────────┘
```

Состояния тайла:
- **online** — последний пакет ≤ 4 секунд назад → зелёная левая граница
- **stale** — > 4 секунд → жёлтая граница + label из последнего пакета
- **never seen** — серая граница, текст «no packets yet»

Цвета меток (можно использовать наши, `web_page.h` в прототипе уже их задаёт):

```
yes=#41d287   no=#f06363   up=#6ec3ff    down=#ffae3a
left=#c5b3ff  right=#ff85c8 on=#b6ff63    off=#888
stop=#ff7373  go=#4ad6ff   unknown/silence=#5a6470
```

### 4.4 Fusion decisions list

Список решений ансамбля, **новейшее сверху**, до 50 записей. Каждая
запись:

```
yes    score=4736 margin=2048 voters=3 · 12:34:56
stop   score=3892 margin=1856 voters=3 · 12:34:51
no     score=4102 margin=1920 voters=2 · 12:34:48
```

`score` и `margin` — из C++ aggregator, в Q8.8 fixed‑point (поделить на
256, чтобы получить float; на UI можно показывать как есть, число как
число — это «уверенность», не вероятность). `voters` ∈ {2, 3}.

Опциональный приятный штрих (мы делали в FastAPI): если последние ≥ 3
строк подряд — одна и та же команда от одного узла внутри 2 секунд, в
карточку добавлять `×N` с количеством. Не критично, можно опустить.

### 4.5 Inference latency (опционально)

В нашем FastAPI был отдельный блок «Inference performance»:
median/min/avg/p95/max + sparkline + warning bands (<200 ms green,
200–500 amber, >500 ms red). На master это отдельная история, потому что
`invoke_ms` приходит в каждом пакете — можно собирать ringbuffer
последних 30 значений per‑node и считать. Если осилить — большой плюс
для защиты диплома; если нет — пропускаем.

### 4.6 Что **не** делать

- НЕ делать графики загрузки CPU master (слишком общо, не относится к ансамблю).
- НЕ делать настройки агрегатора в UI — он compile‑time, перепрошивкой.
- НЕ хранить историю в SPIFFS (в нашем сценарии это лишний слой; если хочется логов — пиши в Serial, юзер уже умеет ловить через USB).

---

## 5. WebSocket протокол

Endpoint: `ws://<master-ip>/ws`. Сообщения JSON, разделяются естественно
рамкой WebSocket. Порядок и формат — как в текущем прототипе, при
переписывании можно слегка поменять, лишь бы был консистентен.

### 5.1 На connect — `snapshot`

Сервер шлёт **один раз**, сразу после успешного `WS_EVT_CONNECT`:

```json
{
  "type": "snapshot",
  "nodes": [
    {"node": 1, "label": 0, "score": 200, "margin": 42, "packets": 142},
    {"node": 2, "label": 0, "score": 215, "margin": 58, "packets": 138},
    {"node": 3, "label": 0, "score": 190, "margin": 38, "packets": 140}
  ],
  "fusion": [
    {"label": 0, "score": 4736, "margin": 2048, "voters": 3, "time_ms": 123456}
  ],
  "counters": {
    "fusion": 47, "packets": 420, "rejected": 0,
    "agg_mode": 0, "uptime_s": 105
  }
}
```

`fusion[]` — последние ≤ 30 решений, новейшее **первым**.

### 5.2 На каждый принятый пакет — `node`

```json
{
  "type": "node",
  "node": 2,
  "label": 4,
  "score": 215,
  "margin": 58,
  "packets": 139,
  "counters": { "fusion": 47, "packets": 421, "rejected": 0, "agg_mode": 0, "uptime_s": 106 }
}
```

### 5.3 На каждое fusion‑решение — `fusion`

```json
{
  "type": "fusion",
  "label": 0,
  "score": 4736,
  "margin": 2048,
  "voters": 3,
  "time_ms": 234567,
  "counters": { ... }
}
```

`time_ms` — `millis()` в момент решения, для сортировки и dedup.

### 5.4 Прочее

- На клиенте reconnect через 1 секунду при `onclose`.
- Не делать heartbeat — JsonDocument в каждом сообщении уже имеет
  `counters`, этого хватает на «живой» индикатор.
- Никакого backpressure (наш трафик ~4 msg/s в активной фазе, AsyncTCP
  очередь spravится).

---

## 6. Технические детали

### 6.1 Библиотеки Arduino

- **ESP Async WebServer** by mathieucarbou (версия с поддержкой esp32 core 3.x)
- **Async TCP** by mathieucarbou
- **ArduinoJson** by bblanchon, ≥ 7.0

> Старый `me-no-dev/ESPAsyncWebServer` НЕ компилируется под core 3.x. Это
> важно — у нас core 3.x по дефолту в Tools → Boards.

### 6.2 Board options (Tools)

Те же, что и в `code/firmware/hash_kws_master/`:

- ESP32S3 Dev Module
- USB CDC On Boot: Enabled
- CPU 240 MHz
- PSRAM: OPI PSRAM
- Flash 16MB (или 8MB по плате)
- Partition Scheme: Default

### 6.3 Ресурсы

- Flash: ~1 MB после компиляции (наш прототип 982 KB / 1310 KB partition).
- RAM: ~50 KB глобалов + AsyncTCP TCP buffers ~30 KB на 1 клиент.
- Стек loop‑task стандартный, в callback'ах ничего тяжелого.

### 6.4 Канал ESP‑NOW

В STA‑режиме мастер залочен на канал WiFi. Inference‑узлы про роутер не
знают — у них канал зашит compile‑time. Spec для пользователя:
посмотреть канал через `netsh wlan show interfaces`, пересобрать все 4
sketches с `-DHASH_KWS_ESPNOW_CHANNEL=<chan>`.

Если нет 2.4 ГГц сети поблизости — fallback в AP‑mode (master сам
точка доступа, канал по умолчанию 1, никаких подгонок).

---

## 7. Acceptance criteria

Дашборд считается готовым, когда:

1. С нуля прошитый master при произнесении команды в любой из 3 микрофонов
   показывает в браузере:
   - Соответствующий тайл узла стал зелёным, текст команды и обновлённые
     score/margin/packets.
   - В fusion‑списке появилась новая строка с правильной командой.
   - Счётчик `fusion` увеличился на 1.
2. После 30 секунд в покое (тишина) тайлы становятся жёлтыми (`stale`).
3. При перезагрузке роутера / разрыве WiFi на 30 сек статус показывает
   `disconnected`, после восстановления — `live`, без потери счётчиков
   на стороне master.
4. Открытие дашборда с **двух** устройств одновременно (ноут + телефон)
   работает: оба видят одно и то же (broadcast по `ws.textAll`).
5. Серийный лог master по‑прежнему печатает `hash_evt kind=fusion ...`
   как и раньше — это нужно для бэкап‑записи логов в JSONL хост‑side
   при необходимости (см. `code/scripts/hash_ensemble_master_bridge.py`).

---

## 8. Где что взять для реализации

| Что | Где |
|---|---|
| Прототип master sketch + HTML | `code/firmware/hash_kws_master_web/` (ветка `feat/hash-ensemble-deploy`) |
| C++ агрегатор (готовый) | `code/firmware/hash_kws_aggregator/hash_ensemble_aggregator.{h,cpp}` |
| Калиброванные параметры | `code/deploy/hash_ensemble/reports/aggregator_params.h` |
| Layout пакета inference‑узла | `code/firmware/micro_speech_sim/micro_speech/micro_speech.ino` (поиск `HashKwsEspNowPacket`) |
| Обращайся к нашему хост‑дашборду за вдохновением UX | `code/dashboard/templates/index.html` |
| Метрики и пример target‑accuracy | `code/deploy/hash_ensemble/reports/ensemble_results.json` |
| Журнальный план + решения | `notes/Journal/2026-05-09_hash_ensemble_plan.md` |

Если что‑то непонятно по агрегатору, лейблам, формату пакета или
WebSocket‑контракту — пиши, разберём прямо в этом файле или в issues.

---

## 9. Текущий прототип — короткий обзор

В `hash_kws_master_web/hash_kws_master_web.ino` основные блоки:

- `bringUpWifi()` — STA / AP fallback.
- `bringUpEspNow()` — `esp_now_register_recv_cb(onDataRecv)`.
- `onDataRecv()` — валидация magic/version/CRC, `aggregator.submit(...)`,
  обновление `g_nodes[]`, `broadcastNode(...)`.
- `pollAggregator()` — в `loop()`, `aggregator.resolve(...)`, dedup по
  одинаковой метке внутри 800 мс, `broadcastFusion(...)`, печать в Serial.
- `onWsEvent()` — на `WS_EVT_CONNECT` шлёт `sendSnapshot(client)`.
- `web_page.h` — HTML с inline CSS+JS, дашборд в одном статичном файле
  размером ~5 KB после gzip‑компрессии (gzip не включён, но можно — у
  AsyncWebServer есть поддержка).

Можно использовать как базу или как референс — оба варианта валидные.

# Видео-MCU → master_web: протокол обмена

Документ описывает **ровно то, что должен сделать LLM-агент**, пишущий
прошивку для отдельного микроконтроллера с видео-нейронкой, чтобы его
результаты корректно принимались мастером (`code/firmware/hash_kws_master_web/`)
и появлялись в его веб-дашборде в отдельной карточке `Video MCU`.

Мастер уже умеет всё нужное: принимает ESP-NOW-пакеты, маршрутизирует
пакеты с `node = 4` в обособленную видео-ветку, кормит их в собственный
временной аггрегатор (mean over 1.2 s sliding window), рисует UI.
**Менять код мастера не требуется.** Видео-MCU должен лишь правильно
сформировать ESP-NOW-пакет и отправить его на MAC мастера.

---

## 1. Транспорт

ESP-NOW поверх 2.4 GHz Wi-Fi. Без TCP/IP, без подключения к роутеру со
стороны видео-MCU.

### 1.1 Жёсткие требования к радио

- **Канал** видео-MCU должен совпадать с каналом мастера.
  - Если мастер запускается в STA-режиме и подключён к роутеру, его
    канал = канал роутера. На Windows смотри `netsh wlan show interfaces`,
    строка `Channel`.
  - Если мастер ушёл в AP-fallback (`KWS-Master` / `kwsmaster1`), он
    использует `HASH_KWS_ESPNOW_CHANNEL` (по умолчанию `1`).
  - На видео-MCU перед `esp_now_init()` выставь канал так:
    ```cpp
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    esp_wifi_set_channel(MASTER_CHANNEL, WIFI_SECOND_CHAN_NONE);
    ```
- **MAC мастера** надо знать заранее. Узнать его проще всего так:
  - После загрузки мастер печатает в Serial:
    `hash_evt kind=espnow phase=init status=ok node=master mac=AA:BB:CC:DD:EE:FF`.
    Это и есть MAC, который указывать `peer_addr` на видео-MCU.
  - В STA-режиме это `WiFi.macAddress()`, в AP-fallback —
    `WiFi.softAPmacAddress()`.
- **Шифрование ESP-NOW** не используется. Не включай его на стороне
  отправителя.
- **Broadcast** (`FF:FF:FF:FF:FF:FF`) тоже работает, но требует, чтобы
  мастер сидел на том же канале. Unicast предпочтительнее — меньше шума
  в эфире и стабильнее.

---

## 2. Структура пакета

Используется **тот же бинарный пакет**, что и для аудио-нод. Видео-MCU
прикидывается «4-й нодой» в общем кластере. Размер строго **34 байта**,
выравнивание упаковано.

### 2.1 Магия и версия

```cpp
constexpr uint32_t kHashKwsEspNowMagic   = 0x4B485731UL;  // "KHW1"
constexpr uint8_t  kHashKwsEspNowVersion = 1;
```

`magic` мастер сверяет в первую очередь. Не угадаешь — пакет молча
посчитается `rejected`.

### 2.2 C-струтура (копировать как есть)

```cpp
struct __attribute__((packed)) HashKwsEspNowPacket {
  uint32_t magic;        // = kHashKwsEspNowMagic
  uint8_t  version;      // = kHashKwsEspNowVersion
  uint8_t  node;         // = 4 (см. §3.1)
  uint16_t seq;          // монотонный счётчик кадров, для дебага
  uint32_t t_ms;         // millis() в момент готовности кадра
  uint16_t invoke_ms;    // длительность инференса (ms)
  uint8_t  kind;         // 0 = infer, 1 = episode, 2 = emit (см. §3.2)
  uint8_t  label;        // индекс top-1 в видео-таблице меток (0..11)
  uint8_t  score;        // raw-уверенность top-1, 0..255 (см. §3.4)
  uint8_t  margin;       // raw top1 - top2, 0..255 (см. §3.4)
  uint8_t  recent_max;   // для видео можно оставить 0 (см. §3.5)
  uint8_t  flags;        // зарезервировано, = 0
  int8_t   logits[12];   // ВАЖНО: реальные int8-логиты (см. §3.3)
  uint16_t crc16;        // CRC-16/ARC по первым 32 байтам (см. §4)
};
static_assert(sizeof(HashKwsEspNowPacket) == 34, "packet must be 34 bytes");
```

`#pragma pack` / `__attribute__((packed))` обязателен — иначе компилятор
вставит padding, итоговый размер не сойдётся с приёмником.

---

## 3. Семантика полей под видео

### 3.1 `node`

Всегда **`4`**. Это значение `HASH_KWS_VIDEO_NODE_ID` в `main.cpp` мастера.
Не путать с аудио-нодами 1..3 — те имеют другую таблицу классов и идут
в свой ансамбль. Если пришлёшь `node = 1..3`, мастер скормит логиты в
аудио-агрегатор и интерпретирует их как KWS-классы (yes / no / up / down
/ ...) — будет мусор.

### 3.2 `kind`

Перечисление берётся из `enum class SourceKind`:

| Значение | Смысл | Когда слать |
|---|---|---|
| `0` | `infer` | Каждый кадр инференса. **Базовый режим для видео.** |
| `1` | `episode` | Не используется на видео-стороне (специфика аудио). |
| `2` | `emit` | Не используется. |

Для видео практически всегда `kind = 0`. Поле есть в пакете для
бинарной совместимости с аудио-нодами.

### 3.3 `logits[12]` — главное поле

**Это то, что реально использует временной аггрегатор мастера.**

- Тип: `int8_t`, диапазон `[-128, 127]`.
- Длина: ровно **12 элементов**, даже если у видео-модели меньше классов
  (лишние слоты заполни нулями) или больше (нужно сократить до 12).
- Семантика: «logits» — выход последнего fully-connected слоя ДО softmax,
  квантованные в int8. Главное — сохранить относительный порядок: класс
  с большим логитом должен быть «более вероятным».
- Квантование: например, если float-логиты лежат в `[-6.0, +6.0]`, делай
  `int8 = round(float * 21.0)`. Конкретный коэффициент не критичен —
  аггрегатор просто усреднит int8-логиты по последним кадрам в окне
  1200 мс и возьмёт argmax. Главное — одинаковый масштаб от кадра к
  кадру.
- Индексы классов в `logits[]` **должны совпадать** с таблицей `kVideoLabels`
  на мастере:

| Индекс | Метка (`kVideoLabels[i]`) |
|---|---|
| 0  | `person` |
| 1  | `face` |
| 2  | `car` |
| 3  | `bicycle` |
| 4  | `motorbike` |
| 5  | `dog` |
| 6  | `cat` |
| 7  | `bird` |
| 8  | `hand` |
| 9  | `stop_sign` |
| 10 | `unknown` |
| 11 | `no_obj` |

Если у твоей модели другой набор классов — либо переиндексируй на
видео-MCU (рекомендуется), либо отредактируй `kVideoLabels[]` в
`hash_kws_master_web/src/main.cpp` (тогда оба места останутся в синке).

### 3.4 `label`, `score`, `margin` — для UI-карточки

Не участвуют в усреднении (его делает аггрегатор по `logits[]`), но
показываются в правой колонке `Last raw frame` видео-карточки.

- `label`: индекс top-1 этого кадра, `0..11`. Маппинг — та же таблица
  `kVideoLabels`. Если `label >= 12`, мастер отвергнет пакет.
- `score`: «сырая» уверенность top-1, `0..255`. Можно посчитать как
  `score = clamp(round(softmax(logits)[top1] * 255), 0, 255)`. Или взять
  «как есть» из int8-логита top-1, сместив в `0..255`. Лишь бы число
  росло вместе с уверенностью — мастер только показывает его.
- `margin`: `top1_logit - top2_logit`, тоже в `0..255`. Тоже только
  показывается.

### 3.5 `recent_max`, `flags`, `seq`, `t_ms`, `invoke_ms`

- `recent_max`: аудио-специфика (пик RMS микрофона). Для видео ставь `0`.
- `flags`: пока не используется. Ставь `0`.
- `seq`: монотонный 16-bit счётчик отправленных кадров от этого видео-MCU.
  Может переполняться. Мастер показывает в UI, полезно для дебага потерь.
- `t_ms`: `millis()` в момент готовности кадра (твоего, не мастера).
  Аггрегатор сам ставит метку прибытия `host_arrival_ms = millis()` на
  стороне мастера, так что `t_ms` тоже только для дебага.
- `invoke_ms`: сколько ушло на инференс. Попадает в `lat` плашки на UI
  (`med`, `p95`). Если меришь не весь инференс, а его кусок — это твоё
  дело, но цифра должна отражать что-то осмысленное.

---

## 4. CRC-16

Алгоритм — **CRC-16/ARC (он же CRC-16/IBM)**.

- Полином: `0xA001` (отражённый `0x8005`).
- Начальное значение: `0xFFFF`.
- Без финального XOR.
- Считается по **первым 32 байтам** пакета — то есть `sizeof(HashKwsEspNowPacket) - sizeof(uint16_t)`.

Эталонная реализация (битовая, без таблицы — компактно, не лезет в
производительность):

```cpp
static uint16_t HashCrc16(const uint8_t* data, size_t len) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < len; ++i) {
    crc ^= static_cast<uint16_t>(data[i]);
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc & 1u) ? static_cast<uint16_t>((crc >> 1) ^ 0xA001u)
                       : static_cast<uint16_t>(crc >> 1);
    }
  }
  return crc;
}
```

Использование:

```cpp
HashKwsEspNowPacket pkt = { /* …заполнено… */ };
pkt.crc16 = HashCrc16(reinterpret_cast<const uint8_t*>(&pkt),
                      sizeof(pkt) - sizeof(pkt.crc16));
```

---

## 5. Что валидирует мастер

`ValidatePacket()` в `main.cpp` мастера отвергает пакет, если:

1. `len != sizeof(HashKwsEspNowPacket)` (т.е. не 34 байта).
2. `magic != 0x4B485731`.
3. `version != 1`.
4. `node == 0`, либо `node ∉ {1,2,3,4}`.
5. Для видео (`node == 4`): `label >= 12`.
6. `crc16` не сходится с тем, что мастер пересчитал по первым 32 байтам.

При любом провале счётчик `rejected` инкрементируется — это видно в
карточке `nodes` (`packets` vs `rejected`) и в `hash_evt kind=agg_diag`
строке в Serial.

---

## 6. Что делает мастер с принятым видео-пакетом

Зачем тебе это знать — чтобы понимать, какие гарантии нужно держать со
стороны отправителя.

Маршрутизация:

```
ESP-NOW recv → ValidatePacket → (node == 4) →
    update g_video (label/score/margin/seq/packets/latency ring)
    VideoAggSubmit(now_ms, pkt.logits)  // кладёт логиты в кольцо из 16 слотов
    mark g_video_dirty
```

В фоне `PollVideoAggregator()` пробегается по кольцу, оставляет только
слоты с `(now - ts) <= 1200ms`, усредняет логиты, берёт argmax → это
«сглаженный» результат, который рисуется крупно слева в видео-карточке
(`Smoothed (mean over 1.2 s window)`). Дополнительно — дедупликация по
лейблу с окном 800 мс, чтобы карточка не дёргалась.

Параметры на стороне мастера (`#define` в `main.cpp`, можно переопределять
через `build_flags`):

| `#define` | Default | Что значит |
|---|---|---|
| `HASH_KWS_VIDEO_NODE_ID` | `4` | Какой `node` считать видео-нодой. |
| `HASH_KWS_VIDEO_NUM_CLASSES` | `12` | Длина `kVideoLabels` и range-check `label`. |
| `HASH_KWS_VIDEO_STALE_MS` | `5000` | Через сколько ms тишины карточка станет «stale» в UI. |
| `HASH_KWS_VIDEO_AGG_WINDOW_MS` | `1200` | Окно усреднения логитов. |
| `HASH_KWS_VIDEO_AGG_RING_SIZE` | `16` | Сколько кадров держим в кольце. |
| `HASH_KWS_VIDEO_AGG_DEDUP_MS` | `800` | Дедуп сглаженного результата. |

Эти числа диктуют рекомендации к темпу отправки (§7).

---

## 7. Рекомендации по темпу отправки

- Кольцо аггрегатора **16 слотов**, окно **1200 мс**. Это даёт ровный
  «потолок» ≈ `16 / 1.2 ≈ 13.3` кадра/сек. Сверх него старые слоты
  начнут затираться раньше, чем выпадут по таймауту — окно станет
  «эффективно короче», но математика останется корректной.
- **Практический оптимум**: 5–10 fps. Хватает для сглаживания (5–10
  слотов в окне), не забивает эфир ESP-NOW.
- Меньше 1 fps — окно усреднения вырождается в один кадр, сглаживания
  фактически нет. UI начнёт показывать `voters = 1` или `0`, и в карточке
  будет «no frames in window» когда нода молчит дольше окна.
- Если у тебя «событийная» детекция (шлёшь только когда что-то
  обнаружили) — задумайся, имеет ли смысл сглаживать вообще; возможно
  стоит просто слать `kind=infer` периодически с low-confidence
  «no_obj», чтобы аггрегатор не висел на старом срабатывании.

---

## 8. Что мастер пишет в Serial после твоего пакета

Чтобы можно было однозначно понять, что пакет принят и распарсен.

На **каждый** успешно принятый видео-пакет (после `g_video_dirty` →
`loop()`):

```
hash_evt kind=video node=4 label=person score=214 margin=72 packets=147 seq=847
```

При **смене сглаженного top-1** (после дедупа):

```
hash_evt kind=video_fusion node=4 label=person score=1842 margin=724 voters=9 window_ms=1200 total=14
```

Раз в 3 секунды — общая агрегатор-диагностика по аудио (видео не
включает, но `packets`/`rejected` там общие):

```
hash_evt kind=agg_diag node=master resolves=... last_voters=... min_voters=2 window_ms=1200
```

При старте мастера (один раз):

```
hash_evt kind=video_agg_init node=master video_node=4 window_ms=1200 ring_size=16 num_classes=12 dedup_ms=800
```

Эти строки удобно ловить любым `pio device monitor` или скриптом-мостом
(`code/scripts/hash_kws_master_demux_bridge.py` в репо).

---

## 9. Reference Arduino sender (минимально работающий)

Целиком собирается под любую ESP32-плату с включённым WiFi (включая
ESP32-C3 / S3 / WROOM). Заменить нужно только `MASTER_MAC`, `MASTER_CHANNEL`
и логику инференса (`RunVideoInference()`).

```cpp
#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <string.h>

// ─── Настройка под твою установку ───────────────────────────────────────────
// MAC мастера: возьми из его Serial-лога строкой
// "hash_evt kind=espnow phase=init status=ok node=master mac=..."
static const uint8_t MASTER_MAC[6] = {0xAA,0xBB,0xCC,0xDD,0xEE,0xFF};
// Канал должен совпадать с каналом мастера (router channel в STA, либо
// HASH_KWS_ESPNOW_CHANNEL в AP-fallback).
static const uint8_t MASTER_CHANNEL = 1;

// ─── Mirror of HashKwsEspNowPacket (must stay byte-identical) ───────────────
constexpr uint32_t kHashKwsEspNowMagic   = 0x4B485731UL;  // "KHW1"
constexpr uint8_t  kHashKwsEspNowVersion = 1;
constexpr uint8_t  kVideoNodeId          = 4;
constexpr uint8_t  kNumClasses           = 12;

struct __attribute__((packed)) HashKwsEspNowPacket {
  uint32_t magic;
  uint8_t  version;
  uint8_t  node;
  uint16_t seq;
  uint32_t t_ms;
  uint16_t invoke_ms;
  uint8_t  kind;
  uint8_t  label;
  uint8_t  score;
  uint8_t  margin;
  uint8_t  recent_max;
  uint8_t  flags;
  int8_t   logits[kNumClasses];
  uint16_t crc16;
};
static_assert(sizeof(HashKwsEspNowPacket) == 34, "packet must be 34 bytes");

// ─── CRC-16/ARC (poly 0xA001, init 0xFFFF, reflected) ───────────────────────
static uint16_t HashCrc16(const uint8_t* data, size_t len) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < len; ++i) {
    crc ^= static_cast<uint16_t>(data[i]);
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc & 1u) ? static_cast<uint16_t>((crc >> 1) ^ 0xA001u)
                       : static_cast<uint16_t>(crc >> 1);
    }
  }
  return crc;
}

// ─── Заглушка инференса. Подменить на реальный вызов модели. ─────────────────
struct InferResult {
  int8_t  logits[kNumClasses];
  uint8_t top1_idx;
  uint8_t top1_score;
  uint8_t top1_margin;
  uint16_t invoke_ms;
};

static InferResult RunVideoInference() {
  InferResult out{};
  // TODO: твой реальный инференс. Здесь — фейковый person с высокой
  // уверенностью, чтобы видеть результат в UI.
  out.logits[0] = 100;   // person
  out.logits[1] = 40;    // face
  for (int i = 2; i < kNumClasses; ++i) out.logits[i] = -20;
  out.top1_idx    = 0;
  out.top1_score  = 220;
  out.top1_margin = 60;
  out.invoke_ms   = 30;
  return out;
}

// ─── ESP-NOW callback на статус доставки (опционально) ──────────────────────
static void OnDataSent(const wifi_tx_info_t* /*info*/, esp_now_send_status_t s) {
  // ESP_NOW_SEND_SUCCESS / ESP_NOW_SEND_FAIL. Логировать по желанию.
  (void)s;
}

// ─── Глобал ─────────────────────────────────────────────────────────────────
static uint16_t g_seq = 0;

void setup() {
  Serial.begin(115200);
  delay(100);

  // 1. WiFi в STA, без подключения к роутеру.
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  // 2. Фиксируем канал.
  esp_wifi_set_channel(MASTER_CHANNEL, WIFI_SECOND_CHAN_NONE);

  // 3. ESP-NOW.
  if (esp_now_init() != ESP_OK) {
    Serial.println("esp_now_init failed");
    while (true) delay(1000);
  }
  esp_now_register_send_cb(OnDataSent);

  // 4. Регистрируем мастер как peer.
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, MASTER_MAC, 6);
  peer.channel = MASTER_CHANNEL;
  peer.encrypt = false;
  peer.ifidx   = WIFI_IF_STA;
  if (esp_now_add_peer(&peer) != ESP_OK) {
    Serial.println("esp_now_add_peer failed");
  }

  Serial.printf("video sender ready, my mac=%s\n",
                WiFi.macAddress().c_str());
}

void loop() {
  // Темп: ~10 fps. Влезает в кольцо аггрегатора 16 / 1.2с.
  const uint32_t now_ms = millis();
  static uint32_t next_at = 0;
  if (now_ms < next_at) { delay(5); return; }
  next_at = now_ms + 100;

  InferResult r = RunVideoInference();

  HashKwsEspNowPacket pkt = {};
  pkt.magic      = kHashKwsEspNowMagic;
  pkt.version    = kHashKwsEspNowVersion;
  pkt.node       = kVideoNodeId;            // = 4
  pkt.seq        = g_seq++;
  pkt.t_ms       = now_ms;
  pkt.invoke_ms  = r.invoke_ms;
  pkt.kind       = 0;                       // infer
  pkt.label      = r.top1_idx;
  pkt.score      = r.top1_score;
  pkt.margin     = r.top1_margin;
  pkt.recent_max = 0;                       // не используется на видео
  pkt.flags      = 0;
  memcpy(pkt.logits, r.logits, kNumClasses);

  pkt.crc16 = HashCrc16(reinterpret_cast<const uint8_t*>(&pkt),
                        sizeof(pkt) - sizeof(pkt.crc16));

  esp_err_t err = esp_now_send(MASTER_MAC,
                               reinterpret_cast<const uint8_t*>(&pkt),
                               sizeof(pkt));
  if (err != ESP_OK) {
    Serial.printf("esp_now_send err=%d\n", (int)err);
  }
}
```

> **API note для Arduino-ESP32 3.x**: подпись `OnDataSent` использует
> `wifi_tx_info_t*`. Для 2.x — `const uint8_t* mac`. Если собираешь под
> старый core, поправь сигнатуру.

---

## 10. Чек-лист «отправил → получили»

Проверять по порядку, не пропуская:

1. На мастере в Serial есть строка
   `hash_evt kind=espnow phase=init status=ok node=master mac=…`.
   Скопируй MAC в `MASTER_MAC` на видео-MCU.
2. Мастер в Serial показывает свой `channel` (`hash_evt kind=wifi phase=sta_ok ... channel=N`
   или `phase=ap_fallback ... channel=N`). Этот же `N` поставь в
   `MASTER_CHANNEL` на видео-MCU.
3. После прошивки видео-MCU и старта `loop()`, в Serial мастера должны
   начать появляться строки `hash_evt kind=video node=4 label=… seq=…`.
   Если их нет, но `packets`/`rejected` растут — почти наверняка
   расходится `magic`/`version`/`crc16` или `node != 4`.
4. В дашборде в видео-карточке статус становится `● online`,
   `Last raw frame` показывает твою метку, через 1.2 с заполняется
   `Smoothed (mean over 1.2 s window)` слева.
5. Если карточка осталась `○ offline` через 6 секунд после первого
   отправленного пакета — пакеты не доходят до мастера. Проверь канал
   (самая частая причина).

---

## 11. Сводный «контракт» в одну строку

> Видео-MCU шлёт мастеру **34-байтовый ESP-NOW пакет** структуры
> `HashKwsEspNowPacket` с `magic=0x4B485731`, `version=1`, `node=4`,
> `label ∈ [0..11]`, валидным `int8 logits[12]` и `CRC-16/ARC` по первым
> 32 байтам. Канал и MAC мастера берутся из его Serial-лога. Темп
> 5–10 fps. Остальное мастер сделает сам.

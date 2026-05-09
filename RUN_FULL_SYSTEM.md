# Запуск полной системы Hash KWS — 4 платы + веб-дашборд

Пошаговое руководство: прошить 3 inference-ноды и мастер с embedded дашбордом,
проверить что всё работает вместе.

---

## Что понадобится

| Железо | Количество |
|---|---|
| ESP32-S3 Dev Module (N16R8 или N8R8) | 3 (inference-ноды) |
| **ESP32 WROOM-32** | 1 (мастер — без микрофона) |
| INMP441 / MEMS I2S микрофон | 3 (по одному на каждую inference-ноду) |
| USB-кабель (micro-USB или USB-C по модели) | 4 |
| 2.4 ГГц WiFi роутер | 1 (или без роутера — AP-режим, читай ниже) |

---

## Часть 0 — Установка ПО (один раз)

### 0.1 Arduino IDE

Скачай и установи **Arduino IDE 2.3+** с https://www.arduino.cc/en/software

### 0.2 ESP32 board package

`Tools → Board Manager` → найди **"esp32 by Espressif Systems"** → установи версию **3.0.x** (или выше, но 3.x обязательно).

### 0.3 TFLite библиотека (для inference-нод)

`Tools → Manage Libraries` → найди **"TensorFlowLite_ESP32"** → установи.

### 0.4 Библиотеки для мастера

`Tools → Manage Libraries` → установи ВСЕ три:

| Библиотека | Автор | Важно |
|---|---|---|
| **ESP Async WebServer** | mathieucarbou | Именно этот форк — `me-no-dev` не компилируется под core 3.x |
| **AsyncTCP** | mathieucarbou | Зависимость предыдущей |
| **ArduinoJson** | bblanchon | Версия ≥ 7.0 |

> В Library Manager может быть две записи "ESP Async WebServer" — выбирай ту, у которой автор **mathieucarbou**.

### 0.5 Настройки платы (Tools)

**Мастер (ESP32 WROOM-32):**

| Параметр | Значение |
|---|---|
| Board | **ESP32 Dev Module** |
| CPU Frequency | **240 MHz** |
| Flash Size | **4MB** |
| Partition Scheme | Default 4MB with spiffs |
| Upload Speed | 921600 |

> Serial-монитор у WROOM-32 работает через USB-serial чип (CP2102 / CH340) — скорость **115200 baud**, USB CDC не нужен.

**Inference-ноды (ESP32-S3 Dev Module):**

| Параметр | Значение |
|---|---|
| Board | **ESP32S3 Dev Module** |
| USB CDC On Boot | **Enabled** |
| CPU Frequency | **240 MHz** |
| Flash Mode | QIO 80MHz |
| Flash Size | **16MB** (или 8MB по модели) |
| PSRAM | **OPI PSRAM** |
| Partition Scheme | Default 4MB with spiffs |
| Upload Speed | 921600 |

---

## Часть 1 — Узнать канал WiFi

> Пропусти этот шаг если будешь использовать **AP-режим** (мастер создаёт свою точку доступа). В AP-режиме по умолчанию канал 1, все inference-ноды компилируй с `-DHASH_KWS_ESPNOW_CHANNEL=1`.

На Windows:

```
netsh wlan show interfaces
```

В выводе найди строку `Channel` — запомни это число (например `6`).
Оно пойдёт в `HASH_KWS_ESPNOW_CHANNEL=6` на ВСЕХ 4 платах.

На Mac:

```
/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport -I | grep channel
```

---

## Часть 2 — Подготовить и прошить мастер (4-я плата, без микрофона)

Рекомендую прошить мастер ПЕРВЫМ — сразу проверишь дашборд и WiFi.

### 2.1 Скопировать файлы агрегатора в папку скетча

Открой PowerShell в корне репозитория и выполни:

```powershell
$dst = "code\firmware\hash_kws_master_web"
$agg = "code\firmware\hash_kws_aggregator"
$par = "code\deploy\hash_ensemble\reports"

Copy-Item "$agg\hash_ensemble_aggregator.h"   $dst -Force
Copy-Item "$agg\hash_ensemble_aggregator.cpp"  $dst -Force
Copy-Item "$par\aggregator_params.h"           $dst -Force
```

Проверь: в `code/firmware/hash_kws_master_web/` должно быть 6 файлов:
`hash_kws_master_web.ino`, `web_page.h`, `hash_ensemble_aggregator.h`,
`hash_ensemble_aggregator.cpp`, `aggregator_params.h`, `README.md`.

### 2.2 Настроить WiFi и канал

Открой `code/firmware/hash_kws_master_web/hash_kws_master_web.ino`.
Найди в самом верху:

```cpp
#define WIFI_SSID     "YourSSID"
#define WIFI_PASSWORD "YourPassword"
```

Замени на свои данные. Если используешь AP-режим (без роутера) — оставь как есть,
мастер создаст точку `KWS-Master` / `kwsmaster1` сам.

Если роутер есть, добавь или измени (подставь свой канал):

```cpp
#define HASH_KWS_ESPNOW_CHANNEL 6   // ← твой канал из Части 1
```

### 2.3 Открыть скетч в Arduino IDE

`File → Open` → выбери `code/firmware/hash_kws_master_web/hash_kws_master_web.ino`

### 2.4 Build flags мастера

В `Tools → Build flags` (или вручную в начале `.ino` через `#define`):

```
-DHASH_KWS_ESPNOW_CHANNEL=6
-DHASH_KWS_AGG_MODE=0
```

`HASH_KWS_AGG_MODE=0` — mean_logits (рекомендуется). Не трогай если не знаешь зачем.

### 2.5 Подключить плату и прошить

- Подключи **4-ю плату (без микрофона)** по USB
- `Tools → Port` → выбери её COM-порт
- Нажми **Upload** (→ или Ctrl+U)

### 2.6 Проверить Serial Monitor

`Tools → Serial Monitor`, скорость **115200 baud**.

Должно появиться:

```
hash_evt kind=boot node=master role=master_web channel=6 agg_mode=0
hash_evt kind=wifi phase=sta_connect ssid=YourSSID
....
hash_evt kind=wifi phase=sta_ok ip=192.168.1.105 channel=6
hash_evt kind=espnow phase=init status=ok node=master mac=XX:XX:XX:XX:XX:XX
hash_evt kind=mdns hostname=micro_network.local
hash_evt kind=http phase=ready ip=192.168.1.105 url=http://192.168.1.105/ mdns=http://micro_network.local/
```

**Запомни IP** из последней строки.

Если вместо `sta_ok` видишь `ap_fallback` — мастер не подключился к роутеру и поднял свою точку `KWS-Master`. В этом случае подключись к ней ноутбуком и используй IP `192.168.4.1`.

### 2.7 Открыть дашборд

В браузере (на том же WiFi что и мастер):

```
http://micro_network.local
```

или по прямому IP:

```
http://192.168.1.105
```

Должна открыться страница с заголовком **Hash KWS Master** и статусом **● live**.
Три тайла нод пока серые — пакетов ещё нет.

---

## Часть 3 — Прошить inference-ноду №1 (variant `ens_a`, NODE_ID=1)

### 3.1 Подключить микрофон INMP441 к плате №1

| INMP441 | ESP32-S3 |
|---|---|
| VDD | 3V3 |
| GND | GND |
| L/R | GND |
| WS | GPIO 16 |
| SCK | GPIO 15 |
| SD | GPIO 17 |

### 3.2 Активировать вариант `ens_a`

```powershell
pwsh -File code\scripts\select_hash_kws_variant.ps1 -Variant ens_a
```

Вывод должен быть:
```
copied hash_model_data.cpp
copied hash_model_data.h
...
Active hash_kws_runtime now reflects variant: ens_a
```

### 3.3 Открыть скетч inference-ноды

`File → Open` → `code/firmware/micro_speech_sim/micro_speech/micro_speech.ino`

### 3.4 Build flags ноды №1

В `Tools → Build flags`:

```
-DHASH_KWS_USE_ESPNOW=1
-DHASH_KWS_NODE_ID=1
-DHASH_KWS_ESPNOW_CHANNEL=6
-DHASH_KWS_TELEMETRY_STREAM=1
```

Если IDE не позволяет задать build flags — вставь в самый верх `micro_speech.ino`:

```cpp
// ← добавь ЭТИ строки самым первым делом
#define HASH_KWS_USE_ESPNOW      1
#define HASH_KWS_NODE_ID         1
#define HASH_KWS_ESPNOW_CHANNEL  6
#define HASH_KWS_TELEMETRY_STREAM 1
```

### 3.5 Прошить

- `Tools → Port` → выбери COM-порт платы №1
- **Upload**

### 3.6 Проверить Serial Monitor ноды №1

```
hash_evt kind=boot node=1 cpu_mhz=240 esp_nn=1 ...
hash_evt kind=boot_post node=1 esp_nn_active=1
hash_evt kind=espnow phase=init status=ok
```

- `esp_nn_active=1` — SIMD путь активен (~230 ms инференс). Хорошо.
- `esp_nn_active=0` — проверь PSRAM = OPI PSRAM в Tools.

Скажи слово в микрофон — должны пойти строки:
```
hash_evt kind=infer node=1 label=0 top1=yes score=... ...
hash_evt kind=emit  node=1 label=0 ...
```

---

## Часть 4 — Прошить inference-ноды №2 и №3

Повтори **Часть 3** полностью для каждой ноды, меняя только две вещи:

### Нода №2 (variant `ens_b`)

```powershell
pwsh -File code\scripts\select_hash_kws_variant.ps1 -Variant ens_b
```

Build flags:
```
-DHASH_KWS_USE_ESPNOW=1
-DHASH_KWS_NODE_ID=2
-DHASH_KWS_ESPNOW_CHANNEL=6
-DHASH_KWS_TELEMETRY_STREAM=1
```

Boot log должен начинаться с `hash_evt kind=boot node=2 ...`

### Нода №3 (variant `ens_c`)

```powershell
pwsh -File code\scripts\select_hash_kws_variant.ps1 -Variant ens_c
```

Build flags:
```
-DHASH_KWS_USE_ESPNOW=1
-DHASH_KWS_NODE_ID=3
-DHASH_KWS_ESPNOW_CHANNEL=6
-DHASH_KWS_TELEMETRY_STREAM=1
```

Boot log: `hash_evt kind=boot node=3 ...`

> После прошивки ноды №2/№3 нода №1 **не теряет прошивку** — модель уже во flash.

---

## Часть 5 — Итоговый тест

### 5.1 Все 4 платы включить одновременно

Подключи все по USB (к зарядкам или хабу — питание важнее чем Serial).

### 5.2 Открыть дашборд мастера

```
http://micro_network.local
```

### 5.3 Произнести команду

Поставь 3 микрофонных ноды рядом, скажи одно из:
**yes, no, up, down, left, right, on, off, stop, go**

**Ожидаемый результат на дашборде:**
- Все 3 тайла нод получают зелёную левую границу
- В тайлах появляется распознанная метка (крупным шрифтом, с цветом)
- Список "Fusion decisions" пополняется строкой с командой, score, voters=3
- Счётчик `fusion` растёт
- Синий LED мастера (GPIO2) коротко мигает

**В Serial мастера:**
```
hash_evt kind=fusion node=master label=yes score=4736 margin=2048 voters=3 mode=0 packets=N rejected=0
```

### 5.4 Проверить stale-состояние

Замолчи на 30 секунд — все три тайла должны пожелтеть (граница становится amber).

### 5.5 Проверить два устройства

Открой дашборд на телефоне и на ноутбуке одновременно — оба должны видеть
одно и то же в реальном времени.

---

## Чек-лист

| # | Проверка | Ожидаемо |
|---|---|---|
| 1 | Serial мастера: `sta_ok` или `ap_fallback` | да |
| 2 | Serial мастера: `mdns hostname=micro_network.local` | да |
| 3 | Браузер: `http://micro_network.local` открывается | да |
| 4 | Статус дашборда: `● live` | да |
| 5 | Boot log каждой ноды: `esp_nn_active=1` | да |
| 6 | Boot log каждой ноды: `espnow phase=init status=ok` | да |
| 7 | Команда → тайлы зеленеют, label появляется | да |
| 8 | Команда → fusion decisions пополняется | да |
| 9 | `voters` в fusion ≥ 2 (идеально = 3) | да |
| 10 | `rejected` на мастере растёт медленно или = 0 | да |
| 11 | 30 с тишины → тайлы желтеют | да |
| 12 | 2 браузера одновременно — оба синхронны | да |

---

## Типичные проблемы

### Дашборд не открывается по micro_network.local

- mDNS бывает ненадёжным на Windows — попробуй прямой IP из Serial.
- На Android mDNS работает через браузер Chrome/Firefox, но не всегда.
- Убедись, что ноутбук и мастер на **одном WiFi** (или ты подключён к `KWS-Master` в AP-режиме).

### Статус дашборда `○ disconnected` после загрузки страницы

WebSocket не подключился. Возможно:
- Браузер открыт по HTTP, но WebSocket блокируется корпоративным файрволом — попробуй другую сеть.
- Мастер перезагрузился. Обнови страницу — она авто-реконнектится через 1 сек.

### `fusion` нет, хотя `infer` на нодах идёт

Ноды не достигают мастера по ESP-NOW:
1. Убедись что канал совпадает у всех 4 плат (`HASH_KWS_ESPNOW_CHANNEL` одинаковый).
2. В STA-режиме канал диктует роутер — inference-ноды должны быть скомпилированы с **тем же** каналом что печатает мастер в `sta_ok`.
3. В Serial inference-ноды должна быть строка `hash_evt kind=espnow phase=tx status=ok`. Если `status=fail` — channel mismatch.

### `esp_nn_active=0` на inference-ноде (ESP32-S3)

Время инференса будет ~1–2 сек вместо 230 мс. Причина — неверный PSRAM.
- `Tools → PSRAM → OPI PSRAM` (только для ESP32-S3 inference-нод)
- Пересобери и перепрошей.

> Это не относится к мастеру (WROOM-32): у него нет PSRAM и inference не запускается.

### `voters=2` всегда (одна нода не попадает в окно)

Одна из нод запаздывает (медленный инференс или WiFi помехи).
- Проверь `esp_nn_active=1` на ней.
- Попробуй увеличить окно агрегатора: в `hash_kws_master_web.ino` измени
  `#define HASH_KWS_AGG_WINDOW_MS 1200` на `1800` и перепрошей мастер.

### `packets=0, rejected>N` на мастере

Пакеты приходят, но не проходят валидацию (CRC / magic / version).
- Убедись что inference-ноды и мастер собраны из **одной и той же** версии кода.
- Перекомпилируй все 4 платы заново.

### AP-режим: дашборд на http://192.168.4.1

Если SSID/пароль не заданы или роутер недоступен — мастер поднимает точку доступа.
- Подключись к WiFi `KWS-Master` (пароль `kwsmaster1`)
- Открой `http://192.168.4.1`
- mDNS в этом режиме тоже работает: `http://micro_network.local` если OS поддерживает.
- Inference-ноды **должны** быть скомпилированы с `HASH_KWS_ESPNOW_CHANNEL=1`
  (канал AP по умолчанию).

---

## Краткая шпаргалка каналов

```
Роутер на канале X  →  все 4 платы: -DHASH_KWS_ESPNOW_CHANNEL=X
AP-режим            →  все 4 платы: -DHASH_KWS_ESPNOW_CHANNEL=1
```

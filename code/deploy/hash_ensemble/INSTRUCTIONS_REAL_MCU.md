# Распределённый KWS на 4 ESP32 — пошаговый гайд

Полный сценарий запуска: 3 микрофонных узла с разными моделями (`ens_a`,
`ens_b`, `ens_c`) + 1 master‑агрегатор. Все 4 платы общаются по ESP‑NOW.
Хост подключается только к master по USB и читает Serial — никакого WiFi.

> Все пути относительно корня репозитория.

---

## Bill of Materials

| Что | Сколько | Зачем |
|---|---|---|
| ESP32‑S3 dev board (S3‑WROOM‑1 N16R8 или N8R8) | 4 | 3 inference + 1 master |
| INMP441 / MEMS I2S микрофон | 3 | по одному на каждый inference‑узел |
| USB‑C / micro‑USB кабель | 4 | питание + flash + Serial |
| Перемычки, breadboard | по вкусу | для подключения микрофонов |
| Опционально: 4‑портовый USB‑hub | 1 | если на хосте мало портов |

Master может быть любой ESP32‑S3 без микрофона (нужен только Wi‑Fi MAC + USB).
Onboard RGB светодиод (LED_RGB_PIN=48 на типовых S3‑боардах) у master сигнализирует распознанной командой.

### Распиновка INMP441 → ESP32‑S3

| INMP441 | ESP32‑S3 | Роль |
|---|---|---|
| VDD | 3V3 | Питание |
| GND | GND | Земля |
| L/R | GND | Левый канал (соответствует `I2S_CHANNEL_FMT_ONLY_LEFT`) |
| WS | GPIO 16 | Word‑select (LRCLK) |
| SCK | GPIO 15 | Bit‑clock (BCLK) |
| SD | GPIO 17 | Data out (микрофон → MCU) |

Источник распиновки: `code/firmware/micro_speech_sim/micro_speech/audio_provider.cpp:96‑101`. Если у платы конфликт по этим GPIO — изменить там и пересобрать.

---

## Software Prerequisites (хост)

1. **Arduino IDE 2.x** (рекомендую 2.3+).
2. **ESP32 board package 3.x** (Tools → Board Manager → "esp32 by Espressif Systems" → версия 3.0.x). Сергеева ветка использовала PlatformIO; у нас Arduino IDE.
3. Открыть Tools и выставить:
   - Board: **ESP32S3 Dev Module**
   - USB CDC On Boot: **Enabled** (для Serial.print через нативный USB)
   - CPU Frequency: **240 MHz**
   - Flash Mode: **QIO 80MHz**
   - Flash Size: **16MB** (или 8MB, в зависимости от платы)
   - PSRAM: **OPI PSRAM** (важно для tensor arena)
   - Partition Scheme: **Default 4MB with spiffs** (или 16M Flash для N16R8)
   - Upload Speed: **921600**
4. Python ≥ 3.10 в PATH (для `select_hash_kws_variant.ps1`, в нём вызовов python нет, но скрипт PowerShell — должен запускаться `pwsh` или `powershell`).

---

## Phase 0. Preflight на хосте

```powershell
# 1) Убедимся, что bundles на месте
python code\scripts\hash_ensemble_sim.py verify --bundles code\deploy\hash_ensemble\models\ens_a\student_best.pt code\deploy\hash_ensemble\models\ens_b\student_best.pt code\deploy\hash_ensemble\models\ens_c\student_best.pt --variant-names ens_a ens_b ens_c --device cpu
```

Должно напечатать `All bundles loaded and forward-pass clean.` с recorded top1 ≈ 0.9304 / 0.9325 / 0.9306.

```powershell
# 2) Проверим C++ агрегатор
python code\firmware\hash_kws_aggregator\test_aggregator_match.py
```

Должно напечатать `All cases match.`

```powershell
# 3) Запишем COM‑порты будущих узлов (после подключения по USB)
#    Открыть в Arduino IDE → Tools → Port — увидеть 4 порта (например COM3..COM6).
```

---

## Phase 1. Прошить inference‑узел №1 (variant `ens_a`)

### 1.1 Подключить микрофон INMP441 к плате №1 (см. распиновку выше)

### 1.2 Активировать вариант `ens_a`

```powershell
pwsh -File code\scripts\select_hash_kws_variant.ps1 -Variant ens_a
```

Что делает: копирует `hash_model_data.cpp` + `hash_model_export_metadata.json` из `code/firmware/hash_kws_runtime_ens_a/` в `code/firmware/hash_kws_runtime/`. Sketch всегда компилирует именно из `hash_kws_runtime/`.

Проверка: `git status code/firmware/hash_kws_runtime/hash_model_data.cpp` должен показать модификацию (или открыть файл — в первой же строке комментарий `// Generated from bundle: ... ens_a_s13`).

### 1.3 Открыть скетч в Arduino IDE

`code/firmware/micro_speech_sim/micro_speech/micro_speech.ino`

### 1.4 Build flags

В `Tools → Manage Libraries` ничего не нужно ставить (esp_now / WiFi уже в core 3.x; esp‑nn идёт с пакетом).

В **`Tools → Build flags`** (или скопировать в `platform.local.txt` рядом с скетчем — оба пути работают):

```text
-DHASH_KWS_USE_ESPNOW=1
-DHASH_KWS_NODE_ID=1
-DHASH_KWS_ESPNOW_CHANNEL=1
-DHASH_KWS_TELEMETRY_STREAM=1
```

Если IDE не даёт указать build flags из меню — открыть файл `micro_speech.ino` и в самом верху поставить:

```cpp
#define HASH_KWS_USE_ESPNOW 1
#define HASH_KWS_NODE_ID 1
#define HASH_KWS_ESPNOW_CHANNEL 1
#define HASH_KWS_TELEMETRY_STREAM 1
```

(тогда не нужно ничего лезть в build flags).

### 1.5 Подключить плату №1, выбрать её COM‑порт, **Upload**

После загрузки откроем Serial Monitor (115200 baud). В первые 2 секунды должны появиться строки:

```
hash_evt kind=boot node=1 cpu_mhz=240 esp_nn=1 int_mac_pw=1 ...
hash_evt kind=boot_post node=1 esp_nn_active=1
hash_evt kind=espnow phase=init status=ok
```

Если `esp_nn_active=0` — Prepare() не материализовал веса (см. журнал `2026-04-25_perf_pass4_002.yaml`); проверить PSRAM=OPI PSRAM.

Если `phase=init status=fail` — проверить, что собрано с `HASH_KWS_USE_ESPNOW=1`.

**Сохранить MAC платы №1** из лога — пригодится для дебага (master печатает MAC отправителя пакета только в Wireshark, но если что‑то идёт не так, сверка MAC решает).

Скажите команду в микрофон — должны пойти строки `hash_evt kind=infer ... top1=...` и `hash_evt kind=emit ...`. Это значит модель работает локально.

---

## Phase 2. Прошить inference‑узлы №2 (variant `ens_b`) и №3 (variant `ens_c`)

Повторить **полностью** Phase 1, изменив две вещи:

### Узел 2

```powershell
pwsh -File code\scripts\select_hash_kws_variant.ps1 -Variant ens_b
```

```cpp
#define HASH_KWS_NODE_ID 2
```

Подключить плату №2 → выбрать её COM → Upload. Boot log должен начинаться с `hash_evt kind=boot node=2 ...`.

### Узел 3

```powershell
pwsh -File code\scripts\select_hash_kws_variant.ps1 -Variant ens_c
```

```cpp
#define HASH_KWS_NODE_ID 3
```

Подключить плату №3 → выбрать её COM → Upload. Boot log должен начинаться с `hash_evt kind=boot node=3 ...`.

> **Важно:** после загрузки узла 1 не надо его трогать на узлах 2/3. PS1‑свитчер просто перезаписывает `hash_kws_runtime/`, узлу 1 это уже всё равно — у него прошивка во flash содержит свои веса.

---

## Phase 3. Прошить master (4‑я плата, без микрофона)

### 3.1 Открыть скетч

`code/firmware/hash_kws_master/hash_kws_master.ino`

В этой папке уже лежат:
- `hash_ensemble_aggregator.h`
- `hash_ensemble_aggregator.cpp`
- `aggregator_params.h` (калиброванные T и weights из обучения)

Arduino IDE подхватит их автоматически (они в той же папке, что и .ino).

### 3.2 Build flags / макросы в начале скетча

```cpp
#define HASH_KWS_ESPNOW_CHANNEL 1     // должен совпадать с inference‑узлами!
#define HASH_KWS_AGG_MODE 0           // 0=mean_logits (рекомендуется)
                                      // 1=temperature_scaled
                                      // 2=learned_weights
```

### 3.3 Подключить master, выбрать COM, **Upload**

### 3.4 Открыть Serial Monitor (master)

Должны появиться:

```
hash_evt kind=boot node=master role=master_aggregator channel=1 agg_mode=0
hash_evt kind=espnow phase=init status=ok node=master mac=XX:XX:XX:XX:XX:XX
```

Master теперь сидит в receive‑mode и ждёт пакеты от 3 inference‑узлов.

---

## Phase 4. End‑to‑end проверка

### 4.1 Включить все 4 платы одновременно

Все по USB подключены к хосту (или master к хосту, остальные через USB‑зарядки — питание им нужно одинаковое).

### 4.2 Открыть Serial Monitor master'а на 115200

### 4.3 Произнести команду в микрофоны

Поставить 3 микрофонных узла рядом и сказать одно из 10 ключевых слов: **`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`** (`unknown` и `silence` — служебные).

Master должен вывести строку:

```
hash_evt kind=fusion node=master label=yes score=4736 margin=2048 voters=3 mode=0 packets=137 rejected=0
```

Расшифровка:
- `label` — финальная метка ансамбля (`mean_logits` от 3 узлов).
- `score` — top‑1 логит после агрегации в Q8.8 (делить на 256 → реальное значение).
- `margin` — top1−top2 в Q8.8.
- `voters` — сколько узлов попали в окно 1.2 сек (требуется ≥ 2).
- `mode` — какой режим агрегации использовался (0/1/2).
- `packets` — сколько валидных пакетов master принял всего.
- `rejected` — сколько пакетов отбросил (CRC / magic / version mismatch).

RGB светодиод на master меняет цвет на цвет распознанной команды.

### 4.4 Проверить per‑узел

Параллельно открыть Serial Monitor для одного из inference‑узлов — там должно идти `hash_evt kind=infer ... top1=...` для каждой произнесённой команды.

### 4.5 Опционально: прицепить host‑dashboard

Master печатает в Serial те же `hash_evt kind=fusion ...`, что и старый симметричный режим. Чтобы это увидеть в FastAPI dashboard:

```powershell
python code\scripts\run_distributed_demo.py --port COM_OF_MASTER
```

В другом окне:

```powershell
python run_dashboard.py
```

Открыть `http://127.0.0.1:8765/` — fusion‑строка наполнится решениями master'а.

---

## Чек‑лист первой проверки

| Проверка | Ожидаемо |
|---|---|
| 4× `kind=boot` строки во всех Serial | да |
| 4× `kind=espnow phase=init status=ok` | да |
| Master: `kind=fusion` после произнесения команды | да |
| `voters` на master ≥ 2 | да |
| `rejected` на master растёт медленно или = 0 | да |
| Fusion‑label согласуется с большинством `top1=` от узлов | да |

---

## Troubleshooting

### Master печатает только `kind=boot`, но `kind=fusion` нет

Проверить:
1. Все 4 платы на одном `HASH_KWS_ESPNOW_CHANNEL`. Если хоть один узел на `=2`, а master на `=1` — пакеты не дойдут.
2. INMP441 действительно слышит. На узле‑источнике в Serial должны идти `hash_evt kind=infer ... speech=1 ...`. Если `speech=0` всегда — проверить пайку микрофона / уровень сигнала / `recent_max`.
3. На inference‑узле в Serial должны идти `hash_evt kind=espnow phase=tx status=ok` после каждого emit. Если `status=fail` — channel mismatch или магистрали Wi‑Fi заняты.

### Master: `packets=0`, `rejected>0`

CRC / magic / version не сходятся. Чаще всего значит, что ты собрал inference и master с **разной версией** `HashKwsEspNowPacket` (разный layout struct). Перекомпилировать master заново — он берёт layout из своего же `hash_kws_master.ino`, который синхронизирован с `micro_speech.ino` на момент написания этой инструкции.

### Один из узлов даёт неверные метки

Проверить `code/firmware/hash_kws_runtime/hash_model_export_metadata.json` **перед** компиляцией каждого узла. Поле `experiment.tag` должно содержать имя варианта: `..._hn95_ens_a_s13`, `..._hn95_ens_b_s29`, `..._hn95_ens_c_s47`. Если ошибся вариантом — сделать `select_hash_kws_variant.ps1 -Variant ens_X` ещё раз и перепрошить.

### Latency высокая (`invoke_ms > 500`)

Проверить:
1. PSRAM включён (`Tools → PSRAM = OPI PSRAM`).
2. `esp_nn_active=1` в `kind=boot_post`. Если `=0` — материализация весов не сработала, см. журнал `2026-04-25_perf_pass4_002.yaml`.
3. CPU frequency 240 MHz.

Цель — `invoke_ms ≈ 230` (pass 4 SIMD path).

### `kind=fusion voters=2` всегда (не 3)

Один из узлов не успевает попасть в окно 1.2 сек. Проверить, что у проблемного узла нет ESP‑NOW таймаутов в Serial. Если у конкретной платы хронически 1‑2 voter — возможно у неё хуже Wi‑Fi (антенна) — переставить ближе к master или поднять окно `HASH_KWS_AGG_WINDOW_MS` до 1800.

### Хочется сменить режим агрегации на лету

Перепрошить только master с другим `HASH_KWS_AGG_MODE`:
- 0 = `mean_logits` (рекомендуется)
- 1 = `temperature_scaled` (читает T из `aggregator_params.h`)
- 2 = `learned_weights` (читает w из `aggregator_params.h`)

На наших данных все три дают почти одинаковую top1 (разница 0.06 п.п.).

---

## Что показать на защите

1. 4 платы, объясняем роль каждой (3 inference + master).
2. Открытый Serial Monitor master'а — видны `kind=fusion` строки с разными командами.
3. График из `code/deploy/hash_ensemble/reports/aggregator_comparison.png` — `mean_logits ensemble = 0.9423` против `best single ens_b = 0.9323` (+1.06 п.п.) и oracle 0.9661.
4. Тепловая карта `pairwise_disagreement.png` — модели реально разные (~6% несогласий между парами), не схлопнулись в одну.
5. JSON `ensemble_results.json` — per‑class disagreement: труднее всего `no/go/down`, легче всего `silence`.

---

## Где смотреть, если что‑то непонятно

- План работы и принятые решения: `notes/Journal/2026-05-09_hash_ensemble_plan.md`
- Журнал реализации: `notes/Journal/2026-05-09_hash_ensemble_implementation_001.yaml`
- Журнал deploy: `notes/Journal/2026-05-09_hash_ensemble_deploy_001.yaml`
- C++ агрегатор: `code/firmware/hash_kws_aggregator/README.md`
- Master sketch: `code/firmware/hash_kws_master/README.md`

# Прошивка v4-ансамбля (с wire-fix) — гайд

Изолированный firmware-комплект для распределённого hash-KWS ансамбля **v4** с
исправленным форматом провода. Лежит отдельно от рабочего образца
(`WORKED_ENSEMBLE` / `code/firmware/`) — **его не трогает**.

## Что нового vs WORKED

1. **Новые модели v4** (`hash_kws_runtime_ens_{a,b,c}/`) — обучены робастным
   рецептом, диверсия ансамбля от seed (m1=13, m2=29, m3=47). Архитектура у всех
   трёх одинаковая (compact ≈ 8.8K параметров, 12 классов, вход 40×49).
2. **Wire-fix.** Раньше нода слала на ESP-NOW центрированный softmax
   (`255·p − 128`), а мастер трактовал его как логиты — из-за чего
   `temperature_scaled` / `learned_weights` вырождались. Теперь:
   - нода шлёт **истинные логиты**, квантованные `q = round(logit / 0.25)`
     (clip ±127) в `packet.logits`;
   - мастер **деквантует** `×0.25` перед усреднением/температурой → агрегатор
     работает в реальном логит-домене (совпадает с обучением и ноут-стендом);
   - `HASH_KWS_AGG_NOISE_BOOST` снижен `24 → 0.5` (логит-домен; 24 утопил бы все
     команды в unknown/silence).
   - `output_scores` (centered-softmax) на ноде **не тронут** — нода по-прежнему
     использует его для top1/score/эпизодов; на провод идёт отдельный
     квантованный вектор логитов.

## Железо

- **3× ESP32-S3 (N16R8)** — аудио-ноды, микрофон INMP441 (I2S).
- **1× ESP32-S3 (N16R8)** — master-агрегатор, без микрофона.
- (опц.) камера-нода — **не меняется**, прошивается из WORKED как есть
  (пакет/триггер не трогали, согласована).

## ⚠️ Tools-опции Arduino IDE (ВСЁ обязательно)

| Опция | Значение |
|---|---|
| Board | ESP32S3 Dev Module |
| Flash Size | `16MB (128Mb)` |
| Flash Mode | `QIO 80MHz` |
| **PSRAM** | **`OPI PSRAM`** ← без этого упадёт аудио-буфер |
| Partition Scheme | `Huge APP (3MB No OTA/1MB SPIFFS)` |
| Erase All Flash Before Sketch Upload | `Enabled` (при первой прошивке) |

> **Почему PSRAM критичен:** модель + scratch (~131 КБ) рантайм аллоцирует
> сначала из PSRAM. Если PSRAM выключен (`free_psram=0` в boot-логе), всё ложится
> во внутреннюю SRAM, её фрагментирует, и I2S ring-buffer (131072 байт) не влезает:
> `RINGBUF: rb_init failed ... / Feature generation failed`.
>
> **Cold-boot race S3:** при холодном включении OPI-PSRAM иногда не успевает
> подняться (`octal_psram: chip is not connected`). Нажми RST 5–10 раз или поставь
> конденсатор 10–22 µF на 3V3/GND. Признак успеха: `free_psram` ≈ 8 МБ.

## Прошивка аудио-нод (×3)

Перед каждой нодой выбери её вариант модели (скрипт кладёт модель в активный
`code/firmware_v4/hash_kws_runtime/`, откуда её включает скетч):

```powershell
pwsh -File code\scripts\select_hash_kws_variant_v4.ps1 -Variant ens_a   # node 1
# затем ens_b для node 2, ens_c для node 3
```

Открой `code/firmware_v4/micro_speech_v4/micro_speech/micro_speech.ino`, выставь
`#define HASH_KWS_NODE_ID` под номер ноды (1 / 2 / 3) и прошей. Маппинг:

| Вариант | Узел | hash_seed |
|---|---|---|
| ens_a | node 1 | 13 |
| ens_b | node 2 | 29 |
| ens_c | node 3 | 47 |

## Прошивка master

Открой `code/firmware_v4/hash_kws_master_web_v4/hash_kws_master_web.ino`, те же
Tools-опции, прошей. Модель мастеру не нужна (только приём ESP-NOW + фьюжн + дашборд).

## Проверка (Serial)

Нода (успех):
```
hash_mem phase=after_hash_alloc free_8bit=... free_psram=<БОЛЬШОЙ, не 0>
hash_kws ready: classes=12 input=40x49 esp_nn_active=1
hash_evt kind=infer node=N ... top1=<слово> ...
```
— и **нет** строк `RINGBUF ... failed` / `Feature generation failed`.

Master:
```
hash_evt kind=fusion node=master label=<слово> voters=N ...
```

Дашборд: WiFi `esp32-hash-master` / `12345678` → `http://192.168.4.1/`.

## Деталь сборки

Скетч ноды включает рантайм по относительному пути `../../hash_kws_runtime/`;
`hash_runtime_bridge.cpp` напрямую `#include`-ит рантайм-`.cpp` (Arduino single-TU),
включая `hash_kws_runner.cpp` с wire-fix. Сборка самодостаточна внутри
`code/firmware_v4/` (не зависит от `code/training/` или `code/firmware/`).

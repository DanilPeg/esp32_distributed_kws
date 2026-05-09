# mic1 — первый микрофонный узел распределённой сети

ESP32-S3 + INMP441. Принимает звук, прогоняет через hash-KWS модель
(91.28% accuracy, 12 классов), отправляет результат координатору в формате
`InferencePacket` через ESP-NOW.

## Сборка

```powershell
# Заглушка — посылает фейковый top-3 в формате InferencePacket.
# Удобно для проверки сетевого слоя без модели и микрофона.
pio run -e esp32s3

# Полная схема: hash-KWS модель + audio_provider из Моедли/.
# По умолчанию USE_FAKE_MIC=1, чтобы не требовался физический INMP441.
# Чтобы включить реальный микрофон — выставить USE_FAKE_MIC=0 в build_flags.
pio run -e esp32s3-hashkws
```

## Что было исправлено в `Моедли/.../micro_speech_sim`

1. **`hash_model_data.h` — рекурсивный шим без include guard'ов.** Файл
   содержал только `#include "../firmware/hash_kws_runtime/hash_model_data.h"`,
   что из `firmware/hash_kws_runtime/` указывает на самого себя. Реального
   `extern HashDscnnModelData g_hash_model;` не было ни в одном `.h`. Демо
   собиралось только потому, что `hash_runtime_bridge.cpp` напрямую инлайнит
   `.cpp` файл с моделью.
   - Заменено на полноценный заголовок с `extern const HashDscnnModelData
     g_hash_model;` и проперными guard'ами. См.
     `Моедли/.../firmware/hash_kws_runtime/hash_model_data.h`.

2. **`Serial.printf` до `Serial.begin()`.** В `setup()` строка 879 (cpu)
   и 881 (`hash_evt kind=boot ...`) выполнялись до `Serial.begin(115200)`
   на строке 885 — первая boot-телеметрия терялась.
   - В `mic1/src/main.cpp` `Serial.begin()` идёт строго первым.

3. **Канал ESP-NOW.** В оригинале `HASH_KWS_ESPNOW_CHANNEL = 1`, а в
   `coordinator/src/main.cpp` ноды-приёмники и AP сидят на 11. Пакеты
   уходили в эфир, но координатор их не видел.
   - Зафиксирован канал 11 в `mic1/include/config.h::WIFI_CHANNEL`.

4. **Несовместимый формат пакета.** Оригинал шлёт собственный
   `HashKwsEspNowPacket` (38 байт, magic `KHW1`, CRC16, broadcast
   FF:FF:FF:FF:FF:FF). Координатор `Distibuded_Network` ждёт
   `InferencePacket` 74 байта.
   - В `mic1/src/main.cpp` всегда отправляется `InferencePacket`. Top-3
     метки берутся из 12-классового набора hash-KWS:
     `yes/no/up/down/left/right/on/off/stop/go/unknown/silence`. Адрес
     назначения — unicast `COORDINATOR_MAC`.

5. **Busy-loop `while (!g_latest_audio_timestamp){}` без таймаута** —
   при сбое I2S setup() зависает навсегда. Поправлено косвенно: в
   stub-режиме I2S не запускается; в hash-KWS-режиме `setupHashRunner()`
   возвращает false, узел уходит в режим heartbeat и шлёт InferencePacket
   с уцелевшим silence/unknown.

## Структура

```
mic1/
├── platformio.ini       # два env: stub и hashkws
├── include/
│   ├── protocol.h       # InferencePacket, копия shared/
│   └── config.h         # NODE_ID="mic_01", COORDINATOR_MAC, WIFI_CHANNEL=11,
│                        # 12-классовый MIC_LABELS под hash-KWS
└── src/
    └── main.cpp         # объединяет stub-путь и реальный hash-KWS pipeline
```

`hash_kws_runtime` и `audio_provider` подтягиваются из
`../../../Моедли/board1_model_9128_node1/code/firmware/` через
`lib_extra_dirs` в `platformio.ini` (только в env `esp32s3-hashkws`).

## Pin-out (когда подключён реальный INMP441)

| Сигнал | GPIO ESP32-S3 |
|--------|--------------:|
| BCLK   | 15            |
| WS     | 16            |
| SD     | 17            |

(Совпадает с `audio_provider.cpp` в Моедли/.) Для смены пинов править
`audio_provider.cpp::i2s_init()` или прокидывать через build flags.

## Что увидит координатор

После прошивки и старта (на serial 115200):

```
=== mic1 boot ===
cpu=240 MHz  node=mic_01
MAC: A8:42:E3:7F:4C:A0  channel: 11
mode=stub (USE_HASH_KWS_MODEL=0)        # либо: hash_kws ready
Coordinator: B4:3A:45:3F:AC:BC
Ready.
[820] send mic_01  top: silence=78%  unknown=14%  no=8%  OK
[1640] send mic_01  top: silence=72%  unknown=20%  yes=8%  OK
...
```

В дашборде координатора (`http://<coord-ip>/`) карточка `mic_01` появится
сразу после первого пакета.

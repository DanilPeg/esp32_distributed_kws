# mic2 — второй микрофонный узел распределённой сети

ESP32-S3 + INMP441. Аналог `mic1/`, отличия:

* `NODE_ID = "mic_02"` (см. `include/config.h`)
* hash-KWS модель тянется из
  `../../../Моедли/board2_model_9261_node2/code/firmware/` (точность 92.61%)
* подсветка LED при boot — magenta (255, 0, 192) вместо синей у mic1, чтобы
  визуально различать платы

## Сборка

```powershell
# Заглушка
pio run -e esp32s3

# Реальная hash-KWS схема
pio run -e esp32s3-hashkws
```

См. `../mic1/README.md` для подробного списка исправлений из исходного
`micro_speech_sim` (рекурсивный шим `hash_model_data.h`, порядок
`Serial.begin`, канал ESP-NOW, формат пакета, busy-loop в I2S init).
Модель и runtime у mic2 имеют ту же архитектуру, что и у mic1, поэтому
все исправления применимы и здесь — `hash_model_data.h` в
`Моедли/board2_model_9261_node2/code/firmware/hash_kws_runtime/` уже
заменён на нормальный заголовок с `extern const HashDscnnModelData
g_hash_model;` и include guard'ами.

## Структура

```
mic2/
├── platformio.ini       # два env: stub и hashkws
├── include/
│   ├── protocol.h       # InferencePacket, копия shared/
│   └── config.h         # NODE_ID="mic_02"
└── src/
    └── main.cpp         # объединяет stub-путь и реальный hash-KWS pipeline
```

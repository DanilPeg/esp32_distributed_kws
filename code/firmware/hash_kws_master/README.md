# Hash KWS master (4th ESP32)

Dedicated aggregator board. Does **not** run a KWS model. Receives
`HashKwsEspNowPacket` packets (12 int8 logits) from the three inference nodes
(ens_a, ens_b, ens_c) over ESP-NOW, runs `hash_ensemble_aggregator` and
prints `hash_evt kind=fusion ...` to Serial.

## Required files

Open `hash_kws_master.ino` in Arduino IDE, then add the following from
`Sketch > Add File`:

- `code/firmware/hash_kws_aggregator/hash_ensemble_aggregator.cpp`
- `code/firmware/hash_kws_aggregator/hash_ensemble_aggregator.h`
- `code/firmware/hash_kws_aggregator/aggregator_params.h`

(Or copy them next to `hash_kws_master.ino`.)

## Build flags

In `Tools > Build flags` (or `platformio.ini`):

```text
-DHASH_KWS_ESPNOW_CHANNEL=1   # must match the three inference boards
-DHASH_KWS_AGG_MODE=0          # 0=mean_logits, 1=temperature_scaled, 2=learned_weights
```

`mean_logits` is the default and the recommended choice from
`code/deploy/hash_ensemble/reports/ensemble_results.json` (the three
calibration / weight modes give an indistinguishable +0.06 p.p. on test).

## Wiring the four-board demo

| Role          | Board          | Variant       | HASH_KWS_NODE_ID |
|---------------|----------------|----------------|-------------------|
| inference 1   | ESP32-S3 + mic | `ens_a`       | 1                 |
| inference 2   | ESP32-S3 + mic | `ens_b`       | 2                 |
| inference 3   | ESP32-S3 + mic | `ens_c`       | 3                 |
| **master**    | ESP32-S3       | (this sketch) | n/a               |

Inference boards keep using `code/firmware/micro_speech_sim/` with the right
runtime overlay. The master uses **this** sketch.

## Serial protocol

Master prints two kinds of lines:

- `hash_evt kind=boot node=master role=master_aggregator channel=N agg_mode=M`
- `hash_evt kind=fusion node=master label=<word> score=<int16> margin=<int16> voters=N mode=M packets=K rejected=R`

`score` and `margin` are Q8.8 fixed-point. The host dashboard already parses
`hash_evt kind=fusion ...`; nothing on the host side needs to change.

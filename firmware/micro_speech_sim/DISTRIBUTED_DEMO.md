# Distributed hash-KWS demo — runbook

Three-node demo on current hardware reality: **one real ESP32 with a mic,
one emulated node on the host, one master aggregator server with a live
dashboard.** Transport between real and emulated is ESP-NOW-shaped: the real
board genuinely transmits ESP-NOW packets into the air; the emulator
manufactures consistent peer events on the host by tailing the real board's
telemetry. The master aggregator runs entirely on the host.

```
         ┌──────────────────────────────┐
         │  node1  —  real ESP32-S3     │
         │  mic → hash-KWS (91.28%)     │
         │  → ESP-NOW TX + serial log   │
         └──────────┬───────────────────┘
                    │ USB serial
                    ▼
         ┌──────────────────────────────┐
         │  hash_kws_serial_bridge.py   │
         │  notes/Journal/hash_kws_     │
         │  telemetry/node1/*           │
         └──────────┬───────────────────┘
                    │ JSONL tail
                    ▼
         ┌──────────────────────────────┐
         │  hash_kws_cluster_sim.py     │
         │  → synthesises node2 events  │
         │  → synthesises node3 fusion  │
         │    (hash_kws_fusion/*)       │
         └──────────┬───────────────────┘
                    │ JSONL tail
                    ▼
         ┌──────────────────────────────┐
         │  code/dashboard/ (FastAPI)   │
         │  http://127.0.0.1:8765/      │
         └──────────────────────────────┘
```

## Prereqs

* Arduino IDE with the **ESP32** core (tested against arduino-esp32 3.x).
* Host Python 3.10+ with all four deps installed in the same interpreter:

  ```powershell
  pip install fastapi uvicorn jinja2 pyserial
  ```

  The launcher pre-flights these and aborts with a clear message if any is
  missing, so you won't see cryptic child crashes.
* The 91.28% hash-KWS model bundle is already committed at
  `code/training/hash_model_data.cpp` and pulled into the firmware via
  `code/firmware/micro_speech_sim/micro_speech/hash_runtime_bridge.cpp`.
  No Colab / retraining required.

## One-time firmware build & flash

1. Open
   `code/firmware/micro_speech_sim/micro_speech/micro_speech.ino` in
   Arduino IDE.
2. Board: **ESP32S3 Dev Module** (or the variant that matches the board in
   hand). USB CDC on Boot: enabled, partition scheme: default.
3. Confirm the defaults at the top of the sketch (no edits needed for the
   demo):

   ```cpp
   #define HASH_KWS_NODE_ID            1   // logical id of this board
   #define HASH_KWS_USE_ESPNOW         1   // distributed demo = ON
   #define HASH_KWS_ESPNOW_FUSION      1   // emit kind=fusion on peer agreement
   #define HASH_KWS_TELEMETRY_STREAM   1   // emit hash_evt lines over serial
   ```
4. Compile (`Ctrl+R`). Upload (`Ctrl+U`). The first compile warms up the
   TFLM + ESP-NOW core caches and can take several minutes; subsequent
   rebuilds are much faster.
5. Open the serial monitor at **115200 baud**. Expected first lines:
   * `hash_evt kind=ready node=1 runtime=hash_kws ...`
   * `hash_evt kind=activity node=1 t=... speech=0 ...`
   * When you say "yes" / "no" etc.: `hash_evt kind=infer ...`,
     `hash_evt kind=emit ...`, `hash_evt kind=espnow phase=tx ...`.

If compile fails, paste the error — the most likely causes are wrong core
version (needs 3.x), missing `TensorFlowLite_ESP32` library, or a stale
Arduino core cache (`File → Preferences → reset` plus delete
`%LOCALAPPDATA%/Arduino15`).

## Run the host side (one command)

Note the board's COM port in Device Manager (e.g. `COM5`). Then from the
project root:

```powershell
python code\scripts\run_distributed_demo.py --port COM5
```

The launcher starts, in this order:

1. `hash_kws_serial_bridge.py` — reads serial, writes per-node JSONL into
   `notes/Journal/hash_kws_telemetry/node1/`.
2. `hash_kws_cluster_sim.py` — tails node1 events, writes virtual
   node2/node3 events and fusion decisions.
3. `run_dashboard.py` — FastAPI dashboard on `http://127.0.0.1:8765/`.

Press **Ctrl+C once** to stop everything. All three subprocesses inherit
the parent's terminal, so their logs mix in the one window — that's the
quickest way to spot issues during the demo.

### Useful flags

* `--port COMx` — required.
* `--dashboard-port 8000` — change the dashboard port.
* `--no-dashboard` — if you prefer to run `python run_dashboard.py`
  separately (useful when iterating on the UI).
* `--no-sim` — disable the emulated peer + master (single-node debug mode).
* `--no-bridge` — replay existing JSONL without opening the serial port.

## What to look for in the dashboard

* **Cluster overview**: three cards.
  * `node1 / real` should go **online** within ~2 seconds of the bridge
    starting. Latest top1 / score updates whenever the board emits a
    `kind=infer` event.
  * `node2 / emulated` goes online as soon as the cluster sim has
    manufactured its first event based on node1's inference.
  * `node3 / master` flips online when the sim writes the first
    `kind=fusion` virtual event.
* **Counters (tail window)**: tail-scoped totals of `infer`, `emit`,
  `espnow tx`, `fusion`, plus the fusion agreement rate. `espnow tx` is a
  good proxy for "the real board is actually transmitting into the air".
* **Live inference feed**: newest-first stream across all three nodes.
* **Fusion decisions**: `audio_fusion_agree` entries should appear within
  ~1 s of matching inferences on node1 and node2.

## Troubleshooting

* **Dashboard online=false for all three nodes**: the bridge is probably
  not reading the serial. Confirm the COM port and that the Arduino serial
  monitor is **closed** (only one process can hold the port).
* **node1 online, others not**: cluster sim is not running. Re-run without
  `--no-sim`, or launch it manually:
  `python code\scripts\hash_kws_cluster_sim.py --watch --print-state`.
* **ESP-NOW counters stay at zero**: confirm `HASH_KWS_USE_ESPNOW=1` in
  the compiled sketch (this is now the default — but if you're on a stale
  build, the macro may still be 0).
* **Duplicate events / stale data**: the bridge rotates through
  `notes/Journal/hash_kws_telemetry/node1/events.jsonl`. To reset,
  stop everything and truncate that file plus the matching
  `hash_kws_fusion/decisions.jsonl` and `hash_kws_cluster/state.json`.

## Single-board reality

The demo is consciously wired so the user-visible architecture — three
nodes, ESP-NOW transport, master aggregator — is preserved even when only
one physical board is available. The emulator keeps the event contract
faithful (same JSON schema as a real second board would produce, same
fusion logic as the firmware runs on-device). When a second real MCU is
wired in later, flash it with `HASH_KWS_NODE_ID=2` and disable the sim's
virtual peer (future work).


## Single-node mode

If you want to watch only the real ESP32 with no emulated peer or virtual
master (useful for isolating inference behaviour), pass ``--single-node``:

```powershell
python code\scripts\run_distributed_demo.py --port COM8 --single-node
```

Same as ``--no-sim``. The dashboard drops the node2 / node3 cards as soon as
their JSONL streams stop producing events, so the UI stays clean.

## Inference performance notes

The hash-KWS runtime is a custom float engine that reconstructs weights from
hash indices at inference time. For the dedicated ``micro_speech_sim`` sketch
we apply three host-free optimisations (all require a reflash to take effect):

- ``#pragma GCC optimize ("O3", "unroll-loops", "tree-vectorize")`` at the top
  of ``hash_runtime_bridge.cpp``. Arduino's default ``-Os`` disables both
  unrolling and auto-vectorisation — this pragma overrides it for every TU
  the bridge pulls in (runner, recogniser, hash_model_data).
- ``heap_caps_malloc(..., MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)`` for
  ``scratch_a``/``scratch_b`` and the input buffer, so activation reads/writes
  never touch PSRAM.
- Integer-MAC fast path in ``RunPointwiseConv1x1`` / ``RunPointwiseResidualConv1x1``:
  the hash-derived weights are packed to signed int8 (sign baked in, codebook
  scale factored out), and the inner loop does ``int32 += int8 * int8``
  instead of ``float += float * float``. Correctness is preserved because
  the per-layer weight scale is just ``codebook_scale``.

Expectations after these changes:

- Cold first invoke: 400-900 ms (TFLM-less lazy init of activation scratch).
- Warm invoke: target 150-400 ms on ESP32-S3 at 240 MHz.
- If the dashboard still shows median > 500 ms across 10+ invokes, open a
  dedicated journal entry — next optimisation pass would move the depthwise
  3x3 and stem 3x3 convs to int-MAC too, which is a larger rewrite.

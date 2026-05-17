# Micro Speech Example

## Simulation Mode (No Microphone)
This repo copy runs in **fake mic** mode by default:
- `USE_FAKE_MIC=1` in `audio_provider.cpp`
- It feeds zeroed audio and increments time to exercise the inference pipeline.

Result: the model runs and logs output over Serial **without a physical microphone**.
To use a real mic, set `USE_FAKE_MIC=0` and configure I2S pins in `audio_provider.cpp`.

This example shows how to run a 20 kB model that can recognize 2 keywords,
"yes" and "no", from speech data.

The application listens to its surroundings with a microphone and indicates
when it has detected a word by displaying data on a screen.

## Deploy to ESP32

The following instructions will help you build and deploy this sample
to [ESP32](https://www.espressif.com/en/products/hardware/esp32/overview)
devices using the [ESP IDF](https://github.com/espressif/esp-idf).

The sample has been tested on ESP-IDF version `release/v4.2` and `release/v4.4` with the following devices:
- [ESP32-DevKitC](http://esp-idf.readthedocs.io/en/latest/get-started/get-started-devkitc.html)
- [ESP32-S3-DevKitC](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/hw-reference/esp32s3/user-guide-devkitc-1.html)
- [ESP-EYE](https://github.com/espressif/esp-who/blob/master/docs/en/get-started/ESP-EYE_Getting_Started_Guide.md)

### Install the ESP IDF

Follow the instructions of the
[ESP-IDF get started guide](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html)
to setup the toolchain and the ESP-IDF itself.

The next steps assume that the
[IDF environment variables are set](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html#step-4-set-up-the-environment-variables) :

 * The `IDF_PATH` environment variable is set
 * `idf.py` and Xtensa-esp32 tools (e.g. `xtensa-esp32-elf-gcc`) are in `$PATH`


### Building the example

Set the chip target (For esp32s3 target, IDF version `release/v4.4` is needed):

```
idf.py set-target esp32s3
```

Then build with `idf.py`
```
idf.py build
```

### Load and run the example

To flash (replace `/dev/ttyUSB0` with the device serial port):
```
idf.py --port /dev/ttyUSB0 flash
```

Monitor the serial output:
```
idf.py --port /dev/ttyUSB0 monitor
```

Use `Ctrl+]` to exit.

### Recommended live monitoring path

Raw serial logs are still available, and the hash runtime can also emit
structured `hash_evt ...` lines when needed.

Recommended host-side workflow:

```
pip install pyserial
python code/scripts/hash_kws_serial_bridge.py --port COM7 --baud 115200
```

This bridge writes:

- `notes/Journal/hash_kws_telemetry/state.json`
- `notes/Journal/hash_kws_telemetry/events.jsonl`
- `notes/Journal/hash_kws_telemetry/raw.log`

If the web portal is running, open:

```
http://127.0.0.1:8000/hash-kws/live
```

This is more useful than only watching `hash_dbg` because it shows:

- fast activity-gate updates from recent slices;
- invoke timing and top-k snapshots;
- episode start/end and emitted command events.

Note:

- current default firmware output is concise for live board checks;
- compile-time toggles now default to:
  - `HASH_KWS_DEBUG_STREAM = 0`
  - `HASH_KWS_TELEMETRY_STREAM = 1` for live testing through the serial bridge
  - `HASH_KWS_USE_ESPNOW = 0` for one-board testing with host simulation
- re-enable them only when targeted tuning is needed.

## Current board status

- The current hybrid hash-KWS scheduler is now the recommended default branch.
- Live board feedback says it works tolerably.
- The remaining issue is invoke cost, so the next improvements should focus on reducing per-inference cost rather than increasing temporal input length.
- The 128-channel model needs `128000` bytes of scratch arena. Runtime buffers now prefer PSRAM and fall back to internal heap.
- The live audio ring buffer is reduced to `32768` bytes and no longer asserts on allocation failure.

## Two-node audio distribution with ESP-NOW

The current distributed demo path is audio+audio, not audio+image. Each ESP32
runs the local hash-KWS model and broadcasts a compact ESP-NOW packet after a
local inference. The packet contains node id, sequence, local timestamp, top
label, score, top1-top2 margin, recent activity, and the 12 int8 logits.

The sketch defaults to ESP-NOW disabled for the current one-board simulation.
For real two-board ESP-NOW testing, enable it explicitly:

```c
HASH_KWS_USE_ESPNOW = 1
HASH_KWS_ESPNOW_CHANNEL = 1
HASH_KWS_ESPNOW_FUSION = 1
```

Build/upload the same sketch twice with different node ids:

```c
#define HASH_KWS_NODE_ID 1
```

and:

```c
#define HASH_KWS_NODE_ID 2
```

For evidence logs, enable:

```c
#define HASH_KWS_TELEMETRY_STREAM 1
```

Expected serial evidence:

```text
hash_evt kind=espnow node=1 phase=init status=ok ...
hash_evt kind=espnow node=2 phase=init status=ok ...
hash_evt kind=fusion node=1 peer=2 label=yes local_score=... peer_score=... fused_score=...
```

Recommended host logging while both boards are connected:

```powershell
python code/scripts/hash_kws_serial_bridge.py --port COM7 --baud 115200 --node-id 1 --node-label audio_a --state-path notes/Journal/hash_kws_telemetry/node1/state.json --events-path notes/Journal/hash_kws_telemetry/node1/events.jsonl --raw-path notes/Journal/hash_kws_telemetry/node1/raw.log --echo
python code/scripts/hash_kws_serial_bridge.py --port COM8 --baud 115200 --node-id 2 --node-label audio_b --state-path notes/Journal/hash_kws_telemetry/node2/state.json --events-path notes/Journal/hash_kws_telemetry/node2/events.jsonl --raw-path notes/Journal/hash_kws_telemetry/node2/raw.log --echo
```

The host-side fusion script is kept only as a fallback and audit helper:

```powershell
python code/scripts/hash_kws_dual_audio_fusion.py --events notes/Journal/hash_kws_telemetry/node1/events.jsonl notes/Journal/hash_kws_telemetry/node2/events.jsonl --print-decisions
```

### Hybrid simulation with one real board

If only one microphone-equipped ESP32 is available, run node 1 as the real
board and emulate node 2 plus node 3 on the host:

```powershell
python code/scripts/hash_kws_serial_bridge.py --port COM7 --baud 115200 --node-id 1 --node-label real_audio --state-path notes/Journal/hash_kws_telemetry/node1/state.json --events-path notes/Journal/hash_kws_telemetry/node1/events.jsonl --raw-path notes/Journal/hash_kws_telemetry/node1/raw.log --echo
python code/scripts/hash_kws_cluster_sim.py --watch --print-state
```

This creates:

- virtual peer node 2 events in `notes/Journal/hash_kws_telemetry/node2/`;
- virtual master node 3 events in `notes/Journal/hash_kws_telemetry/node3/`;
- fused decisions in `notes/Journal/hash_kws_fusion/`;
- cluster state in `notes/Journal/hash_kws_cluster/state.json`.

Open the portal page:

```text
http://127.0.0.1:8000/hash-kws/live
```

The final hardware target is still three ESP32 roles: two audio KWS sensor
nodes and one master aggregator. The host simulation only stands in for the
missing boards while preserving the same event-level contract.

The previous two commands can be combined:
```
idf.py --port /dev/ttyUSB0 flash monitor
```

### Sample output

  * When a keyword is detected you will see following output sample output on the log screen:

```
Heard yes (<score>) at <time>
```

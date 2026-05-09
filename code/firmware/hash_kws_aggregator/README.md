# Hash KWS ensemble aggregator (MCU)

Drop-in C++ module that mirrors `code/training/hash_ensemble/aggregation.py`.
Implements three aggregation modes over per-node int8 logits:

- `kModeMeanLogits` — research headline (`A4 best on KWS = mean_logits`).
- `kModeTemperatureScaled` — mean of `softmax(logits_k / T_k)`.
  Falls back to `kModeMeanLogits` if `setTemperatures()` was never called.
- `kModeLearnedWeights` — `Σ w_k · logits_k` with non-negative softmax-normalized
  weights fit on validation logits. Falls back to `kModeMeanLogits` if weights
  are missing.

Calibration parameters and learned weights are written to
`code/training/hash_artifacts/hash_ensemble/aggregator_params.h` by the
training notebook. The notebook never leaks test-set information into those
parameters.

## Usage sketch

```cpp
#include "hash_ensemble_aggregator.h"
#include "../hash_kws_aggregator_params/aggregator_params.h"  // auto-generated

using hash_kws_ensemble::Aggregator;
using hash_kws_ensemble::Mode;
using hash_kws_ensemble::SourceKind;

Aggregator agg;
agg.reset(/*num_nodes=*/3, /*num_classes=*/12, /*window_ms=*/1200);
agg.setTemperatures(kHashEnsembleTemperatures);
agg.setLearnedWeights(kHashEnsembleLearnedWeights);
agg.setMode(Mode::kModeLearnedWeights);

// On every received ESP-NOW packet from peer / local infer:
agg.submit(node_id, SourceKind::kEmit, packet.device_t, millis(),
           packet.logits_int8, /*num_classes=*/12);

// Periodically (or on every emit):
hash_kws_ensemble::Resolved out;
agg.resolve(millis(), &out);
if (out.has_decision) {
  // out.label, out.score (Q8.8), out.margin (Q8.8), out.num_voters
}
```

## Constraints

- No heap allocation; all buffers static (`HASH_KWS_AGG_MAX_NODES = 4`,
  `HASH_KWS_AGG_MAX_CLASSES = 12` by default).
- Depends only on `<stdint.h>`, `<stddef.h>`, `<string.h>`, `<math.h>`.
- Decisions require at least 2 voters in the window.
- `submit()` keeps higher-priority sources (`emit > episode > infer`) when a
  fresh vote already exists from the same node within the window.

## Where it gets called from

Today: integrated by the host simulator
`code/scripts/hash_ensemble_sim.py` so we can validate the contract before
flashing three boards.

Once three real ESP32-S3 boards are available: include from `micro_speech.ino`
on the master node (or symmetric mode), feed it the int8-logit packets that
already flow over ESP-NOW (see `2026-04-24_hw_espnow_001.yaml`).

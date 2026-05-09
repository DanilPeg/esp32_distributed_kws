# Hash KWS distributed ensemble — deploy bundle

Complete deploy artefacts for the 3‑model + master‑aggregator demo.
All paths in this file are relative to the repository root.

## What is here

```
code/deploy/hash_ensemble/
├── README.md                              ← this file
├── INSTRUCTIONS_TEST_NO_MCU.md            ← run end-to-end on host only
├── INSTRUCTIONS_REAL_MCU.md               ← flash 4 boards, run live demo
├── reports/
│   ├── ensemble_results.json              ← per-model + per-aggregator metrics
│   ├── aggregator_params.h                ← calibrated temperatures + learned weights (val-fit)
│   ├── aggregator_comparison.png          ← bar(top1) per aggregator + best-single + oracle
│   └── pairwise_disagreement.png          ← 3×3 heatmap
└── models/
    ├── ens_a/
    │   ├── student_best.pt                ← PyTorch checkpoint with experiment + state_dict
    │   └── firmware_export/
    │       ├── hash_model_data.cpp        ← variant-specific weights bundle
    │       └── hash_model_export_metadata.json
    ├── ens_b/   (this is the variant of the existing hn95_kd128_layerwise_signed_residual recipe; unchanged accuracy)
    └── ens_c/
```

## Headline numbers (from `reports/ensemble_results.json`)

| Variant | top1   | top3   | compact params | virtual params | MACs |
|---------|--------|--------|----------------|-----------------|---------|
| ens_a   | 0.9309 | 0.9893 |   8 780        |  72 832         |  35.65M |
| ens_b   | 0.9323 | 0.9896 |   9 804        |  72 832         |  35.65M |
| ens_c   | 0.9306 | 0.9892 |  10 828        |  72 832         |  35.65M |
| **mean_logits ens.** | **0.9423** | 0.9925 | – | – | – |
| mean_probs            | 0.9429 | 0.9922 | – | – | – |
| temperature_scaled    | 0.9425 | 0.9920 | – | – | – |
| learned_weights       | 0.9423 | 0.9925 | – | – | – |
| oracle (any-correct)  | 0.9661 | – | – | – | – |
| dispersion (single std) | 0.000716 | – | – | – | – |

`ens_b` is the same recipe and architecture that produced the prior 93.25%
result; the new run gives 93.23% — within seed noise. Across all five
"smart" aggregators the spread is 0.20 p.p., so the practical choice is
**`mean_logits`** (no `exp()` on the MCU, matches the research headline).

The temperatures and learned weights in `reports/aggregator_params.h` are
**fitted on the validation split only** (no test leakage). They are baked
into the master MCU build and selected at compile time via
`HASH_KWS_AGG_MODE`.

## Where these files are wired into the rest of the repo

- `code/firmware/hash_kws_runtime_ens_{a,b,c}/` — full sketch overlays for
  each inference board (variant-specific `hash_model_data.cpp` + scaffolding
  copied from `hash_kws_runtime/`). Use
  `code/scripts/select_hash_kws_variant.ps1 -Variant ens_b` before flashing.
- `code/firmware/hash_kws_master/` — minimal Arduino sketch for the **4th**
  ESP32 (no microphone, no KWS model). Already includes the aggregator code
  + `aggregator_params.h`.
- `code/firmware/hash_kws_aggregator/aggregator_params.h` — same header as in
  `reports/`, kept here for non-master use cases (e.g. on-host evaluation).

## Two ways to use this bundle

1. **Without real MCUs** — replay test samples through the three
   `student_best.pt` files, run the aggregator, write JSONL events to the
   existing dashboard contract. See `INSTRUCTIONS_TEST_NO_MCU.md`.

2. **With four ESP32 boards** — flash 3 inference nodes (each its variant)
   and 1 master node, watch live `kind=fusion` events in the dashboard.
   See `INSTRUCTIONS_REAL_MCU.md`.

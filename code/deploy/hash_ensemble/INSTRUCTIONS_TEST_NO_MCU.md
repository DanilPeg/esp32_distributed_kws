# Run the ensemble demo without real MCUs

Three students live in `code/deploy/hash_ensemble/models/`. The host
simulator loads them, runs the test split (or a subset), feeds logits into
the same aggregator the master MCU will use, and writes the same JSONL
events that the dashboard already reads.

All commands assume the repository root as the working directory and Python
≥ 3.10 with `torch`, `torchaudio`, `numpy`, `tqdm` installed (same env you
used during training; if you trained in Colab, re-create locally with
`code/training/requirements-kws-hash-exact-frontend.txt`).

> **NumPy / TensorFlow import error:** if you see
> `AttributeError: _ARRAY_API not found` or
> `ImportError: numpy.core._multiarray_umath failed to import`, your numpy
> is 2.x and the installed TensorFlow needs 1.x. Fix in the same env:
>
> ```powershell
> python -m pip install "numpy<2"
> ```
>
> The repo's `requirements-kws-hash-exact-frontend.txt` already pins
> `numpy<2` for this exact reason.

## 1. Smoke check (no torch, no checkpoints)

Make sure the JSONL contract works end-to-end before involving real models.

```powershell
python code\scripts\hash_ensemble_sim.py smoke --samples 30 --reset-streams
```

Expected output:

```
smoke: 30/30 ensemble correct -> 1.000
streams under .../notes/Journal/hash_kws_telemetry / .../hash_kws_fusion / .../hash_kws_cluster
```

Open the dashboard (`run_dashboard.py`) — three node panels and the fusion
strip should populate within seconds.

## 1b. Verify the three real bundles load (no TF, no Speech Commands)

Before pulling in the dataset, confirm the `.pt` files load and the
architectures match. This path doesn't import TensorFlow — useful when the
numpy 2.x / TF mismatch above isn't fixed yet.

```powershell
python code\scripts\hash_ensemble_sim.py verify --bundles code\deploy\hash_ensemble\models\ens_a\student_best.pt code\deploy\hash_ensemble\models\ens_b\student_best.pt code\deploy\hash_ensemble\models\ens_c\student_best.pt --variant-names ens_a ens_b ens_c --device cpu
```

Expected output (per-line `OK  tag=... recorded_test_top1=0.93...`):

```
[ens_a] OK  tag=hash_kws12_iterlab_v1_hn95_ens_a_s13  input_shape=(1, 40, 49)  logits_shape=(1, 12)  recorded_test_top1=0.9304156535834202
[ens_b] OK  tag=hash_kws12_iterlab_v1_hn95_ens_b_s29  ...                                            recorded_test_top1=0.9324997105476439
[ens_c] OK  tag=hash_kws12_iterlab_v1_hn95_ens_c_s47  ...                                            recorded_test_top1=0.930647215468334

All bundles loaded and forward-pass clean.
```

## 2. Full eval — independent verification of `ensemble_results.json`

Run the three real students through the test split. PowerShell:

```powershell
python code\scripts\hash_ensemble_sim.py eval `
    --bundles `
        code\deploy\hash_ensemble\models\ens_a\student_best.pt `
        code\deploy\hash_ensemble\models\ens_b\student_best.pt `
        code\deploy\hash_ensemble\models\ens_c\student_best.pt `
    --variant-names ens_a ens_b ens_c `
    --output code\deploy\hash_ensemble\reports\ensemble_results_local.json
```

If your shell doesn't like multi-line, the same as one line:

```powershell
python code\scripts\hash_ensemble_sim.py eval --bundles code\deploy\hash_ensemble\models\ens_a\student_best.pt code\deploy\hash_ensemble\models\ens_b\student_best.pt code\deploy\hash_ensemble\models\ens_c\student_best.pt --variant-names ens_a ens_b ens_c --output code\deploy\hash_ensemble\reports\ensemble_results_local.json
```

Expected: per-model top1 within ±0.001 of the numbers in
`reports/ensemble_results.json` (full deterministic match if you also keep
the same Speech Commands cache). All seven aggregators printed.

## 3. Live demo — replay test samples through aggregator + dashboard

PowerShell:

```powershell
python code\scripts\hash_ensemble_sim.py demo `
    --bundles `
        code\deploy\hash_ensemble\models\ens_a\student_best.pt `
        code\deploy\hash_ensemble\models\ens_b\student_best.pt `
        code\deploy\hash_ensemble\models\ens_c\student_best.pt `
    --variant-names ens_a ens_b ens_c `
    --aggregator mean_logits `
    --params-json code\deploy\hash_ensemble\reports\ensemble_results.json `
    --samples 200 --reset-streams --verbose
```

One-line variant:

```powershell
python code\scripts\hash_ensemble_sim.py demo --bundles code\deploy\hash_ensemble\models\ens_a\student_best.pt code\deploy\hash_ensemble\models\ens_b\student_best.pt code\deploy\hash_ensemble\models\ens_c\student_best.pt --variant-names ens_a ens_b ens_c --aggregator mean_logits --params-json code\deploy\hash_ensemble\reports\ensemble_results.json --samples 200 --reset-streams --verbose
```

Each sample emits one `infer` event under
`notes/Journal/hash_kws_telemetry/node{1,2,3}/events.jsonl` and one
`audio_fusion_agree` decision under `notes/Journal/hash_kws_fusion/decisions.jsonl`.
The existing dashboard renders these in real time.

Try the other two aggregators by changing `--aggregator`:

- `temperature_scaled` — uses the calibrated `T = [0.84, 0.84, 0.86]`.
- `learned_weights` — uses the fitted (uniform) weights.

`--params-json` is what the simulator reads to find calibration parameters;
you can also use `code/firmware/hash_kws_aggregator/aggregator_params.h` if
you regenerate it.

## 4. Cross-check the MCU aggregator against the host

Pure smoke: the C++ master aggregator is byte-equivalent on label decisions
to the numpy reference on six fixed test vectors.

```powershell
python code\firmware\hash_kws_aggregator\test_aggregator_match.py
```

Expected: `All cases match.`

## Where the files live (quick reference)

- Models: `code/deploy/hash_ensemble/models/ens_{a,b,c}/student_best.pt`
- Calibration: `code/deploy/hash_ensemble/reports/aggregator_params.h`
- Simulator: `code/scripts/hash_ensemble_sim.py`
- Numpy aggregators: `code/training/hash_ensemble/aggregation.py`
- Per-board firmware overlays: `code/firmware/hash_kws_runtime_ens_{a,b,c}/`
- Master MCU sketch: `code/firmware/hash_kws_master/`

# Hash KWS Runtime

This folder contains a custom low-level inference path for hash-compressed KWS models on ESP32.

The design goal is different from the existing TFLite Micro path:

- preserve the memory win of analytic-hash codebooks;
- avoid dense weight materialization;
- reuse the existing `49 x 40` audio feature pipeline from the firmware side;
- keep the runtime simple enough to optimize incrementally on `ESP32-S3`.

## Current Runtime Strategy

- Input features are expected as a `49 x 40` `int8` spectrogram buffer.
- Internal activations use `int8` double-buffer scratch memory.
- Hash codebooks are stored as quantized `int8` arrays plus one scale per layer.
- BatchNorm is folded into per-output-channel post-affine parameters instead of into dense weights.
- The final layer produces logits.
- The current recommended firmware policy is not pure always-on smoothing:
  - sparse idle probes during quiet periods;
  - speech episodes when recent slices show activity;
  - peak-hold command selection inside the episode;
  - concise serial output by default for live board checks.

## Current Board Status

- Current live-board feedback says this branch works tolerably as a deployment baseline.
- The main remaining problem is heavy invoke latency on the deeper hash model, not microphone capture.
- Because of that, increasing temporal input size above `49` frames is not the right next step for the latency branch.
- If training-side changes are needed, prefer:
  - exact frontend alignment;
  - equal or smaller time footprint;
  - lower-cost temporal resolution before trying larger windows.

## Export Path

Generate the firmware arrays from a trained hash bundle with:

```powershell
python code/scripts/export_hash_kws_firmware.py `
  --bundle code/training/hash_artifacts/<experiment-tag>/hash_kws_student_student.pt
```

The exporter will:

- restore the PyTorch hash model from the compact bundle;
- fold `BatchNorm` into post-convolution affine parameters;
- calibrate per-stage activation scales on a few dataset batches;
- quantize each codebook to `int8` without materializing dense weights;
- overwrite `hash_model_data.cpp` and emit `hash_model_export_metadata.json`.

## Important Limitations

- The placeholder `hash_model_data.cpp` intentionally marks the model as unavailable until a real export is generated.
- Current runtime math preserves the hash-compressed weights, but accuracy still depends on the on-device frontend matching the training frontend.
- Current runtime uses two full `int8` activation buffers. For the `64 x 20 x 25` deeper hash model this is about `64 KB` of scratch, so the next memory optimization target is fused `depthwise -> pointwise` streaming.

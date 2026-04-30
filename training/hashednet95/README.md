# HashedNet95 KWS Lab

This folder contains a compact notebook track for the next hash-KWS push.

The goal is to test the HashedNets ideas that were still weak in the previous
branch:

- signed hashing as a first-class setting;
- per-layer codebook budgets instead of one uniform bucket count;
- virtual-width inflation under a mostly fixed storage budget;
- optional HashNetDK-style teacher distillation;
- teacher checkpoint reuse and teacher-logit caching for follow-up student runs;
- KD alpha/temperature schedules and explicit hard-label polish smoothing;
- direct Speech Commands preparation from inside the notebook;
- persistent Google Drive caching for exact microfrontend feature tensors and
  run artifacts.

Generate the notebook:

```powershell
python code/training/hashednet95/build_hashednet95_notebook.py
```

Open:

```text
code/training/hashednet95/hashednet95_kws_colab.ipynb
```

The default next-push recipe is:

```text
hn95_kd128_cached_schedule
```

Suggested sweep order:

1. `hn95_kd128_cached_schedule`
2. `hn95_kd128_hard_polish_cached`
3. `hn95_kd128_pw2048_budget_cached_schedule`
4. `hn95_kd112_latency_balanced_cached_schedule`
5. `hn95_kd128_rich_teacher160_cached_schedule`
6. `hn95_kd128_cached_schedule_s29`
7. `hn95_kd128_cached_schedule_s47`

The notebook intentionally avoids embedding the whole runtime payload. It
does embed the minimal files required for the hash-KWS runtime, so it can
recover `/content/diploma_esp32_distributed_nn` even when only the notebook is
opened in Colab.

Dataset/cache behavior:

- Speech Commands is prepared deterministically under
  `/content/diploma_esp32_distributed_nn/data` by default, matching the older
  notebooks;
- the notebook validates the exact expected extracted directory
  `SpeechCommands/speech_commands_v0.02` and its `validation_list.txt` /
  `testing_list.txt` manifests;
- if that exact extraction is incomplete, the notebook removes only the broken
  `SpeechCommands` extraction folder and lets torchaudio download/extract the
  dataset again;
- no recursive search over Google Drive is used for dataset discovery;
- exact microfrontend feature caches are stored under
  `/content/drive/MyDrive/diploma_kws_cache/hashednet95/hash_feature_cache`
  through `HASH_KWS_FEATURE_CACHE_ROOT`.
- teacher logits are stored under
  `/content/drive/MyDrive/diploma_kws_cache/hashednet95/teacher_logits_cache`
  through `HASH_KWS_TEACHER_LOGITS_CACHE_ROOT`.

Optional teacher reuse:

- set `TEACHER_CHECKPOINT_PATH` in the notebook to a previous
  `teacher_best.pt`;
- keep `FORCE_TEACHER_RETRAIN = False` to evaluate and reuse that teacher
  directly;
- leave `TEACHER_CHECKPOINT_PATH = ""` to train the teacher inside the run.

Seed repeats:

- the `_s29` and `_s47` recipes vary `train.seed`, not `dataset.seed`, so the
  Speech Commands split composition remains fixed while initialization and
  dataloader shuffling change.

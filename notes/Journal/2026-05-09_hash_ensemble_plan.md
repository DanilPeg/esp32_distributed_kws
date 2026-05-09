# Hash Ensemble — Locked Plan (v2)

Дата: 2026-05-09
Автор: Danil
Статус: locked, implementation pending
Изменения v2: переход с hetero‑K на «почти‑homo» (PW‑perturbed) — точность важнее per‑node истории; добавлены два развивающих идею агрегатора (temperature‑scaled mean_logits и learnable 3‑weight head).

Этот файл — единственный источник правды для трека «3 hash‑KWS‑модели + master‑агрегатор».
Все последующие изменения архитектуры/рецептов/контрактов агрегации фиксируются ПРАВКАМИ этого файла, а не через PR/чат.

## 0. Что мы делаем и зачем

Берём текущий рабочий baseline `hn95_kd128_layerwise_signed_residual` (KWS‑12, exact microfrontend, 128ch×4 blocks, signed+residual hash, KD α=0.55 / T=4.5) — student test acc **0.9325**, teacher 0.9484 (`notes/Journal/2026-04-24_ml_001.yaml`). Превращаем его в три однородных по архитектуре, но **разных по кодбук‑бюджету и seed‑у** ученика и собираем ансамбль с агрегацией поверх логитов.

Опираемся на результаты `code/analysis/hash_ensemble/` (см. `NOTES.md` §7.3–7.4):

- Подтверждено: ансамбль даёт +1.5–2.5 п.п. над best single (баггинг‑амплитуда).
- Headline: на KWS std упал ×3.84 при N=3 — устойчивость к неудачным хеш‑коллизиям, выше √N.
- `mean_logits` достаточен; smart‑агрегаторы системного выигрыша не дают.
- Hetero‑K на iid‑тесте **проигрывает** homo (A3=0.9232 < A2=0.9269 при равном бюджете).
- A5 «PW‑heavy hetero» (только PW‑бюджет варьируется, остальное одинаково) дал 0.9267 ≈ homo — почти не теряет в accuracy и сохраняет лёгкую диверсификацию.

Решение по конфигурации: «**почти‑homo + лёгкие PW‑возмущения**». Все три модели имеют идентичные stem / DW / linear codebooks и идентичную архитектуру (каналы, блоки, residual). Различия только в:
- PW codebooks: −20% / baseline / +13% от текущего рабочего рецепта;
- `train.seed`: 13 / 29 / 47 (отличается init и dataloader shuffle);
- внутри прежнего «латентного бюджета» текущей рабочей модели — без раздувания.

Обоснование: A2 (homo) был лучшим ансамблем в исследовании; user попросил «побольше и поменьше», что мы реализуем минимальной PW‑покачкой по типу A5 — это даёт slight diversity без потери accuracy. Per‑node history (small для слабого узла) сознательно отбрасываем — diploma‑story теперь «N идентичных микроконтроллеров с независимыми хеш‑скетчами».

## 1. Baseline и три варианта

Источник истины: `code/training/hashednet95/hashednet95_recipes.py::hn95_kd128_layerwise_signed_residual`.

Зафиксированные общие параметры (всех трёх):

- vocabulary: `kws12` (10 команд + unknown + silence = 12 классов)
- feature: exact_microfrontend, normalize_mode=none, cache_features=true (int8)
- channels=128, num_blocks=4, signed_hash=true, use_residual=true
- teacher: dense_dscnn_teacher, channels=128, num_blocks=6, dropout=0.05
- KD: α=0.55, T=4.5; teacher_logits_cache=ON (`hn95_kd128_cached_schedule`‑style), кэш дзвучит один раз для всех трёх
- schedule: 6 CE pretrain → 40 KD → 4 polish; cosine LR; label_smoothing=0.02; grad_clip=1.0
- export: `student_best.pt` + bundle + firmware export — каждый в свой подкаталог

Три варианта (отличаются ТОЛЬКО PW‑codebooks и seed):

| Вариант    | seed | stem | dw per‑layer            | pw per‑layer                   | linear | Δ pw‑budget |
|------------|------|------|-------------------------|--------------------------------|--------|-------------|
| `ens_a`    | 13   | 512  | (288, 256, 224, 192)    | (1024, 1280, 1536, 1792)       | 512    | −20%        |
| `ens_b`    | 29   | 512  | (288, 256, 224, 192)    | (1280, 1536, 1792, 2048)       | 512    | baseline    |
| `ens_c`    | 47   | 512  | (288, 256, 224, 192)    | (1536, 1792, 2048, 2304)       | 512    | +13%        |

`ens_b` — точно текущий рабочий рецепт `hn95_kd128_layerwise_signed_residual` + seed=29 (текущая модель остаётся валидной — её можно скачать из предыдущих ходок и не переобучать, но в этом рецепте мы переобучаем её честно с того же конфига для согласованной обработки teacher‑logits и preprocessing‑кэша).

`ens_a` — сдвиг PW‑codebooks на одно «деление» вниз; сохраняет рантайм MAC‑бюджет (channels одинаковы), уменьшает только размер кодбука → больше collision pressure → даёт нужную диверсификацию через семантически отличающиеся коллизии.

`ens_c` — сдвиг PW‑codebooks на одно деление вверх; немного выше виртуальный/compact ratio, но всё ещё легко в S3 SRAM (PW dominantly hashed; реальные int8 веса в Prepare ≈19→24 KB).

Все три используют идентичную архитектуру backbone, поэтому одинаково совместимы с прошивкой `hash_kws_runtime` и с pass‑4 SIMD‑путём (PW через `esp_nn_conv_s8_esp32s3`). Per‑model bundle отличается только содержимым codebook‑таблиц.

Tag‑схема:
- `hash_kws12_iterlab_v1_hn95_ens_a_s13`
- `hash_kws12_iterlab_v1_hn95_ens_b_s29`
- `hash_kws12_iterlab_v1_hn95_ens_c_s47`

## 2. Сохранение на диск (контракт)

Каждой модели соответствует независимый `run_dir` (`code/training/hash_runs/<tag>/`) и набор артефактов:

- `student_best.pt` — `{state_dict, experiment, result}` (формат текущего ноутбука).
- `hash_artifacts/<tag>/hash_kws_student_student.pt` + `..._metadata.json` — bundle для дальнейшей работы (через `export_model_bundle`).
- `hash_artifacts/<tag>/firmware_export/` — готовый firmware bundle (`hash_model_data.{h,cpp}`, `hash_model_settings.{h,cpp}`, `hash_model_export_metadata.json`). Конечная цель — три параллельных каталога:
  - `code/firmware/hash_kws_runtime_small/`
  - `code/firmware/hash_kws_runtime_base/`
  - `code/firmware/hash_kws_runtime_large/`
  чтобы прошивка выбирала вариант через `HASH_KWS_MODEL_VARIANT`.
- В конце ноутбука всё это zip‑уется и копируется на Drive под `DRIVE_CACHE_ROOT/runs/hash_ensemble_<timestamp>.zip`.

Юзер скачивает один zip и получает: 3× `student_best.pt`, 3× firmware‑бандла, `ensemble_results.json` + графики.

## 3. Препроцессинг

Без изменений относительно `hn95_kd128_layerwise_signed_residual`:

- `frontend_name="exact_microfrontend"` (точный TFLM‑микрофронтэнд, биты которого совпадают с прошивкой)
- `require_exact_microfrontend=True`, `normalize_mode="none"`, `cache_features=True`, `cache_dtype="int8"`
- SpecAugment отключён (как в основном рецепте; cached features и так дают одну детерминированную аугментированную копию)
- Speech Commands v0.02 готовится локально под `/content/diploma_esp32_distributed_nn/data` ровно как в `hashednet95_kws_colab.ipynb` (download через torchaudio, валидация наличия `validation_list.txt` / `testing_list.txt`)
- Кэш фич и кэш teacher‑логитов — на Drive (`HASH_KWS_FEATURE_CACHE_ROOT`, `HASH_KWS_TEACHER_LOGITS_CACHE_ROOT`)

Teacher тренируется один раз на `ens_base`‑seed (=29) и переиспользуется через `teacher_reuse_tag` для small/large; teacher‑logits кэшируются и подгружаются для всех трёх студентов. Это и время экономит, и гарантирует, что все 3 ученика дистиллируются от одного и того же учителя — диверсификация остаётся только в codebook+seed.

## 4. Оценка и агрегация в ноутбуке

После обучения все три студента грузятся с диска (best‑state) и считаются на полном test split.

### 4.1 Per‑model
`top1`, `top3`, per‑class precision/recall/F1, NLL, ECE (10‑bin), confusion matrix.

### 4.2 Базовые агрегаторы (как в исследовании, §5 ниже)
- `mean_logits` — headline.
- `mean_probs`.
- `conf_weighted` (вес ∝ 1/H(softmax)).
- `trimmed` (с N=3 вырождается в median — держим как контроль).
- `majority_vote`.

### 4.3 Развивающие агрегаторы (новое в v2)
Идея — взять то, что в исследовании оказалось headline (mean_logits), и **дать ему калиброванные / выученные веса**, не выходя за пределы пары флоатов на узел.

1. **Temperature‑scaled mean_logits.** На validation split подбираем по одной температуре T_k для каждой модели (минимизация NLL, line‑search). Затем агрегатор: `mean_k softmax(logits_k / T_k)`. Известный приём из калибровки сетей; стоит ровно 3 float‑а на флот, пересчёт на каждый infer тривиален. Гипотеза: если уверенности 3 моделей плохо согласованы (что подтверждается ECE per‑model), калибровка тянет mean‑агрегатор вверх.

2. **Learned 3‑weight aggregator.** На validation split тренируем 3 неотрицательных скаляра `w = (w_1, w_2, w_3)` через softmax‑параметризацию: `w = softmax(z)` где `z ∈ R^3` — обучаемые. Loss: CE на `sum_k w_k · logits_k`. Штрафа на разреженность нет (N=3, переобучения нет). Это упрощённая learnable‑version pwise‑weighted ensemble; в исследовании этого не пробовали, но идея прямо вытекает из «conf_weighted как worst smart aggregator» — если глобальные веса по моделям лучше per‑input weights, мы это увидим.

3. **Bonus diagnostic — per‑class disagreement.** Не агрегатор, но честная диагностика: для каждого класса считаем долю input‑ов, где хотя бы 2 модели расходятся. Помогает аргументировать в дипломе, почему ансамбль вообще нужен (на каких классах три скетча систематически коллизионируют по‑разному).

### 4.4 Прочее
- Oracle acc: hit@k=1, если хотя бы одна из трёх моделей даёт верный top‑1.
- Pairwise disagreement matrix 3×3.
- Std тестовой точности по 3 seed‑ам (проверка ×3.84 super‑bagging‑эффекта на нашей конкретной архитектуре).
- Hold‑out этикет: калибровка и learned weights настраиваются ТОЛЬКО на validation split, отчитываются на test split — методологически корректно, без leakage.

### 4.5 Запись результатов
`hash_artifacts/hash_ensemble/ensemble_results.json` со схемой:
```json
{
  "per_model": {"ens_a": {...}, "ens_b": {...}, "ens_c": {...}},
  "aggregators_test": {"mean_logits": {...}, "mean_probs": {...},
                       "temperature_scaled": {"T": [..], "metrics": {...}},
                       "learned_weights": {"w": [..], "metrics": {...}}, ...},
  "oracle_top1": float,
  "pairwise_disagreement": [[..3×3..]],
  "per_class_disagreement_rate": {...},
  "dispersion": {"single_std": float, "ensemble_std_subset": float}
}
```
Плюс 2 plot‑а: bar(accuracy by aggregator) и heatmap(pairwise disagreement).

## 5. Aggregation API (host + MCU)

### Host: `code/training/hash_ensemble/aggregation.py`

```python
# all funcs operate on logits: np.ndarray with shape [N_models, B_batch, C_classes]
def mean_logits(logits) -> np.ndarray                                # -> [B, C]
def mean_probs(logits) -> np.ndarray
def conf_weighted(logits) -> np.ndarray                              # 1/H(softmax) weights
def trimmed_mean(logits, drop: int = 1) -> np.ndarray
def majority_vote(logits) -> np.ndarray

# new in v2 — learned/calibrated aggregators
def fit_per_model_temperatures(val_logits, val_labels) -> np.ndarray # -> [N], NLL line-search
def temperature_scaled_mean(logits, temps: np.ndarray) -> np.ndarray
def fit_learned_weights(val_logits, val_labels, n_iters: int = 200,
                        lr: float = 0.05) -> np.ndarray              # -> [N], softmax-parametrized
def learned_weights_mean(logits, weights: np.ndarray) -> np.ndarray

# diagnostics
def oracle_topk(logits, labels, k: int = 1) -> float
def pairwise_disagreement(logits) -> np.ndarray                      # [N, N]
def per_class_disagreement_rate(logits, labels) -> dict[str, float]
```

Все функции чистые, без зависимости от torch (только numpy) — чтобы host‑симулятор стартовал быстро и не тащил модель в путь агрегации. `fit_*` функции возвращают компактные параметры (3 float‑а), которые сериализуются в JSON и могут быть зашиты в прошивку как константы — никакого ML‑рантайма на MCU не нужно.

### MCU: `code/firmware/hash_kws_aggregator/`

C++‑модуль:

- `HashEnsembleAggregator agg(num_nodes=3, num_classes=12, window_ms=1200);`
- `agg.setTemperatures(const float* temps);` // optional, если калиброванный путь активен
- `agg.setWeights(const float* w);`           // optional, learned weights (нормализованы, sum=1)
- `agg.submit(node_id, int8_logits[12], device_t, source_kind);` — кладёт пакет в кольцо.
- `agg.tryResolve(now_ms, &out_label, &out_score) -> bool` — если в окне есть пакеты от ≥2 узлов → mean_logits (или калиброванный/взвешенный путь), tie‑break по margin.
- Source‑promotion: `emit > infer`. Tie‑break: max margin (top1−top2).
- Совпадает с тем, что уже декодирует прошивка: 12 int8 логитов уже в пакете ESP‑NOW (`micro_speech.ino`).

Зависимости — только `<stdint.h>` / `<stddef.h>` / `<math.h>` (для exp/log в softmax). Без STL, без heap allocation. Буферы статические. Калибровочные параметры и веса задаются compile‑time константами из JSON — генератор шапки `aggregator_params.h` пишет ноутбук в конце ходки.

## 6. Host‑симулятор: `code/scripts/hash_ensemble_sim.py`

Подкоманды:

- `python hash_ensemble_sim.py eval --bundles small.pt base.pt large.pt --split test`
  Прогоняет полный test и печатает per‑model + ensemble метрики. Совпадает с тем, что считает ноутбук, как независимая проверка.
- `python hash_ensemble_sim.py demo --bundles small.pt base.pt large.pt --samples 50`
  Случайные test‑семплы, печатает вердикт каждой модели + результат агрегатора + истину; пишет `notes/Journal/hash_kws_telemetry/node{1,2,3}/events.jsonl` и `hash_kws_fusion/decisions.jsonl`, чтобы dashboard «зажил» без живой платы.
- `python hash_ensemble_sim.py serve --bundles ... --port COMx`
  Один реальный ESP32 + два виртуальных hash‑узла из локальных моделей, master‑агрегация на хосте. Параллельно отдаёт JSONL под существующие пути (`notes/Journal/hash_kws_telemetry/...`).

Модели грузятся через `torch.load(...)` + `experiment_from_dict(...)` + `build_student_model(...)` + `state_dict`.

## 7. MCU build‑switch

Без изменения общего скетча:

- В `micro_speech.ino` добавляем `#ifndef HASH_KWS_MODEL_VARIANT` → default `base`.
- `hash_runtime_bridge.cpp` инклюдит `hash_model_data.cpp` из соответствующего варианта‑директории.
- README — три однострочных примера компиляции под small/base/large.
- Aggregator‑модуль либо живёт на каждой плате (симметричный режим, как сейчас), либо на одной master‑плате.

## 8. Журналирование и приёмка

- Создать `notes/Journal/2026-05-09_hash_ensemble_plan.md` (этот файл) — DONE на этапе фиксации плана.
- На каждый этап (рецепты / ноутбук / aggregator host / aggregator MCU / симулятор / firmware switch) — короткая yaml‑запись `notes/Journal/2026-05-XX_hash_ensemble_<topic>.yaml`.
- Acceptance:
  1. py_compile проходит на всех новых .py.
  2. Build script ноутбука детерминистично генерирует валидный .ipynb (jinja/json check, как в `build_hashednet95_notebook.py`).
  3. Smoke‑run симулятора с заглушками‑моделями (random init) выдаёт корректные JSONL и не падает.
  4. После обучения в Colab: `ensemble_results.json` содержит per_model для 3 моделей и aggregators_test для ≥7 вариантов (mean_logits, mean_probs, conf_weighted, trimmed, majority_vote, temperature_scaled, learned_weights), oracle, pairwise + per‑class disagreement, dispersion.
  5. Один zip с тремя моделями + тремя firmware‑бандлами + `aggregator_params.h` скачивается с Drive.

## 9. Что вне scope этой итерации

- Дообучение/изменение teacher (используем ровно тот, что в `hn95_kd128_cached_schedule`).
- Добавление branches=2 / joint training (опровергнуты в исследовании; п.7.4 NOTES.md явно их дропает).
- Image‑модальность — отдельный трек (`2026-05-03_image_training_plan_004.yaml`).
- Полная замена `hash_kws_lab/models.py` — изменений в core нет, расширяемся через recipes.

## 10. Риски

- Drive‑квота: 3 экспортных бандла + 3 firmware‑бандла на одну ходку — мало. Кэш фич/логитов уже занят и многоразов; новая ходка только инкрементальна.
- Latency `ens_c`: больший PW‑бюджет → MAC‑ы те же (channels одинаковы), но кэш весов в `Prepare()` распухает с ~19 KB до ~24 KB. На S3 N16R8 запас есть; задокументируем фактический `Prepare RAM` в firmware‑exporter‑метаданных.
- Calibration leakage: temperature и learned weights настраиваются ТОЛЬКО на val split, метрики отчитываются на test. Если на test видим скачок learned vs mean_logits → проверить, что val/test действительно разделены (`validation_list.txt` / `testing_list.txt` в SC v0.02).
- «Почти‑homo» риск: если PW‑возмущение слишком мало, диверсификация уйдёт целиком в seed → ансамбль = чистое баггинг. Это OK (мы и так ожидаем баггинг‑амплитуду +1.5–2.5 п.п.), но захотим увидеть super‑bagging ×3.84 — это требование корректно сходящихся, но различных моделей. Pairwise disagreement в результатах покажет, не схлопнулись ли модели.
- Learned weights риск: при N=3 переобучение на val практически невозможно, но если все 3 модели почти идентичны по качеству, weights уйдут в ~(0.33, 0.33, 0.33) и mean_logits ≡ learned. Это не ошибка — это означает, что хитрая агрегация не нужна; так и пишем в diploma.

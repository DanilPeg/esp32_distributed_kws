# Hash Ensemble — research notes

Working notes for the "ensemble of HashedNets with different hashes and codebook sizes + smart aggregation" track.
This file is the running log: formalization, hypotheses, ablation plan, results, open questions.
The notebook `hash_ensemble_ablations.ipynb` is the main experimental tool.

---

## 1. Problem statement

Standard HashedNet:
a layer's weight tensor `W` is parametrized by a codebook `c ∈ R^K` and fixed analytic hash `π` plus
a fixed sign `s: indices → {-1, +1}`. The effective weight at position `i` is:

```
W[i] = s(i) · c[π(i)]
```

Only `c` is learned; `π`, `s` are deterministic at init. `K << numel(W)` gives compression.

**Open empirical question** for this track: does an ensemble of `N` HashedNets, each with its own `(π_k, s_k, K_k)`,
combined with a non-trivial aggregation, outperform a single HashedNet with the same total parameter budget,
and by how much, and under what conditions?

User's prior result (Fashion-MNIST): three models with different codebooks gave 0.81 ensemble vs
0.79 best single. Modest gain, likely because Fashion-MNIST is too easy. CIFAR-10 should leave more room.

---

## 2. Architecture

Mirrors `hash_kws_lab/models.py` with a few hash-specific upgrades:

- **DSCNN** for CIFAR-10: hashed `Conv2d` stem (3→C, 3×3, stride 2), `num_blocks` × hashed depthwise+pointwise
  blocks with optional residual, global avg pool, hashed final `Linear`. Defaults: `channels=64`, `num_blocks=3`.
- **Analytic hash** with multiplicative residue using large coprime primes; per-layer `layer_id` decorrelates
  collisions between layers; per-model `hash_seed` gives independent hash patterns across ensemble members.
- **Signed hash** mod 2 → ±1, kills systematic bias to codebook mean.

**Hash-specific upgrades** (additions on top of the diploma's `hash_kws_lab` baseline):

- **Hash-aware init** (`HASH_AWARE_INIT=True`): codebook init std `= 1/sqrt(fan_in_equiv)` instead of fixed
  `0.01`. Without this, effective `Var(W)` is constant in K and mismatches Kaiming for layers with
  large fan_in. Free win, no extra params.
- **Squeeze-and-excitation** (`USE_SE=True`): per-block channel attention. Two extra hashed FC layers
  per block (squeeze C→C/r, excite C/r→C). Cheap (≈2·C²/r virtual params per block), typical +1-2% boost
  on classification, naturally hashable since SE is highly redundant by design.
- **Parallel hash branches** (`HASH_BRANCHES=B`, default 1): two independent hash functions onto the **same**
  codebook, summed and 1/√B-normalized. Like double hashing in Bloom filters — same K, but each weight
  gets B independent collision attempts. Tested as ablation A9.

---

## 3. Formalization

Let `f(x; π, s, c)` denote a HashedNet's output (logits) on input `x`. An ensemble of `N` such models is
`{f_k(x) : k ∈ 1..N}`. An aggregator `A: R^{N×C} → R^C` maps stacked outputs to a final prediction:

```
ŷ(x) = A(f_1(x), ..., f_N(x))
```

Aggregators studied:

- `A_meanp`: mean of softmax probabilities.
- `A_meanl`: mean of logits.
- `A_conf`: confidence-weighted mean of probs. `w_k(x) ∝ 1/H(softmax(f_k(x)))`.
- `A_trim`: trimmed mean of logits per (B,C) cell, drops top/bottom α-fraction.
- `A_vote`: hard majority vote (one-hot averaged).
- `A_joint`: same as `A_meanp`, but the ensemble is trained end-to-end with the aggregated loss
  `L(A_meanp(f_1, ..., f_N), y)`, not independent per-model losses.

The motivation for variety: HashedNet collisions produce structured, input-dependent errors that vary per
model. Robust and confidence-weighted aggregators should exploit per-input reliability differences and
catastrophic outliers, going beyond what plain averaging gives.

---

## 4. Hypotheses

**H1.** Ensemble of `N` hashed nets with same `K` outperforms single hashed net with `K_total = N·K` at
matched total parameter budget. If true: ensembling buys something beyond raw parameter count.
If false: this whole track is "just have more parameters", and the architectural angle is empty. **Falsifier.**

**H2.** Ensemble of nets with **different** `K_k` (heterogeneous codebook sizes) outperforms an ensemble of
the same total budget but with all `K_k` equal. Reason: different `K` produce qualitatively different failure
modes. Small `K` is coarse; large `K` keeps fine detail but overfits hash-specific noise. Multi-resolution
diversity from heterogeneous `K` should beat the same-`K` ensemble. **Signature observation.**

**H3.** `A_conf > A_meanp ≥ A_meanl` on the heterogeneous-K ensemble. Confidence-weighted aggregation should
help on inputs where some models are clearly worse than others.

**H4.** `A_trim` is strictly better than `A_meanp` only when there exists a non-trivial fraction of inputs
where some single model produces catastrophically wrong logits.

**H5.** `A_joint` outperforms independent training under the same aggregator. Joint training lets models
specialize to cover failure patterns of others. Diagnostic: pairwise agreement rate. If models collapse
(>95% agreement), specialization didn't emerge.

**H6.** Variance of test accuracy across hash-seeds (fixed config) decreases significantly with `N` —
the "robustness, not magic" promise. Var(acc(ensemble)) ≪ Var(acc(single)) at matched total budget.

**H7.** Layer-wise heterogeneous K (e.g., `pw_heavy` or `proportional` profiles) brings diversity beyond
just same-K-different-seed. Tests where the hashing pressure should land for best results.

**H8.** Parallel hash branches (`B=2` on the same codebook, summed and normalized) reduce collision damage
inside a single model. Mean accuracy at fixed K should be higher for B=2 than for B=1, with smaller variance
across seeds. **If H8 holds strongly, much of the per-model collision-resilience benefit can be obtained
without ensembling at all** — important for separating "ensemble effect" from "double-hashing effect".

---

## 5. Ablation plan

Each item is a numbered cell-block in the notebook. Time estimates assume Colab T4 GPU.

- **A0 dense baseline** — DenseDSCNN, no hashing anywhere. Upper anchor. (~3 min)
- **A1 single hashed sweep** — sweep `K ∈ K_SWEEP`. Reports acc per K. (~25 min)
- **A2 same-K ensemble** — `N` models at `K=K_HOMO`, different hash seeds. Tests **H1**. (~20 min)
- **A3 different-K ensemble** — `N` models with `K_k` from `K_HETERO` (sum matched to A2). Tests **H2**. (~20 min)
- **A4 aggregator comparison** — five aggregators on A2 and A3 models. Tests **H3, H4**. (~1 min, no training)
- **A5 joint training** — same K profile as A3, end-to-end aggregated loss. Tests **H5**. (~20 min)
- **A6 layer-wise heterogeneity** — three profiles: uniform / pw_heavy / proportional. Tests **H7**. (~12 min)
- **A7 dispersion test** — `DISPERSION_N` seeds at fixed K, std of single vs std of N=3 subsets. Tests **H6**. (~25 min)
- **A8 OOD-lite robustness** — A2/A3 models on noise+brightness corrupted CIFAR-10. Tangential. (~2 min)
- **A9 parallel hash branches** — branches=1 vs branches=2 at fixed K, multiple seeds. Tests **H8**. (~15 min)

**Estimated full-mode runtime on T4: ~140 min.** FAST_MODE shrinks to ~30 min while still touching all
ablations.

---

## 6. Failure modes to watch for

- **Saturation.** If A1 accuracy is flat across K, the dataset is too easy for this model size and there's
  no headroom for ensembling to show. Switch to CIFAR-100.
- **Bagging-equivalence** (kills H1). If A2 ≈ A1 at `K = N·K_HOMO`, ensembling = "more parameters". The
  whole architectural angle is empty. **The first thing to look at in results.**
- **Hetero ≈ homo** (kills H2). If A3 ≈ A2 at matched budget, the multi-resolution diversity story doesn't
  hold and we're back to plain bagging.
- **Joint collapse** (kills H5). If A5 pairwise agreement > 95%, joint training pulled all models together
  and ensembling is meaningless after joint training.
- **Confidence miscalibration.** If A_conf does not exceed A_meanp, model confidence isn't well-calibrated
  and entropy weighting is no better than uniform.
- **Branches reduce single-model variance — but how much?** If A9 b=2 gives most of the gain that A2 ensembling
  gives, the ensemble story is partially redundant with simply double-hashing every model.

---

## 7. Results log

### 7.1 CIFAR-10 first run, 2026-04-29

Architecture: full-hashed DSCNN (channels=64, num_blocks=3, signed, residual, SE, hash-aware init).
Note: too low capacity overall — dense baseline only 0.6769, headroom small. Treat as preliminary.

| Ablation | Config                    | Test acc | Note |
|----------|---------------------------|----------|------|
| A0       | Dense                     | 0.6769   | low-capacity ceiling |
| A1       | K=128 → 4096 sweep        | 0.495 → 0.630 | monotone, no saturation |
| A2       | 3×K=1024 ensemble         | 0.6217   | best single 0.5974 |
| A3       | K=[256..2048] hetero, matched | 0.6197 | best single 0.6275 — **hetero ens BELOW best single** |
| A4 best  | conf_weighted on hetero    | 0.6239   | gain over mean_probs only +0.4% |
| A5 joint | hybrid loss not yet — plain mean-of-probs | 0.5771 | **−4% vs A3 indep**, agreement 0.328 (catastrophic) |
| A6       | uniform / pw_heavy / proportional | 0.591 / **0.648** / 0.567 | **pw_heavy single dominant** |
| A7       | single std/ens std         | 0.0054 / 0.0028 | reduction ×1.91 ≈ √3, plain bagging |
| A8 OOD   | hetero ens                 | 0.572 vs single 0.552 | **hetero gains more on OOD** |

Hypothesis verdict (CIFAR-10 alone):
- H1 ✗ — A2 ens loses to A1 at K = N·K_HOMO budget
- H2 ✗ — hetero ens below best single
- H3 ~ — conf_weighted slightly higher only on hetero
- H4 ✗ — trimmed worse than mean
- H5 ✗ — joint catastrophic without hybrid loss
- H6 ✓ trivial — bagging-style √N
- H7 ✓ — pw_heavy strongest single (+5% over uniform)
- H8 not run on CIFAR (added to plan, not yet executed there)

### 7.2 Speech Commands V2 / KWS-12, 2026-04-30

Architecture: full-hashed DSCNN (channels=64, num_blocks=4, signed, residual, hash-aware init, no SE).
Mirrors `hash_kws_lab` non-distillation recipes. Cached log-mel features.

| Ablation | Config                    | Test acc | Note |
|----------|---------------------------|----------|------|
| A0       | single K=500              | **0.9098** | reproduces ~0.91 baseline ✓ |
| A1       | K=128 → 1024 sweep        | 0.853 → 0.923 | monotone, A1@K=1024 = 0.9227 |
| A2       | 3×K=500 homo ensemble     | **0.9257** mean_probs / **0.9269** mean_logits | best single 0.9122 |
| A3       | K=[256,500,768] hetero, matched | 0.9232 | best single 0.9212 |
| A4 best  | mean_logits on homo       | 0.9269   | conf_weighted ≤ simple mean |
| A5       | pw_heavy single / ensemble | 0.9121 / 0.9267 | ~ on par with A0/A2 |
| A6       | branches=1 / branches=2   | 0.9117 / 0.9091 | **double hash hurts** |
| A7 joint | hybrid loss λ=0.4         | 0.9147 best agg, agreement 0.887 | non-collapsed but below A3 indep |
| A8       | dispersion: single / ens N=3 | 0.9108±0.0028 / **0.9273±0.00073** | reduction **×3.84**, ABOVE √3 |

Hypothesis verdict (KWS):
- H1 ~ formal — A2 (0.9257, codebook 1500) > A1@K=1024 (0.9227, codebook 1024); but extrapolating A1 to codebook 1500 ≈ 0.927 → bagging-equivalent. **Per-node** memory still wins.
- H2 ✗ — hetero (0.9232) below homo (0.9257) at matched codebook
- H3 ✗ — `mean_logits` ≥ `conf_weighted`, simple aggregation suffices
- H4 ✗ — trimmed strictly worse
- H5 ✗ — hybrid loss prevents collapse, but ensemble below independent
- H6 ✓✓ — std reduction ×3.84 > √3 = 1.73 (super-bagging), the strongest signal of the experiment
- H7 ~ — pw_heavy ≈ uniform (CIFAR-10 result not replicated; deployment-friendly though)
- H8 ✗ — branches=2 hurts by −0.26%, std grows

### 7.3 Cross-task synthesis

Consistently confirmed across both tasks:
- Ensembling provides ordinary bagging-magnitude accuracy gain (+1.5-2.5% over best single)
- Variance reduction with N is real and on KWS exceeds bagging theory (×3.84 reduction at N=3)
- Simple mean aggregation is sufficient — smart aggregators do not buy anything

Consistently refuted across both tasks:
- H2 (heterogeneous K): no advantage at matched budget on iid test, often hurts
- H4 (trimmed mean): no catastrophic collisions to remove
- H5 (joint training): even with hybrid loss, fails to beat independent

Task-dependent (not a robust pattern):
- H7 (pw_heavy): wins big on undersized CIFAR-10 model, neutral on appropriately-sized KWS model.
  Interpretation: matters only when uniform K causes destructive compression on small layers (dw, SE).
- H1 (ensemble vs big single): KWS shows a small but real ensemble advantage at matched codebook.
  CIFAR-10 was inverse — ensemble lost to big single. Likely reflects whether headroom exists.

OOD (only CIFAR-10 tested): heterogeneous ensemble benefits emerge under distribution shift.
The single argument that survives for K-heterogeneity story.

### 7.4 External analysis intake (2026-04-30)

A second-pass review (external) on the KWS results contributed several methodologically important
points which we accept and act on:

1. **Confounded diversity sources.** In A2/A3 we set `set_all_seeds(200+k)` AND `hash_seed=200+k`, i.e.
   per-model seeds simultaneously control hash, weight init, and DataLoader shuffle. We cannot claim
   the ensemble gain comes specifically from hash diversity. Need three control ablations:
   - hash-only-different (init+order shared, hash varied)
   - init-only-different (hash+order shared, init varied)
   - order-only-different (hash+init shared, order varied)
   - all-different (current setup)
   Only with these four can we attribute the ensemble gain to its actual source.

2. **Codebook-budget accounting is misleading.** `total_codebook_budget = 1500` understates real cost.
   Per A1: K=500 single → real=6740 (so ≈12 hashed-layer codebooks of 500 each + biases/BN); K=1024
   single → real=11980. Three K=500 models = ~20k real params on the fleet, which is **more** than a
   single K=1024 model (~12k). The honest framing for the diploma is **per-node** vs **fleet-total**:
   each node holds a small model; together they exceed a single-node-feasible big model. The
   distributed-systems framing is the one that survives this accounting; the "matched-codebook" framing
   used in the notebook so far does not.

3. **A8 dispersion is the strongest single result.** Std drops from 0.0028 → 0.00073, a ×3.84
   reduction at N=3 vs theoretical √3=1.73. Take this as the headline practical claim for the diploma:
   distributed multi-sketch makes the system robust to unlucky hash collisions, not just slightly
   more accurate. (Caveat: ×3.84 may be an artifact of saturation — all members near a common ceiling.
   Confirm with a wider N-sweep.)

4. **Heterogeneous K negative result is informative, not a failure.** The K=256 member adds noise,
   the K=768 member already does most of the work. Multi-resolution as a uniform principle does not
   work; comparable-strength sketches do.

5. **Aggregators don't matter much.** No catastrophic outliers to trim, no useful confidence
   differential to weight. Use `mean_logits` and stop optimizing.

6. **Branches and joint training: drop them.** Both refuted, neither passes the cross-task
   replication bar. They will not appear in v2.

7. **Pw_heavy as a deployment-friendly variant.** Not a quality win on KWS but not a quality loss
   either, and substantially simpler to implement on ESP32 (only pw layers need a hashed-conv runtime;
   stem/dw/fc can stay dense and use stock kernels). Worth keeping for the firmware track even if it
   doesn't headline the accuracy story.

Final reformulation, accepted: **N microcontrollers each store an independent hashed-sketch
representation of one compact KWS architecture. Per-node footprint is small; logit aggregation
across the fleet reduces hash-collision damage and yields higher and more stable accuracy than any
single hashed model with the same per-node budget.**

---

## 8. Open questions and follow-ups

- Is the A8 super-bagging variance reduction (×3.84 vs √N=1.73) real or saturation-driven? Resolved
  by an N-sweep with wider N range.
- For H1: does the per-node-vs-fleet reframing change the answer? At fixed *per-node* budget, the
  ensemble strictly beats single (because single can't run on one node either). At fixed *total*
  budget the picture is bagging-equivalent. Both framings should be reported.
- OOD-lite for KWS (noise injection, time shift) — does heterogeneity reactivate as on CIFAR-10?
- Oracle aggregator headroom: if oracle is +5% over mean_logits, a learned router might be worth it;
  if +1%, stop bothering.
- Port `aware_init` to `hash_kws_lab/models.py` regardless of ensemble outcome — it's a pure
  improvement to single hashed training.

---

## 9. Plan for v2 notebook (2026-05-01 onwards)

A new notebook `hash_ensemble_kws_v2.ipynb` to address the methodological gaps. Key additions:

- **C1 control ablations** (the central methodological fix): four 3-model groups —
  hash-only / init-only / order-only / all-different — measuring each source's contribution to
  ensemble gain.
- **C2 N-sweep** with a pool of 6 trained models. Plot accuracy and std as functions of N=1..6.
  Confirms whether the A8 super-bagging is real.
- **C3 oracle and disagreement diagnostics.** Oracle accuracy = upper bound (any-correct vote).
  Pairwise disagreement matrix. Per-class disagreement. Tells us whether smart routing has any
  remaining headroom.
- **C4 hash-design micro-sweep.** Signed vs unsigned hash, hash-aware init on/off — best-practice
  validation for the underlying hashed model itself, not just for ensembling.
- **Strict accounting.** Every result reports `per_node_real`, `total_real`, `per_node_codebook`,
  `total_codebook`, `virtual_params` explicitly, so per-node-vs-fleet framings are unambiguous.
- **Drop:** A6 branches, A7 joint training (refuted on both tasks).
- **Keep:** A0 baseline, A1 sweep, A2 homo ens (becomes "all-different" arm of C1), A3 hetero ens,
  A4 aggregators (compressed: just report mean_logits + best alt), A5 pw_heavy single.

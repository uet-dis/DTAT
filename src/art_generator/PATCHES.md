# Generator patches - what we changed in ART and why

Baseline: `adversarial-robustness-toolbox==1.20.1`. Every `file:line` below refers to that version.

## The problem

ART's attacks are designed for images: every pixel has the same scale and a global `[0, 1]` box.
Network-flow features are the opposite - they span ~8 orders of magnitude (bytes ≈ 1e8, rates in [0, 1]),
are never negative, and some columns are categorical or binary. Run unmodified, the attacks produce
pools that look strong but are not valid evidence of evasion:

| Symptom with stock ART on tabular data | Consequence |
|---|---|
| Raw-space L2/L-inf budgets (eps = 0.1–0.2) | a step that is meaningful for one feature is a no-op on large-scale features (JSMA left most rows unchanged) and destructive on small-scale ones (DeepFool moved samples enormous distances in σ units) |
| No domain constraint | C&W and DeepFool produced negative counts/bytes; clamping them back removed most of the evasions, i.e. the evasion existed only in an unrealizable region |
| `targeted=True` / ground-truth `y` passed as the target | PGD optimised *toward* the true label; JSMA's ASR was 0 |
| Image-style init and loops | HSJA drew starting points far from every real record; its epsilon loop could spin forever |

## The principle

**Change the coordinates, the box and the declared configuration - never the attack itself.**
The attacked model is exactly the trained model; FGSM, PGD and C&W keep ART's own algorithm;
DeepFool, ZOO, HSJA and JSMA are thin subclasses that copy ART's code verbatim and alter one
clearly identified line. Any difference in results is therefore due to the measurement frame, not to a
different attack.

## Shared infrastructure

- **InputNorm z-space** (`inputnorm_adapter.py`). Minimum-norm attacks and JSMA run in
  `z = (x − μ) / σ` taken from the trained DNN checkpoint, through an adapter that shares the model's
  weights. "Minimal" now means minimal in σ units, with the same geometry on both datasets.
- **Per-feature data box, `pad = 0`** (`data_zspace_clip`). Lower edge = feature minimum, upper edge =
  feature maximum, so every adversarial value lies inside the observed data range: no negatives, no
  out-of-domain values.
- **σ-scaled budgets** (`eps_scaling.py`, FGSM/PGD). `eps_i = k · σ_i`, with `eps = 0` on frozen columns.
  ART already accepts per-feature eps arrays, so only the array is built.
- **Safe targets** (`targets.py`, `attack_generator.py`). When an attack is targeted, the target is the
  highest-scoring class other than the true label (the true label's score is set to −∞ before ranking,
  so it can never be picked, even for samples the model misclassifies); the ground truth is never handed to ART as a target.
- **Frozen columns.** Categorical/binary features are masked during the attack and restored bit-exact
  afterwards; pool shape and label column are unchanged, so downstream code needs no change.

## One pipeline, seven attacks

Every generator wrapper (`AttackGenerator` subclass) runs the same stages. The patches are exactly the
cells below that differ from stock ART, so each change has a fixed place.

| # | Stage | What happens | Where |
|---|---|---|---|
| 1 | **Parameters** | `ATTACK_PARAMETERS` overrides the wrapper defaults and builds the ART attack | `update_generator_params`, constructor |
| 2 | **Targets** | untargeted: y = ground truth; targeted: best class other than the true label | `resolve_targets` |
| 3 | **Enter attack space** | raw → z (`to_z`), or stay raw | wrapper `generate` |
| 4 | **Declare the domain** | per-feature box, or gradient mask + per-feature eps | wrapper `generate` / `generate_adv.py` |
| 5 | **Core attack** | ART's optimisation loop | ART or a one-line subclass |
| 6 | **Run control** | batching, retry, timeout, placeholder rows | HSJA, JSMA only |
| 7 | **Return to raw and clamp** | z → raw, clip to the raw minimum, restore frozen columns | wrapper `generate` |
| 8 | **Package** | DataFrame with the original feature names and the label column | every wrapper |

Which stages differ from stock ART (● = changed, ○ = stock/not applicable):

| Stage | FGSM | PGD | C&W | DeepFool | ZOO | HSJA | JSMA |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 Parameters | ● σ-eps | ● σ-eps, untargeted | ● retuned | ● retuned | ● retuned | ● retuned | ● θ = 1 σ, γ = 0.1 |
| 2 Targets | ● safe target if targeted | ● same | ● same | ○ untargeted | ○ untargeted | ○ untargeted | ● own target rule |
| 3 Attack space | ○ raw | ○ raw | ● z | ● z | ● z | ● z | ● z |
| 4 Domain | ● mask, eps = 0 on frozen | ● same | ● box on estimator | ● box on subclass | ● box (full-set when sharded) | ● box | ● box on estimator |
| 5 Core attack | ○ stock | ○ stock | ○ stock | ● subclass | ● subclass | ● subclass | ● subclass |
| 6 Run control | ○ | ○ | ○ | ○ | ○ | ● | ● |
| 7 Clamp and restore | ○ mask suffices | ○ mask suffices | ● box clip + floor + restore | ● floor + restore | ● floor + restore | ● floor + restore | ● floor + restore |
| 8 Package | ○ | ○ | ○ | ○ | ○ | ○ | ○ |

This gives three kinds of change, in increasing intrusiveness:

1. **Outside ART (stages 1–4, 6–7)** - FGSM, PGD, C&W. Nothing inside ART is modified.
2. **Inside ART's loop (stage 5)** - DeepFool, ZOO, HSJA, JSMA. Each subclass copies the ART method
   verbatim and changes one identified spot (listed below), because the spot cannot be reached from outside.
3. **Configuration only (stage 1)** - strength values, always recorded with a reason.

Stages 3, 4 and 7 are the same code idea in every z-space attack, so a reviewer can check them once.

## Per-attack walk-through

Stage numbers refer to the pipeline table above. A stage that is not listed for an attack is stock ART.

### FGSM - outside ART (stages 1, 2, 4)

1. **Parameters.** CIC uses `eps_per_std = 0.1`: `generate_adv.py` turns it into a per-feature array
   `eps_i = 0.1 · σ_i` (σ from the DNN/TabNet InputNorm) and passes it as `eps`. A raw `eps = 0.1` is a
   no-op on CIC features whose σ is many orders of magnitude larger. NSL keeps the raw `eps = 0.1`. The unused `eps_step` was
   removed from the NSL config (ART reads it only when `minimal=True`).
2. **Targets.** With `--targeted`, the target is the best class other than the true label.
3. **Space.** Raw, as in stock ART.
4. **Domain.** The wrapper builds a gradient mask that is 0 on categorical/binary columns, and the eps
   array is 0 on the same columns, so the L-inf projection freezes them exactly.
5. **Core.** Stock `FastGradientMethod`.
7. **Return.** Nothing to clamp: the mask already guarantees that frozen columns are untouched.

### PGD - outside ART (stages 1, 2, 4)

1. **Parameters.**
   - The wrapper default `targeted=True` pushed samples toward their own label (it minimised the loss), so
     the "evasion" was really regularisation. It is now `False` (untargeted ascent).
   - `max_iter` 200 → 100 (both datasets): ART's PGD has no early stop, so 200 was a 10× over-provision.
   - CIC uses `eps_per_std = 0.2`, `eps_step_per_std = 0.02` (same σ-array mechanism as FGSM); NSL uses raw
     `eps = 0.2`, `eps_step = 0.02`, which traverses the ball in 10 steps.
2. **Targets.** As for FGSM.
4. **Domain.** Gradient mask and eps = 0 on frozen columns, as for FGSM.
5. **Core.** Stock `ProjectedGradientDescentPyTorch`.

### C&W (L2) - outside ART (stages 1, 2, 3, 4, 7)

1. **Parameters.**
   - `confidence` 0.0 → 0.1: a positive margin keeps evasions valid under small logit drift.
   - `binary_search_steps` 3 → 9 and `max_iter` 3 → 10: the old setting made only 9 gradient steps in
     total, and the constant search stayed far below the useful range.
   - `initial_const` 0.01 → 2.0 (CIC) / 1.0 (NSL). 0.01 is ART's default for unnormalised inputs; in
     z-space the distance term is O(0.1–1), so the classification term vanished and the attack
     under-converged (almost no sample evaded). The constant was raised until ASR plateaued while the
     perturbation norm stayed small; much larger values lose the minimum-norm property.
2. **Targets.** As for FGSM.
3. **Space.** z-space through an adapter that shares the model's weights, so the attack sees exactly the
   trained model. The distance term is now measured in σ units (in raw space the optimiser destroyed
   some rows and left many others unchanged).
4. **Domain.** The per-feature box (`pad = 0`) is attached to the estimator before every `generate`
   (ART reads it once per call). Neither alternative works: unbounded z lets the optimiser produce
   negative counts (the v1 pool did), and the raw box (0 … 4e8) makes the tanh remap lose many σ of
   precision.
5. **Core.** Stock `CarliniL2Method`.
7. **Return.** Clip z into the box (tanh-remap round-off), convert to raw, clamp to the raw per-feature
   minimum (float32 round-off), restore frozen columns bit-exact.

### DeepFool - inside ART's loop (stages 1, 3, 4, 5, 7)

1. **Parameters.** `nb_grads` 5 → 10 on CIC (12 classes; 5 misses boundaries outside the top 5; NSL keeps
   5 = all classes). `epsilon` 1e-6 → 0.02 (the paper's overshoot; 1e-6 lands exactly on the boundary,
   which is fragile).
3. **Space.** z-space, as for C&W (DNN/TabNet only).
4. **Domain.** The per-feature box is stored on the subclass (`_data_box`).
5. **Core.** Stock ART has two bad options on tabular data:
   - with `clip_values`, every step is multiplied by the box width (`deepfool.py:177`): on a 4e8 box the
     step covers the whole feature range and destroys the sample (the model's accuracy on the pool
     collapses);
   - without `clip_values`, steps are unscaled but the trajectory is unbounded, so the minimal step lands
     at negative raw values (clamping back removes most of the evasions).

   `_DataBoxDeepFool` keeps the step **unscaled** and clips the batch into the box **after every step and
   after the overshoot**. Everything else is ART 1.20.1 verbatim (`deepfool.py:101-214`).
7. **Return.** Convert to raw, clamp to the raw minimum, restore frozen columns.

### ZOO - inside ART's loop (stages 1, 3, 4, 5, 7)

1. **Parameters.** CIC `confidence` 10.0 → 0.5 (aligned with NSL); `variable_h` 0.002 on CIC
   (0.005 on NSL); `batch_size = 1`, `nb_parallel = 16` on both.
3. **Space.** z-space through a black-box estimator that overrides only `predict` (z → raw, then the real
   model). ZOO only calls `predict`, so this also serves the tree models (xgb/catb/bagging/gbm).
   Prediction is limited to one thread: otherwise each 1-row call starts a thread pool, thousands of
   times per sample.
4. **Domain.** Per-feature box on the estimator. In sharded runs the box (`zspace_clip`) and the raw
   floor (`raw_floor`) are computed once on the full set and passed to every shard, so all shards declare
   the same domain.
5. **Core.** ART touches the box in two places. Place 1 (the elementwise clip after each iteration,
   `zoo.py:241-243`) already broadcasts per-feature arrays and is inherited. Place 2 (the ADAM noise
   projection, `zoo.py:564-566`) assumes a scalar box. `_DataBoxZOO` copies `zoo.py:525-567` verbatim and
   changes that single line to clip by the edges of the feature the noise index belongs to
   (`feat = idx % n_features`); with `batch_size = 1`, as configured, this is identical to ART's semantics.
7. **Return.** Convert to raw, clamp to the raw minimum (the full-set minimum when sharded), restore
   frozen columns.

### HSJA - inside ART's loop (stages 1, 3, 4, 5, 6, 7)

1. **Parameters.** CIC `max_iter` 5 → 10, `max_eval` 300 → 600, `init_eval` 10 → 20 (the same query
   budget as NSL). `init_radius = 2.0` is a new config key.
3. **Space.** z-space through the same black-box estimator as ZOO.
4. **Domain.** Per-feature box on the estimator (full-set box when sharded) and the frozen-column mask
   passed to ART natively. ART's starting step `0.1 · (clip_max − clip_min)` automatically becomes 10% of
   each feature's observed range.
5. **Core.** Two subclasses:
   - `_LocalInitHopSkipJump` - stock ART draws the first adversarial point uniformly over the whole box
     (`hop_skip_jump.py:309/344`), an image habit that places the point far from any real record.
     The subclass draws `x + uniform(−1, 1) · init_radius` as in the paper (2 σ in z-space) and clamps the
     draw into the box, because an anchor outside the box leaves the binary search with an unreachable
     boundary.
   - `_GuardedHSJA` - ART halves epsilon with `while not success`, which never ends if the anchor is
     outside the box or stuck in a corner. The guard stops when the clipped candidate equals the current
     sample (no smaller epsilon could move it, so no feasible success is missed), caps halvings at 200
     (float64 needs ~60), and skips non-finite updates. The loop is `hop_skip_jump.py:376-462`, verbatim
     except the halving loop at `:441-449`.

   The per-sample "Found initial adversarial image" message was lowered to DEBUG; it used to break the
   progress bar.
6. **Run control.** Batching, retry, a SIGALRM timeout and the placeholder rule (a failed batch keeps its
   original rows, or is dropped on request).
7. **Return.** Convert to raw, clamp to the raw minimum (full-set when sharded), restore frozen columns.

### JSMA - inside ART's loop (stages 1, 2, 3, 4, 5, 6, 7)

1. **Parameters.** `theta = 1.0` - one σ per step (a raw `theta = 0.02` leaves most rows unchanged on
   large-scale features); `gamma = 0.1` - a deliberately small budget, at most ~4 modified features, stated
   in the threat model.
2. **Targets.** JSMA reads `y` as the *target*, so v1 passed the ground truth and the attack raised the
   confidence of the true label (ASR = 0). The target is now the highest-scoring class other than the
   true label, computed with the adapter estimator; `y` only labels the output rows. Always targeted,
   deterministic.
3. **Space.** z-space, DNN/TabNet only (the trees have no gradient).
4. **Domain.** Per-feature box attached to the estimator once for the whole dataset.
5. **Core.** `_DataBoxJSMA` copies ART's `generate()` verbatim (`saliency_map.py:92-204`) and changes
   two things:
   - ART's pairwise update clips with a scalar (`saliency_map.py:154-160`); the subclass indexes the box
     edge by the feature selected for each row.
   - The box edges are cast to float32. ART resets its search space with exact equality
     (`batch == clip_value`); with float64 edges the stored float32 value never equals the edge, so a
     feature at the boundary is chosen again and again and the loop never ends.
6. **Run control.** As for HSJA.
7. **Return.** Convert to raw, clamp to the raw minimum, restore frozen columns.

## Why the results can be trusted

- **Validity is checked independently of the generators.** A read-only audit re-derives, from the clean
  files, that frozen columns are untouched, that no value is negative or beyond the observed maximum
  (except inside a declared bounded budget), and that budgets are respected. Displacement is reported
  but is not a pass/fail criterion.
- **Reproducible.** Budgets live in `ATTACK_PARAMETERS` with a recorded reason for each change; sharded
  runs (ZOO/HSJA) share the full-set box and floor, and the same `--seed` regenerates the same pool.
- **Honest scope.** The patches enforce domain validity (non-negative, in-range). They do not yet
  guarantee semantic validity of a flow (integer-valued counts, protocol consistency).

## What we deliberately did not touch

- **The attack algorithms.** FGSM, PGD and C&W run ART's own code, with no subclass and no copied
  functions; we only add three layers around the call: (1) *before* - coordinates and budgets (z-space
  for C&W, σ-scaled per-feature eps for FGSM/PGD), a gradient mask for frozen columns, and a safe target
  when targeted; (2) *the box* - for C&W, the per-feature data box is attached to the estimator before
  each `generate`; (3) *after* - clamp into the box and the raw minimum, then restore frozen columns
  bit-exact.
- **Attack strength.** `eps`, `confidence`, `max_iter` and similar values change only in
  `ATTACK_PARAMETERS`, each with a comment giving the reason - never implicitly in code.
- **Attack goal.** Attacks are untargeted (any misclassification counts); only JSMA is targeted.
- **The pool format.** Output has the same rows and columns as the clean input. If an attack fails on a
  sample, that row keeps its original values (`placeholder = original`) instead of being dropped, so
  training and evaluation code downstream needs no change.

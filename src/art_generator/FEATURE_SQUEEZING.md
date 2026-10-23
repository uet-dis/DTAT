# Feature Squeezing (FS) - notes

Status: **removed from the defense and from the step-8 evaluators.** This file records what existed
and how it related to the attack generators. It is meant to stay local (to be added to `.gitignore`).

## Where it lives

- `src/art_classifier/feature_squeezing.py` - `build_feature_squeezing_defences(params)`; local-only
  (listed in the root `.gitignore`).
- `src/art_classifier/art_classifier.py` - `_build_preprocessing_defences` imports the module
  optionally; when the file is absent it returns `None` and the wrappers build no defences. All five
  wrappers call it, but nothing passes FS params any more.
- `cli/apelid/8_attack_evaluation/classifier_evaluate_adv.py`, `classifier_parallel_ens_eval.py` -
  the `--fs-enable`, `--fs-bit-depth`, `--fs-config` flags and the `wrapper_params` FS blocks were removed.
- `cli/apelid/preprocessing/prepare.py` - the `clip_values_per_feature` meta key (FS grid input) was removed.

## Design (as last implemented)

- ART `FeatureSqueezing` quantises each feature to `2^bit_depth` levels over a clip range.
- **Per-feature clip range** `(min_i, max_i)` from the full reference set, so the grid step is
  `(max_i − min_i) / (2^b − 1)` per feature. The earlier global `(0, 1)` image grid was scale-blind on
  tabular data (a no-op for byte/count features, destructive for unit-scale ones) and was removed;
  without `fs_clip_values` FS is disabled with a warning rather than guessing a grid.
- Constant features (`lo_i == hi_i`, e.g. `num_outbound_cmds` on NSL-KDD) are coerced to
  `(lo_i, lo_i + eps)` so ART does not reject the whole box.
- **One bit depth per model.** Grouped per-feature bit depths (`fs_group_bits`) were dropped on
  2026-10-12: tuning did not beat a single per-model bit depth (CIC: +6.4 pp on xgb but −8.6 pp on
  catb; NSL: strictly worse than single b = 6).

## Interaction with the generators

- C&W `confidence` 0.0 → 0.1 was chosen so evasions keep a positive margin against small logit drift
  such as quantisation; the expectation was that pooled evasions would survive FS.
- Generation always targeted the model **without** FS. An adaptive pool (`generate_adv.py
  --using-robust`) therefore still attacks the undefended wrapper; it must go to its own
  `--output-dir`, never overwrite `adv_samples/test`.
- The attack-side box (`data_zspace_clip`, `pad = 0`) and the FS grid use the same per-feature
  min/max idea, but are independent code paths.

## Published history

FS is present in the already-published history (`README.md` section "Lightweight Feature-Squeezing
Layer", the old evaluators, the notebook, the old `art_classifier.py`). `.gitignore` cannot remove it;
that would need a history rewrite.

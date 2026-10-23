# Pool audit — how to read the metrics

Scope: the read-only evaluation of adversarial pools (`{res}_{model}_{attack}_adv.csv`), run by
`cli/apelid/8_attack_evaluation/pool_evaluation.py`. Metrics come from `pool_audit.py` (integrity,
displacement, validity), `distance_metrics.py` (paired log-space distance) and `asr_metrics.py`
(active coverage, ASR).

This file explains **why each metric exists and how to interpret it**. It makes no claim about
attack results.

## Two questions, kept apart

| Question | Answered by | Role |
|---|---|---|
| **Proximity** — how far did an adversarial row move from its clean row? | displacement metrics | reported, never a pass/fail gate |
| **Effectiveness** — does the model get fooled? | ASR | reported separately |
| **Validity** — is the row still inside the data domain? | frozen-column and domain checks | the only gate |

Low proximity distance does not prove a strong attack, and a high ASR does not prove a small
perturbation. Neither is used to excuse the other.

## The metrics

All distances are computed per row on the clean/adversarial pair, then summarised by median and p90
over rows.

| Metric | Definition | Why it is there |
|---|---|---|
| **Active coverage** | share of rows with `max_i |z_adv_i − z_orig_i| > 1e-5` over continuous features | distances are summarised over active rows only; without this number a small distance on very few active rows would be read like the same distance on almost all rows. Rows left unchanged (failed batches, placeholders) show up here instead of diluting the distance |
| **InputNorm L2** | `sqrt(Σ dz_i²)`, `dz_i = (x_adv_i − x_orig_i) / σ_i`, continuous features | the standard norm-based proximity measure; the σ units put features of very different scale on one axis |
| **InputNorm L∞** | `max_i |dz_i|` | L2 can hide one large change among many small ones; L∞ exposes the most-changed feature |
| **L0 / sparsity** | number of features with `|dz_i|` above a small tolerance | fewer changed features are easier to explain; categorical/binary columns are excluded and checked separately |
| **Frozen integrity** | `frozen_changed_rows = 0` | categorical/binary columns must be bit-exact; a pool that changes them is invalid for the pipeline however small its L2 |
| **Domain range** | `min_orig_i ≤ x_adv_i ≤ max_orig_i` per feature | no value outside the observed data range (for bounded attacks, only within the declared budget) |
| **Paired log-RMS** | `delta_i = log1p(x_adv_i) − log1p(x_orig_i)`; `log_rms = sqrt(mean(delta_i²))`; also `log_l2`, `log_l1_mean` | heavy-tailed features (bytes, duration, packet counts) make raw-scale distances unreadable; the log scale measures relative change. Computed on mutable, finite, non-negative raw features and active rows only |

Notes on the less standard choices:

- **σ comes from the DNN checkpoint.** Using one model's σ as the common geometry for every attacked
  model is a project convention, not a standard tabular metric. It is what makes pools of different
  models comparable.
- **Log-RMS is a diagnostic, not a standard metric.** It has the same form as RMSLE
  (`sqrt(mean((log(1+ŷ) − log(1+y))²))`), applied across the features of one row pair instead of across
  predictions. It should be described as an *RMSLE-inspired paired log-space proximity diagnostic*.
- **Active coverage is project-specific.** No cited source defines it; it exists so that unchanged rows
  cannot hide inside a summary.
- **Median and p90 carry no universal threshold.** Typical rows are read from the median, tails from p90.

## Where each idea comes from

| Component | Basis | What the project changes |
|---|---|---|
| L1/L2/L∞ proximity | Ballet et al. 2019; He et al. 2024; TabAttackBench 2025; Carlini & Wagner 2017 | norm-based proximity on continuous InputNorm coordinates |
| L0 / sparsity | He et al. 2024; TabAttackBench 2025 | tolerance applied in z-space |
| Immutable features | Ballet et al. 2019; He et al. 2024 | enforced as a bit-exact gate for categorical/binary columns |
| Feasibility / domain range | Ballet et al. 2019; He et al. 2024 | observed per-feature min/max, not a semantic profile |
| Log-RMS | RMSLE (scikit-learn `root_mean_squared_log_error`) | adapted from prediction error to paired feature displacement |

References:

- Ballet et al., *Imperceptible Adversarial Attacks on Tabular Data*, 2019 — <https://arxiv.org/abs/1911.03274>
- He et al., *Investigating Imperceptibility of Adversarial Attacks on Tabular Data*, 2024 — <https://arxiv.org/abs/2407.11463>
- He et al., *TabAttackBench: A Benchmark for Adversarial Attacks on Tabular Data*, 2025 — <https://arxiv.org/html/2505.21027v2>
- Carlini & Wagner, *Towards Evaluating the Robustness of Neural Networks*, 2017 — <https://arxiv.org/abs/1608.04644>

## How to report a result

Main evidence:

```text
active coverage
InputNorm L2 median / p90
InputNorm L∞ p90
L0 median
frozen_changed_rows
domain violations
```

Supporting evidence for heavy-tailed data: `log_rms` median/p90 and `log_l1_mean` median.
Reported on their own line: ASR over active rows and over all rows.

A defensible claim is a comparison on the same dataset, with its coverage stated:

> In log-transformed raw feature space, pool X has lower paired displacement than pool Y on the same
> dataset, with active coverage A% and B%, and `log_rms` median M and p90 P.

Avoid "`log_rms < X` means a small perturbation": there is no threshold that holds for both datasets.
A low ASR weakens only the claim about attack effectiveness; it says nothing against proximity.

`PASS` means the pool passed the integrity, frozen-column and domain checks. It does not mean the rows
are semantically valid or physically realistic.

## Limits

These metrics do not show that:

- integer/count fields keep their semantic type;
- rate and categorical relationships stay consistent;
- feature interdependencies are preserved;
- a row can be turned back into real traffic;
- the adversarial row is physically realistic.

Semantic checks and projection are a separate, later phase. Until then, the accurate wording is
**bounded paired displacement in normalised and log-transformed raw feature spaces**, not
"semantic-valid" or "physically realistic" data.

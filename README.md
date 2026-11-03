# Transfer-Aware and Constraint-Consistent Adversarial Curricula for Heterogeneous Intrusion Detection Ensembles

Code, data, adversarial pools, checkpoints, and all result files for the paper
*"Transfer-Aware and Constraint-Consistent Adversarial Curricula for Heterogeneous Intrusion
Detection Ensembles"* (submitted to *Applied Soft Computing*). The method is called **DTAT**
(Directional Transfer-Aware Adversarial Training). The installable package keeps the earlier
project name **HEDGE** (`hedge`).

Architectural diversity in a heterogeneous intrusion-detection ensemble does not guarantee
adversarial independence: a perturbation generated against one component can also affect other
learners that share its features. DTAT builds a **target-specific adversarial curriculum** for every
member of an ensemble of deep-learning (DL) and tree-based learners, and a **domain-constraint layer**
(Ω) audits and repairs adversarial vectors and training records so that they remain consistent
network-flow records.

## Contents

- [Method](#method)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data and artifacts](#data-and-artifacts)
- [Rebuilding the paper's tables and figures](#rebuilding-the-papers-tables-and-figures)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Main results](#main-results)
- [Citation](#citation)

## Method

DTAT has four stages:

1. **Source-indexed generation.** Each clean-trained source generates one pool per procedure.
   Gradient-based procedures (FGSM, PGD, DeepFool, C&W, JSMA) need a differentiable source
   (DNN, TabNet); query-based procedures (ZOO, HSJA) apply to every component.
2. **Routing policy.** Each tree target receives its own query-based pools plus the gradient-based
   pools of its designated DL sources; each DL target receives its own pools.
3. **Per-procedure curation.** A quota τ per procedure (12,000 on CSE-CIC-IDS2018, 5,000 on
   NSL-KDD) is split equally among the contributing sources and among the classes.
4. **Learner-specific fitting.** Each component is refitted on the balanced training set plus its
   curriculum; the ensemble combines the components by fixed weighted soft voting.

**Domain constraints.** `src/art_generator/domain_constraints.py` defines two levels:

- **Ω₁ (consistency):** immutable fields, integrality, value ranges, and the arithmetic relations
  among flow statistics (CSE-CIC-IDS2018), or integrality, ranges, and rate granularity (NSL-KDD).
- **Ω₂ (attacker capability):** Ω₁ plus only additive changes of the forward direction
  (padding, packet injection, delays) on CSE-CIC-IDS2018, or of `duration` and `src_bytes` on NSL-KDD.

The repair operator Π_Ω recomputes every derived statistic from the primary ones; its strict
variant is used to repair the WGAN-synthesized training records. **DTAT-Ω** builds the
curriculum from Π_Ω-repaired pools.

### Ensemble components

| Family | Model (paper abbreviation, code name) | Voting weight |
|---|---|---|
| Deep learning | DNN (`dnn`), TabNet (`tabnet`) | 0.09, 0.10 |
| Tree-based | XGBoost (XGB, `xgb`), Gradient Boosting Machine (GBM, `gbm`), CatBoost (CBT, `catb`), bagging meta-estimator (BME, `bagging`) | 0.27, 0.18, 0.18, 0.18 |

### Curricula compared in the paper

Every adversarial curriculum is fitted with four curation seeds (1337–1340).

| Arm (code name) | Tree targets | DL targets |
|---|---|---|
| Base | clean | clean |
| IAT (`iat`) | own ZOO/HSJA, 2τ | DTAT networks |
| IAT-vol (`iatvol`) | own ZOO/HSJA, 3.5τ each (7τ) | DTAT networks |
| S={DNN} (`dnnonly`), S={TabNet} (`tabonly`) | own query + one network's gradient pools | DTAT networks |
| Pooled (`pooled`) | all sources, quota split equally | all sources |
| Reverse (`reverse`) | own ZOO/HSJA | own pools + the trees' query pools |
| Random (`random`) | one random source per procedure | same rule |
| **DTAT** | own query + DNN and TabNet gradient pools, τ/2 each | own pools |
| DTAT-Ω₁ / DTAT-Ω₂ (`dtat_o1`, `dtat_o2`) | as DTAT, pools repaired with Π_Ω₁ / Π_Ω₂ | as DTAT, repaired |
| DTAT-Ω₁′ / DTAT-Ω₂′ (`dtat_o1s`, `dtat_o2s`) | as above, on the strictly repaired training set | as above |

## Repository layout

```
HELGE/
├── src/                         # installable package (top-level modules)
│   ├── art_classifier/          # ART wrappers for xgb / catb / gbm / bagging / dnn / tabnet
│   ├── art_generator/           # attack generators, eps scaling, repair R0
│   │   └── domain_constraints.py   # Ω1/Ω2, repair Π_Ω (and strict variant), audit
│   ├── resampling/, training/, argparsers/, utils/
├── cli/hedge/                  # pipeline entry points
│   ├── 1_preprocessing_*/ … 4_merge/   # cleaning, split, WGAN + K-means balancing
│   ├── 5_training/              # clean and adversarial training
│   ├── 7_generating_samples/    # pool generation, merging, per-procedure curation
│   ├── 8_attack_evaluation/     # component-level and per-branch ensemble ASR
│   └── 9_revision/              # constraints, ablation arms, constrained/adaptive attacks,
│                                #   statistics, tables, figures (see below)
├── params/training/             # learner hyperparameters
├── notebooks/evaluate.ipynb     # original evaluation runs
└── results/revision/            # every measured result of the paper (CSV) + generated
    ├── NSLKDD/, CIC-2018/       #   audits, eval/, constrained/, xfer/, fit_times.csv
    ├── tables/, figures/        #   tables (LaTeX) and figures regenerated from the CSVs
    └── stats.md, summary.md
```

Scripts in `cli/hedge/9_revision/`:

| Script | Purpose |
|---|---|
| `project_test_vectors.py` | audit vectors against Ω and write Π_Ω-repaired copies (`--strict` for training pools) |
| `repair_dbal.py` | strict Π_Ω₁ repair of the balanced training set (D_bal′) |
| `build_ablation_pools.py` | build the curricula of every arm with the original merge/curation code |
| `train_ablation.py` | fit the arm checkpoints (GPU/CPU lanes) |
| `link_ablation_models.py` | restore the links to shared checkpoints after extracting the archives |
| `run_eval_matrix.py` | paper protocol (component and per-branch ensemble) on original and repaired vectors |
| `constrained_attack.py` | constrained PGD (cPGD), constrained random search (cRS), constrained Square Attack (cSq) |
| `run_constrained_matrix.py` | all constrained/adaptive attacks over arms, seeds, constraint levels, radii |
| `stats_revision.py` | Friedman/Nemenyi and Holm-corrected Wilcoxon tests |
| `make_tables.py`, `generate_figures.py`, `summarize_revision.py` | tables, figures, and summary from the result CSVs |

## Installation

Python 3.10. The experiments used torch 1.12.1, xgboost 2.1.4, catboost 1.2.8, ART 1.20.1,
pytorch-tabnet 4.1.0, scipy, pandas, and matplotlib.

```bash
pip install -r requirements.txt
pip install -e .        # installs the package from src/
export PYTHONPATH=src   # the revision scripts import the package modules directly
```

Run every command from the repository root.

## Data and artifacts

| Artifact | Where | Needed for |
|---|---|---|
| Result CSVs of every table and figure | `results/revision/` (in the repository) | rebuilding tables, figures, statistics |
| Balanced training/test data | `balanced_data.7z.001` | every experiment |
| Clean and DTAT checkpoints, adversarial pools, evaluation vectors | `resources_NSLKDD.7z.*`, `resources_CIC-2018.7z.*` | rerunning evaluations |
| Ablation-arm checkpoints | release assets `ablation_checkpoints_<DS>.tar.zst.part-*` | evaluating the arms without retraining |

```bash
7z x balanced_data.7z.001          # -> data/
7z x resources_NSLKDD.7z.001       # -> resources/NSLKDD/
7z x resources_CIC-2018.7z.001     # -> resources/CIC-2018/

# ablation checkpoints (download the release assets first)
cat ablation_checkpoints_NSLKDD.tar.zst.part-*   | zstd -d | tar -x
cat ablation_checkpoints_CIC-2018.tar.zst.part-* | zstd -d | tar -x
python cli/hedge/9_revision/link_ablation_models.py
```

The curated curricula, the Π_Ω-repaired vectors, and D_bal′ are not shipped: they are regenerated
deterministically by the scripts below (the curriculum builder reproduces the original DTAT and
IAT curricula byte for byte).

## Rebuilding the paper's tables and figures

No experiment needs to be rerun; the scripts read `results/revision/`:

```bash
python cli/hedge/9_revision/stats_revision.py       # -> results/revision/stats.md
python cli/hedge/9_revision/make_tables.py          # -> results/revision/tables/*.tex
python cli/hedge/9_revision/generate_figures.py     # -> results/revision/figures/*.{pdf,png}
python cli/hedge/9_revision/summarize_revision.py   # -> results/revision/summary.md
```

Standard deviations are taken over curation seeds (and attack or generation seeds where they
apply); source cases and procedures are averaged within each run first.

## Reproducing the experiments

`{res}` is `nslkdd` or `cic2018`. Steps 1–3 are the original pipeline; steps 4–9 add the
revision experiments.

**1. Clean training and source-indexed pools**

```bash
python cli/hedge/5_training/model_training.py -m xgb -r cic2018 --params-dir params/training
python cli/hedge/7_generating_samples/generate_adv.py -r cic2018 --subset train -m dnn -a pgd
python cli/hedge/7_generating_samples/generate_adv.py -r cic2018 --subset test  -m dnn -a pgd
```

**2. DTAT curricula and fitting** (curation quota τ; 5000 for nslkdd)

```bash
python cli/hedge/7_generating_samples/merge_training_samples.py -r cic2018 -m xgb
python cli/hedge/7_generating_samples/postprocess_merge.py -r cic2018 -m xgb --sample-threshold 12000
./cli/hedge/5_training/train_robust_lanes.sh cic2018
```

**3. Paper protocol on the original vectors** (case D: `--adv-fallback-src dnn`, case T: `tabnet`)

```bash
python cli/hedge/9_revision/run_eval_matrix.py --subsets test --variants base iat dtat
```

**4. Constraint audit and repaired vectors**

```bash
python cli/hedge/9_revision/project_test_vectors.py -r cic2018                    # test_omega1/2
python cli/hedge/9_revision/project_test_vectors.py -r cic2018 --subset train     # train_omega1/2
python cli/hedge/9_revision/repair_dbal.py -r cic2018                             # D_bal'
python cli/hedge/9_revision/project_test_vectors.py -r cic2018 --subset train --strict \
  --clean-in data/cic2018_merged_train_raw_processed_omega1s.csv                   # train_omega1s/2s
```

**5. Ablation curricula and checkpoints**

```bash
python cli/hedge/9_revision/build_ablation_pools.py -r nslkdd cic2018 \
  -a iat iatvol dnnonly tabonly pooled reverse random dtat_o1 dtat_o2 dtat_o1s dtat_o2s
python cli/hedge/9_revision/train_ablation.py -r nslkdd cic2018 \
  -a iat iatvol dnnonly tabonly pooled reverse random dtat_o1 dtat_o2 dtat_o1s dtat_o2s
```

**6. Paper protocol on original and repaired vectors, all arms**

```bash
python cli/hedge/9_revision/run_eval_matrix.py --subsets test test_omega1 test_omega2 \
  --variants base iat iatvol dnnonly tabonly pooled reverse random dtat dtat_o1 dtat_o2 dtat_o1s dtat_o2s
```

**7. Constrained and defense-aware attacks** (three attack seeds at ε = 0.5σ)

```bash
python cli/hedge/9_revision/run_constrained_matrix.py --attack-seeds 0 1 2
# a single run, e.g. the Square Attack on the complete DTAT ensemble:
python cli/hedge/9_revision/constrained_attack.py -r cic2018 -e ens-sq --variant dtat --seed 1337 \
  --level omega2 --eps 0.5 --iters 1000
```

**8–9. Statistics, tables, figures:** see [Rebuilding the paper's tables and figures](#rebuilding-the-papers-tables-and-figures).

## Main results

Tree-mean ASR averaged over the four trees; case D uses DNN-generated and case T TabNet-generated
gradient vectors.

- **Transfer is asymmetric.** DNN-generated gradient vectors disrupt the clean trees by up to
  91.70%; tree-generated query vectors reach the networks by at most 4.05%.
- **Routing, not volume.** DTAT keeps transferred tree disruption at or below 3.48% (C&W on
  NSL-KDD: 8.22% and 7.97%), whereas self-training leaves up to 63.81%, and 61.98% at matched
  volume. A single designated source leaves 13.39–18.42% on CSE-CIC-IDS2018 for the other source;
  direction-blind pooling performs comparably to DTAT.
- **Constraints matter.** Nearly all unconstrained evaluation vectors, and 30.3–99.6% of the
  WGAN-synthesized training records per class, violate Ω. Under constrained PGD on
  CSE-CIC-IDS2018, DTAT-Ω lowers transferred tree disruption from 47.31% to 15.01%.
- **Adaptive attacks remain effective.** Against a defense-aware Square Attack on the complete
  ensemble by a forward-only attacker (Ω₂, CSE-CIC-IDS2018), constraint-consistent curricula
  lower disruption from 72.60% (DTAT) to 27.85–30.58%; no curriculum is robust in this setting.
- **Clean accuracy** changes by at most 0.11 points (95.68–95.84% and 99.26–99.35%).

**Scope.** Ω removes flow-meter inconsistencies but does not establish protocol validity or
preserved malicious functionality; no traffic was replayed.

## Citation

```bibtex
@article{dtat2026,
  title  = {Transfer-Aware and Constraint-Consistent Adversarial Curricula for Heterogeneous Intrusion Detection Ensembles},
  author = {Tuyen T. Nguyen and Minh Q. Tran and Uyen H. Tran and Hanh P. Du and Hai H. Le and Hoa N. Nguyen},
  note   = {Submitted to Applied Soft Computing},
  year   = {2026}
}
```

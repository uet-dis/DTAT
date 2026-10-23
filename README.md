# DTAT: Directional Transfer-Aware Adversarial Training for Robust Heterogeneous Intrusion Detection Ensembles

Code, processed datasets, adversarial pools, and trained checkpoints for the paper
"Directional Transfer-Aware Adversarial Training for Robust Heterogeneous Intrusion Detection Ensembles", which introduces **Directional Transfer-Aware Adversarial Training (DTAT)**. The repository and the installable package keep
the earlier project name **HEDGE** (`hedge`).

Architectural diversity in a heterogeneous IDS ensemble does not guarantee adversarial
independence: a perturbation generated against one component can also affect other learners
that share its features. DTAT builds a **target-specific adversarial curriculum** for each
component of an ensemble of deep-learning (DL) and tree-based learners. It specifies explicitly
which source–attack–target pathways supply each component's training set and how the examples
from each pathway are curated.

## Method overview

DTAT has four stages:

1. **Source-indexed generation.** Each clean-trained, compatible source generates one pool per
   procedure. Gradient-based procedures (FGSM, PGD, DeepFool, C&W, JSMA) require a
   differentiable source. Query-based procedures (ZOO, HSJA) apply to every component.
2. **Directional policy.** Each tree target receives its own query-based pools plus the
   gradient-based pools of its designated DL sources. Each DL target receives only its own
   pools. Tree-generated pools are not admitted to any DL curriculum.
3. **Per-procedure curation.** A fixed quota τ per procedure caps each pool's contribution and
   is divided equally among classes. For a tree target, the quota of each gradient-based
   procedure is split equally between the two DL sources.
4. **Learner-specific fitting.** Each component is refit on the balanced clean training set plus
   its curated curriculum. The ensemble combines the components by weighted voting with fixed
   weights.

Three design inputs are specified separately:

- **Source eligibility:** which sources supply each target.
- **Procedure admission:** which procedures are admitted.
- **Curation quota:** τ, the number of examples kept per procedure.

The evaluation records the source of each branch's input, which separates a component's
direct exposure from the unchanged transfer it receives from other components.

### Ensemble components

| Family | Models (code name) | Default voting weight |
|---|---|---|
| Deep learning | DNN (`dnn`), TabNet (`tabnet`) | 0.09, 0.10 |
| Tree-based | XGBoost (`xgb`), GBM (`gbm`), CatBoost (`catb`), Bagging (`bagging`) | 0.27, 0.18, 0.18, 0.18 |

The weights are self-normalizing and are defined in `cli/apelid/helpers/ensemble_helper.py`.

### Defense variants

| Variant | Description | Checkpoints | Evaluator flag |
|---|---|---|---|
| **Base** | Clean-trained ensemble | `{res}_{model}.{pkl,pth}` | _(none)_ |
| **IAT** | Isolated adversarial training: each tree is retrained only on its own ZOO + HSJA pools; DNN/TabNet use their DTAT checkpoints | `{res}_{model}_isolated_robust.pkl` | `--using-isolated-robust` |
| **DTAT** | Every component is retrained under the directional policy; four curricula per target, each built with a different curation seed | `{res}_{model}_robust_seed{1337..1340}.{pkl,pth}` | `--using-robust --model-seeds 1337 1338 1339 1340` |

## Repository layout

```
HELGE/
├── src/                       # installable package (top-level modules, no `hedge.` prefix)
│   ├── art_classifier/        # ART wrappers for xgb / catb / gbm / bagging / dnn / tabnet
│   ├── art_generator/         # FGSM, PGD, DeepFool, C&W, JSMA, ZOO, HSJA + eps scaling, repair, metrics
│   ├── resampling/            # WGAN minority augmentation, K-means majority compression
│   ├── training/              # learner training routines and classification metrics
│   ├── argparsers/, utils/
├── cli/apelid/                # pipeline entry points
│   ├── 1_preprocessing_*/     # cleaning, encoders, stratified 70:30 split
│   ├── 2_major/, 3_minor/     # K-means compression (majority) / WGAN augmentation (minority)
│   ├── 4_merge/               # balanced training set (14,000 records per class) and capped test set
│   ├── 5_training/            # clean and adversarial training, clean-accuracy evaluation
│   ├── 6_classifiers/         # classifier-level evaluation helpers
│   ├── 7_generating_samples/  # adversarial pool generation, merging, per-procedure curation
│   ├── 8_attack_evaluation/   # component-level and ensemble ASR evaluation
│   ├── configs/               # dataset paths, labels, attack parameters (ATTACK_PARAMETERS)
│   └── helpers/               # ensemble weights, model loading
├── params/training/           # learner hyperparameters (JSON)
├── notebooks/evaluate.ipynb   # all evaluation runs reported in the paper
├── data/, encoders/, resources/   # extracted from the archives below
└── reports/                   # metrics JSON and confusion matrices
```

## Installation

Requires Python 3.10. The paper's runs used torch 1.12.1, xgboost 2.1.4, catboost 1.2.8,
ART 1.20.1, and pytorch-tabnet 4.1.0.

```bash
pip install -r requirements.txt
pip install -e .        # installs the package from src/
```

Run every command from the repository root.

## Data and artifacts

```bash
7z x balanced_data.7z.001          # -> data/
7z x resources_NSLKDD.7z.001       # -> resources/NSLKDD/
7z x resources_CIC-2018.7z.001     # -> resources/CIC-2018/
```

```
data/
├── {nslkdd,cic2018}_merged_train_raw_processed.csv        # balanced training set (D_bal)
└── {nslkdd,cic2018}_test_random_sample_clean_merged.csv   # held-out test set
resources/{NSLKDD,CIC-2018}/
├── models/          # Base, IAT (_isolated_robust) and DTAT (_robust_seed{1337..1340}) checkpoints
└── adv_samples/
    ├── train/       # raw source-indexed pools: {res}_{source}_{attack}_adv.csv
    ├── isolated/    # IAT curricula of the trees
    ├── robust/      # DTAT curricula: *_robust_train_post.csv (seed 1337), seed_{1338,1339,1340}/
    └── test/        # evaluation inputs; seed_1..3/ hold re-generated ZOO/HSJA inputs
```

| Dataset | Inputs | Classes | Balanced train | Test |
|---|---|---|---|---|
| NSL-KDD | 41 | 5 | 70,000 | 17,336 |
| CSE-CIC-IDS2018 | 62 | 12 | 168,000 | 51,129 |

## Reproducing the pipeline

`{res}` is `nslkdd` or `cic2018`. The examples below use `cic2018`. Attack settings are read
from `ATTACK_PARAMETERS` in `cli/apelid/configs/{nslkdd,cic2018}.py`.

**1. Clean training**

```bash
python cli/apelid/5_training/model_training.py -m xgb -r cic2018 --params-dir params/training
python cli/apelid/5_training/tabnet_training.py -r cic2018 --params-dir params/training
python cli/apelid/5_training/model_evaluation.py -r cic2018 --mode model --models xgb catb gbm bagging dnn tabnet
```

**2. Source-indexed pools** (gradient-based procedures from `dnn`/`tabnet`; ZOO/HSJA from every model)

```bash
python cli/apelid/7_generating_samples/generate_adv.py -r cic2018 --subset train -m dnn -a pgd
python cli/apelid/7_generating_samples/generate_adv.py -r cic2018 --subset train -m xgb -a zoo
```

**3. Directional selection and curation** (DTAT curricula, curation seed 1337)

```bash
python cli/apelid/7_generating_samples/merge_training_samples.py -r cic2018 -m xgb
python cli/apelid/7_generating_samples/postprocess_merge.py -r cic2018 -m xgb --sample-threshold 12000   # τ; 5000 for nslkdd
```

**4. Additional curation seeds and DTAT fitting**

```bash
./cli/apelid/7_generating_samples/sample_robust_seeds.sh cic2018 xgb catb bagging gbm dnn tabnet   # seeds 1338..1340
./cli/apelid/5_training/train_robust_lanes.sh cic2018                                              # all models x seeds
./cli/apelid/5_training/retrain_robust.sh cic2018 xgb --seed 1338                                  # a single model/seed
```

**5. Evaluation inputs**

```bash
python cli/apelid/7_generating_samples/generate_adv.py -r cic2018 --subset test -m dnn -a fgsm
```

**6. Component-level exposure.** `--adv-fallback-src` selects the source case: `dnn` is case D
and `tabnet` is case T. In both cases the trees receive the chosen DL source's gradient-based
vectors.

```bash
python cli/apelid/8_attack_evaluation/classifier_evaluate_adv.py -r cic2018 \
  -m xgb catb gbm bagging dnn tabnet -a zoo hsja fgsm jsma pgd deepfool cw \
  --adv-fallback-src dnn --using-robust --model-seeds 1337 1338 1339 1340 \
  --seed-dirs resources/CIC-2018/adv_samples/test/seed_{1,2,3}
```

The base `test/` directory plus `seed_1..3` give the four ZOO/HSJA attack-data sources. Results
are reported as mean ± std over all applicable seed combinations. For Base, drop the robust
flags; for IAT, use `--using-isolated-robust` instead.

**7. Ensemble ASR.** Each branch receives the input generated for it. Rates are conditioned on
the clean correctness of the full ensemble.

```bash
python cli/apelid/8_attack_evaluation/classifier_parallel_ens_eval.py -r cic2018 \
  -a zoo hsja fgsm jsma pgd deepfool cw --adv-fallback-src dnn \
  --using-robust --model-seeds 1337 1338 1339 1340 \
  --seed-dirs resources/CIC-2018/adv_samples/test/seed_{1,2,3}
```

`notebooks/evaluate.ipynb` contains every run for both datasets, all three variants, and both
source cases.

## Main results

These are the tree-target ASRs averaged over the four trees. Case D uses DNN-generated vectors;
case T uses TabNet-generated vectors.

| | Base | IAT | DTAT |
|---|---|---|---|
| Transferred gradient-based vectors, CSE-CIC-IDS2018 (max over procedures) | 77.74% | 61.41% | ≤ 3.48% |
| Transferred gradient-based vectors, NSL-KDD, excluding C&W (max) | 81.95% | 57.83% | ≤ 2.64% |
| C&W on NSL-KDD (case D / case T) | 59.99% / 52.29% | 38.49% / 31.27% | 8.22% / 7.96% |

- **Transfer depends on attack, source, and target.** Exposure differs across trees for the same
  source, and the most exposed tree can change with the source.
- **Reverse direction.** Tree-generated ZOO/HSJA vectors reach at most 4.05% ASR on the
  networks, which is consistent with keeping tree pools out of DL curricula.
- **Ensemble.** Both adversarially trained variants reduce the maximum ensemble ASR from
  77.33% (CSE-CIC-IDS2018) and 74.91% (NSL-KDD) to below 4%. The IAT − DTAT contrast is small
  and uneven, ranging from −0.09 to 2.90 percentage points.
- **Clean accuracy** is essentially unchanged: 95.72–95.73% on CSE-CIC-IDS2018 and
  99.31–99.33% on NSL-KDD.

**Scope.** All perturbations are generated against clean-trained checkpoints, in feature space,
and without adaptation to the defended models. The results characterize the DTAT curriculum as
a whole. They do not isolate source assignment from procedure composition or augmentation
volume, and they do not establish adaptive robustness.

## Citation

```bibtex
@article{dtat2026,
  title  = {Directional Transfer-Aware Adversarial Training for Robust Heterogeneous Intrusion Detection Ensembles},
  author = {Tuyen T. Nguyen and Minh Q. Tran and Hanh P. Du and Hai H. Le and Hoa N. Nguyen},
  year   = {2026}
}
```

This work extends the APELID-ATH conference paper (VCRIS 2025).

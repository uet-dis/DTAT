from .resources import Resources


class NSLKDDResources(Resources):
    resources_name = "nslkdd"
    DATA_FOLDER = "resources/NSLKDD"

    KDD_TEXT_PATH = f"{DATA_FOLDER}/KDD+.txt"
    NSLKDD_ORIGINAL_CSV_PATH = f"{DATA_FOLDER}/nslkdd_original.csv"

    CLEAN_MERGED_DATA_FOLDER = f"{DATA_FOLDER}/clean_merged"
    ENCODED_DATA_FOLDER = f"{DATA_FOLDER}/encoded"
    RAW_PROCESSED_DATA_FOLDER = f"{DATA_FOLDER}/raw_processed"

    ENCODERS_FOLDER = "encoders/nslkdd"
    REPORT_FOLDER = "reports/nslkdd"

    MAJORITY_LABELS = ['Benign', 'DoS']
    MINORITY_LABELS = ['Probe', 'R2L', 'U2R']

    # Attack mode (paper: "Attack mode" row):
    #   fgsm/pgd/cw/zoo/hsja/deepfool = untargeted (y = ground truth for ascent / predictions for black-box)
    #   jsma = targeted, target = runner-up class excluding the true label (deterministic)
    #   targeted variants: fgsm/pgd/cw via --targeted (target chosen automatically, never the true label)
    #   zoo/hsja/deepfool: NOT targeted-capable (do not receive y / ignore y)
    ATTACK_PARAMETERS = {
        'zoo': {
            'confidence': 0.5,
            'learning_rate': 0.05,
            'max_iter': 50,
            'binary_search_steps': 5,
            'initial_const': 1e-3,
            'abort_early': True,
            'use_resize': False,
            'use_importance': False,
            'nb_parallel': 16,
            'batch_size': 1,
            'variable_h': 0.005,
            'verbose': True
        },
        'hsja': {
            "batch_size": 64,
            "norm": 2,
            "max_iter": 10,
            "max_eval": 600,
            "init_eval": 20,
            "init_size": 20,
            "init_radius": 2.0,
            "verbose": True
        },
        'jsma': {
            # theta is in InputNorm sigma units (attack runs in z space, one
            # sigma per step); gamma stays the declared low budget.
            "theta": 1.0,
            "gamma": 0.1,  # low-budget by design: <= ~4 modified features (declare in threat model)
            "batch_size": 64,
            "verbose": True,
        },
        'fgsm': {
            "eps": 0.1,
            "batch_size": 64,
            # eps_step removed: dead parameter (ART FastGradientMethod only uses
            # it with minimal=True; see fast_gradient.py:262-266)
            "targeted": False,
            "verbose": True,
        },
        'pgd': {
            "eps": 0.2,
            "eps_step": 0.02,  # traverses the eps=0.2 ball in 10 steps
            "batch_size": 64,
            "targeted": False,  # untargeted evasion (was True — bug: pushed samples toward the true label)
            "max_iter": 100,  # was 200 — 10x over-provisioned (no early stop in ART PGD)
            "verbose": True
        },
        'deepfool': {
            "max_iter": 100,
            "batch_size": 64,
            "nb_grads": 5,  # = nb_classes(NSL) -> full gradients (ART auto-disables the subset path)
            "epsilon": 0.02,  # was 1e-6 — paper-value overshoot; 1e-6 lands exactly ON the boundary (fragile)
            "verbose": True,
        },
        'cw': {
            "confidence": 0.1,  # was 0.0 — positive margin makes evasions robust to small logit drift
            "learning_rate": 0.01,
            "binary_search_steps": 9,  # was 3 — c-search only reached ~0.04 with initial_const=0.01
            "max_iter": 10,  # was 3 — only 9 gradient steps total, below the ART default of 10
            "batch_size": 64,
            "verbose": True,
            # 0.01 is the ART default, tuned for raw/unnormalized inputs; in InputNorm
            # z-space it leaves the classification term negligible and C&W under-converges.
            # NSL probe (2000 samples, full-set box): ASR saturates by c~1.0
            # (0.5->37.8%, 1.0->40.0%, 2.0->40.0%) while norms keep drifting up
            # (z-L2 median 1.18/1.47/1.58 sigma), so 1.0 is the minimal value that
            # reaches the plateau. CIC keeps 2.0 (not yet saturated there).
            "initial_const": 1.0,
            "max_halving": 5,
            "max_doubling": 5
        },
    }

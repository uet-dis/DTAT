from .resources import Resources


class CIC2018Resources(Resources):
    resources_name = "cic2018"
    ORIGINAL_DATA_FOLDER = "resources/CIC-2018/original"
    DATA_FOLDER = "resources/CIC-2018/"
    REPORT_FOLDER = "reports/cic2018"
    ENCODERS_FOLDER = "encoders/cic2018"

    ENCODED_DATA_FOLDER = f"{DATA_FOLDER}/encoded"
    RAW_PROCESSED_DATA_FOLDER = f"{DATA_FOLDER}/raw_processed"
    CLEAN_MERGED_DATA_FOLDER = f"{DATA_FOLDER}/clean_merged"

    EMBEDDINGS_FOLDER = f"{DATA_FOLDER}/embeddings"
    PCA_CACHE_FOLDER = f"{EMBEDDINGS_FOLDER}/pca_cache"

    LABEL_COLUMN = "Label"

    MAJORITY_LABELS = [
        'Benign', 'DDoS attacks-LOIC-HTTP', 'DDOS attack-HOIC', 'DoS attacks-Hulk', 'Bot', 
        'Infilteration', 'SSH-Bruteforce', 'DoS attacks-GoldenEye'
    ]

    MINORITY_LABELS = [
        'DoS attacks-Slowloris',
        # 'DDOS attack-LOIC-UDP',
        'Brute Force -Web',
        'Brute Force -XSS',
        'SQL Injection',
        # 'DoS attacks-SlowHTTPTest',
        # 'FTP-BruteForce'
    ]

    ACCEPT_RATE_MAP = {
        'DoS attacks-Slowloris': 0.30,
        'Brute Force -Web': 0.35,
        'Brute Force -XSS': 0.50,
        'SQL Injection': 0.3,
    }
    # Attack mode (paper: "Attack mode" row):
    #   fgsm/pgd/cw/zoo/hsja/deepfool = untargeted (y = ground truth for ascent / predictions for black-box)
    #   jsma = targeted, target = runner-up class excluding the true label (deterministic)
    #   targeted variants: fgsm/pgd/cw via --targeted (target chosen automatically, never the true label)
    #   zoo/hsja/deepfool: NOT targeted-capable (do not receive y / ignore y)
    #
    # Option B (eps scaling): raw feature scales on CIC-2018 span ~1e8 (IAT/duration),
    # so a fixed raw eps is a no-op for most features. fgsm/pgd budgets are instead
    # defined per feature as k * sigma_i (InputNorm sigma of the trained DNN,
    # resolved in generate_adv.py via art_generator/eps_scaling.py):
    #   eps_i = k_eps * sigma_i, eps_step_i = k_step * sigma_i.
    # The budget in InputNorm space is exactly k_eps standard deviations per feature.
    ATTACK_PARAMETERS = {
        'zoo': {
            'confidence': 0.5,  # was 10.0 — unintentionally aggressive asymmetry vs NSL (0.5)
            'learning_rate': 0.04,
            'max_iter': 40,
            'binary_search_steps': 3,
            'initial_const': 1e-3,
            'abort_early': True,
            'use_resize': False,
            'use_importance': False,
            'nb_parallel': 16,
            'batch_size': 1,
            'variable_h': 0.002,
            'verbose': True
        },
        'hsja': {
            "batch_size": 64,
            "norm": 2,
            "max_iter": 10,  # was 5 — sync query budget with NSL (~600 queries/sample vs ~150)
            "max_eval": 600,  # was 300
            "init_eval": 20,  # was 10
            "init_size": 20,
            "init_radius": 2.0,
            "verbose": True
        },
        'jsma': {
            # theta is in InputNorm sigma units (attack runs in z space, one
            # sigma per step — the raw-scale comment below is obsolete).
            "theta": 1.0,
            "gamma": 0.1,  # low-budget by design: <= ~4 modified features (declare in threat model)
            "batch_size": 64,
            "verbose": True,
        },
        'fgsm': {
            "eps_per_std": 0.1,  # budget = 0.1 sigma per continuous feature (was raw eps=0.1 — a no-op on CIC)
            "eps_step_per_std": 0.1,  # ignored by FGSM (minimal=False) but required as an array for ART type validation
            "batch_size": 64,
            "targeted": False,
            "verbose": True,
        },
        'pgd': {
            "eps_per_std": 0.2,  # budget = 0.2 sigma per continuous feature (was raw eps=0.2 — a no-op on CIC)
            "eps_step_per_std": 0.02,  # traverses the ball in 10 steps
            "batch_size": 64,
            "targeted": False,  # untargeted evasion (was True — bug: pushed samples toward the true label)
            "max_iter": 100,  # was 200 — 10x over-provisioned (no early stop in ART PGD)
            "verbose": True
        },
        'deepfool': {
            "max_iter": 100,
            "batch_size": 64,
            "nb_grads": 10,  # was 5 — CIC has 12 classes; 5 missed boundaries to classes outside the top-5
            "epsilon": 0.02,  # was 1e-6 — paper-value overshoot; 1e-6 lands exactly ON the boundary (fragile)
        },
        'cw': {
            "confidence": 0.1,  # was 0.0 — positive margin makes evasions robust to small logit drift
            "learning_rate": 0.01,
            "binary_search_steps": 9,  # was 3 — c-search only reached ~0.04 with initial_const=0.01
            "max_iter": 10,  # was 3 — only 9 gradient steps total, below the ART default of 10
            "batch_size": 64,
            "verbose": True,
            # 0.01 is the ART default, tuned for raw/unnormalized inputs. In InputNorm
            # z-space the data term is O(0.1-1), so c=0.01 leaves the classification
            # term negligible and C&W under-converges (full-set pool ASR 0.34% vs
            # 29.7% at c=2.0 on a 1000-sample probe; p90 z-L2 1.9 sigma, still minimal).
            "initial_const": 2.0,
            "max_halving": 5,
            "max_doubling": 5
        },
    }

    

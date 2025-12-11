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
    ATTACK_PARAMETERS = {
        'zoo': {
            'confidence': 10.0,
            'targeted': False,
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
            "targeted": False,
            "norm": 2,
            "max_iter": 5,
            "max_eval": 300,
            "init_eval": 10,
            "init_size": 20,
            "verbose": True
        },
        'jsma': {
            "theta": 0.02,
            "gamma": 0.1,
            "batch_size": 64,
            "verbose": True,
        },
        'fgsm': {
            "eps": 0.1,
            "batch_size": 64,
            "eps_step": 0.01,
            "targeted": False,
        },
        'pgd': {
            "eps": 0.2,
            "eps_step": 0.01,
            "batch_size": 64,
            "targeted": True,
            "max_iter": 200,
            "verbose": True
        },
        'deepfool': {
            "max_iter": 100,
            "batch_size": 64,
            "nb_grads": 5,
            "epsilon": 1e-6
        },
        'cw': {
            "confidence": 0.0,
            "learning_rate": 0.01,
            "binary_search_steps": 3,
            "max_iter": 3,
            "batch_size": 64,
            "verbose": False,
            "initial_const": 0.01,
            "max_halving": 5,
            "max_doubling": 5
        },
    }

    
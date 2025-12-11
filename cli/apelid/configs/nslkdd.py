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

    ATTACK_PARAMETERS = {
        'zoo': {
            'confidence': 0.5,
            'targeted': False,
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
            "targeted": False,
            "norm": 2,
            "max_iter": 10,
            "max_eval": 600,
            "init_eval": 20,
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
            "verbose": True,
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
            "epsilon": 1e-6,
            "verbose": True,
        },
        'cw': {
            "confidence": 0.0,
            "learning_rate": 0.01,
            "binary_search_steps": 3,
            "max_iter": 3,
            "batch_size": 64,
            "verbose": True,
            "initial_const": 0.01,
            "max_halving": 5,
            "max_doubling": 5
        },
    }
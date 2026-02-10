from art.attacks.evasion import ZooAttack
from art_generator.attack_generator import AttackGenerator
import numpy as np
import pandas as pd
from utils.logging import get_logger
logger = get_logger(__name__)

class ZooAttackGenerator(AttackGenerator):
    def __init__(self, classifier, generator_params: dict = None):
        super().__init__(classifier, generator_params)
        attack_params = {
            "confidence": 0.0,
            "targeted": False,
            "learning_rate": 1e-1,
            "max_iter": 100,
            "binary_search_steps": 3,
            "initial_const": 1e-3,
            "abort_early": True,
            "use_resize": False,
            "use_importance": False,
            "nb_parallel": 10,
            "batch_size": 1,
            "variable_h": 0.02,
            "verbose": True
        }
        attack_params = self.update_generator_params(attack_params)
        logger.debug(f"[+] ZooAttack parameters: {attack_params}")

        self.attack = ZooAttack(classifier = self.classifier,
            **attack_params
        )

    def generate(self, x: np.ndarray, y: np.ndarray, input_metadata: dict, mutate_indices: list[int] = [], **kwargs):
        # Generate adversarial features
        logger.info(f"[+] Generating adversarial samples with ZooAttack, mutate_indices: {mutate_indices}")
        x_adv = self.attack.generate(x)
        if mutate_indices:
            x_adv[:, mutate_indices] = x[:, mutate_indices]
        # y is mandatory: return DataFrame combining X_adv and y
        y_arr = y.reshape(-1) if y.ndim > 1 else y
        feature_names = input_metadata['feature_names']
        df = pd.DataFrame(x_adv, columns=feature_names)
        df[input_metadata['label_column']] = y_arr
        return df
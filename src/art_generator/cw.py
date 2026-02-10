from art.attacks.evasion import CarliniL2Method
from art_generator.attack_generator import AttackGenerator
import numpy as np
import pandas as pd
from utils.logging import get_logger
logger = get_logger(__name__)

class CWAttackGenerator(AttackGenerator):
    def __init__(self, classifier, generator_params: dict = None):
        super().__init__(classifier, generator_params)
        attack_params = {
            "confidence": 0.0,
            "learning_rate": 0.01,
            "binary_search_steps": 3,
            "max_iter": 3,
            "batch_size": 64,
            "verbose": False,
            "initial_const": 0.01,
            "max_halving": 5,
            "max_doubling": 5
        }
        attack_params = self.update_generator_params(attack_params)

        self.attack = CarliniL2Method(classifier = self.classifier,
            **attack_params
        )

    def generate(self, x: np.ndarray, y: np.ndarray, input_metadata: dict, mutate_indices: list[int] = [], **kwargs):
        # Generate adversarial features
        x_adv = self.attack.generate(x=x, y=y)
        if mutate_indices:
            x_adv[:, mutate_indices] = x[:, mutate_indices]
        # y is mandatory: return DataFrame combining X_adv and y
        y_arr = y.reshape(-1) if y.ndim > 1 else y
        feature_names = input_metadata['feature_names']
        df = pd.DataFrame(x_adv, columns=feature_names)
        df[input_metadata['label_column']] = y_arr
        return df
from art.attacks.evasion import DeepFool
from art_generator.attack_generator import AttackGenerator
import numpy as np
import pandas as pd
from utils.logging import get_logger
logger = get_logger(__name__)

class DeepFoolAttackGenerator(AttackGenerator):
    def __init__(self, classifier, generator_params: dict = None):
        super().__init__(classifier, generator_params)
        attack_params = {
            "max_iter": 100,
            "batch_size": 64,
            "nb_grads": 5,
            "epsilon": 1e-6
        }

        attack_params = self.update_generator_params(attack_params)
        self.attack = DeepFool(classifier = self.classifier,
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
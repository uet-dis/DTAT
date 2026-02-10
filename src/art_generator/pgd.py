from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art_generator.attack_generator import AttackGenerator
import numpy as np
import pandas as pd
from utils.logging import get_logger
logger = get_logger(__name__)

class PGDAttackGenerator(AttackGenerator):
    def __init__(self, classifier, generator_params: dict = None):
        super().__init__(classifier, generator_params)
        attack_params = {
            "eps": 0.2,
            "eps_step": 0.01,
            "batch_size": 64,
            "targeted": True,
            "max_iter": 200,
            "verbose": True
        }
        attack_params = self.update_generator_params(attack_params)

        self.attack = ProjectedGradientDescentPyTorch(estimator = self.classifier,
            **attack_params
        )

    def generate(self, x: np.ndarray, y: np.ndarray, input_metadata: dict, mutate_indices: list[int] = [], **kwargs):
        # Generate adversarial features
        mask = np.ones(x.shape)
        mask[:, mutate_indices] = 0
        logger.info(f"[+] Generating adversarial samples with PGD, mutate_indices: {mutate_indices}")
        x_adv = self.attack.generate(x=x, y=y, mask=mask)
        # y is mandatory: return DataFrame combining X_adv and y
        y_arr = y.reshape(-1) if y.ndim > 1 else y
        feature_names = input_metadata['feature_names']
        df = pd.DataFrame(x_adv, columns=feature_names)
        df[input_metadata['label_column']] = y_arr
        return df
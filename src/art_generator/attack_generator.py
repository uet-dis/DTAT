from abc import ABC, abstractmethod
import numpy as np
from utils.logging import get_logger
logger = get_logger(__name__)


class AttackGenerator(ABC):

    def __init__(self, classifier, generator_params: dict = None):
        self.attack = None
        self.generator_params = generator_params
        self.classifier = classifier

    @abstractmethod
    def generate(self, x: np.ndarray, y: np.ndarray, input_metadata: dict, mutate_indices: list[int] = [], **kwargs):
        """Generate adversarial samples with required labels y.

        Implementations should return a pandas DataFrame combining adversarial
        features and the original labels, or an equivalent labeled structure.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def update_generator_params(self, attack_params: dict):
        if self.generator_params is None:
            self.generator_params = attack_params.copy()
            return attack_params
        return_params = attack_params.copy()
        for key in return_params.keys():
            if key in self.generator_params:
                return_params[key] = self.generator_params[key]
        return return_params
        
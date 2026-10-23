from abc import ABC, abstractmethod
import numpy as np
from art_generator.targets import masked_second_most_likely
from utils.logging import get_logger
logger = get_logger(__name__)


class AttackGenerator(ABC):

    def __init__(self, classifier, generator_params: dict = None):
        self.attack = None
        self.generator_params = generator_params
        self.classifier = classifier

    def resolve_targets(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Labels to hand to the underlying ART attack.

        Untargeted attacks interpret y as the reference labels for gradient
        ascent, so the ground truth is correct there. But if the ART attack is
        configured targeted (`attack.targeted`), y is interpreted as ATTACK
        TARGETS — passing ground truth would optimize FOR the true label (the
        original PGD targeted-to-true-label bug). In that case compute
        deterministic masked second-most-likely targets instead. Never return
        ground-truth y to a targeted attack.
        """
        if getattr(self.attack, "targeted", False):
            logger.info("[+] Targeted mode: targets = masked second-most-likely class (deterministic)")
            return masked_second_most_likely(x, self.classifier, y)
        return y

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
        
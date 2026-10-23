from art.attacks.evasion import CarliniL2Method
from art_generator.attack_generator import AttackGenerator
from art_generator.inputnorm_adapter import InputNormStats, data_zspace_clip, build_zspace_torch_estimator
import numpy as np
import pandas as pd
from utils.logging import get_logger
logger = get_logger(__name__)

class CWAttackGenerator(AttackGenerator):
    """C&W L2 in InputNorm coordinates (scale-coherent minimal perturbation).

    ART CarliniL2Method minimizes ||delta||_2^2 + c * f(x + delta) in the
    estimator's input space. On raw tabular features the raw-L2 distance term
    is meaningless with heterogeneous feature scales: steps that are cheap in
    raw L2 move low-sigma features by tens of standard deviations (v1 pool:
    10.8% destroyed rows, L2 max 1.2e4 sigma), and the raw loss landscape is
    so anisotropic that the optimizer stalls (v1 pool: 71% unchanged rows).

    Fix: the attack estimator wraps the SAME torch model (shared weights) in
    InputNorm coordinates via inputnorm_adapter — the L2 distance term and
    the optimization steps act on sigma units. Before each generate call the
    estimator gets a data-derived PER-FEATURE z clip box (data_zspace_clip):
    lower edge = per-feature data min (raw >= 0 — valid domain), upper edge =
    data max (pad=0: no value outside the observed data range). Without a box the unbounded z optimizer exploits
    raw-negative values (invalid, unrealizable records); with the RAW clip
    box the tanh remap precision error is ~1e4 sigma. The data box keeps
    remap error ~1e-5 sigma and the domain valid.

    The generator converts raw -> z before the attack and z_adv -> raw after;
    frozen cat/binary positions are restored bit-exactly from the originals.
    """

    def __init__(self, classifier, generator_params: dict = None):
        super().__init__(classifier, generator_params)
        stats = InputNormStats.from_estimator(self.classifier)
        if stats is None or not stats.cont_idx:
            raise RuntimeError(
                "C&W requires the tabular DNN with an input_norm module "
                "(DNN-only attack, InputNorm coordinates)."
            )
        self.stats = stats
        attack_params = {
            "confidence": 0.0,
            "learning_rate": 0.01,
            "binary_search_steps": 3,
            "max_iter": 3,
            "batch_size": 64,
            "verbose": False,
            "initial_const": 0.01,
            "max_halving": 5,
            "max_doubling": 5,
            "targeted": False,  # flip to True in ATTACK_PARAMETERS for targeted mode (targets auto-resolved to masked second-most-likely)
        }
        attack_params = self.update_generator_params(attack_params)

        self.attack_estimator = build_zspace_torch_estimator(self.classifier, stats)
        self.attack = CarliniL2Method(classifier=self.attack_estimator,
            **attack_params
        )

    def generate(self, x: np.ndarray, y: np.ndarray, input_metadata: dict, mutate_indices: list[int] = [], **kwargs):
        # Generate adversarial features (ground truth y as correct labels for
        # the untargeted objective; masked second-most-likely targets if the
        # attack is configured targeted). NOTE: C&W is natively targeted-capable
        # (Carlini's original objective); never hand ground-truth y to a
        # targeted configuration — resolve_targets() guards against that.
        x_in = self.stats.to_z(x)
        raw_clip = getattr(self.classifier, "clip_values", None)
        raw_clip_max = raw_clip[1] if raw_clip is not None else None
        lo, hi = data_zspace_clip(self.stats, x_in, raw_clip_max=raw_clip_max)
        # ART CW reads estimator.clip_values once per generate (carlini.py:244-247);
        # PyTorchClassifier stores it as the private _clip_values attribute.
        self.attack_estimator._clip_values = (lo.astype(np.float32), hi.astype(np.float32))
        x_adv_z = self.attack.generate(x=x_in, y=self.resolve_targets(x, y))
        # tanh-remap roundoff can land a hair below the box lower edge
        # (float32); clip back into the box so raw stays >= data min >= 0.
        x_adv_z = np.clip(x_adv_z, lo.astype(np.float32), hi.astype(np.float32))
        x_adv = self.stats.from_z(x_adv_z)
        # from_z (z->raw in float32) can still land a hair below the raw data
        # minimum (float roundoff at the box edge); clamp to the per-feature
        # RAW data minimum computed directly on x (roundoff-free), so the
        # pool is strictly inside the valid domain.
        x_adv = np.maximum(x_adv, x.min(axis=0))
        if mutate_indices:
            x_adv[:, mutate_indices] = x[:, mutate_indices]
        # y is mandatory: return DataFrame combining X_adv and y
        y_arr = y.reshape(-1) if y.ndim > 1 else y
        feature_names = input_metadata['feature_names']
        df = pd.DataFrame(x_adv, columns=feature_names)
        df[input_metadata['label_column']] = y_arr
        return df

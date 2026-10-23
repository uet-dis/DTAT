from art.attacks.evasion import ZooAttack
from art_generator.attack_generator import AttackGenerator
from art_generator.inputnorm_adapter import (
    build_zspace_estimator_for,
    data_zspace_clip,
    set_attackgen_predict_nthread,
    set_zspace_clip_box,
)
import numpy as np
import pandas as pd
from utils.logging import get_logger
logger = get_logger(__name__)


class _DataBoxZOO(ZooAttack):
    """ART ZooAttack 1.20.1 with a per-feature data-derived clip box.

    Why a copy: ART ZOO touches the clip box in exactly two places
    (zoo.py:241-243, :564-566). The first (elementwise ``np.clip(x_adv, ...)``
    after each iteration) broadcasts correctly with per-feature arrays and is
    inherited verbatim. The second, the ADAM noise projection in
    ``_optimizer_adam_coordinate`` (zoo.py:564-566), selects flat coordinates
    of the (batch, features)-shaped noise — ``np.clip(noise[index], clip_min,
    clip_max)`` assumes a SCALAR box and fails to broadcast against a
    per-feature array. This subclass maps each flat index to its feature
    (row-major: ``feat = idx % n_features``) and clips the coordinate by its
    own box edge. With ``batch_size=1`` (our config) the mapping is the
    identity, so semantics are identical to ART's scalar behavior; the
    mapping keeps it correct for any batch size.

    Semantics note: the noise projection (:566) is a loose bound on the
    displacement variable; the binding domain constraint is the elementwise
    x_adv clip after each iteration (:241-243), also inherited verbatim.
    """

    def _optimizer_adam_coordinate(
        self,
        losses: np.ndarray,
        index: np.ndarray,
        mean: np.ndarray,
        var: np.ndarray,
        current_noise: np.ndarray,
        learning_rate: float,
        adam_epochs: np.ndarray,
        proj: bool,
    ) -> np.ndarray:
        """
        Implementation of the ADAM optimizer for coordinate descent
        (verbatim from ART 1.20.1 zoo.py:525-567, with the clip line
        extended to per-feature boxes).

        :param losses: Overall loss.
        :param index: Indices of the coordinates to update.
        :param mean: The mean of the gradient (first moment).
        :param var: The uncentered variance of the gradient (second moment).
        :param current_noise: Current noise.
        :param learning_rate: Learning rate for Adam optimizer.
        :param adam_epochs: Epochs to run the Adam optimizer.
        :param proj: Whether to project the noise to the L_p ball.
        :return: Updated noise for coordinate descent.
        """
        beta1, beta2 = 0.9, 0.999

        # Estimate grads from loss variation (constant `h` from the paper is fixed to .0001)
        grads = np.array([(losses[i] - losses[i + 1]) / (2 * self.variable_h) for i in range(0, len(losses), 2)])

        # ADAM update
        mean[index] = beta1 * mean[index] + (1 - beta1) * grads
        var[index] = beta2 * var[index] + (1 - beta2) * grads**2

        corr = (np.sqrt(1 - np.power(beta2, adam_epochs[index]))) / (1 - np.power(beta1, adam_epochs[index]))
        orig_shape = current_noise.shape
        current_noise = current_noise.reshape(-1)
        current_noise[index] -= learning_rate * corr * mean[index] / (np.sqrt(var[index]) + 1e-8)
        adam_epochs[index] += 1

        if proj and hasattr(self.estimator, "clip_values") and self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
            # flat index of a (batch, n_features) array -> per-coordinate
            # feature id (row-major); per-feature box support
            n_feat = int(np.prod(self.estimator.input_shape))
            feat = np.asarray(index) % n_feat
            current_noise[index] = np.clip(current_noise[index], clip_min[feat], clip_max[feat])

        return current_noise.reshape(orig_shape)

class ZooAttackGenerator(AttackGenerator):
    """ZOO in InputNorm coordinates (scale-coherent minimal perturbation).

    ART ZOO minimizes the L2 (raw-space) norm of the perturbation plus the
    confidence loss, using finite-difference coordinate descent that only
    ever calls estimator.predict. On raw tabular features a raw-L2-cheap step
    can move a low-sigma feature by tens of standard deviations, and ART
    clips the result to the raw clip box (zoo.py:240-243) — the same
    geometry mismatch that destroyed the DeepFool pool.

    Fix: the attack estimator is the SAME model wrapped by the z-space
    adapter from inputnorm_adapter (continuous columns z -> raw before the
    real predict; for the DNN a shared-weight torch module, for GBT models a
    black-box remap subclass). A per-feature valid-domain box derived from
    the data itself (data_zspace_clip; lower edge = per-feature data min ->
    raw >= 0) is attached to the estimator before each generate call — ART
    ZOO only ever clips elementwise with it, so no step is distorted. The
    declared confidence/max_iter parameters are unchanged; only the
    coordinate system changes.

    ``inputnorm_stats`` (from the DNN checkpoint) is required when the
    attacked model is not the DNN itself, because the canonical sigma units
    live there.
    """

    def __init__(self, classifier, generator_params: dict = None, inputnorm_stats=None):
        super().__init__(classifier, generator_params)
        set_attackgen_predict_nthread(self.classifier)
        attack_estimator, self.stats = build_zspace_estimator_for(classifier, inputnorm_stats)
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

        self.attack = _DataBoxZOO(classifier = attack_estimator,
            **attack_params
        )

    def generate(self, x: np.ndarray, y: np.ndarray, input_metadata: dict, mutate_indices: list[int] = [],
                 zspace_clip=None, raw_floor=None, **kwargs):
        # Attack in InputNorm coordinates, convert back to raw units.
        # Frozen positions are restored from the raw originals (bit-exact).
        x_in = self.stats.to_z(x)
        # Per-feature valid-domain box derived from the data itself (lower
        # edge = per-feature data min -> raw >= 0; see inputnorm_adapter).
        # ``zspace_clip`` (sharded generation): a box precomputed on the FULL
        # input, overriding the per-batch derivation — the declared domain
        # box must be identical across every shard of the pool.
        raw_clip = getattr(self.classifier, "clip_values", None)
        raw_clip_max = raw_clip[1] if raw_clip is not None else None
        if zspace_clip is None:
            zspace_clip = data_zspace_clip(self.stats, x_in, raw_clip_max=raw_clip_max)
        set_zspace_clip_box(self.attack.estimator, zspace_clip)
        x_adv_z = self.attack.generate(x_in)
        x_adv = self.stats.from_z(x_adv_z)
        # from_z (z->raw in float32) can land a hair below the raw data
        # minimum (float roundoff at the box edge); clamp to the per-feature
        # RAW data minimum computed directly on x (roundoff-free).
        # ``raw_floor`` (sharded generation): the full-set per-feature minimum,
        # so the clamp is consistent across shards (a shard min can exceed the
        # full-set min and would leave roundoff dips below the declared floor).
        floor = raw_floor if raw_floor is not None else x.min(axis=0)
        x_adv = np.maximum(x_adv, floor)
        if mutate_indices:
            x_adv[:, mutate_indices] = x[:, mutate_indices]
        # y is mandatory: return DataFrame combining X_adv and y
        y_arr = y.reshape(-1) if y.ndim > 1 else y
        feature_names = input_metadata['feature_names']
        df = pd.DataFrame(x_adv, columns=feature_names)
        df[input_metadata['label_column']] = y_arr
        return df

from art.attacks.evasion import DeepFool
from art.config import ART_NUMPY_DTYPE
from art.utils import is_probability
from tqdm.auto import trange
from art_generator.attack_generator import AttackGenerator
from art_generator.inputnorm_adapter import (
    InputNormStats,
    build_zspace_torch_estimator,
    data_zspace_clip,
)
import numpy as np
import pandas as pd
from utils.logging import get_logger
logger = get_logger(__name__)


class _DataBoxDeepFool(DeepFool):
    """ART DeepFool 1.20.1 with a per-feature data-derived clip box.

    Why a copy: passing ``clip_values`` to the estimator would make ART scale
    every step by (clip_max - clip_min) (deepfool.py:177) — for tabular boxes
    that inflates steps by the whole feature range and destroys samples; and
    an UNBOUNDED z space (clip_values=None) lets the minimal step park
    perturbations at raw < 0 on count-like features that are never negative
    in the source data (measured on the NSL pool: 100% of rows used negative
    values; clamping back reverts over half the evasions — the evasion was
    partly bought in an unrealizable domain).

    This subclass keeps the validated UNSCALED step (deepfool.py:182) and
    clips the batch into ``self._data_box`` after each step and after the
    overshoot, so the trajectory never leaves the valid domain. Everything
    else is verbatim from ART 1.20.1 generate() (lines 101-214).
    """

    _data_box = None  # (clip_min_z, clip_max_z), full-width per-feature arrays

    def generate(self, x: np.ndarray, y: np.ndarray = None, **kwargs):
        clip_min, clip_max = self._data_box
        x_adv = x.astype(ART_NUMPY_DTYPE)
        preds = self.estimator.predict(x, batch_size=self.batch_size)

        if self.estimator.nb_classes == 2 and preds.shape[1] == 1:
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        if is_probability(preds[0]):
            logger.warning(
                "It seems that the attacked model is predicting probabilities. DeepFool expects logits as model output "
                "to achieve its full attack strength."
            )

        # Determine the class labels for which to compute the gradients
        use_grads_subset = self.nb_grads < self.estimator.nb_classes
        if use_grads_subset:
            grad_labels = np.argsort(-preds, axis=1)[:, : self.nb_grads]
            labels_set = np.unique(grad_labels)
        else:
            labels_set = np.arange(self.estimator.nb_classes)
        sorter = np.arange(len(labels_set))

        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Compute perturbation with implicit batching
        for batch_id in trange(
            int(np.ceil(x_adv.shape[0] / float(self.batch_size))), desc="DeepFool", disable=not self.verbose
        ):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_adv[batch_index_1:batch_index_2].copy()

            # Get predictions and gradients for batch
            f_batch = preds[batch_index_1:batch_index_2]
            fk_hat = np.argmax(f_batch, axis=1)
            if use_grads_subset:
                # Compute gradients only for top predicted classes
                grd = np.array([self.estimator.class_gradient(batch, label=int(label_i)) for label_i in labels_set])
                grd = np.squeeze(np.swapaxes(grd, 0, 2), axis=0)
            else:
                # Compute gradients for all classes
                grd = self.estimator.class_gradient(batch)

            # Get current predictions
            active_indices = np.arange(len(batch))
            current_step = 0
            while active_indices.size > 0 and current_step < self.max_iter:
                # Compute difference in predictions and gradients only for selected top predictions
                labels_indices = sorter[np.searchsorted(labels_set, fk_hat, sorter=sorter)]
                grad_diff = grd - grd[np.arange(len(grd)), labels_indices][:, None]
                f_diff = f_batch[:, labels_set] - f_batch[np.arange(len(f_batch)), labels_indices][:, None]

                # Choose coordinate and compute perturbation
                norm = np.linalg.norm(grad_diff.reshape(len(grad_diff), len(labels_set), -1), axis=2) + tol
                value = np.abs(f_diff) / norm
                value[np.arange(len(value)), labels_indices] = np.inf
                l_var = np.argmin(value, axis=1)
                absolute1 = abs(f_diff[np.arange(len(f_diff)), l_var])
                draddiff = grad_diff[np.arange(len(grad_diff)), l_var].reshape(len(grad_diff), -1)
                pow1 = (
                    pow(
                        np.linalg.norm(draddiff, axis=1),
                        2,
                    )
                    + tol
                )
                r_var = absolute1 / pow1
                r_var = r_var.reshape((-1,) + (1,) * (len(x.shape) - 1))
                r_var = r_var * grad_diff[np.arange(len(grad_diff)), l_var]

                # Add perturbation (UNSCALED, as with clip_values=None) and
                # clip into the valid-domain box
                batch[active_indices] = np.clip(
                    batch[active_indices] + r_var[active_indices], clip_min, clip_max
                )

                # Recompute prediction for new x
                f_batch = self.estimator.predict(batch)
                fk_i_hat = np.argmax(f_batch, axis=1)

                # Recompute gradients for new x
                if use_grads_subset:
                    # Compute gradients only for (originally) top predicted classes
                    grd = np.array([self.estimator.class_gradient(batch, label=int(label_i)) for label_i in labels_set])
                    grd = np.squeeze(np.swapaxes(grd, 0, 2), axis=0)
                else:
                    # Compute gradients for all classes
                    grd = self.estimator.class_gradient(batch)

                # Stop if misclassification has been achieved
                active_indices = np.where(fk_i_hat == fk_hat)[0]

                current_step += 1

            # Apply overshoot parameter
            x_adv1 = x_adv[batch_index_1:batch_index_2]
            x_adv2 = (1 + self.epsilon) * (batch - x_adv[batch_index_1:batch_index_2])
            x_adv[batch_index_1:batch_index_2] = x_adv1 + x_adv2
            # overshoot extrapolation can leave the box -> final clip
            np.clip(
                x_adv[batch_index_1:batch_index_2],
                clip_min,
                clip_max,
                out=x_adv[batch_index_1:batch_index_2],
            )

        return x_adv



class DeepFoolAttackGenerator(AttackGenerator):
    """DeepFool in InputNorm coordinates (scale-coherent minimal perturbation).

    ART DeepFool minimizes the L2 norm of the perturbation in the space the
    estimator consumes (art/attacks/evasion/deepfool.py:156-171), and
    multiplies every incremental step by (clip_values[1] - clip_values[0])
    when the estimator has clip_values set (deepfool.py:174-180). Attacking
    the DNN on RAW tabular features therefore breaks "minimal" in two ways:

      1. clip-range scaling: clip_values are the raw train min/max
         (e.g. 4e8 on NSL-KDD), so each step is inflated by the whole feature
         range -> samples destroyed (accuracy collapse to ~1-2%,
         displacement ~1e7-1e8 sigma).
      2. raw-L2 geometry: even without clip_values, a step that is cheap in
         raw L2 can move a low-sigma feature by tens of sigmas.

    Fix: the attack estimator wraps the SAME torch model (shared weights, no
    copy) through the z-space adapter from inputnorm_adapter — the function
    DeepFool sees is exactly the trained model expressed in z coordinates —
    and is built with clip_values=None so steps are added unscaled
    (deepfool.py:182) and the overshoot epsilon keeps its paper meaning
    (deepfool.py:202-205). See art_generator/inputnorm_adapter.py.

    The generator converts raw -> z before the attack and z_adv -> raw after,
    so callers keep working on raw-space DataFrames with unchanged dimensions.
    Gradient masks are unaffected: categorical/binary positions have zero
    gradient through round/clamp, so DeepFool only moves continuous features.
    """

    def __init__(self, classifier, generator_params: dict = None):
        super().__init__(classifier, generator_params)
        stats = InputNormStats.from_estimator(self.classifier)
        if stats is None or not stats.cont_idx:
            raise RuntimeError(
                "DeepFool requires the tabular DNN with an input_norm module "
                "(DNN-only attack, InputNorm coordinates)."
            )
        self.stats = stats
        logger.info(
            "[+] DeepFool InputNorm space: %d continuous features, "
            "denom range [%.4g, %.4g]",
            len(stats.cont_idx), stats.denom.min(), stats.denom.max(),
        )

        attack_params = {
            "max_iter": 100,
            "batch_size": 64,
            "nb_grads": 5,
            "epsilon": 1e-6
        }
        attack_params = self.update_generator_params(attack_params)
        attack_estimator = build_zspace_torch_estimator(self.classifier, stats)
        self.attack = _DataBoxDeepFool(classifier=attack_estimator, **attack_params)

    # ---------- coordinate conversion (kept for diagnostics/audits) ----------
    def _to_inputnorm_space(self, x: np.ndarray) -> np.ndarray:
        return self.stats.to_z(x)

    def _from_inputnorm_space(self, z_adv: np.ndarray) -> np.ndarray:
        return self.stats.from_z(z_adv)

    def generate(self, x: np.ndarray, y: np.ndarray, input_metadata: dict, mutate_indices: list[int] = [], **kwargs):
        # Attack in InputNorm coordinates, convert back to raw units
        x_in = self._to_inputnorm_space(x)
        # Per-feature valid-domain box derived from the data itself (lower
        # edge = per-feature data min -> raw >= 0; see inputnorm_adapter).
        raw_clip = getattr(self.classifier, "clip_values", None)
        raw_clip_max = raw_clip[1] if raw_clip is not None else None
        self.attack._data_box = data_zspace_clip(self.stats, x_in, raw_clip_max=raw_clip_max)
        x_adv_norm = self.attack.generate(x=x_in, y=y)
        x_adv = self._from_inputnorm_space(x_adv_norm)
        # from_z (z->raw in float32) can land a hair below the raw data
        # minimum (float roundoff at the box edge); clamp to the per-feature
        # RAW data minimum computed directly on x (roundoff-free).
        x_adv = np.maximum(x_adv, x.min(axis=0))
        if mutate_indices:
            x_adv[:, mutate_indices] = x[:, mutate_indices]
        # y is mandatory: return DataFrame combining X_adv and y
        y_arr = y.reshape(-1) if y.ndim > 1 else y
        feature_names = input_metadata['feature_names']
        df = pd.DataFrame(x_adv, columns=feature_names)
        df[input_metadata['label_column']] = y_arr
        return df

"""InputNorm coordinate system for minimum-norm attacks (shared adapter).

Rationale
---------
DeepFool, C&W, ZOO and HSJA all minimize the L2 norm of the perturbation in
the coordinate space of the estimator input. On raw tabular features with
heterogeneous scales this "minimal" is meaningless: a step that is cheap in
raw L2 can move a low-sigma feature by tens of standard deviations, and ART's
image-domain habits (clip box from clip_values, HSJA init uniform over the
whole box) destroy samples outright (see SEED_AND_ATTACK_FIX_PLAN.md,
DeepFool diagnosis D1-D3).

Fix: every minimum-norm attack runs in InputNorm coordinates
``z = (x - mu) / (sigma + eps)`` of the trained DNN — the canonical sigma
units of the project (eps_scaling.py). The attack function stays EXACTLY the
trained model: an adapter maps the continuous columns z -> raw before the
real model, which re-applies InputNorm internally (including its trained
gamma/beta affine, untouched). Gradients in z units are the raw gradients
times denom (chain rule), and the attack estimator is built with
``clip_values=None`` so ART adds steps unscaled and infers any needed box
from the (already sane) batch data instead of the raw clip range 0..4e8.

Three builders:
  - ``InputNormStats``: mu/denom buffers of a DNN checkpoint + z<->raw
    conversions (float64 math, float32 results; frozen cat/binary columns
    pass through untouched).
  - ``build_zspace_torch_estimator``: PyTorchClassifier over a shared-weight
    adapter module (for the DNN; validated by deepfool step-check:
    prediction identical to the real model, gradients = raw * denom).
  - ``build_zspace_blackbox_estimator``: dynamic subclass of any non-torch
    ART estimator (XGBoostClassifier, ...) overriding ``predict`` with the
    coordinate map — black-box attacks (ZOO, HSJA) only ever call predict.

Interface used by ART ZOO/HopSkipJump on the estimator (verified in
art/attacks/evasion/zoo.py, hop_skip_jump.py): ``predict``, ``nb_classes``,
``input_shape``, ``clip_values``, ``channels_first`` — all inherited from
the inner instance; only ``predict`` and ``clip_values`` are overridden.
"""
import numpy as np
import torch
import torch.nn as nn

from utils.logging import get_logger

logger = get_logger(__name__)


class InputNormStats:
    """Continuous-column statistics of the trained DNN InputNorm."""

    def __init__(self, cont_idx, mu, denom):
        self.cont_idx = [int(i) for i in cont_idx]
        self.mu = np.asarray(mu, dtype=np.float64)
        self.denom = np.asarray(denom, dtype=np.float64)

    @classmethod
    def from_torch_model(cls, torch_model) -> "InputNormStats | None":
        """Extract (cont_idx, mu, sigma + eps) from a tabular DNN instance."""
        norm = getattr(torch_model, "input_norm", None)
        if norm is None:
            return None
        cont_idx = [int(i) for i in getattr(torch_model, "cont_feature_indices", [])]
        mu = norm.mu.detach().cpu().numpy().astype(np.float64)
        sigma = norm.sigma.detach().cpu().numpy().astype(np.float64)
        eps = float(getattr(norm, "eps", 1e-6))
        return cls(cont_idx, mu, sigma + eps)

    @classmethod
    def from_estimator(cls, estimator) -> "InputNormStats | None":
        """From an ART estimator whose ``.model`` is the tabular DNN."""
        torch_model = getattr(estimator, "model", None)
        if torch_model is None:
            return None
        return cls.from_torch_model(torch_model)

    def to_z(self, x: np.ndarray) -> np.ndarray:
        """raw -> z on continuous columns (cat/binary pass through)."""
        z = np.array(x, dtype=np.float64, copy=True)
        if not self.cont_idx:
            return z.astype(np.float32)
        idx = np.asarray(self.cont_idx, dtype=np.int64)
        z[:, idx] = (z[:, idx] - self.mu[None, :]) / self.denom[None, :]
        return z.astype(np.float32)

    def from_z(self, z: np.ndarray) -> np.ndarray:
        """z -> raw on continuous columns (cat/binary pass through)."""
        x = np.array(z, dtype=np.float64, copy=True)
        if not self.cont_idx:
            return x.astype(np.float32)
        idx = np.asarray(self.cont_idx, dtype=np.int64)
        x[:, idx] = x[:, idx] * self.denom[None, :] + self.mu[None, :]
        return x.astype(np.float32)


class _InputNormSpaceAdapter(nn.Module):
    """Continuous columns arrive in InputNorm coordinates; map them back to
    raw and call the real net (shared reference, weights untouched, InputNorm
    intact)."""

    def __init__(self, net, cont_idx, mu, denom):
        super().__init__()
        self.net = net
        self.register_buffer("cont_idx", cont_idx)
        self.register_buffer("mu_t", mu)
        self.register_buffer("denom_t", denom)

    def forward(self, x):
        cont_raw = x.index_select(1, self.cont_idx) * self.denom_t + self.mu_t
        x2 = x.clone()
        x2.index_copy_(1, self.cont_idx, cont_raw)
        return self.net(x2)


def data_zspace_clip(
    stats: "InputNormStats",
    z: np.ndarray,
    raw_clip_max: float | None = None,
    pad: float = 0.0,
    floor: float = 1e-4,
):
    """Per-feature clip box in z coordinates, derived from the data itself.

    Purpose: C&W's tanh remap and (later) ZOO/HSJA candidate clipping need a
    box, and an UNBOUNDED z space lets the optimizer park perturbations at
    raw < 0 on count-like features that are never negative in the source data
    (measured: 100% of the CW-v2 pool rows went negative, and clamping them
    back to >= 0 reverts ~2/3 of the evasions). The lower edge is the
    per-feature data minimum — always valid, because every batch sample is a
    valid raw record (raw >= 0). The upper edge is the data max padded by
    ``pad`` times the feature's data range: attacks live within a few sigma
    of x, so the padding is generous while keeping the C&W tanh-remap
    precision error (5e-7 * box width, art/utils.py:704-706) near the 1e-5
    sigma level instead of the 1e4 sigma catastrophe of the raw clip box
    0..4e8. ``pad=0`` (default) keeps every adversarial cell within the
    OBSERVED data range — the strongest realizability claim (no value outside
    what the dataset itself contains; a 10% pad was measured to let rate
    features exceed 1.0, which is semantically impossible). Constant features
    (width 0, e.g. all-zero columns) get a floor
    width of ``floor`` sigma (1e-4, below the audit L0 tolerance 1e-3).

    Returns full-width arrays (n_features,) suitable for
    ``PyTorchClassifier.clip_values`` / ART attacks that clip elementwise.
    """
    lo = z.min(axis=0).astype(np.float64)
    hi = z.max(axis=0).astype(np.float64)
    width = hi - lo

    hi2 = hi + pad * width
    zero_width = width < 1e-12
    hi2[zero_width] = hi[zero_width] + floor
    if raw_clip_max is not None and len(stats.cont_idx):
        # upper bound must not exceed the raw clip range of continuous features
        idx = np.asarray(stats.cont_idx, dtype=np.int64)
        hi2[idx] = np.minimum(hi2[idx], (float(raw_clip_max) - stats.mu) / stats.denom)
    return lo, hi2


def build_zspace_torch_estimator(classifier, stats: InputNormStats, clip_values=None):
    """PyTorchClassifier over the SAME weights through a z-space adapter.

    clip_values=None (default) keeps DeepFool-style unscaled steps. For C&W
    the generator assigns a data-derived per-feature z box to the estimator's
    ``_clip_values`` before each generate call (see data_zspace_clip) — never
    the raw clip range, whose tanh-remap precision error is catastrophic.
    """
    from art.estimators.classification import PyTorchClassifier

    torch_model = getattr(classifier, "model", None)
    if torch_model is None or stats is None or not stats.cont_idx:
        raise RuntimeError(
            "z-space torch adapter requires a PyTorchClassifier over the "
            "tabular DNN with an input_norm module."
        )
    device = next(torch_model.parameters()).device
    net = _InputNormSpaceAdapter(
        torch_model,
        torch.tensor(stats.cont_idx, dtype=torch.long, device=device),
        torch.as_tensor(stats.mu, dtype=torch.float32, device=device),
        torch.as_tensor(stats.denom, dtype=torch.float32, device=device),
    )
    device_type = "gpu" if device.type == "cuda" else "cpu"
    logger.info(
        "[+] InputNorm adapter (torch): %d cont features, clip_values=%s",
        len(stats.cont_idx), clip_values,
    )
    return PyTorchClassifier(
        model=net,
        loss=nn.CrossEntropyLoss(),
        input_shape=tuple(classifier.input_shape),
        nb_classes=int(classifier.nb_classes),
        clip_values=clip_values,
        device_type=device_type,
    )


def build_zspace_blackbox_estimator(estimator, stats: InputNormStats):
    """Same type as ``estimator`` with ``predict`` remapped z -> raw.

    Copies the inner estimator's instance state into a dynamic subclass so
    every other interface member (nb_classes, input_shape, channels_first,
    ...) is inherited untouched; only ``predict`` is overridden. ``clip_values``
    starts as None (no clip box in z coordinates — ART then infers any needed
    box from the batch data, which is sane in z space); generators attach a
    data-derived per-feature z box per generate call via ``set_zspace_clip_box``
    (the backing attribute ``_clip_box_z`` is settable by design).
    """
    if stats is None or not stats.cont_idx:
        raise RuntimeError(
            "z-space black-box adapter requires InputNorm statistics "
            "(pass inputnorm_stats from the DNN checkpoint)."
        )

    def predict(self, x, batch_size=128, **kwargs):
        raw = self._z_stats.from_z(np.asarray(x))
        return self._z_inner.predict(raw, batch_size=batch_size, **kwargs)

    cls = type(
        "_ZSpaceEstimatorAdapter",
        (type(estimator),),
        {"predict": predict, "clip_values": property(lambda self: self._clip_box_z)},
    )
    inst = object.__new__(cls)
    inst.__dict__.update(estimator.__dict__)
    inst._z_inner = estimator
    inst._z_stats = stats
    inst._clip_box_z = None
    logger.info(
        "[+] InputNorm adapter (black-box): %s -> predict remapped, "
        "clip_values=None, %d cont features",
        type(estimator).__name__, len(stats.cont_idx),
    )
    return inst


def set_attackgen_predict_nthread(classifier, nthread: int = 1):
    """Limit thread count for single-row-heavy attack-time prediction.

    Black-box attacks (HSJA/ZOO) call estimator.predict one row at a time
    thousands of times per sample. XGBoost Booster predict spawns nthread
    threads per call, so the default (all cores) turns every single-row call
    into pure thread-thrash overhead (~200 ms under load, while a 1000-row
    batch costs nearly the same). Measured on NSL xgb: nthread=1 halves the
    per-call latency even on a loaded machine, far more when idle. Only used
    in attack-generation context — evaluation predicts in large batches,
    where the default threading is beneficial. Harmless for models without
    the Booster API (torch/sklearn wrappers are skipped).
    """
    model_obj = getattr(classifier, "model", None)
    if model_obj is not None and hasattr(model_obj, "set_param"):
        model_obj.set_param({"nthread": nthread})
        logger.info("[+] Attack-gen predict nthread=%d (%s)", nthread, type(model_obj).__name__)


def set_zspace_clip_box(attack_estimator, box):
    """Attach a z-space per-feature clip box to an adapted estimator.

    PyTorchClassifier stores its box as ``_clip_values`` (read back through
    the standard ``clip_values`` property); the black-box z-space adapter
    stores it as ``_clip_box_z``. ART attacks (ZOO/HSJA/JSMA/C&W/DeepFool
    subclasses) all read ``estimator.clip_values`` and only ever clip
    elementwise with these arrays — never scale steps by the box width, so a
    data-derived per-feature z box (data_zspace_clip) is safe to hand over.
    """
    if hasattr(attack_estimator, "_clip_box_z"):
        attack_estimator._clip_box_z = box
    else:
        attack_estimator._clip_values = box


def build_zspace_estimator_for(classifier, stats: InputNormStats | None = None):
    """Dispatcher: torch models adapt through the shared-weight module; any
    other ART estimator (XGBoost/CatBoost/sklearn) through the black-box
    remap. ``stats`` is only needed for non-torch classifiers (it must come
    from the DNN checkpoint — the canonical sigma units)."""
    torch_stats = InputNormStats.from_estimator(classifier)
    if torch_stats is not None:
        return build_zspace_torch_estimator(classifier, torch_stats), torch_stats
    return build_zspace_blackbox_estimator(classifier, stats), stats

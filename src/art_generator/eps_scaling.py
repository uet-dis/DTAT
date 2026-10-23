"""Per-feature epsilon scaling (Option B) — scale-coherent attack budgets.

Rationale
---------
The DNN model applies InputNorm internally: ``z = (x - mu) / (sigma + eps)``
(training/dnn.py:63). An attack step of ``eps_raw`` in raw feature units
therefore moves the model input by ``eps_raw / sigma_i`` standard deviations,
which for datasets with heterogeneous feature scales (e.g. CIC-2018 IAT /
duration features with sigma ~ 1e5-1e7) makes a fixed raw budget like
``eps=0.2`` a no-op for most features.

Option B fixes this by defining the budget in *normalized* units:

    eps_i     = k_eps     * sigma_i   (per continuous feature)
    eps_step_i= k_step    * sigma_i

where ``sigma_i`` is the InputNorm std actually used by the trained DNN
(restored from the checkpoint metadata, dnn.py:405-417). In InputNorm space
the attack budget is then exactly ``k_eps`` standard deviations for every
continuous feature — the same geometry for every dataset.

Frozen positions (categorical/binary features) get ``eps = 0``: the L-inf
projection then freezes them exactly (in addition to the gradient mask used
by the wrappers). Their ``eps_step`` is set to a strictly positive dummy
value because ART validates ``eps_step > 0`` element-wise for arrays
(art/attacks/evasion/fast_gradient.py:361-366) while the gradient at frozen
positions is zeroed by the mask anyway.

Supported natively by ART: per-feature eps arrays for L-inf projection
(art/utils.py:509-548, projected_gradient_descent_pytorch.py:459-508).
"""
import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)


def get_inputnorm_sigma(wrapper) -> tuple[list[int], np.ndarray]:
    """Return (cont_feature_indices, sigma) from a DNN ART wrapper.

    ``sigma`` is the InputNorm buffer restored from the training checkpoint
    (exact statistics the model normalizes with at inference).
    """
    torch_model = getattr(wrapper.model, "model", None) or wrapper.model
    norm = getattr(torch_model, "input_norm", None)
    if norm is None:
        raise RuntimeError(
            "eps_per_std requires the tabular DNN with an input_norm module "
            "(continuous branch with InputNorm)."
        )
    sigma = norm.sigma.detach().cpu().numpy().astype(float)
    cont_idx = [int(i) for i in getattr(torch_model, "cont_feature_indices", [])]
    return cont_idx, sigma


def build_per_feature_eps_arrays(
    wrapper,
    meta: dict,
    eps_scale: float,
    eps_step_scale: float,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-feature (eps, eps_step) arrays aligned to the input dim.

    - Continuous features: eps_i = eps_scale * sigma_i, step_i = eps_step_scale * sigma_i
      (sigma from the trained DNN's InputNorm, aligned with cont_feature_indices).
    - All other positions (categorical/binary, frozen by mask): eps = 0
      (exact L-inf freeze) and eps_step = eps_step_scale (dummy > 0, ART
      validates eps_step > 0 element-wise; the gradient there is zeroed).
    """
    cont_idx, sigma = get_inputnorm_sigma(wrapper)
    if len(cont_idx) != len(sigma):
        raise RuntimeError(
            f"cont_feature_indices ({len(cont_idx)}) does not match sigma ({len(sigma)})"
        )
    meta_idx = [int(i) for i in meta.get("cont_feature_indices", [])]
    if cont_idx != meta_idx:
        logger.warning(
            "[!] cont_feature_indices differ between model and data meta "
            f"(model={cont_idx}, meta={meta_idx}) — using the MODEL's indices."
        )

    eps = np.zeros(feature_dim, dtype=float)
    eps_step = np.full(feature_dim, max(eps_step_scale, 1e-12), dtype=float)
    eps[cont_idx] = eps_scale * sigma
    eps_step[cont_idx] = eps_step_scale * sigma

    logger.info(
        "[+] Per-feature eps (Option B): eps_i = %.4g * sigma_i, "
        "eps_step_i = %.4g * sigma_i (InputNorm sigma of the trained DNN); "
        "raw eps range over continuous features: [%.4g, %.4g]; "
        "budget in InputNorm space: %.4g sigma for %d features",
        eps_scale, eps_step_scale,
        eps[cont_idx].min(), eps[cont_idx].max(), eps_scale, len(cont_idx),
    )
    return eps, eps_step

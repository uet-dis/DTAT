"""Prediction-based metrics for paired adversarial pool evaluation.

This module deliberately does not load models or preprocess CSV files.  The
CLI owns those dataset-specific concerns; these functions operate on prepared
numpy arrays and an ART-style wrapper.  This keeps the pool audit
model-agnostic while allowing a separate report to combine integrity,
proximity, and attack-effectiveness metrics.
"""
from __future__ import annotations

import numpy as np


def predict_labels(wrapper, x: np.ndarray) -> np.ndarray:
    """Return integer class predictions from either labels or probabilities."""
    predictions = np.asarray(wrapper.predict(x))
    if predictions.ndim == 2:
        return np.argmax(predictions, axis=1)
    return predictions.astype(int, copy=False)


def active_row_mask(
    x_orig: np.ndarray,
    x_adv: np.ndarray,
    cont_idx: list[int],
    sigma: np.ndarray,
    *,
    unchanged_tol: float = 1e-5,
) -> np.ndarray:
    """Return rows with a displacement above the audit unchanged tolerance.

    The comparison uses the same canonical InputNorm coordinates as
    ``pool_audit``.  Categorical and binary columns are not part of this
    active-row decision; they are checked separately by the raw pool audit.
    """
    if x_orig.shape != x_adv.shape:
        raise ValueError(
            f"shape mismatch: original {x_orig.shape} vs adversarial {x_adv.shape}"
        )
    if not cont_idx:
        return np.zeros(x_orig.shape[0], dtype=bool)

    denom = np.asarray(sigma, dtype=np.float64) + 1e-6
    if len(cont_idx) != len(denom):
        raise ValueError(
            f"continuous index count {len(cont_idx)} != sigma count {len(denom)}"
        )
    delta = (
        x_adv[:, cont_idx].astype(np.float64)
        - x_orig[:, cont_idx].astype(np.float64)
    ) / denom[None, :]
    return np.max(np.abs(delta), axis=1) > unchanged_tol


def attack_effectiveness_report(
    y_true: np.ndarray,
    clean_pred: np.ndarray,
    adv_pred: np.ndarray,
    active: np.ndarray,
) -> dict[str, float | int]:
    """Compute all-row and active-row attack effectiveness statistics.

    ``asr_*_clean_correct`` follows the standard convention of evaluating
    attack success only on rows correctly classified before the attack.
    ``active_evasion_pct`` is an additional descriptive statistic whose
    denominator includes all active rows, including clean-incorrect rows.
    """
    y_true = np.asarray(y_true)
    clean_pred = np.asarray(clean_pred)
    adv_pred = np.asarray(adv_pred)
    active = np.asarray(active, dtype=bool)
    n = len(y_true)
    if clean_pred.shape != (n,) or adv_pred.shape != (n,) or active.shape != (n,):
        raise ValueError("y_true, predictions, and active must be one-dimensional and aligned")

    clean_correct = clean_pred == y_true
    success = adv_pred != y_true
    active_clean = active & clean_correct

    def pct(count: int, denominator: int) -> float:
        return 100.0 * count / denominator if denominator else 0.0

    return {
        'clean_correct': int(clean_correct.sum()),
        'clean_accuracy_pct': pct(int(clean_correct.sum()), n),
        'active_rows': int(active.sum()),
        'active_pct': pct(int(active.sum()), n),
        'unchanged_rows': int((~active).sum()),
        'asr_all_clean_correct_pct': pct(
            int((success & clean_correct).sum()), int(clean_correct.sum())
        ),
        'asr_active_clean_correct_pct': pct(
            int((success & active_clean).sum()), int(active_clean.sum())
        ),
        'active_evasion_pct': pct(int((success & active).sum()), int(active.sum())),
    }


def evaluate_attack_predictions(
    wrapper,
    x_orig: np.ndarray,
    x_adv: np.ndarray,
    y_true: np.ndarray,
    cont_idx: list[int],
    sigma: np.ndarray,
    *,
    unchanged_tol: float = 1e-5,
) -> dict[str, float | int]:
    """Predict clean/adv rows and return the paired effectiveness report."""
    active = active_row_mask(
        x_orig,
        x_adv,
        cont_idx,
        sigma,
        unchanged_tol=unchanged_tol,
    )
    return attack_effectiveness_report(
        y_true,
        predict_labels(wrapper, x_orig),
        predict_labels(wrapper, x_adv),
        active,
    )

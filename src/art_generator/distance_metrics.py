"""Report-only paired distance metrics for raw adversarial tabular data.

The attack audit already compares each adversarial pool with the clean source
row by row.  This module adds a second, raw-space view of that same pairing.
Unlike InputNorm L2, the metric applies ``log1p`` to nonnegative mutable
features so large byte/duration/count tails do not dominate the summary.

This is deliberately not a validity gate.  Features with negative or
non-finite raw values are excluded from the log metric and reported explicitly
instead of being silently clipped.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def _summary(values: np.ndarray) -> dict[str, float]:
    """Return stable summary statistics for one value per row."""
    if values.size == 0:
        return {
            'median': float('nan'),
            'p90': float('nan'),
            'max': float('nan'),
        }
    return {
        'median': float(np.median(values)),
        'p90': float(np.percentile(values, 90)),
        'max': float(np.max(values)),
    }


def _raw_numeric(df: pd.DataFrame, names: list[str]) -> np.ndarray:
    """Read selected raw columns without applying the model encoders."""
    return df.loc[:, names].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)


def paired_log_distance(
    df_orig: pd.DataFrame,
    df_adv: pd.DataFrame,
    feature_names: list[str],
    mutable_indices: list[int],
    *,
    label_values: np.ndarray | None = None,
    row_mask: np.ndarray | None = None,
) -> dict:
    """Compute paired ``log1p`` distance summaries in raw feature space.

    Parameters
    ----------
    df_orig, df_adv:
        Clean and adversarial raw frames.  Rows must have the same order.
    feature_names:
        Feature order used by the model, excluding the label column.
    mutable_indices:
        Indices of model-mutable columns.  The audit passes continuous
        feature indices, thereby excluding categorical and binary columns.
    label_values:
        Optional clean labels used to add a compact per-label JSON summary.
    row_mask:
        Optional boolean mask selecting rows to include. The audit passes
        active rows only, excluding its existing unchanged-row mask.

    Notes
    -----
    ``log1p`` is only defined here for columns whose paired raw values are all
    finite and nonnegative.  Such columns are included; other columns are
    skipped and named in the returned report.  No clipping or repair occurs.
    """
    indices = [int(i) for i in mutable_indices if 0 <= int(i) < len(feature_names)]
    names = [feature_names[i] for i in indices]

    base = {
        'log_distance_total_rows': int(len(df_orig)),
        'log_distance_active_rows': 0,
        'log_distance_excluded_rows': int(len(df_orig)),
        'log_distance_features': 0,
        'log_distance_skipped_features': '',
        'log_distance_invalid_cells': 0,
        'log_distance_top_features': '',
        'log_distance_by_label': '',
    }
    for prefix in ('log_l2', 'log_rms', 'log_l1_mean'):
        base[f'{prefix}_median'] = float('nan')
        base[f'{prefix}_p90'] = float('nan')
        base[f'{prefix}_max'] = float('nan')

    if len(df_orig) != len(df_adv) or not names:
        base['log_distance_skipped_features'] = ';'.join(names)
        return base

    if row_mask is None:
        include = np.ones(len(df_orig), dtype=bool)
    else:
        include = np.asarray(row_mask, dtype=bool)
        if include.shape != (len(df_orig),):
            raise ValueError(
                f"row_mask shape {include.shape} does not match rows {(len(df_orig),)}"
            )
    base['log_distance_active_rows'] = int(include.sum())
    base['log_distance_excluded_rows'] = int((~include).sum())
    if not include.any():
        return base

    orig = _raw_numeric(df_orig, names)[include]
    adv = _raw_numeric(df_adv, names)[include]
    pair = np.concatenate((orig, adv), axis=0)
    valid_cols = np.isfinite(pair).all(axis=0) & (pair >= 0).all(axis=0)
    invalid_cells = int((~np.isfinite(pair) | (pair < 0)).sum())
    skipped = [name for name, valid in zip(names, valid_cols) if not valid]
    base['log_distance_features'] = int(valid_cols.sum())
    base['log_distance_skipped_features'] = ';'.join(skipped)
    base['log_distance_invalid_cells'] = invalid_cells

    if not valid_cols.any():
        return base

    valid_names = [name for name, valid in zip(names, valid_cols) if valid]
    delta = np.log1p(adv[:, valid_cols]) - np.log1p(orig[:, valid_cols])
    abs_delta = np.abs(delta)
    row_l2 = np.sqrt(np.sum(delta ** 2, axis=1))
    row_rms = np.sqrt(np.mean(delta ** 2, axis=1))
    row_l1_mean = np.mean(abs_delta, axis=1)

    for prefix, values in (
        ('log_l2', row_l2),
        ('log_rms', row_rms),
        ('log_l1_mean', row_l1_mean),
    ):
        stats = _summary(values)
        for suffix, value in stats.items():
            base[f'{prefix}_{suffix}'] = value

    feature_contribution = np.median(abs_delta, axis=0)
    order = np.argsort(-feature_contribution, kind='stable')[:5]
    base['log_distance_top_features'] = json.dumps([
        {'feature': valid_names[int(i)], 'median_abs_delta': float(feature_contribution[int(i)])}
        for i in order
    ], separators=(',', ':'))

    if label_values is not None and len(label_values) == len(df_orig):
        labels = np.asarray(label_values)[include]
        by_label = {}
        for label in pd.unique(labels):
            mask = labels == label
            by_label[str(label)] = {
                'n': int(mask.sum()),
                'log_rms': _summary(row_rms[mask]),
                'log_l1_mean': _summary(row_l1_mean[mask]),
            }
        base['log_distance_by_label'] = json.dumps(by_label, separators=(',', ':'))

    return base

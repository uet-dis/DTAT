"""Post-generation quality audit for adversarial sample pools.

Why
---
A quality gate applied AFTER generation and BEFORE any robustness
evaluation. Its purpose is to catch destroyed / incoherent samples from the
deviation against the original test set, per the standard practice in the
adversarial-robustness literature:

  - Minimum-norm attacks (DeepFool, C&W, ZOO, HSJA) are reported through their
    perturbation-size distribution, not only success rate (Moosavi-Dezfooli
    et al. 2015; Carlini et al., "On Evaluating Adversarial Robustness",
    arXiv:1902.06705 — median/average minimal distance).
  - Benchmarks discard or neutralize minimum-norm results above a declared
    budget (arXiv:1912.11852: A(x) = x if ||delta||_p > eps).
  - Tabular data requires validity checks beyond images: unchanged immutable
    features, feasibility of continuous values (LowProFool, arXiv:1911.03274;
    "Investigating Imperceptibility of Adversarial Attacks on Tabular Data",
    arXiv:2407.11463).

What is checked (per file `{resource}_{model}_{attack}_adv.csv`)
---------------------------------------------------------------
integrity      row count identical to the source test set, label column
               unchanged, no NaN/Inf in the perturbed features.
frozen         categorical (ordinal integers) and binary features are
               bit-exactly unchanged — they are frozen by the gradient mask
               plus mutate_indices at generation time, so any change is a
               pipeline bug, not an attack property.
displacement   L2 and L-inf (max |dz|) of the perturbation in InputNorm
               coordinates z = (x - mu) / (sigma + eps), computed on the
               continuous features only, plus L0 (modified-feature count at
               tolerance). This is the primary metric for minimum-norm
               attacks. The InputNorm statistics are the DNN checkpoint's
               (get_inputnorm_sigma) — the canonical sigma units of the
               project (see eps_scaling.py).
budget         for bounded attacks: share of rows whose per-feature deviation
               stays within the DECLARED budget read from the generation
               config. Two modes are supported:
                 'sigma' (eps_per_std, Option B): |dz_i| <= k_eps for every
                   continuous feature (budget defined in InputNorm space).
                 'raw' (scalar eps): |dx_i| <= eps_raw, i.e. the per-feature
                   budget in z units is eps_raw / (sigma_i + eps).
unchanged      share of rows with NO modified feature: placeholder rows of
               failed batches (--placeholder original) or fully failed
               attacks. Large values mean the pool under-reports the attack.
destroyed      rows whose L2 deviation exceeds a destruction threshold
               (default 10 sigma) — REPORT ONLY, not a status gate. The
               threshold was calibrated for the DeepFool v2 artifact
               (1e7-1e8 sigma, accuracy collapse); on the v2 in-box pool the
               natural heavy-tail rows legitimately reach 10-130 sigma while
               staying inside the observed data range, so a displacement
               threshold cannot serve as a validity gate (measured:
               naive position metrics Mahalanobis/quantile/kNN are all
               ill-conditioned on network data — see art_generator/
               CHANGELOG.md). Validity below is the gate instead.
validity       data-driven out-of-support check in BOTH directions: a
               continuous feature that never takes negative values in the
               source data (counts, bytes, rates) must not go negative, and
               no value may exceed the observed per-feature maximum.
               Reference domain = per-feature min/max of the ORIGINAL file
               itself. This is what defines destroyed/invalid in v2: every
               v1-artifact value (raw 1e8-1e9, >1e7 sigma) was
               out-of-support, while in-box natural tails are not flagged.

The audit is inference-only (encode + numpy + one inverse transform); no
attack or gradient computation, no GPU. Run it with a single thread if it
must not compete with other jobs.
"""
import os
import re

import numpy as np
import pandas as pd

from utils.logging import get_logger
from art_generator.distance_metrics import paired_log_distance

logger = get_logger(__name__)

# Destruction threshold (sigma units) — DeepFool v2 artifact sat at 1e7+.
DESTRUCTION_THRESHOLD = 10.0
FILE_RE = re.compile(r"^(?P<resource>.+)_(?P<model>[^_]+)_(?P<attack>[^_]+)_adv\.csv$")


def budgets_from_attack_params(attack_params: dict) -> dict | None:
    """Interpret a generation-config entry as a declared perturbation budget.

    Returns ``{'mode': 'sigma', 'value': k}`` when the config declares
    ``eps_per_std`` (Option B — budget in InputNorm space), or
    ``{'mode': 'raw', 'value': eps}`` when it declares a scalar raw ``eps``
    (ART L-inf ball in raw feature units). ``None`` for minimum-norm attacks
    without a declared budget.
    """
    if not attack_params:
        return None
    if attack_params.get('eps_per_std') is not None:
        return {'mode': 'sigma', 'value': float(attack_params['eps_per_std'])}
    if attack_params.get('eps') is not None:
        return {'mode': 'raw', 'value': float(attack_params['eps'])}
    return None


def _z(x: np.ndarray, cont_idx: list[int], denom: np.ndarray) -> np.ndarray:
    """Raw -> InputNorm coordinates for the continuous columns only."""
    z = x.astype(np.float64, copy=True)
    z[:, cont_idx] = (z[:, cont_idx] - 0.0) / denom[None, :]
    return z


def audit_attack_file(
    adv_csv: str,
    orig_csv: str,
    pre,
    meta: dict,
    cont_idx: list[int],
    sigma: np.ndarray,
    *,
    budgets: dict | None = None,
    ztol: float = 1e-3,
    unchanged_tol: float = 1e-5,
    destruction_threshold: float = DESTRUCTION_THRESHOLD,
) -> dict:
    """Audit one adversarial CSV against its source test CSV.

    Parameters
    ----------
    adv_csv, orig_csv : paths. ``orig_csv`` is the raw test CSV the pool was
        generated from; ``adv_csv`` follows
        ``{resource}_{model}_{attack}_adv.csv``.
    pre : fitted preprocessor (ordinal/binary/label encoders loaded).
    meta : metadata from ``PrepareData.prepare_input_data`` on the ORIGINAL
        file (feature order, cat/binary/cont indices, label column).
    cont_idx, sigma : continuous-feature input positions and InputNorm sigma
        (canonical units) — from ``eps_scaling.get_inputnorm_sigma``.
    budgets : mapping attack -> {'mode': 'sigma'|'raw', 'value': k} as
        produced by ``budgets_from_attack_params`` (declared generation
        config). Attacks missing from the mapping are minimum-norm and get
        no budget check.
    ztol : |dz| tolerance (sigma units) below which a feature counts as
        unmodified for L0/displacement statistics (float32 roundtrip noise
        ~1e-6 sigma at worst). Separately, ``unchanged_tol`` is the tighter
        bit-precision tolerance for counting rows the attack left entirely
        untouched (placeholder rows of failed batches).
    unchanged_tol : |dz| threshold (sigma units) for the unchanged-row count.
        Default 1e-5 sigma: comfortably above float32 roundtrip noise
        (~1e-6 sigma) and below real hairline perturbations (~1e-4 sigma).
    """
    budgets = budgets or {}
    filename = os.path.basename(adv_csv)
    m = FILE_RE.match(filename)
    if not m:
        raise ValueError(f"cannot parse model/attack from filename: {filename}")
    model, attack = m.group('model'), m.group('attack')

    df_orig = pd.read_csv(orig_csv, low_memory=False)
    df_adv = pd.read_csv(adv_csv, low_memory=False)
    label_col = meta['label_column']
    feature_names = meta['feature_names']

    rep: dict = {'file': filename, 'model': model, 'attack': attack}
    rep['n_rows_orig'], rep['n_rows_adv'] = len(df_orig), len(df_adv)

    # ---------- integrity ----------
    rep['rows_match'] = bool(len(df_orig) == len(df_adv))
    if not rep['rows_match']:
        logger.error("[audit] %s: row count %d != %d", filename, len(df_adv), len(df_orig))
        rep['status'] = 'FAIL'
        return rep

    y_orig = df_orig[label_col].to_numpy()
    y_adv = df_adv[label_col].to_numpy()
    rep['labels_unchanged'] = bool(np.array_equal(y_orig, y_adv))

    # prepare both through the identical encode path (ordinal + binary + label)
    Xo, yo, mo = _prepare(df_orig, pre)
    Xa, ya, _ = _prepare(df_adv, pre)
    meta_use = mo or meta

    cat_idx = list(meta_use['cat_feature_indices'])
    bin_idx = list(meta_use['binary_feature_indices'])
    cont_idx_use = list(meta_use['cont_feature_indices'])
    if cont_idx_use != list(cont_idx):
        logger.warning(
            "[audit] %s: cont_feature_indices differ between meta %s and model %s — using meta.",
            filename, cont_idx_use, list(cont_idx),
        )
    denom = np.asarray(sigma, dtype=np.float64) + 1e-6

    # NaN/Inf on the perturbed features
    rep['finite'] = bool(np.isfinite(Xa).all())

    # ---------- frozen positions: cat (ordinal ints) + binary ----------
    frozen_idx = cat_idx + bin_idx
    if frozen_idx:
        frozen_equal = np.all(Xo[:, frozen_idx] == Xa[:, frozen_idx], axis=1)
        rep['frozen_changed_rows'] = int((~frozen_equal).sum())
    else:
        rep['frozen_changed_rows'] = 0

    # ---------- continuous displacement in InputNorm coordinates ----------
    zo = _z(Xo.astype(np.float64), cont_idx_use, denom)
    za = _z(Xa.astype(np.float64), cont_idx_use, denom)
    dz = za - zo

    l2 = np.sqrt((dz ** 2).sum(axis=1))
    linf = np.abs(dz).max(axis=1)
    modified = np.abs(dz) > ztol
    l0 = modified.sum(axis=1)

    rep['l2_median'] = float(np.median(l2))
    rep['l2_p90'] = float(np.percentile(l2, 90))
    rep['l2_max'] = float(l2.max())
    rep['linf_median'] = float(np.median(linf))
    rep['linf_p90'] = float(np.percentile(linf, 90))
    rep['l0_median'] = float(np.median(l0))
    rep['l0_max'] = int(l0.max())

    # unchanged rows: attack produced nothing (placeholder policy or failure).
    # Bit-precision tolerance (1e-5 sigma) — must NOT use the L0 tolerance:
    # HSJA hairline evasions (single tree-threshold crossing) move features
    # by ~1e-4 sigma and would be miscounted as unchanged otherwise.
    unchanged_mask = np.abs(dz).max(axis=1) <= unchanged_tol
    rep['unchanged_rows'] = int(unchanged_mask.sum())
    rep['unchanged_pct'] = float(100.0 * unchanged_mask.mean())

    # ---------- paired raw-space distance on active rows (REPORT ONLY) ----------
    # Keep this separate from the encoded/InputNorm audit. The raw values are
    # the values written to the pool, while the continuous indices define the
    # model-mutable columns and exclude cat/binary fields. Exclude the same
    # unchanged rows already reported above so placeholder/failure rows do not
    # dilute the paired-distance summaries.
    rep.update(paired_log_distance(
        df_orig,
        df_adv,
        list(meta['feature_names']),
        list(meta['cont_feature_indices']),
        label_values=y_orig,
        row_mask=~unchanged_mask,
    ))

    # destroyed rows (DeepFool-v2 signature) — REPORT ONLY, not a status
    # gate: on heavy-tailed network features the observed range extends
    # hundreds of sigma from the mean (CIC box edge: 200-237 sigma), so
    # in-range natural tails legitimately exceed 10 sigma. The artifact
    # this count was calibrated for (raw-space 1e7-1e8 sigma) is
    # out-of-support and is caught by the domain check below instead.
    rep['destroyed_rows'] = int((l2 > destruction_threshold).sum())
    rep['destroyed_pct'] = float(100.0 * (l2 > destruction_threshold).mean())

    # ---------- budget conformance (bounded attacks) ----------
    b = budgets.get(attack)
    if b is not None and b.get('value') is not None:
        k = float(b['value'])
        dzc = dz[:, cont_idx_use]  # cont columns only (frozen cols are zero anyway)
        if b.get('mode') == 'sigma':
            # |dz_i| <= k for every continuous feature
            viol_cells = np.abs(dzc) > (k + ztol)
        else:
            # raw mode: |dx_i| = |dz_i| * denom_i <= k, with float tolerance
            viol_cells = np.abs(dzc) > (k / denom[None, :] + ztol)
        ok_rows = ~viol_cells.any(axis=1)
        rep['budget_mode'] = b.get('mode')
        rep['budget_value'] = k
        rep['budget_ok_pct'] = float(100.0 * ok_rows.mean())
        rep['budget_violations'] = int((~ok_rows).sum())
        if rep['budget_violations']:
            logger.warning(
                "[audit] %s: %d rows exceed the declared %s budget %.4g (%s)",
                filename, rep['budget_violations'],
                b.get('mode'), k, attack,
            )
    else:
        rep['budget_mode'] = None
        rep['budget_value'] = None
        rep['budget_ok_pct'] = None
        rep['budget_violations'] = 0

    # ---------- data-driven validity (domain of continuous features) ----------
    # Out-of-support check in BOTH directions (this is the v2 validity gate):
    #   below: a feature that never goes negative in the source data must
    #          stay >= 0 — no attack config permits negatives (ART clips
    #          bounded attacks to the global raw box, whose lower edge is 0).
    #   above: no adversarial value may exceed the observed per-feature max,
    #          EXCEPT within the attack's DECLARED budget for bounded
    #          attacks (raw mode: k raw units; sigma mode: k * denom raw
    #          units) — the standard truncation convention A(x) = x if
    #          ||delta|| > eps (arXiv:1912.11852) explicitly permits moves
    #          up to eps above the observed max for L-inf attacks. Attacks
    #          with no declared budget (minimum-norm) get no slack: the
    #          pad=0 z-space box keeps them inside the observed range.
    lo = Xo[:, cont_idx_use].min(axis=0)
    hi = Xo[:, cont_idx_use].max(axis=0)
    neg_allowed = lo < 0
    below_cols = np.where(~neg_allowed)[0]
    violations = np.zeros(len(cont_idx_use), dtype=bool)
    below_cells = 0
    if below_cols.size:
        below_cells = int(
            sum((Xa[:, cont_idx_use][:, j] < -ztol * denom[j]).sum()
                for j in below_cols)
        )
        violations[below_cols] = (
            Xa[:, cont_idx_use][:, below_cols] < -ztol * denom[below_cols][None, :]
        ).any(axis=0)
    # declared-budget slack above the observed max (raw units, per feature)
    slack = np.zeros(len(cont_idx_use))
    if b is not None and b.get('value') is not None:
        slack[:] = float(b['value']) * (1.0 if b.get('mode') == 'raw' else denom)
    above_mask = Xa[:, cont_idx_use] > (hi + slack + ztol * denom)[None, :]
    above_cells = int(above_mask.sum())
    violations |= above_mask.any(axis=0)
    rep['domain_violation_features'] = int(violations.sum())
    rep['domain_violation_cells'] = below_cells + above_cells

    # ---------- status ----------
    rep['status'] = _status(rep)
    return rep


def _prepare(df: pd.DataFrame, pre) -> tuple[np.ndarray, np.ndarray, dict]:
    """Encode a raw CSV exactly like the pipeline does (ordinal+binary+label)."""
    from preprocessing.prepare import PrepareData
    X, y, meta = PrepareData.prepare_input_data(df, pre, include_label=True)
    return X.astype(np.float64), y, meta


def _status(rep: dict) -> str:
    """v2 status: validity (out-of-support), not displacement, is the gate.

    ``destroyed_rows`` (L2 > 10 sigma) is informational: in-box natural
    tails on heavy-tailed features legitimately exceed it, and naive
    position metrics (Mahalanobis / quantile / kNN) were measured to be
    ill-conditioned on this data — see CHANGELOG.md. Any other hard
    violation (integrity, labels, frozen, budget, domain) that is not the
    domain gate itself reports WARN.
    """
    if not rep.get('rows_match', False) or not rep.get('finite', False):
        return 'FAIL'
    hard = [
        rep.get('labels_unchanged', False),
        rep.get('frozen_changed_rows', 0) == 0,
        rep.get('budget_violations', 0) == 0,
        rep.get('domain_violation_cells', 0) == 0,
    ]
    return 'PASS' if all(hard) else 'WARN'


def audit_pool(
    pool_dir: str,
    orig_csv: str,
    pre,
    meta: dict,
    cont_idx: list[int],
    sigma: np.ndarray,
    *,
    budgets: dict | None = None,
    ztol: float = 1e-3,
    unchanged_tol: float = 1e-5,
    destruction_threshold: float = DESTRUCTION_THRESHOLD,
) -> pd.DataFrame:
    """Audit every ``*_adv.csv`` in ``pool_dir``; return one row per file."""
    budgets = dict(budgets or {})
    files = sorted(f for f in os.listdir(pool_dir) if f.endswith('_adv.csv'))
    if not files:
        raise FileNotFoundError(f"no *_adv.csv files in {pool_dir}")
    rows = []
    for f in files:
        path = os.path.join(pool_dir, f)
        logger.info("[audit] %s", f)
        try:
            rows.append(audit_attack_file(
                path, orig_csv, pre, meta, cont_idx, sigma,
                budgets=budgets, ztol=ztol, unchanged_tol=unchanged_tol,
                destruction_threshold=destruction_threshold,
            ))
        except Exception as e:  # keep auditing the remaining files
            logger.error("[audit] %s failed: %s", f, e)
            mm = FILE_RE.match(f)
            rows.append({
                'file': f,
                'model': mm.group('model') if mm else '?',
                'attack': mm.group('attack') if mm else '?',
                'status': 'ERROR', 'error': str(e),
            })
    return pd.DataFrame(rows)


def print_report(report: pd.DataFrame) -> None:
    """Compact console rendering of the audit report."""
    cols = ['file', 'status', 'n_rows_adv', 'labels_unchanged', 'frozen_changed_rows',
            'unchanged_pct', 'l2_median', 'l2_p90', 'l2_max', 'linf_median',
            'l0_median', 'destroyed_rows', 'log_l2_median', 'log_rms_median',
            'log_rms_p90', 'log_rms_max', 'log_l1_mean_median',
            'log_distance_active_rows', 'log_distance_excluded_rows',
            'log_distance_features', 'log_distance_invalid_cells',
            'budget_ok_pct',
            'domain_violation_cells']
    cols = [c for c in cols if c in report.columns]
    with pd.option_context('display.max_columns', None, 'display.width', 220,
                           'display.float_format', lambda v: f"{v:.4g}"):
        print(report[cols].to_string(index=False))
    n_fail = (report['status'] == 'FAIL').sum()
    n_warn = (report['status'] == 'WARN').sum()
    print(f"\nSummary: {len(report)} files — PASS {len(report) - n_fail - n_warn}, "
          f"WARN {n_warn}, FAIL {n_fail}")

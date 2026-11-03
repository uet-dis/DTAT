"""Domain constraints for NIDS feature vectors and a repair operator Pi_Omega.

The constraints are restricted to relations that hold on the real held-out records
(checked on the test splits; the WGAN-augmented training set is not used to define
them). Two levels are provided:

  omega1  consistency: immutable coordinates, integrality, non-negativity, value
          ranges, and the arithmetic relations between flow statistics
          (CSE-CIC-IDS2018), or integrality / ranges / rate granularity (NSL-KDD).
  omega2  omega1 plus an attacker-capability model: on CSE-CIC-IDS2018 only the
          forward direction may change and only additively (more or larger
          forward packets, longer duration); backward-direction, header-option
          and activity/idle features stay at the original record's values. On
          NSL-KDD only duration and src_bytes may change, non-decreasingly.

repair(adv, orig, level) maps a perturbed vector to a vector satisfying the level.
It is a deterministic repair that recomputes every derived statistic from the
primary ones; it is not a nearest-point projection.

audit(adv, orig) reports, per record, whether the vector satisfies every relation
that its original record satisfies, so residual irregularities of the source data
are not counted as violations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-9

# --------------------------------------------------------------------------- #
# CSE-CIC-IDS2018                                                              #
# --------------------------------------------------------------------------- #
CIC_IMMUTABLE = ['Dst Port', 'Protocol', 'Init Fwd Win Byts', 'Fwd Seg Size Min',
                 'ACK Flag Cnt', 'ECE Flag Cnt', 'PSH Flag Cnt', 'RST Flag Cnt']
CIC_INT = ['Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
           'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
           'Flow Duration', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Act Data Pkts',
           'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Max', 'Fwd IAT Min',
           'Bwd IAT Tot', 'Bwd IAT Max', 'Bwd IAT Min',
           'Active Max', 'Active Min', 'Idle Max', 'Idle Min']
# (min, mean, max) triples whose mean is not derived from totals.
CIC_TRIPLES = [('Flow IAT Min', 'Flow IAT Mean', 'Flow IAT Max'),
               ('Fwd IAT Min', 'Fwd IAT Mean', 'Fwd IAT Max'),
               ('Bwd IAT Min', 'Bwd IAT Mean', 'Bwd IAT Max'),
               ('Active Min', 'Active Mean', 'Active Max'),
               ('Idle Min', 'Idle Mean', 'Idle Max')]
CIC_BWD_FIXED = ['Tot Bwd Pkts', 'TotLen Bwd Pkts', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
                 'Bwd Pkt Len Std', 'Bwd Header Len', 'Bwd IAT Tot', 'Bwd IAT Mean',
                 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
                 'Active Max', 'Active Mean', 'Active Min', 'Active Std',
                 'Idle Max', 'Idle Mean', 'Idle Min', 'Idle Std']
CIC_FWD_NONDECREASING = ['Tot Fwd Pkts', 'TotLen Fwd Pkts', 'Fwd Pkt Len Max', 'Flow Duration',
                         'Fwd IAT Tot', 'Fwd IAT Max', 'Fwd Act Data Pkts']


def _cic_relations(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean satisfaction of each CIC relation, per record."""
    F, B = df['Tot Fwd Pkts'], df['Tot Bwd Pkts']
    LF, LB = df['TotLen Fwd Pkts'], df['TotLen Bwd Pkts']
    D = df['Flow Duration']
    sec = (D / 1e6).where(D > 0)

    def close(a, b, tol=1e-3):
        a, b = np.asarray(a, float), np.asarray(b, float)
        both_nan = np.isnan(a) & np.isnan(b)
        return np.isclose(a, b, rtol=tol, atol=tol) | both_nan

    rel = {
        'nonneg': (df.select_dtypes('number') >= -_EPS).all(axis=1).values,
        'integer': np.all([np.isclose(df[c], np.round(df[c])) for c in CIC_INT], axis=0),
        'fwd_mean': close(df['Fwd Pkt Len Mean'], np.where(F > 0, LF / F.clip(lower=1), 0)),
        'bwd_mean': close(df['Bwd Pkt Len Mean'], np.where(B > 0, LB / B.clip(lower=1), 0)),
        'fwd_seg': close(df['Fwd Seg Size Avg'], df['Fwd Pkt Len Mean']),
        'bwd_seg': close(df['Bwd Seg Size Avg'], df['Bwd Pkt Len Mean']),
        'subflow': (close(df['Subflow Fwd Pkts'], F) & close(df['Subflow Bwd Pkts'], B)
                    & close(df['Subflow Fwd Byts'], LF) & close(df['Subflow Bwd Byts'], LB)),
        'pkt_max': close(df['Pkt Len Max'], np.maximum(df['Fwd Pkt Len Max'], df['Bwd Pkt Len Max'])),
        'pkt_min': close(df['Pkt Len Min'], np.minimum(df['Fwd Pkt Len Min'],
                                                       np.where(B > 0, df['Bwd Pkt Len Min'], np.inf))),
        'pkt_mean': close(df['Pkt Len Mean'], (LF + LB) / (F + B + 1)),
        'pkt_size': close(df['Pkt Size Avg'], (LF + LB) / (F + B).clip(lower=1)),
        'var_std': close(df['Pkt Len Var'], df['Pkt Len Std'] ** 2),
        'down_up': close(df['Down/Up Ratio'], np.floor(B / F.clip(lower=1))),
        'rates': (close(df['Flow Byts/s'].where(D > 0), ((LF + LB) / sec))
                  & close(df['Flow Pkts/s'].where(D > 0), ((F + B) / sec))
                  & close(df['Fwd Pkts/s'].where(D > 0), (F / sec))
                  & close(df['Bwd Pkts/s'].where(D > 0), (B / sec))),
        'len_order': ((df['Fwd Pkt Len Min'] <= df['Fwd Pkt Len Mean'] + 1e-6)
                      & (df['Fwd Pkt Len Mean'] <= df['Fwd Pkt Len Max'] + 1e-6)
                      & (df['Bwd Pkt Len Min'] <= df['Bwd Pkt Len Mean'] + 1e-6)
                      & (df['Bwd Pkt Len Mean'] <= df['Bwd Pkt Len Max'] + 1e-6)).values,
        'iat_order': np.all([((df[lo] <= df[mu] + 1e-6) & (df[mu] <= df[hi] + 1e-6)).values
                             for lo, mu, hi in CIC_TRIPLES], axis=0),
        'act_data': (df['Fwd Act Data Pkts'] <= F).values,
    }
    return {k: np.asarray(v, bool) for k, v in rel.items()}


def _cic_repair(adv: pd.DataFrame, orig: pd.DataFrame, level: str, strict: bool = False) -> pd.DataFrame:
    x = adv.copy()
    o = orig
    num = x.select_dtypes('number').columns
    x[num] = x[num].clip(lower=0)
    for c in _present(x, CIC_IMMUTABLE):
        x[c] = o[c].values
    if level == 'omega2':
        for c in CIC_BWD_FIXED:
            x[c] = o[c].values
        for c in CIC_FWD_NONDECREASING:
            x[c] = np.maximum(x[c].values, o[c].values)
        # Additional forward packets carry the record's own per-packet header size.
        hdr = np.where(o['Tot Fwd Pkts'] > 0, o['Fwd Header Len'] / o['Tot Fwd Pkts'].clip(lower=1), 0)
        x['Fwd Header Len'] = o['Fwd Header Len'].values + (x['Tot Fwd Pkts'].values
                                                            - o['Tot Fwd Pkts'].values) * hdr
    for c in CIC_INT:
        x[c] = np.round(x[c])
    x['Fwd Act Data Pkts'] = np.minimum(x['Fwd Act Data Pkts'], x['Tot Fwd Pkts'])

    # A few real records deviate slightly from the CICFlowMeter relations (e.g. the
    # packet-mean denominator). Their original deviation is carried over, so a clean
    # record maps to itself and only relations the original satisfies are imposed.
    # strict=True imposes every relation regardless of the original (used to repair
    # synthetic training records, whose deviations are generator artifacts).
    derived = _cic_derive(x, o)
    if strict:
        num_cols = derived.select_dtypes('number').columns
        derived[num_cols] = derived[num_cols].clip(lower=0)
        return derived
    o_num = o.select_dtypes('number')
    o_der = _cic_derive(o.copy(), o)[o_num.columns]
    deviates = ~np.isclose(o_num.values, o_der.values, rtol=1e-3, atol=1e-3)
    offset = np.where(deviates, o_num.values - o_der.values, 0.0)
    derived[o_num.columns] = np.maximum(derived[o_num.columns].values + offset, 0.0)
    return derived


def _cic_derive(x: pd.DataFrame, o: pd.DataFrame) -> pd.DataFrame:
    """Recompute every derived CIC statistic from the primary ones."""
    F, B = x['Tot Fwd Pkts'], x['Tot Bwd Pkts']
    for d, n, L in (('Fwd', F, 'TotLen Fwd Pkts'), ('Bwd', B, 'TotLen Bwd Pkts')):
        mean = np.where(n > 0, x[L] / n.clip(lower=1), 0.0)
        x[f'{d} Pkt Len Mean'] = mean
        x[f'{d} Pkt Len Min'] = np.where(n > 0, np.minimum(x[f'{d} Pkt Len Min'], np.floor(mean)), 0)
        x[f'{d} Pkt Len Max'] = np.where(n > 0, np.maximum(x[f'{d} Pkt Len Max'], np.ceil(mean)), 0)
        x[f'{d} Pkt Len Std'] = np.where(n > 1, x[f'{d} Pkt Len Std'], 0.0)
        x[f'{d} Seg Size Avg'] = mean
    LF, LB = x['TotLen Fwd Pkts'], x['TotLen Bwd Pkts']
    x['Subflow Fwd Pkts'], x['Subflow Bwd Pkts'] = F, B
    x['Subflow Fwd Byts'], x['Subflow Bwd Byts'] = LF, LB
    x['Pkt Len Max'] = np.maximum(x['Fwd Pkt Len Max'], x['Bwd Pkt Len Max'])
    x['Pkt Len Min'] = np.minimum(x['Fwd Pkt Len Min'], np.where(B > 0, x['Bwd Pkt Len Min'], np.inf))
    x['Pkt Len Mean'] = (LF + LB) / (F + B + 1)
    x['Pkt Size Avg'] = (LF + LB) / (F + B).clip(lower=1)
    x['Pkt Len Var'] = x['Pkt Len Std'] ** 2
    x['Down/Up Ratio'] = np.floor(B / F.clip(lower=1))
    for lo, mu, hi in CIC_TRIPLES:
        x[lo] = np.minimum(x[lo], np.floor(x[mu]))
        x[hi] = np.maximum(x[hi], np.ceil(x[mu]))
    D = x['Flow Duration']
    sec = D / 1e6
    pos = D > 0
    for col, val in (('Flow Byts/s', LF + LB), ('Flow Pkts/s', F + B), ('Fwd Pkts/s', F), ('Bwd Pkts/s', B)):
        x[col] = np.where(pos, val / sec.where(pos, 1.0), o[col].values)
    return x


# --------------------------------------------------------------------------- #
# NSL-KDD                                                                      #
# --------------------------------------------------------------------------- #
NSL_IMMUTABLE = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
NSL_RANGE_INT = {'count': 511, 'srv_count': 511, 'dst_host_count': 255, 'dst_host_srv_count': 255,
                 'root_shell': 1, 'su_attempted': 2, 'wrong_fragment': 3, 'urgent': 3}
NSL_INT = ['duration', 'src_bytes', 'dst_bytes', 'hot', 'num_failed_logins', 'num_compromised',
           'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds']
NSL_RATE_PAIRS = [('serror_rate', 'rerror_rate'), ('srv_serror_rate', 'srv_rerror_rate'),
                  ('dst_host_serror_rate', 'dst_host_rerror_rate'),
                  ('dst_host_srv_serror_rate', 'dst_host_srv_rerror_rate')]
NSL_CAPABILITY = ['duration', 'src_bytes']


def _nsl_rates(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith('_rate')]


def _nsl_relations(df: pd.DataFrame) -> dict[str, np.ndarray]:
    rates = _nsl_rates(df)
    ints = NSL_INT + list(NSL_RANGE_INT)
    r = df[rates].values
    rel = {
        'nonneg': (df.select_dtypes('number') >= -_EPS).all(axis=1).values,
        'integer': np.isclose(df[ints].values, np.round(df[ints].values)).all(axis=1),
        'ranges': np.all([(df[c] <= hi + _EPS).values for c, hi in NSL_RANGE_INT.items()], axis=0),
        'rate_box': ((r >= -_EPS) & (r <= 1 + _EPS)).all(axis=1),
        'rate_grid': np.isclose(r * 100, np.round(r * 100), atol=1e-6).all(axis=1),
        'rate_pairs': np.all([(df[a] + df[b] <= 1 + 1e-6).values for a, b in NSL_RATE_PAIRS], axis=0),
    }
    return {k: np.asarray(v, bool) for k, v in rel.items()}


def _nsl_repair(adv: pd.DataFrame, orig: pd.DataFrame, level: str) -> pd.DataFrame:
    x = adv.copy()
    o = orig
    for c in _present(x, NSL_IMMUTABLE):
        x[c] = o[c].values
    if level == 'omega2':
        keep = [c for c in x.columns if c not in NSL_CAPABILITY and c in o.columns and c != 'Label']
        for c in keep:
            x[c] = o[c].values
        for c in NSL_CAPABILITY:
            x[c] = np.maximum(x[c].values, o[c].values)
    num = [c for c in x.select_dtypes('number').columns if c not in NSL_IMMUTABLE]
    x[num] = x[num].clip(lower=0)
    for c in NSL_INT:
        x[c] = np.round(x[c])
    for c, hi in NSL_RANGE_INT.items():
        x[c] = np.round(x[c]).clip(0, hi)
    for c in _nsl_rates(x):
        x[c] = np.round(x[c].clip(0, 1) * 100) / 100
    for a, b in NSL_RATE_PAIRS:
        excess = (x[a] + x[b] - 1).clip(lower=0)
        x[b] = np.round((x[b] - excess) * 100) / 100
    return x


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
CIC_DERIVED = ['Fwd Pkt Len Mean', 'Bwd Pkt Len Mean', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
               'Subflow Fwd Pkts', 'Subflow Bwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Byts',
               'Pkt Len Max', 'Pkt Len Min', 'Pkt Len Mean', 'Pkt Size Avg', 'Pkt Len Var',
               'Down/Up Ratio', 'Flow Byts/s', 'Flow Pkts/s', 'Fwd Pkts/s', 'Bwd Pkts/s']


def mutable_columns(columns: list[str], level: str) -> tuple[list[str], list[str]]:
    """Primary coordinates an attacker may move under a level.

    Returns (mutable, nondecreasing): derived statistics are excluded because the
    repair recomputes them; under omega2 every mutable coordinate may only grow.
    """
    cols = list(columns)
    if 'Flow Duration' in cols:
        if level == 'omega2':
            mut = [c for c in cols if c.startswith('Fwd') or c.startswith('Flow') or c in
                   ('Tot Fwd Pkts', 'TotLen Fwd Pkts', 'Pkt Len Std')]
            mut = [c for c in mut if c not in CIC_DERIVED + CIC_IMMUTABLE + ['Fwd Header Len']]
            return mut, mut
        mut = [c for c in cols if c not in CIC_DERIVED + CIC_IMMUTABLE]
        return mut, []
    if level == 'omega2':
        return list(NSL_CAPABILITY), list(NSL_CAPABILITY)
    return [c for c in cols if c not in NSL_IMMUTABLE], []


def _present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    # Constant features (e.g. Protocol on CSE-CIC-IDS2018) are dropped by the preprocessor.
    return [c for c in cols if c in df.columns]


def _dataset(df: pd.DataFrame) -> str:
    return 'cic2018' if 'Flow Duration' in df.columns else 'nslkdd'


def relations(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return _cic_relations(df) if _dataset(df) == 'cic2018' else _nsl_relations(df)


def repair(adv: pd.DataFrame, orig: pd.DataFrame, level: str = 'omega1', strict: bool = False) -> pd.DataFrame:
    """Return Pi_Omega(adv) for row-aligned adversarial and original records.

    strict=False keeps relation deviations already present in the original record;
    strict=True imposes every relation (absolute consistency).
    """
    if level not in ('omega1', 'omega2'):
        raise ValueError(f"unknown constraint level: {level}")
    if len(adv) != len(orig):
        raise ValueError("adversarial and original records must be row-aligned")
    adv = adv.reset_index(drop=True)
    orig = orig.reset_index(drop=True)
    out = _cic_repair(adv, orig, level, strict) if _dataset(adv) == 'cic2018' else _nsl_repair(adv, orig, level)
    return out[adv.columns]


def violations_absolute(df: pd.DataFrame) -> pd.DataFrame:
    """Per-record violation of every relation, irrespective of any original record."""
    rel = relations(df)
    out = pd.DataFrame({k: ~v for k, v in rel.items()})
    out['any'] = out.any(axis=1)
    return out


def capability_ok(adv: pd.DataFrame, orig: pd.DataFrame) -> np.ndarray:
    """Per-record check of the omega2 attacker-capability model."""
    adv = adv.reset_index(drop=True)
    orig = orig.reset_index(drop=True)
    if _dataset(adv) == 'cic2018':
        fixed = _present(adv, CIC_IMMUTABLE) + CIC_BWD_FIXED
        same = np.isclose(adv[fixed].values, orig[fixed].values).all(axis=1)
        grow = (adv[CIC_FWD_NONDECREASING].values >= orig[CIC_FWD_NONDECREASING].values - 1e-6).all(axis=1)
        return same & grow
    fixed = [c for c in adv.select_dtypes('number').columns if c not in NSL_CAPABILITY]
    same = np.isclose(adv[fixed].values, orig[fixed].values).all(axis=1)
    grow = (adv[NSL_CAPABILITY].values >= orig[NSL_CAPABILITY].values - 1e-6).all(axis=1)
    return same & grow


def audit(adv: pd.DataFrame, orig: pd.DataFrame) -> pd.DataFrame:
    """Per-record violations of relations that the original record satisfies.

    Returns one boolean column per relation (True = violated) plus 'any', 'immutable'
    and 'capability' (omega2 model violated).
    """
    adv = adv.reset_index(drop=True)
    orig = orig.reset_index(drop=True)
    ra, ro = relations(adv), relations(orig)
    out = pd.DataFrame({k: ro[k] & ~ra[k] for k in ra})
    imm = _present(adv, CIC_IMMUTABLE if _dataset(adv) == 'cic2018' else NSL_IMMUTABLE)
    out['immutable'] = ~np.all([(adv[c].values == orig[c].values) for c in imm], axis=0)
    out['any'] = out.any(axis=1)
    out['capability'] = ~capability_ok(adv, orig)
    return out

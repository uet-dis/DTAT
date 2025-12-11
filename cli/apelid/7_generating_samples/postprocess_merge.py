import os
import sys
import argparse
import math
from typing import List, Dict, Tuple

import pandas as pd

from utils.logging import setup_logging, get_logger


logger = get_logger(__name__)

# Optional pretty table for compact logging
try:
    from prettytable import PrettyTable  # type: ignore
    _HAS_PRETTYTABLE = True
except Exception:  # pragma: no cover
    _HAS_PRETTYTABLE = False


# Align path discovery with merge_training_samples.py
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from configs import CIC2018Resources, NSLKDDResources  # noqa: E402
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor  # noqa: E402
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor  # noqa: E402

REGISTRY = {
    'cic2018': (CIC2018Resources, CIC2018Preprocessor),
    'nslkdd': (NSLKDDResources, NSLKDDPreprocessor),
}

MODEL_TYPES = ['xgb', 'dnn', 'catb', 'bagging', 'histgbm', 'gbm']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process merged robust TRAIN CSV to balanced subset per attack and label")
    parser.add_argument('--resource', '-r', type=str, required=True, choices=list(REGISTRY.keys()))
    parser.add_argument('--model', '-m', type=str, required=True, choices=MODEL_TYPES)
    parser.add_argument('--input', '-i', type=str, default=None, help="Path to merged robust TRAIN CSV (default resolves from resource/model)")
    parser.add_argument('--output', '-o', type=str, default=None, help="Path to save the postprocessed CSV (default resolves from resource/model)")
    parser.add_argument('--label-col', type=str, default=None, help="Label column name (default: auto-detect common names)")
    parser.add_argument('--sample-threshold', type=int, required=True, help="Target number of samples per attack after balancing")
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    return args


def autodetect_label_column(df: pd.DataFrame, explicit: str | None, preferred: str | None = None) -> str:
    # 1) explicit arg
    if explicit is not None:
        if explicit not in df.columns:
            raise SystemExit(f"Label column not found: {explicit}")
        return explicit
    # 2) preferred from preprocessor
    if preferred is not None and preferred in df.columns:
        return preferred
    # 3) common candidates
    candidates = ['label', 'y', 'target', 'class', 'Label']
    for c in candidates:
        if c in df.columns:
            return c
    # 4) fallback: last column that is not __attack__
    for col in reversed(list(df.columns)):
        if col != '__attack__':
            return col
    return df.columns[-1]


def explode_attacks(df: pd.DataFrame) -> pd.DataFrame:
    if '__attack__' not in df.columns:
        raise SystemExit("Missing __attack__ column in merged robust CSV")
    col = df.loc[:, '__attack__']
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    col = col.astype(str).str.split('|')
    return df.assign(__attack__=col).explode('__attack__', ignore_index=True)


def allocate_label_quotas(labels: List, total: int) -> Dict:
    # Evenly distribute 'total' across labels deterministically
    k = len(labels)
    if k == 0:
        return {}
    base = total // k
    rem = total % k
    quotas = {lbl: base for lbl in sorted(labels, key=lambda x: str(x))}
    # Give one extra to the first 'rem' labels
    for lbl in list(quotas.keys())[:rem]:
        quotas[lbl] += 1
    return quotas


def sample_balanced_per_label(df: pd.DataFrame, label_col: str, total_target: int, random_state: int) -> pd.DataFrame:
    labels = sorted(df[label_col].dropna().unique().tolist(), key=lambda x: str(x))
    quotas = allocate_label_quotas(labels, total_target)
    chunks: List[pd.DataFrame] = []
    for lbl, g in df.groupby(label_col):
        want = quotas.get(lbl, 0)
        if want <= 0:
            continue
        n = len(g)
        if n <= want:
            chunks.append(g)
        else:
            chunks.append(g.sample(want, random_state=random_state))
    return pd.concat(chunks, axis=0, ignore_index=True) if chunks else df.iloc[0:0]


def _update_label_counts(df: pd.DataFrame, label_col: str, counts: Dict) -> None:
    vc = df[label_col].value_counts()
    for lbl, cnt in vc.items():
        counts[lbl] = counts.get(lbl, 0) + int(cnt)


def sample_balanced_per_label_capped(
    df: pd.DataFrame,
    label_col: str,
    total_target: int,
    per_label_cap: int,
    current_counts: Dict,
    random_state: int,
) -> pd.DataFrame:
    # Balanced quotas but each label limited by per_label_cap - current_counts[label]
    labels = sorted(df[label_col].dropna().unique().tolist(), key=lambda x: str(x))
    if not labels or total_target <= 0:
        return df.iloc[0:0]
    quotas = allocate_label_quotas(labels, total_target)
    chunks: List[pd.DataFrame] = []
    for lbl in labels:
        g = df[df[label_col] == lbl]
        base_quota = quotas.get(lbl, 0)
        allowed_extra = max(0, per_label_cap - int(current_counts.get(lbl, 0)))
        take = min(base_quota, allowed_extra, len(g))
        if take <= 0:
            continue
        if len(g) <= take:
            sel = g
        else:
            sel = g.sample(take, random_state=random_state)
        chunks.append(sel)
    return pd.concat(chunks, axis=0, ignore_index=True) if chunks else df.iloc[0:0]


def sample_with_per_label_cap(df: pd.DataFrame, label_col: str, current_counts: Dict, per_label_cap: int,
                              total_target: int, random_state: int) -> pd.DataFrame:
    # Deterministic per-label capped sampling up to total_target
    if total_target <= 0 or df.empty:
        return df.iloc[0:0]
    need = total_target
    parts: List[pd.DataFrame] = []
    for lbl in sorted(df[label_col].dropna().unique().tolist(), key=lambda x: str(x)):
        if need <= 0:
            break
        g = df[df[label_col] == lbl]
        allowed = max(0, per_label_cap - int(current_counts.get(lbl, 0)))
        if allowed <= 0:
            continue
        take = min(len(g), allowed, need)
        if take <= 0:
            continue
        if len(g) <= take:
            sel = g
        else:
            sel = g.sample(take, random_state=random_state)
        parts.append(sel)
        need -= len(sel)
        # update local counts to reflect reserved rows (caller should update global counts with actual selection)
        current_counts[lbl] = current_counts.get(lbl, 0) + len(sel)
    return pd.concat(parts, axis=0, ignore_index=True) if parts else df.iloc[0:0]


def main():
    args = parse_args()
    setup_logging(args.log_level)

    # Resolve defaults based on resource/model
    ResClass, PreprocessorClass = REGISTRY[args.resource]
    res = ResClass
    if args.input is None:
        robust_dir = os.path.join(res.DATA_FOLDER, 'adv_samples', 'robust')
        args.input = os.path.join(robust_dir, f"{res.resources_name}_{args.model}_robust_train.csv")
    if args.output is None:
        robust_dir = os.path.join(res.DATA_FOLDER, 'adv_samples', 'robust')
        args.output = os.path.join(robust_dir, f"{res.resources_name}_{args.model}_robust_train_post.csv")

    if not os.path.exists(args.input):
        raise SystemExit(f"Input merged CSV not found: {args.input}")
    logger.info(f"[+] Loading merged robust CSV: {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    pre = PreprocessorClass()
    preferred_label = getattr(pre, 'label_column', None)
    label_col = autodetect_label_column(df, args.label_col, preferred=preferred_label)
    if label_col == '__attack__':
        # guard against bad detection
        label_col = autodetect_label_column(df, None, preferred=None)
    logger.info(f"[+] Using label column: {label_col}")

    # Build unique-key to detect overlapping vs unique samples across attacks
    feature_cols = [c for c in df.columns if c not in ['__attack__', label_col]]
    if not feature_cols:
        raise SystemExit("No feature columns found (only __attack__ and label present)")
    df['_key_'] = pd.util.hash_pandas_object(df[feature_cols], index=False).astype('int64')

    # Prepare per-attack dataframes of unique-only and overlap-only
    # unique keys appear exactly once across exploded rows
    # Avoid duplicate column names when slicing
    base_cols = ['__attack__', label_col, '_key_']
    base_cols_unique = []
    seen = set()
    for c in base_cols:
        if c not in seen:
            base_cols_unique.append(c)
            seen.add(c)
    exploded = explode_attacks(df[base_cols_unique].copy())
    key_counts = exploded['_key_'].value_counts()
    unique_keys = set(key_counts[key_counts == 1].index.tolist())
    overlap_keys = set(key_counts[key_counts > 1].index.tolist())

    # For deterministic sampling
    rng_seed = 1337

    attacks = sorted(exploded['__attack__'].dropna().unique().tolist())
    if not attacks:
        raise SystemExit("No attacks found in __attack__ column")
    num_labels = df[label_col].nunique(dropna=False)
    logger.info(f"[+] Attacks: {attacks} | Labels: {num_labels} | per-attack target: {args.sample_threshold}")

    selected_rows: List[pd.DataFrame] = []

    selected_keys = set()
    for atk in attacks:
        need = args.sample_threshold
        picked_parts: List[pd.DataFrame] = []

        # All rows for this attack excluding already selected keys
        atk_rows = df[
            df['__attack__'].astype(str).str.contains(fr'(^|\|){atk}(\||$)')
            & (~df['_key_'].isin(selected_keys))
        ].copy()

        # Prepare per-label cap: at most ceil(threshold / num_labels * 2) per label within this attack
        per_label_cap = int(math.ceil(float(args.sample_threshold) / float(max(1, num_labels)) * 2.0))
        # Track current label counts already picked in this attack
        current_counts: Dict = {}

        # Phase 1: unique-only for this attack, balanced per label with early cap
        atk_unique = atk_rows[atk_rows['_key_'].isin(unique_keys)].copy()
        logger.debug(f"[+] Attack={atk} unique rows (available): {len(atk_unique)}")
        if len(atk_unique) > 0 and need > 0:
            take_balanced = min(need, len(atk_unique))
            picked1 = sample_balanced_per_label_capped(
                atk_unique, label_col, take_balanced, per_label_cap, current_counts, rng_seed
            )
            picked_parts.append(picked1)
            selected_keys.update(picked1['_key_'].tolist())
            need -= len(picked1)
            if len(picked1) > 0:
                _update_label_counts(picked1, label_col, current_counts)
        logger.debug(f"[+] Attack={atk} phase1 picked: {sum(len(p) for p in picked_parts)} | remain {need}")

        # Phase 2: remaining unique-only random but respect per-label cap
        if need > 0:
            leftover_unique = atk_unique[~atk_unique['_key_'].isin(selected_keys)]
            if len(leftover_unique) > 0:
                add = sample_with_per_label_cap(leftover_unique, label_col, current_counts, per_label_cap, need, rng_seed)
                if len(add) > 0:
                    picked_parts.append(add)
                    selected_keys.update(add['_key_'].tolist())
                    need -= len(add)
        logger.debug(f"[+] Attack={atk} phase2 picked: {sum(len(p) for p in picked_parts)} | remain {need}")

        # Phase 3: overlap rows random, still respect per-label cap
        if need > 0:
            atk_overlap = atk_rows[atk_rows['_key_'].isin(overlap_keys) & (~atk_rows['_key_'].isin(selected_keys))]
            if len(atk_overlap) > 0:
                add = sample_with_per_label_cap(atk_overlap, label_col, current_counts, per_label_cap, need, rng_seed)
                if len(add) > 0:
                    picked_parts.append(add)
                    selected_keys.update(add['_key_'].tolist())
                    need -= len(add)
        picked = pd.concat(picked_parts, axis=0, ignore_index=True) if picked_parts else atk_rows.iloc[0:0]
        # Reassign attack label to the chosen attack name
        if len(picked) > 0:
            picked.loc[:, '__attack__'] = atk
        logger.info(f"[+] Attack={atk} final picked: {len(picked)} / target {args.sample_threshold}")
        selected_rows.append(picked)

    # Combine; rows are unique by key by construction
    result = pd.concat(selected_rows, axis=0, ignore_index=True)
    if '_key_' in result.columns:
        result = result.drop(columns=['_key_'])

    # Log label x attack distribution (labels as rows, attacks as columns); no duplicate stats here
    try:
        df_attack_label = (
            result.assign(__attack__=result['__attack__'].astype(str).str.split('|'))
            .explode('__attack__', ignore_index=True)
            .groupby(['__attack__', label_col])
            .size()
            .rename('count')
            .reset_index()
        )
        df_pivot = (
            df_attack_label
            .pivot_table(index=label_col, columns='__attack__', values='count', aggfunc='sum', fill_value=0)
            .sort_index()
        )
        attacks = sorted(df_pivot.columns, key=lambda x: str(x))
        df_pivot = df_pivot[attacks]
        df_with_total = df_pivot.copy()
        df_with_total['total'] = df_with_total.sum(axis=1)
        if _HAS_PRETTYTABLE:
            headers = ['label'] + [str(a) for a in attacks] + ['total']
            tbl = PrettyTable(field_names=headers)
            for lbl, row in df_with_total.iterrows():
                row_vals = [int(row[a]) for a in attacks] + [int(row['total'])]
                tbl.add_row([lbl] + row_vals)
            # Add 'all' row: sums per attack and grand total
            col_sums = df_with_total[attacks].sum(axis=0)
            all_total = int(df_with_total['total'].sum())
            tbl.add_row(['all'] + [int(col_sums.get(a, 0)) for a in attacks] + [all_total])
            logger.info(f"[+] Label x Attack counts (postprocess):\n{tbl}")
        else:
            tmp = df_with_total.copy()
            tmp.index.name = 'label'
            # Append 'all' row
            col_sums = tmp[attacks].sum(axis=0)
            all_row = pd.Series({a: int(col_sums.get(a, 0)) for a in attacks}, name='all')
            all_row['total'] = int(tmp['total'].sum())
            tmp = pd.concat([tmp, all_row.to_frame().T[tmp.columns]], axis=0)
            logger.info("[+] Label x Attack counts (postprocess):\n" + tmp.to_string())
    except Exception as e:
        logger.warning(f"[!] Failed to compute label distribution per attack (postprocess): {e}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.to_csv(args.output, index=False)
    logger.info(f"[+] Saved postprocessed CSV: {args.output} | rows={len(result)}")


if __name__ == "__main__":
    main()


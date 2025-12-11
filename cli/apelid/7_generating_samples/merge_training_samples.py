import os
import sys
import argparse
from typing import List, Tuple

import pandas as pd

from utils.logging import setup_logging, get_logger

# Optional pretty table for compact logging
try:
    from prettytable import PrettyTable  # type: ignore
    _HAS_PRETTYTABLE = True
except Exception:  # pragma: no cover
    _HAS_PRETTYTABLE = False


logger = get_logger(__name__)


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


def _default_inputs(res) -> tuple[str, str]:
    data_dir = res.BALENCED_DATA_FOLDER
    train_path = os.path.join(data_dir, f"{res.resources_name}_merged_train_raw_processed.csv")
    test_path = os.path.join(data_dir, f"{res.resources_name}_test_random_sample_clean_merged.csv")
    return train_path, test_path


def _default_adv_train_dir(res) -> str:
    return os.path.join(res.DATA_FOLDER, 'adv_samples', 'train')


def _parse_attack_from_filename(filename: str, resource_name: str, model_type: str) -> str:
    stem = os.path.basename(filename)
    prefix = f"{resource_name}_{model_type}_"
    suffix = "_adv.csv"
    if stem.startswith(prefix) and stem.endswith(suffix):
        return stem[len(prefix):-len(suffix)]
    # Fallback to filename without extension
    return os.path.splitext(stem)[0]


def _discover_attack_files(base_dir: str, resource_name: str, model_type: str) -> List[Tuple[str, str]]:
    if not os.path.isdir(base_dir):
        return []
    results: List[Tuple[str, str]] = []
    for fname in os.listdir(base_dir):
        if not fname.endswith('_adv.csv'):
            continue
        if not fname.startswith(f"{resource_name}_{model_type}_"):
            continue
        attack = _parse_attack_from_filename(fname, resource_name, model_type)
        results.append((os.path.join(base_dir, fname), attack))
    return results


def _map_attack_to_path(base_dir: str, resource_name: str, model_type: str) -> dict:
    mapping = {}
    if not os.path.isdir(base_dir):
        return mapping
    prefix = f"{resource_name}_{model_type}_"
    for fname in os.listdir(base_dir):
        if not fname.endswith('_adv.csv'):
            continue
        if not fname.startswith(prefix):
            continue
        attack = _parse_attack_from_filename(fname, resource_name, model_type)
        mapping[attack] = os.path.join(base_dir, fname)
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Merge adversarial TRAIN samples across attacks for robust training")
    parser.add_argument('--resource', '-r', type=str, required=True, choices=list(REGISTRY.keys()))
    parser.add_argument('--model', '-m', type=str, required=True, choices=MODEL_TYPES, help="Model type for which to merge")
    parser.add_argument('--attack', '-a', type=str, nargs='+', default=None,
                        help="Specific attack names to include (if omitted, auto-discover)")
    parser.add_argument('--adv-in', type=str, nargs='+', default=None,
                        help="Explicit adversarial TRAIN CSV paths; overrides --attack and auto-discovery")
    parser.add_argument('--plain-train', type=str, default=None,
                        help="Plain training CSV for dedup (default: dataset default train)")
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Directory to save merged robust CSV (default: <data>/adv_samples/robust)")
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Resource and preprocessor
    ResClass, PreprocessorClass = REGISTRY[args.resource]
    res = ResClass
    pre = PreprocessorClass()
    label_col = pre.label_column

    # Defaults
    if args.output_dir is None:
        args.output_dir = os.path.join(res.DATA_FOLDER, 'adv_samples', 'robust')

    # Resolve plain training CSV
    if args.plain_train is None:
        default_train, _ = _default_inputs(res)
        args.plain_train = default_train

    if not os.path.exists(args.plain_train):
        raise SystemExit(f"Plain training CSV not found: {args.plain_train}")
    logger.info(f"[+] Plain training CSV: {args.plain_train}")

    # Load plain training set and build feature keys
    df_plain = pd.read_csv(args.plain_train, low_memory=False)
    df_plain = pre.select_features_and_label(df_plain)
    # Shape of df_plain
    logger.info(f"[+] Shape of plain training set: {df_plain.shape}")
    if label_col not in df_plain.columns:
        raise SystemExit(f"Label column '{label_col}' not found in plain training CSV")
    feature_cols = [c for c in df_plain.columns if c != label_col]
    base_keys = set(map(tuple, df_plain[feature_cols].values)) if not df_plain.empty else set()
    logger.info(f"[+] Plain training rows: {len(df_plain)} | Feature columns: {len(feature_cols)}")

    # Resolve adversarial TRAIN files to merge
    adv_files: List[Tuple[str, str]] = []  # (path, attack)
    train_dir = _default_adv_train_dir(res)

    if args.adv_in:
        for p in args.adv_in:
            if not os.path.exists(p):
                logger.warning(f"[!] Adversarial TRAIN CSV not found: {p}; skipping")
                continue
            attack = _parse_attack_from_filename(p, res.resources_name, args.model)
            adv_files.append((p, attack))
    elif args.attack:
        for atk in args.attack:
            fname_model = f"{res.resources_name}_{args.model}_{atk}_adv.csv"
            p_model = os.path.join(train_dir, fname_model)
            if os.path.exists(p_model):
                adv_files.append((p_model, atk))
                continue
            # Fallback to DNN if model-specific file is missing
            if args.model != 'dnn':
                fname_dnn = f"{res.resources_name}_dnn_{atk}_adv.csv"
                p_dnn = os.path.join(train_dir, fname_dnn)
                if os.path.exists(p_dnn):
                    logger.debug(f"[+] Fallback to DNN attack file for atk={atk}: {p_dnn}")
                    adv_files.append((p_dnn, atk))
                    continue
            logger.warning(f"[!] Expected adversarial TRAIN CSV missing: {p_model}; skipping")
    else:
        # Auto-discover with fallback: use model-specific files if present; otherwise use DNN's
        model_map = _map_attack_to_path(train_dir, res.resources_name, args.model)
        dnn_map = {} if args.model == 'dnn' else _map_attack_to_path(train_dir, res.resources_name, 'dnn')
        union_attacks = set(model_map.keys()) | set(dnn_map.keys())
        adv_files = []
        for atk in sorted(union_attacks):
            path = model_map.get(atk) or dnn_map.get(atk)
            if path:
                adv_files.append((path, atk))

    if not adv_files:
        raise SystemExit("No adversarial TRAIN CSVs found to merge. Provide --adv-in/--attack or generate them first.")

    logger.info(f"[+] Found {len(adv_files)} adversarial TRAIN file(s) to merge for model={args.model}")

    # Load and concatenate, adding __attack__
    merged_list: List[pd.DataFrame] = []
    # Collect per-file counts to aggregate into 2D tables
    # - pre_internal_rows: counts BEFORE internal dedup within each file
    # - pre_counts_rows: counts AFTER internal dedup but BEFORE merge
    pre_internal_rows: List[Tuple[str, object, int]] = []
    pre_counts_rows: List[Tuple[str, object, int]] = []
    for path, attack in adv_files:
        try:
            df = pd.read_csv(path, low_memory=False)
            df = pre.select_features_and_label(df)
            if label_col not in df.columns:
                logger.warning(f"[!] Missing label column in {path}; skipping")
                continue
            # Keep only known feature columns + label if present
            missing_feats = [c for c in feature_cols if c not in df.columns]
            if missing_feats:
                logger.warning(f"[!] Missing expected feature columns in {path}: {missing_feats[:5]}{'...' if len(missing_feats)>5 else ''}; skipping")
                continue
            df = df[feature_cols + [label_col]].copy()
            # Accumulate per-file label counts BEFORE internal dedup
            try:
                counts_before = df[label_col].value_counts(dropna=False)
                for lbl, cnt in counts_before.items():
                    pre_internal_rows.append((attack, lbl, int(cnt)))
            except Exception as e:
                logger.warning(f"[!] Failed to collect pre-internal counts for attack={attack}: {e}")
            # Internal dedup by features within this attack file
            _before = len(df)
            df = df.drop_duplicates(subset=feature_cols, ignore_index=True)
            _after = len(df)
            if _after < _before:
                logger.debug(f"[+] Internal dedup for attack={attack}: {_before} -> {_after}")
            df['__attack__'] = attack
            merged_list.append(df)
            logger.debug(f"[+] Loaded {path} | rows={len(df)} | attack={attack}")
            # Accumulate per-file label counts AFTER internal dedup for a combined 2D table later
            try:
                counts = df[label_col].value_counts(dropna=False)
                for lbl, cnt in counts.items():
                    pre_counts_rows.append((attack, lbl, int(cnt)))
            except Exception as e:
                logger.warning(f"[!] Failed to collect per-file counts for attack={attack}: {e}")
        except Exception as e:
            logger.warning(f"[!] Failed to load {path}: {e}")

    if not merged_list:
        raise SystemExit("No valid adversarial TRAIN CSVs to merge after filtering.")

    # Combined two-dimensional table across attacks BEFORE internal dedup: attack x label (counts only)
    try:
        if pre_internal_rows:
            df_pre_internal = pd.DataFrame(pre_internal_rows, columns=['attack', label_col, 'count'])
            df_pre_internal_pivot = (
                df_pre_internal
                .pivot_table(index='attack', columns=label_col, values='count', aggfunc='sum', fill_value=0)
                .sort_index()
            )
            df_pre_internal_pivot = df_pre_internal_pivot.reindex(sorted(df_pre_internal_pivot.columns, key=lambda x: str(x)), axis=1)
            df_pre_internal_pivot['total'] = df_pre_internal_pivot.sum(axis=1)
            # Chunk label columns into groups of 6 for display
            label_cols = [c for c in df_pre_internal_pivot.columns if c != 'total']
            chunks = [label_cols[i:i+6] for i in range(0, len(label_cols), 6)] or [[]]
            for idx, chunk in enumerate(chunks, start=1):
                if _HAS_PRETTYTABLE:
                    headers = ['attack'] + [str(c) for c in (chunk + ['total'])]
                    tbl = PrettyTable(field_names=headers)
                    for atk, row in df_pre_internal_pivot.iterrows():
                        vals = [int(row[c]) for c in chunk] + [int(row['total'])]
                        tbl.add_row([atk] + vals)
                    # Add 'all' row from column sums
                    col_sums = df_pre_internal_pivot[chunk].sum(axis=0)
                    all_total = int(df_pre_internal_pivot['total'].sum())
                    tbl.add_row(['all'] + [int(col_sums.get(c, 0)) for c in chunk] + [all_total])
                    logger.info(f"[+] Attack x Label counts across attacks (pre-internal-dedup) - part {idx}:\n{tbl}")
                else:
                    tmp = df_pre_internal_pivot[chunk + ['total']].copy()
                    tmp.index.name = 'attack'
                    col_sums = tmp[chunk].sum(axis=0)
                    all_row = pd.Series({c: int(col_sums.get(c, 0)) for c in chunk}, name='all')
                    all_row['total'] = int(tmp['total'].sum())
                    tmp = pd.concat([tmp, all_row.to_frame().T[tmp.columns]], axis=0)
                    logger.info(f"[+] Attack x Label counts across attacks (pre-internal-dedup) - part {idx}:\n" + tmp.to_string())
    except Exception as e:
        logger.warning(f"[!] Failed to compute pre-internal-dedup combined attack x label table: {e}")

    # Combined two-dimensional table across attacks BEFORE aggregation/dedup: attack x label (counts only)
    try:
        if pre_counts_rows:
            df_pre_counts = pd.DataFrame(pre_counts_rows, columns=['attack', label_col, 'count'])
            df_pre_pivot = (
                df_pre_counts
                .pivot_table(index='attack', columns=label_col, values='count', aggfunc='sum', fill_value=0)
                .sort_index()
            )
        df_pre_pivot = df_pre_pivot.reindex(sorted(df_pre_pivot.columns, key=lambda x: str(x)), axis=1)
        df_pre_pivot['total'] = df_pre_pivot.sum(axis=1)
            # Chunk label columns (exclude 'total') into groups of 6 for display
        label_cols = [c for c in df_pre_pivot.columns if c != 'total']
        chunks = [label_cols[i:i+6] for i in range(0, len(label_cols), 6)] or [[]]
        for idx, chunk in enumerate(chunks, start=1):
            if _HAS_PRETTYTABLE:
                headers = ['attack'] + [str(c) for c in (chunk + ['total'])]
                tbl = PrettyTable(field_names=headers)
                for atk, row in df_pre_pivot.iterrows():
                    vals = [int(row[c]) for c in chunk] + [int(row['total'])]
                    tbl.add_row([atk] + vals)
                # Add 'all' row from column sums
                col_sums = df_pre_pivot[chunk].sum(axis=0)
                all_total = int(df_pre_pivot['total'].sum())
                tbl.add_row(['all'] + [int(col_sums.get(c, 0)) for c in chunk] + [all_total])
                logger.info(f"[+] Attack x Label counts across attacks (pre-merge) - part {idx}:\n{tbl}")
            else:
                tmp = df_pre_pivot[chunk + ['total']].copy()
                tmp.index.name = 'attack'
                col_sums = tmp[chunk].sum(axis=0)
                all_row = pd.Series({c: int(col_sums.get(c, 0)) for c in chunk}, name='all')
                all_row['total'] = int(tmp['total'].sum())
                tmp = pd.concat([tmp, all_row.to_frame().T[tmp.columns]], axis=0)
                logger.info(f"[+] Attack x Label counts across attacks (pre-merge) - part {idx}:\n" + tmp.to_string())
    except Exception as e:
        logger.warning(f"[!] Failed to compute pre-dedup combined attack x label table: {e}")

    df_adv_all = pd.concat(merged_list, axis=0, ignore_index=True)
    logger.info(f"[+] Concatenated adversarial rows (pre-dedup): {len(df_adv_all)}")

    # 1) Dedup across attacks by feature values and preserve all attack sources
    #    - Keep first label
    #    - Join unique attack names with '|'
    def _join_unique_attacks(s: pd.Series) -> str:
        return '|'.join(sorted(pd.unique(s.astype(str))))

    df_adv_all = (
        df_adv_all
        .groupby(feature_cols, as_index=False)
        .agg({label_col: 'first', '__attack__': _join_unique_attacks})
    )
    logger.info(f"[+] After cross-attack aggregation by features: {len(df_adv_all)}")

    # 2D table BEFORE removing duplicates with plain: attack x label (counts only)
    try:
        df_before_plain = (
            df_adv_all.assign(__attack__=df_adv_all['__attack__'].astype(str).str.split('|'))
            .explode('__attack__', ignore_index=True)
        )
        df_before_counts = (
            df_before_plain.groupby(['__attack__', label_col])
            .size()
            .rename('count')
            .reset_index()
        )
        df_before_pivot = (
            df_before_counts
            .pivot_table(index='__attack__', columns=label_col, values='count', aggfunc='sum', fill_value=0)
            .sort_index()
        )
        df_before_pivot = df_before_pivot.reindex(sorted(df_before_pivot.columns, key=lambda x: str(x)), axis=1)
        df_before_pivot['total'] = df_before_pivot.sum(axis=1)
        # Compute cross-attack duplicate counts per attack (columns dup-<attack>)
        try:
            unique_attacks = sorted({atk for s in df_adv_all['__attack__'].astype(str) for atk in s.split('|') if atk})
            from collections import defaultdict
            pair_counts = defaultdict(int)
            for s in df_adv_all['__attack__'].astype(str):
                parts = [p for p in s.split('|') if p]
                S = set(parts)
                for a in S:
                    for b in S:
                        if a == b:
                            continue
                        pair_counts[(a, b)] += 1
            dup_cols = [f"dup-{b}" for b in unique_attacks]
            dup_df = pd.DataFrame(0, index=unique_attacks, columns=dup_cols)
            for (a, b), v in pair_counts.items():
                col = f"dup-{b}"
                if col in dup_df.columns and a in dup_df.index:
                    dup_df.loc[a, col] = int(v)
            dup_df = dup_df.reindex(df_before_pivot.index).fillna(0).astype(int)
            df_before_pivot = pd.concat([df_before_pivot, dup_df], axis=1)
        except Exception as e:
            logger.warning(f"[!] Failed to compute cross-attack duplicate columns (before-plain-dedup): {e}")
        label_cols = [c for c in df_before_pivot.columns if not str(c).startswith('dup-') and c != 'total']
        dup_cols = [c for c in df_before_pivot.columns if str(c).startswith('dup-')]
        if _HAS_PRETTYTABLE:
            # Table 1: labels + total (chunked by 6 labels)
            chunks = [label_cols[i:i+6] for i in range(0, len(label_cols), 6)] or [[]]
            for idx, chunk in enumerate(chunks, start=1):
                headers1 = ['attack'] + [str(c) for c in (chunk + ['total'])]
                tbl1 = PrettyTable(field_names=headers1)
                for atk, row in df_before_pivot.iterrows():
                    row_vals = [int(row[c]) for c in (chunk + ['total'])]
                    tbl1.add_row([atk] + row_vals)
                # Add 'all' row from column sums
                col_sums = df_before_pivot[chunk].sum(axis=0)
                all_total = int(df_before_pivot['total'].sum())
                tbl1.add_row(['all'] + [int(col_sums.get(c, 0)) for c in chunk] + [all_total])
                logger.info(f"[+] Attack x Label counts across attacks (before-plain-dedup) - labels part {idx}:\n{tbl1}")
            # Table 2: dup-* columns
            if dup_cols:
                headers2 = ['attack'] + dup_cols
                tbl2 = PrettyTable(field_names=headers2)
                for atk, row in df_before_pivot.iterrows():
                    row_vals = [int(row[c]) for c in dup_cols]
                    tbl2.add_row([atk] + row_vals)
                tbl2.add_row(['all'] + ['--' for _ in dup_cols])
                logger.info(f"[+] Attack x Attack duplicate counts (before-plain-dedup) - dup:\n{tbl2}")
        else:
            # DataFrame fallback - Table 1: labels + total (chunked)
            chunks = [label_cols[i:i+6] for i in range(0, len(label_cols), 6)] or [[]]
            for idx, chunk in enumerate(chunks, start=1):
                tmp_labels = df_before_pivot[chunk + ['total']].copy()
                tmp_labels.index.name = 'attack'
                col_sums = tmp_labels[chunk].sum(axis=0)
                all_row = pd.Series({c: int(col_sums.get(c, 0)) for c in chunk}, name='all')
                all_row['total'] = int(tmp_labels['total'].sum())
                tmp_labels = pd.concat([tmp_labels, all_row.to_frame().T[tmp_labels.columns]], axis=0)
                logger.info(f"[+] Attack x Label counts across attacks (before-plain-dedup) - labels part {idx}:\n" + tmp_labels.to_string())
            # Table 2: dup-* columns
            if dup_cols:
                tmp_dup = df_before_pivot[dup_cols].copy()
                tmp_dup.loc['all'] = ['--' for _ in dup_cols]
                tmp_dup.index.name = 'attack'
                logger.info("[+] Attack x Attack duplicate counts (before-plain-dedup) - dup:\n" + tmp_dup.to_string())
    except Exception as e:
        logger.warning(f"[!] Failed to compute before-plain-dedup attack x label table: {e}")

    # 2) Dedup against plain training by feature values
    if base_keys:
        keep_mask = [tuple(row) not in base_keys for row in df_adv_all[feature_cols].values]
        df_adv_all = df_adv_all.loc[keep_mask].reset_index(drop=True)
        logger.info(f"[+] After removing duplicates against plain training: {len(df_adv_all)}")

    # Output
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f"{res.resources_name}_{args.model}_robust_train.csv")
    # Label distribution per individual attack (explode pipe-separated sources)
    try:
        df_attack_label = (
            df_adv_all.assign(__attack__=df_adv_all['__attack__'].astype(str).str.split('|'))
            .explode('__attack__', ignore_index=True)
            .groupby(['__attack__', label_col])
            .size()
            .rename('count')
            .reset_index()
        )
        # Combined two-dimensional table across attacks: attack x label (counts only)
        df_pivot = (
            df_attack_label
            .pivot_table(index='__attack__', columns=label_col, values='count', aggfunc='sum', fill_value=0)
            .sort_index()
        )
        df_pivot = df_pivot.reindex(sorted(df_pivot.columns, key=lambda x: str(x)), axis=1)
        df_pivot['total'] = df_pivot.sum(axis=1)
        # Compute cross-attack duplicate counts per attack (columns dup-<attack>) on post-dedup set
        try:
            unique_attacks_post = sorted({atk for s in df_adv_all['__attack__'].astype(str) for atk in s.split('|') if atk})
            from collections import defaultdict
            pair_counts_post = defaultdict(int)
            for s in df_adv_all['__attack__'].astype(str):
                parts = [p for p in s.split('|') if p]
                S = set(parts)
                for a in S:
                    for b in S:
                        if a == b:
                            continue
                        pair_counts_post[(a, b)] += 1
            dup_cols_post = [f"dup-{b}" for b in unique_attacks_post]
            dup_df_post = pd.DataFrame(0, index=unique_attacks_post, columns=dup_cols_post)
            for (a, b), v in pair_counts_post.items():
                col = f"dup-{b}"
                if col in dup_df_post.columns and a in dup_df_post.index:
                    dup_df_post.loc[a, col] = int(v)
            dup_df_post = dup_df_post.reindex(df_pivot.index).fillna(0).astype(int)
            df_pivot = pd.concat([df_pivot, dup_df_post], axis=1)
        except Exception as e:
            logger.warning(f"[!] Failed to compute cross-attack duplicate columns (post-dedup): {e}")
        label_cols = [c for c in df_pivot.columns if not str(c).startswith('dup-') and c != 'total']
        dup_cols = [c for c in df_pivot.columns if str(c).startswith('dup-')]
        if _HAS_PRETTYTABLE:
            # Table 1: labels + total (chunked by 6 labels)
            chunks = [label_cols[i:i+6] for i in range(0, len(label_cols), 6)] or [[]]
            for idx, chunk in enumerate(chunks, start=1):
                headers1 = ['attack'] + [str(c) for c in (chunk + ['total'])]
                tbl1 = PrettyTable(field_names=headers1)
                for atk, row in df_pivot.iterrows():
                    row_vals = [int(row[c]) for c in (chunk + ['total'])]
                    tbl1.add_row([atk] + row_vals)
                # Add 'all' row from column sums
                col_sums = df_pivot[chunk].sum(axis=0)
                all_total = int(df_pivot['total'].sum())
                tbl1.add_row(['all'] + [int(col_sums.get(c, 0)) for c in chunk] + [all_total])
                logger.info(f"[+] Attack x Label counts across attacks (post-dedup) - labels part {idx}:\n{tbl1}")
            # Table 2: dup-* columns
            if dup_cols:
                headers2 = ['attack'] + dup_cols
                tbl2 = PrettyTable(field_names=headers2)
                for atk, row in df_pivot.iterrows():
                    row_vals = [int(row[c]) for c in dup_cols]
                    tbl2.add_row([atk] + row_vals)
                tbl2.add_row(['all'] + ['--' for _ in dup_cols])
                logger.info(f"[+] Attack x Attack duplicate counts (post-dedup) - dup:\n{tbl2}")
        else:
            # DataFrame fallback - Table 1: labels + total (chunked)
            chunks = [label_cols[i:i+6] for i in range(0, len(label_cols), 6)] or [[]]
            for idx, chunk in enumerate(chunks, start=1):
                tmp_labels = df_pivot[chunk + ['total']].copy()
                tmp_labels.index.name = 'attack'
                col_sums = tmp_labels[chunk].sum(axis=0)
                all_row = pd.Series({c: int(col_sums.get(c, 0)) for c in chunk}, name='all')
                all_row['total'] = int(tmp_labels['total'].sum())
                tmp_labels = pd.concat([tmp_labels, all_row.to_frame().T[tmp_labels.columns]], axis=0)
                logger.info(f"[+] Attack x Label counts across attacks (post-dedup) - labels part {idx}:\n" + tmp_labels.to_string())
            # Table 2: dup-* columns
            if dup_cols:
                tmp_dup = df_pivot[dup_cols].copy()
                tmp_dup.loc['all'] = ['--' for _ in dup_cols]
                tmp_dup.index.name = 'attack'
                logger.info("[+] Attack x Attack duplicate counts (post-dedup) - dup:\n" + tmp_dup.to_string())
    except Exception as e:
        logger.warning(f"[!] Failed to compute label distribution per attack: {e}")
    df_adv_all.to_csv(out_csv, index=False)
    logger.info(f"[+] Saved merged robust TRAIN CSV: {out_csv}")


if __name__ == "__main__":
    main()



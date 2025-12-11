import os
import sys
import argparse
from typing import List, Dict

import pandas as pd

from utils.logging import setup_logging, get_logger


logger = get_logger(__name__)

# Optional pretty table for compact logging
try:
    from prettytable import PrettyTable  # type: ignore
    _HAS_PRETTYTABLE = True
except Exception:  # pragma: no cover
    _HAS_PRETTYTABLE = False


# Reuse registry/discovery style from postprocess_merge.py
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
    parser = argparse.ArgumentParser(description="Report label x attack distribution for postprocessed robust train files")
    parser.add_argument('--resource', '-r', type=str, required=True, choices=list(REGISTRY.keys()))
    parser.add_argument('--model', '-m', type=str, required=True, nargs='+', choices=MODEL_TYPES,
                        help="One or more model types to include")
    parser.add_argument('--attack', '-a', type=str, nargs='*', default=None,
                        help="Optional list of attacks to include (default: all)")
    parser.add_argument('--label-col', type=str, default=None, help="Label column name (default: auto-detect)")
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    return parser.parse_args()


def autodetect_label_column(df: pd.DataFrame, explicit: str | None, preferred: str | None = None) -> str:
    if explicit is not None:
        if explicit not in df.columns:
            raise SystemExit(f"Label column not found: {explicit}")
        return explicit
    if preferred is not None and preferred in df.columns:
        return preferred
    candidates = ['label', 'y', 'target', 'class', 'Label']
    for c in candidates:
        if c in df.columns:
            return c
    for col in reversed(list(df.columns)):
        if col != '__attack__':
            return col
    return df.columns[-1]


def normalize_attacks_column(df: pd.DataFrame) -> pd.DataFrame:
    # Handle both single-attack entries and multi-valued 'a|b' strings
    if '__attack__' not in df.columns:
        raise SystemExit("Missing __attack__ column in robust file")
    col = df.loc[:, '__attack__']
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    col = col.astype(str).str.split('|')
    return df.assign(__attack__=col).explode('__attack__', ignore_index=True)


def build_label_attack_table(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(['__attack__', label_col])
        .size()
        .rename('count')
        .reset_index()
    )
    pivot = (
        grouped.pivot_table(index=label_col, columns='__attack__', values='count', aggfunc='sum', fill_value=0)
        .sort_index()
    )
    attacks = sorted(pivot.columns, key=lambda x: str(x))
    pivot = pivot[attacks]
    with_total = pivot.copy()
    with_total['total'] = with_total.sum(axis=1)
    return with_total


def pretty_print_table(model: str, table_df: pd.DataFrame) -> None:
    attacks = [c for c in table_df.columns if c != 'total']
    if _HAS_PRETTYTABLE:
        headers = ['label'] + [str(a) for a in attacks] + ['total']
        tbl = PrettyTable(field_names=headers)
        for lbl, row in table_df.iterrows():
            row_vals = [int(row[a]) for a in attacks] + [int(row['total'])]
            tbl.add_row([lbl] + row_vals)
        col_sums = table_df[attacks].sum(axis=0)
        all_total = int(table_df['total'].sum())
        tbl.add_row(['all'] + [int(col_sums.get(a, 0)) for a in attacks] + [all_total])
        logger.info(f"[Model={model}] Label x Attack counts:\n{tbl}")
    else:
        tmp = table_df.copy()
        tmp.index.name = 'label'
        col_sums = tmp[attacks].sum(axis=0)
        all_row = pd.Series({a: int(col_sums.get(a, 0)) for a in attacks}, name='all')
        all_row['total'] = int(tmp['total'].sum())
        tmp = pd.concat([tmp, all_row.to_frame().T[tmp.columns]], axis=0)
        logger.info(f"[Model={model}] Label x Attack counts:\n" + tmp.to_string())


def main():
    args = parse_args()
    setup_logging(args.log_level)

    ResClass, PreprocessorClass = REGISTRY[args.resource]
    res = ResClass

    # Preprocessor preferred label
    pre = PreprocessorClass()
    preferred_label = getattr(pre, 'label_column', None)

    robust_dir = os.path.join(res.DATA_FOLDER, 'adv_samples', 'robust')

    for model in args.model:
        input_path = os.path.join(robust_dir, f"{res.resources_name}_{model}_robust_train_post.csv")
        if not os.path.exists(input_path):
            logger.warning(f"[!] Missing robust post file for model={model}: {input_path}")
            continue
        logger.info(f"[+] Loading: {input_path}")
        df = pd.read_csv(input_path, low_memory=False)

        label_col = autodetect_label_column(df, args.label_col, preferred=preferred_label)
        if label_col == '__attack__':
            label_col = autodetect_label_column(df, None, preferred=None)

        # Normalize attacks; after postprocess it should be single-valued, but support multi anyway
        exploded = normalize_attacks_column(df[[label_col, '__attack__']].copy())

        # Optional attack filter
        if args.attack:
            wanted = set(args.attack)
            exploded = exploded[exploded['__attack__'].astype(str).isin(wanted)]
            if exploded.empty:
                logger.warning(f"[!] No rows left after attack filter for model={model}; skipping")
                continue

        table_df = build_label_attack_table(exploded, label_col)
        pretty_print_table(model, table_df)


if __name__ == "__main__":
    main()





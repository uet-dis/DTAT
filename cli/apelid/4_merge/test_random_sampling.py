import os
import sys
import argparse
import pandas as pd

from utils.logging import setup_logging, get_logger

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from configs import CIC2018Resources, NSLKDDResources


logger = get_logger(__name__)


REGISTRY = {
    'cic2018': CIC2018Resources,
    'nslkdd': NSLKDDResources,
}


def _safe_concat(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    cols = set()
    for d in dfs:
        cols.update(d.columns.tolist())
    ordered = sorted(cols)
    normed = [d.reindex(columns=ordered) for d in dfs]
    return pd.concat(normed, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Randomly sample up to tau rows per label from original clean_merged TEST data (multi-resource)")
    parser.add_argument('--resource', '-r', type=str, required=True, choices=list(REGISTRY.keys()))
    parser.add_argument('--tau', type=int, required=True, help='Max rows per label to sample')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'label'], help='Sample all labels or a provided list')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='Labels to sample when mode=label')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--out', type=str, default=None,
                        help='Output CSV path for the combined sample')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    ResClass = REGISTRY[args.resource]
    res = ResClass

    # Resolve target labels
    if args.mode == 'label':
        if not args.labels:
            raise SystemExit("--labels is required when --mode label")
        # validate
        valid = set(res.MAJORITY_LABELS + res.MINORITY_LABELS)
        invalid = [lb for lb in args.labels if lb not in valid]
        if invalid:
            raise SystemExit(f"Invalid labels for {res.resources_name}: {invalid}")
        target_labels = args.labels
    else:
        target_labels = res.MAJORITY_LABELS + res.MINORITY_LABELS

    logger.info(f"[+] Sampling TEST clean_merged with tau={args.tau} per label | resource={res.resources_name} | labels={target_labels}")

    rng = args.seed
    parts: list[pd.DataFrame] = []
    total_before = 0
    total_after = 0

    for lbl in target_labels:
        p = res.clean_merged_path_for('test', res.get_label_name(lbl), compressed=False)
        if not os.path.exists(p):
            logger.warning(f"[!] Missing test clean_merged for {lbl}: {p} (skip)")
            continue
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as e:
            logger.warning(f"[!] Failed to read {p}: {e} (skip)")
            continue

        n_before = len(df)
        total_before += n_before
        n_take = min(int(args.tau), n_before) if args.tau and args.tau > 0 else n_before
        if n_take < n_before:
            df = df.sample(n=n_take, random_state=rng).reset_index(drop=True)
        parts.append(df)
        total_after += len(df)
        logger.info(f"[+] {lbl}: {n_before} => {len(df)} (cap={args.tau})")

    out_df = _safe_concat(parts)
    if len(out_df) == 0:
        logger.error("[!] No rows sampled; nothing to write")
        return

    if args.out is None:
        out_path = os.path.join(res.DATA_FOLDER, f'{res.resources_name}_test_random_sample_clean_merged_{args.tau}.csv')
    else:
        out_path = args.out

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info(f"[+] Saved combined TEST random sample => {out_path} (rows={len(out_df)})")
    logger.info(f"[+] Totals: before={total_before}, after={total_after}")


if __name__ == "__main__":
    main()



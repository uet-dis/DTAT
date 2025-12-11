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


def _path_minor_train_augmented(res, label_safe: str, strategy: str) -> str:
    return os.path.join(res.RAW_PROCESSED_DATA_FOLDER, 'train', f"{res.resources_name}_{label_safe}_minority_{strategy}_train_augmented_raw_processed.csv")


def main():
    parser = argparse.ArgumentParser(description="Merge datasets into single TRAIN CSV | Major: clean_merged_compressed(train); Minor: train augmented raw_processed (multi-resource)")
    parser.add_argument('--resource', '-r', type=str, required=True, choices=list(REGISTRY.keys()))
    parser.add_argument('--strategy', type=str, default='wgan', choices=['wgan', 'cfm', 'fdm'],
                        help='Augment strategy name used for minor train filenames (default: wgan)')
    parser.add_argument('--train-out', type=str, default=None,
                        help='Output CSV path for merged train (default inside resource DATA_FOLDER)')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    ResClass = REGISTRY[args.resource]
    res = ResClass

    train_out = args.train_out or os.path.join(res.DATA_FOLDER, f"{res.resources_name}_merged_train_raw_processed.csv")

    logger.info("[+] Merging TRAIN only: major clean_merged_compressed + minor augmented raw_processed …")
    train_parts: list[pd.DataFrame] = []

    # Major train: ONLY clean_merged compressed
    for lbl in res.MAJORITY_LABELS:
        safe = res.get_label_name(lbl)
        p_comp = res.clean_merged_path_for('train', safe, compressed=True)
        if os.path.exists(p_comp):
            try:
                logger.debug(f"[+] Loading major train compressed: {p_comp}")
                df = pd.read_csv(p_comp, low_memory=False)
                df = df.drop_duplicates(ignore_index=True)
                df['__source__'] = f"major:{lbl}"
                train_parts.append(df)
                logger.info(f"[+] Major train OK: {lbl} ({len(df)}) | source=compressed")
            except Exception as e:
                logger.warning(f"[!] Skip major train {lbl}: {e}")
        else:
            logger.warning(f"[-] Missing major train compressed: {p_comp}")

    # Minor train (augmented)
    for lbl in res.MINORITY_LABELS:
        safe = res.get_label_name(lbl)
        p = _path_minor_train_augmented(res, safe, args.strategy)
        if os.path.exists(p):
            try:
                logger.debug(f"[+] Loading minor train augmented: {p}")
                df = pd.read_csv(p, low_memory=False)
                df = df.drop_duplicates(ignore_index=True)
                df['__source__'] = f"minor:{lbl}:{args.strategy}"
                train_parts.append(df)
                logger.info(f"[+] Minor train OK: {lbl} ({len(df)})")
            except Exception as e:
                logger.warning(f"[!] Skip minor train {lbl}: {e}")
        else:
            logger.debug(f"[-] Missing minor train: {p}")

    train_merged = _safe_concat(train_parts)
    if len(train_merged) == 0:
        logger.error("[!] No train parts found; nothing to merge")
        return

    # Shuffle for safety (reproducible)
    train_merged = train_merged.sample(frac=1.0, random_state=42).reset_index(drop=True)
    logger.info("[+] Shuffled merged TRAIN (seed=42)")
    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    train_merged.to_csv(train_out, index=False)
    logger.info(f"[+] Saved merged TRAIN -> {train_out} ({len(train_merged)})")


if __name__ == "__main__":
    main()



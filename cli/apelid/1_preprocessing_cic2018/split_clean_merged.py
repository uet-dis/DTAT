import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

import configs.cic2018
from utils.logging import setup_logging, get_logger


logger = get_logger(__name__)


CLEAN_MERGED_DIR = configs.cic2018.CLEAN_MERGED_DATA_FOLDER


def _list_all_label_files(input_dir: str) -> list[str]:
    files: list[str] = []
    for fname in sorted(os.listdir(input_dir)):
        if fname.endswith('_clean_merged.csv') and fname.startswith('cic2018_'):
            files.append(os.path.join(input_dir, fname))
    return files


def _filter_files_by_labels(files: list[str], label_names: list[str] | None) -> list[str]:
    if not label_names:
        return files
    allowed_safe = {configs.cic2018.get_label_name(lbl) for lbl in label_names}
    out: list[str] = []
    for fp in files:
        base = os.path.basename(fp)
        label_safe = base[len('cic2018_'):-len('_clean_merged.csv')]
        if label_safe in allowed_safe:
            out.append(fp)
    return out


def _split_one_file(fp: str, train_dir: str, test_dir: str, test_size: float, random_state: int) -> Tuple[str, int, int, int]:
    base = os.path.basename(fp)
    label_safe = base[len('cic2018_'):-len('_clean_merged.csv')]
    df = pd.read_csv(fp, low_memory=False)
    if len(df) < 2:
        return (label_safe, len(df), 0, 0)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )
    out_train = os.path.join(train_dir, f"cic2018_{label_safe}_train_clean_merged.csv")
    out_test = os.path.join(test_dir, f"cic2018_{label_safe}_test_clean_merged.csv")
    train_df.to_csv(out_train, index=False)
    test_df.to_csv(out_test, index=False)
    return (label_safe, len(df), len(train_df), len(test_df))


def main():
    parser = argparse.ArgumentParser(description="Split per-label clean-merged datasets into train/test to avoid leakage")
    parser.add_argument('--input-dir', type=str, default=CLEAN_MERGED_DIR,
                        help='Directory containing per-label clean merged CSVs (cic2018_<label>_clean_merged.csv)')
    parser.add_argument('--mode', '-m', type=str, default='all', choices=['all', 'label'],
                        help='Split all labels found in input-dir or a provided list of labels')
    parser.add_argument('--labels', '-c', type=str, nargs='+', default=None,
                        help='List of label names to split when mode=label (original names, not safe)')
    parser.add_argument('--exclude-labels', '-x', type=str, nargs='+', default=None,
                        help='Labels to exclude (original names, not safe)')
    parser.add_argument('--test-size', type=float, default=0.30, help='Test fraction')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for shuffling')
    parser.add_argument('--n-jobs', type=int, default=1, help='Parallel jobs across files (labels). 1 = sequential')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")

    all_files = _list_all_label_files(input_dir)
    if not all_files:
        raise SystemExit(f"No per-label clean merged files found in {input_dir}")

    if args.mode == 'label' and not args.labels:
        raise SystemExit("--labels is required when --mode label")

    target_files = _filter_files_by_labels(all_files, args.labels if args.mode == 'label' else None)
    if args.exclude_labels:
        excludes_safe = {configs.cic2018.get_label_name(lbl) for lbl in args.exclude_labels}
        kept: list[str] = []
        for fp in target_files:
            base = os.path.basename(fp)
            lbl_safe = base[len('cic2018_'):-len('_clean_merged.csv')]
            if lbl_safe not in excludes_safe:
                kept.append(fp)
        target_files = kept

    if not target_files:
        raise SystemExit("No files to split after applying filters.")

    logger.info("===========================================")
    logger.info("CLEAN-MERGED PER-LABEL TRAIN/TEST SPLIT STARTED")
    logger.info("===========================================")
    logger.info(f"Input dir: {input_dir}")
    logger.info(f"Files selected: {len(target_files)}")

    # Prepare output subdirectories
    train_dir = os.path.join(input_dir, 'train')
    test_dir = os.path.join(input_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    n_jobs = int(args.n_jobs or 1)
    if n_jobs > 1 and len(target_files) > 1:
        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs, backend='loky')([
                delayed(_split_one_file)(fp, train_dir, test_dir, args.test_size, args.random_state)
                for fp in target_files
            ])
            for label_safe, total, trn, tst in results:
                if total < 2:
                    logger.warning(f"Skip {label_safe}: not enough rows ({total})")
                else:
                    logger.info(f"{label_safe}: total={total} -> train={trn}, test={tst}")
        except Exception as e:
            logger.warning(f"Parallel split failed ({e}); falling back to sequential")
            for fp in target_files:
                try:
                    label_safe, total, trn, tst = _split_one_file(fp, train_dir, test_dir, args.test_size, args.random_state)
                    if total < 2:
                        logger.warning(f"Skip {label_safe}: not enough rows ({total})")
                    else:
                        logger.info(f"{label_safe}: total={total} -> train={trn}, test={tst}")
                except Exception as ex:
                    logger.warning(f"Failed on {os.path.basename(fp)}: {ex}")
    else:
        for fp in target_files:
            try:
                label_safe, total, trn, tst = _split_one_file(fp, train_dir, test_dir, args.test_size, args.random_state)
                if total < 2:
                    logger.warning(f"Skip {label_safe}: not enough rows ({total})")
                else:
                    logger.info(f"{label_safe}: total={total} -> train={trn}, test={tst}")
            except Exception as ex:
                logger.warning(f"Failed on {os.path.basename(fp)}: {ex}")

    logger.info("===========================================")
    logger.info("SPLIT COMPLETED")
    logger.info("===========================================")


if __name__ == "__main__":
    main()



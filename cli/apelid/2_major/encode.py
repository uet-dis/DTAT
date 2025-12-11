import os
import pandas as pd
import numpy as np

from utils.logging import setup_logging, get_logger
from argparsers.baseparser import BaseParser
import sys

from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from configs import CIC2018Resources, NSLKDDResources

logger = get_logger(__name__)


REGISTRY = {
    'cic2018': (CIC2018Resources, CIC2018Preprocessor),
    'nslkdd': (NSLKDDResources, NSLKDDPreprocessor),
}


def deduplicate_dataframe(df: pd.DataFrame, stage_name: str = "",
                          preprocessor=None) -> pd.DataFrame:
    logger.info(f"[+] Deduplicating {stage_name} data...")
    if preprocessor is None:
        return df.drop_duplicates()

    tech_cols = [c for c in ['__rowid__', '__subset__'] if c in df.columns]
    df_key = df.drop(columns=tech_cols) if tech_cols else df
    try:
        has_duplicates = preprocessor.check_duplicates(df_key)
    except Exception:
        has_duplicates = df_key.duplicated().any()
    if has_duplicates:
        before_dedup = len(df)
        df_clean = df_key.drop_duplicates().copy()
        for c in tech_cols:
            if c in df.columns and c not in df_clean.columns:
                df_clean[c] = df.loc[df_clean.index, c]
        logger.info(f"[+] {stage_name} deduplication: {before_dedup - len(df_clean)} duplicates removed")
        logger.info(f"[+] Final {stage_name} shape: {df_clean.shape}")
        return df_clean
    else:
        logger.info(f"[+] No duplicates found in {stage_name} data")
        return df


def main():
    parser = BaseParser(description="Encode per-label datasets for MAJORITY classes for multiple resources")
    parser.add_argument('--resource', type=str, default='cic2018', choices=list(REGISTRY.keys()))
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Input directory containing per-label clean merged CSVs')
    parser.add_argument('--num-encoder', '-n', type=str, required=True, choices=['standard'],
                        help='Numerical encoder to use')
    parser.add_argument('--subset', type=str, default='full', choices=['full', 'train', 'test'],
                        help="Which subset of clean-merged to encode: full (unsplit), train or test")
    parser.add_argument('--mode', '-m', type=str, default='all', choices=['all', 'label'],
                        help='Encode all majority labels or a provided list')
    parser.add_argument('--labels', '-c', type=str, nargs='+', default=None,
                        help='List of label names to encode when mode=label')
    parser.add_argument('--exclude-labels', '-x', type=str, nargs='+', default=None,
                        help='Labels to exclude from encoding (applied after --mode/--labels)')
    args = parser.parse_args()
    setup_logging(args.log_level)

    ResourcesClass, PreprocessorClass = REGISTRY[args.resource]
    res = ResourcesClass
    preprocessor = PreprocessorClass()

    input_dir = args.input_dir or res.CLEAN_MERGED_DATA_FOLDER
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")

    encoded_dir = res.ENCODED_DATA_FOLDER

    logger.info("===========================================")
    logger.info(f"MAJORITY ENCODING PIPELINE STARTED (resource={res.resources_name})")
    logger.info("===========================================")
    logger.info(f"[+] Numerical encoder: {args.num_encoder}; subset={args.subset}")

    # Resolve target labels
    if args.mode == 'label':
        if not args.labels:
            raise SystemExit("--labels is required when --mode label")
        # validate against resource labels
        valid = set(res.MAJORITY_LABELS)
        invalid = [lbl for lbl in args.labels if lbl not in valid]
        if invalid:
            raise SystemExit(f"Invalid labels for {res.resources_name}: {invalid}")
        target_labels = args.labels
    else:
        target_labels = res.MAJORITY_LABELS

    # Apply excludes if provided
    if args.exclude_labels:
        excludes = set(args.exclude_labels)
        target_labels = [lbl for lbl in target_labels if lbl not in excludes]
        logger.info(f"[+] Excluding labels: {sorted(excludes)}")

    if not target_labels:
        raise SystemExit("No labels to encode after applying excludes.")

    logger.info(f"[+] Target labels: {target_labels}")
    allowed_safe = {res.get_label_name(lbl) for lbl in target_labels}

    if not preprocessor.load_encoders():
        raise SystemExit(
            f"Encoders not found. Fit encoders for {res.resources_name} first."
        )

    def process_subset(subset: str):
        label_files = res.list_clean_merged_label_files(input_dir, allowed_safe, subset)
        if not label_files:
            logger.info(f"[+] Skip subset={subset}: no files found")
            return

        encoded_out_dir = res.encoded_dir_for_subset(subset)
        os.makedirs(encoded_out_dir, exist_ok=True)

        logger.info(f"[+] Found {len(label_files)} majority label files (subset={subset})")
        logger.info("[+] Encoding and exporting MAJORITY per-label datasets (encoded only)...")
        encoded_counts: dict[str, int] = {}

        for lf in label_files:
            base = os.path.basename(lf)
            suffix = '_clean_merged.csv' if subset == 'full' else f'_{subset}_clean_merged.csv'
            label_safe = base[len(f"{res.resources_name}_"):-len(suffix)]
            logger.info(f"[+] Encoding {base} -> label={label_safe} (subset={subset})")

            df_label = pd.read_csv(lf, low_memory=False)
            df_label['__rowid__'] = np.arange(len(df_label), dtype=np.int64)

            df_label = preprocessor.select_features_and_label(df_label)
            df_label['__rowid__'] = df_label.index.to_series().map(lambda i: i).astype(np.int64)

            if args.num_encoder == 'standard':
                if hasattr(preprocessor, 'preprocess_encode_numerical_features_standard'):
                    enc_df = preprocessor.preprocess_encode_numerical_features_standard(df_label.copy())
                else:
                    logger.error("No compatible numerical encoder found on preprocessor")
                    raise ValueError("No compatible numerical encoder found on preprocessor")
            else:
                raise ValueError(f"Invalid numerical encoder: {args.num_encoder}")

            # Binary and label/categorical encoding
            if hasattr(preprocessor, 'preprocess_encode_binary_features'):
                enc_df = preprocessor.preprocess_encode_binary_features(enc_df)
            else:
                logger.error("No compatible binary encoder found on preprocessor")
                raise ValueError("No compatible binary encoder found on preprocessor")
            if hasattr(preprocessor, 'preprocess_encode_label'):
                enc_df = preprocessor.preprocess_encode_label(enc_df)
            else:
                logger.error("No compatible label encoder found on preprocessor")
                raise ValueError("No compatible label encoder found on preprocessor")
            if hasattr(preprocessor, 'preprocess_encode_categorical_features'):
                enc_df = preprocessor.preprocess_encode_categorical_features(enc_df)
            else:
                logger.error("No compatible categorical encoder found on preprocessor")
                raise ValueError("No compatible categorical encoder found on preprocessor")

            enc_df['__subset__'] = subset

            enc_df = deduplicate_dataframe(
                enc_df,
                stage_name=f"{res.resources_name}_{label_safe} (majority, encoded)",
                preprocessor=preprocessor,
            )

            # Export encoded per-label
            enc_fname = res.encoded_filename_for_label(label_safe)
            enc_path = os.path.join(encoded_out_dir, enc_fname)
            enc_df.to_csv(enc_path, index=False)
            encoded_counts[label_safe] = len(enc_df)
            logger.info(f"[+] Saved encoded: {enc_path} ({len(enc_df)})")

        logger.info("===========================================")
        logger.info(f"[+] MAJORITY ENCODING COMPLETED (subset={subset})")
        logger.info("===========================================")
        logger.info(f"[+] Encoded outputs -> {encoded_out_dir}")
        if encoded_counts:
            sample_items = list(encoded_counts.items())[:15]
            logger.info("[+] Sample encoded counts:")
            for name, cnt in sample_items:
                logger.info(f"    {name}: {cnt}")
            if len(encoded_counts) > 15:
                logger.info(f"    ... and {len(encoded_counts) - 15} more labels")

    if args.subset == 'full':
        for sub in ['train', 'test']:
            process_subset(sub)
    else:
        process_subset(args.subset)


if __name__ == "__main__":
    main()



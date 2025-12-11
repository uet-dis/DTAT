import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from utils.logging import setup_logging, get_logger
from configs import CIC2018Resources

logger = get_logger(__name__)

data_folder = CIC2018Resources.ORIGINAL_DATA_FOLDER
dest_folder = CIC2018Resources.DATA_FOLDER
dest_folder_by_label = CIC2018Resources.CLEAN_MERGED_DATA_FOLDER

os.makedirs(dest_folder_by_label, exist_ok=True)


def read_header_columns(csv_path: str) -> List[str]:
    """Read header columns from CSV file"""
    header_df = pd.read_csv(
        csv_path, nrows=0, low_memory=False, skipinitialspace=True
    )
    return [c.strip() for c in header_df.columns.tolist()]


def detect_majority_schema(csv_files: List[str]) -> Tuple[List[str], dict]:
    """
    Detect the majority schema across all CSV files.
    """
    schema_counter: Counter = Counter()
    file_to_schema = {}

    for fp in csv_files:
        cols = read_header_columns(fp)
        file_to_schema[fp] = cols
        schema_counter[tuple(cols)] += 1

    if not schema_counter:
        raise RuntimeError("No CSV headers detected in the provided folder")

    majority_schema_tuple, majority_count = schema_counter.most_common(1)[0]
    majority_schema = list(majority_schema_tuple)

    logger.info(f"[+] Majority schema detected from {majority_count}/{len(csv_files)} files")
    logger.info(f"[+] Majority schema: {len(majority_schema)} columns")
    
    # Check for files with different schemas
    outliers = [fp for fp, cols in file_to_schema.items() if cols != majority_schema]
    if outliers:
        logger.info(f"[+] Found {len(outliers)} files with different schemas:")
        for fp in outliers:
            cols = file_to_schema[fp]
            extra = [c for c in cols if c not in majority_schema]
            missing = [c for c in majority_schema if c not in cols]
            logger.info(f"    - {os.path.basename(fp)}: +{len(extra)} extras, -{len(missing)} missing")
            if extra:
                logger.info(f"      Extra: {extra[:4]}{'...' if len(extra) > 4 else ''}")
    else:
        logger.info(f"[+] All {len(csv_files)} files have identical schema")

    return majority_schema, file_to_schema


def merge_csv_files_with_early_cleaning(
    csv_files: List[str],
    majority_schema: List[str],
    preprocessor: CIC2018Preprocessor,
    output_dir: str,
    chunksize: int = 500_000,
    label_distribution: bool = False,
    protocol_distribution: bool = False,
    ignore_sentinel_cols: bool = True,
):
    """
    Phase 1: Stream-merge with early cleaning, then route rows per label into
    per-label CSVs under output_dir using naming convention from CIC2018Resources.
    """
    logger.info("[+] Phase 1: Merge CSV files with early cleaning (stream -> per label files)...")
    os.makedirs(output_dir, exist_ok=True)

    # Remove existing per-label merged files to avoid appending across runs
    existing_label_files = [
        os.path.join(output_dir, f)
        for f in sorted(os.listdir(output_dir))
        if f.endswith('_clean_merged.csv')
    ]
    if existing_label_files:
        logger.info(f"[+] Removing {len(existing_label_files)} existing per-label files before writing (append mode)")
        for lf in existing_label_files:
            try:
                os.remove(lf)
                logger.debug(f"    removed: {os.path.basename(lf)}")
            except Exception as e:
                logger.warning(f"[!] Failed to remove {lf}: {e}")

    total_read = 0
    dropped_header_rows = 0
    dropped_rows = 0
    logger.info(f"[+] Dropping columns: {preprocessor.columns_to_drop}")
    per_label_counts: Counter = Counter()

    for fp in csv_files:
        logger.info(f"[+] Processing: {os.path.basename(fp)}")
        # Per-file aggregations for debugging
        file_label_counter: Counter = Counter()
        protocol_seen_raw: set = set()
        protocol_seen_clean: set = set()

        for chunk in pd.read_csv(
            fp,
            usecols=majority_schema,  # Use only majority schema columns
            low_memory=False,
            skipinitialspace=True,
            chunksize=chunksize,
        ):
            total_read += len(chunk)


            chunk = preprocessor.select_features_and_label(chunk)

            # Coerce dtypes per schema (cat/binary -> Int32, continuous -> float32)
            chunk = preprocessor.coerce_feature_dtypes(chunk)

            # Normalize and drop repeated header rows
            if "Label" not in chunk.columns:
                raise RuntimeError(f"Missing 'Label' column in file {fp}")

            chunk["Label"] = chunk["Label"].astype(str).str.strip()
            before = len(chunk)
            chunk = chunk[chunk["Label"] != "Label"]
            dropped_header_rows += before - len(chunk)

            # Aggregate per-file label distribution (pre-clean) and Protocol raw values
            if label_distribution:
                file_label_counter.update(chunk["Label"].value_counts().to_dict())
                if 'Protocol' in chunk.columns:
                    try:
                        protocol_seen_raw.update(pd.Series(chunk['Protocol']).dropna().unique().tolist())
                    except Exception:
                        pass

            before_clean = len(chunk)
            # Clean missing/inf and intra-chunk duplicates
            chunk = preprocessor.remove_missing_and_inf_values(chunk)


            # Drop rows with any negative numeric values
            chunk = preprocessor.remove_negative_numeric_rows(chunk)

            # Drop rows with many zeros in continuous features (>=50% of cont_features == 0)
            # chunk = preprocessor.drop_rows_with_zero_heavy_continuous(chunk, threshold_frac=0.5)

            # Drop duplicates (intra-chunk)
            chunk = preprocessor.fix_duplicates(chunk)

            after_clean = len(chunk)
            dropped_rows += before_clean - after_clean


            # Track Protocol values after cleaning (per-file)
            if label_distribution and 'Protocol' in chunk.columns and len(chunk) > 0:
                try:
                    protocol_seen_clean.update(pd.Series(chunk['Protocol']).dropna().unique().tolist())
                except Exception:
                    pass

            if len(chunk) == 0:
                continue

            # Route rows per label to per-label CSVs (append mode)
            for lbl, sub_df in chunk.groupby('Label'):
                if len(sub_df) == 0:
                    continue
                if label_distribution:
                    logger.debug(f"[+] Writing {len(sub_df)} rows (after cleaning) for label {lbl}")
                safe_label = CIC2018Resources.get_label_name(lbl)
                out_path = os.path.join(output_dir, f"cic2018_{safe_label}_clean_merged.csv")
                write_header = not os.path.exists(out_path)
                sub_df.to_csv(out_path, index=False, mode='a', header=write_header)
                per_label_counts[lbl] += len(sub_df)

        # End of file: emit aggregated label distribution and protocol values if requested
        if label_distribution:
            logger.info(f"[+] File summary (pre-clean) for {os.path.basename(fp)}:")
            for lbl, cnt in file_label_counter.most_common():
                logger.info(f"    {lbl}: {cnt}")
        if protocol_distribution:
            if protocol_seen_raw:
                logger.info(f"[+] Protocol values observed (raw) in {os.path.basename(fp)}: {sorted(list(protocol_seen_raw))}")
            if protocol_seen_clean:
                logger.info(f"[+] Protocol values observed (after cleaning) in {os.path.basename(fp)}: {sorted(list(protocol_seen_clean))}")
        
    logger.info(f"[+] Phase 1 complete:")
    logger.info(f"    - Total rows read: {total_read}")
    logger.info(f"    - Dropped header rows: {dropped_header_rows}")
    logger.info(f"    - Dropped rows with NaN/Inf/negative values and duplication intra-chunk (feature selection was applied): {dropped_rows}")
    logger.info(f"    - Labels written: {len(per_label_counts)}")
    return per_label_counts


def deduplicate_dataframe(df: pd.DataFrame, stage_name: str = "", precheck: bool = False, postcheck: bool = False, preprocessor: CIC2018Preprocessor = None):
    """
    Remove duplicates from dataframe using DataService with optimized methods for dataset size.
    """
    logger.info(f"[+] Deduplicating {stage_name} data...")
    
    # Use DataService static methods for consistent deduplication
    has_duplicates = preprocessor.check_duplicates(df) if precheck else True
    
    if has_duplicates:
        before_dedup = len(df)
        if precheck:
            duplicate_count = df.duplicated().sum()
            logger.info(f"[+] Found {duplicate_count} duplicates in {stage_name} data")
        
        # Use DataService with auto method selection for optimal performance
        df_clean = preprocessor.fix_duplicates(df)
        after_dedup = len(df_clean)
        duplicates_removed = before_dedup - after_dedup
        
        logger.info(f"[+] {stage_name} deduplication: {duplicates_removed} duplicates removed")
        logger.info(f"[+] Final {stage_name} shape: {df_clean.shape}")
        
        if postcheck:
            final_dup_check = preprocessor.check_duplicates(df_clean)
            if final_dup_check:
                remaining_dups = df_clean.duplicated().sum()
                logger.warning(f"[!] WARNING: Still {remaining_dups} duplicates remaining!")
            else:
                logger.info(f"[+] Verification: No duplicates remaining in {stage_name} data")
        
        return df_clean
    else:
        logger.info(f"[+] No duplicates found in {stage_name} data")
        return df


def main():
    """
    Phase 1-2 preprocessing pipeline:
    1. Stream-merge CSV files with early cleaning and write per-label CSVs to CLEAN_MERGED_DATA_FOLDER
    2. Per-label final deduplication pass over written files
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--precheck", action="store_true", help="Precheck for duplicates")
    parser.add_argument("--log-level", '-l', type=str, default="INFO", help="Logging level")
    parser.add_argument("--postcheck", action="store_true", help="Postcheck for duplicates")
    parser.add_argument("--label-distribution", '-d', action="store_true", help="Get label distribution")
    parser.add_argument("--protocol-distribution", action="store_true", help="Enable per-file protocol distribution diagnostics")
    parser.add_argument("--ignore-sentinel-cols", action="store_true", help="Ignore sentinel columns when removing negative numeric rows")
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    logger.info(f"[+] Ignoring sentinel columns: {args.ignore_sentinel_cols}")
    if not os.path.isdir(data_folder):
        raise SystemExit(f"Data folder not found: {data_folder}")
    
    # Discover CSV files
    csv_files = [
        os.path.join(data_folder, f)
        for f in sorted(os.listdir(data_folder))
        if f.lower().endswith(".csv")
    ]
    
    if not csv_files:
        raise SystemExit(f"No CSV files found in {data_folder}")
    
    logger.info(f"[+] Found {len(csv_files)} CSV files")
    logger.info("[+] ===========================================")
    logger.info("[+] PHASE 1-2 PREPROCESSING PIPELINE STARTED")
    logger.info("[+] ===========================================")
    
    # Initialize preprocessor
    preprocessor = CIC2018Preprocessor()
    
    # Detect majority schema across all files
    logger.info("[+] Detecting schema across CSV files...")
    majority_schema, file_to_schema = detect_majority_schema(csv_files)
    
    # Phase 1: Stream-merge and write per-label files
    per_label_counts = merge_csv_files_with_early_cleaning(
        csv_files, majority_schema, preprocessor, dest_folder_by_label,
        label_distribution=args.label_distribution,
        protocol_distribution=args.protocol_distribution,
        ignore_sentinel_cols=args.ignore_sentinel_cols,
    )

    if args.label_distribution and per_label_counts:
        logger.info("[+] Label distribution (written in Phase 1):")
        for label, count in per_label_counts.most_common():
            logger.info(f"    {label}: {count}")

    # Phase 2: Final deduplication pass per label file
    logger.info("[+] Phase 2: Final deduplication per label file...")
    os.makedirs(dest_folder_by_label, exist_ok=True)
    label_files = [
        os.path.join(dest_folder_by_label, f)
        for f in sorted(os.listdir(dest_folder_by_label))
        if f.endswith('_clean_merged.csv')
    ]

    final_counts = {}
    for lf in label_files:
        try:
            df_label = pd.read_csv(lf)
            df_label = deduplicate_dataframe(df_label, stage_name=os.path.basename(lf), postcheck=args.postcheck, preprocessor=preprocessor)
            df_label.to_csv(lf, index=False)
            final_counts[os.path.basename(lf)] = len(df_label)
        except Exception as e:
            logger.warning(f"[!] Failed to deduplicate {lf}: {e}")

    # Final summary for Phase 1-2
    logger.info("[+] ===========================================")
    logger.info("[+] PHASE 1-2 PREPROCESSING COMPLETED!")
    logger.info("[+] ===========================================")
    logger.info(f"[+] Input files processed: {len(csv_files)}")
    logger.info(f"[+] Per-label files written: {len(label_files)} @ {dest_folder_by_label}")
    if final_counts:
        logger.info("[+] Final per-label row counts (post-dedup):")
        sorted_items = sorted(final_counts.items(), key=lambda kv: kv[1], reverse=True)

        for name, cnt in sorted_items:
            logger.info(f"    {name}: {cnt}")
    logger.info("[+] ===========================================")
    logger.info("[+] Ready for Phase 3-4 encoding pipeline!")
    logger.info("[+] Run: encode_data to continue processing")


if __name__ == "__main__":
    main()

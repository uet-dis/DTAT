import os
import sys
import argparse
import pandas as pd

from utils.logging import get_logger, setup_logging



SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from configs import CIC2018Resources, NSLKDDResources
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor

# WGAN
from resampling.data_augmentation.augmented_wgan.pipeline import (
    AugmentOptions as WGANOptions,
    train_wgan_with_critic,
    generate_encoded,
    trim_to_need,
    final_fill,
)
from resampling.data_augmentation.augmented_wgan.wgan import WGAN


logger = get_logger(__name__)


REGISTRY = {
    'cic2018': (CIC2018Resources, CIC2018Preprocessor),
    'nslkdd': (NSLKDDResources, NSLKDDPreprocessor),
}


def _encode_like_minor(pre, df: pd.DataFrame, num_encoder: str) -> pd.DataFrame:
    df2 = pre.select_features_and_label(df.copy())
    logger.info(f"[+] Encoding like minority with {num_encoder} encoder")

    if num_encoder == 'quantile_uniform' and hasattr(pre, 'preprocess_encode_numerical_features_quantile_uniform'):
        enc_df = pre.preprocess_encode_numerical_features_quantile_uniform(df2)
    elif num_encoder == 'minmax' and hasattr(pre, 'preprocess_encode_numerical_features_minmax'):
        enc_df = pre.preprocess_encode_numerical_features_minmax(df2)
    else:
        raise ValueError(f"Invalid numerical encoder for opposite sampling: {num_encoder}")

    if hasattr(pre, 'preprocess_encode_binary_features'):
        enc_df = pre.preprocess_encode_binary_features(enc_df)
    else:
        raise AttributeError(f"No compatible binary encoder found on preprocessor")
    if hasattr(pre, 'preprocess_encode_label'):
        enc_df = pre.preprocess_encode_label(enc_df)
    else:
        raise AttributeError(f"No compatible label encoder found on preprocessor")
    if hasattr(pre, 'preprocess_encode_categorical_features'):
        enc_df = pre.preprocess_encode_categorical_features(enc_df)
    else:
        raise AttributeError(f"No compatible categorical encoder found on preprocessor")
    return enc_df


def _load_clean_merged_opposite(res, pre, label_name: str, subset: str = 'train', num_encoder: str = 'quantile_uniform'):
    try:
        safe = res.get_label_name(label_name)
        # Prefer compressed if available (for Benign), else fall back to original clean_merged
        cmc = res.clean_merged_path_for(subset, safe, compressed=True)
        if os.path.exists(cmc):
            df = pd.read_csv(cmc, low_memory=False)
        else:
            cm = res.clean_merged_path_for(subset, safe, compressed=False)
            if not os.path.exists(cm):
                raise FileNotFoundError(f"Opposite clean_merged not found: {cm}")
            df = pd.read_csv(cm, low_memory=False)

        try:
            return _encode_like_minor(pre, df, num_encoder=num_encoder)
        except Exception as e:
            logger.warning(f"[WGAN] Opposite encoding failed: {e}")
            return None
    except Exception as e:
        logger.warning(f"[WGAN] Opposite clean_merged loader failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Minority augmenting (WGAN) for multiple resources")
    parser.add_argument('--resource', '-r', type=str, required=True, choices=list(REGISTRY.keys()))
    parser.add_argument('--augmenting-strategy', '-a', type=str, required=True, choices=['wgan'],
                        help='Augmenting strategy to use (currently: wgan)')
    parser.add_argument('--mode', '-m', type=str, default='all', choices=['all', 'label'],
                        help='Augment all predefined minority labels or provided labels')
    parser.add_argument('--labels', '-c', type=str, nargs='+', default=None,
                        help='Label names to augment when mode=label')
    parser.add_argument('--exclude-labels', '-x', type=str, nargs='+', default=None,
                        help='Labels to exclude from augmentation')
    parser.add_argument('--opposite-label', type=str, default='Benign',
                        help='Label to use as critic opposite class (default Benign)')
    parser.add_argument('--num-encoder', '-n', type=str, required=True,
                        choices=['minmax', 'quantile_uniform'],
                        help='Numerical encoder for opposite sampling encode (match minority encoding)')
    parser.add_argument('--tau', '-t', type=int, required=True, help='Base target samples per class after augmentation')
    parser.add_argument('--log-level', '-L', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    args = parser.parse_args()
    setup_logging(args.log_level)

    ResourcesClass, PreprocessorClass = REGISTRY[args.resource]
    res = ResourcesClass
    pre = PreprocessorClass()

    # Load encoders required by augmentation pipelines
    if not pre.load_encoders():
        raise SystemExit(f"Encoders not found. Fit encoders for {res.resources_name} first.")

    # Resolve labels
    if args.mode == 'label':
        if not args.labels:
            raise SystemExit("--labels is required when --mode label")
        minority_labels = args.labels
        # validate
        valid = set(res.MINORITY_LABELS)
        invalid = [lb for lb in minority_labels if lb not in valid]
        if invalid:
            raise SystemExit(f"Invalid labels for {res.resources_name}: {invalid}")
    else:
        minority_labels = res.MINORITY_LABELS
    if args.exclude_labels:
        excludes = set(args.exclude_labels)
        minority_labels = [lbl for lbl in minority_labels if lbl not in excludes]

    logger.info(f"[+] Minority labels to augment ({res.resources_name}): {minority_labels}")

    # Directories and accept rates
    encoded_train_dir = res.encoded_dir_for_subset('train')
    os.makedirs(encoded_train_dir, exist_ok=True)
    accept_rate_map = getattr(res, 'ACCEPT_RATE_MAP', {}) or {}

    # Opposite label id (optional)
    try:
        critic_id = pre.encoders['label'].transform([args.opposite_label])[0]
    except Exception:
        critic_id = None

    for lbl in minority_labels:
        safe = res.get_label_name(lbl)
        enc_train_path = os.path.join(encoded_train_dir, res.encoded_filename_for_label(safe))
        if not os.path.exists(enc_train_path):
            logger.warning(f"[+] Encoded train not found for {lbl}: {enc_train_path}; skip")
            continue
        train_df = pd.read_csv(enc_train_path, low_memory=False)

        # Load encoded test if available (for dedup keys in WGAN)
        encoded_test_dir = res.encoded_dir_for_subset('test')
        enc_test_path = os.path.join(encoded_test_dir, res.encoded_filename_for_label(safe))
        if os.path.exists(enc_test_path):
            test_df = pd.read_csv(enc_test_path, low_memory=False)
            logger.info(f"[+] Loaded encoded test: {len(test_df)} rows")
        else:
            # Create minimal test_df by sampling from train_df to avoid pipeline issues
            test_df = train_df.sample(n=min(100, len(train_df)), random_state=42).copy()
            logger.info(f"[+] No encoded test found, using train sample as test_df for {lbl} ({len(test_df)} rows)")

        if args.augmenting_strategy == 'wgan':
            # Accept rate per label with sensible fallbacks
            ar = float(accept_rate_map.get(lbl, 0.30))
            opts = WGANOptions(
                use_benign_for_critic=True,
                critic_epochs=60,
                wgan_iterations=10000,
                d_iter=5,
                use_gp=True,
                accept_rate=ar,
                request_multiplier=3.0,
                max_rounds=40,
                min_precision=0.95,
                trim_to_need=True,
                use_final_fill=True,
            )

            def _benign_loader():
                return _load_clean_merged_opposite(
                    res,
                    pre,
                    label_name=args.opposite_label,
                    subset='train',
                    num_encoder=args.num_encoder,
                )

            # Orchestrate generation externally (encoded → decode → raw-dedup → refill)
            feat_cols = [c for c in train_df.columns if c != 'Label']
            benign_enc = _benign_loader() if opts.use_benign_for_critic else None
            # Try to load saved WGAN/critic for this label to speed up
            models_dir = f"models/{res.resources_name}"
            safe_name = safe.lower().replace(' ', '_').replace('/', '_')
            ckpt_dir = os.path.join(models_dir, f"wgan_{safe_name}")
            wgan = None
            try:
                if os.path.isdir(ckpt_dir) and \
                   os.path.exists(os.path.join(ckpt_dir, 'wgan_model.pth')) and \
                   os.path.exists(os.path.join(ckpt_dir, 'critic_model.pth')):
                    wgan = WGAN(x_dim=len(feat_cols), device='auto', use_gp=opts.use_gp, use_critic_loss=True, lambda_critic=0.5)
                    wgan.load_models(ckpt_dir)
                    logger.info(f"[+] Loaded saved WGAN models from {ckpt_dir}")
            except Exception as e:
                logger.warning(f"[Load] Failed to load saved WGAN, will retrain: {e}")

            if wgan is None:
                wgan = train_wgan_with_critic(
                    encoded_train=train_df,
                    feat_cols=feat_cols,
                    pre=pre,
                    benign_encoded=benign_enc,
                    device='auto',
                    use_gp=opts.use_gp,
                    critic_epochs=opts.critic_epochs,
                    wgan_iterations=opts.wgan_iterations,
                    d_iter=opts.d_iter,
                    critic_id=critic_id,
                )
                # Save for future reuse
                try:
                    os.makedirs(ckpt_dir, exist_ok=True)
                    wgan.save_models(ckpt_dir)
                    logger.info(f"[+] Saved WGAN models to {ckpt_dir}")
                except Exception as e:
                    logger.warning(f"[Save] Failed to save WGAN models: {e}")

            need = int(args.tau) - len(train_df)
            if need <= 0:
                logger.info(f"[+] {lbl} already >= tau")
                continue

            # Build base raw keys from original raw train/test files to avoid duplicates against original data
            base_raw_keys = set()
            base_feats = None
            def _load_raw_subset(subset_name: str):
                try:
                    # Prefer compressed file if available
                    cmc = res.clean_merged_path_for(subset_name, safe, compressed=True)
                    if os.path.exists(cmc):
                        return pd.read_csv(cmc, low_memory=False)
                    cm = res.clean_merged_path_for(subset_name, safe, compressed=False)
                    if os.path.exists(cm):
                        return pd.read_csv(cm, low_memory=False)
                except Exception:
                    return None
                return None

            raw_train_df = _load_raw_subset('train')
            raw_test_df = _load_raw_subset('test')
            try:
                if raw_train_df is not None and not raw_train_df.empty:
                    tr_sel = pre.select_features_and_label(raw_train_df.copy())
                    base_feats = [c for c in tr_sel.columns if c != 'Label']
                    arr = tr_sel[base_feats].values
                    base_raw_keys |= set(map(tuple, arr))
                if raw_test_df is not None and not raw_test_df.empty:
                    te_sel = pre.select_features_and_label(raw_test_df.copy())
                    if base_feats is None:
                        base_feats = [c for c in te_sel.columns if c != 'Label']
                    arr = te_sel[[c for c in te_sel.columns if c != 'Label']].values
                    base_raw_keys |= set(map(tuple, arr))
            except Exception as e:
                logger.warning(f"[Dedup] Failed to build base raw keys for {lbl}: {e}")

            accepted_raw_batches = []
            rounds = 0
            # raw-level dedup using original raw keys and already-accepted batches
            def _dedup_raw(df_raw, existing_batches):
                feats = base_feats or [c for c in df_raw.columns if c != 'Label']
                keys_existing = set(base_raw_keys)
                for b in existing_batches:
                    arr = b[feats].values
                    keys_existing |= set(map(tuple, arr))
                mask = [tuple(row) not in keys_existing for row in df_raw[feats].values]
                return df_raw.loc[mask]

            while sum(len(b) for b in accepted_raw_batches) < need and rounds < opts.max_rounds:
                rounds += 1
                need_now = need - sum(len(b) for b in accepted_raw_batches)
                request = int(min(max(need_now * opts.request_multiplier, (need_now / max(opts.accept_rate, 1e-3)) * opts.request_multiplier), 60000))
                logger.info(f"[+] Round {rounds}/{opts.max_rounds}: need_now={need_now}, request={request}, accept_rate={opts.accept_rate}")

                gen_enc_df = generate_encoded(wgan, request, feat_cols, opts.accept_rate)
                # Assign encoded label id
                class_id = pre.encoders['label'].transform([lbl])[0]
                gen_enc_df['Label'] = class_id
                # Decode to raw
                try:
                    gen_raw_df = pre.inverse_transform(gen_enc_df, numerical_inverse=args.num_encoder)
                except Exception as e:
                    logger.warning(f"[Decode] skipped batch: {e}")
                    continue

                # Dedup across already accepted raw batches
                feats_for_batch = base_feats or [c for c in gen_raw_df.columns if c != 'Label']
                # 1) Dedup internal batch by features
                try:
                    gen_raw_df = gen_raw_df.drop_duplicates(subset=feats_for_batch, ignore_index=True)
                except Exception:
                    gen_raw_df = gen_raw_df.drop_duplicates(ignore_index=True)
                # 2) Dedup with base_raw_keys + accepted batches
                gen_raw_unique = _dedup_raw(gen_raw_df, accepted_raw_batches)
                if len(gen_raw_unique) > 0:
                    accepted_raw_batches.append(gen_raw_unique)

            # Combine and trim to need
            if accepted_raw_batches:
                acc_raw = pd.concat(accepted_raw_batches, ignore_index=True)
            else:
                acc_raw = pd.DataFrame(columns=[c for c in gen_raw_df.columns] if 'gen_raw_df' in locals() else [])

            if len(acc_raw) > need:
                acc_raw = acc_raw.head(need)

            # If still short, generate extra encoded and decode, then append (no dedup to ensure completion)
            deficit = need - len(acc_raw)
            if deficit > 0:
                try:
                    extra_enc = final_fill(wgan, feat_cols, pd.DataFrame(columns=feat_cols), deficit, opts.accept_rate)
                    if len(extra_enc) > 0:
                        extra_enc['Label'] = class_id
                        extra_raw = pre.inverse_transform(extra_enc, numerical_inverse=args.num_encoder)
                        # Dedup nội bộ batch fill theo features
                        feats_for_fill = base_feats or [c for c in extra_raw.columns if c != 'Label']
                        try:
                            extra_raw = extra_raw.drop_duplicates(subset=feats_for_fill, ignore_index=True)
                        except Exception:
                            extra_raw = extra_raw.drop_duplicates(ignore_index=True)
                        # Dedup với base_raw_keys + acc_raw hiện có
                        keys_existing = set(base_raw_keys)
                        if len(acc_raw) > 0:
                            try:
                                arr = acc_raw[feats_for_fill].values
                            except Exception:
                                arr = acc_raw.values
                            keys_existing |= set(map(tuple, arr))
                        try:
                            mask = [tuple(row) not in keys_existing for row in extra_raw[feats_for_fill].values]
                        except Exception:
                            mask = [tuple(row) not in keys_existing for row in extra_raw.values]
                        extra_raw_unique = extra_raw.loc[mask]
                        if len(extra_raw_unique) > 0:
                            acc_raw = pd.concat([acc_raw, extra_raw_unique.head(deficit)], ignore_index=True)
                except Exception as e:
                    logger.warning(f"[Fill] skipped: {e}")

            # Save merged output (original raw train + augmented) into the legacy augmented filename
            raw_train_dir = os.path.join(res.RAW_PROCESSED_DATA_FOLDER, 'train')
            out_path = os.path.join(raw_train_dir, f"{res.resources_name}_{safe}_minority_{args.augmenting_strategy}_train_augmented_raw_processed.csv")
            try:
                base_sel = pre.select_features_and_label(raw_train_df.copy()) if (raw_train_df is not None and not raw_train_df.empty) else pd.DataFrame(columns=acc_raw.columns)
            except Exception:
                base_sel = pd.DataFrame(columns=acc_raw.columns)
            merged_raw = pd.concat([base_sel, acc_raw], ignore_index=True)
            merged_raw = merged_raw.drop_duplicates(ignore_index=True)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            merged_raw.to_csv(out_path, index=False)
            logger.info(f"[+] Saved RAW_PROCESSED ({lbl}) -> {out_path} ({len(merged_raw)})")
        else:
            logger.error(f"[+] Augmenting strategy not implemented: {args.augmenting_strategy}")
            continue

        # (Legacy direct-decode path removed; output saved above via acc_raw)

    logger.info(f"[+] {res.resources_name} minority augmenting completed")


if __name__ == "__main__":
    main()



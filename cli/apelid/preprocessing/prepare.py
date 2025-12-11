import numpy as np
import pandas as pd
from .preprocessor import Preprocessor
from sklearn.model_selection import train_test_split
from utils.logging import get_logger

logger = get_logger(__name__)


class PrepareData:
    @staticmethod
    def prepare_training_data(
        df: pd.DataFrame,
        pre: Preprocessor,
        *,
        use_validation: bool = False,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, dict]:
        """Return numpy arrays and metadata for training.

        Returns: X_tr, X_val, y_tr, y_val, meta
        meta: { feature_names, class_names, cat_feature_indices, cont_feature_indices }
        """
        train_df = pre.select_features_and_label(df.copy())

        logger.debug(f"[+] train_df shape: {train_df.shape}")

        train_df = pre.fix_duplicates(train_df)
        logger.info(f"[+] train_df shape after fix_duplicates: {train_df.shape}")
        logger.debug(f"[+] train_df label distribution after fix_duplicates: {train_df[pre.label_column].value_counts()}")

        # stats params for input normalization (computed after ordinal/binary encodes, before any numerical scaling)
        cont_features = getattr(pre, 'cont_features', []) or []
        
        # Ordinal for categorical
        train_df = pre.preprocess_encode_ordinal_features(train_df)
        # Binary to 0/1
        train_df = pre.preprocess_encode_binary_features(train_df)
        
        # Compute continuous stats on encoded-but-unscaled data
        cont_cols_for_stats = [c for c in cont_features if c in train_df.columns]
        if cont_cols_for_stats:
            cont_mat = train_df[cont_cols_for_stats].astype(np.float32)
            cont_means_all = cont_mat.mean(axis=0).to_numpy()
            cont_stds_all = cont_mat.std(axis=0, ddof=0).to_numpy()
            # Clamp tiny stds to 1.0 to avoid division blow-up at runtime
            cont_stds_all = np.where(cont_stds_all < 1e-6, 1.0, cont_stds_all)
        else:
            cont_means_all = np.array([], dtype=np.float32)
            cont_stds_all = np.array([], dtype=np.float32)
        # NOTE: Numerical scaling removed; handled in-model (InputNorm) for DNN.
        # Label encode
        train_df = pre.preprocess_encode_label(train_df)

        # Split
        if use_validation:
            strat = train_df[pre.label_column] if train_df[pre.label_column].nunique() > 1 else None
            train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state, stratify=strat)
        else:
            val_df = None

        feature_names = list(pre.ordered_features)
        clip_values = (np.min(train_df[feature_names].to_numpy()), np.max(train_df[feature_names].to_numpy()))
        X_tr = train_df[feature_names].to_numpy()
        y_tr = train_df[pre.label_column].to_numpy()
        X_val = val_df[feature_names].to_numpy() if val_df is not None else None
        y_val = val_df[pre.label_column].to_numpy() if val_df is not None else None

        # Build indices for cat/cont if available on preprocessor
        cat_cols = getattr(pre, 'cat_features', []) or []
        cont_cols = getattr(pre, 'cont_features', []) or []
        binary_cols = getattr(pre, 'binary_features', []) or []
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        cat_idx = [name_to_idx[c] for c in cat_cols if c in name_to_idx]
        cont_idx = [name_to_idx[c] for c in cont_cols if c in name_to_idx]
        binary_idx = [name_to_idx[c] for c in binary_cols if c in name_to_idx]
        class_names = list(getattr(getattr(pre, 'encoders', {}), 'get', lambda k: None)('label').classes_) if getattr(pre, 'encoders', None) and 'label' in pre.encoders else []

        # Align cont stats to the order of cont_idx (feature_names order)
        if cont_idx:
            cont_names_in_order = [feature_names[i] for i in cont_idx]
            if cont_cols_for_stats:
                stats_df = {
                    'mean': {name: float(m) for name, m in zip(cont_cols_for_stats, cont_means_all)},
                    'std': {name: float(s) for name, s in zip(cont_cols_for_stats, cont_stds_all)},
                }
                cont_means = [stats_df['mean'].get(n, 0.0) for n in cont_names_in_order]
                cont_stds = [stats_df['std'].get(n, 1.0) for n in cont_names_in_order]
            else:
                cont_means = []
                cont_stds = []
        else:
            cont_means = []
            cont_stds = []

        meta = {
            'feature_names': feature_names,
            'class_names': class_names,
            'cat_feature_indices': cat_idx,
            'cont_feature_indices': cont_idx,
            'binary_feature_indices': binary_idx,
            # input-norm stats aligned to cont_feature_indices
            'cont_means': cont_means,
            'cont_stds': cont_stds,
            'inputnorm_eps': 1e-6,
            'clip_values': clip_values,
            'label_column': pre.label_column,
        }
        return X_tr, X_val, y_tr, y_val, meta

    @staticmethod
    def prepare_input_data(
        df: pd.DataFrame,
        pre: Preprocessor,
        *,
        include_label: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None, dict]:
        """Return numpy arrays and metadata for test/inference.

        Returns: X_te, y_te (or None), meta
        """
        input_df = pre.select_features_and_label(df.copy()) if include_label else df.copy()
        input_df = pre.preprocess_encode_ordinal_features(input_df)
        input_df = pre.preprocess_encode_binary_features(input_df)
        # NOTE: Numerical scaling removed; handled in-model (InputNorm) for DNN.
        if include_label:
            input_df = pre.preprocess_encode_label(input_df)

        feature_names = list(pre.ordered_features)
        X_te = input_df[feature_names].to_numpy()
        clip_values = (np.min(X_te), np.max(X_te))
        y_te = input_df[pre.label_column].to_numpy() if include_label and pre.label_column in input_df.columns else None

        cat_cols = getattr(pre, 'cat_features', []) or []
        cont_cols = getattr(pre, 'cont_features', []) or []
        binary_cols = getattr(pre, 'binary_features', []) or []
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        cat_idx = [name_to_idx[c] for c in cat_cols if c in name_to_idx]
        cont_idx = [name_to_idx[c] for c in cont_cols if c in name_to_idx]
        binary_idx = [name_to_idx[c] for c in binary_cols if c in name_to_idx]
        class_names = list(getattr(getattr(pre, 'encoders', {}), 'get', lambda k: None)('label').classes_) if getattr(pre, 'encoders', None) and 'label' in pre.encoders else []

        meta = {
            'feature_names': feature_names,
            'class_names': class_names,
            'cat_feature_indices': cat_idx,
            'cont_feature_indices': cont_idx,
            'binary_feature_indices': binary_idx,
            'clip_values': clip_values,
            'label_column': pre.label_column,
        }
        return X_te, y_te, meta
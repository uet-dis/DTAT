import pandas as pd
import numpy as np
from utils.logging import get_logger
import joblib

logger = get_logger(__name__)

class Preprocessor:
    def __init__(self, features: list[str], label_column: str):
        self.features = features
        self.label_column = label_column
        self.cat_features = []
        self.cont_features = []
        self.binary_features = []
        self.encoders = {}

    def info_dataset(self, df: pd.DataFrame):
        logger.info(f"[+] Dataset shape: {df.shape}")
        logger.info(f"[+] Dataset columns: {df.columns.tolist()}")
        logger.info(f"[+] Dataset label distribution:")
        logger.info(df[self.label_column].value_counts())

    def remove_missing_and_inf_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace infinite values with NaN and drop all rows containing NaN or infinite values"""
        logger.debug(f"[+] Cleaning missing and infinite values from {df.shape[0]} rows")
        # Replace +inf and -inf with NaN
        result_df = df.replace([np.inf, -np.inf], np.nan)
        # Drop rows containing NaN (including original NaN and converted inf values)
        result_df = result_df.dropna()
        logger.debug(f"[+] to {result_df.shape[0]} rows")
        return result_df
    
    def check_missing_and_inf_values(self, df: pd.DataFrame) -> bool:
        """Check if the dataframe contains any missing values (NaN) or infinite values (+inf or -inf)"""
        has_missing = df.isnull().any().any()
        has_inf = np.isinf(df.select_dtypes(include=[np.number])).any().any()
        return has_missing or has_inf
    
    def align_schema(self, df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
        """Ensure dataframe has all expected columns; add missing ones as NaN and reorder."""
        out = df.copy()
        for c in expected_cols:
            if c not in out.columns:
                out[c] = np.nan
        return out[expected_cols]

    def fix_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(ignore_index=True)
    
    def check_duplicates(self, df: pd.DataFrame) -> bool:
        """Check if dataframe contains duplicates"""
        return df.duplicated().any()
    
    def remove_negative_numeric_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows containing any negative values in numeric columns."""
        logger.debug(f"[+] Dropping rows with any negative values in numeric columns from {df.shape[0]} rows")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            logger.debug(f"[+] No numeric columns found")
            return df
        keep_mask = df[numeric_cols].ge(0).all(axis=1)
        logger.debug(f"[+] to {df[keep_mask].shape[0]} rows")
        return df[keep_mask]
    
    def label_distribution(self, df: pd.DataFrame):
        return df[self.label_column].value_counts()

from .preprocessor import Preprocessor
import pandas as pd
import numpy as np
import os
from utils.logging import get_logger, setup_logging
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder, QuantileTransformer, StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib

logger = get_logger(__name__)

class CIC2018Preprocessor(Preprocessor):
    def __init__(self):

        # Columns to drop
        self.columns_to_drop = [
            'Timestamp',  # Drop timestamp to avoid overfitting
            'Bwd PSH Flags', 'Bwd URG Flags',  # Constant columns
            'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg',  # Constant columns
            'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', # Constant columns
            'Fwd PSH Flags', 'Fwd URG Flags', 'FIN Flag Cnt', # Binary columns (number 0 < 8%)
            'SYN Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count',   # Binary columns (number 0 < 8%)
            'Init Bwd Win Byts', # more than 50% is negative
            'Protocol', # just one value after remove negative rows
        ]
        # Original features
        # 'Dst Port', 'Protocol', # 2
        # 'Timestamp', # dropped(1) # 3
        # 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', # 7
        # 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', # 11
        # 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', # 15
        # 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', # 18
        # 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', # 22
        # 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', # 27
        # 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', # 32
        # 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', # dropped(5) # 36
        # 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', # 40
        # 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', # 45
        # 'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt', # dropped(13) # 53
        # 'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', # 57
        # 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', # dropped(9) # 63
        # 'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', # 67
        # 'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', # 71
        # 'Active Mean', 'Active Std', 'Active Max', 'Active Min', # 75
        # 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', # 79
        # 'Label' # 80
        
        # Features to keep after dropping constants and timestamp
        self.features = [
            'Dst Port', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', # 4
            'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min', # 8
            'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min', # 12
            'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', # 16
            'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', # 20
            'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', # 25
            'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', # 30
            'Fwd Header Len', 'Bwd Header Len', # 32
            'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', # 37
            'Pkt Len Std', 'Pkt Len Var', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'ECE Flag Cnt', # 43
            'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', # 47
            'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', # 51
            'Init Fwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', # 54
            'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', # 60
            'Idle Max', 'Idle Min' # 62
        ]
        
        self.label_column = 'Label'

        self.cat_features = ['Dst Port']

        self.cont_features =  [
            'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', # 3
            'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min', # 7
            'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min', # 11
            'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', # 15
            'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', # 19
            'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', # 24
            'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', # 29
            'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', # 33
            'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', # 38
            'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', # 42
            'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', # 46
            'Init Fwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', # 49
            'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', # 55
            'Idle Max', 'Idle Min' # 57
        ]

        # Binary features (keep as 0/1, no scaling)
        self.binary_features = [
            'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'ECE Flag Cnt' # 4
        ]

        # Categorical features for OneHot encoding
        self.encoded_categorical_features = [] # 0

        # Ordinal-encoded features (for classical models); include 'Dst Port'
        # For 'Dst Port', we will fit full range [1..65535] to handle unseen values
        self.encoded_categorical_features_ordinal = self.cat_features
        
        
        # Numerical features for MinMax scaling (all features minus categorical and binary)
        self.cat_features_in_large_scale = ['Dst Port']
        self.encoded_numerical_features = self.cat_features_in_large_scale + self.cont_features

        self.numerical_features_for_approx_dedup = self.cont_features
        
        self.encoders = {
            'categorical': OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
            'numerical_minmax': MinMaxScaler(),
            'numerical_standard': StandardScaler(),
            'label': LabelEncoder(),
            'numerical_quantile_normal': QuantileTransformer(
                n_quantiles=1000,
                output_distribution='normal',
                subsample=100000,
                random_state=42,
            ),
            'numerical_quantile_uniform': QuantileTransformer(
                n_quantiles=1000,
                output_distribution='uniform',
                subsample=100000,
                random_state=42,
            ),
        }
        self.encoders_dir = "encoders/cic2018"
        self.ordered_features = self.cat_features + self.binary_features + self.cont_features
    def coerce_feature_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce feature dtypes per schema: continuous->float32, categorical/binary->Int32.
        Assumes df already filtered to selected features + Label. Uses 'coerce' so invalid
        tokens become NaN and get dropped at the missing/inf cleaning step.
        """
        logger.debug(f"[+] Coercing dtypes: cont->float32, cat/binary->Int32 for {df.shape[0]} rows")
        # Continuous features
        for col in self.cont_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

        # Categorical and binary features (nullable integer to allow NA before dropna)
        for col in (self.cat_features + self.binary_features):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int32')

        return df

    def remove_negative_numeric_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with negative values in any numeric column (no sentinel exceptions)."""
        logger.debug(f"[+] Removing negative numeric rows from {df.shape[0]} rows")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return df
        check_cols = numeric_cols
        keep_mask = df[check_cols].ge(0).all(axis=1)
        logger.debug(f"[+] to {df[keep_mask].shape[0]} rows")
        return df[keep_mask]

    def select_features_and_label(self, df: pd.DataFrame):
        """Select features and label, dropping unwanted columns"""
        # Drop unwanted columns if they exist
        columns_to_drop = self.columns_to_drop
        
        available_features = [col for col in self.features if col in df.columns]
        available_drops = [col for col in columns_to_drop if col in df.columns]
        
        if available_drops:
            # logger.debug(f"[+] Dropping columns: {available_drops}")
            df = df.drop(columns=available_drops)
        
        selected_columns = available_features + [self.label_column]
        return df[selected_columns]
    
    def drop_rows_with_zero_heavy_continuous(self, df: pd.DataFrame, threshold_frac: float = 0.5, threshold_count: int | None = None) -> pd.DataFrame:
        """Drop rows where the number of continuous features equal to 0 exceeds a threshold.

        Args:
            df: Input DataFrame containing continuous feature columns
            threshold_frac: Fraction of continuous features that being zero would mark a row for dropping
            threshold_count: Absolute number of zero-valued continuous features to trigger dropping.
                              If provided, overrides threshold_frac.

        Returns:
            Filtered DataFrame with zero-heavy rows removed.
        """
        present_cont_cols = [c for c in self.cont_features if c in df.columns]
        if not present_cont_cols:
            return df
        if threshold_count is None:
            threshold_count = int(len(present_cont_cols) * float(threshold_frac))
        try:
            zero_counts = (df[present_cont_cols] == 0).sum(axis=1)
            keep_mask = zero_counts < int(threshold_count)
            return df[keep_mask]
        except Exception as e:
            logger.warning(f"[!] Failed zero-heavy row filtering: {e}")
            return df
        
    def setup_encoders(self, df: pd.DataFrame):
        """Setup encoders for CIC-2018 features"""
        logger.debug(f"[+] Setting up encoders for CIC-2018 features")
        
        # Handle infinity values before fitting encoders
        df_clean = df.copy()
        
        # Fit encoders
        if self.encoded_categorical_features and all(col in df_clean.columns for col in self.encoded_categorical_features):
            self.encoders['categorical'].fit(df_clean[self.encoded_categorical_features])
            logger.debug(f"[+] Unique values for categorical features:")
            for col in self.encoded_categorical_features:
                logger.debug(f"    - {col}: {df_clean[col].unique()}")

        # Prepare OrdinalEncoder with categories (Protocol from data, Dst Port full range)
        if self.encoded_categorical_features_ordinal and all(col in df_clean.columns for col in self.encoded_categorical_features_ordinal):
            categories = []
            for col in self.encoded_categorical_features_ordinal:
                if col == 'Dst Port':
                    categories.append(np.arange(1, 65536, dtype=int))
                else:
                    cats = pd.Series(df_clean[col]).dropna().unique().tolist()
                    categories.append(sorted(cats))
            self.encoders['ordinal'] = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=-1)
            self.encoders['ordinal'].fit(df_clean[self.encoded_categorical_features_ordinal])
        
        if self.encoded_numerical_features and all(col in df_clean.columns for col in self.encoded_numerical_features):
            self.encoders['numerical_minmax'].fit(df_clean[self.encoded_numerical_features])
            # Fit standard scaler on numerical features
            self.encoders['numerical_standard'].fit(df_clean[self.encoded_numerical_features])
            # Fit quantile transformer independently on the same numerical features
            self.encoders['numerical_quantile_normal'].fit(df_clean[self.encoded_numerical_features])
            self.encoders['numerical_quantile_uniform'].fit(df_clean[self.encoded_numerical_features])
        
        if self.label_column in df_clean.columns:
            self.encoders['label'].fit(df_clean[self.label_column])
        
        logger.info(f"[+] Encoders fitted for CIC-2018 dataset")
        logger.info(f"[+] Categorical features: {self.encoded_categorical_features}")
        logger.info(f"[+] Binary features: {self.binary_features}")
        logger.info(f"[+] Numerical features: {len(self.encoded_numerical_features)} features")

    def load_encoders(self, encoders_dir=None, fixed_label_encoder: bool = False):
        """Load pre-trained encoders from saved files"""
        
        if encoders_dir is None:
            encoders_dir = self.encoders_dir
            
        try:
            self.encoders = {
                'categorical': joblib.load(f"{encoders_dir}/categorical_encoder.pkl"),
                'numerical_minmax': joblib.load(f"{encoders_dir}/numerical_minmax_encoder.pkl"),
                'numerical_standard': joblib.load(f"{encoders_dir}/numerical_standard_encoder.pkl"),
                'numerical_quantile_normal': joblib.load(f"{encoders_dir}/numerical_quantile_normal_encoder.pkl"),
                'numerical_quantile_uniform': joblib.load(f"{encoders_dir}/numerical_quantile_uniform_encoder.pkl"),
                'label': joblib.load(f"{encoders_dir}/label_encoder_fixed.pkl" if fixed_label_encoder else f"{encoders_dir}/label_encoder.pkl"),
                'ordinal': joblib.load(f"{encoders_dir}/ordinal_encoder.pkl")
            }
            logger.info(f"[+] CIC-2018 encoders loaded from {encoders_dir}")
            return True
        except FileNotFoundError as e:
            logger.error(f"[-] CIC-2018 encoder files not found in {encoders_dir}: {e}")
            return False
        except Exception as e:
            logger.error(f"[-] Error loading CIC-2018 encoders: {e}")
            return False

    def save_encoders(self, encoders_dir=None):
        """Save trained encoders to files"""
        
        if encoders_dir is None:
            encoders_dir = self.encoders_dir
            
        os.makedirs(encoders_dir, exist_ok=True)
        
        try:
            joblib.dump(self.encoders['categorical'], f"{encoders_dir}/categorical_encoder.pkl")
            joblib.dump(self.encoders['numerical_minmax'], f"{encoders_dir}/numerical_minmax_encoder.pkl")
            joblib.dump(self.encoders['numerical_standard'], f"{encoders_dir}/numerical_standard_encoder.pkl")
            joblib.dump(self.encoders['numerical_quantile_normal'], f"{encoders_dir}/numerical_quantile_normal_encoder.pkl")
            joblib.dump(self.encoders['numerical_quantile_uniform'], f"{encoders_dir}/numerical_quantile_uniform_encoder.pkl")
            joblib.dump(self.encoders['label'], f"{encoders_dir}/label_encoder.pkl")
            joblib.dump(self.encoders['ordinal'], f"{encoders_dir}/ordinal_encoder.pkl")
            logger.info(f"[+] CIC-2018 encoders saved to {encoders_dir}")
            return True
        except Exception as e:
            logger.error(f"[-] Error saving CIC-2018 encoders: {e}")
            return False
    
    def preprocess_encode_categorical_features(self, df: pd.DataFrame):
        """Encode categorical features using OneHot encoding"""
        if not self.encoded_categorical_features or not all(col in df.columns for col in self.encoded_categorical_features):
            return df
            
        encoder = self.encoders['categorical']
        encoded_train = encoder.transform(df[self.encoded_categorical_features])
        encoded_train_df = pd.DataFrame(
            encoded_train, 
            columns=encoder.get_feature_names_out(self.encoded_categorical_features),
            index=df.index
        )
        df = pd.concat([encoded_train_df, df], axis=1)
        df = df.drop(columns=self.encoded_categorical_features)
        return df

    def preprocess_encode_numerical_features_minmax(self, df: pd.DataFrame):
        """Encode numerical features using MinMax scaling"""
        if not self.encoded_numerical_features or not all(col in df.columns for col in self.encoded_numerical_features):
            return df
        df_clean = df.copy()
        scaler = self.encoders['numerical_minmax']
        df_clean[self.encoded_numerical_features] = scaler.transform(df_clean[self.encoded_numerical_features])
        return df_clean

    def preprocess_encode_numerical_features_quantile_normal(self, df: pd.DataFrame):
        """Encode numerical features using QuantileTransformer (independent of MinMax)."""
        if not self.encoded_numerical_features or not all(col in df.columns for col in self.encoded_numerical_features):
            return df     
        df_clean = df.copy()
        qtx = self.encoders.get('numerical_quantile_normal')
        if qtx is None:
            raise RuntimeError("Quantile encoder not initialized. Call setup_encoders() first.")
        df_clean[self.encoded_numerical_features] = qtx.transform(df_clean[self.encoded_numerical_features])
        return df_clean

    def preprocess_encode_numerical_features_standard(self, df: pd.DataFrame, exclude_large_scale_categories: bool = False):
        """Encode numerical features using StandardScaler (mean=0, std=1)."""
        logger.debug(f"[+] Encoding numerical features using StandardScaler (mean=0, std=1)")
        if not self.encoded_numerical_features or not all(col in df.columns for col in self.encoded_numerical_features):
            return df
        scaler = self.encoders.get('numerical_standard')
        if scaler is None:
            raise RuntimeError("Standard scaler not initialized. Call setup_encoders() first.")
        df_clean = df.copy()        
        df_clean[self.encoded_numerical_features] = scaler.transform(df_clean[self.encoded_numerical_features])
        if exclude_large_scale_categories:
            logger.debug(f"[+] Excluding large scale categories: {self.cat_features_in_large_scale}")
            df_clean[self.cat_features_in_large_scale] = df[self.cat_features_in_large_scale]
        return df_clean

    def preprocess_encode_numerical_features_quantile_uniform(self, df: pd.DataFrame):
        """Encode numerical features using QuantileTransformer (independent of MinMax)."""
        if not self.encoded_numerical_features or not all(col in df.columns for col in self.encoded_numerical_features):
            return df
        df_clean = df.copy()
            
        qtx = self.encoders.get('numerical_quantile_uniform')
        if qtx is None:
            raise RuntimeError("Quantile encoder not initialized. Call setup_encoders() first.")
            
        df_clean[self.encoded_numerical_features] = qtx.transform(df_clean[self.encoded_numerical_features])
        return df_clean

    def preprocess_encode_ordinal_features(self, df: pd.DataFrame):
        """Encode categorical features using Ordinal encoding"""
        if not self.encoded_categorical_features_ordinal or not all(col in df.columns for col in self.encoded_categorical_features_ordinal):
            return df
            
        # Sanitize 'Dst Port' to avoid unknown mapping (-1)
        if 'Dst Port' in df.columns:
            # Coerce to numeric
            df['Dst Port'] = pd.to_numeric(df['Dst Port'], errors='coerce')
            before_na = df['Dst Port'].isna().sum()
            # Round and clip to valid range, fill NaN to 1
            df['Dst Port'] = (
                df['Dst Port']
                .round()
                .clip(lower=1, upper=65535)
                .fillna(1)
                .astype(int)
            )
            if before_na:
                logger.debug(f"[+] Coerced and filled {before_na} NaN 'Dst Port' values before ordinal encoding")

        encoder = self.encoders['ordinal']
        df[self.encoded_categorical_features_ordinal] = encoder.transform(df[self.encoded_categorical_features_ordinal])
        df[self.encoded_categorical_features_ordinal] = df[self.encoded_categorical_features_ordinal].astype(int)
        return df
    
    def inverse_transform_ordinal_features(self, df: pd.DataFrame):
        """Inverse transform ordinal features"""
        if not self.encoded_categorical_features_ordinal or not all(col in df.columns for col in self.encoded_categorical_features_ordinal):
            return df
            
        encoder = self.encoders['ordinal']
        df[self.encoded_categorical_features_ordinal] = encoder.inverse_transform(df[self.encoded_categorical_features_ordinal])
        return df

    def preprocess_encode_binary_features(self, df: pd.DataFrame):
        """Encode binary features - keep them as 0/1"""
        for feature in self.binary_features:
            if feature in df.columns:
                # Ensure binary values (0 or 1)
                df[feature] = df[feature].astype(int)
        return df

    def preprocess_encode_label(self, df: pd.DataFrame):
        """Encode labels using LabelEncoder"""
        if self.label_column not in df.columns:
            return df
            
        encoder = self.encoders['label']
        df[self.label_column] = encoder.transform(df[self.label_column])
        return df

    def inverse_transform(self, df: pd.DataFrame, numerical_inverse: str = 'quantile_normal'):
        """
        Inverse transform the encoded DataFrame back to original format
        """
        df_inverse = df.copy()
        
        # 1. Inverse transform categorical features (one-hot -> original strings)
        if 'categorical' in self.encoders and self.encoded_categorical_features:
            encoder = self.encoders['categorical']
            # Find one-hot columns for Protocol
            onehot_columns = [col for col in df_inverse.columns if col.startswith('Protocol_')]
            
            if onehot_columns:
                # Extract one-hot data
                onehot_data = df_inverse[onehot_columns].values
                
                # Inverse transform to get original categorical values
                original_cat = encoder.inverse_transform(onehot_data)
                original_cat_df = pd.DataFrame(original_cat, 
                                             columns=['Protocol'],
                                             index=df_inverse.index)
                
                # Replace one-hot columns with original categorical columns
                df_inverse = df_inverse.drop(columns=onehot_columns)
                df_inverse = pd.concat([df_inverse, original_cat_df], axis=1)
                logger.debug(f"[+] Inverse transformed categorical features")
        
        # 2. Inverse transform numerical features (scaled -> original values)
        if self.encoded_numerical_features:
            existing_numerical = [col for col in self.encoded_numerical_features if col in df_inverse.columns]
            if existing_numerical:
                if numerical_inverse == 'quantile_normal' and 'numerical_quantile_normal' in self.encoders:
                    try:
                        df_inverse = self.inverse_transform_numerical_features_quantile_normal(df_inverse)
                        logger.debug(f"[+] Inverse transformed {len(existing_numerical)} numerical features via QuantileTransformer")
                    except Exception as e:
                        logger.warning(f"[!] Quantile inverse failed, skipping numerical inverse: {e}")
                elif numerical_inverse == 'quantile_uniform' and 'numerical_quantile_uniform' in self.encoders:
                    try:
                        df_inverse = self.inverse_transform_numerical_features_quantile_uniform(df_inverse)
                        logger.debug(f"[+] Inverse transformed {len(existing_numerical)} numerical features via QuantileTransformer (uniform)")
                    except Exception as e:
                        logger.warning(f"[!] Quantile inverse failed, skipping numerical inverse: {e}")
                elif numerical_inverse == 'standard' and 'numerical_standard' in self.encoders:
                    scaler = self.encoders['numerical_standard']
                    try:
                        df_inverse[existing_numerical] = scaler.inverse_transform(df_inverse[existing_numerical])
                        logger.debug(f"[+] Inverse transformed {len(existing_numerical)} numerical features via StandardScaler")
                    except Exception as e:
                        logger.warning(f"[!] StandardScaler inverse failed, skipping numerical inverse: {e}")
                elif numerical_inverse == 'minmax' and 'numerical_minmax' in self.encoders:
                    scaler = self.encoders['numerical_minmax']
                    try:
                        df_inverse[existing_numerical] = scaler.inverse_transform(df_inverse[existing_numerical])
                        logger.debug(f"[+] Inverse transformed {len(existing_numerical)} numerical features via MinMaxScaler")
                    except Exception as e:
                        logger.warning(f"[!] MinMax inverse failed, skipping numerical inverse: {e}")

        # 2b. Fix 'Dst Port' to be an integer in [1, 65535]
        if 'Dst Port' in df_inverse.columns:
            try:
                df_inverse['Dst Port'] = (
                    df_inverse['Dst Port']
                    .round()
                    .clip(lower=1, upper=65535)
                    .astype(int)
                )
                logger.debug("[+] Normalized 'Dst Port' to integer range [1, 65535]")
            except Exception as e:
                logger.warning(f"[!] Could not normalize 'Dst Port': {e}")
        
        # 3. Inverse transform labels (encoded -> original strings)
        if 'label' in self.encoders and self.label_column in df_inverse.columns:
            label_encoder = self.encoders['label']
            df_inverse[self.label_column] = label_encoder.inverse_transform(df_inverse[self.label_column])
            logger.debug(f"[+] Inverse transformed labels")
        
        # 4. Fix binary features - convert back to 0/1
        for feature in self.binary_features:
            if feature in df_inverse.columns:
                # Convert to binary using threshold 0.5
                df_inverse[feature] = (df_inverse[feature] > 0.5).astype(int)
                logger.debug(f"[+] Fixed binary feature {feature}: converted to 0/1")
        
        # 4b. Clamp tiny numerical magnitudes and enforce non-negativity on numerical features
        try:
            epsilon = 1e-9
            num_cols_present = [col for col in self.cont_features if col in df_inverse.columns]
            if num_cols_present:
                # Set |x| < epsilon to 0 and negatives to 0
                df_inverse[num_cols_present] = df_inverse[num_cols_present].mask(df_inverse[num_cols_present].abs() < epsilon, 0.0)
                df_inverse[num_cols_present] = df_inverse[num_cols_present].mask(df_inverse[num_cols_present] < 0, 0.0)
                logger.debug(f"[+] Clamped tiny and negative values to 0 for {len(num_cols_present)} numerical features")
                # Soft rounding to reduce floating noise without being too strict
                rounding_decimals = 8
                df_inverse[num_cols_present] = df_inverse[num_cols_present].round(rounding_decimals)
                logger.debug(f"[+] Rounded numerical features to {rounding_decimals} decimals")
        except Exception as e:
            logger.warning(f"[!] Failed to clamp tiny/negative numerical values: {e}")
        
        # 5. Reorder columns to match original order
        original_order = self.features + [self.label_column]
        
        # Only keep columns that exist
        existing_columns = [col for col in original_order if col in df_inverse.columns]
        df_inverse = df_inverse[existing_columns]
        
        logger.debug(f"[+] Inverse transform completed. Shape: {df_inverse.shape}")
        return df_inverse

    def inverse_transform_numerical_features_quantile_normal(self, df: pd.DataFrame):
        """Inverse-transform numerical features that were transformed by QuantileTransformer."""
        if 'numerical_quantile_normal' not in self.encoders or not self.encoded_numerical_features:
            return df

        df_inv = df.copy()
        qtx = self.encoders['numerical_quantile_normal']
        # Only inverse-transform columns that are present
        existing_numerical = [col for col in self.encoded_numerical_features if col in df_inv.columns]
        if existing_numerical:
            try:
                df_inv[existing_numerical] = qtx.inverse_transform(df_inv[existing_numerical])
                logger.debug(f"[+] Inverse transformed {len(existing_numerical)} numerical features via QuantileTransformer")
            except Exception as e:
                logger.warning(f"[!] Quantile inverse_transform failed: {e}")
        return df_inv

    def inverse_transform_numerical_features_quantile_uniform(self, df: pd.DataFrame):
        """Inverse-transform numerical features that were transformed by QuantileTransformer (uniform)."""
        if 'numerical_quantile_uniform' not in self.encoders or not self.encoded_numerical_features:
            return df

        df_inv = df.copy()
        qtx = self.encoders['numerical_quantile_uniform']
        # Only inverse-transform columns that are present
        existing_numerical = [col for col in self.encoded_numerical_features if col in df_inv.columns]
        if existing_numerical:
            try:
                df_inv[existing_numerical] = qtx.inverse_transform(df_inv[existing_numerical])
                logger.debug(f"[+] Inverse transformed {len(existing_numerical)} numerical features via QuantileTransformer (uniform)")
            except Exception as e:
                logger.warning(f"[!] Quantile inverse_transform failed: {e}")
        return df_inv

    def inverse_transform_label(self, df: pd.DataFrame):
        if 'label' in self.encoders and self.label_column in df.columns:
            label_encoder = self.encoders['label']
            df[self.label_column] = label_encoder.inverse_transform(df[self.label_column])
        return df

    def extract_categorical_cardinalities(self):
        features_and_cardinalities = {}
        features = self.encoders['ordinal'].feature_names_in_
        categories = self.encoders['ordinal'].categories_
        for feature, category in zip(features, categories):
            features_and_cardinalities[feature] = len(category)
        return features_and_cardinalities


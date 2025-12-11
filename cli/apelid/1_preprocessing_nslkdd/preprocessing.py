import os
import sys
import pandas as pd
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor
from utils.logging import setup_logging, get_logger
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from configs import nslkdd

logger = get_logger(__name__)


if __name__ == "__main__":
    setup_logging("INFO")
    # Load data
    preprocessor = NSLKDDPreprocessor()
    df = pd.read_csv(nslkdd.KDD_TEXT_PATH, names=preprocessor.nslkdd_columns)
    df.to_csv(nslkdd.NSLKDD_ORIGINAL_CSV_PATH, index=False)
    logger.info(f"Saved: {nslkdd.NSLKDD_ORIGINAL_CSV_PATH}")
    
    # Step 1: Map labels and select features
    df = preprocessor.map_label(df)
    df = preprocessor.select_features_and_label(df)
    
    # Step 2: Data cleaning
    df = preprocessor.remove_missing_and_inf_values(df)
    df = preprocessor.fix_duplicates(df)
    
    preprocessor.info_dataset(df)

    # print unique values of these features
    categorical_features = preprocessor.encoded_categorical_features
    logger.info(f"Categorical features: {categorical_features}")
    for feature in categorical_features:
        logger.info(f"{feature}: {df[feature].nunique()}")

    # divide per label
    dst_folder = nslkdd.CLEAN_MERGED_DATA_FOLDER
    os.makedirs(dst_folder, exist_ok=True)
    for label in nslkdd.MAJORITY_LABELS + nslkdd.MINORITY_LABELS:
        df_label = df[df[preprocessor.label_column] == label]
        df_label.to_csv(os.path.join(dst_folder, f"nslkdd_{nslkdd.get_label_name(label)}_clean_merged.csv"), index=False)
        logger.info(f"Saved: {os.path.join(dst_folder, f'nslkdd_{nslkdd.get_label_name(label)}_clean_merged.csv')}")


    # divdie per label into train and test (7:3)
    train_dst_folder = os.path.join(dst_folder, "train")
    test_dst_folder = os.path.join(dst_folder, "test")
    os.makedirs(train_dst_folder, exist_ok=True)
    os.makedirs(test_dst_folder, exist_ok=True)
    for label in nslkdd.MAJORITY_LABELS + nslkdd.MINORITY_LABELS:
        df_label = pd.read_csv(os.path.join(dst_folder, f"nslkdd_{nslkdd.get_label_name(label)}_clean_merged.csv"))
        df_label_train, df_label_test = train_test_split(df_label, test_size=0.3, random_state=42)
        df_label_train.to_csv(os.path.join(train_dst_folder, f"nslkdd_{nslkdd.get_label_name(label)}_train_clean_merged.csv"), index=False)
        df_label_test.to_csv(os.path.join(test_dst_folder, f"nslkdd_{nslkdd.get_label_name(label)}_test_clean_merged.csv"), index=False)

    # merge training dataset from each label to a dataframe and setup encoders
    train_df = pd.concat([pd.read_csv(os.path.join(train_dst_folder, f)) for f in os.listdir(train_dst_folder)])
    logger.info(f"Categorical features: {categorical_features} (at train dataset)")
    for feature in categorical_features:
        logger.info(f"{feature}: {train_df[feature].nunique()}")

    preprocessor.setup_encoders(train_df)
    preprocessor.save_encoders()


    

    

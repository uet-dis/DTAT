import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch

from utils.logging import setup_logging, get_logger

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from configs import CIC2018Resources, NSLKDDResources
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor
from preprocessing.prepare import PrepareData

from training.tabnet import TabNetModel
from training.model import evaluate_classification, confusion_matrix_from_indices, save_confusion_matrix_png


logger = get_logger(__name__)


REGISTRY = {
    'cic2018': (CIC2018Resources, CIC2018Preprocessor),
    'nslkdd': (NSLKDDResources, NSLKDDPreprocessor),
}

MODEL_TAG = 'tabnet'


def _default_inputs(res) -> tuple[str, str]:
    data_dir = res.BALENCED_DATA_FOLDER
    train_path = os.path.join(data_dir, f"{res.resources_name}_merged_train_raw_processed.csv")
    test_path = os.path.join(data_dir, f"{res.resources_name}_test_random_sample_clean_merged.csv")
    return train_path, test_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained TabNet model (standalone; ensemble integration comes after validation)")
    parser.add_argument('--resource', '-r', type=str, required=True, choices=list(REGISTRY.keys()))
    parser.add_argument('--model-in', type=str, default=None,
                        help="Path to the trained TabNet checkpoint (default: {resource}_tabnet.pth under models dir)")
    parser.add_argument('--models-dir', type=str, default=None, help="Directory containing trained models")
    parser.add_argument('--data-in', type=str, default=None, help="Input test data file (default: test sample)")
    parser.add_argument('--model-tag', type=str, default=MODEL_TAG,
                        help="Tag used for report filenames (e.g. 'tabnet_robust' when evaluating a robust checkpoint)")
    parser.add_argument('--report-out', type=str, default=None)
    parser.add_argument('--cm-out', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    ResClass, PreprocessorClass = REGISTRY[args.resource]
    res = ResClass
    pre = PreprocessorClass()

    # Get class names from configs (not from encoder)
    all_labels = res.MAJORITY_LABELS + res.MINORITY_LABELS
    config_class_names = sorted(all_labels)

    if args.models_dir is None:
        args.models_dir = os.path.join(res.DATA_FOLDER, 'models')
    if args.model_in is None:
        args.model_in = os.path.join(args.models_dir, f"{res.resources_name}_{MODEL_TAG}.pth")
    if not os.path.exists(args.model_in):
        raise SystemExit(f"Model file not found: {args.model_in}")

    if args.data_in is None:
        _, default_test = _default_inputs(res)
        args.data_in = default_test
    if not os.path.exists(args.data_in):
        raise SystemExit(f"Input data file not found: {args.data_in}")

    logger.info(f"[+] Loading test data: {args.data_in}")
    df_test = pd.read_csv(args.data_in, low_memory=False)
    logger.info(f"[+] Test data shape: {df_test.shape}")

    if not pre.load_encoders(fixed_label_encoder=True):
        raise SystemExit("Encoders not found. Train models first.")

    # Prepare test data using the unified PrepareData interface (model-agnostic)
    X_test, y_test, meta_test = PrepareData.prepare_input_data(
        df_test, pre, include_label=True
    )
    class_names = config_class_names

    logger.info(f"[+] Loading TabNet model from {args.model_in}")
    model = TabNetModel.load_model(args.model_in, device=None if args.device == 'auto' else args.device)

    y_pred_idx = model.predict(X_test)
    metrics = evaluate_classification(y_test, y_pred_idx, class_names, metric='weighted')
    logger.info(f"[+] Metrics: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}")

    # Outputs
    report_folder = f"{res.REPORT_FOLDER}/{MODEL_TAG}"
    os.makedirs(report_folder, exist_ok=True)
    report_out = args.report_out or os.path.join(report_folder, f'{args.model_tag}_metrics.json')
    cm_out = args.cm_out or os.path.join(report_folder, f'{args.model_tag}_cm.png')

    with open(report_out, 'w') as f:
        json.dump(metrics, f, indent=2)

    cm_mat = confusion_matrix_from_indices(y_test, y_pred_idx, num_classes=len(class_names))
    os.makedirs(os.path.dirname(cm_out), exist_ok=True)
    save_confusion_matrix_png(cm_mat, class_names, cm_out)

    logger.info(f"[+] Metrics JSON -> {report_out}")
    logger.info(f"[+] Confusion matrix PNG -> {cm_out}")
    logger.info(f"  {args.model_tag}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")


if __name__ == "__main__":
    main()

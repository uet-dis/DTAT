import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from utils.logging import setup_logging, get_logger


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from configs import CIC2018Resources, NSLKDDResources  # noqa: E402
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor  # noqa: E402
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor  # noqa: E402
from preprocessing.prepare import PrepareData  # noqa: E402

from training.model import evaluate_classification, confusion_matrix_from_indices, save_confusion_matrix_png  # noqa: E402
from helpers.wrapping_helper import load_model_as_wrapper


logger = get_logger(__name__)


REGISTRY = {
    'cic2018': (CIC2018Resources, CIC2018Preprocessor),
    'nslkdd': (NSLKDDResources, NSLKDDPreprocessor),
}


def _default_inputs(res) -> Tuple[str, str]:
    data_dir = res.BALENCED_DATA_FOLDER
    train_path = os.path.join(data_dir, f"{res.resources_name}_merged_train_raw_processed.csv")
    test_path = os.path.join(data_dir, f"{res.resources_name}_test_random_sample_clean_merged.csv")
    return train_path, test_path


def _find_model_files(models_dir: str, resource_name: str) -> Dict[str, str]:
    model_files: Dict[str, str] = {}
    models_dir_path = Path(models_dir)
    if not models_dir_path.exists():
        logger.warning(f"Models directory not found: {models_dir}")
        return model_files

    for model_type in ['xgb', 'dnn', 'catb', 'bagging', 'histgbm', 'gbm']:
        ext = '.pth' if model_type == 'dnn' else '.pkl'
        pattern = f"{resource_name}_{model_type}{ext}"
        model_path = models_dir_path / pattern
        if model_path.exists():
            model_files[model_type] = str(model_path)
            logger.debug(f"[+] Found {model_type} model: {model_path}")
        else:
            logger.warning(f"[!] Model not found: {model_path}")
    return model_files


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models via ART classifiers on test data")
    parser.add_argument('--mode', type=str, required=True, choices=['all', 'model'],
                        help="Evaluation mode: 'all' for all found models, 'model' for specific models")
    parser.add_argument('--models', type=str, nargs='*', default=[],
                        help="Specific models to evaluate (when mode=model)")
    parser.add_argument('--exclude-models', type=str, nargs='*', default=[],
                        help="Models to exclude from evaluation")
    parser.add_argument('--resource', '-r', type=str, required=True, choices=list(REGISTRY.keys()))
    parser.add_argument('--data-in', type=str, default=None, help="Input test data file")
    parser.add_argument('--models-dir', type=str, default=None, help="Directory containing trained models")
    parser.add_argument('--output-dir', type=str, default=None, help="Output directory for results")
    parser.add_argument('--save-cm', action='store_true', help="Save confusion matrix for each model")
    parser.add_argument('--save-mt', action='store_true', help="Save metrics for each model")
    parser.add_argument('--using-histgbm', action='store_true',
                        help="Use HistGradientBoosting (histgbm) instead of GBM (default uses gbm)")
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Setup resource and preprocessor
    ResClass, PreprocessorClass = REGISTRY[args.resource]
    res = ResClass
    pre = PreprocessorClass()

    # Class names from configs
    all_labels = res.MAJORITY_LABELS + res.MINORITY_LABELS
    class_names = sorted(all_labels)

    # Paths
    if args.models_dir is None:
        args.models_dir = os.path.join(res.DATA_FOLDER, 'models')
    if args.output_dir is None:
        args.output_dir = os.path.join(res.REPORT_FOLDER, 'classifier_eval')

    # Default test data
    if args.data_in is None:
        _, default_test = _default_inputs(res)
        args.data_in = default_test

    # Load data
    if not os.path.exists(args.data_in):
        raise SystemExit(f"Input data file not found: {args.data_in}")
    logger.info(f"[+] Loading test data: {args.data_in}")
    df_test = pd.read_csv(args.data_in, low_memory=False)
    logger.info(f"[+] Test data shape: {df_test.shape}")

    # Load encoders
    if not pre.load_encoders(fixed_label_encoder=True):
        raise SystemExit("Encoders not found. Train models first.")

    # Model files
    if args.mode == 'all':
        model_files = _find_model_files(args.models_dir, res.resources_name)
        if not model_files:
            raise SystemExit(f"No model files found in {args.models_dir}")
    else:
        if not args.models:
            raise SystemExit("Must specify --models when mode=model")
        model_files = {}
        for model_type in args.models:
            ext = '.pth' if model_type == 'dnn' else '.pkl'
            p = os.path.join(args.models_dir, f"{res.resources_name}_{model_type}{ext}")
            if os.path.exists(p):
                model_files[model_type] = p
            else:
                logger.warning(f"Model file not found: {p}")

    # Exclusions
    for ex in args.exclude_models:
        if ex in model_files:
            del model_files[ex]
            logger.info(f"[+] Excluded model: {ex}")
    # Prefer one GBM family based on flag (default: gbm)
    if args.using_histgbm:
        if 'gbm' in model_files:
            del model_files['gbm']
            logger.info("[+] using_histgbm is ON: disabled 'gbm', keeping 'histgbm' if present")
        if 'histgbm' not in model_files:
            logger.warning("[!] using_histgbm is ON but 'histgbm' model not found")
    else:
        if 'histgbm' in model_files:
            del model_files['histgbm']
            logger.info("[+] using_histgbm is OFF: disabled 'histgbm', using 'gbm' if present")
        if 'gbm' not in model_files:
            logger.warning("[!] using_histgbm is OFF but 'gbm' model not found")
    if not model_files:
        raise SystemExit("No models to evaluate after exclusions")

    logger.info(f"[+] Evaluating {len(model_files)} models with ART: {list(model_files.keys())}")

    # Prepare features using unified interface
    X_test, y_test, meta_test = PrepareData.prepare_input_data(
        df_test, pre, include_label=True
    )

    os.makedirs(args.output_dir, exist_ok=True)

    all_results: Dict[str, Dict] = {}
    for model_type, model_path in model_files.items():
        try:
            logger.info(f"=== Evaluating {model_type} ===")
            # Load wrapper
            wrapper = load_model_as_wrapper(
                model_type,
                model_path,
                num_classes=len(class_names),
                input_dim=X_test.shape[1],
                clip_values=meta_test['clip_values'],
                device=args.device,
            )
            X_eval = X_test

            # Predict via wrapper
            y_pred_idx = wrapper.predict(X_eval)

            # Metrics
            metrics = evaluate_classification(y_test, y_pred_idx, class_names)

            # Save or debug-log
            out_dir = os.path.join(args.output_dir, model_type)
            os.makedirs(out_dir, exist_ok=True)

            if args.save_mt:
                metrics_file = os.path.join(out_dir, f"{res.resources_name}_{model_type}_metrics.json")
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                logger.info(f"[+] Metrics saved to: {metrics_file}")
            else:
                logger.debug(f"{model_type} metrics: {json.dumps(metrics, indent=2)}")

            if args.save_cm:
                cm_mat = confusion_matrix_from_indices(y_test, y_pred_idx, num_classes=len(class_names))
                cm_file = os.path.join(out_dir, f"{res.resources_name}_{model_type}_cm.png")
                save_confusion_matrix_png(cm_mat, class_names, cm_file)
                logger.info(f"[+] Confusion matrix saved to: {cm_file}")
            else:
                cm_mat = confusion_matrix_from_indices(y_test, y_pred_idx, num_classes=len(class_names))
                logger.debug(f"{model_type} confusion matrix:")
                logger.debug(f"Labels: {class_names}")
                logger.debug(f"Matrix:\n{cm_mat}")

            logger.info(f"[+] {model_type} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
            all_results[model_type] = metrics

        except Exception as e:
            logger.error(f"[!] Failed to evaluate {model_type} via ART: {e}")
            continue

    # Combined summary
    if all_results:
        combined_file = os.path.join(args.output_dir, f"{res.resources_name}_all_art_metrics.json")
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"[+] Combined results saved to: {combined_file}")
        logger.info("[+] Evaluation Summary:")
        for m, mt in all_results.items():
            logger.info(f"  {m}: Accuracy={mt['accuracy']:.4f}, F1={mt['f1']:.4f}, Precision={mt['precision']:.4f}, Recall={mt['recall']:.4f}")
    else:
        logger.warning("[!] No models were successfully evaluated via ART")


if __name__ == "__main__":
    main()



import os
import sys
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from helpers.wrapping_helper import (
    load_model_as_wrapper,
    default_inputs as helper_default_inputs,
    find_classifier_model_files,
    predict_with_batching_wrapper,
    combine_weighted,
)  # noqa: E402


logger = get_logger(__name__)


REGISTRY = {
    'cic2018': (CIC2018Resources, CIC2018Preprocessor),
    'nslkdd': (NSLKDDResources, NSLKDDPreprocessor),
}

MODEL_TYPES = ['xgb', 'dnn', 'catb', 'bagging', 'histgbm', 'gbm']

# Default ensemble weights
ENSEMBLE_WEIGHTS = {
    'xgb': 0.3,
    'dnn': 0.1,
    'catb': 0.2,
    'bagging': 0.2,
    'histgbm': 0.2,
    'gbm': 0.2,
}


def _default_inputs(res) -> tuple[str, str]:
    return helper_default_inputs(res)


def _find_model_files(models_dir: str, resource_name: str) -> dict:
    return find_classifier_model_files(models_dir, resource_name)


def _predict_with_batching(wrapper, model_type: str, X: np.ndarray, num_class: int, batch_size: int) -> np.ndarray | None:
    return predict_with_batching_wrapper(wrapper, model_type, X, num_class, batch_size)


def _parallel_ensemble_predict(model_files: dict, X_test: np.ndarray, *, num_class: int, input_dim: int,
                               device: str = 'auto', batch_size: int = -1, max_workers: int = 4):
    logger.info(f"[+] Parallel ensemble prediction with {len(model_files)} models (max_workers={max_workers}, batch_size={batch_size})")

    # Prepare tasks: each worker loads its own wrapper and predicts (with optional batching)
    tasks = []
    for mt, path in model_files.items():
        tasks.append((mt, path))

    results: dict[str, np.ndarray] = {}

    def _worker(mt: str, path: str):
        # load wrapper
        wrapper= load_model_as_wrapper(mt, path, num_classes=num_class, input_dim=input_dim, device=device)
        # predict with optional batching
        return mt, _predict_with_batching(wrapper, mt, X_test, num_class, batch_size)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut2mt = {ex.submit(_worker, *t): t[0] for t in tasks}
        for fut in as_completed(fut2mt):
            mt = fut2mt[fut]
            try:
                mt_out, proba = fut.result()
                if proba is not None:
                    results[mt_out] = proba
                    logger.debug(f"[+] {mt_out} prediction completed: {proba.shape}")
                else:
                    logger.warning(f"[!] {mt_out} prediction failed")
            except Exception as e:
                logger.error(f"[!] {mt} prediction error: {e}")

    if not results:
        raise RuntimeError("No models produced valid predictions")

    # Weighted ensemble via helper
    return combine_weighted(results, num_class, ENSEMBLE_WEIGHTS)


def main():
    parser = argparse.ArgumentParser(description="Parallel ensemble prediction using ART-wrapped classifiers")
    parser.add_argument('--resource', '-r', type=str, required=True, choices=list(REGISTRY.keys()))
    parser.add_argument('--data-in', type=str, default=None, help="Input test data file")
    parser.add_argument('--models-dir', type=str, default=None, help="Directory containing trained models")
    parser.add_argument('--output-dir', type=str, default=None, help="Output directory for results")
    parser.add_argument('--exclude-models', type=str, nargs='*', default=[], help="Models to exclude from ensemble")
    parser.add_argument('--using-histgbm', action='store_true',
                        help="Use HistGradientBoosting (histgbm) instead of GBM (default uses gbm)")
    parser.add_argument('--max-workers', type=int, default=4, help="Maximum parallel workers")
    parser.add_argument('--batch-size', type=int, default=-1, help="Batch size for prediction (-1 for full dataset)")
    parser.add_argument('--save-cm', action='store_true', help="Save confusion matrix")
    parser.add_argument('--save-mt', action='store_true', help="Save metrics")
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Setup resource and preprocessor
    ResClass, PreprocessorClass = REGISTRY[args.resource]
    res = ResClass
    pre = PreprocessorClass()

    # Preload ART modules to avoid ModuleLock deadlocks in threads
    try:
        from art.estimators.classification import (
            PyTorchClassifier, XGBoostClassifier, CatBoostARTClassifier, SklearnClassifier
        )  # noqa: F401
        from art.preprocessing.standardisation_mean_std import StandardisationMeanStd  # noqa: F401
        logger.debug("[+] Preloaded ART modules for thread safety")
    except Exception as e:
        logger.debug(f"[!] ART preload skipped or not available: {e}")

    # Class names from configs
    all_labels = res.MAJORITY_LABELS + res.MINORITY_LABELS
    class_names = sorted(all_labels)

    # Paths
    if args.models_dir is None:
        args.models_dir = os.path.join(res.DATA_FOLDER, 'models')
    if args.output_dir is None:
        args.output_dir = os.path.join(res.REPORT_FOLDER, 'classifier_parallel_ensemble')

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
    model_files = _find_model_files(args.models_dir, res.resources_name)
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
        raise SystemExit("No models to ensemble after exclusions")
    logger.info(f"[+] Parallel ensemble with {len(model_files)} models: {list(model_files.keys())}")

    # Prepare features (single unified input for all wrappers)
    X_test, y_test, _ = PrepareData.prepare_input_data(
        df_test, pre, include_label=True
    )

    # Parallel ensemble prediction
    try:
        ensemble_proba, ensemble_pred = _parallel_ensemble_predict(
            model_files,
            X_test,
            num_class=len(class_names),
            input_dim=X_test.shape[1],
            device=args.device,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
        )

        # Calculate metrics
        metrics = evaluate_classification(y_test, ensemble_pred, class_names)

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        if args.save_mt:
            metrics_file = os.path.join(args.output_dir, f"{res.resources_name}_classifier_parallel_ensemble_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"[+] Metrics saved to: {metrics_file}")
        else:
            logger.debug(f"Parallel ensemble metrics: {json.dumps(metrics, indent=2)}")

        if args.save_cm:
            cm_mat = confusion_matrix_from_indices(y_test, ensemble_pred, num_classes=len(class_names))
            cm_file = os.path.join(args.output_dir, f"{res.resources_name}_classifier_parallel_ensemble_cm.png")
            save_confusion_matrix_png(cm_mat, class_names, cm_file)
            logger.info(f"[+] Confusion matrix saved to: {cm_file}")
        else:
            cm_mat = confusion_matrix_from_indices(y_test, ensemble_pred, num_classes=len(class_names))
            logger.debug("Parallel ensemble confusion matrix:")
            logger.debug(f"Labels: {class_names}")
            logger.debug(f"Matrix:\n{cm_mat}")

        logger.info(f"[+] Parallel Ensemble - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        logger.info(f"[+] Ensemble weights: {ENSEMBLE_WEIGHTS}")

    except Exception as e:
        logger.error(f"[!] Parallel ensemble prediction failed: {e}")
        raise


if __name__ == "__main__":
    main()



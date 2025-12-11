import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.logging import setup_logging, get_logger

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from configs import CIC2018Resources, NSLKDDResources
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor
from preprocessing.prepare import PrepareData

from helpers.ensemble_helper import (
    find_model_files, load_model, predict_with_batching, 
    combine_predictions, save_ensemble_results
)


logger = get_logger(__name__)


REGISTRY = {
    'cic2018': (CIC2018Resources, CIC2018Preprocessor),
    'nslkdd': (NSLKDDResources, NSLKDDPreprocessor),
}


def _predict_single_model_parallel(model_path: str, model_type: str, X_test: np.ndarray, 
                                  num_class: int, device: str = 'auto',
                                  batch_size: int = -1):
    """Predict probabilities for a single model in parallel mode with batching."""
    try:
        logger.debug(f"[+] Loading {model_type} model from {model_path}")
        
        # Load model using helper function
        model = load_model(model_path, model_type, num_class=num_class, device=device)
        
        # Predict with batching using helper function
        proba = predict_with_batching(model, model_type, X_test, num_class, batch_size)
        
        if proba is not None:
            logger.debug(f"[+] {model_type} prediction shape: {proba.shape}")
            return model_type, proba
        else:
            return model_type, None
        
    except Exception as e:
        logger.error(f"[!] Failed to predict with {model_type}: {e}")
        return model_type, None


def _parallel_ensemble_predict(model_files: dict, X_test: np.ndarray,
                              num_class: int, max_workers: int = 4, device: str = 'auto',
                              batch_size: int = -1):
    """Parallel ensemble prediction - each worker loads its own model and predicts with batching."""
    logger.info(f"[+] Parallel ensemble prediction with {len(model_files)} models (max_workers={max_workers}, batch_size={batch_size})")
    
    results = {}
    
    # Prepare tasks for parallel processing
    tasks = []
    for model_type, model_path in model_files.items():
        tasks.append((model_path, model_type, X_test, num_class, device, batch_size))
    
    # Execute parallel prediction
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_model = {
            executor.submit(_predict_single_model_parallel, *task): task[1] 
            for task in tasks
        }
        
        # Collect results
        for future in as_completed(future_to_model):
            model_type = future_to_model[future]
            try:
                model_type_result, proba = future.result()
                if proba is not None:
                    results[model_type_result] = proba
                    logger.debug(f"[+] {model_type_result} prediction completed: {proba.shape}")
                else:
                    logger.warning(f"[!] {model_type_result} prediction failed")
            except Exception as e:
                logger.error(f"[!] {model_type} prediction error: {e}")
    
    # Combine predictions using helper function
    return combine_predictions(results, X_test, num_class)


def _default_inputs(res) -> tuple[str, str]:
    data_dir = res.BALENCED_DATA_FOLDER
    train_path = os.path.join(data_dir, f"{res.resources_name}_merged_train_raw_processed.csv")
    test_path = os.path.join(data_dir, f"{res.resources_name}_test_random_sample_clean_merged.csv")
    return train_path, test_path


def main():
    parser = argparse.ArgumentParser(description="Parallel ensemble prediction using multiple trained models")
    parser.add_argument('--resource', '-r', type=str, required=True, choices=list(REGISTRY.keys()))
    parser.add_argument('--data-in', type=str, default=None, help="Input test data file")
    parser.add_argument('--models-dir', type=str, default=None, help="Directory containing trained models")
    parser.add_argument('--output-dir', type=str, default=None, help="Output directory for results")
    parser.add_argument('--exclude-models', type=str, nargs='*', default=[], 
                       help="Models to exclude from ensemble")
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
    
    # Get class names from configs (not from encoder)
    all_labels = res.MAJORITY_LABELS + res.MINORITY_LABELS
    config_class_names = sorted(all_labels)
    
    # Default paths
    if args.models_dir is None:
        args.models_dir = os.path.join(res.DATA_FOLDER, 'models')
    if args.output_dir is None:
        args.output_dir = os.path.join(res.REPORT_FOLDER, 'ensemble')
    
    # Default test data path
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
    
    # Find model files
    model_files = find_model_files(args.models_dir, res.resources_name)
    if not model_files:
        raise SystemExit(f"No model files found in {args.models_dir}")
    
    # Apply exclusions
    for exclude_model in args.exclude_models:
        if exclude_model in model_files:
            del model_files[exclude_model]
            logger.info(f"[+] Excluded model: {exclude_model}")
    
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
    
    # Prepare test data using unified PrepareData interface
    X_test, y_test, meta_test = PrepareData.prepare_input_data(
        df_test, pre, include_label=True
    )
    class_names = config_class_names
    
    # Parallel ensemble prediction
    try:
        ensemble_proba, ensemble_pred = _parallel_ensemble_predict(
            model_files, X_test,
            len(class_names), args.max_workers, args.device, args.batch_size
        )
        
        # Save results using helper function
        save_ensemble_results(
            ensemble_proba, ensemble_pred, y_test, class_names, 
            args.output_dir, res.resources_name, "parallel",
            save_metrics=args.save_mt, save_cm=args.save_cm
        )
        
    except Exception as e:
        logger.error(f"[!] Parallel ensemble prediction failed: {e}")
        raise


if __name__ == "__main__":
    main()
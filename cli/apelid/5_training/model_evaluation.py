import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path

from utils.logging import setup_logging, get_logger

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from configs import CIC2018Resources, NSLKDDResources
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor
from preprocessing.prepare import PrepareData

from helpers.ensemble_helper import load_model, find_model_files, evaluate_single_model


logger = get_logger(__name__)


REGISTRY = {
    'cic2018': (CIC2018Resources, CIC2018Preprocessor),
    'nslkdd': (NSLKDDResources, NSLKDDPreprocessor),
}

MODEL_CLASSES = ['xgb', 'dnn', 'catb', 'bagging', 'histgbm', 'gbm']

def _default_inputs(res) -> tuple[str, str]:
    """Get default train and test data paths."""
    data_dir = res.BALENCED_DATA_FOLDER
    train_path = os.path.join(data_dir, f"{res.resources_name}_merged_train_raw_processed.csv")
    test_path = os.path.join(data_dir, f"{res.resources_name}_test_random_sample_clean_merged.csv")
    return train_path, test_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on test data")
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
        args.output_dir = os.path.join(res.REPORT_FOLDER, 'evaluation')
    
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
    if args.mode == 'all':
        model_files = find_model_files(args.models_dir, res.resources_name)
        if not model_files:
            raise SystemExit(f"No model files found in {args.models_dir}")
    else:  # mode == 'model'
        if not args.models:
            raise SystemExit("Must specify --models when mode=model")
        model_files = {}
        for model_type in args.models:
            if model_type not in MODEL_CLASSES:
                raise SystemExit(f"Unknown model type: {model_type}")
            ext = '.pth' if model_type == 'dnn' else '.pkl'
            model_path = os.path.join(args.models_dir, f"{res.resources_name}_{model_type}{ext}")
            if os.path.exists(model_path):
                model_files[model_type] = model_path
            else:
                logger.warning(f"Model file not found: {model_path}")
    
    # Apply exclusions
    for exclude_model in args.exclude_models:
        if exclude_model in model_files:
            del model_files[exclude_model]
            logger.info(f"[+] Excluded model: {exclude_model}")
    
    if not model_files:
        raise SystemExit("No models to evaluate after exclusions")
    
    logger.info(f"[+] Evaluating {len(model_files)} models: {list(model_files.keys())}")
    
    # Prepare test data using unified PrepareData interface
    X_test, y_test, meta_test = PrepareData.prepare_input_data(
        df_test, pre, include_label=True
    )
    class_names = config_class_names
    
    # Results storage
    all_results = {}
    
    # Evaluate each model
    for model_type, model_path in model_files.items():
        try:
            # Load model
            logger.debug(f"[+] Loading {model_type} model from {model_path}")
            
            model = load_model(model_path, model_type, num_class=len(class_names), device=args.device)
            
            X_eval = X_test
            
            # Evaluate
            metrics = evaluate_single_model(
                model, model_type, X_eval, y_test, class_names, 
                args.output_dir, f"{res.resources_name}_{model_type}",
                save_metrics=args.save_mt, save_cm=args.save_cm
            )
            all_results[model_type] = metrics
            
        except Exception as e:
            logger.error(f"[!] Failed to evaluate {model_type}: {e}")
            continue
    
    # Save combined results
    if all_results:
        combined_file = os.path.join(args.output_dir, f"{res.resources_name}_all_models_metrics.json")
        os.makedirs(os.path.dirname(combined_file), exist_ok=True)
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"[+] Combined results saved to: {combined_file}")
        
        # Print summary
        logger.info("[+] Evaluation Summary:")
        for model_type, metrics in all_results.items():
            logger.info(f"  {model_type}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    else:
        logger.warning("[!] No models were successfully evaluated")


if __name__ == "__main__":
    main()

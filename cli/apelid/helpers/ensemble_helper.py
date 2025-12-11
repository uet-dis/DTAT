import os
import json
import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
import xgboost as xgb

from utils.logging import get_logger

from training.xgb import XGBModel
from training.dnn import DNNModel
from training.catb import CatBoostModel
from training.bagging import BaggingModel
from training.histgbm import HistGBMModel
from training.gbm import GBMModel
from training.model import evaluate_classification, confusion_matrix_from_indices, save_confusion_matrix_png


logger = get_logger(__name__)


MODEL_CLASSES = {
    'xgb': XGBModel,
    'dnn': DNNModel,
    'catb': CatBoostModel,
    'bagging': BaggingModel,
    'histgbm': HistGBMModel,
    'gbm': GBMModel,
}

# Ensemble weights
ENSEMBLE_WEIGHTS = {
    'xgb': 0.3,
    'dnn': 0.1,
    'catb': 0.2,
    'bagging': 0.2,
    'histgbm': 0.2,
    'gbm': 0.2,
}


def load_model(model_path: str, model_type: str, num_class: int = None, device: str = 'auto'):
    """Load a trained model from file using the model's own load_model method."""
    if model_type == 'dnn':
        return DNNModel.load_model(model_path, device=device)
    elif model_type == 'xgb':
        return XGBModel.load_model(model_path, num_class=num_class, device=device)
    elif model_type == 'catb':
        return CatBoostModel.load_model(model_path, num_class=num_class, device=device)
    elif model_type == 'bagging':
        return BaggingModel.load_model(model_path, num_class=num_class, device=device)
    elif model_type == 'histgbm':
        return HistGBMModel.load_model(model_path, num_class=num_class, device=device)
    elif model_type == 'gbm':
        return GBMModel.load_model(model_path, num_class=num_class, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def find_model_files(models_dir: str, resource_name: str) -> dict:
    """Find all model files in the models directory."""
    model_files = {}
    models_dir = Path(models_dir)
    
    if not models_dir.exists():
        logger.warning(f"Models directory not found: {models_dir}")
        return model_files
    
    for model_type in MODEL_CLASSES.keys():
        # Look for .pth (DNN) or .pkl (tree models)
        ext = '.pth' if model_type == 'dnn' else '.pkl'
        pattern = f"{resource_name}_{model_type}{ext}"
        model_path = models_dir / pattern
        
        if model_path.exists():
            model_files[model_type] = str(model_path)
            logger.debug(f"[+] Found {model_type} model: {model_path}")
        else:
            logger.warning(f"[!] Model not found: {model_path}")
    
    return model_files


def predict_with_batching(model, model_type: str, X_test: np.ndarray, num_class: int, batch_size: int = -1):
    """Predict with batching for large datasets using pre-loaded model."""
    if batch_size == -1 or X_test.shape[0] <= batch_size:
        # No batching needed
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)
            else:
                pred = model.predict(X_test)
                proba = np.zeros((len(pred), num_class))
                proba[np.arange(len(pred)), pred] = 1.0
            return proba
        except Exception as e:
            logger.error(f"[!] Failed to predict with {model_type}: {e}")
            return None
    
    # Batching needed
    logger.info(f"[+] Predicting {model_type} in batches of {batch_size}")
    all_proba = []
    
    for i in range(0, X_test.shape[0], batch_size):
        batch_end = min(i + batch_size, X_test.shape[0])
        X_batch = X_test[i:batch_end]
        
        logger.debug(f"[+] Processing batch {i//batch_size + 1}/{(X_test.shape[0] + batch_size - 1)//batch_size}")
        
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_batch)
            else:
                pred = model.predict(X_batch)
                proba = np.zeros((len(pred), num_class))
                proba[np.arange(len(pred)), pred] = 1.0
            
            if proba is not None:
                all_proba.append(proba)
            else:
                logger.warning(f"[!] Batch {i//batch_size + 1} failed for {model_type}")
                return None
        except Exception as e:
            logger.error(f"[!] Batch {i//batch_size + 1} error for {model_type}: {e}")
            return None
    
    if all_proba:
        final_proba = np.vstack(all_proba)
        logger.info(f"[+] {model_type} batch prediction completed: {final_proba.shape}")
        return final_proba
    else:
        return None


def combine_predictions(results: dict, X_test: np.ndarray, num_class: int, weights: dict = None):
    """Combine predictions from multiple models using weighted voting."""
    if weights is None:
        weights = ENSEMBLE_WEIGHTS
    
    if not results:
        raise RuntimeError("No models produced valid predictions")
    
    # Weighted ensemble
    logger.info(f"[+] Combining predictions from {len(results)} models")
    ensemble_proba = np.zeros((X_test.shape[0], num_class))
    total_weight = 0.0
    
    for model_type, proba in results.items():
        weight = weights.get(model_type, 0.0)
        if weight > 0:
            # Check if proba shape matches expected num_class
            if proba.shape[1] != num_class:
                logger.warning(f"[!] {model_type} has {proba.shape[1]} classes, expected {num_class}. Skipping.")
                continue
            
            ensemble_proba += weight * proba
            total_weight += weight
            logger.info(f"[+] Added {model_type} with weight {weight:.2f} (shape: {proba.shape})")
    
    if total_weight > 0:
        ensemble_proba /= total_weight
        logger.info(f"[+] Ensemble prediction completed (total weight: {total_weight:.2f})")
    else:
        logger.warning("[!] No valid weights found, using uniform voting")
        ensemble_proba = np.mean(list(results.values()), axis=0)
    
    # Convert to predictions
    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    return ensemble_proba, ensemble_pred


def save_ensemble_results(ensemble_proba: np.ndarray, ensemble_pred: np.ndarray, y_test: np.ndarray, 
                        class_names: list, output_dir: str, resource_name: str, ensemble_type: str,
                        save_metrics: bool = False, save_cm: bool = False):
    """Save ensemble results (metrics and confusion matrix)."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate metrics
    metrics = evaluate_classification(y_test, ensemble_pred, class_names)
    
    # Save metrics if requested
    if save_metrics:
        metrics_file = os.path.join(output_dir, f"{resource_name}_{ensemble_type}_ensemble_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"[+] Metrics saved to: {metrics_file}")
    else:
        logger.debug(f"{ensemble_type} ensemble metrics: {json.dumps(metrics, indent=2)}")
    
    # Save confusion matrix if requested
    if save_cm:
        cm_mat = confusion_matrix_from_indices(y_test, ensemble_pred, num_classes=len(class_names))
        cm_file = os.path.join(output_dir, f"{resource_name}_{ensemble_type}_ensemble_cm.png")
        save_confusion_matrix_png(cm_mat, class_names, cm_file)
        logger.info(f"[+] Confusion matrix saved to: {cm_file}")
    else:
        cm_mat = confusion_matrix_from_indices(y_test, ensemble_pred, num_classes=len(class_names))
        logger.debug(f"{ensemble_type} ensemble confusion matrix:")
        logger.debug(f"Labels: {class_names}")
        logger.debug(f"Matrix:\n{cm_mat}")
    
    # Print summary
    logger.info(f"[+] {ensemble_type.title()} Ensemble - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    # logger.debug(f"[+] Ensemble weights: {ENSEMBLE_WEIGHTS}")
    
    return metrics


def evaluate_single_model(model, model_type: str, X_test: np.ndarray, y_test: np.ndarray, 
                         class_names: list, output_dir: str, model_name: str, 
                         save_metrics: bool = False, save_cm: bool = False):
    """Evaluate a single model and save results."""
    logger.debug(f"=== Evaluating {model_name} ===")
    
    # Predict
    y_pred_idx = model.predict(X_test)
    
    # Calculate metrics
    metrics = evaluate_classification(y_test, y_pred_idx, class_names)
    
    # Save metrics if requested
    if save_metrics:
        metrics_file = os.path.join(output_dir, model_type, f"{model_name}_metrics.json")
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"[+] Metrics saved to: {metrics_file}")
    else:
        # Print metrics to debug log if not saving
        logger.debug(f"{model_name} metrics: {json.dumps(metrics, indent=2)}")
    
    # Generate and save confusion matrix if requested
    if save_cm:
        cm_mat = confusion_matrix_from_indices(y_test, y_pred_idx, num_classes=len(class_names))
        cm_file = os.path.join(output_dir, model_type, f"{model_name}_cm.png")
        os.makedirs(os.path.dirname(cm_file), exist_ok=True)
        save_confusion_matrix_png(cm_mat, class_names, cm_file)
        logger.info(f"[+] Confusion matrix saved to: {cm_file}")
    else:
        # Print confusion matrix to debug log if not saving
        cm_mat = confusion_matrix_from_indices(y_test, y_pred_idx, num_classes=len(class_names))
        logger.debug(f"{model_name} confusion matrix:")
        logger.debug(f"Labels: {class_names}")
        logger.debug(f"Matrix:\n{cm_mat}")
    
    # logger.info(f"[+] {model_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    
    return metrics

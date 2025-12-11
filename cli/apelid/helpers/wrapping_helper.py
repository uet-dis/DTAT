from art_classifier.dnn_classifier import DNNClassifier
from art_classifier.xgb_classifier import XGBWrapper
from art_classifier.catb_classifier import CatBoostWrapper
from art_classifier.sklearn_classifier import SkleanWrapper
from typing import Tuple, Dict
import os
import numpy as np
from pathlib import Path


def load_model_as_wrapper(model_type: str, model_path: str, *,
                           num_classes: int, input_dim: int, clip_values: Tuple[float, float], device: str,
                           wrapper_params: dict | None = None) -> Tuple[object, int]:
    """Load saved model via training.*.load_model and wrap with appropriate ART wrapper.

    Returns (wrapper, input_dim_used).
    """
    num_classes = int(num_classes)
    if model_type == 'dnn':
        # DNN loads from .pth with embedded metadata + InputNorm
        wrapper = DNNClassifier.from_checkpoint(
            ckpt_path=model_path,
            num_classes=num_classes,
            device=device,
            input_dim=input_dim,
            clip_values=clip_values,
            params=(wrapper_params or {}),
        )
        return wrapper

    if model_type == 'xgb':
        # Use ART wrapper factory loading via training model
        wrapper = XGBWrapper.from_model_path(
            path=model_path,
            num_classes=num_classes,
            input_dim=input_dim,
            device=device,
            clip_values=clip_values,
            params=(wrapper_params or {}),
        )
        return wrapper

    if model_type == 'catb':
        # Use ART wrapper factory loading via training model (no fallback)
        wrapper = CatBoostWrapper.from_model_path(
            path=model_path,
            num_classes=num_classes,
            input_dim=input_dim,
            device=device,
            clip_values=clip_values,
            params=(wrapper_params or {}),
        )
        return wrapper

    if model_type == 'bagging':
        from training.bagging import BaggingModel
        model_inst = BaggingModel.load_model(model_path, num_class=num_classes, device=device)
        wrapper = SkleanWrapper(
            model=model_inst.model,
            num_classes=num_classes,
            input_shape=(input_dim,),
            device='cpu',
            clip_values=clip_values,
            params=(wrapper_params or {}),
        )
        return wrapper

    if model_type == 'histgbm':
        from training.histgbm import HistGBMModel
        model_inst = HistGBMModel.load_model(model_path, num_class=num_classes, device=device)
        wrapper = SkleanWrapper(
            model=model_inst.model,
            num_classes=num_classes,
            input_shape=(input_dim,),
            device='cpu',
            clip_values=clip_values,
            params=(wrapper_params or {}),
        )
        return wrapper

    if model_type == 'gbm':
        from training.gbm import GBMModel
        model_inst = GBMModel.load_model(model_path, num_class=num_classes, device=device)
        wrapper = SkleanWrapper(
            model=model_inst.model,
            num_classes=num_classes,
            input_shape=(input_dim,),
            device='cpu',
            clip_values=clip_values,
            params=(wrapper_params or {}),
        )
        return wrapper

    raise ValueError(f"Unknown model type: {model_type}")


# ---------------- Shared helpers for classifier ensembles ---------------- #

MODEL_TYPES = ['xgb', 'dnn', 'catb', 'bagging', 'histgbm', 'gbm']


def default_inputs(res) -> Tuple[str, str]:
    data_dir = res.BALENCED_DATA_FOLDER
    train_path = os.path.join(data_dir, f"{res.resources_name}_merged_train_raw_processed.csv")
    test_path = os.path.join(data_dir, f"{res.resources_name}_test_random_sample_clean_merged.csv")
    return train_path, test_path


def find_classifier_model_files(models_dir: str, resource_name: str) -> Dict[str, str]:
    model_files: Dict[str, str] = {}
    models_dir_path = Path(models_dir)
    if not models_dir_path.exists():
        return model_files
    for mt in MODEL_TYPES:
        ext = '.pth' if mt == 'dnn' else '.pkl'
        p = models_dir_path / f"{resource_name}_{mt}{ext}"
        if p.exists():
            model_files[mt] = str(p)
    return model_files


def predict_one_wrapper(wrapper, model_type: str, X: np.ndarray, num_class: int):
    proba = wrapper.predict_proba(X)
    if proba is None:
        pred = wrapper.predict(X)
        proba = np.zeros((len(pred), num_class), dtype=float)
        proba[np.arange(len(pred)), pred] = 1.0
    return proba


def predict_with_batching_wrapper(wrapper, model_type: str, X: np.ndarray, num_class: int, batch_size: int):
    if batch_size == -1 or X.shape[0] <= batch_size:
        return predict_one_wrapper(wrapper, model_type, X, num_class)
    outs = []
    for i in range(0, X.shape[0], batch_size):
        batch = X[i:i + batch_size]
        proba = predict_one_wrapper(wrapper, model_type, batch, num_class)
        outs.append(proba)
    return np.vstack(outs) if outs else None


def combine_weighted(results: Dict[str, np.ndarray], num_class: int, weights: Dict[str, float]):
    if not results:
        raise RuntimeError("No models produced valid predictions")
    import numpy as _np
    first_key = next(iter(results))
    n = results[first_key].shape[0]
    ensemble = _np.zeros((n, num_class), dtype=float)
    total_w = 0.0
    for mt, proba in results.items():
        w = float(weights.get(mt, 0.0))
        if w <= 0:
            continue
        if proba.shape[1] != num_class:
            continue
        ensemble += w * proba
        total_w += w
    if total_w > 0:
        ensemble /= total_w
    else:
        ensemble = _np.mean(list(results.values()), axis=0)
    y_pred = _np.argmax(ensemble, axis=1)
    return ensemble, y_pred
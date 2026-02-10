from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from utils.logging import get_logger

# Ensure headless environments can save figures
matplotlib.use('Agg')

logger = get_logger(__name__)


def _check_gpu_status(device: str = None) -> dict:
    """Check GPU availability and capacity for any model."""
    status = {
        'available': False,
        'device_count': 0,
        'current_device': None,
        'memory_total': 0,
        'memory_allocated': 0,
        'memory_free': 0,
        'cuda_version': None
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            status['available'] = True
            status['device_count'] = torch.cuda.device_count()
            status['current_device'] = torch.cuda.current_device()
            status['cuda_version'] = torch.version.cuda
            
            if device and 'cuda' in str(device):
                try:
                    status['memory_total'] = torch.cuda.get_device_properties(0).total_memory
                    status['memory_allocated'] = torch.cuda.memory_allocated(0)
                    status['memory_free'] = status['memory_total'] - status['memory_allocated']
                except Exception:
                    pass
    except ImportError:
        pass
    
    return status


def log_gpu_status(device: str = None, model_name: str = "Model") -> None:
    """Log GPU status for any model."""
    gpu_status = _check_gpu_status(device)
    if gpu_status['available']:
        logger.debug(f"{model_name} - Available: {gpu_status['device_count']} device(s)")
        logger.debug(f"{model_name} - Current device: {gpu_status['current_device']}")
        logger.debug(f"{model_name} - CUDA version: {gpu_status['cuda_version']}")
        if gpu_status['memory_total'] > 0:
            total_gb = gpu_status['memory_total'] / (1024**3)
            allocated_gb = gpu_status['memory_allocated'] / (1024**3)
            free_gb = gpu_status['memory_free'] / (1024**3)
            logger.debug(f"GPU info: {model_name} - Memory - Total: {total_gb:.2f}GB, Allocated: {allocated_gb:.2f}GB, Free: {free_gb:.2f}GB")
    else:
        logger.warning(f"{model_name} - CUDA not available, using CPU")


class Model(ABC):
    """Minimal training interface for models used in this project.

    Required implementations:
    - fit: train the model (optionally with a validation split)
    - predict: deterministic predictions for inputs
    - save_model/load_model: persist and restore trained models

    Optional:
    - predict_proba: return class probabilities for classifiers
    - get_params/set_params: expose hyperparameters
    """

    def __init__(self, random_state: Optional[int] = 42) -> None:
        self.model: Any = None
        self.random_state: Optional[int] = random_state
        self._is_fitted: bool = False

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train the model. Implementations should set self._is_fitted = True on success."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for X."""
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Optional: return class probabilities [n_samples, n_classes] if supported."""
        return None

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Persist the trained model to `path`."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_model(cls, path: str) -> "Model":
        """Load a trained model from `path` and return an instance."""
        raise NotImplementedError

    def get_params(self) -> Dict[str, Any]:
        """Return current hyperparameters (override in concrete implementations if needed)."""
        return {}

    def set_params(self, **params: Any) -> "Model":
        """Set hyperparameters and return self (override if needed)."""
        return self


# ---------- Generic evaluation helpers (for classifiers) ----------
def evaluate_classification(
    y_true: np.ndarray,
    y_pred_idx: np.ndarray,
    class_names: List[str],
    metric: str = 'weighted',
) -> Dict[str, Any]:
    """Compute common classification metrics using label indices.

    Returns a dict with accuracy, F1/precision/recall and classes.
    """
    acc = float(accuracy_score(y_true, y_pred_idx))
    f1 = float(f1_score(y_true, y_pred_idx, average=metric))
    precision = float(precision_score(y_true, y_pred_idx, average=metric, zero_division=0))
    recall = float(recall_score(y_true, y_pred_idx, average=metric, zero_division=0))
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'classes': class_names,
    }


def confusion_matrix_from_indices(
    y_true_idx: np.ndarray,
    y_pred_idx: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Return confusion matrix (rows=true, cols=pred) using integer label indices [0..num_classes-1]."""
    labels = list(range(int(num_classes)))
    return confusion_matrix(y_true_idx, y_pred_idx, labels=labels)


def save_confusion_matrix_png(
    cm: np.ndarray,
    labels: List[str],
    out_path: str,
    *,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 150,
) -> None:
    """Save confusion matrix as a PNG using a consistent style."""
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=figsize)
    disp.plot(include_values=True, cmap='Blues', ax=ax, xticks_rotation=45, colorbar=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

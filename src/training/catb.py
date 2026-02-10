from __future__ import annotations

from typing import Optional, Any, Dict

import numpy as np
from catboost import CatBoostClassifier
from utils.logging import setup_logging, get_logger

from .model import Model, log_gpu_status, _check_gpu_status

logger = get_logger(__name__)


class CatBoostModel(Model):
    def __init__(
        self,
        *,
        num_class: int,
        params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        base = {
            'loss_function': 'MultiClass',
            'random_seed': int(random_state),
            'verbose': False,
        }
        if params:
            base.update(params)
        
        # Check and log GPU status if task_type is GPU
        if params and params.get('task_type') == 'GPU':
            log_gpu_status(None, "CatBoost")
            if not _check_gpu_status()['available']:
                logger.warning("[GPU] CatBoost GPU requested but CUDA not available, falling back to CPU")
                base['task_type'] = 'CPU'
        
        self.model = CatBoostClassifier(**base)
        self.num_class = int(num_class)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        if X_val is not None and y_val is not None:
            self.model.fit(X_train, y_train, eval_set=(X_val, y_val))
        else:
            self.model.fit(X_train, y_train)
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        return self.model.predict_proba(X)

    def save_model(self, path: str) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    @classmethod
    def load_model(cls, path: str, num_class: int = None, device: str = 'auto') -> "CatBoostModel":
        import pickle
        with open(path, 'rb') as f:
            mdl = pickle.load(f)
        inst = cls(num_class=num_class or int(getattr(mdl, 'classes_', []).__len__() or 2))
        inst.model = mdl
        inst._is_fitted = True
        
        # Try enable GPU if requested
        try:
            if device in ('cuda', 'GPU', 'auto') and device != 'cpu':
                if hasattr(inst.model, 'set_param'):
                    try:
                        inst.model.set_param({'task_type': 'GPU'})
                    except Exception:
                        pass
                logger.debug("[CatBoost] GPU requested; proceeding if supported by build")
            else:
                logger.debug("[CatBoost] Using CPU")
        except Exception:
            pass
        
        return inst



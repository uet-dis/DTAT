from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import xgboost as xgb
from .model import Model, log_gpu_status, _check_gpu_status
from utils.logging import get_logger

logger = get_logger(__name__)

class XGBModel(Model):
    def __init__(
        self,
        *,
        num_class: int,
        params: Optional[Dict[str, Any]] = None,
        num_round: int = 1200,
        early_stopping: int = 20,
        random_state: int = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        self.num_class = int(num_class)
        self.params = {
            'objective': 'multi:softprob',
            'num_class': self.num_class,
            'eta': 0.08,
            'max_depth': 8,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'lambda': 1.0,
            'alpha': 0.0,
            'tree_method': 'hist',
            'random_state': int(random_state),
            'eval_metric': 'mlogloss',
            'verbose_eval': False,
        }
        if params:
            self.params.update(params)
        
        # Check and log GPU status if tree_method is gpu_hist or device is cuda
        if self.params.get('tree_method') == 'gpu_hist':
            log_gpu_status(None, "XGBoost")
            if not _check_gpu_status()['available']:
                logger.warning("XGBoost GPU requested but CUDA not available, falling back to hist")
                self.params['tree_method'] = 'hist'
            else:
                # Convert deprecated gpu_hist to new device parameter
                logger.warning("Converting deprecated gpu_hist to device=cuda")
                self.params['tree_method'] = 'hist'
                self.params['device'] = 'cuda'
        elif self.params.get('device') == 'cuda':
            log_gpu_status(None, "XGBoost")
            if not _check_gpu_status()['available']:
                logger.warning("XGBoost GPU requested but CUDA not available, falling back to CPU")
                self.params['device'] = 'cpu'
        
        self.num_round = int(num_round)
        self.early_stopping = int(early_stopping)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'val'))
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping if len(evals) > 1 else None,
        )
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        dmat = xgb.DMatrix(X)
        proba = self.model.predict(dmat)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        dmat = xgb.DMatrix(X)
        return self.model.predict(dmat)

    def save_model(self, path: str) -> None:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    @classmethod
    def load_model(cls, path: str, num_class: int = None, device: str = 'auto') -> "XGBModel":
        import pickle
        with open(path, 'rb') as f:
            booster = pickle.load(f)
        # num_class is not directly exposed; require user to set afterwards if needed
        inst = cls(num_class=num_class or 2)
        inst.model = booster
        inst._is_fitted = True
        
        # Try enable GPU predictor if requested
        try:
            if device in ('cuda', 'GPU', 'auto') and device != 'cpu':
                if hasattr(inst.model, 'set_param'):
                    try:
                        inst.model.set_param({'device': 'cuda'})
                    except Exception:
                        pass
                logger.debug("[XGB] Using GPU (device=cuda) when available")
            else:
                logger.debug("[XGB] Using CPU predictor")
                inst.model.set_param({'device': 'cpu'})
        except Exception:
            pass
        
        return inst


 



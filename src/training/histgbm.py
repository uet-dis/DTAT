from __future__ import annotations

from typing import Optional, Any, Dict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .model import Model


class HistGBMModel(Model):
    def __init__(
        self,
        *,
        num_class: int,
        params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        kwargs: Dict[str, Any] = dict(
            learning_rate=0.1,
            max_depth=None,
            max_iter=300,
            random_state=int(random_state),
            l2_regularization=0.0,
        )
        if params:
            kwargs.update(params)
        self.model = HistGradientBoostingClassifier(**kwargs)
        self.num_class = int(num_class)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        self.model.fit(X_train, y_train)
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X).astype(int)

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
    def load_model(cls, path: str, num_class: int = None, device: str = 'auto') -> "HistGBMModel":
        import pickle
        with open(path, 'rb') as f:
            mdl = pickle.load(f)
        inst = cls(num_class=num_class or int(getattr(mdl, 'classes_', []).__len__() or 2))
        inst.model = mdl
        inst._is_fitted = True
        return inst



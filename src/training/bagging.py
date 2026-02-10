from __future__ import annotations

from typing import Optional, Any, Dict

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from .model import Model


class BaggingModel(Model):
    def __init__(
        self,
        *,
        base_estimator: Optional[Any] = None,
        n_estimators: int = 200,
        max_samples: float | int = 1.0,
        max_features: float | int = 1.0,
        random_state: int = 42,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(random_state=random_state)
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier(max_depth=None, random_state=random_state, min_samples_split=2, min_samples_leaf=1)
        kwargs: Dict[str, Any] = dict(
            estimator=base_estimator,
            n_estimators=int(n_estimators),
            max_samples=max_samples,
            max_features=max_features,
            random_state=int(random_state),
            # n_jobs=-1,
        )
        if params:
            kwargs.update(params)
        self.model = BaggingClassifier(**kwargs)

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
    def load_model(cls, path: str, num_class: int = None, device: str = 'auto') -> "BaggingModel":
        import pickle
        with open(path, 'rb') as f:
            mdl = pickle.load(f)
        inst = cls()
        inst.model = mdl
        inst._is_fitted = True
        return inst



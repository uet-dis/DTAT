import abc
from typing import Any, Dict, Optional, Tuple

import numpy as np



class AdversarialWrapper(abc.ABC):
    """Common interface for adversarial robustness wrappers using ART.

    This wrapper adapts a trained model to ART's estimator interface and exposes
    convenience methods to generate adversarial samples and evaluate robustness.
    """

    def __init__(
        self,
        *,
        model: Any,
        num_classes: int,
        input_shape: Tuple[int, ...],
        clip_values: Tuple[float, float],
        device: str | None = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.num_classes = int(num_classes)
        self.input_shape = tuple(input_shape)
        self.clip_values = tuple(clip_values)
        self.device = device or "cpu"
        self.params = params.copy() if params else {}
        self._estimator = None  # Underlying ART estimator

    @abc.abstractmethod
    def build_estimator(self) -> Any:
        """Create and return the ART estimator for the underlying model."""

    def get_estimator(self) -> Any:
        if self._estimator is None:
            self._estimator = self.build_estimator()
        return self._estimator

    # Optional preprocessing defences (Feature Squeezing)
    def _build_preprocessing_defences(self) -> Optional[list]:
        """Create preprocessing defences list based on params.

        Supported params:
          - feature_squeezing: bool (enable/disable)
          - fs_bit_depth: int (default 8)
          - fs_apply_fit: bool (default True)
          - fs_apply_predict: bool (default True)
        """
        try:
            enabled = bool(self.params.get("feature_squeezing", False) or ("fs_bit_depth" in self.params))
            if not enabled:
                return None
            try:
                from art.defences.preprocessor import FeatureSqueezing  # type: ignore
            except Exception:
                # ART not available or defence not present; skip silently
                return None
            bit_depth = int(self.params.get("fs_bit_depth", self.params.get("feature_squeezing_bit_depth", 8)))
            apply_fit = bool(self.params.get("fs_apply_fit", True))
            apply_predict = bool(self.params.get("fs_apply_predict", True))
            fs = FeatureSqueezing(
                clip_values=(0, 1),
                bit_depth=bit_depth,
                apply_fit=apply_fit,
                apply_predict=apply_predict,
            )
            return [fs]
        except Exception:
            return None

    # Basic API
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        estimator = self.get_estimator()
        # Ensure float32 to avoid dtype mismatch for torch-based estimators
        X_safe = X.astype(np.float32, copy=False)
        return estimator.predict(X_safe)  # ART returns probabilities for classifiers

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def score(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        accuracy = float((y_pred == y_true).mean())
        return {"accuracy": accuracy}

    # Params API
    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)






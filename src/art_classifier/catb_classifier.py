from typing import Any, Optional, Dict

from .art_classifier import AdversarialWrapper

from utils.logging import get_logger
from typing import Tuple
logger = get_logger(__name__)
class CatBoostWrapper(AdversarialWrapper):
    """ART wrapper for CatBoostClassifier models.

    Expects a fitted catboost.CatBoostClassifier or its sklearn-compatible wrapper.
    """

    def build_estimator(self) -> Any:  # type: ignore[override]
        try:
            from art.estimators.classification import CatBoostARTClassifier
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("ART is required for CatBoostWrapper") from exc
        # Try to log/enable GPU intent
        if self.device and (str(self.device).startswith("cuda") or self.device == "auto"):
            logger.debug("Device: CUDA (requested). Inference GPU usage depends on CatBoost build; proceeding.")
        else:
            logger.debug("Device: CPU")

        # logger.info(f"Input shape: {self.input_shape}")

        defences = self._build_preprocessing_defences() or []
        return CatBoostARTClassifier(
            model=self.model,
            nb_features=self.input_shape[0],
            clip_values=self.clip_values,
            preprocessing_defences=defences,
        )


    # ---------- Factory helpers ----------
    @classmethod
    def from_model_path(
        cls,
        path: str,
        *,
        num_classes: int,
        input_dim: int,
        clip_values: Tuple[float, float],
        device: str | None = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> "CatBoostWrapper":
        """Create a wrapper by loading a trained CatBoost model via training.CatBoostModel.

        Compatible with the current pickle-based persistence.
        """
        from training.catb import CatBoostModel

        model_inst = CatBoostModel.load_model(path, num_class=int(num_classes), device=device)
        wrapper = cls(
            model=model_inst.model,
            num_classes=int(num_classes),
            input_shape=(int(input_dim),),
            clip_values=clip_values,
            device=device,
            params=(params or {}),
        )
        return wrapper



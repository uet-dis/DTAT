from typing import Any, Optional, Dict

from .art_classifier import AdversarialWrapper

from utils.logging import get_logger
from typing import Tuple
logger = get_logger(__name__)


class XGBWrapper(AdversarialWrapper):
    """ART wrapper for XGBoost models via ART's XGBoostClassifier.

    Note: ART's XGBoostClassifier expects a sklearn-like XGBClassifier; if your
    training flow saves native Booster, consider constructing an XGBClassifier and
    loading the model with `load_model` for compatibility.
    """

    def build_estimator(self) -> Any:  # type: ignore[override]
        try:
            from art.estimators.classification import XGBoostClassifier
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("ART is required for XGBWrapper") from exc
        # Try to enable GPU prediction if requested and supported
        try:
            if self.device and (str(self.device).startswith("cuda") or self.device == "auto"):
                # sklearn API
                if hasattr(self.model, "set_params"):
                    try:
                        # xgboost>=2.0 prefers device='cuda'
                        self.model.set_params(**{"device": "cuda"})
                    except Exception:
                        pass
                # native Booster API
                if hasattr(self.model, "set_param"):
                    try:
                        self.model.set_param({"device": "cuda"})
                    except Exception:
                        pass
                
                logger.debug("Device: CUDA (requested), attempting device=cuda")
            else:
                logger.debug("Device: CPU")
        except Exception:
            # Best-effort; ignore if not applicable
            pass

        defences = self._build_preprocessing_defences() or []
        return XGBoostClassifier(
            model=self.model,
            nb_features=self.input_shape[0],
            clip_values=self.clip_values,
            nb_classes=self.num_classes,
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
    ) -> "XGBWrapper":
        """Create a wrapper by loading a trained XGB model via training.XGBModel.

        Compatible with the current pickle-based persistence.
        """
        from training.xgb import XGBModel

        model_inst = XGBModel.load_model(path, num_class=int(num_classes), device=device)
        # ART expects the underlying xgboost model (Booster) as the model argument
        wrapper = cls(
            model=model_inst.model,
            num_classes=int(num_classes),
            input_shape=(int(input_dim),),
            clip_values=clip_values,
            device=device,
            params=(params or {}),
        )
        return wrapper



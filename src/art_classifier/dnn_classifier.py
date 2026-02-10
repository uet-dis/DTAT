import os
from typing import Any, Dict, Optional, Tuple

from utils.logging import get_logger

from .art_classifier import AdversarialWrapper


logger = get_logger(__name__)


class DNNClassifier(AdversarialWrapper):
    """ART wrapper for the DNN model (PyTorch) trained under src/training/dnn.py.

    Usage:
      - Construct from an existing DNNModel instance
      - Or use `from_checkpoint` to load state_dict from .pth file
    """

    def build_estimator(self) -> Any:  # type: ignore[override]
        try:
            from art.estimators.classification import PyTorchClassifier
            import torch.nn as nn
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PyTorch and ART are required for DNNClassifier") from exc

        # Underlying torch module
        torch_model = getattr(self.model, "model", None) or self.model
        loss = getattr(self.model, "criterion", None) or nn.CrossEntropyLoss()
        optimizer = getattr(self.model, "optimizer", None)

        if self.device is None or self.device.startswith('cuda') or self.device == 'auto':
            device_type = "gpu"
            logger.debug("Device: CUDA (requested)")
        else:
            device_type = "cpu"
            logger.debug("Device: CPU")
        defences = self._build_preprocessing_defences() or []
        estimator = PyTorchClassifier(
            model=torch_model,
            loss=loss,
            optimizer=optimizer,
            input_shape=self.input_shape,
            nb_classes=self.num_classes,
            clip_values=self.clip_values,
            device_type=device_type,
            preprocessing_defences=defences,
        )
        return estimator


    # ---------- Factory helpers ----------
    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        *,
        num_classes: int,
        device: Optional[str] = None,
        input_dim: Optional[int] = None,
        clip_values: Tuple[float, float],
        dnn_hparams: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> "DNNClassifier":
        """Create a DNNClassifier by loading a full DNNModel from .pth.

        Supports new format where the .pth contains both state_dict and metadata
        required to fully reconstruct the model (including embeddings and InputNorm).
        For backward compatibility, if needed, an optional input_dim can be passed
        but will be ignored when metadata is present.
        """
        from training.dnn import DNNModel

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # Prefer new unified load that reconstructs model from metadata
        try:
            model = DNNModel.load_model(ckpt_path, device=device)
        except ValueError:
            # Legacy format fallback is no longer supported via ART wrapper,
            # as embedding/InputNorm config is required for consistent behavior.
            raise RuntimeError(
                "Legacy DNN .pth format detected. Please retrain to save metadata-enabled checkpoint."
            )

        # Infer input_dim from model if not provided
        inferred_input_dim: int
        if hasattr(model, "input_dim") and isinstance(getattr(model, "input_dim"), int):
            inferred_input_dim = int(getattr(model, "input_dim"))
        else:
            # Fallback best-effort: try to read first Linear layer in underlying torch model
            torch_model = getattr(model, "model", None) or model
            first_layer = None
            try:
                for m in getattr(torch_model, "modules")():
                    if hasattr(m, "in_features"):
                        first_layer = m
                        break
            except Exception:
                first_layer = None
            if first_layer is not None and hasattr(first_layer, "in_features"):
                inferred_input_dim = int(getattr(first_layer, "in_features"))
            else:
                if input_dim is None:
                    raise RuntimeError("Unable to infer input_dim for DNNClassifier")
                inferred_input_dim = int(input_dim)

        hp = dnn_hparams or {}
        wrapper = cls(
            model=model,
            num_classes=int(num_classes),
            input_shape=(inferred_input_dim,),
            clip_values=clip_values,
            device=device,
            params={**hp, **(params or {})},
        )
        logger.debug(f"Loaded DNN (with metadata) from {ckpt_path} on device={device}")
        return wrapper



import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from utils.logging import get_logger

from .art_classifier import AdversarialWrapper


logger = get_logger(__name__)


class _TabNetLogitsView(nn.Module):
    """Logits-only view of the tabular TabNet training module.

    ART's PyTorchClassifier requires ``forward`` to return logits, while the
    training module (``training/tabnet.py::_TabularTabNetWithEmbedding``)
    returns ``(logits, M_loss)``. The InputNorm contract for z-space attacks
    (``art_generator/inputnorm_adapter.py`` reads ``input_norm`` /
    ``cont_feature_indices`` attributes off the estimator's torch model) is
    preserved by delegating those attributes to the underlying module.
    """

    def __init__(self, tabnet_module: nn.Module) -> None:
        super().__init__()
        self._tabnet = tabnet_module
        # Delegated attributes (contract A of z-space attacks)
        self.input_norm = tabnet_module.input_norm
        self.cont_feature_indices = tabnet_module.cont_feature_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self._tabnet(x)
        return logits


class TabNetClassifier(AdversarialWrapper):
    """ART wrapper for the TabNet model (PyTorch) trained under src/training/tabnet.py.

    Usage:
      - Construct from an existing TabNetModel instance
      - Or use `from_checkpoint` to load state_dict from .pth file
    """

    def build_estimator(self) -> Any:  # type: ignore[override]
        try:
            from art.estimators.classification import PyTorchClassifier
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PyTorch and ART are required for TabNetClassifier") from exc

        # Underlying torch module (training model -> _TabularTabNetWithEmbedding)
        torch_model = getattr(self.model, "model", None) or self.model
        logits_view = _TabNetLogitsView(torch_model)
        loss = nn.CrossEntropyLoss()

        if self.device is None or self.device.startswith('cuda') or self.device == 'auto':
            device_type = "gpu"
            logger.debug("Device: CUDA (requested)")
        else:
            device_type = "cpu"
            logger.debug("Device: CPU")
        defences = self._build_preprocessing_defences() or []
        estimator = PyTorchClassifier(
            model=logits_view,
            loss=loss,
            optimizer=None,
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
        tabnet_hparams: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> "TabNetClassifier":
        """Create a TabNetClassifier by loading a full TabNetModel from .pth.

        Requires the metadata-enabled checkpoint format ({state_dict, metadata})
        written by TabNetModel.save_model.
        """
        from training.tabnet import TabNetModel

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        try:
            model = TabNetModel.load_model(ckpt_path, device=device)
        except ValueError as exc:
            raise RuntimeError(
                f"Unsupported TabNet checkpoint at {ckpt_path}: {exc}"
            ) from exc

        inferred_input_dim: int
        if hasattr(model, "input_dim") and isinstance(getattr(model, "input_dim"), int):
            inferred_input_dim = int(getattr(model, "input_dim"))
        elif input_dim is not None:
            inferred_input_dim = int(input_dim)
        else:
            raise RuntimeError("Unable to infer input_dim for TabNetClassifier")

        hp = tabnet_hparams or {}
        wrapper = cls(
            model=model,
            num_classes=int(num_classes),
            input_shape=(inferred_input_dim,),
            clip_values=clip_values,
            device=device,
            params={**hp, **(params or {})},
        )
        logger.debug(f"Loaded TabNet (with metadata) from {ckpt_path} on device={device}")
        return wrapper

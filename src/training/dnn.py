from __future__ import annotations

from typing import Optional, Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.logging import get_logger

from .model import Model, log_gpu_status, _check_gpu_status

logger = get_logger(__name__)


class InputNorm(nn.Module):
    """Per-feature input normalization for continuous features.

    Defaults:
    - Fits mean/std from the first train batch (no external stats required)
    - Applies learnable affine (gamma/beta)
    - eps=1e-6 to avoid division by zero
    """
    def __init__(self, num_features: int, *, eps: float = 1e-6, affine: bool = True, learnable_stats: bool = False) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)
        self.learnable_stats = bool(learnable_stats)

        if learnable_stats:
            self.mu = nn.Parameter(torch.zeros(self.num_features), requires_grad=True)
            self.sigma = nn.Parameter(torch.ones(self.num_features), requires_grad=True)
            self._fitted = True
        else:
            self.register_buffer('mu', torch.zeros(self.num_features))
            self.register_buffer('sigma', torch.ones(self.num_features))
            self._fitted = False

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.num_features), requires_grad=True)
            self.beta = nn.Parameter(torch.zeros(self.num_features), requires_grad=True)
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, num_features]
        if not self._fitted and self.training and x.numel() > 0:
            with torch.no_grad():
                mu = x.mean(dim=0)
                sigma = x.std(dim=0, unbiased=False)
                sigma = torch.where(sigma < self.eps, torch.ones_like(sigma), sigma)
                if isinstance(self.mu, torch.nn.Parameter):
                    self.mu.data = mu
                    self.sigma.data = sigma
                else:
                    self.mu.copy_(mu)
                    self.sigma.copy_(sigma)
                self._fitted = True

        x_norm = (x - self.mu) / (self.sigma + self.eps)
        if self.affine:
            x_norm = self.gamma * x_norm + self.beta
        return x_norm


class _TabularNetWithEmbedding(nn.Module):
    def __init__(
        self,
        *,
        cat_cardinalities: Dict[str, int],
        cat_feature_indices: List[int],
        binary_feature_indices: List[int],
        cont_feature_indices: List[int],
        num_class: int,
        embedding_dim: int = 16,
        hidden_dims: List[int] = [256, 128],
        inputnorm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.cat_cardinalities = cat_cardinalities
        self.cat_feature_indices = cat_feature_indices
        self.binary_feature_indices = binary_feature_indices
        self.cont_feature_indices = cont_feature_indices
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.inputnorm_eps = inputnorm_eps

        # Embeddings
        self.embeddings = nn.ModuleDict()
        feature_names = list(cat_cardinalities.keys())
        if len(feature_names) != len(cat_feature_indices):
            raise ValueError(f"Mismatch: {len(feature_names)} feature names but {len(cat_feature_indices)} categorical indices")
        actual_embedding_sizes: List[int] = []
        for i, _ in enumerate(cat_feature_indices):
            name = feature_names[i]
            card = cat_cardinalities[name]
            emb_size = min(embedding_dim, max(2, card // 2))
            actual_embedding_sizes.append(emb_size)
            self.embeddings[name] = nn.Embedding(card, emb_size)

        # Input dims
        embedding_total_dim = sum(actual_embedding_sizes)
        binary_dim = len(binary_feature_indices)
        cont_dim = len(cont_feature_indices)
        mlp_input_dim = embedding_total_dim + binary_dim + cont_dim

        # MLP
        layers: List[nn.Module] = []
        prev = mlp_input_dim
        for hd in hidden_dims:
            layers.extend([nn.Linear(prev, hd), nn.LayerNorm(hd), nn.ReLU(), nn.Dropout(0.2)])
            prev = hd
        layers.append(nn.Linear(prev, num_class))
        self.mlp = nn.Sequential(*layers)

        # InputNorm for continuous branch
        self.input_norm = InputNorm(cont_dim, eps=inputnorm_eps, affine=True, learnable_stats=False) if cont_dim > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Slice branches
        cat_raw = x[:, self.cat_feature_indices] if len(self.cat_feature_indices) > 0 else None
        bin_x = x[:, self.binary_feature_indices].float() if len(self.binary_feature_indices) > 0 else None
        cont_x = x[:, self.cont_feature_indices].float()

        # Embedding lookup
        feature_names = list(self.cat_cardinalities.keys())
        emb_vecs: List[torch.Tensor] = []
        for i, _ in enumerate(self.cat_feature_indices):
            name = feature_names[i]
            # Round and clamp categorical indices to valid range [0, cardinality-1]
            idx = torch.round(cat_raw[:, i])
            max_idx = self.embeddings[name].num_embeddings - 1
            idx = idx.clamp_(0, max_idx).long()
            emb = self.embeddings[name](idx)
            emb_vecs.append(emb)

        parts: List[torch.Tensor] = []
        if emb_vecs:
            parts.append(torch.cat(emb_vecs, dim=1))
        if bin_x is not None and bin_x.size(1) > 0:
            # Force binary features to {0,1}
            bin_x = torch.round(bin_x).clamp_(0, 1)
            parts.append(bin_x)
        if cont_x.size(1) > 0:
            if self.input_norm is not None:
                cont_x = self.input_norm(cont_x)
            parts.append(cont_x)
        combined = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        return self.mlp(combined)


class DNNModel(Model):
    def __init__(
        self,
        *,
        input_dim: int,
        num_class: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        max_epochs: int = 50,
        patience: int = 8,
        device: Optional[str] = None,
        random_state: int = 42,
        # Embedding-mode params (optional). If provided, model uses embedding+inputnorm
        cat_cardinalities: Optional[Dict[str, int]] = None,
        cat_feature_indices: Optional[List[int]] = None,
        binary_feature_indices: Optional[List[int]] = None,
        cont_feature_indices: Optional[List[int]] = None,
        embedding_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        # Optional precomputed stats for InputNorm
        cont_means: Optional[List[float]] = None,
        cont_stds: Optional[List[float]] = None,
        inputnorm_eps: float = 1e-6,
    ) -> None:
        super().__init__(random_state=random_state)
        self.input_dim = int(input_dim)
        self.num_class = int(num_class)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        if device is None or device == 'cuda' or device == 'auto':
            log_gpu_status("cuda", "DNN")
            if not _check_gpu_status("cuda")['available']:
                logger.warning("DNN GPU requested but CUDA not available, falling back to CPU")
                device = "cpu"
            else:
                device = "cuda"
        self.device = torch.device(device)
         
        
        torch.manual_seed(self.random_state or 42)
        # Require embedding parameters; no fallback MLP
        if cat_cardinalities is None or cat_feature_indices is None or binary_feature_indices is None or cont_feature_indices is None:
            raise ValueError("Embedding configuration required: cat_cardinalities, cat_feature_indices, binary_feature_indices, cont_feature_indices")

        hd = hidden_dims if hidden_dims is not None else [256, 128]
        self.model = _TabularNetWithEmbedding(
            cat_cardinalities=cat_cardinalities,
            cat_feature_indices=cat_feature_indices,
            binary_feature_indices=binary_feature_indices,
            cont_feature_indices=cont_feature_indices,
            num_class=self.num_class,
            embedding_dim=int(embedding_dim),
            hidden_dims=hd,
            inputnorm_eps=float(inputnorm_eps),
        ).to(self.device)
        
        
        # Store InputNorm stats for saving
        self._cont_means = cont_means
        self._cont_stds = cont_stds
        # Initialize InputNorm stats if provided
        try:
            inp = getattr(self.model, 'input_norm', None)
            if inp is not None and cont_means is not None and cont_stds is not None and len(cont_means) == len(cont_stds) and len(cont_stds) == len(cont_feature_indices):
                mu = torch.tensor(cont_means, dtype=torch.float32, device=self.device)
                sigma = torch.tensor(cont_stds, dtype=torch.float32, device=self.device)
                sigma = torch.where(sigma < inputnorm_eps, torch.ones_like(sigma), sigma)
                if isinstance(inp.mu, torch.nn.Parameter):
                    inp.mu.data = mu
                    inp.sigma.data = sigma
                else:
                    inp.mu.copy_(mu)
                    inp.sigma.copy_(sigma)
                inp._fitted = True
        except Exception as e:
            logger.warning(f"[!] Could not initialize InputNorm from provided stats: {e}")
        
        # Initialize weights with Xavier normal
        self._initialize_weights()
        
        self._best_state: Optional[dict[str, Any]] = None

    def _initialize_weights(self):
        """Initialize weights using Xavier normal initialization"""
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_loader(self, X: np.ndarray, y: Optional[np.ndarray], shuffle: bool) -> DataLoader:
        X_t = torch.from_numpy(X).float()
        if y is None:
            ds = TensorDataset(X_t)
        else:
            y_t = torch.from_numpy(y).long()
            ds = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False) if X_val is not None and y_val is not None else None
        criterion = nn.CrossEntropyLoss()
        
        # Use AdamW optimizer with weight decay
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Learning rate scheduler
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

        best_val = float('inf')
        wait = 0
        for epoch in range(self.max_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_n = 0
            for batch in train_loader:
                xb, yb = batch
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                bs = yb.size(0)
                train_loss += float(loss.item()) * bs
                train_n += bs
            
            avg_train_loss = train_loss / max(1, train_n)

            if val_loader is None:
                if (epoch + 1) % 10 == 0:
                    logger.debug(f"Epoch {epoch+1}/{self.max_epochs} - Train Loss: {avg_train_loss:.6f}")
                continue
                
            # Validation phase
            self.model.eval()
            total_loss = 0.0
            total_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    xb, yb = batch
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                    bs = yb.size(0)
                    total_loss += float(loss.item()) * bs
                    total_n += bs
            avg_val = total_loss / max(1, total_n)
            
            # Update learning rate
            scheduler.step(avg_val)
            
            if avg_val < best_val - 1e-6:
                best_val = avg_val
                self._best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
                improved = "improved"
            else:
                wait += 1
                improved = ""
                if wait >= self.patience:
                    logger.debug(f"Epoch {epoch+1}/{self.max_epochs} - Train: {avg_train_loss:.6f}, Val: {avg_val:.6f} (Best: {best_val:.6f}, Wait: {wait}/{self.patience}) {improved}")
                    logger.debug(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Log every epoch or every 10 epochs if early stopping
            if wait < self.patience and (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch {epoch+1}/{self.max_epochs} - Train: {avg_train_loss:.6f}, Val: {avg_val:.6f} (Best: {best_val:.6f}, Wait: {wait}/{self.patience}) {improved}")

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        self.model.eval()
        loader = self._make_loader(X, None, shuffle=False)
        preds: list[int] = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                pred = torch.argmax(logits, dim=1)
                preds.extend(pred.cpu().numpy().tolist())
        return np.asarray(preds, dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        self.model.eval()
        loader = self._make_loader(X, None, shuffle=False)
        probas: list[np.ndarray] = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                prob = torch.softmax(logits, dim=1)
                probas.append(prob.cpu().numpy())
        return np.concatenate(probas, axis=0) if probas else None

    def save_model(self, path: str) -> None:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        
        # Save essential metadata + InputNorm stats for stable inference
        save_data = {
            'state_dict': self.model.state_dict(),
            'metadata': {
                # Essential for model architecture
                'input_dim': self.input_dim,
                'num_class': self.num_class,
                'cat_cardinalities': getattr(self.model, 'cat_cardinalities', None),
                'cat_feature_indices': getattr(self.model, 'cat_feature_indices', None),
                'binary_feature_indices': getattr(self.model, 'binary_feature_indices', None),
                'cont_feature_indices': getattr(self.model, 'cont_feature_indices', None),
                'embedding_dim': getattr(self.model, 'embedding_dim', 16),
                'hidden_dims': getattr(self.model, 'hidden_dims', [256, 128]),
                'inputnorm_eps': getattr(self.model, 'inputnorm_eps', 1e-6),
                # InputNorm stats from training for stable inference
                'cont_means': getattr(self, '_cont_means', None),
                'cont_stds': getattr(self, '_cont_stds', None),
            }
        }
        logger.debug(f"metadata: {save_data['metadata']}")
        torch.save(save_data, path)

    @classmethod
    def load_model(cls, path: str, device: Optional[str] = None) -> "DNNModel":
        """Load a complete DNN model from file with metadata."""
        save_data = torch.load(path, map_location='cpu' if device == 'cpu' else None)
        
        if isinstance(save_data, dict) and 'state_dict' in save_data and 'metadata' in save_data:
            # New format with metadata
            metadata = save_data['metadata']
            state_dict = save_data['state_dict']
            
            # Create model with saved metadata + InputNorm stats
            model = cls(
                input_dim=metadata['input_dim'],
                num_class=metadata['num_class'],
                device=device,
                cat_cardinalities=metadata['cat_cardinalities'],
                cat_feature_indices=metadata['cat_feature_indices'],
                binary_feature_indices=metadata['binary_feature_indices'],
                cont_feature_indices=metadata['cont_feature_indices'],
                embedding_dim=metadata.get('embedding_dim', 16),
                hidden_dims=metadata.get('hidden_dims', [256, 128]),
                inputnorm_eps=metadata.get('inputnorm_eps', 1e-6),
                # Restore InputNorm stats from training for stable inference
                cont_means=metadata.get('cont_means'),
                cont_stds=metadata.get('cont_stds'),
            )
            
            # Load state dict
            model.model.load_state_dict(state_dict)
            model._is_fitted = True
            return model
        else:
            # Legacy format - only state_dict
            raise ValueError("Legacy .pth format detected. Please retrain model to get new format with metadata.")



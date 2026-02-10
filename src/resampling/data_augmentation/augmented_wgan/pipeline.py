import os
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from utils.logging import get_logger
from resampling.data_augmentation.augmented_wgan.wgan import WGAN


logger = get_logger(__name__)


## NOTE: The pipeline now expects inputs already encoded. No auto-detect or re-encode.


# (Encoded dedup helpers removed)


def train_wgan_with_critic(encoded_train: pd.DataFrame,
                           feat_cols: Iterable[str],
                           pre,
                           benign_encoded: Optional[pd.DataFrame],
                           device: str = 'auto',
                           use_gp: bool = True,
                           critic_epochs: int = 60,
                           wgan_iterations: int = 10000,
                           d_iter: int = 5,
                           critic_id: Optional[int] = None) -> WGAN:
    x_dim = len(list(feat_cols))
    wgan = WGAN(x_dim=x_dim, device=device, use_gp=use_gp, use_critic_loss=True, lambda_critic=0.5)

    # Critic train set (no internal sampling; expect pre-sampled opposite set)
    if benign_encoded is not None and not benign_encoded.empty:
        pos = encoded_train.copy()
        neg = benign_encoded.copy()
        critic_df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=1)
        logger.info(f"[+] Critic set: pos={len(pos)}, neg={len(neg)} (no internal sampling)")
    else:
        critic_df = encoded_train.copy()

    critic_loader, _ = wgan.prepare_data(critic_df, use_label_column=True, critic_id=critic_id)
    wgan.train_critic(critic_loader, epochs=critic_epochs)

    attack_loader, _ = wgan.prepare_data(encoded_train, use_label_column=False, critic_id=critic_id)
    wgan.train_wgan(attack_loader, iterations=wgan_iterations, d_iter=d_iter, save_interval=1000)
    return wgan


@dataclass
class AugmentOptions:
    # Data/critic
    use_benign_for_critic: bool = True
    critic_epochs: int = 60
    wgan_iterations: int = 10000
    d_iter: int = 5
    use_gp: bool = True

    # Generation/selection
    accept_rate: float = 0.2
    request_multiplier: float = 3.0
    max_rounds: int = 40

    # (Post-filtering removed)
    min_precision: float = 0.95

    # Trim/fill
    trim_to_need: bool = True
    use_final_fill: bool = True


def _build_critic_df(encoded_train: pd.DataFrame, benign_encoded: Optional[pd.DataFrame]) -> pd.DataFrame:
    if benign_encoded is not None and not benign_encoded.empty:
        pos = encoded_train.copy()
        neg = benign_encoded.copy()
        return pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=1)
    return encoded_train.copy()


# (Post-filtering utilities removed)


# (Encoded dedup removed)


def trim_to_need(accepted_df: pd.DataFrame, need: int) -> pd.DataFrame:
    """Trim a dataframe to the desired number of rows using a simple policy.

    Prefer deterministic head() to keep reproducibility.
    """
    if len(accepted_df) <= need:
        return accepted_df
    try:
        return accepted_df.head(need)
    except Exception:
        return accepted_df.iloc[:need]


def final_fill(wgan: WGAN,
                feat_cols: Iterable[str],
                accepted_enc: pd.DataFrame,
                deficit: int,
                accept_rate: float) -> pd.DataFrame:
    """Generate extra encoded samples to fill remaining deficit.

    Uses a simple policy: request scaled by deficit and accept_rate, then take head(deficit).
    """
    request = int(min(max(deficit * 6, (deficit / max(accept_rate, 1e-3)) * 6), 80000))
    gen_extra = wgan.generate_samples(request, critic_threshold=None, accept_rate=accept_rate)
    gen_extra_df = pd.DataFrame(gen_extra, columns=list(feat_cols))
    if len(gen_extra_df) > 0:
        gen_extra_df = gen_extra_df.head(deficit)
        accepted_enc = pd.concat([accepted_enc, gen_extra_df], ignore_index=True)
    return accepted_enc


def generate_encoded(wgan: WGAN,
                     num_samples: int,
                     feat_cols: Iterable[str],
                     accept_rate: float) -> pd.DataFrame:
    """Generate an encoded batch (features only) as a DataFrame with feat_cols."""
    arr = wgan.generate_samples(num_samples, critic_threshold=None, accept_rate=accept_rate)
    return pd.DataFrame(arr, columns=list(feat_cols))


def generate_augmented_samples(*args, **kwargs):
    """Deprecated: Orchestration moved to CLI. Kept for backward compatibility."""
    raise NotImplementedError("Use helpers: train_wgan_with_critic, generate_encoded, trim_to_need, final_fill")



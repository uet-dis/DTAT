from art.attacks.evasion import SaliencyMapMethod
from art.config import ART_NUMPY_DTYPE
from art.utils import check_and_transform_label_format
from tqdm.auto import trange
from art_generator.attack_generator import AttackGenerator
from art_generator.inputnorm_adapter import (
    InputNormStats,
    build_zspace_torch_estimator,
    data_zspace_clip,
)
from art_generator.targets import masked_second_most_likely
import numpy as np
import pandas as pd
from utils.logging import get_logger
from contextlib import contextmanager
import signal
logger = get_logger(__name__)


class _DataBoxJSMA(SaliencyMapMethod):
    """ART SaliencyMapMethod 1.20.1 with a per-feature data-derived clip box.

    Why a copy: ART's pairwise update assumes a SCALAR clip value —
    ``clip_func(clip_value, tmp_batch[..., feat] + theta)`` selects one cell
    per row (saliency_map.py:154-160), so a per-feature array box of shape
    (n_features,) fails to broadcast against the (n_active,) selection. The
    rest of the attack (search-space init ``batch < clip_max`` and the
    search-space reset ``batch == clip_value``) broadcasts correctly with
    per-feature arrays and is kept verbatim. This subclass indexes the box
    edge by the SELECTED feature of each row (``clip_value[feat_ind[:, k]]``)
    — semantics identical to the scalar case, elementwise clipping only, no
    width scaling. Everything else is verbatim from ART 1.20.1 generate()
    (lines 92-204).
    """

    def generate(self, x: np.ndarray, y: np.ndarray | None = None, **kwargs) -> np.ndarray:
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

        # Initialize variables
        dims = list(x.shape[1:])
        self._nb_features = np.prod(dims)
        x_adv = np.reshape(x.astype(ART_NUMPY_DTYPE), (-1, self._nb_features))
        preds = np.argmax(self.estimator.predict(x, batch_size=self.batch_size), axis=1)

        # Determine target classes for attack (y is always provided by our
        # generator — masked second-most-likely targets; the random fallback
        # is kept verbatim for API fidelity but is unreachable here)
        if y is None:
            # Randomly choose target from the incorrect classes for each sample
            from art.utils import random_targets

            targets = np.argmax(random_targets(preds, self.estimator.nb_classes), axis=1)
        else:
            if self.estimator.nb_classes == 2 and y.shape[1] == 1:  # pragma: no cover
                raise ValueError(
                    "This attack has not yet been tested for binary classification with a single output classifier."
                )

            targets = np.argmax(y, axis=1)

        # Compute perturbation with implicit batching
        for batch_id in trange(
            int(np.ceil(x_adv.shape[0] / float(self.batch_size))), desc="JSMA", disable=not self.verbose
        ):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_adv[batch_index_1:batch_index_2]

            # Main algorithm for each batch
            # Initialize the search space; optimize to remove features that can't be changed
            if self.estimator.clip_values is not None:
                search_space = np.zeros(batch.shape)
                clip_min, clip_max = self.estimator.clip_values
                # Cast the box to the batch dtype (float32): the search-space
                # reset below (search_space[batch == clip_value] = 0) relies
                # on EXACT equality between the stored clipped value and the
                # box edge. With a float64 edge, min(edge, v+theta) rounds
                # down when stored into the float32 batch, equality never
                # holds, the edge feature is re-selected forever and the
                # while loop never terminates (measured: spin at ~3000 CPU-min
                # on NSL). float32 edges make min() land exactly on the edge.
                clip_min = clip_min.astype(ART_NUMPY_DTYPE)
                clip_max = clip_max.astype(ART_NUMPY_DTYPE)
                if self.theta > 0:
                    search_space[batch < clip_max] = 1
                else:  # pragma: no cover
                    search_space[batch > clip_min] = 1
            else:
                search_space = np.ones(batch.shape)

            # Get current predictions
            current_pred = preds[batch_index_1:batch_index_2]
            target = targets[batch_index_1:batch_index_2]
            active_indices = np.where(current_pred != target)[0]
            all_feat = np.zeros_like(batch)

            while active_indices.size != 0:
                # Compute saliency map
                feat_ind = self._saliency_map(
                    np.reshape(batch, [batch.shape[0]] + dims)[active_indices],
                    target[active_indices],
                    search_space[active_indices],
                )

                # Update used features
                all_feat[active_indices, feat_ind[:, 0]] = 1
                all_feat[active_indices, feat_ind[:, 1]] = 1

                # Apply attack with clipping
                if self.estimator.clip_values is not None:
                    # Prepare update depending on theta
                    if self.theta > 0:
                        clip_func, clip_value = np.minimum, clip_max  # type: ignore
                    else:  # pragma: no cover
                        clip_func, clip_value = np.maximum, clip_min  # type: ignore

                    # Update adversarial examples — box edge indexed by the
                    # row's selected feature (per-feature box support)
                    tmp_batch = batch[active_indices]
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 0]] = clip_func(
                        clip_value[feat_ind[:, 0]],
                        tmp_batch[np.arange(len(active_indices)), feat_ind[:, 0]] + self.theta,
                    )
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 1]] = clip_func(
                        clip_value[feat_ind[:, 1]],
                        tmp_batch[np.arange(len(active_indices)), feat_ind[:, 1]] + self.theta,
                    )
                    batch[active_indices] = tmp_batch

                    # Remove indices from search space if max/min values were reached
                    search_space[batch == clip_value] = 0

                # Apply attack without clipping
                else:
                    tmp_batch = batch[active_indices]
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 0]] += self.theta
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 1]] += self.theta
                    batch[active_indices] = tmp_batch

                # Recompute model prediction
                current_pred = np.argmax(
                    self.estimator.predict(np.reshape(batch, [batch.shape[0]] + dims)),
                    axis=1,
                )

                # Update active_indices
                active_indices = np.where(
                    (current_pred != target)
                    * (np.sum(all_feat, axis=1) / self._nb_features <= self.gamma)
                    * (np.sum(search_space, axis=1) > 0)
                )[0]

            x_adv[batch_index_1:batch_index_2] = batch

        x_adv = np.reshape(x_adv, x.shape)

        return x_adv


class JSMAAttackGenerator(AttackGenerator):
    """JSMA in InputNorm coordinates (scale-coherent unit step).

    ART SaliencyMapMethod moves each selected pair of features by +/- theta
    per step (saliency_map.py:147-152) and derives its search space from the
    estimator's clip_values (saliency_map.py:118-124). In RAW tabular space
    a scalar theta is meaningless: features span ~1e8 (dst_bytes) down to
    [0,1] (rate features), so one theta is simultaneously a no-op (1e8-scale
    features: measured 98.6% unchanged on NSL v1) and an overshoot (rate
    features).

    Fix: the attack estimator wraps the SAME torch model (shared weights) in
    InputNorm z coordinates through the adapter from inputnorm_adapter, so
    theta=1.0 means one SIGMA per step on every feature. The per-feature
    valid-domain box from data_zspace_clip is assigned to the estimator's
    ``_clip_values`` before each generate call; ART JSMA then (a) excludes
    features already at a box edge from the search space and (b) clips every
    step at the box edge (saliency_map.py:147-152) — elementwise clipping,
    no width scaling (unlike DeepFool's step inflation, deepfool.py:174-182),
    so the z box is safe to hand to the attack directly. With the box's lower
    edge at the per-feature data min, a positive-theta targeted JSMA can
    never park a value below raw 0.

    Targets are the masked second-most-likely class computed with the ADAPTER
    estimator on z inputs (predict == raw predict, argmax preserved — see
    inputnorm_adapter validation), so the attack and its targets see the same
    function. The generator converts raw -> z before the attack and z_adv ->
    raw after, so callers keep working on raw-space DataFrames.
    """

    def __init__(self, classifier, generator_params: dict = None):
        super().__init__(classifier, generator_params)
        stats = InputNormStats.from_estimator(self.classifier)
        if stats is None or not stats.cont_idx:
            raise RuntimeError(
                "JSMA requires the tabular DNN with an input_norm module "
                "(DNN-only attack, InputNorm coordinates)."
            )
        self.stats = stats
        logger.info(
            "[+] JSMA InputNorm space: %d continuous features, "
            "denom range [%.4g, %.4g]",
            len(stats.cont_idx), stats.denom.min(), stats.denom.max(),
        )

        attack_params = {
            "theta": 1.0,  # one sigma per step (was 0.02 raw units)
            "gamma": 0.1,  # low budget by design: <= ~4 modified features
            "batch_size": 64,
            "verbose": True,
        }
        attack_params = self.update_generator_params(attack_params)
        attack_estimator = build_zspace_torch_estimator(self.classifier, stats)
        self.attack = _DataBoxJSMA(classifier=attack_estimator, **attack_params)

    def generate(self, x: np.ndarray, y: np.ndarray, input_metadata: dict, mutate_indices: list[int] = [], **kwargs) -> pd.DataFrame:
        """Generate adversarial examples using JSMA with optional batching.

        Args:
            x: Input samples, shape (N, F)
            y: True labels, required by our interface
            input_metadata: Dict containing at least 'feature_names' and 'label_column'
            mutate_indices: Indices to keep unchanged in outputs (e.g., categorical/binary)
            **kwargs: Optional 'batch_size' to process in chunks

        Returns:
            DataFrame of adversarial samples with original labels appended.
        """
        if x is None or x.size == 0:
            return pd.DataFrame(columns=(input_metadata.get('feature_names', []) + [input_metadata.get('label_column', 'label')]))

        # Attack in InputNorm coordinates, convert back to raw units
        x_in = self.stats.to_z(x)
        # Per-feature valid-domain box derived from the data itself (lower
        # edge = per-feature data min -> raw >= 0; see inputnorm_adapter).
        # Dataset-wide box: covers every row, all batch chunks alike.
        raw_clip = getattr(self.classifier, "clip_values", None)
        raw_clip_max = raw_clip[1] if raw_clip is not None else None
        attack_estimator = self.attack.estimator
        attack_estimator._clip_values = data_zspace_clip(self.stats, x_in, raw_clip_max=raw_clip_max)

        # Target selection (deterministic): target = highest-scoring class other than the true label,
        # with the true label masked out. SaliencyMapMethod treats y as ATTACK
        # TARGETS (not ground truth), so passing ground-truth y previously made
        # the attack increase confidence in the true label (ASR = 0). Masking
        # guarantees the target is never the true label, even for samples the
        # model already mispredicts. The second most-likely class is the easiest
        # target (shortest decision boundary), i.e. an upper bound on evasion
        # strength, and keeps the attack RNG-free.
        # `y` remains the ground-truth labels, used only to tag output rows.
        logger.info("[+] JSMA target selection: masked second-most-likely class (deterministic)")

        batch_size = kwargs.get('batch_size', -1)
        max_retries = int(kwargs.get('max_retries', 3))
        placeholder_mode = kwargs.get('placeholder', 'original')  # 'original' or 'drop'
        timeout_seconds = int(kwargs.get('timeout', -1))  # -1 = no timeout
        verbose_every = int(kwargs.get('verbose', 0))  # log progress every N batches; 0 = disable
        n_samples = x.shape[0]

        if batch_size is None or batch_size <= 0 or batch_size >= n_samples:
            logger.info("[+] Generating adversarial samples with JSMA (single batch)")
            targets = masked_second_most_likely(x_in, attack_estimator, y)
            x_adv_norm = self.attack.generate(x=x_in, y=targets)
            x_adv = self.stats.from_z(x_adv_norm)
            if mutate_indices:
                x_adv[:, mutate_indices] = x[:, mutate_indices]
            if verbose_every and verbose_every > 0:
                logger.info("[JSMA] Progress: 1/1 batches, failed=0")
        else:
            logger.info(f"[+] Generating adversarial samples with JSMA in batches of {batch_size}")
            adv_parts = []
            y_parts = []
            total_batches = (n_samples + batch_size - 1) // batch_size
            processed_batches = 0
            failed_batches = 0
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                xb = x[start:end]
                yb = y[start:end]
                xb_in = x_in[start:end]
                targets_b = masked_second_most_likely(xb_in, attack_estimator, yb)
                # retries
                attempt = 0
                last_err = None
                xb_adv = None
                batch_failed = False
                while attempt < max_retries:
                    try:
                        if timeout_seconds is not None and timeout_seconds > 0:
                            with _timeout(timeout_seconds):
                                xb_adv = self.attack.generate(x=xb_in, y=targets_b)
                        else:
                            xb_adv = self.attack.generate(x=xb_in, y=targets_b)
                        break
                    except Exception as e:
                        last_err = e
                        attempt += 1
                        logger.warning(f"Batch {start}:{end} attempt {attempt}/{max_retries} failed: {e}")
                if xb_adv is None:
                    batch_failed = True
                    if placeholder_mode == 'drop':
                        logger.error(f"Dropping failed batch {start}:{end} after {max_retries} retries")
                        failed_batches += 1
                        processed_batches += 1
                        if verbose_every and verbose_every > 0 and (processed_batches % verbose_every == 0 or processed_batches == total_batches):
                            logger.info(f"[JSMA] Progress: {processed_batches}/{total_batches} batches, failed={failed_batches}")
                        continue
                    else:  # 'original'
                        logger.error(f"Using original samples for failed batch {start}:{end} after {max_retries} retries")
                        xb_adv = xb
                        failed_batches += 1
                if mutate_indices:
                    xb_adv[:, mutate_indices] = xb[:, mutate_indices]
                adv_parts.append(xb_adv)
                y_parts.append(yb)
                processed_batches += 1
                if verbose_every and verbose_every > 0 and (processed_batches % verbose_every == 0 or processed_batches == total_batches):
                    logger.info(f"[JSMA] Progress: {processed_batches}/{total_batches} batches, failed={failed_batches}")
            if not adv_parts:
                # all batches dropped
                return pd.DataFrame(columns=(input_metadata.get('feature_names', []) + [input_metadata.get('label_column', 'label')]))
            x_adv = np.concatenate(adv_parts, axis=0)
            y = np.concatenate(y_parts, axis=0) if y_parts else y[:0]
            x_adv = self.stats.from_z(x_adv)

        # from_z (z->raw in float32) can land a hair below the raw data
        # minimum (float roundoff at the box edge); clamp to the per-feature
        # RAW data minimum computed directly on x (roundoff-free).
        x_adv = np.maximum(x_adv, x.min(axis=0))
        y_arr = y.reshape(-1) if y.ndim > 1 else y
        feature_names = input_metadata['feature_names']
        df = pd.DataFrame(x_adv, columns=feature_names)
        df[input_metadata['label_column']] = y_arr
        return df


@contextmanager
def _timeout(seconds: int):
    if seconds is None or seconds <= 0:
        yield
        return
    def _handle(signum, frame):
        raise TimeoutError("JSMA generation timed out")
    old = signal.signal(signal.SIGALRM, _handle)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
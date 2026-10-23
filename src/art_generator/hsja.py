from art.attacks.evasion import HopSkipJump
from art_generator.attack_generator import AttackGenerator
from art_generator.inputnorm_adapter import (
    build_zspace_estimator_for,
    data_zspace_clip,
    set_attackgen_predict_nthread,
    set_zspace_clip_box,
)
import numpy as np
import pandas as pd
from utils.logging import get_logger
from contextlib import contextmanager
import signal
logger = get_logger(__name__)


class _LocalInitHopSkipJump(HopSkipJump):
    """HopSkipJump with LOCAL initial adversarial draws.

    ART 1.20.1 draws the initial adversarial point uniform over the WHOLE
    clip box (hop_skip_jump.py:309/344) — an image-domain habit that is
    catastrophic for tabular features with a global raw clip range of
    0..4e8 (the DeepFool-v2-class destruction), and in z coordinates still
    wastes draws in irrelevant regions (init failures -> unchanged rows).

    Faithful to the HopSkipJump paper instead: start from a point CLOSE to
    the original sample — x + uniform(-1, 1) * init_radius (radius in the
    attack's coordinate units, i.e. sigma for z-space attacks) — then the
    inherited binary search pulls it to the nearest boundary crossing.
    Everything else is copied verbatim from ART 1.20.1 _init_sample,
    including the unseeded RandomState (covered by generate_adv --seed).
    """

    def __init__(self, classifier, init_radius: float = 2.0, **kwargs):
        super().__init__(classifier, **kwargs)
        self._init_radius = float(init_radius)

    def _init_sample(self, x, y, y_p, init_pred, adv_init, mask, clip_min, clip_max):
        nprd = np.random.RandomState()
        initial_sample = None

        if self.targeted:
            # Attack satisfied
            if y == y_p:
                return None

            # Attack unsatisfied yet and the initial image satisfied
            if adv_init is not None and init_pred == y:
                return adv_init.astype(np.float32), init_pred

            # Attack unsatisfied yet and the initial image unsatisfied
            for _ in range(self.init_size):
                # Draw near the original sample, then CLAMP into the
                # per-feature valid-domain box: the init point's
                # adversariality must hold AS CLIPPED. An out-of-box anchor
                # makes the binary search return an unreachable boundary and
                # the epsilon step search spin forever (observed: success
                # never true, candidate stuck at clip(current) for 100k+
                # halvings on the NSL smoke).
                random_img = np.clip(
                    (x + self._init_radius * nprd.uniform(low=-1.0, high=1.0, size=x.shape).astype(x.dtype)).astype(
                        x.dtype
                    ),
                    clip_min,
                    clip_max,
                )

                if mask is not None:
                    random_img = random_img * mask + x * (1 - mask)

                random_class = np.argmax(
                    self.estimator.predict(np.array([random_img]), batch_size=self.batch_size),
                    axis=1,
                )[0]

                if random_class == y:
                    # Binary search to reduce the l2 distance to the original image
                    random_img = self._binary_search(
                        current_sample=random_img,
                        original_sample=x,
                        target=y,
                        norm=2,
                        clip_min=clip_min,
                        clip_max=clip_max,
                        threshold=0.001,
                    )
                    initial_sample = random_img, random_class

                    # Per-sample init message: downgraded to DEBUG (see the
                    # untargeted branch below for rationale).
                    logger.debug("Found initial adversarial image for targeted attack.")
                    break
            else:
                logger.warning("Failed to draw a random image that is adversarial, attack failed.")

        else:
            # The initial image satisfied
            if adv_init is not None and init_pred != y_p:
                return adv_init.astype(np.float32), y_p

            # The initial image unsatisfied
            for _ in range(self.init_size):
                # Draw near the original sample, then CLAMP into the
                # per-feature valid-domain box: the init point's
                # adversariality must hold AS CLIPPED. An out-of-box anchor
                # makes the binary search return an unreachable boundary and
                # the epsilon step search spin forever (observed: success
                # never true, candidate stuck at clip(current) for 100k+
                # halvings on the NSL smoke).
                random_img = np.clip(
                    (x + self._init_radius * nprd.uniform(low=-1.0, high=1.0, size=x.shape).astype(x.dtype)).astype(
                        x.dtype
                    ),
                    clip_min,
                    clip_max,
                )

                if mask is not None:
                    random_img = random_img * mask + x * (1 - mask)

                random_class = np.argmax(
                    self.estimator.predict(np.array([random_img]), batch_size=self.batch_size),
                    axis=1,
                )[0]

                if random_class != y_p:
                    # Binary search to reduce the l2 distance to the original image
                    random_img = self._binary_search(
                        current_sample=random_img,
                        original_sample=x,
                        target=y_p,
                        norm=2,
                        clip_min=clip_min,
                        clip_max=clip_max,
                        threshold=0.001,
                    )
                    initial_sample = random_img, y_p

                    # Per-sample init message: downgraded to DEBUG so the
                    # per-row progress bar stays readable (fires once per
                    # sample; visible again with --log-level DEBUG).
                    logger.debug("Found initial adversarial image for untargeted attack.")
                    break
            else:
                logger.warning("Failed to draw a random image that is adversarial, attack failed.")

        return initial_sample


class _GuardedHSJA(_LocalInitHopSkipJump):
    """HopSkipJump with a blocked-movement guard in the epsilon step search.

    ART 1.20.1 ``_attack`` searches the step size with an UNGUARDED halving
    loop (hop_skip_jump.py:441-449): ``while not success: epsilon /= 2``.
    Two regimes make this spin without bound:

    (a) an anchor outside the per-feature box (prevented at the source by
        clamping the init draw in ``_LocalInitHopSkipJump``): every candidate
        is clipped into the box while the reachable adversarial region lies
        outside it, so success never comes and, at epsilon == 0, the clipped
        candidate keeps differing from the (out-of-box) current sample —
        measured on the NSL xgb smoke: >100k halvings on ONE outer
        iteration, ~28x the healthy call budget overall;
    (b) an anchor parked at a box corner with an outward-pointing update:
        the candidate is clipped back to ITSELF for every epsilon.

    The guard stops halving as soon as the clipped candidate is IDENTICAL to
    ``current_sample``: no smaller epsilon can move the sample either, so the
    guard can never skip a reachable success. A hard cap of 200 halvings
    (float64 ulp level needs ~60) terminates any remaining pathology.
    Semantics are otherwise verbatim from ART 1.20.1 ``_attack``
    (lines 376-462): the candidate is clipped before the check, which is
    what ``_adversarial_satisfactory`` does internally anyway
    (hop_skip_jump.py:644), so predictions are identical.
    """

    def _attack(
        self,
        initial_sample: np.ndarray,
        original_sample: np.ndarray,
        target: int,
        mask: np.ndarray | None,
        clip_min: float,
        clip_max: float,
    ) -> np.ndarray:
        """
        Main function for the boundary attack (verbatim ART 1.20.1 with the
        guarded epsilon loop).

        :param initial_sample: An initial adversarial example.
        :param original_sample: The original input.
        :param target: The target label.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :return: an adversarial example.
        """
        # Set current perturbed image to the initial image
        current_sample = initial_sample

        # Main loop to wander around the boundary
        for _ in range(self.max_iter):
            # First compute delta
            delta = self._compute_delta(
                current_sample=current_sample,
                original_sample=original_sample,
                clip_min=clip_min,
                clip_max=clip_max,
            )

            # Then run binary search
            current_sample = self._binary_search(
                current_sample=current_sample,
                original_sample=original_sample,
                norm=self.norm,
                target=target,
                clip_min=clip_min,
                clip_max=clip_max,
            )
            if not np.isfinite(current_sample).all():
                logger.debug(
                    "[guard] iter=%d non-finite CURRENT after binary search "
                    "(%d NaN cells) — attack cannot proceed, returning original",
                    self.curr_iter, int((~np.isfinite(current_sample)).sum()),
                )
                return original_sample

            # Next compute the number of evaluations and compute the update
            num_eval = min(int(self.init_eval * np.sqrt(self.curr_iter + 1)), self.max_eval)

            update = self._compute_update(
                current_sample=current_sample,
                num_eval=num_eval,
                delta=delta,
                target=target,
                mask=mask,
                clip_min=clip_min,
                clip_max=clip_max,
            )
            if not np.isfinite(update).all():
                logger.debug(
                    "[guard] iter=%d non-finite UPDATE detected (grad degenerate) — "
                    "keeping current sample, skipping epsilon search",
                    self.curr_iter,
                )
                self.curr_iter += 1
                continue

            # Finally run step size search by first computing epsilon
            if self.norm == 2:
                dist = np.linalg.norm(original_sample - current_sample)
            else:
                dist = np.max(abs(original_sample - current_sample))

            epsilon = 2.0 * dist / np.sqrt(self.curr_iter + 1)
            success = False
            halvings = 0

            while not success:
                epsilon /= 2.0
                potential_sample = current_sample + epsilon * update
                # Guard (ours): stop halving when the clipped candidate is
                # identical to the current sample — either the box blocks the
                # update direction entirely, or eps*update fell below one ulp.
                # Further halving cannot change any prediction.
                candidate = np.clip(potential_sample, clip_min, clip_max)
                if np.array_equal(candidate, current_sample):
                    logger.debug(
                        "[guard] iter=%d epsilon search blocked after %d halvings "
                        "(clipped candidate == current sample)",
                        self.curr_iter,
                        halvings + 1,
                    )
                    break
                halvings += 1
                # Hard termination bound (ours): ART has none and would spin
                # to float underflow; 200 halvings is far beyond anything
                # reachable in float64 (ulp level needs ~60).
                if halvings >= 200:
                    logger.debug(
                        "[guard] iter=%d epsilon search capped at %d halvings",
                        self.curr_iter,
                        halvings,
                    )
                    break
                success = self._adversarial_satisfactory(  # type: ignore
                    samples=candidate[None],
                    target=target,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

            # Update current sample
            current_sample = np.clip(potential_sample, clip_min, clip_max)

            # Update current iteration
            self.curr_iter += 1

            # If attack failed. return original sample
            if np.isnan(current_sample).any():  # pragma: no cover
                logger.debug("NaN detected in sample, returning original sample.")
                return original_sample

        return current_sample


class HSJAAttackGenerator(AttackGenerator):
    """HSJA in InputNorm coordinates (scale-coherent minimal perturbation).

    ART HopSkipJump minimizes the distance to the decision boundary in the
    estimator's input space, using only estimator.predict (black box). On
    raw tabular features two image-domain habits corrupt the pool:

      1. init: if the original sample is not adversarial, ART draws the
         initial adversarial point uniform over the WHOLE clip box
         (hop_skip_jump.py:309/344) — with the raw clip range 0..4e8 that
         point can be ~1e8 raw units away from any real traffic, and the
         binary search only guarantees proximity to the boundary, not to the
         original sample.
      2. raw-L2 geometry: the descent minimizes raw distance, so
         low-sigma features are moved by tens of sigmas for free.

    Fix: the attack estimator is the SAME model wrapped by the z-space
    adapter from inputnorm_adapter and the descent minimizes distance in
    sigma units. A per-feature valid-domain box derived from the data itself
    (data_zspace_clip; lower edge = per-feature data min -> raw >= 0) is
    attached to the estimator before each generate call. ART HopSkipJump
    only clips elementwise with it (hop_skip_jump.py:452/605/644) and draws
    its fallback init uniform within the box (:309/344 — a random record
    with every feature inside its observed range), so the box is safe to
    hand over; the initial delta 0.1*(clip_max - clip_min) (:551) becomes a
    per-feature radius — 10% of each feature's observed range, exactly the
    scale-coherent quantity the raw-space version could never be.
    ``inputnorm_stats`` (from the DNN checkpoint) is required for non-DNN
    models.
    """

    def __init__(self, classifier, generator_params: dict = None, inputnorm_stats=None):
        generator_params = dict(generator_params or {})
        fallback_radius = generator_params.pop('fallback_radius', None)
        self.fallback_radius = None if fallback_radius is None else float(fallback_radius)
        super().__init__(classifier, generator_params)
        set_attackgen_predict_nthread(self.classifier)
        attack_estimator, self.stats = build_zspace_estimator_for(classifier, inputnorm_stats)
        attack_params = {
            "batch_size": 16,
            "targeted": False,
            "norm": 2,
            "max_iter": 10,
            "max_eval": 1000,
            "init_eval": 20,
            "init_size": 20,
            "init_radius": 2.0,
            "verbose": True
        }
        attack_params = self.update_generator_params(attack_params)
        init_radius = float(attack_params.pop('init_radius', 2.0))

        self.attack = _GuardedHSJA(classifier=attack_estimator,
                                   init_radius=init_radius, **attack_params)

    def generate(self, x: np.ndarray, y: np.ndarray, input_metadata: dict, mutate_indices: list[int] = [],
                 zspace_clip=None, raw_floor=None, **kwargs) -> pd.DataFrame:
        """Generate adversarial examples using HSJA with optional batching.

        Args:
            x: Input samples (raw space), shape (N, F)
            y: True labels, required by our interface
            input_metadata: Dict containing at least 'feature_names' and 'label_column'
            mutate_indices: Indices to keep unchanged in outputs (e.g., categorical/binary)
            zspace_clip: Optional precomputed (lo, hi) z-space box (sharded
                generation: full-set box, overriding the per-batch derivation).
            raw_floor: Optional per-feature raw minimum for the post-conversion
                clamp (sharded generation: full-set floor, not the chunk's).
            **kwargs: Optional 'batch_size' to process in chunks

        Returns:
            DataFrame of adversarial samples (raw space) with original labels appended.
        """
        if x is None or x.size == 0:
            return pd.DataFrame(columns=(input_metadata.get('feature_names', []) + [input_metadata.get('label_column', 'label')]))

        # Per-feature valid-domain box derived from the data itself (lower
        # edge = per-feature data min -> raw >= 0; see inputnorm_adapter).
        # Dataset-wide box: covers every row, all batch chunks alike.
        raw_clip = getattr(self.classifier, "clip_values", None)
        raw_clip_max = raw_clip[1] if raw_clip is not None else None
        if zspace_clip is None:
            zspace_clip = data_zspace_clip(self.stats, self.stats.to_z(x), raw_clip_max=raw_clip_max)
        set_zspace_clip_box(self.attack.estimator, zspace_clip)

        # Create masking array: 1s except for mutate_indices which are 0s
        mask = np.ones(x.shape)
        if mutate_indices:
            mask[:, mutate_indices] = 0
        logger.info(f"[+] Generating adversarial samples with HSJA (InputNorm coordinates), mutate_indices: {mutate_indices}")

        def _attack_batch(xb: np.ndarray, maskb: np.ndarray) -> np.ndarray:
            """raw in -> raw out: convert to z, attack, convert back."""
            zb = self.stats.to_z(xb)
            zb_adv = self.attack.generate(x=zb, mask=maskb)

            # Optional hybrid fallback: local HSJA can fail before reaching
            # the boundary because no initial adversarial point is found.
            # Retry only those unchanged rows at a wider radius and keep the
            # first result for rows that already moved.  This remains
            # reportable as a two-stage threat model and does not silently
            # replace the baseline behavior.
            if self.fallback_radius is not None:
                unchanged = np.max(np.abs(zb_adv - zb), axis=1) <= 1e-5
                if np.any(unchanged):
                    old_radius = self.attack._init_radius
                    self.attack._init_radius = self.fallback_radius
                    try:
                        retry_adv = self.attack.generate(
                            x=zb[unchanged], mask=maskb[unchanged]
                        )
                    finally:
                        self.attack._init_radius = old_radius
                    retry_active = np.max(
                        np.abs(retry_adv - zb[unchanged]), axis=1
                    ) > 1e-5
                    retry_rows = np.flatnonzero(unchanged)
                    zb_adv[retry_rows[retry_active]] = retry_adv[retry_active]
                    logger.info(
                        "[HSJA] fallback radius %.4g: retried %d, recovered %d",
                        self.fallback_radius,
                        int(unchanged.sum()),
                        int(retry_active.sum()),
                    )
            x_adv_b = self.stats.from_z(zb_adv)
            # from_z (z->raw in float32) can land a hair below the raw data
            # minimum (float roundoff at the box edge); clamp to the
            # per-feature RAW data minimum computed on xb (roundoff-free).
            # Sharded generation passes the full-set floor instead (raw_floor).
            floor = xb.min(axis=0) if raw_floor is None else raw_floor
            return np.maximum(x_adv_b, floor)

        batch_size = kwargs.get('batch_size', -1)
        max_retries = int(kwargs.get('max_retries', 3))
        placeholder_mode = kwargs.get('placeholder', 'original')  # 'original' or 'drop'
        timeout_seconds = int(kwargs.get('timeout', -1))  # -1 = no timeout
        verbose_every = int(kwargs.get('verbose', 0))  # log progress every N batches; 0 disables
        n_samples = x.shape[0]

        if batch_size is None or batch_size <= 0 or batch_size >= n_samples:
            logger.info("[+] Generating adversarial samples with HSJA (single batch)")
            x_adv = _attack_batch(x, mask)
            if verbose_every and verbose_every > 0:
                logger.info("[HSJA] Progress: 1/1 batches, failed=0")
        else:
            logger.info(f"[+] Generating adversarial samples with HSJA in batches of {batch_size}")
            adv_parts = []
            y_parts = []
            total_batches = (n_samples + batch_size - 1) // batch_size
            processed_batches = 0
            failed_batches = 0
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                xb = x[start:end]
                yb = y[start:end]
                maskb = mask[start:end]
                # retries
                attempt = 0
                last_err = None
                xb_adv = None
                batch_failed = False
                while attempt < max_retries:
                    try:
                        if timeout_seconds is not None and timeout_seconds > 0:
                            with _timeout(timeout_seconds):
                                xb_adv = _attack_batch(xb, maskb)
                        else:
                            xb_adv = _attack_batch(xb, maskb)
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
                            logger.info(f"[HSJA] Progress: {processed_batches}/{total_batches} batches, failed={failed_batches}")
                        continue
                    else:  # 'original'
                        logger.error(f"Using original samples for failed batch {start}:{end} after {max_retries} retries")
                        xb_adv = xb
                        failed_batches += 1
                adv_parts.append(xb_adv)
                y_parts.append(yb)
                processed_batches += 1
                if verbose_every and verbose_every > 0 and (processed_batches % verbose_every == 0 or processed_batches == total_batches):
                    logger.info(f"[HSJA] Progress: {processed_batches}/{total_batches} batches, failed={failed_batches}")
            if not adv_parts:
                # all batches dropped
                return pd.DataFrame(columns=(input_metadata.get('feature_names', []) + [input_metadata.get('label_column', 'label')]))
            x_adv = np.concatenate(adv_parts, axis=0)
            y = np.concatenate(y_parts, axis=0) if y_parts else y[:0]

        # Frozen positions restored from the raw originals (bit-exact)
        if mutate_indices:
            x_adv[:, mutate_indices] = x[:, mutate_indices]
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
        raise TimeoutError("HSJA generation timed out")
    old = signal.signal(signal.SIGALRM, _handle)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

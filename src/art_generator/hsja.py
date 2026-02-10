from art.attacks.evasion import HopSkipJump
from art_generator.attack_generator import AttackGenerator
import numpy as np
import pandas as pd
from utils.logging import get_logger
from contextlib import contextmanager
import signal
logger = get_logger(__name__)

class HSJAAttackGenerator(AttackGenerator):
    def __init__(self, classifier, generator_params: dict = None):
        super().__init__(classifier, generator_params)
        attack_params = {
            "batch_size": 16,
            "targeted": False,
            "norm": 2,
            "max_iter": 10,
            "max_eval": 1000,
            "init_eval": 20,
            "init_size": 20,
            "verbose": True
        }
        attack_params = self.update_generator_params(attack_params)

        self.attack = HopSkipJump(classifier=self.classifier, **attack_params)

    def generate(self, x: np.ndarray, y: np.ndarray, input_metadata: dict, mutate_indices: list[int] = [], **kwargs) -> pd.DataFrame:
        """Generate adversarial examples using HSJA with optional batching.

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

        # Create masking array: 1s except for mutate_indices which are 0s
        mask = np.ones(x.shape)
        if mutate_indices:
            mask[:, mutate_indices] = 0
        logger.info(f"[+] Generating adversarial samples with HSJA, mutate_indices: {mutate_indices}")

        batch_size = kwargs.get('batch_size', -1)
        max_retries = int(kwargs.get('max_retries', 3))
        placeholder_mode = kwargs.get('placeholder', 'original')  # 'original' or 'drop'
        timeout_seconds = int(kwargs.get('timeout', -1))  # -1 = no timeout
        verbose_every = int(kwargs.get('verbose', 0))  # log progress every N batches; 0 disables
        n_samples = x.shape[0]

        if batch_size is None or batch_size <= 0 or batch_size >= n_samples:
            logger.info("[+] Generating adversarial samples with HSJA (single batch)")
            x_adv = self.attack.generate(x=x, mask=mask)
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
                                xb_adv = self.attack.generate(x=xb, mask=maskb)
                        else:
                            xb_adv = self.attack.generate(x=xb, mask=maskb)
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
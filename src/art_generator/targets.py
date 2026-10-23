"""Target-selection helpers for targeted attacks.

Target rule: the highest-scoring class other than the true label (the true label is
masked out of the prediction table first), so the chosen target is never the true
label — even for samples the model already mispredicts (where the plain
second-most-likely class could coincide with the true label and reproduce
the targeted-to-true-label bug). Deterministic: no RNG involved.
"""
import numpy as np
from art.utils import to_categorical
from utils.logging import get_logger

logger = get_logger(__name__)


def true_label_indices(y: np.ndarray) -> np.ndarray:
    """Return class indices from one-hot (2-D) or integer labels (1-D)."""
    y_arr = np.asarray(y)
    if y_arr.ndim > 1:
        return np.argmax(y_arr, axis=1)
    return y_arr.astype(int)


def masked_second_most_likely(x: np.ndarray, classifier, y: np.ndarray) -> np.ndarray:
    """One-hot targets: highest-confidence class excluding the true label.

    For correctly classified samples this is identical to ART's
    ``second_most_likely_class``; it only differs for mispredicted samples,
    where it prevents targeting the true label. Note: for a mispredicted
    sample the masked target is the model's current top-1 class, i.e. the
    sample is "already evaded" — targeted ASR should be interpreted on the
    clean-correct subset (or those rows trivially succeed by definition).
    """
    preds = classifier.predict(x)
    p = preds.astype(np.float64).copy()
    y_idx = true_label_indices(y)
    p[np.arange(p.shape[0]), y_idx] = -np.inf
    return to_categorical(np.argmax(p, axis=1), nb_classes=classifier.nb_classes)

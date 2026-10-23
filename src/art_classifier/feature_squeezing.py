"""Feature Squeezing preprocessing defence builder (local-only module).

This file is intentionally listed in .gitignore and is NOT published.
``art_classifier.AdversarialWrapper`` loads it optionally: when this file is absent
the wrappers simply build no preprocessing defences.
"""
from typing import Any, Dict, Optional

import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)


def build_feature_squeezing_defences(params: Dict[str, Any]) -> Optional[list]:
    """Create preprocessing defences list based on params.

    Supported params:
      - feature_squeezing: bool (enable/disable)
      - fs_clip_values: (lo, hi) per-feature arrays (REQUIRED for FS:
        min/max per feature from the full reference set — declared box)
      - fs_bit_depth: int (default 8), one grid per feature with
        (max_i - min_i) / (2^b - 1) cells
      - fs_apply_fit / fs_apply_predict: bool (default True)

    Without fs_clip_values, FS is disabled with a warning (the legacy
    global (0, 1) image grid was removed: scale-blind on tabular data).
    """
    try:
        enabled = bool(params.get("feature_squeezing", False) or ("fs_bit_depth" in params))
        if not enabled:
            return None
        try:
            from art.defences.preprocessor import FeatureSqueezing  # type: ignore
        except Exception:
            # ART not available or defence not present; skip silently
            return None
        bit_depth = int(params.get("fs_bit_depth", params.get("feature_squeezing_bit_depth", 8)))
        apply_fit = bool(params.get("fs_apply_fit", True))
        apply_predict = bool(params.get("fs_apply_predict", True))
        # Prefer PER-FEATURE clip range (min_i, max_i): ART FeatureSqueezing
        # quantizes on a uniform grid of (max - min) / (2^bit_depth - 1) per
        # feature. With a single global scalar pair the grid is shared by
        # features spanning ~8 orders of magnitude on the tabular datasets
        # (legacy (0, 1) hardcode made the grid scale-blind: absolute step
        # 1/(2^b - 1) in raw units — a no-op for byte/count features and the
        # only SANER fallback, since the true global range 0..4e8 would
        # destroy all values below ~1e6). The per-feature pair restores the
        # image-domain semantics: 2^bit_depth levels over each feature's own
        # observed data range. Arrays are supported by ART FS (broadcasting
        # in __call__, elementwise min>=max check in _check_params).
        per_feature = params.get("fs_clip_values")
        if per_feature is not None:
            try:
                lo = np.asarray(per_feature[0], dtype=np.float64)
                hi = np.asarray(per_feature[1], dtype=np.float64)
                if lo.ndim != 1 or hi.ndim != 1 or lo.shape != hi.shape:
                    raise ValueError(f"expected two equal-shape 1-D arrays, got {lo.shape}/{hi.shape}")
            except Exception as exc:
                logger.warning(f"[!] fs_clip_values unusable ({exc}); Feature Squeezing DISABLED")
                per_feature = None
            else:
                # Constant features (e.g. num_outbound_cmds ≡ 0 on KDD sets)
                # give lo_i == hi_i, which ART rejects as an invalid box.
                # Rejecting the whole box silently disabled FS on NSL-KDD;
                # instead coerce only those dims to a point-like range
                # (lo_i, lo_i + eps): FS stays active on all informative
                # features while the constant dims quantize to a no-op.
                degenerate = lo >= hi
                if degenerate.any():
                    eps = np.maximum(1.0, np.abs(lo)) * 1e-6
                    hi = np.where(degenerate, lo + eps, hi)
                    logger.warning(
                        "[+] fs_clip_values: %d/%d constant feature ranges (lo_i == hi_i) "
                        "coerced to (lo_i, lo_i + eps)",
                        int(degenerate.sum()), lo.size,
                    )
        # GROUPED per-feature bit depths (fs_group_bits) REMOVED (2026-10-12):
        # tuning showed per-model scale groups did NOT beat a single
        # per-model bit depth over the per-feature clip range (CIC: +6.4pp
        # xgb but -8.6pp catb; NSL: strictly worse than single b=6) — the
        # actionable semantics are: per-feature clip range + ONE bit depth
        # per model (fs_bit_depth). See reports/fs_tuning/ for sweeps.
        if per_feature is not None:
            logger.info(
                "[+] Feature Squeezing: per-feature clip range (bit_depth=%d, %d features, grid steps [%.4g, %.4g])",
                bit_depth, lo.size, np.min((hi - lo) / (2**bit_depth - 1)), np.max((hi - lo) / (2**bit_depth - 1)),
            )
            fs = FeatureSqueezing(
                clip_values=(lo, hi),
                bit_depth=bit_depth,
                apply_fit=apply_fit,
                apply_predict=apply_predict,
            )
            return [fs]
        # No legacy (0, 1) fallback by design (2026-10-12): a global (0, 1)
        # grid is image-domain semantics — on this tabular data it is
        # scale-blind (absolute step 1/(2^b - 1) raw units): a near no-op
        # for byte/count features (~1e4..1e8) and destructive-by-typo for
        # unit-scale ones. Without a declared per-feature clip range the
        # defense cannot be specified correctly, so DISABLE it loudly
        # instead of guessing an image grid.
        logger.warning(
            "[!] Feature Squeezing requested but no fs_clip_values provided "
            "-> Feature Squeezing DISABLED (per-feature clip range is required; "
            "the legacy (0, 1) image grid was removed as inappropriate for tabular data)"
        )
        return None
    except Exception:
        logger.exception("[!] build_feature_squeezing_defences failed; defences disabled")
        return None

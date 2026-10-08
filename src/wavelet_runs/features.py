"""Extract orientation-specific spatial variance of DT-CWT coefficients."""

import dtcwt
import numpy as np

from .preprocessing import validate_volumes


def extract_features(volumes: np.ndarray, levels: int = 5) -> np.ndarray:
    """Return log-magnitude variances with axes ``(time, scale, orientation)``.

    Input axes are ``(time, x, y, z)``. Spatial dimensions must be even.
    Each volume is transformed independently, using dtcwt's default filters
    and boundary extension. Variance uses ddof=0 over the three spatial axes.
    Zero coefficients are masked before taking natural logarithms. Entirely
    undefined orientations are NaN; downstream classification rejects them.
    """
    volumes = validate_volumes(volumes)
    if any(size % 2 for size in volumes.shape[1:]):
        raise ValueError("Spatial dimensions must be even; pad or crop explicitly.")
    if isinstance(levels, bool) or not isinstance(levels, (int, np.integer)) or levels < 1:
        raise ValueError("levels must be a positive integer.")
    transform = dtcwt.Transform3d()
    features = np.empty((len(volumes), levels, 28), dtype=np.float64)
    for time, volume in enumerate(volumes):
        pyramid = transform.forward(volume, nlevels=levels)
        for scale, coefficients in enumerate(pyramid.highpasses):
            log_magnitude = np.ma.log(np.abs(coefficients))
            features[time, scale] = log_magnitude.var(axis=(0, 1, 2)).filled(np.nan)
    return features

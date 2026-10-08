"""Explicit ROI masking and within-run temporal normalization."""

import numpy as np


def validate_volumes(volumes: np.ndarray) -> np.ndarray:
    """Validate a real, finite time-by-space array and return float64 values."""
    volumes = np.asarray(volumes)
    if volumes.ndim != 4 or any(size == 0 for size in volumes.shape):
        raise ValueError("Expected nonempty volumes with axes (time, x, y, z).")
    if volumes.dtype.kind not in "fiu" or not np.isfinite(volumes).all():
        raise ValueError("Volumes must contain finite real numeric values.")
    return volumes.astype(np.float64, copy=False)


def validate_chunks(chunks: np.ndarray, samples: int) -> np.ndarray:
    """Require one nonnegative integer run ID per volume."""
    chunks = np.asarray(chunks)
    if chunks.shape != (samples,) or chunks.dtype.kind not in "iu" or np.any(chunks < 0):
        raise ValueError("chunks must be a vector of nonnegative integer run IDs.")
    return chunks


def apply_mask(volumes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero voxels outside a binary ROI without cropping or reorienting data."""
    volumes = validate_volumes(volumes)
    mask = np.asarray(mask)
    if mask.shape != volumes.shape[1:] or not np.isin(mask, [0, 1]).all():
        raise ValueError("Mask must be binary and match the spatial shape of the volumes.")
    if not np.any(mask):
        raise ValueError("Mask must include at least one voxel.")
    return volumes * mask.astype(bool)


def zscore_runs(volumes: np.ndarray, chunks: np.ndarray) -> np.ndarray:
    """Z-score each voxel across time separately in each run (ddof=0).

    Constant voxels become zero. No normalization parameters are shared
    across runs. This is an offline analysis and uses all time points within
    each run, including the held-out run's unlabeled signals.
    """
    volumes = validate_volumes(volumes)
    chunks = validate_chunks(chunks, len(volumes))
    normalized = np.zeros_like(volumes)
    for run in np.unique(chunks):
        selected = chunks == run
        if selected.sum() < 2:
            raise ValueError("Each run needs at least two volumes for z-scoring.")
        values = volumes[selected]
        centered = values - values.mean(axis=0)
        std = values.std(axis=0)
        normalized[selected] = np.divide(centered, std, out=np.zeros_like(centered), where=std > 0)
    return normalized

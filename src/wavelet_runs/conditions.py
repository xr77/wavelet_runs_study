"""Convert condition matrices into categorical labels without interpolation."""

import numpy as np

from .preprocessing import validate_chunks


def condition_labels(
    conditions: np.ndarray, shift: int = 0, chunks: np.ndarray | None = None
) -> np.ndarray:
    """Convert a binary (condition, time) matrix to labels; zero means rest.

    Conditions are numbered from one. Overlapping active conditions are
    rejected. A positive shift delays labels and zero-fills the beginning.
    With chunks, shifting is independent within each run; without chunks it
    applies to the whole series. Neither mode wraps labels at boundaries.
    """
    conditions = np.asarray(conditions)
    if conditions.ndim != 2 or any(size == 0 for size in conditions.shape):
        raise ValueError("Conditions must be a nonempty (condition, time) matrix.")
    if not np.isin(conditions, [0, 1]).all() or np.any(conditions.sum(axis=0) > 1):
        raise ValueError("Conditions must be binary with at most one active condition per time.")
    if isinstance(shift, bool) or not isinstance(shift, (int, np.integer)) or shift < 0:
        raise ValueError("shift must be a nonnegative integer.")
    labels = np.where(conditions.any(axis=0), conditions.argmax(axis=0) + 1, 0)
    if chunks is None:
        chunks = np.zeros(len(labels), dtype=int)
    chunks = validate_chunks(chunks, len(labels))
    shifted = np.zeros_like(labels)
    for run in np.unique(chunks):
        indices = np.flatnonzero(chunks == run)
        if np.any(np.diff(indices) != 1):
            raise ValueError("Runs must occupy contiguous time points for label shifting.")
        if shift < len(indices):
            shifted[indices[shift:]] = labels[indices[: len(indices) - shift]]
    return shifted

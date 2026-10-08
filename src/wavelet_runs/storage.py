"""Local feature storage with safe loading and no implicit file overwrites."""

from pathlib import Path

import numpy as np

from .preprocessing import validate_chunks


def save_features(path, features, chunks):
    """Write a numeric NPZ with documented feature and run axes."""
    path = Path(path)
    features = np.asarray(features)
    if path.suffix != ".npz":
        raise ValueError("Feature output must end in .npz.")
    if features.ndim != 3 or features.dtype.kind not in "fiu":
        raise ValueError("Features must be a numeric (time, scale, orientation) array.")
    chunks = validate_chunks(chunks, len(features))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, features=features, chunks=chunks)


def load_features(path):
    """Load numeric features without enabling pickle deserialization."""
    with np.load(path, allow_pickle=False) as archive:
        features = archive["features"]
        chunks = archive["chunks"]
    if features.ndim != 3 or features.dtype.kind not in "fiu":
        raise ValueError("Expected numeric features with axes (time, scale, orientation).")
    return features, validate_chunks(chunks, len(features))


def stack_subjects(subject_features):
    """Stack identically shaped subjects as (time,scale,orientation,subject).

    Input list order defines the subject axis. Callers must check that labels,
    preprocessing, and run order correspond before combining subjects.
    """
    arrays = [np.asarray(values) for values in subject_features]
    if not arrays or any(a.ndim != 3 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Subjects must have identical (time, scale, orientation) shapes.")
    return np.stack(arrays, axis=3)

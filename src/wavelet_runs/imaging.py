"""Read local NIfTI runs with explicit spatial and run-order checks."""

from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np

from .preprocessing import apply_mask, validate_volumes


def load_nifti_runs(
    paths: Sequence[str | Path], mask_path: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    """Load ordered 4-D NIfTI files; return (time,x,y,z) volumes and run IDs.

    The 3-D binary mask and every run must have the same grid and affine.
    No resampling, registration, reorientation, or scan preprocessing occurs.
    All runs are loaded into memory. The caller supplies preprocessed scans.
    """
    if not paths:
        raise ValueError("At least one NIfTI run is required.")
    mask_image = nib.load(str(mask_path))
    if len(mask_image.shape) != 3:
        raise ValueError("The mask must be a three-dimensional NIfTI image.")
    mask = mask_image.get_fdata()
    if not np.isfinite(mask_image.affine).all():
        raise ValueError("Mask affine must be finite.")
    volumes, chunks = [], []
    for run, path in enumerate(paths):
        image = nib.load(str(path))
        if len(image.shape) != 4 or image.shape[:3] != mask.shape:
            raise ValueError(f"Run {run} must be 4-D and match the mask's spatial shape.")
        if not np.allclose(image.affine, mask_image.affine, atol=1e-5, rtol=0):
            raise ValueError(f"Run {run} and mask have different affines; align them first.")
        data = validate_volumes(np.moveaxis(image.get_fdata(), -1, 0))
        volumes.append(apply_mask(data, mask))
        chunks.append(np.full(len(data), run, dtype=int))
    return np.concatenate(volumes), np.concatenate(chunks)

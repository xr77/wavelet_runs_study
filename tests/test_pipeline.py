"""Workflow tests use generated arrays and temporary NIfTI files only."""

import json

import nibabel as nib
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from wavelet_runs.classification import classify_runs
from wavelet_runs.cli import main
from wavelet_runs.conditions import condition_labels
from wavelet_runs.imaging import load_nifti_runs
from wavelet_runs.preprocessing import apply_mask, zscore_runs
from wavelet_runs.storage import load_features, save_features, stack_subjects


def test_zscore_separates_runs():
    volumes = np.array([0, 2, 100, 104], dtype=float).reshape(4, 1, 1, 1)
    chunks = np.array([0, 0, 1, 1])
    np.testing.assert_allclose(zscore_runs(volumes, chunks).ravel(), [-1, 1, -1, 1])
    assert not zscore_runs(np.ones_like(volumes), chunks).any()
    np.testing.assert_array_equal(volumes.ravel(), [0, 2, 100, 104])


def test_binary_mask_preserves_grid():
    volumes = np.ones((2, 2, 2, 2))
    mask = np.zeros((2, 2, 2))
    mask[0, 0, 0] = 1
    actual = apply_mask(volumes, mask)
    assert actual.shape == volumes.shape and actual.sum() == 2
    with pytest.raises(ValueError, match="binary"):
        apply_mask(volumes, mask + 0.5)
    with pytest.raises(ValueError, match="at least one"):
        apply_mask(volumes, np.zeros_like(mask))


def test_label_delay_respects_explicit_run_boundaries():
    conditions = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    np.testing.assert_array_equal(condition_labels(conditions), [1, 2, 2, 1])
    np.testing.assert_array_equal(condition_labels(conditions, 1), [0, 1, 2, 2])
    np.testing.assert_array_equal(
        condition_labels(conditions, 1, np.array([0, 0, 1, 1])), [0, 1, 0, 2]
    )
    assert not condition_labels(conditions, 8).any()
    with pytest.raises(ValueError, match="at most one"):
        condition_labels(np.ones((2, 4)))


def test_nifti_axes_order_and_affine_validation(tmp_path):
    affine = np.eye(4)
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 4, 6)), affine), mask_path)
    paths = []
    for run, length in enumerate([2, 3]):
        path = tmp_path / f"run_{run}.nii.gz"
        nib.save(nib.Nifti1Image(np.full((2, 4, 6, length), run + 1.0), affine), path)
        paths.append(path)
    volumes, chunks = load_nifti_runs(paths, mask_path)
    assert volumes.shape == (5, 2, 4, 6)
    np.testing.assert_array_equal(chunks, [0, 0, 1, 1, 1])
    np.testing.assert_array_equal(volumes[:, 0, 0, 0], [1, 1, 2, 2, 2])
    affine[0, 3] = 2
    nib.save(nib.Nifti1Image(np.ones((2, 4, 6, 3)), affine), paths[1])
    with pytest.raises(ValueError, match="affines"):
        load_nifti_runs(paths, mask_path)


class SplitSpy(ClassifierMixin, BaseEstimator):
    observations = []

    def fit(self, features, labels):
        self.train_ids_ = set(features[:, 0])
        self.classes_ = np.unique(labels)
        return self

    def predict(self, features):
        self.observations.append((self.train_ids_, set(features[:, 0])))
        return np.zeros(len(features), dtype=int)


def test_cv_runs_are_disjoint_and_rest_is_excluded():
    features = np.arange(12.0).reshape(12, 1, 1)
    labels = np.tile([0, 1, 2, 1], 3)
    chunks = np.repeat(np.arange(3), 4)
    SplitSpy.observations.clear()
    result = classify_runs(features, labels, chunks, SplitSpy())
    assert len(SplitSpy.observations) == 3
    for run, (train_ids, test_ids) in enumerate(SplitSpy.observations):
        assert train_ids.isdisjoint(test_ids)
        assert test_ids == set(np.flatnonzero((chunks == run) & (labels != 0)))
        assert not (train_ids | test_ids) & {0, 4, 8}
    assert result["test_samples_per_run"] == [3, 3, 3]
    assert np.asarray(result["confusion_matrices"]).sum() == 9


def test_unseen_test_class_is_rejected():
    with pytest.raises(ValueError, match="absent from training"):
        classify_runs(
            np.ones((6, 1, 2)), np.array([1, 2, 3, 1, 2, 2]), np.array([0, 0, 0, 1, 1, 1])
        )


def test_storage_and_subject_axis(tmp_path):
    features = np.arange(48.0).reshape(4, 2, 6)
    chunks = np.array([0, 0, 1, 1])
    path = tmp_path / "features.npz"
    save_features(path, features, chunks)
    actual, actual_chunks = load_features(path)
    np.testing.assert_array_equal(actual, features)
    np.testing.assert_array_equal(actual_chunks, chunks)
    with pytest.raises(FileExistsError):
        save_features(path, features, chunks)
    stacked = stack_subjects([features, features + 1])
    assert stacked.shape == (4, 2, 6, 2)
    np.testing.assert_array_equal(stacked[..., 1], features + 1)
    with pytest.raises(ValueError):
        stack_subjects([features, features[:2]])


def test_cli_labels_and_classification(tmp_path):
    condition_path, labels_path = tmp_path / "conditions.npy", tmp_path / "labels.npy"
    feature_path, output = tmp_path / "features.npz", tmp_path / "result.json"
    labels = np.tile([1, 2], 12)
    np.save(condition_path, np.vstack([labels == 1, labels == 2]))
    features = labels[:, None, None] * 10 + np.random.default_rng(9).normal(size=(24, 2, 3))
    save_features(feature_path, features, np.repeat(np.arange(3), 8))
    main(["labels", "--input", str(condition_path), "--output", str(labels_path)])
    main(
        [
            "classify",
            "--features",
            str(feature_path),
            "--labels",
            str(labels_path),
            "--output",
            str(output),
        ]
    )
    assert json.loads(output.read_text())["mean_accuracy"] == [1.0, 1.0]
    original = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(["demo", "--output", str(output)])
    assert error.value.code == 2 and output.read_bytes() == original

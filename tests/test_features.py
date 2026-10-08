"""Tests of the public utility, without original participant data."""

import unittest

import dtcwt
import numpy as np

from wavelet_runs.features import extract_features


class FeatureTests(unittest.TestCase):
    def test_matches_original_statistic(self):
        data = np.random.default_rng(2).normal(size=(2, 16, 16, 16))
        actual = extract_features(data, levels=2)
        self.assertEqual(actual.shape, (2, 2, 28))
        # Independent spelling of the original per-scale/orientation statistic.
        pyramid = dtcwt.Transform3d().forward(data[0], nlevels=2)
        for level, coefficients in enumerate(pyramid.highpasses):
            for orientation in range(28):
                magnitudes = np.abs(coefficients[..., orientation]).ravel()
                expected = np.log(magnitudes[magnitudes > 0]).var()
                self.assertAlmostEqual(actual[0, level, orientation], expected, places=12)

    def test_zero_volume_preserves_undefined_features(self):
        result = extract_features(np.zeros((1, 16, 16, 16)), levels=2)
        self.assertTrue(np.isnan(result).all())

    def test_invalid_input(self):
        for data in [
            np.zeros((16, 16, 16)),
            np.zeros((0, 16, 16, 16)),
            np.zeros((1, 15, 16, 16)),
            np.full((1, 16, 16, 16), np.nan),
        ]:
            with self.subTest(shape=data.shape), self.assertRaises(ValueError):
                extract_features(data)
        with self.assertRaises(ValueError):
            extract_features(np.zeros((1, 16, 16, 16)), levels=0)


if __name__ == "__main__":
    unittest.main()

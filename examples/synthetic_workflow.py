"""Run a synthetic demonstration; no laboratory data are used or downloaded.

Usage after installation: python examples/synthetic_workflow.py
All arrays stay in memory. Reported accuracy is a smoke test, not a finding.
"""

import numpy as np

from wavelet_runs.classification import classify_runs
from wavelet_runs.features import extract_features


def main():
    rng = np.random.default_rng(42)
    chunks = np.repeat(np.arange(3), 8)
    labels = np.tile([1, 2], 12)
    volumes = rng.normal(size=(24, 16, 16, 16))
    features = extract_features(volumes, levels=2)
    result = classify_runs(features, labels, chunks)
    print("Synthetic demonstration only — no participant data or scientific inference.")
    print("Feature shape:", features.shape)
    print("Mean held-out accuracy by scale:", result["mean_accuracy"])


if __name__ == "__main__":
    main()

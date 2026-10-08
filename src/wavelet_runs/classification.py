"""Within-subject, leave-one-run-out evaluation at each wavelet scale."""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from .preprocessing import validate_chunks


def make_classifier(name: str = "gaussian-nb", seed: int = 42):
    """Create an unfitted classifier; XGBoost is an optional dependency."""
    if name == "gaussian-nb":
        return GaussianNB()
    if name == "svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale")
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError('Install XGBoost with pip install ".[xgboost]".') from error
        return XGBClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    raise ValueError("Classifier must be gaussian-nb, svm, or xgboost.")


def classify_runs(features, labels, chunks, estimator=None) -> dict:
    """Evaluate one subject; return fold accuracies and held-out confusion counts.

    Features have axes (time, scale, orientation). Labels are nonnegative
    integers; label zero (rest) is excluded from training and evaluation.
    Each fold fits a fresh estimator on all other runs. Labels are encoded
    from training data only, and unseen test classes are rejected explicitly.
    Means weight runs equally, matching the original analysis convention.
    """
    features = np.asarray(features)
    if features.ndim != 3 or any(size == 0 for size in features.shape):
        raise ValueError("Features must have nonempty axes (time, scale, orientation).")
    if features.dtype.kind not in "fiu" or not np.isfinite(features).all():
        raise ValueError("Features must be finite; resolve undefined wavelet statistics first.")
    chunks = validate_chunks(chunks, len(features))
    labels = np.asarray(labels)
    if labels.shape != (len(features),) or labels.dtype.kind not in "iu" or np.any(labels < 0):
        raise ValueError("Labels must be one nonnegative integer per time point.")
    runs = np.unique(chunks)
    classes = np.unique(labels[labels != 0])
    if len(runs) < 2 or len(classes) < 2:
        raise ValueError("Classification needs at least two runs and two non-rest classes.")
    if estimator is None:
        estimator = make_classifier()
    accuracy = np.empty((features.shape[1], len(runs)))
    confusion = np.zeros((features.shape[1], len(classes), len(classes)), dtype=int)
    sample_counts = []
    for fold, run in enumerate(runs):
        train = (chunks != run) & (labels != 0)
        test = (chunks == run) & (labels != 0)
        if not test.any() or len(np.unique(labels[train])) < 2:
            raise ValueError(f"Run {run} has no test samples or insufficient training classes.")
        encoder = LabelEncoder().fit(labels[train])
        if not np.isin(labels[test], encoder.classes_).all():
            raise ValueError(f"Run {run} contains a class absent from training runs.")
        sample_counts.append(int(test.sum()))
        for scale in range(features.shape[1]):
            model = clone(estimator)
            model.fit(features[train, scale], encoder.transform(labels[train]))
            predicted = encoder.inverse_transform(model.predict(features[test, scale]))
            accuracy[scale, fold] = accuracy_score(labels[test], predicted)
            confusion[scale] += confusion_matrix(labels[test], predicted, labels=classes)
    return {
        "run_ids": runs.tolist(),
        "classes": classes.tolist(),
        "test_samples_per_run": sample_counts,
        "fold_accuracy": accuracy.tolist(),
        "mean_accuracy": accuracy.mean(axis=1).tolist(),
        "confusion_matrices": confusion.tolist(),
    }

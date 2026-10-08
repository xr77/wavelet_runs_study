"""Command-line entry points for local analysis and a synthetic demonstration."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from . import __version__
from .classification import classify_runs, make_classifier
from .conditions import condition_labels
from .features import extract_features
from .imaging import load_nifti_runs
from .preprocessing import zscore_runs
from .storage import load_features, save_features


def build_parser():
    parser = argparse.ArgumentParser(description="Local spatial-wavelet analysis. No data uploads.")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="Extract features from seeded synthetic volumes")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--levels", type=int, default=5)

    extract = commands.add_parser("extract", help="Extract features from ordered local NIfTI runs")
    extract.add_argument("--bold", nargs="+", type=Path, required=True)
    extract.add_argument("--mask", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--levels", type=int, default=5)
    extract.add_argument(
        "--zscore", action="store_true", help="Normalize each voxel within each run"
    )

    labels = commands.add_parser("labels", help="Convert a local binary condition matrix to labels")
    labels.add_argument(
        "--input", type=Path, required=True, help="MAT or NPY condition-by-time matrix"
    )
    labels.add_argument("--key", default="conds_short_tlrc", help="Variable name for MAT input")
    labels.add_argument("--shift", type=int, default=0, help="Nonnegative delay in time points")
    labels.add_argument(
        "--runs-from", type=Path, help="Feature NPZ; shift separately within its runs"
    )
    labels.add_argument("--output", type=Path, required=True)

    classify = commands.add_parser("classify", help="Leave one run out, per scale, for one subject")
    classify.add_argument("--features", type=Path, required=True)
    classify.add_argument("--labels", type=Path, required=True)
    classify.add_argument(
        "--classifier", choices=["gaussian-nb", "svm", "xgboost"], default="gaussian-nb"
    )
    classify.add_argument("--seed", type=int, default=42)
    classify.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("Output already exists; choose a new path.")
        if args.command in ("demo", "extract"):
            if args.output.suffix != ".npz":
                raise ValueError("Feature output must end in .npz.")
            if args.command == "demo":
                volumes = np.random.default_rng(args.seed).normal(size=(4, 32, 32, 32))
                chunks = np.repeat(np.arange(2), 2)
            else:
                volumes, chunks = load_nifti_runs(args.bold, args.mask)
                if args.zscore:
                    volumes = zscore_runs(volumes, chunks)
            features = extract_features(volumes, args.levels)
            save_features(args.output, features, chunks)
            print(f"Saved {features.shape} (time, scale, orientation) to {args.output}")
            if np.isnan(features).any():
                print(
                    "Undefined coefficients produced NaN features; inspect before classification."
                )
        elif args.command == "labels":
            if args.output.suffix != ".npy":
                raise ValueError("Labels output must end in .npy.")
            if args.input.suffix == ".mat":
                conditions = loadmat(args.input)[args.key]
            elif args.input.suffix == ".npy":
                conditions = np.load(args.input, allow_pickle=False)
            else:
                raise ValueError("Condition input must be .mat or .npy.")
            chunks = load_features(args.runs_from)[1] if args.runs_from else None
            labels = condition_labels(conditions, args.shift, chunks)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("xb") as stream:
                np.save(stream, labels, allow_pickle=False)
            print(f"Saved {len(labels)} labels to {args.output}")
        else:
            if args.output.suffix != ".json":
                raise ValueError("Classification output must end in .json.")
            features, chunks = load_features(args.features)
            labels = np.load(args.labels, allow_pickle=False)
            result = classify_runs(
                features, labels, chunks, make_classifier(args.classifier, args.seed)
            )
            result.update(version=__version__, classifier=args.classifier, seed=args.seed)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
            print(f"Saved held-out classification results to {args.output}")
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.error(str(error))

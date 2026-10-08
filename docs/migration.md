# Migration from exploratory research code

Version 0.2.0 consolidates repeated notebook cells into a Python package. The public tree contains no `.ipynb` files, embedded figures, participant arrays, or saved research results. Originals are preserved only in the owner's local backup, outside the release manifest.

| Original analysis | Supported replacement |
| --- | --- |
| Occipital/ventral-temporal `step1` scripts | `imaging.load_nifti_runs`, `preprocessing.apply_mask`, `preprocessing.zscore_runs`, `features.extract_features`; choose ROI with `--mask` |
| Subject-stacking notebook | `storage.stack_subjects` with explicit subject order |
| Primary classification notebook | `classification.classify_runs` and `wavelet-runs classify` |
| Confusion-matrix notebook | Held-out confusion counts from `classify_runs` |
| Repeated condition conversion | `conditions.condition_labels` and `wavelet-runs labels` |
| First-release demonstration | `wavelet-runs demo` and `examples/synthetic_workflow.py` |

Earlier MVPA/searchlight work, plotting experiments, mask-editing cells, post-analysis work, and permutation/Bayesian experiments are not silently promoted into supported methods. They have divergent assumptions or depend on interactive state. They remain in the owner's private backup; adding them to the public package requires a separate reviewed implementation.

## Deliberate changes

- Paths, subject counts, run lengths, and ROI choices are arguments instead of constants.
- Explicit imports replace wildcard imports and `%pylab` state.
- Array axes are `(time, x, y, z)`; compare historical PyMVPA mapping before claiming parity.
- Label delay uses integer indexing instead of spline interpolation. Run boundaries are explicit, and the default delay is zero.
- Binary masks, aligned affines, nonoverlapping conditions, and train/test classes are validated.
- Undefined wavelet statistics remain NaN instead of exposing masked-array backing storage.
- Gaussian naive Bayes is the lightweight default; SVM and XGBoost are explicit alternatives. Historical classifier equivalence is not claimed.
- No research data, figures, condition matrices, or original numerical results are distributed.

Generated-input tests establish software behavior, not reproduction of the original study.

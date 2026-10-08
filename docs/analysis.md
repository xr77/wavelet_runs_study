# Analysis guide

## Data stays local

There are no study data or saved outputs in this repository. Obtain any data-access permission independently. Commands operate only on paths you supply. Keep private inputs outside the repository and generated outputs in an ignored directory such as `outputs/`.

## Feature extraction

`load_nifti_runs` reads ordered four-dimensional NIfTI files `(x, y, z, time)`, checks them against a three-dimensional binary ROI mask, and returns `(time, x, y, z)` arrays. Spatial shapes and affines must agree. No resampling, reorientation, or registration occurs. The runs are loaded into memory.

`apply_mask` zeros voxels outside the ROI while preserving the grid. `zscore_runs` optionally normalizes each voxel across time separately in each run, using population standard deviation. Constant voxels become zero. This offline normalization uses all unlabeled time points within the held-out run; it is not a prospective online protocol.

`extract_features` applies `dtcwt.Transform3d` independently to each volume using the library's default filters and boundary handling. For each scale and 28 orientations it calculates `var(log(abs(coefficients)))` over space with population variance. Zero coefficients are masked; entirely undefined statistics are `NaN`. Spatial dimensions must be even; any padding or cropping is the caller's explicit choice.

Output NPZ files contain numeric arrays `features` `(time, scale, orientation)` and `chunks` `(time,)`. Run IDs follow input-file order. Pickle loading is disabled. Output files are never overwritten implicitly.

## Conditions

`condition_labels` accepts a binary `(condition, time)` matrix with at most one active condition per time point. Classes are numbered from one; zero means rest. Nonnegative shifts delay labels and zero-fill exposed samples without interpolation or wrapping.

With `--runs-from`, shifts occur within each run. Without it, the entire concatenated series is shifted. The default delay is zero. Original scripts used two TRs across the concatenated series; this choice is not automatically appropriate for another experiment.

## Classification

`classify_runs` analyzes one subject and one scale at a time. Each fold holds out all non-rest samples of one run and trains a fresh estimator on other runs. Reports include ordered run IDs and classes, test counts, scale-by-run accuracies, equally weighted run means, and pooled held-out confusion matrices (rows=true, columns=predicted).

Gaussian naive Bayes is the default. RBF SVM and optional XGBoost are explicit alternatives. No hyperparameter selection, cross-subject pooling, or additional feature scaling is implicit. Label encoding is fitted on training labels; unseen test classes trigger an error. Undefined features must be handled explicitly.

Choose preprocessing and model settings before interpreting performance. Synthetic accuracy is not biological evidence. Permutation inference, Bayesian comparisons, and multiple-comparison corrections are outside this release's supported API.

## Reproducibility limits

Original notebooks mixed Python 2/3, interactive state, fixed paths/run lengths, and classifier defaults. This package retains the core feature statistic and run-wise evaluation structure while making choices explicit. Axis mapping, normalization, timing, classifiers, and results require validation against an independently authorized reference dataset before claiming replication. No original study results have been regenerated.

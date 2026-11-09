Emulator model and training.

- `helper.py` the parametric matrix model (eigenvalue emulator) and its cost functions.
- `main_only_HL_mod.py` trains a half-life emulator on one nucleus (`--hl-file`, `--params-file`), with k-fold
  cross-validation options; `main_only_centroid_mod.py` the same for the windowed GT resonance centroid from
  strength functions (`--data-dir`).
- `cv.sh` runs the k folds of one nucleus; `run_cv.sh` runs the paper's five-fold cross-validation for all
  nuclei of a functional from `data/training`.
- `pick_best_cv.py` selects, per nucleus, the fold with the lowest validation cost and deploys its parameters
  and training set to `data/emulators` (the committed emulators were chosen this way).
- `combine_full_cv.py` combines K = 0 and K = 1 strength emulators (not used for the paper's centroid emulators).

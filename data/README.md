Data used by the paper. Nucleus names are element symbol plus mass number (Cd126); GT nuclei carry an
underscore in `config.py` keys (Cd_126) and none in directory names.

- `posteriors/<F>/<set>_<likelihood>/` MCMC posterior samples (`trace.nc`, ArviZ NetCDF), `summary.csv`, and the
  run `manifest.txt`. F = SKOP, SLY4, HFB1 (SkO', SLy4, UNEDF1-HFB); set = setA, G10, L10, full; likelihood =
  studentt, normal. For HFB1 the L10 set equals G10 and only `G10_*` exists. The manifests record the run
  settings with the set keys used at run time (A, G10V7Q = G10_SKOP, A10V7Q = L10_SKOP, G10_SLY4V7Q = G10_SLY4,
  A10_SLY4V7Q = L10_SLY4, G10_HFB1V7Q = G10_HFB1, POOL40_<F> = full) and the file paths of the original
  machine.
- `emulators/<F>/HL/<nuc>/` and `emulators/<F>/GT/<nuc>/full/` trained emulator parameters (`params_best_n5_*.txt`),
  the scaled training points (`train_set*.txt`), and the fold they were taken from (`picker_meta.txt`).
- `errors/` test emulator errors from the 40-point test batch 21001 (`sigma_em_frozen_<F>_batch21001.txt`,
  per-sample residuals in `test_residuals_batch21001.tsv`), the per-nucleus Q-value error term in decades
  (`sigma_q_<F>.dat`), and under `cv/` the cross-validation parity files (true vs emulated value per fold).
- `training/<F>/HL/<nuc>/` the 100-point Sobol design (`params.txt`: V0, C1s) and the pnFAM half-lives
  (`halflives.txt`, s); `training/<F>/GT/<nuc>/full/` the GT strength functions per sample (`sample_<i>.dat`)
  and the design. The two nuclei excluded from each functional's pool (see `config.HL_EXCL_QCORR`) are included.
- `test_batch21001/<F>/HL/<nuc>/` the test design and pnFAM half-lives behind `errors/`.
- `n126/<F>/<nuc>/batch30001/` pnFAM half-lives of the N = 126 isotones at posterior draws (Figs. 8-10, Table VI);
  `batch30201/` the same with the first-forbidden part separated.
- `experiment/` NUBASE2020 and AME2020 tables.
Measured half-lives, GT resonance energies, and their errors are in `calibration/config.py`.

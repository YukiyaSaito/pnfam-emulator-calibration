Bayesian calibration.

- `bayes_mcmc.py` likelihood, priors, emulator wrappers, and sampler (PyMC). Run through `run_bayes_mcmc.sh`,
  whose settings block documents every option; `runs/run_paper_posteriors.sh` reproduces the posteriors of the paper.
- `bayes_plot.py` diagnostic plots per run.
- `config.py` measured half-lives and GT resonance energies with errors, GT windows and energy shifts, the
  candidate pools, the calibration sets of the paper (`HL_SETS`, `GT_SETS`), and parameter bounds.
- `predict_from_draws.py`, `frozen_common.py` posterior predictions for the N = 126 isotones from the pnFAM
  runs at posterior draws (`data/n126`), producing `paper/data/predictions_<F>_batch30001.tsv`.
New runs are written to `$PNFAM_CALIB_RESULTS` (default `<repo>/results`).

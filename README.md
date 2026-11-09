# pnfam-emulator-calibration

Code, trained emulators, and posterior samples for the Bayesian calibration of the isovector
pairing strength V0 and the spin-isospin coupling C1s of three Skyrme functionals (SkO', SLy4,
UNEDF1-HFB) to measured beta-decay half-lives and Gamow-Teller (GT) resonance energies, using
emulators of the proton-neutron finite-amplitude method (pnFAM). Companion to

Y. Saito, A. Ravlić, and W. Nazarewicz, "Emulator-Enabled Calibration and Uncertainty Quantification of
beta-decay Calculations within the Energy Density Functional Theory", submitted to Phys. Rev. C (2026)
[arXiv identifier to be added].

## Layout

| directory | content |
|---|---|
| `calibration/` | Bayesian inference driver (`bayes_mcmc.py`), set definitions and experimental data (`config.py`), run scripts |
| `emulator/` | emulator model (`helper.py`), training scripts, five-fold cross-validation launcher, fold selection |
| `design/` | E-optimal set selection: information matrix, coverage rule, sensitivity tables, search outputs |
| `qcorr/` | Q-value correction fit and shift tables |
| `data/` | trained emulators, emulator error tables, posterior samples, training inputs, experimental inputs |
| `paper/` | figure and table scripts, their data files, and the figures and tables of the paper |
| `environment/` | conda and pip environment files |
| `tests/` | reproduction checks |

All paths are relative to the repository root (`repo_paths.py`); a fresh clone runs without edits.

## Reproducing the paper

1. Figures and tables from the committed posteriors (environment/environment-paper.yml, about 15 min):
   `tests/reproduce_figures.sh`. Outputs go to `paper/figs` and `paper/tables` and are compared with the committed copies.
2. Posteriors from the committed emulators (environment/environment-mcmc.yml, about a minute per posterior):
   `calibration/runs/run_paper_posteriors.sh`, or the single-posterior check `tests/reproduce_posterior.sh`.
   The sampler (PyMC SMC, fixed seed) is deterministic, so `summary.csv` agrees with the committed one.
3. E-optimal sets (environment-paper.yml, about 1.5 h per functional for the exact ten-nucleus search):
   `design/constrained_set_search.py --functional SKOP --sigma-the-hl 0.464 --sigma-the-gt 1.197 --loc-lo -325 --loc-hi -199 --kmin 10 --kmax 10 --pools screened`
   (SLy4: 0.475 0.820 -370 -197; UNEDF1-HFB: 0.406 0.738 -322 -143). The theory errors are the Set-A
   posterior medians of Table III (Normal likelihood), the windows the Set-A 16-84 % intervals of V0.
4. Emulators from the training inputs (environment/requirements-emulator.txt, TensorFlow):
   `emulator/run_cv.sh HL SKOP` and `emulator/run_cv.sh GT SKOP`, then `emulator/pick_best_cv.py`.

The raw pnFAM outputs from which the training inputs were extracted are in the companion archive
pnfam-emulator-calibration-data [Zenodo DOI to be added]; nothing in steps 1-4 needs them.

## Citation

See CITATION.cff. The emulator model (emulator/helper.py) is a modified version of SMLR, Surrogate Models for
Linear Response, by L. Jin, A. Ravlić, P. Giuliani, K. Godbey, and W. Nazarewicz, https://github.com/ascsn/SMLR.
The pnFAM and pynfam codes that produced the training data are those of M. T. Mustonen, E. M. Ney, J. Engel and
collaborators and are not part of this repository.

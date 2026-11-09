Q-value correction (Sec. on the Q values, Fig. 1, Table I).

- `q_correction.py` the adopted correction dQ(F, Z, N) = a + b (N - Z) + c (N + Z) per functional, with a
  self-test against `results/correction_refit.txt`.
- `fit_correction.py`, `refit_correction.py` the fit to the 154 nuclei with measured Q values;
  `results/dQ_<F>.txt` HFB minus measured Q per nucleus; `results/q_shift_<F>.txt` the fitted coefficients;
  `collect_q.py` extracts the HFB Q values from the raw survey outputs (not in the companion archive, 225 GB).
- `make_nucleus_list.py` AME2020 reader used by the N = 126 table.

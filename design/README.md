E-optimal set selection (Sec. on the dataset design).

- `skop_nuclei_fisher.py` half-life sensitivities per nucleus from the training data -> `<f>_hl_fisher.csv`;
  `gt_nuclei_fisher.py` the same for the GT centroids -> `<f>_gt_fisher.csv`, including the cross-validation
  emulator error of the centroid (`sigma_em_MeV`, read from data/errors/cv); `fisher_matrix_report.py` a summary.
- `nuclei_tag_table.py` -> `nuclei_tag_table[_<F>].csv`: the region tags (mass, Q value, shell closure,
  deformation, first-forbidden share) used by the coverage rule. Regenerating it needs the raw pnFAM outputs.
- `constrained_set_search.py` exhaustive search for the set of m half-life nuclei maximizing the smallest
  eigenvalue of the information matrix under the coverage rule, with global and local (Set-A window)
  sensitivities, including the per-nucleus Q-value error term (`--no-sigma-q` omits it). The variances are the
  same decomposition as the likelihood: half-lives sigma_exp^2 + sigma_em^2 + sigma_th^2 + sigma_Q^2, GT centroids
  sigma_exp^2 + sigma_em^2 + sigma_th^2 (the GT emulator term, below 0.03 MeV, was added 2026-11-10; it changes
  the smallest eigenvalues by about 1e-4 and no selected set). Outputs
  `constrained_set_search[_<F>]<suffix>.{csv,txt}`; the committed `*_sigq_k10.*` are the ten-nucleus searches
  behind Table II, `*_setsV7sigq.*` the eight-nucleus searches whose local-slope tables Fig. 4 reads.
- Coverage rule of the adopted search (scheme `Qter-Dthree-Xboth` in the reports, the strictest of the 24 variants;
  `fig_information_saturation.py` applies the same rule at every m). Every selected set must contain at least one
  nucleus of each class: mass light (A <= 82), medium (88 <= A <= 148), heavy (A >= 152); near the neutron shell
  closures N = 50, 82, 126 (`shell` tag); calculated Q_beta (`HFB_Qval`) below 3 MeV, 3-6 MeV, above 6 MeV;
  deformation |beta2| < 0.10 (spherical), 0.10-0.20 (transitional), >= 0.20 (deformed); one heavy deformed nucleus
  with Q_beta < 3 MeV (the rare-earth/actinide member: Sm156, Er174, or Ra234); first-forbidden share >= 50 %
  (`pcFF`). At m = 10 the schemes differ only for SLy4 with global slopes (Fe66 in place of Ti58 under the
  weaker rules); see the per-scheme tables in `*_sigq_k10.txt`.
- `monotonic_check_<F>_2026-10-08.txt` screening of the candidate pool; `vpair_preference_trend*.csv`
  per-nucleus preferred V0 (informational).

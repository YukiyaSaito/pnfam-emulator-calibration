Figures and tables of the paper.

| file | script | inputs |
|---|---|---|
| FIG1 `fig_q_correction` | Q-correction fit | qcorr/results |
| FIG2 `fig_emulator_summary` | emulator errors | data/errors, data/errors/cv |
| FIG3 `posterior_A` (fig_posterior_A.py) | Set-A posteriors, Normal | data/posteriors |
| FIG4 `fig_information_saturation` | information vs set size | design/, data/errors; `--replot` redraws from `figs/fig_information_saturation.txt` (full run about 10 min) |
| FIG5 `fig_posteriors_final` | final posteriors | data/posteriors |
| FIG6 `fig_pool_residuals` | half-life residuals | data/posteriors, data/emulators, paper/data/literature |
| FIG7 `fig_gt_residuals` | GT residuals | data/posteriors |
| FIG8-10 `fig_n126_*` | N = 126 predictions | paper/data/predictions_<F>_batch30001.tsv, data/n126, paper/data/literature |
| TABLE1 `table_qcorr` | | qcorr/results |
| TABLE2 `table_calibsets`, TABLE3 `table_posteriors` | | config.py, data/posteriors |
| TABLE4-5 `table_pool_nuclei` | | design/nuclei_tag_table*.csv |
| TABLE6 `table_n126_gs` | | paper/data/n126_gs.txt (cache), data/n126 |

`scripts/make_tables.sh` writes all tables; each figure script writes `figs/<name>.{pdf,png}`.
`paper/data/literature/` holds published half-life tables used for comparison: data_Evan.txt (Ney et al. 2020, SkO'),
data_FRDM.txt (FRDM), supplementary_halflives.txt (Ney et al.), half-lives.dat (RQRPA/RQTBA), data_DDPC1.txt
(DD-PC1, Ravlic et al.).

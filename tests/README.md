- `reproduce_figures.sh` runs every figure and table script and compares the PNG and .tex outputs byte for byte
  with the committed copies in paper/figs and paper/tables (PDFs differ only in their timestamp).
- `reproduce_posterior.sh [F set likelihood]` reruns one calibration (default HFB1 setA studentt) and compares
  summary.csv with data/posteriors.

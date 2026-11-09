"""Shared matplotlib style for the paper figures (user 2026-10-13: LaTeX fonts).

Usage in a figure script:
    import paper_style
    ...
    paper_style.save(fig, OUT)      # writes OUT.pdf (LaTeX text) and OUT.png

Text is rendered by LaTeX (Computer Modern, same as the REVTeX document).
Requires a working `latex` and the type1ec/cm-super package; both are in
~/texmf on this machine (installed 2026-10-13). PNG previews are made from
the PDF with pdftocairo, since dvipng is not installed.
"""
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OKABE_ITO = {"SKOP": "#0072B2", "SLY4": "#D55E00", "HFB1": "#009E73"}
MARKERS = {"SKOP": "o", "SLY4": "s", "HFB1": "^"}
LABELS = {"SKOP": r"SkO$'$", "SLY4": "SLy4", "HFB1": "UNEDF1-HFB"}
COLUMN_WIDTH = 3.4      # inches, PRC single column
TEXT_WIDTH = 7.0        # inches, PRC double column

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def nuc(name):
    """'Sn134' -> r'$^{134}$Sn' for LaTeX labels."""
    sym = "".join(c for c in name if c.isalpha())
    A = "".join(c for c in name if c.isdigit())
    return rf"$^{{{A}}}${sym}"


def save(fig, stem, dpi=200):
    """Write stem.pdf (vector, LaTeX text) and stem.png (preview)."""
    pdf = stem + ".pdf"
    fig.savefig(pdf)
    try:
        subprocess.run(["pdftocairo", "-png", "-r", str(dpi), "-singlefile", pdf, stem],
                       check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as e:
        print("png preview not written:", e)
    print("wrote", os.path.abspath(pdf), "(+ .png)")

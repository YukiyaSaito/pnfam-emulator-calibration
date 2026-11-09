#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the nucleus tag table for representation-constrained calibration-set
selection (SKO' 2D). One row per rates nucleus in the scratch pool, tagging:

  mass region, Q-window (HFB_Qval, MeV), shell-closure proximity (N and Z),
  ground-state deformation beta2 (HFB 'Def', sample-independent), forbidden
  fraction %FF at the (vpair_t0=0, cs0=0) reference sample, plus sensitivity
  (b_vp, b_cs, R2 from skop_hl_fisher.csv), preferred-vpair tension
  (vpair_preference_trend.csv), and training-data availability.

Physics note: vpair_t0 and override_cs0 enter only the QRPA/FAM stage, so the
HFB-level tags (HFB_Qval, Def, E_gs) are identical across samples of a nucleus.
%FF and Total-HL are sample-dependent; here they are taken at sample_0, the
(0,0) override reference, and tagged as such.

Inputs (pre-extracted by the driver; see EXTRACT_DIR):
  <EXTRACT_DIR>/<Nuc>.dat : meta/logfile_master.dat of sample_0 (or fallback:
                            hfb_soln/logfile_mini.dat, or another sample's
                            master when sample_0 crashed -- Q/beta2 unaffected)
  <EXTRACT_DIR>/provenance.txt (optional): "<Nuc> sample_<i> master" lines;
                            merged as tag_sample so non-sample_0 pcFF /
                            Total_HL provenance stays visible.
Outputs (this directory):
  nuclei_tag_table.csv (SKOP) / nuclei_tag_table_<FUNCTIONAL>.csv

2026-08-05: generalized to --functional (default SKOP, unchanged behavior).
HFB tags (Q, beta2, E_gs) are FUNCTIONAL-DEPENDENT: masters must be
extracted from that functional's tarballs (set TAG_MASTERS). The trend
merge is optional -- no vpair_preference_trend exists for UNE1 yet.
"""
import argparse
import os
import glob
import numpy as np
import pandas as pd

_ap = argparse.ArgumentParser()
_ap.add_argument("--functional", default="SKOP", help="SKOP | UNE1 | ...")
_args = _ap.parse_args()
FUNCTIONAL = _args.functional

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRACT_DIR = os.environ.get("TAG_MASTERS", "")   # directory of per-nucleus master .dat files extracted from the raw pnFAM
                                                    # outputs (companion archive); needed only to regenerate the committed tag tables
FISHER_CSV = os.path.join(HERE, f"{FUNCTIONAL.lower()}_hl_fisher.csv")
TREND_CSV  = os.path.join(HERE, "vpair_preference_trend.csv") \
    if FUNCTIONAL == "SKOP" else None
sys.path.insert(0, os.path.join(HERE, ".."))
import repo_paths as RP  # noqa: E402
TRAIN_DIR  = os.path.join(RP.TRAINING, FUNCTIONAL, "HL")
OUT_CSV    = os.path.join(HERE, "nuclei_tag_table.csv" if FUNCTIONAL == "SKOP"
                          else f"nuclei_tag_table_{FUNCTIONAL}.csv")

# Magic numbers for shell-closure proximity tags
N_MAGIC = (50, 82, 126)
Z_MAGIC = (28, 50, 82)

# Strata cut points (tunable; defaults chosen from the pool's distribution)
Q_LOW_MEV   = 3.0    # HFB_Qval below this = low-Q / narrow window [MeV]
Q_HIGH_MEV  = 6.0    # above this = high-Q / very neutron-rich    [MeV]
B2_SPH      = 0.10   # |beta2| below this = spherical
B2_DEF      = 0.20   # |beta2| above this = well-deformed
SHELL_NEAR  = 6      # |N - magic| (or |Z - magic|) <= this = "near closure"


def parse_master(path):
    """Parse one logfile_master.dat / logfile_mini.dat (whitespace table with
    a header line). Returns dict of the columns we need; NaN when absent."""
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if len(lines) < 2:
        return None
    header = lines[0].split()
    row = lines[1].split()
    # 'Label' value ('000000/beta_soln') has no embedded spaces, but the row
    # begins with a pandas index integer -- align from the right instead.
    if len(row) == len(header) + 1:
        row = row[1:]
    d = dict(zip(header, row))
    out = {}
    for key, col in [("Z", "Z"), ("N", "N"), ("HFB_Qval", "HFB_Qval"),
                     ("EQRPA_max", "EQRPA_max"), ("E_gs", "E_gs"),
                     ("beta2", "Def"), ("pcFF", "%FF"),
                     ("Total_HL_s", "Total-HL")]:
        try:
            out[key] = float(d[col])
        except (KeyError, ValueError):
            out[key] = np.nan
    out["HFB_Conv"] = d.get("HFB_Conv", "?")
    return out


def shell_tag(N, Z):
    tags = []
    for m in N_MAGIC:
        if abs(N - m) <= SHELL_NEAR:
            tags.append(f"N~{m}" if N != m else f"N={m}")
    for m in Z_MAGIC:
        if abs(Z - m) <= SHELL_NEAR:
            tags.append(f"Z~{m}" if Z != m else f"Z={m}")
    return "+".join(tags) if tags else "midshell"


def mass_class(A):
    if A <= 82:
        return "light"
    if A < 150:
        return "medium"
    return "heavy"


def q_class(q):
    if not np.isfinite(q):
        return "unknown"
    if q < Q_LOW_MEV:
        return "lowQ"
    if q > Q_HIGH_MEV:
        return "highQ"
    return "midQ"


def def_class(b2):
    if not np.isfinite(b2):
        return "unknown"
    if abs(b2) < B2_SPH:
        return "spherical"
    if abs(b2) > B2_DEF:
        return "deformed"
    return "transitional"


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(EXTRACT_DIR, "*.dat"))):
        nuc = os.path.splitext(os.path.basename(path))[0]
        rec = parse_master(path)
        if rec is None:
            print(f"WARNING: unparseable {path}")
            continue
        rec["nucleus"] = nuc
        rows.append(rec)
    tab = pd.DataFrame(rows)
    tab["Z"] = tab["Z"].astype(int)
    tab["N"] = tab["N"].astype(int)
    tab["A"] = tab["Z"] + tab["N"]

    # sanity: A parsed from the master must match the A in the nucleus name
    a_name = tab["nucleus"].str.extract(r"(\d+)").astype(int)[0]
    bad = tab[tab["A"] != a_name]
    if len(bad):
        raise SystemExit(f"A mismatch vs name: {bad['nucleus'].tolist()}")

    fisher = pd.read_csv(FISHER_CSV)[
        ["nucleus", "n", "R2", "b_vp", "b_cs", "pR2_vp", "pR2_cs", "T_s"]]
    tab = tab.merge(fisher, on="nucleus", how="left")
    trend_cols = ["exp_log10T", "vpair_at_exp", "nmatch", "covered"]
    if TREND_CSV is not None and os.path.isfile(TREND_CSV):
        trend = pd.read_csv(TREND_CSV)[["nucleus"] + trend_cols]
        tab = tab.merge(trend, on="nucleus", how="left")
    else:
        print(f"NOTE: no vpair preference trend for {FUNCTIONAL} yet; "
              "columns left NaN")
        for c in trend_cols:
            tab[c] = np.nan

    prov = os.path.join(EXTRACT_DIR, "provenance.txt")
    if os.path.isfile(prov):
        pmap = {}
        for ln in open(prov):
            c = ln.split()
            if len(c) >= 2:
                pmap[c[0]] = c[1]
        tab["tag_sample"] = tab["nucleus"].map(pmap)
    else:
        tab["tag_sample"] = "sample_0"

    trained = set(os.listdir(TRAIN_DIR)) if os.path.isdir(TRAIN_DIR) else set()
    tab["has_training"] = tab["nucleus"].isin(trained)

    tab["mass_class"]  = tab["A"].apply(mass_class)
    tab["Q_class"]     = tab["HFB_Qval"].apply(q_class)
    tab["def_class"]   = tab["beta2"].apply(def_class)
    tab["shell"]       = [shell_tag(n, z) for n, z in zip(tab["N"], tab["Z"])]

    cols = ["nucleus", "Z", "N", "A", "mass_class",
            "HFB_Qval", "EQRPA_max", "E_gs", "Q_class",
            "beta2", "def_class", "shell", "pcFF",
            "R2", "b_vp", "b_cs", "pR2_vp", "pR2_cs",
            "vpair_at_exp", "nmatch", "covered",
            "exp_log10T", "n", "has_training", "HFB_Conv", "tag_sample"]
    tab = tab[cols].sort_values("A").reset_index(drop=True)
    tab.to_csv(OUT_CSV, index=False, float_format="%.4g")
    print(f"Wrote {OUT_CSV}  ({len(tab)} nuclei)")

    show = tab[["nucleus", "A", "mass_class", "HFB_Qval", "Q_class",
                "beta2", "def_class", "shell", "pcFF", "R2", "b_vp",
                "vpair_at_exp", "has_training"]]
    with pd.option_context("display.width", 200, "display.max_rows", 60):
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()

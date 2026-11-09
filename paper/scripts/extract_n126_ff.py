#!/usr/bin/env python3
"""Extract the Gamow-Teller and first-forbidden rates of the N = 126 batch-30001 runs (20 posterior draws
per nucleus and functional) from the run archives, for the FF-fraction figure (2026-10-28).

Source per run: <scratch>/<F>/outputs/rates/<nuc>_sample_<id>_final.tar.gz
                -> <nuc>/sample_<id>/meta/logfile_master.dat, columns Total, Total-GT, Total-Forbidden [1/s],
                   %FF, HFB_Conv, FAM_Conv. The couplings come from the batch table params_batch30001.txt.
Output: ../data/n126_ff_rates.txt (one row per run):
        functional nucleus Z sample vpair cs0 rate_total rate_GT rate_FF pcFF hfb_conv fam_conv
No new pnFAM runs: this only reads archives that already exist.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import repo_paths as RP  # noqa: E402  (all data paths of the repository)
SCRATCH = RP.PNFAM_RAW    # raw pnFAM outputs (companion archive), set PNFAM_RAW
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "n126_ff_rates.txt")
FUNCS = ["SKOP", "SLY4", "HFB1"]
NUCS = ["Ba182", "Ce184", "Nd186", "Sm188", "Gd190", "Dy192", "Er194", "Yb196", "Hf198", "W200", "Os202", "Pt204", "Hg206"]
ZOF = {n: 56 + 2 * i for i, n in enumerate(NUCS)}
B = 30001


def logfile_row(F, nuc, sid):
    tgz = os.path.join(SCRATCH, F, "outputs", "rates", f"{nuc}_sample_{sid}_final.tar.gz")
    member = f"{nuc}/sample_{sid}/meta/logfile_master.dat"
    r = subprocess.run(["tar", "xzf", tgz, "-O", member], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    head = lines[0].split(); vals = lines[1].split()
    row = dict(zip(head, vals[len(vals) - len(head):]))
    return row


rows = ["# GT and first-forbidden rates of the N = 126 batch-30001 runs (draft/scripts/extract_n126_ff.py)",
        "# rates in 1/s from logfile_master.dat (Total, Total-GT, Total-Forbidden); pcFF = %FF column",
        "functional\tnucleus\tZ\tsample\tvpair\tcs0\trate_total\trate_GT\trate_FF\tpcFF\thfb_conv\tfam_conv"]
nmiss = 0
for F in FUNCS:
    for nuc in NUCS:
        pf = os.path.join(RP.N126, F, nuc, f"batch{B}", f"params_batch{B}.txt")
        P = {int(float(l.split()[0])): (float(l.split()[1]), float(l.split()[2]))
             for l in open(pf) if not l.startswith("#") and l.strip()}
        for sid in sorted(P):
            row = logfile_row(F, nuc, sid)
            if row is None:
                nmiss += 1; print("missing archive", F, nuc, sid, file=sys.stderr); continue
            rows.append("\t".join([F, nuc, str(ZOF[nuc]), str(sid), f"{P[sid][0]:.4f}", f"{P[sid][1]:.4f}",
                                   row["Total"], row["Total-GT"], row["Total-Forbidden"], row["%FF"],
                                   row["HFB_Conv"], row["FAM_Conv"]]))
        print(F, nuc, "done", file=sys.stderr)
open(OUT, "w").write("\n".join(rows) + "\n")
print("wrote", os.path.abspath(OUT), f"({len(rows) - 3} runs, {nmiss} missing)")

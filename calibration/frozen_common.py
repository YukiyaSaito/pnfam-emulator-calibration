#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared helpers for the frozen-validation and prediction pipeline
(extract_frozen_batch.py, frozen_sigma_em.py, closure_loglik.py,
gen_prediction_draws.py, predict_from_draws.py). Nothing here writes files.

Units: vpair_t0 and override_cs0 in MeV fm^3; half-lives in seconds; all
errors in log10(T_1/2 / s).
"""
import ast
import os
import re
import subprocess
import sys

import numpy as np

SMLR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
CAL = os.path.join(SMLR, "calibration")
RESULTS = os.path.join(SMLR, "results")
SCRATCH = os.environ.get("PNFAM_RAW", "")   # raw pnFAM outputs (companion archive)
PRIOR = {"vpair": (-450.0, 0.0), "cs0": (0.0, 250.0)}

sys.path.insert(0, os.path.join(SMLR, "Beta_decay"))


def read_cols(path):
    """'# header' + whitespace rows -> {int(id): [float, ...]}."""
    d = {}
    for ln in open(path):
        if ln.lstrip().startswith("#") or not ln.strip():
            continue
        q = ln.split()
        d[int(float(q[0]))] = [float(x) for x in q[1:]]
    return d


def master_row(functional, nuc, sample_id):
    """(Total-HL [s], HFB_Conv, FAM_Conv) from a final tarball's master
    logfile; None if the tarball is absent; 'unreadable' on parse failure.
    Same reader as extract_densify_batch.py."""
    tar = os.path.join(SCRATCH, functional, "outputs", "rates",
                       f"{nuc}_sample_{sample_id}_final.tar.gz")
    if not os.path.isfile(tar):
        return None
    member = f"{nuc}/sample_{sample_id}/meta/logfile_master.dat"
    try:
        txt = subprocess.run(["tar", "-xzf", tar, "-O", member],
                             capture_output=True, text=True, timeout=300,
                             check=True).stdout
    except subprocess.CalledProcessError:
        return "unreadable"
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    if len(lines) < 2:
        return "unreadable"
    header, row = lines[0].split(), lines[1].split()
    if len(row) == len(header) + 1:
        row = row[1:]
    d = dict(zip(header, row))
    try:
        return (float(d["Total-HL"]), d.get("HFB_Conv", "?"),
                d.get("FAM_Conv", "?"))
    except (KeyError, ValueError):
        return "unreadable"


def emulator_fn(nuc, namespace):
    """Deployed AE half-life emulator as a numpy callable f(vpair, cs0) ->
    log10(T_1/2 / s). Uses bayes_mcmc.HalfLifeEmulator so it is the SAME
    object the MCMC evaluates (namespace e.g. 'SKOP' or 'SKOP_PT200LOC')."""
    import pytensor
    import pytensor.tensor as pt
    import bayes_mcmc as bm
    emu = bm.HalfLifeEmulator(nuc, namespace, n=5)
    v, c = pt.dscalar("v"), pt.dscalar("c")
    f = pytensor.function([v, c], emu(v, c))
    return lambda vp, cs: float(f(float(vp), float(cs)))


def load_posterior(trace_dir):
    """(vpair, cs0, sigma_hl_the) flat sample arrays from <trace_dir>/trace.nc."""
    import arviz as az
    po = az.from_netcdf(os.path.join(trace_dir, "trace.nc")).posterior
    g = lambda k: np.asarray(po[k].values).ravel()
    return g("vpair"), g("cs0"), g("sigma_hl_the")


def read_manifest(trace_dir):
    """<trace_dir>/manifest.txt -> dict; python-literal values parsed."""
    d = {}
    for ln in open(os.path.join(trace_dir, "manifest.txt")):
        if "=" not in ln:
            continue
        k, v = ln.rstrip("\n").split("=", 1)
        try:
            d[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            d[k] = v
    return d


def read_boxes(batch_manifest_path):
    """batch_manifest.txt 'box_<name>=V0[lo,hi] C1s[lo,hi] n=N ...' ->
    {name: dict(v=(lo,hi), c=(lo,hi), n=N, area=..)}."""
    boxes = {}
    pat = re.compile(r"^box_(\w+)=V0\[([-\d.]+),([-\d.]+)\] C1s\[([-\d.]+),([-\d.]+)\] n=(\d+)")
    for ln in open(batch_manifest_path):
        m = pat.match(ln)
        if m:
            name, vlo, vhi, clo, chi, n = m.groups()
            vlo, vhi, clo, chi = map(float, (vlo, vhi, clo, chi))
            boxes[name] = dict(v=(vlo, vhi), c=(clo, chi), n=int(n),
                               area=(vhi - vlo) * (chi - clo))
    return boxes


def design_density(pts, boxes):
    """Number density q(theta) of a union of uniform boxes at each point:
    sum over boxes containing the point of n_box / area_box."""
    q = np.zeros(len(pts))
    for b in boxes.values():
        inside = ((pts[:, 0] >= b["v"][0] - 1e-9) & (pts[:, 0] <= b["v"][1] + 1e-9) &
                  (pts[:, 1] >= b["c"][0] - 1e-9) & (pts[:, 1] <= b["c"][1] + 1e-9))
        q[inside] += b["n"] / b["area"]
    return q


def posterior_weights(pts, vpair, cs0, boxes):
    """Importance weights p_post(theta)/q(theta) with p_post a Gaussian fit
    to the posterior samples and q the design density. Returns (w, ESS)."""
    from scipy.stats import multivariate_normal
    mu = np.array([vpair.mean(), cs0.mean()])
    cov = np.cov(np.vstack([vpair, cs0]))
    p = multivariate_normal(mu, cov).pdf(pts)
    q = design_density(pts, boxes)
    w = np.where(q > 0, p / np.where(q > 0, q, 1.0), 0.0)
    ess = w.sum() ** 2 / max((w ** 2).sum(), 1e-300)
    return w, ess


def sigma_exp_log10(nuc):
    """Experimental error converted to log10 space, as in bayes_mcmc.build_model."""
    import config
    return config.HL_ERR[nuc] / (config.HL_EXP[nuc] * np.log(10.0))


def loglik_hl(pred_log10, nuc, sigma_the, sigma_em, family="normal", nu=4.0):
    """Per-nucleus HL log-likelihood of the experimental half-life given a
    predicted log10(T), matching bayes_mcmc (variance-matched Student-t)."""
    import config
    from scipy.stats import norm, t
    x = np.log10(config.HL_EXP[nuc])
    s = np.sqrt(sigma_the ** 2 + sigma_exp_log10(nuc) ** 2 + sigma_em ** 2)
    if family == "studentt":
        return t.logpdf(x, nu, loc=pred_log10, scale=s * np.sqrt((nu - 2.0) / nu))
    return norm.logpdf(x, loc=pred_log10, scale=s)

"""
Post-MCMC plotting for bayes_mcmc.py traces.

Reads  SMLR/results/UNE1/bayes/<gt_mode>/trace.nc
Writes SMLR/results/UNE1/bayes/<gt_mode>/figs/*.pdf

Produces:
  joint_posterior_seaborn.pdf      2D KDE jointplot (seaborn style; same density as contour)
  joint_posterior_hist2d.pdf       matplotlib hist2d + marginals
  joint_posterior_contour.pdf      boundary-aware KDE (boundary_kde) filled-contour + marginals
  joint_prior_seaborn.pdf          prior counterpart of seaborn jointplot
  joint_prior_hist2d.pdf           prior counterpart of hist2d + marginals
  joint_prior_contour.pdf          prior counterpart of KDE contour + marginals
  trace.pdf                        arviz traceplot of (vpair, cs0)
  hl_posterior_parity.pdf          log-log parity of predicted vs experimental T_1/2
  hl_posterior_violin.pdf          posterior-predictive distributions per nucleus
  gt_posterior_<Nuc>.pdf           posterior-predictive GT strength (requires full-dist emulators)
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from scipy.stats import truncnorm
from matplotlib.gridspec import GridSpec
import arviz as az
import pytensor
import pytensor.tensor as pt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
import config
from bayes_mcmc import (
    HL_EXP, HL_ERR, GT_E_EXP, GT_E_ERR,
    HalfLifeEmulator, GTStrengthEmulator, _bayes_root,
    PRIORS, make_prior_spec, sample_prior_numpy, read_prior_spec_from_manifest,
    describe_prior_spec,
)


def sample_prior(n_samples, rng, spec=None):
    """Draw independent joint prior samples. ``spec`` is the prior spec recorded
    in the run's manifest (bayes_mcmc.read_prior_spec_from_manifest); None =
    the legacy PRIORS (runs before 2026-09-22)."""
    if spec is None:
        spec = make_prior_spec('legacy', 'SKOP')
    return sample_prior_numpy(spec, n_samples, rng)

# Mustonen & Engel reference point
vpair_ME = -346.352
cs0_ME   = 128.279

# Shared E-grid for posterior-predictive GT plots (math-E frame)
E_GRID = np.linspace(-5.0, 30.0, 351)


# =====================================================================
# Joint posterior plots
# =====================================================================
def _reflect(v, c, support):
    """Boundary-corrected KDE input: mirror the samples across each edge of the
    rectangular support (3x3 copies) so the kernel smoothing does not leak mass
    across a hard edge; the caller clips the result to the support. The density
    then goes to zero smoothly where the samples thin out (2026-09-23)."""
    (vlo, vhi), (clo, chi) = support
    vs = [v, 2 * vlo - v, 2 * vhi - v]
    cs = [c, 2 * clo - c, 2 * chi - c]
    V = np.concatenate([x for x in vs for _ in cs])
    C = np.concatenate([y for _ in vs for y in cs])
    return V, C


def prior_density(spec, v, c):
    """Unnormalized joint prior density on arrays v, c (broadcastable), from the
    manifest's prior spec: TruncatedNormal(V0) x tanh^2 taper x flat/TN(C1s)."""
    pv, pc = spec['vpair'], spec['cs0']
    d = np.exp(-0.5 * ((v - pv['mu']) / pv['sigma']) ** 2)
    if pv.get('taper_w', 0.0) > 0:
        d = d * np.tanh(np.clip(v - pv['lower'], 0, None) / pv['taper_w']) ** 2
    d = np.where((v >= pv['lower']) & (v <= pv['upper']), d, 0.0)
    if pc.get('dist', 'tnorm') == 'flat':
        dc = np.ones_like(c, dtype=float)
    else:
        dc = np.exp(-0.5 * ((c - pc['mu']) / pc['sigma']) ** 2)
    dc = np.where((c >= pc['lower']) & (c <= pc['upper']), dc, 0.0)
    return d * dc


def _taper(x, spec):
    """tanh^2 taper of the V0 prior: tanh^2((V0 - lower)/taper_w) above the lower
    edge, 0 below. All ones when taper_w = 0 (legacy prior)."""
    pv = spec['vpair']
    w = pv.get('taper_w', 0.0)
    x = np.asarray(x, dtype=float)
    if w <= 0:
        return np.ones_like(x)
    return np.tanh(np.clip(x - pv['lower'], 0, None) / w) ** 2


def taper_weights(v, spec):
    """KDE weights 1/taper(V0) of the posterior samples, clipped at their 99th
    percentile so a few samples next to the V0 edge do not dominate. All ones
    when taper_w = 0 (2026-10-12)."""
    wt = 1.0 / np.clip(_taper(v, spec), 1e-6, None)
    return np.minimum(wt, np.quantile(wt, 0.99))


class _FixedCovKDE(st.gaussian_kde):
    """scipy gaussian_kde with an explicitly given kernel covariance.
    gaussian_kde scales its kernel to the covariance of the data it is given; for
    the mirrored copies (_reflect) that covariance is set by the distance between
    the copies, not by the spread of the samples, so the kernel would be 10-20x
    too wide. Here the kernel covariance is passed in directly (2026-10-12)."""
    def __init__(self, dataset, kernel_cov, weights=None):
        self._kernel_cov = np.atleast_2d(np.asarray(kernel_cov, dtype=float))
        super().__init__(dataset, weights=weights)

    def _compute_covariance(self):   # evaluate() uses cho_cov + log_det; the inv_cov property is not valid here
        self.factor = 1.0
        self.covariance = self._kernel_cov
        self.cho_cov = np.linalg.cholesky(self._kernel_cov).astype(np.float64)
        self.log_det = 2 * np.log(np.diag(self.cho_cov * np.sqrt(2 * np.pi))).sum()


def _spec_from_support(spec, support):
    """Old calling convention: support = ((v_lo, v_hi), (c_lo, c_hi)) without a
    prior spec -> hard edges only, no taper."""
    if spec is None and support is not None:
        (vlo, vhi), (clo, chi) = support
        spec = {'vpair': dict(lower=vlo, upper=vhi, taper_w=0.0),
                'cs0': dict(lower=clo, upper=chi)}
    return spec


def boundary_kde(post_v, post_c, spec=None, vg=None, cg=None, bw_factor=0.5):
    """Posterior density on a (V0, C1s) grid for the joint-posterior figures.

    spec = prior spec from the run's manifest. The posterior is the prior taper
    tanh^2((V0 - lower)/taper_w) times a smooth part. The smooth part is a
    Gaussian KDE of the samples weighted by 1/taper (taper_weights), with the
    samples mirrored across the edges of the prior support (_reflect); kernel
    covariance = (0.5 x Scott's factor)^2 x (weighted) covariance of the samples.
    The estimate is multiplied by the taper again and set to NaN outside the
    support. taper_w = 0: weights 1, i.e. the plain reflection KDE.
    spec = None: plain scipy KDE (Scott's factor), no edges.
    bw_factor: kernel bandwidth as a multiple of Scott's factor (default 0.5,
    as adopted 2026-10-12 for the run diagnostics; 1.0 for smoother paper figures).

    Returns a dict: vg, cg (1D grids), VV, CC (meshgrid, ij), f (joint density,
    arbitrary normalization), mv, mc (marginal densities, each normalized to
    integrate to 1 over the support on the grid)  (2026-10-12)."""
    v = np.asarray(post_v, dtype=float); c = np.asarray(post_c, dtype=float)
    vg = np.linspace(-500, 0, 401) if vg is None else np.asarray(vg, dtype=float)
    cg = np.linspace(0, 250, 251) if cg is None else np.asarray(cg, dtype=float)
    VV, CC = np.meshgrid(vg, cg, indexing="ij")
    pts = np.vstack([VV.ravel(), CC.ravel()])
    if spec is None:
        f = st.gaussian_kde(np.vstack([v, c]))(pts).reshape(VV.shape)
        mv = st.gaussian_kde(v)(vg)
        mc = st.gaussian_kde(c)(cg)
    else:
        vlo, vhi = spec['vpair']['lower'], spec['vpair']['upper']
        clo, chi = spec['cs0']['lower'], spec['cs0']['upper']
        wt = taper_weights(v, spec)
        bw = bw_factor * st.gaussian_kde(np.vstack([v, c])).factor    # bw_factor x Scott's factor (default 0.5)
        H = bw ** 2 * np.cov(np.vstack([v, c]), aweights=wt)          # 2D kernel covariance
        Hc = bw ** 2 * np.var(c, ddof=1)                              # C1s marginal: unweighted samples
        rv, rc = _reflect(v, c, ((vlo, vhi), (clo, chi))); rw = np.tile(wt, 9)
        # speed only: drop mirrored points more than 10 kernel widths outside the support
        # (their contribution inside the support is < exp(-50) of a kernel peak)
        sv, sc = np.sqrt(H[0, 0]), np.sqrt(H[1, 1])
        near = (rv > vlo - 10 * sv) & (rv < vhi + 10 * sv) & (rc > clo - 10 * sc) & (rc < chi + 10 * sc)
        inside = (pts[0] >= vlo) & (pts[0] <= vhi) & (pts[1] >= clo) & (pts[1] <= chi)
        f = np.full(pts.shape[1], np.nan)
        f[inside] = _FixedCovKDE(np.vstack([rv[near], rc[near]]), H, weights=rw[near])(pts[:, inside])
        f = f.reshape(VV.shape)
        f = f * _taper(VV, spec)
        f = np.where((VV >= vlo) & (VV <= vhi) & (CC >= clo) & (CC <= chi), f, np.nan)
        mv = _FixedCovKDE(np.concatenate([v, 2 * vlo - v, 2 * vhi - v]), H[0, 0],
                          weights=np.tile(wt, 3))(vg) * _taper(vg, spec)
        mv = np.where((vg >= vlo) & (vg <= vhi), mv, np.nan)
        mc = _FixedCovKDE(np.concatenate([c, 2 * clo - c, 2 * chi - c]), Hc)(cg)
        mc = np.where((cg >= clo) & (cg <= chi), mc, np.nan)
    mv = mv / (np.nansum(mv) * (vg[1] - vg[0]))
    mc = mc / (np.nansum(mc) * (cg[1] - cg[0]))
    return dict(vg=vg, cg=cg, VV=VV, CC=CC, f=f, mv=mv, mc=mc)


def _shade_excluded(spec, ax_joint, ax_marg_x, ax_marg_y, xmin=-500, ymin=0):
    """Grey out the regions the prior excludes: V0 below its lower edge and C1s
    below its lower bound (joint panel and both marginal panels)."""
    vlo, clo = spec['vpair']['lower'], spec['cs0']['lower']
    if vlo > xmin:
        for ax in (ax_joint, ax_marg_x):
            ax.axvspan(xmin, vlo, color="0.85", zorder=0); ax.axvline(vlo, color="0.4", lw=1.2)
    if clo > ymin:
        for ax in (ax_joint, ax_marg_y):
            ax.axhspan(ymin, clo, color="0.85", zorder=0); ax.axhline(clo, color="0.4", lw=1.2)


def plot_joint_prior_analytic(spec, out_path, title=r"Joint Prior: $V_0$ vs $C^S_1$"):
    """Exact prior density (no sample smoothing): filled contours + marginals."""
    vg = np.linspace(-500, 0, 501); cg = np.linspace(0, 250, 251)
    VV, CC = np.meshgrid(vg, cg, indexing="ij")
    f = prior_density(spec, VV, CC); f = f / f.max()
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(4, 4, figure=fig, hspace=0.1, wspace=0.1)
    ax_joint  = fig.add_subplot(gs[1:4, 0:3])
    ax_marg_x = fig.add_subplot(gs[0,   0:3], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1:4, 3],   sharey=ax_joint)
    ax_joint.contourf(VV, CC, np.where(f > 0, f, np.nan), levels=np.linspace(0, 1, 11), cmap='Blues')
    ax_joint.scatter(vpair_ME, cs0_ME, color='red', marker='*', s=200, edgecolor='black', linewidth=1.0,
                     label='Mustonen & Engel (2016)', zorder=5)
    mx = f.sum(axis=1); my = f.sum(axis=0)
    ax_marg_x.fill_between(vg, 0, mx / mx.max(), color='royalblue', alpha=0.6); ax_marg_x.plot(vg, mx / mx.max(), color='C0')
    ax_marg_y.fill_betweenx(cg, 0, my / my.max(), color='royalblue', alpha=0.6); ax_marg_y.plot(my / my.max(), cg, color='C0')
    ax_marg_x.axvline(vpair_ME, color='red', linestyle='--', linewidth=1.0)
    ax_marg_y.axhline(cs0_ME,   color='red', linestyle='--', linewidth=1.0)
    plt.setp(ax_marg_x.get_xticklabels(), visible=False); plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    ax_marg_x.set_yticks([]); ax_marg_y.set_xticks([])
    ax_joint.set_xlim(-500, 0); ax_joint.set_ylim(0, 250)
    ax_joint.set_xlabel(r"$V_0$", fontsize=18); ax_joint.set_ylabel(r"$C^S_1$", fontsize=18)
    ax_joint.tick_params(labelsize=16); ax_joint.legend(loc='upper right', frameon=True, fontsize=16)
    ax_joint.grid(True, alpha=0.3, linestyle=':')
    fig.savefig(out_path, bbox_inches='tight'); plt.close(fig)


def plot_joint_seaborn(post_v, post_c, out_path,
                       title=r"Joint Posterior: $V_0$ vs $C^S_1$", spec=None, support=None, est=None):
    """Seaborn-jointplot-styled counterpart of plot_joint_contour, drawn from the
    same density estimate (boundary_kde; est = its precomputed result). Built
    with matplotlib so seaborn does not smooth the samples again (2026-10-12).
    Style as sns.jointplot(kind="kde", fill=True, thresh=0.05, levels=10,
    height=7, space=0): filled contours at the density values enclosing
    5%, ..., 100% of the probability mass; filled marginal curves.
    spec = prior spec (shades the excluded regions); None = plain KDE."""
    spec = _spec_from_support(spec, support)
    if est is None:
        est = boundary_kde(post_v, post_c, spec)
    xmin, xmax = -500, 0
    ymin, ymax = 0, 250
    fig = plt.figure(figsize=(7, 7))
    gs = GridSpec(6, 6, figure=fig, hspace=0, wspace=0)          # seaborn ratio=5, space=0
    ax_joint  = fig.add_subplot(gs[1:, :-1])
    ax_marg_x = fig.add_subplot(gs[0, :-1], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1:, -1], sharey=ax_joint)

    f = est['f']
    vals = np.sort(f[np.isfinite(f)].ravel())[::-1]
    cum = np.cumsum(vals) / vals.sum()
    levels = np.take(vals, np.searchsorted(cum, 1 - np.linspace(0.05, 1, 10)), mode="clip")   # seaborn iso-proportion levels
    levels = np.unique(np.append(levels[:-1], vals[0] * (1 + 1e-9)))
    ax_joint.contourf(est['VV'], est['CC'], f, levels=levels, cmap="Blues")
    if spec is not None:
        _shade_excluded(spec, ax_joint, ax_marg_x, ax_marg_y, xmin, ymin)
    ax_joint.scatter(vpair_ME, cs0_ME, color='red', marker='*', s=300,
                     edgecolor='black', linewidth=1.0,
                     label='Mustonen & Engel (2016)', zorder=5)
    ax_marg_x.hist(post_v, bins=np.linspace(xmin, xmax, 201), density=True, color="royalblue", alpha=0.35)   # bin width 2.5
    ax_marg_y.hist(post_c, bins=np.linspace(ymin, ymax, 101), density=True, color="royalblue", alpha=0.35, orientation="horizontal")
    ax_marg_x.fill_between(est['vg'], 0, est['mv'], color="C0", alpha=0.25); ax_marg_x.plot(est['vg'], est['mv'], color="C0")
    ax_marg_y.fill_betweenx(est['cg'], 0, est['mc'], color="C0", alpha=0.25); ax_marg_y.plot(est['mc'], est['cg'], color="C0")
    ax_marg_x.axvline(vpair_ME, color='red', linestyle='--', linewidth=1.0)
    ax_marg_y.axhline(cs0_ME,   color='red', linestyle='--', linewidth=1.0)
    sns.despine(ax=ax_joint); sns.despine(ax=ax_marg_x, left=True); sns.despine(ax=ax_marg_y, bottom=True)
    ax_marg_x.set_yticks([]); ax_marg_y.set_xticks([])
    plt.setp(ax_marg_x.get_xticklabels(), visible=False); plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    ax_joint.set_xlabel(r"$V_0$", fontsize=18); ax_joint.set_ylabel(r"$C^S_1$", fontsize=18)
    # tick label size
    ax_marg_x.tick_params(labelsize=14); ax_marg_y.tick_params(labelsize=14)
    ax_joint.legend(loc='upper right', frameon=True, fontsize=16)
    ax_joint.set_xlim(xmin, xmax)
    ax_joint.set_ylim(ymin, ymax)
    # tick label size
    ax_joint.tick_params(labelsize=16)
    # fig.suptitle(title, y=1.03, fontsize=15)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_joint_hist2d(post_v, post_c, out_path,
                      title=r"Joint Posterior: $V_0$ vs $C^S_1$"):
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(4, 4, figure=fig, hspace=0.1, wspace=0.1)
    ax_joint  = fig.add_subplot(gs[1:4, 0:3])
    ax_marg_x = fig.add_subplot(gs[0,   0:3], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1:4, 3],   sharey=ax_joint)

    ax_joint.hist2d(post_v, post_c, bins=60, cmap='Blues', density=True)
    ax_joint.scatter(vpair_ME, cs0_ME, color='red', marker='*', s=200,
                     edgecolor='black', linewidth=1.0,
                     label='Mustonen & Engel (2016)', zorder=5)
    ax_marg_x.hist(post_v, bins=60, color='royalblue', alpha=0.7, density=True)
    ax_marg_y.hist(post_c, bins=60, color='royalblue', alpha=0.7, density=True,
                   orientation='horizontal')
    ax_marg_x.axvline(vpair_ME, color='red', linestyle='--', linewidth=1.0)
    ax_marg_y.axhline(cs0_ME,   color='red', linestyle='--', linewidth=1.0)
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    ax_marg_x.set_yticks([]); ax_marg_y.set_xticks([])
    ax_joint.set_xlim(-500, 0); ax_joint.set_ylim(0, 250)
    ax_joint.tick_params(labelsize=16)
    ax_joint.set_xlabel(r"$V_0$", fontsize=18)
    ax_joint.set_ylabel(r"$C^S_1$", fontsize=18)
    ax_joint.legend(loc='upper right', frameon=True, fontsize=16)
    ax_joint.grid(True, alpha=0.3, linestyle=':')
    # fig.suptitle(title, y=0.92, fontsize=14)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_joint_contour(post_v, post_c, out_path,
                       title=r"Joint Posterior: $V_0$ vs $C^S_1$", spec=None, support=None, est=None):
    """Filled contours of the boundary-aware posterior density (boundary_kde;
    est = its precomputed result) with histogram marginals (bin width 2.5) and
    the normalized marginal densities on top. spec = prior spec from the
    manifest (shades the excluded regions); None = plain KDE (2026-10-12)."""
    spec = _spec_from_support(spec, support)
    if est is None:
        est = boundary_kde(post_v, post_c, spec)
    xmin, xmax = -500, 0
    ymin, ymax = 0, 250
    xx, yy, f = est['VV'], est['CC'], est['f']

    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(4, 4, figure=fig, hspace=0.1, wspace=0.1)
    ax_joint  = fig.add_subplot(gs[1:4, 0:3])
    ax_marg_x = fig.add_subplot(gs[0,   0:3], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1:4, 3],   sharey=ax_joint)

    fmax = np.nanmax(f)
    ax_joint.contourf(xx, yy, np.where(f > 0.02 * fmax, f, np.nan), levels=np.linspace(0.02 * fmax, fmax, 11), cmap='Blues')   # near-zero left white
    if spec is not None:
        _shade_excluded(spec, ax_joint, ax_marg_x, ax_marg_y, xmin, ymin)
    ax_joint.scatter(vpair_ME, cs0_ME, color='red', marker='*', s=200,
                     edgecolor='black', linewidth=1.0,
                     label='Mustonen & Engel (2016)', zorder=5)
    ax_marg_x.hist(post_v, bins=np.linspace(xmin, xmax, 201), color='royalblue', alpha=0.35, density=True)   # bin width 2.5
    ax_marg_y.hist(post_c, bins=np.linspace(ymin, ymax, 101), color='royalblue', alpha=0.35, density=True,
                   orientation='horizontal')
    ax_marg_x.plot(est['vg'], est['mv'], color='C0', lw=1.8)
    ax_marg_y.plot(est['mc'], est['cg'], color='C0', lw=1.8)
    ax_marg_x.axvline(vpair_ME, color='red', linestyle='--', linewidth=1.0)
    ax_marg_y.axhline(cs0_ME,   color='red', linestyle='--', linewidth=1.0)
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    ax_marg_x.set_yticks([]); ax_marg_y.set_xticks([])
    ax_joint.set_xlabel(r"$V_0$", fontsize=18)
    ax_joint.set_ylabel(r"$C^S_1$", fontsize=18)
    # tick label size
    ax_joint.tick_params(labelsize=16)
    ax_joint.legend(loc='upper right', frameon=True, fontsize=16)
    ax_joint.grid(True, alpha=0.3, linestyle=':')
    ax_joint.set_xlim(xmin, xmax); ax_joint.set_ylim(ymin, ymax)
    # fig.suptitle(title, y=0.92, fontsize=14)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


# =====================================================================
# HL posterior-predictive
# =====================================================================
def _compile_hl_evaluators(functional, hl_keys):
    v_s = pt.dscalar('v'); c_s = pt.dscalar('c')
    funcs = {}
    missing = []
    for nuc in hl_keys:
        try:
            emu = HalfLifeEmulator(nuc, functional, n=5)
            funcs[nuc] = pytensor.function([v_s, c_s], emu(v_s, c_s))
        except FileNotFoundError as e:
            missing.append((nuc, str(e)))
    if missing:
        print("  (skipping HL nuclei without emulators):")
        for nuc, msg in missing:
            print(f"    {nuc}: {msg}")
    return funcs


def plot_hl(post_v, post_c, fig_dir, functional, hl_keys, n_samples=500, rng=None):
    rng = rng or np.random.default_rng(0)
    print("Compiling HL evaluators...")
    funcs = _compile_hl_evaluators(functional, hl_keys)
    if not funcs:
        print("  no HL emulators available; skipping HL plots")
        return
    idxs = rng.choice(len(post_v), size=min(n_samples, len(post_v)), replace=False)

    nuclei = list(funcs.keys())
    preds = {nuc: [] for nuc in nuclei}
    for i in idxs:
        v, c = float(post_v[i]), float(post_c[i])
        for nuc in nuclei:
            log10_hl = float(funcs[nuc](v, c))
            preds[nuc].append(10.0 ** log10_hl)

    exp_vals = [HL_EXP[n] for n in nuclei]
    medians, err_lo, err_hi, plot_data = [], [], [], []
    for nuc in nuclei:
        arr = np.array(preds[nuc])
        plot_data.append(arr)
        m = np.percentile(arr, 50)
        lo = np.percentile(arr, 2.5)
        hi = np.percentile(arr, 97.5)
        medians.append(m); err_lo.append(m - lo); err_hi.append(hi - m)

    # Parity plot
    plt.figure(figsize=(8, 7))
    # lo_all = min(min(exp_vals), min(np.array(medians) - np.array(err_lo))) * 0.5
    # hi_all = max(max(exp_vals), max(np.array(medians) + np.array(err_hi))) * 2.0
    lo_all = 1e-10
    hi_all = 1e+10
    plt.plot([lo_all, hi_all], [lo_all, hi_all], 'k--', alpha=0.5, label='Perfect Agreement')
    plt.errorbar(exp_vals, medians, yerr=[err_lo, err_hi], fmt='o',
                 color='royalblue', ecolor='lightblue', capsize=4, elinewidth=1.5,
                 label='Posterior Median (95% CI)')
    for i, nuc in enumerate(nuclei):
        plt.annotate(nuc, (exp_vals[i], medians[i]),
                     textcoords="offset points", xytext=(5, 5), ha='left', fontsize=12)
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Experimental Half-Life [s]', fontsize=16)
    plt.ylabel('Posterior Predictive Half-Life [s]', fontsize=16)
    # tick label size
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    # plt.title('Bayesian Calibration: Half-Life Parity')

    #Set the limits +/− 5 of the largest/smallest value in the data
    smallest_hl = min(min(exp_vals), min(np.array(medians) - np.array(err_lo)))
    largest_hl = max(max(exp_vals), max(np.array(medians) + np.array(err_hi)))
    plt.xlim(smallest_hl / 5, largest_hl * 5)
    plt.ylim(smallest_hl / 5, largest_hl * 5)

    plt.legend(fontsize=14); plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "hl_posterior_parity.pdf"))
    plt.close()

    # Violin
    plt.figure(figsize=(10, 6))
    parts = plt.violinplot(plot_data, positions=range(len(nuclei)),
                           showmeans=False, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('royalblue'); pc.set_alpha(0.5)
    plt.scatter(range(len(nuclei)), exp_vals, marker='*', color='red', s=100,
                zorder=5, label='Experimental Target')
    plt.errorbar(range(len(nuclei)), exp_vals,
                 yerr=[HL_ERR[n] for n in nuclei], fmt='none',
                 ecolor='red', capsize=10)
    plt.xticks(range(len(nuclei)), nuclei, rotation=45)
    plt.yscale('log'); plt.ylabel('Half-Life [s]')
    plt.title('Posterior Predictive Distributions: Half-Lives')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "hl_posterior_violin.pdf"))
    plt.close()


# =====================================================================
# GT posterior-predictive (broadened spectrum — requires full-dist emulators)
# =====================================================================
def _compile_gt_spectrum_evaluator(nuc, functional):
    """Build S_full(E) = S_K0(E) + 2*S_K1(E); training now separates K=0 and
    K=1 for every GT nucleus."""
    v_s = pt.dscalar('v'); c_s = pt.dscalar('c'); eg = pt.dvector('eg')
    emu0 = GTStrengthEmulator(nuc, K='0', functional=functional)
    emu1 = GTStrengthEmulator(nuc, K='1', functional=functional)
    spec = emu0.broadened(v_s, c_s, eg) + 2.0 * emu1.broadened(v_s, c_s, eg)
    return pytensor.function([v_s, c_s, eg], spec)


def plot_gt_posterior(post_v, post_c, fig_dir, functional, gt_keys,
                      n_samples=100, rng=None):
    rng = rng or np.random.default_rng(1)
    idxs = rng.choice(len(post_v), size=min(n_samples, len(post_v)), replace=False)

    for nuc in gt_keys:
        try:
            eval_func = _compile_gt_spectrum_evaluator(nuc, functional)
        except FileNotFoundError as e:
            print(f"  skipping GT posterior-predictive for {nuc} "
                  f"(full-dist emulator missing: {e})")
            continue

        print(f"  -> GT posterior-predictive: {nuc}")
        all_specs = np.array([eval_func(float(post_v[i]), float(post_c[i]), E_GRID)
                              for i in idxs])
        median  = np.percentile(all_specs, 50,   axis=0)
        high_95 = np.percentile(all_specs, 97.5, axis=0)
        Ex = E_GRID - config.get_gt_shift(nuc, functional)

        plt.figure(figsize=(8, 5))
        for spec in all_specs:
            plt.plot(Ex, spec, color='blue', alpha=0.15)
        plt.plot(Ex, median, color='darkblue', lw=2, label='Posterior Median')
        plt.axvline(GT_E_EXP[nuc], color='red', linestyle='--',
                    label=f'Exp Target: {GT_E_EXP[nuc]} MeV')
        w_lo, w_hi = config.get_gt_window(nuc)
        plt.axvspan(w_lo, w_hi, color='gray', alpha=0.2, label='Window')
        plt.title(f"Posterior GT Strength: {nuc}")
        plt.xlabel(r"Excitation Energy $E_x$ [MeV]")
        plt.ylabel("Strength [1/MeV]")
        plt.xlim(0, 25)
        ymax = float(np.max(high_95))
        plt.ylim(0, ymax * 1.2 if ymax > 0 else 1.0)
        plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"gt_posterior_{nuc}.pdf"))
        plt.close()


# =====================================================================
# Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt-mode',
                    choices=['full', 'full_centroid', 'full_peak', 'full_window_peak',
                             'centroid', 'centroid_total', 'peak'],
                    required=True)
    ap.add_argument('--functional', default='UNE1',
                    help="Functional name (subdir of results/). Default: UNE1.")
    ap.add_argument('--save-dir', default=None)
    ap.add_argument('--bayes-subdir', default='bayes',
                    help="Subdirectory under results/<functional>/ to read traces from. "
                         "Default: 'bayes'. Match what was passed to bayes_mcmc.py.")
    ap.add_argument('--n-hl-samples', type=int, default=500)
    ap.add_argument('--n-gt-samples', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-gt-spectrum', action='store_true',
                    help="Skip GT posterior-predictive spectrum plots "
                         "(only useful when full-dist GT emulators are available).")
    ap.add_argument('--hl-set', default='all',
                    choices=['all'] + list(config.HL_SETS),
                    help="HL calibration set to plot. Default: all. "
                         "Match what was passed to bayes_mcmc.py.")
    ap.add_argument('--gt-set', default='all',
                    choices=['all'] + list(config.GT_SETS),
                    help="GT calibration set to plot. Default: all. "
                         "Match what was passed to bayes_mcmc.py.")
    args = ap.parse_args()

    hl_keys = config.select_hl_nuclei(args.hl_set)
    gt_keys = config.select_gt_nuclei(args.gt_set)

    save_dir = args.save_dir or os.path.join(
        _bayes_root(args.functional, args.bayes_subdir), args.gt_mode)
    fig_dir = os.path.join(save_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    trace_path = os.path.join(save_dir, "trace.nc")
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Trace not found: {trace_path}")
    print(f"Loading trace: {trace_path}")
    trace = az.from_netcdf(trace_path)

    post_v = trace.posterior['vpair'].values.flatten()
    post_c = trace.posterior['cs0'].values.flatten()

    prior_spec = read_prior_spec_from_manifest(os.path.join(save_dir, "manifest.txt"))
    support = ((prior_spec['vpair']['lower'], prior_spec['vpair']['upper']),
               (prior_spec['cs0']['lower'], prior_spec['cs0']['upper']))
    print(f"Prior support for the KDE plots: V0 {support[0]}, C1s {support[1]}, "
          f"taper_w {prior_spec['vpair'].get('taper_w', 0.0)}")
    print("Joint posterior plots...")
    est = boundary_kde(post_v, post_c, prior_spec)   # shared by the seaborn-style and contour figures
    plot_joint_seaborn(post_v, post_c, os.path.join(fig_dir, "joint_posterior_seaborn.pdf"), spec=prior_spec, est=est)
    plot_joint_hist2d (post_v, post_c, os.path.join(fig_dir, "joint_posterior_hist2d.pdf"))
    plot_joint_contour(post_v, post_c, os.path.join(fig_dir, "joint_posterior_contour.pdf"), spec=prior_spec, est=est)

    print("Joint prior plots...")
    rng_prior = np.random.default_rng(args.seed + 1)
    print(f"Prior (from manifest): {describe_prior_spec(prior_spec)}")
    prior_v, prior_c = sample_prior(len(post_v), rng_prior, prior_spec)
    prior_title = r"Joint Prior: $V_0$ vs $C^S_1$"
    plot_joint_prior_analytic(prior_spec, os.path.join(fig_dir, "joint_prior_seaborn.pdf"), title=prior_title)   # exact density, no smoothing
    plot_joint_hist2d (prior_v, prior_c,
                       os.path.join(fig_dir, "joint_prior_hist2d.pdf"),
                       title=prior_title)
    plot_joint_prior_analytic(prior_spec, os.path.join(fig_dir, "joint_prior_contour.pdf"), title=prior_title)   # exact density, no smoothing

    print("Trace plot...")
    az.plot_trace(trace, var_names=['vpair', 'cs0'])
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "trace.pdf"))
    plt.close()

    print("HL posterior-predictive...")
    rng = np.random.default_rng(args.seed)
    plot_hl(post_v, post_c, fig_dir, args.functional, hl_keys,
            n_samples=args.n_hl_samples, rng=rng)

    if not args.no_gt_spectrum:
        print("GT posterior-predictive spectra (requires full-dist emulators)...")
        plot_gt_posterior(post_v, post_c, fig_dir,
                          functional=args.functional,
                          gt_keys=gt_keys,
                          n_samples=args.n_gt_samples, rng=rng)

    print(f"Done. Figures in {fig_dir}")


if __name__ == '__main__':
    main()

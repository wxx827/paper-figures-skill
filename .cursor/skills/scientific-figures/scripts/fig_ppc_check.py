"""Posterior predictive check for a Gamma likelihood fitted to a
contaminated sample.

(a) Density overlay: the observed sample against 60 posterior predictive
    replicates plus the pointwise 50% / 95% predictive envelope.
(b-e) Posterior predictive distributions of four test statistics with the
    observed value and the two-sided Bayesian p-value; the tail beyond the
    observation is shaded so the p-value is readable off the picture.

The generating process adds a small heavy-tailed contamination the Gamma
model cannot represent, so location and spread check out while the maximum
shows a clear misfit -- the situation a PPC is meant to expose.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import gaussian_kde
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

OBS_C = "#1a1a1a"
REP_C = "#6fa8cf"
BAND_C = "#cfe1ee"
BAD_C = "#b23a48"
GOOD_C = "#2e7d5b"


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(2718)

    # ---- observed data: Gamma core + rare heavy-tailed contamination -------
    n = 180
    core = rng.gamma(4.0, 1.5, n)
    hit = rng.random(n) < 0.045
    core[hit] *= rng.uniform(1.9, 2.8, hit.sum())
    y_obs = core

    # ---- posterior over (shape, scale) via a normal approximation ---------
    a_hat, _, s_hat = stats.gamma.fit(y_obs, floc=0)
    n_rep = 3000
    a_draw = a_hat * np.exp(rng.normal(0, 0.075, n_rep))
    s_draw = s_hat * np.exp(rng.normal(0, 0.080, n_rep))
    y_rep = rng.gamma(a_draw[:, None], s_draw[:, None], size=(n_rep, n))

    fig = plt.figure(figsize=(7.4, 6.4))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.42, 1.0, 1.0],
                  width_ratios=[1.62, 1.0], hspace=0.60, wspace=0.20,
                  left=0.072, right=0.985, top=0.868, bottom=0.098)

    # ---- (a) density overlay ----------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    style_axes(ax)
    grid = np.linspace(0, np.percentile(y_obs, 99.9) * 1.12, 400)
    dens = np.vstack([gaussian_kde(r)(grid) for r in y_rep[:400]])
    q = np.percentile(dens, [2.5, 25, 75, 97.5], axis=0)
    ax.fill_between(grid, q[0], q[3], color=BAND_C, alpha=0.75, lw=0,
                    label="95% envelope")
    ax.fill_between(grid, q[1], q[2], color="#9cc3dd", alpha=0.9, lw=0,
                    label="50% envelope")
    for r in y_rep[:60]:
        ax.plot(grid, gaussian_kde(r)(grid), color=REP_C, lw=0.35, alpha=0.30)
    ax.plot([], [], color=REP_C, lw=0.8, alpha=0.7,
            label=r"replicates $y^{rep}$")
    obs_dens = gaussian_kde(y_obs)(grid)
    exceed = obs_dens > q[3]
    ax.fill_between(grid, q[3], obs_dens, where=exceed, color=BAD_C,
                    alpha=0.30, lw=0, zorder=4,
                    label="observed above envelope")
    ax.plot(grid, obs_dens, color=OBS_C, lw=2.0,
            label=r"observed $y$", zorder=6)
    ax.plot(y_obs, np.full(n, -0.004), marker="|", ls="none", ms=3.4,
            mew=0.5, color="#555555", alpha=0.55, clip_on=False, zorder=5)
    ax.set_xlim(0, grid[-1])
    ax.set_ylim(-0.007, q[3].max() * 1.22)
    ax.set_xlabel("response $y$ (arbitrary units)", labelpad=1)
    ax.set_ylabel("density")
    add_panel_label(ax, "(a)", x=-0.085, y=1.02, size=10)

    # ---- (b) tail zoom: the misfit is invisible on the common scale --------
    cut = 13.0
    m = grid >= cut
    axi = fig.add_subplot(gs[0, 1])
    style_axes(axi)
    axi.fill_between(grid[m], q[0][m], q[3][m], color=BAND_C, alpha=0.8, lw=0)
    axi.fill_between(grid[m], q[1][m], q[2][m], color="#9cc3dd", alpha=0.9,
                     lw=0)
    axi.fill_between(grid[m], q[3][m], obs_dens[m], where=exceed[m],
                     color=BAD_C, alpha=0.35, lw=0)
    for r in y_rep[:60]:
        axi.plot(grid[m], gaussian_kde(r)(grid[m]), color=REP_C, lw=0.3,
                 alpha=0.28)
    axi.plot(grid[m], obs_dens[m], color=OBS_C, lw=1.6, zorder=6)
    tail_pts = y_obs[y_obs >= cut]
    axi.plot(tail_pts, np.full(tail_pts.size, 0.0), marker="|", ls="none",
             ms=3.6, mew=0.6, color="#555555", alpha=0.7, clip_on=False,
             zorder=5)
    axi.set_xlim(cut, grid[-1])
    axi.set_ylim(0, max(q[3][m].max(), obs_dens[m].max()) * 1.22)
    axi.set_xlabel("response $y$", labelpad=1)
    axi.set_ylabel("density")
    axi.tick_params(labelsize=7)
    axi.text(0.97, 0.97, "tail zoom  ($y \\geq 13$)", transform=axi.transAxes,
             ha="right", va="top", fontsize=7.0, color="#555555",
             style="italic")
    add_panel_label(axi, "(b)", x=-0.135, y=1.02, size=10)

    # ---- (b-e) test statistics --------------------------------------------
    tests = [
        ("mean", r"$T = \mathrm{mean}(y)$", np.mean),
        ("sd", r"$T = \mathrm{sd}(y)$", np.std),
        ("min", r"$T = \min(y)$", np.min),
        ("max", r"$T = \max(y)$", np.max),
    ]
    for k, (key, tex, fn) in enumerate(tests):
        axk = fig.add_subplot(gs[1 + k // 2, k % 2])
        style_axes(axk)
        t_rep = fn(y_rep, axis=1)
        t_obs = float(fn(y_obs))
        p_hi = float(np.mean(t_rep >= t_obs))
        p_two = 2 * min(p_hi, 1 - p_hi)
        flagged = p_two < 0.05
        col = BAD_C if flagged else GOOD_C

        cnt, edges = np.histogram(t_rep, bins=42)
        centres = 0.5 * (edges[:-1] + edges[1:])
        wid = edges[1] - edges[0]
        tail = centres >= t_obs if p_hi < 0.5 else centres <= t_obs
        axk.bar(centres[~tail], cnt[~tail], width=wid, color="#cfe1ee",
                edgecolor="#7ba7c7", lw=0.3)
        axk.bar(centres[tail], cnt[tail], width=wid, color=col,
                edgecolor="none", alpha=0.55)
        axk.axvline(t_obs, color=OBS_C, lw=1.6, zorder=6)
        axk.set_ylim(0, cnt.max() * 1.42)
        lo = min(edges[0], t_obs)
        hi = max(edges[-1], t_obs)
        pad = 0.07 * (hi - lo)
        axk.set_xlim(lo - pad, hi + pad)
        axk.set_yticks([])
        axk.spines["left"].set_visible(False)
        axk.set_xlabel(tex, labelpad=1, fontsize=8.2)
        axk.tick_params(labelsize=7)

        side = "left" if t_obs > np.median(t_rep) else "right"
        tx = 0.015 if side == "left" else 0.985
        axk.text(tx, 0.99, f"$T(y)$ = {t_obs:.2f}", transform=axk.transAxes,
                 ha=side, va="top", fontsize=6.8, color=OBS_C,
                 fontweight="bold")
        axk.text(tx, 0.855,
                 f"$p_B$ = {p_two:.3f}" + ("  misfit" if flagged else "  ok"),
                 transform=axk.transAxes, ha=side, va="top", fontsize=6.8,
                 color=col, fontweight="bold")
        add_panel_label(axk, f"({'cdef'[k]})", x=-0.045, y=1.02, size=10)

    handles, labels = ax.get_legend_handles_labels()

    savefig(fig, "130_ppc_check.png")


if __name__ == "__main__":
    main()

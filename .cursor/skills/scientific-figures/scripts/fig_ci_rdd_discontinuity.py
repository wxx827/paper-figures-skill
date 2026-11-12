"""Figure 770 - Sharp regression discontinuity at an eligibility cutoff.

Main panel: binned means of the outcome against the running variable, with
marker area proportional to the number of observations in the bin, a
triangular-kernel local linear fit computed separately on each side of the
cutoff with a pointwise 95 % band, and a vertical double arrow at the cutoff
whose length is the local average treatment effect.  Bins never straddle the
cutoff - a single bin spanning it would smear away the very jump the figure
exists to show.

Auxiliary (both at 58 % of the main panel width): the McCrary density test,
where the same local linear machinery is run on binned counts to check that
the running variable is not manipulated at the threshold; and the bandwidth
sensitivity curve, the estimate and its interval as the kernel bandwidth is
swept, with the reference bandwidth marked.

Every number - the jump, its robust standard error, the t statistic, the
log-density discontinuity and the effective sample sizes - is computed from
the simulated observations that are plotted.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_label_inches

INK, SUB, MUTED, FAINT = "#1a1a1a", "#555555", "#777777", "#8a8a8a"
HAIR, GRID, BASE = "#cccccc", "#d9d9d9", "#808080"
CARD = "#f5f5f5"

RIGHT = "#b23a48"        # eligible side (treated): the emphasis
LEFT = "#3b5b92"         # ineligible side (control)

N_OBS = 2600
SPAN = 34.0              # plotted range of the running variable, in points
H_REF = 12.0             # reference bandwidth
TAU = 8.4                # true jump, in percentage points
BIN_W = 2.0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def smooth_part(x):
    """Continuous, deliberately non-linear baseline in the running variable."""
    return 41.0 + 0.30 * x + 0.0048 * x ** 2 + 3.1 * np.tanh(x / 15.0)


def make_data(rng):
    # A mildly skewed, perfectly continuous density: no bunching at zero.
    x = np.concatenate([rng.normal(-4.0, 15.0, int(N_OBS * 0.62)),
                        rng.normal(9.0, 12.0, N_OBS - int(N_OBS * 0.62))])
    x = x[np.abs(x) <= SPAN]
    y = (smooth_part(x) + TAU * (x >= 0.0)
         + rng.normal(0.0, 6.4, x.size)
         + 1.6 * rng.standard_t(6, x.size))
    return x, y


def loclin(x, y, x0, h, side):
    """Triangular-kernel local linear fit; returns intercept, its SE and n.

    The heteroskedasticity-robust sandwich is the honest variance here: the
    outcome spread is not constant across the running variable, and a
    homoskedastic SE would make the band at the cutoff look tighter than the
    data can support.
    """
    m = (x >= x0) & (x <= x0 + h) if side > 0 else (x <= x0) & (x >= x0 - h)
    if m.sum() < 8:
        return np.nan, np.nan, int(m.sum())
    xs, ys = x[m] - x0, y[m]
    w = np.clip(1.0 - np.abs(xs) / h, 1e-6, None)
    xm = np.column_stack([np.ones_like(xs), xs])
    xtwx = xm.T @ (xm * w[:, None])
    try:
        b = np.linalg.solve(xtwx, xm.T @ (w * ys))
    except np.linalg.LinAlgError:
        return np.nan, np.nan, int(m.sum())
    e = ys - xm @ b
    meat = xm.T @ (xm * (w ** 2 * e ** 2)[:, None])
    inv = np.linalg.inv(xtwx)
    v = inv @ meat @ inv
    return float(b[0]), float(np.sqrt(max(v[0, 0], 0.0))), int(m.sum())


def rd_estimate(x, y, h):
    a_l, s_l, n_l = loclin(x, y, 0.0, h, -1)
    a_r, s_r, n_r = loclin(x, y, 0.0, h, +1)
    tau = a_r - a_l
    se = float(np.sqrt(s_l ** 2 + s_r ** 2))
    return tau, se, a_l, a_r, n_l, n_r


def fit_curve(x, y, h, side, n_grid=70, reach=None):
    edge = SPAN if reach is None else reach
    grid = (np.linspace(0.0, edge, n_grid) if side > 0
            else np.linspace(-edge, 0.0, n_grid))
    est = np.array([loclin(x, y, g, h, side)[:2] for g in grid])
    return grid, est[:, 0], est[:, 1]


def binned(x, y, side):
    edges = (np.arange(0.0, SPAN + BIN_W, BIN_W) if side > 0
             else np.arange(-SPAN, BIN_W, BIN_W))
    idx = np.digitize(x, edges) - 1
    out = []
    for b in range(edges.size - 1):
        m = idx == b
        if m.sum() >= 4:
            out.append((0.5 * (edges[b] + edges[b + 1]), y[m].mean(),
                        int(m.sum())))
    return np.array(out)


def mccrary(x, h=15.0, bw=1.4):
    """Local linear density on binned counts, run separately on each side."""
    edges = np.arange(-SPAN, SPAN + bw, bw)
    cnt, _ = np.histogram(x, bins=edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    dens = cnt / (x.size * bw)
    left = mid < 0
    a_l, s_l, _ = loclin(mid[left], dens[left], 0.0, h, -1)
    a_r, s_r, _ = loclin(mid[~left], dens[~left], 0.0, h, +1)
    theta = np.log(max(a_r, 1e-9)) - np.log(max(a_l, 1e-9))
    se = float(np.sqrt((s_r / a_r) ** 2 + (s_l / a_l) ** 2))
    curves = {}
    for side, sub in ((+1, ~left), (-1, left)):
        g, f, s = fit_curve(mid[sub], dens[sub], h, side, n_grid=48)
        curves[side] = (g, f, s)
    return mid, dens, curves, theta, se, (a_l, a_r)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def rd_panel(ax, x, y, tau, se, a_l, a_r, n_l, n_r):
    ax.axvspan(0.0, SPAN + 1.0, facecolor=CARD, edgecolor="none", zorder=0)
    ax.axvline(0.0, color="#3a3a3a", lw=1.1, ls=(0, (3.4, 2.2)), zorder=5)

    smax = max(binned(x, y, -1)[:, 2].max(), binned(x, y, +1)[:, 2].max())
    for side, col, lab in ((-1, LEFT, "below cutoff (ineligible)"),
                           (+1, RIGHT, "above cutoff (eligible)")):
        b = binned(x, y, side)
        ax.scatter(b[:, 0], b[:, 1], s=9.0 + 62.0 * b[:, 2] / smax,
                   facecolor=lighten(col, 0.42), edgecolor=col, linewidth=0.6,
                   zorder=3, label=lab)
        g, f, s = fit_curve(x, y, H_REF, side, reach=SPAN - 3.0)
        ok = np.isfinite(f)
        ax.fill_between(g[ok], (f - 1.96 * s)[ok], (f + 1.96 * s)[ok],
                        facecolor=col, alpha=0.20, edgecolor="none", zorder=2)
        ax.plot(g[ok], f[ok], lw=1.8, color=col, zorder=4)

    ax.plot([0.0], [a_l], marker="o", ms=5.0, mfc="white", mec=LEFT, mew=1.4,
            zorder=6)
    ax.plot([0.0], [a_r], marker="o", ms=5.0, mfc="white", mec=RIGHT, mew=1.4,
            zorder=6)
    ax.annotate("", xy=(0.0, a_r), xytext=(0.0, a_l),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.3,
                                shrinkA=0, shrinkB=0, mutation_scale=9),
                zorder=7)
    ax.text(1.4, 0.5 * (a_l + a_r),
            f"LATE = {tau:+.1f} pp\n(SE {se:.1f}, "
            f"t = {tau / se:.2f})", ha="left", va="center", fontsize=7.2,
            color=INK, fontweight="bold", linespacing=1.45, zorder=8)

    ax.text(-1.0, 0.035, "ineligible", transform=ax.get_xaxis_transform(),
            ha="right", va="bottom", fontsize=6.6, color=MUTED)
    ax.text(1.0, 0.035, "eligible", transform=ax.get_xaxis_transform(),
            ha="left", va="bottom", fontsize=6.6, color=SUB)

    ax.set_xlim(-SPAN - 1.0, SPAN + 1.0)
    ax.set_xlabel("running variable: exam score minus cutoff (points)",
                  fontsize=8.0)
    ax.set_ylabel("programme enrolment (pp)", fontsize=8.0)
    ax.tick_params(labelsize=7.0, length=2.5)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_title("A discontinuity in enrolment where the score rule bites",
                 fontsize=9.0, fontweight="bold", pad=3.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    leg = ax.legend(loc="upper left", fontsize=7.0, frameon=False,
                    handlelength=1.0, borderaxespad=0.35, labelspacing=0.32,
                    scatterpoints=1)
    for h in leg.legend_handles:
        h.set_sizes([22.0])
    ax.text(0.975, 0.055,
            f"marker area scales with bin count, {BIN_W:.0f}-point bins\n"
            f"triangular kernel, h = {H_REF:.0f} points\n"
            f"n = {x.size:,} ({n_l} left / {n_r} right in window)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6,
            color=SUB, linespacing=1.45)


def density_panel(ax, mid, dens, curves, theta, se):
    ax.bar(mid, dens, width=1.25, facecolor="#e2e2e2", edgecolor="white",
           linewidth=0.3, zorder=1)
    for side, col in ((-1, LEFT), (+1, RIGHT)):
        g, f, s = curves[side]
        ok = np.isfinite(f)
        ax.fill_between(g[ok], (f - 1.96 * s)[ok], (f + 1.96 * s)[ok],
                        facecolor=col, alpha=0.22, edgecolor="none", zorder=2)
        ax.plot(g[ok], f[ok], lw=1.5, color=col, zorder=3)
    ax.axvline(0.0, color="#3a3a3a", lw=0.9, ls=(0, (3.0, 2.2)), zorder=4)

    p = 2.0 * stats.norm.sf(abs(theta / se))
    ax.set_xlim(-SPAN - 1.0, SPAN + 1.0)
    ax.set_ylim(0.0, dens.max() * 1.38)
    ax.set_xlabel("running variable (points)", fontsize=7.4)
    ax.set_ylabel("density (1/point)", fontsize=7.4)
    ax.tick_params(labelsize=6.5, length=2.2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_title("McCrary test: no bunching at the cutoff", fontsize=8.5,
                 fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.02, 0.96,
            f"log density jump {theta:+.3f} (SE {se:.3f}), p = {p:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
            color=SUB)


def bandwidth_panel(ax, hs, taus, ses):
    ax.fill_between(hs, taus - 1.96 * ses, taus + 1.96 * ses,
                    facecolor=RIGHT, alpha=0.20, edgecolor="none", zorder=2)
    ax.plot(hs, taus, lw=1.7, color=RIGHT, zorder=3)
    ax.axhline(0.0, color=BASE, lw=0.8, ls=(0, (3.0, 2.2)), zorder=1)
    ax.axvline(H_REF, color="#3a3a3a", lw=0.9, ls=(0, (2.6, 2.0)), zorder=4)
    j = int(np.argmin(np.abs(hs - H_REF)))
    ax.plot([H_REF], [taus[j]], marker="o", ms=4.2, mfc="white", mec=RIGHT,
            mew=1.3, zorder=5)
    ax.text(H_REF + 0.6, ax.get_ylim()[0] + 0.06 * np.ptp(ax.get_ylim()),
            f"reference h = {H_REF:.0f}", ha="left", va="bottom",
            fontsize=6.3, color=SUB)

    ax.set_xlim(hs[0], hs[-1])
    ax.set_xlabel("bandwidth (score points)", fontsize=7.4)
    ax.set_ylabel("LATE (pp, 95 % CI)", fontsize=7.4)
    ax.tick_params(labelsize=6.5, length=2.2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_title("Estimate is flat over a wide bandwidth range",
                 fontsize=8.5, fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(770)
    x, y = make_data(rng)

    tau, se, a_l, a_r, n_l, n_r = rd_estimate(x, y, H_REF)
    mid, dens, curves, theta, th_se, _ = mccrary(x)

    hs = np.arange(4.0, 30.5, 0.5)
    res = np.array([rd_estimate(x, y, h)[:2] for h in hs])
    taus, ses = res[:, 0], res[:, 1]

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (0.86, 0.82, 5.24, 4.52))
    ax_den = axes_inches(fig, (7.05, 3.48, 3.05, 1.86))
    ax_bw = axes_inches(fig, (7.05, 0.82, 3.05, 1.86))

    rd_panel(ax_main, x, y, tau, se, a_l, a_r, n_l, n_r)
    density_panel(ax_den, mid, dens, curves, theta, th_se)
    bandwidth_panel(ax_bw, hs, taus, ses)


    for ax, lab in ((ax_main, "(a)"), (ax_den, "(b)"), (ax_bw, "(c)")):
        panel_label_inches(fig, ax, lab, dx=-0.62, dy=0.10)
    savefig(fig, "770_ci_rdd_discontinuity.png")


if __name__ == "__main__":
    main()

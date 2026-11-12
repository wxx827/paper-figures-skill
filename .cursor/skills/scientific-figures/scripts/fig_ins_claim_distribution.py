"""Claim severity: a heavy tail, and what a deductible does to it.

Main panel: the severity histogram on a log money axis, in density per decade
of USD so the body and the tail are legible in one frame, with a lognormal
body fit and a Pareto tail fit drawn over it.  The mean, the median and the
99th percentile are marked to show how far apart three "typical" numbers can
be once the tail is heavy.  The two narrow right-hand panels carry the pricing
consequences: the mean excess-loss function, whose upward slope is the tail
signature, and the deductible trade-off between claim count and loss volume.

All severities come from ``gen_claims``; swap it for a real loss run and the
fits, statistics and annotations follow automatically.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, style_axes
from figkit import panel_labels_inches

BLUE = "#3b5b92"
GOLD = "#c9a227"
RED = "#b23a48"
GREEN = "#2e7d5b"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRID = "#ededed"
BAR = "#cfe1ee"

N_CLAIMS = 24000
TAIL_SHARE = 0.045        # fraction of claims drawn from the Pareto component
LN_MU, LN_SIGMA = 7.85, 1.02
PAR_SCALE, PAR_ALPHA = 45000.0, 1.55
FLOOR = 60.0              # smallest reportable claim, USD


def gen_claims(seed=637):
    """Ground-up claim severities in USD: lognormal body, Pareto tail."""
    rng = np.random.default_rng(seed)
    n_tail = int(round(N_CLAIMS * TAIL_SHARE))
    body = rng.lognormal(LN_MU, LN_SIGMA, N_CLAIMS - n_tail)
    u = rng.random(n_tail)
    tail = PAR_SCALE * (1.0 - u) ** (-1.0 / PAR_ALPHA)
    x = np.concatenate([body, tail])
    return np.maximum(x, FLOOR)


def lognormal_fit(x):
    """MLE of the lognormal body, returned as (mu, sigma) of log-severity."""
    lx = np.log(x)
    return float(lx.mean()), float(lx.std(ddof=1))


def hill_fit(x, q=0.95):
    """Hill tail index above the q-quantile; returns (threshold, alpha)."""
    u = float(np.quantile(x, q))
    exc = x[x > u]
    alpha = 1.0 / float(np.mean(np.log(exc / u)))
    return u, alpha, exc.size / x.size


def mean_excess(x, grid):
    """e(d) = E[X - d | X > d] on a deductible grid, in USD."""
    out = np.full(grid.size, np.nan)
    for i, d in enumerate(grid):
        exc = x[x > d]
        if exc.size >= 60:
            out[i] = exc.mean() - d
    return out


def main():
    set_pub_style(base_font=8)
    x = gen_claims()
    n = x.size

    mu, sigma = lognormal_fit(x)
    u_thr, alpha, tail_frac = hill_fit(x)

    med = float(np.median(x))
    mean = float(x.mean())
    var99 = float(np.quantile(x, 0.99))
    tvar99 = float(x[x >= var99].mean())
    ratio = mean / med
    tail_load = float(x[x >= var99].sum() / x.sum())

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          left=0.062, right=0.978, top=0.882, bottom=0.128,
                          hspace=0.50, wspace=0.215)
    ax = fig.add_subplot(gs[:, 0])
    ax_e = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1])

    for a in (ax, ax_e, ax_d):
        a.set_axisbelow(True)
        a.grid(color=GRID, lw=0.7, zorder=0)
        style_axes(a)

    # ---- main: severity histogram in density per decade ---------------------
    lo, hi = 1.6, 7.2                      # log10 USD
    edges = np.logspace(lo, hi, 46)
    counts, _ = np.histogram(x, bins=edges)
    dex = np.diff(np.log10(edges))
    dens = counts / (n * dex)
    centres = np.sqrt(edges[:-1] * edges[1:])
    ax.bar(centres, dens, width=np.diff(edges), align="center", color=BAR,
           edgecolor="#9cc3dd", linewidth=0.5, zorder=3)

    xg = np.logspace(lo, hi, 700)
    ln_pdf = (np.log(10.0) / (sigma * np.sqrt(2 * np.pi))
              * np.exp(-0.5 * ((np.log(xg) - mu) / sigma) ** 2))
    ax.plot(xg, ln_pdf, color=GOLD, lw=1.9, zorder=6)

    xt = xg[xg >= u_thr]
    par_pdf = (np.log(10.0) * tail_frac * alpha
               * (u_thr / xt) ** alpha)
    ax.plot(xt, par_pdf, color=RED, lw=2.0, zorder=7)
    ax.axvspan(u_thr, 10 ** hi, facecolor="#f7eef0", lw=0, zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10 ** lo, 10 ** hi)
    ax.set_ylim(2e-4, 6.0)
    ax.set_xlabel("Ground-up claim severity (USD, log scale)", fontsize=9.0)
    ax.set_ylabel("Probability density (per decade of USD, log scale)",
                  fontsize=9.0)
    ax.set_title("One tail, three very different 'typical' claims",
                 fontsize=9.0, fontweight="bold", pad=5.0)

    xtr = ax.get_xaxis_transform()
    marks = [(med, "median", "#808080", 0.52, True),
             (mean, "mean", BLUE, 0.94, False),
             (var99, "$VaR_{99\\%}$", RED, 0.94, False)]
    for v, lab, col, yf, boxed in marks:
        ax.axvline(v, color=col, lw=1.2, ls=(0, (3.0, 2.0)), zorder=8)
        bbox = dict(boxstyle="round,pad=0.28", facecolor="white",
                    edgecolor="none", alpha=0.88) if boxed else None
        ax.text(v * 1.10, yf, f"{lab}\n{v:,.0f} USD", ha="left", va="top",
                fontsize=6.9, color=col, linespacing=1.35, transform=xtr,
                bbox=bbox, zorder=9)

    ax.annotate("", xy=(var99, 0.085), xytext=(med, 0.085), xycoords=xtr,
                arrowprops=dict(arrowstyle="<->", lw=1.0, color=SUB,
                                shrinkA=0, shrinkB=0), zorder=8)
    ax.text(np.sqrt(med * var99), 0.100,
            f"{var99 / med:,.0f}$\\times$ from median to $VaR_{{99\\%}}$",
            ha="center", va="bottom", fontsize=7.0, color=SUB,
            transform=xtr, zorder=9)

    ax.text(0.985, 0.700,
            f"mean / median = {ratio:.2f}\n"
            f"$TVaR_{{99\\%}}$ = {tvar99:,.0f} USD\n"
            f"top 1% of claims carry {tail_load:.0%} of loss\n"
            f"Hill index $\\alpha$ = {alpha:.2f} above {u_thr:,.0f} USD",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.0,
            color=INK, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.94),
            zorder=10)

    handles = [
        Patch(facecolor=BAR, edgecolor="#9cc3dd", lw=0.5,
              label=f"empirical, n = {n:,}"),
        Line2D([], [], color=GOLD, lw=1.9,
               label=f"lognormal body ($\\sigma$ = {sigma:.2f})"),
        Line2D([], [], color=RED, lw=2.0, label="Pareto tail fit"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7.4,
              handlelength=1.9, labelspacing=0.45, borderpad=0.35)

    # ---- aux (b): mean excess loss -----------------------------------------
    grid = np.logspace(2.0, 6.0, 90)
    me = mean_excess(x, grid)
    ok = np.isfinite(me)
    ax_e.plot(grid[ok], me[ok] / 1e3, color=RED, lw=1.8, zorder=5)
    ax_e.set_xscale("log")
    ax_e.set_xlim(1e2, 1e6)
    ax_e.set_ylim(0, np.nanmax(me[ok]) / 1e3 * 1.20)
    ax_e.tick_params(labelsize=6.8)
    ax_e.set_xlabel("deductible  $d$ (USD, log scale)", fontsize=8.0,
                    labelpad=1.5)
    ax_e.set_ylabel("$e(d)$ (thousand USD)", fontsize=8.0)
    ax_e.set_title("Mean excess loss keeps rising", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    slope = np.polyfit(np.log(grid[ok][-30:]), me[ok][-30:] / 1e3, 1)[0]
    ax_e.text(0.035, 0.955,
              "an increasing $e(d)$ is the\nheavy-tail signature\n"
              f"(slope {slope:.0f}k USD per $e$-fold)",
              transform=ax_e.transAxes, ha="left", va="top", fontsize=6.5,
              color=SUB, linespacing=1.5)

    # ---- aux (c): deductible trade-off -------------------------------------
    dg = np.logspace(1.8, 5.0, 120)
    kept_loss = np.array([np.maximum(x - d, 0.0).sum() for d in dg]) / x.sum()
    kept_count = np.array([(x > d).mean() for d in dg])
    ax_d.plot(dg, kept_loss * 100.0, color=BLUE, lw=1.8, zorder=6)
    ax_d.plot(dg, kept_count * 100.0, color=GREEN, lw=1.8, ls=(0, (3.4, 2.0)),
              zorder=5)
    ax_d.set_xscale("log")
    ax_d.set_xlim(10 ** 1.8, 1e5)
    ax_d.set_ylim(0, 105)
    ax_d.set_yticks([0, 25, 50, 75, 100])
    ax_d.tick_params(labelsize=6.8)
    ax_d.set_xlabel("deductible  $d$ (USD, log scale)", fontsize=8.0,
                    labelpad=1.5)
    ax_d.set_ylabel("share retained (%)", fontsize=8.0)
    ax_d.set_title("A deductible cuts count, not loss", fontsize=8.5,
                   fontweight="bold", pad=4.0)

    d_star = 2500.0
    i = int(np.argmin(np.abs(dg - d_star)))
    ax_d.axvline(dg[i], color="#9a9a9a", lw=0.8, ls=(0, (2.4, 2.0)), zorder=3)
    ax_d.plot([dg[i], dg[i]], [kept_loss[i] * 100, kept_count[i] * 100],
              marker="o", ms=3.4, mfc="white", mew=1.2, ls="none",
              mec="#333333", zorder=8)
    ax_d.text(dg[i] * 2.0, 52,
              f"at {dg[i]:,.0f} USD:\n{kept_count[i]:.0%} of claims remain\n"
              f"but {kept_loss[i]:.0%} of loss does",
              ha="left", va="top", fontsize=6.5, color=SUB, linespacing=1.5)
    ax_d.legend(handles=[
        Line2D([], [], color=BLUE, lw=1.8, label="loss volume above $d$"),
        Line2D([], [], color=GREEN, lw=1.8, ls=(0, (3.4, 2.0)),
               label="claim count above $d$")],
        loc="lower left", frameon=False, fontsize=6.6, handlelength=2.0,
        labelspacing=0.35, borderpad=0.3)

    panel_labels_inches(fig, [ax, ax_e, ax_d], dx=-0.42, dy=0.06)

    savefig(fig, "637_ins_claim_distribution.png")


if __name__ == "__main__":
    main()

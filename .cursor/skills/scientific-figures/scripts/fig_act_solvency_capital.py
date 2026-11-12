"""Solvency capital: where the 99.5% line falls, and what it costs to stand.

Main panel: the simulated one-year aggregate loss distribution with the two
risk measures marked on it -- value at risk at 99.5% and the tail value at
risk beyond it -- and the 0.5% tail shaded.  The capital requirement is the
distance from the expected loss to the VaR, and the own-funds rule drawn over
it shows how many times that requirement is actually covered.  The two narrow
right-hand panels carry the aggregation and the sensitivity: how the module
capitals combine into a smaller total once correlation is admitted, and how
the requirement moves with the chosen confidence level.

Losses are a compound Poisson-lognormal simulation; the quantiles, tail means,
diversification benefit and coverage ratio are all read off that sample, so
TVaR is above VaR and the aggregate is below the sum of the modules by
construction rather than by assertion.  Values are illustrative.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

LOSS = "#b23a48"
DEEP = "#8c2f2f"
CAP = "#3b5b92"
DIV = "#2a7f7a"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"
DARK = "#3a3a3a"
FILL = lighten("#b23a48", 0.80)
TAIL = lighten("#b23a48", 0.52)

FIG_W, FIG_H = 10.4, 6.4
MAIN_RECT = (0.90, 0.74, 5.20, 4.68)
AUX_T_RECT = (7.10, 3.47, 2.90, 1.95)
AUX_B_RECT = (7.10, 0.74, 2.90, 1.95)

N_SIMS = 60000
LAMBDA = 140.0            # claims per year
SEV_MU = np.log(0.62e6)   # lognormal severity, USD
SEV_SIGMA = 1.05
ALPHA = 0.995             # supervisory confidence level
COVERAGE = 1.31           # own funds as a multiple of the requirement

# non-underwriting module capital (M USD) and the aggregation correlations
MODULES = ["market", "credit", "underwriting"]
MODULE_CAP = {"market": 46.0, "credit": 19.0}
CORR = np.array([[1.00, 0.25, 0.25],
                 [0.25, 1.00, 0.25],
                 [0.25, 0.25, 1.00]])
OP_RISK = 14.0


def simulate_losses(seed=886):
    """Aggregate one-year loss per simulation, in M USD."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(LAMBDA, N_SIMS)
    sev = rng.lognormal(SEV_MU, SEV_SIGMA, int(counts.sum()))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return np.add.reduceat(sev, starts) / 1e6


def risk_measures(x, alpha):
    var = float(np.quantile(x, alpha))
    tvar = float(np.mean(x[x >= var]))
    return var, tvar


def main():
    set_pub_style(base_font=8)
    loss = simulate_losses()
    mean_loss = float(np.mean(loss))
    var, tvar = risk_measures(loss, ALPHA)
    scr_uw = var - mean_loss
    own_funds = COVERAGE * scr_uw

    caps = np.array([MODULE_CAP["market"], MODULE_CAP["credit"], scr_uw])
    total_simple = float(caps.sum())
    bscr = float(np.sqrt(caps @ CORR @ caps))
    diversification = total_simple - bscr
    scr = bscr + OP_RISK

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_m = axes_inches(fig, AUX_T_RECT)
    ax_a = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_m, ax_a):
        a.set_axisbelow(True)
        style_axes(a)
    ax.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)
    ax_m.grid(axis="y", color=GRID, lw=0.55, alpha=0.75, zorder=0)
    ax_a.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)

    # ---- main: the loss distribution and the two risk measures -------------
    lo = float(np.quantile(loss, 0.0005))
    hi = float(np.quantile(loss, 0.99965))
    edges = np.linspace(lo, hi, 190)
    dens, _ = np.histogram(loss, bins=edges, density=True)
    mids = 0.5 * (edges[:-1] + edges[1:])
    kern = np.ones(5) / 5.0
    dens = np.convolve(dens, kern, mode="same")

    ax.fill_between(mids, dens, color=FILL, lw=0.0, zorder=2)
    ax.plot(mids, dens, color=LOSS, lw=1.6, zorder=4)
    tail = mids >= var
    ax.fill_between(mids[tail], dens[tail], color=TAIL, lw=0.0, zorder=3)

    top = float(dens.max())
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, top * 1.44)

    marks = ((mean_loss, "E[L]", DARK, (0, (2.6, 2.0)), 1.235),
             (var, "VaR", LOSS, "-", 1.235),
             (tvar, "TVaR", DEEP, (0, (4.0, 1.8)), 1.235),
             (mean_loss + own_funds, "own funds", CAP, "-", 1.315))
    for xv, tag, colour, dash, ytop in marks:
        ax.plot([xv, xv], [0, top * ytop], color=colour, lw=1.5, ls=dash,
                zorder=6)
        ax.text(xv, top * (ytop + 0.020), tag, ha="center", va="bottom",
                fontsize=6.8, color=colour, zorder=7)

    # measurement stack above the density, so no rule crosses the curve
    ax.annotate("", xy=(var, top * 1.03), xytext=(mean_loss, top * 1.03),
                arrowprops=dict(arrowstyle="<|-|>", lw=1.0, color=LOSS,
                                shrinkA=0, shrinkB=0), zorder=7)
    ax.text(0.5 * (mean_loss + var), top * 1.055,
            f"capital requirement = VaR - E[L] = {scr_uw:,.0f} M USD",
            ha="center", va="bottom", fontsize=7.0, color=LOSS,
            fontweight="bold", zorder=7)
    ax.annotate("", xy=(mean_loss + own_funds, top * 1.135),
                xytext=(mean_loss, top * 1.135),
                arrowprops=dict(arrowstyle="<|-|>", lw=1.0, color=CAP,
                                shrinkA=0, shrinkB=0), zorder=7)
    ax.text(0.5 * (2 * mean_loss + own_funds), top * 1.160,
            f"own funds cover it {own_funds / scr_uw:.2f}x",
            ha="center", va="bottom", fontsize=7.0, color=CAP, zorder=7)

    ax.set_xlabel("aggregate one-year loss (M USD)", fontsize=8.5)
    ax.set_ylabel("probability density (per M USD)", fontsize=8.5)
    ax.set_title("The 0.5% tail sets the capital requirement", fontsize=9.0,
                 fontweight="bold", pad=5.0)
    rows = [f"{'E[L]':<11}{mean_loss:>7,.0f} M USD",
            f"{'VaR 99.5%':<11}{var:>7,.0f} M USD",
            f"{'TVaR 99.5%':<11}{tvar:>7,.0f} M USD",
            f"{'own funds':<11}{mean_loss + own_funds:>7,.0f} M USD",
            f"shaded = the {100 * (1 - ALPHA):.1f}% tail,",
            f"TVaR is its mean, {tvar - var:,.0f} M USD",
            f"above the VaR"]
    ax.text(0.018, 0.985, "\n".join(rows), transform=ax.transAxes,
            ha="left", va="top", fontsize=6.6, color=SUB,
            family="DejaVu Sans Mono", linespacing=1.55,
            bbox=dict(boxstyle="round,pad=0.42", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.94))

    # ---- aux (b): modules, diversification, SCR ----------------------------
    steps = [("market", MODULE_CAP["market"], CAP, "total"),
             ("credit", MODULE_CAP["credit"], CAP, "total"),
             ("u/w", scr_uw, LOSS, "total"),
             ("sum", total_simple, DARK, "total"),
             ("diver-\nsification", -diversification, DIV, "delta"),
             ("BSCR", bscr, DARK, "total"),
             ("op risk", OP_RISK, MUTED, "delta"),
             ("SCR", scr, DARK, "total")]
    run = 0.0
    for k, (name, val, colour, kind) in enumerate(steps):
        if kind == "total":
            ax_m.bar(k, val, width=0.66, color=colour, edgecolor="white",
                     linewidth=0.6, zorder=3)
            run, top_k = val, val
        else:
            base = run + val if val < 0 else run
            ax_m.bar(k, abs(val), bottom=base, width=0.66, color=colour,
                     edgecolor="white", linewidth=0.6, zorder=3)
            run = run + val
            top_k = max(run, run - val)
        ax_m.text(k, top_k + 0.02 * scr, f"{val:+,.0f}" if kind == "delta"
                  else f"{val:,.0f}", ha="center", va="bottom", fontsize=6.0,
                  color=DIV if kind == "delta" and val < 0 else SUB)
    ax_m.set_xticks(np.arange(len(steps)))
    ax_m.set_xticklabels([s[0] for s in steps], fontsize=6.0)
    ax_m.set_ylim(0, total_simple * 1.24)
    ax_m.tick_params(labelsize=6.8)
    ax_m.set_ylabel("capital (M USD)", fontsize=7.8)
    ax_m.set_xlabel("risk module, then aggregation", fontsize=7.8,
                    labelpad=1.5)
    ax_m.set_title("Correlation returns a diversification benefit",
                   fontsize=8.5, fontweight="bold", pad=4.0)
    ax_m.text(0.03, 0.955,
              f"benefit = {diversification:,.0f} M USD, "
              f"{100 * diversification / total_simple:.1f}% of the\n"
              f"simple sum (average correlation "
              f"{CORR[np.triu_indices(3, 1)].mean():.2f})",
              transform=ax_m.transAxes, ha="left", va="top", fontsize=6.4,
              color=SUB, linespacing=1.45)

    # ---- aux (c): capital versus confidence level --------------------------
    ps = np.linspace(0.900, 0.9985, 60)
    var_p = np.quantile(loss, ps)
    tvar_p = np.array([float(np.mean(loss[loss >= v])) for v in var_p])
    ax_a.plot(100.0 * ps, var_p - mean_loss, color=LOSS, lw=1.7, zorder=5,
              label="VaR based")
    ax_a.plot(100.0 * ps, tvar_p - mean_loss, color=DEEP, lw=1.3,
              ls=(0, (4.0, 1.8)), zorder=5, label="TVaR based")
    ax_a.plot([100 * ALPHA], [scr_uw], marker="o", ms=4.2, mfc="white",
              mec=LOSS, mew=1.3, zorder=7)
    ax_a.axvline(100 * ALPHA, color=FAINT, lw=0.8, ls=(0, (2.4, 2.0)),
                 zorder=2)
    ax_a.set_xlim(90.0, 99.9)
    ax_a.set_ylim(0.88 * float((var_p - mean_loss).min()),
                  1.10 * float((tvar_p - mean_loss).max()))
    ax_a.tick_params(labelsize=6.8)
    ax_a.set_xlabel("confidence level (%)", fontsize=7.8, labelpad=1.5)
    ax_a.set_ylabel("capital requirement (M USD)", fontsize=7.8)
    ax_a.set_title("Cost of the last half percent", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_a.legend(loc="upper left", frameon=False, fontsize=6.4,
                handlelength=1.8, labelspacing=0.35, borderpad=0.25)
    ax_a.text(0.975, 0.06,
              f"moving from 99.0% to {100 * ALPHA:.1f}% adds\n"
              f"{scr_uw - (float(np.quantile(loss, 0.99)) - mean_loss):,.0f}"
              f" M USD of capital",
              transform=ax_a.transAxes, ha="right", va="bottom",
              fontsize=6.4, color=SUB, linespacing=1.45)

    panel_labels_inches(fig, [ax, ax_m, ax_a], dx=-0.62, dy=0.10)
    savefig(fig, "886_act_solvency_capital.png")


if __name__ == "__main__":
    main()

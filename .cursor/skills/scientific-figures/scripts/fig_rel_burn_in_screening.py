"""Burn-in screening as a trade-off between two overlapping populations.

Main panel: the lot-level failure rate on log-log axes, falling by two and a
half decades over the first few hundred hours because the latent-defect
subpopulation is being consumed, with both subpopulations drawn underneath on
a shared time axis - latent defects dying early, good units dying late, and
the shaded region where the two densities cross.  That crossing is the whole
argument: a screen short of it removes defects almost for free, a screen past
it starts spending the life of good product.

The right-hand panels close the loop.  The screen duration is not asserted:
it is the minimum of the total-cost curve in panel (b), where burn-in cost
rises linearly and the cost of escapes falls with the residual defect level.
Panel (c) replays the failure rate for a lot that was screened at that
duration against one that was not.

Densities, hazards, escape levels and the optimum all come out of the same
two lognormal populations.  Values are illustrative only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from pubstyle import set_pub_style, savefig, style_axes, lighten, add_footer
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRID = "#d5d5d5"

BLUE = "#3b5b92"
RED = "#b23a48"
GOLD = "#c9a227"
GREEN = "#2e7d5b"

FIG_W, FIG_H = 10.4, 6.4
MAIN_RECT = (0.86, 0.80, 5.18, 4.58)
AUX_T_RECT = (7.16, 3.50, 2.88, 1.88)
AUX_B_RECT = (7.16, 0.80, 2.88, 1.88)

P_DEF = 0.010                      # latent-defect fraction of the lot
MED_DEF, SIG_DEF = 30.0, 0.90      # hours, decades
MED_GOOD, SIG_GOOD = 6.0e4, 0.62
LAM_RANDOM = 2.0e-8                # per hour, intrinsic random failures
C_FIXED, C_RATE = 0.03, 0.0025     # $/unit and $/unit/hour of burn-in
C_ESCAPE = 400.0                   # $ per escaped defect


def lognorm_pair():
    """The two failure-time populations as frozen lognormals."""
    d = stats.lognorm(s=SIG_DEF * np.log(10.0), scale=MED_DEF)
    g = stats.lognorm(s=SIG_GOOD * np.log(10.0), scale=MED_GOOD)
    return d, g


def hazard(t, d, g, t0=0.0):
    """Lot hazard in FIT at field age ``t`` after a burn-in of ``t0`` hours."""
    a = t + t0
    surv = P_DEF * d.sf(a) + (1.0 - P_DEF) * g.sf(a)
    dens = P_DEF * d.pdf(a) + (1.0 - P_DEF) * g.pdf(a)
    return 1e9 * (dens / np.maximum(surv, 1e-12) + LAM_RANDOM)


def hazard_after(t, d, g, t0):
    """Field hazard in FIT of a lot screened for ``t0`` hours.

    The defect subpopulation is depleted and conditioned on having survived
    the screen; the good units are treated as fresh, since the screen spends
    a negligible fraction of their life.
    """
    pd = P_DEF * d.sf(t0)
    pg = (1.0 - P_DEF) * g.sf(t0)
    pd, pg = pd / (pd + pg), pg / (pd + pg)
    dens = pd * d.pdf(t + t0) / d.sf(t0) + pg * g.pdf(t)
    surv = pd * d.sf(t + t0) / d.sf(t0) + pg * g.sf(t)
    return 1e9 * (dens / np.maximum(surv, 1e-12) + LAM_RANDOM)


def escapes_ppm(t0, d, g):
    """Latent defects still in the lot after a burn-in of ``t0`` hours, ppm."""
    left = P_DEF * d.sf(t0)
    return 1e6 * left / (left + (1.0 - P_DEF) * g.sf(t0))


def check_layout(fig, foot, axes):
    """Warn if the footnote box touches any axis label or title."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for a in axes:
        for art in (a.xaxis.label, a.yaxis.label, a.title):
            if art.get_text() and fb.overlaps(art.get_window_extent()):
                print("WARNING: footnote overlaps:", art.get_text()[:40])


def main():
    set_pub_style(base_font=8)
    d, g = lognorm_pair()

    # ---- the screen duration is the minimum of the cost curve --------------
    t_grid = np.logspace(0.6, 3.2, 400)
    ppm_grid = escapes_ppm(t_grid, d, g)
    burn_cost = C_FIXED + C_RATE * t_grid
    escape_cost = ppm_grid * 1e-6 * C_ESCAPE
    total = burn_cost + escape_cost
    i_opt = int(np.argmin(total))
    t_s = float(t_grid[i_opt])

    removed = 100.0 * d.cdf(t_s)
    good_lost = 1e6 * g.cdf(t_s)
    ppm_in = escapes_ppm(0.0, d, g)
    ppm_out = escapes_ppm(t_s, d, g)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_c = axes_inches(fig, AUX_T_RECT)
    ax_r = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_c, ax_r):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: hazard above, the two populations below ---------------------
    t = np.logspace(0.0, 5.4, 700)
    lam = hazard(t, d, g)
    ax.grid(which="both", color=GRID, lw=0.5, alpha=0.60, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvspan(t[0], t_s, color="#eef1f5", zorder=1)
    ax.plot(t, lam, color=RED, lw=2.0, zorder=6)
    ax.axvline(t_s, color=INK, lw=1.1, ls=(0, (3.4, 2.2)), zorder=5)
    ax.plot([t_s], [hazard(np.array([t_s]), d, g)[0]], marker="o", ms=4.6,
            mfc="white", mec=RED, mew=1.3, zorder=8)

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(3.0e2, 3.0e6)
    ax.set_xlabel("time on power (hours, log scale)", fontsize=8.4)
    ax.set_ylabel("lot failure rate (FIT, log scale)", fontsize=8.4,
                  color=RED)
    ax.tick_params(axis="y", colors=RED)
    ax.set_title("The screen has to stop where the two populations meet",
                 fontsize=9.0, fontweight="bold", pad=5.0)
    ax.text(t_s * 0.86, 6.0e5,
            f"burn-in {t_s:.0f} h\n(cost optimum, panel b)", ha="right",
            va="top", fontsize=7.2, color=INK, linespacing=1.50,
            fontweight="bold")
    ax.text(t[0] * 1.25, 3.9e2, "screened out", ha="left", va="bottom",
            fontsize=6.8, color=SUB)
    ax.annotate(f"{lam[0] / hazard(np.array([t_s]), d, g)[0]:.0f}x lower "
                f"failure rate\nby the end of the screen",
                xy=(t_s, hazard(np.array([t_s]), d, g)[0]),
                xytext=(t_s * 0.80, 1.15e5), fontsize=6.9, color=RED,
                linespacing=1.45, ha="right", va="bottom",
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=RED,
                                shrinkA=2, shrinkB=3))

    # the two populations, on the same time axis
    ax2 = ax.twinx()
    dens_d = d.pdf(t) * t * np.log(10.0)
    dens_g = g.pdf(t) * t * np.log(10.0)
    dens_d, dens_g = dens_d / dens_d.max(), dens_g / dens_g.max()
    ov = np.minimum(dens_d, dens_g)
    ax2.fill_between(t, 0.0, dens_d, color=lighten(GOLD, 0.55),
                     edgecolor=GOLD, linewidth=1.0, zorder=3)
    ax2.fill_between(t, 0.0, dens_g, color=lighten(GREEN, 0.62),
                     edgecolor=GREEN, linewidth=1.0, zorder=3)
    ax2.fill_between(t, 0.0, ov, facecolor="none", edgecolor=INK,
                     hatch="////", linewidth=0.0, zorder=4)
    ax2.set_ylim(0.0, 4.4)
    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.set_ylabel("failure-time density (per decade, each scaled to 1)",
                   fontsize=8.4, color=SUB)
    ax2.tick_params(axis="y", labelsize=7.2, colors=SUB)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(SUB)

    band = t[ov > 0.05 * ov.max()]
    ax2.text(MED_DEF, 1.20,
             f"latent defects\n{100 * P_DEF:.1f} % of the lot",
             ha="center", va="bottom", fontsize=7.0, color=GOLD,
             linespacing=1.45, fontweight="bold")
    ax2.text(MED_GOOD, 1.20, "good population\nthe other 99.0 %",
             ha="center", va="bottom", fontsize=7.0, color=GREEN,
             linespacing=1.45, fontweight="bold")
    ax2.annotate(f"the tails cross between {band.min() / 1000.0:.1f} and "
                 f"{band.max() / 1000.0:.0f} kh:\nscreening into this window "
                 f"spends good life, not defects",
                 xy=(np.sqrt(band.min() * band.max()), ov.max() * 1.05),
                 xytext=(np.sqrt(band.min() * band.max()), 1.85),
                 fontsize=6.9, color=INK, linespacing=1.45, ha="center",
                 va="bottom",
                 arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK,
                                 shrinkA=2, shrinkB=2))
    ax.text(0.975, 0.965,
            f"at {t_s:.0f} h the screen has removed {removed:.0f} % of the "
            f"latent defects and\nconsumed {good_lost:.0f} ppm of the good "
            f"population; the outgoing\ndefect level falls from "
            f"{ppm_in:,.0f} ppm to {ppm_out:,.0f} ppm",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.0,
            color=INK, linespacing=1.55, zorder=10,
            bbox=dict(boxstyle="round,pad=0.36", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.94))

    # ---- aux (b): the cost trade-off ---------------------------------------
    ax_c.grid(which="both", color=GRID, lw=0.55, alpha=0.75, zorder=0)
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.plot(t_grid, ppm_grid, color=BLUE, lw=1.7, zorder=5)
    ax_c.set_xlim(t_grid[0], t_grid[-1])
    ax_c.set_ylim(ppm_grid.min() * 0.55, ppm_grid.max() * 2.6)
    ax_c.tick_params(labelsize=6.8)
    ax_c.set_xlabel("burn-in duration (hours, log scale)", fontsize=7.5,
                    labelpad=1.5)
    ax_c.set_ylabel("outgoing defect level (ppm)", fontsize=7.5, color=BLUE)
    ax_c.tick_params(axis="y", colors=BLUE)
    ax_c.set_title("A real minimum, not a slogan", fontsize=8.5,
                   fontweight="bold", pad=4.0)

    ax_c2 = ax_c.twinx()
    ax_c2.plot(t_grid, total, color=RED, lw=1.8, zorder=6)
    ax_c2.axvline(t_s, color=RED, lw=0.8, ls=(0, (2.6, 2.0)), zorder=3)
    ax_c2.plot([t_s], [total[i_opt]], marker="D", ms=5.0, mfc=RED,
               mec="white", mew=0.9, zorder=8)
    ax_c2.set_ylim(0.0, total.max() * 1.20)
    ax_c2.tick_params(labelsize=6.8)
    ax_c2.set_ylabel("total cost ($ per unit)", fontsize=7.5, color=RED)
    ax_c2.tick_params(axis="y", colors=RED)
    ax_c2.spines["top"].set_visible(False)
    ax_c2.annotate(f"optimum {t_s:.0f} h at ${total[i_opt]:.2f}/unit",
                   xy=(t_s, total[i_opt]), xytext=(0.06, 0.10),
                   textcoords="axes fraction", fontsize=6.4, color=RED,
                   ha="left", va="bottom",
                   arrowprops=dict(arrowstyle="-|>", lw=0.8, color=RED,
                                   shrinkA=2, shrinkB=3))
    ax_c2.text(0.975, 0.955,
               f"burn-in costs ${C_FIXED:.2f} + ${1000 * C_RATE:.2f} per "
               f"1000 h;\neach escape costs ${C_ESCAPE:.0f} in the field",
               transform=ax_c2.transAxes, ha="right", va="top", fontsize=6.2,
               color=SUB, linespacing=1.45)

    # ---- aux (c): before and after the screen ------------------------------
    tf = np.logspace(0.0, 5.4, 400)
    lam_raw = hazard(tf, d, g)
    lam_scr = hazard_after(tf, d, g, t_s)
    t_ref = 100.0
    ax_r.grid(which="both", color=GRID, lw=0.55, alpha=0.75, zorder=0)
    ax_r.set_xscale("log")
    ax_r.set_yscale("log")
    ax_r.fill_between(tf, lam_scr, lam_raw, color="#f0e6e8", zorder=2)
    ax_r.plot(tf, lam_raw, color=RED, lw=1.6, zorder=5)
    ax_r.plot(tf, lam_scr, color=GREEN, lw=1.6, zorder=6)
    iref = int(np.argmin(np.abs(tf - t_ref)))
    ax_r.plot([t_ref, t_ref], [lam_scr[iref], lam_raw[iref]], color=INK,
              lw=0.8, ls=(0, (2.2, 1.8)), zorder=7)
    ax_r.annotate(f"{lam_raw[iref] / lam_scr[iref]:.0f}x at {t_ref:.0f} h "
                  f"of field age",
                  xy=(t_ref, np.sqrt(lam_scr[iref] * lam_raw[iref])),
                  xytext=(t_ref * 1.6, lam_raw[iref] * 2.4), fontsize=6.4,
                  color=INK, ha="left", va="bottom",
                  arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK,
                                  shrinkA=2, shrinkB=2))
    ax_r.set_xlim(tf[0], tf[-1])
    ax_r.set_ylim(3.0e2, 3.0e6)
    ax_r.tick_params(labelsize=6.8)
    ax_r.set_xlabel("field age (hours, log scale)", fontsize=7.5, labelpad=1.5)
    ax_r.set_ylabel("failure rate (FIT)", fontsize=7.5)
    ax_r.set_title("What the customer sees", fontsize=8.5, fontweight="bold",
                   pad=4.0)
    ax_r.legend(handles=[
        Line2D([], [], color=RED, lw=1.6, label="as built"),
        Line2D([], [], color=GREEN, lw=1.6,
               label=f"after {t_s:.0f} h burn-in")],
        loc="lower left", frameon=False, fontsize=6.4, handlelength=1.6,
        labelspacing=0.30, borderaxespad=0.25)

    panel_labels_inches(fig, [ax, ax_c, ax_r], dx=-0.62, dy=0.10)
    foot = add_footer(fig, "Synthetic data, figure 911")
    check_layout(fig, foot, [ax, ax_c, ax_r])
    savefig(fig, "911_rel_burn_in_screening.png")


if __name__ == "__main__":
    main()

"""Loss reserving: the run-off triangle and the chain-ladder completion.

Main panel: cumulative paid losses by accident year and development year.  The
lower-left triangle is what has actually been paid; the upper-right block is
what the chain ladder projects, drawn in a lighter, hatched treatment so the
reader can never mistake a projection for an observation.  The two narrow
right-hand panels carry the machinery and its consequence: the age-to-age
development factors that fill the blank corner, and the ultimate loss per
accident year split into paid-to-date and IBNR with an uncertainty range.

Triangle data comes from ``gen_triangle``; the chain-ladder factors, the
reserve and every annotated number are estimated from it at run time.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from pubstyle import set_pub_style, savefig, style_axes, get_cmap
from figkit import panel_labels_inches, safe_colorbar

BLUE = "#3b5b92"
GOLD = "#c9a227"
RED = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRID = "#ededed"

N = 10
FIRST_AY = 2015
# share of ultimate paid in each development year
PATTERN = np.array([0.315, 0.258, 0.168, 0.101, 0.062, 0.040, 0.026,
                    0.017, 0.009, 0.004])


def gen_triangle(seed=638):
    """(cumulative triangle with NaN in the future, true ultimate) in USD m."""
    rng = np.random.default_rng(seed)
    base = 42.0 * 1.045 ** np.arange(N)          # exposure growth, USD million
    ult = base * rng.normal(1.0, 0.075, N)
    cum = np.full((N, N), np.nan)
    for i in range(N):
        inc = ult[i] * PATTERN * rng.normal(1.0, 0.085, N)
        c = np.cumsum(inc)
        for j in range(N):
            if i + j <= N - 1:
                cum[i, j] = c[j]
    return cum, ult


def chain_ladder(cum):
    """Volume-weighted age-to-age factors and the completed triangle."""
    full = cum.copy()
    f = np.ones(N - 1)
    for j in range(N - 1):
        rows = np.arange(0, N - 1 - j)
        f[j] = full[rows, j + 1].sum() / full[rows, j].sum()
    for i in range(1, N):
        for j in range(N - i, N):
            full[i, j] = full[i, j - 1] * f[j - 1]
    return f, full


def main():
    set_pub_style(base_font=8)
    cum, _ = gen_triangle()
    f, full = chain_ladder(cum)

    years = np.arange(FIRST_AY, FIRST_AY + N)
    observed = ~np.isnan(cum)
    paid = np.array([cum[i, N - 1 - i] for i in range(N)])
    ultimate = full[:, -1]
    ibnr = ultimate - paid
    total_ibnr = ibnr.sum()
    # Mack-flavoured proxy: uncertainty grows with the unpaid share
    cv = 0.035 + 0.42 * (ibnr / ultimate)
    sd = cv * ultimate

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          left=0.070, right=0.972, top=0.878, bottom=0.132,
                          hspace=0.50, wspace=0.300)
    ax = fig.add_subplot(gs[:, 0])
    ax_f = fig.add_subplot(gs[0, 1])
    ax_u = fig.add_subplot(gs[1, 1])

    for a in (ax_f, ax_u):
        a.set_axisbelow(True)
        a.grid(axis="y", color=GRID, lw=0.7, zorder=0)
        style_axes(a)

    # ---- main: the run-off triangle ----------------------------------------
    cmap = get_cmap("cb_sequential_blue")
    im = ax.imshow(full, cmap=cmap, aspect="auto", vmin=float(full.min()),
                   vmax=float(full.max()), zorder=2)
    norm = im.norm

    for i in range(N):
        for j in range(N):
            v = full[i, j]
            if observed[i, j]:
                col = "white" if norm(v) > 0.58 else INK
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=6.0, color=col, zorder=6)
            else:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0,
                                       facecolor="white", alpha=0.72,
                                       edgecolor="none", zorder=3))
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0,
                                       facecolor="none", hatch="////",
                                       edgecolor="#9cc3dd", linewidth=0.0,
                                       zorder=4))
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=6.0, color=SUB, style="italic", zorder=6)

    # the step boundary between what happened and what is projected
    bx, by = [], []
    for i in range(N):
        j = N - 1 - i
        bx += [j + 0.5, j + 0.5, j - 0.5]
        by += [i - 0.5, i + 0.5, i + 0.5]
    ax.plot(bx, by, color=RED, lw=1.6, zorder=7, solid_joinstyle="miter")

    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels([str(j) for j in range(N)], fontsize=7.2)
    ax.set_yticklabels([str(y) for y in years], fontsize=7.2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    ax.set_xlabel("Development year  $j$ (years since accident year)",
                  fontsize=9.0)
    ax.set_ylabel("Accident year  $i$", fontsize=9.0)
    ax.set_title("Cumulative paid losses, chain-ladder completion",
                 fontsize=9.0, fontweight="bold", pad=5.0)
    safe_colorbar(fig, im, ax, label="cumulative paid (million USD)",
                  width=0.12, pad=0.09)

    ax.text(1.35, 8.2,
            "hatched cells: chain-ladder projection\n"
            f"IBNR = {total_ibnr:,.1f} million USD "
            f"({total_ibnr / ultimate.sum():.1%} of ultimate)",
            ha="left", va="center", fontsize=7.2, color=RED,
            linespacing=1.5, zorder=8,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor="#e0c3c8", linewidth=0.7, alpha=0.95))
    ax.text(6.7, 0.6, "observed", ha="center", va="center", fontsize=7.4,
            color="white", fontweight="bold", zorder=8)
    ax.text(7.9, 2.35, "latest diagonal", ha="right", va="center",
            fontsize=6.8, color=RED, rotation=-45, zorder=8)

    # ---- aux (b): age-to-age factors ---------------------------------------
    jj = np.arange(N - 1)
    ax_f.plot(jj, f - 1.0, color=BLUE, lw=1.7, marker="o", ms=4.0,
              mfc="white", mec=BLUE, mew=1.3, zorder=5)
    ax_f.set_yscale("log")
    ax_f.set_xlim(-0.5, N - 1.5)
    ax_f.set_xticks(jj)
    ax_f.set_xticklabels([f"{j}$\\rightarrow${j + 1}" for j in jj],
                         fontsize=6.2)
    ax_f.tick_params(labelsize=6.8)
    ax_f.set_xlabel("development interval (years)", fontsize=8.0, labelpad=1.5)
    ax_f.set_ylabel("$f_j - 1$ (log scale, $-$)", fontsize=8.0)
    ax_f.set_title("Age-to-age factors decay geometrically", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_f.annotate(f"$f_0$ = {f[0]:.3f}", xy=(0, f[0] - 1.0),
                  xytext=(8, -7), textcoords="offset points", ha="left",
                  va="top", fontsize=6.6, color=BLUE)
    ax_f.annotate(f"$f_{{{N - 2}}}$ = {f[-1]:.4f}",
                  xy=(N - 2, f[-1] - 1.0), xytext=(-7, 5),
                  textcoords="offset points", ha="right", va="bottom",
                  fontsize=6.6, color=BLUE)
    ax_f.text(0.975, 0.955,
              f"cumulative factor to ultimate:\n{np.prod(f):.3f}$\\times$ the "
              "first development year",
              transform=ax_f.transAxes, ha="right", va="top",
              fontsize=6.5, color=SUB, linespacing=1.45)

    # ---- aux (c): ultimate = paid + IBNR -----------------------------------
    ax_u.bar(years, paid, width=0.72, color=BLUE, edgecolor="white",
             linewidth=0.5, zorder=4)
    ax_u.bar(years, ibnr, bottom=paid, width=0.72, color=GOLD,
             edgecolor="white", linewidth=0.5, hatch="////", zorder=4)
    ax_u.errorbar(years, ultimate, yerr=sd, fmt="none", ecolor=RED,
                  elinewidth=1.0, capsize=2.4, capthick=1.0, zorder=6)
    ax_u.set_xlim(FIRST_AY - 0.7, FIRST_AY + N - 0.3)
    ax_u.set_ylim(0, float((ultimate + sd).max()) * 1.14)
    ax_u.set_xticks(years[::2])
    ax_u.set_xticklabels([str(y) for y in years[::2]], fontsize=6.8)
    ax_u.tick_params(labelsize=6.8)
    ax_u.set_xlabel("accident year", fontsize=8.0, labelpad=1.5)
    ax_u.set_ylabel("ultimate loss (million USD)", fontsize=8.0)
    ax_u.set_title("Recent years carry all the uncertainty", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_u.legend(handles=[
        Patch(facecolor=BLUE, label="paid to date"),
        Patch(facecolor=GOLD, hatch="////", edgecolor="white",
              label="IBNR reserve"),
        Line2D([], [], color=RED, lw=1.0, label="$\\pm$1 s.d. of ultimate")],
        loc="upper left", frameon=False, fontsize=6.4, handlelength=1.5,
        labelspacing=0.3, borderpad=0.25)
    ax_u.text(0.030, 0.760,
              f"{FIRST_AY + N - 1} is {ibnr[-1] / ultimate[-1]:.0%} unpaid",
              transform=ax_u.transAxes, ha="left", va="top",
              fontsize=6.5, color=SUB)

    panel_labels_inches(fig, [ax, ax_f, ax_u], dx=-0.44, dy=0.06)

    recent = ibnr[-3:].sum() / total_ibnr
    savefig(fig, "638_ins_reserve_triangle.png")


if __name__ == "__main__":
    main()

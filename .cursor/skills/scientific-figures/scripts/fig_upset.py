"""UpSet plot: set-intersection sizes with a dot-matrix legend.

Alternative to a Venn diagram for more than three sets.  Membership is drawn
for 600 grid cells first and the exclusive intersections are counted from that
table, so the intersection bars, the set totals and the dot matrix are
guaranteed to be mutually consistent.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pubstyle import set_pub_style, savefig, style_axes

SETS = ["Water yield", "Carbon", "Habitat", "Soil retention"]
BAR = "#2e7d5b"
BAR_HI = "#14523a"
TOT = "#9a9a9a"
N_CELLS = 600
TOP_K = 12


def simulate(rng):
    """Hotspot membership with a shared 'overall quality' latent."""
    latent = rng.normal(0, 1, N_CELLS)
    probs = [0.34, 0.30, 0.27, 0.24]
    loads = [0.9, 0.7, 1.0, 0.6]
    member = np.zeros((N_CELLS, len(SETS)), dtype=bool)
    for k, (p, w) in enumerate(zip(probs, loads)):
        z = w * latent + rng.normal(0, 1, N_CELLS)
        member[:, k] = z > np.quantile(z, 1 - p)
    return member


def main():
    set_pub_style()
    rng = np.random.default_rng(161)
    member = simulate(rng)
    n = len(SETS)

    codes = member @ (1 << np.arange(n))
    counts = np.bincount(codes, minlength=1 << n)
    combos = [(tuple(k for k in range(n) if c >> k & 1), counts[c])
              for c in range(1, 1 << n) if counts[c] > 0]
    combos.sort(key=lambda t: -t[1])
    shown = combos[:TOP_K]
    sizes = np.array([s for _, s in shown])
    set_totals = member.sum(axis=0)

    fig = plt.figure(figsize=(9.6, 6.0))
    gs = GridSpec(2, 2, width_ratios=[1, 3.6], height_ratios=[3, 2],
                  wspace=0.26, hspace=0.08, left=0.055, right=0.985,
                  top=0.90, bottom=0.10)

    ax_bar = fig.add_subplot(gs[0, 1])
    xpos = np.arange(len(shown))
    cols = [BAR_HI if i == 0 else BAR for i in range(len(shown))]
    ax_bar.bar(xpos, sizes, color=cols, width=0.62, edgecolor="white",
               zorder=3)
    for x, s in zip(xpos, sizes):
        ax_bar.text(x, s + sizes.max() * 0.02, str(s), ha="center",
                    fontsize=6.5, color="#333333")
    ax_bar.set_ylabel("Exclusive intersection size\n(number of grid cells)",
                      fontsize=8, linespacing=1.4)
    ax_bar.set_xticks([])
    ax_bar.set_xlim(-0.6, len(shown) - 0.4)
    ax_bar.set_ylim(0, sizes.max() * 1.16)
    ax_bar.grid(axis="y", color="#eeeeee", lw=0.6, zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.tick_params(labelsize=7.5, length=2.5)
    style_axes(ax_bar)

    ax_tot = fig.add_subplot(gs[1, 0])
    ax_tot.barh(range(n), set_totals, color=TOT, height=0.45, zorder=3)
    for k, v in enumerate(set_totals):
        ax_tot.text(v + set_totals.max() * 0.04, k, str(v), va="center",
                    ha="right", fontsize=6.5, color="#555555")
    ax_tot.set_yticks(range(n))
    ax_tot.set_yticklabels([])
    ax_tot.invert_xaxis()
    ax_tot.set_xlabel("Cells in set", fontsize=8)
    ax_tot.set_ylim(-0.6, n - 0.4)
    ax_tot.set_xlim(set_totals.max() * 1.25, 0)
    ax_tot.grid(axis="x", color="#eeeeee", lw=0.6, zorder=0)
    ax_tot.set_axisbelow(True)
    ax_tot.tick_params(labelsize=7.5, length=2.5)
    style_axes(ax_tot)

    ax_dot = fig.add_subplot(gs[1, 1], sharex=ax_bar)
    for j in range(n):
        ax_dot.axhspan(j - 0.34, j + 0.34, color="#f4f4f4", zorder=0)
    for x, (combo, _s) in enumerate(shown):
        members = list(combo)
        ax_dot.scatter([x] * n, range(n), s=42, color="#dcdcdc", zorder=1)
        ax_dot.scatter([x] * len(members), members, s=52, color=BAR, zorder=3)
        if len(members) > 1:
            ax_dot.plot([x, x], [min(members), max(members)], color=BAR,
                        lw=1.6, zorder=2)
    ax_dot.set_yticks(range(n))
    ax_dot.set_yticklabels(SETS, fontsize=8)
    ax_dot.set_ylim(-0.6, n - 0.4)
    ax_dot.set_xticks([])
    ax_dot.tick_params(length=0)
    for s in ax_dot.spines.values():
        s.set_visible(False)

    ax_note = fig.add_subplot(gs[0, 0])
    ax_note.axis("off")
    covered = sizes.sum()
    ax_note.text(0.0, 0.02,
                 f"{N_CELLS} grid cells\n{len(combos)} non-empty "
                 f"intersections\ntop {len(shown)} shown "
                 f"({covered / member.any(axis=1).sum() * 100:.0f}% of "
                 "cells in\nat least one set)",
                 transform=ax_note.transAxes, ha="left", va="bottom",
                 fontsize=7.5, color="#555555", linespacing=1.5)

    savefig(fig, "26_upset.png")


if __name__ == "__main__":
    main()

"""One-year credit rating migration of a synthetic corporate bond portfolio.

(a) the migration matrix on a logarithmic colour scale - linear shading would
    leave everything except the diagonal white, because the off-diagonal mass
    spans four orders of magnitude; cells above the boxed diagonal are
    downgrades, cells below it are upgrades.  (b) the one-year default
    probability per rating on a log axis; (c) a flow thumbnail that weights the
    same matrix by portfolio exposure, so the reader sees where the money, not
    just the probability, actually goes.

The bars and the ribbons are derived from the matrix drawn in (a).
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, style_axes, get_cmap, add_colorbar
from figkit import panel_labels_inches, halo_text

BLUE = "#3b5b92"
RED = "#b23a48"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
RUST = "#c0603f"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRID = "#ececec"

RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
COLS = RATINGS + ["D"]

# Rows are read as "start of year"; columns as "end of year", in per cent.
RAW = np.array([
    [90.80, 8.30, 0.70, 0.10, 0.05, 0.02, 0.01, 0.01],
    [0.70, 90.60, 7.70, 0.60, 0.06, 0.10, 0.02, 0.02],
    [0.09, 2.20, 91.20, 5.20, 0.70, 0.20, 0.02, 0.06],
    [0.02, 0.33, 5.90, 87.50, 5.00, 1.20, 0.12, 0.18],
    [0.03, 0.14, 0.67, 7.70, 81.10, 8.40, 1.00, 1.06],
    [0.01, 0.11, 0.24, 0.43, 6.50, 83.50, 4.10, 5.20],
    [0.02, 0.03, 0.22, 1.30, 2.40, 11.20, 64.90, 19.93],
])
# USD mn of exposure at the start of the year, by rating.
EXPOSURE = np.array([45.0, 130.0, 275.0, 390.0, 250.0, 95.0, 32.0])

OUTCOMES = ["Upgrade", "Unchanged", "Downgrade", "Default"]
OUT_COLOR = {"Upgrade": GREEN, "Unchanged": BLUE, "Downgrade": GOLD,
             "Default": RED}


def migration_matrix():
    return 100.0 * RAW / RAW.sum(axis=1, keepdims=True)


def panel_matrix(fig, ax, m):
    cmap = get_cmap("cb_sequential_blue")
    norm = mpl.colors.LogNorm(vmin=0.01, vmax=100.0)
    im = ax.imshow(m, cmap=cmap, norm=norm, aspect="auto",
                   interpolation="nearest")
    n_r, n_c = m.shape
    ax.set_xticks(np.arange(n_c))
    ax.set_yticks(np.arange(n_r))
    ax.set_xticklabels(COLS, fontsize=7.2)
    ax.set_yticklabels(RATINGS, fontsize=7.2)
    ax.set_xticks(np.arange(-0.5, n_c, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_r, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.7)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(length=2.0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(n_r):
        for j in range(n_c):
            v = m[i, j]
            txt = "%.1f" % v if v >= 1.0 else "%.2f" % v
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.0,
                    color="white" if v > 12.0 else INK)
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1.0, 1.0, fill=False,
                               edgecolor=INK, lw=1.0, zorder=5))
    ax.add_patch(Rectangle((n_c - 1.5, -0.5), 1.0, n_r, fill=False,
                           edgecolor=RED, lw=1.1, zorder=5))
    ax.text(n_c - 1.0, -0.72, "default\ncolumn", ha="center", va="bottom",
            fontsize=6.4, color=RED, linespacing=1.2)
    halo_text(ax, 5.95, 0.42, "downgrades $\\rightarrow$", color=SUB,
              halo="white", halo_lw=2.6, fontsize=7.0, ha="center",
              va="center", zorder=6)
    halo_text(ax, 1.05, 5.62, "$\\leftarrow$ upgrades", color=SUB,
              halo="white", halo_lw=2.6, fontsize=7.0, ha="center",
              va="center", zorder=6)
    ax.set_xlabel("rating at end of year")
    ax.set_ylabel("rating at start of year")
    cb = add_colorbar(fig, im, ax, label="migration probability (%)",
                      orientation="horizontal", fraction=0.055, pad=0.13,
                      shrink=0.86, aspect=30,
                      ticks=[0.01, 0.1, 1.0, 10.0, 100.0])
    cb.ax.set_xticklabels(["0.01", "0.1", "1", "10", "100"])
    return im


def panel_pd(ax, pd):
    y = np.arange(len(RATINGS))
    cols = [RED if p > 1.0 else BLUE for p in pd]
    ax.barh(y, pd, height=0.62, color=cols, alpha=0.88, lw=0)
    for yi, p in zip(y, pd):
        ax.text(p * 1.22, yi, "%.2f%%" % p if p < 1 else "%.1f%%" % p,
                va="center", ha="left", fontsize=6.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(RATINGS, fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.008, 90.0)
    ax.set_xlabel("one-year default probability (%, log axis)")
    ax.axvline(1.0, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.text(1.12, 6.55, "1% line", fontsize=6.4, color=MUTED, ha="left",
            va="center")
    ax.grid(axis="x", lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.8)
    style_axes(ax)


def outcome_flows(m):
    """Exposure (USD mn) moving from each rating into the four outcomes."""
    flows = np.zeros((len(RATINGS), len(OUTCOMES)))
    for i in range(len(RATINGS)):
        p = m[i] / 100.0
        flows[i, 0] = p[:i].sum()
        flows[i, 1] = p[i]
        flows[i, 2] = p[i + 1:len(RATINGS)].sum()
        flows[i, 3] = p[-1]
    return flows * EXPOSURE[:, None]


def panel_sankey(ax, flows):
    total = flows.sum()
    gap_l, gap_r = 0.020, 0.115
    lh = 1.0 - gap_l * (len(RATINGS) - 1)
    rh = 1.0 - gap_r * (len(OUTCOMES) - 1)
    left_h = flows.sum(axis=1) / total * lh
    right_h = flows.sum(axis=0) / total * rh

    ly, l_top = {}, 1.0
    for i, h in enumerate(left_h):
        ly[i] = l_top
        l_top -= h + gap_l
    ry, r_top = {}, 1.0
    for j, h in enumerate(right_h):
        ry[j] = r_top
        r_top -= h + gap_r

    x = np.linspace(0.0, 1.0, 70)
    s = 0.5 - 0.5 * np.cos(np.pi * x)
    xs = 0.13 + x * 0.62
    l_cur = dict(ly)
    r_cur = dict(ry)
    order = np.argsort(-flows.sum(axis=1))
    for i in order:
        for j in range(len(OUTCOMES)):
            w = flows[i, j] / total * lh
            if w <= 0:
                continue
            wr = flows[i, j] / total * rh
            top = l_cur[i] + (r_cur[j] - l_cur[i]) * s
            bot = (l_cur[i] - w) + ((r_cur[j] - wr) - (l_cur[i] - w)) * s
            ax.fill_between(xs, bot, top, color=OUT_COLOR[OUTCOMES[j]],
                            alpha=0.42, lw=0, zorder=2)
            l_cur[i] -= w
            r_cur[j] -= wr

    for i, h in enumerate(left_h):
        ax.add_patch(Rectangle((0.09, ly[i] - h), 0.04, h, facecolor="#5a5a5a",
                               edgecolor="none", zorder=4))
        ax.text(0.075, ly[i] - h / 2.0, RATINGS[i], ha="right", va="center",
                fontsize=6.4, color=INK)
    for j, h in enumerate(right_h):
        ax.add_patch(Rectangle((0.75, ry[j] - h), 0.04, h,
                               facecolor=OUT_COLOR[OUTCOMES[j]],
                               edgecolor="none", zorder=4))
        share = 100.0 * flows.sum(axis=0)[j] / total
        ax.text(0.80, ry[j] - h / 2.0,
                "%s\n%.0f USD mn (%.1f%%)"
                % (OUTCOMES[j], flows.sum(axis=0)[j], share),
                ha="left", va="center", fontsize=6.3, color=INK,
                linespacing=1.2)
    ax.text(0.60, 1.075, "exposure-weighted one-year flow (USD mn)",
            fontsize=6.8, color=INK, ha="center", va="bottom",
            fontweight="bold")
    ax.set_xlim(0.0, 1.34)
    ax.set_ylim(-0.06, 1.16)
    ax.axis("off")


def main():
    set_pub_style()
    m = migration_matrix()
    pd = m[:, -1]
    flows = outcome_flows(m)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.60],
                          height_ratios=[1.0, 0.92], hspace=0.40,
                          wspace=0.20, left=0.075, right=0.975, top=0.900,
                          bottom=0.115)
    ax_m = fig.add_subplot(gs[:, 0])
    ax_pd = fig.add_subplot(gs[0, 1])
    ax_sk = fig.add_subplot(gs[1, 1])

    panel_matrix(fig, ax_m, m)
    panel_pd(ax_pd, pd)
    panel_sankey(ax_sk, flows)

    stay = float(np.mean(np.diag(m[:, :len(RATINGS)])))
    dflt = 100.0 * flows[:, 3].sum() / flows.sum()

    panel_labels_inches(fig, [ax_m, ax_pd, ax_sk], dx=-0.46, dy=0.06)
    savefig(fig, "574_fin_credit_migration.png")


if __name__ == "__main__":
    main()

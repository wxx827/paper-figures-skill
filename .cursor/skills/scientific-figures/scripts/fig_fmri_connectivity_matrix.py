"""Figure 536 - Resting-state functional connectivity matrix, module sorted.

Main panel: a 90-region Pearson correlation matrix with the rows and columns
re-ordered by network membership, white rules on the module boundaries and a
colour band on the top and left edge carrying the assignment.  The diverging
scale is symmetric about zero, so the neutral tint really does mean r = 0 -
an asymmetric range would quietly turn weak positive coupling into "red".
The subordinate panel contrasts within- against between-network coupling,
both averaged from the same matrix.

The network palette here is the house qualitative order and is reused verbatim
by figures 538 and 539 so that a module keeps its colour across the set.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, get_cmap, COLORS
from figkit import axes_inches, panel_labels_inches, safe_colorbar, shared_norm

TR = 2.0
N_VOL = 320

# (name, number of parcels, colour) - house qualitative order, positions 1-5
# plus one extra family colour for the sixth network.
NETWORKS = [
    ("Visual", 16, "#b23a48"),
    ("Somatomotor", 18, "#3b5b92"),
    ("Dorsal attention", 12, "#2e7d5b"),
    ("Salience", 10, "#c9a227"),
    ("Frontoparietal", 14, "#8e6bbf"),
    ("Default mode", 20, "#c0603f"),
]

# Latent coupling between network time courses; the classic task-positive /
# default-mode anticorrelation is the -0.42 entry.
NET_COV = np.array([
    [1.00, 0.30, 0.22, 0.10, 0.06, -0.10],
    [0.30, 1.00, 0.28, 0.18, 0.08, -0.16],
    [0.22, 0.28, 1.00, 0.34, 0.40, -0.42],
    [0.10, 0.18, 0.34, 1.00, 0.38, -0.24],
    [0.06, 0.08, 0.40, 0.38, 1.00, -0.18],
    [-0.10, -0.16, -0.42, -0.24, -0.18, 1.00],
])


def simulate(rng):
    """Parcel time courses from a network factor model; returns (R, labels)."""
    n_net = len(NETWORKS)
    sizes = np.array([n for _, n, _ in NETWORKS])
    labels = np.repeat(np.arange(n_net), sizes)
    L = np.linalg.cholesky(NET_COV + 1e-9 * np.eye(n_net))
    factors = L @ rng.standard_normal((n_net, N_VOL))
    globalsig = rng.standard_normal(N_VOL)

    ts = np.empty((labels.size, N_VOL))
    for i, m in enumerate(labels):
        w = rng.uniform(0.62, 1.05)
        ts[i] = (w * factors[m] + 0.22 * globalsig +
                 rng.uniform(0.55, 0.95) * rng.standard_normal(N_VOL))
    ts += 0.10 * rng.standard_normal((labels.size, 1)) * np.linspace(
        -1, 1, N_VOL)                                    # residual drift
    return np.corrcoef(ts), labels


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(4)
    R, labels = simulate(rng)
    n = labels.size
    bounds = np.cumsum([m for _, m, _ in NETWORKS])
    colors = [c for _, _, c in NETWORKS]
    names = [s for s, _, _ in NETWORKS]

    off = ~np.eye(n, dtype=bool)
    norm = shared_norm([R[off]], symmetric=True, robust=(1, 99))
    disp = R.copy()
    np.fill_diagonal(disp, np.nan)
    cmap = get_cmap("cb_diverging")
    cmap.set_bad("#e8e8e8")

    FW, FH = 10.4, 6.4
    mx, my, mw = 1.06, 0.72, 4.62                     # main matrix, inches
    strip = 0.13
    fig = plt.figure(figsize=(FW, FH))

    ax = axes_inches(fig, (mx, my, mw, mw))
    im = ax.imshow(disp, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_aspect("equal")
    for b in bounds[:-1]:
        ax.axvline(b - 0.5, color="white", lw=1.1)
        ax.axhline(b - 0.5, color="white", lw=1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    ax.set_xlabel(f"Parcel, sorted by network (n = {n})", fontsize=8.0)
    ax.set_ylabel("Parcel, sorted by network", fontsize=8.0,
                  labelpad=strip * 72 + 8)
    ax.set_title("Resting-state functional connectivity", fontsize=8.8,
                 fontweight="bold", pad=strip * 72 + 11)

    # membership bands: one axes each, so they cannot be clipped by the matrix
    band = mpl.colors.ListedColormap(colors)
    axt = axes_inches(fig, (mx, my + mw + 0.035, mw, strip))
    axt.imshow(labels[None, :], cmap=band, aspect="auto", vmin=-0.5,
               vmax=len(NETWORKS) - 0.5, interpolation="nearest")
    axl = axes_inches(fig, (mx - strip - 0.035, my, strip, mw))
    axl.imshow(labels[:, None], cmap=band, aspect="auto", vmin=-0.5,
               vmax=len(NETWORKS) - 0.5, interpolation="nearest")
    for a in (axt, axl):
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)

    safe_colorbar(fig, im, ax, label="Pearson $r$ (-)", side="right",
                  width=0.11, pad=0.10, shrink=1.0,
                  ticks=np.arange(-0.6, 0.61, 0.3))

    # ---- subordinate panel: within vs between network coupling ------------
    within, between, wsd = [], [], []
    for k in range(len(NETWORKS)):
        m = labels == k
        blk = R[np.ix_(m, m)][~np.eye(m.sum(), dtype=bool)]
        within.append(blk.mean())
        wsd.append(blk.std(ddof=1))
        between.append(R[np.ix_(m, ~m)].mean())

    aw, ah = 2.52, mw * 0.62
    axb = axes_inches(fig, (7.46, my + (mw - ah) / 2.0, aw, ah))
    ypos = np.arange(len(NETWORKS))[::-1]
    axb.barh(ypos + 0.19, within, height=0.34, color=colors, lw=0,
             label="within network")
    axb.barh(ypos - 0.19, between, height=0.34, color="#b3b3b3", lw=0,
             label="between networks")
    axb.errorbar(within, ypos + 0.19, xerr=wsd, fmt="none", ecolor="#555555",
                 elinewidth=0.7, capsize=1.8)
    axb.axvline(0.0, color=COLORS["rule"], lw=0.8)
    axb.set_yticks(ypos)
    axb.set_yticklabels(names, fontsize=7.0)
    for tick, c in zip(axb.get_yticklabels(), colors):
        tick.set_color(c)
    axb.set_xlabel("Mean Pearson $r$ (-)", fontsize=7.6)
    axb.tick_params(labelsize=6.8, length=2.2)
    axb.set_ylim(-0.65, len(NETWORKS) - 0.35)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)
    axb.set_title("Within- vs between-network coupling", fontsize=8.2,
                  fontweight="bold", pad=4)
    axb.legend(fontsize=6.8, loc="lower right", frameon=False,
               handlelength=1.1, borderaxespad=0.2)

    # ---- shared network key, drawn once -----------------------------------
    handles = [Line2D([], [], marker="s", ls="none", ms=5.0, mec="none",
                      mfc=c, label=s) for s, _, c in NETWORKS]

    panel_labels_inches(fig, [ax, axb], dx=-0.52, dy=0.42)

    i, j = np.unravel_index(np.argmin(np.where(off, R, np.inf)), R.shape)

    savefig(fig, "536_fmri_connectivity_matrix.png")


if __name__ == "__main__":
    main()

"""Point-to-point communication matrix of a 32-rank domain decomposition.

Rank i sends to rank j: the halo exchange of an 8x4 Cartesian grid puts most
of the traffic on a few off-diagonal bands (the x-neighbours are one rank
away, the y-neighbours eight), the gather/broadcast pair through the I/O root
adds the cross through rank 0, and a light all-to-all floor sits underneath
everything. The volumes span four orders of magnitude, so the colour scale is
logarithmic; without it the halo bands would saturate and the collective
pattern would be invisible. The marginal bars give the per-rank send and
receive totals on one shared value axis, which is what makes the receive-side
imbalance readable at a glance.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator
from pubstyle import set_pub_style, savefig, get_cmap
from figkit import axes_inches, safe_colorbar, shared_limits

NX, NY = 8, 4          # Cartesian process grid
NP = NX * NY
ROOT = 0               # I/O aggregator rank

BAR = "#3b5b92"        # house blue
ACCENT = "#b23a48"     # house red, the single emphasis object
RULE = "#333333"
FRAME = "#8a8a8a"
SUB = "#555555"
MUTED = "#777777"

FW, FH = 5.8, 4.5
HM = (0.62, 0.78, 2.72, 2.72)          # x0, y0, w, h in inches
TOP = (0.62, 3.54, 2.72, 0.62)
RIGHT = (3.48, 0.78, 0.62, 2.72)


def build_matrix(rng):
    """Volumes in MB from rank i to rank j; the diagonal is not traffic."""
    c = 0.045 * np.exp(rng.normal(0.0, 0.55, (NP, NP)))   # all-to-all floor
    for r in range(NP):
        rx, ry = r % NX, r // NX
        for dx, dy, vol in ((1, 0, 62.0), (-1, 0, 62.0),
                            (0, 1, 31.0), (0, -1, 31.0)):
            s = ((rx + dx) % NX) + ((ry + dy) % NY) * NX
            c[r, s] += vol * np.exp(rng.normal(0.0, 0.10))
    c[:, ROOT] += 7.6 * np.exp(rng.normal(0.0, 0.20, NP))  # gather to root
    c[ROOT, :] += 3.4 * np.exp(rng.normal(0.0, 0.20, NP))  # broadcast
    np.fill_diagonal(c, np.nan)
    return c


def main():
    set_pub_style()
    rng = np.random.default_rng(5661)
    c = build_matrix(rng)

    sent = np.nansum(c, axis=1)
    recv = np.nansum(c, axis=0)
    total_gb = np.nansum(c) / 1024.0
    imbalance = recv.max() / recv.mean()
    hot = int(np.argmax(recv))

    fig = plt.figure(figsize=(FW, FH))
    ax = axes_inches(fig, HM)
    ax_top = axes_inches(fig, TOP)
    ax_rgt = axes_inches(fig, RIGHT)

    # ---- matrix -----------------------------------------------------------
    cmap = get_cmap("cb_sequential_blue")
    cmap = cmap.with_extremes(bad="#f5f5f5")
    norm = mpl.colors.LogNorm(vmin=np.nanmin(c), vmax=np.nanmax(c))
    im = ax.imshow(c, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color(FRAME)
    ticks = np.arange(0, NP, 4)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=7.0, length=2.4)
    ax.set_xlabel("receiver rank (-)", fontsize=8.0)
    ax.set_ylabel("sender rank (-)", fontsize=8.0)
    ax.set_xlim(-0.5, NP - 0.5)
    ax.set_ylim(NP - 0.5, -0.5)

    ax.axhline(ROOT, color=ACCENT, lw=0.7, ls=":", alpha=0.85, zorder=4)
    ax.axvline(ROOT, color=ACCENT, lw=0.7, ls=":", alpha=0.85, zorder=4)
    # Labels sit on the map itself, so they carry a white stroke rather than
    # an opaque box that would hide cells.
    stroke = [pe.withStroke(linewidth=2.0, foreground="white")]
    a1 = ax.annotate("collective cross\nthrough rank 0",
                     xy=(ROOT, NP * 0.72), xytext=(NP * 0.26, NP * 0.83),
                     fontsize=6.4, color=ACCENT, ha="left", va="center",
                     linespacing=1.30, zorder=8,
                     arrowprops=dict(arrowstyle="-", lw=0.7, color=ACCENT,
                                     shrinkA=1.0, shrinkB=1.0))
    a2 = ax.annotate("halo bands: +/-1 in x, +/-8 in y",
                     xy=(NP * 0.52, NP * 0.52 - 1.0),
                     xytext=(NP * 0.30, NP * 0.24), fontsize=6.4, color=RULE,
                     ha="left", va="center", zorder=8,
                     arrowprops=dict(arrowstyle="-", lw=0.7, color=MUTED,
                                     shrinkA=1.0, shrinkB=1.0))
    for a in (a1, a2):
        a.set_path_effects(stroke)

    # ---- marginals, one shared value axis ---------------------------------
    lo, hi = shared_limits([sent, recv], pad=0.10, lo=0.0)
    ranks = np.arange(NP)

    ax_top.bar(ranks, sent, width=0.78, color=BAR, edgecolor="none")
    ax_top.set_xlim(-0.5, NP - 0.5)
    ax_top.set_ylim(lo, hi)
    ax_top.set_xticks(ticks)
    ax_top.set_xticklabels([])
    ax_top.tick_params(labelsize=6.5, length=2.2)
    ax_top.set_ylabel("sent (MB)", fontsize=7.0)
    ax_top.grid(axis="y", lw=0.45, alpha=0.5)

    cols = [ACCENT if i == hot else BAR for i in ranks]
    ax_rgt.barh(ranks, recv, height=0.78, color=cols, edgecolor="none")
    ax_rgt.set_ylim(NP - 0.5, -0.5)
    ax_rgt.set_xlim(lo, hi)
    ax_rgt.set_yticks(ticks)
    ax_rgt.set_yticklabels([])
    ax_rgt.tick_params(labelsize=6.5, length=2.2)
    ax_rgt.set_xlabel("received (MB)", fontsize=7.0)
    ax_rgt.grid(axis="x", lw=0.45, alpha=0.5)
    ax_rgt.xaxis.set_major_locator(MaxNLocator(3))
    plt.setp(ax_rgt.get_xticklabels(), rotation=45, ha="right")
    ax_top.yaxis.set_major_locator(MaxNLocator(3))
    ax_rgt.annotate(f"rank {hot}\n{recv[hot]:.0f} MB",
                    xy=(recv[hot], hot), xytext=(hi * 0.98, hot + 5.5),
                    fontsize=6.2, color=ACCENT, ha="right", va="top",
                    linespacing=1.30,
                    arrowprops=dict(arrowstyle="-", lw=0.7, color=ACCENT,
                                    shrinkA=1.0, shrinkB=1.5))

    cb = safe_colorbar(fig, im, ax_rgt, label="message volume (MB)",
                       width=0.12, pad=0.34, labelsize=6.8)
    cb.ax.tick_params(labelsize=6.8)

    x_mid = (HM[0] + 5.04) / 2.0 / FW

    savefig(fig, "566_hpc_communication_matrix.png")


if __name__ == "__main__":
    main()

"""Figure 645 - Source-destination traffic matrix with its two margins.

Eighteen points of presence in four regions, one log colour scale.  Traffic
between backbone sites spans four orders of magnitude, so a linear ramp would
paint everything except the two or three biggest flows the same colour; the
log scale is what makes the intra-region blocks readable at all.  Sites are
ordered by region and the region boundaries are cut in white, so the block
structure is visible without a dendrogram.  The two margin panels are the row
and column sums of the same matrix, and every number quoted in the corner is
reduced from it.

Reference-gallery figure: the numbers are generated, the layout is the point.
Canvas aspect follows DESIGN_SYSTEM section 1.2 exception E6 (matrix figure).
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from pubstyle import set_pub_style, savefig, get_cmap, COLORS
from figkit import panel_label_inches

ACCENT = "#b23a48"
REGIONS = [("EU-West", 5, "#3b5b92"), ("US-East", 5, "#2e7d5b"),
           ("APAC", 4, "#c9a227"), ("LATAM", 4, "#8e6bbf")]


def build(rng):
    """Return (names, colours, boundaries, matrix in Gbps)."""
    names, cols, bounds = [], [], []
    tag = {"EU-West": "EU", "US-East": "US", "APAC": "AP", "LATAM": "LA"}
    k = 0
    for reg, n, col in REGIONS:
        for i in range(n):
            names.append("%s%d" % (tag[reg], i + 1))
            cols.append(col)
        k += n
        bounds.append(k)
    n = len(names)

    group = np.concatenate([np.full(g[1], i)
                            for i, g in enumerate(REGIONS)])
    same = group[:, None] == group[None, :]
    # log-normal core, one decade louder inside a region than across regions
    base = rng.lognormal(np.log(0.9), 1.05, size=(n, n))
    mat = base * np.where(same, 11.0, 1.0)
    pop = rng.lognormal(0.0, 0.55, n)            # per-site size factor
    mat *= np.outer(pop, pop)
    # three transit backbones carry an order of magnitude more than the rest
    for a, b in ((0, 6), (5, 10), (2, 15)):
        mat[a, b] *= 14.0
        mat[b, a] *= 9.0
    np.fill_diagonal(mat, np.nan)
    return names, cols, bounds[:-1], mat


def gini(x):
    x = np.sort(np.asarray(x, dtype=float).ravel())
    n = x.size
    return float((2.0 * np.arange(1, n + 1) - n - 1.0) @ x / (n * x.sum()))


def main():
    set_pub_style()
    rng = np.random.default_rng(645)
    names, cols, bounds, mat = build(rng)
    n = len(names)

    flows = mat[np.isfinite(mat)]
    total = flows.sum()
    egress = np.nansum(mat, axis=1)
    ingress = np.nansum(mat, axis=0)
    i_max, j_max = np.unravel_index(np.nanargmax(mat), mat.shape)
    g = gini(flows)

    cmap = get_cmap("cb_sequential_blue")
    cmap.set_bad("#ededed")
    norm = LogNorm(vmin=np.percentile(flows, 1.5), vmax=flows.max())

    fig = plt.figure(figsize=(7.4, 6.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.30],
                          height_ratios=[0.30, 1.0],
                          left=0.105, right=0.955, top=0.905, bottom=0.150,
                          wspace=0.06, hspace=0.06)
    ax_t = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[1, 0])
    ax_r = fig.add_subplot(gs[1, 1])
    ax_c = fig.add_subplot(gs[0, 1])

    im = ax_m.imshow(mat, cmap=cmap, norm=norm, interpolation="nearest",
                     aspect="auto", zorder=2)
    for b in bounds:
        ax_m.axhline(b - 0.5, color="white", lw=1.8, zorder=4)
        ax_m.axvline(b - 0.5, color="white", lw=1.8, zorder=4)
    ax_m.plot([j_max], [i_max], marker="s", ms=7.5, mfc="none", mec=ACCENT,
              mew=1.4, zorder=6)

    ax_m.set_xticks(np.arange(n))
    ax_m.set_yticks(np.arange(n))
    ax_m.set_xticklabels(names, fontsize=6.2, rotation=90)
    ax_m.set_yticklabels(names, fontsize=6.2)
    for lab, c in zip(ax_m.get_xticklabels(), cols):
        lab.set_color(c)
    for lab, c in zip(ax_m.get_yticklabels(), cols):
        lab.set_color(c)
    ax_m.tick_params(length=2.0)
    ax_m.set_xlabel("destination point of presence", fontsize=8.0)
    ax_m.set_ylabel("source point of presence", fontsize=8.0)
    for s in ax_m.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")

    ax_t.bar(np.arange(n), ingress, width=0.74, color=cols, edgecolor="white",
             linewidth=0.4, zorder=3)
    ax_t.set_xlim(-0.5, n - 0.5)
    ax_t.set_ylim(0.0, max(ingress.max(), egress.max()) * 1.10)
    ax_t.tick_params(labelbottom=False, labelsize=6.2, length=2.0)
    ax_t.set_ylabel("ingress\n(Gbps)", fontsize=7.0, linespacing=1.25)
    ax_t.grid(axis="y", lw=0.45, alpha=0.5)
    ax_t.set_axisbelow(True)
    for s in ("top", "right"):
        ax_t.spines[s].set_visible(False)

    start = 0
    for (reg, _, col), end in zip(REGIONS, list(bounds) + [n]):
        ax_t.text(start + (end - start - 1) / 2.0, 1.07, reg,
                  transform=ax_t.get_xaxis_transform(), ha="center",
                  va="bottom", fontsize=6.8, color=col, fontweight="bold")
        start = end

    ax_r.barh(np.arange(n), egress, height=0.74, color=cols,
              edgecolor="white", linewidth=0.4, zorder=3)
    ax_r.set_ylim(n - 0.5, -0.5)
    ax_r.set_xlim(0.0, max(ingress.max(), egress.max()) * 1.10)
    ax_r.tick_params(labelleft=False, labelsize=6.2, length=2.0)
    ax_r.set_xlabel("egress (Gbps)", fontsize=7.0)
    ax_r.grid(axis="x", lw=0.45, alpha=0.5)
    ax_r.set_axisbelow(True)
    for s in ("top", "right"):
        ax_r.spines[s].set_visible(False)

    ax_c.axis("off")
    cax = ax_c.inset_axes([0.03, 0.80, 0.94, 0.105])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("flow rate (Gbps)", fontsize=6.4, color=COLORS["ink"],
                 labelpad=2.0)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(COLORS["rule"])
    cb.ax.tick_params(labelsize=6.0, width=0.7, length=2.2,
                      color=COLORS["rule"])
    ax_c.text(0.03, 0.40, "total carried  %.0f Gbps\nGini of flows  %.3f"
              % (total, g), ha="left", va="top", fontsize=6.0,
              color=COLORS["sub"], linespacing=1.32)
    ax_c.text(0.03, 0.21, "top pair %s to %s\n%.0f Gbps (%.1f %% of total)"
              % (names[i_max], names[j_max], mat[i_max, j_max],
                 100.0 * mat[i_max, j_max] / total), ha="left", va="top",
              fontsize=6.0, color=ACCENT, linespacing=1.32)

    for a, lab, dx in ((ax_t, "(a)", -0.56), (ax_m, "(b)", -0.56),
                       (ax_r, "(c)", -0.22)):
        panel_label_inches(fig, a, lab, dx=dx, dy=0.07)

    intra = np.nansum(np.where(
        np.concatenate([np.full(s, i) for i, (_, s, _) in
                        enumerate(REGIONS)])[:, None] ==
        np.concatenate([np.full(s, i) for i, (_, s, _) in
                        enumerate(REGIONS)])[None, :], mat, 0.0))

    savefig(fig, "645_net_traffic_matrix.png")


if __name__ == "__main__":
    main()

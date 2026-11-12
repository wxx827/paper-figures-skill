"""A food web drawn on a trophic-level axis, with the pyramid it implies.

Nothing here is placed by eye on the vertical axis.  Each consumer carries a
diet composition, and the trophic levels are the solution of the standard
linear system TL_i = 1 + sum_j DC_ij TL_j - a 14 x 14 solve - so the height of
every node is derived from the same diet matrix that draws the edges.  A
producer sits at 1.0 because its row is empty, and the apex predator lands
just above 5 because of what it eats, not because the layout wanted a node up
there.

Edge width encodes energy flow, Q_i = B_i (Q/B)_i split by diet fraction, on a
log scale because flows span four decades; node area encodes biomass on the
same kind of log scale, keyed by the three reference circles at the top left.
The red path is one top-down cascade traced through the web.

The side panels are aggregations of exactly those numbers: biomass summed into
integer trophic bins as a pyramid, and the transfer efficiency between
neighbouring bins as their biomass ratio, which is where the textbook ten
percent comes from.  Illustrative values; this is a layout reference.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, get_cmap, lighten, darken
from figkit import axes_inches, panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
GREY = "#808080"
CASCADE = "#b23a48"
EDGE = "#9aa7b4"

FW, FH = 10.4, 6.4
MAIN = (0.86, 1.04, 4.86, 4.16)
AUX1 = (6.62, 3.14, 2.78, 2.06)
AUX2 = (6.62, 1.04, 2.78, 1.44)

# key, display name, x, biomass (t/km2), Q/B (1/yr), label offset, ha, note
NODES = [
    ("PHY", "phytoplankton", 2.2, 22.0, 0.0, (-13, 0), "right",
     "primary producer"),
    ("MAC", "macroalgae", 6.5, 8.0, 0.0, (13, 0), "left", None),
    ("DET", "detritus", 9.2, 12.0, 0.0, (12, 0), "left", None),
    ("ZOO", "zooplankton", 2.0, 3.00, 30.0, (-12, 0), "right", None),
    ("BEN", "benthic grazers", 6.4, 1.10, 14.0, (12, 0), "left", None),
    ("FIL", "filter feeders", 8.9, 1.40, 12.0, (11, 0), "left", None),
    ("ANC", "small pelagics", 2.4, 0.300, 9.0, (-11, 0), "right",
     "forage-fish bottleneck"),
    ("SHR", "shrimp", 5.4, 0.160, 10.0, (0, -13), "center", None),
    ("DEM", "demersal fish", 8.4, 0.120, 5.0, (10, 0), "left", None),
    ("SQD", "squid", 4.4, 0.0300, 6.0, (-10, 0), "right", None),
    ("MAK", "mackerel", 1.9, 0.0260, 5.0, (-10, 0), "right", None),
    ("COD", "large demersal", 7.8, 0.0140, 3.4, (10, 0), "left", None),
    ("TUN", "tuna", 3.0, 0.00550, 2.8, (-10, 0), "right", None),
    ("SHK", "shark", 6.2, 0.00200, 2.4, (10, 0), "left", "keystone predator"),
]

DIET = {
    "ZOO": {"PHY": 0.85, "DET": 0.15},
    "BEN": {"MAC": 0.55, "DET": 0.45},
    "FIL": {"PHY": 0.60, "DET": 0.40},
    "ANC": {"ZOO": 0.90, "PHY": 0.10},
    "SHR": {"DET": 0.40, "BEN": 0.35, "ZOO": 0.25},
    "DEM": {"BEN": 0.40, "SHR": 0.35, "FIL": 0.25},
    "SQD": {"ANC": 0.70, "SHR": 0.30},
    "MAK": {"ANC": 0.65, "ZOO": 0.35},
    "COD": {"DEM": 0.45, "SHR": 0.25, "ANC": 0.30},
    "TUN": {"MAK": 0.45, "SQD": 0.35, "ANC": 0.20},
    "SHK": {"TUN": 0.25, "COD": 0.35, "SQD": 0.25, "MAK": 0.15},
}

CASCADE_PATH = ["PHY", "ZOO", "ANC", "MAK", "TUN", "SHK"]


def solve_trophic():
    """TL = 1 + DC @ TL, solved as (I - DC) TL = 1."""
    keys = [n[0] for n in NODES]
    pos = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    dc = np.zeros((n, n))
    for pred, diet in DIET.items():
        for prey, frac in diet.items():
            dc[pos[pred], pos[prey]] = frac
    tl = np.linalg.solve(np.eye(n) - dc, np.ones(n))
    return keys, pos, dc, tl


def energy_flows(pos, dc):
    """Flow from prey to predator, Q_i = B_i (Q/B)_i split by diet share."""
    b = np.array([n[3] for n in NODES])
    qb = np.array([n[4] for n in NODES])
    cons = b * qb
    flow = dc * cons[:, None]
    return b, cons, flow


def draw_web(ax, keys, pos, tl, b, flow):
    lb = np.log10(b)
    lo, hi = lb.min(), lb.max()
    size = 20.0 + 540.0 * (lb - lo) / (hi - lo)
    cmap = get_cmap("cb_sequential_green")
    tnorm = (tl - 1.0) / (tl.max() - 1.0)

    pairs = set(zip(CASCADE_PATH[:-1], CASCADE_PATH[1:]))
    fpos = flow[flow > 0]
    flo, fhi = np.log10(fpos.min()), np.log10(fpos.max())
    for pred, diet in DIET.items():
        i = pos[pred]
        for prey in diet:
            j = pos[prey]
            f = flow[i, j]
            w = 0.35 + 2.35 * (np.log10(f) - flo) / (fhi - flo)
            hot = (prey, pred) in pairs
            ax.annotate("", xy=(NODES[i][2], tl[i]),
                        xytext=(NODES[j][2], tl[j]),
                        arrowprops=dict(
                            arrowstyle="-|>", shrinkA=3.0, shrinkB=5.0,
                            linewidth=w if not hot else w + 0.55,
                            color=CASCADE if hot else EDGE,
                            alpha=0.95 if hot else 0.62,
                            mutation_scale=8.0,
                            connectionstyle="arc3,rad=0.10"),
                        zorder=5 if hot else 3)

    for i, nd in enumerate(NODES):
        ax.scatter([nd[2]], [tl[i]], s=size[i], color=cmap(tnorm[i]),
                   edgecolors="white", linewidths=0.7, zorder=8)
    for i, nd in enumerate(NODES):
        dx, dy = nd[5]
        txt = nd[1]
        ax.annotate(txt, xy=(nd[2], tl[i]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=6.5, color=INK,
                    ha=nd[6], va="center", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                              edgecolor="none", alpha=0.80))
        if nd[7]:
            ax.annotate(nd[7], xy=(nd[2], tl[i]), xytext=(dx, dy - 8.6),
                        textcoords="offset points", fontsize=6,
                        color=CASCADE, ha=nd[6], va="center", zorder=10)
        ax.annotate("TL {:.2f}".format(tl[i]), xy=(nd[2], tl[i]),
                    xytext=(dx, dy + (8.6 if dy >= 0 else -8.6)),
                    textcoords="offset points",
                    fontsize=6, color=MUTED, ha=nd[6], va="center",
                    zorder=10)

    key_b = [0.01, 1.0, 25.0]
    key_x = [0.55, 1.45, 2.55]
    for kb, kx in zip(key_b, key_x):
        ks = 20.0 + 540.0 * (np.log10(kb) - lo) / (hi - lo)
        ax.scatter([kx], [5.28], s=ks, facecolor=lighten(GREY, 0.62),
                   edgecolors="white", linewidths=0.6, zorder=8)
        ax.text(kx, 5.04, "{:g}".format(kb), fontsize=6, color=SUB,
                ha="center", va="top", zorder=9)
    ax.text(0.20, 5.28, "biomass\n(t/km$^2$)", fontsize=6.0, color=SUB,
            ha="right", va="center", linespacing=1.30, zorder=9)

    ax.plot([], [], color=CASCADE, lw=1.6,
            label="top-down cascade path")
    ax.plot([], [], color=EDGE, lw=1.2,
            label="energy flow, width ~ log$_{10}$ t km$^{-2}$ yr$^{-1}$")
    leg = ax.legend(loc="upper right", fontsize=6.4, frameon=True,
                    framealpha=0.94, borderpad=0.40, handlelength=1.8,
                    labelspacing=0.32)
    leg.get_frame().set_edgecolor("#cccccc")
    leg.get_frame().set_linewidth(0.6)
    leg.set_zorder(12)

    for lv in range(1, 6):
        ax.axhline(lv, color="#ececec", lw=0.6, zorder=1)
    ax.set_xlim(-0.35, 10.7)
    ax.set_ylim(0.62, 5.98)
    ax.set_xticks([])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylabel("trophic level (-)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("Food web with nodes placed at their computed trophic level",
                 fontsize=8.5, fontweight="bold")
    return size


def bin_biomass(tl, b):
    lev = np.clip(np.rint(tl).astype(int), 1, 5)
    tot = np.array([b[lev == k].sum() for k in range(1, 6)])
    return lev, tot


def draw_pyramid(ax, tot):
    cmap = get_cmap("cb_sequential_green")
    lt = np.log10(tot)
    half = (lt - lt.min()) + 0.42
    half = half / half.max()
    for k in range(5):
        y = k + 1
        ax.barh(y, 2.0 * half[k], left=-half[k], height=0.74,
                color=cmap(0.22 + 0.70 * k / 4.0), edgecolor="white",
                linewidth=0.6, zorder=4)
        # A white pill keeps one text colour legible across the whole ramp,
        # including the narrow apex bar the label overhangs.
        ax.text(0.0, y, "TL {:d}   {:s} t/km$^2$"
                .format(k + 1, "{:.4g}".format(tot[k])), fontsize=6.3,
                color=INK, ha="center", va="center", zorder=6,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.86))
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(0.35, 5.75)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("bar width $\\propto$ log$_{10}$ biomass (t/km$^2$ shown)",
                  fontsize=7.4)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title("Biomass pyramid by trophic level", fontsize=8.5,
                 fontweight="bold")


def draw_efficiency(ax, tot):
    te = tot[1:] / tot[:-1]
    x = np.arange(4)
    cmap = get_cmap("cb_sequential_blue")
    ax.bar(x, te * 100.0, width=0.60, color=[cmap(0.34 + 0.16 * i)
                                             for i in range(4)],
           edgecolor="white", linewidth=0.5, zorder=4)
    ax.axhline(10.0, color=CASCADE, lw=0.9, ls=(0, (3.4, 2.2)), zorder=5)
    ax.text(-0.55, 10.5, "10% rule", fontsize=6.2, color=CASCADE, ha="left",
            va="bottom", zorder=6)
    for xi, v in zip(x, te):
        ax.text(xi, v * 100.0 + 0.5, "{:.1f}%".format(v * 100.0),
                fontsize=6.3, color=SUB, ha="center", va="bottom", zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels(["1 $\\rightarrow$ 2", "2 $\\rightarrow$ 3",
                        "3 $\\rightarrow$ 4", "4 $\\rightarrow$ 5"],
                       fontsize=6.5)
    ax.set_xlim(-0.62, 3.62)
    ax.set_ylim(0.0, te.max() * 100.0 * 1.34)
    ax.set_ylabel("transfer efficiency (%)", fontsize=7.6)
    ax.set_xlabel("trophic level step (-)", fontsize=7.6)
    ax.tick_params(labelsize=6.6)
    ax.set_title("Biomass transfer between levels", fontsize=8.5,
                 fontweight="bold")
    return te


def main():
    set_pub_style()
    rng = np.random.default_rng(9030)
    keys, pos, dc, tl = solve_trophic()
    b, cons, flow = energy_flows(pos, dc)
    lev, tot = bin_biomass(tl, b)

    fig = plt.figure(figsize=(FW, FH))
    ax_w = axes_inches(fig, MAIN)
    ax_p = axes_inches(fig, AUX1)
    ax_e = axes_inches(fig, AUX2)

    draw_web(ax_w, keys, pos, tl, b, flow)
    draw_pyramid(ax_p, tot)
    te = draw_efficiency(ax_e, tot)

    panel_labels_inches(fig, [ax_w, ax_p, ax_e], dx=-0.46, dy=0.20)

    top = keys[int(np.argmax(tl))]
    top_name = NODES[int(np.argmax(tl))][1]
    n_edge = int(np.sum(flow > 0))

    print("tl_max={:.3f} ({:s}) edges={:d} te={:s}"
          .format(tl.max(), top, n_edge,
                  " ".join("{:.3f}".format(v) for v in te)))
    savefig(fig, "903_fsh_trophic_network.png")


if __name__ == "__main__":
    main()

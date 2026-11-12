"""Locational marginal prices and the congestion that creates them.

(a) a price surface interpolated over the footprint, with the priced nodes
drawn on top and the two binding constraints marked as red cut lines: the
contours bunch up across a binding interface, which is the point of the map.
(b) the LMP of the twelve nodes decomposed into energy, congestion and loss
components, sorted by total price.
(c) how many hours each interface was binding.

Layout note: the map is the result, so it takes ~57 % of the width and the
full height; (b) and (c) are stacked read-outs in the right column and (c) is
the shortest panel of the three.  (b) and the map share one price scale.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import Rbf

from pubstyle import set_pub_style, PALETTES, savefig, style_axes

INK = "#333333"
GREY = "#8d959c"
ALERT = "#b23a48"
# LMP decomposition: energy is the common component, congestion is the story
COMP = [("Energy", "#7ba5bd"), ("Congestion", "#b23a48"),
        ("Loss", "#c9c2b6")]

# node -> (x, y, energy, congestion, loss) in $/MWh
NODES = {
    "N01": (0.68, 4.42, 41.0, -6.2, 0.9),
    "N02": (1.95, 5.05, 41.0, -5.4, 1.1),
    "N03": (1.24, 3.05, 41.0, -4.1, 1.4),
    "N04": (2.86, 3.72, 41.0, -2.6, 1.6),
    "N05": (3.55, 5.28, 41.0, -1.2, 1.3),
    "N06": (4.42, 4.05, 41.0, 3.8, 2.0),
    "N07": (4.05, 2.42, 41.0, 6.6, 2.3),
    "N08": (5.62, 3.18, 41.0, 18.4, 2.8),
    "N09": (5.18, 1.42, 41.0, 21.7, 3.1),
    "N10": (6.55, 2.05, 41.0, 32.4, 3.6),
    "N11": (6.88, 4.32, 41.0, 27.9, 3.2),
    "N12": (2.28, 1.28, 41.0, -3.3, 1.2),
}

# binding interfaces: (x0, y0, x1, y1, name, hours binding)
CUTS = [
    (3.72, 0.55, 4.05, 5.95, "Interface A", 1284),
    (5.95, 0.90, 6.20, 5.60, "Interface B", 742),
]
OTHER_IF = [("Interface C", 386), ("Interface D", 205),
            ("Interface E", 118), ("Interface F", 64)]


def main():
    set_pub_style(base_font=8)
    names = list(NODES)
    xs = np.array([NODES[n][0] for n in names])
    ys = np.array([NODES[n][1] for n in names])
    en = np.array([NODES[n][2] for n in names])
    cg = np.array([NODES[n][3] for n in names])
    ls = np.array([NODES[n][4] for n in names])
    lmp = en + cg + ls

    cmap = LinearSegmentedColormap.from_list(
        "lmp", PALETTES["cb_diverging"], N=256)
    norm = Normalize(np.floor(lmp.min() / 5) * 5, np.ceil(lmp.max() / 5) * 5)

    fig = plt.figure(figsize=(7.7, 4.7))
    # the colour scale rides directly under the map instead of in its own
    # full-height column, where its ticks collided with the right-hand panels
    outer = fig.add_gridspec(1, 2, left=0.072, right=0.978, top=0.885,
                             bottom=0.115, wspace=0.21,
                             width_ratios=[1.0, 0.66])
    left = outer[0, 0].subgridspec(2, 1, height_ratios=[1.0, 0.045],
                                   hspace=0.075)
    ax = fig.add_subplot(left[0])
    axc = fig.add_subplot(left[1])
    right = outer[0, 1].subgridspec(2, 1, height_ratios=[1.0, 0.62],
                                    hspace=0.62)
    axb = fig.add_subplot(right[0])
    axd = fig.add_subplot(right[1])

    # ---- (a) interpolated price surface ----------------------------------
    gx = np.linspace(0.0, 7.6, 260)
    gy = np.linspace(0.0, 6.5, 230)
    GX, GY = np.meshgrid(gx, gy)
    rbf = Rbf(xs, ys, lmp, function="multiquadric", smooth=0.6, epsilon=2.2)
    Z = rbf(GX, GY)
    im = ax.imshow(Z, origin="lower", extent=(0, 7.6, 0, 6.5), cmap=cmap,
                   norm=norm, aspect="auto", interpolation="bilinear",
                   zorder=1)
    ax.contour(GX, GY, Z, levels=np.arange(35, 81, 5), colors="white",
               linewidths=0.45, alpha=0.55, zorder=2)

    for x0, y0, x1, y1, name, _ in CUTS:
        ax.plot([x0, x1], [y0, y1], color="white", lw=3.2, zorder=3,
                solid_capstyle="round")
        ax.plot([x0, x1], [y0, y1], color=ALERT, lw=1.6,
                ls=(0, (5.5, 2.2)), zorder=4, solid_capstyle="butt")
        ax.text(x1 + 0.06, y1 + 0.12, name, fontsize=6.4, color=ALERT,
                fontweight="bold", ha="center", va="bottom", zorder=6,
                bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                          edgecolor="none", alpha=0.85))

    ax.scatter(xs, ys, s=52, c=lmp, cmap=cmap, norm=norm, edgecolors=INK,
               linewidths=0.7, zorder=5)
    for n, x, y, v in zip(names, xs, ys, lmp):
        ax.text(x, y - 0.30, f"{v:.0f}", fontsize=6.0, color=INK,
                ha="center", va="top", zorder=6,
                bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                          edgecolor="none", alpha=0.78))

    ax.set_xlim(0, 7.6)
    ax.set_ylim(0, 6.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_edgecolor(INK)
    ax.set_title("Locational marginal price surface", fontsize=8.8,
                 fontweight="bold", loc="left", x=0.0)
    ax.text(0.014, 0.975, f"spread {lmp.max() - lmp.min():.0f} $/MWh "
            f"across {len(names)} nodes", transform=ax.transAxes,
            fontsize=6.4, color=INK, ha="left", va="top", zorder=7,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor="#d5d5d5", lw=0.6, alpha=0.92))

    cb = fig.colorbar(im, cax=axc, orientation="horizontal")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(INK)
    cb.ax.tick_params(labelsize=6.4, width=0.7, length=2.4, color=INK,
                      pad=1.5)
    cb.set_label("LMP ($/MWh)", fontsize=6.9, labelpad=2)

    # ---- (b) price decomposition -----------------------------------------
    order = np.argsort(lmp)
    ypos = np.arange(len(names))
    neg = np.zeros(len(names))
    pos = np.zeros(len(names))
    for vals, (_, col) in zip((en[order], cg[order], ls[order]), COMP):
        base = np.where(vals < 0, neg, pos)
        axb.barh(ypos, vals, left=base, height=0.66, color=col, lw=0,
                 zorder=3)
        neg = np.where(vals < 0, neg + vals, neg)
        pos = np.where(vals < 0, pos, pos + vals)
    axb.plot(lmp[order], ypos, marker="o", ms=3.0, lw=0, color=INK,
             mec="white", mew=0.6, zorder=5)
    axb.axvline(0, color=INK, lw=0.7, zorder=4)
    axb.set_yticks(ypos)
    axb.set_yticklabels([names[i] for i in order], fontsize=6.2)
    axb.set_ylim(-0.7, len(names) - 0.3)
    axb.set_xlim(-12, 82)
    axb.set_xticks([0, 20, 40, 60, 80])
    axb.tick_params(axis="y", length=0)
    axb.set_xlabel("$/MWh", fontsize=7.2)
    axb.set_title("LMP decomposition", fontsize=8.0, fontweight="bold",
                  loc="left", x=0.0)
    axb.grid(axis="x", lw=0.4, alpha=0.5)
    style_axes(axb)

    handles = [Patch(facecolor=c, label=n) for n, c in COMP]
    handles.append(Line2D([], [], marker="o", color="none",
                          markerfacecolor=INK, markeredgecolor="white",
                          markeredgewidth=0.6, markersize=3.0,
                          label="Total LMP"))
    leg = axb.legend(handles=handles, loc="lower right",
                     bbox_to_anchor=(1.01, -0.02), frameon=True,
                     facecolor="white", edgecolor="#dcdcdc", framealpha=0.95,
                     fontsize=6.4, handlelength=1.4, labelspacing=0.35,
                     borderpad=0.42)
    leg.get_frame().set_linewidth(0.6)
    leg.set_clip_on(False)

    # ---- (c) binding hours (shortest panel) ------------------------------
    ifs = [(n, h) for *_, n, h in CUTS] + OTHER_IF
    ifs.sort(key=lambda p: -p[1])
    yy = np.arange(len(ifs))[::-1]
    cols = [ALERT if n in ("Interface A", "Interface B") else "#c8ccd0"
            for n, _ in ifs]
    axd.barh(yy, [h for _, h in ifs], height=0.62, color=cols, lw=0,
             zorder=3)
    for y, (n, h) in zip(yy, ifs):
        axd.text(h + 40, y, f"{h:,}", fontsize=6.2, color=INK, va="center",
                 ha="left", zorder=4)
    axd.set_yticks(yy)
    axd.set_yticklabels([n.replace("Interface ", "") for n, _ in ifs],
                        fontsize=6.4)
    axd.set_xlim(0, 1620)
    axd.set_xticks([0, 500, 1000, 1500])
    axd.tick_params(axis="y", length=0)
    axd.set_xlabel("Hours binding per year", fontsize=7.2)
    axd.set_ylabel("Interface", fontsize=7.2, labelpad=1)
    axd.set_title("Congestion frequency", fontsize=8.0, fontweight="bold",
                  loc="left", x=0.0)
    axd.grid(axis="x", lw=0.4, alpha=0.5)
    axd.text(0.985, 0.06, "red = mapped in (a)", transform=axd.transAxes,
             fontsize=6.2, color=ALERT, ha="right", va="bottom")
    style_axes(axd)

    fw, fh = fig.get_size_inches()
    for a, lab in ((ax, "(a)"), (axb, "(b)"), (axd, "(c)")):
        bb = a.get_position()
        a.text(-0.30 / (bb.width * fw), 1.0 + 0.055 / (bb.height * fh), lab,
               transform=a.transAxes, ha="left", va="bottom",
               fontweight="bold", fontsize=9.5)

    print(f"LMP {lmp.min():.1f}\u2013{lmp.max():.1f} $/MWh  "
          f"spread {lmp.max() - lmp.min():.1f}")
    print(f"congestion component {cg.min():+.1f} to {cg.max():+.1f} $/MWh  "
          f"loss {ls.min():.1f}\u2013{ls.max():.1f}")
    savefig(fig, "201_lmp_congestion.png")


if __name__ == "__main__":
    main()

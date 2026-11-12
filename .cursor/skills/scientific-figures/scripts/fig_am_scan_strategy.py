"""Six laser scan strategies for one powder-bed layer, and what they cost.

The six tiles are the same layer footprint, the same hatch spacing and the same
colour ruler, so the only thing that changes between them is the order in which
the vectors are laid down - which is exactly the variable the figure is about.
Colour runs from the first vector to the last, so the reader can follow the
build sequence without arrows.  The strip underneath turns the qualitative
difference into the two numbers a process engineer actually buys: peak
longitudinal residual stress and relative density, both plotted against the
same six strategies in the same left-to-right order.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import axes_inches, halo_text, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RULE = "#333333"
ACCENT = "#b23a48"      # residual stress, the quantity to be minimised
GREEN = "#2e7d5b"       # relative density
GREY = "#808080"
HAIR = "#cccccc"

# The pale end of the sequential ramp disappears against the tile, so the
# ruler starts a quarter of the way in; it is still monotonic in lightness.
ORDER_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "scan_order", get_cmap("cb_sequential_blue")(np.linspace(0.26, 1.0, 256)))

W_PART, H_PART = 12.0, 6.0      # mm, layer footprint
HATCH = 0.30                    # mm, hatch spacing


def clip_segment(px, py, ux, uy, rect):
    """Liang-Barsky clip of an infinite line against an axis-aligned box."""
    x0, y0, x1, y1 = rect
    t0, t1 = -1.0e9, 1.0e9
    for p, q in ((-ux, px - x0), (ux, x1 - px), (-uy, py - y0), (uy, y1 - py)):
        if abs(p) < 1.0e-12:
            if q < 0.0:
                return None
        else:
            r = q / p
            if p < 0.0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
    return (t0, t1) if t1 - t0 > 1.0e-9 else None


def hatch_rect(rect, angle_deg, spacing=HATCH, meander=True):
    """Parallel vectors filling a box, returned in the order they are drawn."""
    x0, y0, x1, y1 = rect
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    a = np.deg2rad(angle_deg)
    ux, uy = np.cos(a), np.sin(a)
    nx, ny = -np.sin(a), np.cos(a)
    reach = 0.5 * np.hypot(x1 - x0, y1 - y0)
    offs = np.arange(-reach, reach + spacing, spacing)
    segs = []
    for i, s in enumerate(offs):
        hit = clip_segment(cx + s * nx, cy + s * ny, ux, uy, rect)
        if hit is None:
            continue
        t0, t1 = hit
        p = np.array([[cx + s * nx + t0 * ux, cy + s * ny + t0 * uy],
                      [cx + s * nx + t1 * ux, cy + s * ny + t1 * uy]])
        if meander and i % 2:
            p = p[::-1]
        segs.append(p)
    return segs


def spiral_segments(rect, spacing=HATCH):
    x0, y0, x1, y1 = rect
    segs = []
    while (x1 - x0) > 2.2 * spacing and (y1 - y0) > 2.2 * spacing:
        segs.append(np.array([[x0, y0], [x1, y0]]))
        segs.append(np.array([[x1, y0], [x1, y1]]))
        segs.append(np.array([[x1, y1], [x0, y1]]))
        segs.append(np.array([[x0, y1], [x0, y0 + spacing]]))
        x0, y0, x1, y1 = x0 + spacing, y0 + spacing, x1 - spacing, y1 - spacing
    return segs


def strategies():
    full = (0.0, 0.0, W_PART, H_PART)
    out = {}
    out["Unidirectional"] = hatch_rect(full, 0.0, meander=False)
    out["Bidirectional"] = hatch_rect(full, 0.0, meander=True)

    stripes = []
    for k in range(4):
        stripes += hatch_rect((3.0 * k, 0.0, 3.0 * (k + 1), H_PART), 90.0)
    out["Stripe"] = stripes

    board = []
    tiles = [(c, r) for r in range(2) for c in range(4)]
    for idx in [0, 5, 2, 7, 4, 1, 6, 3]:
        c, r = tiles[idx]
        board += hatch_rect((3.0 * c, 3.0 * r, 3.0 * (c + 1), 3.0 * (r + 1)),
                            90.0 * ((c + r) % 2))
    out["Chessboard"] = board

    out["Spiral"] = spiral_segments(full)

    sector = []
    for k, (c, r) in enumerate([(c, r) for r in range(2) for c in range(3)]):
        sector += hatch_rect((4.0 * c, 3.0 * r, 4.0 * (c + 1), 3.0 * (r + 1)),
                             30.0 * k)
    out["Sector rotation"] = sector
    return out


def outcomes(names, seed=7):
    """Stress and density per strategy; sub-division lowers both extremes."""
    rng = np.random.default_rng(seed)
    base = {"Unidirectional": (342.0, 99.06), "Bidirectional": (308.0, 99.24),
            "Stripe": (247.0, 99.47), "Chessboard": (198.0, 99.63),
            "Spiral": (263.0, 99.21), "Sector rotation": (176.0, 99.71)}
    stress = np.array([base[n][0] for n in names])
    dens = np.array([base[n][1] for n in names])
    s_err = 9.0 + 7.0 * rng.random(len(names))
    d_err = 0.05 + 0.07 * rng.random(len(names))
    return stress, dens, s_err, d_err


def tile(ax, segs, title, norm):
    ax.add_patch(Rectangle((0.0, 0.0), W_PART, H_PART, facecolor="white",
                           edgecolor=HAIR, linewidth=0.7, zorder=0))
    order = np.linspace(0.0, 1.0, len(segs))
    lc = LineCollection(segs, cmap=ORDER_CMAP, norm=norm, linewidths=0.80)
    lc.set_array(order)
    ax.add_collection(lc)
    # Direction is the whole point of the first two tiles, and a line has
    # none, so every sixth vector carries a head at its midpoint.
    step = max(len(segs) // 5, 1)
    step += 1 - step % 2      # odd stride, or a meander's heads all agree
    for i in range(0, len(segs), step):
        (xa, ya), (xb, yb) = segs[i]
        u = np.array([xb - xa, yb - ya])
        n = float(np.hypot(*u)) or 1.0
        mid = np.array([0.5 * (xa + xb), 0.5 * (ya + yb)])
        ax.annotate("", xy=mid + 0.55 * u / n, xytext=mid - 0.55 * u / n,
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=7.5,
                                    lw=0.9, color=ORDER_CMAP(order[i]),
                                    shrinkA=0, shrinkB=0), zorder=4)
    ax.plot([segs[0][0, 0]], [segs[0][0, 1]], marker="o", ms=3.4,
            mfc="white", mec=ACCENT, mew=1.0, zorder=5)
    halo_text(ax, W_PART - 0.18, 0.20, f"n = {len(segs)} vectors", ha="right",
              va="bottom", fontsize=6.2, color=SUB, halo="white",
              halo_lw=2.2, zorder=6)
    ax.set_aspect("equal")
    ax.set_xlim(-0.35, W_PART + 0.35)
    ax.set_ylim(-0.35, H_PART + 0.35)
    ax.set_xticks([0, 4, 8, 12])
    ax.set_yticks([0, 3, 6])
    ax.tick_params(labelsize=6.3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=8.4, fontweight="bold", color=INK, pad=3.0)


def bar_panel(ax, names, stress, dens, s_err, d_err):
    x = np.arange(len(names))
    ax.bar(x, stress, width=0.56, color=ACCENT, alpha=0.88, edgecolor="white",
           linewidth=0.6, yerr=s_err, error_kw=dict(elinewidth=0.8,
                                                    capsize=2.2,
                                                    ecolor=RULE), zorder=3)
    ax.set_ylabel(r"peak $\sigma_{res}$ (MPa)", fontsize=7.6, color=ACCENT)
    ax.set_ylim(0.0, stress.max() * 1.40)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7.0)
    ax.tick_params(axis="y", labelsize=6.6, colors=ACCENT)
    ax.tick_params(axis="x", labelsize=7.0, colors=RULE)
    ax.grid(axis="y", lw=0.5, alpha=0.6)

    ax2 = ax.twinx()
    ax2.plot(x, dens, marker="D", ms=4.2, lw=1.2, color=GREEN, mfc="white",
             mew=1.1, zorder=5)
    ax2.errorbar(x, dens, yerr=d_err, fmt="none", elinewidth=0.8, capsize=2.2,
                 ecolor=GREEN, zorder=4)
    ax2.set_ylabel("relative density (%)", fontsize=7.6, color=GREEN)
    ax2.set_ylim(dens.min() - 0.28, dens.max() + 0.20)
    ax2.tick_params(axis="y", labelsize=6.6, colors=GREEN)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(GREEN)
    ax2.spines["top"].set_visible(False)

    best = int(np.argmin(stress))
    worst = int(np.argmax(stress))
    drop = 100.0 * (stress[worst] - stress[best]) / stress[worst]
    gain = dens[best] - dens[worst]
    handles = [Line2D([], [], color=ACCENT, lw=5.0, alpha=0.88,
                      label=r"peak $\sigma_{res}$ (MPa)"),
               Line2D([], [], color=GREEN, marker="D", ms=4.2, lw=1.2,
                      mfc="white", label="relative density (%)")]
    ax.legend(handles=handles, frameon=False, fontsize=7.0, ncol=2,
              loc="upper right", borderaxespad=0.2, handlelength=1.6)
    ax.set_title(f"{names[best]} drops the peak stress {drop:.0f} % below "
                 f"{names[worst].lower()} and adds {gain:.2f} pp of density",
                 fontsize=8.6, fontweight="bold", color=INK)


def main():
    set_pub_style()
    strat = strategies()
    names = list(strat)
    stress, dens, s_err, d_err = outcomes(names)
    norm = mpl.colors.Normalize(0.0, 1.0)

    fig = plt.figure(figsize=(10.4, 6.4))
    axes = []
    for i, name in enumerate(names):
        col, row = i % 3, i // 3
        ax = axes_inches(fig, (0.70 + col * 2.91, 4.28 - row * 1.82,
                               2.45, 1.225))
        tile(ax, strat[name], name, norm)
        axes.append(ax)
    for ax in axes[:3]:
        ax.set_xticklabels([])
    for i, ax in enumerate(axes):
        if i % 3:
            ax.set_yticklabels([])
    for ax in axes[3:]:
        ax.set_xlabel("x (mm)", fontsize=7.2)
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("y (mm)", fontsize=7.2)

    cax = axes_inches(fig, (9.16, 2.46, 0.13, 3.045))
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=ORDER_CMAP),
                      cax=cax, ticks=[0.0, 0.5, 1.0])
    cb.set_label("scan order, first to last (-)", fontsize=6.6, color=INK)
    cb.ax.set_yticklabels(["start", "half", "end"])
    cb.ax.tick_params(labelsize=6.4, width=0.7, length=2.5, color=RULE)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(RULE)

    ax_bar = axes_inches(fig, (0.95, 0.62, 8.05, 1.10))
    bar_panel(ax_bar, names, stress, dens, s_err, d_err)

    panel_labels_inches(fig, axes + [ax_bar], dx=-0.30, dy=0.10)
    savefig(fig, "745_am_scan_strategy.png")


if __name__ == "__main__":
    main()

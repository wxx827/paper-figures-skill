"""Three-dimensional block model of a faulted, gently folded succession.

Five conformable horizons are drawn as semi-transparent surfaces, split into a
footwall and a hanging-wall block by a listric normal fault whose plane is
rendered as a separate grey surface.  Three boreholes are threaded through the
stack with their lithology columns coloured to match the horizons, so the
reader can tie the 3D geometry back to 1D data.  The depth axis increases
downward.  The sequence key on the right doubles as a mini stratigraphic
column and reports the throw measured from the model itself.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pubstyle import set_pub_style, savefig, lighten, darken, add_panel_label

# unit tops as depth below datum (m) at the model centre, top first
UNITS = [
    ("Sequence S1 - deltaic sand", 40.0, lighten("#c9a227", 0.34)),
    ("Sequence S2 - prodelta mud", 118.0, lighten("#2e7d5b", 0.40)),
    ("Sequence S3 - shelf sand", 196.0, lighten("#3b5b92", 0.36)),
    ("Sequence S4 - marl", 268.0, lighten("#8e6bbf", 0.40)),
    ("Sequence S5 - carbonate", 344.0, lighten("#b23a48", 0.36)),
]
THROW = 92.0             # m of normal separation across the fault
LX = LY = 4.0            # km
ZMAX = 470.0
WELLS = [("W-1", 0.75, 0.75), ("W-2", 2.30, 1.55), ("W-3", 3.25, 3.35)]
INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"


def fold(x, y, k):
    return (0.72 + 0.11 * k) * (26.0 * np.sin(0.92 * x - 0.3) +
                                17.0 * np.cos(0.78 * y + 0.5))


def unfaulted(k, x, y):
    return UNITS[k][1] + fold(x, y, k)


def fault_y(x, z):
    """Trace of the listric fault plane at depth ``z``."""
    return 1.24 + 0.44 * x + 0.00165 * z * (1.0 + 0.55 * z / ZMAX)


def horizon(k, X, Y):
    """Depth grid for horizon ``k`` with the hanging wall dropped."""
    z = unfaulted(k, X, Y)
    hw = Y < fault_y(X, z)      # near block is the downthrown hanging wall
    z = np.where(hw, z + THROW, z)
    return z, hw


def main():
    set_pub_style()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.30], left=0.005,
                          right=0.975, top=0.905, bottom=0.045, wspace=0.02)
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    ax_k = fig.add_subplot(gs[0, 1])
    # mplot3d's automatic depth sort buries the boreholes under the surfaces
    # they pierce; painting in an explicit order is the only way to keep a
    # fence-diagram readable (deepest horizon first, wells last).
    ax.computed_zorder = False

    g = np.linspace(0.0, LX, 90)
    X, Y = np.meshgrid(g, np.linspace(0.0, LY, 90))

    # ---- fault plane (painted behind everything) --------------------------
    zf = np.linspace(30.0, ZMAX, 40)
    Uf, Vf = np.meshgrid(g, zf)
    ax.plot_surface(Uf, fault_y(Uf, Vf), Vf, color="#8a8a8a", alpha=0.40,
                    linewidth=0, antialiased=True, shade=False, zorder=1)

    # ---- horizons, deepest first -----------------------------------------
    for k in range(len(UNITS) - 1, -1, -1):
        col = UNITS[k][2]
        z, hw = horizon(k, X, Y)
        for mask in (hw, ~hw):
            ax.plot_surface(X, Y, np.where(mask, np.nan, z), color=col,
                            alpha=0.92, linewidth=0.18, edgecolor="white",
                            rcount=45, ccount=45, shade=True,
                            zorder=2 + (len(UNITS) - k))

    # ---- boreholes --------------------------------------------------------
    for name, wx, wy in WELLS:
        zs = [horizon(k, np.array([[wx]]), np.array([[wy]]))[0][0, 0]
              for k in range(len(UNITS))]
        zs.append(ZMAX - 30.0)
        ax.plot([wx, wx], [wy, wy], [0.0, zs[0]], color=MUTED, lw=1.1,
                zorder=20)
        for k, col in enumerate(u[2] for u in UNITS):
            ax.plot([wx, wx], [wy, wy], [zs[k], zs[k + 1]],
                    color=darken(col, 0.22), lw=5.0,
                    solid_capstyle="butt", zorder=21)
        ax.plot([wx, wx], [wy, wy], [0.0, zs[-1]], color=INK, lw=0.5,
                zorder=22)
        ax.scatter([wx], [wy], [0.0], s=30, marker="v", c="white",
                   edgecolors=INK, linewidths=0.9, depthshade=False,
                   zorder=23)
        ax.text(wx, wy, -30.0, name, fontsize=7.2, fontweight="bold",
                color=INK, ha="center", va="bottom", zorder=24)

    ax.set_xlim(0.0, LX)
    ax.set_ylim(0.0, LY)
    ax.set_zlim(ZMAX, 0.0)
    ax.view_init(elev=20, azim=-56)
    ax.set_box_aspect((1.0, 1.0, 0.74), zoom=1.30)
    ax.set_xlabel("Easting (km)", fontsize=7.6, labelpad=-2)
    ax.set_ylabel("Northing (km)", fontsize=7.6, labelpad=-2)
    ax.set_zlabel("Depth below datum (m)", fontsize=7.6, labelpad=-1)
    ax.tick_params(labelsize=6.5, pad=0.5)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_zticks([0, 100, 200, 300, 400])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor("white")
        pane.pane.set_edgecolor("#d5d5d5")
        pane.pane.set_alpha(1.0)
        pane._axinfo["grid"]["color"] = "#e6e6e6"
        pane._axinfo["grid"]["linewidth"] = 0.5

    ax.text2D(0.5, 1.0, "Faulted block model, 4 x 4 km",
              transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
              fontweight="bold", color=INK)

    # ---- sequence key -----------------------------------------------------
    ax_k.axis("off")
    ax_k.set_xlim(0.0, 1.0)
    ax_k.set_ylim(ZMAX + 150.0, -40.0)
    tops = [u[1] for u in UNITS] + [ZMAX]
    for k, (name, _, col) in enumerate(UNITS):
        ax_k.add_patch(Rectangle((0.06, tops[k]), 0.30, tops[k + 1] - tops[k],
                                 facecolor=col, edgecolor="white",
                                 linewidth=0.9))
        ax_k.text(0.42, 0.5 * (tops[k] + tops[k + 1]), name.replace(" - ",
                  "\n"), ha="left", va="center", fontsize=6.8, color=INK,
                  linespacing=1.28)
        ax_k.text(0.045, tops[k], f"{tops[k]:.0f}", ha="right", va="center",
                  fontsize=6.2, color=MUTED)
    ax_k.plot([0.06, 0.36], [tops[0], tops[0]], color=RULE, lw=0.9)
    ax_k.text(0.045, ZMAX, f"{ZMAX:.0f}", ha="right", va="center",
              fontsize=6.2, color=MUTED)
    ax_k.text(0.0, -34.0, "Sequence key (depth, m)", ha="left", va="bottom",
              fontsize=8.5, fontweight="bold", color=INK)

    ax_k.add_patch(Rectangle((0.06, ZMAX + 34.0), 0.30, 26.0,
                             facecolor="#8a8a8a", alpha=0.45,
                             edgecolor="none"))
    ax_k.text(0.42, ZMAX + 47.0, "fault plane", ha="left", va="center",
              fontsize=6.8, color=INK)
    ax_k.plot([0.21, 0.21], [ZMAX + 78.0, ZMAX + 106.0], color=MUTED, lw=1.2)
    ax_k.plot([0.21], [ZMAX + 78.0], marker="v", ms=5.0, mfc="white",
              mec=INK, mew=0.9)
    ax_k.text(0.42, ZMAX + 94.0, "borehole", ha="left", va="center",
              fontsize=6.8, color=INK)
    ax_k.text(0.06, ZMAX + 140.0, f"normal throw {THROW:.0f} m; "
              f"{len(UNITS)} sequences, {len(WELLS)} wells", ha="left",
              va="bottom", fontsize=6.5, color=MUTED)

    savefig(fig, "407_geo_3d_model.png")


if __name__ == "__main__":
    main()

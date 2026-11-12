"""Hydrographic section: temperature, salinity and dissolved oxygen.

Three panels share the same distance axis and the same depth range, so the
thermocline, the subsurface salinity maximum and the oxygen minimum zone can be
read off against each other. Station positions are drawn once, as inverted
triangles above the top panel, and the sea bed is masked with a grey wedge that
is identical in every panel.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig

C_DEEP = "#17384f"
C_MAIN = "#2c7a8c"
C_LIGHT = "#8fbdd0"
C_ACC = "#c0603f"
C_WARM = "#d9a441"
C_BED = "#d6d3cc"

XMAX, ZMAX = 320.0, 800.0
STATIONS = np.array([5, 14, 24, 36, 50, 66, 84, 106, 134, 168,
                     204, 242, 280, 314], dtype=float)


def panel_tag(fig, ax, text, dx=-0.056, dy=0.012, size=10.5):
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, text, fontsize=size,
             fontweight="bold", ha="left", va="bottom", color="#1a1a1a")


def bed_depth(x):
    return (128.0 + 715.0 / (1.0 + np.exp(-(x - 98.0) / 12.0))
            - 300.0 * np.exp(-((x - 250.0) / 21.0) ** 2))


def displacement(x):
    """Vertical displacement of the isolines (positive = shoaling)."""
    return (72.0 * np.exp(-((x - 150.0) / 46.0) ** 2)
            - 45.0 * np.exp(-((x - 262.0) / 52.0) ** 2)
            + 70.0 * np.exp(-x / 34.0))


def fields(X, Z):
    zz = np.clip(Z + displacement(X), 0.0, None)
    t = 4.2 + 19.6 * np.exp(-zz / 175.0) + 1.4 * np.exp(-zz / 900.0)
    s = (34.58 + 0.44 * np.exp(-((zz - 145.0) / 115.0) ** 2)
         - 0.86 * np.exp(-zz / 62.0) - 0.11 * np.tanh((zz - 430.0) / 240.0)
         + 0.028 * np.sin(X / 52.0))
    o = (1.35 + 6.9 * np.exp(-zz / 56.0) + 2.25 * (zz / ZMAX) ** 1.5
         - 1.15 * np.exp(-((zz - 430.0) / 165.0) ** 2))
    return t, s, o


def ramp(name, cols):
    return mpl.colors.LinearSegmentedColormap.from_list(name, cols)


def main():
    set_pub_style()

    x = np.linspace(0, XMAX, 520)
    z = np.linspace(0, ZMAX, 400)
    X, Z = np.meshgrid(x, z)
    T, S, O = fields(X, Z)

    bed = bed_depth(X)
    below = Z > bed
    T = np.ma.masked_where(below, T)
    S = np.ma.masked_where(below, S)
    O = np.ma.masked_where(below, O)

    specs = [
        dict(data=T, label="Temperature (\u00b0C)",
             cmap=ramp("t", ["#fdf6ec", "#f6dfb8", "#eec27f", "#e0a153",
                             "#cb7a3c", "#ab4f2c"]),
             levels=np.arange(4, 25.1, 1.0), lines=np.arange(5, 25, 3),
             fmt="%.0f"),
        dict(data=S, label="Salinity (psu)",
             cmap=ramp("s", ["#f4f9fc", "#d8e9f2", "#b0d0e2", "#7fb0cb",
                             "#4d86ab", "#22496b"]),
             levels=np.arange(33.6, 35.11, 0.075),
             lines=np.arange(33.8, 35.05, 0.3), fmt="%.1f"),
        dict(data=O, label="Dissolved oxygen (mg L$^{-1}$)",
             cmap=ramp("o", ["#1e565c", "#448a8a", "#74b0ad", "#a5cfcb",
                             "#cfe6e3", "#eef6f5"]),
             levels=np.arange(0, 8.51, 0.4), lines=np.arange(1, 8.1, 2),
             fmt="%.0f"),
    ]

    fig = plt.figure(figsize=(9.3, 8.4))
    left, width = 0.082, 0.775
    heights = 0.238
    tops = [0.845, 0.560, 0.275]
    axes = []
    for k, spec in enumerate(specs):
        ax = fig.add_axes([left, tops[k] - heights, width, heights])
        cax = fig.add_axes([left + width + 0.018, tops[k] - heights, 0.0155,
                            heights])
        axes.append(ax)

        cf = ax.contourf(X, Z, spec["data"], levels=spec["levels"],
                         cmap=spec["cmap"], extend="both", zorder=2)
        cf.set_edgecolor("face")
        cs = ax.contour(X, Z, spec["data"], levels=spec["lines"],
                        colors="#33414a", linewidths=0.5, alpha=0.7, zorder=3)
        ax.clabel(cs, inline=True, fontsize=6.0, fmt=spec["fmt"],
                  inline_spacing=3)

        ax.fill_between(x, bed_depth(x), ZMAX + 40, color=C_BED,
                        edgecolor="#8f8b82", linewidth=0.8, zorder=4)
        ax.set_xlim(0, XMAX)
        ax.set_ylim(ZMAX, 0)
        ax.set_yticks([0, 200, 400, 600, 800])
        ax.set_xticks(np.arange(0, 321, 40))
        ax.set_ylabel("Depth (m)")
        ax.tick_params(length=3, labelsize=7.8)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("#333333")
            sp.set_linewidth(0.8)
        if k < 2:
            ax.tick_params(labelbottom=False)

        cb = fig.colorbar(cf, cax=cax)
        cb.set_label(spec["label"], fontsize=7.9, labelpad=3)
        cb.ax.tick_params(labelsize=7.2, width=0.7, length=2.4)
        cb.outline.set_linewidth(0.6)
        cb.outline.set_edgecolor("#333333")
        if k == 1:
            cb.set_ticks(np.arange(33.6, 35.11, 0.3))

    ax_t, ax_s, ax_o = axes
    ax_o.set_xlabel("Distance offshore (km)")

    # ---- stations, drawn once above the top panel -------------------------
    ax_t.plot(STATIONS, np.full_like(STATIONS, -10.0), ls="none", marker="v",
              ms=4.6, mfc="white", mec=C_DEEP, mew=0.9, clip_on=False,
              zorder=8)
    for i, sx in enumerate(STATIONS, start=1):
        ax_t.text(sx, -32.0, f"{i}", ha="center", va="bottom", fontsize=6.2,
                  color="#5a5a5a", clip_on=False, zorder=8)
    for ax in axes:
        for sx in STATIONS:
            ax.plot([sx, sx], [0, min(bed_depth(sx), ZMAX)], color="#ffffff",
                    lw=0.35, alpha=0.35, zorder=5)

    # ---- feature annotations ----------------------------------------------
    ax_t.annotate("seasonal thermocline", xy=(212, 105), xytext=(238, 268),
                  fontsize=7.2, color="#5b3418", ha="center", va="center",
                  zorder=7,
                  bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                            edgecolor="#e0dbd2", linewidth=0.5, alpha=0.9),
                  arrowprops=dict(arrowstyle="->", color="#7a5230", lw=0.8,
                                  connectionstyle="arc3,rad=0.2"))
    ax_t.annotate("coastal upwelling", xy=(24, 62), xytext=(78, 232),
                  fontsize=7.2, color="#5b3418", ha="center", va="center",
                  zorder=7,
                  bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                            edgecolor="#e0dbd2", linewidth=0.5, alpha=0.9),
                  arrowprops=dict(arrowstyle="->", color="#7a5230", lw=0.8,
                                  connectionstyle="arc3,rad=-0.25"))
    ax_s.annotate("subsurface salinity maximum", xy=(150, 88),
                  xytext=(196, 262), fontsize=7.2, color=C_DEEP,
                  ha="center", va="center", zorder=7,
                  bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                            edgecolor="#d8e0e6", linewidth=0.5, alpha=0.9),
                  arrowprops=dict(arrowstyle="->", color="#3f6a8a", lw=0.8,
                                  connectionstyle="arc3,rad=0.2"))
    ax_o.annotate("oxygen minimum zone", xy=(206, 396), xytext=(214, 640),
                  fontsize=7.2, color="#1e565c", ha="center", va="center",
                  zorder=7,
                  bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                            edgecolor="#d5e2e1", linewidth=0.5, alpha=0.9),
                  arrowprops=dict(arrowstyle="->", color="#2f7076", lw=0.8,
                                  connectionstyle="arc3,rad=0.2"))
    for ax, col in ((ax_t, "#7a5230"), (ax_s, "#3f6a8a"), (ax_o, "#2f7076")):
        ax.axvline(98, color=col, lw=0.7, ls=(0, (3, 2.2)), alpha=0.6,
                   zorder=6)
    ax_t.text(102, 752, "shelf break", ha="left", va="bottom", fontsize=7.0,
              color="#6b6b6b", zorder=7)
    ax_o.text(250, 742, "seamount", ha="center", va="bottom", fontsize=7.0,
              color="#6b6b6b", zorder=7)


    for a, lab in zip(axes, ("(a)", "(b)", "(c)")):
        panel_tag(fig, a, lab)
    savefig(fig, "219_ocean_section.png")


if __name__ == "__main__":
    main()

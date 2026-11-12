"""Water-table map of an alluvial aquifer with monitoring-well hydrographs.

Panel (a) contours the simulated water table, shades it with a sequential ramp
and adds unit vectors of the hydraulic gradient, which are by construction
normal to the equipotentials. A pumping well drives a cone of depression, a
recharge mound sits in the north-west and a gaining river bounds the south.
Panel (b) is a deliberately small companion card holding the hydrographs of
three of the monitoring wells.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from pubstyle import set_pub_style, savefig, style_axes

C_DEEP = "#17384f"
C_MAIN = "#2c7a8c"
C_LIGHT = "#8fbdd0"
C_ACC = "#c0603f"

XMAX, YMAX = 12.0, 9.0
PUMP = (7.6, 4.3)

WELLS = [
    ("MW-1", 1.5, 6.8, (0.0, 0.30), "center"),
    ("MW-2", 4.4, 7.3, (0.0, 0.30), "center"),
    ("MW-3", 9.6, 6.7, (0.0, 0.30), "center"),
    ("MW-4", 2.5, 4.6, (-0.30, 0.0), "right"),
    ("MW-5", 6.1, 5.6, (0.0, 0.30), "center"),
    ("MW-6", 9.9, 4.0, (0.32, 0.0), "left"),
    ("MW-7", 4.6, 1.9, (0.0, 0.30), "center"),
    ("MW-8", 8.4, 1.7, (0.0, 0.30), "center"),
]
SERIES = [("MW-2", C_MAIN, 0.85, 2.5, -0.012),
          ("MW-5", C_ACC, 0.62, 3.4, -0.055),
          ("MW-8", C_DEEP, 1.10, 1.4, -0.004)]


def panel_tag(fig, ax, text, dx=-0.048, dy=0.014, size=10.5):
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, text, fontsize=size,
             fontweight="bold", ha="left", va="bottom", color="#1a1a1a")


def head(x, y):
    """Water-table elevation (m above datum)."""
    h = 92.0 - 1.95 * x - 0.95 * y
    h += 5.6 * np.exp(-(((x - 2.8) / 2.3) ** 2 + ((y - 7.4) / 1.9) ** 2))
    h -= 10.5 * np.exp(-(((x - PUMP[0]) / 1.55) ** 2
                         + ((y - PUMP[1]) / 1.45) ** 2))
    h -= 2.4 * np.exp(-(y / 1.6) ** 2)            # gaining river to the south
    return h


def main():
    set_pub_style()
    rng = np.random.default_rng(23)

    x = np.linspace(0, XMAX, 320)
    y = np.linspace(0, YMAX, 250)
    X, Y = np.meshgrid(x, y)
    H = head(X, Y)

    fig = plt.figure(figsize=(9.5, 6.3))
    # box sized so that aspect="equal" needs no shrink: 0.640*9.5*(9/12) in
    ax = fig.add_axes([0.052, 0.122, 0.640, 0.7238])
    cax = fig.add_axes([0.706, 0.452, 0.0155, 0.394])
    ax_h = fig.add_axes([0.790, 0.150, 0.185, 0.215])
    ax.set_zorder(2)
    ax_h.set_zorder(2)
    ax_h.patch.set_alpha(0.0)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "head", ["#f4f9fb", "#dceaf2", "#bfd9e8", "#9cc2d9", "#77a8c8",
                 "#4f87ad", "#2c6389"])
    levels = np.arange(56, 94.1, 1.0)
    cf = ax.contourf(X, Y, H, levels=levels, cmap=cmap, extend="both",
                     zorder=1)
    cf.set_edgecolor("face")
    cs = ax.contour(X, Y, H, levels=np.arange(56, 94.1, 2.0),
                    colors="#2f4c5e", linewidths=0.55, alpha=0.8, zorder=3)
    ax.clabel(cs, cs.levels[::2], inline=True, fontsize=6.4, fmt="%.0f",
              inline_spacing=4)

    # ---- hydraulic gradient vectors (normal to the equipotentials) --------
    xq = np.linspace(0.55, XMAX - 0.55, 13)
    yq = np.linspace(0.55, YMAX - 0.55, 10)
    XQ, YQ = np.meshgrid(xq, yq)
    eps = 0.02
    u = -(head(XQ + eps, YQ) - head(XQ - eps, YQ)) / (2 * eps)
    v = -(head(XQ, YQ + eps) - head(XQ, YQ - eps)) / (2 * eps)
    mag = np.hypot(u, v)
    ax.quiver(XQ, YQ, u / mag, v / mag, scale=32, width=0.0032,
              headwidth=4.2, headlength=4.6, headaxislength=4.0,
              color="#1d3d52", alpha=0.72, zorder=5)

    # ---- river along the southern boundary --------------------------------
    xr = np.linspace(0, XMAX, 200)
    yr = 0.42 + 0.20 * np.sin(xr / 1.7) + 0.05 * xr
    ax.plot(xr, yr, color="#4f87ad", lw=2.6, alpha=0.85, zorder=6,
            solid_capstyle="round")
    ax.plot(xr, yr, color="#dceaf2", lw=1.0, alpha=0.9, zorder=6)
    ax.text(10.7, yr[-1] + 0.22, "River Ash", ha="right", va="bottom",
            fontsize=7.2, color="#356a8c", style="italic", zorder=8)

    # ---- wells -------------------------------------------------------------
    wx = np.array([w[1] for w in WELLS])
    wy = np.array([w[2] for w in WELLS])
    ax.plot(wx, wy, ls="none", marker="^", ms=6.0, mfc="white", mec=C_DEEP,
            mew=1.1, zorder=8, label="Monitoring well")
    for name, px, py, (dx, dy), ha in WELLS:
        col = C_ACC if name in {s[0] for s in SERIES} else "#1f3c4d"
        ax.text(px + dx, py + dy, name, ha=ha,
                va="bottom" if dy > 0 else "center", fontsize=6.6,
                color=col, zorder=9,
                bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                          edgecolor="none", alpha=0.82))
    ax.plot([PUMP[0]], [PUMP[1]], ls="none", marker="o", ms=7.2, mfc=C_ACC,
            mec="white", mew=1.1, zorder=9, label="Abstraction well")
    ax.annotate("PW-1\ncone of depression", xy=(PUMP[0] + 0.9, PUMP[1] - 0.7),
                xytext=(PUMP[0] + 2.5, PUMP[1] - 2.2), fontsize=7.2,
                color=C_ACC, ha="center", va="center", linespacing=1.4,
                zorder=10,
                bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                          edgecolor="#e4d3cc", linewidth=0.6, alpha=0.94),
                arrowprops=dict(arrowstyle="->", color=C_ACC, lw=0.8,
                                connectionstyle="arc3,rad=0.25"))
    ax.annotate("recharge mound", xy=(2.8, 7.4), xytext=(1.5, 5.9),
                fontsize=7.2, color="#2a5670", ha="center", va="center",
                zorder=10,
                bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                          edgecolor="#d3dade", linewidth=0.6, alpha=0.94),
                arrowprops=dict(arrowstyle="->", color="#3f6a8a", lw=0.8,
                                connectionstyle="arc3,rad=-0.22"))

    # ---- map furniture -----------------------------------------------------
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, 12.1, 2))
    ax.set_yticks(np.arange(0, 9.1, 2))
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.tick_params(length=3, labelsize=7.8)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#333333")
        sp.set_linewidth(0.8)

    leg_m = ax.legend(loc="upper left", bbox_to_anchor=(0.006, 0.995),
                      ncol=2, fontsize=7.0, frameon=True, facecolor="white",
                      edgecolor="#d5dade", framealpha=0.92,
                      handletextpad=0.3, columnspacing=1.1, borderpad=0.45)
    leg_m.set_zorder(12)
    leg_m.set_clip_on(False)

    ax.add_patch(Rectangle((8.90, 8.06), 3.00, 0.80, facecolor="white",
                           edgecolor="#d5dade", linewidth=0.6, alpha=0.92,
                           zorder=11))
    ax.annotate("", xy=(11.62, 8.76), xytext=(11.62, 8.18),
                arrowprops=dict(arrowstyle="-|>", color="#3a3a3a", lw=1.0,
                                mutation_scale=7), zorder=12)
    ax.text(11.62, 8.02, "N", ha="center", va="bottom", fontsize=7.0,
            fontweight="bold", color="#3a3a3a", zorder=12)
    ax.plot([9.15, 11.15], [8.34, 8.34], color="#333333", lw=1.4, zorder=12,
            solid_capstyle="butt")
    for xx in (9.15, 10.15, 11.15):
        ax.plot([xx, xx], [8.34, 8.47], color="#333333", lw=0.9, zorder=12)
    ax.text(10.15, 8.52, "2 km", ha="center", va="bottom", fontsize=6.8,
            color="#333333", zorder=12)
    ax.text(0.28, 0.16, "arrows: unit hydraulic gradient", ha="left",
            va="bottom", fontsize=6.8, color="#2c4d63", zorder=12,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.8))

    cb = fig.colorbar(cf, cax=cax, ticks=np.arange(56, 94.1, 6))
    cb.set_label("Water-table elevation (m a.s.l.)", fontsize=8.0, labelpad=4)
    cb.ax.tick_params(labelsize=7.4, width=0.7, length=2.5)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    # ---- (b) well hydrographs (subordinate card) --------------------------
    box = ax_h.get_position()
    card = FancyBboxPatch((box.x0 - 0.058, box.y0 - 0.075),
                          box.width + 0.078, box.height + 0.170,
                          transform=fig.transFigure,
                          boxstyle="round,pad=0.004,rounding_size=0.010",
                          facecolor="#f2f5f6", edgecolor="#dde3e6",
                          linewidth=0.7, zorder=-5)
    fig.patches.append(card)

    months = np.arange(0, 25)
    for name, col, amp, ph, tr in SERIES:
        seas = amp * np.sin(2 * np.pi * (months - ph) / 12.0)
        series = seas + tr * months + rng.normal(0, 0.08, months.size)
        ax_h.plot(months, series, color=col, lw=1.3, zorder=4, label=name)
    ax_h.axhline(0, color="#b6bfc4", lw=0.7, zorder=2)
    ax_h.set_xlim(0, 24)
    ax_h.set_xticks([0, 6, 12, 18, 24])
    ax_h.set_ylim(-2.6, 1.8)
    ax_h.set_yticks([-2, -1, 0, 1])
    ax_h.tick_params(labelsize=7.0, length=2.4)
    ax_h.set_xlabel("Month", fontsize=7.4, labelpad=2.0)
    ax_h.set_ylabel("Head anomaly (m)", fontsize=7.4, labelpad=2.0)
    ax_h.grid(color="#e4e8ea", lw=0.5)
    ax_h.set_axisbelow(True)
    style_axes(ax_h)
    leg = ax_h.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=3,
                      fontsize=6.6, handlelength=1.2, columnspacing=0.9,
                      handletextpad=0.4, borderpad=0.2)
    leg.set_clip_on(False)

    ax_h.text(0.0, 1.15, "Monitoring-well hydrographs",
              transform=ax_h.transAxes, fontsize=8.0, fontweight="bold",
              ha="left", va="bottom", color="#1a1a1a")
    ax_h.text(0.0, 1.035, "deviation from the mapped head",
              transform=ax_h.transAxes, fontsize=6.9, ha="left", va="bottom",
              color="#6b6b6b")


    panel_tag(fig, ax, "(a)")
    panel_tag(fig, ax_h, "(b)", dx=-0.026, dy=0.094)
    savefig(fig, "221_groundwater_contour.png")


if __name__ == "__main__":
    main()

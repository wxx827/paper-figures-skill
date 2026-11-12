"""Wave climate at an offshore buoy: rose, directional spectrum, 1-D spectrum.

Panel (a) is a wave rose - the joint frequency of significant wave height and
mean wave direction, stacked in 16 compass sectors. Panel (b) shows a two-peak
two-dimensional spectrum on a polar frequency-direction grid, with a swell peak
from the south-east and a younger wind sea from the north-east. Panel (c) is the
subordinate one-dimensional spectrum obtained by integrating (b) over direction.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pubstyle import set_pub_style, savefig, style_axes

C_DEEP = "#17384f"
C_MAIN = "#2c7a8c"
C_LIGHT = "#8fbdd0"
C_ACC = "#c0603f"

HS_BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.5), (4.5, 7.0)]
HS_COLORS = ["#dbeaf1", "#a9cddc", "#75aac4", "#43829f", "#1d4e68"]
SECTORS = 16
CALM = 3.4


def panel_tag(fig, ax, text, dx=-0.048, dy=0.020, size=10.5):
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, text, fontsize=size,
             fontweight="bold", ha="left", va="bottom", color="#1a1a1a")


def rose_table(rng):
    """Percent occurrence, sector x height bin."""
    centres = np.arange(SECTORS) * 360.0 / SECTORS
    # two modes: dominant SE swell, secondary NE wind sea
    w = (1.00 * np.exp(-0.5 * (((centres - 135) + 180) % 360 - 180) ** 2 / 34 ** 2)
         + 0.66 * np.exp(-0.5 * (((centres - 45) + 180) % 360 - 180) ** 2 / 30 ** 2)
         + 0.22)
    w = w / w.sum()
    shape = np.array([0.30, 0.33, 0.21, 0.11, 0.05])
    tab = np.outer(w, shape)
    # bigger waves are steered towards the swell sector
    steer = np.exp(-0.5 * (((centres - 130) + 180) % 360 - 180) ** 2 / 42 ** 2)
    tab[:, 3] *= 0.5 + 1.6 * steer
    tab[:, 4] *= 0.3 + 2.4 * steer
    tab *= 1.0 + 0.05 * rng.standard_normal(tab.shape)
    tab = np.clip(tab, 0.0, None)
    return centres, tab / tab.sum() * (100.0 - CALM)


def partition(f, fp, width, amp, skew=2.4):
    """Smooth, slightly skewed spectral peak (energy on the high-f tail)."""
    z = (f - fp) / width
    return amp * np.exp(-0.5 * z ** 2) * (1.0 + 0.5 * np.tanh(-z / skew))


def spread(theta, th0, s):
    d = np.cos(0.5 * (theta - np.deg2rad(th0))) ** (2 * s)
    return d


def main():
    set_pub_style()
    rng = np.random.default_rng(19)

    fig = plt.figure(figsize=(9.9, 6.2))
    ax_r = fig.add_axes([0.038, 0.205, 0.345, 0.575], projection="polar")
    ax_s = fig.add_axes([0.455, 0.345, 0.300, 0.480], projection="polar")
    ax_1 = fig.add_axes([0.480, 0.105, 0.250, 0.155])
    cax = fig.add_axes([0.800, 0.370, 0.015, 0.430])

    # ---- (a) wave rose ----------------------------------------------------
    centres, tab = rose_table(rng)
    width = np.deg2rad(360.0 / SECTORS) * 0.86
    bottom = np.zeros(SECTORS)
    r0 = 1.6                       # blank hub for the calm annotation
    for j, (lo, hi) in enumerate(HS_BINS):
        ax_r.bar(np.deg2rad(centres), tab[:, j], width=width,
                 bottom=bottom + r0, color=HS_COLORS[j], edgecolor="white",
                 linewidth=0.4, zorder=3,
                 label=f"{lo:.1f}–{hi:.1f}" if hi < 7 else f"> {lo:.1f}")
        bottom += tab[:, j]

    ax_r.set_theta_zero_location("N")
    ax_r.set_theta_direction(-1)
    ax_r.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax_r.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                         fontsize=8.0)
    ax_r.tick_params(axis="x", pad=0.5)
    rings = np.array([3, 6, 9, 12])
    ax_r.set_yticks(rings + r0)
    ax_r.set_yticklabels([f"{v:.0f}%" for v in rings], fontsize=6.8,
                         color="#5f5f5f")
    ax_r.set_rlabel_position(202)
    ax_r.set_ylim(0, 13.6 + r0)
    for t in ax_r.get_yticklabels():
        t.set_bbox(dict(boxstyle="round,pad=0.12", facecolor="white",
                        edgecolor="none", alpha=0.75))
    ax_r.grid(color="#d8dee2", lw=0.5)
    ax_r.set_axisbelow(True)
    ax_r.spines["polar"].set_color("#c8ced2")
    ax_r.spines["polar"].set_linewidth(0.7)
    ax_r.text(0, 0, f"calm\n{CALM:.1f}%", ha="center", va="center",
              fontsize=6.8, color="#6b6b6b", linespacing=1.3, zorder=5)

    handles = [Patch(facecolor=c, edgecolor="white", linewidth=0.4)
               for c in HS_COLORS]
    labels = [f"{lo:.0f}–{hi:.0f}" if lo > 0 else f"0–{hi:.0f}"
              for lo, hi in HS_BINS]
    labels[-1] = "> 4.5"
    leg = ax_r.legend(handles, labels, loc="upper center",
                      bbox_to_anchor=(0.5, -0.045), ncol=5, fontsize=7.4,
                      handlelength=1.1, handleheight=0.85,
                      columnspacing=1.0, handletextpad=0.45,
                      title="Significant wave height $H_s$ (m)")
    leg.get_title().set_fontsize(7.6)
    leg.get_title().set_fontweight("bold")
    leg.set_clip_on(False)

    # ---- (b) directional spectrum -----------------------------------------
    f = np.linspace(0.0, 0.30, 180)
    th = np.linspace(0, 2 * np.pi, 241)
    F, TH = np.meshgrid(f, th)

    e_sw = partition(f, 1 / 12.5, 0.0165, 1.00)[None, :] * spread(TH, 135, 5.0)
    e_ws = partition(f, 1 / 5.6, 0.0295, 0.62)[None, :] * spread(TH, 48, 3.0)
    E = (e_sw / np.trapezoid(spread(th, 135, 5.0), th)
         + e_ws / np.trapezoid(spread(th, 48, 3.0), th))
    s1 = np.trapezoid(E, th, axis=0)
    E *= (3.4 / 4.0) ** 2 / np.trapezoid(s1, f)     # target H_m0 = 3.4 m
    s1 = np.trapezoid(E, th, axis=0)
    vmax = float(np.ceil(E.max() / 2.0) * 2.0)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "wave", ["#ffffff", "#e5f0f5", "#c6dfea", "#9dc6d9", "#6fa7c3",
                 "#437f9f", "#22536f"])
    levels = np.linspace(0, vmax, 15)
    cf = ax_s.contourf(TH, F, E, levels=levels, cmap=cmap, zorder=2,
                       extend="max")
    cf.set_edgecolor("face")
    ax_s.contour(TH, F, E, levels=levels[3::3], colors="#ffffff",
                 linewidths=0.4, alpha=0.7, zorder=3)

    ax_s.set_theta_zero_location("N")
    ax_s.set_theta_direction(-1)
    ax_s.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax_s.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                         fontsize=7.6)
    ax_s.tick_params(axis="x", pad=0.0)
    ax_s.set_ylim(0, 0.26)
    ax_s.set_yticks([0.1, 0.2])
    ax_s.set_yticklabels(["0.1", "0.2 Hz"], fontsize=6.6, color="#4a4a4a")
    ax_s.set_rlabel_position(288)
    ax_s.grid(color="#b9c4ca", lw=0.4, alpha=0.8)
    ax_s.set_axisbelow(False)
    ax_s.spines["polar"].set_color("#c8ced2")
    ax_s.spines["polar"].set_linewidth(0.7)

    ax_s.annotate("swell\n$T_p$ = 12.5 s", xy=(np.deg2rad(135), 0.095),
                  xytext=(np.deg2rad(172), 0.285), fontsize=7.4,
                  color=C_DEEP, ha="center", va="center", linespacing=1.4,
                  zorder=6,
                  bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                            edgecolor="#d3dade", linewidth=0.5, alpha=0.92),
                  arrowprops=dict(arrowstyle="->", color="#4a6a7c", lw=0.8,
                                  connectionstyle="arc3,rad=0.2"))
    ax_s.annotate("wind sea\n$T_p$ = 5.6 s", xy=(np.deg2rad(48), 0.178),
                  xytext=(np.deg2rad(6), 0.285), fontsize=7.4,
                  color=C_DEEP, ha="center", va="center", linespacing=1.4,
                  zorder=6,
                  bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                            edgecolor="#d3dade", linewidth=0.5, alpha=0.92),
                  arrowprops=dict(arrowstyle="->", color="#4a6a7c", lw=0.8,
                                  connectionstyle="arc3,rad=-0.2"))

    cb = fig.colorbar(cf, cax=cax, ticks=np.linspace(0, vmax, 6))
    cb.set_label("Energy density  $S(f,\\theta)$  (m$^2$ Hz$^{-1}$ rad$^{-1}$)",
                 fontsize=7.8, labelpad=4)
    cb.ax.tick_params(labelsize=7.2, width=0.7, length=2.5)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    # ---- (c) 1-D spectrum (subordinate) -----------------------------------
    ax_1.fill_between(f, 0, s1, color=C_MAIN, alpha=0.20, lw=0, zorder=2)
    ax_1.plot(f, s1, color=C_DEEP, lw=1.5, zorder=4)
    for fp, lab, col in ((1 / 12.5, "swell", C_MAIN), (1 / 5.6, "wind sea",
                                                       C_ACC)):
        yv = float(np.interp(fp, f, s1))
        ax_1.plot([fp, fp], [0, yv], color=col, lw=0.8, ls=(0, (3, 2)),
                  zorder=3)
        ax_1.text(fp, yv + 0.055 * s1.max(), lab, ha="center", va="bottom",
                  fontsize=6.9, color=col)
    ax_1.set_xlim(0, 0.30)
    ax_1.set_ylim(0, 1.32 * s1.max())
    ax_1.set_xticks([0, 0.1, 0.2, 0.3])
    ax_1.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax_1.tick_params(labelsize=7.0, length=2.4)
    ax_1.set_xlabel("Frequency (Hz)", fontsize=7.6, labelpad=2.0)
    ax_1.set_ylabel("$S(f)$\n(m$^2$ Hz$^{-1}$)", fontsize=7.4,
                    linespacing=1.3, labelpad=2.0)
    ax_1.grid(axis="y", color="#ececec", lw=0.5)
    ax_1.set_axisbelow(True)
    style_axes(ax_1)

    hs = 4.0 * np.sqrt(np.trapezoid(s1, f))
    ax_1.text(0.985, 0.90, f"$H_{{m0}}$ = {hs:.1f} m", transform=ax_1.transAxes,
              ha="right", va="top", fontsize=7.2, color="#333333")


    for a in (ax_r, ax_s, ax_1):
        panel_tag(fig, a, "(%s)" % "abc"[(ax_r, ax_s, ax_1).index(a)],
                  dx=-0.032, dy=0.014)
    savefig(fig, "217_wave_rose_spectrum.png")


if __name__ == "__main__":
    main()

"""Boundary-layer structure: profiles, a time-height curtain and the column.

The curtain is the primary panel - nitrogen dioxide as a function of hour and
altitude - and the other two are readings taken out of it.  The profile panel
on the left is the curtain sliced at one hour and shares its altitude axis, so
the inversion band and the mixing-layer top drawn on both cannot disagree; the
series underneath shares the time axis and is the curtain integrated over
altitude, which is exactly what a satellite sees as a tropospheric column.
Ozone and sulphur dioxide are drawn in the same mixing-ratio unit as nitrogen
dioxide, so the three profiles legitimately share one axis.
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, get_cmap, COLORS
from figkit import panel_labels_inches, safe_colorbar

INK = COLORS["ink"]
SUB = COLORS["sub"]
MUTED = COLORS["muted"]
ACCENT = "#b23a48"

O3_C = "#3b5b92"
NO2_C = "#c0603f"
SO2_C = "#8e6bbf"
INV_C = "#d5d5d5"

Z_TOP = 2000.0          # m above ground
PROFILE_HOUR = 8.0
# molecules per cm3 per ppb near the surface, for the column integral
MOLEC_PER_PPB = 2.46e10


def rasterize(artist):
    """3.10's ContourSet mis-reports 'rasterization will be ignored'."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Rasterization of", UserWarning)
        artist.set_rasterized(True)


def mixing_height(t):
    """Mixing-layer depth in metres over the day."""
    return 220.0 + 1150.0 * np.clip(np.sin(np.pi * (t - 6.0) / 12.5),
                                    0.0, None) ** 1.3


def no2_curtain(t, z):
    """NO2 mixing ratio (ppb) on a (height, time) grid."""
    T, Z = np.meshgrid(t, z)
    h = mixing_height(T)
    src = (1.0 + 0.95 * np.exp(-0.5 * ((T - 8.0) / 1.6) ** 2)
           + 0.85 * np.exp(-0.5 * ((T - 20.0) / 2.2) ** 2))
    return 14.0 * src / (h / 600.0) ** 0.85 * \
        np.exp(-(Z / (0.75 * h)) ** 1.6) + 0.8


def main():
    set_pub_style()
    rng = np.random.default_rng(557)

    t = np.linspace(0.0, 24.0, 193)
    z = np.linspace(0.0, Z_TOP, 151)
    no2 = no2_curtain(t, z)
    no2 = no2 * rng.lognormal(0.0, 0.035, no2.shape)      # instrument noise
    h_t = mixing_height(t)

    it = int(np.argmin(np.abs(t - PROFILE_HOUR)))
    h14 = float(h_t[it])
    no2_p = no2[:, it]
    o3_p = 26.0 + 34.0 * (z / Z_TOP) ** 0.55 + \
        6.0 * np.tanh((z - h14) / 220.0)
    so2_p = 2.0 + 7.0 * np.exp(-0.5 * ((z - 380.0) / 210.0) ** 2) + \
        1.5 * np.exp(-z / 900.0)

    # Tropospheric sub-column, integrated over the very field that is plotted.
    dz_cm = (z[1] - z[0]) * 100.0
    column = np.trapezoid(no2, dx=dz_cm, axis=0) * MOLEC_PER_PPB / 1e15
    o3_du = 292.0 + 9.0 * np.sin(2.0 * np.pi * (t - 4.0) / 24.0) + \
        rng.normal(0.0, 0.8, t.size)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.58, 1.0],
                          height_ratios=[1.0, 0.58], left=0.060, right=0.905,
                          top=0.875, bottom=0.100, hspace=0.30, wspace=0.24)
    axp = fig.add_subplot(gs[0, 0])
    axc = fig.add_subplot(gs[0, 1])
    axk = fig.add_subplot(gs[1, 0])
    axs = fig.add_subplot(gs[1, 1])

    # ------------------------------------------------- (a) vertical profiles
    axp.axhspan(h14, h14 + 200.0, color=INV_C, alpha=0.75, zorder=0)
    for prof, col, name in ((o3_p, O3_C, "$O_3$"), (no2_p, NO2_C, "$NO_2$"),
                            (so2_p, SO2_C, "$SO_2$")):
        axp.plot(prof, z, color=col, lw=1.8, zorder=4)
    axp.axhline(h14, color="#333333", lw=1.0, ls=(0, (3.4, 2.0)), zorder=5)
    tag = dict(boxstyle="round,pad=0.20", facecolor="white",
               edgecolor="none", alpha=0.88)
    axp.text(0.975, h14 + 250.0, f"mixing-layer top {h14:.0f} m",
             transform=axp.get_yaxis_transform(), ha="right", va="bottom",
             fontsize=6.5, color=INK, zorder=7, bbox=tag)
    axp.text(0.975, h14 + 100.0, "inversion",
             transform=axp.get_yaxis_transform(), ha="right", va="center",
             fontsize=6.4, color=SUB, zorder=7, bbox=tag)

    # Curves are labelled where they are furthest apart, not at a common
    # height, so the three tags never stack on top of one another.
    for prof, col, name, z_lab, side in ((o3_p, O3_C, "$O_3$", 1620.0, 1),
                                         (no2_p, NO2_C, "$NO_2$", 150.0, 1),
                                         (so2_p, SO2_C, "$SO_2$", 380.0, -1)):
        x_lab = float(np.interp(z_lab, z, prof)) + 2.0 * side
        axp.text(x_lab, z_lab, name, fontsize=7.6, color=col,
                 ha="left" if side > 0 else "right", va="center", zorder=6)

    axp.set_xlim(0.0, 74.0)
    axp.set_ylim(0.0, Z_TOP)
    axp.set_xlabel("mixing ratio (ppb)", fontsize=8.0)
    axp.set_ylabel("height above ground (m)", fontsize=8.0)
    axp.set_title(f"Profiles at {PROFILE_HOUR:02.0f}:00", fontsize=9.0,
                  fontweight="bold", color=INK)
    axp.tick_params(labelsize=7.0)
    axp.grid(axis="x", color="#ececec", lw=0.55, zorder=1)

    # ------------------------------------------------ (b) time-height curtain
    # Power-spaced levels: the nocturnal surface layer is five times the bulk
    # value, and equal steps would wash the daytime structure out entirely.
    vmax = float(np.percentile(no2, 99.6))
    levels = vmax * np.linspace(0.0, 1.0, 15) ** 1.7
    cf = axc.contourf(t, z, np.clip(no2, 0.0, vmax), levels=levels,
                      cmap=get_cmap("cb_sequential_warm"), extend="max")
    rasterize(cf)
    cf.set_edgecolor("face")
    axc.plot(t, h_t, color="#1a1a1a", lw=1.3, zorder=5)
    axc.text(12.2, mixing_height(12.2) + 70.0, "mixing-layer top",
             fontsize=6.6, color="#1a1a1a", ha="center", va="bottom",
             zorder=6)
    axc.axvline(PROFILE_HOUR, color="#3a3a3a", lw=0.9, ls=(0, (2.6, 1.8)),
                zorder=6)
    axc.text(PROFILE_HOUR + 0.25, Z_TOP * 0.965, "profile hour", fontsize=6.4,
             color="#3a3a3a", ha="left", va="top", zorder=6)

    axc.set_xlim(0.0, 24.0)
    axc.set_ylim(0.0, Z_TOP)
    axc.set_xticks(np.arange(0, 25, 3))
    axc.set_yticklabels([])
    axc.set_title("$NO_2$ curtain over the diurnal cycle", fontsize=9.0,
                  fontweight="bold", color=INK)
    axc.tick_params(labelsize=7.0, labelbottom=False)
    safe_colorbar(fig, cf, axc, label="$NO_2$ mixing ratio (ppb)",
                  width=0.11, pad=0.07, labelsize=7.0,
                  ticks=[0, 2, 5, 10, 20, 30, 40])

    # ------------------------------------------------ (c) column time series
    axs.plot(t, column, color=NO2_C, lw=1.7, zorder=4)
    axs.fill_between(t, 0.0, column, color=NO2_C, alpha=0.16, linewidth=0,
                     zorder=3)
    hi = int(np.argmax(column))
    axs.plot([t[hi]], [column[hi]], marker="o", ms=4.0, mfc="white",
             mec=NO2_C, mew=1.2, zorder=6)
    axs.annotate(f"peak {column[hi]:.0f} at {t[hi]:04.1f} h",
                 xy=(t[hi], column[hi]), xytext=(t[hi] + 1.1,
                                                 column[hi] * 1.02),
                 fontsize=6.4, color=NO2_C, ha="left", va="center")
    axs.set_xlim(0.0, 24.0)
    axs.set_ylim(0.0, float(column.max()) * 1.22)
    axs.set_xticks(np.arange(0, 25, 3))
    axs.set_xlabel("hour of day (local time, h)", fontsize=8.0)
    axs.set_ylabel("$NO_2$ column\n($10^{15}$ molec cm$^{-2}$)", fontsize=7.5,
                   color=NO2_C, linespacing=1.30)
    axs.tick_params(axis="y", labelsize=6.8, colors=NO2_C, length=2.4)
    axs.tick_params(axis="x", labelsize=7.0)
    axs.spines["left"].set_color(NO2_C)
    axs.grid(axis="y", color="#ececec", lw=0.55, zorder=0)
    axs.set_title("Column amounts", fontsize=8.5, fontweight="bold",
                  color=INK)

    axo = axs.twinx()
    axo.plot(t, o3_du, color=O3_C, lw=1.3, zorder=5)
    axo.set_ylabel("total $O_3$ column (DU)", fontsize=7.5, color=O3_C)
    axo.set_ylim(o3_du.min() - 6.0, o3_du.max() + 14.0)
    axo.tick_params(axis="y", labelsize=6.8, colors=O3_C, length=2.4)
    axo.spines["top"].set_visible(False)
    axo.spines["right"].set_visible(True)
    axo.spines["right"].set_color(O3_C)

    # --------------------------------------------------------- legend block
    axk.axis("off")
    axk.legend(handles=[
        Line2D([], [], color=O3_C, lw=1.8, label="$O_3$"),
        Line2D([], [], color=NO2_C, lw=1.8, label="$NO_2$"),
        Line2D([], [], color=SO2_C, lw=1.8, label="$SO_2$"),
        Line2D([], [], color="#1a1a1a", lw=1.2, ls=(0, (3.4, 2.0)),
               label="mixing-layer top (m)"),
        Patch(facecolor=INV_C, edgecolor="none", label="inversion layer")],
        loc="upper left", bbox_to_anchor=(-0.02, 0.98), fontsize=7.0,
        frameon=False, handlelength=1.6, labelspacing=0.55)
    axk.text(-0.02, 0.02, f"curtain: {z.size} levels x {t.size} times\n"
             f"column integrated 0-{Z_TOP:.0f} m from the plotted field\n"
             f"daily range {column.min():.0f}-{column.max():.0f} "
             "$\\times10^{15}$ molec cm$^{-2}$",
             transform=axk.transAxes, fontsize=6.5, color=MUTED, ha="left",
             va="bottom", linespacing=1.45)

    swing = column.max() / column.min()
    panel_labels_inches(fig, [axp, axc, axs], dx=-0.40, dy=0.12)

    savefig(fig, "557_atm_vertical_column.png")


if __name__ == "__main__":
    main()

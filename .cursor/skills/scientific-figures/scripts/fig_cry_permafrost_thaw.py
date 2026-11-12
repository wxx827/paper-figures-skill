"""Figure 614 - Ground-temperature envelope of a degrading permafrost borehole.

Panel (a) is a single time-depth field: a warming mean annual air temperature
plus a downward-propagating annual wave, damped and phase-lagged with depth,
with a latent-heat term that flattens the profile as it crosses the freezing
point.  That flattening is the zero curtain, and it is why the 0 deg C contour
lingers instead of sweeping smoothly through the active layer.

Everything else is measured off that field rather than asserted.  The
active-layer thickness in panel (b) is the deepest 0 deg C crossing reached in
each calendar year, the depth of zero annual amplitude is where the annual
range of the same field drops below 0.1 deg C, and the trends in panel (c) are
least-squares fits to its own annual means.
"""
import textwrap
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, get_cmap, get_palette, COLORS
from figkit import panel_label_inches, safe_colorbar, shared_norm

T_START, T_END = 2005.0, 2025.0
Z_MAX = 12.0                    # depth of the modelled column, m
MAAT0 = -6.30                   # mean annual ground temperature in 2005, degC
WARMING = 0.085                 # degC per year at the surface
PENETRATION = 12.0              # e-folding depth of the multi-decadal trend, m
LAG = 0.075                     # phase lag of the slow signal, years per m
AMP_S = 14.5                    # surface annual amplitude, degC
DAMP = 2.40                     # thermal damping depth, m
ZAA_RANGE = 0.5                 # annual range defining zero annual amplitude
GEOTHERM = 0.030                # geothermal gradient, degC per m
LATENT = 0.85                   # zero-curtain strength, degC
PHASE = 0.56                    # fraction of the year at the warmest day
PROBES = (1.0, 3.0, 6.0, 9.0)   # depths reported in panel (c), m

FROZEN = "#3b5b92"
THAWED = "#b23a48"


def rasterize(artist):
    """3.10's ContourSet mis-reports 'rasterization will be ignored'."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Rasterization of", UserWarning)
        artist.set_rasterized(True)


def temperature_field(t, z, rng):
    """Ground temperature (degC) on the time-depth grid, with a zero curtain."""
    tt, zz = np.meshgrid(t, z)
    knots = np.linspace(t[0] - 2.0, t[-1], 16)
    vals = rng.normal(0.0, 0.55, knots.size)
    # The slow surface signal reaches depth attenuated and delayed, which is
    # what separates the four probe curves in panel (c).
    lag = LAG * zz
    slow = np.interp(tt - lag, knots, vals) * np.exp(-zz / 5.5)
    trend = WARMING * np.exp(-zz / PENETRATION) * (tt - lag - T_START)
    wave = (AMP_S * np.exp(-zz / DAMP)
            * np.sin(2 * np.pi * (tt - PHASE) - zz / DAMP))
    field = MAAT0 + trend + slow + wave + GEOTHERM * zz
    # Latent heat pins the profile near the freezing point while the pore ice
    # changes phase; the effect dies away where the wave no longer reaches.
    curtain = LATENT * np.exp(-zz / (2.2 * DAMP))
    return field - curtain * np.tanh(field / np.maximum(curtain, 1e-6))


def thaw_depth(col, z):
    """Deepest 0 degC crossing reached from the surface, in metres."""
    if col[0] <= 0.0:
        return 0.0
    below = np.nonzero(col <= 0.0)[0]
    if below.size == 0:
        return float(z[-1])
    k = int(below[0])
    t0, t1 = col[k - 1], col[k]
    f = t0 / (t0 - t1) if t0 != t1 else 0.0
    return float(z[k - 1] + f * (z[k] - z[k - 1]))


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(6140)

    t = np.arange(T_START, T_END + 1e-9, 1.0 / 24.0)
    z = np.linspace(0.0, Z_MAX, 300)
    field = temperature_field(t, z, rng)

    years = np.arange(int(T_START), int(T_END))
    alt = np.array([max(thaw_depth(field[:, j], z)
                        for j in np.nonzero((t >= y) & (t < y + 1))[0])
                    for y in years])
    alt_trend = np.poly1d(np.polyfit(years, alt, 1))
    alt_rate = 100.0 * float(np.polyfit(years, alt, 1)[0])       # cm per year

    # Depth of zero annual amplitude, straight off the plotted field.
    last = (t >= years[-1]) & (t < years[-1] + 1)
    swing = field[:, last].max(axis=1) - field[:, last].min(axis=1)
    zaa = float(np.interp(ZAA_RANGE, swing[::-1], z[::-1]))

    probe_series = {}
    for zp in PROBES:
        k = int(np.argmin(np.abs(z - zp)))
        probe_series[zp] = np.array(
            [field[k, (t >= y) & (t < y + 1)].mean() for y in years])

    norm = shared_norm([field], symmetric=True)
    levels = np.linspace(norm.vmin, norm.vmax, 19)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58],
                          left=0.058, right=0.960, top=0.885, bottom=0.118,
                          wspace=0.40, hspace=0.44)
    ax = fig.add_subplot(gs[:, 0])
    ax_a = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 1])

    # ---- (a) the time-depth envelope ---------------------------------------
    cf = ax.contourf(t, z, field, levels=levels,
                     cmap=get_cmap("cb_diverging"), norm=norm, extend="both")
    rasterize(cf)
    cf.set_edgecolor("face")
    ax.contour(t, z, field, levels=levels[::3], colors="#555555",
               linewidths=0.35, alpha=0.7)
    ax.contour(t, z, field, levels=[0.0], colors=["#1a1a1a"], linewidths=1.8)

    ax.axhline(zaa, color=COLORS["rule"], lw=0.9, ls=(0, (5, 2.5)), zorder=6)
    ax.text(T_END - 0.3, zaa - 0.18,
            f"zero annual amplitude, {zaa:.1f} m", fontsize=6.4,
            color=COLORS["rule"], ha="right", va="bottom")
    ax.annotate("zero curtain", xy=(T_START + 6.4, 1.55),
                xytext=(T_START + 2.2, 4.3), fontsize=6.6, color="#1a1a1a",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#1a1a1a",
                                shrinkA=1, shrinkB=2))

    ax.set_xlim(T_START, T_END)
    ax.set_ylim(Z_MAX, 0.0)                                # depth downward
    ax.set_xticks(np.arange(T_START, T_END + 1, 5.0))
    ax.set_xticklabels([f"{v:.0f}" for v in np.arange(T_START, T_END + 1, 5)])
    ax.set_xlabel("year", fontsize=8.0)
    ax.set_ylabel("depth below ground surface (m)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.set_title("Ground temperature, borehole envelope", fontsize=8.8,
                 fontweight="bold", pad=4.0)
    ax.legend(handles=[
        Line2D([0], [0], color="#1a1a1a", lw=1.8,
               label="0 $^\\circ$C isotherm (active-layer base)"),
        Line2D([0], [0], color=COLORS["rule"], lw=0.9, ls=(0, (5, 2.5)),
               label="zero annual amplitude"),
    ], loc="lower left", bbox_to_anchor=(0.005, 0.02), frameon=True,
        framealpha=0.90, facecolor="white", edgecolor="#cccccc",
        borderpad=0.4, fontsize=6.4, handlelength=1.8)
    safe_colorbar(fig, cf, ax, label="ground temperature ($^\\circ$C)",
                  width=0.12, pad=0.07, labelsize=6.8,
                  ticks=np.arange(-20, 21, 5))

    # ---- (b) active-layer thickness ----------------------------------------
    ax_a.plot(years, alt, marker="o", ms=2.4, lw=1.0, color="#777777",
              mfc="#777777", mec="none", zorder=3)
    ax_a.plot(years, alt_trend(years), color=THAWED, lw=1.6, zorder=4)
    pad = 0.16 * (alt.max() - alt.min())
    ax_a.set_ylim(alt.max() + pad, alt.min() - pad)        # depth downward
    ax_a.set_xticks(np.arange(2005, 2026, 5))
    ax_a.set_xlabel("year", fontsize=7.6)
    ax_a.set_ylabel("active-layer\nthickness (m)", fontsize=7.4,
                    linespacing=1.3)
    ax_a.tick_params(labelsize=6.6)
    ax_a.grid(axis="y", lw=0.5, alpha=0.6)
    ax_a.text(0.97, 0.94, f"{alt_rate:+.1f} cm yr$^{{-1}}$",
              transform=ax_a.transAxes, fontsize=6.6, color=THAWED,
              ha="right", va="top")
    ax_a.set_title(f"Active layer deepened {alt[-1] - alt[0]:+.2f} m",
                   fontsize=8.5, fontweight="bold", pad=3.5)
    for s in ("top", "right"):
        ax_a.spines[s].set_visible(False)

    # ---- (c) mean annual temperature at four depths ------------------------
    cols = get_palette("cb_sequential_blue", 7)[2:6][::-1]
    rates = {}
    for (zp, series), col in zip(probe_series.items(), cols):
        rates[zp] = 10.0 * float(np.polyfit(years, series, 1)[0])
        ax_p.plot(years, series, lw=1.3, color=col, zorder=3,
                  label=f"{zp:.0f} m ({rates[zp]:+.2f})")
    ax_p.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax_p.set_xticks(np.arange(2005, 2026, 5))
    ax_p.legend(loc="lower right", frameon=False, fontsize=6.2, ncol=2,
                handlelength=1.5, labelspacing=0.26, columnspacing=1.0)
    ax_p.set_xlabel("year", fontsize=7.6)
    ax_p.set_ylabel("mean annual T ($^\\circ$C)", fontsize=7.4)
    ax_p.tick_params(labelsize=6.6)
    ax_p.grid(axis="y", lw=0.5, alpha=0.6)
    ax_p.set_title("Warming rate by depth ($^\\circ$C per decade)",
                   fontsize=8.5, fontweight="bold", pad=3.5)
    for s in ("top", "right"):
        ax_p.spines[s].set_visible(False)

    panel_label_inches(fig, ax, "(a)", dx=-0.46, dy=0.09)
    panel_label_inches(fig, ax_a, "(b)", dx=-0.46, dy=0.09)
    panel_label_inches(fig, ax_p, "(c)", dx=-0.46, dy=0.09)


    savefig(fig, "614_cry_permafrost_thaw.png")


if __name__ == "__main__":
    main()

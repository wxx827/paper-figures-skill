"""Single-turbine wake: plan-view velocity field, recovery and lateral cuts.

The main panel is a plan view of the horizontal velocity at hub height behind
one 126 m rotor, drawn on an equal-aspect axes so the wake really does open
at the modelled expansion rate.  A Gaussian far-wake model supplies the
deficit, a smoothed random field adds turbulent meandering, and the dashed
cone marks the two-sigma wake width.  Every deficit percentage printed at a
downstream station is measured back off the field that is plotted.

The companion panels resolve the same field two ways: recovery of the
centreline speed with downstream distance, and lateral cuts at four
stations.  Both share the speed scale of the colour bar, so a shallow cut
cannot be mistaken for a deep one.

Canvas aspect is below the house band by exception E1: the main panel is
locked to ``set_aspect("equal")`` and a taller canvas would only add margin.
"""
import textwrap

import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, get_cmap, get_palette
from figkit import axes_inches, panel_labels_inches, halo_text

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
GRID = "#d9d9d9"

BLUE = "#3b5b92"
RED = "#b23a48"
GOLD = "#c9a227"

DIAM = 126.0
U_INF = 9.0          # free-stream speed at hub height (m/s)
CT = 0.80            # thrust coefficient (-)
KSTAR = 0.052        # wake growth rate (-)
STATIONS = np.array([2.0, 4.0, 6.0, 8.0, 10.0])     # x / D
CUTS = np.array([2.0, 4.0, 6.0, 8.0])               # x / D for lateral cuts


def sigma_over_d(xd):
    """Gaussian wake width sigma / D at downstream distance x / D."""
    beta = 0.5 * (1.0 + np.sqrt(1.0 - CT)) / np.sqrt(1.0 - CT)
    eps = 0.2 * np.sqrt(beta)
    return KSTAR * xd + eps


def deficit(xd, yd):
    """Bastankhah-Porte-Agel deficit, held flat through the near wake."""
    xd_eff = np.maximum(xd, 1.6)
    s = sigma_over_d(xd_eff)
    core = 1.0 - np.sqrt(np.maximum(1.0 - CT / (8.0 * s ** 2), 1e-6))
    return np.where(xd >= -0.05, core * np.exp(-0.5 * (yd / s) ** 2), 0.0)


def main():
    set_pub_style()
    rng = np.random.default_rng(716)

    nx, ny = 620, 320
    xd = np.linspace(-1.0, 11.0, nx)
    yd = np.linspace(-3.0, 3.0, ny)
    XD, YD = np.meshgrid(xd, yd)

    field = U_INF * (1.0 - deficit(XD, YD))
    # turbulent meandering: smoothed noise, strongest inside the shear layer
    noise = gaussian_filter(rng.normal(0.0, 1.0, field.shape), (7.0, 11.0))
    noise /= noise.std()
    weight = 0.30 + 1.10 * np.exp(-0.5 * ((np.abs(YD) - 0.55) / 0.55) ** 2)
    field = field + 0.11 * noise * weight * np.clip(XD + 0.6, 0.0, 3.0) / 3.0

    ext = [xd[0] * DIAM, xd[-1] * DIAM, yd[0] * DIAM, yd[-1] * DIAM]
    vmin, vmax = float(field.min()), float(field.max())

    # measured back off the plotted field
    ix = [int(np.argmin(np.abs(xd - s))) for s in STATIONS]
    u_cl = field[np.argmin(np.abs(yd)), :]
    core = np.abs(yd) <= 1.5                    # look for the deficit in the wake
    def_pct = [100.0 * (1.0 - field[core, i].min() / U_INF) for i in ix]
    rec = u_cl / U_INF
    far = np.where((xd > 2.0) & (rec >= 0.90))[0]
    x_rec90 = float(xd[far[0]]) if far.size else float("nan")

    fig = plt.figure(figsize=(10.4, 5.5))
    ax = axes_inches(fig, (0.88, 1.15, 5.30, 2.65))
    ax1 = axes_inches(fig, (6.75, 3.25, 3.05, 1.42))
    ax2 = axes_inches(fig, (6.75, 1.15, 3.05, 1.42))

    # ---- main panel: plan view of the wake -------------------------------
    im = ax.imshow(field, origin="lower", extent=ext, aspect="equal",
                   cmap=get_cmap("cb_sequential_blue"), vmin=vmin, vmax=vmax,
                   interpolation="bilinear", zorder=1)
    cs = ax.contour(XD * DIAM, YD * DIAM, field, levels=[6.0, 7.0, 8.0, 8.6],
                    colors="#333333", linewidths=0.55, alpha=0.85, zorder=3)
    ax.clabel(cs, fmt="%.1f", fontsize=6.0, inline=True, inline_spacing=3)

    xs = np.linspace(0.0, 11.0, 200)
    r_edge = (0.5 + KSTAR * xs) * DIAM           # cone anchored on the tips
    for sgn in (+1, -1):
        ax.plot(xs * DIAM, sgn * r_edge, color=RED, lw=1.1, ls=(0, (4, 2)),
                zorder=4)
    halo_text(ax, 8.9 * DIAM, (0.5 + KSTAR * 8.9) * DIAM + 26.0,
              "wake edge, $r = D/2 + k^* x$", fontsize=6.6, color="white",
              halo=RED, ha="center", va="bottom", zorder=7)

    ax.plot([0.0, 0.0], [-DIAM / 2, DIAM / 2], color=INK, lw=3.0,
            solid_capstyle="butt", zorder=6)
    halo_text(ax, 0.10 * DIAM, 0.62 * DIAM, "rotor, D = 126 m", fontsize=6.6,
              ha="left", va="bottom", zorder=7)

    for s, i, d in zip(STATIONS, ix, def_pct):
        ax.plot([s * DIAM, s * DIAM], [-2.88 * DIAM, -2.52 * DIAM],
                color="white", lw=1.0, zorder=6)
        halo_text(ax, s * DIAM, -2.46 * DIAM, f"{s:.0f}D\n{d:.0f}%",
                  fontsize=6.5, ha="center", va="bottom", linespacing=1.2,
                  zorder=7)
    halo_text(ax, -0.90 * DIAM, -2.70 * DIAM, "peak deficit", fontsize=6.5,
              ha="left", va="center", zorder=7)

    ax.set_xlabel("Downstream distance $x$ (m)", fontsize=8.0)
    ax.set_ylabel("Lateral distance $y$ (m)", fontsize=8.0)
    ax.set_title("Plan view of hub-height wind speed behind one rotor",
                 fontsize=9.0, fontweight="bold")
    ax.tick_params(labelsize=7.2)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
        sp.set_color("#8a8a8a")

    # The bar is a figure-level axes, not ``ax.inset_axes``: a child axes above
    # the panel drags the panel title up with it (matplotlib raises the title
    # clear of every child), which detaches the title from its panel.
    cax = axes_inches(fig, (1.72, 4.34, 3.62, 0.13))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.5, width=0.7, length=2.5, color="#333333")
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("top")
    cb.set_label("wind speed $u$ (m/s)", fontsize=6.5, color=INK)

    # ---- aux 1: centreline recovery ---------------------------------------
    ax1.axhline(U_INF, color=GREY, lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax1.text(-0.85, U_INF - 0.18, f"free stream {U_INF:.1f} m/s", fontsize=6.5,
             color=SUB, ha="left", va="top")
    dn = xd >= 0.0
    ax1.plot(xd[dn], u_cl[dn], color=BLUE, lw=1.6, zorder=4)
    ax1.plot(STATIONS, [u_cl[i] for i in ix], marker="o", ls="none", ms=3.6,
             mfc="white", mec=INK, mew=0.8, zorder=5)
    ax1.axvline(x_rec90, color=RED, lw=0.9, ls=(0, (3, 2)), zorder=3)
    ax1.text(x_rec90 + 0.2, 0.06, f"90% recovery\nat {x_rec90:.1f}D",
             transform=ax1.get_xaxis_transform(), fontsize=6.5, color=RED,
             ha="left", va="bottom", linespacing=1.25)
    ax1.set_xlim(-1.0, 11.0)
    ax1.set_ylim(vmin, vmax)
    ax1.set_xlabel("Downstream distance $x/D$ (-)", fontsize=7.2)
    ax1.set_ylabel("Centreline $u$ (m/s)", fontsize=7.2)
    ax1.set_title("Centreline speed recovery", fontsize=8.5,
                  fontweight="bold")
    ax1.tick_params(labelsize=6.5)
    ax1.grid(True, color=GRID, lw=0.5, alpha=0.8)
    ax1.set_axisbelow(True)

    # ---- aux 2: lateral cuts ----------------------------------------------
    ramp = get_palette("cb_sequential_blue", 7)[2:6][::-1]
    for c, col in zip(CUTS, ramp):
        j = int(np.argmin(np.abs(xd - c)))
        ax2.plot(yd * DIAM, field[:, j], color=col, lw=1.4, zorder=4,
                 label=f"{c:.0f}D")
    ax2.axhline(U_INF, color=GREY, lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax2.axvline(0.0, color=GREY, lw=0.7, ls=(0, (2, 2)), alpha=0.8, zorder=2)
    ax2.set_xlim(yd[0] * DIAM, yd[-1] * DIAM)
    ax2.set_ylim(vmin, vmax)
    ax2.set_xlabel("Lateral distance $y$ (m)", fontsize=7.2)
    ax2.set_ylabel("Wind speed $u$ (m/s)", fontsize=7.2)
    ax2.set_title("Lateral cuts at four stations", fontsize=8.5,
                  fontweight="bold")
    ax2.tick_params(labelsize=6.5)
    ax2.grid(True, color=GRID, lw=0.5, alpha=0.8)
    ax2.set_axisbelow(True)
    ax2.legend(loc="lower right", frameon=False, fontsize=6.5, ncol=4,
               handlelength=1.2, columnspacing=0.9, handletextpad=0.4,
               title="station $x/D$ (-)")
    ax2.get_legend().get_title().set_fontsize(6.5)

    panel_labels_inches(fig, [ax, ax1, ax2], dx=-0.50, dy=0.10)


    savefig(fig, "716_wnd_wake_field.png")


if __name__ == "__main__":
    main()

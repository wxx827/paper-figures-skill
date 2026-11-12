"""Melt-pool thermal field of a moving heat source in directed energy deposition.

The main pair of panels is the claim: a Rosenthal-style temperature field for a
point source travelling over a semi-infinite substrate, seen from above and in
longitudinal section, with the liquidus isotherm picked out as the melt-pool
boundary so the elongated trailing tail is unmistakable.  Every annotated
number - pool length, width, depth, peak temperature, the cooling rate at the
solidification front - is measured from the same field that is contoured, so a
change of process parameters moves the drawing and the labels together.  The
two smaller panels carry the supporting evidence along the scan centreline.
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import axes_inches, halo_text, panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RULE = "#333333"
ACCENT = "#b23a48"      # melt-pool boundary: the single emphasis object
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GREY = "#808080"

WARM_CMAP = get_cmap("cb_sequential_warm")

# Absorbed-power Rosenthal solution with a distributed-source radius, so the
# field stays finite at the origin instead of diverging like 1/R.
T0 = 25.0            # deg C, substrate preheat
Q_ABS = 300.0        # W, absorbed beam power
K_TH = 0.025         # W / (mm K)
ALPHA = 5.3          # mm^2 / s
V_SCAN = 40.0        # mm / s
R_REG = 0.30         # mm
T_LIQ = 1400.0       # deg C
T_SOL = 1320.0       # deg C

XI0, XI1 = -2.2, 0.6      # mm, moving frame (source at xi = 0)
Y0, Y1 = -0.78, 0.78      # mm
Z0, Z1 = -0.68, 0.0       # mm, depth below the surface


def rasterize(artist):
    """3.10 mis-reports contour rasterisation as ignored; it is not."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Rasterization of", UserWarning)
        artist.set_rasterized(True)


def temperature(xi, y, z):
    r = np.sqrt(xi ** 2 + y ** 2 + z ** 2 + R_REG ** 2)
    return T0 + Q_ABS / (2.0 * np.pi * K_TH * r) * np.exp(
        -V_SCAN * (r + xi) / (2.0 * ALPHA))


def fields(n=421):
    xi = np.linspace(XI0, XI1, n)
    y = np.linspace(Y0, Y1, 241)
    z = np.linspace(Z0, Z1, 161)
    XT, YT = np.meshgrid(xi, y, indexing="xy")
    XS, ZS = np.meshgrid(xi, z, indexing="xy")
    top = temperature(XT, YT, 0.0)
    side = temperature(XS, 0.0, ZS)
    return xi, y, z, top, side


def pool_metrics(xi, y, z, top, side):
    """Length, width, depth and peak, all read off the plotted fields."""
    mt = top >= T_LIQ
    ms = side >= T_LIQ
    length = float(np.ptp(xi[np.any(mt, axis=0)]))
    width = 2.0 * float(np.abs(y[np.any(mt, axis=1)]).max())
    depth = float(np.abs(z[np.any(ms, axis=1)]).max())
    # widest station, used to place the width dimension line
    col = mt.sum(axis=0)
    xi_wide = xi[int(np.argmax(col))]
    xi_lo = xi[np.any(mt, axis=0)].min()
    xi_hi = xi[np.any(mt, axis=0)].max()
    peak = float(top.max())
    return dict(length=length, width=width, depth=depth, peak=peak,
                xi_wide=xi_wide, xi_lo=xi_lo, xi_hi=xi_hi)


def centreline(n=3001):
    xi = np.linspace(XI0, XI1, n)
    t = temperature(xi, 0.0, 0.0)
    # A material point sees T(xi = x - v t) with xi falling as time advances,
    # so dT/dt = -v dT/dxi; the tail of the track is where that is negative.
    cool = V_SCAN * np.gradient(t, xi)
    return xi, t, cool


def draw_field(ax, xa, yb, temp, levels, norm, vertical=False):
    cf = ax.contourf(xa, yb, temp, levels=levels, cmap=WARM_CMAP, norm=norm,
                     extend="max")
    rasterize(cf)
    cf.set_edgecolor("face")
    thin = [t for t in (500, 800, 1100, 2000, 2500) if t < temp.max()]
    cs = ax.contour(xa, yb, temp, levels=thin, colors=["#8a8a8a"],
                    linewidths=0.55, alpha=0.9)
    ax.clabel(cs, fmt="%d", fontsize=6.0, colors=["#5a5a5a"], inline_spacing=6)
    ax.contour(xa, yb, temp, levels=[T_SOL], colors=[ACCENT], linewidths=0.7,
               linestyles=[(0, (2.6, 1.8))])
    ax.contour(xa, yb, temp, levels=[T_LIQ], colors=[ACCENT], linewidths=1.9)
    ax.set_aspect("equal")
    ax.set_xlim(XI0, XI1)
    ax.set_ylim(*( (Z0, Z1) if vertical else (Y0, Y1) ))
    ax.tick_params(labelsize=6.8)
    return cf


def top_panel(ax, xi, y, top, levels, norm, m):
    cf = draw_field(ax, xi, y, top, levels, norm)
    ax.plot([0.0], [0.0], marker="v", ms=5.0, color=INK, mec="white",
            mew=0.7, zorder=8, clip_on=False)
    ya = Y1 - 0.13
    ax.annotate("", xy=(XI1 - 0.05, ya), xytext=(XI1 - 0.50, ya),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
    halo_text(ax, XI1 - 0.55, ya, f"beam, v = {V_SCAN:.0f} mm/s", ha="right",
              va="center", fontsize=6.6, color=INK, halo="white", halo_lw=2.0)

    yl = Y0 + 0.13
    ax.annotate("", xy=(m["xi_hi"], yl), xytext=(m["xi_lo"], yl),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
    halo_text(ax, 0.5 * (m["xi_lo"] + m["xi_hi"]), yl + 0.04,
              f"pool length L = {m['length']:.2f} mm", ha="center",
              va="bottom", fontsize=6.8, color=INK, halo="white", halo_lw=2.0)
    xw = m["xi_wide"]
    ax.annotate("", xy=(xw, m["width"] / 2.0), xytext=(xw, -m["width"] / 2.0),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
    halo_text(ax, xw - 0.06, 0.0, f"W = {m['width']:.2f} mm", ha="right",
              va="center", fontsize=6.8, color=INK, halo="white", halo_lw=2.0)
    halo_text(ax, XI0 + 0.06, Y1 - 0.07,
              "$T_{peak}$ = " + f"{m['peak']:.0f} " + r"$^{\circ}$C",
              ha="left", va="top", fontsize=7.0, color=INK, halo="white",
              halo_lw=2.2)

    ax.set_ylabel("transverse y (mm)", fontsize=7.6)
    ax.set_xticklabels([])
    ax.set_title("Trailing melt pool at the liquidus isotherm",
                 fontsize=9.0, fontweight="bold", color=INK)
    return cf


def side_panel(ax, xi, z, side, levels, norm, m):
    draw_field(ax, xi, z, side, levels, norm, vertical=True)
    ax.axhline(0.0, color=RULE, lw=1.0)
    ax.annotate("", xy=(m["xi_wide"], 0.0), xytext=(m["xi_wide"], -m["depth"]),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
    halo_text(ax, m["xi_wide"] - 0.06, -m["depth"] / 2.0,
              f"D = {m['depth']:.2f} mm", ha="right", va="center",
              fontsize=6.8, color=INK, halo="white", halo_lw=2.0)
    halo_text(ax, XI1 - 0.04, Z0 + 0.04,
              f"aspect D/W = {m['depth'] / m['width']:.2f}", ha="right",
              va="bottom", fontsize=6.6, color=SUB, halo="white", halo_lw=2.0)
    ax.set_xlabel(r"scan-direction coordinate $\xi = x - v t$ (mm)",
                  fontsize=7.6)
    ax.set_ylabel("depth z (mm)", fontsize=7.6)
    ax.set_title("Longitudinal section on the centreline", fontsize=8.2,
                 fontweight="bold", color=INK)


def profile_panel(ax, xi, t, m):
    ax.plot(xi, t, color=BLUE, lw=1.6, zorder=4)
    inside = t >= T_LIQ
    ax.fill_between(xi, T0, t, where=inside, color=ACCENT, alpha=0.16,
                    linewidth=0.0, zorder=2)
    ax.axhline(T_LIQ, color=ACCENT, lw=1.0)
    ax.axhline(T_SOL, color=ACCENT, lw=0.8, ls=(0, (2.6, 1.8)))
    ax.text(XI0 + 0.06, T_LIQ + 80.0, r"$T_{liq}$ = " + f"{T_LIQ:.0f}" +
            r" $^{\circ}$C", fontsize=6.5, color=ACCENT, va="bottom")
    ax.text(XI0 + 0.06, T_SOL - 80.0, r"$T_{sol}$ = " + f"{T_SOL:.0f}" +
            r" $^{\circ}$C", fontsize=6.5, color=ACCENT, va="top")
    ax.annotate(f"molten span {m['length']:.2f} mm",
                xy=(m["xi_lo"] + 0.30, T_LIQ + 300.0),
                xytext=(XI0 + 0.10, m["peak"] * 0.86),
                fontsize=6.8, color=INK, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.set_xlim(XI0, XI1)
    ax.set_ylim(0.0, m["peak"] * 1.14)
    ax.set_xlabel(r"$\xi$ (mm)", fontsize=7.4)
    ax.set_ylabel(r"temperature ($^{\circ}$C)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title("Centreline temperature", fontsize=8.2, fontweight="bold",
                 color=INK)


def sci_label(v):
    e = int(np.floor(np.log10(abs(v))))
    return f"{v / 10.0 ** e:.1f}" + r" $\times$ 10$^{" + str(e) + r"}$ K/s"


def rate_panel(ax, xi, t, cool, m):
    decade = np.floor(np.log10(cool.max()))
    lo, hi = 10.0 ** (decade - 2.0), 10.0 ** (decade + 1.0)

    xi_peak = xi[int(np.argmax(t))]
    tail = (xi <= xi_peak) & (cool > lo)
    front = (xi > xi_peak) & (-cool > lo)
    ax.semilogy(xi[tail], cool[tail], color=GREEN, lw=1.6, zorder=4)
    ax.semilogy(xi[front], -cool[front], color=GREY, lw=1.1,
                ls=(0, (3.2, 2.0)), zorder=3)

    sel = xi < xi_peak
    i_sol = int(np.argmin(np.abs(t[sel] - T_SOL)))
    xi_sol = float(xi[sel][i_sol])
    r_sol = float(cool[sel][i_sol])
    ax.plot([xi_sol], [r_sol], marker="o", ms=4.2, mfc="white", mec=ACCENT,
            mew=1.2, zorder=6)
    ax.annotate("at the solidification front\n" + sci_label(r_sol),
                xy=(xi_sol, r_sol), xytext=(XI0 + 0.08, lo * 2.2),
                fontsize=6.6, color=INK, ha="left", linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.text(0.03, 0.97, "solid: cooling behind the beam\n"
                        "dashed: heating ahead of it",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.3,
            color=SUB, linespacing=1.35)
    ax.set_xlim(XI0, XI1)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$\xi$ (mm)", fontsize=7.4)
    ax.set_ylabel(r"$|dT/dt|$ (K/s)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title("Cooling rate along the track", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    xi, y, z, top, side = fields()
    m = pool_metrics(xi, y, z, top, side)
    xc, tc, rc = centreline()

    levels = np.linspace(T0, m["peak"], 26)
    norm = plt.Normalize(vmin=T0, vmax=m["peak"])

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_top = axes_inches(fig, (0.62, 2.72, 5.05, 2.81))
    ax_side = axes_inches(fig, (0.62, 1.15, 5.05, 1.23))
    ax_prof = axes_inches(fig, (7.20, 3.78, 2.70, 1.75))
    ax_rate = axes_inches(fig, (7.20, 1.15, 2.70, 1.75))

    cf = top_panel(ax_top, xi, y, top, levels, norm, m)
    side_panel(ax_side, xi, z, side, levels, norm, m)
    profile_panel(ax_prof, xc, tc, m)
    rate_panel(ax_rate, xc, tc, rc, m)

    safe_colorbar(fig, cf, ax_top, label=r"temperature ($^{\circ}$C)",
                  width=0.11, pad=0.08, labelsize=6.6,
                  ticks=np.arange(500.0, m["peak"], 500.0))

    panel_labels_inches(fig, [ax_top, ax_side, ax_prof, ax_rate], dx=-0.46,
                        dy=0.10)
    savefig(fig, "744_am_meltpool_thermal.png")


if __name__ == "__main__":
    main()

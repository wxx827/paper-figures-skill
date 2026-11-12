"""Rayleigh-wave dispersion along four paths, and the depths each period sees.

The dispersion curves are derived, not drawn.  For every period the phase
velocity is the depth average of that path's shear-velocity profile under a
Rayleigh sensitivity kernel, scaled by the half-space factor 0.92, and the
kernel depth is itself proportional to the wavelength, so phase velocity and
kernel are iterated to consistency.  Group velocity then follows from the
finite-difference form U = c / (1 + (T/c) dc/dT), which is why the shield path
separates from the rift path exactly where the kernels start sampling the
mantle lid.  The Airy minimum quoted in the lower panel is the argmin of that
computed group-velocity curve, not a marked point.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
DEEP = "#22406e"
BLUE = "#3b5b92"
PALE = "#a8bcd8"
RED = "#b23a48"
GOLD = "#c9a227"
TEAL = "#2a7f7a"
GREY = "#8a8a8a"

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.02, 4.66, 4.46)
AUX_T = (7.42, 3.61, 2.70, 1.87)
AUX_B = (7.42, 1.02, 2.70, 1.82)

Z_MAX, DZ = 900.0, 1.0        # km
KERNEL_FRAC = 0.30            # kernel depth scale as a fraction of wavelength
RAY_FACTOR = 0.92             # half-space Rayleigh / shear velocity ratio
PERIODS = np.linspace(10.0, 150.0, 141)      # s
MEAS_T = np.array([12.0, 16.0, 22.0, 30.0, 40.0, 55.0, 70.0, 90.0, 110.0,
                   135.0])
MEAS_SD = 0.035               # km/s, quoted measurement error
KERNEL_SHOW = [20.0, 60.0, 120.0]            # s
AIRY_PATH = "Archean shield"

PATHS = [
    dict(name="Archean shield", color=DEEP, marker="o",
         nodes=[(0, 3.55), (42, 3.75), (44, 4.72), (200, 4.76), (260, 4.58),
                (400, 4.72), (900, 5.30)]),
    dict(name="Stable platform", color=TEAL, marker="s",
         nodes=[(0, 3.45), (36, 3.70), (38, 4.58), (110, 4.60), (190, 4.42),
                (400, 4.68), (900, 5.28)]),
    dict(name="Young ocean basin", color=BLUE, marker="^",
         nodes=[(0, 3.20), (8, 3.60), (10, 4.42), (45, 4.44), (110, 4.16),
                (200, 4.35), (400, 4.66), (900, 5.26)]),
    dict(name="Continental rift", color=RED, marker="D",
         nodes=[(0, 3.10), (30, 3.55), (32, 4.28), (80, 4.10), (160, 4.14),
                (300, 4.50), (900, 5.22)]),
]


def profile(nodes, z):
    n = np.asarray(nodes, dtype=float)
    return np.interp(z, n[:, 0], n[:, 1])


def kernel(z, zc):
    """Rayleigh depth sensitivity: rises from zero, peaks at ``zc``, decays."""
    return (z / zc) * np.exp(-z / zc)


def phase_velocity(vs, z, period, n_iter=8):
    """Kernel-weighted depth average of Vs, iterated to self-consistency."""
    c = 4.0
    for _ in range(n_iter):
        k = kernel(z, KERNEL_FRAC * c * period)
        c = RAY_FACTOR * np.trapezoid(vs * k, z) / np.trapezoid(k, z)
    return float(c)


def dispersion(vs, z, periods):
    c = np.array([phase_velocity(vs, z, t) for t in periods])
    dcdt = np.gradient(c, periods)
    return c, c / (1.0 + periods / c * dcdt)


def peak_depth(c, period):
    return KERNEL_FRAC * c * period


def main_panel(ax, paths, rng):
    for p in paths:
        ax.plot(PERIODS, p["u"], color=p["color"], lw=1.7, zorder=6)
        ax.plot(PERIODS, p["c"], color=p["color"], lw=0.9,
                ls=(0, (3.6, 2.2)), alpha=0.75, zorder=5)
        ax.errorbar(MEAS_T, p["meas"], yerr=MEAS_SD, fmt=p["marker"],
                    ms=3.6, mfc="white", mec=p["color"], mew=0.9,
                    ecolor=p["color"], elinewidth=0.7, capsize=1.6,
                    capthick=0.7, zorder=7)

    ax.set_xscale("log")
    ax.set_xlim(9.5, 158.0)
    ax.set_ylim(3.05, 4.55)
    ax.set_xticks([10, 15, 20, 30, 40, 60, 80, 100, 150])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlabel("period (s, log scale)")
    ax.set_ylabel("velocity (km/s)")
    ax.tick_params(labelsize=6.8)
    ax.grid(lw=0.45, alpha=0.32)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Fundamental-mode Rayleigh dispersion for four paths",
                 fontsize=9.0, fontweight="bold", color=INK, pad=5.0)

    for p in paths:
        ax.text(160.0, p["u"][-1], p["name"], fontsize=6.4, color=p["color"],
                ha="left", va="center", fontweight="bold", clip_on=False,
                zorder=8)

    shield = paths[0]
    rift = paths[3]
    i = int(np.argmax(np.abs(shield["u"] - rift["u"])))
    t0 = PERIODS[i]
    ax.annotate("shield and rift differ most at {:.0f} s,\n"
                "which samples about {:.0f} km depth"
                .format(t0, peak_depth(shield["c"][i], t0)),
                xy=(t0, 0.5 * (shield["u"][i] + rift["u"][i])),
                xytext=(11.5, 4.42), fontsize=6.4, color=INK, ha="left",
                va="top", linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7,
                                shrinkA=2, shrinkB=3), zorder=9)

    handles = [Line2D([], [], color=SUB, lw=1.7, label="group velocity U"),
               Line2D([], [], color=SUB, lw=0.9, ls=(0, (3.6, 2.2)),
                      label="phase velocity c"),
               Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white",
                      mec=SUB, mew=0.9,
                      label="measured U, +/- {:.3f} km/s".format(MEAS_SD))]
    ax.legend(handles=handles, loc="lower right", fontsize=6.4, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.8, labelspacing=0.28, borderaxespad=0.4)


def profile_panel(ax, paths, z):
    for p in paths:
        ax.plot(p["vs"], z, color=p["color"], lw=1.3, zorder=6,
                label=p["name"])
    ax.set_xlim(2.9, 5.1)
    ax.set_ylim(360.0, 0.0)
    ax.set_xlabel("shear velocity Vs (km/s)", fontsize=6.9)
    ax.set_ylabel("depth (km, positive down)", fontsize=6.9)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.45, alpha=0.32)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.4, labelspacing=0.22, borderaxespad=0.28)
    ax.set_title("The four shear-velocity profiles", fontsize=8.0,
                 fontweight="bold", color=INK, pad=3.0)


def airy_panel(ax, path, z):
    ax.plot(PERIODS, path["c"], color=path["color"], lw=1.5,
            ls=(0, (3.6, 2.2)), zorder=6, label="phase velocity c")
    ax.plot(PERIODS, path["u"], color=path["color"], lw=1.7, zorder=7,
            label="group velocity U")
    k = int(np.argmin(path["u"]))
    ax.plot([PERIODS[k]], [path["u"][k]], marker="v", ms=5.5, mfc=GOLD,
            mec=path["color"], mew=0.7, zorder=9)
    ax.annotate("Airy minimum: U = {:.2f} km/s at {:.0f} s,\n"
                "peak sensitivity near {:.0f} km depth"
                .format(path["u"][k], PERIODS[k],
                        peak_depth(path["c"][k], PERIODS[k])),
                xy=(PERIODS[k], path["u"][k]), xytext=(30.0, 3.35),
                fontsize=6.1, color=INK, ha="left", va="center",
                linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7,
                                shrinkA=2, shrinkB=4), zorder=9)
    ax.fill_between(PERIODS, path["u"], path["c"], facecolor=PALE,
                    alpha=0.40, edgecolor="none", zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(9.5, 158.0)
    ax.set_ylim(3.20, 4.45)
    ax.set_xticks([10, 20, 40, 80, 150])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlabel("period (s, log scale)", fontsize=6.9)
    ax.set_ylabel("velocity (km/s)", fontsize=6.9)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.45, alpha=0.32)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.5, labelspacing=0.22, borderaxespad=0.28)
    ax.set_title("Group lags phase: {:s}".format(path["name"]),
                 fontsize=8.0, fontweight="bold", color=INK, pad=3.0)


def audit(fig, foot, axes):
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for ax in axes:
        ab = ax.get_window_extent()
        for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            tb = t.get_window_extent()
            if tb.width <= 0.0 or not t.get_text():
                continue
            if tb.x1 < ab.x0 - 2.0 or tb.x0 > ab.x1 + 2.0:
                continue
            if fb.overlaps(tb):
                print("OVERLAP with", repr(t.get_text()))


def main():
    set_pub_style()
    rng = np.random.default_rng(30871456)

    z = np.arange(0.0, Z_MAX + DZ, DZ)
    paths = []
    for spec in PATHS:
        vs = profile(spec["nodes"], z)
        c, u = dispersion(vs, z, PERIODS)
        meas = np.interp(MEAS_T, PERIODS, u) + rng.normal(0.0, MEAS_SD,
                                                          MEAS_T.size)
        paths.append(dict(spec, vs=vs, c=c, u=u, meas=meas))

    shield = next(p for p in paths if p["name"] == AIRY_PATH)
    rift = paths[3]

    assert np.all(shield["c"] > shield["u"]), "group velocity must lag phase"
    assert shield["u"].max() > rift["u"].max(), "shield should be the fastest"
    assert 3.0 < shield["u"].min() < 4.2, "Airy minimum off the plotted range"
    assert np.all(np.abs(np.diff(shield["c"])) < 0.06), "phase curve is ragged"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_prof = axes_inches(fig, AUX_T)
    ax_airy = axes_inches(fig, AUX_B)

    main_panel(ax_main, paths, rng)
    profile_panel(ax_prof, paths, z)
    airy_panel(ax_airy, shield, z)

    spread = float(np.max(shield["u"] - rift["u"]))


    panel_labels_inches(fig, [ax_main, ax_prof, ax_airy], dx=-0.60, dy=0.12)
    savefig(fig, "1055_eqs_surface_wave_dispersion.png")


if __name__ == "__main__":
    main()

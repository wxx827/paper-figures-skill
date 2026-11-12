"""Morison wave loading on a monopile, split into drag and inertia.

The main panel puts the sea state and the structure in the same frame and
hangs the load distribution off the pile axis on a second, top-mounted scale:
drag, inertia and their sum, all evaluated at the wave phase that maximises
base shear, which the script searches for rather than assumes.  That search is
the point of the figure - for a six-metre pile the peak arrives at the zero
up-crossing, not under the crest.  The two smaller panels carry the design
consequences: how base shear grows with period and wave height, and how long
the fatigue budget lasts at three sites.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, annotate_box
from figkit import panel_labels_inches

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
BLUE = "#3b5b92"        # drag component, water
GREEN = "#2e7d5b"       # inertia component
GOLD = "#c9a227"        # third sea state
RED = "#b23a48"         # peak load and the exceeded fatigue budget
WATER = "#dbe7f1"
SAND = "#e3ddd1"
STEEL = "#c4c4c4"

RHO = 1025.0            # sea water density, kg/m3
G = 9.81
DEPTH = 30.0            # water depth, m
DIAM = 2.0              # jacket leg outer diameter, m
CD, CM = 1.05, 2.0      # Morison coefficients
HS, TP = 10.0, 9.0      # design sea state: significant height m, period s
DESIGN_LIFE = 25.0      # years
SN_M = 3.0              # SN curve slope
FATIGUE_TARGET = 0.85   # damage consumed by the design site at design life


def wave_number(t, d=DEPTH):
    w = 2.0 * np.pi / t
    k = w ** 2 / G
    for _ in range(60):
        k = w ** 2 / (G * np.tanh(k * d))
    return k


def morison(z, phase, h=HS, t=TP, d=DEPTH):
    """Drag and inertia load per unit length in kN/m at a given wave phase."""
    k = wave_number(t, d)
    w = 2.0 * np.pi / t
    prof = np.cosh(k * (z + d)) / np.sinh(k * d)
    u = 0.5 * h * w * prof * np.cos(phase)
    a = 0.5 * h * w ** 2 * prof * np.sin(phase)
    fd = 0.5 * RHO * CD * DIAM * u * np.abs(u) / 1.0e3
    fi = CM * RHO * (np.pi * DIAM ** 2 / 4.0) * a / 1.0e3
    return fd, fi


def base_shear(phase, h=HS, t=TP, d=DEPTH, n=400):
    eta = 0.5 * h * np.cos(phase)
    z = np.linspace(-d, min(eta, 0.0), n)
    fd, fi = morison(z, phase, h, t, d)
    f = fd + fi
    return np.trapezoid(f, z), np.trapezoid(f * (z + d), z)


def worst_phase(h=HS, t=TP, n=361):
    ph = np.linspace(0.0, 2.0 * np.pi, n)
    shear = np.array([base_shear(p, h, t)[0] for p in ph])
    return float(ph[np.argmax(np.abs(shear))])


def response_surface(periods, heights):
    return {h: np.array([abs(base_shear(worst_phase(h, t), h, t)[0])
                         for t in periods]) for h in heights}


def fatigue(years, sites, ref="design site"):
    """Miner damage per year from a two-sea-state scatter at each site.

    The SN intercept is not a physical constant here: it is back-solved so the
    reference site consumes ``FATIGUE_TARGET`` of its budget at the design
    life, which keeps the three slopes in their correct ratio while putting
    the crossing points inside the plotted window.
    """
    rate = {}
    for name, states in sites.items():
        acc = 0.0
        for h, t, prob in states:
            shear = abs(base_shear(worst_phase(h, t), h, t)[0])
            cycles = prob * 365.25 * 24.0 * 3600.0 / t
            acc += cycles * shear ** SN_M
        rate[name] = acc
    scale = FATIGUE_TARGET / (rate[ref] * DESIGN_LIFE)
    return {name: r * scale * years for name, r in rate.items()}


def main_panel(ax, phase):
    k = wave_number(TP)
    lam = 2.0 * np.pi / k
    x = np.linspace(-0.90 * lam, 0.90 * lam, 600)
    eta = 0.5 * HS * np.cos(k * x - phase)

    ax.fill_between(x, -DEPTH, eta, color=WATER, edgecolor="none", zorder=1)
    ax.plot(x, eta, color=BLUE, lw=1.3, zorder=3)
    ax.fill_between(x, -DEPTH - 6.0, -DEPTH, color=SAND, edgecolor="none",
                    zorder=1)
    ax.plot(x, np.full_like(x, -DEPTH), color="#a89f8c", lw=0.9, zorder=2)
    ax.axhline(0.0, color=GREY, lw=0.7, ls=(0, (4, 2.6)), zorder=2)

    ax.fill_betweenx([-DEPTH - 6.0, 12.0], -DIAM / 2.0, DIAM / 2.0,
                     color=STEEL, edgecolor="#555555", linewidth=0.8,
                     zorder=4)
    ax.text(0.0, 13.0, f"jacket leg D = {DIAM:.1f} m", ha="center",
            va="bottom", fontsize=7.0, color=SUB)
    ax.text(-0.86 * lam, -DEPTH + 1.4, f"water depth {DEPTH:.0f} m",
            ha="left", va="bottom", fontsize=6.8, color=SUB)
    ax.text(-0.86 * lam, 1.0, f"still water level, "
            r"$H_s$ = " f"{HS:.0f} m, " r"$T_p$ = " f"{TP:.0f} s",
            ha="left", va="bottom", fontsize=6.8, color=SUB)

    ax.set_xlim(-0.90 * lam, 0.90 * lam)
    ax.set_ylim(-DEPTH - 6.0, 17.0)
    ax.set_xlabel("horizontal distance from pile (m)")
    ax.set_ylabel("elevation relative to still water (m)")
    return k, lam


def load_overlay(ax, phase, lam):
    eta0 = 0.5 * HS * np.cos(phase)
    z = np.linspace(-DEPTH, min(eta0, 0.0), 400)
    fd, fi = morison(z, phase)
    tot = fd + fi
    scale = (0.72 * lam) / np.abs(tot).max()            # m per (kN/m)

    ax2 = ax.twiny()
    lo, hi = ax.get_xlim()
    ax2.set_xlim(lo / scale, hi / scale)
    ax2.set_ylim(ax.get_ylim())
    ax2.spines["top"].set_visible(True)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_color(RULE)
    ax2.tick_params(axis="x", colors=RULE, labelsize=7.0, length=3)
    ax2.set_xlabel("Morison load per unit length (kN/m)", fontsize=8.0,
                   color=RULE)
    ax2.set_xticks([0, 5, 10, 15, 20, 25])

    ax2.plot(fd, z, color=BLUE, lw=1.3, ls=(0, (5, 2.2)), zorder=6,
             label="drag  " r"$\frac{1}{2}\rho C_D D u|u|$")
    ax2.plot(fi, z, color=GREEN, lw=1.3, ls=(0, (1.8, 1.8)), zorder=6,
             label="inertia  " r"$C_M \rho \frac{\pi D^2}{4} \dot{u}$")
    ax2.plot(tot, z, color=RULE, lw=2.0, zorder=7, label="total")
    ax2.fill_betweenx(z, 0.0, tot, color=RULE, alpha=0.08, linewidth=0.0,
                      zorder=5)

    k = int(np.argmax(tot))
    ax2.plot([tot[k]], [z[k]], marker="o", ms=4.2, mfc="white", mec=RED,
             mew=1.3, zorder=8)
    ax2.annotate(f"peak {tot[k]:.1f} kN/m at z = {z[k]:.1f} m",
                 xy=(tot[k], z[k]), xytext=(tot[k] * 0.42, z[k] + 9.5),
                 ha="left", va="center", fontsize=7.0, color=RED,
                 arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
    ax2.legend(fontsize=7.0, frameon=False, loc="center right",
               handlelength=2.0, borderaxespad=0.6)

    shear, moment = base_shear(phase)
    annotate_box(ax, f"base shear {abs(shear):.0f} kN\n"
                     f"overturning {abs(moment) / 1e3:.1f} MN"
                     r"$\cdot$m" "\n"
                     f"lever arm {abs(moment / shear):.1f} m above seabed\n"
                     f"inertia carries "
                     f"{np.trapezoid(fi, z) / np.trapezoid(tot, z) * 100:.0f}"
                     f" % of the shear",
                 loc="lower right", fontsize=6.8)
    return tot, z


def response_panel(ax, periods, curves):
    for (h, y), color in zip(curves.items(), [BLUE, GREEN, RED]):
        ax.plot(periods, y, color=color, lw=1.5, zorder=4,
                label=r"$H_s$ = " f"{h:.0f} m")
    y_des = curves[HS][int(np.argmin(np.abs(periods - TP)))]
    ax.plot([TP], [y_des], marker="o", ms=4.0, mfc="white", mec=RED, mew=1.2,
            zorder=6)
    ax.annotate(f"design point\n{y_des:.0f} kN at {TP:.0f} s",
                xy=(TP, y_des), xytext=(TP + 1.6, y_des * 1.30), ha="left",
                va="center", fontsize=6.5, color=RED, linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    ax.set_xlabel(r"peak period $T_p$ (s)", fontsize=7.4)
    ax.set_ylabel("peak base shear (kN)", fontsize=7.4)
    ax.set_xlim(periods[0], periods[-1])
    ax.set_ylim(0.0, max(v.max() for v in curves.values()) * 1.42)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left", ncol=3,
              handlelength=1.5, borderaxespad=0.2)
    ax.set_title("Load response across the sea-state scatter", fontsize=8.2,
                 fontweight="bold", color=INK)


def fatigue_panel(ax, years, damage):
    ax.axhspan(1.0, 2.4, facecolor="#f5f5f5", edgecolor="none", zorder=0)
    ax.axhline(1.0, color=RED, lw=1.0, ls=(0, (4, 2.4)), zorder=3)
    ax.text(years[0] + 0.4, 1.05, "Miner limit D = 1", ha="left", va="bottom",
            fontsize=6.5, color=RED)
    ax.axvline(DESIGN_LIFE, color=GREY, lw=0.9, ls=(0, (2.4, 2.0)), zorder=3)
    ax.text(DESIGN_LIFE - 0.6, 2.2, f"design life {DESIGN_LIFE:.0f} years",
            ha="right", va="top", fontsize=6.5, color=SUB)

    for (name, d), color in zip(damage.items(), [BLUE, GREEN, RED]):
        ax.plot(years, d, color=color, lw=1.5, zorder=4, label=name)
        if d[-1] > 1.0:
            yr = float(np.interp(1.0, d, years))
            ax.plot([yr], [1.0], marker="o", ms=3.6, mfc="white", mec=color,
                    mew=1.1, zorder=6)
            ax.text(yr, 0.90, f"{yr:.0f} yr", ha="center", va="top",
                    fontsize=6.2, color=color)
    ax.set_xlabel("service time (years)", fontsize=7.4)
    ax.set_ylabel("cumulative damage (-)", fontsize=7.4)
    ax.set_xlim(years[0], years[-1])
    ax.set_ylim(0.0, 2.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", handlelength=1.5,
              borderaxespad=0.2)
    ax.set_title("Fatigue budget under Miner accumulation", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    phase = worst_phase()
    periods = np.linspace(6.0, 18.0, 40)
    curves = response_surface(periods, [5.0, 7.5, HS])
    years = np.linspace(0.0, 40.0, 200)
    sites = {
        "sheltered site": [(1.5, 6.0, 0.72), (3.5, 8.5, 0.28)],
        "design site": [(2.0, 6.5, 0.65), (4.5, 9.0, 0.35)],
        "exposed site": [(2.5, 7.0, 0.60), (5.5, 9.5, 0.40)],
    }
    damage = fatigue(years, sites)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], left=0.070, right=0.985,
                          top=0.825, bottom=0.125, wspace=0.30, hspace=0.62)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_res = fig.add_subplot(gs[0, 1])
    ax_fat = fig.add_subplot(gs[1, 1])

    _, lam = main_panel(ax_main, phase)
    tot, z = load_overlay(ax_main, phase, lam)
    response_panel(ax_res, periods, curves)
    fatigue_panel(ax_fat, years, damage)

    bb = ax_main.get_position()
    fw, fh = fig.get_size_inches()
    xs = np.ptp(ax_main.get_xlim()) / (bb.width * fw)
    ys = np.ptp(ax_main.get_ylim()) / (bb.height * fh)
    ax_main.text(0.985, 0.985, f"vertical exaggeration {xs / ys:.1f}x",
                 transform=ax_main.transAxes, ha="right", va="top",
                 fontsize=6.8, color=MUTED)
    ax_main.set_title("Sea state, structure and Morison load in one frame",
                      fontsize=9.0, fontweight="bold", color=INK, pad=24.0)

    shear, _ = base_shear(phase)
    panel_labels_inches(fig, [ax_main, ax_res, ax_fat], dx=-0.44, dy=0.12)
    savefig(fig, "657_oce_wave_loading.png")


if __name__ == "__main__":
    main()

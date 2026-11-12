"""Single-screw extrusion: axial temperature, pressure and shear-rate profiles.

The channel depth profile is the one thing defined by hand; the shear rate is
then computed from it as pi D N / h, the melt temperature is integrated with a
viscous-heating term proportional to that shear rate, and the melt overheat,
the peak pressure and its position are read back off the plotted arrays.  The
screw outline above the curves is drawn from the same depth profile with a
pitch that shortens along the machine, so the schematic and the curves cannot
drift apart.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
HAIR = "#cccccc"
WARM = "#c0603f"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
RED = "#b23a48"
DEG = chr(176)

D_MM = 45.0             # screw diameter
N_RPM = 60.0            # screw speed for the axial profiles
LD = np.linspace(0.0, 24.0, 1400)
ZONES = [(0.0, 8.0, "feed"), (8.0, 16.0, "compression"),
         (16.0, 22.0, "metering"), (22.0, 24.0, "die")]
SET_T = [180.0, 210.0, 235.0, 245.0]


def depth_profile():
    """Channel depth in mm along the screw."""
    h = np.full_like(LD, 9.0)
    ramp = (LD > 8.0) & (LD <= 16.0)
    h[ramp] = 9.0 + (3.0 - 9.0) * (LD[ramp] - 8.0) / 8.0
    h[(LD > 16.0) & (LD <= 22.0)] = 3.0
    h[LD > 22.0] = 1.5
    return h


def barrel_setpoint():
    t = np.empty_like(LD)
    for (lo, hi, _), val in zip(ZONES, SET_T):
        t[(LD >= lo) & (LD <= hi)] = val
    return t


def melt_temperature(gamma, tset, k=0.40, c=0.078):
    """Relax toward the barrel setpoint while shear heating adds to it."""
    t = np.empty_like(LD)
    t[0] = 40.0
    dx = LD[1] - LD[0]
    for i in range(1, LD.size):
        t[i] = t[i - 1] + dx * (k * (tset[i - 1] - t[i - 1]) +
                                c * gamma[i - 1])
    return t


def pressure_profile(pmax=32.0):
    ramp = 1.0 / (1.0 + np.exp(-(LD - 13.5) / 2.6))
    p = pmax * (ramp - ramp[0]) / (ramp[-1] - ramp[0])
    tip = LD > 22.0
    p_tip = float(np.interp(22.0, LD, p))
    p[tip] = p_tip * (1.0 - (LD[tip] - 22.0) / 2.0) ** 1.4
    return p


def draw_screw(ax, h_mm):
    root = 0.5 - (h_mm / D_MM)
    ax.fill_between(LD, -root, root, facecolor="#d5d5d5", edgecolor="#8a8a8a",
                    linewidth=0.7, zorder=3)
    for y in (-0.5, 0.5):
        ax.plot([0.0, 24.0], [y, y], color=SUB, lw=1.2, zorder=4)
    x, pitch = 0.55, 1.15
    while x < 22.0:
        ax.plot([x, x + 0.30], [-0.5, 0.5], color="#8a8a8a", lw=1.5,
                solid_capstyle="round", zorder=2)
        pitch = 1.15 - 0.45 * min(x / 20.0, 1.0)
        x += pitch
    ax.add_patch(Polygon([[1.1, 0.5], [3.4, 0.5], [3.0, 1.15], [1.5, 1.15]],
                         closed=True, facecolor="#f0f0f0", edgecolor=SUB,
                         linewidth=0.8, zorder=4))
    ax.text(2.25, 1.22, "hopper", ha="center", va="bottom", fontsize=6.3,
            color=SUB)
    ax.add_patch(Polygon([[22.0, 0.5], [24.0, 0.34], [24.0, -0.34],
                          [22.0, -0.5]], closed=True, facecolor="#e8e8e8",
                         edgecolor=SUB, linewidth=0.8, zorder=4))
    ax.text(19.0, -0.68, "flight pitch shortens toward the die", ha="center",
            va="top", fontsize=6.3, color=MUTED)


def zone_bands(ax, label=False):
    for i, (lo, hi, name) in enumerate(ZONES):
        ax.axvspan(lo, hi, facecolor="#f5f5f5" if i % 2 else "#ebebeb",
                   edgecolor="none", zorder=0)
        if label:
            ax.text(0.5 * (lo + hi), 1.02, name,
                    transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=6.6, color=SUB)


def main():
    set_pub_style()
    rng = np.random.default_rng(839)

    h_mm = depth_profile()
    gamma = np.pi * D_MM * N_RPM / (60.0 * h_mm)
    tset = barrel_setpoint()
    tmelt = melt_temperature(gamma, tset)
    tmelt += 0.35 * np.convolve(rng.standard_normal(LD.size),
                                np.ones(45) / 45.0, mode="same")
    press = pressure_profile()

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_s = axes_inches(fig, (0.86, 4.42, 5.44, 0.74))
    ax_t = axes_inches(fig, (0.86, 3.24, 5.44, 1.04))
    ax_p = axes_inches(fig, (0.86, 2.14, 5.44, 1.04))
    ax_g = axes_inches(fig, (0.86, 1.04, 5.44, 1.04))
    ax_q = axes_inches(fig, (7.52, 3.30, 2.48, 1.88))
    ax_r = axes_inches(fig, (7.52, 0.98, 2.48, 1.70))

    draw_screw(ax_s, h_mm)
    zone_bands(ax_s, label=True)
    ax_s.set_xlim(0.0, 24.0)
    ax_s.set_ylim(-0.95, 1.30)
    ax_s.axis("off")
    ax_s.set_title("Single-screw extruder, D = %.0f mm, %g rpm"
                   % (D_MM, N_RPM), fontsize=9.0, fontweight="bold",
                   color=INK, pad=13.0)

    over = float(np.max(tmelt - tset))
    i_over = int(np.argmax(tmelt - tset))
    ax_t.plot(LD, tset, color=GREY, lw=1.0, ls=(0, (3.4, 2.0)),
              label="barrel set point")
    ax_t.plot(LD, tmelt, color=WARM, lw=1.7, label="melt temperature")
    ax_t.annotate("melt runs %.1f " % over + DEG + "C above the set point",
                  (LD[i_over], tmelt[i_over]),
                  xytext=(LD[i_over] - 11.0, tmelt[i_over] - 96.0),
                  fontsize=6.8, color=WARM, ha="left",
                  arrowprops=dict(arrowstyle="-|>", color=WARM, lw=0.8,
                                  shrinkA=1.0, shrinkB=2.0))
    ax_t.set_ylabel("temperature\n(" + DEG + "C)", fontsize=7.6)
    ax_t.set_ylim(20.0, 300.0)
    ax_t.set_yticks([50.0, 150.0, 250.0])
    ax_t.legend(fontsize=6.4, frameon=False, loc="lower right",
                bbox_to_anchor=(1.0, -0.04), handlelength=1.7, ncol=2,
                columnspacing=1.2)

    i_max = int(np.argmax(press))
    ax_p.plot(LD, press, color=BLUE, lw=1.7)
    ax_p.fill_between(LD, 0.0, press, color=BLUE, alpha=0.10)
    ax_p.plot(LD[i_max], press[i_max], marker="o", ms=5.0, mfc="white",
              mec=BLUE, mew=1.4, zorder=6)
    ax_p.annotate("peak %.1f MPa at L/D = %.1f" % (press[i_max], LD[i_max]),
                  (LD[i_max], press[i_max]),
                  xytext=(LD[i_max] - 10.6, press[i_max] * 0.52),
                  fontsize=6.8, color=BLUE, ha="left",
                  arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.8,
                                  shrinkA=1.0, shrinkB=2.0))
    ax_p.set_ylabel("pressure\n(MPa)", fontsize=7.6)
    ax_p.set_ylim(-2.0, 40.0)
    ax_p.set_yticks([0.0, 15.0, 30.0])

    ax_g.plot(LD, gamma, color=GREEN, lw=1.7)
    h_met = float(h_mm[int(np.argmin(np.abs(LD - 18.0)))])
    ax_g.annotate("channel depth %.0f mm in the feed, %.0f mm in metering"
                  % (h_mm[0], h_met),
                  (16.0, gamma[np.argmin(np.abs(LD - 16.0))]),
                  xytext=(4.6, 78.0), fontsize=6.8, color=GREEN, ha="left",
                  arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=0.8,
                                  shrinkA=1.0, shrinkB=2.0))
    ax_g.set_ylabel("shear rate\n($s^{-1}$)", fontsize=7.6)
    ax_g.set_ylim(0.0, 118.0)
    ax_g.set_yticks([0.0, 40.0, 80.0])
    ax_g.set_xlabel("axial position along the screw (L/D, -)", fontsize=8.0)

    for a in (ax_t, ax_p, ax_g):
        zone_bands(a)
        a.set_xlim(0.0, 24.0)
        a.tick_params(labelsize=6.8)
        for lo, _, _ in ZONES[1:]:
            a.axvline(lo, color=HAIR, lw=0.7, zorder=1)
    for a in (ax_t, ax_p):
        a.tick_params(axis="x", labelbottom=False)

    n = np.linspace(20.0, 140.0, 60)
    dps = [10.0, 20.0, 30.0, 40.0]
    blues = get_cmap("cb_sequential_blue")
    for i, dp in enumerate(dps):
        q = np.maximum(0.62 * n - 0.85 * dp, 0.0)
        ax_q.plot(n, q, color=blues(0.35 + 0.19 * i), lw=1.5, zorder=3)
        ax_q.text(n[-1] - 1.0, q[-1] + 1.4, "%.0f MPa" % dp, fontsize=6.3,
                  ha="right", va="bottom", color=blues(0.35 + 0.19 * i))
    q_ref = 0.62 * N_RPM - 0.85 * 20.0
    ax_q.plot(N_RPM, q_ref, marker="o", ms=5.2, color=RED, mec="white",
              mew=0.7, zorder=5)
    ax_q.annotate("operating point\n%g rpm, %.1f kg/h" % (N_RPM, q_ref),
                  (N_RPM, q_ref), textcoords="offset points",
                  xytext=(-7.0, -3.0), fontsize=6.3, color=RED, va="top",
                  ha="right", linespacing=1.5)
    ax_q.set_xlim(n[0], n[-1] + 4.0)
    ax_q.set_ylim(0.0, 92.0)
    ax_q.set_xlabel("screw speed (rpm)", fontsize=7.4)
    ax_q.set_ylabel("output (kg/h)", fontsize=7.4)
    ax_q.tick_params(labelsize=6.5)
    ax_q.set_title("Output falls as head pressure rises", fontsize=8.5,
                   fontweight="bold", color=INK)

    t = np.linspace(0.0, 260.0, 900)
    t0, s = 46.0, 0.52
    e = np.zeros_like(t)
    m = t > t0
    e[m] = np.exp(-0.5 * (np.log((t[m] - t0) / 34.0) / s) ** 2) / (t[m] - t0)
    e /= np.trapezoid(e, t)
    mean = float(np.trapezoid(t * e, t))
    tp = float(t[int(np.argmax(e))])
    ax_r.fill_between(t, 0.0, e, color=GREEN, alpha=0.16, zorder=2)
    ax_r.plot(t, e, color=GREEN, lw=1.6, zorder=4)
    for val, col, lab in [(t0, GREY, "$t_{min}$ = %.0f s" % t0),
                          (tp, GREEN, "mode %.0f s" % tp),
                          (mean, RED, "mean %.0f s" % mean)]:
        ax_r.axvline(val, color=col, lw=0.9, ls=(0, (3.0, 2.0)), zorder=3)
    ax_r.text(t0 + 4.0, e.max() * 0.96, "$t_{min}$ = %.0f s" % t0,
              rotation=90, fontsize=6.3, color=SUB, va="top", ha="left")
    ax_r.text(mean + 5.0, e.max() * 0.62,
              "mean %.0f s\nmode %.0f s" % (mean, tp), fontsize=6.3,
              color=INK, va="top", ha="left", linespacing=1.5)
    ax_r.set_xlim(0.0, 260.0)
    ax_r.set_ylim(0.0, e.max() * 1.18)
    ax_r.set_xlabel("residence time (s)", fontsize=7.4)
    ax_r.set_ylabel("$E(t)$ (s$^{-1}$)", fontsize=7.4)
    ax_r.tick_params(labelsize=6.5)
    ax_r.set_title("Residence time distribution", fontsize=8.5,
                   fontweight="bold", color=INK)


    panel_labels_inches(fig, [ax_s], labels=["(a)"], dx=-0.62, dy=0.16)
    panel_labels_inches(fig, [ax_q], labels=["(b)"], dx=-0.54, dy=0.14)
    panel_labels_inches(fig, [ax_r], labels=["(c)"], dx=-0.54, dy=0.14)
    savefig(fig, "839_pol_extrusion_process.png")


if __name__ == "__main__":
    main()

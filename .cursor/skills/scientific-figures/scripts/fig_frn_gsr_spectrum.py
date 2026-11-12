"""Gunshot residue: SEM-EDX spectra, Pb-Sb-Ba ternary and particle sizes.

The three spectra are Gaussian line sets on a bremsstrahlung continuum, and the
peak labels are anchored to the height the summed spectrum actually reaches at
that energy, so a label can never float above or below its own peak.  The
ternary panel classifies particles by the same 12 % minimum-fraction rule that
shades its decision region, and the counts quoted in the caption are taken from
that classification.

Synthetic illustration for a visual reference library; not a case record.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

ORANGE = "#c0603f"          # GSR particle
BLUE = "#3b5b92"            # environmental particle
GREY = "#808080"            # instrument background
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

FW, FH = 10.4, 6.4
MAIN = (0.82, 0.88, 5.08, 4.46)
AUX_T = (6.88, 3.59, 1.97, 1.75)     # equal aspect ternary
AUX_B = (6.88, 0.88, 2.96, 1.75)

RES = 0.058                 # detector resolution, keV (Gaussian sigma)
E0 = 20.0                   # accelerating voltage, keV
KEY = [("Pb", 2.342), ("Sb", 3.604), ("Ba", 4.466)]
T_MIN = 0.12                # minimum mass fraction for a GSR-consistent grain

GSR = [("C K", 0.277, 900.0), ("O K", 0.525, 2600.0),
       ("Na K", 1.041, 700.0), ("Si K", 1.740, 1500.0),
       ("S K", 2.307, 3400.0), ("Pb M", 2.342, 9000.0),
       ("Cl K", 2.622, 600.0), ("Sb L", 3.604, 7200.0),
       ("Sb L2", 3.843, 1800.0), ("Ba L", 4.466, 8200.0),
       ("Ba L2", 4.828, 2100.0), ("Fe K", 6.400, 320.0),
       ("Cu K", 8.040, 180.0), ("Pb L", 10.551, 900.0)]
ENV = [("C K", 0.277, 700.0), ("O K", 0.525, 4200.0),
       ("Na K", 1.041, 400.0), ("Mg K", 1.253, 900.0),
       ("Al K", 1.487, 2600.0), ("Si K", 1.740, 7800.0),
       ("K K", 3.312, 700.0), ("Ca K", 3.691, 2400.0),
       ("Ti K", 4.508, 600.0), ("Fe K", 6.400, 3100.0),
       ("Fe Kb", 7.058, 500.0)]

# label text, energy used for placement
GSR_TAGS = [("C 0.28", 0.277), ("O 0.53", 0.525), ("Na 1.04", 1.041),
            ("Si 1.74", 1.740), ("S K / Pb M 2.33", 2.342),
            ("Sb L 3.60", 3.604), ("Ba L 4.47", 4.466),
            ("Fe 6.40", 6.400), ("Pb L 10.55", 10.551)]
ENV_TAGS = [("Si 1.74", 1.740), ("Ca 3.69", 3.691), ("Fe 6.40", 6.400)]


def spectrum(energy, lines, cont, rng):
    y = cont * np.clip(E0 - energy, 0.0, None) / np.clip(energy, 0.06, None)
    for _lab, e, amp in lines:
        y = y + amp * np.exp(-0.5 * ((energy - e) / RES) ** 2)
    return np.clip(y * np.exp(rng.normal(0.0, 0.055, energy.size)), 0.6, None)


def tag(ax, energy, spec, tags, scale, color, up=True):
    for text, e in tags:
        j = int(np.argmin(np.abs(energy - e)))
        y = spec[j] * scale
        ax.text(e, y * (1.55 if up else 0.40), text, rotation=90,
                ha="center", va="bottom" if up else "top", fontsize=6.0,
                color=color, zorder=6)


def draw_spectra(ax, energy, s_gsr, s_env, s_bg, scales):
    band = lighten(ORANGE, 0.86)
    for sym, e in KEY:
        ax.axvspan(e - 0.20, e + 0.20, facecolor=band, edgecolor="none",
                   zorder=0)
        ax.text(e, 0.985, sym, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.4, fontweight="bold",
                color=ORANGE, zorder=6)
    ax.axvspan(10.551 - 0.24, 10.551 + 0.24, facecolor=band,
               edgecolor="none", zorder=0)

    ax.plot(energy, s_bg * scales[2], color=GREY, lw=0.9, zorder=3,
            label="instrument background")
    ax.plot(energy, s_env * scales[1], color=BLUE, lw=1.0, zorder=4,
            label="environmental particle (brake dust)")
    ax.plot(energy, s_gsr * scales[0], color=ORANGE, lw=1.1, zorder=5,
            label="GSR-consistent particle")

    tag(ax, energy, s_gsr, GSR_TAGS, scales[0], INK, up=True)
    tag(ax, energy, s_env, ENV_TAGS, scales[1], BLUE, up=False)

    ax.set_yscale("log")
    ax.set_xlim(0.0, 12.0)
    ax.set_ylim(0.35, 1.4e5)
    ax.set_xticks(np.arange(0.0, 12.1, 1.0))
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel("X-ray energy (keV)", fontsize=8.0)
    ax.set_ylabel("counts (offset for clarity)", fontsize=8.0)
    ax.set_title("SEM-EDX spectra of a residue grain and a control grain",
                 fontsize=9.0, fontweight="bold", color=INK)
    ax.legend(loc="upper right", fontsize=6.8, frameon=True, framealpha=0.92,
              facecolor="white", edgecolor=HAIR, handlelength=1.6,
              borderpad=0.4)


def bary(a, b, c):
    return b + 0.5 * c, (np.sqrt(3.0) / 2.0) * c


def draw_ternary(ax, comp_g, comp_e):
    cx, cy = bary(*np.eye(3))
    ax.plot(np.r_[cx, cx[0]], np.r_[cy, cy[0]], color=SUB, lw=0.9)
    inner = np.array([[1 - 2 * T_MIN, T_MIN, T_MIN],
                      [T_MIN, 1 - 2 * T_MIN, T_MIN],
                      [T_MIN, T_MIN, 1 - 2 * T_MIN]])
    ix, iy = bary(inner[:, 0], inner[:, 1], inner[:, 2])
    ax.fill(ix, iy, facecolor=lighten(ORANGE, 0.78), edgecolor=ORANGE,
            linewidth=0.9, linestyle=(0, (3.0, 2.0)), zorder=1)

    for grid in np.arange(0.2, 1.0, 0.2):
        for perm in range(3):
            p = np.zeros((2, 3))
            p[0, perm] = grid
            p[0, (perm + 1) % 3] = 1.0 - grid
            p[1, perm] = grid
            p[1, (perm + 2) % 3] = 1.0 - grid
            gx, gy = bary(p[:, 0], p[:, 1], p[:, 2])
            ax.plot(gx, gy, color="#e2e2e2", lw=0.45, zorder=0)

    ex, ey = bary(comp_e[:, 0], comp_e[:, 1], comp_e[:, 2])
    gx, gy = bary(comp_g[:, 0], comp_g[:, 1], comp_g[:, 2])
    ax.scatter(ex, ey, s=7.0, color=BLUE, alpha=0.65, linewidths=0.0,
               zorder=3)
    ax.scatter(gx, gy, s=9.0, color=ORANGE, alpha=0.80, linewidths=0.0,
               zorder=4)

    for lab, (px, py), ha, va in [("Pb", (0.0, 0.0), "right", "top"),
                                  ("Sb", (1.0, 0.0), "left", "top"),
                                  ("Ba", (0.5, 0.866), "center", "bottom")]:
        ax.text(px, py + (0.02 if va == "bottom" else -0.02), lab, ha=ha,
                va=va, fontsize=7.4, fontweight="bold", color=INK)
    ax.text(0.5, 0.285, f"GSR zone\nall > {T_MIN * 100:.0f} %", ha="center",
            va="center", fontsize=6.0, color=ORANGE, linespacing=1.3,
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="none", alpha=0.85))

    ax.set_xlim(-0.14, 1.14)
    ax.set_ylim(-0.13, 0.99)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Pb-Sb-Ba mass fractions (%)", fontsize=8.5,
                 fontweight="bold", color=INK)


def draw_sizes(ax, d_gsr, d_env):
    edges = np.linspace(0.0, 12.0, 31)
    ax.hist(d_env, bins=edges, color=BLUE, alpha=0.55, edgecolor="white",
            linewidth=0.4, label=f"environmental (n = {d_env.size})")
    ax.hist(d_gsr, bins=edges, color=ORANGE, alpha=0.75, edgecolor="white",
            linewidth=0.4, label=f"GSR-consistent (n = {d_gsr.size})")
    for d, col in ((d_env, BLUE), (d_gsr, ORANGE)):
        ax.axvline(float(np.median(d)), color=col, lw=1.0,
                   ls=(0, (3.0, 2.0)))
    ax.text(0.97, 0.94, f"median {np.median(d_gsr):.1f} " + r"$\mu$m"
            + f" vs {np.median(d_env):.1f} " + r"$\mu$m",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.2,
            color=SUB)
    ax.set_xlim(0.0, 12.0)
    ax.tick_params(labelsize=6.4)
    ax.set_xlabel(r"equivalent circular diameter ($\mu$m)", fontsize=7.0)
    ax.set_ylabel("particles (count)", fontsize=7.0)
    ax.set_title("Particle size distribution", fontsize=8.5,
                 fontweight="bold", color=INK)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=6.2,
              frameon=False, handlelength=1.3)


def main():
    set_pub_style()
    rng = np.random.default_rng(881)

    energy = np.linspace(0.03, 12.0, 3600)
    s_gsr = spectrum(energy, GSR, 130.0, rng)
    s_env = spectrum(energy, ENV, 110.0, rng)
    s_bg = spectrum(energy, [], 90.0, rng)
    scales = (1.0, 0.045, 0.0016)

    n_g, n_e = 46, 58
    comp_g = rng.dirichlet([5.2, 3.4, 3.8], n_g)
    edge = rng.dirichlet([0.30, 0.22, 0.26], n_e)
    comp_e = edge / edge.sum(axis=1, keepdims=True)
    n_ok = int(np.sum(comp_g.min(axis=1) >= T_MIN))
    n_bad = int(np.sum(comp_e.min(axis=1) >= T_MIN))

    d_gsr = np.clip(rng.lognormal(np.log(1.7), 0.46, n_g * 5), 0.2, 12.0)
    d_env = np.clip(rng.lognormal(np.log(4.6), 0.52, n_e * 5), 0.2, 12.0)

    fig = plt.figure(figsize=(FW, FH))
    ax_main = axes_inches(fig, MAIN)
    ax_tri = axes_inches(fig, AUX_T)
    ax_sz = axes_inches(fig, AUX_B)

    draw_spectra(ax_main, energy, s_gsr, s_env, s_bg, scales)
    draw_ternary(ax_tri, comp_g, comp_e)
    draw_sizes(ax_sz, d_gsr, d_env)

    fig.text((AUX_T[0] + AUX_T[2] + 0.14) / FW, (AUX_T[1] + 1.40) / FH,
             f"grains examined\n  {n_g + n_e}\n\nin the GSR zone\n  "
             f"{n_ok + n_bad} of {n_g + n_e}\n\ncontrol grains\n  reaching "
             f"it  {n_bad}\n\nkey lines\n  Pb M 2.34 keV\n  Sb L 3.60 keV\n"
             f"  Ba L 4.47 keV",
             ha="left", va="top", fontsize=6.4, color=SUB, linespacing=1.5)


    panel_labels_inches(fig, [ax_main], labels=["(a)"], dx=-0.66, dy=0.14)
    panel_labels_inches(fig, [ax_tri, ax_sz], labels=["(b)", "(c)"],
                        dx=-0.54, dy=0.14)
    savefig(fig, "796_frn_gsr_spectrum.png")


if __name__ == "__main__":
    main()

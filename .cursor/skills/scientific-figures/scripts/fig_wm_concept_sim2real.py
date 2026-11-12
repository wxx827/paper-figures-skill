"""Domain randomisation: what the simulator spans, and what the robot needs.

Two overlapping supports.  The wide dashed ellipse is everything the
randomised simulator can produce, seeded by five parameter groups whose
sampled draws are drawn as small satellites around each parameter.  The
tighter solid ellipse is the real world.  Their lens is labelled coverage,
and the point of the figure is the two real configurations that fall
*outside* it -- the only element in the accent colour, because they are the
only thing here that can make a transfer fail.

Below the topology, one axis: how hard each parameter is being randomised.
That axis is the knob a practitioner actually turns, and the figure exists to
say which way to turn it.

Amplitudes and configurations are illustrative, not tuned values.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon

from pubstyle import COLORS, savefig, set_pub_style
from figkit import arrow, halo_text, inch_canvas

FW, FH = 10.4, 6.4

SIM_C, SIM_R = (4.15, 3.85), (2.80, 1.55)         # centre, semi-axes
REAL_C, REAL_R = (7.35, 3.85), (2.00, 1.25)
SIM_FILL, SIM_EDGE = "#dce6f1", "#5b83b5"
REAL_FILL, REAL_EDGE = "#8fb0d6", "#2f4f7f"
ACCENT = "#b23a48"

# name, randomisation amplitude in [0, 1], blue rung
PARAMS = [
    ("Latency", 0.22, "#7ba0cb"),
    ("Mass", 0.38, "#5b83b5"),
    ("Friction", 0.55, "#4a72a4"),
    ("Lighting", 0.72, "#3d6294"),
    ("Texture", 0.85, "#2f4f7f"),
]

BALL_Y = 4.30
BALL_X = [1.95 + 0.775 * i for i in range(5)]
COVERED = [(6.05, 4.05), (6.30, 3.55), (5.95, 3.35)]
UNCOVERED = [(8.15, 4.20), (8.50, 3.45)]

AX_Y, AX_X0, AX_X1 = 1.05, 1.60, 9.10
TICKS = [(0.00, "none"), (0.25, "mild"), (0.50, "moderate"),
         (0.75, "wide"), (1.00, "extreme")]


def ax_x(m):
    return AX_X0 + (AX_X1 - AX_X0) * m


def main():
    set_pub_style()
    rng = np.random.default_rng(1490)
    fig, ax = inch_canvas(FW, FH)

    # ---- the two supports ---------------------------------------------------
    ax.add_patch(Ellipse(SIM_C, 2 * SIM_R[0], 2 * SIM_R[1],
                         facecolor=SIM_FILL, edgecolor=SIM_EDGE, linewidth=1.5,
                         linestyle=(0, (5.0, 2.6)), zorder=2))
    ax.add_patch(Ellipse(REAL_C, 2 * REAL_R[0], 2 * REAL_R[1],
                         facecolor=REAL_FILL, edgecolor=REAL_EDGE,
                         linewidth=1.6, alpha=0.46, zorder=3))
    ax.add_patch(Ellipse(REAL_C, 2 * REAL_R[0], 2 * REAL_R[1],
                         facecolor="none", edgecolor=REAL_EDGE, linewidth=1.6,
                         zorder=4))

    halo_text(ax, 3.30, 5.00, "SIMULATION  DOMAIN", ha="center", va="center",
              fontsize=8.4, fontweight="bold", color="#2f4f7f", halo="white",
              halo_lw=3.2, zorder=6)
    halo_text(ax, 7.95, 4.85, "REAL  DOMAIN", ha="center", va="center",
              fontsize=8.4, fontweight="bold", color="#22364f", halo="white",
              halo_lw=3.2, zorder=6)

    # ---- randomisation parameters, each with its sampled draws --------------
    for (name, mag, col), bx in zip(PARAMS, BALL_X):
        ang = rng.uniform(0, 2 * np.pi, 9)
        rad = rng.uniform(0.30, 0.36, 9)
        ax.plot(bx + rad * np.cos(ang), BALL_Y + rad * np.sin(ang),
                marker="o", ls="none", ms=2.8, color=col, alpha=0.55,
                markeredgecolor="none", zorder=5)
        ax.plot([bx], [BALL_Y], marker="o", ms=11.0, markerfacecolor=col,
                markeredgecolor="white", markeredgewidth=1.2, zorder=6)
        halo_text(ax, bx, BALL_Y - 0.60, name, ha="center", va="center",
                  fontsize=6.8, color="#2b3a4c", halo="white", halo_lw=3.0,
                  zorder=6)

    # ---- what the real world actually asks for ------------------------------
    for x, y in COVERED:
        ax.plot([x], [y], marker="o", ms=7.5, markerfacecolor="#22364f",
                markeredgecolor="white", markeredgewidth=1.1, zorder=7)
    for x, y in UNCOVERED:
        ax.plot([x], [y], marker="o", ms=7.5, markerfacecolor="white",
                markeredgecolor=ACCENT, markeredgewidth=1.6, zorder=7)

    halo_text(ax, 6.22, 4.42, "coverage", ha="center", va="center",
              fontsize=8.6, fontweight="bold", color="#22364f", halo="white",
              halo_lw=3.4, zorder=7)

    # A ringed marker reads as "empty", not as "failure"; say which is which.
    key = [("#22364f", "#22364f", 1.1, "inside sim support"),
           ("white", ACCENT, 1.6, "outside sim support")]
    kx = 6.86
    for face, edge, mew, lab in key:
        ax.plot([kx], [5.46], marker="o", ms=7.5, markerfacecolor=face,
                markeredgecolor=edge, markeredgewidth=mew, zorder=7)
        ax.text(kx + 0.16, 5.46, lab, ha="left", va="center", fontsize=6.9,
                color=COLORS["sub"], zorder=7)
        kx += 1.62
    arrow(ax, (9.02, 3.02), (8.66, 3.37), color=ACCENT, lw=1.1, ms=9,
          zorder=7, rad=0.18)
    halo_text(ax, 9.98, 2.92, "two configurations fall\noutside the "
              "randomised support", ha="right", va="top", fontsize=6.9,
              color=ACCENT, halo="white", halo_lw=3.0, zorder=7,
              linespacing=1.55)

    # ---- the knob: randomisation amplitude ----------------------------------
    ax.text(AX_X0, 2.16, "RANDOMISATION  AMPLITUDE", ha="left", va="center",
            fontsize=7.4, fontweight="bold", color="#3d6294")
    ax.text(AX_X1 + 0.30, 2.16, "wider randomisation buys coverage and costs "
            "sample efficiency", ha="right", va="center", fontsize=6.8,
            color=COLORS["muted"])

    ax.add_patch(FancyBboxPatch((AX_X0, AX_Y - 0.085), AX_X1 - AX_X0, 0.17,
                                boxstyle="round,pad=0,rounding_size=0.085",
                                facecolor="#eef3f9", edgecolor="#c3d3e3",
                                linewidth=0.8, mutation_aspect=1.0, zorder=2))
    ax.add_patch(Polygon([(AX_X1, AX_Y + 0.155), (AX_X1 + 0.30, AX_Y),
                          (AX_X1, AX_Y - 0.155)], closed=True,
                         facecolor="#c3d3e3", edgecolor="none", zorder=2))
    for m, lab in TICKS:
        ax.plot([ax_x(m)] * 2, [AX_Y - 0.085, AX_Y - 0.20], color="#a8b8ca",
                lw=0.8, zorder=2)
        ax.text(ax_x(m), 0.66, lab, ha="center", va="center", fontsize=6.8,
                color=COLORS["sub"])

    for i, (name, mag, col) in enumerate(PARAMS):
        x = ax_x(mag)
        ly = 1.54 if i % 2 == 0 else 1.82
        ax.plot([x, x], [AX_Y + 0.085, ly - 0.13], color=col, lw=0.9,
                zorder=3)
        ax.plot([x], [AX_Y], marker="o", ms=7.5, markerfacecolor=col,
                markeredgecolor="white", markeredgewidth=1.1, zorder=4)
        halo_text(ax, x, ly, f"{name}  {mag:.2f}", ha="center", va="center",
                  fontsize=6.8, color="#2b3a4c", halo="white", halo_lw=3.0,
                  zorder=5)

    # ---- titles -------------------------------------------------------------
    ax.text(0.35, 6.12, "Domain randomisation and the coverage gap",
            ha="left", va="center", fontsize=11.0, fontweight="bold",
            color=COLORS["ink"])
    ax.text(0.35, 5.82, "Transfer succeeds only where the randomised "
            "simulator's support already contains the real configuration",
            ha="left", va="center", fontsize=8.0, color=COLORS["sub"])
    ax.text(0.35, 0.26, "Ellipse areas are schematic and the amplitudes are "
            "illustrative settings, not tuned values",
            ha="left", va="center", fontsize=6.6, color=COLORS["muted"])

    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    savefig(fig, "1085_wm_concept_sim2real.png")


if __name__ == "__main__":
    main()

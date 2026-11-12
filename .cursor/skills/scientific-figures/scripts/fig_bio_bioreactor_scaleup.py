"""Scale-up train from shake flask to 5 m3, on normalised parallel axes.

Holding one engineering parameter constant is the whole game in scale-up, and
the main panel shows the price of that choice: the train is designed at
constant volumetric power input, so the P/V axis is nearly flat while mixing
time and shear climb and oxygen transfer decays.  Each vertical axis carries
its own real units; only the vertical position is normalised, so the lines can
be compared without pretending the quantities share a scale.  The right-hand
panels close the loop - the titre actually reached at each scale, and a
breakdown of the gap at the largest scale whose components are rescaled to sum
to exactly the measured deviation.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, get_palette, savefig
from figkit import axes_inches, panel_label_inches

GREY = "#808080"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
RED = "#b23a48"
INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

W, H = 10.4, 6.4
SCALES = ["shake flask", "5 L", "50 L", "500 L", "5000 L"]
VOLUME = [0.25, 5.0, 50.0, 500.0, 5000.0]
REF = 1                                   # 5 L bench reactor is the reference

# metric -> (label, unit, values per scale, higher_is_better)
METRICS = [
    (r"$k_La$", "1/h", [95.0, 210.0, 175.0, 148.0, 126.0]),
    ("mixing time", "s", [2.0, 6.0, 14.0, 32.0, 68.0]),
    ("max shear rate", "1/s", [380.0, 1450.0, 1720.0, 2050.0, 2400.0]),
    ("power input P/V", r"W/m$^3$", [120.0, 950.0, 900.0, 860.0, 820.0]),
]
TITRE = [11.2, 14.3, 13.6, 12.4, 11.1]    # g/L at harvest
GAP_TERMS = [("oxygen transfer limitation", 9.0),
             ("mixing and gradients", 7.0),
             ("dissolved CO$_2$ build-up", 4.0),
             ("hydrodynamic shear", 2.0)]


def colors():
    ramp = get_palette("cb_sequential_blue", 7)[2:6]
    return [GREY] + list(ramp)


def parallel_panel(ax, cols):
    n = len(METRICS)
    ax.set_xlim(-0.42, n - 1 + 0.42)
    ax.set_ylim(-0.26, 1.20)
    ax.axis("off")

    for j, (name, unit, vals) in enumerate(METRICS):
        v = np.array(vals, dtype=float)
        lo, hi = v.min(), v.max()
        ax.plot([j, j], [0.0, 1.0], color=RULE, lw=1.0, zorder=3)
        for frac in np.linspace(0.0, 1.0, 5):
            ax.plot([j - 0.028, j + 0.028], [frac, frac], color=RULE, lw=0.8,
                    zorder=3)
            ax.text(j + 0.055, frac, f"{lo + frac * (hi - lo):.0f}",
                    ha="left", va="center", fontsize=6.2, color=MUTED,
                    zorder=6,
                    bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                              edgecolor="none", alpha=0.85))
        ax.text(j, 1.10, name, ha="center", va="bottom", fontsize=7.6,
                color=INK)
        ax.text(j, 1.055, f"({unit})", ha="center", va="bottom", fontsize=6.6,
                color=MUTED)

    norm = []
    for _, _, vals in METRICS:
        v = np.array(vals, dtype=float)
        norm.append((v - v.min()) / (v.max() - v.min()))
    norm = np.array(norm).T

    xs = np.arange(n, dtype=float)
    for i, (name, col) in enumerate(zip(SCALES, cols)):
        lw = 2.2 if i == REF else 1.5
        ax.plot(xs, norm[i], color=col, lw=lw, marker="o", ms=4.2,
                mfc="white", mew=1.0, mec=col, zorder=5,
                label=f"{name} ({VOLUME[i]:.2f} L)" if i == 0 else name)

    band = norm[1:, -1]
    ax.plot([n - 1 - 0.075, n - 1 - 0.075], [band.min(), band.max()],
            color=RED, lw=2.4, solid_capstyle="butt", zorder=6)
    ax.text(-0.42, -0.13, "normalised position on each axis: bottom = "
            "smallest value in the train, top = largest", ha="left",
            va="center", fontsize=6.5, color=MUTED)
    ax.text(n - 1 + 0.42, -0.13, "red bar: constant P/V above 5 L",
            ha="right", va="center", fontsize=6.8, color=RED)
    ax.legend(fontsize=7.0, frameon=False, loc="lower center", ncol=5,
              bbox_to_anchor=(0.5, -0.075), handlelength=1.5,
              columnspacing=1.4)
    ax.set_title("Constant power input does not keep mixing or oxygen "
                 "transfer constant", fontsize=9.0, fontweight="bold",
                 color=INK)


def titre_panel(ax, cols):
    x = np.arange(len(SCALES), dtype=float)
    ax.bar(x, TITRE, width=0.60, color=cols, edgecolor="white", lw=0.7,
           zorder=3)
    ax.axhline(TITRE[REF], color=RED, lw=1.0, ls=(0, (3.2, 2.2)), zorder=4)
    ax.text(-0.42, max(TITRE) * 1.25,
            f"dashed rule: bench reference {TITRE[REF]:.1f} g/L", ha="left",
            va="top", fontsize=6.4, color=RED)
    for xi, v in zip(x, TITRE):
        dev = (v / TITRE[REF] - 1.0) * 100.0
        ax.text(xi, v + 0.30, f"{dev:+.0f}%", ha="center", va="bottom",
                fontsize=6.4, color=GREEN if dev >= 0 else RED)
    ax.set_xticks(x)
    ax.set_xticklabels(SCALES, fontsize=6.6)
    ax.set_ylabel("harvest titre (g/L)", fontsize=7.4)
    ax.set_ylim(0.0, max(TITRE) * 1.28)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title("Titre erodes with every scale step", fontsize=8.2,
                 fontweight="bold", color=INK)
    return (TITRE[-1] / TITRE[REF] - 1.0) * 100.0


def gap_panel(ax, deficit):
    names = [t[0] for t in GAP_TERMS]
    raw = np.array([t[1] for t in GAP_TERMS], dtype=float)
    share = raw / raw.sum() * abs(deficit)

    y = np.arange(len(names), dtype=float)[::-1]
    ax.barh(y, -share, height=0.56, color=RED, alpha=0.88, edgecolor="white",
            lw=0.7, zorder=3)
    for yi, v in zip(y, share):
        ax.text(-v - 0.35, yi, f"-{v:.1f}%", ha="right", va="center",
                fontsize=6.4, color=RED)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.8)
    ax.set_xlabel("contribution to the 5000 L titre gap (%)", fontsize=7.4)
    ax.set_xlim(-abs(deficit) * 0.62, 0.0)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.7)
    ax.set_title(f"Attribution of the {deficit:.0f} % gap at 5000 L",
                 fontsize=8.2, fontweight="bold", color=INK)


def main():
    set_pub_style()
    cols = colors()

    fig = plt.figure(figsize=(W, H))
    ax_pc = axes_inches(fig, (0.62, 0.95, 5.45, 4.18))
    ax_ti = axes_inches(fig, (7.10, 3.16, 2.95, 1.97))
    ax_gp = axes_inches(fig, (7.55, 0.90, 2.50, 1.58))

    parallel_panel(ax_pc, cols)
    deficit = titre_panel(ax_ti, cols)
    gap_panel(ax_gp, deficit)

    kla = np.array(METRICS[0][2])
    tmix = np.array(METRICS[1][2])
    fig.text(0.021, 6.14 / H,
             "Scaling at constant P/V costs "
             f"{(1 - kla[-1] / kla[REF]) * 100:.0f} % of the oxygen transfer "
             f"and stretches mixing time {tmix[-1] / tmix[REF]:.0f}-fold",
             ha="left", va="center", fontsize=11.0, fontweight="bold",
             color=INK)
    fig.text(0.021, 0.11 / H,
             f"Working volumes span {VOLUME[0]:.2f} L to {VOLUME[-1]:.0f} L, "
             f"a {VOLUME[-1] / VOLUME[0]:.0f}-fold range. The 5 L bench "
             f"reactor is the reference for every deviation quoted here; "
             f"gap components are rescaled to sum to the measured "
             f"{deficit:.1f} % shortfall.",
             ha="left", va="bottom", fontsize=6.5, color=MUTED)

    panel_label_inches(fig, ax_pc, "(a)", dx=-0.42, dy=0.16)
    panel_label_inches(fig, ax_ti, "(b)", dx=-0.62, dy=0.14)
    panel_label_inches(fig, ax_gp, "(c)", dx=-1.07, dy=0.14)

    savefig(fig, "712_bio_bioreactor_scaleup.png")


if __name__ == "__main__":
    main()

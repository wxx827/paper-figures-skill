"""Ablation study drawn as a figure instead of a LaTeX table.

Left block: a component matrix (filled tick = module enabled, open cross =
disabled).  Middle block: one cell per metric where the bar length and the
cell tint both encode the value, normalised within its own column.  Right
block: the aggregated score.  The proposed configuration is the last row and
is highlighted throughout.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

from pubstyle import set_pub_style, savefig, lighten

OURS = "#3b5b92"
GREY = "#8c8c8c"
LGREY = "#b8b8b8"

MODULES = ["A", "B", "C", "D"]
MODULE_LONG = ["A: adaptive gating", "B: cross-modal fusion",
               "C: contrastive loss", "D: EMA teacher"]

ROWS = [
    ("Baseline",        (0, 0, 0, 0)),
    ("+ A",             (1, 0, 0, 0)),
    ("+ B",             (0, 1, 0, 0)),
    ("+ A + B",         (1, 1, 0, 0)),
    ("+ A + B + C",     (1, 1, 1, 0)),
    ("Full (Ours)",     (1, 1, 1, 1)),
]

# metric -> (values per row, higher-is-better, unit, format)
METRICS = [
    ("Top-1", [76.2, 78.0, 77.4, 79.6, 80.7, 81.9], True, "%"),
    ("mIoU", [68.4, 70.1, 70.8, 72.3, 73.1, 74.6], True, "%"),
    ("F1", [71.0, 72.6, 72.2, 74.1, 75.0, 76.4], True, "%"),
    ("ECE $\\downarrow$", [8.4, 7.1, 7.6, 5.9, 5.1, 4.2], False, "%"),
    ("Robust", [52.1, 55.0, 54.2, 58.3, 60.4, 63.1], True, "%"),
]

NR = len(ROWS)
CW = 1.62          # metric cell width
X_MOD = 0.0        # first module column centre
X_MET = 4.35       # left edge of the first metric cell
X_TOT = X_MET + len(METRICS) * CW + 0.55


def main():
    set_pub_style(base_font=8)

    vals = np.array([m[1] for m in METRICS], dtype=float)      # (nmet, nrow)
    norm = np.zeros_like(vals)
    for i, (_, _, higher, _) in enumerate(METRICS):
        v = vals[i]
        lo, hi = v.min(), v.max()
        n = (v - lo) / (hi - lo)
        norm[i] = n if higher else 1.0 - n
    total = 100.0 * norm.mean(axis=0)

    fig = plt.figure(figsize=(7.5, 3.5))
    ax = fig.add_axes([0.005, 0.045, 0.99, 0.80])
    ax.set_xlim(-3.9, X_TOT + 3.5)
    ax.set_ylim(NR - 0.45, -1.35)
    ax.axis("off")

    # ---- row bands ---------------------------------------------------------
    for r, (name, flags) in enumerate(ROWS):
        hot = "Ours" in name
        ax.add_patch(Rectangle((-3.85, r - 0.42), X_TOT + 7.3, 0.84,
                               facecolor="#eef3f8" if hot else
                               ("#fafafa" if r % 2 else "white"),
                               edgecolor="none", zorder=0))
        ax.text(-3.75, r, name, ha="left", va="center",
                fontsize=8.2 if hot else 7.8,
                color=OURS if hot else "#1a1a1a",
                fontweight="bold" if hot else "normal", zorder=3)

    # ---- component matrix --------------------------------------------------
    for r, (_, flags) in enumerate(ROWS):
        for c, on in enumerate(flags):
            x = X_MOD + c * 0.85
            if on:
                ax.plot([x], [r], marker="o", ms=9.5, mfc=lighten(OURS, 0.12),
                        mec=OURS, mew=0.8, zorder=3)
                ax.plot([x - 0.115, x - 0.03, x + 0.135],
                        [r - 0.02, r + 0.13, r - 0.16],
                        color="white", lw=1.35, solid_capstyle="round",
                        solid_joinstyle="miter", zorder=4)
            else:
                ax.plot([x], [r], marker="o", ms=9.5, mfc="white",
                        mec="#dcdcdc", mew=0.8, zorder=3)
                ax.plot([x - 0.1, x + 0.1], [r - 0.11, r + 0.11],
                        color="#c9c9c9", lw=1.0, zorder=4)
                ax.plot([x - 0.1, x + 0.1], [r + 0.11, r - 0.11],
                        color="#c9c9c9", lw=1.0, zorder=4)

    # ---- metric cells ------------------------------------------------------
    for i, (mname, _, higher, unit) in enumerate(METRICS):
        x0 = X_MET + i * CW
        for r in range(NR):
            hot = "Ours" in ROWS[r][0]
            n = norm[i, r]
            ax.add_patch(FancyBboxPatch(
                (x0 + 0.06, r - 0.31), CW - 0.20, 0.62,
                boxstyle="round,pad=0,rounding_size=0.07",
                facecolor=lighten(OURS, 0.97 - 0.16 * n),
                edgecolor="#dde5ec", lw=0.5, zorder=1))
            bw = (CW - 0.32) * (0.10 + 0.90 * n)
            ax.add_patch(Rectangle(
                (x0 + 0.13, r - 0.155), bw, 0.31,
                facecolor=OURS if hot else lighten(OURS, 0.76 - 0.30 * n),
                edgecolor="none", zorder=2))
            inside = hot
            ax.text(x0 + 0.24, r, f"{vals[i, r]:.1f}", ha="left",
                    va="center", fontsize=6.7,
                    color="white" if inside else "#4a4a4a",
                    fontweight="bold" if hot else "normal", zorder=5)

    # ---- overall score bars ------------------------------------------------
    smax = 3.05
    for r in range(NR):
        hot = "Ours" in ROWS[r][0]
        w = smax * (0.02 + 0.98 * total[r] / 100.0)
        ax.add_patch(Rectangle((X_TOT, r - 0.19), smax, 0.38,
                               facecolor="#f4f4f4", edgecolor="none",
                               zorder=1))
        ax.add_patch(Rectangle((X_TOT, r - 0.19), w, 0.38,
                               facecolor=OURS if hot else LGREY,
                               edgecolor="none", zorder=2))
        ax.text(X_TOT + smax + 0.12, r, f"{total[r]:.0f}", ha="left",
                va="center", fontsize=7.4 if hot else 7.0,
                color=OURS if hot else "#5a5a5a",
                fontweight="bold" if hot else "normal", zorder=3)

    # ---- headers -----------------------------------------------------------
    hy = -0.72
    for c, m in enumerate(MODULES):
        ax.text(X_MOD + c * 0.85, hy, m, ha="center", va="center",
                fontsize=7.6, fontweight="bold", color="#333333")
    for i, (mname, _, _, unit) in enumerate(METRICS):
        ax.text(X_MET + i * CW + (CW - 0.14) / 2, hy, mname, ha="center",
                va="center", fontsize=7.6, fontweight="bold", color="#333333")
    ax.text(X_TOT + smax / 2, hy, "Relative gain", ha="center", va="center",
            fontsize=7.6, fontweight="bold", color="#333333")

    groups = [(-3.85, X_MOD + 3 * 0.85 + 0.35, "Components"),
              (X_MET + 0.06, X_MET + len(METRICS) * CW - 0.14,
               "Metrics (%)"),
              (X_TOT, X_TOT + smax, "0 = Baseline, 100 = Full")]
    for xa, xb, lab in groups:
        ax.plot([xa, xb], [-1.02, -1.02], color="#333333", lw=0.9,
                clip_on=False)
        ax.text((xa + xb) / 2, -1.14, lab, ha="center", va="center",
                fontsize=7.0, color="#666666", style="italic")

    ax.plot([-3.85, X_TOT + smax], [NR - 0.47, NR - 0.47], color="#cccccc",
            lw=0.7, clip_on=False)


    for r, (name, _) in enumerate(ROWS):
        print(f"  {name:14s} total {total[r]:5.1f}   " +
              "  ".join(f"{METRICS[i][0].split(' ')[0]}={vals[i, r]:5.1f}"
                        for i in range(len(METRICS))))
    savefig(fig, "305_ablation_table_viz.png")


if __name__ == "__main__":
    main()

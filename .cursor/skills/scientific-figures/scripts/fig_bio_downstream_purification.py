"""Downstream train: what each unit operation costs and what it buys.

The main panel is the trade the whole process is built around - every step
takes yield away and gives purity back - so step yield, cumulative yield and
purity share one staircase with the cumulative loss shaded behind them.  The
chromatography column is not assumed: its recovery and its pool purity are
integrated from the elution trace in panel (b) over the collection window
drawn there, and the same two numbers are the ones the staircase steps
through.  Panel (c) is the visual confirmation the wet lab actually looks at.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, inch_axes, panel_label_inches

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GREY = "#808080"
RED = "#b23a48"
INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
SOFT = "#f5f5f5"

W, H = 10.4, 6.4
STEPS = ["centrifugation", "homogenisation", "clarification",
         "chromatography", "ultrafiltration", "lyophilisation"]
STEP_Y = [95.0, 88.0, 92.0, None, 94.0, 97.0]      # None = from panel (b)
PURITY = [9.0, 11.0, 18.0, None, 91.0, 94.0]
CHROM = 3
FEED_PURITY = 6.0
POOL = (27.6, 33.4)                                 # collection window, mL

# elution peaks: (centre mL, sigma mL, height mAU, is_target)
PEAKS = [(13.0, 1.9, 260.0, False), (30.6, 2.6, 420.0, True),
         (32.6, 2.0, 92.0, False), (38.8, 2.2, 150.0, False)]

# SDS-PAGE lanes: name -> list of (kDa, intensity)
MARKERS = [250.0, 130.0, 70.0, 45.0, 35.0, 25.0, 15.0]
TARGET_KDA = 45.0
LANES = [
    ("M", [(m, 0.55) for m in MARKERS]),
    ("lysate", [(210.0, 0.45), (150.0, 0.55), (95.0, 0.60), (66.0, 0.75),
                (45.0, 0.35), (38.0, 0.70), (28.0, 0.62), (22.0, 0.50),
                (17.0, 0.45)]),
    ("clarified", [(150.0, 0.35), (95.0, 0.40), (66.0, 0.58), (45.0, 0.48),
                   (38.0, 0.52), (28.0, 0.40), (17.0, 0.28)]),
    ("eluate", [(66.0, 0.18), (45.0, 0.88), (38.0, 0.22), (28.0, 0.14)]),
    ("final", [(45.0, 0.95), (38.0, 0.08)]),
]


def gauss(x, c, s, a):
    return a * np.exp(-0.5 * ((x - c) / s) ** 2)


def elution(n=900):
    v = np.linspace(0.0, 50.0, n)
    target = np.zeros_like(v)
    imp = np.zeros_like(v)
    for c, s, a, is_t in PEAKS:
        (target if is_t else imp).__iadd__(gauss(v, c, s, a))
    base = 6.0 + 3.0 * np.exp(-((v - 4.0) / 2.5) ** 2)
    grad = 100.0 * np.clip((v - 8.0) / 30.0, 0.0, 1.0)
    m = (v >= POOL[0]) & (v <= POOL[1])
    rec = np.trapezoid(target[m], v[m]) / np.trapezoid(target, v)
    pur = (np.trapezoid(target[m], v[m]) /
           np.trapezoid(target[m] + imp[m], v[m]))
    return v, target, imp, base, grad, m, rec, pur


def staircase(ax, step_y, purity):
    x = np.arange(-1, len(STEPS), dtype=float)
    cum = np.concatenate([[100.0], np.cumprod(np.array(step_y) / 100.0) * 100.0])
    pur = np.concatenate([[FEED_PURITY], purity])

    ax.bar(np.arange(len(STEPS)), step_y, width=0.56, color=GREY, alpha=0.35,
           edgecolor="white", lw=0.7, zorder=2, label="step yield")
    ax.fill_between(x, cum, 100.0, step="post", color=RED, alpha=0.09,
                    linewidth=0.0, zorder=1)
    ax.plot(x, cum, drawstyle="steps-post", color=BLUE, lw=1.9, zorder=5,
            label="cumulative yield")
    ax.plot(x, cum, ls="none", marker="o", ms=4.2, mfc="white", mew=1.1,
            mec=BLUE, zorder=6)
    for xi, v in zip(x, cum):
        ax.text(xi + 0.06, v + 2.6, f"{v:.0f}", ha="left", va="bottom",
                fontsize=6.4, color=BLUE)

    ax2 = ax.twinx()
    ax2.plot(x, pur, drawstyle="steps-post", color=GREEN, lw=1.9, zorder=5,
             label="purity")
    ax2.plot(x, pur, ls="none", marker="s", ms=3.8, mfc="white", mew=1.1,
             mec=GREEN, zorder=6)
    for xi, v in zip(x, pur):
        ax2.text(xi + 0.06, v - 3.0, f"{v:.0f}", ha="left", va="top",
                 fontsize=6.4, color=GREEN)
    ax2.set_ylabel("product purity (%)", color=GREEN)
    ax2.set_ylim(0.0, 105.0)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(GREEN)
    ax2.tick_params(axis="y", colors=GREEN)

    worst = int(np.argmin(step_y))
    ax.annotate(f"largest single loss:\n{100 - step_y[worst]:.0f} % at the "
                f"{STEPS[worst]} step",
                xy=(worst, cum[worst + 1]), xytext=(worst + 0.35, 26.0),
                ha="left", va="center", fontsize=6.8, color=RED,
                linespacing=1.5,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    ax.text(-0.85, 92.0, "shaded band = cumulative loss", ha="left",
            va="center", fontsize=6.6, color=RED)

    ax.set_xticks(x)
    ax.set_xticklabels(["harvest"] + STEPS, rotation=18.0, ha="right",
                       fontsize=7.0)
    ax.set_ylabel("yield (%)", color=BLUE)
    ax.set_ylim(0.0, 105.0)
    ax.set_xlim(-1.55, len(STEPS) - 0.35)
    ax.tick_params(axis="y", colors=BLUE)
    ax.spines["left"].set_color(BLUE)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.0, frameon=False, loc="lower left",
              bbox_to_anchor=(0.015, 0.02), handlelength=1.6)
    ax.set_title(f"Six unit operations keep {cum[-1]:.0f} % of the product "
                 f"and lift purity from {FEED_PURITY:.0f} % to "
                 f"{pur[-1]:.0f} %", fontsize=9.0, fontweight="bold",
                 color=INK)
    return cum[-1]


def elution_panel(ax, v, target, imp, base, grad, m, rec, pur):
    total = base + target + imp
    ax.axvspan(POOL[0], POOL[1], facecolor="#f0f0f0", edgecolor="none",
               zorder=0)
    ax.fill_between(v, base, base + imp, color=RED, alpha=0.30,
                    linewidth=0.0, zorder=2)
    ax.fill_between(v, base, base + target, color=GREEN, alpha=0.30,
                    linewidth=0.0, zorder=3)
    ax.plot(v, total, color=RULE, lw=1.3, zorder=5)
    ax.text(13.0, gauss(13.0, 13.0, 1.9, 260.0) + 30.0, "flow-through",
            ha="center", va="bottom", fontsize=6.2, color=RED)
    ax.text(38.8, gauss(38.8, 38.8, 2.2, 150.0) + 26.0, "late impurity",
            ha="center", va="bottom", fontsize=6.2, color=RED)
    ax.text(np.mean(POOL), 22.0, "pooled", ha="center", va="bottom",
            fontsize=6.6, color=SUB)
    ax.text(30.6, 484.0, "target", ha="center", va="bottom", fontsize=6.6,
            color=GREEN)

    ax.set_xlabel("elution volume (mL)", fontsize=7.4)
    ax.set_ylabel(r"A$_{280}$ (mAU)", fontsize=7.4)
    ax.set_xlim(0.0, 50.0)
    ax.set_ylim(0.0, 520.0)
    ax.tick_params(labelsize=6.5)

    ax2 = ax.twinx()
    ax2.plot(v, grad, color=GREY, lw=1.1, ls=(0, (4.5, 2.2)), zorder=4)
    ax2.set_ylabel("eluent B (%)", fontsize=7.4, color=GREY)
    ax2.set_ylim(0.0, 105.0)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(GREY)
    ax2.tick_params(axis="y", colors=GREY, labelsize=6.5)

    ax.text(1.0, 508.0, f"pool {POOL[0]:.1f} - {POOL[1]:.1f} mL\n"
            f"{rec * 100:.0f} % recovery,  {pur * 100:.0f} % purity",
            ha="left", va="top", fontsize=6.6, color=INK, linespacing=1.5)
    ax.set_title("Collection window sets the recovery / purity trade",
                 fontsize=8.2, fontweight="bold", color=INK)


def gel_panel(ax, w, h):
    ax.text(0.04, h - 0.08, "SDS-PAGE of the pooled fractions", ha="left",
            va="top", fontsize=8.6, fontweight="bold", color=INK)
    x0, x1 = 0.52, w - 0.14
    y0, y1 = 0.34, h - 0.46
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=SOFT,
                           edgecolor=HAIR, lw=0.7, zorder=2))

    def ypos(kda):
        f = (np.log10(250.0) - np.log10(kda)) / (np.log10(250.0) -
                                                 np.log10(12.0))
        return y1 - 0.055 - f * (y1 - y0 - 0.11)

    for kda in MARKERS:
        ax.text(x0 - 0.07, ypos(kda), f"{kda:.0f}", ha="right", va="center",
                fontsize=6.2, color=MUTED, zorder=4)
    ax.text(x0 - 0.07, y1 + 0.10, "kDa", ha="right", va="center",
            fontsize=6.4, color=SUB, zorder=4)

    lane_w = (x1 - x0) / (len(LANES) + 0.6)
    for i, (name, bands) in enumerate(LANES):
        cx = x0 + (i + 0.8) * lane_w
        for kda, inten in bands:
            is_target = abs(kda - TARGET_KDA) < 1e-6 and i > 0
            col = BLUE if is_target else "#3a3a3a"
            ax.add_patch(Rectangle((cx - lane_w * 0.34, ypos(kda) - 0.028),
                                   lane_w * 0.68, 0.056, facecolor=col,
                                   edgecolor="none", alpha=0.20 + 0.78 * inten,
                                   zorder=3))
        ax.text(cx, y0 - 0.09, name, ha="center", va="top", fontsize=6.6,
                color=INK, rotation=0.0)

    cx_last = x0 + (len(LANES) - 0.2) * lane_w
    ax.annotate(f"target {TARGET_KDA:.0f} kDa",
                xy=(cx_last, ypos(TARGET_KDA)),
                xytext=(x1 - 0.02, ypos(TARGET_KDA) + 0.34), ha="right",
                va="bottom", fontsize=6.4, color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6))
    ax.text(x0, 0.06, "band darkness is proportional to protein load",
            ha="left", va="bottom", fontsize=6.2, color=MUTED)


def main():
    set_pub_style()
    v, target, imp, base, grad, m, rec, pur = elution()
    step_y = [rec * 100.0 if s is None else s for s in STEP_Y]
    purity = [pur * 100.0 if s is None else s for s in PURITY]

    fig = plt.figure(figsize=(W, H))
    ax_main = axes_inches(fig, (0.82, 1.02, 5.20, 4.10))
    ax_el = axes_inches(fig, (7.08, 3.28, 2.86, 1.84))
    ax_gel = inch_axes(fig, (6.68, 0.72, 3.45, 2.05))

    overall = staircase(ax_main, step_y, purity)
    elution_panel(ax_el, v, target, imp, base, grad, m, rec, pur)
    gel_panel(ax_gel, 3.45, 2.05)

    fig.text(0.021, 6.16 / H,
             "Chromatography buys almost all of the purity and pays almost "
             f"all of the yield: {overall:.0f} % overall recovery",
             ha="left", va="center", fontsize=11.0, fontweight="bold",
             color=INK)
    fig.text(0.021, 0.11 / H,
             f"Chromatography recovery ({rec * 100:.0f} %) and pool purity "
             f"({pur * 100:.0f} %) are integrated from the elution trace over "
             f"the {POOL[0]:.1f} - {POOL[1]:.1f} mL window; the other five "
             f"steps use campaign averages. Overall recovery "
             f"{overall:.1f} %.",
             ha="left", va="bottom", fontsize=6.5, color=MUTED)

    panel_label_inches(fig, ax_main, "(a)", dx=-0.62, dy=0.14)
    panel_label_inches(fig, ax_el, "(b)", dx=-0.62, dy=0.14)
    panel_label_inches(fig, ax_gel, "(c)", dx=-0.22, dy=0.02)

    savefig(fig, "713_bio_downstream_purification.png")


if __name__ == "__main__":
    main()

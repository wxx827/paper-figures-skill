"""Figure 479 - Recommendation funnel from impression to order.

Left: the funnel itself, one rounded slab per stage with the connecting necks
filled, the absolute volume inside each slab and the step conversion printed on
the neck.  Slab width is proportional to volume, so the shape carries the drop
-off rather than merely decorating it.

Right: the same three conversions broken out by user segment.  The three steps
differ by an order of magnitude (4 % against 32 %), so a single colour scale
over the raw rates would flatten two of the columns; the cells are coloured by
their lift against the column average on a symmetric diverging scale and
annotated with the raw rate, which keeps both readings available.

Bottom: a 30-day trend per stage, indexed to day 1 so that four volumes spanning
five orders of magnitude can honestly share one axis.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

from pubstyle import (set_pub_style, savefig, get_cmap, add_colorbar,
                      lighten, darken)
from figkit import (axes_inches, inch_axes, panel_label_inches, shared_limits,
                    shared_norm)

INK, SUB, MUTED, FAINT = "#1a1a1a", "#555555", "#777777", "#8a8a8a"
HAIR, GRID = "#cccccc", "#d9d9d9"

STAGES = ["Impressions", "Clicks", "Add to cart", "Orders"]
STEPS = ["CTR (%)", "cart rate (%)", "order rate (%)"]
SEGMENTS = ["New", "Casual", "Core", "Lapsed", "Premium"]
# Slabs walk a single blue ramp: one hue, monotone lightness, no rainbow.
SLAB = ["#9cc3dd", "#5f9dc6", "#2e74a8", "#134a76"]
ACCENT = "#b23a48"      # the single emphasis: the weakest step
DAYS = 30
BASE = 12_400_000       # impressions in the window
RATES = (0.0438, 0.1820, 0.3240)


def segment_rates(rng):
    """Per-segment step conversion, multiplicative around the overall rate."""
    tilt = np.array([[1.24, 0.72, 0.61],      # New: clicks a lot, buys little
                     [1.02, 0.94, 0.88],
                     [0.88, 1.18, 1.26],      # Core: fewer clicks, more orders
                     [0.71, 0.83, 0.74],
                     [0.96, 1.31, 1.44]])
    noise = np.exp(rng.normal(0.0, 0.045, tilt.shape))
    return np.array(RATES)[None, :] * tilt * noise


def trends(rng):
    """Daily volume per stage over the window, indexed to day 1."""
    day = np.arange(1, DAYS + 1)
    week = 1.0 + 0.085 * np.sin(2 * np.pi * day / 7.0)
    imp = BASE / DAYS * week * np.exp(np.cumsum(rng.normal(0, 0.011, DAYS)))
    out = [imp]
    for r, drift in zip(RATES, (0.0022, 0.0014, -0.0009)):
        prev = out[-1]
        step = r * (1.0 + drift * (day - 1)) * np.exp(
            rng.normal(0.0, 0.017, DAYS))
        out.append(prev * step)
    return day, out


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(479)

    vols = [BASE]
    for r in RATES:
        vols.append(vols[-1] * r)
    vols = np.array(vols)
    conv = vols[1:] / vols[:-1]

    fig = plt.figure(figsize=(10.4, 6.4))

    # ---- (a) funnel --------------------------------------------------------
    axf = inch_axes(fig, (0.30, 2.02, 4.80, 3.52))
    FW, FH = axf.get_xlim()[1], axf.get_ylim()[1]
    cx, wmax = FW * 0.44, 3.20
    slab_h, gap = 0.50, 0.40
    widths = wmax * (vols / vols[0]) ** 0.34      # cube-ish root keeps the
    widths = np.maximum(widths, 0.70)             # last slab legible
    tops = FH - 0.30 - np.arange(len(STAGES)) * (slab_h + gap)
    weakest = int(np.argmin(conv))

    for i, (name, v, w, top) in enumerate(zip(STAGES, vols, widths, tops)):
        if i:
            # Neck: the trapezoid joining this slab to the one above.
            pw, pt = widths[i - 1], tops[i - 1]
            axf.add_patch(Polygon(
                [(cx - pw / 2, pt - slab_h), (cx + pw / 2, pt - slab_h),
                 (cx + w / 2, top), (cx - w / 2, top)],
                closed=True, facecolor="#f0f0f0", edgecolor="none", zorder=2))
            hot = i - 1 == weakest
            axf.annotate("", xy=(cx, top + 0.045),
                         xytext=(cx, pt - slab_h - 0.045),
                         arrowprops=dict(arrowstyle="-|>", lw=1.0,
                                         color=ACCENT if hot else "#5a5a5a"),
                         zorder=4)
            axf.text(cx + 0.16, top + gap / 2,
                     f"{100 * conv[i - 1]:.1f} %",
                     ha="left", va="center", fontsize=7.4,
                     color=ACCENT if hot else INK,
                     fontweight="bold" if hot else "normal", zorder=5)
            if hot:
                axf.text(cx + 0.16, top + gap / 2 - 0.155, "tightest step",
                         ha="left", va="center", fontsize=6.4, color=ACCENT,
                         zorder=5)

        axf.add_patch(FancyBboxPatch(
            (cx - w / 2, top - slab_h), w, slab_h,
            boxstyle="round,pad=0,rounding_size=0.10", facecolor=SLAB[i],
            edgecolor="white", linewidth=1.0, mutation_aspect=1.0, zorder=3))
        axf.text(cx, top - slab_h / 2, f"{v / 1e6:.2f} M" if v > 1e6
                 else f"{v / 1e3:.0f} k", ha="center", va="center",
                 fontsize=7.6, color="white" if i else INK, fontweight="bold",
                 zorder=5)
        axf.text(cx - wmax / 2 - 0.30, top - slab_h / 2, name, ha="right",
                 va="center", fontsize=8.0, color=INK, zorder=5)
        axf.text(cx + wmax / 2 + 0.66, top - slab_h / 2,
                 f"{100 * v / vols[0]:.2f} %".rstrip(), ha="right",
                 va="center", fontsize=6.8, color=SUB, zorder=5)

    axf.text(cx + wmax / 2 + 0.66, tops[0] + 0.12, "of impressions",
             ha="right", va="bottom", fontsize=6.4, color=MUTED)
    axf.text(cx - wmax / 2, FH - 0.06, "slab width $\\propto$ volume$^{0.34}$",
             ha="left", va="top", fontsize=6.4, color=FAINT)
    axf.set_title("Impression to order, one window", fontsize=9.0,
                  fontweight="bold", pad=2.0)

    # ---- (b) conversion by segment ----------------------------------------
    axh = axes_inches(fig, (6.30, 2.28, 2.86, 2.72))
    rates = segment_rates(rng)
    lift = 100.0 * (rates / rates.mean(axis=0, keepdims=True) - 1.0)
    norm = shared_norm([lift], symmetric=True)
    im = axh.imshow(lift, cmap=get_cmap("cb_diverging"), norm=norm,
                    aspect="auto")
    axh.set_xticks(range(len(STEPS)))
    axh.set_xticklabels(STEPS, rotation=18, ha="right")
    axh.set_yticks(range(len(SEGMENTS)))
    axh.set_yticklabels(SEGMENTS)
    axh.tick_params(labelsize=6.8, length=2.0)
    for i in range(len(SEGMENTS)):
        for j in range(len(STEPS)):
            deep = abs(lift[i, j]) > 0.48 * norm.vmax
            axh.text(j, i - 0.13, f"{100 * rates[i, j]:.1f}", ha="center",
                     va="center", fontsize=6.8,
                     color="white" if deep else INK)
            axh.text(j, i + 0.19, f"{lift[i, j]:+.0f} %", ha="center",
                     va="center", fontsize=6.0,
                     color="#e8e8e8" if deep else SUB)
    axh.set_title("Step conversion by segment", fontsize=8.5,
                  fontweight="bold", pad=3.0)
    for s in axh.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    add_colorbar(fig, im, axh, label="lift vs column mean (%)", aspect=16,
                 ticks=[-30, 0, 30])

    # ---- (c) daily trend per stage ----------------------------------------
    day, series = trends(rng)
    index = [100.0 * s / s[0] for s in series]
    lo, hi = shared_limits(index, pad=0.08)
    axes_row = []
    for i, (name, idx_s, col) in enumerate(zip(STAGES, index, SLAB)):
        axm = axes_inches(fig, (0.72 + i * 2.40, 0.62, 2.06, 1.02))
        axes_row.append(axm)
        axm.axhline(100.0, color="#808080", lw=0.8, ls=(0, (3.0, 2.0)),
                    zorder=2)
        axm.fill_between(day, 100.0, idx_s, facecolor=lighten(col, 0.72),
                         edgecolor="none", zorder=3)
        axm.plot(day, idx_s, lw=1.3, color=col, zorder=4)
        axm.set_ylim(lo, hi)
        axm.set_xlim(1, DAYS)
        axm.set_xticks([1, 10, 20, 30])
        axm.tick_params(labelsize=6.5, length=2.2)
        axm.grid(axis="y", color=GRID, linewidth=0.6)
        axm.set_xlabel("day (d)", fontsize=7.0)
        drift = idx_s[-1] - 100.0
        axm.set_title(f"{name}   {drift:+.0f} %", fontsize=8.0,
                      fontweight="bold", color=darken(col, 0.32), pad=2.5)
        if i:
            axm.tick_params(labelleft=False)
        else:
            axm.set_ylabel("index, day 1 = 100", fontsize=7.0)
        for s in ("top", "right"):
            axm.spines[s].set_visible(False)

    end2end = 100.0 * vols[-1] / vols[0]
    panel_label_inches(fig, axf, "(a)", dx=-0.12, dy=-0.06)
    panel_label_inches(fig, axh, "(b)", dx=-0.58, dy=0.10)
    panel_label_inches(fig, axes_row[0], "(c)", dx=-0.58, dy=0.10)
    savefig(fig, "479_rec_funnel_metrics.png")


if __name__ == "__main__":
    main()

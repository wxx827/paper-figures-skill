"""Figure 1192 - Quantisation: rollout throughput against rollout quality.

Eight quantisation recipes, one table of measurements, two views:

    (a) scatter over (throughput, quality) with the Pareto front joined up
    (b) the same eight rows, with memory and the quality delta from FP16

The front is *computed*, not drawn by eye: a recipe is on it when no other
recipe is at least as fast and at least as good.  The two recipes that fail
that test are named in the footer together with the point that beats them and
the two comparisons that settle it, so the reader can check the verdict
without re-reading the scatter.

Every delta in (b) is the subtraction of two *printed* qualities, so the
column adds up against the axis labels in (a) digit for digit.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

from pubstyle import set_pub_style, savefig, style_axes
from _wm_metrics import closed_ratio

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#d5d5d5"
GRID, CARD = "#ededed", "#e2e8ee"
# Row banding is the deeper card grey on purpose: 0.906 luminance sits clearly
# below the 0.94 ink threshold, so the table reads as content under either
# ruler instead of flickering between "all ink" and "all paper".
ROW = "#e2e8ee"
# weight bit-width -> one step of the blue ladder, deepest = most precise
BITCOL = {16: "#08263f", 8: "#2e74a8", 6: "#5f9dc6", 4: "#9cc3dd",
          3: "#cfe1ee"}
FRONT = "#0b3d63"

# name, weight bits, throughput (clips/s), quality, memory (GB), label offset
RECIPES = [
    ("FP16  W16A16", 16, 42.0, 0.941, 26.8, (0.0, -0.011, "center")),
    ("W6A16", 6, 52.4, 0.929, 11.2, (-1.6, -0.010, "right")),
    ("W8A16", 8, 55.2, 0.940, 14.9, (0.0, 0.008, "center")),
    ("W4A16", 4, 63.0, 0.918, 8.1, (1.8, -0.007, "left")),
    ("W8A8", 8, 71.4, 0.936, 13.7, (2.0, 0.005, "left")),
    ("W4A8 mixed", 4, 88.2, 0.913, 7.9, (2.0, 0.006, "left")),
    ("W4A8 uniform", 4, 96.6, 0.874, 7.4, (2.0, 0.007, "left")),
    ("W3A8", 3, 104.2, 0.802, 6.2, (-2.4, 0.008, "right")),
]
DP_Q = 3


def pareto(rows):
    """Index -> True when nothing else is both at least as fast and as good."""
    out = {}
    for i, a in enumerate(rows):
        out[i] = not any(
            j != i and b["tp"] >= a["tp"] and b["q"] >= a["q"]
            and (b["tp"] > a["tp"] or b["q"] > a["q"]) for j, b in
            enumerate(rows))
    return out


def main():
    set_pub_style(base_font=8.5)

    rows = [dict(name=n, bits=b, tp=t, q=q, mem=m, off=o)
            for (n, b, t, q, m, o) in RECIPES]
    on = pareto(rows)
    base = rows[0]
    for i, r in enumerate(rows):         # deltas from the printed qualities
        r["dq"] = float(f"{base['q']:.{DP_Q}f}") - float(f"{r['q']:.{DP_Q}f}")
        r["front"] = on[i]
    losers = [r for r in rows if not r["front"]]
    assert len(losers) == 2, "the number of dominated recipes changed"
    beats = []
    for r in losers:
        w = max((b for b in rows if b["tp"] >= r["tp"] and b["q"] >= r["q"]
                 and b is not r), key=lambda b: b["q"])
        beats.append((r, w))
    assert abs(rows[5]["dq"] - 0.028) < 1e-9, \
        "the headline quality gap no longer matches the table"

    fig = plt.figure(figsize=(10.4, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.66], left=0.070,
                          right=0.988, top=0.800, bottom=0.250, wspace=0.16)
    axa = fig.add_subplot(gs[0, 0])
    axt = fig.add_subplot(gs[0, 1])

    # ---- (a) the trade-off plane -------------------------------------------
    style_axes(axa)
    axa.set_axisbelow(True)
    axa.grid(color=GRID, lw=0.7, zorder=0)
    front = sorted([r for r in rows if r["front"]], key=lambda r: r["tp"])
    fx = [r["tp"] for r in front]
    fq = [r["q"] for r in front]
    # everything under the polyline is reachable but beaten by it
    axa.fill_between(fx, 0.784, fq, facecolor="#dfeaf3", edgecolor="none",
                     zorder=1)
    axa.plot(fx, fq, color=FRONT, lw=1.5, zorder=3, solid_capstyle="round")
    for r, w in beats:                    # the comparison that rules it out
        axa.plot([r["tp"], w["tp"]], [r["q"], w["q"]], color="#9a9a9a",
                 lw=0.8, ls=(0, (2.2, 1.8)), zorder=2)
    for r in rows:
        axa.plot([r["tp"]], [r["q"]], marker="o", ms=9.5 if r["front"] else 8.0,
                 mfc=BITCOL[r["bits"]] if r["front"] else "white",
                 mec=FRONT if r["front"] else "#8a8a8a",
                 mew=1.5 if r["front"] else 1.2, zorder=6)
        if not r["front"]:                # keep the hue, drop the fill
            axa.plot([r["tp"]], [r["q"]], marker="o", ms=3.4,
                     mfc=BITCOL[r["bits"]], mec="none", zorder=7)
        dx, dy, ha = r["off"]
        axa.text(r["tp"] + dx, r["q"] + dy, r["name"],
                 ha=ha, va="center", fontsize=7.6,
                 color=INK if r["front"] else MUTED,
                 fontweight="bold" if r["front"] else "normal", zorder=8,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                           alpha=0.90))
    axa.set_xlim(33.0, 118.0)
    axa.set_ylim(0.784, 0.960)
    axa.set_xlabel("rollout throughput  (clips / s, batch 8)", fontsize=8.6)
    axa.set_ylabel("rollout quality  (action-conditioned score)",
                   fontsize=8.6)
    axa.tick_params(labelsize=7.4, length=2.4)
    axa.set_title("(a)  Throughput against quality, Pareto front joined up",
                  fontsize=9.4, fontweight="bold", pad=5.0, loc="left")
    axa.text(0.018, 0.055, "a recipe is on the front when nothing else is "
             "both faster and\nbetter; the shaded region is what the front "
             "already beats", transform=axa.transAxes, ha="left",
             va="bottom", fontsize=7.4, color=SUB, linespacing=1.5, zorder=9,
             bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=HAIR,
                       lw=0.6, alpha=0.95))

    handles = [Line2D([], [], color=FRONT, lw=1.5, marker="o", ms=8.0,
                      mfc=BITCOL[8], mec=FRONT, mew=1.5,
                      label="on the front"),
               Line2D([], [], color="#9a9a9a", lw=0.8, ls=(0, (2.2, 1.8)),
                      marker="o", ms=7.0, mfc="white", mec="#8a8a8a", mew=1.2,
                      label="dominated, joined to the recipe that beats it")]
    handles += [Line2D([], [], ls="none", marker="o", ms=7.5,
                       mfc=BITCOL[b], mec=FRONT, mew=1.0,
                       label=f"{b}-bit weights") for b in (16, 8, 6, 4, 3)]
    leg = fig.legend(handles=handles, loc="lower center",
                     bbox_to_anchor=(0.5, 0.093), ncol=4, frameon=True,
                     facecolor="white", edgecolor=HAIR, framealpha=0.95,
                     fontsize=7.6, handlelength=1.8, columnspacing=1.4,
                     borderpad=0.42, labelspacing=0.35)
    leg.get_frame().set_linewidth(0.5)

    # ---- (b) the same eight rows -------------------------------------------
    axt.set_xlim(0.0, 1.0)
    axt.set_ylim(0.0, 1.0)
    axt.set_axis_off()
    axt.set_title("(b)  Memory and the quality lost against FP16",
                  fontsize=9.4, fontweight="bold", pad=5.0, loc="left")
    n = len(rows)
    top, rh = 0.925, 0.925 / (n + 1.35)
    mmax = max(r["mem"] for r in rows)
    x_bar, w_bar = 0.365, 0.335
    axt.text(0.012, top + 0.030, "recipe", fontsize=7.6, color=SUB,
             fontweight="bold", ha="left", va="bottom")
    axt.text(x_bar, top + 0.030, "memory  (GB)", fontsize=7.6, color=SUB,
             fontweight="bold", ha="left", va="bottom")
    axt.text(0.995, top + 0.030, "$\\Delta$ quality vs FP16", fontsize=7.6,
             color=SUB, fontweight="bold", ha="right", va="bottom")
    axt.plot([0.0, 1.0], [top + 0.018, top + 0.018], color="#b8c4ce", lw=0.8)
    for i, r in enumerate(rows):
        yc = top - (i + 0.5) * rh
        if i % 2 == 0:
            axt.add_patch(Rectangle((0.0, yc - rh / 2), 1.0, rh,
                                    facecolor=ROW, edgecolor="none",
                                    zorder=0))
        axt.plot([0.030], [yc], marker="o", ms=6.4 if r["front"] else 5.6,
                 mfc=BITCOL[r["bits"]] if r["front"] else "white",
                 mec=FRONT if r["front"] else "#8a8a8a",
                 mew=1.3 if r["front"] else 1.1, zorder=3,
                 clip_on=False)
        axt.text(0.060, yc, r["name"], fontsize=7.8, va="center", ha="left",
                 color=INK if r["front"] else MUTED,
                 fontweight="bold" if r["front"] else "normal", zorder=3)
        w = w_bar * r["mem"] / mmax
        axt.add_patch(Rectangle((x_bar, yc - rh * 0.26), w, rh * 0.52,
                                facecolor=BITCOL[r["bits"]],
                                edgecolor="white", linewidth=0.6, zorder=3))
        axt.text(x_bar + w + 0.012, yc, f"{r['mem']:.1f}", fontsize=7.6,
                 va="center", ha="left", color=INK, zorder=3)
        axt.text(0.995, yc, "-" if r["dq"] == 0 else f"-{r['dq']:.{DP_Q}f}",
                 fontsize=7.8, va="center", ha="right", zorder=3,
                 color=INK if r["dq"] < 0.03 else SUB,
                 fontweight="bold" if r["dq"] < 0.03 else "normal")
    axt.plot([0.0, 1.0], [top - n * rh - 0.012, top - n * rh - 0.012],
             color="#b8c4ce", lw=0.8)
    axt.text(0.012, top - n * rh - 0.055, "bold rows keep the front; bold "
             "deltas stay within 0.030 of FP16", fontsize=7.2, color=MUTED,
             ha="left", va="top")

    # ---- headline and the verdicts -----------------------------------------
    pick = rows[5]
    ns, ds, q, _ = closed_ratio(base["mem"], pick["mem"])

    band = FancyBboxPatch((0.022, 0.012), 0.956, 0.078,
                          boxstyle="round,pad=0.004,rounding_size=0.008",
                          transform=fig.transFigure, facecolor=CARD,
                          edgecolor="#c3cdd6", linewidth=0.6, zorder=-5)
    fig.add_artist(band)
    band.set_in_layout(False)
    verdict = [f"{r['name']} is dominated by {w['name']} "
               f"({w['tp']:.1f} > {r['tp']:.1f} clips / s and "
               f"{w['q']:.{DP_Q}f} > {r['q']:.{DP_Q}f} quality)"
               for r, w in beats]
    savefig(fig, "1192_wm_sys_quantization.png")


if __name__ == "__main__":
    main()

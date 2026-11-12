"""Figure 681 - Flexible assembly line under a mixed-variant order.

Main panel: a U-shaped ten-station cell drawn on an inch canvas, so the
station boxes, the material-flow arrows and the AGV corridor all share one
ruler and the rounded corners stay circular (DESIGN_SYSTEM 12.8).  Station
fill encodes the load the current order actually puts on it; the bottleneck
carries the only red outline in the figure.

Right column at 56 % width: (b) the changeover-visible schedule for three
cells over one shift, (c) how the balance rate reacts to the variant mix.

One process-time matrix drives all three panels.  Station load is the mix
weighted sum of that matrix, the balance rate is mean load over bottleneck
load, and the takt printed on the layout is the reciprocal of the same
bottleneck, so the panels cannot disagree (DESIGN_SYSTEM 13.2).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from pubstyle import set_pub_style, savefig, get_cmap, lighten
from figkit import (axes_inches, inch_axes, block, arrow, route,
                    safe_colorbar, panel_label_inches, lowess)

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#cccccc"
GRID, BASE = "#d9d9d9", "#808080"
BLUE, RED, GREEN = "#3b5b92", "#b23a48", "#2e7d5b"
GOLD, PURPLE, SOLID = "#c9a227", "#8e6bbf", "#3a3a3a"

N_ST = 10                   # stations OP10 .. OP100
VARIANTS = ["A", "B", "C", "D"]
VAR_COLOR = [BLUE, GREEN, GOLD, PURPLE]
CELLS = ["Cell 1", "Cell 2", "Cell 3"]
SHIFT_H = 8.0
TARGET_BAL = 85.0
ORDER_MIX = np.array([0.40, 0.28, 0.20, 0.12])


def balance(load):
    return 100.0 * load.mean() / load.max()


def evenness(w):
    """Normalised entropy of a mix, 0 % single variant to 100 % even."""
    p = w[w > 0]
    return 100.0 * float(-(p * np.log(p)).sum() / np.log(len(w)))


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(681)
    fig = plt.figure(figsize=(10.4, 6.4))

    # ---- one process-time matrix drives every panel ----------------------
    proc = 1.4 + 1.5 * rng.random((len(VARIANTS), N_ST))   # min per unit
    proc[:, 4] *= 1.35                                     # a heavy operation
    proc[:, 7] *= 1.22
    load = ORDER_MIX @ proc                                # min per unit
    util = 100.0 * load / load.max()
    bott = int(np.argmax(load))
    takt = float(load.max())
    thr = 60.0 / takt                                      # units per hour
    bal_now = balance(load)

    cmap = get_cmap("cb_sequential_blue")
    norm = mpl.colors.Normalize(vmin=float(util.min()) - 6.0, vmax=100.0)

    # =====================================================================
    # (a) line layout
    # =====================================================================
    axl = inch_axes(fig, (0.72, 1.06, 5.30, 4.06))
    xs = np.array([0.80, 1.72, 2.64, 3.56, 4.48])
    y_top, y_bot, bw, bh = 3.14, 1.28, 0.72, 0.60
    corridor = 2.21

    boxes = []
    for i in range(N_ST):
        top = i < 5
        x = xs[i] if top else xs[9 - i]
        y = y_top if top else y_bot
        u = util[i]
        fc = cmap(0.24 + 0.68 * norm(u))
        a = block(axl, x, y, bw, bh, label="OP%d" % (10 * (i + 1)),
                  color=fc, sub="%.0f %%" % u, fontsize=7.4,
                  ec=RED if i == bott else "white",
                  lw=1.8 if i == bott else 0.9, sub_color="#f0f0f0")
        boxes.append(a)

    for i in range(4):
        arrow(axl, boxes[i]["r"], boxes[i + 1]["l"], color=SOLID, lw=1.1,
              ms=9, gap=0.03)
    route(axl, [boxes[4]["r"], (5.12, y_top), (5.12, y_bot), boxes[5]["r"]],
          color=SOLID, lw=1.1, ms=9, gap=0.03)
    for i in range(5, 9):
        arrow(axl, boxes[i]["l"], boxes[i + 1]["r"], color=SOLID, lw=1.1,
              ms=9, gap=0.03)
    arrow(axl, (0.16, y_top), boxes[0]["l"], color=SOLID, lw=1.1, ms=9,
          gap_b=0.03)
    arrow(axl, boxes[9]["l"], (0.16, y_bot), color=SOLID, lw=1.1, ms=9,
          gap_a=0.03)
    axl.text(0.10, y_top + 0.20, "raw material", ha="left", va="bottom",
             fontsize=6.4, color=SUB)
    axl.text(0.10, y_bot - 0.36, "finished units", ha="left", va="top",
             fontsize=6.4, color=SUB)

    axl.plot([0.30, 5.12], [corridor, corridor], lw=1.3, color=GOLD,
             ls=(0, (5.0, 3.0)), zorder=2)
    for x in xs:
        axl.plot([x, x], [corridor, y_top - bh / 2], lw=0.8, color=GOLD,
                 ls=(0, (2.0, 2.0)), alpha=0.8, zorder=2)
        axl.plot([x, x], [y_bot + bh / 2, corridor], lw=0.8, color=GOLD,
                 ls=(0, (2.0, 2.0)), alpha=0.8, zorder=2)
    for x in (1.26, 4.02):
        axl.plot([x], [corridor], marker="s", ms=5.0, mfc=GOLD, mec="white",
                 mew=0.8, zorder=5)
    axl.text(5.10, corridor + 0.10, "AGV corridor, 2 vehicles", ha="right",
             va="bottom", fontsize=6.4, color=GOLD)

    axl.text(0.10, 0.34, "takt %.2f min/unit at OP%d   |   throughput "
             "%.1f units/h   |   balance %.1f %%"
             % (takt, 10 * (bott + 1), thr, bal_now), ha="left", va="bottom",
             fontsize=7.0, color=INK)
    axl.text(0.10, 0.10, "red outline marks the bottleneck station",
             ha="left", va="bottom", fontsize=6.4, color=RED)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    safe_colorbar(fig, sm, axl, label="station load under the current order "
                  "(% of bottleneck)", side="bottom", width=0.11, pad=0.06,
                  shrink=0.72, labelsize=6.8)
    axl.set_title("Ten-station U cell running a four-variant order",
                  fontsize=9.0, fontweight="bold", pad=4.0)

    # =====================================================================
    # (b) changeover-visible shift schedule
    # =====================================================================
    axg = axes_inches(fig, (7.02, 3.44, 2.94, 1.76))
    co = 8.0 + 34.0 * rng.random((len(VARIANTS), len(VARIANTS)))
    np.fill_diagonal(co, 0.0)
    co_total = 0.0
    for r, cell in enumerate(CELLS):
        seq = list(rng.permutation(len(VARIANTS)))
        t = 0.0
        prev = None
        while t < SHIFT_H * 60.0:
            v = seq[0]
            seq = seq[1:] + [v]
            if prev is not None:
                d = co[prev, v]
                d = min(d, SHIFT_H * 60.0 - t)
                axg.barh(r, d / 60.0, left=t / 60.0, height=0.56,
                         color=lighten(RED, 0.55), edgecolor=RED,
                         linewidth=0.7, hatch="///", zorder=3)
                t += d
                co_total += d
            run = min(58.0 + 62.0 * rng.random(), SHIFT_H * 60.0 - t)
            if run <= 0.0:
                break
            axg.barh(r, run / 60.0, left=t / 60.0, height=0.56,
                     color=VAR_COLOR[v], edgecolor="white", linewidth=0.7,
                     zorder=3)
            if run > 42.0:
                axg.text((t + run / 2.0) / 60.0, r, VARIANTS[v], ha="center",
                         va="center", fontsize=6.6, color="white",
                         fontweight="bold", zorder=5)
            t += run
            prev = v
    co_share = 100.0 * co_total / (len(CELLS) * SHIFT_H * 60.0)

    axg.set_yticks(np.arange(len(CELLS)))
    axg.set_yticklabels(CELLS, fontsize=6.5)
    axg.invert_yaxis()
    axg.set_xlim(0.0, SHIFT_H)
    axg.set_xticks(np.arange(0.0, SHIFT_H + 0.1, 1.0))
    axg.set_xlabel("shift time (h)", fontsize=7.4)
    axg.tick_params(labelsize=6.5, length=2.2)
    axg.grid(axis="x", color=GRID, lw=0.55)
    axg.set_title("Shift schedule, hatched = changeover (%.1f %%)"
                  % co_share, fontsize=8.5, fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        axg.spines[s].set_visible(False)

    # =====================================================================
    # (c) balance rate against mix evenness
    # =====================================================================
    axb = axes_inches(fig, (7.02, 0.95, 2.94, 1.76))
    mixes = rng.dirichlet(np.full(len(VARIANTS), 0.55), 220)
    ev = np.array([evenness(w) for w in mixes])
    bl = np.array([balance(w @ proc) for w in mixes])
    axb.scatter(ev, bl, s=7.0, color=lighten(BLUE, 0.45), edgecolor="none",
                zorder=2)
    gx, gy = lowess(ev, bl, frac=0.45, n_out=120)
    axb.plot(gx, gy, lw=1.7, color=BLUE, zorder=4)
    axb.axhline(TARGET_BAL, color=GREEN, lw=1.1, ls=(0, (4.0, 2.2)), zorder=3)
    axb.text(2.0, TARGET_BAL + 0.5, "%.0f %% balance target" % TARGET_BAL,
             ha="left", va="bottom", fontsize=6.4, color=GREEN)
    ev_now = evenness(ORDER_MIX)
    axb.plot([ev_now], [bal_now], marker="*", ms=11.0, mfc=RED, mec="white",
             mew=0.8, zorder=6)
    axb.annotate("current order, %.0f %% even" % ev_now,
                 xy=(ev_now, bal_now), xytext=(-6.0, -8.0),
                 textcoords="offset points", ha="right", va="top",
                 fontsize=6.4, color=RED)
    axb.set_xlim(0.0, 101.0)
    axb.set_xlabel("mix evenness, normalised entropy (%)", fontsize=7.4)
    axb.set_ylabel("line balance rate (%)", fontsize=7.4)
    axb.tick_params(labelsize=6.5, length=2.2)
    axb.grid(axis="y", color=GRID, lw=0.55)
    axb.set_title("Balance over %d sampled mixes" % mixes.shape[0],
                  fontsize=8.5, fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)

    for ax, lab in ((axl, "(a)"), (axg, "(b)"), (axb, "(c)")):
        panel_label_inches(fig, ax, lab, dx=-0.46, dy=0.12)
    savefig(fig, "681_mfg_flexible_line.png")


if __name__ == "__main__":
    main()

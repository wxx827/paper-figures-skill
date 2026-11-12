"""Weave construction sheet: design, threading, peg plan and appearance.

The three basic weaves are generated from one rule each, and everything else on
the page is counted off those Boolean repeat matrices: the threading is the
straight draw implied by the rule, the peg plan is read back out of the design,
the float lengths are periodic run lengths in the repeat, and the interlacing
density is the number of warp/weft changes per cell.  The design grids are
drawn on an inch canvas with equal aspect so a cell is a true square, which is
the only way a weave diagram is readable.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, inch_axes, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
DARK = "#3a3a3a"
WARP = "#3b5b92"
WEFT = "#c9a227"
RED = "#b23a48"
GREEN = "#2e7d5b"
CAT = [RED, WARP, GREEN]

CELL = 0.14
NDESIGN = 8
APP_CELL = 0.15

WEAVES = [
    dict(key="plain", name="Plain 1/1", shafts=2, rep=2),
    dict(key="twill", name="Twill 2/2 Z", shafts=4, rep=4),
    dict(key="satin", name="Satin 5-end, step 2", shafts=5, rep=5),
]


def weave(key, rows, cols):
    """Warp-up mask; row = pick (weft), column = end (warp)."""
    i = np.arange(rows)[:, None]
    j = np.arange(cols)[None, :]
    if key == "plain":
        return ((i + j) % 2) == 0
    if key == "twill":
        return ((j - i) % 4) < 2
    return ((j - 2 * i) % 5) != 0


def periodic_runs(v):
    """Run lengths of True in a cyclic Boolean vector."""
    v = np.asarray(v, dtype=bool)
    n = v.size
    if v.all():
        return [n]
    if not v.any():
        return []
    v = np.roll(v, -int(np.argmax(~v)))
    runs, c = [], 0
    for x in v:
        if x:
            c += 1
        elif c:
            runs.append(c)
            c = 0
    if c:
        runs.append(c)
    return runs


def repeat_stats(key, rep):
    r = weave(key, rep, rep)
    warp = [L for c in range(rep) for L in periodic_runs(r[:, c])]
    weft = [L for i in range(rep) for L in periodic_runs(~r[i, :])]
    changes = 0
    for c in range(rep):
        changes += int(np.sum(r[:, c] != np.roll(r[:, c], -1)))
    for i in range(rep):
        changes += int(np.sum(r[i, :] != np.roll(r[i, :], -1)))
    return dict(warp=np.array(warp, dtype=float),
                weft=np.array(weft, dtype=float),
                density=changes / (2.0 * rep * rep))


def grid_block(ax, x0, y0, mask, cell, fill, edge="#a8a8a8", lw=0.45,
               border=SUB, blw=0.9):
    nrow, ncol = mask.shape
    for r in range(nrow):
        for c in range(ncol):
            ax.add_patch(Rectangle(
                (x0 + c * cell, y0 + r * cell), cell, cell,
                facecolor=fill if mask[r, c] else "white",
                edgecolor=edge, linewidth=lw, zorder=2))
    ax.add_patch(Rectangle((x0, y0), ncol * cell, nrow * cell,
                           facecolor="none", edgecolor=border,
                           linewidth=blw, zorder=4))


def appearance(ax, x0, y0, mask, cell, rng):
    n = mask.shape[0]
    inset = cell * 0.11
    ax.add_patch(Rectangle((x0, y0), n * cell, n * cell, facecolor="#f5f5f5",
                           edgecolor="none", zorder=1))
    for r in range(n):
        for c in range(n):
            if mask[r, c]:
                ax.add_patch(Rectangle(
                    (x0 + c * cell + inset, y0 + r * cell),
                    cell - 2 * inset, cell,
                    facecolor=lighten(WARP, float(rng.uniform(0.0, 0.13))),
                    edgecolor="none", zorder=2))
            else:
                ax.add_patch(Rectangle(
                    (x0 + c * cell, y0 + r * cell + inset),
                    cell, cell - 2 * inset,
                    facecolor=lighten(WEFT, float(rng.uniform(0.0, 0.13))),
                    edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((x0, y0), n * cell, n * cell, facecolor="none",
                           edgecolor=SUB, linewidth=0.9, zorder=5))


def grouped_bars(ax, groups, series, colors, hatches=None, width=0.78):
    pos = np.arange(len(groups), dtype=float)
    k = len(series)
    w = width / k
    for s, (label, vals) in enumerate(series):
        off = -width / 2.0 + w * (s + 0.5)
        ax.bar(pos + off, vals, width=w * 0.92, color=colors[s],
               edgecolor="white", linewidth=0.5, label=label, zorder=3,
               hatch=None if hatches is None else hatches[s])
    ax.set_xticks(pos)
    ax.set_xticklabels(groups, fontsize=6.5, color=INK)
    return pos


def main():
    set_pub_style()
    rng = np.random.default_rng(834)

    stats = {w["key"]: repeat_stats(w["key"], w["rep"]) for w in WEAVES}
    base_float = float(np.mean(np.concatenate(
        [stats["plain"]["warp"], stats["plain"]["weft"]])))
    base_dens = stats["plain"]["density"]

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = inch_axes(fig, (0.28, 0.86, 6.85, 4.56))
    ax_fl = axes_inches(fig, (7.80, 3.32, 2.26, 1.78))
    ax_tg = axes_inches(fig, (7.80, 0.96, 2.26, 1.78))

    y_des_top, y_des_bot = 3.36, 3.36 - NDESIGN * CELL
    y_thr_bot = y_des_top + 0.09
    y_app_bot = 0.50
    x_cols = [0.42, 2.58, 4.74]

    for col, (w, x0) in enumerate(zip(WEAVES, x_cols)):
        key, ns, rep = w["key"], w["shafts"], w["rep"]
        design = weave(key, NDESIGN, NDESIGN)
        thread = np.zeros((ns, NDESIGN), dtype=bool)
        for j in range(NDESIGN):
            thread[j % ns, j] = True
        peg = weave(key, NDESIGN, ns)

        grid_block(ax_main, x0, y_des_bot, design, CELL, DARK)
        ax_main.add_patch(Rectangle(
            (x0, y_des_bot + (NDESIGN - rep) * CELL), rep * CELL, rep * CELL,
            facecolor="none", edgecolor=RED, linewidth=1.25, zorder=6))
        grid_block(ax_main, x0, y_thr_bot, thread, CELL, WARP)
        x_peg = x0 + NDESIGN * CELL + 0.10
        grid_block(ax_main, x_peg, y_des_bot, peg, CELL, GREEN)

        ax_main.text(x0 - 0.07, y_thr_bot + ns * CELL / 2.0, "threading",
                     rotation=90, ha="right", va="center", fontsize=6.0,
                     color=MUTED)
        ax_main.text(x0 - 0.07, y_des_bot + NDESIGN * CELL / 2.0, "design",
                     rotation=90, ha="right", va="center", fontsize=6.0,
                     color=MUTED)
        ax_main.text(x_peg, y_des_top + 0.09, "peg plan", ha="left",
                     va="bottom", fontsize=6.0, color=MUTED)
        ax_main.text(x0, 4.52, w["name"], ha="left", va="top", fontsize=8.6,
                     fontweight="bold", color=INK)
        ax_main.text(x0, 4.36, "%d shafts   repeat %d x %d   %d picks/repeat"
                     % (ns, rep, rep, rep), ha="left", va="top", fontsize=6.2,
                     color=MUTED)

        appearance(ax_main, x0, y_app_bot, weave(key, 8, 8), APP_CELL, rng)
        st = stats[key]
        ax_main.text(x0, y_app_bot + 8 * APP_CELL + 0.07,
                     "appearance  (warp float %d, weft float %d)"
                     % (int(st["warp"].max()), int(st["weft"].max())),
                     ha="left", va="bottom", fontsize=6.2, color=SUB)

    ax_main.text(x_cols[0], y_des_bot - 0.15,
                 "filled = warp end over pick        red outline = one repeat",
                 ha="left", va="top", fontsize=6.4, color=SUB)
    ax_main.set_ylim(0.0, 4.56)

    names = ["Plain\n1/1", "Twill\n2/2", "Satin\n5-end"]
    mean_warp = [float(stats[w["key"]]["warp"].mean()) for w in WEAVES]
    mean_weft = [float(stats[w["key"]]["weft"].mean()) for w in WEAVES]
    longest = [float(max(stats[w["key"]]["warp"].max(),
                         stats[w["key"]]["weft"].max())) for w in WEAVES]
    grouped_bars(ax_fl, names,
                 [("mean warp float", mean_warp),
                  ("mean weft float", mean_weft),
                  ("longest float", longest)],
                 [WARP, WEFT, "#8a8a8a"])
    for i, v in enumerate(longest):
        ax_fl.text(i + 0.26, v + 0.09, "%d" % v, ha="center", va="bottom",
                   fontsize=6.2, color=SUB)
    ax_fl.set_ylim(0.0, max(longest) * 1.30)
    ax_fl.set_ylabel("float length (yarns crossed)", fontsize=7.4)
    ax_fl.tick_params(axis="y", labelsize=6.5)
    ax_fl.legend(fontsize=6.2, frameon=False, ncol=1, loc="upper left",
                 handlelength=1.2, borderpad=0.2, labelspacing=0.28)
    ax_fl.set_title("Float length grows with the repeat", fontsize=8.5,
                    fontweight="bold", color=INK)

    dens = [100.0 * stats[w["key"]]["density"] / base_dens for w in WEAVES]
    tear = [100.0 * float(np.mean(np.concatenate(
        [stats[w["key"]]["warp"], stats[w["key"]]["weft"]]))) / base_float
        for w in WEAVES]
    grouped_bars(ax_tg, names,
                 [("interlacing density", dens),
                  ("tear index (mean float)", tear)],
                 [RED, GREEN], width=0.62)
    for i, (d, t) in enumerate(zip(dens, tear)):
        ax_tg.text(i - 0.155, d + 5.0, "%.0f" % d, ha="center", va="bottom",
                   fontsize=6.2, color=SUB)
        ax_tg.text(i + 0.155, t + 5.0, "%.0f" % t, ha="center", va="bottom",
                   fontsize=6.2, color=SUB)
    ax_tg.axhline(100.0, color="#808080", lw=0.8, ls=(0, (3.0, 2.0)),
                  zorder=2)
    ax_tg.text(2.42, 103.0, "plain = 100", ha="right", va="bottom",
               fontsize=6.2, color=MUTED)
    ax_tg.set_ylim(0.0, max(tear) * 1.24)
    ax_tg.set_ylabel("relative to plain weave (%)", fontsize=7.4)
    ax_tg.tick_params(axis="y", labelsize=6.5)
    ax_tg.legend(fontsize=6.2, frameon=False, ncol=1, loc="upper left",
                 handlelength=1.2, borderpad=0.2, labelspacing=0.28)
    ax_tg.set_title("Fewer interlacings, looser cloth", fontsize=8.5,
                    fontweight="bold", color=INK)


    panel_labels_inches(fig, [ax_main], labels=["(a)"], dx=0.02, dy=0.10)
    panel_labels_inches(fig, [ax_fl], labels=["(b)"], dx=-0.44, dy=0.14)
    panel_labels_inches(fig, [ax_tg], labels=["(c)"], dx=-0.44, dy=0.14)
    savefig(fig, "834_txt_weave_pattern.png")


if __name__ == "__main__":
    main()

"""Grouped success-rate bars for five policies over seven manipulation tasks.

The seven tasks fall into three evaluation regimes, marked by a bracketed group
header above the panel.  A narrow "Average" panel on a grey card carries the
across-task mean for the same five policies, on exactly the same y range as the
main panel so the two can be read against each other.

Synthetic data lives in ``gen_scores`` only -- swap that one function for a real
loader and leave the styling untouched.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, style_axes
from figkit import background_card

METHODS = ["Diffusion Policy", "RT-2-X", "Octo", "OpenVLA", "Ours (WM-VLA)"]
# cool-blue ramp for the baselines, one warm accent reserved for Ours
COL = {"Diffusion Policy": "#5f9dc6", "RT-2-X": "#2a7f7a", "Octo": "#8e6bbf",
       "OpenVLA": "#3b5b92", "Ours (WM-VLA)": "#b23a48"}

# (group label, [(task label, mean success rate of the five methods)])
GROUPS = [
    ("Narrow Single-Instruction", [
        ("Put Carrot\nin Bowl", 58.0),
        ("Flip Pot\nUpright", 62.0),
    ]),
    ("Diverse Multi-Instruction", [
        ("Move <object>\nonto Plate", 34.0),
        ("Knock <object>\nOver", 47.0),
        ("Cover <object>\nwith Towel", 41.0),
    ]),
    ("Visual Robustness", [
        ("Unseen\nDistractors", 39.0),
        ("Novel Table\nSurface", 30.0),
    ]),
]

# method -> (skill offset in points, spread of the per-task wobble)
SKILL = {
    "Diffusion Policy": (-14.0, 7.0),
    "RT-2-X": (-8.5, 6.5),
    "Octo": (-2.0, 6.0),
    "OpenVLA": (7.0, 6.0),
    "Ours (WM-VLA)": (19.0, 5.0),
}

GRID = "#ededed"
GAP = 0.85          # blank slots inserted between task groups
BARW = 0.165
VAL_FS = 6.2


def gen_scores(seed=1499):
    """{method: array over the seven tasks} of success rates in percent."""
    rng = np.random.default_rng(seed)
    base = np.array([lv for _, tasks in GROUPS for _, lv in tasks])
    out = {}
    for m in METHODS:
        off, spread = SKILL[m]
        v = base + off + rng.normal(0.0, spread, base.size)
        out[m] = np.clip(v, 4.0, 96.0)
    return out


def bar_group(ax, centers, scores, width=BARW, val_fs=VAL_FS):
    """Five bars per centre, value printed on top of each bar."""
    n = len(METHODS)
    for j, m in enumerate(METHODS):
        off = (j - (n - 1) / 2.0) * width
        v = scores[m]
        ax.bar(centers + off, v, width * 0.94, color=COL[m], lw=0.0,
               zorder=3, label=m)
        # bar pitch is narrower than a horizontal "88.8" at 6 pt, so the
        # per-bar values are set upright instead of letting them touch
        for xc, vi in zip(centers + off, np.atleast_1d(v)):
            ax.text(xc, vi + 1.6, f"{vi:.1f}", ha="center", va="bottom",
                    rotation=90, fontsize=val_fs, color="#333333", zorder=4)


def group_header(ax, x0, x1, label, y=1.045, tick=0.028):
    """Reference-style ``|---- Group name ----|`` bracket above the panel."""
    tr = ax.get_xaxis_transform()
    ax.plot([x0, x0, x1, x1], [y - tick, y, y, y - tick], transform=tr,
            color="#555555", lw=0.9, solid_joinstyle="miter", clip_on=False,
            zorder=5)
    ax.text((x0 + x1) / 2.0, y, label, transform=tr, ha="center",
            va="center", fontsize=8.6, color="#1a1a1a", clip_on=False,
            zorder=6,
            bbox=dict(boxstyle="square,pad=0.28", fc="white", ec="none"))


def main():
    set_pub_style(base_font=8)
    scores = gen_scores()

    # x layout: one slot per task, GAP blank slots between groups
    centers, edges, labels = [], [], []
    cur = 0.0
    for _, tasks in GROUPS:
        lo = cur
        for name, _ in tasks:
            centers.append(cur)
            labels.append(name)
            cur += 1.0
        edges.append((lo - 0.5, cur - 0.5))
        cur += GAP
    centers = np.asarray(centers)
    xlim = (edges[0][0] - 0.12, edges[-1][1] + 0.12)

    avg = {m: float(np.mean(scores[m])) for m in METHODS}

    # one shared y range for both panels, from every value that gets drawn
    top = max(max(scores[m].max() for m in METHODS), max(avg.values()))
    ylo, yhi = 0.0, top + 13.0

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.22, 1.0], left=0.058,
                          right=0.988, top=0.795, bottom=0.185, wspace=0.075)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[0, 1])

    for ax in (ax_a, ax_m):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
        ax.set_ylim(ylo, yhi)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        style_axes(ax)

    # ---- Average side panel -------------------------------------------------
    xa = np.arange(len(METHODS), dtype=float)
    for j, m in enumerate(METHODS):
        ax_a.bar(xa[j], avg[m], 0.68, color=COL[m], lw=0.0, zorder=3)
        ax_a.text(xa[j], avg[m] + 1.8, f"{avg[m]:.1f}", ha="center",
                  va="bottom", fontsize=6.6, color="#333333", zorder=4)
    ax_a.set_xlim(-0.75, len(METHODS) - 0.25)
    ax_a.set_xticks([])
    ax_a.set_ylabel("Success rate (%)", fontsize=8.6)
    ax_a.set_title("Average", fontsize=9.4, fontweight="bold", pad=4.0)
    ax_a.text(0.5, -0.045, "7 tasks $\\times$ 20 trials", transform=ax_a.transAxes,
              ha="center", va="top", fontsize=6.8, color="#777777")

    # ---- Main panel ---------------------------------------------------------
    bar_group(ax_m, centers, scores)
    ax_m.set_xlim(*xlim)
    ax_m.set_xticks(centers)
    ax_m.set_xticklabels(labels, fontsize=7.4)
    ax_m.tick_params(axis="x", length=0, pad=3.5)
    ax_m.tick_params(labelleft=False)
    for (lo, hi), (label, _) in zip(edges, GROUPS):
        group_header(ax_m, lo + 0.04, hi - 0.04, label)
    for k in range(len(edges) - 1):
        xsep = (edges[k][1] + edges[k + 1][0]) / 2.0
        ax_m.axvline(xsep, color="#d5d5d5", lw=0.8, zorder=1)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[m]) for m in METHODS]
    leg = fig.legend(handles, METHODS, loc="lower center", ncol=5,
                     bbox_to_anchor=(0.53, 0.052), frameon=False, fontsize=8.6,
                     handlelength=1.5, handleheight=0.9, columnspacing=2.0,
                     handletextpad=0.55)
    leg.get_texts()[-1].set_fontweight("bold")

    fig.canvas.draw()
    p = ax_a.get_position()
    background_card(fig, (p.x0 - 0.048, p.y0 - 0.075, p.width + 0.062,
                          p.height + 0.135),
                    facecolor="#f2f2f2", zorder=-5, radius=0.010)

    savefig(fig, "1094_wm_quant_grouped_bars.png")


if __name__ == "__main__":
    main()

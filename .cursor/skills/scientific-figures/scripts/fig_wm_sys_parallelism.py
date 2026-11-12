"""Figure 1191 - Four ways to cut one model across four GPUs.

Every panel is the same 4 x 4 board: rows are GPUs, columns are the four
layer blocks of the model, and the coloured area inside a cell is the *part*
of that block the GPU holds.  Only the shape of the shard changes:

    data      whole block, one quarter of the batch      -> full cell
    tensor    one quarter of every block                 -> vertical quarter
    pipeline  one whole block, nothing else              -> diagonal
    sequence  whole block, one quarter of the sequence   -> horizontal quarter

Shard identity is carried by one blue ladder plus a fill texture, never by a
second hue, so the four boards stay inside the batch colour budget.

The strip underneath is measured on the same four strategies: bars are traffic
per optimiser step, the markers on the right axis are scaling efficiency, and
efficiency is printed as the division it is - ``3.72 / 4 = 0.930`` - so the
reader never has to trust a lone percentage.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

from pubstyle import set_pub_style, savefig, style_axes

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#d5d5d5"
GRID, CARD, EMPTY = "#ededed", "#e2e8ee", "#e6ebf0"
SHARD = ["#cfe1ee", "#9cc3dd", "#2e74a8", "#134a76"]
HATCH = ["///", "", "", "\\\\\\"]
TXT = [INK, INK, "white", "white"]
NG = 4                                    # GPUs, and blocks, and shards

# name, cell-shape code, one-line reading of the board
STRATS = [
    ("Data parallel", "data", "whole model per GPU,\n1/4 of the batch"),
    ("Tensor parallel", "tensor", "every block cut 4 ways\nacross the GPUs"),
    ("Pipeline parallel", "pipe", "one block per GPU,\nmicro-batches flow"),
    ("Sequence parallel", "seq", "whole model per GPU,\n1/4 of the sequence"),
]
# traffic in GB moved per optimiser step, and measured speed-up on 4 GPUs
COMM = {"data": 20.1, "tensor": 34.8, "pipe": 2.6, "seq": 12.4}
SPEEDUP = {"data": 3.72, "tensor": 2.84, "pipe": 3.12, "seq": 3.44}


def draw_board(ax, code):
    """One 4 x 4 board; returns nothing, every cell is drawn the same way."""
    for r in range(NG):                       # r = GPU index, top row = GPU 0
        y = NG - 1 - r
        for c in range(NG):                   # c = layer block index
            ax.add_patch(Rectangle((c + 0.06, y + 0.06), 0.88, 0.88,
                                   facecolor=EMPTY, edgecolor="#c8d2da",
                                   linewidth=0.6, zorder=2))
            if code == "data":
                box, s = (c + 0.06, y + 0.06, 0.88, 0.88), r
            elif code == "tensor":
                box, s = (c + 0.06 + 0.22 * r, y + 0.06, 0.22, 0.88), r
            elif code == "seq":
                box, s = (c + 0.06, y + 0.06 + 0.22 * (NG - 1 - r),
                          0.88, 0.22), r
            else:                              # pipeline: only the diagonal
                if r != c:
                    continue
                box, s = (c + 0.06, y + 0.06, 0.88, 0.88), c
            ax.add_patch(Rectangle(box[:2], box[2], box[3],
                                   facecolor=SHARD[s], edgecolor="white",
                                   linewidth=0.5, hatch=HATCH[s], zorder=3))
    ax.set_xlim(-0.04, NG + 0.04)
    ax.set_ylim(-0.04, NG + 0.04)
    ax.set_xticks(np.arange(NG) + 0.5)
    ax.set_xticklabels([f"L{c}" for c in range(NG)], fontsize=7.0)
    ax.set_yticks(np.arange(NG) + 0.5)
    ax.set_yticklabels([f"GPU {r}" for r in range(NG)][::-1], fontsize=7.0)
    ax.tick_params(length=0.0, pad=2.0)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    set_pub_style(base_font=8.5)
    keys = [s[1] for s in STRATS]
    NAME = {s[1]: s[0] for s in STRATS}
    eff = {k: SPEEDUP[k] / NG for k in keys}
    for k in keys:
        assert abs(float(f"{SPEEDUP[k]:.2f}") / NG
                   - float(f"{eff[k]:.3f}")) < 5e-6, \
            f"the printed efficiency for {k} is not the printed division"
    best = max(keys, key=lambda k: eff[k])
    lean = min(keys, key=lambda k: COMM[k])
    assert lean != best, "the headline contrast has disappeared"

    fig = plt.figure(figsize=(11.0, 4.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.72], left=0.058,
                          right=0.988, top=0.740, bottom=0.190, wspace=0.20,
                          hspace=0.92)
    boards = [fig.add_subplot(gs[0, j]) for j in range(4)]
    axs = fig.add_subplot(gs[1, :])

    for j, (name, code, note) in enumerate(STRATS):
        ax = boards[j]
        draw_board(ax, code)
        ax.set_title(f"({'abcd'[j]})  {name}", fontsize=9.0,
                     fontweight="bold", pad=4.5)
        if j:
            ax.set_yticklabels([])
        ax.set_xlabel(note, fontsize=7.4, color=SUB, labelpad=3.0,
                      linespacing=1.5)

    # ---- traffic bars and efficiency markers, same four strategies ----------
    style_axes(axs)
    axs.set_axisbelow(True)
    axs.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    xs = np.arange(4, dtype=float)
    vals = [COMM[k] for k in keys]
    axs.bar(xs, vals, width=0.44, color="#5f9dc6", edgecolor="white",
            linewidth=0.8, zorder=3)
    for x, v in zip(xs, vals):
        axs.text(x, v + 0.9, f"{v:.1f} GB", ha="center", va="bottom",
                 fontsize=7.6, color=INK, fontweight="bold", zorder=5)
    axs.set_xlim(-0.55, 3.55)
    axs.set_ylim(0.0, max(vals) * 1.42)
    axs.set_xticks(xs)
    axs.set_xticklabels([s[0].split()[0] for s in STRATS], fontsize=8.2)
    axs.set_ylabel("traffic per step\n(GB)", fontsize=7.8, linespacing=1.5)
    axs.tick_params(labelsize=7.4, length=2.4)

    axe = axs.twinx()
    axe.spines["top"].set_visible(False)
    axe.plot(xs, [eff[k] for k in keys], color="#0b3d63", lw=1.2,
             ls=(0, (3.0, 1.8)), marker="o", ms=5.4, mfc="white",
             mec="#0b3d63", mew=1.5, zorder=6)
    for x, k in zip(xs, keys):
        axe.text(x + 0.13, eff[k] + 0.012,
                 f"{SPEEDUP[k]:.2f} / {NG} = {eff[k]:.3f}", ha="left",
                 va="bottom", fontsize=7.4, color="#0b3d63",
                 fontweight="bold", zorder=7,
                 bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=HAIR,
                           lw=0.5, alpha=0.96))
    axe.set_ylim(0.62, 1.04)
    axe.set_yticks([0.7, 0.8, 0.9, 1.0])
    axe.set_ylabel("scaling efficiency\n(speed-up / 4 GPUs)", fontsize=7.8,
                   linespacing=1.5, color="#0b3d63")
    axe.tick_params(labelsize=7.4, length=2.4, colors="#0b3d63")
    axs.set_title("(e)  What each board costs: traffic per step (bars) and "
                  "scaling efficiency (markers)", fontsize=9.0,
                  fontweight="bold", pad=4.5, loc="left")

    # ---- legend: one entry per element that is actually drawn ---------------
    handles = [Patch(facecolor=SHARD[s], edgecolor="white", hatch=HATCH[s],
                     label=f"shard {s}") for s in range(NG)]
    handles.append(Patch(facecolor=EMPTY, edgecolor="#c8d2da",
                         label="not resident"))
    handles.append(Line2D([], [], color="#0b3d63", lw=1.2, ls=(0, (3.0, 1.8)),
                          marker="o", ms=5.0, mfc="white", mec="#0b3d63",
                          mew=1.4, label="scaling efficiency"))
    leg = fig.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.53, 0.845), ncol=6, frameon=True,
                     facecolor="white", edgecolor=HAIR, framealpha=0.95,
                     fontsize=7.8, handlelength=1.5, columnspacing=1.5,
                     borderpad=0.42)
    leg.get_frame().set_linewidth(0.5)

    # ---- headline and the divisions -----------------------------------------
    # wrapped, or the one-line version widens the exported canvas by 42 %

    band = FancyBboxPatch((0.016, 0.020), 0.968, 0.062,
                          boxstyle="round,pad=0.004,rounding_size=0.008",
                          transform=fig.transFigure, facecolor=CARD,
                          edgecolor="#c3cdd6", linewidth=0.6, zorder=-5)
    fig.add_artist(band)
    band.set_in_layout(False)
    savefig(fig, "1191_wm_sys_parallelism.png")


if __name__ == "__main__":
    main()

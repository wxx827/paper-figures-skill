"""Figure 316 - Token-level confidence read-out for one generated passage.

The passage itself is the main panel: every token sits in a cell shaded by the
probability the model assigned to it, so low-confidence spans (proper nouns,
dates, the pivot word "compromise") stand out as pale gaps in an otherwise
dark sentence.  Four of those positions are opened up underneath as top-5
candidate distributions, drawn small and boxed in grey because they are
evidence for the strip above, and the per-token surprisal trace on the right
turns the same numbers into the sequence perplexity.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, style_axes, get_cmap

ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
FAINT = "#8a8a8a"
BLOCK = "#f5f5f5"

FIG_W, FIG_H = 10.4, 6.4
STRIP = [0.055, 0.700, 0.890, 0.200]        # x0, y0, w, h in fig fractions
CBAR = [0.720, 0.648, 0.225, 0.017]
CAND_Y, CAND_H, CAND_W, CAND_GAP = 0.180, 0.315, 0.115, 0.033
CAND_X0 = 0.055
PPL_RECT = [0.700, 0.180, 0.262, 0.385]
MONO_ADV = 0.6022                            # DejaVu Sans Mono advance width
FS_TOKEN = 9.3
WRAP = 100

TOKENS = [
    ("The", 0.44), (" capital", 0.58), (" of", 0.94), (" Australia", 0.41),
    (" is", 0.96), (" Canberra", 0.47), (",", 0.72), (" not", 0.33),
    (" Sydney", 0.86), (".", 0.93), (" The", 0.55), (" city", 0.49),
    (" was", 0.78), (" chosen", 0.52), (" in", 0.89), (" 1908", 0.29),
    (" as", 0.66), (" a", 0.84), (" compromise", 0.37), (" between", 0.74),
    (" the", 0.91), (" two", 0.63), (" rival", 0.31), (" centres", 0.58),
    (",", 0.80), (" and", 0.69), (" construction", 0.44), (" of", 0.57),
    (" the", 0.87), (" parliament", 0.40), (" began", 0.61), (" in", 0.90),
    (" 1913", 0.26), (".", 0.95),
]
MARKS = [5, 15, 18, 32]                      # positions opened up below
CANDIDATES = {
    5: [("Canberra", 0.47), ("Sydney", 0.21), ("Melbourne", 0.12),
        ("the", 0.07), ("Adelaide", 0.04)],
    15: [("1908", 0.29), ("1901", 0.18), ("1913", 0.11), ("the", 0.09),
         ("1911", 0.07)],
    18: [("compromise", 0.37), ("capital", 0.16), ("city", 0.12),
         ("site", 0.08), ("result", 0.05)],
    32: [("1913", 0.26), ("1927", 0.19), ("1911", 0.13), ("1901", 0.10),
         ("the", 0.06)],
}
CAND_XMAX = 0.68


def label(fig, ax, text, dx=-0.34, dy=0.06):
    """Panel label placed a fixed distance (inches) from the axes corner."""
    p = ax.get_position()
    fig.text(p.x0 + dx / FIG_W, p.y1 + dy / FIG_H, text, fontsize=11,
             fontweight="bold", ha="left", va="bottom")


def wrap_tokens():
    lines, cur, used = [], [], 0
    for i, (t, p) in enumerate(TOKENS):
        if used + len(t) > WRAP and cur:
            lines.append(cur)
            cur, used = [], 0
            t = t.lstrip()
        cur.append((i, t, p, used))
        used += len(t)
    lines.append(cur)
    return lines


def draw_strip(fig, cmap, norm):
    ax = fig.add_axes(STRIP)
    ax.set_axis_off()
    lines = wrap_tokens()
    n_line = len(lines)
    width_in = STRIP[2] * FIG_W
    n_chars = width_in * 72.0 / (FS_TOKEN * MONO_ADV)
    ax.set_xlim(0, n_chars)
    ax.set_ylim(0, n_line)

    for li, line in enumerate(lines):
        y = n_line - li - 0.5
        for k, (idx, tok, p, x0) in enumerate(line):
            marked = idx in MARKS
            ax.add_patch(Rectangle((x0, y - 0.31), len(tok), 0.62,
                                   facecolor=cmap(norm(p)),
                                   edgecolor=ACCENT if marked else "white",
                                   linewidth=1.1 if marked else 0.5,
                                   zorder=3 if marked else 2))
            ax.text(x0 + 0.10, y, tok, fontsize=FS_TOKEN,
                    fontfamily="monospace", ha="left", va="center", zorder=4,
                    color="white" if p > 0.62 else INK)
            if marked:
                ax.text(x0 + len(tok) / 2.0, y + 0.34,
                        f"{MARKS.index(idx) + 1}", fontsize=6.6,
                        color=ACCENT, fontweight="bold", ha="center",
                        va="bottom", zorder=5)
    return ax


def draw_candidates(fig):
    axes = []
    for k, pos in enumerate(MARKS):
        rect = [CAND_X0 + k * (CAND_W + CAND_GAP), CAND_Y, CAND_W, CAND_H]
        ax = fig.add_axes(rect)
        ax.set_facecolor("none")
        cand = CANDIDATES[pos]
        ys = np.arange(len(cand))[::-1]
        for y, (tok, p) in zip(ys, cand):
            top = tok == TOKENS[pos][0].strip()
            ax.barh(y, p, height=0.66, color=ACCENT if top else "#b8b8b8",
                    lw=0, zorder=3)
            ax.text(p + 0.012, y, f"{p:.2f}", va="center", ha="left",
                    fontsize=6.0, color=ACCENT if top else SUB,
                    fontweight="bold" if top else "normal")
        ax.set_yticks(ys)
        ax.set_yticklabels([t for t, _ in cand], fontsize=6.2,
                           fontfamily="monospace")
        for lab, (tok, _) in zip(ax.get_yticklabels(), cand):
            if tok == TOKENS[pos][0].strip():
                lab.set_color(ACCENT)
                lab.set_fontweight("bold")
        ax.set_xlim(0, CAND_XMAX)
        ax.set_xticks([0, 0.2, 0.4, 0.6])
        ax.tick_params(labelsize=6.2, length=2.2)
        ax.xaxis.grid(True, lw=0.5, color="#e2e2e2")
        ax.set_axisbelow(True)
        style_axes(ax)
        ax.set_title(f"{k + 1}  ·  position {pos + 1}", fontsize=7.5,
                     fontweight="bold", color=INK, pad=3.0)
        if k == 0:
            ax.set_ylabel("candidate", fontsize=7.0, labelpad=1.5)
        ax.set_xlabel("$p$ (-)", fontsize=7.0, labelpad=1.0)
        axes.append(ax)
    return axes


def draw_perplexity(fig):
    ax = fig.add_axes(PPL_RECT)
    p = np.array([q for _, q in TOKENS])
    bits = -np.log2(p)
    x = np.arange(len(p)) + 1
    ax.fill_between(x, 0, bits, step="mid", color="#3b5b92", alpha=0.16, lw=0)
    ax.step(x, bits, where="mid", color="#3b5b92", lw=1.3, zorder=3)
    ax.axhline(bits.mean(), color=SUB, ls=(0, (4, 2.5)), lw=0.9, zorder=2)
    ax.text(len(p), bits.mean() + 0.06, "mean surprisal", fontsize=6.4,
            color=SUB, ha="right", va="bottom")
    for k, pos in enumerate(MARKS):
        ax.plot(pos + 1, bits[pos], marker="o", ms=4.0, color=ACCENT,
                zorder=5)
        ax.text(pos + 1, bits[pos] + 0.14, f"{k + 1}", fontsize=6.6,
                color=ACCENT, fontweight="bold", ha="center", va="bottom")
    ax.set_xlim(0.5, len(p) + 0.5)
    ax.set_ylim(0, bits.max() * 1.30)
    ax.set_xlabel("token index", fontsize=7.5, labelpad=1.5)
    ax.set_ylabel("surprisal $-\\log_2 p$ (bits)", fontsize=7.5, labelpad=2.0)
    ax.tick_params(labelsize=6.8, length=2.5)
    ax.set_title("Sequence surprisal", fontsize=8.5, fontweight="bold",
                 pad=3.5)
    style_axes(ax)
    ax.text(0.03, 0.955, f"PPL = ${2 ** bits.mean():.2f}$",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.0,
            fontweight="bold", color=INK,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor="#dcdcdc", linewidth=0.6, alpha=0.95))
    return ax


def main():
    set_pub_style(base_font=9)
    cmap = get_cmap("cb_sequential_blue")
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    x0 = CAND_X0 - 0.020
    x1 = CAND_X0 + 4 * CAND_W + 3 * CAND_GAP + 0.010
    fig.patches.append(Rectangle((x0, 0.098), x1 - x0, 0.495,
                                 transform=fig.transFigure, facecolor=BLOCK,
                                 edgecolor="none", zorder=-1))
    fig.text((x0 + x1) / 2.0, 0.552,
             "Top-5 candidates at the four marked positions", fontsize=7.8,
             fontweight="bold", color=INK, ha="center", va="bottom")

    ax_s = draw_strip(fig, cmap, norm)
    ax_c = draw_candidates(fig)
    ax_p = draw_perplexity(fig)

    cax = fig.add_axes(CBAR)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal", ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=8.0, width=0.7, length=2.5, color="#333333")
    cb.set_label("probability of the sampled token (-)", fontsize=8.0,
                 color=INK, labelpad=2.5)

    label(fig, ax_s, "(a)", dy=0.02)
    label(fig, ax_c[0], "(b)", dy=0.10)
    label(fig, ax_p, "(c)", dy=0.10)

    savefig(fig, "316_llm_token_probability.png")


if __name__ == "__main__":
    main()

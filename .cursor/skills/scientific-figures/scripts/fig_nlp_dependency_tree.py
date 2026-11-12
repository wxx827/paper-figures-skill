"""Figure 466 - Projective dependency parse of a single sentence.

Main panel: every token is an independent rounded chip, the dependency arcs
ride above them as half-ellipses whose height grows with the head-dependent
distance, and a part-of-speech colour strip runs underneath.  The parse is
projective and the height rule is monotone in distance, so a nested arc is
always strictly shorter than the arc containing it and no two arcs can ever
cross - the layout is correct by construction rather than by tuning.

Subordinate row: how often each relation label and each arc length occurs in
the surrounding treebank sample, with this sentence marked on both.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from pubstyle import set_pub_style, savefig, lighten, darken
from figkit import inch_axes, panel_label_inches

# ---------------------------------------------------------------------------
# Shared NLP-batch vocabulary (464-473)
# ---------------------------------------------------------------------------
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
HAIR = "#cccccc"
GRID = "#d9d9d9"
ACCENT = "#b23a48"

MONO = {"fontfamily": "monospace"}
CHAR_W = 0.6022 / 72.0        # DejaVu Sans Mono advance, inches per pt
CHIP_H = 0.205                # batch-wide chip height, inches
CHIP_PAD = 0.058              # batch-wide horizontal padding, inches
CHIP_R = 0.045                # batch-wide corner radius, inches

POS_COLORS = {
    "NOUN": "#3b5b92",
    "VERB": "#b23a48",
    "ADJ": "#2e7d5b",
    "DET": "#c9a227",
    "PRON": "#c9a227",
    "ADP": "#8e6bbf",
    "PUNCT": "#b3b3b3",
}
POS_LEGEND = [("NOUN", "#3b5b92"), ("VERB", "#b23a48"), ("ADJ", "#2e7d5b"),
              ("DET / PRON", "#c9a227"), ("ADP", "#8e6bbf"),
              ("PUNCT", "#b3b3b3")]

TOKENS = ["The", "research", "team", "published", "their", "new", "findings",
          "in", "a", "major", "journal", "."]
POS = ["DET", "NOUN", "NOUN", "VERB", "PRON", "ADJ", "NOUN", "ADP", "DET",
       "ADJ", "NOUN", "PUNCT"]
# (head, dependent, relation); the parse is projective by design
DEPS = [(2, 0, "det"), (2, 1, "compound"), (3, 2, "nsubj"), (6, 4, "poss"),
        (6, 5, "amod"), (3, 6, "dobj"), (3, 7, "prep"), (10, 8, "det"),
        (10, 9, "amod"), (7, 10, "pobj"), (3, 11, "punct")]
ROOT = 3
RELATIONS = ["det", "nsubj", "dobj", "amod", "prep", "pobj", "compound",
             "poss", "punct", "advmod", "aux", "cc", "conj", "nummod"]

TOK_FS = 8.5
TOK_GAP = 0.225
Y_BASE = 0.99
H0, H_STEP = 0.30, 0.19


def chip_width(text, fs=7.0, pad=CHIP_PAD):
    """Exact chip width in inches for monospaced text of ``fs`` points."""
    return 2.0 * pad + len(text) * CHAR_W * fs


def chip(ax, x, y, text, fs=7.0, h=CHIP_H, fc="#f0f0f0", ec=HAIR, color=INK,
         weight="normal", pad=CHIP_PAD, lw=0.7, zorder=6):
    """Rounded token chip with its lower-left corner at ``(x, y)`` in inches."""
    w = chip_width(text, fs, pad)
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={CHIP_R}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
        mutation_aspect=1.0))
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center",
            fontsize=fs, color=color, fontweight=weight, zorder=zorder + 2,
            **MONO)
    return w


def treebank_sample(rng, n=4200):
    """Synthetic treebank: relation labels and head-dependent distances."""
    w = np.array([2.6, 2.1, 1.7, 2.0, 1.5, 1.4, 0.9, 0.6, 2.2, 1.1,
                  1.3, 1.0, 1.2, 0.5])
    labels = rng.choice(len(RELATIONS), size=n, p=w / w.sum())
    dist = 1 + rng.geometric(0.42, size=n) - 1
    dist = np.clip(dist, 1, 12)
    return labels, dist


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(466)
    labels, dist = treebank_sample(rng)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = inch_axes(fig, (0.42, 2.95, 9.62, 3.05))
    CW = ax.get_xlim()[1]

    # ---- token chips, centred as one block --------------------------------
    widths = [chip_width(t, TOK_FS) for t in TOKENS]
    total = sum(widths) + TOK_GAP * (len(TOKENS) - 1)
    x = (CW - total) / 2.0
    xc, xl, xr = [], [], []
    for t, w in zip(TOKENS, widths):
        xl.append(x)
        xr.append(x + w)
        xc.append(x + w / 2.0)
        x += w + TOK_GAP
    xc = np.array(xc)

    ax.plot([xl[0] - 0.10, xr[-1] + 0.10], [Y_BASE, Y_BASE], lw=0.7,
            color=HAIR, zorder=2)
    for t, p, x0 in zip(TOKENS, POS, xl):
        col = POS_COLORS[p]
        chip(ax, x0, Y_BASE - 0.30 - CHIP_H, t, fs=TOK_FS,
             fc=lighten(col, 0.90), ec=lighten(col, 0.45),
             color=darken(col, 0.18))

    # ---- part-of-speech strip --------------------------------------------
    strip_y = Y_BASE - 0.30 - CHIP_H - 0.19
    for p, x0, x1 in zip(POS, xl, xr):
        col = POS_COLORS[p]
        ax.add_patch(FancyBboxPatch(
            (x0, strip_y), x1 - x0, 0.115,
            boxstyle=f"round,pad=0,rounding_size={CHIP_R * 0.6}",
            facecolor=col, edgecolor="white", linewidth=0.5, zorder=5,
            mutation_aspect=1.0))
    for p, cx in zip(POS, xc):
        ax.text(cx, strip_y - 0.055, p, ha="center", va="top", fontsize=6.0,
                color=darken(POS_COLORS[p], 0.20), **MONO)
    ax.text(xl[0] - 0.16, strip_y + 0.058, "POS", ha="right", va="center",
            fontsize=6.6, color=FAINT)

    # ---- dependency arcs --------------------------------------------------
    t = np.linspace(0.0, np.pi, 160)
    for head, dep, rel in DEPS:
        d = abs(head - dep)
        h = H0 + H_STEP * (d - 1)
        xm, a = (xc[head] + xc[dep]) / 2.0, abs(xc[head] - xc[dep]) / 2.0
        col = POS_COLORS[POS[dep]]
        ax.plot(xm + a * np.cos(t), Y_BASE + h * np.sin(t), lw=1.15,
                color=col, alpha=0.92, zorder=4, solid_capstyle="round")
        ax.plot([xc[dep]], [Y_BASE + 0.028], marker="v", ms=4.0, color=col,
                zorder=5, clip_on=False)
        ax.text(xm, Y_BASE + h + 0.015, rel, ha="center", va="center",
                fontsize=6.4, color=darken(col, 0.22), zorder=6, **MONO,
                bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                          edgecolor="none"))

    # root marker
    ax.annotate("", xy=(xc[ROOT], Y_BASE + 0.055),
                xytext=(xc[ROOT], Y_BASE + 0.46),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.4,
                                shrinkA=0, shrinkB=0))
    ax.text(xc[ROOT], Y_BASE + 0.50, "root", ha="center", va="bottom",
            fontsize=6.6, color=ACCENT, fontweight="bold", **MONO)

    # ---- legend across the top -------------------------------------------
    ly = ax.get_ylim()[1] - 0.14
    lx = xl[0]
    for name, col in POS_LEGEND:
        ax.add_patch(FancyBboxPatch(
            (lx, ly - 0.055), 0.17, 0.105,
            boxstyle=f"round,pad=0,rounding_size={CHIP_R * 0.6}",
            facecolor=col, edgecolor="none", zorder=5, mutation_aspect=1.0))
        ax.text(lx + 0.23, ly, name, ha="left", va="center", fontsize=6.8,
                color=SUB, zorder=5)
        lx += 0.23 + len(name) * CHAR_W * 6.8 + 0.42
    ax.text(xr[-1], ly, "arc height $\\propto$ head-dependent distance",
            ha="right", va="center", fontsize=6.8, color=FAINT)

    # ---- (b) relation frequency in the treebank sample --------------------
    axb = fig.add_axes([0.075, 0.100, 0.395, 0.280])
    counts = np.array([(labels == i).sum() for i in range(len(RELATIONS))])
    order = np.argsort(counts)
    here = {r for _h, _d, r in DEPS}
    cols = [ACCENT if RELATIONS[i] in here else "#9fb0c8" for i in order]
    axb.barh(np.arange(len(order)), counts[order], height=0.68, color=cols,
             edgecolor="white", linewidth=0.4, zorder=3)
    axb.set_yticks(np.arange(len(order)))
    axb.set_yticklabels([RELATIONS[i] for i in order], fontsize=6.2, **MONO)
    axb.set_xlabel("occurrences in the treebank sample (-)", fontsize=7.0)
    axb.tick_params(labelsize=6.5, length=2.2)
    axb.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    axb.set_title("Relation labels, red where this sentence uses them",
                  fontsize=8.5, fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)

    # ---- (c) arc length distribution --------------------------------------
    axc = fig.add_axes([0.585, 0.100, 0.375, 0.280])
    edges = np.arange(0.5, 12.6, 1.0)
    axc.hist(dist, bins=edges, color="#9fb0c8", edgecolor="white",
             linewidth=0.5, zorder=3)
    sent_d = np.array([abs(h - d) for h, d, _r in DEPS])
    axc.axvline(sent_d.mean(), color=ACCENT, lw=1.3, ls=(0, (3.2, 1.8)),
                zorder=4)
    axc.text(sent_d.mean() + 0.28, axc.get_ylim()[1] * 0.90,
             f"this sentence\nmean {sent_d.mean():.2f} tokens", fontsize=6.4,
             color=ACCENT, ha="left", va="top", linespacing=1.35)
    axc.plot(sent_d, np.full(sent_d.size, -axc.get_ylim()[1] * 0.045), "|",
             color=ACCENT, ms=5.0, mew=1.0, clip_on=False, zorder=5)
    axc.set_xlim(0.4, 12.6)
    axc.set_xlabel("head-dependent distance (tokens)", fontsize=7.0)
    axc.set_ylabel("arcs (-)", fontsize=7.0)
    axc.tick_params(labelsize=6.5, length=2.2)
    axc.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    axc.set_title(f"Arc length over {dist.size:,} sampled arcs",
                  fontsize=8.5, fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)

    panel_label_inches(fig, ax, "(a)", dx=-0.16, dy=0.02)
    panel_label_inches(fig, axb, "(b)", dx=-0.50, dy=0.08)
    panel_label_inches(fig, axc, "(c)", dx=-0.50, dy=0.08)
    savefig(fig, "466_nlp_dependency_tree.png")


if __name__ == "__main__":
    main()

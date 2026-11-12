"""Figure 470 - Named entity recognition over a tagged passage.

Main panel: a news paragraph laid out as one rounded chip per token, wrapped
at a fixed column width.  Entity chips carry the type colour and a small type
tag above them; every other token stays grey, so the entities read as figure
and the rest as ground.  Multi-word entities occupy a single chip, which is
what makes the span boundaries visible.

Subordinate column: how many entities of each type the test set holds, and
the precision, recall and F1 the tagger reaches on them.  Precision and recall
come from one simulated confusion count, so F1 is always their harmonic mean.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

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

ENT_COLORS = {"PER": "#3b5b92", "ORG": "#b23a48", "LOC": "#2e7d5b",
              "DATE": "#c9a227", "MISC": "#8e6bbf"}
PLAIN_FC, PLAIN_EC, PLAIN_TX = "#f7f7f7", "#e6e6e6", "#8a8a8a"

MONO = {"fontfamily": "monospace"}
CHAR_W = 0.6022 / 72.0        # DejaVu Sans Mono advance, inches per pt
CHIP_H = 0.205                # batch-wide chip height, inches
CHIP_PAD = 0.058              # batch-wide horizontal padding, inches
CHIP_R = 0.045                # batch-wide corner radius, inches
TOK_FS = 7.5
GAP = 0.070
TAG_H = 0.135
LEAD = CHIP_H + TAG_H + 0.075

PASSAGE = [
    ("On", None), ("12 March 2024,", "DATE"), ("Dr. Elena Vasquez", "PER"),
    ("of", None), ("the", None), ("University of Toronto", "ORG"),
    ("announced", None), ("that", None), ("NeuroLink", "ORG"),
    ("had", None), ("signed", None), ("a", None), ("research", None),
    ("agreement", None), ("with", None), ("Siemens Healthineers", "ORG"),
    ("in", None), ("Berlin.", "LOC"), ("The", None), ("project,", None),
    ("funded", None), ("by", None), ("the", None),
    ("European Research Council,", "ORG"), ("will", None), ("run", None),
    ("until", None), ("June 2027", "DATE"), ("and", None), ("is", None),
    ("led", None), ("jointly", None), ("by", None), ("Vasquez", "PER"),
    ("and", None), ("Prof. Kenji Tanaka", "PER"), ("of", None),
    ("RIKEN", "ORG"), ("in", None), ("Wako,", "LOC"), ("Japan.", "LOC"),
    ("Early", None), ("trials", None), ("at", None),
    ("Charite Hospital", "ORG"), ("began", None), ("in", None),
    ("October 2023,", "DATE"), ("with", None), ("Ada Okonkwo", "PER"),
    ("coordinating", None), ("the", None), ("Lagos", "LOC"),
    ("site.", None), ("The", None), ("team", None), ("presented", None),
    ("results", None), ("at", None), ("NeurIPS", "MISC"), ("and", None),
    ("released", None), ("the", None), ("OpenSpike", "MISC"),
    ("toolkit", None), ("under", None), ("an", None), ("Apache 2.0", "MISC"),
    ("licence.", None), ("A", None), ("follow-up", None), ("study", None),
    ("with", None), ("Stanford University", "ORG"), ("and", None),
    ("the", None), ("Allen Institute", "ORG"), ("is", None),
    ("planned", None), ("for", None), ("March 2026", "DATE"), ("in", None),
    ("Seattle,", "LOC"), ("Washington.", "LOC"), ("Reviewers", None),
    ("from", None), ("Nature Methods", "MISC"), ("praised", None),
    ("the", None), ("open", None), ("protocol,", None), ("which", None),
    ("Okonkwo", "PER"), ("will", None), ("present", None), ("at", None),
    ("ACL 2026", "MISC"), ("in", None), ("Nairobi.", "LOC"),
]
TYPES = list(ENT_COLORS)
SUPPORT_W = np.array([1.00, 1.18, 0.86, 0.62, 0.38])
RECALL_BASE = np.array([0.912, 0.861, 0.893, 0.947, 0.706])
FP_RATE = np.array([0.088, 0.132, 0.098, 0.047, 0.225])
N_TEST = 3200


def chip_width(text, fs=TOK_FS, pad=CHIP_PAD):
    """Exact chip width in inches for monospaced text of ``fs`` points."""
    return 2.0 * pad + len(text) * CHAR_W * fs


def chip(ax, x, y, text, fs=TOK_FS, h=CHIP_H, fc=PLAIN_FC, ec=PLAIN_EC,
         color=PLAIN_TX, weight="normal", lw=0.7, zorder=5):
    """Rounded token chip with its lower-left corner at ``(x, y)`` in inches."""
    w = chip_width(text, fs)
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={CHIP_R}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
        mutation_aspect=1.0))
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center",
            fontsize=fs, color=color, fontweight=weight, zorder=zorder + 2,
            **MONO)
    return w


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(470)

    support = np.round(SUPPORT_W / SUPPORT_W.sum() * N_TEST).astype(int)
    tp = rng.binomial(support, RECALL_BASE)
    fp = rng.binomial(tp, FP_RATE)
    prec = tp / (tp + fp)
    rec = tp / support
    f1 = 2 * prec * rec / (prec + rec)
    micro_f1 = (2 * tp.sum() / (2 * tp.sum() + fp.sum() +
                                (support - tp).sum()))

    fig = plt.figure(figsize=(10.4, 6.4))

    # ---- (a) the tagged passage -------------------------------------------
    ax = inch_axes(fig, (0.26, 0.48, 6.72, 5.50))
    CW, CH = ax.get_xlim()[1], ax.get_ylim()[1]
    ax.text(0.04, CH - 0.06, "Tagged passage", fontsize=9.0,
            fontweight="bold", ha="left", va="top", color=INK)
    ax.text(CW - 0.04, CH - 0.09,
            f"{sum(1 for _t, e in PASSAGE if e)} entities "
            f"in {len(PASSAGE)} tokens", fontsize=6.8, ha="right", va="top",
            color=FAINT)

    lx = 0.04
    for t, col in ENT_COLORS.items():
        lx += chip(ax, lx, CH - 0.50, t, fs=6.6, h=0.185,
                   fc=lighten(col, 0.84), ec=lighten(col, 0.40),
                   color=darken(col, 0.20), weight="bold") + 0.14
    lx += chip(ax, lx, CH - 0.50, "other", fs=6.6, h=0.185) + 0.14

    x, y = 0.04, CH - 0.98
    for text, ent in PASSAGE:
        w = chip_width(text)
        if x > 0.04 and x + w > CW - 0.04:
            x, y = 0.04, y - LEAD
        if ent is None:
            chip(ax, x, y, text)
        else:
            col = ENT_COLORS[ent]
            chip(ax, x, y, text, fc=lighten(col, 0.86),
                 ec=lighten(col, 0.38), color=darken(col, 0.22),
                 weight="bold", lw=0.9)
            ax.text(x + w / 2.0, y + CHIP_H + 0.028, ent, ha="center",
                    va="bottom", fontsize=6.0, color=col, fontweight="bold",
                    zorder=8, **MONO)
        x += w + GAP

    counts = {t: sum(1 for _w, e in PASSAGE if e == t) for t in TYPES}
    ax.text(0.04, y - 0.34,
            "  ".join(f"{t}: {counts[t]}" for t in TYPES) +
            f"   |   test-set micro F1 {micro_f1:.3f}",
            fontsize=6.8, color=SUB, ha="left", va="top", **MONO)

    # ---- (b) entity counts in the test set --------------------------------
    axb = fig.add_axes([0.735, 0.575, 0.235, 0.320])
    ypos = np.arange(len(TYPES))[::-1]
    axb.barh(ypos, support, height=0.64,
             color=[ENT_COLORS[t] for t in TYPES], edgecolor="white",
             linewidth=0.4, zorder=3)
    for yy, v in zip(ypos, support):
        axb.text(v + support.max() * 0.02, yy, f"{v:,}", va="center",
                 ha="left", fontsize=6.4, color=SUB, **MONO)
    axb.set_yticks(ypos)
    axb.set_yticklabels(TYPES, fontsize=7.0, **MONO)
    axb.set_xlim(0, support.max() * 1.24)
    axb.set_xlabel("entities in the test set (-)", fontsize=7.0)
    axb.tick_params(labelsize=6.5, length=2.2)
    axb.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    axb.set_title("Support per type", fontsize=8.5, fontweight="bold",
                  pad=3.0, loc="left")
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)

    # ---- (c) precision, recall, F1 ----------------------------------------
    axc = fig.add_axes([0.735, 0.110, 0.235, 0.300])
    for yy, t, p, r, f in zip(ypos, TYPES, prec, rec, f1):
        col = ENT_COLORS[t]
        axc.barh(yy, f, height=0.52, color=lighten(col, 0.55),
                 edgecolor="white", linewidth=0.4, zorder=3)
        axc.plot([p], [yy], marker="o", ms=4.0, mfc="white", mec=col,
                 mew=1.1, zorder=5)
        axc.plot([r], [yy], marker="s", ms=3.6, mfc=col, mec="white",
                 mew=0.6, zorder=5)
        axc.text(0.562, yy, f"{f:.3f}", va="center", ha="left",
                 fontsize=6.2, color=darken(col, 0.30), zorder=6, **MONO)
    axc.set_yticks(ypos)
    axc.set_yticklabels(TYPES, fontsize=7.0, **MONO)
    axc.set_xlim(0.55, 1.0)
    axc.set_xlabel("score (-)", fontsize=7.0)
    axc.tick_params(labelsize=6.5, length=2.2)
    axc.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    axc.set_title("Bar = F1, markers = P and R", fontsize=8.5,
                  fontweight="bold", pad=3.0, loc="left")
    axc.legend(handles=[
        Line2D([], [], marker="o", ls="none", mfc="white", mec=SUB, mew=1.1,
               ms=4.0, label="precision"),
        Line2D([], [], marker="s", ls="none", mfc=SUB, mec="white", mew=0.6,
               ms=3.6, label="recall")],
        loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
        fontsize=6.8, frameon=False, handletextpad=0.4)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)

    worst = TYPES[int(np.argmin(f1))]
    panel_label_inches(fig, ax, "(a)", dx=-0.18, dy=-0.02)
    panel_label_inches(fig, axb, "(b)", dx=-0.50, dy=0.08)
    panel_label_inches(fig, axc, "(c)", dx=-0.50, dy=0.08)
    savefig(fig, "470_nlp_named_entity.png")


if __name__ == "__main__":
    main()

"""473 - Summarisation evaluation: copy behaviour, metrics and human ratings.

Left: the source document and one generated summary as word chips.  A chip is
tinted when the content word is shared by both texts, so the copied backbone of
the summary and its few novel words are visible at a glance; the copy rate
printed underneath is counted from exactly those chips.  Top right: the four
automatic metrics on one shared radial scale - they are all scores in [0, 1],
and rescaling each axis to its own range is what makes radar charts lie.
Bottom right: the human ratings as full Likert distributions rather than bare
means, with the mean printed beside each bar.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

from pubstyle import (set_pub_style, savefig, get_palette, lighten, darken,
                      add_panel_label)
from figkit import use_inch_units

INK, RULE, SUB = "#1a1a1a", "#333333", "#555555"
MUTED, HAIR, SOFT, GRID = "#777777", "#cccccc", "#f2f2f2", "#e2e2e2"
SHARE = "#3b5b92"        # word present in both texts
NOVEL = "#b23a48"        # summary word that is not in the source

MODELS = ["LEAD-3", "BART", "PEGASUS", "Ours"]
MCOL = {"LEAD-3": "#8a8a8a", "BART": "#2e7d5b", "PEGASUS": "#c9a227",
        "Ours": "#b23a48"}
METRICS = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"]
SCORES = {                      # corpus-level, one row per model
    "LEAD-3":  [0.401, 0.172, 0.363, 0.848],
    "BART":    [0.442, 0.213, 0.411, 0.876],
    "PEGASUS": [0.447, 0.219, 0.415, 0.879],
    "Ours":    [0.468, 0.238, 0.434, 0.891],
}
CRITERIA = ["Fluency", "Faithfulness", "Informativeness"]
LIKERT = ["1 poor", "2", "3", "4", "5 excellent"]
# rating distributions, per criterion and model, over 120 annotator judgements
DIST = {
    ("Fluency", "LEAD-3"): [3, 8, 22, 41, 26],
    ("Fluency", "BART"): [1, 4, 15, 44, 36],
    ("Fluency", "PEGASUS"): [1, 4, 14, 43, 38],
    ("Fluency", "Ours"): [1, 3, 12, 42, 42],
    ("Faithfulness", "LEAD-3"): [1, 3, 11, 38, 47],
    ("Faithfulness", "BART"): [6, 13, 21, 36, 24],
    ("Faithfulness", "PEGASUS"): [5, 12, 20, 38, 25],
    ("Faithfulness", "Ours"): [2, 6, 16, 40, 36],
    ("Informativeness", "LEAD-3"): [9, 19, 28, 30, 14],
    ("Informativeness", "BART"): [3, 9, 22, 40, 26],
    ("Informativeness", "PEGASUS"): [3, 8, 21, 41, 27],
    ("Informativeness", "Ours"): [2, 6, 18, 42, 32],
}
N_JUDGE = 120

SOURCE = (
    "Regulators approved the merger of two regional airlines on Tuesday, "
    "clearing the way for a combined carrier that will serve 140 cities. "
    "The decision followed a two-year review in which the agency examined "
    "fares on overlapping routes and heard from twelve city governments. "
    "Both companies agreed to give up eight gates at the busiest hub and to "
    "hold fares flat on contested routes for three years. Consumer groups "
    "had argued that the deal would remove the last low-cost operator from "
    "several markets in the north, while the airlines said the combination "
    "was the only way to keep serving small airports that lose money in "
    "winter. Shares of both carriers rose in late trading, and the unions "
    "representing pilots said they would seek a single contract before the "
    "networks are merged next spring."
)
SUMMARY = (
    "Regulators approved the regional airline merger after a two-year review, "
    "forcing the carriers to surrender eight gates and freeze fares on "
    "contested routes."
)
STOP = set("a an the of to on in and or for that which will be is was were "
           "at from by with as it its their both up after".split())

FS_CHIP = 6.6
CH_W = 0.6 * FS_CHIP / 72.0      # monospace advance width, inches
CH_H = 0.155
PAD = 0.036
GAP = 0.036
LINE = 0.215


def norm_tok(tok):
    return tok.strip(".,;:()\u2014\u2019'\"").lower()


def chip(ax, x, y, w, h, text, fc, ec, tc, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={0.045}",
        facecolor=fc, edgecolor=ec, linewidth=0.5, mutation_aspect=1.0,
        zorder=2))
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center",
            fontsize=FS_CHIP, color=tc, fontfamily="monospace",
            fontweight=weight, zorder=3)


def flow(ax, x0, y0, width, tokens, shared, novel=()):
    """Lay chips out left to right with wrapping; returns the bottom y."""
    x, y = x0, y0
    for tok in tokens:
        key = norm_tok(tok)
        w = len(tok) * CH_W + 2 * PAD
        if x + w > x0 + width:
            x, y = x0, y - LINE
        if key in novel:
            fc, ec, tc, wt = lighten(NOVEL, 0.80), lighten(NOVEL, 0.45), \
                darken(NOVEL, 0.25), "bold"
        elif key in shared:
            fc, ec, tc, wt = lighten(SHARE, 0.82), lighten(SHARE, 0.50), \
                darken(SHARE, 0.30), "bold"
        else:
            fc, ec, tc, wt = SOFT, HAIR, SUB, "normal"
        chip(ax, x, y - CH_H, w, CH_H, tok, fc, ec, tc, wt)
        x += w + GAP
    return y - CH_H


def main():
    set_pub_style(base_font=8)

    src_tok = SOURCE.split()
    sum_tok = SUMMARY.split()
    src_set = {norm_tok(w) for w in src_tok} - STOP
    sum_set = {norm_tok(w) for w in sum_tok} - STOP
    shared = src_set & sum_set
    novel = sum_set - src_set
    copy_rate = len(shared) / max(len(sum_set), 1)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.06, 0.94],
                          height_ratios=[0.92, 1.00],
                          left=0.045, right=0.965, top=0.845, bottom=0.075,
                          hspace=0.34, wspace=0.13)
    ax_tx = fig.add_subplot(gs[:, 0])
    ax_rd = fig.add_subplot(gs[0, 1], polar=True)
    ax_hb = fig.add_subplot(gs[1, 1])

    # ---- (a) source and summary as chips -----------------------------------
    ax_tx.set_facecolor("white")
    for s in ax_tx.spines.values():
        s.set_visible(False)
    W, Hh = use_inch_units(ax_tx, off=True)
    ax_tx.set_aspect("auto")
    ax_tx.set_xlim(0, W)
    ax_tx.set_ylim(0, Hh)

    y = Hh - 0.10
    ax_tx.text(0.0, y, "Source document", fontsize=8.6, fontweight="bold",
               color=INK, va="top")
    ax_tx.text(W, y, f"{len(src_tok)} tokens", fontsize=6.6, color=MUTED,
               va="top", ha="right")
    y = flow(ax_tx, 0.0, y - 0.22, W, src_tok, shared)

    y -= 0.34
    ax_tx.plot([0.0, W], [y + 0.10, y + 0.10], color=HAIR, lw=0.8)
    ax_tx.text(0.0, y, "Generated summary  \u00b7  Ours", fontsize=8.6,
               fontweight="bold", color=INK, va="top")
    ax_tx.text(W, y, f"{len(sum_tok)} tokens", fontsize=6.6, color=MUTED,
               va="top", ha="right")
    y = flow(ax_tx, 0.0, y - 0.22, W, sum_tok, shared, novel)

    y -= 0.30
    ax_tx.add_patch(FancyBboxPatch(
        (0.0, y - 0.52), W, 0.60,
        boxstyle="round,pad=0,rounding_size=0.045", facecolor=SOFT,
        edgecolor=HAIR, lw=0.6, mutation_aspect=1.0, zorder=1))
    ax_tx.text(0.10, y - 0.06,
               f"{len(shared)} of {len(sum_set)} content words in the summary "
               f"also occur in the source ({copy_rate:.0%} copied); "
               f"{len(novel)} are new: "
               + ", ".join(sorted(novel)[:5]),
               fontsize=6.6, color=SUB, va="top", ha="left", zorder=3)
    ax_tx.text(0.10, y - 0.30,
               "Stop words are excluded from the count; tinted chips are the "
               "shared content words.", fontsize=6.4, color=MUTED, va="top",
               ha="left", zorder=3)

    handles = [
        Patch(facecolor=lighten(SHARE, 0.82), edgecolor=lighten(SHARE, 0.50),
              lw=0.5, label="content word in both texts"),
        Patch(facecolor=lighten(NOVEL, 0.80), edgecolor=lighten(NOVEL, 0.45),
              lw=0.5, label="new word, summary only"),
        Patch(facecolor=SOFT, edgecolor=HAIR, lw=0.5,
              label="stop word or source-only word"),
    ]
    ax_tx.legend(handles=handles, loc="lower left",
                 bbox_to_anchor=(0.0, -0.045), ncol=3, frameon=False,
                 fontsize=6.6, handlelength=1.4, handleheight=1.0,
                 columnspacing=1.2)
    add_panel_label(ax_tx, "(a)", x=-0.035, y=1.045)

    # ---- (b) metric radar on one shared radial scale -----------------------
    ang = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False)
    ring = np.r_[ang, ang[0]]
    for m in MODELS:
        v = np.r_[SCORES[m], SCORES[m][0]]
        hot = m == "Ours"
        ax_rd.plot(ring, v, color=MCOL[m], lw=1.9 if hot else 1.2,
                   marker="o", ms=3.4 if hot else 2.6, mfc="white",
                   mec=MCOL[m], mew=1.0, zorder=5 if hot else 4, label=m)
        if hot:
            ax_rd.fill(ring, v, color=MCOL[m], alpha=0.12, lw=0, zorder=3)
            for a, s in zip(ang, SCORES[m]):
                # near the rim the value would sit on the axis label, so it
                # is tucked inside and rotated a little off the spoke
                a = a if s < 0.72 else a + 0.12
                r = s + 0.085 if s < 0.72 else s - 0.115
                ax_rd.text(a, r, f"{s:.3f}", ha="center", va="center",
                           fontsize=6.2, color=MCOL[m], fontweight="bold",
                           zorder=8)
    ax_rd.set_theta_offset(np.pi / 2)
    ax_rd.set_theta_direction(-1)
    ax_rd.set_xticks(ang)
    ax_rd.set_xticklabels(METRICS, fontsize=7.2)
    ax_rd.tick_params(axis="x", pad=6.5)
    ax_rd.set_ylim(0.0, 1.0)
    ax_rd.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax_rd.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6.0,
                          color=MUTED)
    ax_rd.set_rlabel_position(228)
    ax_rd.grid(color=GRID, lw=0.6)
    ax_rd.spines["polar"].set_color(HAIR)
    ax_rd.spines["polar"].set_linewidth(0.8)
    ax_rd.legend(loc="lower left", bbox_to_anchor=(-0.24, -0.13), ncol=2,
                 frameon=False, fontsize=6.8, handlelength=1.5,
                 columnspacing=1.0, labelspacing=0.3)
    bb = ax_rd.get_position()

    # ---- (c) human ratings as Likert distributions -------------------------
    cols = get_palette("cb_diverging", len(LIKERT))
    ypos, ylab, ycol = [], [], []
    p = 0.0
    for ci, crit in enumerate(CRITERIA):
        p += 0.85
        for m in MODELS:
            ypos.append(p)
            ylab.append(m)
            ycol.append(m)
            p += 1.0
    means = {}
    for y, m, crit in zip(ypos, ylab,
                          [c for c in CRITERIA for _ in MODELS]):
        d = np.asarray(DIST[(crit, m)], dtype=float)
        pct = 100.0 * d / d.sum()
        means[(crit, m)] = float((np.arange(1, 6) * d).sum() / d.sum())
        left = 0.0
        for k, (v, c) in enumerate(zip(pct, cols)):
            ax_hb.barh(y, v, left=left, height=0.68, color=c,
                       edgecolor="white", lw=0.5, zorder=3)
            if v >= 11.0:
                ax_hb.text(left + v / 2.0, y, f"{v:.0f}", ha="center",
                           va="center", fontsize=6,
                           color="white" if k in (0, 4) else INK, zorder=4)
            left += v
        ax_hb.text(101.5, y, f"{means[(crit, m)]:.2f}", ha="left",
                   va="center", fontsize=6.5, color=MCOL[m],
                   fontweight="bold" if m == "Ours" else "normal", zorder=4)

    ax_hb.set_yticks(ypos)
    ax_hb.set_yticklabels(ylab, fontsize=6.8)
    for lab, m in zip(ax_hb.get_yticklabels(), ylab):
        lab.set_color(MCOL[m])
        if m == "Ours":
            lab.set_fontweight("bold")
    ax_hb.invert_yaxis()
    ax_hb.set_xlim(0, 100)
    ax_hb.set_xticks([0, 25, 50, 75, 100])
    ax_hb.set_xlabel("share of judgements (%)", fontsize=7.6)
    ax_hb.tick_params(labelsize=6.8, length=2.2)
    ax_hb.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax_hb.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax_hb.spines[s].set_visible(False)
    for ci, crit in enumerate(CRITERIA):
        y0 = ypos[ci * len(MODELS)]
        ax_hb.text(-0.145, y0 - 0.62, crit, transform=ax_hb.get_yaxis_transform(),
                   ha="left", va="center", fontsize=7.4, fontweight="bold",
                   color=INK)
        if ci:
            ax_hb.axhline(y0 - 0.95, color=HAIR, lw=0.7, zorder=1)
    ax_hb.text(101.5, ypos[0] - 1.15, "mean", ha="left", va="center",
               fontsize=6.4, color=MUTED)
    ax_hb.set_title("Human ratings, full Likert distribution "
                    f"($n$ = {N_JUDGE} judgements per bar)", fontsize=8.4,
                    fontweight="bold", pad=16.0, loc="left", x=-0.145)
    ax_hb.legend(handles=[Patch(facecolor=c, edgecolor="white", lw=0.5,
                                label=l) for c, l in zip(cols, LIKERT)],
                 loc="lower left", bbox_to_anchor=(-0.145, 1.005), ncol=5,
                 frameon=False, fontsize=6.4, handlelength=1.3,
                 handleheight=1.0, columnspacing=1.0)
    add_panel_label(ax_hb, "(c)", x=-0.155, y=1.16)

    gap_r = SCORES["Ours"][0] - SCORES["PEGASUS"][0]
    faith = means[("Faithfulness", "Ours")] - means[("Faithfulness", "PEGASUS")]
    lead = means[("Faithfulness", "LEAD-3")]
    savefig(fig, "473_nlp_summarization_eval.png")


if __name__ == "__main__":
    main()

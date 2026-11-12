"""461 - ASR error analysis: word alignment, error mix, stratified WER.

The alignment on the left is a real Levenshtein backtrace over the example
sentences; the donut and the stratified bars are counted over a simulated
corpus whose error probability depends on SNR, speaker and speaking rate.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Patch

from pubstyle import set_pub_style, savefig, style_axes, lighten

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
OP_COLOR = {"C": "#b3b3b3", "S": "#b23a48", "D": "#3b5b92", "I": "#c9a227"}
OP_NAME = {"S": "substitution", "D": "deletion", "I": "insertion"}

PAIRS = [
    ("the lone man sold the red silk suit",
     "the loan man sold a red silk suit"),
    ("please call stella and ask her to bring these things",
     "please call stella and ask to bring these things"),
    ("six spoons of fresh snow peas",
     "six spoons of fresh snow peas and"),
    ("we can meet at nine tomorrow morning",
     "we can beat at nine tomorrow warning"),
]

STRATA = [
    ("Input SNR", ["5 dB", "10 dB", "20 dB"], [1.85, 1.15, 0.70]),
    ("Speaker", ["female", "male", "child"], [0.92, 1.00, 1.55]),
    ("Speaking rate", ["slow", "normal", "fast"], [0.86, 1.00, 1.42]),
]


def align(ref, hyp):
    """Levenshtein alignment; returns a list of (op, ref_word, hyp_word)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (
                ref[i - 1] != hyp[j - 1]):
            ops.append(("C" if ref[i - 1] == hyp[j - 1] else "S",
                        ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("D", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            j -= 1
    return ops[::-1]


def simulate_corpus(seed=17, n_utt=420):
    """Per-utterance error counts under a multiplicative risk model."""
    rng = np.random.default_rng(seed)
    base = 0.085
    rows = []
    for _ in range(n_utt):
        levels = [rng.integers(0, 3) for _ in STRATA]
        p = base
        for (_, _, mult), k in zip(STRATA, levels):
            p *= mult[k]
        p = float(np.clip(p, 0.005, 0.75))
        n_words = int(rng.integers(7, 16))
        n_err = rng.binomial(n_words, p)
        s, dl, ins = rng.multinomial(n_err, [0.58, 0.26, 0.16])
        rows.append((levels, n_words, s, dl, ins))
    return rows


def main():
    set_pub_style(base_font=8)

    aligns = [align(r.split(), h.split()) for r, h in PAIRS]
    counts = {k: 0 for k in "CSDI"}
    n_ref = 0
    for ops in aligns:
        for op, r, h in ops:
            counts[op] += 1
            n_ref += r is not None
    wer_ex = 100.0 * (counts["S"] + counts["D"] + counts["I"]) / n_ref

    corpus = simulate_corpus()
    tot_w = sum(r[1] for r in corpus)
    tot = {"S": sum(r[2] for r in corpus), "D": sum(r[3] for r in corpus),
           "I": sum(r[4] for r in corpus)}
    wer_corpus = 100.0 * sum(tot.values()) / tot_w

    fig = plt.figure(figsize=(8.6, 5.4))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 0.52],
                  height_ratios=[1.0, 1.05], hspace=0.42, wspace=0.16,
                  left=0.045, right=0.985, top=0.860, bottom=0.090)

    # ---- (a) alignment view -------------------------------------------------
    ax = fig.add_subplot(gs[:, 0])
    ax.set_axis_off()
    pad, gap = 0.30, 0.16
    row_h, row_gap, blk_gap = 0.40, 0.46, 1.28
    y = 0.0
    xmax = 0.0
    for u, ops in enumerate(aligns):
        y_ref, y_hyp = -y, -y - row_gap
        x = 1.55
        for op, r, h in ops:
            wr = 0.128 * max(len(r or ""), len(h or "")) + 2 * pad
            col = OP_COLOR[op]
            for txt, yy in ((r, y_ref), (h, y_hyp)):
                if txt is None:
                    ax.plot([x + 0.20, x + wr - 0.20], [yy, yy], color=col,
                            lw=1.1, ls=(0, (2.2, 1.6)), zorder=3)
                    continue
                ax.add_patch(FancyBboxPatch(
                    (x, yy - row_h / 2), wr - gap, row_h,
                    boxstyle="round,pad=0,rounding_size=0.07",
                    facecolor=lighten(col, 0.80 if op == "C" else 0.66),
                    edgecolor=col, linewidth=0.7, zorder=3,
                    mutation_aspect=1.0))
                ax.text(x + (wr - gap) / 2, yy, txt, ha="center", va="center",
                        fontsize=6.8, color=INK if op != "C" else SUB,
                        zorder=5)
            if op != "C":
                ax.text(x + (wr - gap) / 2, y_hyp - row_h / 2 - 0.09, op,
                        ha="center", va="top", fontsize=6.2, color=col,
                        fontweight="bold", zorder=5)
            x += wr
        xmax = max(xmax, x)
        ax.text(1.42, y_ref, "REF", ha="right", va="center", fontsize=6.8,
                color=SUB, fontweight="bold")
        ax.text(1.42, y_hyp, "HYP", ha="right", va="center", fontsize=6.8,
                color=SUB, fontweight="bold")
        e = sum(1 for o, _, _ in ops if o != "C")
        nr = sum(1 for _, r, _ in ops if r is not None)
        ax.text(0.05, (y_ref + y_hyp) / 2, f"u{u + 1}", ha="left",
                va="center", fontsize=7.6, color=INK, fontweight="bold")
        ax.text(0.05, (y_ref + y_hyp) / 2 - 0.30, f"{100 * e / nr:.0f} %",
                ha="left", va="center", fontsize=6.3, color=MUTED)
        y += blk_gap
    ax.set_xlim(0, xmax + 0.1)
    ax.set_ylim(-y - 0.05, 0.85)
    ax.set_aspect("auto")

    handles = [Patch(facecolor=lighten(OP_COLOR[k], 0.66),
                     edgecolor=OP_COLOR[k], linewidth=0.7,
                     label=f"{OP_NAME[k]} ({k})") for k in "SDI"]
    handles.insert(0, Patch(facecolor=lighten(OP_COLOR["C"], 0.80),
                            edgecolor=OP_COLOR["C"], linewidth=0.7,
                            label="correct (C)"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.02),
              ncol=4, frameon=False, fontsize=6.8, handlelength=1.2,
              columnspacing=1.4, handleheight=0.9)
    ax.text(0.998, 1.02, f"{len(PAIRS)} example utterances   |   "
            f"WER {wer_ex:.1f} %   ({counts['S']}S / {counts['D']}D / "
            f"{counts['I']}I over {n_ref} reference words)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8,
            color=INK)
    ax.text(-0.005, 1.075, "(a)", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=11, fontweight="bold", color=INK)

    # ---- (b) error-type donut (subordinate) --------------------------------
    ax_d = fig.add_subplot(gs[0, 1])
    keys = ["S", "D", "I"]
    vals = [tot[k] for k in keys]
    share = 100.0 * np.asarray(vals) / sum(vals)
    wedges, _ = ax_d.pie(vals, colors=[OP_COLOR[k] for k in keys],
                         startangle=90, counterclock=False, radius=0.70,
                         wedgeprops=dict(width=0.30, edgecolor="white",
                                         linewidth=1.0))
    ax_d.set_aspect("equal")
    a0 = 90.0
    for k, w, v, s in zip(keys, wedges, vals, share):
        a1 = a0 - 360.0 * v / sum(vals)
        am = np.deg2rad(0.5 * (a0 + a1))
        cx, cy = np.cos(am), np.sin(am)
        ha = "left" if cx > 0.15 else ("right" if cx < -0.15 else "center")
        ax_d.text(0.78 * cx, 0.78 * cy,
                  f"{OP_NAME[k]}\n{s:.0f} %  ({v})", ha=ha,
                  va="center", fontsize=6.6, color=OP_COLOR[k],
                  fontweight="bold", linespacing=1.3)
        a0 = a1
    ax_d.text(0, 0.07, f"{wer_corpus:.1f} %", ha="center", va="center",
              fontsize=11, fontweight="bold", color=INK)
    ax_d.text(0, -0.11, f"corpus WER\n{sum(vals)} errors / {tot_w} words",
              ha="center", va="center", fontsize=6.3, color=MUTED,
              linespacing=1.3)
    ax_d.set_title("Error mix", fontsize=8.0, fontweight="bold", pad=2)
    ax_d.text(-0.06, 1.02, "(b)", transform=ax_d.transAxes, ha="left",
              va="bottom", fontsize=11, fontweight="bold", color=INK)

    # ---- (c) stratified WER (subordinate) ----------------------------------
    ax_b = fig.add_subplot(gs[1, 1])
    ypos, labels, vals_b, errs, cols = [], [], [], [], []
    y = 0.0
    group_mid = []
    for gi, (gname, lv_names, _) in enumerate(STRATA):
        ys = []
        for k, lv in enumerate(lv_names):
            sel = [r for r in corpus if r[0][gi] == k]
            w = sum(r[1] for r in sel)
            e = sum(r[2] + r[3] + r[4] for r in sel)
            p = e / max(w, 1)
            ypos.append(y)
            ys.append(y)
            labels.append(lv)
            vals_b.append(100.0 * p)
            errs.append(100.0 * 1.96 * np.sqrt(p * (1 - p) / max(w, 1)))
            cols.append(lighten("#3b5b92", 0.10 + 0.24 * k))
            y += 1.0
        group_mid.append((gname, float(np.mean(ys))))
        y += 0.75
    ypos = np.asarray(ypos)
    ax_b.barh(ypos, vals_b, height=0.72, color=cols, edgecolor="#3b5b92",
              linewidth=0.7, zorder=3,
              xerr=errs, error_kw=dict(ecolor=RULE, elinewidth=0.7,
                                       capsize=1.8))
    for yy, v in zip(ypos, vals_b):
        ax_b.text(v + max(errs) + 0.6, yy, f"{v:.1f}", va="center", ha="left",
                  fontsize=6.2, color=SUB, zorder=4)
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels(labels, fontsize=6.8)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, max(vals_b) * 1.30)
    ax_b.set_xlabel("Word error rate (%)", fontsize=7.5)
    ax_b.axvline(wer_corpus, color="#b23a48", lw=0.9, ls="--", zorder=2)
    ax_b.text(wer_corpus + 0.3, ypos.min() - 0.75,
              f"corpus {wer_corpus:.1f} %", fontsize=6.3, color="#b23a48",
              va="top", ha="left")
    ax_b.tick_params(labelsize=6.5, length=2)
    ax_b.grid(axis="x", lw=0.4, alpha=0.5)
    style_axes(ax_b)
    for gname, ym in group_mid:
        ax_b.text(-0.215, ym, gname, transform=ax_b.get_yaxis_transform(),
                  ha="left", va="center", fontsize=7.0, color=INK,
                  fontweight="bold", rotation=90)
    ax_b.set_title(f"WER by condition, {len(corpus)} utterances",
                   fontsize=8.0, fontweight="bold")
    ax_b.text(-0.30, 1.03, "(c)", transform=ax_b.transAxes, ha="left",
              va="bottom", fontsize=11, fontweight="bold", color=INK)


    savefig(fig, "461_spx_asr_errors.png")


if __name__ == "__main__":
    main()

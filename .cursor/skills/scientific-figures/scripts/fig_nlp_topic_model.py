"""Figure 464 - Topic model overview for a document corpus.

Main panel: the top-8 words of every discovered topic as grouped horizontal
bars, one hue per topic and a darker tint for the higher-probability words.
Subordinate column: a donut of the corpus-wide topic mix with elbow leader
labels, and a stacked area chart of how that mix drifts across twelve
publication years.

All three panels read the *same* share array, so the header percentages, the
donut wedges and the last column of the area chart cannot disagree.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, lighten, darken
from figkit import place_outside_labels, panel_labels_inches

# ---------------------------------------------------------------------------
# House vocabulary shared by the NLP batch (464-473)
# ---------------------------------------------------------------------------
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
HAIR = "#cccccc"
GRID = "#d9d9d9"

TOPIC_COLORS = ["#b23a48", "#3b5b92", "#2e7d5b", "#c9a227"]
OTHER_COLOR = "#b3b3b3"

TOPIC_WORDS = {
    "Machine translation": ["translation", "decoder", "encoder", "bleu",
                            "parallel", "alignment", "corpus", "beam"],
    "Clinical notes": ["patient", "clinical", "diagnosis", "cohort",
                       "record", "symptom", "admission", "coding"],
    "Social sentiment": ["tweet", "sentiment", "hashtag", "emoji",
                         "polarity", "sarcasm", "viral", "reply"],
    "Legal documents": ["clause", "contract", "statute", "court",
                        "liability", "ruling", "precedent", "party"],
}
SHORT = ["Translation", "Clinical", "Sentiment", "Legal", "Other"]
N_DOCS = 24800
YEARS = np.arange(2016, 2028)


def topic_shares(rng):
    """Smooth, softmax-normalised topic mix over ``YEARS``; rows sum to one."""
    t = np.linspace(0.0, 1.0, YEARS.size)
    a = np.array([1.05, 0.55, 0.05, 0.35, 0.45])
    b = np.array([-1.15, 0.35, 1.35, 0.30, -0.30])
    walk = np.cumsum(rng.standard_normal((a.size, YEARS.size)), axis=1)
    walk = (walk - walk.mean(axis=1, keepdims=True)) / (walk.std() + 1e-9)
    z = a[:, None] + b[:, None] * t[None, :] + 0.18 * walk
    s = np.exp(z)
    return s / s.sum(axis=0, keepdims=True)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(464)

    shares = topic_shares(rng)                 # (5 topics, 12 years)
    mean_share = shares.mean(axis=1)
    doc_counts = np.round(mean_share * N_DOCS).astype(int)
    colors = TOPIC_COLORS + [OTHER_COLOR]

    # top-8 word probabilities per topic, always sorted descending
    probs = {}
    for name in TOPIC_WORDS:
        p = np.sort(rng.dirichlet(np.full(8, 1.25)))[::-1] * 0.44
        probs[name] = p

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58],
                          left=0.100, right=0.975, top=0.905, bottom=0.085,
                          hspace=0.40, wspace=0.20)
    ax_bar = fig.add_subplot(gs[:, 0])
    ax_pie = fig.add_subplot(gs[0, 1])
    ax_area = fig.add_subplot(gs[1, 1])

    # ---- (a) topic x word bars -------------------------------------------
    pos = 0.0
    head_y, tick_y, tick_lab = [], [], []
    for ti, (name, words) in enumerate(TOPIC_WORDS.items()):
        pos += 1.70
        head_y.append(pos)
        pos += 1.05
        base = TOPIC_COLORS[ti]
        for wi, word in enumerate(words):
            v = probs[name][wi]
            col = lighten(base, 0.42 * wi / 7.0)
            ax_bar.barh(pos, v, height=0.70, color=col, edgecolor="white",
                        linewidth=0.4, zorder=3)
            ax_bar.text(v + 0.0022, pos, f"{v:.3f}", va="center", ha="left",
                        fontsize=6.0, color=SUB, zorder=4,
                        fontfamily="monospace")
            tick_y.append(pos)
            tick_lab.append(word)
            pos += 1.0
    pos += 0.6

    xmax = max(p.max() for p in probs.values()) * 1.30
    ax_bar.set_xlim(0.0, xmax)
    ax_bar.set_ylim(pos, -0.5)
    ax_bar.set_yticks(tick_y)
    ax_bar.set_yticklabels(tick_lab, fontsize=6.2, fontfamily="monospace")
    ax_bar.tick_params(axis="y", length=0, pad=2.0)
    ax_bar.tick_params(axis="x", labelsize=7.0, length=2.5)
    ax_bar.set_xlabel("topic-word probability p(w | z) (-)", fontsize=8.0)
    ax_bar.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax_bar.set_title("Top-8 words of every topic", fontsize=9.0,
                     fontweight="bold")
    for ti, (name, y) in enumerate(zip(TOPIC_WORDS, head_y)):
        ax_bar.text(0.0, y, name, fontsize=7.6, fontweight="bold",
                    color=darken(TOPIC_COLORS[ti], 0.12), va="center",
                    ha="left")
        ax_bar.text(xmax, y,
                    f"{mean_share[ti]:.1%} of corpus  ·  "
                    f"n={doc_counts[ti]:,}",
                    fontsize=6.6, color=MUTED, va="center", ha="right")
        ax_bar.axhline(y - 0.95, color=HAIR, linewidth=0.6, zorder=1)

    # ---- (b) corpus-wide topic mix ---------------------------------------
    ax_pie.pie(mean_share, colors=colors, startangle=96, counterclock=False,
               wedgeprops=dict(width=0.40, edgecolor="white", linewidth=0.9),
               radius=1.0)
    ax_pie.set_xlim(-1.86, 1.86)
    ax_pie.set_ylim(-1.24, 1.24)
    ax_pie.set_aspect("equal")

    ang0 = 96.0
    items = []
    for sh, lab, col in zip(mean_share, SHORT, colors):
        sweep = 360.0 * sh
        items.append((ang0 - sweep / 2.0, lab, f"{sh:.1%}", col))
        ang0 -= sweep
    place_outside_labels(ax_pie, items, r_edge=1.02, r_stub=1.09, x_knee=1.18,
                         x_text=1.26, min_gap=0.34, y_span=1.02,
                         name_size=7.0, stat_size=6.4)
    ax_pie.text(0, 0.10, f"{doc_counts.sum():,}", ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=INK)
    ax_pie.text(0, -0.16, "documents", ha="center", va="center",
                fontsize=6.8, color=MUTED)
    ax_pie.set_title("Corpus-wide topic mix", fontsize=8.5,
                     fontweight="bold", pad=2.0)

    # ---- (c) topic mix over time -----------------------------------------
    ax_area.stackplot(YEARS, shares, colors=colors, edgecolor="white",
                      linewidth=0.5)
    ax_area.set_xlim(YEARS[0], YEARS[-1])
    ax_area.set_ylim(0.0, 1.0)
    ax_area.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_area.set_yticklabels(["0", "25", "50", "75", "100"])
    ax_area.set_xticks(YEARS[::3])
    ax_area.tick_params(labelsize=6.5, length=2.2)
    ax_area.set_ylabel("share of documents (%)", fontsize=7.0)
    ax_area.set_xlabel("publication year", fontsize=7.0)
    ax_area.set_title("Topic mix drifts across twelve years", fontsize=8.5,
                      fontweight="bold", pad=2.0)
    for s in ("top", "right"):
        ax_area.spines[s].set_visible(False)

    mid = np.cumsum(shares[:, -1]) - shares[:, -1] / 2.0
    for lab, y, sh in zip(SHORT, mid, shares[:, -1]):
        if sh > 0.10:
            ax_area.text(YEARS[-1] - 0.18, y, lab, ha="right", va="center",
                         fontsize=6.2, color="white", fontweight="bold")

    grew = int(np.argmax(shares[:, -1] - shares[:, 0]))
    delta = shares[grew, -1] - shares[grew, 0]
    panel_labels_inches(fig, [ax_bar, ax_pie, ax_area], dx=-0.62, dy=0.10)
    savefig(fig, "464_nlp_topic_model.png")


if __name__ == "__main__":
    main()

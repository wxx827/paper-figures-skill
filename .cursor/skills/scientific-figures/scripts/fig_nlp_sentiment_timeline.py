"""Figure 469 - Sentiment of a review stream over two years.

Main panel: the weekly mean polarity, filled green above zero and red below,
with a bootstrap confidence band and four annotated events.  Below it, the
monthly positive / neutral / negative mix as stacked bars.  Right column: the
sentiment lexicon as a chip cloud, chip size scaling with mention count and
chip colour with polarity - a word cloud that keeps a consistent baseline,
padding and corner radius instead of rotating words to fill space.

Every review is simulated once; the curve, the band, the monthly mix and the
event deltas are all reductions of that single sample.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

POS_C = "#2e7d5b"
NEU_C = "#b3b3b3"
NEG_C = "#b23a48"

MONO = {"fontfamily": "monospace"}
CHAR_W = 0.6022 / 72.0        # DejaVu Sans Mono advance, inches per pt
CHIP_PAD = 0.058              # batch-wide horizontal padding, inches
CHIP_R = 0.045                # batch-wide corner radius, inches

THR = 0.15                    # polarity band that counts as neutral
EVENTS = [(11, +0.62, "product launch"), (33, -0.95, "data breach"),
          (57, +0.44, "price cut"), (84, -0.68, "battery recall")]
LEXICON = {
    "excellent": 0.85, "love": 0.78, "recommend": 0.72, "smooth": 0.61,
    "reliable": 0.66, "brilliant": 0.80, "helpful": 0.58, "worth": 0.49,
    "fast": 0.55, "sturdy": 0.44, "quiet": 0.38, "responsive": 0.52,
    "broken": -0.88, "refund": -0.74, "crash": -0.81, "overpriced": -0.69,
    "useless": -0.90, "delay": -0.55, "faulty": -0.83, "noisy": -0.47,
    "slow": -0.58, "leaking": -0.72, "flimsy": -0.63, "return": -0.41,
    "delivery": 0.04, "battery": -0.08, "screen": 0.02, "update": -0.05,
    "support": 0.07, "price": -0.11, "package": 0.05, "manual": -0.02,
    "setup": 0.09, "weight": -0.06, "colour": 0.03, "size": -0.04,
}


def chip_width(text, fs, pad=CHIP_PAD):
    """Exact chip width in inches for monospaced text of ``fs`` points."""
    return 2.0 * pad + len(text) * CHAR_W * fs


def chip(ax, x, y, text, fs, h, fc, ec, color, weight="normal", lw=0.7,
         zorder=5):
    """Rounded chip with its lower-left corner at ``(x, y)`` in inches."""
    w = chip_width(text, fs)
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={CHIP_R}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
        mutation_aspect=1.0))
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center",
            fontsize=fs, color=color, fontweight=weight, zorder=zorder + 2,
            **MONO)
    return w


def simulate(rng, n_weeks=105):
    """One synthetic review stream; everything else is derived from it."""
    k = np.arange(n_weeks)
    mu = np.cumsum(rng.standard_normal(n_weeks)) * 0.048
    mu -= mu.mean()
    for i, amp, _lab in EVENTS:
        mu = mu + amp * np.exp(-np.clip(k - i, 0, None) / 7.0) * (k >= i)
    mu = np.clip(mu, -1.0, 1.0)
    counts = rng.integers(90, 260, size=n_weeks)
    return [np.clip(rng.normal(m, 0.52, size=c), -1.0, 1.0)
            for m, c in zip(mu, counts)]


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(469)

    reviews = simulate(rng)
    weeks = np.arange("2024-01-01", "2026-01-06",
                      np.timedelta64(7, "D"), dtype="datetime64[D]")
    weeks = weeks[:len(reviews)]
    mean = np.array([r.mean() for r in reviews])
    sem = np.array([r.std(ddof=1) / np.sqrt(r.size) for r in reviews])
    n_reviews = sum(r.size for r in reviews)

    months = np.unique(weeks.astype("datetime64[M]"))
    mix = np.zeros((3, months.size))
    for mi, m in enumerate(months):
        sel = np.concatenate([r for r, w in zip(reviews, weeks)
                              if w.astype("datetime64[M]") == m])
        mix[0, mi] = (sel > THR).mean()
        mix[2, mi] = (sel < -THR).mean()
        mix[1, mi] = 1.0 - mix[0, mi] - mix[2, mi]

    fig = plt.figure(figsize=(10.4, 6.4))
    axm = fig.add_axes([0.058, 0.480, 0.578, 0.420])
    axb = fig.add_axes([0.058, 0.098, 0.578, 0.262])

    # ---- (a) weekly polarity ---------------------------------------------
    axm.axhline(0.0, color="#808080", lw=0.8, zorder=2)
    axm.fill_between(weeks, 0.0, mean, where=mean >= 0, interpolate=True,
                     color=lighten(POS_C, 0.62), zorder=3)
    axm.fill_between(weeks, 0.0, mean, where=mean <= 0, interpolate=True,
                     color=lighten(NEG_C, 0.62), zorder=3)
    axm.fill_between(weeks, mean - 1.96 * sem, mean + 1.96 * sem,
                     color="#5a5a5a", alpha=0.20, linewidth=0, zorder=4)
    axm.plot(weeks, mean, lw=1.5, color="#3a3a3a", zorder=5)

    ymax = float(np.abs(mean).max()) * 1.55
    axm.set_ylim(-ymax, ymax)

    def shock(i):
        """Polarity change across the four weeks bracketing an event."""
        return float(mean[min(i + 3, mean.size - 1)] - mean[max(i - 1, 0)])

    for ei, (i, _amp, lab) in enumerate(EVENTS):
        axm.axvline(weeks[i], color=MUTED, lw=0.8, ls=(0, (3.0, 2.0)),
                    zorder=6)
        d = shock(i)
        axm.text(weeks[i] + np.timedelta64(5, "D"),
                 ymax * (0.93 if ei % 2 == 0 else 0.64),
                 f"{lab}\n{d:+.2f}", fontsize=6.4, color=INK, ha="left",
                 va="top", linespacing=1.35, zorder=7,
                 bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                           edgecolor=HAIR, linewidth=0.6, alpha=0.94))
    axm.set_ylabel("mean polarity (-)", fontsize=8.0)
    axm.tick_params(labelsize=7.0, length=2.5)
    axm.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    axm.set_title(f"Weekly mean polarity over {n_reviews:,} reviews, "
                  "95 % band", fontsize=9.0, fontweight="bold")
    for s in ("top", "right"):
        axm.spines[s].set_visible(False)

    # ---- (b) monthly mix ---------------------------------------------------
    mx = months.astype("datetime64[D]")
    bottom = np.zeros(months.size)
    for row, col, lab in ((0, POS_C, "positive"), (1, NEU_C, "neutral"),
                          (2, NEG_C, "negative")):
        axb.bar(mx + np.timedelta64(15, "D"), mix[row], bottom=bottom,
                width=24, color=col, edgecolor="white", linewidth=0.5,
                zorder=3, label=lab)
        bottom = bottom + mix[row]
    axb.set_ylim(0.0, 1.0)
    axb.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axb.set_yticklabels(["0", "25", "50", "75", "100"])
    axb.set_ylabel("share of reviews (%)", fontsize=8.0)
    axb.tick_params(labelsize=7.0, length=2.5)
    axb.set_title("Monthly positive / neutral / negative mix", fontsize=8.5,
                  fontweight="bold", pad=3.0)
    axb.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3,
               fontsize=7.0, frameon=False, handlelength=1.2)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)

    for a in (axm, axb):
        a.set_xlim(weeks[0] - np.timedelta64(9, "D"),
                   weeks[-1] + np.timedelta64(9, "D"))
        a.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    axm.tick_params(labelbottom=False)

    # ---- (c) lexicon chip cloud -------------------------------------------
    axc = inch_axes(fig, (7.02, 0.62, 3.14, 5.18))
    CW, CH = axc.get_xlim()[1], axc.get_ylim()[1]
    freq = {w: int(v) for w, v in
            zip(LEXICON, np.round(rng.pareto(1.35, len(LEXICON)) * 340 + 55))}

    axc.text(0.06, CH - 0.06, "Sentiment lexicon", fontsize=9.0,
             fontweight="bold", ha="left", va="top", color=INK)
    axc.text(CW - 0.04, CH - 0.09, "chip size $\\propto$ mentions",
             fontsize=6.8, ha="right", va="top", color=FAINT)

    lx = 0.06
    for lab, col in (("positive", POS_C), ("neutral", NEU_C),
                     ("negative", NEG_C)):
        lx += chip(axc, lx, CH - 0.52, lab, 6.6, 0.185,
                   lighten(col, 0.84), lighten(col, 0.42),
                   darken(col, 0.20)) + 0.12

    fmin, fmax = min(freq.values()), max(freq.values())
    order = sorted(freq, key=lambda w: -freq[w])
    x, y, row_h, gap = 0.06, CH - 0.82, 0.0, 0.105
    for w in order:
        t = (np.log(freq[w]) - np.log(fmin)) / (np.log(fmax) - np.log(fmin))
        fs = 7.2 + 3.8 * t
        h = 0.100 + 0.0175 * fs
        cw = chip_width(w, fs)
        if x > 0.06 and x + cw > CW - 0.06:
            y -= row_h + 0.135
            x, row_h = 0.06, 0.0
        col = POS_C if LEXICON[w] > 0.2 else (
            NEG_C if LEXICON[w] < -0.2 else NEU_C)
        chip(axc, x, y - h, w, fs, h, lighten(col, 0.86),
             lighten(col, 0.40), darken(col, 0.22),
             weight="bold" if abs(LEXICON[w]) > 0.6 else "normal")
        x += cw + gap
        row_h = max(row_h, h)
    y -= row_h

    net = np.mean([LEXICON[w] * freq[w] for w in freq]) / np.mean(
        list(freq.values()))
    axc.text(0.06, y - 0.26,
             f"{len(freq)} lexicon entries, "
             f"{sum(freq.values()):,} mentions\n"
             f"frequency-weighted polarity {net:+.3f}",
             fontsize=6.8, color=SUB, ha="left", va="top", linespacing=1.4)

    worst, worst_lab = min((shock(i), lab) for i, _a, lab in EVENTS)
    panel_label_inches(fig, axm, "(a)", dx=-0.50, dy=0.08)
    panel_label_inches(fig, axb, "(b)", dx=-0.50, dy=0.08)
    panel_label_inches(fig, axc, "(c)", dx=-0.20, dy=-0.02)
    savefig(fig, "469_nlp_sentiment_timeline.png")


if __name__ == "__main__":
    main()

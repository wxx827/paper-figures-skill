"""Retractions: how many, for what, and how long the citations keep coming.

Main panel: annual retractions of a fictional bibliographic corpus stacked by
the reason given, from a period when the whole category was small to one
dominated by industrial-scale fabrication.  Counts alone would only show that
the literature got bigger, so the retraction *rate* per ten thousand
published papers is drawn on a second axis - the two disagree, and that
disagreement is the point.

Auxiliary (both at 58 % of the main panel width): annual citations to a
retracted paper in the years around the retraction notice, against matched
papers that were never retracted, so the residual citation flow after the
notice can be read off directly; and the delay between publication and
retraction, which is heavy tailed and therefore drawn on a logarithmic
month axis, split by the kind of problem that triggered it.

The reason shares follow smooth trends and every count is a Poisson draw on
them; the percentages, the rate curve, the post-notice citation share and the
median delays are all measured from the sampled arrays.  Fictional corpus,
illustrative values only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
DEEP = "#2b4470"
RUST = "#c0603f"
GOLD = "#c9a227"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.98, 0.90, 5.30, 4.58)
AUX_T_RECT = (7.64, 3.48, 3.14, 2.00)
AUX_B_RECT = (7.64, 0.86, 3.14, 2.00)

Y0, Y1 = 2000, 2025
YEARS = np.arange(Y0, Y1 + 1)
TT = (YEARS - Y0) / (Y1 - Y0)

# reason, colour, share at the start, share at the end (renormalised later)
REASONS = [
    ("Honest error", lighten(BLUE, 0.62), 0.30, 0.10),
    ("Plagiarism", lighten(BLUE, 0.40), 0.26, 0.13),
    ("Authorship / ethics", lighten(BLUE, 0.16), 0.18, 0.14),
    ("Image duplication", lighten(ACCENT, 0.42), 0.18, 0.22),
    ("Fabricated data / paper mills", ACCENT, 0.08, 0.41),
]
LAG_GROUPS = [
    ("Honest error", DEEP, 2.35, 0.75),
    ("Misconduct", ACCENT, 3.35, 0.80),
]


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(995)

    # published papers and total retractions
    papers = 3.2e5 * np.exp(1.05 * TT)                  # papers per year
    total_rate = 1.15 * np.exp(2.25 * TT)               # per 10 000 papers
    lam = total_rate * papers / 1.0e4

    shares = np.stack([s0 + (s1 - s0) * TT ** 1.35
                       for _, _, s0, s1 in REASONS])
    shares = shares / shares.sum(axis=0)
    counts = rng.poisson(shares * lam)
    tot = counts.sum(axis=0)
    rate = 1.0e4 * tot / papers

    mill = counts[-1]
    mill_first = 100.0 * mill[:3].sum() / tot[:3].sum()
    mill_last = 100.0 * mill[-3:].sum() / tot[-3:].sum()
    growth = tot[-3:].sum() / tot[:3].sum()
    rate_growth = rate[-3:].mean() / rate[:3].mean()

    # citations around the retraction notice
    rel = np.arange(-5, 6)
    base = 14.0 * np.exp(-((rel + 1.0) ** 2) / 26.0) + 3.0
    ctrl = rng.poisson(np.tile(base * 1.05, (240, 1))).mean(axis=0)
    hit = base * np.where(rel < 0, 1.12, 0.34 + 0.10 * (rel - 0))
    hit = np.clip(hit, 0.6, None)
    retr = rng.poisson(np.tile(hit, (240, 1))).mean(axis=0)
    post_share = 100.0 * retr[rel > 0].sum() / retr.sum()
    drop = 100.0 * (1.0 - retr[rel == 1][0] / retr[rel == -1][0])

    lags = {}
    for name, col, mu, sg in LAG_GROUPS:
        lags[name] = np.clip(rng.lognormal(mu, sg, 900), 1.0, 400.0)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: stacked reasons plus the rate --------------------------------
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.55, zorder=0)
    ax.stackplot(YEARS, counts, colors=[c for _, c, _, _ in REASONS],
                 edgecolor="white", linewidth=0.5, zorder=3)
    # the bands narrow toward the left, so in-band labels would spill out of
    # them; a key over the empty upper-left corner is the safe placement
    keys = [Patch(facecolor=c, edgecolor="white", lw=0.5,
                  label=f"{n} ({100.0 * counts[i][-3:].sum() / tot[-3:].sum():.0f} "
                        f"% of recent notices)")
            for i, (n, c, _, _) in enumerate(REASONS)][::-1]
    kl = ax.legend(handles=keys, loc="upper left", frameon=False,
                   fontsize=6.7, labelspacing=0.42, handlelength=1.5,
                   handleheight=0.9, borderpad=0.2,
                   bbox_to_anchor=(-0.008, 0.845))
    for t in kl.get_texts():
        t.set_color(SUB)

    ax2 = ax.twinx()
    ax2.plot(YEARS, rate, color=INK, lw=1.8, zorder=6)
    ax2.plot(YEARS, rate, ls="none", marker="o", ms=2.8, mfc="white",
             mec=INK, mew=0.8, zorder=7)
    ax2.set_ylim(0, max(rate) * 1.55)
    ax2.set_ylabel("retractions per 10 000 published papers", fontsize=8.5,
                   color=INK)
    ax2.tick_params(labelsize=7.4)
    ax2.spines["top"].set_visible(False)
    ax2.set_zorder(ax.get_zorder() + 1)
    ax2.patch.set_visible(False)
    ax2.annotate(f"rate: x{rate_growth:.1f} over the period",
                 xy=(YEARS[-4], rate[-4]),
                 xytext=(Y0 + 10.5, max(rate) * 1.34), fontsize=6.8,
                 color=INK, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK,
                                 connectionstyle="arc3,rad=-0.22"))

    ax.set_xlim(Y0 - 0.6, Y1 + 0.6)
    ax.set_ylim(0, tot.max() * 1.55)
    ax.set_xticks(np.arange(Y0, Y1 + 1, 5))
    ax.set_xlabel("year of the retraction notice (years)", fontsize=8.5)
    ax.set_ylabel("retraction notices (notices/year)", fontsize=8.5)
    ax.set_title("Retraction notices of a fictional corpus, by reason given",
                 fontsize=9.0, fontweight="bold", pad=5.0)
    ax.text(Y0 + 0.4, tot.max() * 1.50,
            f"notices per year grew x{growth:.1f} from {Y0}-{Y0 + 2} to "
            f"{Y1 - 2}-{Y1}, but the corpus grew too:\nthe rate per 10 000 "
            f"papers (black line, right axis) grew x{rate_growth:.1f}\n"
            f"fabricated data and paper mills went from {mill_first:.0f} % "
            f"to {mill_last:.0f} % of all notices",
            ha="left", va="top", fontsize=6.8, color=SUB, linespacing=1.6)

    # ---- aux (b): citations around the notice -------------------------------
    ax_t.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    ax_t.axvline(0.0, color=SUB, lw=0.9, ls=(0, (2.6, 2.0)), zorder=3)
    ax_t.plot(rel, ctrl, color=DEEP, lw=1.6, marker="o", ms=3.0, mfc="white",
              mew=0.9, zorder=6)
    ax_t.plot(rel, retr, color=ACCENT, lw=1.8, marker="D", ms=3.2,
              mfc="white", mew=0.9, zorder=7)
    ax_t.fill_between(rel[rel >= 0], 0.0, retr[rel >= 0],
                      facecolor=lighten(ACCENT, 0.62), edgecolor="none",
                      zorder=2)
    ax_t.set_xlim(-5.4, 5.4)
    ax_t.set_ylim(0, max(ctrl.max(), retr.max()) * 1.35)
    ax_t.set_xticks(np.arange(-5, 6, 2))
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("years relative to the retraction notice (years)",
                    fontsize=7.6, labelpad=1.5)
    ax_t.set_ylabel("citations per year", fontsize=7.6)
    ax_t.set_title("Citation flow survives the notice", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_t.text(0.030, 0.955,
              f"{drop:.0f} % drop in the first year after the notice,\n"
              f"but {post_share:.0f} % of the paper's plotted citations "
              f"still arrive afterwards",
              transform=ax_t.transAxes, ha="left", va="top", fontsize=6.1,
              color=SUB, linespacing=1.5)
    hl = [Line2D([], [], color=DEEP, lw=1.6, marker="o", ms=3.0, mfc="white",
                 label="matched papers, never retracted"),
          Line2D([], [], color=ACCENT, lw=1.8, marker="D", ms=3.2,
                 mfc="white", label="retracted papers (n = 240)")]
    lg = ax_t.legend(handles=hl, loc="lower left", frameon=False,
                     fontsize=6.1, handlelength=1.9, labelspacing=0.30,
                     borderpad=0.15, bbox_to_anchor=(-0.010, -0.015))
    for t in lg.get_texts():
        t.set_color(SUB)

    # ---- aux (c): delay to retraction ---------------------------------------
    ax_b.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    bins = np.logspace(0, np.log10(400.0), 30)
    peak = 0.0
    for name, col, _, _ in LAG_GROUPS:
        counts_b, _, _ = ax_b.hist(lags[name], bins=bins, histtype="step",
                                   color=col, lw=1.4, zorder=5)
        peak = max(peak, counts_b.max())
    ax_b.set_ylim(0, peak * 1.42)
    for (name, col, _, _), fy in zip(LAG_GROUPS, (0.60, 0.74)):
        med = float(np.median(lags[name]))
        ax_b.axvline(med, color=col, lw=0.9, ls=(0, (2.4, 1.8)), zorder=6)
        ax_b.text(med * 1.08, peak * 1.42 * fy, f"median {med:.0f} months",
                  ha="left", va="center", fontsize=6.1, color=col, zorder=8)
    ax_b.set_xscale("log")
    ax_b.set_xlim(1.0, 400.0)
    ax_b.set_xticks([1, 6, 12, 36, 120, 360])
    ax_b.set_xticklabels(["1", "6", "12", "36", "120", "360"])
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("publication to retraction (months, log scale)",
                    fontsize=7.6, labelpad=1.5)
    ax_b.set_ylabel("papers per bin", fontsize=7.6)
    ax_b.set_title("Misconduct takes years to surface", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    lb = [Line2D([], [], color=c, lw=1.4, label=f"{n} (n = 900)")
          for n, c, _, _ in LAG_GROUPS]
    lg2 = ax_b.legend(handles=lb, loc="upper left", frameon=False,
                      fontsize=6.1, handlelength=1.8, labelspacing=0.30,
                      borderpad=0.15, bbox_to_anchor=(-0.010, 1.030))
    for t in lg2.get_texts():
        t.set_color(SUB)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.70, dy=0.12)
    savefig(fig, "995_sci_retraction.png")


if __name__ == "__main__":
    main()

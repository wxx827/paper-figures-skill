"""Likelihood-ratio evidence strength with a verbal-equivalent scale.

Two synthetic score populations - same-source and different-source pairs - are
drawn once and every panel reads from them.  The main panel shows the two
log10 LR densities over the banded verbal scale; the two smaller panels on the
right give the Tippett plot and the ROC curve computed from exactly the same
samples, so the misleading-evidence rates quoted in the main panel and the
operating point marked on the ROC cannot drift apart.

Synthetic illustration for a visual reference library; not a case record.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score, roc_curve

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

ORANGE = "#c0603f"          # same-source (Hp) population
BLUE = "#3b5b92"            # different-source (Hd) population
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

FW, FH = 10.4, 6.4
MAIN = (0.78, 0.88, 5.12, 4.46)
AUX_T = (6.88, 3.59, 2.96, 1.75)
AUX_B = (6.88, 0.88, 2.96, 1.75)

# Verbal-equivalent bands on log10 LR, with light monotone blue tints.
BANDS = [(0.0, 1.0, "limited", "#f8fbfd"),
         (1.0, 2.0, "moderate", "#eff5fa"),
         (2.0, 3.0, "strong", "#e5eef6"),
         (3.0, 4.0, "very strong", "#dae7f2"),
         (4.0, 8.0, "extremely strong", "#cfdfed")]
BOUNDS = [0.0, 1.0, 2.0, 3.0, 4.0]
BOUND_TXT = ["1", "10", "100", r"$10^{3}$", r"$10^{4}$"]
XLO, XHI = -8.0, 8.0


def draw_main(ax, xs, d_hs, d_hd, ls_hs, ls_hd, rates):
    ax.axvspan(XLO, 0.0, facecolor="#f0f0f0", edgecolor="none", zorder=0)
    for lo, hi, name, tint in BANDS:
        ax.axvspan(lo, hi, facecolor=tint, edgecolor="none", zorder=0)

    top = float(max(d_hs.max(), d_hd.max())) * 1.20
    for lo, hi, name, _t in BANDS:
        ax.text(0.5 * (lo + hi) if hi <= 4.0 else 5.6, top * 0.985, name,
                rotation=90, ha="center", va="top", fontsize=6.2, color=SUB,
                zorder=3)
    ax.text(-7.78, top * 0.985, "support for $H_d$", rotation=90,
            ha="center", va="top", fontsize=6.2, color=SUB, zorder=3)

    overlap = np.minimum(d_hs, d_hd)
    ax.fill_between(xs, 0.0, overlap, facecolor="#8a8a8a", alpha=0.52,
                    lw=0.0, zorder=4)
    ax.fill_between(xs, 0.0, d_hd, facecolor=BLUE, alpha=0.20, lw=0.0,
                    zorder=3)
    ax.fill_between(xs, 0.0, d_hs, facecolor=ORANGE, alpha=0.20, lw=0.0,
                    zorder=3)
    ax.plot(xs, d_hd, color=BLUE, lw=1.5, zorder=5,
            label=f"different source, $H_d$ (n = {ls_hd.size})")
    ax.plot(xs, d_hs, color=ORANGE, lw=1.5, zorder=5,
            label=f"same source, $H_p$ (n = {ls_hs.size})")

    ax.axvline(0.0, color="#333333", lw=1.0, zorder=6)
    chip = dict(boxstyle="square,pad=0.14", facecolor="white",
                edgecolor="none", alpha=0.88)
    for b, txt in zip(BOUNDS, BOUND_TXT):
        ax.text(b, top * 0.030, txt, ha="center", va="bottom", fontsize=6.2,
                color=SUB, zorder=7, bbox=chip)
    ax.text(-0.34, top * 0.030, "LR =", ha="right", va="bottom",
            fontsize=6.2, color=SUB, zorder=7, bbox=chip)

    ax.annotate(
        f"misleading evidence\nfor $H_p$: {rates[0] * 100:.2f} %\n"
        f"for $H_d$: {rates[1] * 100:.2f} %",
        xy=(0.0, float(overlap.max()) * 1.05), xytext=(-5.4, top * 0.62),
        ha="center", va="center", fontsize=6.6, color=INK, linespacing=1.45,
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                  edgecolor=HAIR, linewidth=0.6, alpha=0.94),
        arrowprops=dict(arrowstyle="-|>", lw=0.8, color=SUB, shrinkA=2.0,
                        shrinkB=2.0))

    ax.set_xlim(XLO, XHI)
    ax.set_ylim(0.0, top)
    ax.set_xticks(np.arange(-8.0, 8.1, 2.0))
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel(r"$\log_{10}$ LR (-)", fontsize=8.0)
    ax.set_ylabel("probability density (-)", fontsize=8.0)
    ax.set_title("Evidence strength under the two competing hypotheses",
                 fontsize=9.0, fontweight="bold", color=INK)
    ax.legend(loc="upper left", bbox_to_anchor=(0.050, 0.985), fontsize=6.8,
              frameon=False, handlelength=1.5, borderaxespad=0.0)


def draw_tippett(ax, xs, ls_hs, ls_hd, rates):
    p_hs = np.array([np.mean(ls_hs > t) for t in xs]) * 100.0
    p_hd = np.array([np.mean(ls_hd > t) for t in xs]) * 100.0
    ax.plot(xs, p_hd, color=BLUE, lw=1.3)
    ax.plot(xs, p_hs, color=ORANGE, lw=1.3)
    ax.axvline(0.0, color=GREY, lw=0.8, ls=(0, (3.0, 2.0)))
    ax.plot([0.0], [rates[0] * 100.0], marker="o", ms=3.6, color=BLUE,
            mec="white", mew=0.5)
    ax.text(0.35, rates[0] * 100.0 + 4.0, f"{rates[0] * 100:.2f} % of $H_d$\n"
            "pairs give LR > 1", ha="left", va="bottom", fontsize=6.0,
            color=SUB, linespacing=1.3)
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(0.0, 104.0)
    ax.set_xticks(np.arange(-8.0, 8.1, 4.0))
    ax.tick_params(labelsize=6.4)
    ax.set_xlabel(r"$\log_{10}$ LR threshold (-)", fontsize=7.0)
    ax.set_ylabel("pairs exceeding (%)", fontsize=7.0)
    ax.set_title("Tippett plot", fontsize=8.5, fontweight="bold", color=INK)


def draw_det(ax, ls_hs, ls_hd):
    y = np.r_[np.ones(ls_hs.size), np.zeros(ls_hd.size)]
    s = np.r_[ls_hs, ls_hd]
    fpr, tpr, thr = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    fnr = 1.0 - tpr

    floor = 0.02
    f_p = np.clip(fpr * 100.0, floor, 100.0)
    f_n = np.clip(fnr * 100.0, floor, 100.0)
    ax.plot([floor, 100.0], [floor, 100.0], color=GREY, lw=0.8,
            ls=(0, (3.0, 2.0)))
    ax.plot(f_p, f_n, color=ORANGE, lw=1.5)

    k_eer = int(np.argmin(np.abs(fnr - fpr)))
    eer = 0.5 * (fpr[k_eer] + fnr[k_eer]) * 100.0
    ax.plot([eer], [eer], marker="o", ms=4.0, color=INK, mec="white",
            mew=0.6, zorder=5)
    ax.annotate(f"EER = {eer:.2f} %", xy=(eer, eer), xytext=(eer * 3.2, 0.05),
                fontsize=6.2, color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=SUB))

    k1 = int(np.argmin(np.abs(thr - 0.0)))
    ax.plot([max(fpr[k1] * 100.0, floor)], [max(fnr[k1] * 100.0, floor)],
            marker="s", ms=3.8, color=BLUE, mec="white", mew=0.6, zorder=5)
    ax.annotate("LR = 1", xy=(max(fpr[k1] * 100.0, floor),
                              max(fnr[k1] * 100.0, floor)),
                xytext=(0.10, 6.0), fontsize=6.2, color=INK, ha="left",
                va="center", arrowprops=dict(arrowstyle="-", lw=0.7,
                                             color=SUB))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(floor, 100.0)
    ax.set_ylim(floor, 100.0)
    ticks = [0.1, 1.0, 10.0, 100.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(["0.1", "1", "10", "100"])
    ax.set_yticklabels(["0.1", "1", "10", "100"])
    ax.tick_params(labelsize=6.4, which="both")
    ax.set_xlabel("false positive rate (%)", fontsize=7.0)
    ax.set_ylabel("false negative rate (%)", fontsize=7.0)
    ax.set_title(f"DET curve, AUC = {auc:.4f}", fontsize=8.5,
                 fontweight="bold", color=INK)
    return auc


def main():
    set_pub_style()
    rng = np.random.default_rng(19)

    n = 4000
    ls_hs = rng.normal(2.90, 1.35, n)
    ls_hd = rng.normal(-2.20, 1.30, n)
    xs = np.linspace(XLO, XHI, 900)
    d_hs = gaussian_kde(ls_hs)(xs)
    d_hd = gaussian_kde(ls_hd)(xs)

    rate_pros = float(np.mean(ls_hd > 0.0))     # misleading, favours Hp
    rate_def = float(np.mean(ls_hs < 0.0))      # misleading, favours Hd
    rates = (rate_pros, rate_def)

    fig = plt.figure(figsize=(FW, FH))
    ax_main = axes_inches(fig, MAIN)
    ax_tip = axes_inches(fig, AUX_T)
    ax_det = axes_inches(fig, AUX_B)

    draw_main(ax_main, xs, d_hs, d_hd, ls_hs, ls_hd, rates)
    draw_tippett(ax_tip, xs, ls_hs, ls_hd, rates)
    auc = draw_det(ax_det, ls_hs, ls_hd)

    med_hs = float(np.median(ls_hs))
    q_hd = float(np.percentile(ls_hd, 99.0))

    panel_labels_inches(fig, [ax_main], labels=["(a)"], dx=-0.62, dy=0.14)
    panel_labels_inches(fig, [ax_tip, ax_det], labels=["(b)", "(c)"],
                        dx=-0.54, dy=0.14)
    savefig(fig, "793_frn_likelihood_ratio.png")


if __name__ == "__main__":
    main()

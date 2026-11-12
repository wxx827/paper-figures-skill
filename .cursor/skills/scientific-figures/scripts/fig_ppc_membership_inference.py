"""Membership inference: what the attacker actually sees is two overlapping
score distributions.

Main panel: logit-scaled confidence of the target model on its own training
records against held-out records, drawn as mirrored densities - the
undefended model above the axis, the same model trained with a strict
differential-privacy budget below it.  The grey wedge is the overlap of the
two densities, which is exactly the attacker's error region; the balanced
accuracy quoted for each half is the best a single threshold can do, found
by sweeping the threshold over the sampled scores rather than assumed.

Auxiliary (both 58 % of the main panel width): the same three models on a
log-log ROC, where the low-false-positive corner that a linear ROC hides is
the part that decides whether an attack is a real disclosure risk; and
attack balanced accuracy against the privacy budget, with the random-guess
floor and the pure differential-privacy ceiling (e^eps / (1 + e^eps)) drawn
as limits, so the curve runs from a coin flip at a tight budget to the
undefended level as the budget opens up.

Illustrative synthetic scores; no real model, dataset or individual.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
DEEP = "#2b4470"
GREEN = "#2e7d5b"
TEAL = "#2a7f7a"
RUST = "#c0603f"
ACCENT = "#b23a48"
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRID = "#d5d5d5"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.92, 1.00, 5.50, 4.48)
AUX_T_RECT = (7.64, 3.48, 3.14, 2.00)
AUX_B_RECT = (7.64, 1.00, 3.14, 1.86)

N_REC = 20000          # records per population
BASE_MU = 1.20         # held-out mean logit-confidence
BASE_SD = 2.00         # held-out spread
SD_SHRINK = 0.38       # members are also tighter, in proportion to the gap
GAP_MAX = 2.00         # member-minus-holdout gap of the undefended model
EPS_HALF = 3.0         # budget at which half the gap has come back
GRID_X = np.linspace(-7.5, 10.5, 500)


def gap_at(eps):
    """Member/held-out separation left after training at budget eps.

    Zero at a vanishing budget (the two populations are then identical and
    the attacker can only guess) and saturating at the undefended gap.
    """
    eps = np.asarray(eps, dtype=float)
    return GAP_MAX * eps / (eps + EPS_HALF)


def draw_scores(rng, gap, n=N_REC):
    """(member, held-out) logit-scaled confidences for one training run."""
    sd_mem = BASE_SD - SD_SHRINK * gap / GAP_MAX
    mem = rng.normal(BASE_MU + gap, sd_mem, n)
    out = rng.normal(BASE_MU, BASE_SD, n)
    return mem, out


def roc_from_scores(mem, out, thr):
    """TPR / FPR of the threshold rule "score > thr means member"."""
    tpr = 1.0 - np.searchsorted(np.sort(mem), thr, side="right") / mem.size
    fpr = 1.0 - np.searchsorted(np.sort(out), thr, side="right") / out.size
    return tpr, fpr


def best_balanced_accuracy(mem, out, n_thr=400):
    """Best single-threshold balanced accuracy, swept over the samples."""
    thr = np.quantile(np.concatenate([mem, out]),
                      np.linspace(0.002, 0.998, n_thr))
    tpr, fpr = roc_from_scores(mem, out, thr)
    acc = 0.5 * (tpr + (1.0 - fpr))
    i = int(np.argmax(acc))
    return float(acc[i]), float(thr[i])


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(1015)

    models = []
    for name, eps, col in (("undefended", np.inf, ACCENT),
                           ("DP-SGD, eps = 8", 8.0, BLUE),
                           ("DP-SGD, eps = 1", 1.0, GREEN)):
        gap = GAP_MAX if not np.isfinite(eps) else float(gap_at(eps))
        mem, out = draw_scores(rng, gap)
        acc, thr = best_balanced_accuracy(mem, out)
        models.append(dict(name=name, eps=eps, col=col, mem=mem, out=out,
                           gap=gap, acc=acc, thr=thr))

    # ---- densities for the mirrored main panel ------------------------------
    for m in (models[0], models[2]):
        m["d_mem"] = gaussian_kde(m["mem"])(GRID_X)
        m["d_out"] = gaussian_kde(m["out"])(GRID_X)
        m["d_ovl"] = np.minimum(m["d_mem"], m["d_out"])
        # total-variation distance between the plotted curves; the optimal
        # attacker's advantage is exactly this quantity
        step = GRID_X[1] - GRID_X[0]
        m["tv"] = float(0.5 * np.abs(m["d_mem"] - m["d_out"]).sum() * step)
        m["ovl"] = float(m["d_ovl"].sum() * step)

    # ---- ROC ----------------------------------------------------------------
    thr_roc = np.quantile(np.concatenate([models[0]["mem"], models[0]["out"]]),
                          np.linspace(1e-4, 1.0 - 1e-4, 4000))
    for m in models:
        tpr, fpr = roc_from_scores(m["mem"], m["out"], thr_roc)
        m["tpr"], m["fpr"] = tpr, fpr
        m["tpr_at"] = float(np.interp(1e-3, fpr[::-1], tpr[::-1]))

    # ---- attack accuracy against the privacy budget -------------------------
    eps_grid = np.logspace(np.log10(0.1), np.log10(100.0), 24)
    acc_eps = np.array([best_balanced_accuracy(
        *draw_scores(rng, float(gap_at(e)), 40000), n_thr=220)[0]
        for e in eps_grid])
    bound = np.exp(eps_grid) / (1.0 + np.exp(eps_grid))
    ceiling = models[0]["acc"]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main panel: mirrored densities -------------------------------------
    ax.grid(color=GRID, lw=0.5, alpha=0.5, axis="x", zorder=0)
    top, bot = models[0], models[2]
    for m, sign in ((top, +1.0), (bot, -1.0)):
        ax.fill_between(GRID_X, 0.0, sign * m["d_out"], color=lighten(BLUE, .55),
                        alpha=0.85, lw=0, zorder=3)
        ax.fill_between(GRID_X, 0.0, sign * m["d_mem"], color=lighten(RUST, .45),
                        alpha=0.70, lw=0, zorder=4)
        ax.fill_between(GRID_X, 0.0, sign * m["d_ovl"], color="#9a9a9a",
                        alpha=0.55, lw=0, zorder=5)
        ax.plot(GRID_X, sign * m["d_out"], color=DEEP, lw=1.5, zorder=6)
        ax.plot(GRID_X, sign * m["d_mem"], color=ACCENT, lw=1.7, zorder=6)
        ax.plot([m["thr"], m["thr"]], [0.0, sign * 0.272], color=INK, lw=1.0,
                ls=(0, (3.0, 2.0)), zorder=7)
    ax.axhline(0.0, color="#333333", lw=0.9, zorder=8)

    ax.text(-7.2, 0.300,
            "undefended target model\n"
            f"attacker's balanced accuracy {100 * top['acc']:.1f} %  "
            f"(advantage {100 * (2 * top['acc'] - 1):.1f} points)\n"
            f"score distributions overlap on {100 * top['ovl']:.0f} % of "
            "their mass",
            ha="left", va="top", fontsize=7.0, color=INK, linespacing=1.55,
            fontweight="bold")
    ax.text(-7.2, -0.300,
            f"same model trained with DP-SGD at $\\varepsilon$ = 1\n"
            f"attacker's balanced accuracy {100 * bot['acc']:.1f} %  "
            f"(advantage {100 * (2 * bot['acc'] - 1):.1f} points)\n"
            f"score distributions overlap on {100 * bot['ovl']:.0f} % of "
            "their mass",
            ha="left", va="bottom", fontsize=7.0, color=INK, linespacing=1.55,
            fontweight="bold")
    ax.annotate("overlap = records the attacker\nmust get wrong",
                xy=(1.0, 0.5 * float(np.interp(1.0, GRID_X, top["d_ovl"]))),
                xytext=(-6.9, 0.115), fontsize=6.6, color="#5d5d5d",
                ha="left", va="center", linespacing=1.5,
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#8a8a8a",
                                connectionstyle="arc3,rad=-0.2"))
    ax.annotate(f"best threshold {top['thr']:.2f}",
                xy=(top["thr"], 0.250), xytext=(5.2, 0.145), fontsize=6.6,
                color=INK, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=INK,
                                connectionstyle="arc3,rad=-0.2"))

    ax.set_xlim(GRID_X[0], GRID_X[-1])
    ax.set_ylim(-0.325, 0.325)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{abs(v):.2f}"))
    ax.set_yticks([-0.25, -0.125, 0.0, 0.125, 0.25])
    ax.set_xlabel("logit-scaled confidence on the true class "
                  "(dimensionless)", fontsize=8.5)
    ax.set_ylabel("probability density (per unit of logit-confidence)",
                  fontsize=8.5)
    ax.set_title(f"Attack score densities, {N_REC:,} records per group",
                 fontsize=9.0, fontweight="bold", pad=5.0)
    hm = [Line2D([], [], color=ACCENT, lw=1.7, label="training members"),
          Line2D([], [], color=DEEP, lw=1.5, label="held-out non-members"),
          Line2D([], [], color="#9a9a9a", lw=5.0, alpha=0.55,
                 label="overlap (attacker error)"),
          Line2D([], [], color=INK, lw=1.0, ls=(0, (3.0, 2.0)),
                 label="best single threshold")]
    leg = ax.legend(handles=hm, loc="upper right", frameon=False, fontsize=6.8,
                    labelspacing=0.42, handlelength=2.0, borderpad=0.2,
                    bbox_to_anchor=(1.008, 1.010))
    for t in leg.get_texts():
        t.set_color(SUB)

    # ---- aux (b): log-log ROC ----------------------------------------------
    ax_t.grid(color=GRID, lw=0.5, alpha=0.7, which="major", zorder=0)
    ax_t.plot([1e-4, 1.0], [1e-4, 1.0], color=GREY, lw=0.9,
              ls=(0, (3.2, 2.2)), zorder=3)
    for m in models:
        ax_t.loglog(np.clip(m["fpr"], 1e-5, None), np.clip(m["tpr"], 1e-5, None),
                    color=m["col"], lw=1.6, zorder=5,
                    label=f"{m['name']}: TPR {100 * m['tpr_at']:.2f} % "
                          "at 0.1 % FPR")
    ax_t.axvline(1e-3, color=INK, lw=0.7, ls=(0, (2.2, 1.8)), zorder=4)
    ax_t.set_xlim(1e-4, 1.0)
    ax_t.set_ylim(1e-3, 1.0)
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("false positive rate (fraction, log scale)", fontsize=7.6,
                    labelpad=1.5)
    ax_t.set_ylabel("true positive rate\n(fraction, log scale)", fontsize=7.6)
    ax_t.set_title("The low-false-positive corner is the one that matters",
                   fontsize=8.4, fontweight="bold", pad=4.0)
    lt = ax_t.legend(loc="upper left", frameon=False, fontsize=6.0,
                     handlelength=1.7, labelspacing=0.30, borderpad=0.15,
                     bbox_to_anchor=(-0.012, 1.035))
    for t in lt.get_texts():
        t.set_color(SUB)

    # ---- aux (c): attack success against the budget -------------------------
    ax_b.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    ax_b.axhline(50.0, color=GREY, lw=1.0, ls=(0, (3.2, 2.2)), zorder=3)
    ax_b.axhline(100.0 * ceiling, color=ACCENT, lw=1.0, ls=(0, (3.2, 2.2)),
                 zorder=3)
    ax_b.plot(eps_grid, 100.0 * bound, color=INK, lw=1.1, ls=(0, (1.6, 1.6)),
              zorder=5)
    ax_b.plot(eps_grid, 100.0 * acc_eps, color=TEAL, lw=1.9, zorder=6)
    ax_b.plot(eps_grid, 100.0 * acc_eps, marker="o", ls="none", ms=2.8,
              mfc="white", mec=TEAL, mew=0.9, zorder=7)
    ax_b.set_xscale("log")
    ax_b.set_xlim(0.085, 130.0)
    ax_b.set_ylim(46.0, 104.0)
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel(r"privacy budget $\varepsilon$ "
                    "(dimensionless, log scale)", fontsize=7.6, labelpad=1.5)
    ax_b.set_ylabel("attack balanced\naccuracy (%)", fontsize=7.6)
    ax_b.set_title("Attack success between the two limits", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_b.text(0.095, 49.4, "random guessing, 50 %", fontsize=6.0, color=SUB,
              ha="left", va="top")
    ax_b.text(0.095, 100.0 * ceiling - 1.6,
              f"undefended model, {100 * ceiling:.1f} %", fontsize=6.0,
              color=ACCENT, ha="left", va="top")
    ax_b.text(115.0, 95.5,
              r"pure-DP ceiling $e^{\varepsilon}/(1+e^{\varepsilon})$",
              fontsize=6.0, color=INK, ha="right", va="top")

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.64, dy=0.12)
    savefig(fig, "1015_ppc_membership_inference.png")


if __name__ == "__main__":
    main()

"""Visual working memory capacity estimated from a change-detection task.

Main panel: hit rate and correct-rejection rate against memory set size for
two encoding durations, with Cowan's K plotted on a second axis.  K is
computed per set size from the same hit and false-alarm arrays that are drawn
below it, so the capacity ceiling on the figure is a consequence of the
accuracy curves rather than an annotation placed on top of them.

Auxiliary (both at 58 % of the main panel width): the distribution of the
per-participant capacity estimate with its mean and interquartile range; and
the slot-plus-averaging prediction, where the fitted capacity limit is
recovered by least squares from the accuracy curve and compared with the
generating value.

Fictional 32-participant experiment, illustrative values only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
ORANGE = "#d98032"
TEAL = "#2e7d5b"
RUST = "#c0603f"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.86, 0.86, 5.54, 4.62)
AUX_T_RECT = (7.62, 3.48, 3.16, 2.00)
AUX_B_RECT = (7.62, 0.86, 3.16, 2.00)

SET_SIZES = np.array([1, 2, 3, 4, 6, 8])
N_SUBJ = 32
N_TRIALS = 40          # change trials and no-change trials per cell
GUESS_BIAS = 0.52      # probability of saying "change" when guessing
K_TRUE = {"200 ms encoding": 2.35, "1000 ms encoding": 3.30}
COLORS = {"200 ms encoding": ORANGE, "1000 ms encoding": BLUE}


def slot_accuracy(n, k, guess=GUESS_BIAS):
    """Slot model: an item in memory is judged, otherwise the observer guesses."""
    p_in = np.minimum(k / n, 1.0)
    hit = p_in + (1.0 - p_in) * guess
    fa = (1.0 - p_in) * guess
    return hit, fa


def cowan_k(hit, fa, n):
    """Cowan's single-probe capacity, in items."""
    return n * (np.asarray(hit) - np.asarray(fa))


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(958)

    subj_k = {c: np.clip(rng.normal(k, 0.62, N_SUBJ), 0.6, 6.5)
              for c, k in K_TRUE.items()}

    hits, fas, k_per_subj = {}, {}, {}
    for cond, ks in subj_k.items():
        h_mat = np.zeros((N_SUBJ, SET_SIZES.size))
        f_mat = np.zeros((N_SUBJ, SET_SIZES.size))
        for s in range(N_SUBJ):
            h_p, f_p = slot_accuracy(SET_SIZES.astype(float), ks[s])
            h_mat[s] = rng.binomial(N_TRIALS, h_p) / N_TRIALS
            f_mat[s] = rng.binomial(N_TRIALS, f_p) / N_TRIALS
        hits[cond] = h_mat
        fas[cond] = f_mat
        big = SET_SIZES >= 4
        k_per_subj[cond] = cowan_k(h_mat[:, big], f_mat[:, big],
                                   SET_SIZES[big]).mean(axis=1)

    k_curve = {c: cowan_k(hits[c].mean(axis=0), fas[c].mean(axis=0),
                          SET_SIZES) for c in K_TRUE}

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: accuracy curves with capacity on a second axis ---------------
    ax.grid(color=GRID, lw=0.5, alpha=0.6, zorder=0)
    ax.axhline(50.0, color=FAINT, lw=0.9, ls=(0, (3.2, 2.4)), zorder=1)
    ax.text(8.2, 51.0, "chance = 50 %", ha="right", va="bottom", fontsize=6.5,
            color=MUTED)

    for cond in K_TRUE:
        col = COLORS[cond]
        h = 100 * hits[cond].mean(axis=0)
        cr = 100 * (1.0 - fas[cond].mean(axis=0))
        h_se = 100 * hits[cond].std(axis=0, ddof=1) / np.sqrt(N_SUBJ)
        cr_se = 100 * fas[cond].std(axis=0, ddof=1) / np.sqrt(N_SUBJ)
        ax.errorbar(SET_SIZES, h, yerr=1.96 * h_se, color=col, lw=1.6,
                    marker="o", ms=4.6, mfc=col, mec="white", mew=0.8,
                    ecolor=col, elinewidth=0.8, capsize=1.8, zorder=5)
        ax.errorbar(SET_SIZES, cr, yerr=1.96 * cr_se, color=col, lw=1.1,
                    ls=(0, (3.4, 2.0)), marker="s", ms=3.8, mfc="white",
                    mec=col, mew=0.9, ecolor=col, elinewidth=0.8, capsize=1.8,
                    zorder=5)

    ax.set_xlim(0.4, 8.6)
    ax.set_ylim(44.0, 104.0)
    ax.set_xticks(SET_SIZES)
    ax.set_yticks(np.arange(50, 101, 10))
    ax.set_xlabel("memory set size (items)", fontsize=8.5)
    ax.set_ylabel("response rate (%)", fontsize=8.5)
    ax.set_title("Accuracy falls once the array outgrows the store, and the "
                 "capacity estimate flattens", fontsize=9.0,
                 fontweight="bold", pad=5.0)

    ax_k = ax.twinx()
    ax_k.spines["top"].set_visible(False)
    ax_k.set_zorder(0.5)
    ax.set_zorder(2.0)
    ax.patch.set_visible(False)
    for cond in K_TRUE:
        col = COLORS[cond]
        ax_k.fill_between(SET_SIZES, 0.0, k_curve[cond],
                          facecolor=lighten(col, 0.88), edgecolor="none",
                          zorder=0)
        ax_k.plot(SET_SIZES, k_curve[cond], color=col, lw=1.3,
                  ls=(0, (1.4, 1.6)), marker="^", ms=4.0, mfc="white",
                  mec=col, mew=0.9, zorder=3)
        plateau = float(k_curve[cond][SET_SIZES >= 4].mean())
        ax_k.axhline(plateau, color=col, lw=0.8, alpha=0.55, zorder=2)
        ax_k.text(8.5, plateau + 0.06, f"K = {plateau:.2f} items",
                  ha="right", va="bottom", fontsize=6.8, color=col)
    ax_k.set_ylim(0.0, 9.6)
    ax_k.set_yticks([0, 1, 2, 3, 4])
    ax_k.set_ylabel("Cowan's K (items)", fontsize=8.5, labelpad=2.0)
    ax_k.tick_params(labelsize=7.2)

    handles = [
        Line2D([], [], color=SUB, lw=1.6, marker="o", ms=4.6, mfc=SUB,
               mec="white", mew=0.8, label="hit rate (change detected)"),
        Line2D([], [], color=SUB, lw=1.1, ls=(0, (3.4, 2.0)), marker="s",
               ms=3.8, mfc="white", mec=SUB, label="correct rejection rate"),
        Line2D([], [], color=SUB, lw=1.3, ls=(0, (1.4, 1.6)), marker="^",
               ms=4.0, mfc="white", mec=SUB, label="Cowan's K, right axis"),
        Line2D([], [], color=ORANGE, lw=3, label="200 ms encoding"),
        Line2D([], [], color=BLUE, lw=3, label="1000 ms encoding"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=6.9,
              handlelength=2.1, labelspacing=0.34, borderpad=0.12,
              bbox_to_anchor=(-0.004, -0.008))
    ax.text(0.55, 103.0,
            f"n={N_SUBJ} participants, {N_TRIALS} change and {N_TRIALS} "
            "no-change trials per cell\n"
            "K is computed per set size as n x (hit rate - false alarm rate)",
            ha="left", va="top", fontsize=6.5, color=MUTED, linespacing=1.5)

    # ---- aux (b): the spread of individual capacity -------------------------
    ax_t.grid(axis="x", color=GRID, lw=0.5, alpha=0.6, zorder=0)
    for i, cond in enumerate(K_TRUE):
        col = COLORS[cond]
        vals = k_per_subj[cond]
        y = 1 - i
        jit = (rng.random(vals.size) - 0.5) * 0.26
        ax_t.plot(vals, y + jit, marker="o", ls="none", ms=3.0, mfc="white",
                  mec=col, mew=0.8, alpha=0.9, zorder=4)
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax_t.plot([q1, q3], [y - 0.30, y - 0.30], color=col, lw=3.4,
                  solid_capstyle="butt", zorder=5)
        ax_t.plot([med], [y - 0.30], marker="|", ms=7.0, color="white",
                  mew=1.4, zorder=6)
        ax_t.text(vals.mean(), y + 0.36,
                  f"mean {vals.mean():.2f}, IQR {q1:.2f}-{q3:.2f} items",
                  ha="center", va="bottom", fontsize=6.3, color=col)
        ax_t.plot([K_TRUE[cond]], [y - 0.30], marker="v", ms=4.2, mfc=INK,
                  mec="white", mew=0.7, zorder=7)
    ax_t.set_yticks([0, 1])
    ax_t.set_yticklabels(["1000 ms", "200 ms"], fontsize=6.8)
    ax_t.set_ylim(-0.72, 1.78)
    ax_t.set_xlim(0.4, 6.0)
    ax_t.tick_params(axis="x", labelsize=6.6)
    ax_t.set_xlabel("per-participant capacity K (items)", fontsize=7.6,
                    labelpad=1.5)
    ax_t.set_title("Capacity varies two-fold across people", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_t.text(5.9, -0.66, "black triangles are the generating capacities",
              ha="right", va="bottom", fontsize=6.1, color=MUTED)

    # ---- aux (c): slot-model fit to the accuracy curve ----------------------
    ax_b.grid(color=GRID, lw=0.5, alpha=0.6, zorder=0)
    xs = np.linspace(0.8, 8.4, 200)
    fit_txt = []
    for cond in K_TRUE:
        col = COLORS[cond]
        acc = 0.5 * (hits[cond].mean(axis=0) + 1.0 - fas[cond].mean(axis=0))

        def model(n, k):
            h, f = slot_accuracy(n, k)
            return 0.5 * (h + 1.0 - f)

        (k_hat,), _ = curve_fit(model, SET_SIZES.astype(float), acc,
                                p0=[3.0], bounds=([0.2], [10.0]))
        ax_b.plot(SET_SIZES, 100 * acc, marker="o", ls="none", ms=4.4,
                  mfc=col, mec="white", mew=0.8, zorder=5)
        ax_b.plot(xs, 100 * model(xs, k_hat), color=col, lw=1.4, zorder=4)
        fit_txt.append(f"{cond}: K fitted {k_hat:.2f} vs generating "
                       f"{K_TRUE[cond]:.2f} items")
    ax_b.text(0.75, 60.0, "\n".join(fit_txt) +
              "\noverall accuracy is (1 + K/n)/2, so the guess bias "
              "cancels here",
              ha="left", va="bottom", fontsize=6.2, color=SUB,
              linespacing=1.5)
    ax_b.set_xlim(0.4, 8.6)
    ax_b.set_ylim(55.0, 104.0)
    ax_b.set_xticks(SET_SIZES)
    ax_b.set_yticks([60, 70, 80, 90, 100])
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("memory set size (items)", fontsize=7.6, labelpad=1.5)
    ax_b.set_ylabel("overall accuracy (%)", fontsize=7.6, labelpad=2.0)
    ax_b.set_title("A slot model recovers the generating capacity",
                   fontsize=8.4, fontweight="bold", pad=4.0)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.62, dy=0.12)
    plateau_short = float(k_curve["200 ms encoding"][SET_SIZES >= 4].mean())
    plateau_long = float(k_curve["1000 ms encoding"][SET_SIZES >= 4].mean())
    savefig(fig, "958_cog_wm_capacity.png")


if __name__ == "__main__":
    main()

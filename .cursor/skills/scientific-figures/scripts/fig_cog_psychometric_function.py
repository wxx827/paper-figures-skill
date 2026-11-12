"""Psychometric function and the adaptive staircase that finds its threshold.

Main panel: proportion correct against log contrast for an unmasked and a
masked two-alternative forced-choice detection task.  Binomial data are
simulated from a Weibull generating function, a Weibull is then re-fitted by
maximum likelihood, and the 75 % threshold, the slope at threshold and the
bootstrap confidence band are all read back out of the fitted curve rather
than written down.

Auxiliary (both at 58 % of the main panel width): the trial-by-trial track of
a 3-down-1-up staircase on the masked condition with its reversals marked and
the last-reversal mean drawn through; and a threshold recovery panel that puts
the generating value, the constant-stimuli fit and twelve independent
staircase runs on one contrast axis.

Fictional observer, illustrative values only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import minimize

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
ORANGE = "#d98032"
TEAL = "#2e7d5b"
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

GAMMA = 0.50          # guess rate of a two-alternative forced choice task
LAPSE = 0.02          # stimulus-independent lapse rate
LEVELS = np.array([0.8, 1.3, 2.0, 3.2, 5.0, 8.0, 12.5, 20.0])   # contrast (%)
N_PER_LEVEL = 60      # trials per contrast level per condition

CONDITIONS = [
    ("unmasked", 2.35, 2.60, BLUE),
    ("masked", 5.60, 2.20, ORANGE),
]


def weibull(x, alpha, beta):
    """Proportion correct for a 2AFC Weibull with a fixed lapse rate."""
    x = np.maximum(np.asarray(x, dtype=float), 1e-9)
    return GAMMA + (1.0 - GAMMA - LAPSE) * (1.0 - np.exp(-(x / alpha) ** beta))


def fit_weibull(x, k, n):
    """Maximum-likelihood Weibull fit to binomial counts."""
    def nll(p):
        a, b = np.exp(p)
        q = np.clip(weibull(x, a, b), 1e-9, 1.0 - 1e-9)
        return -float(np.sum(k * np.log(q) + (n - k) * np.log1p(-q)))
    start = np.log([float(np.median(x)), 2.0])
    res = minimize(nll, start, method="Nelder-Mead",
                   options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000))
    return float(np.exp(res.x[0])), float(np.exp(res.x[1]))


def threshold_from_curve(alpha, beta, target=0.75, grid=None):
    """Invert the fitted curve numerically on the grid that gets plotted."""
    g = XGRID if grid is None else grid
    p = weibull(g, alpha, beta)
    return float(np.interp(target, p, g))


def slope_per_log_unit(alpha, beta, thr):
    """d(proportion correct) / d(log10 contrast) at the threshold, in % units."""
    h = 0.01
    lo, hi = thr * 10.0 ** (-h), thr * 10.0 ** (h)
    return 100.0 * float(weibull(hi, alpha, beta) - weibull(lo, alpha, beta)) / (2 * h)


XGRID = np.logspace(np.log10(0.45), np.log10(30.0), 900)


def run_staircase(rng, alpha, beta, n_trials=70, start=14.0):
    """3-down-1-up staircase in log contrast; returns the track and reversals."""
    step = 0.16                       # log10 units, halved after 2 reversals
    x = np.log10(start)
    direction = 0
    correct_run = 0
    track, rev_idx = [], []
    for t in range(n_trials):
        c = 10.0 ** x
        track.append(c)
        hit = rng.random() < weibull(c, alpha, beta)
        if hit:
            correct_run += 1
        else:
            correct_run = 0
        move = 0
        if correct_run >= 3:
            move, correct_run = -1, 0
        elif not hit:
            move = +1
        if move:
            if direction and move != direction:
                rev_idx.append(t)
                if len(rev_idx) == 2:
                    step = 0.08
                elif len(rev_idx) == 4:
                    step = 0.05
            direction = move
            x = float(np.clip(x + move * step, np.log10(0.5), np.log10(28.0)))
    return np.array(track), np.array(rev_idx, dtype=int)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(954)

    fits, data = {}, {}
    for name, a_true, b_true, _ in CONDITIONS:
        p_true = weibull(LEVELS, a_true, b_true)
        k = rng.binomial(N_PER_LEVEL, p_true)
        a_hat, b_hat = fit_weibull(LEVELS, k, N_PER_LEVEL)
        thr = threshold_from_curve(a_hat, b_hat)
        data[name] = k
        fits[name] = dict(a=a_hat, b=b_hat, thr=thr,
                          slope=slope_per_log_unit(a_hat, b_hat, thr),
                          a_true=a_true, b_true=b_true,
                          thr_true=threshold_from_curve(a_true, b_true))

    # bootstrap band and threshold interval from the fitted curve itself
    n_boot = 260
    boot = {}
    for name, _, _, _ in CONDITIONS:
        f = fits[name]
        curves, thrs = [], []
        for _ in range(n_boot):
            kb = rng.binomial(N_PER_LEVEL, weibull(LEVELS, f["a"], f["b"]))
            ab, bb = fit_weibull(LEVELS, kb, N_PER_LEVEL)
            curves.append(weibull(XGRID, ab, bb))
            thrs.append(threshold_from_curve(ab, bb))
        curves = np.array(curves)
        boot[name] = dict(lo=np.percentile(curves, 2.5, axis=0),
                          hi=np.percentile(curves, 97.5, axis=0),
                          t_lo=float(np.percentile(thrs, 2.5)),
                          t_hi=float(np.percentile(thrs, 97.5)))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: the two fitted psychometric functions -----------------------
    ax.grid(color=GRID, lw=0.5, alpha=0.6, zorder=0)
    ax.axhline(100 * GAMMA, color=FAINT, lw=0.9, ls=(0, (3.2, 2.4)), zorder=1)
    ax.text(28.0, 100 * GAMMA + 1.1, "chance = 50 % (2AFC guess rate)",
            fontsize=6.6, color=MUTED, ha="right", va="bottom")
    ax.axhline(75.0, color=FAINT, lw=0.9, ls=(0, (1.6, 1.8)), zorder=1)

    for name, _, _, col in CONDITIONS:
        f, bt = fits[name], boot[name]
        ax.fill_between(XGRID, 100 * bt["lo"], 100 * bt["hi"],
                        facecolor=lighten(col, 0.72), edgecolor="none",
                        zorder=2)
        ax.plot(XGRID, 100 * weibull(XGRID, f["a"], f["b"]), color=col,
                lw=1.8, zorder=5, label=f"{name} condition")
        p_obs = 100.0 * data[name] / N_PER_LEVEL
        se = 100.0 * np.sqrt((data[name] / N_PER_LEVEL) *
                             (1 - data[name] / N_PER_LEVEL) / N_PER_LEVEL)
        ax.errorbar(LEVELS, p_obs, yerr=se, fmt="o", ms=4.4, mfc="white",
                    mec=col, mew=1.2, ecolor=col, elinewidth=0.9, capsize=2.0,
                    zorder=6)
        ax.plot([f["thr"], f["thr"]], [46.0, 75.0], color=col, lw=1.0,
                ls=(0, (2.6, 2.0)), zorder=4)
        ax.plot([f["thr"]], [75.0], marker="D", ms=5.0, mfc=col, mec="white",
                mew=0.8, zorder=7)

    f_u, f_m = fits["unmasked"], fits["masked"]
    ax.annotate("", xy=(f_m["thr"], 55.0), xytext=(f_u["thr"], 55.0),
                arrowprops=dict(arrowstyle="<|-|>", color=SUB, lw=1.0,
                                shrinkA=0, shrinkB=0, mutation_scale=8))
    ax.text(np.sqrt(f_u["thr"] * f_m["thr"]), 56.2,
            f"masking cost = {np.log10(f_m['thr'] / f_u['thr']):.2f} log units "
            f"({f_m['thr'] / f_u['thr']:.2f}x contrast)",
            ha="center", va="bottom", fontsize=6.9, color=SUB)

    ax.text(0.50, 102.0,
            f"maximum-likelihood fit, n={N_PER_LEVEL} trials per point\n"
            f"unmasked  thr {f_u['thr']:.2f} % "
            f"[{boot['unmasked']['t_lo']:.2f}, {boot['unmasked']['t_hi']:.2f}]\n"
            f"                     slope {f_u['slope']:.0f} %/log unit\n"
            f"masked      thr {f_m['thr']:.2f} % "
            f"[{boot['masked']['t_lo']:.2f}, {boot['masked']['t_hi']:.2f}]\n"
            f"                     slope {f_m['slope']:.0f} %/log unit\n"
            f"generating thresholds {f_u['thr_true']:.2f} / "
            f"{f_m['thr_true']:.2f} %",
            ha="left", va="top", fontsize=6.9, color=SUB, linespacing=1.60)

    ax.set_xscale("log")
    ax.set_xlim(0.45, 30.0)
    ax.set_ylim(44.0, 103.0)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax.set_yticks([50, 60, 70, 80, 90, 100])
    ax.set_xlabel("stimulus contrast (%, log axis)", fontsize=8.5)
    ax.set_ylabel("proportion correct (%)", fontsize=8.5)
    ax.set_title("A mask shifts the psychometric function to the right "
                 "without flattening it", fontsize=9.0, fontweight="bold",
                 pad=5.0)
    handles = [
        Line2D([], [], color=BLUE, lw=1.8, label="unmasked"),
        Line2D([], [], color=ORANGE, lw=1.8, label="masked"),
        Line2D([], [], color=FAINT, lw=0.9, ls=(0, (1.6, 1.8)),
               label="75 % threshold criterion"),
        Line2D([], [], color=SUB, lw=0, marker="D", ms=5.0,
               label="threshold read off the fit"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=6.9,
              handlelength=1.5, labelspacing=0.32, borderpad=0.15,
              bbox_to_anchor=(1.004, -0.006))

    # ---- aux (b): one staircase track --------------------------------------
    a_m, b_m = f_m["a"], f_m["b"]
    track, revs = run_staircase(rng, a_m, b_m)
    tail = revs[-8:] if revs.size >= 8 else revs
    stair_thr = float(np.mean(track[tail])) if tail.size else np.nan
    trials = np.arange(1, track.size + 1)

    ax_t.grid(color=GRID, lw=0.5, alpha=0.6, zorder=0)
    ax_t.plot(trials, track, color=ORANGE, lw=1.1, zorder=4)
    ax_t.plot(trials[revs], track[revs], marker="v", ls="none", ms=3.6,
              mfc="white", mec=ORANGE, mew=0.9, zorder=5)
    ax_t.plot(trials[tail], track[tail], marker="v", ls="none", ms=3.8,
              mfc=ACCENT, mec="white", mew=0.7, zorder=6)
    ax_t.axhline(stair_thr, color=ACCENT, lw=1.2, zorder=3)
    ax_t.axhline(f_m["thr"], color=BLUE, lw=1.0, ls=(0, (3.0, 2.2)), zorder=3)
    ax_t.text(track.size - 0.5, stair_thr * 1.10,
              f"last-8-reversal mean = {stair_thr:.2f} %",
              ha="right", va="bottom", fontsize=6.2, color=ACCENT)
    ax_t.text(1.0, f_m["thr"] * 0.80, "constant-stimuli 75 % threshold",
              ha="left", va="top", fontsize=6.2, color=BLUE)
    ax_t.set_yscale("log")
    ax_t.set_ylim(0.8, 30.0)
    ax_t.set_yticks([1, 3, 10, 30])
    ax_t.set_yticklabels(["1", "3", "10", "30"])
    ax_t.set_xlim(0, track.size + 1)
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("trial index (n)", fontsize=7.6, labelpad=1.5)
    ax_t.set_ylabel("contrast (%)", fontsize=7.6, labelpad=2.0)
    ax_t.set_title(f"3-down-1-up staircase, {revs.size} reversals",
                   fontsize=8.4, fontweight="bold", pad=4.0)

    # ---- aux (c): threshold recovery ---------------------------------------
    runs = np.array([np.mean(run_staircase(rng, a_m, b_m)[0][-16:])
                     for _ in range(12)])
    stair_mu, stair_sd = float(runs.mean()), float(runs.std(ddof=1))

    thr79_true = threshold_from_curve(f_m["a_true"], f_m["b_true"], 0.794)

    ax_b.grid(axis="x", color=GRID, lw=0.5, alpha=0.6, zorder=0)
    rows = [
        ("generating, 75 % pt", f_m["thr_true"], None, INK, "s"),
        ("constant stimuli", f_m["thr"],
         (boot["masked"]["t_lo"], boot["masked"]["t_hi"]), BLUE, "D"),
        ("generating, 79 % pt", thr79_true, None, INK, "s"),
        ("staircase, 12 runs", stair_mu,
         (stair_mu - 1.96 * stair_sd / np.sqrt(runs.size),
          stair_mu + 1.96 * stair_sd / np.sqrt(runs.size)), ORANGE, "o"),
    ]
    for ref, col in ((f_m["thr_true"], BLUE), (thr79_true, ORANGE)):
        ax_b.axvline(ref, color=lighten(col, 0.45), lw=0.9,
                     ls=(0, (3.0, 2.2)), zorder=1)
    for i, (lab, val, ci, col, mk) in enumerate(rows):
        y = len(rows) - 1 - i
        if ci is not None:
            ax_b.plot(list(ci), [y, y], color=col, lw=1.6, zorder=4,
                      solid_capstyle="butt")
        ax_b.plot([val], [y], marker=mk, ms=5.0, mfc=col, mec="white",
                  mew=0.8, zorder=6)
        ax_b.text(val, y + 0.22, f"{val:.2f} %", ha="center", va="bottom",
                  fontsize=6.3, color=col)
    ax_b.scatter(runs, np.full(runs.size, 0.34), s=9, marker="|",
                 color=lighten(ORANGE, 0.25), zorder=5, linewidths=0.9)
    ax_b.set_yticks(range(len(rows)))
    ax_b.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.5)
    ax_b.set_ylim(-0.55, len(rows) - 0.28)
    ax_b.set_xlim(3.8, 8.4)
    ax_b.tick_params(axis="x", labelsize=6.6)
    ax_b.set_xlabel("estimated threshold, masked (% contrast)",
                    fontsize=7.6, labelpad=1.5)
    ax_b.set_title("Each rule recovers the criterion it targets",
                   fontsize=8.4, fontweight="bold", pad=4.0)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.62, dy=0.12)
    savefig(fig, "954_cog_psychometric_function.png")


if __name__ == "__main__":
    main()

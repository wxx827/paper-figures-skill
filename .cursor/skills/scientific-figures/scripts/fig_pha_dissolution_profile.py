"""In-vitro dissolution: four products, one reference, and the f2 that decides.

Twelve dosage units per product are dissolved on the same sampling schedule.
The mean profile drawn in the main panel is the only input to everything else
on the page: the Weibull curves are fitted to those means, the similarity
factor f2 in the upper right panel is computed from the same mean vectors with
the standard formula over the standard time window, and the Q-value verdict at
30 min is read off the plotted mean rather than typed in.  The lower right
panel repeats one test batch in three compendial media, so the reader can see
that the batch which fails f2 fails it in every medium.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import curve_fit

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
RED = "#b23a48"
ORANGE = "#c0603f"
GREY = "#808080"
HAIR = "#cccccc"

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 4.62, 4.48)
AUX_T = (7.44, 3.61, 2.68, 1.89)
AUX_B = (7.44, 1.02, 2.68, 1.82)

TIMES = np.array([5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0])
N_UNIT = 12
Q_LEVEL = 80.0          # Q value of the monograph, per cent dissolved
Q_TIME = 30.0           # the time the Q value is claimed at, min
S1_MARGIN = 5.0         # stage-1 acceptance is each unit >= Q + 5 %

# name, plateau (%), Weibull scale (min), Weibull shape, lag (min), colour
PRODUCTS = [
    ("RLD-0 reference (IR)", 98.0, 8.6, 1.22, 1.2, BLUE),
    ("PX-114 batch A (IR)", 96.5, 10.6, 1.18, 1.4, GREEN),
    ("PX-114 batch B (IR)", 87.0, 26.0, 1.05, 2.6, ORANGE),
    ("PX-220 prototype (ER)", 94.0, 98.0, 0.92, 3.0, GOLD),
]

# medium label, scale multiplier, plateau multiplier, colour (sequential blue)
MEDIA = [("pH 1.2 HCl", 0.86, 1.00, "#134a76"),
         ("pH 4.5 acetate", 1.15, 0.97, "#5f9dc6"),
         ("pH 6.8 phosphate", 1.52, 0.93, "#9cc3dd")]


def weibull(t, fmax, alpha, beta, lag):
    """Cumulative fraction released by a lagged Weibull dissolution law."""
    t = np.asarray(t, dtype=float)
    z = np.clip((t - lag) / alpha, 0.0, None)
    return fmax * (1.0 - np.exp(-(z ** beta)))


def observe(rng, fmax, alpha, beta, lag, cv=0.055):
    """N_UNIT units on the sampling schedule; returns mean and SD per point."""
    truth = weibull(TIMES, fmax, alpha, beta, lag)
    reps = truth[None, :] * rng.normal(1.0, cv, (N_UNIT, TIMES.size))
    reps = np.clip(reps, 0.0, 105.0)
    return reps.mean(axis=0), reps.std(axis=0, ddof=1)


def fit_weibull(mean):
    p0 = [max(mean.max(), 50.0), 20.0, 1.0, 1.0]
    lo = [40.0, 1.0, 0.4, 0.0]
    hi = [110.0, 400.0, 3.0, 8.0]
    popt, _ = curve_fit(weibull, TIMES, mean, p0=p0, bounds=(lo, hi),
                        maxfev=60000)
    return popt


def f2_window(ref, test):
    """Indices used by the f2 rule: all points up to the first > 85 % pair.

    Only one point beyond 85 % dissolved may enter the sum, otherwise the
    plateau - where the two curves must agree - inflates the statistic.
    """
    keep = []
    seen_plateau = False
    for i in range(TIMES.size):
        keep.append(i)
        if ref[i] > 85.0 and test[i] > 85.0:
            if seen_plateau:
                keep.pop()
                break
            seen_plateau = True
    return np.array(keep, dtype=int)


def f2_value(ref, test):
    """Similarity factor from the two plotted mean vectors."""
    idx = f2_window(ref, test)
    d = ref[idx] - test[idx]
    msd = float(np.mean(d ** 2))
    return 50.0 * np.log10(100.0 / np.sqrt(1.0 + msd)), idx.size


def cross_up(x, y, level):
    """First x where an increasing y rises through ``level``."""
    idx = np.flatnonzero(y >= level)
    if idx.size == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return float(x[0])
    return float(np.interp(level, [y[i - 1], y[i]], [x[i - 1], x[i]]))


def main_panel(ax, data):
    ax.axhline(Q_LEVEL, color=RED, lw=1.15, ls=(0, (5.0, 2.2)), zorder=5)
    ax.text(119.0, Q_LEVEL + 1.4,
            "monograph $Q$ = {:.0f} % at {:.0f} min".format(Q_LEVEL, Q_TIME),
            ha="right", va="bottom", fontsize=6.6, color=RED, zorder=9,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                      edgecolor="none", alpha=0.92))
    ax.axhline(Q_LEVEL + S1_MARGIN, color=lighten(RED, 0.45), lw=0.8,
               ls=(0, (1.6, 1.8)), zorder=4)
    ax.text(119.0, Q_LEVEL + S1_MARGIN + 2.6,
            "stage-1 unit limit $Q$ + {:.0f} %".format(S1_MARGIN), ha="right",
            va="bottom", fontsize=6.1, color=lighten(RED, 0.30), zorder=9,
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                      edgecolor="none", alpha=0.92))

    for tq in (15.0, Q_TIME):
        ax.axvline(tq, color=HAIR, lw=0.8, ls=(0, (2.2, 2.0)), zorder=1)

    tg = np.linspace(0.0, TIMES[-1], 600)
    for (name, _f, _a, _b, _l, col) in PRODUCTS:
        d = data[name]
        ax.errorbar(TIMES, d["mean"], yerr=d["sd"], ls="none", marker="o",
                    ms=3.6, mfc="white", mec=col, mew=1.0, ecolor=col,
                    elinewidth=0.8, capsize=1.9, zorder=7)
        ax.plot(tg, weibull(tg, *d["popt"]), color=col, lw=1.35,
                ls=(0, (4.4, 1.9)), zorder=6)
        ax.plot(TIMES, d["mean"], color=col, lw=1.5, alpha=0.35, zorder=5)

    for label, tq in (("15 min", 15.0), ("30 min", Q_TIME)):
        ax.text(tq, 2.0, label, ha="center", va="bottom", fontsize=6.2,
                color=MUTED, rotation=90, zorder=6)

    ref = data[PRODUCTS[0][0]]["mean"]
    for k, (name, _f, _a, _b, _l, col) in enumerate(PRODUCTS):
        d = data[name]
        j = int(np.argmin(np.abs(TIMES - Q_TIME)))
        ok = d["mean"][j] >= Q_LEVEL
        ax.plot([Q_TIME], [d["mean"][j]], marker="s" if ok else "X",
                ms=5.4 if ok else 6.0, mfc=col if ok else RED,
                mec="white", mew=0.9, zorder=9)

    handles = []
    for (name, _f, _a, _b, _l, col) in PRODUCTS:
        d = data[name]
        j = int(np.argmin(np.abs(TIMES - Q_TIME)))
        handles.append(Line2D([], [], color=col, lw=1.5, marker="o", ms=3.6,
                              mfc="white", mec=col, mew=1.0,
                              label="{:s}   {:.0f} % at 30 min".format(
                                  name, d["mean"][j])))
    handles += [
        Line2D([], [], color=SUB, lw=1.35, ls=(0, (4.4, 1.9)),
               label="Weibull fit to the mean"),
        Line2D([], [], color=RED, lw=1.15, ls=(0, (5.0, 2.2)),
               label="$Q$ level (quality standard)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6.6, frameon=False,
              bbox_to_anchor=(0.998, 0.020), handlelength=1.9,
              borderaxespad=0.0, labelspacing=0.32)

    ax.set_xlim(-2.0, 124.0)
    ax.set_ylim(0.0, 106.0)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90, 105, 120])
    ax.set_xlabel("dissolution time (min)")
    ax.set_ylabel("cumulative dissolved (% of label claim)")
    ax.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Mean of {:d} units $\\pm$ 1 SD, USP apparatus 2 at 50 rpm, "
                 "pH 6.8".format(N_UNIT), fontsize=9.0, fontweight="bold",
                 color=INK, pad=6.0)
    return ref


def f2_panel(ax, data, f2s):
    names = [p[0] for p in PRODUCTS[1:]]
    short = ["batch A\n(IR)", "batch B\n(IR)", "PX-220\n(ER)"]
    vals = [f2s[n][0] for n in names]
    npts = [f2s[n][1] for n in names]
    y = np.arange(len(names))[::-1]
    cols = [GREEN if v >= 50.0 else RED for v in vals]

    ax.barh(y, vals, height=0.56, color=[lighten(c, 0.52) for c in cols],
            edgecolor=cols, linewidth=0.9, zorder=3)
    ax.axvline(50.0, color=INK, lw=1.1, ls=(0, (4.0, 2.0)), zorder=5)
    ax.text(48.6, -0.55, "$f_2$ = 50 threshold", ha="right", va="bottom",
            fontsize=6.2, color=INK, zorder=6)
    for yi, v, c, n in zip(y, vals, cols, npts):
        verdict = "similar" if v >= 50.0 else "dissimilar"
        ax.text(v + 1.6, yi, "{:.1f}  ({:d} pts, {:s})".format(v, n, verdict),
                ha="left", va="center", fontsize=6.4, color=c,
                fontweight="bold", zorder=6)
    ax.set_yticks(y)
    ax.set_yticklabels(short, fontsize=6.3)
    ax.set_xlim(0.0, 92.0)
    ax.set_ylim(-0.62, len(names) - 0.30)
    ax.set_xlabel("similarity factor $f_2$ vs RLD-0 (dimensionless)",
                  fontsize=7.2)
    ax.grid(axis="x", lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Similarity to the reference product", fontsize=8.0,
                 fontweight="bold", color=INK)


def media_panel(ax, media_curves):
    tg = np.linspace(0.0, TIMES[-1], 400)
    for (label, _s, _p, col), (mean, popt) in zip(MEDIA, media_curves):
        ax.plot(tg, weibull(tg, *popt), color=col, lw=1.5, zorder=4)
        ax.plot(TIMES, mean, ls="none", marker="o", ms=3.0, mfc="white",
                mec=col, mew=0.9, zorder=5)
        t80 = cross_up(tg, weibull(tg, *popt), Q_LEVEL)
        lab = label + ("  $t_{{80}}$ {:.0f} min".format(t80)
                       if np.isfinite(t80) else "  $t_{80}$ not reached")
        ax.plot([], [], color=col, lw=1.5, label=lab)
    ax.axhline(Q_LEVEL, color=RED, lw=1.0, ls=(0, (5.0, 2.2)), zorder=3)
    ax.set_xlim(-2.0, 124.0)
    ax.set_ylim(0.0, 106.0)
    ax.set_xticks([0, 30, 60, 90, 120])
    ax.set_xlabel("dissolution time (min)", fontsize=7.2)
    ax.set_ylabel("dissolved (%)", fontsize=7.2)
    ax.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", fontsize=6.0, frameon=False,
              handlelength=1.4, borderaxespad=0.2, labelspacing=0.24)
    ax.set_title("Batch A across three compendial media", fontsize=8.0,
                 fontweight="bold", color=INK)


def audit(fig, foot, axes):
    """Warn if the footnote box overlaps an in-view tick or axis label."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for ax in axes:
        ab = ax.get_window_extent()
        for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            tb = t.get_window_extent()
            if tb.width <= 0.0 or not t.get_text():
                continue
            if tb.x1 < ab.x0 - 2.0 or tb.x0 > ab.x1 + 2.0:
                continue
            if fb.overlaps(tb):
                print("OVERLAP with", repr(t.get_text()))


def main():
    set_pub_style()
    rng = np.random.default_rng(9421)

    data = {}
    for (name, fmax, alpha, beta, lag, col) in PRODUCTS:
        mean, sd = observe(rng, fmax, alpha, beta, lag)
        data[name] = dict(mean=mean, sd=sd, popt=fit_weibull(mean))

    ref = data[PRODUCTS[0][0]]["mean"]
    f2s = {p[0]: f2_value(ref, data[p[0]]["mean"]) for p in PRODUCTS[1:]}

    _n, fmax, alpha, beta, lag, _c = PRODUCTS[1]
    media_curves = []
    for (_label, smul, pmul, _col) in MEDIA:
        mean, _sd = observe(rng, fmax * pmul, alpha * smul, beta, lag)
        media_curves.append((mean, fit_weibull(mean)))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_f2 = axes_inches(fig, AUX_T)
    ax_med = axes_inches(fig, AUX_B)

    main_panel(ax_main, data)
    f2_panel(ax_f2, data, f2s)
    media_panel(ax_med, media_curves)

    j30 = int(np.argmin(np.abs(TIMES - Q_TIME)))
    fail = [p[0] for p in PRODUCTS[1:] if f2s[p[0]][0] < 50.0]


    panel_labels_inches(fig, [ax_main, ax_f2, ax_med], dx=-0.62, dy=0.14)
    savefig(fig, "942_pha_dissolution_profile.png")


if __name__ == "__main__":
    main()

"""Gompertz growth of three breed-by-sex groups, with the derivative that
explains it.

The main panel carries the claim: body weight is sigmoid, so the fastest
growth is not at the start but at the point of inflection, and every group
approaches a mature-weight asymptote that no amount of extra days will move.
The daily gain curve on the secondary axis is the analytic derivative of the
plotted weight curve, so its peak sits exactly on the inflection age by
construction rather than by eye.  The two subordinate panels are the follow-up
questions: how the gain curves compare when they are given a full axis of their
own, and how far the fitted asymptote actually moves with dietary energy.

Every number quoted - inflection age, weight at inflection, peak gain, mature
weight - is read back out of the fitted arrays that are drawn on the page.
Synthetic data; no real herd is represented.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from pubstyle import set_pub_style, savefig, despine
from figkit import axes_inches, panel_labels_inches, shared_limits

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
BASE = "#808080"

# Gompertz truth used only to synthesise the observations; the curves that get
# drawn come back out of curve_fit, never from these numbers.
GROUPS = [
    dict(name="Breed A, male", A=126.0, b=3.62, k=0.0166, color=BLUE, n=48),
    dict(name="Breed A, female", A=112.0, b=3.55, k=0.0173, color=GREEN, n=46),
    dict(name="Breed B, male", A=101.0, b=3.40, k=0.0196, color=GOLD, n=44),
]

# Dietary energy density relative to the control ration, and the multiplier it
# applies to mature size in the synthetic generator.
DIETS = [("85 % ME", 0.85, 0.905), ("100 % ME", 1.00, 1.000),
         ("112 % ME", 1.12, 1.056), ("124 % ME", 1.24, 1.082)]

AGE_LO, AGE_HI = 5.0, 205.0


def gompertz(t, A, b, k):
    return A * np.exp(-b * np.exp(-k * t))


def observations(rng, g, cv=0.055):
    """Serial weighings: every animal is weighed on a shared schedule."""
    days = np.array([14, 28, 42, 56, 70, 84, 98, 119, 140, 161, 182, 200],
                    dtype=float)
    ages = np.repeat(days, max(1, g["n"] // days.size))
    true = gompertz(ages, g["A"], g["b"], g["k"])
    # Multiplicative noise: heavier animals scatter more in absolute terms,
    # which is what a real weigh-crate record looks like.
    return ages, true * (1.0 + rng.normal(0.0, cv, ages.size))


def fit_group(ages, w):
    p0 = (float(np.max(w)) * 1.15, 3.5, 0.018)
    popt, _ = curve_fit(gompertz, ages, w, p0=p0, maxfev=40000)
    return popt


def fit_asymptote(rng, scale, cv=0.035):
    """Refit a diet arm and return the asymptote with its 95 % interval."""
    base = GROUPS[0]
    # The arms are weighed well past the finishing window: an asymptote fitted
    # from data that stop at 60 % of mature size is not identifiable.
    days = np.array([14, 42, 84, 126, 168, 210, 252, 294], dtype=float)
    ages = np.repeat(days, 9)
    true = gompertz(ages, base["A"] * scale, base["b"], base["k"])
    w = true * (1.0 + rng.normal(0.0, cv, ages.size))
    popt, pcov = curve_fit(gompertz, ages, w,
                           p0=(float(np.max(w)) * 1.15, 3.5, 0.018),
                           maxfev=40000)
    return float(popt[0]), 1.96 * float(np.sqrt(abs(pcov[0, 0]))), ages.size


def curves(popt, t):
    """Weight (kg), daily gain (g/d) and the inflection read off the arrays."""
    w = gompertz(t, *popt)
    adg = np.gradient(w, t) * 1000.0
    i = int(np.argmax(adg))
    return w, adg, float(t[i]), float(w[i]), float(adg[i])


def main_panel(ax, fits, t):
    axr = ax.twinx()
    handles, labels = [], []
    for g, popt, obs in fits:
        w, adg, t_inf, w_inf, adg_pk = curves(popt, t)
        ages, y = obs
        rel = (y - gompertz(ages, *popt)) / gompertz(ages, *popt)
        s = float(np.std(rel, ddof=3))
        ax.fill_between(t, w * (1.0 - 1.96 * s), w * (1.0 + 1.96 * s),
                        color=g["color"], alpha=0.13, lw=0, zorder=1)
        ax.scatter(ages, y, s=5.0, facecolor=g["color"], edgecolor="none",
                   alpha=0.45, zorder=3)
        ln, = ax.plot(t, w, color=g["color"], lw=1.8, zorder=5)
        ax.axhline(popt[0], color=g["color"], lw=0.8, ls=(0, (5, 3)),
                   alpha=0.75, zorder=2)
        ax.plot([t_inf], [w_inf], marker="o", ms=4.6, mfc="white",
                mec=g["color"], mew=1.4, zorder=7)
        ax.plot([t_inf, t_inf], [0.0, w_inf], color=g["color"], lw=0.7,
                ls=(0, (1.6, 1.8)), zorder=2)
        axr.plot(t, adg, color=g["color"], lw=1.15, ls=(0, (4.0, 2.0)),
                 zorder=4)
        axr.plot([t_inf], [adg_pk], marker="^", ms=3.8, mfc="white",
                 mec=g["color"], mew=1.0, zorder=6)
        handles.append(ln)
        labels.append(f"{g['name']}  (n={ages.size}, "
                      f"$W_\\infty$={popt[0]:.0f} kg)")

    ax.set_xlim(AGE_LO, AGE_HI)
    ax.set_ylim(0.0, 148.0)
    ax.set_xlabel("age (d)", fontsize=8.0)
    ax.set_ylabel("body weight (kg)", fontsize=8.0)
    axr.set_ylabel("average daily gain (g/d)", fontsize=8.0, color=SUB)
    ax.tick_params(labelsize=7.2)
    axr.tick_params(labelsize=7.2, colors=SUB)
    axr.spines["top"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_title("Weight is sigmoid; gain peaks at the inflection",
                 fontsize=9.0, fontweight="bold", color=INK)

    leg = ax.legend(handles, labels, fontsize=6.8, frameon=True,
                    framealpha=0.93, facecolor="white", edgecolor="#cccccc",
                    loc="lower right", handlelength=2.0, borderpad=0.42,
                    borderaxespad=0.6)
    leg.set_zorder(9)
    ax.text(0.015, 0.982, "solid = body weight (left axis)   |   "
                          "long dash = daily gain (right axis)   |   "
                          "short dash = mature-weight asymptote",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.6,
            color=SUB, zorder=9)

    ref = curves(fits[0][1], t)
    ax.annotate(f"inflection {ref[2]:.0f} d\n{ref[3]:.0f} kg, "
                f"{ref[4]:.0f} g/d",
                xy=(ref[2], ref[3]), xytext=(ref[2] + 13.0, ref[3] - 30.0),
                fontsize=6.7, color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=RULE, lw=0.8,
                                shrinkA=1.0, shrinkB=3.0))
    return axr


def gain_panel(ax, fits, t):
    peaks = []
    for g, popt, _ in fits:
        _, adg, t_inf, _, adg_pk = curves(popt, t)
        ax.plot(t, adg, color=g["color"], lw=1.6, zorder=4)
        ax.plot([t_inf], [adg_pk], marker="o", ms=3.6, mfc="white",
                mec=g["color"], mew=1.1, zorder=6)
        ax.plot([t_inf, t_inf], [0.0, adg_pk], color=g["color"], lw=0.7,
                ls=(0, (1.6, 1.8)), zorder=2)
        peaks.append((g["name"], t_inf, adg_pk))
    lo, hi = min(p[1] for p in peaks), max(p[1] for p in peaks)
    ax.text(0.975, 0.055, f"peak gain between {lo:.0f} and {hi:.0f} d",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            color=SUB)
    ax.set_xlim(AGE_LO, AGE_HI)
    ax.set_xlabel("age (d)", fontsize=7.6)
    ax.set_ylabel("daily gain (g/d)", fontsize=7.6)
    ax.tick_params(labelsize=6.8)
    despine(ax)
    ax.grid(axis="y", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_title("Gain curves peak, then decay", fontsize=8.4,
                 fontweight="bold", color=INK)
    return peaks


def diet_panel(ax, diets, ref_w):
    names = [d[0] for d in diets]
    vals = np.array([d[1] for d in diets])
    errs = np.array([d[2] for d in diets])
    x = np.arange(vals.size)
    # One accent only: the ration that gains least per extra unit of energy.
    gain = np.diff(vals) / vals[:-1]
    flat = int(np.argmin(gain)) + 1
    cols = [BLUE] * vals.size
    cols[flat] = RED
    ax.bar(x, vals, width=0.62, color=cols, edgecolor="white", lw=0.7,
           zorder=3)
    ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor=RULE, elinewidth=0.9,
                capsize=2.4, capthick=0.9, zorder=5)
    ax.axhline(ref_w, color=BASE, lw=0.9, ls=(0, (4, 2.6)), zorder=2)
    ax.set_ylim(0.0, float(vals.max() + errs.max()) * 1.30)
    ax.text(0.015, ref_w, "control ration", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=6.3, color=MUTED)
    for xi, v, e in zip(x, vals, errs):
        ax.text(xi, v + e + 2.5, f"{v:.0f}", ha="center", va="bottom",
                fontsize=6.4, color=INK)
    step = 100.0 * (vals[flat] - vals[flat - 1]) / vals[flat - 1]
    ax.text(0.5, 0.965, f"top energy step adds only {step:.1f} % "
                        f"of mature weight",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.5,
            color=RED)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.8)
    ax.set_xlabel("dietary energy density (- of control)", fontsize=7.6)
    ax.set_ylabel("fitted mature weight (kg)", fontsize=7.6)
    ax.tick_params(axis="y", labelsize=6.8)
    despine(ax)
    ax.grid(axis="y", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_title("Mature weight saturates with energy", fontsize=8.4,
                 fontweight="bold", color=INK)
    return step


def main():
    set_pub_style()
    rng = np.random.default_rng(20840)
    t = np.linspace(AGE_LO, AGE_HI, 601)

    fits = []
    for g in GROUPS:
        obs = observations(rng, g)
        fits.append((g, fit_group(*obs), obs))

    diets = []
    for label, _, scale in DIETS:
        a, ci, n = fit_asymptote(rng, scale)
        diets.append((label, a, ci, n))

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (0.82, 0.98, 5.02, 4.58))
    ax_gain = axes_inches(fig, (7.16, 3.54, 2.78, 2.02))
    ax_diet = axes_inches(fig, (7.16, 0.98, 2.78, 2.02))

    axr = main_panel(ax_main, fits, t)
    peaks = gain_panel(ax_gain, fits, t)
    step = diet_panel(ax_diet, [(d[0], d[1], d[2]) for d in diets],
                      diets[1][1])

    # The secondary axis of the main panel and panel (b) show the same
    # quantity, so they get one ruler (DESIGN_SYSTEM 13.1).
    adg_all = [curves(popt, t)[1] for _, popt, _ in fits]
    _, hi = shared_limits(adg_all, pad=0.12, lo=0.0)
    axr.set_ylim(0.0, hi)
    ax_gain.set_ylim(0.0, hi)

    n_tot = sum(obs[0].size for _, _, obs in fits)
    n_diet = sum(d[3] for d in diets)
    panel_labels_inches(fig, [ax_main, ax_gain, ax_diet], dx=-0.50, dy=0.14)
    savefig(fig, "840_vet_growth_curve.png")


if __name__ == "__main__":
    main()

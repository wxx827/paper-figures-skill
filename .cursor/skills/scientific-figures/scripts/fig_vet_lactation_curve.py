"""Wood lactation curves by parity, and the composition trade-off underneath.

The main panel shows the shape every dairy record has: yield climbs to a peak
in the second month of lactation and then declines slowly, with older cows
peaking higher and earlier but holding that peak less well.  Fat and protein
percentages ride the secondary axis in the opposite direction - dilution means
the richest milk is produced when there is least of it, which is a genuine
physiological fact and not an artefact of the plot.  The subordinate panels
carry the two summaries the main curve cannot show at a glance: what the whole
305-day lactation adds up to, and how somatic cell count drifts upward once
the peak is past.

Peak day, peak yield, persistency and the 305-day totals are all read back out
of the plotted arrays (arg-max, Wood persistency from the fitted exponents,
trapezoidal integration).  Synthetic data; no real herd is represented.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, despine
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
OCHRE = "#c0603f"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
BASE = "#808080"

DIM_LO, DIM_HI = 5.0, 305.0
SCC_THRESHOLD = 2.0e5          # cells/mL, usual subclinical-mastitis cut-off

# Wood's incomplete-gamma model Y = a t^b exp(-c t).  ``peak`` fixes b/c, so
# the curve peaks where the parity is meant to peak; c is derived, not typed.
PARITIES = [
    dict(name="parity 1", a=13.5, b=0.220, peak=55.0, color=BLUE,
         n=112, scc0=4.62),
    dict(name="parity 2", a=15.5, b=0.245, peak=48.0, color=GREEN,
         n=86, scc0=4.80),
    dict(name="parity 3+", a=17.0, b=0.265, peak=42.0, color=GOLD,
         n=74, scc0=5.00),
]


def wood(t, a, b, c):
    return a * np.power(t, b) * np.exp(-c * t)


def parity_curve(p, t):
    c = p["b"] / p["peak"]
    y = wood(t, p["a"], p["b"], c)
    i = int(np.argmax(y))
    # Wood's persistency index; larger means a flatter post-peak decline.
    persist = -(p["b"] + 1.0) * np.log(c)
    yield_305 = float(np.trapezoid(y, t))
    return y, float(t[i]), float(y[i]), float(persist), yield_305


def test_day_records(rng, p, t, y, k=9, cv=0.085):
    """Monthly test-day means: what a milk-recording scheme actually returns."""
    days = np.linspace(20.0, 290.0, k)
    mu = np.interp(days, t, y)
    return days, mu * (1.0 + rng.normal(0.0, cv, days.size))


def composition(t, y_ref):
    """Fat and protein per cent, driven by dilution against the yield curve."""
    frac = y_ref / y_ref.max()
    fat = 3.42 + 1.34 * (1.0 - frac)
    prot = 2.94 + 0.86 * (1.0 - frac)
    return fat, prot


def scc_curve(p, t):
    """Somatic cell count: colostral spike, a nadir, then a steady climb."""
    log10 = (p["scc0"] + 0.90 * ((t - 70.0) / 235.0) ** 2
             + 0.80 * np.exp(-t / 10.0))
    return np.power(10.0, log10)


def main_panel(ax, rng, series, t):
    axr = ax.twinx()
    handles, labels = [], []
    for p, (y, t_pk, y_pk, persist, y305) in series:
        days, obs = test_day_records(rng, p, t, y)
        ax.scatter(days, obs, s=11.0, facecolor="white", edgecolor=p["color"],
                   lw=0.9, zorder=6)
        ln, = ax.plot(t, y, color=p["color"], lw=1.9, zorder=5)
        ax.plot([t_pk], [y_pk], marker="o", ms=5.0, mfc=p["color"],
                mec="white", mew=1.0, zorder=7)
        ax.plot([t_pk, t_pk], [0.0, y_pk], color=p["color"], lw=0.7,
                ls=(0, (1.6, 1.8)), zorder=2)
        handles.append(ln)
        labels.append(f"{p['name']}  (n={p['n']} cows): peak "
                      f"{y_pk:.1f} kg/d at {t_pk:.0f} DIM, "
                      f"persistency {persist:.2f}")

    ref = series[-1][1][0]
    fat, prot = composition(t, ref)
    lf, = axr.plot(t, fat, color=RED, lw=1.4, ls=(0, (4.5, 2.2)), zorder=4)
    lp, = axr.plot(t, prot, color=OCHRE, lw=1.4, ls=(0, (1.2, 1.6)), zorder=4)
    axr.set_ylim(2.4, 5.4)
    axr.set_ylabel("milk fat and protein (%)", fontsize=8.0, color=SUB)
    axr.tick_params(labelsize=7.2, colors=SUB)
    axr.spines["top"].set_visible(False)

    i_pk = int(np.argmax(ref))
    axr.plot([t[i_pk]], [fat[i_pk]], marker="v", ms=4.2, mfc="white",
             mec=RED, mew=1.1, zorder=6)
    axr.annotate(f"fat minimum {fat[i_pk]:.2f} % coincides with peak yield",
                 xy=(t[i_pk], fat[i_pk]), xytext=(118.0, 5.22), fontsize=6.7,
                 color=RED, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=RED, lw=0.8,
                                 shrinkA=1.0, shrinkB=3.0))

    ax.set_xlim(DIM_LO, DIM_HI)
    ax.set_ylim(0.0, 44.0)
    ax.set_xlabel("days in milk, DIM (d)", fontsize=8.0)
    ax.set_ylabel("milk yield (kg/d)", fontsize=8.0)
    ax.tick_params(labelsize=7.2)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_title("Older cows peak higher and earlier, but hold it less well",
                 fontsize=9.0, fontweight="bold", color=INK)

    handles += [lf, lp,
                plt.Line2D([], [], marker="o", ls="none", ms=3.4,
                           mfc="white", mec=RULE, mew=0.9)]
    labels += ["milk fat % (right axis)", "milk protein % (right axis)",
               "monthly test-day means"]
    leg = ax.legend(handles, labels, fontsize=6.8, frameon=True,
                    framealpha=0.93, facecolor="white", edgecolor="#cccccc",
                    loc="lower left", handlelength=2.0, borderpad=0.42,
                    borderaxespad=0.5)
    leg.set_zorder(9)
    return axr


def total_panel(ax, series):
    names = [p["name"] for p, _ in series]
    tot = np.array([s[4] for _, s in series])
    cols = [p["color"] for p, _ in series]
    x = np.arange(tot.size)
    ax.bar(x, tot, width=0.58, color=cols, edgecolor="white", lw=0.7,
           zorder=3)
    for xi, v in zip(x, tot):
        ax.text(xi, v + 0.012 * tot.max(), f"{v:,.0f}", ha="center",
                va="bottom", fontsize=6.6, color=INK)
    ax.set_ylim(0.0, float(tot.max()) * 1.24)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.9)
    ax.set_ylabel("305-d yield (kg)", fontsize=7.6)
    ax.tick_params(axis="y", labelsize=6.8)
    despine(ax)
    ax.grid(axis="y", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    lift = 100.0 * (tot[-1] - tot[0]) / tot[0]
    ax.text(0.5, 0.965, f"parity 3+ integrates {lift:.0f} % more milk "
                        f"than parity 1",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.5,
            color=SUB)
    ax.set_title("305-d totals rise with parity", fontsize=8.4,
                 fontweight="bold", color=INK)
    return tot, lift


def scc_panel(ax, series, t):
    nadirs = []
    for p, _ in series:
        s = scc_curve(p, t)
        ax.plot(t, s, color=p["color"], lw=1.6, zorder=4)
        i = int(np.argmin(s))
        ax.plot([t[i]], [s[i]], marker="o", ms=3.4, mfc="white",
                mec=p["color"], mew=1.0, zorder=6)
        nadirs.append((float(t[i]), float(s[i]), float(s[-1])))
    ax.axhline(SCC_THRESHOLD, color=RED, lw=1.0, ls=(0, (4, 2.4)), zorder=3)
    ax.text(0.02, SCC_THRESHOLD * 1.14, "subclinical threshold "
                                        "$2 \\times 10^{5}$ cells/mL",
            transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=6.3, color=RED)
    ax.set_yscale("log")
    ax.set_xlim(DIM_LO, DIM_HI)
    ax.set_ylim(2.0e4, 3.0e6)
    ax.set_xlabel("days in milk, DIM (d)", fontsize=7.6)
    ax.set_ylabel("somatic cell count (cells/mL)", fontsize=7.6)
    ax.tick_params(labelsize=6.8)
    despine(ax)
    ax.grid(axis="y", lw=0.45, alpha=0.55, which="major")
    ax.set_axisbelow(True)
    lo = min(n[0] for n in nadirs)
    ax.text(0.5, 0.955, f"nadir near {lo:.0f} DIM, then a steady climb",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.5,
            color=SUB)
    ax.set_title("Cell count bottoms out near peak yield", fontsize=8.4,
                 fontweight="bold", color=INK)
    return nadirs


def main():
    set_pub_style()
    rng = np.random.default_rng(20841)
    t = np.linspace(DIM_LO, DIM_HI, 601)
    series = [(p, parity_curve(p, t)) for p in PARITIES]

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (0.82, 0.98, 5.02, 4.58))
    ax_tot = axes_inches(fig, (7.16, 3.54, 2.78, 2.02))
    ax_scc = axes_inches(fig, (7.16, 0.98, 2.78, 2.02))

    main_panel(ax_main, rng, series, t)
    tot, lift = total_panel(ax_tot, series)
    nadirs = scc_panel(ax_scc, series, t)

    n_cows = sum(p["n"] for p in PARITIES)
    pk = [s[1] for _, s in series]
    panel_labels_inches(fig, [ax_main, ax_tot, ax_scc], dx=-0.50, dy=0.14)
    savefig(fig, "841_vet_lactation_curve.png")


if __name__ == "__main__":
    main()

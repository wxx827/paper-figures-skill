"""Dynamic FDG uptake: how lesion type separates only after the blood clears.

Five time-activity curves are generated from explicit kinetic models on one
time grid, so the malignant, inflammatory and background behaviours can be
compared on a single ordinate.  The bi-exponential overlay on the blood pool
is a real ``curve_fit`` to the plotted samples and the two half-times quoted
next to it come out of that fit; the response rate, the correlation
coefficient and the regression line in the small panels are likewise computed
from the points that are drawn.

Synthetic illustration for a visual reference library - not patient data.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

from pubstyle import set_pub_style, savefig, despine
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
HAIR = "#cccccc"

FIGW, FIGH = 10.4, 6.4
T_END = 120.0                 # end of the dynamic acquisition (min)
WINDOW = (55.0, 75.0)         # routine static acquisition window (min)
RESPONSE_DROP = 30.0          # PERCIST-style response threshold (%)

TUMOUR_A = "#b23a48"
TUMOUR_B = "#8e6bbf"
INFLAM = "#c9a227"
LIVER = "#2e7d5b"
BLOOD = "#3b5b92"


def biexp(t, a1, k1, a2, k2):
    return a1 * np.exp(-k1 * t) + a2 * np.exp(-k2 * t)


def curves(t, rng):
    """Five synthetic time-activity curves with correlated measurement noise."""
    raw = {
        "Lesion 1, malignant": 3.20 * (1.0 - np.exp(-t / 22.0)) + 0.0265 * t,
        "Lesion 2, malignant": 1.95 * (1.0 - np.exp(-t / 25.0)) + 0.0125 * t,
        "Inflammatory node": 6.0 * (np.exp(-t / 70.0) - np.exp(-t / 9.0)),
        "Liver background": 1.90 + 0.0026 * t,
        "Blood pool": biexp(t, 4.60, 1.0 / 6.5, 2.60, 1.0 / 95.0),
    }
    out = {}
    for k, v in raw.items():
        jitter = gaussian_filter1d(rng.normal(0.0, 0.085, t.size), 3.0)
        out[k] = np.maximum(v + jitter * np.sqrt(np.maximum(v, 0.2)), 0.02)
    return out


def main():
    set_pub_style()
    rng = np.random.default_rng(874)
    t = np.linspace(0.0, T_END, 241)
    tac = curves(t, rng)
    style = {"Lesion 1, malignant": (TUMOUR_A, 1.8),
             "Lesion 2, malignant": (TUMOUR_B, 1.4),
             "Inflammatory node": (INFLAM, 1.4),
             "Liver background": (LIVER, 1.3),
             "Blood pool": (BLOOD, 1.3)}

    fig = plt.figure(figsize=(FIGW, FIGH))
    ax = axes_inches(fig, (0.84, 1.02, 5.28, 4.34))
    ax_pair = axes_inches(fig, (7.12, 3.70, 2.88, 1.74))
    ax_sc = axes_inches(fig, (7.12, 0.95, 2.88, 1.74))
    for a in (ax, ax_pair, ax_sc):
        despine(a)

    # ---- main panel: dynamic uptake ---------------------------------------
    ax.axvspan(*WINDOW, color=GREY, alpha=0.13, zorder=0)
    ax.text(0.5 * sum(WINDOW), 7.35,
            f"routine static\nwindow {WINDOW[0]:.0f}-{WINDOW[1]:.0f} min",
            ha="center", va="top", fontsize=6.4, color=SUB, linespacing=1.5)

    for name, y in tac.items():
        col, lw = style[name]
        ax.plot(t, y, color=col, lw=lw, label=name, zorder=4)

    # Bi-exponential fit to sampled blood-pool frames
    idx = np.linspace(4, t.size - 1, 15).astype(int)
    ts, ys = t[idx], tac["Blood pool"][idx]
    popt, _ = curve_fit(biexp, ts, ys, p0=[4.0, 0.15, 2.0, 0.01],
                        maxfev=40000)
    a1, k1, a2, k2 = popt
    if k1 < k2:                                    # order fast then slow
        a1, k1, a2, k2 = a2, k2, a1, k1
    t_fast, t_slow = np.log(2.0) / k1, np.log(2.0) / k2
    ax.plot(ts, ys, ls="none", marker="o", ms=3.6, mfc="white", mec=BLOOD,
            mew=0.9, zorder=6)
    ax.plot(t, biexp(t, a1, k1, a2, k2), color="#1a1a1a", lw=0.9,
            ls=(0, (3.2, 1.8)), zorder=7)
    resid = ys - biexp(ts, a1, k1, a2, k2)
    r2 = 1.0 - resid.var() / ys.var()
    ax.annotate("bi-exponential fit to the blood pool\n"
                f"fast $T_{{1/2}}$ = {t_fast:.1f} min, "
                f"slow $T_{{1/2}}$ = {t_slow:.0f} min   "
                f"($R^2$ = {r2:.3f})",
                xy=(t[70], biexp(t[70], a1, k1, a2, k2)),
                xytext=(30.0, 5.55), fontsize=6.5, color=INK, ha="left",
                va="center", linespacing=1.6,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=MUTED))

    for name, dy in (("Lesion 1, malignant", 0.22),
                     ("Inflammatory node", -0.34)):
        ax.annotate(f"{tac[name][-1]:.1f} at {T_END:.0f} min",
                    xy=(T_END, tac[name][-1]),
                    xytext=(-4.0, dy * 26.0), textcoords="offset points",
                    fontsize=6.3, color=style[name][0], ha="right",
                    va="bottom" if dy > 0 else "top")

    ax.set_xlim(0.0, T_END)
    ax.set_ylim(0.0, 7.6)
    ax.set_xticks(np.arange(0, 121, 20))
    ax.set_xlabel("time after injection (min)", fontsize=8.0)
    ax.set_ylabel("standardised uptake value SUV (-)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.grid(True, color="#ececec", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="lower left", ncol=5, frameon=False,
                    bbox_to_anchor=(-0.005, 1.005), fontsize=6.8,
                    handlelength=1.6, columnspacing=1.1,
                    handletextpad=0.45)
    for h in leg.legend_handles:
        h.set_linewidth(1.7)

    # ---- aux 1: paired SUVmax before and after treatment ------------------
    n_les = 13
    pre = rng.uniform(4.2, 13.5, n_les)
    responder = rng.random(n_les) < 0.60
    factor = np.where(responder, rng.uniform(0.28, 0.66, n_les),
                      rng.uniform(0.82, 1.24, n_les))
    post = pre * factor
    change = 100.0 * (post - pre) / pre
    is_resp = change <= -RESPONSE_DROP
    n_resp = int(is_resp.sum())

    for p0, p1, ok in zip(pre, post, is_resp):
        col = "#2e7d5b" if ok else "#b23a48"
        ax_pair.plot([0.0, 1.0], [p0, p1], color=col, lw=1.0, alpha=0.85,
                     zorder=3)
        ax_pair.plot([0.0, 1.0], [p0, p1], ls="none", marker="o", ms=3.2,
                     color=col, zorder=4)
    ax_pair.plot(np.full(n_les, 1.0),
                 pre * (1.0 - RESPONSE_DROP / 100.0), ls="none", marker="_",
                 ms=7.0, mew=1.0, color=GREY, zorder=5)
    ax_pair.text(1.06, float(np.median(pre)) * 0.7,
                 f"-{RESPONSE_DROP:.0f} %\nthreshold", fontsize=6.2,
                 color=MUTED, ha="left", va="center", linespacing=1.5)
    ax_pair.text(0.0, 15.3,
                 f"{n_resp} of {n_les} lesions respond "
                 f"(median change {np.median(change):+.0f} %)",
                 fontsize=6.4, color=SUB, ha="left", va="top")
    ax_pair.set_xlim(-0.16, 1.42)
    ax_pair.set_ylim(0.0, 15.8)
    ax_pair.set_xticks([0.0, 1.0])
    ax_pair.set_xticklabels(["before", "after"], fontsize=6.5)
    ax_pair.tick_params(axis="y", labelsize=6.5)
    ax_pair.tick_params(axis="x", length=0, pad=2.0)
    ax_pair.set_ylabel("$SUV_{max}$ (-)", fontsize=7.2)
    ax_pair.legend(handles=[
        Line2D([], [], color="#2e7d5b", lw=1.2, marker="o", ms=3.0,
               label="responder"),
        Line2D([], [], color="#b23a48", lw=1.2, marker="o", ms=3.0,
               label="non-responder")],
        loc="lower right", frameon=False, fontsize=6.2, handlelength=1.4,
        handletextpad=0.45, labelspacing=0.3, borderpad=0.1)
    ax_pair.set_title("Paired response of 13 lesions", fontsize=8.5,
                      fontweight="bold", color=INK, pad=4.0)

    # ---- aux 2: lesion volume against peak uptake -------------------------
    n_sc = 30
    vol = 10.0 ** rng.uniform(-0.15, 2.05, n_sc)          # mL
    suv = 2.4 + 2.05 * np.log10(vol) + rng.normal(0.0, 0.85, n_sc)
    suv = np.clip(suv, 0.8, None)
    lv = np.log10(vol)
    slope, icept = np.polyfit(lv, suv, 1)
    r = float(np.corrcoef(lv, suv)[0, 1])
    xf = np.linspace(lv.min(), lv.max(), 60)
    ax_sc.scatter(10.0 ** lv, suv, s=15.0, color=TUMOUR_A, alpha=0.75,
                  edgecolor="white", linewidth=0.4, zorder=4)
    ax_sc.plot(10.0 ** xf, slope * xf + icept, color="#3a3a3a", lw=1.1,
               ls=(0, (3.0, 1.8)), zorder=5)
    ax_sc.set_xscale("log")
    ax_sc.text(0.035, 0.955,
               f"$SUV_{{max}}$ = {slope:.2f} log$_{{10}}$V + {icept:.2f}\n"
               f"r = {r:.2f},  n = {n_sc}",
               transform=ax_sc.transAxes, ha="left", va="top", fontsize=6.3,
               color=SUB, linespacing=1.7)
    ax_sc.set_xlabel("lesion volume (mL)", fontsize=7.2)
    ax_sc.set_ylabel("$SUV_{max}$ (-)", fontsize=7.2)
    ax_sc.tick_params(labelsize=6.5)
    ax_sc.set_title("Uptake rises with lesion volume", fontsize=8.5,
                    fontweight="bold", color=INK, pad=4.0)

    w0 = int(np.argmin(np.abs(t - WINDOW[0])))
    w1 = int(np.argmin(np.abs(t - WINDOW[1])))
    ratio_win = (tac["Lesion 1, malignant"][w0:w1].mean() /
                 tac["Liver background"][w0:w1].mean())



    panel_labels_inches(fig, [ax], labels=["(a)"], dx=-0.62, dy=0.36)
    panel_labels_inches(fig, [ax_pair, ax_sc], labels=["(b)", "(c)"],
                        dx=-0.62, dy=0.22)
    savefig(fig, "874_nmd_pet_suv_timeline.png")


if __name__ == "__main__":
    main()

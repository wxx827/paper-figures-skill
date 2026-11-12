"""DSC heating traces of a semi-crystalline polymer after four cooling rates.

Every reported quantity is measured on the plotted trace: the glass transition
by the classical two-tangent midpoint construction, the cold crystallisation
and melting enthalpies by integrating the trace against a local linear baseline
and dividing by the heating rate, the melting peak by the extremum inside its
window, and the crystallinity as the net enthalpy over the enthalpy of the
fully crystalline polymer.  The exothermic direction is marked with an arrow,
without which a DSC plot cannot be read at all.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
RED = "#b23a48"

RATES = [1.0, 5.0, 20.0, 80.0]        # cooling rate, degC/min
BETA = 10.0                            # heating rate, degC/min
DH100 = 140.0                          # J/g for the fully crystalline polymer
OFFSET = 0.78                          # W/g between stacked traces
TEMP = np.linspace(30.0, 290.0, 2400)
CC_WIN = (104.0, 176.0)
MELT_WIN = (214.0, 272.0)
DEG = chr(176)


def trace(rate, rng):
    lg = np.log10(rate)
    tg, w = 76.0 + 3.0 * lg, 3.6
    tcc, scc = 132.0 - 4.0 * lg, 8.5
    tm, s1, s2 = 250.0 - 1.2 * lg, 7.0, 4.6
    dh_cc, dh_m = 6.0 + 12.6 * lg, 48.0 + 0.5 * lg

    base = 0.020 + 0.00055 * (TEMP - TEMP[0])
    step = 0.062 * 0.5 * (1.0 + np.tanh((TEMP - tg) / w))
    cc = (dh_cc / 6.0) / (scc * np.sqrt(2.0 * np.pi)) * \
        np.exp(-0.5 * ((TEMP - tcc) / scc) ** 2)
    s = np.where(TEMP < tm, s1, s2)
    melt = np.exp(-0.5 * ((TEMP - tm) / s) ** 2)
    melt *= (dh_m / 6.0) / (0.5 * np.sqrt(2.0 * np.pi) * (s1 + s2))
    noise = 0.0035 * np.convolve(rng.standard_normal(TEMP.size),
                                 np.ones(25) / 25.0, mode="same")
    return base - step + cc - melt + noise


def peak_enthalpy(hf, win, sign):
    """Enthalpy in J/g from a local linear baseline over one peak window."""
    m = (TEMP >= win[0]) & (TEMP <= win[1])
    t, y = TEMP[m], hf[m]
    edge = 12
    p = np.polyfit(np.concatenate([t[:edge], t[-edge:]]),
                   np.concatenate([y[:edge], y[-edge:]]), 1)
    base = np.polyval(p, t)
    area = np.trapezoid(sign * (y - base), t)      # W/g * degC
    return area * 60.0 / BETA, t, y, base


def tg_midpoint(hf):
    m1 = (TEMP >= 45.0) & (TEMP <= 64.0)
    m2 = (TEMP >= 94.0) & (TEMP <= 108.0)
    p1 = np.polyfit(TEMP[m1], hf[m1], 1)
    p2 = np.polyfit(TEMP[m2], hf[m2], 1)
    mid = 0.5 * (np.polyval(p1, TEMP) + np.polyval(p2, TEMP))
    w = (TEMP >= 62.0) & (TEMP <= 96.0)
    i = int(np.argmin(np.abs(hf[w] - mid[w])))
    return float(TEMP[w][i]), float(hf[w][i]), p1, p2


def main():
    set_pub_style()
    rng = np.random.default_rng(838)
    cmap = get_cmap("cb_sequential_blue")
    colors = [cmap(v) for v in np.linspace(0.38, 0.95, len(RATES))]

    runs = []
    for i, (rate, col) in enumerate(zip(RATES, colors)):
        hf = trace(rate, rng)
        dh_cc, tcc_t, tcc_y, tcc_b = peak_enthalpy(hf, CC_WIN, +1.0)
        dh_m, tm_t, tm_y, tm_b = peak_enthalpy(hf, MELT_WIN, -1.0)
        tg, tg_y, p1, p2 = tg_midpoint(hf)
        runs.append(dict(
            rate=rate, color=col, hf=hf, off=i * OFFSET, tg=tg, tg_y=tg_y,
            p1=p1, p2=p2, dh_cc=dh_cc, dh_m=dh_m,
            xc=100.0 * (dh_m - dh_cc) / DH100,
            tcc=float(tcc_t[np.argmax(tcc_y - tcc_b)]),
            tm=float(tm_t[np.argmin(tm_y - tm_b)]),
            cc=(tcc_t, tcc_y, tcc_b), mp=(tm_t, tm_y, tm_b)))

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.62, 1.02, 5.72, 4.28))
    ax_xc = axes_inches(fig, (7.52, 3.30, 2.48, 1.88))
    ax_tg = axes_inches(fig, (7.52, 0.98, 2.48, 1.70))

    top = runs[-1]
    for r in runs:
        ax.plot(TEMP, r["hf"] + r["off"], color=r["color"], lw=1.5, zorder=4)
        ax.text(TEMP[0] + 3.0, r["hf"][0] + r["off"] + 0.10,
                "cooled at %g " % r["rate"] + DEG + "C/min", fontsize=6.8,
                color=r["color"], ha="left", va="bottom")
        t, y, b = r["cc"]
        ax.fill_between(t, b + r["off"], y + r["off"], color=r["color"],
                        alpha=0.16, zorder=3)
        t, y, b = r["mp"]
        ax.fill_between(t, y + r["off"], b + r["off"], color=r["color"],
                        alpha=0.16, zorder=3)

    for p, lo, hi in [(top["p1"], 42.0, 92.0), (top["p2"], 66.0, 112.0)]:
        ax.plot([lo, hi], np.polyval(p, [lo, hi]) + top["off"], color=RED,
                lw=0.8, ls=(0, (2.6, 1.8)), zorder=5)
    ax.plot(top["tg"], top["tg_y"] + top["off"], marker="o", ms=4.6,
            mfc="white", mec=RED, mew=1.3, zorder=7)
    ax.annotate("$T_g$ midpoint %.1f " % top["tg"] + DEG + "C",
                (top["tg"], top["tg_y"] + top["off"]),
                xytext=(top["tg"] - 34.0, top["tg_y"] + top["off"] + 0.42),
                fontsize=6.8, color=RED, ha="left",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.8,
                                shrinkA=1.0, shrinkB=2.0))
    ax.annotate("cold crystallisation\n$T_{cc}$ = %.0f " % top["tcc"] + DEG
                + "C,  $\\Delta H_{cc}$ = %.1f J/g" % top["dh_cc"],
                (top["tcc"], top["hf"][np.argmin(np.abs(TEMP - top["tcc"]))]
                 + top["off"]),
                xytext=(top["tcc"] - 12.0, top["off"] + 0.86), fontsize=6.8,
                color=RED, ha="left", linespacing=1.5,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.8,
                                shrinkA=1.0, shrinkB=2.0))
    ax.annotate("melting  $T_m$ = %.0f " % top["tm"] + DEG
                + "C\n$\\Delta H_m$ = %.1f J/g (shaded area)" % top["dh_m"],
                (top["tm"], top["hf"][np.argmin(np.abs(TEMP - top["tm"]))]
                 + top["off"]),
                xytext=(top["tm"] - 74.0, top["off"] - 0.62), fontsize=6.8,
                color=RED, ha="left", linespacing=1.5,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.8,
                                shrinkA=1.0, shrinkB=2.0))

    ax.annotate("", xy=(44.0, 3.02), xytext=(44.0, 2.42),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
    ax.text(48.0, 2.72, "exo up", fontsize=7.6, color=INK, va="center",
            fontweight="bold")
    ax.plot([282.0, 282.0], [-0.55, -0.05], color=INK, lw=1.5,
            solid_capstyle="butt", zorder=6)
    ax.text(278.0, -0.30, "0.5 W/g", fontsize=6.8, color=INK, rotation=90,
            ha="right", va="center")

    ax.set_xlim(TEMP[0], TEMP[-1])
    ax.set_ylim(-0.95, 3.35)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("temperature (" + DEG + "C)", fontsize=8.0)
    ax.set_ylabel("heat flow (W/g), traces offset for clarity", fontsize=8.0)
    ax.tick_params(axis="x", labelsize=7.0)
    ax.set_title("DSC heating scans at %g " % BETA + DEG
                 + "C/min after four cooling rates", fontsize=9.0,
                 fontweight="bold", color=INK)

    rates = np.array(RATES)
    xlim = (rates.min() * 0.55, rates.max() * 1.8)
    xc = np.array([r["xc"] for r in runs])
    tgs = np.array([r["tg"] for r in runs])
    slope_x, icpt_x = np.polyfit(np.log10(rates), xc, 1)
    xg = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 50)
    ax_xc.plot(xg, icpt_x + slope_x * np.log10(xg), color=GREY, lw=0.9,
               ls=(0, (3.0, 2.0)), zorder=2,
               label="%.1f %% per decade" % slope_x)
    for r in runs:
        ax_xc.plot(r["rate"], r["xc"], marker="o", ms=5.6, color=r["color"],
                   mec="white", mew=0.7, zorder=4)
        ax_xc.annotate("%.1f" % r["xc"], (r["rate"], r["xc"]),
                       textcoords="offset points", xytext=(0.0, 7.0),
                       ha="center", fontsize=6.3, color=r["color"])
    ax_xc.set_xscale("log")
    ax_xc.set_xlim(*xlim)
    ax_xc.set_ylim(xc.min() - 4.0, xc.max() + 5.0)
    ax_xc.set_xlabel("cooling rate (" + DEG + "C/min)", fontsize=7.4)
    ax_xc.set_ylabel("crystallinity $X_c$ (%)", fontsize=7.4)
    ax_xc.tick_params(labelsize=6.5)
    ax_xc.legend(fontsize=6.2, frameon=False, loc="lower left",
                 handlelength=1.6)
    ax_xc.set_title("Fast cooling freezes out crystallinity", fontsize=8.5,
                    fontweight="bold", color=INK)

    slope_t, icpt_t = np.polyfit(np.log10(rates), tgs, 1)
    ax_tg.plot(xg, icpt_t + slope_t * np.log10(xg), color=GREY, lw=0.9,
               ls=(0, (3.0, 2.0)), zorder=2,
               label="%+.1f " % slope_t + DEG + "C per decade")
    for r in runs:
        ax_tg.plot(r["rate"], r["tg"], marker="s", ms=5.0, color=r["color"],
                   mec="white", mew=0.7, zorder=4)
    ax_tg.set_xscale("log")
    ax_tg.set_xlim(*xlim)
    ax_tg.set_ylim(tgs.min() - 1.6, tgs.max() + 1.8)
    ax_tg.set_xlabel("cooling rate (" + DEG + "C/min)", fontsize=7.4)
    ax_tg.set_ylabel("$T_g$ midpoint (" + DEG + "C)", fontsize=7.4)
    ax_tg.tick_params(labelsize=6.5)
    ax_tg.legend(fontsize=6.2, frameon=False, loc="upper left",
                 handlelength=1.6)
    ax_tg.set_title("The glass transition is rate dependent", fontsize=8.5,
                    fontweight="bold", color=INK)


    panel_labels_inches(fig, [ax], labels=["(a)"], dx=-0.42, dy=0.16)
    panel_labels_inches(fig, [ax_xc], labels=["(b)"], dx=-0.54, dy=0.14)
    panel_labels_inches(fig, [ax_tg], labels=["(c)"], dx=-0.54, dy=0.14)
    savefig(fig, "838_pol_dsc_thermogram.png")


if __name__ == "__main__":
    main()

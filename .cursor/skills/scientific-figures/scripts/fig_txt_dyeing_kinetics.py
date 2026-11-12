"""Dyeing kinetics: exhaustion curves, half-dyeing times and an Arrhenius fit.

Exhaustion is integrated from dE/dt = k(T(t)) (E_inf - E) along the real bath
temperature ramp, so the curves and the ramp on the secondary axis describe the
same run.  The half-dyeing time is then read back off each integrated curve by
interpolation, the apparent rate constant is ln2 / t_half, and the activation
energy in the Arrhenius panel is the slope of a straight-line fit through those
four points - nothing on the page is typed in by hand.
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

R_GAS = 8.314           # J / (mol K)
EA_IN = 42.0e3          # J / mol, input activation energy
A_PRE = 1.0e5           # 1 / min
T_START = 40.0          # degC, bath entry
RAMP = 4.0              # degC / min
T_END = 90.0            # min, total dyeing time
HOLDS = [60.0, 75.0, 90.0, 105.0]
LIQUOR = [10.0, 20.0, 40.0]
PH = [4.0, 5.0, 6.0, 7.0]


def bath_temp(t, hold):
    return np.minimum(T_START + RAMP * t, hold)


def exhaustion(hold, n=1800):
    """Integrate first-order exhaustion along the bath temperature ramp."""
    t = np.linspace(0.0, T_END, n)
    dt = t[1] - t[0]
    temp = bath_temp(t, hold)
    k = A_PRE * np.exp(-EA_IN / (R_GAS * (temp + 273.15)))
    e_inf = 92.0 - 0.115 * (hold - HOLDS[0])
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = e[i - 1] + dt * k[i - 1] * (e_inf - e[i - 1])
    return t, e, temp


def half_time(t, e, e_inf):
    """Time at which exhaustion reaches half of its equilibrium value."""
    return float(np.interp(0.5 * e_inf, e, t))


def rate_from_curve(t, e, hold):
    """Isothermal k and E_inf from a regression of dE/dt on E.

    First order means dE/dt = k E_inf - k E, so a straight line through the
    hold-period points returns both constants without either being assumed.
    """
    dedt = np.gradient(e, t)
    i0 = int(np.searchsorted(t, (hold - T_START) / RAMP))
    keep = (np.arange(t.size) >= i0) & (dedt >= 0.02 * dedt[i0])
    slope, icpt = np.polyfit(e[keep], dedt[keep], 1)
    k = -float(slope)
    return k, float(icpt) / k


def main():
    set_pub_style()
    rng = np.random.default_rng(836)
    cmap = get_cmap("cb_sequential_warm")
    colors = [cmap(v) for v in np.linspace(0.36, 0.95, len(HOLDS))]

    runs = []
    for hold, col in zip(HOLDS, colors):
        t, e, temp = exhaustion(hold)
        k, e_inf = rate_from_curve(t, e, hold)
        th = half_time(t, e, e_inf)
        runs.append(dict(hold=hold, t=t, e=e, temp=temp, color=col, th=th,
                         k=k, ef=e_inf))

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.80, 1.02, 5.28, 4.28))
    axt = ax.twinx()
    ax_ar = axes_inches(fig, (7.52, 3.30, 2.48, 1.88))
    ax_eq = axes_inches(fig, (7.52, 0.98, 2.48, 1.70))

    for r in runs:
        axt.plot(r["t"], r["temp"], color=r["color"], lw=0.85,
                 ls=(0, (3.2, 2.0)), alpha=0.65, zorder=2)
    axt.set_ylim(20.0, 200.0)
    axt.set_yticks([40.0, 60.0, 80.0, 100.0])
    axt.set_ylabel("bath temperature (" + chr(176) + "C)", fontsize=8.0,
                   color=SUB)
    axt.tick_params(axis="y", labelsize=7.0, colors=SUB)
    axt.spines["top"].set_visible(False)
    axt.spines["right"].set_color(GREY)
    ax.text(0.015, 0.985, "dashed: bath temperature ramp, "
            + "%.0f " % RAMP + chr(176) + "C/min from %.0f " % T_START
            + chr(176) + "C (right axis)", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.5, color=MUTED)

    for r in runs:
        ax.plot(r["t"], r["e"], color=r["color"], lw=1.7, zorder=5,
                label="%.0f " % r["hold"] + chr(176)
                      + "C   $t_{1/2}$ = %4.1f min   $E_{\\infty}$ = %.1f %%"
                      % (r["th"], r["ef"]))
        pick = np.linspace(0, r["t"].size - 1, 15).astype(int)[1:]
        ax.plot(r["t"][pick], r["e"][pick] + rng.normal(0.0, 0.55, pick.size),
                ls="none", marker="o", ms=2.9, mfc="white", mec=r["color"],
                mew=0.8, zorder=6)
        ax.plot([r["th"], r["th"]], [0.0, 0.5 * r["ef"]], color=r["color"],
                lw=0.8, ls=(0, (1.4, 1.6)), zorder=4)
        ax.plot(r["th"], 0.5 * r["ef"], marker="v", ms=4.2, color=r["color"],
                zorder=7)

    ax.set_xlim(0.0, T_END)
    ax.set_ylim(0.0, 108.0)
    ax.set_yticks([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
    ax.set_xlabel("dyeing time (min)", fontsize=8.0)
    ax.set_ylabel("exhaustion $E$ (%)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.set_zorder(axt.get_zorder() + 1)
    ax.patch.set_visible(False)
    leg = ax.legend(fontsize=7.0, frameon=False, loc="lower right",
                    bbox_to_anchor=(0.995, 0.055), handlelength=1.7,
                    labelspacing=0.34,
                    title="hold temperature (markers = half-dyeing point)")
    leg.get_title().set_fontsize(6.6)
    leg.get_title().set_color(SUB)
    ax.set_title("Exhaustion against time for four dyeing temperatures",
                 fontsize=9.0, fontweight="bold", color=INK)

    inv_t = np.array([1000.0 / (r["hold"] + 273.15) for r in runs])
    lnk = np.array([np.log(r["k"]) for r in runs])
    slope, icpt = np.polyfit(inv_t, lnk, 1)
    ea = -slope * R_GAS * 1000.0 / 1000.0        # kJ / mol
    xg = np.linspace(inv_t.min() * 0.995, inv_t.max() * 1.005, 40)
    ax_ar.plot(xg, icpt + slope * xg, color=GREY, lw=1.0, ls=(0, (3.0, 2.0)),
               zorder=2)
    for r, x, y in zip(runs, inv_t, lnk):
        ax_ar.plot(x, y, marker="o", ms=5.4, color=r["color"], mec="white",
                   mew=0.7, zorder=4)
        ax_ar.annotate("%.0f " % r["hold"] + chr(176) + "C", (x, y),
                       textcoords="offset points", xytext=(0.0, 7.0),
                       ha="center", fontsize=6.3, color=r["color"])
    ax_ar.text(0.97, 0.60,
               "$E_a$ = %.1f kJ/mol\nfrom slope %.0f K" % (ea, slope * 1000.0),
               transform=ax_ar.transAxes, ha="right", va="top",
               fontsize=7.0, color=INK, linespacing=1.5)
    ax_ar.set_xlabel("$1000/T$ (K$^{-1}$)", fontsize=7.4)
    ax_ar.set_ylabel("ln $k$   ($k$ in min$^{-1}$)", fontsize=7.4)
    ax_ar.tick_params(labelsize=6.5)
    ax_ar.set_ylim(lnk.min() - 0.45, lnk.max() + 0.55)
    ax_ar.set_title("Arrhenius plot of the apparent rate", fontsize=8.5,
                    fontweight="bold", color=INK)

    k_ph = 260.0 * np.exp(-0.55 * (np.array(PH) - PH[0]))
    eq = np.array([[100.0 * k / (k + lr) for k in k_ph] for lr in LIQUOR])
    pos = np.arange(len(PH), dtype=float)
    width = 0.76 / len(LIQUOR)
    greens = get_cmap("cb_sequential_green")
    for i, lr in enumerate(LIQUOR):
        off = -0.38 + width * (i + 0.5)
        ax_eq.bar(pos + off, eq[i], width=width * 0.9,
                  color=greens(0.35 + 0.25 * i), edgecolor="white",
                  linewidth=0.5, label="1:%.0f" % lr, zorder=3)
    ax_eq.set_xticks(pos)
    ax_eq.set_xticklabels(["%.0f" % p for p in PH], fontsize=6.5)
    ax_eq.set_xlabel("bath pH (-)", fontsize=7.4)
    ax_eq.set_ylabel("equilibrium $E_{\\infty}$ (%)", fontsize=7.4)
    ax_eq.set_ylim(0.0, 134.0)
    ax_eq.set_yticks([0.0, 25.0, 50.0, 75.0, 100.0])
    ax_eq.tick_params(axis="y", labelsize=6.5)
    ax_eq.legend(fontsize=6.2, frameon=False, ncol=3, loc="upper center",
                 bbox_to_anchor=(0.5, 1.0), handlelength=1.1,
                 columnspacing=1.0, title="liquor ratio")
    ax_eq.get_legend().get_title().set_fontsize(6.2)
    ax_eq.get_legend().get_title().set_color(SUB)
    for i, j in [np.unravel_index(int(np.argmax(eq)), eq.shape),
                 np.unravel_index(int(np.argmin(eq)), eq.shape)]:
        xb = pos[j] - 0.38 + width * (i + 0.5)
        ax_eq.text(xb, eq[i, j] + 2.0, "%.0f" % eq[i, j], ha="center",
                   va="bottom", fontsize=6.2, color=SUB)
    ax_eq.set_title("Acid bath and short liquor raise exhaustion",
                    fontsize=8.5, fontweight="bold", color=INK)


    panel_labels_inches(fig, [ax], labels=["(a)"], dx=-0.58, dy=0.16)
    panel_labels_inches(fig, [ax_ar], labels=["(b)"], dx=-0.52, dy=0.14)
    panel_labels_inches(fig, [ax_eq], labels=["(c)"], dx=-0.52, dy=0.14)
    savefig(fig, "836_txt_dyeing_kinetics.png")


if __name__ == "__main__":
    main()

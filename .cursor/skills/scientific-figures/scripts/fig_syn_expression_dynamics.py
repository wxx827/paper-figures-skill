"""Induction time courses of a fluorescent reporter, with dose response.

The main panel carries the claim: six IPTG doses drive the same sigmoidal
approach to a dose-dependent plateau while the host grows along one shared
logistic curve, so the reporter signal is not simply tracking biomass.  Both
supporting panels are measured from the very curves that are drawn - the Hill
parameters come from a fit to the plateaus read off the time courses, and the
noise panel regresses the coefficient of variation of the replicate ensemble
against its own mean, so changing the seed moves every quoted number.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from pubstyle import set_pub_style, get_palette, savefig
from figkit import axes_inches, panel_label_inches, shared_limits

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GREY = "#808080"
RED = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"

W, H = 10.4, 6.4
CONC = np.array([0.0, 10.0, 30.0, 60.0, 150.0, 600.0])   # IPTG, micromolar
T_END, T_IND = 14.0, 1.5                                  # hours
K_TRUE, N_TRUE = 45.0, 1.9
BASAL, SPAN = 0.055, 1.02
TAU, LAG = 1.65, 1.10                                     # hours
MU_MAX, OD0, OD_CAP = 0.62, 0.045, 4.3
N_REP = 6


def hill(c, top, k, n):
    return top * c ** n / (k ** n + c ** n)


def series(seed=7):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T_END, 320)
    plateau = BASAL + hill(CONC, SPAN, K_TRUE, N_TRUE)
    ramp = 1.0 / (1.0 + np.exp(-(t - T_IND - LAG) / TAU))
    ramp = np.clip((ramp - ramp[0]) / (1.0 - ramp[0]), 0.0, None)

    curves, reps = [], []
    for p in plateau:
        base = BASAL + (p - BASAL) * ramp
        cv = 0.15 * (p / plateau.max()) ** -0.42          # cell-to-cell spread
        gain = 1.0 + rng.normal(0.0, cv, (N_REP, 1))
        meas = rng.normal(0.0, 0.008 + 0.015 * np.sqrt(base), (N_REP, t.size))
        rep = np.clip(base * gain + meas, 1e-4, None)
        curves.append(rep.mean(axis=0))
        reps.append(rep)
    od = OD_CAP / (1.0 + (OD_CAP / OD0 - 1.0) * np.exp(-MU_MAX * t))
    od = od * (1.0 + rng.normal(0.0, 0.012, t.size))
    return t, np.array(curves), np.array(reps), od


def colors():
    ramp = get_palette("cb_sequential_blue", 8)[2:]
    return [GREY] + list(ramp[:CONC.size - 1])


def main_panel(ax, t, curves, od, cols):
    ax.axvspan(0.0, T_IND, facecolor="#f0f0f0", edgecolor="none", zorder=0)
    ax.axvline(T_IND, color=GREY, lw=0.9, ls=(0, (3.2, 2.4)), zorder=2)
    ax.text(T_IND + 0.16, 0.42, f"induction at t = {T_IND:.1f} h", ha="left",
            va="center", rotation=90.0, fontsize=6.8, color=SUB)
    ax.text(T_IND - 0.16, 0.42, "pre-induction", ha="right", va="center",
            rotation=90.0, fontsize=6.8, color=MUTED)

    for c, y, col in zip(CONC, curves, cols):
        ax.plot(t, y, color=col, lw=1.7, zorder=4,
                label=f"{c:.0f} " + r"$\mu$M")
    ax.set_xlabel("time after inoculation (h)")
    ax.set_ylabel("reporter fluorescence (RFU)")
    ax.set_xlim(0.0, T_END)
    ax.set_ylim(0.0, 1.20)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    leg = ax.legend(title=r"IPTG (${\mu}$M)", ncol=2, fontsize=7.0,
                    frameon=False, loc="upper left",
                    bbox_to_anchor=(0.015, 0.955), handlelength=1.5,
                    labelspacing=0.30, columnspacing=1.0)
    leg.get_title().set_fontsize(7.0)

    ax2 = ax.twinx()
    ax2.plot(t, od, color=GREEN, lw=1.4, ls=(0, (4.5, 2.2)), zorder=3)
    ax2.set_ylabel(r"biomass OD$_{600}$ (-)", color=GREEN)
    ax2.set_ylim(0.0, 5.2)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(GREEN)
    ax2.tick_params(axis="y", colors=GREEN)
    ax2.annotate(r"host growth, OD$_{600}$", xy=(11.6, np.interp(11.6, t, od)),
                 xytext=(8.1, 2.15), ha="left", va="center", fontsize=7.0,
                 color=GREEN,
                 arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.7))

    top = curves[-1][t >= T_END - 1.0].mean()
    half = np.interp(0.5 * top, curves[-1], t)
    ax.set_title(f"Full induction reaches {top:.2f} RFU with a "
                 f"{half - T_IND:.1f} h half-rise time", fontsize=9.0,
                 fontweight="bold", color=INK)
    return top


def dose_panel(ax, t, curves, cols, ylim):
    ss = curves[:, t >= T_END - 1.0].mean(axis=1)
    m = CONC > 0
    p0 = (ss.max(), 50.0, 1.5)
    fit, _ = curve_fit(lambda c, top, k, n: BASAL + hill(c, top, k, n),
                       CONC[m], ss[m], p0=p0, maxfev=20000)
    top, kd, nh = fit

    grid = np.logspace(np.log10(2.0), np.log10(2000.0), 200)
    ax.plot(grid, BASAL + hill(grid, top, kd, nh), color=BLUE, lw=1.5,
            zorder=3)
    ax.axhline(ss[0], color=GREY, lw=0.9, ls=(0, (2.6, 2.0)), zorder=2)
    ax.text(1800.0, ss[0] + 0.04, f"uninduced basal {ss[0]:.2f} RFU",
            ha="right", va="bottom", fontsize=6.2, color=MUTED)
    for c, v, col in zip(CONC[m], ss[m], np.array(cols)[m]):
        ax.plot([c], [v], marker="o", ms=4.4, color=col, mec="white",
                mew=0.7, zorder=5)

    ax.axvline(kd, color=RED, lw=0.9, ls=(0, (3.0, 2.2)), zorder=2)
    ax.text(kd * 1.12, ylim[1] * 0.035, r"$K_d$", ha="left", va="bottom",
            fontsize=6.8, color=RED)
    ax.text(1800.0, ylim[1] * 0.28,
            f"$K_d$ = {kd:.0f} " + r"$\mu$M" + f"\nHill $n$ = {nh:.2f}",
            ha="right", va="center", fontsize=6.8, color=RED, linespacing=1.5)

    ax.set_xscale("log")
    ax.set_xlabel(r"IPTG concentration ($\mu$M)", fontsize=7.4)
    ax.set_ylabel("plateau (RFU)", fontsize=7.4)
    ax.set_xlim(2.0, 2000.0)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title("Dose response of the plateau level", fontsize=8.2,
                 fontweight="bold", color=INK)
    return ss, kd, nh


def noise_panel(ax, t, reps, cols):
    win = t >= T_END - 3.0
    mu, cv = [], []
    for rep in reps:
        block = rep[:, win]
        m = block.mean()
        mu.append(m)
        cv.append(block.mean(axis=1).std(ddof=1) / m * 100.0)
    mu, cv = np.array(mu), np.array(cv)

    slope, inter = np.polyfit(np.log10(mu), np.log10(cv), 1)
    grid = np.logspace(np.log10(mu.min() * 0.7), np.log10(mu.max() * 1.4), 60)
    ax.plot(grid, 10.0 ** inter * grid ** slope, color=BLUE, lw=1.3, zorder=3)
    ref = cv[0] * (grid / mu[0]) ** -0.5
    ax.plot(grid, ref, color=GREY, lw=0.9, ls=(0, (3.0, 2.2)), zorder=2)
    x_ref = float(np.sqrt(grid[0] * grid[-1]))
    ax.text(x_ref, cv[0] * (x_ref / mu[0]) ** -0.5 * 0.86,
            "Poisson slope $-1/2$", ha="center", va="top", fontsize=6.2,
            color=MUTED)

    for m, v, col, c in zip(mu, cv, cols, CONC):
        ax.plot([m], [v], marker="o", ms=4.6, color=col, mec="white", mew=0.7,
                zorder=5)
    ax.annotate(f"uninduced\n{cv[0]:.0f} % CV", xy=(mu[0], cv[0]),
                xytext=(mu[0] * 1.5, cv[0] * 1.35), fontsize=6.2, color=MUTED,
                ha="left", linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean fluorescence (RFU)", fontsize=7.4)
    ax.set_ylabel("noise CV (%)", fontsize=7.4)
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0])
    ax.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1"])
    ax.set_ylim(6.0, 90.0)
    ax.set_yticks([10.0, 20.0, 40.0, 80.0])
    ax.set_yticklabels(["10", "20", "40", "80"])
    ax.minorticks_off()
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title(f"Expression noise falls as CV $\\propto$ mean$^{{"
                 f"{slope:.2f}}}$", fontsize=8.2, fontweight="bold",
                 color=INK)
    return cv, slope


def main():
    set_pub_style()
    t, curves, reps, od = series()
    cols = colors()

    fig = plt.figure(figsize=(W, H))
    ax_main = axes_inches(fig, (0.82, 0.78, 5.30, 4.42))
    ax_dose = axes_inches(fig, (7.08, 3.13, 3.05, 2.07))
    ax_cv = axes_inches(fig, (7.08, 0.78, 3.05, 1.72))

    top = main_panel(ax_main, t, curves, od, cols)
    lim = shared_limits([curves[:, t >= T_END - 1.0].mean(axis=1),
                         np.array([0.0])], pad=0.10)
    ss, kd, nh = dose_panel(ax_dose, t, curves, cols, (0.0, lim[1]))
    cv, slope = noise_panel(ax_cv, t, reps, cols)

    fig.text(0.021, 6.16 / H,
             "Reporter output saturates well before the culture does: the "
             "plateau is set by inducer dose, not by biomass",
             ha="left", va="center", fontsize=11.0, fontweight="bold",
             color=INK)
    fig.text(0.021, 0.11 / H,
             f"{N_REP} biological replicates per dose; curves are replicate "
             f"means. Plateaus are averaged over the last hour of the run, "
             f"and the Hill fit returns $K_d$ = {kd:.0f} $\\mu$M with "
             f"cooperativity $n$ = {nh:.2f}.",
             ha="left", va="bottom", fontsize=6.5, color=MUTED)

    for ax, lab in [(ax_main, "(a)"), (ax_dose, "(b)"), (ax_cv, "(c)")]:
        panel_label_inches(fig, ax, lab, dx=-0.58, dy=0.14)

    savefig(fig, "709_syn_expression_dynamics.png")


if __name__ == "__main__":
    main()

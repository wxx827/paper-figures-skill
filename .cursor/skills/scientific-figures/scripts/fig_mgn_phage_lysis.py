"""Phage lysis of a growing host culture across four decades of MOI.

One delayed-lysis model is integrated once per condition: susceptible cells
grow logistically, adsorb phage at a mass-action rate, pass through five
infected stages so that the latent period is a real delay rather than an
instant conversion, and release a burst of virions at the end of it.  The
turbidity plotted in (a) is the total cell density from that integration
divided by the standard conversion factor, and the phage titre in (b) is the
free-virion state of the same runs.

Every quoted number is measured off those trajectories: the host specific
growth rate is the slope of a log-linear fit over the uninfected control's
exponential window, the lysis half-time of each condition is the interpolated
crossing of half the turbidity that condition reached, and the dose response
in (c) is an ordinary least-squares fit of those half-times against log10 MOI.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from pubstyle import set_pub_style, savefig, get_palette, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
ORANGE = "#d98032"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

MOIS = np.array([1.0e-3, 1.0e-2, 1.0e-1, 1.0e0, 1.0e1])
T_END = 12.0                  # h
DT = 0.0025                   # h
MU_MAX = 0.95                 # 1/h
K_CARRY = 1.2e9               # CFU/mL
S0 = 4.0e7                    # CFU/mL
K_ADS = 3.0e-10               # mL/(PFU h)
LATENT = 0.70                 # h
N_STAGE = 5
BURST = 100                   # PFU per lysed cell
OD_FACTOR = 8.0e8             # CFU/mL per OD600 unit
NOISE_OD = 0.004

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.85, 1.24, 5.55, 4.22)
AUX_T = (7.44, 3.60, 3.18, 1.86)
AUX_B = (7.44, 1.24, 3.18, 1.72)


def simulate(moi, rng):
    """Explicit Euler on the delayed-lysis system; returns t, OD, titre."""
    n_step = int(round(T_END / DT)) + 1
    t = np.arange(n_step) * DT
    s = S0
    inf = np.zeros(N_STAGE)
    p = max(moi, 0.0) * S0
    rate = N_STAGE / LATENT
    od = np.empty(n_step)
    titre = np.empty(n_step)
    for i in range(n_step):
        total = s + inf.sum()
        od[i] = total / OD_FACTOR
        titre[i] = max(p, 1.0)
        grow = MU_MAX * s * (1.0 - total / K_CARRY)
        ads = K_ADS * s * p
        ds = grow - ads
        di = np.empty(N_STAGE)
        di[0] = ads - rate * inf[0]
        di[1:] = rate * inf[:-1] - rate * inf[1:]
        dp = BURST * rate * inf[-1] - ads
        s = max(s + DT * ds, 0.0)
        inf = np.maximum(inf + DT * di, 0.0)
        p = max(p + DT * dp, 0.0)
    od = np.maximum(od + rng.normal(0.0, NOISE_OD, n_step), 0.0)
    return t, od, titre


def growth_rate(t, od, lo=0.09, hi=0.45):
    """Specific growth rate from the log-linear part of the control curve."""
    m = (od > lo) & (od < hi) & (t < 6.0)
    fit = stats.linregress(t[m], np.log(od[m]))
    return fit, np.log(2.0) / fit.slope, m


def half_lysis_time(t, od):
    """Interpolated time at which turbidity falls to half of its own peak."""
    k = int(np.argmax(od))
    target = 0.5 * od[k]
    after = np.where(od[k:] <= target)[0]
    if after.size == 0:
        return np.nan, float(od[k]), float(od.min())
    j = k + int(after[0])
    y0, y1 = od[j - 1], od[j]
    frac = (y0 - target) / (y0 - y1) if y1 != y0 else 0.0
    return (float(t[j - 1] + frac * (t[j] - t[j - 1])), float(od[k]),
            float(od[k:].min()))


def lysis_panel(ax, t, ctrl, runs, cols, gfit, td, mask, half):
    ax.plot(t, ctrl, color=BLUE, lw=1.6, zorder=6, label="host only, no phage")
    for i, moi in enumerate(MOIS):
        ax.plot(t, runs[i], color=cols[i], lw=1.25, zorder=5,
                label="MOI {:g}".format(moi))
        if np.isfinite(half[i]):
            ax.plot([half[i]], [0.5 * runs[i].max()], marker="v", ms=4.4,
                    mfc="white", mec=cols[i], mew=1.1, zorder=7)

    tw = t[mask]
    span = np.linspace(tw.min() - 0.25, tw.max() + 0.55, 40)
    ax.plot(span, np.exp(gfit.intercept + gfit.slope * span), color="#333333",
            lw=1.0, ls=(0, (3.4, 2.2)), zorder=8)

    ax.set_xlim(0.0, T_END)
    ax.set_ylim(0.0, 1.62)
    ax.set_xlabel("time after infection (h)", fontsize=7.4)
    ax.set_ylabel("optical density at 600 nm, OD600 (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.4)
    ax.grid(lw=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.text(0.985, 0.978,
            "uninfected control\n$\\mu$ = {:.3f} h$^{{-1}}$ over the fitted "
            "window\ndoubling time {:.0f} min\nOD600 1.0 = {:.1f} x "
            "$10^{{8}}$ CFU/mL".format(gfit.slope, 60.0 * td,
                                       OD_FACTOR / 1.0e8),
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
            color=INK, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor=HAIR, linewidth=0.6, alpha=0.93), zorder=9)
    ax.text(0.985, 0.42, "triangle = interpolated half-lysis time\n"
            "of that condition\ndashed = fitted exponential phase",
            transform=ax.transAxes, ha="right", va="center", fontsize=6.0,
            color=MUTED, linespacing=1.45, zorder=9)
    leg = ax.legend(loc="upper left", fontsize=6.8, handlelength=1.6,
                    handletextpad=0.45, labelspacing=0.32, borderpad=0.40,
                    frameon=True, framealpha=0.93, edgecolor=HAIR)
    leg.get_frame().set_linewidth(0.6)
    ax.set_title("Turbidity of an infected culture across four decades of "
                 "multiplicity of infection", fontsize=9.0, fontweight="bold",
                 color=INK, pad=6.0)


def titre_panel(ax, t, titres, cols):
    for i, moi in enumerate(MOIS):
        ax.plot(t, np.log10(titres[i]), color=cols[i], lw=1.15, zorder=4)
    gain = np.log10(titres[0][-1] / titres[0][0])
    ax.set_xlim(0.0, T_END)
    ax.set_xlabel("time after infection (h)", fontsize=7.0)
    ax.set_ylabel("phage titre ($\\log_{10}$ PFU/mL)", fontsize=7.0)
    ax.tick_params(labelsize=6.3)
    ax.grid(lw=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.985, 0.045, "MOI {:g} gains {:.1f} $\\log_{{10}}$ PFU/mL in "
            "{:.0f} h\nline colours as in panel (a)"
            .format(MOIS[0], gain, T_END), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.1, color=SUB,
            linespacing=1.4)
    ax.set_title("Free virions released by the same five runs", fontsize=8.0,
                 fontweight="bold", color=INK, pad=5.0)
    return gain


def dose_panel(ax, half, peak, floor, cols):
    x = np.log10(MOIS)
    ok = np.isfinite(half)
    fit = stats.linregress(x[ok], half[ok])
    grid = np.linspace(x.min() - 0.35, x.max() + 0.35, 40)
    ax.plot(grid, fit.intercept + fit.slope * grid, color="#333333", lw=1.0,
            zorder=3)
    for i in range(len(MOIS)):
        ax.plot([x[i]], [half[i]], marker="o", ms=5.0,
                mfc=lighten(cols[i], 0.30), mec=cols[i], mew=1.0, zorder=5)
        ax.text(x[i], half[i] + 0.28, "peak OD {:.2f}".format(peak[i]),
                ha="center", va="bottom", fontsize=6, color=MUTED,
                rotation=90, zorder=6)

    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0.0, float(np.nanmax(half)) * 1.52)
    ax.set_xticks(x)
    ax.set_xticklabels(["{:g}".format(v) for v in MOIS])
    ax.set_xlabel("multiplicity of infection (PFU per CFU)", fontsize=7.0)
    ax.set_ylabel("half-lysis time (h)", fontsize=7.0)
    ax.tick_params(labelsize=6.3)
    ax.grid(lw=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.985, 0.965, "{:+.2f} h per decade of MOI\n$R^2$ = {:.4f}"
            .format(fit.slope, fit.rvalue ** 2), transform=ax.transAxes,
            ha="right", va="top", fontsize=6.2, color=INK, linespacing=1.45)
    ax.set_title("Dose response of the half-lysis times marked in (a)",
                 fontsize=8.0, fontweight="bold", color=INK, pad=5.0)
    return fit


def audit(fig, foot, axes):
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
    rng = np.random.default_rng(9655)
    t, ctrl, _ = simulate(0.0, rng)
    runs, titres = [], []
    for moi in MOIS:
        _, od, tit = simulate(moi, rng)
        runs.append(od)
        titres.append(tit)
    runs = np.asarray(runs)
    titres = np.asarray(titres)

    gfit, td, mask = growth_rate(t, ctrl)
    half = np.empty(len(MOIS))
    peak = np.empty(len(MOIS))
    floor = np.empty(len(MOIS))
    for i in range(len(MOIS)):
        half[i], peak[i], floor[i] = half_lysis_time(t, runs[i])
    assert np.all(np.isfinite(half)), "every condition must reach half lysis"

    ramp = get_palette("cb_sequential_warm", 8)[2:]
    cols = list(ramp[::-1])[:len(MOIS)][::-1]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_od = axes_inches(fig, MAIN)
    ax_tit = axes_inches(fig, AUX_T)
    ax_dose = axes_inches(fig, AUX_B)

    lysis_panel(ax_od, t, ctrl, runs, cols, gfit, td, mask, half)
    gain = titre_panel(ax_tit, t, titres, cols)
    fit = dose_panel(ax_dose, half, peak, floor, cols)

    panel_labels_inches(fig, [ax_od], labels=["(a)"], dx=-0.50, dy=0.14)
    panel_labels_inches(fig, [ax_tit, ax_dose], labels=["(b)", "(c)"],
                        dx=-0.46, dy=0.14)
    savefig(fig, "965_mgn_phage_lysis.png")


if __name__ == "__main__":
    main()

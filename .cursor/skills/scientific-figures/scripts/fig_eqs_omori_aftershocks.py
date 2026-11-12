"""Aftershock decay of an Mw 6.8 mainshock under the modified Omori law.

The sequence is simulated, not sketched.  Occurrence times are inverse-CDF
draws from the modified Omori intensity K / (t + c)^p, magnitudes come from a
Gutenberg-Richter law, and an event survives only if it clears a
time-dependent detection threshold that starts high in the first minutes and
relaxes as the network digs itself out.  Everything quoted is then measured
back off that catalogue: the rate is counts divided by the width of
logarithmic time bins, and p, c and K are the maximum-likelihood parameters of
the non-homogeneous Poisson process, fitted only after the time at which the
completeness curve has dropped below the magnitude cutoff in use.  Both axes
of the main panel are logarithmic, which is what turns the power law into the
straight line the eye can check.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import minimize

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
DEEP = "#22406e"
BLUE = "#3b5b92"
PALE = "#a8bcd8"
RED = "#b23a48"
ROSE = "#d79aa2"
GOLD = "#c9a227"
GREY = "#8a8a8a"

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.02, 4.66, 4.46)
AUX_T = (7.42, 3.61, 2.70, 1.87)
AUX_B = (7.42, 1.02, 2.70, 1.82)

M_MAIN = 6.8              # mainshock moment magnitude, Mw
N_SEED = 4200             # aftershocks generated above M_MIN
M_MIN = 1.8               # generating magnitude floor, Mw
B_TRUE = 0.98
P_TRUE, C_TRUE = 1.12, 0.015          # Omori exponent (-), c in days
T_END = 365.0             # days of catalogue
DM = 0.1                  # reported magnitude step, Mw
MC_OFF, MC_SLOPE = 5.0, 0.76          # short-term incompleteness constants

CUTS = [dict(mcut=2.5, name="Mw >= 2.5", color=DEEP, fill=PALE, marker="o"),
        dict(mcut=3.5, name="Mw >= 3.5", color=RED, fill=ROSE, marker="s")]


def omori_times(rng, n, p, c, t_end):
    """Inverse-CDF draws from K / (t + c)^p on (0, t_end]."""
    q = 1.0 - p
    a, b = c ** q, (t_end + c) ** q
    u = rng.uniform(0.0, 1.0, n)
    return np.sort((a + u * (b - a)) ** (1.0 / q) - c)


def gr_magnitudes(rng, n, b, m_lo, m_hi):
    beta = b * np.log(10.0)
    u = rng.uniform(0.0, 1.0, n)
    span = 1.0 - np.exp(-beta * (m_hi - m_lo))
    return np.round((m_lo - np.log1p(-u * span) / beta) / DM) * DM


def completeness(t):
    """Detection threshold in Mw at time ``t`` days after the mainshock."""
    return M_MAIN - MC_OFF - MC_SLOPE * np.log10(np.maximum(t, 1e-6))


def complete_after(mcut):
    """First time in days at which the network is complete down to ``mcut``."""
    return 10.0 ** ((M_MAIN - MC_OFF - mcut) / MC_SLOPE)


def build_catalogue(rng):
    t = omori_times(rng, N_SEED, P_TRUE, C_TRUE, T_END)
    m = gr_magnitudes(rng, N_SEED, B_TRUE, M_MIN, M_MAIN - 0.4)
    keep = m >= np.maximum(completeness(t), M_MIN)
    return t[keep], m[keep]


def log_bins(t0, t1, per_decade=6):
    n = int(round(per_decade * np.log10(t1 / t0)))
    return np.logspace(np.log10(t0), np.log10(t1), n + 1)


def binned_rate(t, edges):
    """Counts per day in each logarithmic bin, with the bin geometric centre."""
    cnt, _ = np.histogram(t, bins=edges)
    width = np.diff(edges)
    centre = np.sqrt(edges[:-1] * edges[1:])
    ok = cnt > 0
    return centre[ok], cnt[ok] / width[ok], cnt[ok]


def fit_omori(t, t0, t1):
    """Maximum-likelihood p, c, K for the modified Omori law on [t0, t1]."""
    s = t[(t >= t0) & (t <= t1)]
    n = s.size

    def integral(p, c):
        q = 1.0 - p
        return ((t1 + c) ** q - (t0 + c) ** q) / q

    def nll(theta):
        p, log_c = theta
        c = np.exp(log_c)
        a = integral(p, c)
        if a <= 0.0:
            return 1e9
        return -(n * np.log(n / a) - n - p * np.log(s + c).sum())

    res = minimize(nll, x0=[1.1, np.log(0.02)], method="L-BFGS-B",
                   bounds=[(0.55, 2.0), (np.log(1e-4), np.log(2.0))])
    p, c = float(res.x[0]), float(np.exp(res.x[1]))
    k = n / integral(p, c)
    return dict(p=p, c=c, k=k, n=n, t0=t0, t1=t1)


def omori_rate(fit, t):
    return fit["k"] / (t + fit["c"]) ** fit["p"]


def omori_cumulative(fit, t):
    q = 1.0 - fit["p"]
    return fit["k"] * ((t + fit["c"]) ** q - (fit["t0"] + fit["c"]) ** q) / q


def main_panel(ax, sets):
    for s in sets:
        ax.plot(s["tc"], s["rate"], ls="none", marker=s["marker"], ms=4.0,
                mfc=s["fill"], mec=s["color"], mew=0.7, zorder=5)
        ok = s["tc"] >= s["fit"]["t0"]
        ax.plot(s["tc"][ok], s["rate"][ok], ls="none", marker=s["marker"],
                ms=4.0, mfc=s["color"], mec="white", mew=0.5, zorder=6)
        tt = np.logspace(np.log10(s["fit"]["t0"]), np.log10(T_END), 200)
        ax.plot(tt, omori_rate(s["fit"], tt), color=s["color"], lw=1.6,
                zorder=7)
        ax.axvline(s["fit"]["t0"], color=s["color"], lw=0.9,
                   ls=(0, (3.6, 2.4)), alpha=0.85, zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2.5e-3, T_END * 1.6)
    ax.set_ylim(6e-3, 4.0e4)
    ax.set_xlabel("time since mainshock (days, log scale)")
    ax.set_ylabel("aftershock rate (events per day, log scale)")
    ax.tick_params(labelsize=6.8)
    ax.grid(which="both", lw=0.45, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Modified Omori decay at two magnitude cutoffs",
                 fontsize=9.0, fontweight="bold", color=INK, pad=5.0)

    lines = []
    for s in sets:
        f = s["fit"]
        lines.append("{:s}: p = {:.2f}, c = {:.3f} d, K = {:.0f} "
                     "events/d, N = {:d} above {:.2f} d"
                     .format(s["name"], f["p"], f["c"], f["k"], f["n"],
                             f["t0"]))
    lines.append("open symbols: bins the network could not see to that "
                 "cutoff yet")
    ax.text(0.972, 0.955, "\n".join(lines), transform=ax.transAxes,
            ha="right", va="top", fontsize=6.6, color=INK, linespacing=1.55,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor="#dddddd", linewidth=0.6, alpha=0.94),
            zorder=10)

    handles = [
        Line2D([], [], ls="none", marker="o", ms=4.0, mfc=DEEP, mec="white",
               label="binned rate, complete"),
        Line2D([], [], ls="none", marker="o", ms=4.0, mfc=PALE, mec=DEEP,
               mew=0.7, label="binned rate, incomplete"),
        Line2D([], [], color=SUB, lw=1.6,
               label="maximum-likelihood K / (t + c)^p"),
        Line2D([], [], color=SUB, lw=0.9, ls=(0, (3.6, 2.4)),
               label="start of the fitting window"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=6.3, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.7, labelspacing=0.28, borderaxespad=0.35)


def magnitude_panel(ax, t, m, largest):
    ax.plot(t, m, ls="none", marker="o", ms=1.7, mfc=BLUE, mec="none",
            alpha=0.30, zorder=4)
    tt = np.logspace(-3, np.log10(T_END), 200)
    ax.plot(tt, completeness(tt), color=GOLD, lw=1.4, zorder=6)
    ax.plot([largest[0]], [largest[1]], marker="*", ms=9.5, mfc=RED,
            mec="white", mew=0.8, zorder=8)
    ax.annotate("largest aftershock Mw {:.1f}, Bath difference {:.1f} Mw"
                .format(largest[1], M_MAIN - largest[1]),
                xy=largest, xytext=(4.0e-3, 6.55), fontsize=6.1, color=INK,
                ha="left", va="center", linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7,
                                shrinkA=2, shrinkB=5), zorder=9)
    ax.text(4.5e-3, 1.62, "completeness Mc(t)", fontsize=6.1, color=GOLD,
            ha="left", va="bottom", fontweight="bold", zorder=7)
    ax.set_xscale("log")
    ax.set_xlim(2.5e-3, T_END * 1.6)
    ax.set_ylim(1.4, 7.0)
    ax.set_xlabel("time since mainshock (days, log scale)", fontsize=6.9)
    ax.set_ylabel("magnitude (Mw)", fontsize=6.9)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.45, alpha=0.32)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Small events only appear once Mc(t) falls", fontsize=8.0,
                 fontweight="bold", color=INK, pad=3.0)


def cumulative_panel(ax, sets):
    for s in sets:
        f = s["fit"]
        sel = np.sort(s["t"][(s["t"] >= f["t0"]) & (s["t"] <= f["t1"])])
        ax.plot(sel, np.arange(1, sel.size + 1), color=s["color"], lw=1.4,
                zorder=6, label=s["name"])
        tt = np.logspace(np.log10(f["t0"]), np.log10(f["t1"]), 220)
        ax.plot(tt, omori_cumulative(f, tt), color=GOLD, lw=1.0,
                ls=(0, (3.4, 2.2)), zorder=7)
    ax.set_xscale("log")
    ax.set_xlim(6e-3, T_END * 1.6)
    ax.set_xlabel("time since mainshock (days, log scale)", fontsize=6.9)
    ax.set_ylabel("cumulative events (count)", fontsize=6.9)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.45, alpha=0.32)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=GOLD, lw=1.0, ls=(0, (3.4, 2.2))))
    labels.append("integral of the fit")
    ax.legend(handles, labels, loc="upper left", fontsize=6.1, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.5, labelspacing=0.22, borderaxespad=0.3)
    ax.set_title("Cumulative count against the fitted integral",
                 fontsize=8.0, fontweight="bold", color=INK, pad=3.0)


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
    rng = np.random.default_rng(51099204)

    t_all, m_all = build_catalogue(rng)
    edges = log_bins(2.5e-3, T_END)

    sets = []
    for spec in CUTS:
        sel = m_all >= spec["mcut"] - 1e-9
        t = t_all[sel]
        tc, rate, _ = binned_rate(t, edges)
        t0 = complete_after(spec["mcut"])
        fit = fit_omori(t, t0, T_END)
        sets.append(dict(spec, t=t, tc=tc, rate=rate, fit=fit))

    k_big = int(np.argmax(m_all))
    largest = (float(t_all[k_big]), float(m_all[k_big]))

    assert t_all.size > 900, "catalogue too small to bin"
    assert all(0.8 < s["fit"]["p"] < 1.5 for s in sets), "p far off Omori"
    assert sets[0]["fit"]["t0"] > sets[1]["fit"]["t0"], \
        "the lower cutoff must wait longer for completeness"
    assert largest[1] < M_MAIN, "an aftershock cannot beat the mainshock"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_mag = axes_inches(fig, AUX_T)
    ax_cum = axes_inches(fig, AUX_B)

    main_panel(ax_main, sets)
    magnitude_panel(ax_mag, t_all, m_all, largest)
    cumulative_panel(ax_cum, sets)



    panel_labels_inches(fig, [ax_main, ax_mag, ax_cum], dx=-0.60, dy=0.12)
    savefig(fig, "1051_eqs_omori_aftershocks.png")


if __name__ == "__main__":
    main()

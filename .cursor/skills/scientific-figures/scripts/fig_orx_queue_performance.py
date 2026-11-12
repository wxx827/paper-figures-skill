"""M/M/c waiting time and queue length blowing up as utilisation reaches one.

All curves are the closed-form M/M/c results evaluated on the same offered
load: the Erlang C delay probability drives the mean wait, Little's law turns
that into a queue length, and the service-level panel uses the exponential
waiting-time tail of the same model.  Nothing is fitted or sketched, so the
knee near rho = 1 is the model's own behaviour rather than a drawn flourish.
"""
import textwrap
from math import factorial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches, unclip_legend

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
GREY = "#808080"
RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"
C_COL = {1: BLUE, 2: GREEN, 3: GOLD, 5: PURPLE}

W, H = 10.4, 6.4
SEED = 20260813
MU = 6.0                 # service rate per server, requests per hour
SERVERS = [1, 2, 3, 5]
RHO_REF = 0.80           # operating reference point
RHO_HI = 0.95            # stressed reference point
THRESH_MIN = 10.0        # service-level threshold, minutes
RHO_SL = 0.85            # utilisation used for the service-level panel

FOOT = ("Curves are the closed-form M/M/c results on a common service rate of "
        "6 requests per hour per server: the Erlang C delay probability sets "
        "the mean wait, Little's law converts it to a queue length, and the "
        "service-level panel evaluates the exponential waiting-time tail of "
        "the same model. The knee near rho = 1 is the model's own behaviour, "
        "not a drawn flourish.")


def erlang_c(c, rho):
    """Probability that an arrival has to wait in an M/M/c queue."""
    a = c * rho
    s = sum(a ** k / factorial(k) for k in range(c))
    tail = a ** c / (factorial(c) * (1.0 - rho))
    return tail / (s + tail)


def metrics(c, rho):
    """Mean wait (minutes) and mean queue length (requests)."""
    lam = c * MU * rho
    wq_h = erlang_c(c, rho) / (c * MU - lam)
    return wq_h * 60.0, lam * wq_h


def tail_prob(c, rho, t_min):
    """P(wait > t) for M/M/c: Erlang C times an exponential decay."""
    lam = c * MU * rho
    return erlang_c(c, rho) * np.exp(-(c * MU - lam) * (t_min / 60.0))


def draw_main(ax, rho):
    axr = ax.twinx()
    for c in SERVERS:
        wq = np.array([metrics(c, r)[0] for r in rho])
        lq = np.array([metrics(c, r)[1] for r in rho])
        ax.plot(rho, wq, color=C_COL[c], lw=1.6, zorder=4)
        axr.plot(rho, lq, color=C_COL[c], lw=1.0, ls=(0, (3.4, 2.0)),
                 alpha=0.85, zorder=3)

    for a in (ax, axr):
        a.set_yscale("log")
        a.set_xlim(rho[0], 1.045)
    ax.set_ylim(0.02, 400.0)
    axr.set_ylim(0.005, 80.0)

    ax.axvline(1.0, color=GREY, lw=1.0, ls=(0, (2.6, 2.0)), zorder=2)
    ax.text(1.014, 0.68, r"$\rho \rightarrow 1$ asymptote",
            transform=ax.get_xaxis_transform(), ha="center", va="center",
            fontsize=6.4, color=GREY, rotation=90.0)

    ax.axvline(RHO_REF, color=RED, lw=1.1, ls=(0, (4, 2.2)), zorder=5)
    lab = "  ".join(f"c={c}: {metrics(c, RHO_REF)[0]:.1f} min"
                    for c in SERVERS)
    ax.text(RHO_REF - 0.010, 0.50, f"operating point $\\rho$ = {RHO_REF:g}",
            transform=ax.get_xaxis_transform(), ha="center", va="center",
            fontsize=6.6, color=RED, rotation=90.0)
    for c in SERVERS:
        w = metrics(c, RHO_REF)[0]
        ax.plot([RHO_REF], [w], marker="o", ms=4.4, mfc=C_COL[c],
                mec="white", mew=0.8, zorder=7)
    ax.text(0.018, 0.975, f"At $\\rho$ = {RHO_REF:g}:  {lab}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
            color=INK,
            bbox=dict(boxstyle="round,pad=0.36", facecolor="white",
                      edgecolor=HAIR, linewidth=0.6, alpha=0.93))

    ax.set_xlabel(r"server utilisation $\rho = \lambda / (c\mu)$ "
                  "(dimensionless)", fontsize=8.0)
    ax.set_ylabel("mean waiting time $W_q$ (min)", fontsize=8.0)
    axr.set_ylabel("mean queue length $L_q$ (requests)", fontsize=8.0)
    ax.set_title("Wait and queue length diverge as the servers saturate",
                 fontsize=9.0, fontweight="bold")
    ax.tick_params(labelsize=7.2)
    axr.tick_params(labelsize=7.2)
    ax.spines["top"].set_visible(False)
    axr.spines["top"].set_visible(False)

    h_c = [Line2D([], [], color=C_COL[c], lw=1.6, label=f"c = {c} servers")
           for c in SERVERS]
    h_m = [Line2D([], [], color=RULE, lw=1.6, label=r"$W_q$ (left axis)"),
           Line2D([], [], color=RULE, lw=1.0, ls=(0, (3.4, 2.0)),
                  label=r"$L_q$ (right axis)")]
    leg1 = ax.legend(handles=h_c, loc="lower right",
                     bbox_to_anchor=(0.985, 0.020), fontsize=6.6,
                     frameon=False, handlelength=1.7, labelspacing=0.34)
    leg2 = ax.legend(handles=h_m, loc="lower right",
                     bbox_to_anchor=(0.985, 0.245), fontsize=6.6,
                     frameon=False, handlelength=1.7, labelspacing=0.34)
    ax.add_artist(leg1)
    unclip_legend(leg1)
    unclip_legend(leg2)


def draw_distributions(ax, rng):
    c = 2
    lam = c * MU * RHO_SL
    mean_gap = 60.0 / lam            # minutes between arrivals
    mean_svc = 60.0 / MU             # minutes of service
    gaps = rng.exponential(mean_gap, 4000)
    svc = rng.exponential(mean_svc, 4000)
    hi = float(np.percentile(np.concatenate([gaps, svc]), 99))
    bins = np.linspace(0.0, hi, 34)
    for data, mean, col, lab in ((gaps, mean_gap, BLUE, "interarrival gap"),
                                 (svc, mean_svc, GOLD, "service time")):
        ax.hist(data, bins=bins, density=True, color=lighten(col, 0.62),
                edgecolor="white", linewidth=0.35, zorder=2)
        x = np.linspace(0.0, hi, 240)
        ax.plot(x, np.exp(-x / mean) / mean, color=col, lw=1.4, zorder=4,
                label=f"{lab} (mean {mean:.1f} min)")
        ax.axvline(mean, color=col, lw=0.8, ls=(0, (3, 2.2)), zorder=3)
    ax.set_xlim(0.0, hi)
    ax.set_xlabel("elapsed time (min)", fontsize=7.2)
    ax.set_ylabel("density (1/min)", fontsize=7.2)
    ax.set_title(f"Both exponential at $\\rho$ = {RHO_SL:g}, c = {c:d}",
                 fontsize=8.5, fontweight="bold")
    ax.tick_params(labelsize=6.5)
    ax.legend(loc="upper right", fontsize=6.3, frameon=False,
              handlelength=1.4, labelspacing=0.3)
    style_axes(ax)


def draw_service_level(ax):
    cs = list(SERVERS)
    x = np.arange(len(cs))
    p = np.array([100.0 * tail_prob(c, RHO_SL, THRESH_MIN) for c in cs])
    cols = [C_COL[c] for c in cs]
    ax.bar(x, p, width=0.56, color=[lighten(c, 0.42) for c in cols],
           edgecolor=cols, linewidth=0.9, zorder=3)
    for xx, v in zip(x, p):
        ax.text(xx, v + 1.2, f"{v:.1f}", ha="center", va="bottom",
                fontsize=6.2, color=SUB)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c:d}" for c in cs])
    ax.set_ylim(0.0, p.max() * 1.22)
    ax.set_xlabel("servers $c$ (count)", fontsize=7.2)
    ax.set_ylabel(f"P(wait > {THRESH_MIN:.0f} min) (%)", fontsize=7.2)
    ax.set_title(f"Pooling cuts breaches {p[0] / p[-1]:.1f}-fold at the same "
                 f"$\\rho$", fontsize=8.5, fontweight="bold")
    ax.tick_params(labelsize=6.5)
    style_axes(ax)


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)
    rho = np.linspace(0.30, 0.985, 400)

    fig = plt.figure(figsize=(W, H))
    ax_main = axes_inches(fig, (0.86, 1.16, 5.06, 4.24))
    ax_top = axes_inches(fig, (7.16, 3.56, 2.86, 1.84))
    ax_bot = axes_inches(fig, (7.16, 1.16, 2.86, 1.84))

    draw_main(ax_main, rho)
    draw_distributions(ax_top, rng)
    draw_service_level(ax_bot)

    w_ref = metrics(2, RHO_REF)[0]
    w_hi = metrics(2, RHO_HI)[0]
    panel_labels_inches(fig, [ax_main, ax_top, ax_bot], dx=-0.62, dy=0.08)

    savefig(fig, "863_orx_queue_performance.png")


if __name__ == "__main__":
    main()

"""Weibull life-data analysis: probability plot, reliability and hazard rate
for three production lots sharing one colour key."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes

NAVY = "#2c4a6e"
BLUE = "#4c78a8"
AMBER = "#c9922f"
GREY = "#8b949e"
ALERT = "#b23a48"

LOTS = [
    ("Lot A", NAVY, 1.15, 2400.0, "random"),
    ("Lot B", BLUE, 2.60, 1800.0, "wear-out"),
    ("Lot C", AMBER, 0.75, 3200.0, "infant mortality"),
]
N = 30
T_MAX = 4000.0
T_LOG = (150.0, 12000.0)

F_TICKS = [1, 2, 5, 10, 25, 50, 90, 99]


def gfun(f):
    return np.log(-np.log(1.0 - np.asarray(f, dtype=float)))


def stamp_labels(fig, axes, labels, dx=0.058, dy=0.016, size=9):
    for ax, lab in zip(axes, labels):
        bb = ax.get_position()
        fig.text(bb.x0 - dx, bb.y1 + dy, lab, fontsize=size,
                 fontweight="bold", ha="left", va="bottom")


def sample_lot(rng, beta, eta):
    """Median-rank quantiles with lognormal jitter: natural-looking scatter
    that still recovers the nominal shape parameter."""
    ranks = (np.arange(1, N + 1) - 0.3) / (N + 0.4)
    t = eta * (-np.log(1.0 - ranks)) ** (1.0 / beta)
    t = np.sort(t * np.exp(rng.normal(0.0, 0.30 / beta, N)))
    return t, ranks


def fit_weibull(t, f):
    """Rank regression on X (the standard RRX fit) -> (beta_hat, eta_hat)."""
    xs, ys = np.log(t), gfun(f)
    m, c = np.polyfit(ys, xs, 1)
    return 1.0 / m, float(np.exp(c))


def panel_prob(ax, fits):
    for (name, col, _, _, _), (t, f, bhat, ehat) in zip(LOTS, fits):
        ax.plot(t, gfun(f), "o", ms=3.0, mfc="white", mec=col, mew=0.9,
                zorder=4)
        xs = np.array([t.min() * 0.55, t.max() * 1.7])
        ax.plot(xs, bhat * (np.log(xs) - np.log(ehat)), color=col, lw=1.3,
                zorder=3)
    ax.axhline(0.0, color=GREY, lw=0.8, ls=(0, (4, 2.5)), zorder=2)
    ax.text(T_LOG[0] * 1.12, 0.12, "F = 63.2%  ($t = \\eta$)", fontsize=6.1,
            color=GREY, ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(*T_LOG)
    ax.set_xticks([200, 500, 1000, 2000, 5000, 10000])
    ax.set_xticklabels(["200", "500", "1k", "2k", "5k", "10k"], fontsize=7)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylim(gfun(0.008), gfun(0.992))
    ax.set_yticks(gfun(np.array(F_TICKS) / 100.0))
    ax.set_yticklabels([str(v) for v in F_TICKS], fontsize=7)
    ax.set_ylabel("Unreliability  $F(t)$  (%)", fontsize=8)
    ax.set_xlabel("Time to failure (hours, log scale)", fontsize=8)
    ax.grid(True, which="both", color="#e8ecf0", lw=0.5, zorder=0)
    ax.tick_params(labelsize=7)
    style_axes(ax)


def panel_reliability(ax, fits):
    t = np.linspace(0, T_MAX, 400)
    for (name, col, _, _, _), (_, _, bhat, ehat) in zip(LOTS, fits):
        r = np.exp(-((t / ehat) ** bhat))
        lo = np.exp(-((t / (ehat * 0.93)) ** (bhat * 1.10)))
        hi = np.exp(-((t / (ehat * 1.07)) ** (bhat * 0.90)))
        ax.fill_between(t, np.minimum(lo, hi), np.maximum(lo, hi), color=col,
                        alpha=0.14, lw=0, zorder=2)
        ax.plot(t, r, color=col, lw=1.5, zorder=4)
    ax.axhline(0.90, color=ALERT, lw=0.9, ls=(0, (5, 2.5)), zorder=3)
    ax.text(T_MAX * 0.985, 0.915, "$B_{10}$ life", ha="right", va="bottom",
            fontsize=6.3, fontweight="bold", color=ALERT)
    for (_, col, _, _, _), (_, _, bhat, ehat) in zip(LOTS, fits):
        b10 = ehat * (-np.log(0.90)) ** (1.0 / bhat)
        if b10 < T_MAX:
            ax.plot([b10], [0.90], "o", ms=3.6, mfc="white", mec=col, mew=1.2,
                    zorder=5)
    ax.set_xlim(0, T_MAX)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Operating time (hours)", fontsize=8)
    ax.set_ylabel("Reliability  $R(t)$", fontsize=8)
    ax.grid(True, color="#eef1f4", lw=0.5, zorder=0)
    ax.tick_params(labelsize=7)
    style_axes(ax)


def panel_hazard(ax, fits):
    t = np.linspace(40, T_MAX, 400)
    notes = ("$\\beta \\approx 1$  constant", "$\\beta > 1$  wear-out",
             "$\\beta < 1$  early life")
    offs = (12, 11, -12)
    for (_, col, _, _, _), (_, _, bhat, ehat), note, dy in zip(
            LOTS, fits, notes, offs):
        h = (bhat / ehat) * (t / ehat) ** (bhat - 1.0) * 1000.0
        ax.plot(t, h, color=col, lw=1.5, zorder=4)
        xa = T_MAX * 0.62
        ya = (bhat / ehat) * (xa / ehat) ** (bhat - 1.0) * 1000.0
        ax.annotate(note, (xa, ya), xytext=(0, dy),
                    textcoords="offset points", ha="center",
                    va="bottom" if dy > 0 else "top", fontsize=6.2, color=col,
                    zorder=5)
    ax.set_xlim(0, T_MAX)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 14.0)
    ax.set_xlabel("Operating time (hours)", fontsize=8)
    ax.set_ylabel("Hazard rate  $h(t)$  (fails / 1000 h)", fontsize=8)
    ax.grid(True, which="both", color="#eef1f4", lw=0.5, zorder=0)
    ax.tick_params(labelsize=7)
    style_axes(ax)


def main():
    set_pub_style()
    rng = np.random.default_rng(9317)
    fits = []
    for _, _, beta, eta, _ in LOTS:
        t, f = sample_lot(rng, beta, eta)
        bhat, ehat = fit_weibull(t, f)
        fits.append((t, f, bhat, ehat))

    fig = plt.figure(figsize=(7.8, 3.5))
    gs = fig.add_gridspec(1, 3, wspace=0.34, left=0.068, right=0.988,
                          top=0.735, bottom=0.155)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    panel_prob(axes[0], fits)
    panel_reliability(axes[1], fits)
    panel_hazard(axes[2], fits)
    stamp_labels(fig, axes, ["(a)", "(b)", "(c)"])

    handles = [
        Line2D([], [], color=col, marker="o", ms=3.0, mfc="white", mec=col,
               lw=1.4,
               label=f"{name}  \u00b7  $\\beta$ = {bh:.2f}  \u00b7  "
                     f"$\\eta$ = {eh:,.0f} h  \u00b7  {mode}")
        for (name, col, _, _, mode), (_, _, bh, eh) in zip(LOTS, fits)
    ]


    savefig(fig, "247_weibull_reliability.png")


if __name__ == "__main__":
    main()

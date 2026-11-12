"""Thirty-hour fed-batch fermentation, read on one shared time axis.

The three stacked panels are the run itself: biomass, the substrate/product
pair, and the control loops that keep the culture alive.  They share a single
time axis and a single set of phase bands, so an event in one panel can be
lined up with its consequence in another - the feed start visibly shows up as
a dissolved-oxygen dip and a stirrer ramp.  The two smaller panels on the
right are derived quantities: the specific growth rate is differentiated from
the biomass curve that is plotted above it, and the cumulative yields come
from the same carbon bookkeeping that generates the substrate trace, so no
number on this figure is written by hand.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_label_inches

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"
GREY = "#808080"
RED = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"

W, H = 10.4, 6.4
T_END, T_FEED, T_IND, T_STAT = 30.0, 10.0, 12.0, 24.0
X0, S0 = 0.35, 20.0                 # g/L dry cell weight, g/L glucose
Y_XS, Y_PS, MAINT = 0.45, 0.36, 0.022
P_MAX, P_TAU = 15.0, 6.0
OD_K = 0.38                         # nominal g DCW per OD unit
PHASES = [(0.0, T_FEED, "batch", None),
          (T_FEED, T_STAT, "fed-batch", "#f0f0f0"),
          (T_STAT, T_END, "stationary", "#f5f5f5")]


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def run(seed=21):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T_END, 900)

    s1, s2 = sig((t - T_FEED) / 0.6), sig((t - T_STAT) / 0.8)
    mu = 0.32 * (1.0 - s1) + 0.12 * s1 * (1.0 - s2) + 0.012 * s2
    x = X0 * np.exp(np.concatenate([[0.0], np.cumsum(
        0.5 * (mu[1:] + mu[:-1]) * np.diff(t))]))

    p = P_MAX * (1.0 - np.exp(-np.clip(t - T_IND, 0.0, None) / P_TAU))
    maint = MAINT * np.concatenate([[0.0], np.cumsum(
        0.5 * (x[1:] + x[:-1]) * np.diff(t))])
    cons = (x - X0) / Y_XS + p / Y_PS + maint

    s = np.clip(S0 - cons, 0.30, None)
    s = s + 0.55 * np.exp(-((t - T_FEED - 0.6) / 0.8) ** 2)
    s = s + rng.normal(0.0, 0.06, t.size)

    od = x / (OD_K - 0.05 * sig((t - 14.0) / 4.0))
    do = 100.0 - 70.0 * sig((t - 3.6) / 1.1)
    do = do - 9.0 * np.exp(-((t - T_FEED - 0.2) / 0.5) ** 2)
    do = do + 7.0 * np.exp(-((t - T_STAT - 0.6) / 1.1) ** 2)
    do = do + rng.normal(0.0, 0.7, t.size)
    ph = (7.0 - 0.24 * np.exp(-((t - 8.2) / 2.2) ** 2)
          + 0.09 * np.exp(-((t - 25.5) / 3.0) ** 2)
          + rng.normal(0.0, 0.012, t.size))
    rpm = (400.0 + 480.0 * sig((t - 6.5) / 3.0)
           + 40.0 * sig((t - T_FEED) / 0.5) + rng.normal(0.0, 4.0, t.size))
    return dict(t=t, mu=mu, x=x, od=od, s=s, p=p, cons=cons, do=do, ph=ph,
                rpm=rpm)


def bands(ax, label=False, y=1.03):
    for t0, t1, name, col in PHASES:
        if col:
            ax.axvspan(t0, t1, facecolor=col, edgecolor="none", zorder=0)
        if label:
            ax.text(0.5 * (t0 + t1), y, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=6.8, color=SUB)
    for tv, col in [(T_FEED, RED), (T_IND, GREY)]:
        ax.axvline(tv, color=col, lw=0.9, ls=(0, (3.2, 2.2)), zorder=2)


def biomass_panel(ax, d):
    bands(ax, label=True)
    ax.plot(d["t"], d["od"], color=BLUE, lw=1.7, zorder=4)
    ax.set_ylabel(r"OD$_{600}$ (-)", color=BLUE)
    ax.set_ylim(0.0, 150.0)
    ax.tick_params(axis="y", colors=BLUE)
    ax.spines["left"].set_color(BLUE)
    ax.grid(axis="y", lw=0.5, alpha=0.7)

    ax2 = ax.twinx()
    ax2.plot(d["t"], d["x"], color=INK, lw=1.4, ls=(0, (4.5, 2.2)), zorder=4)
    ax2.set_ylabel("dry cell weight (g/L)")
    ax2.set_ylim(0.0, 150.0 * OD_K)
    ax2.spines["right"].set_visible(True)
    ax.text(0.6, 132.0, r"OD$_{600}$", ha="left", va="center", fontsize=7.0,
            color=BLUE)
    ax.text(T_END - 0.6, 96.0, "dry cell weight", ha="right", va="center",
            fontsize=7.0, color=INK)
    ax.text(T_FEED - 0.35, 62.0, "feed start", ha="right", va="center",
            rotation=90.0, fontsize=6.6, color=RED)
    ax.text(T_IND + 0.35, 62.0, "induction", ha="left", va="center",
            rotation=90.0, fontsize=6.6, color=SUB)


def substrate_panel(ax, d):
    bands(ax)
    ax.plot(d["t"], d["s"], color=GOLD, lw=1.7, zorder=4)
    ax.set_ylabel("glucose (g/L)", color=GOLD)
    ax.set_ylim(0.0, 22.0)
    ax.tick_params(axis="y", colors=GOLD)
    ax.spines["left"].set_color(GOLD)
    ax.grid(axis="y", lw=0.5, alpha=0.7)

    ax2 = ax.twinx()
    ax2.plot(d["t"], d["p"], color=GREEN, lw=1.7, zorder=4)
    ax2.set_ylabel("product titre (g/L)", color=GREEN)
    ax2.set_ylim(0.0, 18.0)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(GREEN)
    ax2.tick_params(axis="y", colors=GREEN)

    ax.text(1.0, 17.0, "glucose", ha="left", va="center", fontsize=7.0,
            color=GOLD)
    ax.text(T_END - 0.6, 15.6, "product", ha="right", va="center",
            fontsize=7.0, color=GREEN)
    ax.annotate("glucose limitation\nholds at "
                f"{d['s'][d['t'] > 15].mean():.1f} g/L",
                xy=(17.0, d["s"][np.argmin(abs(d["t"] - 17.0))]),
                xytext=(15.0, 7.2), ha="center", fontsize=6.4, color=MUTED,
                linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))


def control_panel(ax, d):
    bands(ax)
    ax.plot(d["t"], d["do"], color=BLUE, lw=1.4, zorder=4)
    ax.set_ylabel("DO (%)", color=BLUE)
    ax.set_ylim(0.0, 110.0)
    ax.set_xlabel("process time (h)")
    ax.tick_params(axis="y", colors=BLUE)
    ax.spines["left"].set_color(BLUE)
    ax.axhline(30.0, color=BLUE, lw=0.7, ls=(0, (2.2, 2.0)), zorder=2)
    ax.text(0.6, 34.0, "DO setpoint 30 %", ha="left", va="bottom",
            fontsize=6.4, color=BLUE)

    ax2 = ax.twinx()
    ax2.plot(d["t"], d["ph"], color=PURPLE, lw=1.4, zorder=4)
    ax2.set_ylabel("pH (-)", color=PURPLE)
    ax2.set_ylim(6.4, 7.4)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(PURPLE)
    ax2.tick_params(axis="y", colors=PURPLE)

    ax3 = ax.twinx()
    ax3.plot(d["t"], d["rpm"], color=GREY, lw=1.3, ls=(0, (5.0, 2.2)),
             zorder=3)
    ax3.set_ylabel("agitation (rpm)", color=GREY)
    ax3.set_ylim(300.0, 1050.0)
    ax3.spines["right"].set_visible(True)
    ax3.spines["right"].set_position(("outward", 30))
    ax3.spines["right"].set_color(GREY)
    ax3.tick_params(axis="y", colors=GREY)

    ax.text(T_END - 0.6, 79.0, "agitation", ha="right", va="center",
            fontsize=7.0, color=GREY)
    ax2.text(17.5, 7.13, "pH", ha="center", va="bottom", fontsize=7.0,
             color=PURPLE)


def growth_panel(ax, d):
    for t0, t1, _, col in PHASES:
        if col:
            ax.axvspan(t0, t1, facecolor=col, edgecolor="none", zorder=0)
    ax.axvline(T_FEED, color=RED, lw=0.9, ls=(0, (3.2, 2.2)), zorder=2)
    mu = np.gradient(np.log(d["x"]), d["t"])
    ax.plot(d["t"], mu, color=BLUE, lw=1.6, zorder=4)
    ax.set_xlabel("process time (h)", fontsize=7.4)
    ax.set_ylabel(r"$\mu$ (1/h)", fontsize=7.4)
    ax.set_xlim(0.0, T_END)
    ax.set_ylim(0.0, 0.40)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.annotate(f"$\\mu_{{max}}$ = {mu.max():.2f} 1/h", xy=(4.0, mu.max()),
                xytext=(13.0, 0.33), ha="left", fontsize=6.6, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.text(T_STAT + 0.4, 0.055, "growth arrest", ha="left", va="bottom",
            fontsize=6.4, color=MUTED)
    ax.set_title(r"Specific growth rate drops at the feed switch",
                 fontsize=8.2, fontweight="bold", color=INK)
    return mu


def yield_panel(ax, d):
    for t0, t1, _, col in PHASES:
        if col:
            ax.axvspan(t0, t1, facecolor=col, edgecolor="none", zorder=0)
    ax.axvline(T_FEED, color=RED, lw=0.9, ls=(0, (3.2, 2.2)), zorder=2)
    m = d["cons"] > 0.5
    yxs = (d["x"][m] - X0) / d["cons"][m]
    yps = d["p"][m] / d["cons"][m]
    ax.plot(d["t"][m], yxs, color=BLUE, lw=1.6, zorder=4, label=r"$Y_{X/S}$")
    ax.plot(d["t"][m], yps, color=GREEN, lw=1.6, zorder=4, label=r"$Y_{P/S}$")
    ax.set_xlabel("process time (h)", fontsize=7.4)
    ax.set_ylabel("cumulative yield (g/g)", fontsize=7.4)
    ax.set_xlim(0.0, T_END)
    ax.set_ylim(0.0, 0.52)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.8, frameon=False, loc="upper right", ncol=2,
              handlelength=1.3, borderaxespad=0.2)
    ax.text(T_END - 0.5, yxs[-1] - 0.045, f"{yxs[-1]:.2f} g/g", ha="right",
            va="top", fontsize=6.4, color=BLUE)
    ax.text(T_END - 0.5, yps[-1] + 0.030, f"{yps[-1]:.2f} g/g", ha="right",
            va="bottom", fontsize=6.4, color=GREEN)
    ax.set_title("Carbon is traded from biomass to product", fontsize=8.2,
                 fontweight="bold", color=INK)
    return yxs, yps


def main():
    set_pub_style()
    d = run()

    fig = plt.figure(figsize=(W, H))
    xl, rw, rh, gap = 0.80, 4.95, 1.42, 0.11
    ax_bio = axes_inches(fig, (xl, 0.85 + 2 * (rh + gap), rw, rh))
    ax_sub = axes_inches(fig, (xl, 0.85 + rh + gap, rw, rh))
    ax_ctl = axes_inches(fig, (xl, 0.85, rw, rh))
    ax_mu = axes_inches(fig, (7.22, 3.38, 2.86, 1.91))
    ax_y = axes_inches(fig, (7.22, 0.85, 2.86, 1.91))

    for ax in (ax_bio, ax_sub, ax_ctl):
        ax.set_xlim(0.0, T_END)
        ax.set_xticks(np.arange(0.0, T_END + 1.0, 5.0))
    for ax in (ax_bio, ax_sub):
        ax.set_xticklabels([])

    biomass_panel(ax_bio, d)
    substrate_panel(ax_sub, d)
    control_panel(ax_ctl, d)
    mu = growth_panel(ax_mu, d)
    yxs, yps = yield_panel(ax_y, d)

    fig.text(0.021, 6.11 / H,
             "Switching to an exponential feed trades growth for product: "
             f"{d['p'][-1]:.1f} g/L titre at {d['x'][-1]:.0f} g/L biomass",
             ha="left", va="center", fontsize=11.0, fontweight="bold",
             color=INK)
    fig.text(0.021, 0.11 / H,
             f"Feed starts at {T_FEED:.0f} h and induction at {T_IND:.0f} h "
             f"(dashed rules). Final cumulative yields are "
             f"{yxs[-1]:.2f} g biomass and {yps[-1]:.2f} g product per gram "
             f"of glucose consumed, at $\\mu_{{max}}$ = {mu.max():.2f} 1/h.",
             ha="left", va="bottom", fontsize=6.5, color=MUTED)

    for ax, lab in [(ax_bio, "(a)"), (ax_sub, "(b)"), (ax_ctl, "(c)"),
                    (ax_mu, "(d)"), (ax_y, "(e)")]:
        panel_label_inches(fig, ax, lab, dx=-0.62, dy=0.02)

    savefig(fig, "711_bio_fermentation_profile.png")


if __name__ == "__main__":
    main()

"""Fed-batch process profile for three feeding strategies.

The left column is one run: an exponential feed started once the initial
substrate is gone, the broth volume it produces, and the dissolved-oxygen and
pH traces that follow from the same integration - oxygen falls because the
feed drives the uptake rate towards the transfer capacity, and pH saws because
base is pulsed in whenever the acid produced by growth crosses the dead band.
The right column re-integrates the identical model under constant,
exponential and DO-stat feeding, so the titer bars and the acetate curves are
two views of the same three solutions rather than two independent claims.
Overflow acetate is driven by residual glucose, which is why the two
demand-matched feeds avoid it and the constant feed does not.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

GREEN = "#2e7d5b"        # biomass
OCHRE = "#c0603f"        # product
BLUE = "#3b5b92"         # broth volume
STEEL = "#3d6b8e"        # dissolved oxygen
GOLD = "#c9a227"         # pH
GREY = "#808080"         # constant feed
RED = "#b23a48"          # exponential feed
PURPLE = "#8e6bbf"       # DO-stat feed
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

MU_MAX = 0.45            # h^-1
KS = 0.15                # g/L
Y_XS = 0.45              # g/g
Y_PS = 0.42              # g/g
Y_AS = 0.60              # g/g
Y_XO = 0.90              # g biomass / g oxygen
M_O = 0.008              # g O2 / (g h)
MS = 0.015               # g/(g h)
ALPHA, BETA = 0.10, 0.050            # Luedeking-Piret, g/g and g/(g h)
K_AI = 1.5               # acetate inhibition constant for product, g/L
K_ACE, K_SA = 0.30, 1.5  # overflow acetate: g/(g h) and g/L
K_ACO = 0.055            # acetate re-assimilation, g/(g h)
KLA0, KLA1 = 260.0, 1000.0           # h^-1, agitation cascade over 8 h
C_SAT = 0.0072           # g/L, oxygen solubility at 30 C
DO_SET = 0.90            # DO-stat holds uptake at 90 % of the capacity

V0, V_MAX = 3.0, 4.4     # L
S0, X0 = 6.0, 0.5        # g/L
S_FEED = 350.0           # g/L glucose in the feed
T_FEED, T_END = 4.5, 24.0
MU_SET = 0.14            # h^-1, exponential feed design rate

FIG_W, FIG_H = 11.0, 6.5
TOP_L = (0.88, 3.65, 5.30, 1.85)
MID_L = (0.88, 2.30, 5.30, 1.05)
BOT_L = (0.88, 0.95, 5.30, 1.05)
TOP_R = (7.62, 3.61, 2.96, 1.89)
BOT_R = (7.62, 0.95, 2.96, 1.89)

STRATEGIES = [("constant", GREY, "constant feed"),
              ("exponential", RED, "exponential feed"),
              ("do_stat", PURPLE, "DO-stat feed")]


def otr_max(t):
    """Oxygen transfer capacity; the agitation cascade ramps kLa after feed."""
    r = float(np.clip((t - T_FEED) / 8.0, 0.0, 1.0))
    return (KLA0 + (KLA1 - KLA0) * r) * C_SAT


def growth_rate(t, x, s):
    """Monod rate capped by what the oxygen transfer capacity can support.

    The cap is a p-norm soft minimum rather than ``min``: a hard switch makes
    the right-hand side non-differentiable and the stiff solver grinds to a
    halt hunting for the corner.
    """
    s = max(s, 0.0)
    mu_s = MU_MAX * s / (KS + s)
    mu_o = max(Y_XO * (otr_max(t) / max(x, 1e-9) - M_O), 1e-6)
    soft = mu_s / (1.0 + (mu_s / mu_o) ** 8.0) ** 0.125
    return float(np.clip(soft, 0.0, MU_MAX))


def gate(z, w):
    """Smooth 0 -> 1 switch; a hard cut-off makes the solver chatter on it."""
    return 0.5 * (1.0 + np.tanh(z / w))


def feed_rate(t, y, strategy, f0, f_const):
    x, _s, _p, _a, v = y
    g = gate(t - T_FEED, 0.10) * gate(V_MAX - v, 0.06)
    if g < 1e-7:
        return 0.0
    if strategy == "constant":
        base = f_const
    elif strategy == "exponential":
        base = f0 * np.exp(MU_SET * (t - T_FEED))
    else:
        mu_o = max(Y_XO * (DO_SET * otr_max(t) / max(x, 1e-9) - M_O), 0.0)
        mu_o = mu_o / (1.0 + (mu_o / MU_MAX) ** 8.0) ** 0.125
        base = mu_o * x * v / (Y_XS * S_FEED)
    return float(base * g)


def make_rhs(strategy, f0, f_const):
    def rhs(t, y):
        x, s, p, a, v = y
        s, a = max(s, 0.0), max(a, 0.0)
        mu = growth_rate(t, x, s)
        qp = (ALPHA * mu + BETA * s / (KS + s)) / (1.0 + a / K_AI)
        qa = K_ACE * s / (K_SA + s)
        qac = K_ACO * (a / (0.5 + a)) * (KS / (KS + s))
        qs = mu / Y_XS + qp / Y_PS + qa / Y_AS + MS * s / (KS + s)
        f = feed_rate(t, y, strategy, f0, f_const)
        d = f / max(v, 1e-9)
        return [mu * x - d * x,
                -qs * x + d * (S_FEED - s),
                qp * x - d * p,
                (qa - qac) * x - d * a,
                f]
    return rhs


def integrate(strategy, f0, f_const, n=1601):
    t = np.linspace(0.0, T_END, n)
    sol = solve_ivp(make_rhs(strategy, f0, f_const), (0.0, T_END),
                    [X0, S0, 0.0, 0.0, V0], t_eval=t, method="LSODA",
                    rtol=1e-6, atol=1e-9, max_step=0.20)
    x, s, p, a, v = sol.y
    s, a = np.maximum(s, 0.0), np.maximum(a, 0.0)
    mu = np.array([growth_rate(ti, xi, si)
                   for ti, xi, si in zip(sol.t, x, s)])
    f = np.array([feed_rate(ti, yi, strategy, f0, f_const)
                  for ti, yi in zip(sol.t, sol.y.T)])
    cap = np.array([otr_max(ti) for ti in sol.t])
    our = (mu / Y_XO + M_O) * x
    do = np.clip(100.0 * (1.0 - our / cap), 0.0, 100.0)
    return dict(t=sol.t, x=x, s=s, p=p, a=a, v=v, mu=mu, f=f, do=do)


def design_feed():
    """The batch-phase endpoint fixes the exponential feed constant F0."""
    pre = integrate("constant", 0.0, 0.0, n=1201)
    i = int(np.argmin(np.abs(pre["t"] - T_FEED)))
    f0 = MU_SET * pre["x"][i] * pre["v"][i] / (Y_XS * S_FEED)
    f_const = (V_MAX - V0) / (T_END - T_FEED)
    return f0, f_const, float(pre["x"][i])


def ph_trace(t, x, mu, lo=6.96, hi=7.03, gain=0.010):
    """Base pulses fired whenever growth acid drops pH below the dead band."""
    ph = np.empty_like(t)
    cur, pulses = 7.02, []
    for i in range(t.size):
        if i:
            cur -= gain * (mu[i] * x[i] + 0.4) * (t[i] - t[i - 1])
            if cur < lo:
                cur = hi
                pulses.append(t[i])
        ph[i] = cur
    return ph, np.asarray(pulses)


def cap_time(run):
    hit = run["t"][run["v"] >= V_MAX - 1e-6]
    return float(hit[0]) if hit.size else T_END


def top_panel(ax, run):
    ax_v = ax.twinx()
    ax.set_zorder(2)
    ax.patch.set_visible(False)
    ax_v.set_zorder(1)
    ax_v.axvspan(T_FEED, T_END, facecolor=lighten(GREY, 0.94),
                 edgecolor="none", zorder=0)

    ax.plot(run["t"], run["x"], color=GREEN, lw=1.9, zorder=5)
    ax.plot(run["t"], run["p"], color=OCHRE, lw=1.9, zorder=5)
    ax_v.plot(run["t"], run["v"], color=BLUE, lw=1.7, zorder=4)
    ax_v.axhline(V_MAX, color=BLUE, lw=0.9, ls=(0, (3.2, 2.2)), zorder=3)
    ax_v.text(T_END - 0.3, V_MAX + 0.05,
              "working volume limit {:.1f} L".format(V_MAX), ha="right",
              va="bottom", fontsize=6.6, color=BLUE)

    ax.axvline(T_FEED, color=INK, lw=1.0, ls=(0, (4, 2.4)), zorder=6)
    ax.text(T_FEED - 0.25, run["x"].max() * 0.62,
            "feed start\nt = {:.1f} h".format(T_FEED), ha="right", va="center",
            fontsize=6.8, color=INK)
    ax.text(T_FEED + 6.0, run["x"].max() * 0.20, "fed-batch phase",
            ha="center", va="center", fontsize=7.0, color=SUB)

    ax.set_xlim(0.0, T_END)
    ax.set_ylim(0.0, run["x"].max() * 1.24)
    ax_v.set_ylim(V0 - 0.15, V_MAX + 0.40)
    ax.set_ylabel("X and P (g/L)")
    ax_v.set_ylabel("broth volume V (L)", color=BLUE)
    ax_v.tick_params(axis="y", colors=BLUE)
    ax_v.spines["right"].set_color(BLUE)
    ax_v.spines["top"].set_visible(False)
    ax_v.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelbottom=False)
    ax.grid(axis="y", lw=0.5, alpha=0.6)

    handles = [Line2D([], [], color=GREEN, lw=1.9, label="biomass X (g/L)"),
               Line2D([], [], color=OCHRE, lw=1.9, label="product P (g/L)"),
               Line2D([], [], color=BLUE, lw=1.7, label="volume V (L)")]
    ax.legend(handles=handles, loc="upper left", fontsize=7.0, frameon=False,
              handlelength=1.5, borderaxespad=0.3, labelspacing=0.30)
    ax.set_title("Exponential feed: biomass, titer and broth volume",
                 fontsize=9.0, fontweight="bold", color=INK, pad=5.0)


def feed_panel(ax, runs, f0):
    for run, (_k, col, lab) in zip(runs, STRATEGIES):
        if col == RED:
            ax.fill_between(run["t"], 0.0, run["f"], color=lighten(RED, 0.82),
                            linewidth=0.0, zorder=2)
        ax.plot(run["t"], run["f"], color=col, lw=1.5, zorder=3, label=lab)
    ax.axvline(T_FEED, color=INK, lw=1.0, ls=(0, (4, 2.4)), zorder=4)
    top = max(r["f"].max() for r in runs)
    ax.text(T_FEED + 0.5, top * 1.05,
            r"$F(t)=F_0\,\exp(\mu_{set}\,t)$" +
            ",  $F_0$ = {:.4f} L/h,  ".format(f0) +
            r"$\mu_{set}$ = " + "{:.2f}".format(MU_SET) + r" h$^{-1}$",
            ha="left", va="top", fontsize=6.8, color=RED)
    ax.set_xlim(0.0, T_END)
    ax.set_ylim(0.0, top * 1.32)
    ax.set_ylabel("feed rate F\n(L/h)", fontsize=7.6)
    ax.tick_params(axis="y", labelsize=6.8)
    ax.tick_params(labelbottom=False)
    ax.grid(axis="y", lw=0.5, alpha=0.6)
    ax.legend(fontsize=6.4, frameon=False, loc="upper left", ncol=1,
              handlelength=1.4, borderaxespad=0.25, labelspacing=0.22)


def do_ph_panel(ax, run, ph, pulses):
    ax_p = ax.twinx()
    ax.set_zorder(2)
    ax.patch.set_visible(False)
    ax_p.set_zorder(1)

    ax.plot(run["t"], run["do"], color=STEEL, lw=1.7, zorder=5)
    ax_p.plot(run["t"], ph, color=GOLD, lw=1.1, zorder=4)
    ax.axvline(T_FEED, color=INK, lw=1.0, ls=(0, (4, 2.4)), zorder=6)
    i_min = int(np.argmin(run["do"][run["t"] > T_FEED])) + \
        int(np.argmax(run["t"] > T_FEED))
    ax.annotate("DO minimum {:.0f} % sat.".format(run["do"][i_min]),
                xy=(run["t"][i_min], run["do"][i_min]),
                xytext=(run["t"][i_min] - 1.2, 72.0), ha="right", va="center",
                fontsize=6.6, color=STEEL,
                arrowprops=dict(arrowstyle="-", color=STEEL, lw=0.7))
    ax_p.text(T_END - 0.3, 7.052,
              "{:d} base pulses hold pH in the 6.96-7.03 band".format(
                  int(pulses.size)), ha="right", va="center", fontsize=6.6,
              color=GOLD)

    ax.set_xlim(0.0, T_END)
    ax.set_ylim(0.0, 108.0)
    ax.set_yticks([0, 50, 100])
    ax_p.set_ylim(6.93, 7.075)
    ax_p.set_yticks([6.95, 7.00, 7.05])
    ax.set_xlabel("process time (h)")
    ax.set_ylabel("DO\n(% sat.)", color=STEEL, fontsize=7.6)
    ax_p.set_ylabel("pH (-)", color=GOLD, fontsize=7.6)
    ax.tick_params(axis="y", colors=STEEL, labelsize=6.8)
    ax_p.tick_params(axis="y", colors=GOLD, labelsize=6.8)
    ax.spines["left"].set_color(STEEL)
    ax.spines["right"].set_visible(False)
    ax_p.spines["right"].set_color(GOLD)
    ax_p.spines["top"].set_visible(False)
    ax_p.spines["left"].set_visible(False)


def titer_panel(ax, runs):
    names = [lab for _k, _c, lab in STRATEGIES]
    cols = [c for _k, c, _l in STRATEGIES]
    titer = [float(r["p"][-1]) for r in runs]
    mass = [float(r["p"][-1] * r["v"][-1]) for r in runs]
    pos = np.arange(3, dtype=float)
    ax.bar(pos, titer, width=0.56, color=cols, edgecolor="white",
           linewidth=0.8, zorder=3)
    for px, ti, ms_, c in zip(pos, titer, mass, cols):
        ax.text(px, ti + max(titer) * 0.020,
                "{:.1f} g/L\n{:.0f} g total".format(ti, ms_), ha="center",
                va="bottom", fontsize=6.5, color=c, linespacing=1.30)
    best = int(np.argmax(titer))
    gain = 100.0 * (titer[best] / titer[0] - 1.0)
    ax.text(0.5, 0.985, "{:s} leads constant feed by {:.0f} %".format(
        names[best], gain), transform=ax.transAxes, ha="center", va="top",
        fontsize=6.8, color=INK)
    ax.set_xticks(pos)
    ax.set_xticklabels([n.replace(" feed", "") for n in names], fontsize=7.2)
    ax.set_ylim(0.0, max(titer) * 1.42)
    ax.set_ylabel("final product titer (g/L)", fontsize=7.6)
    ax.tick_params(axis="y", labelsize=6.8)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title("Final titer after {:.0f} h".format(T_END), fontsize=8.4,
                 fontweight="bold", color=INK)
    return titer


def acetate_panel(ax, runs):
    top = max(r["a"].max() for r in runs)
    for run, (_k, col, lab) in zip(runs, STRATEGIES):
        ax.plot(run["t"], run["a"], color=col, lw=1.6, label=lab, zorder=3)
        ax.text(T_END + 0.25, run["a"][-1], "{:.2f}".format(run["a"][-1]),
                ha="left", va="center", fontsize=6.5, color=col)
    ax.axhline(K_AI, color=INK, lw=0.9, ls=(0, (3.2, 2.2)), zorder=2)
    ax.text(0.4, K_AI - 0.06,
            r"$K_{i,ace}$ = " + "{:.1f} g/L for product formation".format(
                K_AI), ha="left", va="top", fontsize=6.5, color=INK)
    ax.axvline(T_FEED, color=HAIR, lw=0.9, zorder=1)
    ax.set_xlim(0.0, T_END + 2.2)
    ax.set_ylim(0.0, top * 1.28)
    ax.set_xlabel("process time (h)", fontsize=7.6)
    ax.set_ylabel("acetate (g/L)", fontsize=7.6)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.tick_params(labelsize=6.8)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left",
              handlelength=1.4, borderaxespad=0.25)
    ax.set_title("Residual glucose sets the overflow by-product",
                 fontsize=8.4, fontweight="bold", color=INK)
    return top


def main():
    set_pub_style()
    f0, f_const, x_tf = design_feed()
    runs = [integrate(k, f0, f_const) for k, _c, _l in STRATEGIES]
    ref = runs[1]
    ph, pulses = ph_trace(ref["t"], ref["x"], ref["mu"])

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_top = axes_inches(fig, TOP_L)
    ax_feed = axes_inches(fig, MID_L)
    ax_do = axes_inches(fig, BOT_L)
    ax_titer = axes_inches(fig, TOP_R)
    ax_ace = axes_inches(fig, BOT_R)

    top_panel(ax_top, ref)
    feed_panel(ax_feed, runs, f0)
    do_ph_panel(ax_do, ref, ph, pulses)
    titer = titer_panel(ax_titer, runs)
    acetate_panel(ax_ace, runs)

    panel_labels_inches(fig, [ax_top, ax_feed, ax_do, ax_titer, ax_ace],
                        dx=-0.66, dy=0.10)
    savefig(fig, "877_bpr_fedbatch_profile.png")


if __name__ == "__main__":
    main()

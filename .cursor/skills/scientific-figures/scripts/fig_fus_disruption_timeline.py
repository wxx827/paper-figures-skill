"""A disruption read off five diagnostics on one shared millisecond axis.

The sequence is generated once and then measured, so the phase boundaries
drawn as vertical rules are found rather than typed.  Edge temperature and
stored energy collapse together at the thermal quench, and the radiated power
spike is that lost energy divided by the collapse time, which is why the spike
and the temperature drop are simultaneous by construction rather than by
drafting.  The current quench that follows is fitted where it counts: the
quench interval is the 80 %-to-20 % crossing of the plasma current found on
the array, the current-quench time is extrapolated from it in the usual linear
way, dI/dt is a central difference of the same trace, and the induced loop
voltage is L_p dI/dt with the internal inductance of the drawn geometry.

Two supporting panels put the single event in context: a log-normal spread of
current-quench times against the ITER-class lower bound, which exists because
too fast a quench drives unacceptable vessel forces, and the runaway current
fraction against injected atoms, whose collisional suppression threshold is
read off the curve.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_label_inches

RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3b5b92"
PURPLE = "#8e6bbf"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

T = np.linspace(-14.0, 30.0, 6000)     # ms, 0 = thermal quench
IP0 = 15.0                             # MA flat top
W0 = 350.0                             # MJ thermal stored energy
TE0 = 3.4                              # keV edge temperature
T_MHD = -9.0                           # MHD precursor onset, ms
T_SPI = -3.2                           # shattered pellet injection, ms
TQ_TAU = 0.85                          # thermal quench e-folding, ms
CQ_TAU = 4.0                           # current quench decay, ms
I_RE = 1.6                             # MA, runaway plateau that survives it
L_P = 13.0e-6                          # H, plasma self-inductance

N_SHOT = 4000
CQ_MED, CQ_SIG = 9.0, 0.42             # log-normal current-quench time, ms
CQ_MIN = 5.5                           # ITER-class lower bound, ms

N_INJ = np.linspace(0.2, 12.0, 300)    # 10^22 atoms injected
RE_MAX, N_HALF, RE_EXP = 0.62, 1.35, 3.2

FW, FH = 10.4, 6.4
LEFT, WIDTH = 0.90, 4.82
STACK_Y0, STACK_H, STACK_GAP = 1.06, 0.784, 0.055
AUX1 = (6.62, 3.30, 2.80, 1.90)
AUX2 = (6.62, 1.06, 2.80, 1.52)

SIGNALS = [("plasma current\nI$_p$ (MA)", RED, False),
           ("thermal energy\nW$_{th}$ (MJ)", ORANGE, False),
           ("radiated power\nP$_{rad}$ (MW, log)", PURPLE, True),
           ("edge temperature\nT$_e$ (keV)", BLUE, False),
           ("vertical shift\n$\\Delta$Z (m)", GREEN, False)]


# ------------------------------------------------------------------- signals
def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def build(rng):
    tq = 1.0 - smoothstep((T - 0.0) / (2.6 * TQ_TAU))       # 1 -> 0
    # The precursor decay is deliberately slow and still rising when the
    # quench starts: a smoothstep that *finishes* first has zero derivative
    # there, which puts a bogus notch in the radiated-power trace.
    mhd = 1.0 - 0.09 * smoothstep((T - T_MHD) / 24.0)
    te = TE0 * mhd * tq + 0.012
    wth = W0 * mhd * tq + 1.5

    # current quench starts when the thermal collapse has run its course
    cq = np.exp(-np.clip(T - 1.1, 0.0, None) / CQ_TAU)
    runaway = I_RE * smoothstep((T - 1.6) / 3.4) * np.exp(
        -np.clip(T - 9.0, 0.0, None) / 90.0)
    ip = np.maximum(np.where(T < 1.1, IP0, IP0 * cq), runaway)
    ip = ip + rng.normal(0.0, 0.012, T.size)

    # radiated power: the thermal energy that vanished, per unit time
    prad = np.clip(-np.gradient(wth, T * 1e-3), 0.0, None)     # MJ/s = MW
    prad += 9.0 + 26.0 * smoothstep((T - T_SPI) / 1.4) * np.exp(
        -np.clip(T - T_SPI, 0.0, None) / 16.0)
    # ohmic power dumped by the decaying current keeps radiation up
    prad += 620.0 * np.exp(-np.clip(T - 1.1, 0.0, None) / (0.8 * CQ_TAU)) * (
        T > 1.1)
    prad = np.clip(prad * np.exp(rng.normal(0.0, 0.035, T.size)), 1.0, None)

    dz = 0.92 * (np.exp(np.clip(T - 1.1, 0.0, None) / 9.5) - 1.0)
    dz = np.clip(dz, 0.0, 1.35) + rng.normal(0.0, 0.004, T.size)

    return dict(ip=ip, wth=wth, prad=prad, te=te, dz=dz)


def last_crossing(y, level):
    """Time of the last downward crossing of ``level``, linearly refined."""
    idx = np.flatnonzero(y >= level)
    i = int(idx[-1])
    if i >= y.size - 1:
        return float(T[-1])
    return float(np.interp(level, [y[i + 1], y[i]], [T[i + 1], T[i]]))


def quench_metrics(sig):
    ip = sig["ip"]
    hi = last_crossing(ip, 0.80 * IP0)
    lo = last_crossing(ip, 0.20 * IP0)
    t_cq = (lo - hi) / 0.6                       # linear 100 %-to-0 estimate
    didt = np.gradient(ip, T * 1e-3)             # MA/s
    k = int(np.argmin(didt))
    v_loop = L_P * abs(didt[k]) * 1e6            # V
    k_tq = int(np.argmin(np.gradient(sig["te"], T)))
    k_rad = int(np.argmax(sig["prad"]))
    return dict(t80=hi, t20=lo, t_cq=t_cq, didt=abs(didt[k]) / 1e3,
                t_didt=float(T[k]), v_loop=v_loop, t_tq=float(T[k_tq]),
                t_rad=float(T[k_rad]), p_rad=float(sig["prad"][k_rad]),
                dz_end=float(sig["dz"][-1]))


# -------------------------------------------------------------------- panels
def stack(fig, sig, met):
    traces = [sig["ip"], sig["wth"], sig["prad"], sig["te"], sig["dz"]]
    tops = [17.4, 400.0, 1.9e6, 4.0, 1.55]
    bots = [0.0, 0.0, 3.0, 0.0, 0.0]
    axes = []
    n = len(SIGNALS)
    for i, ((lab, col, logy), y, top, bot) in enumerate(
            zip(SIGNALS, traces, tops, bots)):
        rect = (LEFT, STACK_Y0 + (n - 1 - i) * (STACK_H + STACK_GAP),
                WIDTH, STACK_H)
        ax = axes_inches(fig, rect)
        axes.append(ax)

        for x0, x1, c in ((T_MHD, 0.0, lighten(GOLD, 0.86)),
                          (0.0, 1.1, lighten(RED, 0.84)),
                          (1.1, met["t20"], lighten(BLUE, 0.88))):
            ax.axvspan(x0, x1, color=c, lw=0.0, zorder=1)
        for xv in (T_MHD, 0.0, 1.1, met["t20"]):
            ax.axvline(xv, color=SUB, lw=0.6, ls=(0, (2.6, 2.0)), zorder=2)
        ax.axvline(T_SPI, color=GREEN, lw=0.9, ls=(0, (4.0, 2.0)), zorder=2)

        ax.plot(T, y, color=col, lw=1.35, zorder=5)
        ax.fill_between(T, bot, y, color=lighten(col, 0.90), lw=0.0, zorder=3)
        if logy:
            ax.set_yscale("log")
            ax.set_yticks([1e1, 1e3, 1e5])
        ax.set_xlim(T[0], T[-1])
        ax.set_ylim(bot, top)
        ax.set_ylabel(lab, fontsize=6.3, color=col, labelpad=2.0,
                      linespacing=1.2)
        ax.tick_params(axis="y", labelsize=6.0, colors=col)
        ax.spines["left"].set_color(col)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", lw=0.4, alpha=0.4)
        if i < n - 1:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=1.8)
        else:
            ax.set_xlabel("time relative to the thermal quench (ms)",
                          fontsize=7.6)
            ax.tick_params(axis="x", labelsize=6.8)

    top_ax = axes[0]
    trans = top_ax.get_xaxis_transform()
    for x, lab, col in ((0.5 * (T_MHD + T[0]), "flat top", SUB),
                        (0.5 * (T_MHD + 0.0), "MHD precursor", "#8a6d10"),
                        (0.5 * (1.1 + met["t20"]), "current quench", BLUE),
                        (0.5 * (met["t20"] + T[-1]), "runaway / halo", SUB)):
        top_ax.text(x, 1.05, lab, transform=trans, ha="center", va="bottom",
                    fontsize=6.3, color=col,
                    fontweight="bold" if col != SUB else "normal")
    top_ax.annotate("thermal quench", xy=(0.55, 1.03), xycoords=trans,
                    xytext=(0.55, 1.42), textcoords=trans, ha="center",
                    va="bottom", fontsize=6.3, color=RED, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    top_ax.annotate("shattered pellet\ninjection", xy=(T_SPI, 0.42),
                    xycoords=trans, xytext=(-13.4, 0.20), textcoords=trans,
                    ha="left", va="center", fontsize=6.2, color=GREEN,
                    linespacing=1.25, zorder=9,
                    arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.7))
    axes[0].annotate(
        "current quench:  I$_p$ 80 % $\\rightarrow$ 20 % in {:.2f} ms,\n"
        "t$_{{CQ}}$ = {:.1f} ms,  max |dI/dt| = {:.1f} GA/s,\n"
        "induced loop voltage L$_p$|dI/dt| = {:.1f} kV".format(
            met["t20"] - met["t80"], met["t_cq"], met["didt"],
            met["v_loop"] / 1e3),
        xy=(met["t_didt"], float(np.interp(met["t_didt"], T, sig["ip"]))),
        xytext=(14.0, 13.6), ha="left", va="top", fontsize=6.3, color=RED,
        linespacing=1.38, zorder=9,
        arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    axes[2].annotate("radiated spike {:.0f} MW at t = {:+.2f} ms,\n"
                     "simultaneous with the T$_e$ collapse".format(
                         met["p_rad"], met["t_rad"]),
                     xy=(met["t_rad"], met["p_rad"]), xytext=(6.0, 0.90),
                     textcoords=axes[2].get_xaxis_transform(), ha="left",
                     va="top", fontsize=6.2, color=PURPLE, linespacing=1.3,
                     zorder=9, arrowprops=dict(arrowstyle="-", color=PURPLE,
                                               lw=0.7))
    axes[4].annotate("vertical displacement event,\n{:.2f} m by the end of "
                     "the window".format(met["dz_end"]),
                     xy=(28.0, float(np.interp(28.0, T, sig["dz"]))),
                     xytext=(-13.4, 1.30), ha="left", va="top", fontsize=6.2,
                     color=GREEN, linespacing=1.3, zorder=9,
                     arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.7))
    return axes


def hist_panel(ax, rng, met):
    samp = CQ_MED * np.exp(rng.normal(0.0, CQ_SIG, N_SHOT))
    samp = samp[samp < 40.0]
    ax.hist(samp, bins=46, range=(0.0, 34.0), color=lighten(BLUE, 0.42),
            edgecolor="white", lw=0.4, zorder=4)
    ax.axvline(CQ_MIN, color=RED, lw=1.1, ls=(0, (4.4, 2.4)), zorder=6)
    frac = 100.0 * float(np.mean(samp < CQ_MIN))
    ax.axvspan(0.0, CQ_MIN, color=lighten(RED, 0.90), lw=0.0, zorder=2)
    ax.annotate("vessel-force limit {:.1f} ms:\n{:.1f} % of {:d} shots "
                "quench faster".format(CQ_MIN, frac, samp.size),
                xy=(CQ_MIN, 0.30), xycoords=ax.get_xaxis_transform(),
                xytext=(15.5, 0.93), textcoords=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=6.2, color=RED,
                linespacing=1.3, arrowprops=dict(arrowstyle="-", color=RED,
                                                 lw=0.7))
    med = float(np.median(samp))
    ax.axvline(med, color=INK, lw=1.0, zorder=6)
    ax.text(med + 0.7, 0.52, "median {:.1f} ms".format(med),
            transform=ax.get_xaxis_transform(), ha="left", va="center",
            fontsize=6.2, color=INK, zorder=9,
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                      edgecolor="none", alpha=0.85))
    ax.plot([met["t_cq"]], [0.0], marker="^", ms=6.0, color=ORANGE,
            clip_on=False, zorder=8)
    ax.text(0.975, 0.66, "$\\blacktriangle$ this shot, t$_{{CQ}}$ = "
            "{:.1f} ms".format(met["t_cq"]), transform=ax.transAxes,
            ha="right", va="center", fontsize=6.2, color=ORANGE)
    ax.set_xlim(0.0, 34.0)
    ax.set_xlabel("current-quench time t$_{CQ}$ (ms)", fontsize=7.4)
    ax.set_ylabel("number of shots (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", lw=0.5, alpha=0.45)
    ax.set_title("How fast the current goes, over {:d} shots".format(
        samp.size), fontsize=8.2, fontweight="bold", color=INK)
    return med, frac


def runaway_panel(ax):
    frac = 100.0 * RE_MAX / (1.0 + (N_INJ / N_HALF) ** RE_EXP)
    ax.plot(N_INJ, frac, color=ORANGE, lw=1.8, zorder=5)
    ax.fill_between(N_INJ, 0.06, frac, color=lighten(ORANGE, 0.88), lw=0.0,
                    zorder=3)
    thr = 1.0
    n_thr = float(np.interp(-thr, -frac, N_INJ))
    ax.axhline(thr, color=RED, lw=1.0, ls=(0, (4.4, 2.4)), zorder=6)
    ax.plot([n_thr], [thr], marker="X", ms=6.0, color=RED, zorder=8)
    ax.annotate("1 % of I$_p$ needs {:.1f} $\\times$ 10$^{{22}}$ atoms".format(
        n_thr), xy=(n_thr, thr), xytext=(n_thr + 0.5, thr * 5.0), ha="left",
        va="bottom", fontsize=6.3, color=RED, zorder=9,
        arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    ax.text(0.35, 0.44 * 100.0 * RE_MAX, "unmitigated\nrunaway plateau",
            ha="left", va="top", fontsize=6.2, color=ORANGE,
            linespacing=1.25)
    ax.set_yscale("log")
    ax.set_xlim(0.0, 12.0)
    ax.set_ylim(0.06, 130.0)
    ax.set_xlabel("injected atoms (10$^{22}$)", fontsize=7.4)
    ax.set_ylabel("runaway current fraction (%)", fontsize=7.4)
    ax.tick_params(labelsize=6.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(lw=0.5, alpha=0.45)
    ax.set_title("Mitigation dose against the runaway beam", fontsize=8.2,
                 fontweight="bold", color=INK)
    return n_thr


def main():
    set_pub_style()
    rng = np.random.default_rng(922)
    sig = build(rng)
    met = quench_metrics(sig)

    fig = plt.figure(figsize=(FW, FH))
    axes = stack(fig, sig, met)
    ax_h = axes_inches(fig, AUX1)
    ax_r = axes_inches(fig, AUX2)
    med, frac = hist_panel(ax_h, rng, met)
    n_thr = runaway_panel(ax_r)

    panel_label_inches(fig, axes[0], "(a)", dx=-0.66, dy=0.44)
    panel_label_inches(fig, ax_h, "(b)", dx=-0.62, dy=0.13)
    panel_label_inches(fig, ax_r, "(c)", dx=-0.62, dy=0.13)
    savefig(fig, "922_fus_disruption_timeline.png")


if __name__ == "__main__":
    main()

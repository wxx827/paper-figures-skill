"""How far the simulator is from the robot, and what closes the distance.

The main panel is one wipe of the same contact task executed twice: once in
the simulator and once on hardware.  The shaded area between the two traces
is the gap itself, so its size is the claim and the rms figure quoted beside
it is integrated from exactly those samples.  The upper side panel asks
whether the gap is a bias or noise by putting the per-trial peak forces
against the identity line; the lower one asks what to do about it, sweeping
the width of the domain-randomisation window against measured transfer
success.  Every annotated quantity is recomputed from the plotted arrays.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches, shared_limits

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"        # simulation
GREEN = "#2e7d5b"       # real robot
RED = "#b23a48"         # the gap
GREY = "#808080"

T_END = 4.0
DT = 0.004
N_TRIAL = 44


def traces(seed=17):
    """Same commanded wipe, executed in simulation and on hardware."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, T_END + DT, DT)

    ref = 18.0 * np.sin(np.pi * np.clip(t / T_END, 0.0, 1.0)) ** 1.4
    sim = ref + 0.25 * np.sin(2.0 * np.pi * 1.1 * t)

    # Hardware adds a servo lag, a stiction ramp on direction change and a
    # steady bias from an unmodelled tool mass.
    lag = np.interp(t - 0.085, t, sim, left=sim[0], right=sim[-1])
    stick = 1.6 * np.exp(-0.5 * ((t - 1.30) / 0.16) ** 2) \
        - 1.2 * np.exp(-0.5 * ((t - 2.75) / 0.20) ** 2)
    noise = rng.normal(0.0, 1.0, t.size)
    k = np.exp(-0.5 * (np.arange(-20, 21) / 6.0) ** 2)
    real = lag * 0.94 - 0.85 + stick + 0.22 * np.convolve(noise, k / k.sum(),
                                                          mode="same")
    return t, sim, real


def peaks(seed=23):
    rng = np.random.default_rng(seed)
    s = rng.uniform(9.0, 34.0, N_TRIAL)
    r = 1.17 * s + 1.1 + rng.normal(0.0, 1.9, N_TRIAL)
    return s, r


def sweep(seed=31):
    rng = np.random.default_rng(seed)
    rngw = np.array([0.0, 5.0, 10.0, 18.0, 26.0, 35.0, 45.0, 58.0, 72.0])
    trials = 40
    p = 0.86 * np.exp(-0.5 * ((rngw - 33.0) / 21.0) ** 2) + 0.04
    hits = rng.binomial(trials, np.clip(p, 0.0, 1.0))
    rate = hits / trials
    se = np.sqrt(rate * (1.0 - rate) / trials)
    return rngw, rate, se, trials


def trace_panel(ax, t, sim, real):
    gap = real - sim
    ax.fill_between(t, sim, real, color=RED, alpha=0.20, linewidth=0.0,
                    zorder=2, label="sim-to-real gap")
    ax.plot(t, sim, color=BLUE, lw=1.7, zorder=4, label="simulation")
    ax.plot(t, real, color=GREEN, lw=1.4, zorder=5, label="real robot")

    rms = float(np.sqrt(np.mean(gap ** 2)))
    imax = int(np.argmax(np.abs(gap)))
    ax.annotate(f"largest gap {abs(gap[imax]):.1f} N",
                xy=(t[imax], (sim[imax] + real[imax]) / 2.0),
                xytext=(t[imax] + 0.24, 4.0), fontsize=7.0, color=RED,
                ha="left", arrowprops=dict(arrowstyle="-", color=RED,
                                           lw=0.7))
    ax.text(0.03, 0.97, f"gap {rms:.2f} N rms over {T_END:.0f} s\n"
                        f"mean bias {gap.mean():+.2f} N",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.0,
            color=SUB, linespacing=1.4)
    ax.set_xlim(0.0, T_END)
    ax.set_ylim(-2.0, 22.0)
    ax.set_xlabel("time $t$ (s)")
    ax.set_ylabel("contact force $F_z$ (N)")
    ax.grid(axis="y", lw=0.45, alpha=0.65)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.4, frameon=False, loc="lower right", ncol=3,
              handlelength=1.7, columnspacing=1.1, borderaxespad=0.3)
    ax.set_title("One wipe, executed twice", fontsize=9.0, fontweight="bold",
                 color=INK)
    return rms, float(gap.mean())


def scatter_panel(ax, s, r):
    lo, hi = shared_limits([s, r], pad=0.08)
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=1.0, ls=(0, (4, 2.4)),
            zorder=3, label="1:1")
    a, b = np.polyfit(s, r, 1)
    xs = np.linspace(lo, hi, 40)
    ax.plot(xs, a * xs + b, color=RED, lw=1.2, zorder=4,
            label=f"fit  {a:.2f}x {b:+.1f}")
    ax.plot(s, r, ls="none", marker="o", ms=3.4, mfc=GREEN, mec="white",
            mew=0.5, alpha=0.9, zorder=5)

    d = r - s
    ax.text(0.035, 0.965, f"bias {d.mean():+.1f} N\n"
                          f"rmse {np.sqrt(np.mean(d ** 2)):.1f} N\n"
                          f"n = {len(s)} trials",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.6,
            color=SUB, linespacing=1.4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("simulated peak $F$ (N)", fontsize=7.4)
    ax.set_ylabel("measured peak $F$ (N)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.45, alpha=0.65)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right",
              handlelength=1.5, borderaxespad=0.25)
    ax.set_title("Peak force, sim against real", fontsize=8.2,
                 fontweight="bold", color=INK)
    return float(a), float(d.mean())


def sweep_panel(ax, w, rate, se, trials):
    ax.fill_between(w, 100 * (rate - se), 100 * (rate + se), color=BLUE,
                    alpha=0.16, linewidth=0.0, zorder=2)
    ax.plot(w, 100 * rate, color=BLUE, lw=1.5, marker="o", ms=3.4,
            mfc="white", mew=0.9, zorder=4)
    k = int(np.argmax(rate))
    ax.plot([w[k]], [100 * rate[k]], marker="o", ms=6.0, mfc="none",
            mec=RED, mew=1.3, zorder=5)
    ax.annotate(f"best {100 * rate[k]:.0f} % at "
                r"$\pm$" f"{w[k]:.0f} %", xy=(w[k], 100 * rate[k]),
                xytext=(2.0, 92.0), fontsize=6.6, color=RED, ha="left",
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    ax.text(1.0, 8.0, "no randomisation:\nsim-only policy", fontsize=6.4,
            color=SUB, ha="left", va="bottom", linespacing=1.35)
    ax.set_xlim(-3.0, 76.0)
    ax.set_ylim(0.0, 108.0)
    ax.set_xlabel("randomisation half-width (% of nominal)", fontsize=7.4)
    ax.set_ylabel("transfer success (%)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.45, alpha=0.65)
    ax.set_axisbelow(True)
    ax.set_title(f"Domain randomisation sweep, {trials} trials each",
                 fontsize=8.2, fontweight="bold", color=INK)
    return float(w[k]), 100.0 * float(rate[k])


def main():
    set_pub_style()
    t, sim, real = traces()
    s, r = peaks()
    w, rate, se, trials = sweep()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.66)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_sc = fig.add_subplot(gs[0, 1])
    ax_sw = fig.add_subplot(gs[1, 1])

    rms, bias = trace_panel(ax_main, t, sim, real)
    slope, sc_bias = scatter_panel(ax_sc, s, r)
    w_best, p_best = sweep_panel(ax_sw, w, rate, se, trials)

    fig.tight_layout(rect=[0, 0.045, 1, 0.945], h_pad=2.6, w_pad=2.4)
    fig.canvas.draw()
    ax_sc.apply_aspect()
    panel_labels_inches(fig, [ax_main, ax_sc, ax_sw], dx=-0.44, dy=0.10)
    savefig(fig, "587_rob_sim2real_gap.png")


if __name__ == "__main__":
    main()

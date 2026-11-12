"""Accelerating-rate calorimetry of an NMC811 18650 cell, 100 % SOC.

(a) Can temperature against time on a linear axis, with onset of
    self-heating, thermal runaway, and peak temperature read off the same
    trace.
(b) Self-heating rate dT/dt against temperature, the ARC signature used
    to define those three events.
(c) Time-to-runaway from the same dT/dt curve, starting at each
    temperature on the heat-wait-search path.

Illustrative synthetic ARC data.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
RED = "#b23a48"
GOLD = "#c9a227"
GREEN = "#2e7d5b"
INK = "#1a1a1a"
MUTED = "#777777"
SEED = 1216


def dtdt(t_c):
    """Self-heating rate, K/min, as a function of can temperature."""
    t = np.asarray(t_c, dtype=float)
    se = 4.2e-3 * np.exp((t - 80.0) / 28.0)
    cathode = 0.08 * np.exp((t - 165.0) / 18.0)
    runaway = 18.0 * np.exp((t - 215.0) / 9.5)
    return np.clip(se + cathode + runaway, 1e-4, 8e3)


def main():
    set_pub_style(base_font=9)
    rng = np.random.default_rng(SEED)
    t_c = np.linspace(40.0, 255.0, 900)
    rate = dtdt(t_c) * rng.uniform(0.92, 1.08, t_c.size)
    dt_min = np.gradient(t_c) / np.clip(rate, 1e-4, None)
    time_h = np.concatenate([[0.0], np.cumsum(dt_min[:-1])]) / 60.0
    onset = t_c[np.where(rate >= 0.02)[0][0]]
    tr = t_c[np.where(rate >= 10.0)[0][0]]
    t_pk = t_c[-1]
    k_on = int(np.argmin(np.abs(t_c - onset)))
    k_tr = int(np.argmin(np.abs(t_c - tr)))
    k0 = int(np.argmin(np.abs(t_c - (onset - 8.0))))
    time_h = time_h - time_h[k0]

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.85, 1.02, 5.35, 4.62))
    axb = axes_inches(fig, (7.30, 3.74, 2.90, 1.90))
    axc = axes_inches(fig, (7.30, 1.02, 2.90, 1.90))

    ax.plot(time_h[k0:], t_c[k0:], color=BLUE, lw=1.7, zorder=4)
    for t_ev, col, name in [(onset, GREEN, "onset"),
                            (tr, GOLD, "runaway"),
                            (t_pk, RED, "peak")]:
        k = int(np.argmin(np.abs(t_c - t_ev)))
        ax.axhline(t_ev, color=col, lw=0.7, ls=(0, (2.6, 1.8)), zorder=2)
        ax.plot([time_h[k]], [t_ev], marker="o", ms=5.0, mfc="white",
                mec=col, mew=1.1, zorder=5)
        ax.text(time_h[k] + 0.15, t_ev + 4.0,
                "{} {:.0f} $^\\circ$C".format(name, t_ev),
                fontsize=7.0, color=col, ha="left", va="bottom")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Can temperature ($^\\circ$C)")
    ax.set_xlim(0, max(time_h[k_tr] * 1.12, 1.5))
    ax.set_ylim(max(onset - 28.0, 70.0), 275)
    ax.tick_params(labelsize=7.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("ARC heat-wait-search, 100 % SOC", fontsize=9.0,
                 fontweight="bold", pad=6)
    ax.text(0.03, 0.96,
            "$T_{{onset}}$ = {:.0f} $^\\circ$C\n"
            "$T_{{TR}}$ = {:.0f} $^\\circ$C".format(onset, tr),
            transform=ax.transAxes, ha="left", va="top", fontsize=7.0,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#dcdcdc",
                      lw=0.6))

    axb.semilogy(t_c, rate, color=BLUE, lw=1.5)
    axb.axhline(0.02, color=GREEN, lw=0.8, ls=(0, (2.4, 1.6)))
    axb.axhline(10.0, color=GOLD, lw=0.8, ls=(0, (2.4, 1.6)))
    axb.axvline(onset, color=GREEN, lw=0.7, ls=(0, (1.6, 1.4)))
    axb.axvline(tr, color=GOLD, lw=0.7, ls=(0, (1.6, 1.4)))
    axb.set_xlabel("Can temperature ($^\\circ$C)", fontsize=8, labelpad=1.5)
    axb.set_ylabel("d$T$/d$t$ (K min$^{-1}$)", fontsize=8, labelpad=1.5)
    axb.tick_params(labelsize=6.5)
    axb.grid(color="#ededed", lw=0.5, which="both", zorder=0)
    for side in ("top", "right"):
        axb.spines[side].set_visible(False)
    axb.set_title("Self-heating rate", fontsize=8.5, fontweight="bold", pad=4)
    axb.set_xlim(40, 250)

    remain = (time_h[k_tr] - time_h) * 60.0
    mask = (t_c >= 80.0) & (t_c <= tr)
    axc.semilogy(t_c[mask], np.clip(remain[mask], 0.2, None), color=RED,
                 lw=1.5)
    axc.set_xlabel("Can temperature ($^\\circ$C)", fontsize=8, labelpad=1.5)
    axc.set_ylabel("Time to runaway (min)", fontsize=8, labelpad=1.5)
    axc.tick_params(labelsize=6.5)
    axc.grid(color="#ededed", lw=0.5, which="both", zorder=0)
    for side in ("top", "right"):
        axc.spines[side].set_visible(False)
    axc.set_title("Time-to-runaway", fontsize=8.5, fontweight="bold", pad=4)
    axc.text(0.04, 0.08, "from onset: {:.0f} min".format(remain[k_on]),
             transform=axc.transAxes, fontsize=6.4, color=INK)

    panel_labels_inches(fig, [ax, axb, axc], dx=-0.34, dy=0.10)
    savefig(fig, "1216_lib_thermal_runaway.png")


if __name__ == "__main__":
    main()

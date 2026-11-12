"""Electroglottographic cycle and closed-quotient histogram.

Main panel: 40 overlaid EGG cycles aligned at the positive peak, with the
mean cycle and the 0.2 / 0.8 crossing that define the closed quotient.
Side panels: closed-quotient histogram of 240 cycles, and a 30 ms raw EGG
trace with those crossings marked.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK, SUB = "#1a1a1a", "#555555"
RED, BLUE, TEAL = "#b23a48", "#3b5b92", "#2a7f7a"
SEED = 1049
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 5.50, 4.46)
AUX_T = (7.62, 3.52, 2.78, 1.96)
AUX_B = (7.62, 1.02, 2.78, 1.92)


def egg_cycle(phase, cq, rng):
    """One glottal cycle: closed plateau then open sinusoid, phase in [0, 1)."""
    y = np.empty_like(phase)
    closed = phase < cq
    y[closed] = 1.0 - 0.04 * np.sin(2 * np.pi * phase[closed] / max(cq, 1e-6))
    po = (phase[~closed] - cq) / (1.0 - cq)
    y[~closed] = 0.5 + 0.5 * np.cos(np.pi * po)
    y += rng.normal(0, 0.012, y.size)
    return y


def closed_quotient(y):
    thr = 0.2 + 0.6 * (y.max() - y.min())
    hi = np.where(y >= thr)[0]
    if hi.size < 2:
        return np.nan
    return (hi[-1] - hi[0] + 1) / y.size


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)
    n = 200
    phase = np.linspace(0, 1, n, endpoint=False)
    cycles = np.stack([egg_cycle(phase, rng.normal(0.54, 0.035), rng)
                       for _ in range(40)])
    mu = cycles.mean(0)
    cq_true = np.array([closed_quotient(c) for c in cycles])
    extra = np.array([closed_quotient(egg_cycle(phase, rng.normal(0.54, 0.04),
                                               rng)) for _ in range(200)])
    cqs = np.concatenate([cq_true, extra])

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN)
    ax_h = axes_inches(fig, AUX_T)
    ax_r = axes_inches(fig, AUX_B)

    for c in cycles:
        ax.plot(phase, c, color=BLUE, lw=0.7, alpha=0.22, zorder=3)
    ax.plot(phase, mu, color=RED, lw=1.7, zorder=6, label="mean cycle")
    lo, hi = 0.2 * mu.max() + 0.8 * mu.min(), 0.8 * mu.max() + 0.2 * mu.min()
    ax.axhline(hi, color=TEAL, lw=0.8, ls=(0, (3, 2)), zorder=5)
    ax.axhline(lo, color=TEAL, lw=0.8, ls=(0, (3, 2)), zorder=5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("cycle phase (-)")
    ax.set_ylabel("EGG (arb.)")
    ax.set_title("Forty cycles aligned at glottal closure", fontsize=9.0,
                 fontweight="bold", pad=6)
    ax.legend(loc="lower left", frameon=False, fontsize=7.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.97, 0.96, f"CQ = {np.nanmean(cqs):.2f} $\\pm$ "
            f"{np.nanstd(cqs):.2f}\n$n$ = {cqs.size} cycles",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.0,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#dcdcdc",
                      lw=0.6))

    ax_h.hist(cqs, bins=18, range=(0.38, 0.70), color=BLUE, edgecolor="white",
              linewidth=0.5, zorder=4)
    ax_h.axvline(np.nanmean(cqs), color=RED, lw=1.4, zorder=5)
    ax_h.set_xlabel("closed quotient (-)")
    ax_h.set_ylabel("count")
    ax_h.set_title("Closed-quotient distribution", fontsize=8.5,
                   fontweight="bold", pad=5)
    ax_h.grid(axis="y", lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax_h.spines[side].set_visible(False)

    fs = 20000.0
    f0 = 148.0
    t = np.arange(int(0.030 * fs)) / fs
    raw = np.zeros_like(t)
    for k, ph0 in enumerate(np.arange(0, 0.030, 1 / f0)):
        sl = (t >= ph0) & (t < ph0 + 1 / f0)
        if sl.any():
            ph = (t[sl] - ph0) * f0
            raw[sl] = egg_cycle(ph, 0.54, rng)
    ax_r.plot(t * 1000.0, raw, color=INK, lw=1.15)
    ax_r.set_xlim(0, 30)
    ax_r.set_xlabel("time (ms)")
    ax_r.set_ylabel("EGG (arb.)")
    ax_r.set_title("30 ms raw EGG", fontsize=8.5, fontweight="bold", pad=5)
    for side in ("top", "right"):
        ax_r.spines[side].set_visible(False)

    panel_labels_inches(fig, [ax, ax_h, ax_r], dx=-0.58, dy=0.14)
    savefig(fig, "1049_vox_egg_closed_quotient.png")


if __name__ == "__main__":
    main()

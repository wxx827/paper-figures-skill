"""Strong and weak scaling of a distributed finite-volume solver.

Left panel: strong scaling. A fixed global problem is spread over more and
more cores, so the speedup curve peels away from the ideal linear line once
communication starts to dominate; an Amdahl fit recovers the serial fraction
that the plateau implies. Right panel: weak scaling, where the per-core work
is held constant and the interesting quantity is parallel efficiency rather
than speedup. Both panels share the same core-count axis so the reader can
carry one horizontal position between them, and the scalability knee -- the
last core count that still keeps 70 % strong-scaling efficiency -- is drawn
in both and quoted in the footer.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter
from scipy.optimize import curve_fit
from pubstyle import set_pub_style, savefig
from figkit import shared_limits, panel_labels_inches

IDEAL = "#808080"      # grey dashed reference
MEAS = "#3b5b92"       # house blue, measured
FIT = "#b23a48"        # house red, the single emphasis object
KNEE = "#c9a227"       # house gold, knee marker
SUB = "#555555"
MUTED = "#777777"

CORES = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256,
                  512, 1024, 2048, 4096, 8192], dtype=float)
TARGET_EFF = 0.70      # definition of the scalability knee


def amdahl(p, f):
    """Speedup of a workload with serial fraction ``f`` on ``p`` cores."""
    return 1.0 / (f + (1.0 - f) / p)


def style_log_cores(ax, lo, hi):
    ax.set_xscale("log", base=2)
    ax.set_xlim(lo, hi)
    ax.set_xticks([1, 4, 16, 64, 256, 1024, 4096])
    ax.set_xticklabels(["1", "4", "16", "64", "256", "1024", "4096"])
    ax.xaxis.set_minor_locator(FixedLocator(CORES))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", labelsize=7.0)
    ax.tick_params(axis="x", which="minor", length=1.8)


def main():
    set_pub_style()
    rng = np.random.default_rng(5641)

    # ---- strong scaling: fixed global problem, growing core count ---------
    f_true = 0.0030
    t_raw = f_true + (1.0 - f_true) / CORES + 4.2e-4 * CORES ** 0.62
    speed = (t_raw[0] / t_raw) * np.exp(rng.normal(0.0, 0.020, CORES.size))
    eff_strong = speed / CORES

    # Fit Amdahl only up to the peak: past it the curve is falling, which no
    # serial fraction can describe, and including it would blame the plateau
    # on serial work rather than on communication.
    i_peak = int(np.argmax(speed))
    f_fit = float(curve_fit(amdahl, CORES[:i_peak + 1], speed[:i_peak + 1],
                            p0=[5e-3], bounds=(1e-6, 0.5))[0][0])
    p_dense = np.logspace(0, np.log2(CORES[-1]), 300, base=2.0)

    ok = np.flatnonzero(eff_strong >= TARGET_EFF)
    knee = CORES[ok[-1]] if ok.size else CORES[0]
    knee_speed = speed[ok[-1]] if ok.size else speed[0]

    # ---- weak scaling: constant work per core -----------------------------
    eff_weak = 1.0 / (1.0 + 0.0125 * np.log2(CORES) ** 1.26)
    eff_weak *= np.exp(rng.normal(0.0, 0.012, CORES.size))
    eff_weak[0] = 1.0

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 4.5))
    lo, hi = shared_limits([CORES], pad=0.10, log=True)

    # ---- (a) strong scaling ----------------------------------------------
    axL.plot(p_dense, p_dense, ls="--", lw=1.0, color=IDEAL,
             label="Ideal linear speedup")
    axL.plot(p_dense, amdahl(p_dense, f_fit), ls="-", lw=1.2, color=FIT,
             alpha=0.9, label=f"Amdahl fit, f = {f_fit:.4f}")
    axL.plot(CORES, speed, marker="o", ms=3.6, lw=1.3, color=MEAS,
             mfc="white", mew=0.9, label="Measured")
    axL.plot([knee], [knee_speed], marker="D", ms=5.2, color=KNEE,
             mec="#333333", mew=0.7, ls="none", zorder=5,
             label=f"Knee ({TARGET_EFF:.0%} efficiency)")
    axL.axvline(knee, color=KNEE, lw=0.8, ls=":", zorder=1)

    axL.set_yscale("log", base=2)
    axL.set_yticks([1, 4, 16, 64, 256, 1024, 4096])
    axL.set_yticklabels(["1", "4", "16", "64", "256", "1024", "4096"])
    axL.yaxis.set_minor_locator(FixedLocator(CORES))
    axL.yaxis.set_minor_formatter(NullFormatter())
    axL.set_ylim(0.8, 1.35 * CORES[-1])
    axL.tick_params(axis="y", which="minor", length=1.8)
    style_log_cores(axL, lo, hi)
    axL.set_xlabel("cores (-)", fontsize=8.0)
    axL.set_ylabel("speedup vs 1 core (x)", fontsize=8.0)
    axL.set_title("Strong scaling: fixed 1.2e9-cell mesh", fontsize=8.8,
                  fontweight="bold")
    axL.grid(True, which="major", lw=0.5, alpha=0.5)
    axL.legend(loc="upper left", fontsize=7.0, frameon=False,
               handlelength=1.8, labelspacing=0.30)

    peak = speed.max()
    axL.annotate(f"peak {peak:.0f}x at {CORES[int(np.argmax(speed))]:.0f} cores",
                 xy=(CORES[int(np.argmax(speed))], peak),
                 xytext=(0.97, 0.13), textcoords="axes fraction",
                 ha="right", va="bottom", fontsize=6.8, color=SUB,
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=MUTED,
                                 shrinkA=1.5, shrinkB=2.5))

    # ---- (b) weak scaling -------------------------------------------------
    axR.axhline(100.0, ls="--", lw=1.0, color=IDEAL,
                label="Ideal (constant work per core)")
    axR.plot(CORES, 100.0 * eff_weak, marker="s", ms=3.6, lw=1.3,
             color=MEAS, mfc="white", mew=0.9, label="Measured")
    axR.fill_between(CORES, 100.0 * eff_weak, 100.0, color=FIT, alpha=0.10,
                     lw=0, zorder=0)
    axR.axvline(knee, color=KNEE, lw=0.8, ls=":", zorder=1)

    style_log_cores(axR, lo, hi)
    axR.set_ylim(50.0, 106.0)
    axR.set_yticks([50, 60, 70, 80, 90, 100])
    axR.set_xlabel("cores (-)", fontsize=8.0)
    axR.set_ylabel("parallel efficiency (%)", fontsize=8.0)
    axR.set_title("Weak scaling: 150k cells per core", fontsize=8.8,
                  fontweight="bold")
    axR.grid(True, which="major", lw=0.5, alpha=0.5)
    axR.legend(loc="lower left", fontsize=7.0, frameon=False,
               handlelength=1.8, labelspacing=0.30)

    axR.annotate(f"{100.0 * eff_weak[-1]:.0f} % at {CORES[-1]:.0f} cores",
                 xy=(CORES[-1], 100.0 * eff_weak[-1]),
                 xytext=(0.97, 0.72), textcoords="axes fraction",
                 ha="right", va="top", fontsize=6.8, color=SUB,
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=MUTED,
                                 shrinkA=1.5, shrinkB=2.5))


    fig.tight_layout(rect=[0, 0.075, 1, 0.945], w_pad=2.6)
    panel_labels_inches(fig, [axL, axR], dx=-0.42, dy=0.06)
    savefig(fig, "564_hpc_scaling.png")


if __name__ == "__main__":
    main()

"""Figure 488 - Privacy budget against utility for four DP mechanisms.

The main panel sweeps the privacy budget over three decades and reads off the
accuracy each noise mechanism can still deliver; the non-private ceiling is
the dashed rule the curves are chasing.  The strip underneath carries the
epsilon axis itself, split into the three regimes reviewers argue about and
marked with the standard (epsilon, delta) budgets.  On the right, the reason
the curves bend: per-coordinate gradient densities widen as the noise
multiplier grows until the clipped signal is buried.

All quoted accuracies are read back out of the plotted curves.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from pubstyle import (set_pub_style, savefig, style_axes, get_palette,
                      lighten)

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRID = "#ececec"

FIG_W, FIG_H = 10.4, 6.4
MAIN = [0.062, 0.285, 0.585, 0.595]
EPSD = [0.062, 0.130, 0.585, 0.105]
DENS = [0.730, 0.500, 0.255, 0.345]
KEY = [0.700, 0.120, 0.300, 0.270]

EPS = np.logspace(-1.0, 2.0, 260)
EPS_LIM = (0.085, 130.0)
CEILING = 91.8
DELTA = 1e-5
BUDGETS = [1.0, 3.0, 8.0, 20.0]

# name, colour, utility gap at eps -> 0, half-recovery budget, sharpness
MECHS = [("Laplace mechanism", GOLD, 62.0, 3.20, 0.95),
         ("Gaussian mechanism", BLUE, 55.0, 1.60, 1.05),
         ("DP-SGD, moments accountant", GREEN, 48.0, 1.00, 1.15),
         ("Adaptive clipping (ours)", ACCENT, 42.0, 0.55, 1.20)]
SIGMAS = [0.0, 0.6, 1.2, 2.4]
CLIP = 1.0


def utility(eps, gap, half, k):
    """Saturating utility curve: noise dominates below the half budget."""
    return CEILING - gap / (1.0 + (eps / half) ** k)


def main():
    set_pub_style()
    rng = np.random.default_rng(488)
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # ---- privacy / utility frontier --------------------------------------
    ax = fig.add_axes(MAIN)
    ax.axhline(CEILING, color="#808080", lw=0.9, ls=(0, (4, 2.4)), zorder=2)
    ax.text(EPS_LIM[1], CEILING + 0.9, f"non-private ceiling {CEILING:.1f}%",
            ha="right", va="bottom", fontsize=6.6, color=SUB)

    curves = {}
    for name, col, gap, half, k in MECHS:
        y = utility(EPS, gap, half, k)
        band = 0.9 + 0.6 * np.exp(-EPS / 4.0)
        lead = col == ACCENT
        ax.fill_between(EPS, y - band, y + band, color=col,
                        alpha=0.14 if lead else 0.07, lw=0, zorder=3)
        ax.plot(EPS, y, color=col, lw=1.9 if lead else 1.35,
                zorder=6 if lead else 4)
        curves[name] = (col, gap, half, k)

    for e in BUDGETS[:-1]:
        ax.axvline(e, color=MUTED, lw=0.7, ls=(0, (3, 2.4)), zorder=1)
        ax.text(e, 97.0, f"$\\varepsilon$ = {e:g}", ha="center", va="top",
                fontsize=6.5, color=MUTED,
                bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                          edgecolor="none"))

    best = MECHS[-1]
    a1 = utility(1.0, best[2], best[3], best[4])
    a8 = utility(8.0, best[2], best[3], best[4])
    ax.annotate(f"{a1:.1f}%", xy=(1.0, a1), xytext=(-9, -12),
                textcoords="offset points", ha="right", va="top",
                fontsize=6.8, color=ACCENT, fontweight="bold")
    ax.scatter([1.0, 8.0], [a1, a8], s=26, facecolor=ACCENT,
               edgecolor="white", linewidth=0.9, zorder=8)
    ax.annotate(f"{CEILING - a8:.1f} pt below ceiling at "
                f"$\\varepsilon$ = 8", xy=(8.0, a8), xytext=(10, -14),
                textcoords="offset points", ha="left", va="top",
                fontsize=6.8, color=ACCENT,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#b0b0b0",
                                shrinkA=0.0, shrinkB=3.0))

    ax.set_xscale("log")
    ax.set_xlim(*EPS_LIM)
    ax.set_ylim(28.0, 99.0)
    ax.set_yticks([30, 45, 60, 75, 90])
    ax.tick_params(labelsize=7.0, length=2.5, labelbottom=False)
    ax.grid(True, axis="y", lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    ax.set_ylabel("global model accuracy (%)", fontsize=8.0, labelpad=2.5)
    style_axes(ax)
    ax.set_title("What each mechanism buys back as the budget loosens",
                 fontsize=8.5, fontweight="bold", pad=4.0)

    handles = [Line2D([], [], color=c, lw=1.9 if c == ACCENT else 1.35,
                      label=n) for n, c, *_ in MECHS]
    handles.append(Line2D([], [], color="#808080", lw=0.9, ls=(0, (4, 2.4)),
                          label="non-private ceiling"))
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=6.9,
              handlelength=1.6, labelspacing=0.34, borderaxespad=0.4)

    # ---- epsilon strip with the (eps, delta) budgets ---------------------
    axe = fig.add_axes(EPSD)
    regimes = [(EPS_LIM[0], 1.0, lighten(BLUE, 0.88), "strong privacy"),
               (1.0, 10.0, "#f0f0f0", "practical range"),
               (10.0, EPS_LIM[1], lighten(ACCENT, 0.90), "weak privacy")]
    for lo, hi, fc, lab in regimes:
        axe.axvspan(lo, hi, facecolor=fc, edgecolor="none", zorder=0)
        axe.text(np.sqrt(lo * hi), 0.74, lab, ha="center", va="center",
                 fontsize=6.4, color=SUB, zorder=3)
    for e in BUDGETS:
        axe.plot([e, e], [0.02, 0.42], lw=0.9, color=MUTED, zorder=3)
        axe.text(e, 0.30, f"  ({e:g}, $10^{{-5}}$)", ha="left", va="center",
                 fontsize=6.2, color=INK, zorder=4)
    axe.set_xscale("log")
    axe.set_xlim(*EPS_LIM)
    axe.set_ylim(0, 1)
    axe.set_yticks([])
    axe.set_xticks([0.1, 1.0, 10.0, 100.0])
    axe.set_xticklabels(["0.1", "1", "10", "100"])
    axe.xaxis.set_minor_formatter(plt.NullFormatter())
    axe.tick_params(labelsize=7.0, length=2.5)
    axe.set_xlabel(r"privacy budget $\varepsilon$ (-),  released with "
                   r"$\delta = 10^{-5}$", fontsize=8.0, labelpad=2.0)
    for s in ("top", "right", "left"):
        axe.spines[s].set_visible(False)

    # ---- gradient densities under increasing noise -----------------------
    axd = fig.add_axes(DENS)
    shades = get_palette("cb_sequential_blue", 6)[2:6]
    grid = np.linspace(-4.2, 4.2, 400)
    signal = np.clip(rng.normal(0.42, 0.38, 24000), -CLIP, CLIP)
    peaks = []
    for sig, col in zip(SIGMAS, shades):
        sample = signal + rng.normal(0.0, sig, signal.size)
        dens = gaussian_kde(sample, bw_method=0.20)(grid)
        peaks.append(dens.max())
        axd.fill_between(grid, dens, color=col, alpha=0.22, lw=0)
        axd.plot(grid, dens, color=col, lw=1.35,
                 label=f"$\\sigma$ = {sig:.1f}")
    for s in (-CLIP, CLIP):
        axd.axvline(s, color=MUTED, lw=0.7, ls=(0, (3, 2.2)), zorder=1)
    axd.text(CLIP + 0.14, max(peaks) * 0.60, f"clip $C$ = {CLIP:.1f}",
             ha="left", va="center", fontsize=6.3, color=MUTED)
    axd.set_xlim(-4.2, 4.2)
    axd.set_ylim(0, max(peaks) * 1.12)
    axd.set_xlabel("clipped gradient coordinate (-)", fontsize=7.5,
                   labelpad=1.8)
    axd.set_ylabel("density (-)", fontsize=7.5, labelpad=2.0)
    axd.tick_params(labelsize=6.8, length=2.2)
    axd.grid(True, axis="y", lw=0.5, color=GRID)
    axd.set_axisbelow(True)
    axd.legend(loc="upper right", frameon=False, fontsize=6.5,
               handlelength=1.3, labelspacing=0.28, borderaxespad=0.25,
               title="noise multiplier", title_fontsize=6.5)
    style_axes(axd)
    axd.set_title("Noise buries the clipped signal", fontsize=8.5,
                  fontweight="bold", pad=4.0)

    # ---- operating-point table -------------------------------------------
    axk = fig.add_axes(KEY)
    axk.set_axis_off()
    axk.set_xlim(0, 1)
    axk.set_ylim(0, 1)
    axk.text(0.02, 0.98, r"Operating points  ($\varepsilon$, $\delta$)",
             fontsize=7.6, fontweight="bold", color=INK, ha="left", va="top")
    cols = [("budget", 0.02), ("Gaussian", 0.40), ("ours", 0.76)]
    for lab, x in cols:
        axk.text(x, 0.76, lab, fontsize=6.6, color=MUTED, ha="left",
                 va="center")
    axk.plot([0.02, 0.98], [0.68, 0.68], lw=0.7, color="#cccccc")
    gauss = MECHS[1]
    for i, e in enumerate(BUDGETS):
        y = 0.55 - 0.155 * i
        axk.text(0.02, y, f"({e:g}, $10^{{-5}}$)", fontsize=6.6, color=INK,
                 ha="left", va="center")
        axk.text(0.40, y, f"{utility(e, *gauss[2:]):.1f}%", fontsize=6.6,
                 color=BLUE, ha="left", va="center")
        axk.text(0.76, y, f"{utility(e, *best[2:]):.1f}%", fontsize=6.6,
                 color=ACCENT, ha="left", va="center", fontweight="bold")

    for tag, rect in (("(a)", MAIN), ("(b)", DENS)):
        fig.text(rect[0] - 0.40 / FIG_W, rect[1] + rect[3] + 0.07 / FIG_H,
                 tag, fontsize=11, fontweight="bold", ha="left", va="bottom")

    savefig(fig, "488_fed_dp_tradeoff.png")


if __name__ == "__main__":
    main()

"""Yield learning across three technology nodes: remaining yield loss on a
log-log ramp axis with the learning rate fitted per generation, plus the
defect-density decay and the defect-type Pareto for the newest node."""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
RED = "#b23a48"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"
BASE = "#808080"

WEEKS = np.array([1, 2, 3, 4, 6, 8, 11, 14, 18, 23, 29, 36, 44, 52],
                 dtype=float)

# label, initial loss %, learning exponent, mature loss floor %, colour
GENS = [("N7 node, 2019", 46.0, -0.42, 3.2, lighten(BLUE, 0.48)),
        ("N5 node, 2021", 55.0, -0.50, 3.6, BLUE),
        ("N3 node, 2023", 68.0, -0.62, 4.0, RED)]

DEFECT_TYPES = [("Particle", GOLD), ("Pattern bridge", BLUE),
                ("CMP scratch", GREEN), ("Etch residue", PURPLE),
                ("Photoresist", lighten(BLUE, 0.45)), ("Other", "#b3b3b3")]


def loss_curve(rng, loss0, b, floor):
    """Remaining yield loss per week, %, with lot-to-lot scatter."""
    clean = loss0 * WEEKS ** b + floor
    return clean * np.exp(rng.normal(0.0, 0.055, WEEKS.size))


def learning_rate(loss):
    """Loss removed per doubling of ramp time, from the plotted points."""
    b = np.polyfit(np.log10(WEEKS), np.log10(loss), 1)[0]
    return b, 100.0 * (1.0 - 2.0 ** b)


def main_panel(ax, series):
    for (name, _l0, _b, _f, col), loss in series.items():
        b, lr = learning_rate(loss)
        ax.plot(WEEKS, loss, "o", ms=3.6, mfc="white", mec=col, mew=1.1,
                zorder=5)
        fit = 10.0 ** np.polyval(np.polyfit(np.log10(WEEKS), np.log10(loss), 1),
                                 np.log10(WEEKS))
        ax.plot(WEEKS, fit, color=col, lw=1.6, zorder=4,
                label=f"{name}   LR {lr:.0f}% per doubling")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.85, 62)
    ax.set_ylim(2.2, 90)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 52])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "52"])
    ax.set_yticks([3, 5, 10, 20, 40, 80])
    ax.set_yticklabels(["3", "5", "10", "20", "40", "80"])
    ax.set_xlabel("Weeks since ramp start (log scale)", fontsize=8.0)
    ax.set_ylabel("Remaining yield loss  100 \u2212 Y  (%, log scale)",
                  fontsize=8.0)
    ax.set_title("Yield learning by technology node", fontsize=9.0,
                 fontweight="bold", pad=4)
    ax.tick_params(labelsize=7.0)
    ax.grid(True, which="major", color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=7.0, frameon=False, handlelength=1.5,
              borderaxespad=0.3)
    style_axes(ax)

    # right-hand ruler in yield units, so the reader never has to subtract
    ax2 = ax.twinx()
    ax2.set_yscale("log")
    ax2.set_ylim(*ax.get_ylim())
    ticks = [3, 5, 10, 20, 40, 80]
    ax2.set_yticks(ticks)
    ax2.set_yticklabels([f"{100 - t:g}" for t in ticks])
    ax2.set_ylabel("Equivalent die yield (%)", fontsize=8.0)
    ax2.tick_params(labelsize=7.0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)


def defect_panel(ax, series, area=1.2):
    """Poisson inversion of the same loss curves: D0 = -ln(Y) / die area."""
    for (_name, _l0, _b, _f, col), loss in series.items():
        d0 = -np.log(1.0 - loss / 100.0) / area
        ax.plot(WEEKS, d0, "-", color=col, lw=1.5, zorder=4)
        ax.plot(WEEKS[::3], d0[::3], "o", ms=3.0, mfc="white", mec=col,
                mew=1.0, zorder=5)
    ax.axhline(0.10, color=BASE, lw=0.9, ls=(0, (4, 2.5)), zorder=3)
    ax.text(0.95, 0.104, "mature target 0.10 /cm$^2$", ha="left", va="bottom",
            fontsize=6.2, color=SUB)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.85, 62)
    ax.set_ylim(0.075, 1.9)
    ax.set_xticks([1, 4, 16, 52])
    ax.set_xticklabels(["1", "4", "16", "52"])
    ax.set_yticks([0.1, 0.2, 0.5, 1.0])
    ax.set_yticklabels(["0.1", "0.2", "0.5", "1.0"])
    ax.set_xlabel("Weeks since ramp start", fontsize=7.5)
    ax.set_ylabel("Defect density  $D_0$ (/cm$^2$)", fontsize=7.5)
    ax.set_title("Defect density decay, 1.2 cm$^2$ die", fontsize=8.5,
                 fontweight="bold", pad=4)
    ax.tick_params(labelsize=6.5)
    ax.grid(True, which="major", color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)


def pareto_panel(ax, total_loss):
    share = np.array([0.31, 0.24, 0.16, 0.12, 0.10, 0.07])
    share = share / share.sum()
    loss = share * total_loss
    order = np.argsort(-loss)
    loss = loss[order]
    names = [DEFECT_TYPES[i][0] for i in order]
    cols = [DEFECT_TYPES[i][1] for i in order]
    x = np.arange(len(loss))
    ax.bar(x, loss, width=0.66, color=cols, edgecolor="white", linewidth=0.6,
           zorder=3)
    for xi, v in zip(x, loss):
        ax.text(xi, v + total_loss * 0.018, f"{v:.1f}", ha="center",
                va="bottom", fontsize=6.1, color=INK, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.2, rotation=32, ha="right")
    ax.set_ylim(0, loss[0] * 1.34)
    ax.set_ylabel("Yield loss (%)", fontsize=7.5)
    ax.set_title("Loss by defect type, N3 week 52", fontsize=8.5,
                 fontweight="bold", pad=4)
    ax.tick_params(labelsize=6.5)
    ax.yaxis.grid(True, color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)

    cum = 100.0 * np.cumsum(loss) / loss.sum()
    ax2 = ax.twinx()
    ax2.plot(x, cum, color="#3a3a3a", lw=1.2, marker="o", ms=3.0,
             mfc="white", mec="#3a3a3a", mew=0.9, zorder=6)
    ax2.axhline(80.0, color=RED, lw=0.8, ls=(0, (3, 2.2)), zorder=5)
    n80 = int(np.searchsorted(cum, 80.0) + 1)
    ax2.text(len(loss) - 0.4, 80.5, f"top {n80} = 80%", ha="right",
             va="bottom", fontsize=6.2, color=RED)
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("Cumulative share (%)", fontsize=7.5)
    ax2.tick_params(labelsize=6.5)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)


def main():
    set_pub_style()
    rng = np.random.default_rng(4217)
    series = {g: loss_curve(rng, g[1], g[2], g[3]) for g in GENS}

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.40, hspace=0.62,
                          left=0.062, right=0.885, top=0.885, bottom=0.115)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_d = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 1])

    main_panel(ax_main, series)
    defect_panel(ax_d, series)
    pareto_panel(ax_p, float(series[GENS[-1]][-1]))

    rates = {g[0]: learning_rate(series[g])[1] for g in GENS}
    best = max(rates, key=lambda k: rates[k])

    panel_labels_inches(fig, [ax_main, ax_d, ax_p], dx=-0.40, dy=0.08)
    savefig(fig, "501_semi_yield_learning.png")


if __name__ == "__main__":
    main()

"""Figure 490 - Dirichlet partitions at three concentration levels.

The triptych is the same 20 clients and 10 classes drawn three times, once
per Dirichlet concentration, on one shared colour scale - which is the whole
argument: at alpha = 0.1 almost every client owns one or two classes
outright, at alpha = 10 the same budget spreads into an almost uniform sheet.
The right column turns four clients from each partition into 100% stacked
compositions on a shared share axis, and traces how the dominant-class share
decays as the concentration grows.

Clients are sorted by their dominant class so the block structure is visible;
every statistic quoted is measured on the arrays that were drawn.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, style_axes, get_cmap, get_palette
from figkit import shared_norm

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
ACCENT = "#b23a48"
GRID = "#ececec"

FIG_W, FIG_H = 10.4, 6.4
HM_Y, HM_H, HM_W = 0.315, 0.545, 0.175
HM_X = [0.055, 0.255, 0.455]
CBAR = [0.130, 0.170, 0.425, 0.022]
STACK = [0.715, 0.560, 0.265, 0.300]
SKEW = [0.715, 0.130, 0.265, 0.300]

ALPHAS = [0.1, 0.5, 10.0]
N_CLIENTS, N_CLASSES, N_K = 20, 10, 600
SHOW = 4                       # clients per group in the stacked panel


def partition(rng, alpha, n_clients=N_CLIENTS):
    """Dirichlet class shares per client, sorted by the dominant class."""
    p = rng.dirichlet(np.full(N_CLASSES, alpha), size=n_clients)
    counts = np.round(p * N_K)
    order = np.lexsort((-counts.max(axis=1), counts.argmax(axis=1)))
    return counts[order]


def dominant_share(counts):
    return counts.max(axis=1) / np.maximum(counts.sum(axis=1), 1.0)


def main():
    set_pub_style()
    rng = np.random.default_rng(490)
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    parts = [partition(rng, a) for a in ALPHAS]
    norm = shared_norm(parts, robust=(0, 99))
    cmap = get_cmap("cb_sequential_blue")
    class_cols = get_palette("cb_sequential_blue", N_CLASSES + 2)[1:-1]

    # ---- three partitions on one colour scale ----------------------------
    axes = []
    for i, (x0, alpha, counts) in enumerate(zip(HM_X, ALPHAS, parts)):
        ax = fig.add_axes([x0, HM_Y, HM_W, HM_H])
        im = ax.imshow(counts, cmap=cmap, norm=norm, aspect="auto",
                       interpolation="nearest")
        ax.set_xticks(range(0, N_CLASSES, 2))
        ax.set_yticks(range(0, N_CLIENTS, 5))
        ax.tick_params(labelsize=6.6, length=2.0)
        share = dominant_share(counts).mean()
        ax.set_title(f"$\\alpha$ = {alpha:g}", fontsize=8.5,
                     fontweight="bold", pad=12.0)
        ax.text(0.5, 1.005, f"top class holds {share:.0%} of a client",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=6.4, color=SUB)
        if i == 0:
            ax.set_ylabel("client, sorted by dominant class", fontsize=7.5,
                          labelpad=2.0)
        else:
            ax.tick_params(labelleft=False)
        if i == 1:
            ax.set_xlabel("class index", fontsize=8.0, labelpad=2.0)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.6)
            s.set_color("#333333")
        axes.append(ax)

    cax = fig.add_axes(CBAR)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.6, width=0.7, length=2.5, color="#333333")
    cb.set_label("samples held by one client in one class (-)", fontsize=7.0,
                 color=INK, labelpad=2.0)

    # ---- stacked class composition ---------------------------------------
    axs = fig.add_axes(STACK)
    pos, ticks, labels = [], [], []
    cursor = 0.0
    for alpha, counts in zip(ALPHAS, parts):
        pick = counts[np.linspace(0, len(counts) - 1, SHOW).astype(int)]
        share = pick / pick.sum(axis=1, keepdims=True) * 100.0
        xs = cursor + np.arange(SHOW)
        bottom = np.zeros(SHOW)
        for c in range(N_CLASSES):
            axs.bar(xs, share[:, c], bottom=bottom, width=0.78,
                    color=class_cols[c], edgecolor="white", linewidth=0.35,
                    zorder=3)
            bottom += share[:, c]
        ticks.append(xs.mean())
        labels.append(f"$\\alpha$ = {alpha:g}")
        pos.extend(xs)
        cursor += SHOW + 1.1
    axs.set_xticks(ticks)
    axs.set_xticklabels(labels, fontsize=7.0)
    axs.set_ylim(0, 100)
    axs.set_yticks([0, 25, 50, 75, 100])
    axs.set_xlim(-0.9, pos[-1] + 0.9)
    axs.set_ylabel("share of $n_k$ (%)", fontsize=7.5, labelpad=2.0)
    axs.tick_params(labelsize=6.8, length=2.2)
    axs.grid(True, axis="y", lw=0.5, color=GRID)
    axs.set_axisbelow(True)
    style_axes(axs)
    axs.set_title("Composition of four clients", fontsize=8.5,
                  fontweight="bold", pad=12.0)
    axs.text(0.5, 1.005, "shared share axis, shade = class index 0 to 9",
             transform=axs.transAxes, ha="center", va="bottom", fontsize=6.4,
             color=SUB)

    # ---- skew against the concentration ----------------------------------
    axk = fig.add_axes(SKEW)
    grid = np.logspace(-1.3, 2.0, 26)
    med, lo, hi = [], [], []
    for a in grid:
        s = dominant_share(partition(rng, a, n_clients=400))
        q = np.percentile(s, [25, 50, 75])
        lo.append(q[0] * 100)
        med.append(q[1] * 100)
        hi.append(q[2] * 100)
    med = np.asarray(med)
    axk.fill_between(grid, lo, hi, color="#3b5b92", alpha=0.14, lw=0)
    axk.plot(grid, med, color="#3b5b92", lw=1.6, zorder=4)
    axk.axhline(100.0 / N_CLASSES, color="#808080", lw=0.8, ls=(0, (3, 2.2)),
                zorder=2)
    axk.text(0.052, 100.0 / N_CLASSES + 2.0,
             f"uniform floor {100.0 / N_CLASSES:.0f}%", ha="left",
             va="bottom", fontsize=6.3, color=SUB)
    for a, counts in zip(ALPHAS, parts):
        s = dominant_share(counts).mean() * 100
        axk.scatter(a, s, s=32, facecolor=ACCENT, edgecolor="white",
                    linewidth=0.9, zorder=6)
        axk.annotate(f"{s:.0f}%", xy=(a, s), xytext=(0, 8),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=6.4, color=ACCENT, fontweight="bold")
    axk.set_xscale("log")
    axk.set_xlim(0.045, 130.0)
    axk.set_ylim(0, 105)
    axk.set_yticks([0, 25, 50, 75, 100])
    axk.set_xticks([0.1, 1.0, 10.0, 100.0])
    axk.set_xticklabels(["0.1", "1", "10", "100"])
    axk.xaxis.set_minor_formatter(plt.NullFormatter())
    axk.set_xlabel(r"Dirichlet concentration $\alpha$ (-)", fontsize=7.5,
                   labelpad=1.8)
    axk.set_ylabel("dominant-class share (%)", fontsize=7.5, labelpad=2.0)
    axk.tick_params(labelsize=6.8, length=2.2)
    axk.grid(True, axis="y", lw=0.5, color=GRID)
    axk.set_axisbelow(True)
    style_axes(axk)
    axk.set_title("Skew is set by one knob", fontsize=8.5, fontweight="bold",
                  pad=4.0)

    marks = ([([x, HM_Y, HM_W, HM_H], 0.30) for x in HM_X]
             + [(STACK, 0.30), (SKEW, 0.07)])
    for tag, (rect, dy) in zip(("(a)", "(b)", "(c)", "(d)", "(e)"), marks):
        fig.text(rect[0] - 0.16 / FIG_W, rect[1] + rect[3] + dy / FIG_H,
                 tag, fontsize=11, fontweight="bold", ha="left", va="bottom")

    savefig(fig, "490_fed_noniid_partition.png")


if __name__ == "__main__":
    main()

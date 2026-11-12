"""Early exit: which samples stop early, and what the saved compute costs.

Main panel: the confidence of a 12-block network at every depth for four
difficulty quartiles of the same test set, drawn as the quartile mean with a
10th to 90th percentile band, against the confidence threshold an early-exit
policy compares to; an inset turns the same run into the share of the batch
still executing at each block.  Auxiliary (both 58 per cent of the main panel
width): the exit block of every quartile as a row-normalised map, and
accuracy against the compute a sample actually consumes for a threshold sweep
next to the fixed-depth networks it has to beat.

The generated quantity is one correctness probability per sample per block;
confidence is that probability shifted by a per-sample calibration offset and
per-block noise, which is what lets an early exit be wrong.  Every accuracy is
the mean
correctness at whichever block each sample stopped at, every compute figure
is the mean exit depth times the measured per-block cost, and the rows of the
map in (b) are counts of the same exits, so they sum to 100 per cent.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, get_palette, get_cmap
from figkit import axes_inches, panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
TEAL = "#2a7f7a"
RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3b5b92"
GREY = "#b3b3b3"

N_SAMP = 6000                 # test samples
N_BLOCK = 12                  # transformer blocks, each with an exit head
N_CLASS = 100
GFLOP_BLOCK = 0.38            # GFLOPs per block for one sample
GFLOP_HEAD = 0.004            # GFLOPs per exit head evaluated
TAU_SHOW = 0.85
TAU_GRID = np.concatenate([np.linspace(0.30, 0.96, 23),
                           np.linspace(0.965, 0.999, 8)])
N_GROUP = 4
STATIC = (4, 6, 8, 12)        # fixed-depth networks to compare against

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.14, 3.16, 1.86)


def dataset(rng):
    """Per-sample correctness probability and confidence at every block."""
    diff = rng.uniform(0.0, 1.0, N_SAMP)
    depth = 1.2 + 12.5 * diff ** 1.5          # block at which it is solved
    lay = np.arange(1, N_BLOCK + 1)[None, :]
    q = 1.0 / N_CLASS + (1.0 - 1.0 / N_CLASS) / (
        1.0 + np.exp(-1.15 * (lay - depth[:, None])))
    bias = rng.normal(-0.075, 0.035, (N_SAMP, 1))
    jitter = rng.normal(0.0, 0.032, (N_SAMP, N_BLOCK))
    c = np.clip(q + bias + jitter, 1.0 / N_CLASS, 0.999)
    return diff, q, c


def exit_block(c, tau):
    """First block whose confidence clears the threshold, else the last one."""
    hit = c >= tau
    first = np.argmax(hit, axis=1) + 1
    return np.where(hit.any(axis=1), first, N_BLOCK)


def cost(ex):
    """GFLOPs a sample consumes, blocks executed plus the heads evaluated."""
    return ex * GFLOP_BLOCK + ex * GFLOP_HEAD


def policy(q, c, tau):
    """Accuracy and mean compute of the threshold policy."""
    ex = exit_block(c, tau)
    acc = float(q[np.arange(q.shape[0]), ex - 1].mean())
    return ex, 100.0 * acc, float(cost(ex).mean())


def trace_panel(ax, c, groups, ex, names):
    shades = get_palette("cb_sequential_blue", 7)
    cols = [shades[2], shades[3], shades[5], shades[6]]
    lay = np.arange(1, N_BLOCK + 1)
    for g, (col, name) in enumerate(zip(cols, names)):
        sel = groups == g
        m = c[sel].mean(axis=0)
        lo, hi = np.percentile(c[sel], [10.0, 90.0], axis=0)
        ax.fill_between(lay, lo, hi, facecolor=col, alpha=0.13,
                        edgecolor="none", zorder=2)
        ax.plot(lay, m, color=col, lw=1.6, marker="o", ms=3.4, mfc=col,
                mec="white", mew=0.5, zorder=5,
                label="{:s} (n = {:d}, mean exit block {:.1f})"
                      .format(name, int(sel.sum()), ex[sel].mean()))
        k = np.argmax(m >= TAU_SHOW)
        if m[k] >= TAU_SHOW:
            ax.plot([lay[k]], [m[k]], marker="o", ms=6.5, mfc="none",
                    mec=col, mew=1.3, zorder=7)
    ax.plot([0.6, N_BLOCK + 0.4], [TAU_SHOW, TAU_SHOW], color=ORANGE, lw=1.1,
            ls=(0, (3.5, 2.0)), zorder=6)
    ax.text(0.7, TAU_SHOW + 0.012,
            "exit threshold {:.2f}; rings mark the block at which each "
            "quartile mean first clears it".format(TAU_SHOW),
            fontsize=6.5, color=ORANGE, ha="left", va="bottom")
    ax.plot([0.6, N_BLOCK + 0.4], [1.0 / N_CLASS, 1.0 / N_CLASS], color=GREY,
            lw=0.9, ls=(0, (3.0, 2.0)), zorder=3)
    ax.text(N_BLOCK + 0.3, 1.0 / N_CLASS + 0.012,
            "chance, {:d} classes".format(N_CLASS), fontsize=6.0,
            color=MUTED, ha="right", va="bottom")
    ax.set_xlim(0.6, N_BLOCK + 0.4)
    ax.set_ylim(-0.02, 1.06)
    ax.set_xticks(np.arange(1, N_BLOCK + 1))
    ax.set_xlabel("block index (each block costs {:.2f} GFLOPs per sample)"
                  .format(GFLOP_BLOCK))
    ax.set_ylabel("exit-head confidence (max softmax probability)")
    ax.grid(lw=0.5, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.7, 0.045,
            "bands are the 10th to 90th percentile of the samples in that "
            "quartile", fontsize=6.2, color=SUB, ha="left", va="bottom")

    ins = ax.inset_axes([0.545, 0.115, 0.425, 0.29])
    running = np.array([100.0 * float((ex > l).mean())
                        for l in range(0, N_BLOCK + 1)])
    ins.step(np.arange(0, N_BLOCK + 1), running, where="post", color=TEAL,
             lw=1.3, zorder=5)
    ins.fill_between(np.arange(0, N_BLOCK + 1), 0.0, running, step="post",
                     facecolor=TEAL, alpha=0.14, zorder=3)
    ins.set_xlim(0.0, N_BLOCK)
    ins.set_ylim(0.0, 108.0)
    ins.set_xlabel("after block", fontsize=6, labelpad=1.5)
    ins.set_ylabel("still running (%)", fontsize=6, labelpad=1.5)
    ins.tick_params(labelsize=6, width=0.6, length=2.0)
    ins.grid(lw=0.4, alpha=0.25)
    for s in ins.spines.values():
        s.set_linewidth(0.6)
        s.set_color("#9a9a9a")
    ins.text(N_BLOCK - 0.3, 100.0,
             "half the batch is gone\nby block {:d}".format(
                 int(np.argmax(running <= 50.0))), fontsize=6, color=SUB,
             ha="right", va="top", linespacing=1.4)

    ax.legend(loc="upper left", fontsize=6.4, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.5, borderaxespad=0.4, labelspacing=0.25)
    ax.set_title("Easy samples are already confident four blocks in",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)


def map_panel(fig, ax, mat, names):
    im = ax.imshow(mat, cmap=get_cmap("cb_sequential_blue"), vmin=0.0,
                   vmax=float(mat.max()), origin="upper",
                   interpolation="nearest", aspect="equal",
                   extent=(0.5, N_BLOCK + 0.5, N_GROUP - 0.5, -0.5))
    for g in range(N_GROUP):
        j = int(np.argmax(mat[g]))
        ax.add_patch(plt.Rectangle((j + 0.5, g - 0.5), 1.0, 1.0, fill=False,
                                   edgecolor=RED, linewidth=1.0, zorder=5))
    ax.set_xticks(np.arange(1, N_BLOCK + 1, 2))
    ax.set_xticklabels([str(i) for i in range(1, N_BLOCK + 1, 2)],
                       fontsize=6.0)
    ax.set_yticks(np.arange(N_GROUP))
    ax.set_yticklabels(names, fontsize=6.0)
    ax.set_xlabel("block the sample exits at", fontsize=6.8)
    ax.tick_params(width=0.7, length=2.5)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    cb = safe_colorbar(fig, im, ax, label="% of quartile", width=0.09,
                       pad=0.07, labelsize=6,
                       ticks=[0.0, float(mat.max()) / 2.0, float(mat.max())])
    cb.ax.set_ylabel("% of quartile (rows sum to 100)", fontsize=6)
    ax.set_title("Where each quartile stops, threshold {:.2f}"
                 .format(TAU_SHOW), fontsize=8.0, fontweight="bold",
                 color=INK)


def trade_panel(ax, sweep, static, op):
    g, a = np.array([s[0] for s in sweep]), np.array([s[1] for s in sweep])
    ax.plot(g, a, color=TEAL, lw=1.5, zorder=5,
            label="early exit, threshold swept")
    ax.plot([s[0] for s in static], [s[1] for s in static], ls="none",
            marker="s", ms=4.2, mfc="white", mec=GREY, mew=1.1, zorder=6,
            label="fixed-depth network")
    for depth, (gs, acc) in zip(STATIC, [(s[0], s[1]) for s in static]):
        ax.text(gs, acc - 0.9, "{:d} blocks".format(depth), fontsize=6,
                color=MUTED, ha="center", va="top")
    ax.plot([op[0]], [op[1]], marker="o", ms=6.0, mfc="none", mec=ORANGE,
            mew=1.3, zorder=8)
    ax.annotate("threshold {:.2f}\n{:.1f} % at {:.2f} GFLOPs"
                .format(TAU_SHOW, op[1], op[0]), xy=(op[0], op[1]),
                xytext=(op[0] + 0.42, op[1] - 12.0), fontsize=6,
                color=ORANGE, ha="left", va="center", linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.7,
                                shrinkA=2, shrinkB=3))
    ax.set_xlim(0.0, cost(N_BLOCK) * 1.12)
    ax.set_ylim(min(a.min(), min(s[1] for s in static)) - 3.0,
                max(a.max(), max(s[1] for s in static)) + 2.4)
    ax.set_xlabel("compute actually spent (GFLOPs per sample, mean)",
                  fontsize=6.8)
    ax.set_ylabel("accuracy (%)", fontsize=6.8)
    ax.grid(lw=0.5, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", fontsize=6, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.4, borderaxespad=0.3, labelspacing=0.22)
    ax.set_title("Spending the compute where it is needed", fontsize=8.0,
                 fontweight="bold", color=INK)


def audit(fig, foot, axes):
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for ax in axes:
        ab = ax.get_window_extent()
        for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            tb = t.get_window_extent()
            if tb.width <= 0.0 or not t.get_text():
                continue
            if tb.x1 < ab.x0 - 2.0 or tb.x0 > ab.x1 + 2.0:
                continue
            if fb.overlaps(tb):
                print("OVERLAP with", repr(t.get_text()))


def main():
    set_pub_style()
    rng = np.random.default_rng(8342)
    diff, q, c = dataset(rng)

    edges = np.quantile(diff, np.linspace(0.0, 1.0, N_GROUP + 1))
    groups = np.clip(np.searchsorted(edges, diff, side="right") - 1,
                     0, N_GROUP - 1)
    names = ["easiest 25 %", "second 25 %", "third 25 %", "hardest 25 %"]

    ex, acc_show, gf_show = policy(q, c, TAU_SHOW)
    mat = np.zeros((N_GROUP, N_BLOCK))
    for g in range(N_GROUP):
        sel = groups == g
        mat[g] = 100.0 * np.bincount(ex[sel] - 1, minlength=N_BLOCK) / sel.sum()
    assert np.allclose(mat.sum(axis=1), 100.0), "rows are shares of a quartile"

    sweep = []
    for tau in TAU_GRID:
        _, a, gfl = policy(q, c, tau)
        sweep.append((gfl, a, tau))
    static = []
    for depth in STATIC:
        static.append((cost(depth), 100.0 * float(q[:, depth - 1].mean()),
                       depth))
    full_gf, full_acc = static[-1][0], static[-1][1]

    print("threshold {:.2f}: {:.2f} % at {:.2f} GFLOPs, mean exit block {:.2f}"
          .format(TAU_SHOW, acc_show, gf_show, ex.mean()))
    print("full depth: {:.2f} % at {:.2f} GFLOPs -> {:.1f} % of the compute, "
          "{:.2f} points given up"
          .format(full_acc, full_gf, 100.0 * gf_show / full_gf,
                  full_acc - acc_show))
    for g, n in enumerate(names):
        sel = groups == g
        print("{:14s} mean exit {:.2f}  accuracy {:.1f} %"
              .format(n, ex[sel].mean(),
                      100.0 * q[sel, ex[sel] - 1].mean()))
    match = [s for s in sweep if s[1] >= full_acc - 0.5]
    saving = 100.0 * (1.0 - min(s[0] for s in match) / full_gf)
    print("within 0.5 points of full depth at {:.2f} GFLOPs, {:.1f} % saved"
          .format(min(s[0] for s in match), saving))
    assert ex[groups == 0].mean() < ex[groups == N_GROUP - 1].mean(), \
        "hard samples run deeper"
    assert acc_show < full_acc, "early exit gives up accuracy"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_t = axes_inches(fig, MAIN)
    ax_m = axes_inches(fig, AUX_T)
    ax_c = axes_inches(fig, AUX_B)

    trace_panel(ax_t, c, groups, ex, names)
    map_panel(fig, ax_m, mat, names)
    trade_panel(ax_c, [(s[0], s[1]) for s in sweep], static,
                (gf_show, acc_show))



    panel_labels_inches(fig, [ax_t, ax_m, ax_c], dx=-0.62, dy=0.14)
    savefig(fig, "1031_eff_early_exit_dynamic.png")


if __name__ == "__main__":
    main()

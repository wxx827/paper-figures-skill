"""P300 speller: from single flashes to one letter on the screen.

Main panel: the 6 x 6 flash matrix at true proportions
(``set_aspect("equal")``), every cell shaded by the classifier evidence for
that character, with the row and column evidence drawn as bars along the two
edges they belong to, the winning row and column highlighted and the selected
letter boxed.  Auxiliary panels (both 58 per cent of the main panel width):
the grand-average response to attended and unattended flashes with the P300
window shaded, and accuracy against the number of repetitions with the
resulting spelling rate on a second axis.

The classifier is a matched filter: the difference between the mean attended
and mean unattended epoch of a held-out training block, projected onto every
test epoch.  Cell evidence is the sum of its row and column score, the
selected letter is the argmax of that sum, the P300 latency and amplitude are
read off the plotted averages, and the spelling rate is the flash schedule
divided into a minute.  Synthetic data, no human subject.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d
from pubstyle import set_pub_style, savefig, get_cmap
from figkit import axes_inches, panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3b5b92"
TEAL = "#2a7f7a"
GREY = "#b3b3b3"
BOX = dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="none",
           alpha=0.86)

FS = 250.0                   # sampling rate, Hz
T_EP = 0.80                  # epoch after each flash, s
SOA = 0.175                  # stimulus onset asynchrony, s
PAUSE = 2.5                  # pause between selections, s
GRID = 6                     # rows and columns of the flash matrix
LETTERS = ("ABCDEF", "GHIJKL", "MNOPQR", "STUVWX", "YZ1234", "56789_")
N_TRAIN, N_TEST = 25, 60     # selections used to train and to test
REPS = 15                    # repetitions of the twelve flashes
REP_GRID = np.arange(1, REPS + 1)
NOISE_UV = 11.0              # single-trial background, microvolts
SHOW_SEL, SHOW_REPS = 3, 8   # selection and repetition count drawn in (a)
P300_WIN = (0.26, 0.44)      # window shaded in (b), s

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.14, 3.16, 1.86)


def erps(t):
    """Common visual response and the extra attended-only deflection."""
    def g(amp, mu, sd):
        return amp * np.exp(-0.5 * ((t - mu) / sd) ** 2)
    common = g(1.9, 0.105, 0.022) + g(-2.6, 0.168, 0.026)
    target = g(-2.4, 0.198, 0.034) + g(4.8, 0.330, 0.058) + g(1.1, 0.560, 0.095)
    return common, target


def epochs(rng, is_target, common, target):
    """One epoch per flash: shared response, attended extra, smoothed noise."""
    n = is_target.size
    noise = gaussian_filter1d(rng.normal(0.0, 1.0, (n, common.size)), 3.0,
                              axis=1)
    noise *= NOISE_UV / noise.std()
    return common[None, :] + is_target[:, None] * target[None, :] + noise


def flash_plan(rng, n_sel):
    """Which of the twelve row/column flashes carries the attended letter."""
    tgt_r = rng.integers(0, GRID, n_sel)
    tgt_c = rng.integers(0, GRID, n_sel)
    order = np.empty((n_sel, REPS, 2 * GRID), dtype=int)
    for s in range(n_sel):
        for r in range(REPS):
            order[s, r] = rng.permutation(2 * GRID)
    is_t = (order == tgt_r[:, None, None]) | \
           (order == (GRID + tgt_c)[:, None, None])
    return tgt_r, tgt_c, order, is_t


def block(rng, n_sel, common, target):
    tgt_r, tgt_c, order, is_t = flash_plan(rng, n_sel)
    x = epochs(rng, is_t.ravel(), common, target)
    return tgt_r, tgt_c, order, is_t, x.reshape(n_sel, REPS, 2 * GRID, -1)


def scores(x, template):
    return x @ template / np.linalg.norm(template)


def decide(sc, order, n_rep):
    """Row and column evidence after n_rep repetitions, and the winner."""
    ev = np.zeros(2 * GRID)
    cnt = np.zeros(2 * GRID)
    for r in range(n_rep):
        for k in range(2 * GRID):
            ev[order[r, k]] += sc[r, k]
            cnt[order[r, k]] += 1.0
    ev /= np.maximum(cnt, 1.0)
    return ev, int(np.argmax(ev[:GRID])), int(np.argmax(ev[GRID:]))


def matrix_panel(fig, ax, ev, win_r, win_c, tgt_r, tgt_c):
    cell = ev[:GRID, None] + ev[None, GRID:]
    vmax = float(np.abs(cell).max())
    im = ax.imshow(cell, cmap=get_cmap("cb_diverging"), vmin=-vmax, vmax=vmax,
                   origin="upper", extent=(-0.5, GRID - 0.5, GRID - 0.5, -0.5),
                   interpolation="nearest", aspect="equal", zorder=2)
    for r in range(GRID):
        for c in range(GRID):
            shade = abs(cell[r, c]) / vmax
            ax.text(c, r, LETTERS[r][c], fontsize=9.5, fontweight="bold",
                    ha="center", va="center", zorder=5,
                    color="white" if shade > 0.62 else INK)
    ax.add_patch(Rectangle((win_c - 0.5, win_r - 0.5), 1.0, 1.0, fill=False,
                           edgecolor=RED, linewidth=2.0, zorder=7))
    ax.add_patch(Rectangle((tgt_c - 0.72, tgt_r - 0.72), 1.44, 1.44,
                           fill=False, edgecolor=INK, linewidth=0.9,
                           linestyle=(0, (3.0, 2.0)), zorder=6))

    # The two bar strips share the matrix axes, so a bar can only ever line
    # up with its own row or column.  Both are signed about their own zero
    # rule, because most of the twelve scores are negative.
    span = float(np.abs(ev).max())
    x0, y0 = GRID + 1.5, GRID + 0.75
    for r in range(GRID):
        ax.barh(r, 1.45 * ev[r] / span, left=x0, height=0.62,
                color=RED if r == win_r else "#c8ccd2", edgecolor="white",
                linewidth=0.6, zorder=4)
    for c in range(GRID):
        ax.bar(c, 0.62 * ev[GRID + c] / span, bottom=y0, width=0.62,
               color=RED if c == win_c else "#c8ccd2", edgecolor="white",
               linewidth=0.6, zorder=4)
    ax.plot([x0] * 2, [-0.5, GRID - 0.5], color="#8a8a8a", lw=0.7, zorder=5)
    ax.plot([-0.5, GRID - 0.5], [y0] * 2, color="#8a8a8a", lw=0.7, zorder=5)
    ax.text(x0, y0, "row evidence (a.u.)", fontsize=6.4, color=SUB,
            ha="center", va="center")
    ax.text(-1.05, y0, "column evidence (a.u.)", fontsize=6.4, color=SUB,
            ha="center", va="center", rotation=90)

    ax.text(-0.5, -1.55,
            "attended {:s}, selected {:s}: row {:d} leads the runner-up by "
            "{:.1f}, column {:d} by {:.1f}"
            .format(LETTERS[tgt_r][tgt_c], LETTERS[win_r][win_c], win_r + 1,
                    np.sort(ev[:GRID])[-1] - np.sort(ev[:GRID])[-2],
                    win_c + 1,
                    np.sort(ev[GRID:])[-1] - np.sort(ev[GRID:])[-2]),
            fontsize=7.0, color=INK, ha="left", va="center")
    ax.text(-0.5, -1.02,
            "cell shade is the row score plus the column score; the dashed "
            "box marks the attended letter", fontsize=6.4, color=SUB,
            ha="left", va="center")

    ax.set_xlim(-1.6, GRID + 3.4)
    ax.set_ylim(GRID + 1.0, -1.94)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = safe_colorbar(fig, im, ax, width=0.11, pad=0.10, labelsize=6.2)
    cb.ax.set_ylabel("classifier evidence for the character (a.u.)",
                     fontsize=6.2)
    ax.set_title("After {:d} repetitions the evidence separates one row and "
                 "one column".format(SHOW_REPS), fontsize=9.0,
                 fontweight="bold", color=INK, pad=6.0)


def erp_panel(ax, t, xt, xn):
    ms = t * 1e3
    mt, mn = xt.mean(axis=0), xn.mean(axis=0)
    st = xt.std(axis=0) / np.sqrt(xt.shape[0])
    sn = xn.std(axis=0) / np.sqrt(xn.shape[0])
    ax.axvspan(P300_WIN[0] * 1e3, P300_WIN[1] * 1e3, facecolor=ORANGE,
               alpha=0.10, edgecolor="none", zorder=1)
    ax.fill_between(ms, mn - sn, mn + sn, facecolor=GREY, alpha=0.30,
                    edgecolor="none", zorder=2)
    ax.fill_between(ms, mt - st, mt + st, facecolor=RED, alpha=0.18,
                    edgecolor="none", zorder=3)
    ax.plot(ms, mn, color="#6f7680", lw=1.3, zorder=5,
            label="unattended flashes (n = {:d})".format(xn.shape[0]))
    ax.plot(ms, mt, color=RED, lw=1.6, zorder=6,
            label="attended flashes (n = {:d})".format(xt.shape[0]))
    ax.axhline(0.0, color=GREY, lw=0.7, zorder=2)
    k = int(np.argmax(mt))
    ax.annotate("P300 peak {:.0f} ms, {:.1f} $\\mu V$\nover the unattended "
                "mean".format(ms[k], mt[k] - mn[k]), xy=(ms[k], mt[k]),
                xytext=(8.0, 6.0), textcoords="offset points", fontsize=6,
                color=RED, ha="left", va="bottom", linespacing=1.4, bbox=BOX,
                zorder=9,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7,
                                shrinkA=1, shrinkB=3))
    ax.set_xlim(ms[0], ms[-1])
    ax.set_ylim(min(mt.min(), mn.min()) - 1.2, max(mt.max(), mn.max()) + 2.6)
    ax.set_xlabel("time from flash onset (ms)", fontsize=6.8)
    ax.set_ylabel("amplitude at Cz ($\\mu V$)", fontsize=6.8)
    ax.tick_params(labelsize=6.0, width=0.7, length=2.5)
    ax.grid(lw=0.5, alpha=0.28)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=6, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.4, borderaxespad=0.3, labelspacing=0.22)
    ax.set_title("Grand average over the test block", fontsize=8.0,
                 fontweight="bold", color=INK)


def reps_panel(ax, acc, rate, n_test):
    ax.plot(REP_GRID, acc, color=BLUE, lw=1.5, marker="o", ms=3.2,
            mfc="white", mec=BLUE, mew=0.9, zorder=6, label="accuracy")
    ax.axhline(100.0 / (GRID * GRID), color=GREY, lw=0.9, ls=(0, (3.0, 2.0)),
               zorder=3)
    ax.text(REP_GRID[-1], 100.0 / (GRID * GRID) + 2.0,
            "chance, {:d} characters".format(GRID * GRID), fontsize=6,
            color=MUTED, ha="right", va="bottom", bbox=BOX, zorder=7)
    k = int(np.argmax(acc >= 90.0)) if (acc >= 90.0).any() else len(acc) - 1
    ax.plot([REP_GRID[k]], [acc[k]], marker="o", ms=5.6, mfc="none", mec=RED,
            mew=1.3, zorder=8)
    ax.annotate("{:.0f} % at {:d} repetitions\n({:.1f} letters per minute)"
                .format(acc[k], REP_GRID[k], rate[k]),
                xy=(REP_GRID[k], acc[k]), xytext=(8.0, -16.0),
                textcoords="offset points", fontsize=6, color=RED,
                ha="left", va="center", linespacing=1.4, bbox=BOX, zorder=9,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7,
                                shrinkA=1, shrinkB=3))
    ax.set_xlim(0.4, REPS + 0.6)
    ax.set_ylim(0.0, 108.0)
    ax.set_xlabel("repetitions averaged (n = {:d} test selections)"
                  .format(n_test), fontsize=6.8)
    ax.set_ylabel("selection accuracy (%)", fontsize=6.8)
    ax.tick_params(labelsize=6.0, width=0.7, length=2.5)
    ax.grid(lw=0.5, alpha=0.28)
    ax.spines["top"].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(REP_GRID, rate, color=TEAL, lw=1.3, ls=(0, (4.0, 2.0)), zorder=5,
             label="spelling rate")
    ax2.set_ylim(0.0, max(rate) * 1.15)
    ax2.set_ylabel("letters per minute", fontsize=6.5, color=TEAL)
    ax2.tick_params(labelsize=6.0, width=0.7, length=2.5, colors=TEAL)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(TEAL)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=6, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.4, borderaxespad=0.3, labelspacing=0.22)
    ax.set_title("More repetitions, surer letters, slower typing",
                 fontsize=8.0, fontweight="bold", color=INK)


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
    rng = np.random.default_rng(3164)
    t = np.arange(int(T_EP * FS)) / FS
    common, target = erps(t)

    _, _, _, tr_is_t, tr_x = block(rng, N_TRAIN, common, target)
    flat = tr_x.reshape(-1, t.size)
    lab = tr_is_t.ravel()
    template = flat[lab].mean(axis=0) - flat[~lab].mean(axis=0)

    te_r, te_c, te_order, te_is_t, te_x = block(rng, N_TEST, common, target)
    te_sc = scores(te_x.reshape(-1, t.size), template).reshape(
        N_TEST, REPS, 2 * GRID)

    acc = []
    for n_rep in REP_GRID:
        hit = 0
        for s in range(N_TEST):
            _, wr, wc = decide(te_sc[s], te_order[s], int(n_rep))
            hit += int(wr == te_r[s] and wc == te_c[s])
        acc.append(100.0 * hit / N_TEST)
    acc = np.array(acc)
    t_sel = REP_GRID * 2 * GRID * SOA + PAUSE
    rate = 60.0 / t_sel

    ev, win_r, win_c = decide(te_sc[SHOW_SEL], te_order[SHOW_SEL], SHOW_REPS)
    flat_te = te_x.reshape(-1, t.size)
    lab_te = te_is_t.ravel()

    print("template peak {:.2f} uV at {:.0f} ms"
          .format(template.max(), 1e3 * t[int(np.argmax(template))]))
    print("shown selection: attended {:s}, selected {:s}"
          .format(LETTERS[te_r[SHOW_SEL]][te_c[SHOW_SEL]],
                  LETTERS[win_r][win_c]))
    for n_rep, a, r in zip(REP_GRID, acc, rate):
        print("{:2d} repetitions: {:5.1f} % correct, {:.1f} letters/min, "
              "{:.1f} s per selection".format(n_rep, a, r,
                                              t_sel[n_rep - 1]))
    assert win_r == te_r[SHOW_SEL] and win_c == te_c[SHOW_SEL], \
        "the drawn selection is correct"
    assert acc[-1] > acc[0], "averaging repetitions helps"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_m = axes_inches(fig, MAIN)
    ax_e = axes_inches(fig, AUX_T)
    ax_r = axes_inches(fig, AUX_B)

    matrix_panel(fig, ax_m, ev, win_r, win_c, te_r[SHOW_SEL], te_c[SHOW_SEL])
    erp_panel(ax_e, t, flat_te[lab_te], flat_te[~lab_te])
    reps_panel(ax_r, acc, rate, N_TEST)

    k = int(np.argmax(acc >= 90.0)) if (acc >= 90.0).any() else len(acc) - 1


    panel_labels_inches(fig, [ax_m, ax_e, ax_r], dx=-0.62, dy=0.14)
    savefig(fig, "1060_bci_p300_speller_matrix.png")


if __name__ == "__main__":
    main()

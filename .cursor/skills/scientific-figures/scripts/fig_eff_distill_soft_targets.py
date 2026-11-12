"""What a temperature actually exposes in a teacher, and where it stops paying.

Main panel: the teacher's class distribution averaged over a batch and drawn
at four softening temperatures, with the classes ordered target first, then
the three that share a superclass with it, then the rest, so the mass that
temperature moves onto the near misses can be read directly.  Auxiliary (both
58 per cent of the main panel width): student accuracy over a temperature
sweep at three loss weights, and a layerwise feature-alignment matrix between
the twelve teacher blocks and the six student blocks.

The soft targets are softmax of one logit array, so the probabilities plotted
are the array.  The sweep in (b) is driven by a contrast measured on those
same probabilities - the gap between the best and the median non-target class
- which rises as temperature lifts the near misses out of the floor and falls
again as everything approaches uniform, so the interior optimum is produced by
the data rather than drawn.  The matrix in (c) is linear CKA between generated
feature blocks, computed pairwise.
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

N_CLS, N_SUP = 16, 4          # classes, and classes per superclass
N_SAMP = 2048                 # batch the distributions are averaged over
T_SHOW = (1.0, 2.0, 4.0, 8.0)
T_GRID = np.concatenate([np.linspace(1.0, 4.0, 13),
                         np.linspace(4.5, 16.0, 24)])
ALPHAS = (0.30, 0.60, 0.90)   # weight on the distillation term

ACC_CE = 71.2                 # per cent, student trained on labels alone
GAIN = 6.8                    # accuracy a fully exploited teacher is worth
HARM = 12.0                   # accuracy lost by copying the teacher's mistakes

N_T_LAYER, N_S_LAYER, D_FEAT = 12, 6, 24

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.30, 3.16, 1.70)


def teacher_logits(rng):
    """Logits with a real superclass structure, plus the labels they target."""
    y = rng.integers(0, N_CLS, N_SAMP)
    sup = np.arange(N_CLS) // N_SUP
    z = rng.normal(0.0, 0.9, (N_SAMP, N_CLS))
    z[np.arange(N_SAMP), y] += rng.normal(5.5, 3.2, N_SAMP)
    same = (sup[None, :] == sup[y][:, None])
    z = z + same * rng.normal(2.3, 0.6, (N_SAMP, N_CLS))
    return z, y, sup


def softmax(z, t):
    e = np.exp((z - z.max(axis=1, keepdims=True)) / t)
    return e / e.sum(axis=1, keepdims=True)


def reorder(p, y, sup):
    """Per sample, put the target first, then its siblings, then the rest."""
    out = np.empty_like(p)
    for i in range(p.shape[0]):
        sib = np.where((sup == sup[y[i]]) & (np.arange(N_CLS) != y[i]))[0]
        rest = np.where(sup != sup[y[i]])[0]
        out[i] = p[i, np.concatenate([[y[i]], sib, rest])]
    return out


def transfer_signal(p):
    """How much ordered non-target mass the soft targets expose.

    Two effects fight each other as the temperature rises.  The mass sitting
    on non-target classes grows, which is the information a hard label cannot
    carry; but the ordering inside that mass washes out towards uniform, and
    a uniform tail teaches nothing.  Multiplying the measured mass by the
    measured contrast between the best and the median non-target class turns
    the pair into one number with an interior maximum.
    """
    q = np.sort(p[:, 1:], axis=1)
    best, mid = q[:, -1], np.median(q, axis=1)
    mass = p[:, 1:].sum(axis=1)
    return float((mass * (best - mid) / (best + mid + 1e-12)).mean())


def cka(a, b):
    """Linear centred kernel alignment between two feature blocks."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    num = np.linalg.norm(a.T @ b, ord="fro") ** 2
    den = (np.linalg.norm(a.T @ a, ord="fro")
           * np.linalg.norm(b.T @ b, ord="fro"))
    return float(num / den)


def features(rng, z, a, b, n_layer, noise0, noise1):
    """A depth-ordered stack of feature blocks over one shared latent.

    Both networks see the same inputs and travel the same representational
    arc from basis ``a`` to basis ``b``, so alignment across the two stacks is
    a real measurement; the student is simply noisier at every depth.
    """
    out = []
    for i in range(n_layer):
        u = i / (n_layer - 1.0)
        m = np.cos(0.5 * np.pi * u) * a + np.sin(0.5 * np.pi * u) * b
        s = noise0 + (noise1 - noise0) * u
        out.append(z @ m + rng.normal(0.0, s, (z.shape[0], D_FEAT)))
    return out


def dist_panel(ax, curves, y_true, sup, stats):
    shades = get_palette("cb_sequential_blue", 7)
    cols = [shades[2], shades[3], shades[4], shades[6]]
    x = np.arange(N_CLS)
    ax.axvspan(0.5, N_SUP - 0.5, facecolor="#eef3f8", edgecolor="none",
               zorder=1)
    for (t, mean_p), col in zip(curves, cols):
        ax.plot(x, mean_p, color=col, lw=1.5, marker="o", ms=3.4, mfc=col,
                mec="white", mew=0.5, zorder=5,
                label="T = {:.0f}".format(t))
    ax.plot([0], [1.0], marker="_", ms=13.0, color=GREY, mew=1.8, zorder=4)
    ax.text(0.16, 1.0, "one-hot label", fontsize=6.2, color=SUB, ha="left",
            va="center")
    ax.set_yscale("log")
    ax.set_xlim(-0.6, N_CLS - 0.4)
    ax.set_ylim(2.0e-5, 2.4)
    ax.set_xticks(x)
    ax.set_xticklabels(["target"] + ["sib {:d}".format(i + 1)
                                     for i in range(N_SUP - 1)]
                       + ["c{:d}".format(i + 1)
                          for i in range(N_CLS - N_SUP)], fontsize=6.2)
    ax.set_xlabel("class, ordered by relation to the target "
                  "(n = {:d} samples averaged)".format(N_SAMP))
    ax.set_ylabel("mean teacher probability (log scale)")
    ax.grid(axis="y", which="major", lw=0.5, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(N_SUP / 2.0 - 0.5, 5.0e-5,
            "the {:d} classes sharing a\nsuperclass with the target"
            .format(N_SUP - 1), fontsize=6.2, color=SUB, ha="center",
            va="bottom", linespacing=1.4)

    lo, hi, out_lo, out_hi = stats
    ax.annotate("sibling mass {:.2f} % at T = 1\nrises to {:.1f} % at T = 8"
                .format(100.0 * lo, 100.0 * hi),
                xy=(2.0, curves[-1][1][2]), xytext=(4.4, 0.34),
                fontsize=6.6, color=INK, ha="left", va="center",
                linespacing=1.4, zorder=9,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                shrinkA=2, shrinkB=3))
    ax.annotate("unrelated classes climb from {:.4f} %\nto {:.1f} % over the "
                "same range, which is\nwhy a very high temperature stops "
                "teaching".format(100.0 * out_lo, 100.0 * out_hi),
                xy=(11.0, curves[-1][1][11]), xytext=(6.6, 2.6e-4),
                fontsize=6.6, color=ORANGE, ha="left", va="bottom",
                linespacing=1.4, zorder=9,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8,
                                shrinkA=2, shrinkB=3))
    ax.legend(loc="upper right", fontsize=6.6, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.6, borderaxespad=0.4, labelspacing=0.25,
              ncol=2, columnspacing=1.1)
    ax.set_title("Temperature moves mass onto the classes the teacher "
                 "confuses", fontsize=9.0, fontweight="bold", color=INK,
                 pad=6.0)


def sweep_panel(ax, acc, t_err):
    style = ((BLUE, "-"), (TEAL, "-"), (ORANGE, (0, (4.0, 1.8))))
    for (a, curve), (col, ls) in zip(acc, style):
        ax.plot(T_GRID, curve, color=col, lw=1.4, ls=ls, zorder=5,
                label="loss weight {:.1f}".format(a))
    ax.plot([T_GRID[0], T_GRID[-1]], [ACC_CE, ACC_CE], color=GREY, lw=0.9,
            ls=(0, (3.0, 2.0)), zorder=3)
    ax.text(T_GRID[-1], ACC_CE - 0.35, "labels only {:.1f} %".format(ACC_CE),
            fontsize=6.0, color=MUTED, ha="right", va="top")
    best = max(((a, T_GRID[int(np.argmax(c))], float(np.max(c)))
                for a, c in acc), key=lambda r: r[2])
    ax.plot([best[1]], [best[2]], marker="o", ms=5.0, mfc="none", mec=TEAL,
            mew=1.2, zorder=7)
    ax.annotate("T = {:.1f}, weight {:.1f}\n{:.1f} % (+{:.1f} over labels)"
                .format(best[1], best[0], best[2], best[2] - ACC_CE),
                xy=(best[1], best[2]), xytext=(best[1] + 1.6, best[2] - 1.9),
                fontsize=6.0, color=TEAL, ha="left", va="center",
                linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.7,
                                shrinkA=2, shrinkB=3))
    ax.set_xlim(T_GRID[0], T_GRID[-1])
    ax.set_xlabel("distillation temperature T", fontsize=6.8)
    ax.set_ylabel("student top-1 (%)", fontsize=6.8)
    ax.grid(lw=0.5, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(T_GRID[-1], ax.get_ylim()[0] + 0.25,
            "teacher itself is wrong on {:.1f} % of the batch"
            .format(100.0 * t_err), fontsize=6, color=RED, ha="right",
            va="bottom")
    ax.legend(loc="lower left", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.5, borderaxespad=0.3, labelspacing=0.22)
    ax.set_title("Where the temperature stops paying", fontsize=8.0,
                 fontweight="bold", color=INK)


def cka_panel(fig, ax, mat):
    im = ax.imshow(mat, cmap=get_cmap("cb_sequential_green"), vmin=0.0,
                   vmax=1.0, origin="lower", interpolation="nearest",
                   aspect="equal", extent=(-0.5, N_T_LAYER - 0.5, -0.5,
                                           N_S_LAYER - 0.5))
    for j in range(N_S_LAYER):
        i = int(np.argmax(mat[j]))
        ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1.0, 1.0, fill=False,
                                   edgecolor=RED, linewidth=1.0, zorder=5))
    ax.set_xticks(np.arange(0, N_T_LAYER, 2))
    ax.set_xticklabels([str(i + 1) for i in range(0, N_T_LAYER, 2)],
                       fontsize=6.0)
    ax.set_yticks(np.arange(N_S_LAYER))
    ax.set_yticklabels([str(j + 1) for j in range(N_S_LAYER)], fontsize=6.0)
    ax.set_xlabel("teacher block (12 deep)", fontsize=6.8)
    ax.set_ylabel("student block", fontsize=6.8)
    ax.tick_params(width=0.7, length=2.5)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    cb = safe_colorbar(fig, im, ax, label="linear CKA", width=0.09, pad=0.07,
                       labelsize=6, ticks=[0.0, 0.5, 1.0])
    cb.ax.set_ylabel("linear CKA", fontsize=6)
    ax.set_title("Which teacher block each student block matches",
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
    rng = np.random.default_rng(7715)
    z, y, sup = teacher_logits(rng)
    t_err = float((z.argmax(axis=1) != y).mean())

    curves = [(t, reorder(softmax(z, t), y, sup).mean(axis=0))
              for t in T_SHOW]
    sib_lo = float(curves[0][1][1:N_SUP].sum())
    sib_hi = float(curves[-1][1][1:N_SUP].sum())
    out_lo = float(curves[0][1][N_SUP:].mean())
    out_hi = float(curves[-1][1][N_SUP:].mean())

    sig = np.array([transfer_signal(reorder(softmax(z, t), y, sup))
                    for t in T_GRID])
    sig = sig / sig.max()
    acc = [(a, ACC_CE + a * GAIN * sig - a * HARM * t_err) for a in ALPHAS]

    zz = rng.normal(0.0, 1.0, (512, D_FEAT))
    aa = np.linalg.qr(rng.normal(0.0, 1.0, (D_FEAT, D_FEAT)))[0]
    bb = np.linalg.qr(rng.normal(0.0, 1.0, (D_FEAT, D_FEAT)))[0]
    t_feat = features(rng, zz, aa, bb, N_T_LAYER, 0.15, 0.45)
    s_feat = features(rng, zz, aa, bb, N_S_LAYER, 0.55, 1.20)
    mat = np.array([[cka(s, t) for t in t_feat] for s in s_feat])

    best = max(((a, T_GRID[int(np.argmax(c))], float(np.max(c)))
                for a, c in acc), key=lambda r: r[2])
    print("teacher error {:.1f} %  sibling mass {:.3f} -> {:.3f}"
          .format(100.0 * t_err, sib_lo, sib_hi))
    print("best T {:.1f} weight {:.1f} acc {:.2f} (CE {:.1f})"
          .format(best[1], best[0], best[2], ACC_CE))
    print("peak transfer signal at T = {:.1f}"
          .format(T_GRID[int(np.argmax(sig))]))
    assert 1.0 < T_GRID[int(np.argmax(sig))] < 16.0, "interior optimum"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_d = axes_inches(fig, MAIN)
    ax_s = axes_inches(fig, AUX_T)
    ax_c = axes_inches(fig, AUX_B)

    dist_panel(ax_d, curves, y, sup, (sib_lo, sib_hi, out_lo, out_hi))
    sweep_panel(ax_s, acc, t_err)
    cka_panel(fig, ax_c, mat)

    curve = dict(acc)[best[0]]
    band = T_GRID[curve >= best[2] - 0.25]


    panel_labels_inches(fig, [ax_d, ax_s, ax_c], dx=-0.62, dy=0.14)
    savefig(fig, "1028_eff_distill_soft_targets.png")


if __name__ == "__main__":
    main()

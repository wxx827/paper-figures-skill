"""Ford-style seriation: eight wares, twelve assemblages, one recovered order.

The battleship diagram is the argument.  Each column is one ware and each bar
is that ware's share of the assemblage it sits in, drawn symmetrically about
the column axis, so a ware that appears, spreads and dies leaves the hull
shape the technique is named after.  The staggered peaks are what makes the
sequence a sequence.

The row order is not assumed.  Counts are drawn once from staggered
popularity curves, the twelve assemblages are then shuffled, and the order
plotted here is the one recovered by the first axis of a correspondence
analysis of the count matrix - the ordination archaeologists actually use for
this.  Its agreement with the excavated depth order is reported as a
Spearman coefficient rather than claimed.  The matrix panel shows the same
result a second way: Brainerd-Robinson similarity, permuted into the
recovered order, concentrates on the diagonal.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr

from pubstyle import (set_pub_style, savefig, get_cmap, CATEGORICAL,
                      CATEGORICAL_EXTRA)
from figkit import axes_inches, panel_labels_inches, safe_colorbar

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HAIR = "#cccccc"

FW, FH = 10.4, 6.4
N_LEV, N_TYPE = 12, 8
TCOL = CATEGORICAL + CATEGORICAL_EXTRA[:3]

# (letter, ware name, peak position on the latent sequence, spread)
WARES = [
    ("A", "coarse burnished jar", 0.6, 2.1),
    ("B", "cord-marked bowl",     2.3, 2.0),
    ("C", "red-slipped dish",     4.0, 1.9),
    ("D", "carinated cup",        5.4, 1.7),
    ("E", "painted beaker",       6.8, 1.8),
    ("F", "ring-base bowl",       8.1, 1.9),
    ("G", "grey stamped ware",    9.6, 2.0),
    ("H", "wheel-thrown flask",  11.0, 2.2),
]


def correspondence_axis(X):
    """First correspondence-analysis axis of a counts matrix (rows, cols).

    The standard chi-square decomposition: centre the correspondence matrix
    on the product of its margins, scale both ways by the square root of the
    margins, take the SVD, and read the leading row and column scores.
    """
    P = X / X.sum()
    r, c = P.sum(1), P.sum(0)
    S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    U, sv, Vt = np.linalg.svd(S, full_matrices=False)
    row = U[:, 0] * sv[0] / np.sqrt(r)
    col = Vt[0] / np.sqrt(c) * sv[0]
    inertia = sv ** 2 / (sv ** 2).sum()
    return row, col, float(inertia[0])


def brainerd_robinson(pct):
    d = np.abs(pct[:, None, :] - pct[None, :, :]).sum(-1)
    return 200.0 - d


def main():
    set_pub_style()
    rng = np.random.default_rng(90210)

    depth = np.round(np.linspace(2.35, 0.35, N_LEV), 2)   # deepest first
    codes = [f"L{N_LEV - i:02d}" for i in range(N_LEV)]   # L12 deepest
    lat = np.arange(N_LEV, dtype=float)                   # 0 = deepest

    inten = np.array([[np.exp(-0.5 * ((i - mu) / s) ** 2)
                       for _, _, mu, s in WARES] for i in lat])
    inten = inten * rng.uniform(0.75, 1.3, (1, N_TYPE))
    n_sherd = rng.integers(90, 260, N_LEV)
    counts = np.array([rng.multinomial(n, p / p.sum())
                       for n, p in zip(n_sherd, inten)], dtype=float)
    pct = 100.0 * counts / counts.sum(1, keepdims=True)

    # Forget the excavated order, then recover it.
    shuffle = rng.permutation(N_LEV)
    row_sc, col_sc, iner = correspondence_axis(counts[shuffle])
    if spearmanr(row_sc, lat[shuffle]).statistic < 0:
        row_sc, col_sc = -row_sc, -col_sc          # orient axis 1 with depth
    order = shuffle[np.argsort(row_sc)]
    rho = spearmanr(np.argsort(np.argsort(row_sc)), lat[shuffle]).statistic
    col_order = np.argsort(col_sc)

    pct_s, cnt_s = pct[order], counts[order]
    codes_s = [codes[i] for i in order]
    depth_s = depth[order]
    bmat = brainerd_robinson(pct_s)

    peak_pos = np.array([float(np.argmax(pct_s[:, j])) for j in range(N_TYPE)])
    spans = []
    for j in range(N_TYPE):
        on = np.where(pct_s[:, j] >= 5.0)[0]
        spans.append((float(on.min()), float(on.max())) if on.size
                     else (peak_pos[j], peak_pos[j]))

    fig = plt.figure(figsize=(FW, FH))
    ax_s = axes_inches(fig, (0.92, 1.30, 5.30, 3.95))
    ax_m = axes_inches(fig, (7.05, 3.20, 2.05, 2.05))
    ax_p = axes_inches(fig, (7.05, 1.30, 2.35, 1.45))

    # ---- (a) the battleship diagram ---------------------------------------
    pmax = pct.max()
    half = 0.40 / pmax                       # column half-width per percent
    for k, j in enumerate(col_order):
        letter, ware, _, _ = WARES[j]
        cx = k + 0.5
        ax_s.axvline(cx, color=HAIR, lw=0.5, zorder=1)
        w = pct_s[:, j] * half
        ax_s.barh(np.arange(N_LEV), 2 * w, left=cx - w, height=0.74,
                  color=TCOL[j], edgecolor="white", linewidth=0.4, zorder=3)
        ax_s.fill_betweenx(np.arange(N_LEV), cx - w, cx + w, color=TCOL[j],
                           alpha=0.22, lw=0, zorder=2)
        ax_s.text(cx, N_LEV - 0.30, letter, ha="center", va="bottom",
                  fontsize=8.0, fontweight="bold", color=TCOL[j])
    for k in range(N_TYPE + 1):
        ax_s.axvline(k, color="#e2e2e2", lw=0.6, zorder=0)

    for i in range(N_LEV):
        ax_s.text(N_TYPE + 0.10, i, f"n={int(cnt_s[i].sum())}", ha="left",
                  va="center", fontsize=6.2, color=MUTED)
    # scale key: one 20 % bar drawn in the same units as the columns
    key = 20.0 * half
    ax_s.plot([0.5 - key, 0.5 + key], [-1.15, -1.15], color=RULE, lw=2.6,
              solid_capstyle="butt", clip_on=False, zorder=4)
    ax_s.text(0.5, -1.55, "20 % of assemblage", ha="center", va="center",
              fontsize=6.4, color=SUB, clip_on=False)

    ax_s.set_xlim(0.0, N_TYPE + 0.02)
    ax_s.set_ylim(-1.9, N_LEV - 0.1)
    ax_s.set_xticks([])
    ax_s.set_yticks(range(N_LEV))
    ax_s.set_yticklabels([f"{c}   {d:.2f} m" for c, d in
                          zip(codes_s, depth_s)], fontsize=6.8)
    ax_s.tick_params(axis="y", length=0)
    ax_s.set_ylabel("assemblage in seriated order, deepest at the bottom "
                    "(context, depth)", fontsize=8.0)
    ax_s.set_title("Ware frequencies per assemblage, columns ordered by the "
                   "same ordination", fontsize=9.0, fontweight="bold",
                   loc="left")
    for s in ("top", "right", "bottom"):
        ax_s.spines[s].set_visible(False)

    # ---- (b) Brainerd-Robinson similarity, permuted ------------------------
    im = ax_m.imshow(bmat, cmap=get_cmap("cb_sequential_blue"), vmin=0.0,
                     vmax=200.0, origin="lower", interpolation="nearest")
    ax_m.set_aspect("equal")
    ax_m.set_xticks(range(N_LEV))
    ax_m.set_yticks(range(N_LEV))
    ax_m.set_xticklabels(codes_s, fontsize=6, rotation=90)
    ax_m.set_yticklabels(codes_s, fontsize=6)
    ax_m.tick_params(length=1.8, pad=1.5)
    ax_m.set_title("Similarity concentrates on the diagonal",
                   fontsize=8.5, fontweight="bold", loc="left")
    for s in ax_m.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color(RULE)
    safe_colorbar(fig, im, ax_m, label="Brainerd-Robinson (0-200)",
                  width=0.11, pad=0.08, ticks=[0, 50, 100, 150, 200],
                  labelsize=6.4)

    # ---- (c) currency spans ------------------------------------------------
    for k, j in enumerate(col_order):
        lo, hi = spans[j]
        ax_p.plot([lo, hi], [k, k], color=TCOL[j], lw=4.0,
                  solid_capstyle="butt", alpha=0.55, zorder=2)
        ax_p.plot([peak_pos[j]], [k], marker="o", ms=4.2, color=TCOL[j],
                  mec="white", mew=0.6, zorder=3)
        ax_p.text(hi + 0.25, k, f"{pct_s[int(peak_pos[j]), j]:.0f} %",
                  ha="left", va="center", fontsize=6.0, color=SUB)
    ax_p.set_xlim(-0.7, N_LEV + 1.4)
    ax_p.set_ylim(-0.8, N_TYPE - 0.2)
    ax_p.set_yticks(range(N_TYPE))
    ax_p.set_yticklabels([f"{WARES[j][0]}  {WARES[j][1]}" for j in col_order],
                         fontsize=6.4)
    ax_p.set_xticks(range(0, N_LEV, 2))
    ax_p.set_xticklabels([codes_s[i] for i in range(0, N_LEV, 2)],
                         fontsize=6.4)
    ax_p.tick_params(length=2.2)
    ax_p.set_xlabel("seriated position, deepest at the left", fontsize=8.0)
    ax_p.set_title("Currency spans stagger through the sequence",
                   fontsize=8.5, fontweight="bold", loc="left")
    ax_p.grid(axis="x", lw=0.5, color=HAIR, alpha=0.7)
    ax_p.set_axisbelow(True)
    for s in ("top", "right"):
        ax_p.spines[s].set_visible(False)
    ax_p.text(0.985, 0.03, "bar: ware holds $\\geq$ 5 % of the assemblage\n"
              "dot: its peak share, printed at the right",
              transform=ax_p.transAxes, ha="right", va="bottom", fontsize=6.0,
              color=MUTED, linespacing=1.3)

    panel_labels_inches(fig, [ax_s, ax_m, ax_p], dx=-0.86, dy=0.16)

    print(f"CA axis 1 inertia {iner:.3f}, Spearman vs depth {rho:.3f}")
    print("seriated order:", " ".join(codes_s))
    for k, j in enumerate(col_order):
        print(f"  ware {WARES[j][0]} {WARES[j][1]:22s} peak at "
              f"{codes_s[int(peak_pos[j])]} = {pct_s[int(peak_pos[j]), j]:.1f} %"
              f"  span {spans[j][0]:.0f}-{spans[j][1]:.0f}")
    savefig(fig, "891_arq_artifact_seriation.png")


if __name__ == "__main__":
    main()

"""Three pruning patterns at one matched sparsity, and what a kernel can use.

Main panel: the surviving weights of one 72 x 48 projection block under
unstructured magnitude pruning, 4:8 semi-structured pruning and input-channel
pruning, all three cut to exactly the same sparsity so the only difference is
the shape of the mask; kept cells carry their magnitude on a sequential blue
scale, pruned cells are flat grey.  Auxiliary (both 58 per cent of the main
panel width): top-1 accuracy against sparsity for the three patterns, and the
speedup each one actually returns against the FLOP-count speedup it promises.

Every number is read off the arrays that are drawn.  Sparsity is a count of
zeros in the mask, the retained energy is the Frobenius share of the kept
weights, and the accuracy curve is a fixed monotone function of that measured
energy, so the ordering of the three curves is produced by the masks and not
asserted.  The speedup bars come from a two-term cost model, compute time
against memory time, evaluated for a decode-shaped GEMM where the dense layer
is already memory bound; the gap between promise and delivery is therefore
traffic, not arithmetic, and the per-bar label names the binding term.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, get_cmap, lighten
from figkit import axes_inches, panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
TEAL = "#2a7f7a"
RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3b5b92"
GREY = "#b3b3b3"

N_OUT, N_IN = 96, 64          # weight block that is drawn
M_GROUP = 8                   # N:M group length along the input dimension
S_SHOW = 0.50                 # matched sparsity of the three drawn masks

D_MODEL = 4096                # cost model: one square projection
TOKENS = 64                   # decode-shaped batch, so dense is memory bound
PEAK = 165.0e12               # FLOP/s
BW = 2.0e12                   # byte/s
BYTES_W = 2.0                 # FP16 weights
IDX_BYTES = 2.0               # per surviving weight, CSR-style column index
NM_META = 0.25                # byte per weight, 2-bit 4:8 metadata

ACC_DENSE = 76.4              # per cent top-1 of the unpruned network
ACC_K = 46.0                  # accuracy lost per unit of missing energy

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.14, 3.16, 1.86)


def build_weights(rng):
    """A weight block whose input channels differ a lot in importance."""
    col = np.exp(rng.normal(0.0, 0.75, N_IN))
    row = np.exp(rng.normal(0.0, 0.30, N_OUT))
    return rng.normal(0.0, 1.0, (N_OUT, N_IN)) * col[None, :] * row[:, None]


def mask_unstructured(w, s):
    """Keep the largest magnitudes anywhere in the block."""
    k = int(round((1.0 - s) * w.size))
    idx = np.argsort(np.abs(w).ravel())[::-1][:k]
    m = np.zeros(w.size, dtype=bool)
    m[idx] = True
    return m.reshape(w.shape)


def mask_nm(w, n_keep, m_group=M_GROUP):
    """Keep the n largest of every consecutive group of m input weights."""
    g = np.abs(w).reshape(w.shape[0], w.shape[1] // m_group, m_group)
    order = np.argsort(-g, axis=2)
    m = np.zeros_like(g, dtype=bool)
    np.put_along_axis(m, order[:, :, :n_keep], True, axis=2)
    return m.reshape(w.shape)


def mask_channel(w, s):
    """Drop whole input channels, weakest L2 norm first."""
    drop = int(round(s * w.shape[1]))
    order = np.argsort(np.linalg.norm(w, axis=0))
    m = np.ones(w.shape, dtype=bool)
    m[:, order[:drop]] = False
    return m


def energy(w, m):
    """Share of the Frobenius energy that survives the mask."""
    return float((w[m] ** 2).sum() / (w ** 2).sum())


def accuracy(e):
    """Top-1 implied by the measured retained energy."""
    return ACC_DENSE - ACC_K * (1.0 - e)


def cost(nnz_frac, bytes_per_weight, eff_c, eff_m):
    """Layer time in seconds under a compute / memory two-term model."""
    flops = 2.0 * TOKENS * D_MODEL * D_MODEL * nnz_frac
    wbytes = D_MODEL * D_MODEL * bytes_per_weight
    abytes = 2.0 * TOKENS * D_MODEL * BYTES_W
    t_c = flops / (PEAK * eff_c)
    t_m = (wbytes + abytes) / (BW * eff_m)
    return max(t_c, t_m), t_c, t_m


def mask_panel(fig, ax, w, masks, names, notes):
    vals = np.abs(w)
    vmax = float(np.percentile(vals, 99.0))
    cmap = get_cmap("cb_sequential_blue")
    cmap = cmap.copy()
    cmap.set_bad("#e8ecf1")

    gap = 10.0
    step = N_IN + gap
    im = None
    for i, (m, name, note) in enumerate(zip(masks, names, notes)):
        shown = np.where(m, vals, np.nan)
        x0 = i * step
        im = ax.imshow(shown, cmap=cmap, vmin=0.0, vmax=vmax,
                       extent=(x0, x0 + N_IN, N_OUT, 0.0), origin="upper",
                       interpolation="nearest", zorder=3)
        ax.add_patch(plt.Rectangle((x0, 0.0), N_IN, N_OUT, fill=False,
                                   edgecolor="#8a8a8a", linewidth=0.6,
                                   zorder=6))
        ax.text(x0 + N_IN / 2.0, -13.0, name, ha="center", va="bottom",
                fontsize=7.6, fontweight="bold", color=INK)
        ax.text(x0 + N_IN / 2.0, -3.0, note, ha="center", va="bottom",
                fontsize=6.4, color=SUB, linespacing=1.35)
        kept = int(m.sum())
        ax.text(x0 + N_IN / 2.0, N_OUT + 4.0,
                "sparsity {:.1f} %\n{:d} / {:d} kept\nenergy {:.1f} %"
                .format(100.0 * (1.0 - kept / m.size), kept, m.size,
                        100.0 * energy(w, m)),
                ha="center", va="top", fontsize=6.3, color=MUTED,
                linespacing=1.5)

    ax.set_aspect("equal")
    ax.set_xlim(-5.0, 2 * step + N_IN + 5.0)
    ax.set_ylim(N_OUT + 34.0, -18.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(-4.0, N_OUT + 32.0,
            "{:d} output channels down, {:d} input channels across; one "
            "block of a projection, energy is the retained Frobenius share"
            .format(N_OUT, N_IN),
            ha="left", va="bottom", fontsize=6.2, color=MUTED)
    cb = safe_colorbar(fig, im, ax, label="|weight| (arb. units)",
                       width=0.11, pad=0.10, labelsize=6.2, shrink=0.72)
    cb.set_ticks([0.0, vmax / 2.0, vmax])
    cb.set_ticklabels(["0", "{:.1f}".format(vmax / 2.0),
                       "{:.1f}+".format(vmax)])
    ax.set_title("The same 50 per cent of the weights removed three ways",
                 fontsize=9.0, fontweight="bold", color=INK, pad=8.0)


def curve_panel(ax, sw, curves, ops):
    style = ((BLUE, "-", "o"), (TEAL, "-", "s"), (ORANGE, "-", "^"))
    for (name, s_grid, acc), (col, ls, mk) in zip(curves, style):
        ax.plot(100.0 * np.asarray(s_grid), acc, color=col, lw=1.4, ls=ls,
                marker=mk, ms=2.6, mfc=col, mec="none", zorder=5, label=name)
    ax.plot([8.0, 92.0], [ACC_DENSE, ACC_DENSE], color=GREY, lw=0.9,
            ls=(0, (3.0, 2.0)), zorder=3)
    ax.text(9.0, ACC_DENSE + 0.25, "dense {:.1f} % top-1".format(ACC_DENSE),
            fontsize=6.0, color=MUTED, ha="left", va="bottom")
    for (name, col), (s_op, a_op) in zip(
            zip([c[0] for c in curves], [s[0] for s in style]), ops):
        ax.plot([100.0 * s_op], [a_op], marker="o", ms=5.0, mfc="none",
                mec=col, mew=1.1, zorder=7)
    lo = min(min(c[2]) for c in curves)
    ax.set_xlim(8.0, 92.0)
    ax.set_ylim(lo - 1.6, ACC_DENSE + 1.4)
    ax.set_xlabel("sparsity (% of weights removed)", fontsize=6.8)
    ax.set_ylabel("top-1 accuracy (%)", fontsize=6.8)
    ax.grid(lw=0.5, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(90.0, lo - 1.3, "rings mark the three masks drawn in (a)",
            fontsize=6, color=MUTED, ha="right", va="bottom")
    ax.legend(loc="lower left", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.5, borderaxespad=0.3, labelspacing=0.22)
    ax.set_title("Accuracy paid for each pattern", fontsize=8.0,
                 fontweight="bold", color=INK)


def speed_panel(ax, rows):
    x = np.arange(len(rows), dtype=float)
    wid = 0.34
    theo = [r[1] for r in rows]
    meas = [r[2] for r in rows]
    cols = [RED if m < 1.0 else TEAL for m in meas]
    ax.bar(x - wid / 2.0, theo, width=wid, facecolor="#e6e6e6",
           edgecolor=GREY, linewidth=0.7, zorder=3,
           label="FLOP-count speedup")
    ax.bar(x + wid / 2.0, meas, width=wid,
           facecolor=[lighten(c, 0.30) for c in cols], edgecolor=cols,
           linewidth=0.9, zorder=4, label="modelled kernel speedup")
    ax.plot([-0.55, len(rows) - 0.45], [1.0, 1.0], color=INK, lw=0.8,
            ls=(0, (3.0, 2.0)), zorder=5)
    ax.text(-0.50, 1.03, "dense baseline 1.00x", fontsize=6, color=SUB,
            ha="left", va="bottom")
    for xi, (name, t, m, note) in zip(x, rows):
        ax.text(xi - wid / 2.0, t + 0.04, "{:.2f}x".format(t), fontsize=6.0,
                color=SUB, ha="center", va="bottom")
        ax.text(xi + wid / 2.0, m + 0.04, "{:.2f}x".format(m), fontsize=6.2,
                color=RED if m < 1.0 else TEAL, ha="center", va="bottom",
                fontweight="bold")
        ax.text(xi, -0.30, note, fontsize=6, color=MUTED, ha="center",
                va="top", linespacing=1.35)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=6.4)
    ax.set_xlim(-0.62, len(rows) - 0.38)
    ax.set_ylim(0.0, max(theo) * 1.30)
    ax.set_ylabel("speedup over dense (x)", fontsize=6.8)
    ax.grid(axis="y", lw=0.5, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.3, borderaxespad=0.3, labelspacing=0.22)
    ax.set_title("What the hardware returns", fontsize=8.0,
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
    rng = np.random.default_rng(4271)
    w = build_weights(rng)

    m_uns = mask_unstructured(w, S_SHOW)
    m_nm = mask_nm(w, M_GROUP // 2)
    m_ch = mask_channel(w, S_SHOW)
    masks = [m_uns, m_nm, m_ch]
    for m in masks:
        assert abs(1.0 - m.sum() / m.size - S_SHOW) < 1e-9, "matched sparsity"

    names = ["unstructured", "semi-structured 4:8", "input-channel"]
    notes = ["any weight may go", "4 of every 8 kept", "whole columns go"]

    s_free = np.linspace(0.10, 0.90, 17)
    acc_uns = [accuracy(energy(w, mask_unstructured(w, s))) for s in s_free]
    acc_ch = [accuracy(energy(w, mask_channel(w, s))) for s in s_free]
    n_keep = np.arange(M_GROUP - 1, 0, -1)
    s_nm = 1.0 - n_keep / M_GROUP
    acc_nm = [accuracy(energy(w, mask_nm(w, int(k)))) for k in n_keep]
    curves = [("unstructured", s_free, acc_uns),
              ("semi-structured 4:8", s_nm, acc_nm),
              ("input-channel", s_free, acc_ch)]
    ops = [(S_SHOW, accuracy(energy(w, m))) for m in masks]

    t_dense, tc_d, tm_d = cost(1.0, BYTES_W, 0.92, 0.85)
    kept = 1.0 - S_SHOW
    cases = [("unstructured", kept, kept * (BYTES_W + IDX_BYTES), 0.55, 0.60),
             ("4:8", kept, kept * BYTES_W + NM_META, 0.85, 0.80),
             ("channel", kept, kept * BYTES_W, 0.90, 0.85)]
    rows = []
    for name, frac, bpw, ec, em in cases:
        t, tc, tm = cost(frac, bpw, ec, em)
        lim = "memory bound" if tm >= tc else "compute bound"
        rows.append((name, 1.0 / frac, t_dense / t,
                     "{:s}\n{:.1f} MB moved".format(
                         lim, (D_MODEL * D_MODEL * bpw
                               + 2.0 * TOKENS * D_MODEL * BYTES_W) / 1e6)))
        print("{:14s} theo {:.2f}x  meas {:.2f}x  t {:.1f} us  {:s}"
              .format(name, 1.0 / frac, t_dense / t, t * 1e6, lim))
    print("dense {:.1f} us (compute {:.1f}, memory {:.1f})"
          .format(t_dense * 1e6, tc_d * 1e6, tm_d * 1e6))
    assert rows[0][2] < rows[1][2] < rows[2][2], "delivered speedup ordering"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_m = axes_inches(fig, MAIN)
    ax_c = axes_inches(fig, AUX_T)
    ax_s = axes_inches(fig, AUX_B)

    mask_panel(fig, ax_m, w, masks, names, notes)
    curve_panel(ax_c, s_free, curves, ops)
    speed_panel(ax_s, rows)

    d_acc = ops[0][1] - ops[2][1]


    panel_labels_inches(fig, [ax_m, ax_c, ax_s], dx=-0.62, dy=0.14)
    savefig(fig, "1026_eff_pruning_sparsity_pattern.png")


if __name__ == "__main__":
    main()

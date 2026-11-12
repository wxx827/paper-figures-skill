"""Figure 1170 - Temporal attention over a 16-frame clip.

(a) is a real row-stochastic causal attention matrix: an exponential decay in
lag, a periodic boost at lag 4 (the clip contains a repeating motion cycle) and
an attention sink on the first frame, then each row is normalised.  It is drawn
on a six-step blue ladder rather than a smooth ramp, because a reader compares
cells, not gradients.

(b)-(d) are the three sparsity patterns a practitioner would consider, as
boolean masks over the same 16x16 causal grid, and (e) counts them: every bar
is ``mask.sum()`` over the 136 pairs a full causal matrix contains, so the
percentages and the pictures cannot drift apart.

Canvas note: a square matrix panel fixes the aspect near 0.56 with a 16x16
grid, the E6 exception in DESIGN_SYSTEM section 1.2.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, get_palette, add_panel_label
from figkit import safe_colorbar

INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"
# below 0.92 relative luminance on purpose: a 0.94 card sits exactly on the
# ink-ratio threshold and flips between "all ink" and "all white"
HAIR, CARD = "#d5d5d5", "#e6e9ec"
DEEP, MID = "#134a76", "#5f9dc6"
SEED = 1575  # original id; figure number is 1170
N = 16
LAG = 4                                        # the clip's motion period
STEPS = 6                                      # rungs of the blue ladder
# same rule for the mask background: 0.90, well clear of the threshold
MASK_ON, MASK_OFF = "#2e74a8", "#dde7f0"


def attention():
    """Causal, row-stochastic; decay + periodic recall + a sink on frame 1."""
    rng = np.random.default_rng(SEED)
    i = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    lag = i - j
    A = np.exp(-lag / 3.4)                     # recency
    A = A * (1.0 + 0.95 * (lag % LAG == 0) * (lag > 0))
    A = A + 0.55 * (j == 0)                    # the attention sink
    A = A * (lag >= 0)                         # causal
    A = A * (1.0 + 0.13 * rng.standard_normal((N, N)))
    A = np.clip(A, 0.0, None)
    return A / A.sum(axis=1, keepdims=True)


def patterns():
    i = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    causal = (i - j) >= 0
    local = causal & ((i - j) <= 3)
    strided = causal & (((i - j) % LAG == 0) | ((i - j) <= 1))
    hubs = np.isin(j, [0, 5, 10, 15])
    pooled = causal & (hubs | (i == j))
    return causal, [("local window, $w$ = 4", local),
                    (f"sparse stride {LAG}", strided),
                    ("global pooling, 4 hubs", pooled)]


def main():
    set_pub_style(base_font=8.5)
    A = attention()
    causal, pats = patterns()
    npair = int(causal.sum())

    lag = np.arange(N)[:, None] - np.arange(N)[None, :]
    reach = float((A * lag).sum() / A.sum())            # mean attended lag
    sink = float(A[1:, 0].mean())
    band = float(np.mean([A[i, i - LAG] for i in range(LAG, N)]))
    near = float(np.mean([A[i, i - 1] for i in range(1, N)]))

    cols = get_palette("cb_sequential_blue", STEPS + 1)[1:]
    cmap = mpl.colors.ListedColormap(cols)
    cmap.set_bad("#ffffff")
    # Row 1 has a single causal entry and therefore a weight of 1.00; scaling
    # the ladder to that one cell would flatten every other row into rung one,
    # so the top rung is an over-range bin instead.
    vtop = float(np.percentile(A[causal], 94))
    bounds = np.linspace(0.0, vtop, STEPS)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend="max")
    Am = np.ma.masked_where(~causal, A)

    fig = plt.figure(figsize=(9.6, 5.4))
    # two grids: the matrix keeps its own slot so its colour bar has somewhere
    # to sit without landing on the mask panels' tick labels
    gs_l = fig.add_gridspec(1, 1, left=0.070, right=0.428, top=0.822,
                            bottom=0.112)
    gs_r = fig.add_gridspec(2, 3, height_ratios=[0.80, 1.0], left=0.545,
                            right=0.978, top=0.822, bottom=0.112,
                            wspace=0.290, hspace=0.360)
    axa = fig.add_subplot(gs_l[0, 0])
    axm = [fig.add_subplot(gs_r[0, k]) for k in range(3)]
    axe = fig.add_subplot(gs_r[1, :])

    # ---- (a) the attention matrix -----------------------------------------
    im = axa.imshow(Am, cmap=cmap, norm=norm, interpolation="nearest")
    axa.set_aspect("equal")
    axa.plot([-0.5, N - 0.5], [-0.5, N - 0.5], color="#8fa2b0", lw=0.7,
             ls=(0, (3, 2)), zorder=4)
    axa.add_patch(Rectangle((-0.5, -0.5), 1.0, N, facecolor="none",
                            edgecolor=DEEP, lw=1.1, zorder=5))
    axa.annotate(f"attention sink: every query keeps\n{100 * sink:.1f} % of "
                 f"its mass on frame 1", xy=(0.5, 11.6), xytext=(4.2, 13.6),
                 fontsize=6.6, color=DEEP, ha="left", va="center",
                 linespacing=1.5, zorder=7,
                 bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#b9c9d8",
                           lw=0.6, alpha=0.95),
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=DEEP,
                                 shrinkA=3, shrinkB=3))
    axa.plot([-0.5, N - 1 - LAG + 0.5], [LAG - 0.5, N - 0.5], color=DEEP,
             lw=0.9, ls=(0, (4, 2)), zorder=4)
    axa.annotate(f"lag-{LAG} band: {100 * band:.1f} % per query,\nagainst "
                 f"{100 * near:.1f} % on the previous frame",
                 xy=(6.0, 10.0), xytext=(6.6, 4.4), fontsize=6.6, color=DEEP,
                 ha="left", va="center", linespacing=1.5, zorder=7,
                 bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#b9c9d8",
                           lw=0.6, alpha=0.95),
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=DEEP,
                                 shrinkA=3, shrinkB=3))
    axa.set_xticks(np.arange(0, N, 3))
    axa.set_xticklabels([str(v + 1) for v in range(0, N, 3)], fontsize=7.0)
    axa.set_yticks(np.arange(0, N, 3))
    axa.set_yticklabels([str(v + 1) for v in range(0, N, 3)], fontsize=7.0)
    axa.set_xlabel("key frame $j$", fontsize=8.2)
    axa.set_ylabel("query frame $i$", fontsize=8.2)
    axa.tick_params(length=2.4)
    for s in axa.spines.values():
        s.set_linewidth(0.7)
        s.set_color("#333333")
    axa.set_title(f"Learned attention, {N} frames, rows sum to 1",
                  fontsize=9.2, fontweight="bold", pad=4.0, color=INK,
                  loc="left")
    add_panel_label(axa, "(a)", x=-0.135, y=1.005, size=9.5)

    # fig.colorbar(ax=...) would carve the bar out of the nominal slot, which
    # on an equal-aspect panel parks it an inch away (DESIGN_SYSTEM 12.4)
    cb = safe_colorbar(fig, im, axa, label="attention weight", width=0.10,
                       pad=0.055, ticks=bounds, extend="max", labelsize=6.4)
    cb.ax.set_yticklabels([f"{b:.2f}" for b in bounds])

    # ---- (b) (c) (d) the three sparsity patterns --------------------------
    pcmap = mpl.colors.ListedColormap([MASK_OFF, MASK_ON])
    for k, (ax, (name, m)) in enumerate(zip(axm, pats)):
        ax.imshow(np.where(causal, m.astype(float), np.nan), cmap=pcmap,
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_aspect("equal")
        ax.plot([-0.5, N - 0.5], [-0.5, N - 0.5], color="#b3b3b3", lw=0.6,
                zorder=3)
        ax.set_xticks([0, N - 1])
        ax.set_xticklabels(["1", str(N)], fontsize=6.4)
        ax.set_yticks([0, N - 1])
        ax.set_yticklabels(["1", str(N)], fontsize=6.4)
        ax.tick_params(length=1.8, pad=1.2)
        for s in ax.spines.values():
            s.set_linewidth(0.6)
            s.set_color("#9a9a9a")
        ax.set_title(name, fontsize=7.6, pad=3.2, color=INK)
        add_panel_label(ax, f"({'bcd'[k]})", x=-0.20, y=1.03, size=9.0)
    axm[0].set_ylabel("query $i$ / key $j$", fontsize=6.6, labelpad=2.0)

    # ---- (e) how much of the causal matrix each pattern keeps -------------
    axe.set_facecolor(CARD)
    names = ["full causal"] + [n for n, _ in pats]
    keep = np.array([npair] + [int(m.sum()) for _, m in pats], dtype=float)
    pct = 100.0 * keep / npair
    bcol = ["#9cc3dd", MASK_ON, "#2a7f7a", "#6b5fa8"]
    xs = np.arange(len(names))
    axe.bar(xs, keep, width=0.56, color=bcol, edgecolor="white", linewidth=0.6,
            zorder=3)
    for x, v, p in zip(xs, keep, pct):
        axe.text(x, v + 0.035 * npair, f"{int(v)} pairs\n{p:.0f} % of full",
                 ha="center", va="bottom", fontsize=6.6, color=SUB,
                 linespacing=1.45, zorder=6,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                           alpha=0.92))
    axe.set_xticks(xs)
    axe.set_xticklabels(["full\ncausal", "local\nwindow", "sparse\nstride",
                         "global\npooling"], fontsize=7.0, linespacing=1.4)
    axe.set_ylim(0, npair * 1.42)
    axe.set_yticks([0, 40, 80, 120])
    axe.set_ylabel("attended pairs", fontsize=7.6)
    axe.tick_params(labelsize=7.0, length=2.2)
    axe.grid(True, axis="y", lw=0.45, color="#e2e2e2")
    axe.set_axisbelow(True)
    axe.set_title(f"Each mask counted over the {npair} pairs of a full "
                  f"{N}x{N} causal matrix", fontsize=8.2, fontweight="bold",
                  pad=4.0, color=INK, loc="left")
    add_panel_label(axe, "(e)", x=-0.072, y=1.005, size=9.5)

    # ---- headline ----------------------------------------------------------
    savefig(fig, "1170_wm_vid_temporal_attention.png")


if __name__ == "__main__":
    main()

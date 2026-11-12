"""Figure 472 - Language-model perplexity across scale, domain and length.

Main panel: perplexity against parameter count on log-log axes, one curve per
training-token budget, with the fitted power-law slope and the irreducible
floor marked.  Subordinate column: the same models scored per text domain as a
heat map, and perplexity against sentence length.

Every number on the figure comes out of one closed-form loss surface
``L = E + A N^-a + B D^-b``; the curves, the heat map cells, the fitted slope
and the length bins are all evaluations of that single function, so they can
never disagree with one another.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import (set_pub_style, savefig, get_cmap, lighten, darken,
                      annotate_box, add_colorbar)
from figkit import panel_labels_inches

# ---------------------------------------------------------------------------
# Shared NLP-batch vocabulary (464-473)
# ---------------------------------------------------------------------------
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d9d9d9"
ACCENT = "#b23a48"

MONO = {"fontfamily": "monospace"}

# Chinchilla-shaped loss surface, in nats per token
E_FLOOR, A_N, ALPHA, B_D, BETA = 1.69, 406.4, 0.34, 410.7, 0.28
BUDGETS = [(3e9, "3 B tokens", "#c9a227"), (1e10, "10 B", "#2e7d5b"),
           (3e10, "30 B", "#3b5b92"), (1e11, "100 B", "#b23a48")]
MODELS = [("124 M", 1.24e8), ("355 M", 3.55e8), ("1.3 B", 1.3e9),
          ("6.7 B", 6.7e9)]
DOMAINS = [("code", 0.71), ("news", 0.92), ("web", 1.00), ("books", 1.08),
           ("dialogue", 1.22), ("legal", 1.35)]
REF_D = 3e10


def loss(n, d):
    """Cross-entropy in nats for ``n`` parameters and ``d`` training tokens."""
    return E_FLOOR + A_N * np.asarray(n, float) ** -ALPHA + \
        B_D * np.asarray(d, float) ** -BETA


def main():
    set_pub_style(base_font=8)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.62],
                          height_ratios=[1.0, 0.62],
                          left=0.070, right=0.930, top=0.900, bottom=0.088,
                          hspace=0.46, wspace=0.30)
    axm = fig.add_subplot(gs[:, 0])
    axh = fig.add_subplot(gs[0, 1])
    axl = fig.add_subplot(gs[1, 1])

    # ---- (a) perplexity against model size --------------------------------
    n_grid = np.logspace(7.0, 11.0, 160)
    for d, lab, col in BUDGETS:
        ppl = np.exp(loss(n_grid, d))
        axm.loglog(n_grid, ppl, lw=1.7, color=col, zorder=4)
        axm.text(n_grid[-1] * 1.10, ppl[-1], lab, fontsize=7.0, va="center",
                 ha="left", color=darken(col, 0.15), fontweight="bold")
    floor = np.exp(E_FLOOR)
    axm.axhline(floor, color="#808080", lw=0.9, ls=(0, (3.2, 1.8)), zorder=3)
    axm.text(n_grid[-1] * 3.2, floor * 1.06,
             f"irreducible floor, exp(E) = {floor:.2f}", fontsize=6.8,
             color=MUTED, ha="right", va="bottom")

    for name, n in MODELS:
        p = float(np.exp(loss(n, REF_D)))
        axm.plot([n], [p], marker="o", ms=5.0, mfc="white", mec="#3b5b92",
                 mew=1.3, zorder=6)
        axm.annotate(name, xy=(n, p), xytext=(0, -12),
                     textcoords="offset points", ha="center", va="top",
                     fontsize=6.6, color=SUB, zorder=7, **MONO,
                     bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                               edgecolor="none", alpha=0.88))

    mid = (n_grid > 3e8) & (n_grid < 3e10)
    slope = np.polyfit(np.log10(n_grid[mid]),
                       np.log10(np.exp(loss(n_grid[mid], REF_D))), 1)[0]
    axm.set_xlabel("non-embedding parameters (-)", fontsize=8.0)
    axm.set_ylabel("validation perplexity (-)", fontsize=8.0)
    axm.tick_params(labelsize=7.0, length=2.5)
    axm.grid(True, which="major", color=GRID, linewidth=0.6, zorder=0)
    axm.set_xlim(n_grid[0], n_grid[-1] * 3.4)
    axm.set_title("Perplexity falls as a power law until the floor bites",
                  fontsize=9.0, fontweight="bold")
    annotate_box(axm, f"fitted slope over 0.3-30 B\nd log ppl / d log N = "
                      f"{slope:+.3f}", loc="lower left", fontsize=6.6)

    # ---- (b) perplexity by domain -----------------------------------------
    M = np.array([[np.exp(loss(n, REF_D)) * f for _dn, f in DOMAINS]
                  for _mn, n in MODELS])
    cmap = get_cmap("cb_sequential_blue")
    im = axh.imshow(M, cmap=cmap, aspect="auto", interpolation="nearest",
                    vmin=M.min(), vmax=M.max(), zorder=3)
    axh.set_xticks(range(len(DOMAINS)))
    axh.set_xticklabels([d for d, _f in DOMAINS], fontsize=6.5, rotation=32,
                        ha="right", **MONO)
    axh.set_yticks(range(len(MODELS)))
    axh.set_yticklabels([m for m, _n in MODELS], fontsize=6.5, **MONO)
    axh.tick_params(length=1.8, width=0.6)
    ink_switch = M.min() + 0.45 * (M.max() - M.min())
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            axh.text(j, i, f"{v:.1f}", ha="center", va="center",
                     fontsize=6.0, zorder=4, **MONO,
                     color="white" if v > ink_switch else SUB)
    for s in axh.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#b8b8b8")
    axh.set_title("Perplexity by text domain", fontsize=8.5,
                  fontweight="bold", pad=3.0)
    add_colorbar(fig, im, axh, label="perplexity (-)", aspect=9)

    # ---- (c) perplexity by sentence length --------------------------------
    bins = np.array([4, 8, 12, 18, 26, 36, 50, 70])
    for (name, n), col in zip(MODELS[1:], ["#c9a227", "#2e7d5b", "#b23a48"]):
        base = float(np.exp(loss(n, REF_D)))
        curve = base * (1.0 + 1.85 * np.exp(-bins / 9.0))
        axl.plot(bins, curve, lw=1.5, color=col, marker="o", ms=3.2,
                 mfc="white", mec=col, mew=1.0, zorder=4, label=name)
    axl.set_xscale("log")
    axl.set_xticks(bins)
    axl.set_xticklabels([str(b) for b in bins], fontsize=6.5)
    axl.minorticks_off()
    axl.set_xlabel("sentence length (tokens)", fontsize=7.0)
    axl.set_ylabel("perplexity (-)", fontsize=7.0)
    axl.tick_params(labelsize=6.5, length=2.2)
    axl.grid(color=GRID, linewidth=0.6, zorder=0)
    axl.legend(loc="upper right", fontsize=6.8, frameon=False,
               handlelength=1.4, borderaxespad=0.2)
    axl.set_title("Short sentences are the hardest", fontsize=8.5,
                  fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        axl.spines[s].set_visible(False)

    gain = 1.0 - np.exp(loss(MODELS[-1][1], BUDGETS[-1][0])) / np.exp(
        loss(MODELS[0][1], BUDGETS[0][0]))
    fx = MODELS[-1][1] / MODELS[0][1]
    fd = BUDGETS[-1][0] / BUDGETS[0][0]
    panel_labels_inches(fig, [axm, axh, axl], dx=-0.52, dy=0.10)
    savefig(fig, "472_nlp_perplexity_compare.png")


if __name__ == "__main__":
    main()

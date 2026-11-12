"""Figure 319 - Where a top-2 mixture-of-experts router actually sends tokens.

The main panel is the router's average gate probability for each token class,
with the winning expert outlined; read down a column and you can see which
experts have become generalists.  The two supporting panels quantify the cost
of that specialisation: the load bars show how far each expert is from the
1/E ideal, and the co-activation matrix shows which pairs are almost always
selected together, which is what makes expert-parallel sharding awkward.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, style_axes, get_cmap

BLUE = "#3b5b92"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
FAINT = "#8a8a8a"

FIG_W, FIG_H = 10.4, 6.4
MAIN = [0.098, 0.115, 0.395, 0.755]
MAIN_CB = [0.505, 0.115, 0.011, 0.755]
LOAD = [0.630, 0.575, 0.330, 0.295]
OVER = [0.630, 0.115, 0.222, 0.295]
OVER_CB = [0.862, 0.115, 0.011, 0.295]

CLASSES = [
    ("punctuation", 0.121), ("whitespace", 0.104), ("function words", 0.163),
    ("nouns (en)", 0.118), ("verbs (en)", 0.087), ("numerals", 0.052),
    ("code / brackets", 0.078), ("math symbols", 0.034),
    ("Chinese", 0.071), ("Cyrillic", 0.043), ("URLs", 0.061),
    ("rare / OOV", 0.068),
]
N_E = 12
GENERALIST = 4                      # expert 5 takes a slice of every class
# (primary, secondary, tertiary) expert per token class
ROUTES = [(0, 1, 7), (0, 2, 8), (1, 3, 6), (3, 4, 1), (3, 5, 0), (6, 7, 10),
          (6, 0, 8), (6, 8, 5), (9, 10, 2), (9, 4, 11), (11, 6, 9),
          (11, 8, 3)]


def gates(rng):
    g = rng.uniform(0.010, 0.032, (len(CLASSES), N_E))
    g[:, GENERALIST] += rng.uniform(0.045, 0.095, len(CLASSES))
    for r, (p, s, t) in enumerate(ROUTES):
        g[r, p] += rng.uniform(0.46, 0.62)
        g[r, s] += rng.uniform(0.15, 0.24)
        g[r, t] += rng.uniform(0.06, 0.12)
    return g / g.sum(axis=1, keepdims=True)


def coactivation(g, rng, n=6000):
    """P(expert j also in the top-2 | expert i in the top-2)."""
    w = np.array([f for _, f in CLASSES])
    joint = np.zeros((N_E, N_E))
    for r in range(len(CLASSES)):
        logits = np.log(g[r])[None, :] + rng.gumbel(0, 0.62, (n, N_E))
        top2 = np.argpartition(-logits, 2, axis=1)[:, :2]
        for a, b in top2:
            joint[a, b] += w[r]
            joint[b, a] += w[r]
            joint[a, a] += w[r]
            joint[b, b] += w[r]
    m = joint / np.clip(np.diag(joint), 1e-9, None)[:, None]
    np.fill_diagonal(m, np.nan)
    return np.clip(m, 0.0, 1.0)


def main():
    set_pub_style(base_font=9)
    rng = np.random.default_rng(319)
    g = gates(rng)
    freq = np.array([f for _, f in CLASSES])
    load = (g * freq[:, None]).sum(axis=0)
    load = load / load.sum() * 100.0
    ideal = 100.0 / N_E
    cv = load.std() / load.mean()

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    cmap = get_cmap("cb_sequential_blue")
    norm = Normalize(vmin=0.0, vmax=0.55)

    # ---- routing heatmap -------------------------------------------------
    ax = fig.add_axes(MAIN)
    ax.imshow(g, cmap=cmap, norm=norm, aspect="auto", origin="upper",
              interpolation="nearest",
              extent=[-0.5, N_E - 0.5, len(CLASSES) - 0.5, -0.5])
    for x in np.arange(0.5, N_E - 1, 1.0):
        ax.axvline(x, color="white", lw=0.5, alpha=0.7)
    for y in np.arange(0.5, len(CLASSES) - 1, 1.0):
        ax.axhline(y, color="white", lw=0.5, alpha=0.7)
    for r in range(len(CLASSES)):
        c = int(np.argmax(g[r]))
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="none",
                               edgecolor=ACCENT, linewidth=1.3, zorder=5))
        ax.text(c, r, f"{g[r, c]:.2f}", ha="center", va="center",
                fontsize=6.0, color="white", fontweight="bold", zorder=6)

    ax.set_xticks(range(N_E))
    ax.set_xticklabels([f"E{i + 1}" for i in range(N_E)], fontsize=7.0)
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels([c for c, _ in CLASSES], fontsize=7.0)
    ax.set_xlabel("expert id", fontsize=8.0, labelpad=2.5)
    ax.set_ylabel("token class", fontsize=8.0, labelpad=2.5)
    ax.tick_params(length=2.5)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color("#333333")
    ax.set_title("Mean gate probability, layer 16 router",
                 fontsize=8.5, fontweight="bold", pad=4.0)
    ax.text(0.995, -0.055, "red outline = top-1 expert",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.4,
            color=ACCENT)

    cax = fig.add_axes(MAIN_CB)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      ticks=[0, 0.25, 0.5])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=8.0, width=0.7, length=2.5, color="#333333")
    cb.set_label("gate probability (-)", fontsize=8.0, color=INK, labelpad=3.0)

    # ---- expert load -----------------------------------------------------
    axl = fig.add_axes(LOAD)
    cols = [ACCENT if v > 1.45 * ideal else
            ("#c9c9c9" if v < 0.55 * ideal else BLUE) for v in load]
    axl.bar(np.arange(N_E), load, color=cols, width=0.72, lw=0, zorder=3)
    axl.axhline(ideal, color=SUB, ls=(0, (4, 2.5)), lw=1.0, zorder=4)
    axl.text(N_E - 0.22, ideal, "1/E", fontsize=6.4, color=SUB, ha="left",
             va="center", clip_on=False)
    for i, v in enumerate(load):
        if v > 1.45 * ideal or v < 0.55 * ideal:
            axl.text(i, v + 0.35, f"{v:.1f}", ha="center", va="bottom",
                     fontsize=6.0, color=ACCENT if v > ideal else SUB,
                     fontweight="bold" if v > ideal else "normal")
    axl.set_xticks(range(N_E))
    axl.set_xticklabels([f"E{i + 1}" for i in range(N_E)], fontsize=6.5)
    axl.set_ylim(0, max(load) * 1.28)
    axl.set_xlim(-0.7, N_E - 0.3)
    axl.set_ylabel("share of routed tokens (%)", fontsize=7.5, labelpad=2.0)
    axl.tick_params(labelsize=6.8, length=2.2)
    axl.grid(True, axis="y", lw=0.5, color="#e6e6e6")
    axl.set_axisbelow(True)
    style_axes(axl)
    axl.set_title("Expert load is far from uniform", fontsize=8.5,
                  fontweight="bold", pad=4.0)
    axl.text(0.015, 0.955, f"CV = {cv:.2f}   ·   max/min = "
             f"{load.max() / load.min():.1f}x", transform=axl.transAxes,
             ha="left", va="top", fontsize=6.6, fontweight="bold", color=INK,
             bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                       edgecolor="#dcdcdc", linewidth=0.6, alpha=0.95))

    # ---- co-activation ---------------------------------------------------
    axo = fig.add_axes(OVER)
    co = coactivation(g, rng)
    cmap2 = get_cmap("cb_sequential_purple").copy()
    cmap2.set_bad("#ededed")
    norm2 = Normalize(vmin=0.0, vmax=1.0)
    axo.imshow(co, cmap=cmap2, norm=norm2, interpolation="nearest",
               extent=[-0.5, N_E - 0.5, N_E - 0.5, -0.5], aspect="auto")
    for k in np.arange(0.5, N_E - 1, 1.0):
        axo.axvline(k, color="white", lw=0.4, alpha=0.6)
        axo.axhline(k, color="white", lw=0.4, alpha=0.6)
    axo.set_xticks(range(N_E))
    axo.set_yticks(range(N_E))
    axo.set_xticklabels([f"{i + 1}" for i in range(N_E)], fontsize=6.2)
    axo.set_yticklabels([f"{i + 1}" for i in range(N_E)], fontsize=6.2)
    axo.set_xlabel("expert id", fontsize=7.5, labelpad=1.5)
    axo.set_ylabel("expert id", fontsize=7.5, labelpad=1.5)
    axo.tick_params(length=2.0)
    for s in axo.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color("#333333")
    axo.set_title("Top-2 co-activation", fontsize=8.5, fontweight="bold",
                  pad=4.0)

    cax2 = fig.add_axes(OVER_CB)
    cb2 = fig.colorbar(ScalarMappable(norm=norm2, cmap=cmap2), cax=cax2,
                       ticks=[0, 0.5, 1.0])
    cb2.outline.set_linewidth(0.6)
    cb2.outline.set_edgecolor("#333333")
    cb2.ax.tick_params(labelsize=8.0, width=0.7, length=2.5, color="#333333")
    cb2.set_label("co-selection rate (-)", fontsize=8.0, color=INK,
                  labelpad=3.0)

    for tag, rect, dy in (("(a)", MAIN, 0.06), ("(b)", LOAD, 0.06),
                          ("(c)", OVER, 0.06)):
        fig.text(rect[0] - 0.36 / FIG_W, rect[1] + rect[3] + dy / FIG_H, tag,
                 fontsize=11, fontweight="bold", ha="left", va="bottom")

    savefig(fig, "319_llm_moe_routing.png")


if __name__ == "__main__":
    main()

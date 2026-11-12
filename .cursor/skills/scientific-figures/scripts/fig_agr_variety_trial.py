"""Figure 401 - Multi-environment variety trial: yield, GGE biplot, stability.

Twelve genotypes tested across nine environments.  The main panel is the yield
matrix with both margins reordered by Ward clustering, which is what exposes the
two mega-environments; the GGE biplot shows the same matrix in two components,
and the two lower panels turn the interaction term into a per-genotype
stability score.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

from pubstyle import (set_pub_style, savefig, get_cmap, add_colorbar,
                      add_panel_label)

SEED = 6613
N_G, N_E = 12, 9
ACCENT = "#b23a48"
MEGA = ["#3b5b92", "#c0603f"]        # two mega-environments
HEAT_CMAP = get_cmap("cb_sequential_green")


def simulate(rng):
    """Yield matrix (environments x genotypes) with a two-cluster GE pattern."""
    g = rng.normal(0.0, 0.55, N_G)                     # genotype main effect
    e = rng.normal(0.0, 1.15, N_E)                     # environment effect
    mega = (rng.random(N_E) < 0.5).astype(int)
    if mega.sum() in (0, N_E):
        mega[:4] = 0
        mega[4:] = 1
    # Genotypes have a crossover response: some win in mega-env 0, some in 1.
    cross = rng.normal(0.0, 1.15, N_G)
    inter = np.outer(np.where(mega == 0, -1.0, 1.0), cross)
    y = (7.9 + e[:, None] + g[None, :] + inter
         + rng.normal(0.0, 0.28, (N_E, N_G)))
    return np.clip(y, 2.0, None), mega


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)
    y, mega = simulate(rng)

    # Cluster the *response pattern*, not the yield level: without centring
    # each environment first, Ward simply groups high-yielding sites together
    # and the mega-environments never appear.
    z = (y - y.mean(axis=1, keepdims=True)) / y.std(axis=1, keepdims=True)
    ro = leaves_list(linkage(pdist(z, "euclidean"), "ward"))
    co = leaves_list(linkage(pdist(z.T, "euclidean"), "ward"))
    ys = y[np.ix_(ro, co)]
    mega_s = mega[ro]
    g_names = [f"G{c + 1:02d}" for c in co]
    e_names = [f"E{r + 1}" for r in ro]

    # Interaction term and Wricke's ecovalence, in the plotted column order.
    grand = ys.mean()
    ge = ys - ys.mean(axis=1, keepdims=True) - ys.mean(axis=0,
                                                       keepdims=True) + grand
    weco = (ge ** 2).sum(axis=0)
    g_mean = ys.mean(axis=0)

    # GGE biplot on the environment-centred matrix.
    yc = ys - ys.mean(axis=1, keepdims=True)
    u, s, vt = np.linalg.svd(yc, full_matrices=False)
    expl = s ** 2 / (s ** 2).sum()
    sc_e = u[:, :2] * np.sqrt(s[:2])
    sc_g = vt[:2, :].T * np.sqrt(s[:2])

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.62],
                          height_ratios=[1.0, 0.45],
                          left=0.062, right=0.985, top=0.878, bottom=0.095,
                          wspace=0.28, hspace=0.36)
    ax_h = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[1, 0], sharex=ax_h)
    ax_q = fig.add_subplot(gs[1, 1])

    # ---- (a) yield matrix --------------------------------------------------
    vlo, vhi = float(np.floor(ys.min())), float(np.ceil(ys.max()))
    im = ax_h.imshow(ys, cmap=HEAT_CMAP, aspect="auto", vmin=vlo, vmax=vhi)
    cut = vlo + 0.58 * (vhi - vlo)
    for i in range(N_E):
        for j in range(N_G):
            v = ys[i, j]
            ax_h.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.0,
                      color="white" if v > cut else "#1a1a1a")
    ax_h.set_xticks(range(N_G))
    ax_h.set_xticklabels(g_names, fontsize=6.8)
    ax_h.set_yticks(range(N_E))
    ax_h.set_yticklabels(e_names, fontsize=6.8)
    for tick, m in zip(ax_h.get_yticklabels(), mega_s):
        tick.set_color(MEGA[m])
        tick.set_fontweight("bold")
    ax_h.set_ylabel("environment (Ward order)", fontsize=8.0)
    ax_h.tick_params(length=2.0)
    plt.setp(ax_h.get_xticklabels(), visible=False)
    ax_h.set_title("Grain yield by genotype and environment",
                   fontsize=9.0, fontweight="bold")

    # Separator between the two mega-environment blocks.
    brk = np.where(np.diff(mega_s) != 0)[0]
    for b in brk:
        ax_h.axhline(b + 0.5, color="#1a1a1a", lw=1.4)
    add_colorbar(fig, im, ax_h, label="grain yield (t/ha)")

    # ---- (b) GGE biplot ----------------------------------------------------
    lim = 1.18 * max(np.abs(sc_g).max(), np.abs(sc_e).max())
    ax_b.axhline(0, color="#cccccc", lw=0.7)
    ax_b.axvline(0, color="#cccccc", lw=0.7)
    for k, (px, py) in enumerate(sc_e):
        ax_b.annotate("", xy=(px, py), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="-|>", lw=0.9,
                                      color=MEGA[mega_s[k]], shrinkA=0,
                                      shrinkB=0))
        ax_b.text(px * 1.10, py * 1.10, e_names[k], fontsize=6.4,
                  color=MEGA[mega_s[k]], ha="center", va="center")
    best = int(np.argmax(g_mean - 0.35 * np.sqrt(weco)))
    ax_b.scatter(sc_g[:, 0], sc_g[:, 1], s=[52 if i == best else 26
                                            for i in range(N_G)],
                 c=[ACCENT if i == best else "#2e7d5b" for i in range(N_G)],
                 marker="o", edgecolor="white", linewidth=0.6, zorder=5)
    for k, (px, py) in enumerate(sc_g):
        ax_b.text(px, py - 0.055 * lim, g_names[k], fontsize=6.2,
                  ha="center", va="top",
                  color=ACCENT if k == best else "#333333")
    ax_b.set_xlim(-lim, lim)
    ax_b.set_ylim(-lim, lim)
    ax_b.set_aspect("equal")
    ax_b.set_xlabel(f"PC1  ({expl[0]:.0%} of G+GE)", fontsize=7.6)
    ax_b.set_ylabel(f"PC2  ({expl[1]:.0%} of G+GE)", fontsize=7.6)
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_title("GGE biplot", fontsize=8.5, fontweight="bold")

    # ---- (c) stability bars ------------------------------------------------
    cols = [ACCENT if i == best else "#8a9a86" for i in range(N_G)]
    ax_s.bar(np.arange(N_G), weco, width=0.66, color=cols, edgecolor="white",
             linewidth=0.6, zorder=3)
    ax_s.axhline(weco.mean(), color="#808080", lw=0.8, ls=(0, (3.0, 2.0)),
                 zorder=2)
    ax_s.text(N_G - 0.55, weco.mean() + 0.03 * weco.max(), "mean", ha="right",
              va="bottom", fontsize=6.4, color="#777777")
    ax_s.set_xticks(range(N_G))
    ax_s.set_xticklabels(g_names, fontsize=6.8)
    ax_s.set_xlim(-0.65, N_G - 0.35)
    ax_s.set_ylabel("Wricke ecovalence\n(t/ha)$^2$", fontsize=7.6)
    ax_s.set_xlabel("genotype (Ward order)", fontsize=8.0)
    ax_s.tick_params(labelsize=6.8)
    ax_s.grid(axis="y", lw=0.5, alpha=0.5)
    ax_s.set_title("Contribution to genotype x environment interaction",
                   fontsize=8.5, fontweight="bold")

    # ---- (d) mean versus stability ----------------------------------------
    stab = np.sqrt(weco)
    ax_q.add_patch(plt.Rectangle((g_mean.mean(), -1.0),
                                 g_mean.max() - g_mean.mean() + 1.0,
                                 stab.mean() + 1.0, facecolor="#2e7d5b",
                                 alpha=0.09, edgecolor="none", zorder=0))
    ax_q.axhline(stab.mean(), color="#808080", lw=0.7, ls=(0, (3.0, 2.0)),
                 zorder=1)
    ax_q.axvline(g_mean.mean(), color="#808080", lw=0.7, ls=(0, (3.0, 2.0)),
                 zorder=1)
    ax_q.scatter(g_mean, stab, s=[46 if i == best else 22
                                  for i in range(N_G)],
                 c=[ACCENT if i == best else "#2e7d5b" for i in range(N_G)],
                 edgecolor="white", linewidth=0.6, zorder=4)
    for k in range(N_G):
        ax_q.text(g_mean[k], stab[k] + 0.035 * stab.max(), g_names[k],
                  fontsize=6.0, ha="center", va="bottom",
                  color=ACCENT if k == best else "#555555")
    ax_q.set_xlabel("mean yield across environments (t/ha)", fontsize=7.6)
    ax_q.set_ylabel("instability\n$\\sqrt{W}$ (t/ha)", fontsize=7.6)
    ax_q.set_xlim(g_mean.min() - 0.45, g_mean.max() + 0.45)
    ax_q.set_ylim(-0.05 * stab.max(), stab.max() * 1.22)
    ax_q.tick_params(labelsize=6.6)
    ax_q.text(0.975, 0.045, "high yield,\nlow instability", ha="right",
              va="bottom", transform=ax_q.transAxes, fontsize=6.4,
              color="#1f6a47", linespacing=1.4)
    ax_q.set_title("Selection quadrants", fontsize=8.5, fontweight="bold")

    for ax in (ax_b, ax_s, ax_q):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    add_panel_label(ax_h, "(a)", x=-0.075, y=1.02)
    add_panel_label(ax_b, "(b)", x=-0.145, y=1.02)
    add_panel_label(ax_s, "(c)", x=-0.075, y=1.05)
    add_panel_label(ax_q, "(d)", x=-0.145, y=1.05)


    savefig(fig, "401_agr_variety_trial.png")


if __name__ == "__main__":
    main()

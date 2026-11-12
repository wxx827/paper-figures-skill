"""Over-smoothing: deeper propagation makes every node look like every other.

Left, the pairwise cosine-similarity matrix of the node representations after
2, 4, 8 and 16 rounds of symmetric-normalised propagation, nodes ordered by
community so the block structure is visible.  All four share one diverging
scale, which is the whole point - at L = 2 the off-diagonal blocks are still
blue, by L = 16 the matrix is uniformly saturated.

Right, the same run measured two ways: the Dirichlet energy (a Rayleigh
quotient, so it is scale-free) on a log axis, and the accuracy of a
nearest-class-mean read-out under a fixed absolute noise floor.  Shallow
propagation denoises and accuracy improves; past the peak the signal decays
faster than the noise and the read-out collapses toward chance.
"""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, get_cmap, add_panel_label
from figkit import shared_norm

CLASS_COLORS = ["#b23a48", "#3b5b92", "#2e7d5b"]     # house序位 1, 2, 3
ENERGY_C = "#3b5b92"
ACC_C = "#b23a48"
INK, SUB, MUTED, RULE = "#1a1a1a", "#555555", "#777777", "#333333"

N_PER, N_CLASS, DIM = 60, 3, 16
P_IN, P_OUT = 0.120, 0.018
DEPTHS = np.arange(0, 17)
SHOWN = [2, 4, 8, 16]
NOISE_FLOOR = 0.20           # fixed absolute read-out noise, never rescaled
N_TRIALS = 40                # noise draws averaged per depth
SEED = 21


def block_model(rng):
    labels = np.repeat(np.arange(N_CLASS), N_PER)
    n = labels.size
    prob = np.where(labels[:, None] == labels[None, :], P_IN, P_OUT)
    adj = np.triu((rng.random((n, n)) < prob).astype(float), 1)
    adj = adj + adj.T
    for i in np.flatnonzero(adj.sum(1) == 0):
        peers = np.flatnonzero((labels == labels[i]) & (np.arange(n) != i))
        j = rng.choice(peers)
        adj[i, j] = adj[j, i] = 1.0
    return adj, labels


def dirichlet_energy(h, adj):
    """0.5 * sum_ij A_ij ||h_i/sqrt(d_i) - h_j/sqrt(d_j)||^2, over ||h||^2."""
    d = np.sqrt(np.maximum(adj.sum(1), 1.0))
    z = h / d[:, None]
    i, j = np.nonzero(np.triu(adj, 1))
    return float(((z[i] - z[j]) ** 2).sum() / max((h ** 2).sum(), 1e-12))


def nearest_mean_accuracy(h, labels, train, test, rng, trials=N_TRIALS):
    """Fixed-scale read-out: the noise floor does not shrink with the signal."""
    hits = []
    for _ in range(trials):
        hn = h + rng.normal(0, NOISE_FLOOR, h.shape)
        means = np.stack([hn[train][labels[train] == k].mean(0)
                          for k in range(N_CLASS)])
        pred = np.argmin(((hn[test][:, None, :] - means[None]) ** 2).sum(-1), 1)
        hits.append((pred == labels[test]).mean())
    return float(np.mean(hits))


def cosine_matrix(h):
    z = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
    return z @ z.T


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)
    adj, labels = block_model(rng)
    n = labels.size

    a_hat = adj + np.eye(n)
    inv = 1.0 / np.sqrt(a_hat.sum(1))
    a_hat = a_hat * inv[:, None] * inv[None, :]

    x = (rng.normal(0, 1.0, (N_CLASS, DIM))[labels]
         + rng.normal(0, 1.0, (n, DIM)))
    x /= np.sqrt((x ** 2).mean())                # unit RMS entry at layer 0

    idx = rng.permutation(n)
    train, test = idx[:n // 2], idx[n // 2:]

    h, states = x.copy(), {0: x.copy()}
    energy, acc = [], []
    for L in DEPTHS:
        if L > 0:
            h = a_hat @ h
            states[L] = h.copy()
        energy.append(dirichlet_energy(states[L], adj))
        acc.append(nearest_mean_accuracy(states[L], labels, train, test,
                                         np.random.default_rng(SEED + 100 + L)))
    energy, acc = np.array(energy), np.array(acc)
    # deepest layer that still keeps 90 % of the best accuracy on offer
    usable = int(DEPTHS[np.flatnonzero(acc >= 0.90 * acc.max())][-1])

    mats = [cosine_matrix(states[L]) for L in SHOWN]
    norm = shared_norm(mats, symmetric=True)
    cmap = get_cmap("cb_diverging")

    fig = plt.figure(figsize=(9.6, 5.6))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.0, 1.0, 1.32],
                          height_ratios=[1.0, 1.0, 0.085],
                          wspace=0.22, hspace=0.34,
                          left=0.055, right=0.955, top=0.845, bottom=0.115)
    mat_axes = [fig.add_subplot(gs[r, c]) for r in (0, 1) for c in (0, 1)]
    ax_cb = fig.add_subplot(gs[2, 0:2])
    ax_c = fig.add_subplot(gs[0:2, 2])

    bounds = np.cumsum([N_PER] * (N_CLASS - 1)) - 0.5
    for ax, L, m in zip(mat_axes, SHOWN, mats):
        im = ax.imshow(m, cmap=cmap, norm=norm, interpolation="nearest")
        for b in bounds:
            ax.axhline(b, color="white", lw=0.8)
            ax.axvline(b, color="white", lw=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.6)
            s.set_color("#8a8a8a")
        off = m[np.repeat(np.arange(n), n).reshape(n, n) //
                N_PER != np.tile(np.arange(n), n).reshape(n, n) // N_PER]
        ax.set_title(f"L = {L}   ·   off-block cos {off.mean():.2f}",
                     fontsize=8.5, fontweight="bold", color=INK, pad=3.5)
    for k in range(N_CLASS):
        for r in (0, 1):
            mat_axes[2 * r].text(-4, N_PER * (k + 0.5), "ABC"[k], ha="right",
                                 va="center", fontsize=6.5,
                                 color=CLASS_COLORS[k], fontweight="bold")

    cb = fig.colorbar(im, cax=ax_cb, orientation="horizontal",
                      ticks=[-1, -0.5, 0, 0.5, 1])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(RULE)
    cb.ax.tick_params(labelsize=6.5, width=0.7, length=2.5, color=RULE)
    cb.set_label("pairwise cosine similarity (-)", fontsize=7.0, color=INK,
                 labelpad=1.5)

    # ---- right: the two depth curves --------------------------------------
    ax_c.plot(DEPTHS, energy, color=ENERGY_C, lw=1.8, marker="o", ms=3.2,
              mfc="white", mew=0.9, label="Dirichlet energy", zorder=4)
    ax_c.set_yscale("log")
    ax_c.set_xlabel("propagation depth L (layers)", fontsize=8.0)
    ax_c.set_ylabel("Dirichlet energy $E(H^{(L)})$ (-)", fontsize=8.0,
                    color=ENERGY_C)
    ax_c.tick_params(axis="y", labelsize=7.0, colors=ENERGY_C)
    ax_c.tick_params(axis="x", labelsize=7.0)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["left"].set_color(ENERGY_C)

    ax_a = ax_c.twinx()
    ax_a.plot(DEPTHS, 100 * acc, color=ACC_C, lw=1.8, marker="s", ms=3.2,
              mfc="white", mew=0.9, label="read-out accuracy", zorder=5)
    ax_a.axhline(100 / N_CLASS, color="#808080", lw=0.8, ls="--", zorder=1)
    ax_a.text(0.4, 100 / N_CLASS + 1.2, "chance", ha="left", va="bottom",
              fontsize=6.5, color=MUTED)
    ax_a.set_ylabel("node-classification accuracy (%)", fontsize=8.0,
                    color=ACC_C)
    ax_a.tick_params(axis="y", labelsize=7.0, colors=ACC_C)
    ax_a.set_ylim(20, 100)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_color(ACC_C)
    ax_a.spines["left"].set_visible(False)

    for L in SHOWN:
        ax_a.axvline(L, color="#cccccc", lw=0.7, ls=":", zorder=0)
        ax_a.text(L, 21.5, f"L={L}", ha="center", va="bottom", fontsize=6.0,
                  color=MUTED, rotation=90)
    ax_a.plot([usable], [100 * acc[usable]], marker="s", ms=6.5, mfc="none",
              mec=ACC_C, mew=1.4, zorder=6)
    ax_a.annotate(f"last usable depth\nL = {usable} "
                  f"({100 * acc[usable]:.0f}%)",
                  xy=(usable, 100 * acc[usable]), xytext=(usable + 2.0, 86),
                  fontsize=6.8, color=ACC_C, ha="left",
                  arrowprops=dict(arrowstyle="-", color=ACC_C, lw=0.7))
    ax_c.set_title("Energy and accuracy fall together", fontsize=9.0,
                   fontweight="bold", color=INK, pad=6)

    add_panel_label(mat_axes[0], "(a)", x=-0.15, y=1.16)
    add_panel_label(ax_c, "(b)", x=-0.22, y=1.06)

    fall = energy[0] / energy[-1]
    savefig(fig, "447_gnn_oversmoothing.png")


if __name__ == "__main__":
    main()

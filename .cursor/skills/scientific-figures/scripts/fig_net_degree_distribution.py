"""Degree distributions of three classic network models on log-log axes.

The main panel is the scale-free (preferential-attachment) network: binned
P(k) on log-log axes with a maximum-likelihood power-law fit above k_min, the
exponent and its standard error taken from the fit itself, and the Poisson
expectation of a degree-matched random graph drawn behind it for contrast.
The subordinate column repeats the comparison as complementary CDFs and as
overlaid densities, both on the shared degree axis of the main panel.

Everything is generated with numpy: Erdos-Renyi by direct binomial draw,
Barabasi-Albert by preferential attachment, Watts-Strogatz by rewiring a ring
lattice.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pubstyle import CATEGORICAL, COLORS, annotate_box, savefig, set_pub_style
from figkit import panel_labels_inches

SEED = 5
N_NODES = 6000
MEAN_DEG = 8.0

MODELS = ["Scale-free (BA)", "Random (ER)", "Small-world (WS)"]
MODEL_COLS = {MODELS[0]: CATEGORICAL[0],      # house red
              MODELS[1]: CATEGORICAL[1],      # house blue
              MODELS[2]: CATEGORICAL[2]}      # house green


# --------------------------------------------------------------------------
# degree sequences (numpy only - no networkx in this environment)
# --------------------------------------------------------------------------
def er_degrees(n, mean_deg, seed):
    """Erdos-Renyi G(n, p): degrees are Binomial(n - 1, p)."""
    rng = np.random.default_rng(seed)
    p = mean_deg / (n - 1.0)
    return rng.binomial(n - 1, p, size=n).astype(float)


def ba_degrees(n, m, seed):
    """Barabasi-Albert preferential attachment via an endpoint urn."""
    rng = np.random.default_rng(seed)
    deg = np.zeros(n)
    urn = np.empty(2 * m * n, dtype=np.int64)          # repeated endpoints
    filled = 0
    for v in range(m + 1):                             # seed clique
        for w in range(v):
            deg[v] += 1
            deg[w] += 1
            urn[filled], urn[filled + 1] = v, w
            filled += 2
    for v in range(m + 1, n):
        picked = set()
        while len(picked) < m:
            picked.add(int(urn[rng.integers(filled)]))
        for w in picked:
            deg[v] += 1
            deg[w] += 1
            urn[filled], urn[filled + 1] = v, w
            filled += 2
    return deg


def ws_degrees(n, k, p, seed):
    """Watts-Strogatz: ring lattice of degree k, each edge rewired w.p. p."""
    rng = np.random.default_rng(seed)
    deg = np.full(n, float(k))
    n_edges = n * k // 2
    rewired = rng.random(n_edges) < p
    for _ in np.flatnonzero(rewired):
        src = rng.integers(n)                          # keeps its endpoint
        old = rng.integers(n)                          # loses one
        new = rng.integers(n)                          # gains one
        if old == new or old == src:
            continue
        deg[old] -= 1
        deg[new] += 1
    return np.maximum(deg, 0.0)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def pdf_points(deg, n_bins=22):
    """Logarithmically binned P(k); returns (k_centre, density)."""
    d = deg[deg > 0]
    edges = np.unique(np.round(np.logspace(0, np.log10(d.max() + 1),
                                           n_bins)).astype(int))
    edges = np.append(edges, edges[-1] + 1)
    cnt, _ = np.histogram(d, bins=edges)
    width = np.diff(edges)
    centre = np.sqrt(edges[:-1] * (edges[1:] - 1e-9))
    dens = cnt / (width * d.size)
    ok = cnt > 0
    return centre[ok], dens[ok]


def ccdf(deg):
    d = np.sort(deg[deg > 0])
    return d, 1.0 - np.arange(d.size) / d.size


def powerlaw_mle(deg, k_min):
    """Clauset MLE for a discrete-ish power law; returns (gamma, se, n_tail)."""
    tail = deg[deg >= k_min]
    n = tail.size
    gamma = 1.0 + n / np.sum(np.log(tail / (k_min - 0.5)))
    return gamma, (gamma - 1.0) / np.sqrt(n), n


def poisson_pmf(k, lam):
    from scipy.special import gammaln
    k = np.asarray(k, dtype=float)
    return np.exp(k * np.log(lam) - lam - gammaln(k + 1.0))


# --------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)

    deg = {
        MODELS[0]: ba_degrees(N_NODES, int(MEAN_DEG // 2), SEED),
        MODELS[1]: er_degrees(N_NODES, MEAN_DEG, SEED + 1),
        MODELS[2]: ws_degrees(N_NODES, int(MEAN_DEG), 0.12, SEED + 2),
    }
    sf = deg[MODELS[0]]
    k_min = 6.0
    gamma, gamma_se, n_tail = powerlaw_mle(sf, k_min)

    # one degree axis for every panel in the figure
    k_hi = max(d.max() for d in deg.values()) * 1.6
    k_lim = (0.75, k_hi)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 0.58],
                  height_ratios=[1.0, 0.58], hspace=0.42, wspace=0.22)
    ax = fig.add_subplot(gs[:, 0])
    ax_cc = fig.add_subplot(gs[0, 1])
    ax_pdf = fig.add_subplot(gs[1, 1])

    # ---- (a) scale-free P(k) with the MLE fit ----------------------------
    kc, pk = pdf_points(sf)
    lam = sf.mean()
    kk = np.arange(1, int(sf.max()) + 1)
    ax.plot(kk, poisson_pmf(kk, lam), color=COLORS["grid"], lw=1.4,
            ls="--", zorder=1,
            label=f"Poisson, $\\langle k \\rangle$ = {lam:.1f}")
    ax.scatter(kc, pk, s=26, color=MODEL_COLS[MODELS[0]], edgecolor="white",
               linewidth=0.6, zorder=4, label="scale-free, binned $P(k)$")

    # normalised MLE density, not a line eyeballed through the first bin
    frac_tail = n_tail / sf.size
    fit_k = np.logspace(np.log10(k_min), np.log10(sf.max()), 60)
    ref = frac_tail * (gamma - 1.0) / k_min * (fit_k / k_min) ** (-gamma)
    ax.plot(fit_k, ref, color=COLORS["ink"], lw=1.5, zorder=5,
            label=f"MLE fit, $\\gamma$ = {gamma:.2f}")
    ax.axvline(k_min, color=COLORS["baseline"], lw=0.8, ls=":", zorder=2)
    ax.text(k_min * 1.10, 3.0e-4, f"$k_{{min}}$ = {k_min:.0f}", fontsize=6.8,
            color=COLORS["sub"], rotation=90, va="bottom", ha="left")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*k_lim)
    ax.set_ylim(0.4 / N_NODES, 2.0)
    ax.set_xlabel("degree k (-)", fontsize=8.0)
    ax.set_ylabel("probability density P(k) (-)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.legend(loc="lower left", frameon=False, fontsize=7.2)
    ax.set_title("Scale-free degree distribution and power-law fit",
                 fontsize=9.0, loc="left", pad=4)
    annotate_box(ax, f"$\\gamma$ = {gamma:.2f} $\\pm$ {gamma_se:.2f}\n"
                     f"tail n = {n_tail}  of  {sf.size}\n"
                     f"$k_{{max}}$ = {int(sf.max())}, "
                     f"$\\langle k \\rangle$ = {lam:.1f}",
                 loc="upper right", fontsize=7.0)

    # ---- (b) complementary CDFs -----------------------------------------
    for name in MODELS:
        x, y = ccdf(deg[name])
        ax_cc.plot(x, y, color=MODEL_COLS[name], lw=1.5, label=name)
    ax_cc.set_xscale("log")
    ax_cc.set_yscale("log")
    ax_cc.set_xlim(*k_lim)
    ax_cc.set_ylim(0.4 / N_NODES, 1.8)
    ax_cc.set_xlabel("degree k (-)", fontsize=7.5)
    ax_cc.set_ylabel("P(K $\\geq$ k) (-)", fontsize=7.5)
    ax_cc.tick_params(labelsize=6.8)
    ax_cc.legend(loc="lower left", frameon=False, fontsize=6.8)
    ax_cc.set_title("Complementary CDF", fontsize=8.5, loc="left", pad=4)

    # ---- (c) the three densities overlaid -------------------------------
    for name in MODELS:
        x, y = pdf_points(deg[name], n_bins=20)
        ax_pdf.plot(x, y, color=MODEL_COLS[name], lw=1.3, marker="o", ms=2.8,
                    mec="white", mew=0.4, label=name)
    ax_pdf.set_xscale("log")
    ax_pdf.set_yscale("log")
    ax_pdf.set_xlim(*k_lim)
    ax_pdf.set_ylim(0.4 / N_NODES, 2.0)
    ax_pdf.set_xlabel("degree k (-)", fontsize=7.5)
    ax_pdf.set_ylabel("P(k) (-)", fontsize=7.5)
    ax_pdf.tick_params(labelsize=6.8)
    ax_pdf.set_title("Same three models, densities", fontsize=8.5,
                     loc="left", pad=4)

    fig.tight_layout(rect=[0, 0.01, 1, 0.925], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax, ax_cc, ax_pdf], dx=-0.42, dy=0.10)

    savefig(fig, "365_net_degree_distribution.png")


if __name__ == "__main__":
    main()

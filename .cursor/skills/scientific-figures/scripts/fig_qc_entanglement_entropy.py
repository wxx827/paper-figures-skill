"""Area law against volume law for the entanglement entropy of a spin chain.

The main panel is the claim: two ensembles of disorder realisations on the
same chain, one whose entropy grows with the boundary and saturates, one whose
entropy grows with the subsystem volume and tracks the maximal-entropy line.
Every annotated number - the saturation value, the fitted slope per site, the
fraction of the maximal entropy reached at half chain - is measured from the
plotted ensembles.  The two smaller panels underneath the same colour code
carry the supporting evidence: where the crossover sits in parameter space,
and how differently the Schmidt spectra decay.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import panel_labels_inches, safe_colorbar

BLUE = "#3b5b92"        # area law
PURPLE = "#8e6bbf"      # volume law
GREY = "#808080"        # maximal-entropy reference
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"

N_SITES = 32
N_REAL = 16
XI = 3.2                # correlation length of the area-law phase
S_SAT = 1.35
S_SLOPE = 0.61          # nats per site in the volume-law phase
SEQ = get_cmap("cb_sequential_purple")


def ensembles(seed=12):
    rng = np.random.default_rng(seed)
    ell = np.arange(1, N_SITES)
    cut = np.minimum(ell, N_SITES - ell).astype(float)

    area = S_SAT * (1.0 - np.exp(-cut / XI))
    vol = S_SLOPE * cut

    area_r = area + rng.normal(0.0, 0.055, (N_REAL, ell.size)) * np.sqrt(cut)
    vol_r = vol * (1.0 + rng.normal(0.0, 0.028, (N_REAL, 1)))
    vol_r = vol_r + rng.normal(0.0, 0.10, (N_REAL, ell.size))
    return ell, cut, area_r, vol_r


def phase_map(n=64):
    w = np.linspace(0.4, 8.0, n)
    j = np.linspace(0.2, 2.0, n)
    W, J = np.meshgrid(w, j, indexing="xy")
    s_max = 0.5 * N_SITES * np.log(2.0)
    return w, j, 0.12 + s_max / (1.0 + (W / (2.4 * J)) ** 3.0)


def spectra(seed=5, k_max=26):
    rng = np.random.default_rng(seed)
    k = np.arange(1, k_max + 1)
    vol = np.exp(-0.085 * k) * (1.0 + 0.10 * rng.normal(size=k.size))
    area = np.exp(-0.62 * k) * (1.0 + 0.10 * rng.normal(size=k.size))
    vol = np.abs(vol) / np.abs(vol).sum()
    area = np.abs(area) / np.abs(area).sum()
    return k, area, vol


def main_panel(ax, ell, cut, area_r, vol_r):
    ref = np.log(2.0) * cut
    ax.plot(ell, ref, color=GREY, lw=1.0, ls=(0, (4, 2.6)), zorder=2)

    for arr, color, name in [(vol_r, PURPLE, "volume law"),
                             (area_r, BLUE, "area law")]:
        mu, sd = arr.mean(axis=0), arr.std(axis=0)
        ax.fill_between(ell, mu - sd, mu + sd, color=color, alpha=0.18,
                        linewidth=0.0, zorder=3)
        ax.plot(ell, mu, color=color, lw=1.7, zorder=4, label=name)

    vol_mu, area_mu = vol_r.mean(axis=0), area_r.mean(axis=0)
    half = N_SITES // 2 - 1
    slope = float(np.polyfit(cut[:8], vol_mu[:8], 1)[0])
    sat = float(area_mu[half - 3:half + 4].mean())
    frac = vol_mu[half] / ref[half]

    ax.axhline(sat, color=BLUE, lw=0.8, ls=(0, (2.4, 2.0)), zorder=2)
    ax.text(N_SITES - 1.0, sat - 0.40,
            f"saturation $S_A^{{\\infty}}$ = {sat:.2f} nats", ha="right",
            va="top", fontsize=7.0, color=BLUE)
    ax.annotate(f"volume law,  {slope:.2f} nats per site",
                xy=(11.0, vol_mu[10]), xytext=(5.4, vol_mu[10] + 2.8),
                ha="left", fontsize=7.4, color=PURPLE,
                arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.7))
    ax.annotate("area law,  boundary limited",
                xy=(21.0, area_mu[20]), xytext=(21.0, area_mu[20] + 1.15),
                ha="center", fontsize=7.4, color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7))
    ax.text(N_SITES / 2.0, ref.max() + 0.22,
            r"maximal entropy  $\ell_A \ln 2$", ha="center", va="bottom",
            fontsize=6.8, color=GREY)

    ax.set_xlabel(r"subsystem size $\ell_A$ (sites)")
    ax.set_ylabel(r"entanglement entropy $S_A$ (nats)")
    ax.set_xlim(0.0, N_SITES)
    ax.set_ylim(0.0, np.log(2.0) * (N_SITES / 2.0) + 1.4)
    ax.set_xticks(np.arange(0, N_SITES + 1, 4))
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title(f"Volume-law half chain reaches {frac * 100:.0f} % of the "
                 f"maximum", fontsize=9.0, fontweight="bold", color=INK)
    return sat, slope


def heat_panel(fig, ax, w, j, s):
    im = ax.imshow(s, origin="lower", aspect="auto", cmap=SEQ,
                   extent=[w[0], w[-1], j[0], j[-1]])
    lev = ax.contour(w, j, s, levels=[0.25 * s.max(), 0.60 * s.max()],
                     colors=["#ffffff"], linewidths=0.8,
                     linestyles=[(0, (3, 2.0))])
    ax.clabel(lev, fmt="%.0f", fontsize=6.0, colors="#ffffff")
    ax.set_xlabel(r"disorder strength $W/J_0$ (-)", fontsize=7.4)
    ax.set_ylabel(r"coupling $J/J_0$ (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.set_title("Half-chain entropy across the crossover", fontsize=8.2,
                 fontweight="bold", color=INK)
    safe_colorbar(fig, im, ax, label=r"$S_{N/2}$ (nats)", width=0.09,
                  pad=0.06, labelsize=6.5)


def spectrum_panel(ax, k, area, vol):
    for lam, color, name in [(vol, PURPLE, "volume law"),
                             (area, BLUE, "area law")]:
        ax.semilogy(k, lam, marker="o", ms=2.8, lw=1.1, color=color,
                    mfc="white", mew=0.8, label=name)
    ax.set_xlabel(r"Schmidt index $k$ (-)", fontsize=7.4)
    ax.set_ylabel(r"$\lambda_k$ (-)", fontsize=7.4)
    ax.set_ylim(1e-8, 1.0)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left",
              handlelength=1.4, borderaxespad=0.2)
    ax.set_title("Entanglement spectrum decay", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    ell, cut, area_r, vol_r = ensembles()
    w, j, s = phase_map()
    k, area_l, vol_l = spectra()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.30, hspace=0.62)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_spec = fig.add_subplot(gs[1, 1])

    sat, slope = main_panel(ax_main, ell, cut, area_r, vol_r)
    heat_panel(fig, ax_heat, w, j, s)
    spectrum_panel(ax_spec, k, area_l, vol_l)

    fig.tight_layout(rect=[0, 0.035, 1, 0.945], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_main, ax_heat, ax_spec], dx=-0.42, dy=0.12)
    savefig(fig, "495_qc_entanglement_entropy.png")


if __name__ == "__main__":
    main()

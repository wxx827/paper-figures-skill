"""MCMC trace plot + marginal posterior densities (ArviZ-style diagnostics).

Left column: per-chain marginal posterior KDE with pooled 94% HDI shading.
Right column: the corresponding trace ("caterpillar") for every chain.
A slim third column reports split-Rhat and bulk-ESS for each parameter.

Chains are synthesised as stationary AR(1) processes, so they are genuinely
well mixed: Rhat lands in 1.000-1.005 and the traces show no drift.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from pubstyle import set_pub_style, savefig, style_axes

CHAIN_COLORS = ["#2e6f9e", "#c4714b", "#5aa17f", "#8e6bbf"]


def ar1_chain(rng, mu, sd, rho, n):
    """Stationary AR(1) draw with marginal N(mu, sd^2) and lag-1 corr rho."""
    x = np.empty(n)
    x[0] = mu + sd * rng.standard_normal()
    innov = sd * np.sqrt(1.0 - rho ** 2)
    eps = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = mu + rho * (x[t - 1] - mu) + innov * eps[t]
    return x


def split_rhat(chains):
    """Classic split-chain potential scale reduction factor."""
    m, n = chains.shape
    half = n // 2
    s = np.vstack([chains[:, :half], chains[:, half:2 * half]])
    m2, n2 = s.shape
    means = s.mean(axis=1)
    W = s.var(axis=1, ddof=1).mean()
    B = n2 * means.var(ddof=1)
    var_hat = (n2 - 1) / n2 * W + B / n2
    return float(np.sqrt(var_hat / W))


def _autocorr(x):
    x = x - x.mean()
    n = x.size
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conjugate(f))[:n].real
    return ac / ac[0]


def bulk_ess(chains):
    """Geyer initial-positive-sequence effective sample size."""
    m, n = chains.shape
    rho = np.mean([_autocorr(c) for c in chains], axis=0)
    total = 0.0
    for t in range(1, n // 2):
        pair = rho[2 * t - 1] + rho[2 * t]
        if pair < 0:
            break
        total += pair
    return float(m * n / (1.0 + 2.0 * total))


def hdi(samples, prob=0.94):
    s = np.sort(samples)
    k = int(np.floor(prob * s.size))
    widths = s[k:] - s[:s.size - k]
    i = int(np.argmin(widths))
    return s[i], s[i + k]


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(20240517)

    n_draw, n_chain = 1200, 4
    specs = [
        (r"$\mu$",        2.41, 0.085, 0.34, None),
        (r"$\beta_1$",   -0.68, 0.120, 0.42, None),
        (r"$\beta_2$",    1.15, 0.095, 0.38, None),
        (r"$\sigma$",     None, None,  0.45, "log"),
        (r"$\tau$",       None, None,  0.52, "log"),
    ]
    log_pars = {r"$\sigma$": (np.log(0.83), 0.11),
                r"$\tau$": (np.log(0.42), 0.19)}

    draws = {}
    for name, mu, sd, rho, kind in specs:
        if kind == "log":
            lmu, lsd = log_pars[name]
            ch = np.vstack([np.exp(ar1_chain(rng, lmu, lsd, rho, n_draw))
                            for _ in range(n_chain)])
        else:
            ch = np.vstack([ar1_chain(rng, mu, sd, rho, n_draw)
                            for _ in range(n_chain)])
        draws[name] = ch

    npar = len(specs)
    fig = plt.figure(figsize=(7.4, 8.2))
    gs = GridSpec(npar, 3, figure=fig, width_ratios=[1.0, 1.75, 0.30],
                  hspace=0.55, wspace=0.16,
                  left=0.075, right=0.985, top=0.885, bottom=0.062)
    head_axes = {}

    for i, (name, *_rest) in enumerate(specs):
        ch = draws[name]
        pooled = ch.ravel()
        lo, hi = hdi(pooled, 0.94)
        med = np.median(pooled)

        # ---- marginal posterior -------------------------------------------
        axd = fig.add_subplot(gs[i, 0])
        grid = np.linspace(pooled.min(), pooled.max(), 400)
        pooled_d = gaussian_kde(pooled)(grid)
        axd.fill_between(grid, 0, pooled_d, color="#d9e4ee", alpha=0.85,
                         lw=0, zorder=1)
        for c, col in zip(ch, CHAIN_COLORS):
            axd.plot(grid, gaussian_kde(c)(grid), color=col, lw=1.0,
                     alpha=0.9, zorder=3)
        inside = (grid >= lo) & (grid <= hi)
        axd.plot(grid[inside], np.zeros(inside.sum()), color="#333333",
                 lw=2.2, solid_capstyle="butt", zorder=4)
        # stop the median rule at the density so it never crosses the labels
        med_h = float(np.interp(med, grid, pooled_d))
        axd.plot([med, med], [0, med_h], color="#333333", lw=0.8,
                 ls=(0, (3, 2)), zorder=4)
        axd.set_ylim(0, pooled_d.max() * 1.34)
        axd.set_yticks([])
        axd.spines["left"].set_visible(False)
        style_axes(axd)
        axd.set_ylabel(name, fontsize=11, labelpad=8, rotation=0,
                       va="center", ha="right")
        axd.text(0.015, 0.995, f"median {med:.2f}", transform=axd.transAxes,
                 ha="left", va="top", fontsize=6.5, color="#555555")
        axd.text(0.985, 0.995, f"94% HDI [{lo:.2f}, {hi:.2f}]",
                 transform=axd.transAxes, ha="right", va="top",
                 fontsize=6.5, color="#555555")
        axd.tick_params(labelsize=7)

        # ---- trace ---------------------------------------------------------
        axt = fig.add_subplot(gs[i, 1])
        it = np.arange(n_draw)
        for c, col in zip(ch, CHAIN_COLORS):
            axt.plot(it, c, color=col, lw=0.32, alpha=0.72)
        axt.axhline(med, color="#333333", lw=0.7, ls=(0, (3, 2)), zorder=5)
        axt.set_xlim(0, n_draw)
        pad = 0.06 * np.ptp(pooled)
        axt.set_ylim(pooled.min() - pad, pooled.max() + pad)
        style_axes(axt)
        axt.tick_params(labelsize=7)

        # ---- diagnostics column -------------------------------------------
        axs = fig.add_subplot(gs[i, 2])
        axs.axis("off")
        rh = split_rhat(ch)
        es = bulk_ess(ch)
        axs.text(0.06, 0.66, r"$\hat{R}$" + f" = {rh:.3f}", fontsize=7.5,
                 fontweight="bold", color="#1a1a1a", va="center")
        axs.text(0.06, 0.36, f"ESS = {es:,.0f}", fontsize=7.2,
                 color="#555555", va="center")

        if i == 0:
            head_axes = {"dens": axd, "trace": axt, "diag": axs}
        if i == npar - 1:
            axd.set_xlabel("parameter value", fontsize=8)
            axt.set_xlabel("post-warmup iteration", fontsize=8)

    # ---- headers & shared legend -------------------------------------------
    for key, label in (("dens", "Marginal posterior"), ("trace", "Trace"),
                       ("diag", "Diagnostics")):
        box = head_axes[key].get_position()

    handles = [plt.Line2D([], [], color=c, lw=1.4, label=f"chain {k + 1}")
               for k, c in enumerate(CHAIN_COLORS)]
    handles.append(plt.Line2D([], [], color="#333333", lw=2.2,
                              label="94% HDI"))
    handles.append(plt.Line2D([], [], color="#333333", lw=0.9,
                              ls=(0, (3, 2)), label="posterior median"))

    savefig(fig, "127_trace_posterior.png")


if __name__ == "__main__":
    main()

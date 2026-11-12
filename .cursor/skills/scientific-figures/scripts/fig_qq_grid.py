"""2x3 Q-Q plot grid: normality check with a pointwise confidence envelope.

Samples are standardised so the reference is the 1:1 line, and the grey
envelope is the true pointwise 95% acceptance region for order statistics
(Beta(i, n-i+1) quantiles pushed through the normal quantile function) instead
of a decorative constant-width ribbon. Points that fall outside the envelope
are outlined, and the count is reported together with the Shapiro-Wilk test.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pubstyle import (set_pub_style, PALETTES, savefig, style_axes,
                      add_panel_label)

C = PALETTES["teal_rust"][0]
REF = "#b23a48"
N = 200

# (panel title, distribution key)
VARS = [("NDVI residuals", "normal"),
        ("Temperature residuals", "heavy"),
        ("Precipitation residuals", "right"),
        ("Runoff residuals", "left"),
        ("N-loss residuals", "outliers"),
        ("Carbon-flux residuals", "bimodal")]


def sample(rng, kind):
    if kind == "normal":
        return rng.normal(0, 1, N)
    if kind == "heavy":
        return rng.standard_t(3, N)
    if kind == "right":
        return rng.exponential(1, N)
    if kind == "left":
        return -rng.gamma(2.2, 0.8, N)
    if kind == "outliers":
        d = rng.normal(0, 1, N)
        d[rng.choice(N, 6, replace=False)] += rng.normal(0, 6, 6)
        return d
    return np.concatenate([rng.normal(-1.7, 0.6, N // 2),
                           rng.normal(1.7, 0.6, N - N // 2)])


def envelope(n, alpha=0.05):
    i = np.arange(1, n + 1)
    lo = stats.norm.ppf(stats.beta.ppf(alpha / 2, i, n - i + 1))
    hi = stats.norm.ppf(stats.beta.ppf(1 - alpha / 2, i, n - i + 1))
    return lo, hi


def panel(ax, rng, idx, spec):
    name, kind = spec
    d = sample(rng, kind)
    z = (d - d.mean()) / d.std(ddof=1)
    osm, osr = stats.probplot(z, dist="norm")[0]
    lo, hi = envelope(len(z))

    ax.fill_between(osm, lo, hi, color="#dcdcdc", alpha=0.75, lw=0, zorder=1)
    lim = (min(osm.min(), osr.min(), lo.min()) - 0.4,
           max(osm.max(), osr.max(), hi.max()) + 0.4)
    ax.plot(lim, lim, color=REF, lw=1.4, zorder=3)

    out = (osr < lo) | (osr > hi)
    ax.scatter(osm, osr, s=16, color=C, alpha=0.7, edgecolors="white",
               linewidths=0.4, zorder=4)
    # ring only the few worst offenders, otherwise a skewed panel turns solid
    dev = np.where(out, np.maximum(lo - osr, osr - hi), -np.inf)
    for i in np.argsort(-dev)[:5]:
        if np.isfinite(dev[i]):
            ax.scatter(osm[i], osr[i], s=34, facecolors="none",
                       edgecolors=REF, linewidths=1.0, zorder=5)

    w, p = stats.shapiro(z)
    ax.text(0.035, 0.965,
            f"Shapiro-Wilk W = {w:.3f}\n"
            f"{'p < 0.001' if p < 1e-3 else f'p = {p:.3f}'}\n"
            f"{int(out.sum())} / {len(z)} outside envelope",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.9,
            linespacing=1.35, zorder=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc",
                      lw=0.6, alpha=0.93))

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Theoretical quantile (z)", fontsize=7.5)
    ax.set_ylabel("Sample quantile (z)", fontsize=7.5)
    ax.set_title(name, fontsize=9, fontweight="bold")
    ax.grid(lw=0.5, alpha=0.3)
    style_axes(ax)
    add_panel_label(ax, f"({'abcdef'[idx]})", x=-0.20, y=1.02, size=10)


def main():
    set_pub_style()
    rng = np.random.default_rng(461)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.0))
    for k, (ax, spec) in enumerate(zip(axes.ravel(), VARS)):
        panel(ax, rng, k, spec)
    fig.tight_layout(rect=[0, 0.025, 1, 0.955], h_pad=2.4, w_pad=2.4)
    savefig(fig, "46_qq_grid.png")


if __name__ == "__main__":
    main()

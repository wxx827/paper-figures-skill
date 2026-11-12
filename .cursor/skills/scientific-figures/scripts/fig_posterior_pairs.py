"""Posterior pair plot: joint geometry of a 4-parameter posterior.

Lower triangle  : 2-D posterior density (hexbin) with 50% / 95% credible
                  region contours and the posterior mode marked.
Diagonal        : marginal posterior with 94% HDI shading and median rule.
Upper triangle  : posterior correlation, value on a diverging background.

Draws come from a correlated multivariate normal, with sigma pushed through
an exponential so the posterior shows realistic skew and funnel geometry.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde
from pubstyle import set_pub_style, PALETTES, savefig, style_axes

DENS_CMAP = LinearSegmentedColormap.from_list(
    "post", ["#ffffff"] + PALETTES["cb_sequential_blue"][1:])
CORR_CMAP = LinearSegmentedColormap.from_list(
    "corr", PALETTES["cb_diverging"])


def hdi(samples, prob=0.94):
    s = np.sort(samples)
    k = int(np.floor(prob * s.size))
    i = int(np.argmin(s[k:] - s[:s.size - k]))
    return s[i], s[i + k]


def credible_levels(z, probs=(0.5, 0.95)):
    """Density levels enclosing the requested posterior mass."""
    flat = np.sort(z.ravel())[::-1]
    csum = np.cumsum(flat)
    csum /= csum[-1]
    return sorted(float(flat[np.searchsorted(csum, p)]) for p in probs)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(7)

    names = [r"$\alpha$", r"$\beta_1$", r"$\beta_2$", r"$\sigma$"]
    units = ["intercept", "slope: dose", "slope: age", "residual scale"]
    mean = np.array([2.40, -0.68, 1.15, np.log(0.83)])
    sd = np.array([0.085, 0.120, 0.095, 0.110])
    corr = np.array([
        [1.00, -0.72,  0.31, -0.12],
        [-0.72, 1.00, -0.45,  0.08],
        [0.31, -0.45,  1.00,  0.44],
        [-0.12, 0.08,  0.44,  1.00],
    ])
    cov = corr * np.outer(sd, sd)
    draws = rng.multivariate_normal(mean, cov, size=6000)
    draws[:, 3] = np.exp(draws[:, 3])

    p = len(names)
    fig, axes = plt.subplots(p, p, figsize=(7.3, 7.3))
    fig.subplots_adjust(left=0.085, right=0.965, top=0.876, bottom=0.085,
                        hspace=0.10, wspace=0.10)
    R = np.corrcoef(draws.T)
    cnorm = Normalize(-1, 1)

    for i in range(p):
        for j in range(p):
            ax = axes[i, j]
            style_axes(ax)
            for s in ax.spines.values():
                s.set_linewidth(0.8)
                s.set_color("#333333")

            if i == j:
                x = draws[:, i]
                grid = np.linspace(x.min(), x.max(), 400)
                dens = gaussian_kde(x)(grid)
                lo, hi = hdi(x)
                ax.fill_between(grid, 0, dens, color="#e4ecf3", lw=0)
                inside = (grid >= lo) & (grid <= hi)
                ax.fill_between(grid[inside], 0, dens[inside],
                                color="#9cc3dd", lw=0, alpha=0.95)
                ax.plot(grid, dens, color="#134a76", lw=1.2)
                med = np.median(x)
                ax.plot([med, med], [0, float(np.interp(med, grid, dens))],
                        color="#b23a48", lw=1.1, ls=(0, (3, 2)), zorder=5)
                ax.set_ylim(0, dens.max() * 1.30)
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
                ax.text(0.5, 0.99, f"{med:.2f}  [{lo:.2f}, {hi:.2f}]",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=6.2, color="#444444")

            elif i > j:
                x, y = draws[:, j], draws[:, i]
                ax.hexbin(x, y, gridsize=32, cmap=DENS_CMAP, mincnt=1,
                          linewidths=0.0)
                kde = gaussian_kde(np.vstack([x, y]))
                gx = np.linspace(x.min(), x.max(), 70)
                gy = np.linspace(y.min(), y.max(), 70)
                GX, GY = np.meshgrid(gx, gy)
                Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
                # levels come back ascending -> outer 95% first, inner 50%
                ax.contour(GX, GY, Z, levels=credible_levels(Z),
                           colors=["#1a1a1a", "#8c2f2f"],
                           linewidths=[0.9, 1.0],
                           linestyles=[(0, (3, 2)), "-"])
                k = np.unravel_index(np.argmax(Z), Z.shape)
                ax.plot(GX[k], GY[k], marker="o", ms=3.4, mfc="#f0e442",
                        mec="#1a1a1a", mew=0.7, zorder=6)
                ax.set_xlim(x.min(), x.max())
                ax.set_ylim(y.min(), y.max())

            else:
                r = R[i, j]
                ax.set_facecolor(CORR_CMAP(cnorm(r)))
                shade = "#ffffff" if abs(r) > 0.55 else "#1a1a1a"
                ax.text(0.5, 0.5, f"{r:+.2f}", transform=ax.transAxes,
                        ha="center", va="center", color=shade,
                        fontsize=9 + 7 * abs(r), fontweight="bold")
                ax.set_xticks([])
                ax.set_yticks([])

            if i == p - 1 and (i != j or True):
                ax.set_xlabel(f"{names[j]}\n{units[j]}", fontsize=8,
                              linespacing=1.5)
            if i < p - 1 or i == j:
                ax.tick_params(labelbottom=(i == p - 1))
            if j == 0 and i > 0:
                ax.set_ylabel(f"{names[i]}\n{units[i]}", fontsize=8,
                              linespacing=1.5)
            if j > 0 or i == 0:
                ax.tick_params(labelleft=False)
            ax.tick_params(labelsize=6.5)

    axes[0, 0].set_ylabel(f"{names[0]}\n{units[0]}", fontsize=8,
                          linespacing=1.5)

    # ---- shared colour key for the correlation panels ----------------------
    cax = fig.add_axes([0.62, 0.905, 0.30, 0.016])
    cb = fig.colorbar(ScalarMappable(norm=cnorm, cmap=CORR_CMAP), cax=cax,
                      orientation="horizontal", ticks=[-1, -0.5, 0, 0.5, 1])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.2, width=0.6, length=2.2, pad=1.5)
    cb.ax.xaxis.set_label_position("top")
    cb.set_label("posterior correlation", fontsize=6.8, labelpad=3)

    handles = [
        plt.Line2D([], [], color="#8c2f2f", lw=1.0, label="50% credible region"),
        plt.Line2D([], [], color="#1a1a1a", lw=1.0, ls=(0, (3, 2)),
                   label="95% credible region"),
        plt.Line2D([], [], color="none", marker="o", ms=4, mfc="#f0e442",
                   mec="#1a1a1a", mew=0.7, label="joint mode"),
        plt.Line2D([], [], color="#9cc3dd", lw=5, label="94% HDI (marginal)"),
        plt.Line2D([], [], color="#b23a48", lw=1.1, ls=(0, (3, 2)),
                   label="posterior median"),
    ]

    savefig(fig, "128_posterior_pairs.png")


if __name__ == "__main__":
    main()

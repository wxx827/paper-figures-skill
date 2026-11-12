"""2x3 residual diagnostic grid: rows are models, columns are diagnostics.

Row 1 is a correctly specified linear model, row 2 a misspecified one (omitted
curvature + heteroscedastic, heavy-tailed errors), so the three classic
diagnostics - residuals vs fitted, normal Q-Q, scale-location - actually show
something. Each panel carries the matching test statistic, and the three most
extreme observations are labelled as in standard regression output.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pubstyle import (set_pub_style, PALETTES, savefig, style_axes,
                      add_panel_label)

MODELS = [("Model A: y ~ x  (well specified)", PALETTES["teal_rust"][0]),
          ("Model B: y ~ x  (omitted curvature)", PALETTES["teal_rust"][1])]
REF = "#b23a48"
N = 180


def lowess(x, y, frac=0.45, n=100):
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    xg = np.linspace(xs.min(), xs.max(), n)
    h = frac * (xs.max() - xs.min() + 1e-9)
    w = np.exp(-0.5 * ((xs[None, :] - xg[:, None]) / h) ** 2)
    return xg, (w @ ys) / w.sum(axis=1)


def simulate(rng, bad):
    x = rng.uniform(0, 10, N)
    if bad:
        # curvature the model does not know about + variance growing with x
        eps = rng.standard_t(4, N) * (0.35 + 0.16 * x)
        y = 2 + 0.7 * x + 0.085 * (x - 5) ** 2 + eps
    else:
        y = 2 + 0.7 * x + rng.normal(0, 0.85, N)
    b1, b0 = np.polyfit(x, y, 1)
    fitted = b0 + b1 * x
    resid = y - fitted
    return fitted, resid


def label_extremes(ax, xs, ys, k=3):
    idx = np.argsort(-np.abs(ys))[:k]
    for i in idx:
        ax.annotate(f"{i}", (xs[i], ys[i]), textcoords="offset points",
                    xytext=(4, 2), fontsize=6.5, color="#666666")


def box(ax, text, loc="upper left"):
    xa, ya, ha, va = ((0.035, 0.965, "left", "top") if loc == "upper left"
                      else (0.965, 0.035, "right", "bottom"))
    ax.text(xa, ya, text, transform=ax.transAxes, ha=ha, va=va, fontsize=7,
            linespacing=1.3, zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc",
                      lw=0.6, alpha=0.93))


def resid_vs_fitted(ax, fitted, resid, col):
    ax.axhline(0, color="#999999", lw=0.8, zorder=1)
    ax.scatter(fitted, resid, s=15, color=col, alpha=0.7, linewidths=0.35,
               edgecolors="white", zorder=3)
    xg, yg = lowess(fitted, resid)
    ax.plot(xg, yg, color=REF, lw=1.5, zorder=4)
    # Breusch-Pagan LM test for heteroscedasticity
    u = resid ** 2 / np.mean(resid ** 2)
    r = stats.linregress(fitted, u)
    lm = len(resid) * r.rvalue ** 2
    p = stats.chi2.sf(lm, 1)
    box(ax, f"Breusch-Pagan\nLM = {lm:.1f}, "
            f"{'p < 0.001' if p < 1e-3 else f'p = {p:.3f}'}")
    ax.set_xlabel("Fitted value", fontsize=7.5)
    ax.set_ylabel("Residual", fontsize=7.5)
    label_extremes(ax, fitted, resid)


def qq(ax, resid, col):
    z = resid / resid.std(ddof=2)
    osm, osr = stats.probplot(z, dist="norm")[0]
    ax.plot([osm.min(), osm.max()], [osm.min(), osm.max()], color=REF,
            lw=1.4, zorder=2)
    ax.scatter(osm, osr, s=15, color=col, alpha=0.7, linewidths=0.35,
               edgecolors="white", zorder=3)
    w, p = stats.shapiro(z)
    box(ax, f"Shapiro-Wilk\nW = {w:.3f}, "
            f"{'p < 0.001' if p < 1e-3 else f'p = {p:.3f}'}")
    ax.set_xlabel("Theoretical quantile", fontsize=7.5)
    ax.set_ylabel("Standardised residual", fontsize=7.5)


def scale_location(ax, fitted, resid, col):
    z = np.sqrt(np.abs(resid / resid.std(ddof=2)))
    ax.scatter(fitted, z, s=15, color=col, alpha=0.7, linewidths=0.35,
               edgecolors="white", zorder=3)
    xg, yg = lowess(fitted, z)
    ax.plot(xg, yg, color=REF, lw=1.5, zorder=4)
    rho, p = stats.spearmanr(fitted, np.abs(resid))
    box(ax, f"Spearman $\\rho$ = {rho:+.2f}\n"
            f"{'p < 0.001' if p < 1e-3 else f'p = {p:.3f}'}")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Fitted value", fontsize=7.5)
    ax.set_ylabel(r"$\sqrt{|\mathrm{std.\ residual}|}$", fontsize=7.5)


def main():
    set_pub_style()
    rng = np.random.default_rng(311)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
    titles = ["Residuals vs fitted", "Normal Q-Q", "Scale-location"]
    for r, (mname, col) in enumerate(MODELS):
        fitted, resid = simulate(rng, bad=bool(r))
        resid_vs_fitted(axes[r, 0], fitted, resid, col)
        qq(axes[r, 1], resid, col)
        scale_location(axes[r, 2], fitted, resid, col)
        for c in range(3):
            ax = axes[r, c]
            ax.set_title(titles[c] if r == 0 else "", fontsize=9,
                         fontweight="bold")
            ax.grid(lw=0.5, alpha=0.35)
            style_axes(ax)
            add_panel_label(ax, f"({'abcdef'[r * 3 + c]})", x=-0.19, y=1.02,
                            size=10)
        axes[r, 0].text(-0.30, 0.5, mname, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=8.5,
                        fontweight="bold", color=col)

    fig.tight_layout(rect=[0.015, 0.025, 1, 0.955], h_pad=2.6, w_pad=2.8)
    savefig(fig, "35_residual_grid.png")


if __name__ == "__main__":
    main()

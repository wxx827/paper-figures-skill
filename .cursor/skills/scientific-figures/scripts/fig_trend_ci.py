"""2x3 time-trend grid: scatter + OLS fit + 95% CI band + slope/p annotation.

Each panel is one nitrogen-budget term over time with raw points, a regression
line, a 95% confidence band, and a stats box giving slope (with units), R2 and
the p-value. The box automatically moves to whichever corner the data leave
free, so it never sits on top of the fit.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pubstyle import (set_pub_style, PALETTES, savefig, style_axes,
                      add_panel_label)

# (short title, unit, level in 1980, slope per year, noise s.d.)
VARS = [
    ("Total N input",       "Tg N yr$^{-1}$",       28.0,  0.42, 1.6),
    ("Biological fixation", "Tg N yr$^{-1}$",       11.5, -0.05, 0.9),
    ("Atmospheric deposit", "kg N ha$^{-1}$ yr$^{-1}$", 18.0, -0.21, 1.1),
    ("Fertilizer use",      "Tg N yr$^{-1}$",       14.0,  0.55, 1.9),
    ("N loss to air",       "Tg N yr$^{-1}$",        9.5,  0.11, 1.4),
    ("N loss to water",     "Tg N yr$^{-1}$",        6.2,  0.19, 0.7),
]
C, C_DARK = PALETTES["teal_rust"][0], "#164a46"


def p_text(p):
    return "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"


def panel(ax, rng, idx, spec):
    name, unit, level, slope, sd = spec
    years = np.arange(1980, 2021)
    x = years - years[0]
    # gentle curvature + AR(1) residuals: no textbook-perfect straight line
    e = np.zeros(len(x))
    for i in range(1, len(x)):
        e[i] = 0.45 * e[i - 1] + rng.normal(0, 1)
    y = (level + slope * x + 0.004 * slope * x ** 2
         + e * sd / np.sqrt(1 / (1 - 0.45 ** 2)) + rng.normal(0, sd * 0.5, len(x)))

    ax.scatter(years, y, s=17, color=C, alpha=0.75, linewidths=0.4,
               edgecolors="white", zorder=3)

    res = stats.linregress(years, y)
    xr = np.linspace(years[0], years[-1], 200)
    yhat = res.intercept + res.slope * xr
    n = len(years)
    se = np.sqrt(np.sum((y - (res.intercept + res.slope * years)) ** 2) / (n - 2))
    tval = stats.t.ppf(0.975, n - 2)
    sx = np.sqrt(np.sum((years - years.mean()) ** 2))
    ci = tval * se * np.sqrt(1 / n + (xr - years.mean()) ** 2 / sx ** 2)
    ax.fill_between(xr, yhat - ci, yhat + ci, color=C, alpha=0.20, lw=0,
                    zorder=2)
    ax.plot(xr, yhat, color=C_DARK, lw=1.7, zorder=4)

    # keep the callout in the corner the regression line does not sweep into
    if res.slope >= 0:
        xa, ya, ha, va = 0.035, 0.965, "left", "top"
    else:
        xa, ya, ha, va = 0.035, 0.035, "left", "bottom"
    ax.text(xa, ya,
            f"slope = {res.slope:+.3f} yr$^{{-1}}$\n"
            f"$R^2$ = {res.rvalue ** 2:.2f},  {p_text(res.pvalue)}",
            transform=ax.transAxes, fontsize=7.2, ha=ha, va=va,
            linespacing=1.35, zorder=6,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#cccccc",
                      lw=0.6, alpha=0.93))

    ax.set_title(name, fontsize=9, fontweight="bold")
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel(unit, fontsize=8)
    ax.set_xlim(1978, 2022)
    ax.set_xticks(np.arange(1980, 2021, 10))
    ax.grid(axis="y", lw=0.5, alpha=0.45)
    style_axes(ax)
    add_panel_label(ax, f"({'abcdef'[idx]})", x=-0.17, y=1.02, size=10)


def main():
    set_pub_style()
    rng = np.random.default_rng(221)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    for k, (ax, spec) in enumerate(zip(axes.ravel(), VARS)):
        panel(ax, rng, k, spec)
    fig.tight_layout(rect=[0, 0.025, 1, 0.955], h_pad=2.5, w_pad=2.6)
    savefig(fig, "29_trend_ci.png")


if __name__ == "__main__":
    main()

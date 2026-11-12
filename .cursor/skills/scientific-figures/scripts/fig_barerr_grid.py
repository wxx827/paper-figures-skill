"""2x3 grouped bar grid with SEM error bars + significance brackets.

Each panel compares two treatments across the four seasons. Bar heights are
sample means of simulated plot-level measurements, whiskers are +/-1 SEM, and
the stars come from an actual Welch t-test on those samples (not decoration).
Shared legend on top, panel labels (a)-(f).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pubstyle import (set_pub_style, PALETTES, savefig, style_axes,
                      add_panel_label)

CATS = ["Spring", "Summer", "Autumn", "Winter"]
C1, C2 = PALETTES["cb_safe"][0], PALETTES["cb_safe"][1]
N_PLOTS = 24

# (title, unit, level, spread, seasonal shape, treatment effect per season)
PANELS = [
    ("Water yield",        "mm month$^{-1}$",      42.0, 6.0,
     [-.35, .85, .15, -.65], [.55, -.10, -.62, -.35]),
    ("Carbon storage",     "Mg C ha$^{-1}$",       64.0, 5.5,
     [-.10, .25, .05, -.20], [-.75, -.90, -.55, -.20]),
    ("Habitat quality",    "index (0-1)",           0.68, 0.09,
     [.05, .30, .00, -.35], [-.30, -.85, -1.15, -.25]),
    ("Soil conservation",  "t ha$^{-1}$ month$^{-1}$", 18.0, 3.2,
     [.45, .95, .20, -.60], [-.65, .35, 1.05, -.55]),
    ("Net primary prod.",  "g C m$^{-2}$ d$^{-1}$", 3.6, 0.55,
     [-.20, 1.05, .35, -1.15], [.42, .55, .90, -.12]),
    ("Comprehensive index", "z-score",              0.00, 0.75,
     [-.15, .35, .10, -.30], [.55, .95, 1.05, -.45]),
]


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def panel(ax, rng, idx, spec):
    name, unit, level, spread, season, effect = spec
    x = np.arange(len(CATS))
    w = 0.36

    a = [rng.normal(level + spread * s, spread * 0.9, N_PLOTS) for s in season]
    b = [rng.normal(level + spread * (s + e), spread * 0.95, N_PLOTS)
         for s, e in zip(season, effect)]
    m1 = np.array([v.mean() for v in a])
    m2 = np.array([v.mean() for v in b])
    e1 = np.array([v.std(ddof=1) / np.sqrt(N_PLOTS) for v in a])
    e2 = np.array([v.std(ddof=1) / np.sqrt(N_PLOTS) for v in b])

    ax.bar(x - w / 2, m1, w, yerr=e1, color=C1, edgecolor="white", lw=0.6,
           capsize=2.5, error_kw=dict(lw=0.9, ecolor="#333333"),
           label="Inside reserve", zorder=3)
    ax.bar(x + w / 2, m2, w, yerr=e2, color=C2, edgecolor="white", lw=0.6,
           capsize=2.5, error_kw=dict(lw=0.9, ecolor="#333333"),
           label="Outside reserve", zorder=3)

    top = float(np.max(np.concatenate([m1 + e1, m2 + e2])))
    bot = min(0.0, float(np.min(np.concatenate([m1 - e1, m2 - e2]))))
    rng_y = top - bot
    for xi, (va, vb) in enumerate(zip(a, b)):
        p = stats.ttest_ind(va, vb, equal_var=False).pvalue
        yb = max(m1[xi] + e1[xi], m2[xi] + e2[xi]) + rng_y * 0.045
        h = rng_y * 0.028
        ax.plot([xi - w / 2, xi - w / 2, xi + w / 2, xi + w / 2],
                [yb, yb + h, yb + h, yb], color="#444444", lw=0.8, zorder=4)
        s = stars(p)
        ax.text(xi, yb + h * 1.1, s, ha="center", va="bottom",
                fontsize=7 if s == "n.s." else 8.5,
                fontweight="normal" if s == "n.s." else "bold",
                color="#333333", zorder=4)

    ax.set_ylim(bot, top + rng_y * 0.22)
    ax.set_xticks(x)
    ax.set_xticklabels(CATS, fontsize=7.5)
    ax.set_xlim(-0.6, len(CATS) - 0.4)
    ax.set_title(name, fontsize=9, fontweight="bold")
    ax.set_ylabel(unit, fontsize=8)
    ax.grid(axis="y", lw=0.5, alpha=0.4)
    style_axes(ax)
    add_panel_label(ax, f"({'abcdef'[idx]})", x=-0.17, y=1.02, size=10)


def main():
    set_pub_style()
    rng = np.random.default_rng(251)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for k, (ax, spec) in enumerate(zip(axes.ravel(), PANELS)):
        panel(ax, rng, k, spec)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.025, 1, 0.95], h_pad=2.5, w_pad=2.6)
    savefig(fig, "32_barerr_grid.png")


if __name__ == "__main__":
    main()

"""Beta-convergence scatter: initial income versus subsequent growth.

Each bubble is an economy, sized by population and coloured by region
group; the negatively sloped OLS fit with its 95% mean-prediction band is
the convergence relationship.  Quadrant rules at the sample means split
catch-up economies from those falling behind, and the call-out reports the
convergence coefficient, the implied speed lambda and the half-life.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes

T_YEARS = 29

GROUPS = [
    ("Advanced economies", 10.75, 0.30, 0.0, "#3b5b92", 22),
    ("Emerging Asia", 9.05, 0.42, 1.35, "#3f8f7a", 26),
    ("Latin America & Caribbean", 9.45, 0.34, -0.55, "#c98b3a", 20),
    ("Sub-Saharan Africa", 8.15, 0.38, -0.30, "#a8564f", 24),
]

# names are group-consistent so the annotation never contradicts the colour
NAMES = {
    0: ["Baltic Rim", "Alpine Core", "Northern Isles"],
    1: ["Coastal East", "Delta Corridor", "Monsoon Basin"],
    2: ["Andes Basin", "Pampas South", "Isthmus North"],
    3: ["Sahel Belt", "Great Lakes", "Savanna East"],
}


def simulate(rng):
    x, g, grp, pop = [], [], [], []
    for k, (_, mu, sd, bonus, _, n) in enumerate(GROUPS):
        xi = rng.normal(mu, sd, n)
        gi = 11.6 - 1.05 * xi + bonus + rng.normal(0, 0.62, n)
        pi = np.exp(rng.normal(2.4, 1.15, n))          # million people
        x.append(xi)
        g.append(gi)
        grp.append(np.full(n, k))
        pop.append(pi)
    return (np.concatenate(x), np.concatenate(g), np.concatenate(grp),
            np.concatenate(pop))


def ols_with_band(x, y, xg):
    n = x.size
    b, a = np.polyfit(x, y, 1)
    fit = a + b * x
    resid = y - fit
    s2 = resid @ resid / (n - 2)
    sxx = np.sum((x - x.mean()) ** 2)
    se_b = np.sqrt(s2 / sxx)
    yg = a + b * xg
    se_fit = np.sqrt(s2 * (1.0 / n + (xg - x.mean()) ** 2 / sxx))
    r2 = 1.0 - resid @ resid / np.sum((y - y.mean()) ** 2)
    return b, se_b, a, r2, yg, 1.96 * se_fit


def main():
    set_pub_style()
    rng = np.random.default_rng(23)
    x, y, grp, pop = simulate(rng)

    xg = np.linspace(x.min() - 0.15, x.max() + 0.15, 200)
    b, se_b, a, r2, yg, band = ols_with_band(x, y, xg)

    beta = b / 100.0                       # decimal growth per log point
    lam = -np.log(1.0 + beta * T_YEARS) / T_YEARS
    half = np.log(2.0) / lam

    fig, ax = plt.subplots(figsize=(7.4, 4.9))

    ax.axvline(x.mean(), color="#b8b8b8", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.axhline(y.mean(), color="#b8b8b8", lw=0.8, ls=(0, (4, 3)), zorder=1)

    ax.fill_between(xg, yg - band, yg + band, color="#4a4a4a", alpha=0.13,
                    lw=0, zorder=2)
    ax.plot(xg, yg, color="#2b2b2b", lw=1.6, zorder=4)

    sizes = 14.0 + 150.0 * np.sqrt(pop / pop.max())
    for k, (name, _, _, _, col, _) in enumerate(GROUPS):
        m = grp == k
        ax.scatter(x[m], y[m], s=sizes[m], facecolor=col, edgecolor="white",
                   lw=0.6, alpha=0.85, zorder=3, label=name)

    # representative economies: strong residuals and the largest economies,
    # thinned so that no two labels sit on top of each other
    resid = y - (a + b * x)
    order = list(np.argsort(-np.abs(resid))) + list(np.argsort(-pop)[:6])
    picked = []
    for idx in dict.fromkeys(int(i) for i in order):
        if all(abs(x[idx] - x[j]) > 0.55 or abs(y[idx] - y[j]) > 0.95
               for j in picked):
            picked.append(idx)
        if len(picked) == 5:
            break

    used = {k: 0 for k in NAMES}
    for idx in picked:
        g = int(grp[idx])
        if used[g] >= len(NAMES[g]):
            continue
        name = NAMES[g][used[g]]
        used[g] += 1
        dx = 0.14 if x[idx] < x.mean() else -0.14
        dy = 0.45 if resid[idx] >= 0 else -0.48
        ax.annotate(name, xy=(x[idx], y[idx]),
                    xytext=(x[idx] + dx, y[idx] + dy), fontsize=6.6,
                    color="#3a3a3a", ha="left" if dx > 0 else "right",
                    va="center", zorder=6,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#9a9a9a",
                                    shrinkA=0, shrinkB=3))

    ticks = np.log(np.array([1500, 4000, 10000, 25000, 60000]))
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1.5k", "4k", "10k", "25k", "60k"])
    ax.set_xlim(x.min() - 0.42, x.max() + 0.42)
    ax.set_ylim(y.min() - 0.9, y.max() + 1.35)
    ax.set_xlabel("GDP per capita in 1995 (constant USD, log scale)")
    ax.set_ylabel("Average annual growth of GDP per capita,\n1995-2024 "
                  "(%, per year)", linespacing=1.4)
    ax.set_title("Beta-convergence across 92 economies", fontsize=9.5,
                 loc="left", pad=8, weight="bold")

    q = dict(fontsize=6.5, color="#9a9a9a", zorder=2)
    ax.text(0.015, 0.975, "poor & fast-growing\n(catching up)",
            transform=ax.transAxes, ha="left", va="top", linespacing=1.3, **q)
    ax.text(0.985, 0.975, "rich & fast-growing", transform=ax.transAxes,
            ha="right", va="top", **q)
    ax.text(0.015, 0.025, "poor & slow-growing\n(falling behind)",
            transform=ax.transAxes, ha="left", va="bottom", linespacing=1.3,
            **q)
    ax.text(0.985, 0.025, "rich & slow-growing", transform=ax.transAxes,
            ha="right", va="bottom", **q)

    ax.text(0.985, 0.845,
            f"$\\beta$ = {b:.3f} (s.e. {se_b:.3f})***\n"
            f"$R^2$ = {r2:.2f},  N = {x.size}\n"
            f"speed $\\lambda$ = {100 * lam:.2f}% per year\n"
            f"half-life = {half:.0f} years",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.2,
            linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#d5d5d5",
                      lw=0.6, alpha=0.95))

    grp_handles = [Line2D([], [], marker="o", ls="none", ms=5.2,
                          mfc=col, mec="white", mew=0.6, label=name)
                   for name, _, _, _, col, _ in GROUPS]
    fig.legend(handles=grp_handles, loc="lower left", fontsize=7,
               frameon=False, handletextpad=0.5, labelspacing=0.45,
               bbox_to_anchor=(0.055, 0.045), ncol=2, columnspacing=1.4)

    ref = np.array([5.0, 40.0, 200.0])
    size_handles = [
        Line2D([], [], marker="o", ls="none",
               ms=np.sqrt(14.0 + 150.0 * np.sqrt(p / pop.max())),
               mfc="#bfbfbf", mec="white", mew=0.6, label=f"{p:.0f}")
        for p in ref]
    fig.legend(handles=size_handles, loc="lower right", fontsize=7,
               frameon=False, title="Population (millions)",
               title_fontsize=7, bbox_to_anchor=(0.985, 0.045), ncol=3,
               handletextpad=0.5, columnspacing=1.2, borderpad=0.2)

    ax.grid(lw=0.5, color="#f0f0f0")
    ax.set_axisbelow(True)
    style_axes(ax)

    fig.tight_layout(rect=(0.0, 0.115, 1.0, 1.0))
    savefig(fig, "110_convergence_scatter.png")


if __name__ == "__main__":
    main()

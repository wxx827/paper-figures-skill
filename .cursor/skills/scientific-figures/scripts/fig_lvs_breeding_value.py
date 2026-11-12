"""Estimated breeding values, the selection cut, and what the cut buys.

The main panel is the decision: a distribution of estimated breeding values
for one cohort of candidates, a truncation threshold, and the tail that gets
kept.  Selection differential and intensity are measured off the kept animals
rather than taken from a table.  Because BLUP EBVs are unbiased predictors,
the realised gain in true breeding value matches the differential - the cost
of imperfect accuracy shows up somewhere else, in the *width* of the EBV
distribution: it has standard deviation r*sigma_A rather than sigma_A, so the
same selection intensity buys less.  The three reliability curves behind the
histogram are where that narrowing comes from, a low-reliability EBV being
pulled toward the mean.

The subordinate panels are the validation and the record: EBV against
phenotype for the same cohort, and twenty-one years of herd genetic trend.
Synthetic data; no real population is represented.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from pubstyle import set_pub_style, get_palette, savefig, despine
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
BASE = "#808080"
BEIGE = "#efe9dc"

SIGMA_A = 6.0          # additive genetic standard deviation, kg
H2 = 0.34              # narrow-sense heritability of the trait
MU_P = 92.0            # phenotypic mean, kg
N_CAND = 1500
SELECT_P = 0.15        # proportion kept
RELIABILITIES = [0.35, 0.60, 0.85]
YEARS = np.arange(2005, 2026)
N_PER_YEAR = 220
TREND = 0.42           # kg of breeding value gained per year


def cohort(rng):
    """True breeding values, EBVs and phenotypes for one cohort."""
    a = rng.normal(0.0, SIGMA_A, N_CAND)
    rel = np.array(RELIABILITIES)[rng.integers(0, len(RELIABILITIES),
                                               N_CAND)]
    # BLUP property: EBV = r2 * A + e with var(e) = r2 (1 - r2) sigma_a^2,
    # so var(EBV) = r2 sigma_a^2 and corr(A, EBV) = sqrt(r2).
    ebv = rel * a + rng.normal(0.0, np.sqrt(rel * (1.0 - rel)) * SIGMA_A)
    env = rng.normal(0.0, SIGMA_A * np.sqrt((1.0 - H2) / H2), N_CAND)
    return a, ebv, rel, MU_P + a + env


def history(rng):
    yrs, ebv = [], []
    for y in YEARS:
        mu = TREND * (y - YEARS[0])
        k = N_PER_YEAR + int(rng.integers(-30, 31))
        ebv.append(rng.normal(mu, SIGMA_A * np.sqrt(0.60), k))
        yrs.append(np.full(k, y))
    return np.concatenate(yrs), np.concatenate(ebv)


def main_panel(ax, a, ebv, rel):
    thr = float(np.quantile(ebv, 1.0 - SELECT_P))
    keep = ebv >= thr
    sd = float(np.std(ebv, ddof=1))
    diff = float(ebv[keep].mean() - ebv.mean())
    inten = diff / sd
    realised = float(a[keep].mean() - a.mean())

    # One histogram, one normalisation: splitting it in two and asking for
    # density twice would scale each half by its own count and make the kept
    # tail tower over the rest.  Bins are built outward from the threshold so
    # it falls exactly on an edge.
    w = (ebv.max() - ebv.min()) / 45.0
    bins = np.unique(np.concatenate([
        np.arange(thr, ebv.min() - w, -w)[::-1],
        np.arange(thr, ebv.max() + w, w)]))
    _, edges, patches = ax.hist(ebv, bins=bins, density=True, color=BEIGE,
                                edgecolor="white", lw=0.5, zorder=2)
    for e, p in zip(edges[:-1], patches):
        if e >= thr - 1e-9:
            p.set_facecolor(GREEN)
            p.set_alpha(0.85)

    grid = np.linspace(ebv.min() - 1.5, ebv.max() + 1.5, 400)
    dens = gaussian_kde(ebv)(grid)
    ax.plot(grid, dens, color=INK, lw=1.7, zorder=6)
    m = grid >= thr
    ax.fill_between(grid[m], 0.0, dens[m], facecolor=GREEN, alpha=0.30,
                    lw=0, zorder=4)

    shades = get_palette("cb_sequential_blue", 7)[3:6]
    peaks = [float(dens.max()), float(max(p.get_height() for p in patches))]
    for r, col in zip(RELIABILITIES, shades):
        s = ebv[np.isclose(rel, r)]
        g = gaussian_kde(s)(grid)
        peaks.append(float(g.max()))
        ax.plot(grid, g, color=col, lw=1.3, ls=(0, (4.0, 2.0)), zorder=5,
                label=f"reliability $r^2$={r:.2f} "
                      f"(n={s.size}, SD {np.std(s, ddof=1):.1f} kg)")

    ax.axvline(thr, color=RED, lw=1.6, zorder=7)
    ax.text(thr + 0.4, 0.70, f"truncation threshold {thr:+.1f} kg",
            transform=ax.get_xaxis_transform(), ha="left", va="center",
            fontsize=6.9, color=RED)
    ax.axvline(0.0, color=BASE, lw=0.9, ls=(0, (3, 2.4)), zorder=2)

    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0.0, max(peaks) * 1.16)
    ax.set_xlabel("estimated breeding value, EBV (kg)", fontsize=8.0)
    ax.set_ylabel("density (1/kg)", fontsize=8.0)
    ax.tick_params(labelsize=7.2)
    despine(ax)
    ax.grid(axis="y", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_title("Truncation selection on EBV, and what it actually "
                 "transmits", fontsize=9.0, fontweight="bold", color=INK)

    h, lab = ax.get_legend_handles_labels()
    h = [plt.Line2D([], [], color=INK, lw=1.7),
         plt.Rectangle((0, 0), 1, 1, facecolor=GREEN, alpha=0.85,
                       edgecolor="white"),
         plt.Rectangle((0, 0), 1, 1, facecolor=BEIGE, edgecolor="white")] + h
    lab = ["kernel density, all candidates",
           f"kept: top {100 * SELECT_P:.0f} % (n={int(keep.sum())})",
           f"culled (n={int((~keep).sum())})"] + lab
    leg = ax.legend(h, lab, fontsize=6.8, frameon=True, framealpha=0.94,
                    facecolor="white", edgecolor="#cccccc", loc="upper left",
                    handlelength=1.9, borderpad=0.42, borderaxespad=0.5)
    leg.set_zorder(9)

    acc = sd / SIGMA_A
    ceiling = inten * SIGMA_A
    ax.text(0.985, 0.965,
            f"$S$ = {diff:+.2f} kg,   $i$ = $S$/SD = {inten:.2f}\n"
            f"realised $\\Delta$BV = {realised:+.2f} kg\n"
            f"SD(EBV)/$\\sigma_A$ = {acc:.2f}\n"
            f"perfect EBVs: $i\\sigma_A$ = {ceiling:+.2f} kg",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
            color=INK, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.38", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.7), zorder=9)
    return thr, diff, inten, realised, acc, ceiling


def scatter_panel(ax, rng, a, ebv, pheno, n_show=420):
    sel = rng.choice(ebv.size, n_show, replace=False)
    ax.scatter(ebv[sel], pheno[sel], s=8.0, facecolor=BLUE, edgecolor="none",
               alpha=0.40, zorder=3)
    slope, icept = np.polyfit(ebv, pheno, 1)
    xs = np.linspace(ebv.min(), ebv.max(), 40)
    ax.plot(xs, slope * xs + icept, color=RED, lw=1.6, zorder=5)
    h2 = float(np.var(a, ddof=1) / np.var(pheno, ddof=1))
    r = float(np.corrcoef(ebv, pheno)[0, 1])
    ax.text(0.035, 0.955, f"slope = {slope:.2f} kg/kg (BLUP expects 1)\n"
                          f"realised $h^2$ = {h2:.2f},  r = {r:.2f}\n"
                          f"n={ebv.size} candidates, {n_show} drawn",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
            color=INK, linespacing=1.55)
    ax.set_xlabel("EBV (kg)", fontsize=7.6)
    ax.set_ylabel("phenotype (kg)", fontsize=7.6)
    ax.tick_params(labelsize=6.8)
    despine(ax)
    ax.grid(lw=0.45, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("EBV against phenotype", fontsize=8.4, fontweight="bold",
                 color=INK)
    return slope, h2


def trend_panel(ax, yrs, ebv):
    ys = np.unique(yrs)
    mu = np.array([ebv[yrs == y].mean() for y in ys])
    se = np.array([ebv[yrs == y].std(ddof=1) / np.sqrt((yrs == y).sum())
                   for y in ys])
    ax.fill_between(ys, mu - 1.96 * se, mu + 1.96 * se, facecolor=BLUE,
                    alpha=0.20, lw=0, zorder=2)
    ax.plot(ys, mu, color=BLUE, lw=1.6, marker="o", ms=2.8, mfc="white",
            mew=0.8, zorder=4)
    slope, icept = np.polyfit(ys, mu, 1)
    ax.plot(ys, slope * ys + icept, color=RED, lw=1.3, ls=(0, (4.5, 2.4)),
            zorder=5)
    ax.axhline(0.0, color=BASE, lw=0.9, ls=(0, (3, 2.4)), zorder=1)
    ax.text(0.035, 0.955, f"genetic trend {slope:+.2f} kg per year\n"
                          f"{mu[-1] - mu[0]:+.1f} kg over "
                          f"{ys[-1] - ys[0]:.0f} years",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
            color=INK, linespacing=1.55)
    ax.set_xlabel("birth year", fontsize=7.6)
    ax.set_ylabel("mean EBV (kg)", fontsize=7.6)
    ax.tick_params(labelsize=6.8)
    ax.set_xticks(ys[::5])
    despine(ax)
    ax.grid(axis="y", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.text(0.985, 0.045, "band = 95 % CI of the annual mean",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3,
            color=SUB)
    ax.set_title("Herd genetic trend", fontsize=8.4, fontweight="bold",
                 color=INK)
    return slope


def main():
    set_pub_style()
    rng = np.random.default_rng(20845)
    a, ebv, rel, pheno = cohort(rng)
    yrs, hist_ebv = history(rng)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (0.86, 0.98, 4.98, 4.58))
    ax_sc = axes_inches(fig, (7.20, 3.54, 2.74, 2.02))
    ax_tr = axes_inches(fig, (7.20, 0.98, 2.74, 2.02))

    thr, diff, inten, realised, acc, ceiling = main_panel(ax_main, a, ebv, rel)
    slope, h2 = scatter_panel(ax_sc, rng, a, ebv, pheno)
    trend = trend_panel(ax_tr, yrs, hist_ebv)

    panel_labels_inches(fig, [ax_main, ax_sc, ax_tr], dx=-0.54, dy=0.14)
    savefig(fig, "845_lvs_breeding_value.png")


if __name__ == "__main__":
    main()

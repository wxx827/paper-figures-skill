"""Species-specific biomass allometry and where the carbon is actually put.

One synthetic stem list drives all three panels.  Each tree gets a total dry
biomass from a species power law with log-normal scatter, then that mass is
split between stem, branch, foliage and root by size-dependent allocation
rules, so the stacked panel is a partition of the very masses plotted in the
main panel rather than a second, unrelated invention.

Every number on the figure is refitted from those trees: the four allometric
exponents and their standard errors come from ordinary least squares on
log10 axes, the shaded bands are the t-based 95 % confidence interval of the
fitted mean, and the residual panel reports the pooled scatter of the same
fits.  Species colours are the shared forest set used in 684 and 688.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from pubstyle import set_pub_style, savefig, get_palette
from figkit import axes_inches, panel_labels_inches, shared_limits

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"

SPECIES = ["Picea abies", "Pinus sylvestris", "Betula pendula",
           "Quercus robur"]
SPECIES_COLOR = {
    "Picea abies": "#2e7d5b",
    "Pinus sylvestris": "#b23a48",
    "Betula pendula": "#c9a227",
    "Quercus robur": "#3b5b92",
}
# generating power law: B = a * D^b (kg dry mass, D in cm)
SPECIES_LAW = {
    "Picea abies": (0.075, 2.45, 0.19),
    "Pinus sylvestris": (0.055, 2.52, 0.22),
    "Betula pendula": (0.098, 2.38, 0.24),
    "Quercus robur": (0.115, 2.42, 0.21),
}

ORGANS = ["stem", "branch", "foliage", "root"]
ORGAN_COLOR = dict(zip(ORGANS, get_palette("greens", 4)))

FW, FH = 10.4, 6.4
MAIN_RECT = (0.70, 1.04, 5.34, 4.14)
D_LO, D_HI = 5.0, 72.0
N_PER_SPECIES = 58
N_CLASS = 8
D_TICKS = [5, 10, 20, 40, 70]


def log_dbh_axis(ax):
    """Readable decade-free ticks; the default log locator shows only 10."""
    ax.set_xticks(D_TICKS)
    ax.set_xticklabels([str(t) for t in D_TICKS])
    ax.set_xticks([], minor=True)


def allocation(d, rng):
    """Organ fractions of total dry mass; foliage falls, branch rises."""
    fol = 0.030 + 0.255 * np.exp(-0.045 * d)
    bra = 0.085 + 0.150 * (1.0 - np.exp(-0.032 * d))
    root = 0.255 - 0.045 * (1.0 - np.exp(-0.028 * d))
    stem = 1.0 - fol - bra - root
    f = np.stack([stem, bra, fol, root], axis=-1)
    f = f * np.exp(rng.normal(0.0, 0.085, f.shape))
    return f / f.sum(axis=-1, keepdims=True)


def build_trees(rng):
    rows = []
    for s in SPECIES:
        a, b, sd = SPECIES_LAW[s]
        d = 10.0 ** rng.uniform(np.log10(D_LO), np.log10(D_HI),
                                N_PER_SPECIES)
        mass = a * d ** b * np.exp(rng.normal(0.0, sd, d.size))
        frac = allocation(d, rng)
        rows.append(dict(species=s, dbh=d, mass=mass, frac=frac))
    return rows


def fit_power(d, mass):
    """OLS on log10 axes; returns slope, its s.e., intercept and residuals."""
    lx, ly = np.log10(d), np.log10(mass)
    n = lx.size
    slope, intercept = np.polyfit(lx, ly, 1)
    resid = ly - (intercept + slope * lx)
    s = np.sqrt((resid ** 2).sum() / (n - 2))
    sxx = ((lx - lx.mean()) ** 2).sum()
    return dict(slope=slope, intercept=intercept, se=s / np.sqrt(sxx),
                s=s, n=n, mean=lx.mean(), sxx=sxx, resid=resid)


def draw_main(ax, trees, fits):
    for tr in trees:
        s = tr["species"]
        f = fits[s]
        ax.scatter(tr["dbh"], tr["mass"], s=8.0, color=SPECIES_COLOR[s],
                   alpha=0.80, edgecolors="white", linewidths=0.3, zorder=4)
        gx = np.linspace(np.log10(D_LO), np.log10(D_HI), 160)
        gy = f["intercept"] + f["slope"] * gx
        tcrit = stats.t.ppf(0.975, f["n"] - 2)
        half = tcrit * f["s"] * np.sqrt(1.0 / f["n"] +
                                        (gx - f["mean"]) ** 2 / f["sxx"])
        ax.fill_between(10.0 ** gx, 10.0 ** (gy - half), 10.0 ** (gy + half),
                        color=SPECIES_COLOR[s], alpha=0.20, linewidth=0.0,
                        zorder=3)
        ax.plot(10.0 ** gx, 10.0 ** gy, color=SPECIES_COLOR[s], lw=1.4,
                zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(D_LO * 0.82, D_HI * 1.20)
    ax.set_xlabel("DBH (cm)", fontsize=8.0)
    ax.set_ylabel("total dry biomass (kg)", fontsize=8.0)
    ax.tick_params(labelsize=7.0, which="both")
    log_dbh_axis(ax)
    ax.grid(True, which="major", color="#e2e2e2", lw=0.5)
    ax.set_title("Log-log allometry with 95 % confidence bands",
                 fontsize=8.5, fontweight="bold")

    lines = ["allometric exponent  b"]
    for s in SPECIES:
        lines.append(f"{s}   {fits[s]['slope']:.3f} "
                     f"$\\pm$ {fits[s]['se']:.3f}")
    ax.text(0.030, 0.965, "\n".join(lines), transform=ax.transAxes,
            ha="left", va="top", fontsize=6.8, color=INK, linespacing=1.55,
            bbox=dict(boxstyle="round,pad=0.38", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.92))
    ax.text(0.975, 0.055,
            "$B = a\\,D^{b}$, geometric scaling predicts $b \\approx 8/3$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6,
            color=SUB)


def draw_allocation(ax, trees):
    d = np.concatenate([tr["dbh"] for tr in trees])
    f = np.concatenate([tr["frac"] for tr in trees], axis=0)
    edges = 10.0 ** np.linspace(np.log10(D_LO), np.log10(D_HI), N_CLASS + 1)
    idx = np.clip(np.digitize(d, edges) - 1, 0, N_CLASS - 1)
    mid = np.sqrt(edges[:-1] * edges[1:])
    prof = np.array([f[idx == k].mean(axis=0) * 100.0 for k in range(N_CLASS)])

    # extend the first and last class to the axis limits so the stack has no
    # white sliver at either end
    xs = np.concatenate([[D_LO], mid, [D_HI]])
    ys = np.vstack([prof[:1], prof, prof[-1:]])
    ax.stackplot(xs, ys.T, colors=[ORGAN_COLOR[o] for o in ORGANS],
                 edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(D_LO, D_HI)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("DBH (cm)", fontsize=7.8)
    ax.set_ylabel("share of total mass (%)", fontsize=7.8)
    ax.tick_params(labelsize=6.8, which="both")
    log_dbh_axis(ax)
    ax.set_title("Organ allocation shifts with size", fontsize=8.5,
                 fontweight="bold")
    cum = np.cumsum(prof, axis=1)
    j = N_CLASS // 2
    for k, o in enumerate(ORGANS):
        lo = 0.0 if k == 0 else cum[j, k - 1]
        y = 0.5 * (lo + cum[j, k])
        col = "white" if k < 2 else INK
        ax.text(mid[j], y, f"{o}  {prof[j, k]:.0f} %", ha="center",
                va="center", fontsize=6.2, color=col, zorder=6)
    return prof, mid


def draw_residuals(ax, trees, fits):
    pooled = []
    for tr in trees:
        s = tr["species"]
        ax.scatter(tr["dbh"], fits[s]["resid"], s=6.5,
                   color=SPECIES_COLOR[s], alpha=0.80, edgecolors="white",
                   linewidths=0.25, zorder=4)
        pooled.append(fits[s]["resid"])
    pooled = np.concatenate(pooled)
    rmse = float(np.sqrt(np.mean(pooled ** 2)))
    ax.axhline(0.0, color="#808080", lw=0.9, zorder=3)
    ax.axhspan(-rmse, rmse, color="#cccccc", alpha=0.35, linewidth=0.0,
               zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(D_LO * 0.82, D_HI * 1.20)
    lo, hi = shared_limits([pooled], pad=0.14, symmetric=True)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("DBH (cm)", fontsize=7.8)
    ax.set_ylabel("residual ($log_{10}$ kg)", fontsize=7.8)
    ax.tick_params(labelsize=6.8, which="both")
    log_dbh_axis(ax)
    ax.set_title("Model residuals show no size trend", fontsize=8.5,
                 fontweight="bold")
    ax.text(0.975, 0.055, f"pooled RMSE {rmse:.3f} dex "
                          f"($\\times${10.0 ** rmse:.2f} in mass)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4,
            color=SUB, zorder=6,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.92))
    return rmse


def main():
    set_pub_style()
    rng = np.random.default_rng(685)
    trees = build_trees(rng)
    fits = {tr["species"]: fit_power(tr["dbh"], tr["mass"]) for tr in trees}

    fig = plt.figure(figsize=(FW, FH))
    ax_m = axes_inches(fig, MAIN_RECT)
    ax_a = axes_inches(fig, (7.14, 3.28, 2.88, 1.90))
    ax_r = axes_inches(fig, (7.14, 1.04, 2.88, 1.60))

    draw_main(ax_m, trees, fits)
    prof, mid = draw_allocation(ax_a, trees)
    rmse = draw_residuals(ax_r, trees, fits)

    handles = [Line2D([0], [0], marker="o", ls="none", ms=4.4,
                      mfc=SPECIES_COLOR[s], mec="white", mew=0.5, label=s)
               for s in SPECIES]
    handles += [Line2D([0], [0], marker="s", ls="none", ms=4.8,
                       mfc=ORGAN_COLOR[o], mec="white", mew=0.5, label=o)
                for o in ORGANS]

    panel_labels_inches(fig, [ax_m, ax_a, ax_r], dx=-0.36, dy=0.20)

    bs = np.array([fits[s]["slope"] for s in SPECIES])
    fol_drop = prof[0, 2] - prof[-1, 2]

    print("exponents:", {s: round(fits[s]["slope"], 3) for s in SPECIES},
          f"rmse={rmse:.4f} fol {prof[0, 2]:.1f}->{prof[-1, 2]:.1f} "
          f"stem {prof[0, 0]:.1f}->{prof[-1, 0]:.1f}")
    savefig(fig, "685_for_biomass_allometry.png")


if __name__ == "__main__":
    main()

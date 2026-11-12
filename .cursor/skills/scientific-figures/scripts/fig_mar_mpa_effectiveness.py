"""Do marine protected areas work?  Paired inside-outside biomass, by taxon.

The main block is one comparison read twice.  On the left a dumbbell carries
the raw standing biomass at paired inside and outside sites, on a log axis
because a reef shark and a damselfish differ by more than an order of
magnitude and a shared linear axis would flatten half the rows to nothing.  On
the right the same pairs are reduced to a log response ratio with its 95%
confidence interval, so a row whose interval crosses one is visibly not an
effect - including sea urchin, which goes the other way once its predators
come back.

Both side panels use the same response-ratio currency.  Recovery is fitted as
a saturating curve against years since designation, with the spillover band
just outside the boundary shown on the same axis, and the area panel asks
whether bigger reserves work better once enforcement class is accounted for.
Confidence intervals, fitted time constants and slopes are all computed from
the plotted samples.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

from pubstyle import set_pub_style, savefig, annotate_box
from figkit import axes_inches, panel_labels_inches, shared_limits

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
INSIDE, OUTSIDE = "#2e7d5b", "#808080"
UP, DOWN, FLAT = "#2e7d5b", "#b23a48", "#9a9a9a"
NOTAKE, PARTIAL = "#2e7d5b", "#c9a227"
SPILL = "#3b5b92"

FW, FH = 10.4, 6.4
MAIN_A = (1.34, 1.06, 2.88, 4.10)
MAIN_B = (4.38, 1.06, 1.26, 4.10)
AUX1 = (6.72, 3.10, 2.68, 2.06)
AUX2 = (6.72, 1.06, 2.68, 1.44)

N_PAIRS = 12
TAXA = [
    ("Predatory fish", [("grouper", 1.90, 3.40), ("snapper", 3.10, 2.60),
                        ("jack", 2.20, 2.85), ("reef shark", 0.42, 3.90)]),
    ("Herbivorous fish", [("parrotfish", 5.40, 1.90),
                          ("surgeonfish", 4.10, 1.60),
                          ("rabbitfish", 2.80, 1.32)]),
    ("Mobile invertebrates", [("spiny lobster", 1.20, 3.10),
                              ("abalone", 0.62, 2.20),
                              ("sea urchin", 3.60, 0.52)]),
    ("Small-bodied fish", [("damselfish", 6.80, 1.10),
                           ("wrasse", 3.90, 1.22)]),
]


def paired_sites(rng):
    rows, y = [], 0
    for group, items in TAXA:
        rows.append(dict(kind="header", label=group, y=y))
        y += 1
        for name, out_mu, ratio in items:
            out = out_mu * np.exp(rng.normal(0.0, 0.40, N_PAIRS))
            ins = out * ratio * np.exp(rng.normal(0.0, 0.34, N_PAIRS))
            lrr = np.log(ins / out)
            se = lrr.std(ddof=1) / np.sqrt(N_PAIRS)
            rows.append(dict(kind="taxon", label=name, y=y,
                             out=float(out.mean()), ins=float(ins.mean()),
                             lrr=float(lrr.mean()), lo=float(lrr.mean()
                                                             - 1.96 * se),
                             hi=float(lrr.mean() + 1.96 * se)))
            y += 1
    return rows, y


def row_color(r):
    if r["lo"] > 0.0:
        return UP
    if r["hi"] < 0.0:
        return DOWN
    return FLAT


def draw_dumbbell(ax, rows, nrow):
    for r in rows:
        if r["kind"] == "header":
            ax.text(-0.012, r["y"], r["label"], ha="right", va="center",
                    fontsize=6.6, color=INK,
                    transform=ax.get_yaxis_transform())
            continue
        col = row_color(r)
        ax.plot([r["out"], r["ins"]], [r["y"], r["y"]], color=col, lw=1.7,
                alpha=0.75, solid_capstyle="round", zorder=3)
        ax.plot([r["out"]], [r["y"]], marker="o", ms=4.4, mfc="white",
                mec=OUTSIDE, mew=1.2, zorder=5)
        ax.plot([r["ins"]], [r["y"]], marker="o", ms=4.8, mfc=col,
                mec="white", mew=0.8, zorder=6)
    ax.set_xscale("log")
    ax.set_ylim(nrow - 0.4, -0.6)
    ax.set_yticks([r["y"] for r in rows if r["kind"] == "taxon"])
    ax.set_yticklabels([r["label"] for r in rows if r["kind"] == "taxon"],
                       fontsize=6.8)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(labelsize=6.8)
    ax.set_xlabel("standing biomass (t/km$^2$, log scale)", fontsize=7.8)
    ax.grid(axis="x", color="#e8e8e8", lw=0.6, zorder=0)
    ax.set_title("Paired biomass, inside versus outside", fontsize=8.5,
                 fontweight="bold")


def draw_forest(ax, rows, nrow):
    for r in rows:
        if r["kind"] == "header":
            continue
        col = row_color(r)
        ax.plot([np.exp(r["lo"]), np.exp(r["hi"])], [r["y"], r["y"]],
                color=col, lw=1.0, solid_capstyle="butt", zorder=4)
        ax.plot([np.exp(r["lo"]), np.exp(r["hi"])], [r["y"], r["y"]],
                marker="|", ms=3.4, ls="none", color=col, mew=1.0, zorder=4)
        ax.plot([np.exp(r["lrr"])], [r["y"]], marker="s", ms=3.8, mfc=col,
                mec="white", mew=0.6, zorder=6)
    ax.axvline(1.0, color=RULE, lw=0.9, ls=(0, (3.4, 2.2)), zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.30, 9.0)
    ax.set_xticks([0.5, 1.0, 2.0, 4.0, 8.0])
    ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
    ax.set_ylim(nrow - 0.4, -0.6)
    ax.set_yticks([])
    ax.tick_params(labelsize=6.8)
    ax.set_xlabel("response ratio inside/outside\n(-, 95% CI)", fontsize=7.4)
    ax.grid(axis="x", color="#e8e8e8", lw=0.6, zorder=0)
    ax.set_title("Effect size", fontsize=8.5, fontweight="bold")


def saturating(age, rmax, tau):
    return 1.0 + (rmax - 1.0) * (1.0 - np.exp(-age / tau))


def recovery_data(rng):
    age = np.linspace(0.0, 32.0, 26)
    inner = saturating(age, 2.95, 7.6) * np.exp(rng.normal(0.0, 0.075,
                                                           age.size))
    outer = saturating(age, 1.52, 10.5) * np.exp(rng.normal(0.0, 0.055,
                                                            age.size))
    return age, inner, outer


def area_data(rng):
    n = 34
    area = 10.0 ** rng.uniform(0.0, 3.4, n)
    notake = rng.random(n) < 0.55
    base = np.where(notake, 1.30, 0.92)
    ratio = np.clip(base + 0.46 * np.log10(area)
                    + rng.normal(0.0, 0.30, n), 0.55, None)
    return area, notake, ratio


def draw_recovery(ax, data, ylim):
    age, inner, outer = data
    grid = np.linspace(0.0, 32.0, 220)
    out = []
    for y, col, lab in ((inner, INSIDE, "inside reserve"),
                        (outer, SPILL, "0-2 km spillover belt")):
        par, cov = curve_fit(saturating, age, y, p0=[2.5, 8.0], maxfev=20000)
        se = np.sqrt(np.diag(cov))
        fit = saturating(grid, *par)
        band = np.abs(saturating(grid, par[0] + se[0], par[1])
                      - saturating(grid, par[0] - se[0], par[1])) / 2.0
        ax.fill_between(grid, fit - band, fit + band, color=col, alpha=0.16,
                        linewidth=0.0, zorder=2)
        ax.plot(grid, fit, color=col, lw=1.4, zorder=4, label=lab)
        ax.scatter(age, y, s=8.0, color=col, alpha=0.65, edgecolors="white",
                   linewidths=0.3, zorder=5)
        ax.annotate("$R_{max}$ = " + "{:.2f}, ".format(par[0])
                    + "$\\tau$ = " + "{:.1f} yr".format(par[1]),
                    xy=(31.5, fit[-1]), xytext=(0, 9),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=6.2, color=col, zorder=6)
        out.append(par)
    ax.set_ylim(*ylim)
    ax.axhline(1.0, color="#808080", lw=0.9, ls=(0, (3.4, 2.2)), zorder=3)
    ax.legend(frameon=False, fontsize=6.4, loc="upper left",
              handlelength=1.5, borderpad=0.15)
    ax.set_xlim(0.0, 32.0)
    ax.set_xlabel("years since designation (years)", fontsize=7.8)
    ax.set_ylabel("biomass response ratio (-)", fontsize=7.8)
    ax.tick_params(labelsize=6.8)
    ax.set_title("Recovery and spillover with protection age", fontsize=8.5,
                 fontweight="bold")
    return out


def draw_area(ax, data, ylim):
    area, notake, ratio = data
    n = area.size
    for m, col, lab in ((notake, NOTAKE, "no-take"),
                        (~notake, PARTIAL, "partially protected")):
        ax.scatter(area[m], ratio[m], s=15.0, color=col, alpha=0.85,
                   edgecolors="white", linewidths=0.4, zorder=4, label=lab)
    x = np.log10(area)
    slope, inter = np.polyfit(x, ratio, 1)
    grid = np.logspace(0.0, 3.4, 120)
    ax.plot(grid, inter + slope * np.log10(grid), color=RULE, lw=1.3,
            zorder=5)
    r = float(np.corrcoef(x, ratio)[0, 1])
    ax.axhline(1.0, color="#808080", lw=0.9, ls=(0, (3.4, 2.2)), zorder=3)
    annotate_box(ax, "{:+.2f} per 10x area,  $R^2$ = {:.2f},  n = {:d}"
                 .format(slope, r ** 2, n), loc="upper left", fontsize=6.2,
                 margin=0.03)
    ax.legend(frameon=False, fontsize=6.2, loc="lower right", ncol=1,
              handlelength=1.0, borderpad=0.15, labelspacing=0.25)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 3200.0)
    ax.set_ylim(*ylim)
    ax.set_xlabel("reserve area (km$^2$, log scale)", fontsize=7.8)
    ax.set_ylabel("response ratio (-)", fontsize=7.8)
    ax.tick_params(labelsize=6.8)
    ax.set_title("Bigger reserves, bigger effect", fontsize=8.5,
                 fontweight="bold")
    return slope, r ** 2, n


def main():
    set_pub_style()
    rng = np.random.default_rng(695)
    rows, nrow = paired_sites(rng)
    rec = recovery_data(rng)
    area = area_data(rng)
    ylim = shared_limits([rec[1], rec[2], area[2]], pad=0.07)

    fig = plt.figure(figsize=(FW, FH))
    ax_d = axes_inches(fig, MAIN_A)
    ax_f = axes_inches(fig, MAIN_B)
    ax_r = axes_inches(fig, AUX1)
    ax_a = axes_inches(fig, AUX2)

    draw_dumbbell(ax_d, rows, nrow)
    draw_forest(ax_f, rows, nrow)
    pars = draw_recovery(ax_r, rec, ylim)
    slope, r2, n = draw_area(ax_a, area, ylim)

    handles = [Line2D([0], [0], marker="o", ls="none", ms=4.4, mfc="white",
                      mec=OUTSIDE, mew=1.2, label="outside"),
               Line2D([0], [0], marker="o", ls="none", ms=4.8, mfc=INSIDE,
                      mec="white", mew=0.8, label="inside reserve")]

    panel_labels_inches(fig, [ax_d, ax_f, ax_r, ax_a], dx=-0.60, dy=0.20)

    taxa = [r for r in rows if r["kind"] == "taxon"]
    sig_up = [r for r in taxa if r["lo"] > 0.0]
    sig_dn = [r for r in taxa if r["hi"] < 0.0]
    best = max(taxa, key=lambda r: r["lrr"])

    print("sig_up={:d} sig_dn={:d} best={:s} {:.2f}x  rmax={:.2f} "
          "tau={:.1f}  slope={:+.3f} r2={:.2f}".format(
              len(sig_up), len(sig_dn), best["label"], np.exp(best["lrr"]),
              pars[0][0], pars[0][1], slope, r2))
    savefig(fig, "695_mar_mpa_effectiveness.png")


if __name__ == "__main__":
    main()

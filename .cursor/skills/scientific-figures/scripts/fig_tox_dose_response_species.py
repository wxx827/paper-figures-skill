"""Species sensitivity to a waterborne contaminant, three ways.

The main panel carries the raw evidence: a log-logistic concentration-response
curve for each of five test species, with the fitted EC50 dropped onto the
concentration axis as a dashed rule.  The two auxiliary panels turn that
evidence into the number a regulator actually uses - the species sensitivity
distribution and the 5th-percentile hazardous concentration HC5 - and rank
every species tested so the reader can see which organism sets the limit.

Numbers are synthetic; every value printed on the figure is computed from the
arrays that are plotted, so the annotations cannot drift away from the curves.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from pubstyle import set_pub_style, savefig, style_axes
from figkit import panel_labels_inches, shared_limits

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
GREY, ACCENT = "#b3b3b3", "#b23a48"

# name, EC50 (ug/L), Hill slope, shown in the main panel?, colour
SPECIES = [
    ("C. dubia",       8.0, 1.55, True,  "#0b3d2e"),
    ("D. magna",      14.0, 1.35, True,  "#2e7d5b"),
    ("H. azteca",     26.0, 1.20, False, GREY),
    ("R. subcapitata", 48.0, 1.75, True,  "#3b5b92"),
    ("L. minor",      72.0, 1.10, False, GREY),
    ("O. mykiss",    130.0, 1.45, True,  "#c9a227"),
    ("P. promelas",  240.0, 1.25, False, GREY),
    ("C. riparius",  380.0, 1.05, False, GREY),
    ("L. variegatus", 690.0, 0.95, True,  "#8e6bbf"),
]


def logistic(c, ec50, slope):
    """Log-logistic response in percent of maximum effect."""
    return 100.0 / (1.0 + (ec50 / c) ** slope)


def main():
    set_pub_style(base_font=9)
    rng = np.random.default_rng(624)

    names = [s[0] for s in SPECIES]
    ec50 = np.array([s[1] for s in SPECIES])
    slopes = np.array([s[2] for s in SPECIES])
    shown = np.array([s[3] for s in SPECIES])
    colors = [s[4] for s in SPECIES]

    # ---- species sensitivity distribution ---------------------------------
    order = np.argsort(ec50)
    log_ec = np.log10(ec50)
    mu, sigma = log_ec.mean(), log_ec.std(ddof=1)
    hc5 = 10.0 ** (mu + norm.ppf(0.05) * sigma)
    n = ec50.size
    emp = (np.arange(1, n + 1) - 0.5) / n           # Hazen plotting position

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0],
                          left=0.062, right=0.975, top=0.865, bottom=0.175,
                          wspace=0.30, hspace=0.52)
    ax = fig.add_subplot(gs[:, 0])
    ax_ssd = fig.add_subplot(gs[0, 1])
    ax_rank = fig.add_subplot(gs[1, 1])

    # ---- (a) concentration-response curves --------------------------------
    grid = np.logspace(np.log10(0.7), np.log10(4.0e3), 500)
    ax.axhline(50.0, color="#9a9a9a", lw=0.8, ls=(0, (5, 3)), zorder=1)
    ax.text(0.78, 51.5, "50% effect level", fontsize=6.4, color=MUTED,
            ha="left", va="bottom")

    for i in np.flatnonzero(shown):
        y = logistic(grid, ec50[i], slopes[i])
        ax.plot(grid, y, lw=1.6, color=colors[i], zorder=4,
                label=f"{names[i]}   $EC_{{50}}$ = {ec50[i]:,.0f}")
        ax.plot([ec50[i]] * 2, [-6.0, 50.0], lw=0.8, ls=(0, (2, 2)),
                color=colors[i], alpha=0.9, zorder=2)
        ax.plot(ec50[i], 50.0, marker="o", ms=4.0, mfc=colors[i], mec="white",
                mew=0.8, zorder=6)
        # a handful of bioassay points so the curves read as fitted, not drawn
        pts = ec50[i] * np.array([0.12, 0.34, 1.0, 2.9, 8.4])
        obs = logistic(pts, ec50[i], slopes[i]) + rng.normal(0.0, 3.2, pts.size)
        ax.plot(pts, np.clip(obs, 0.0, 100.0), ls="none", marker="o", ms=3.0,
                mfc="white", mec=colors[i], mew=0.9, zorder=5)

    ax.axvspan(0.7, hc5, color=ACCENT, alpha=0.07, lw=0, zorder=0)
    ax.axvline(hc5, color=ACCENT, lw=1.1, ls=(0, (4, 2)), zorder=3)
    ax.annotate(f"$HC_5$ = {hc5:.1f} $\\mu$g/L\nprotects 95% of species",
                xy=(hc5, 88.0), xytext=(hc5 * 1.5, 96.0), fontsize=6.8,
                color=ACCENT, ha="left", va="top", linespacing=1.45,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=ACCENT))

    ax.set_xscale("log")
    ax.set_xlim(0.7, 4.0e3)
    ax.set_ylim(-6.0, 118.0)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlabel("Exposure concentration ($\\mu$g/L)")
    ax.set_ylabel("Effect (% of maximum response)")
    ax.grid(color="#eeeeee", lw=0.6, zorder=0)
    style_axes(ax)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(-0.004, 1.005), ncol=2,
                    frameon=False, fontsize=6.6, handlelength=1.7,
                    labelspacing=0.35, columnspacing=1.4)
    leg._legend_box.align = "left"
    ax.set_title("Log-logistic concentration-response, 96 h acute assay",
                 fontsize=9.0, fontweight="bold", loc="left", pad=5)

    # ---- (b) species sensitivity distribution -----------------------------
    xs = np.logspace(np.log10(2.0), np.log10(3.0e3), 300)
    cdf = norm.cdf((np.log10(xs) - mu) / sigma)
    ax_ssd.plot(xs, 100.0 * cdf, lw=1.5, color="#3b5b92", zorder=4)
    band = norm.ppf(0.975) * sigma / np.sqrt(n)
    ax_ssd.fill_between(xs, 100.0 * norm.cdf((np.log10(xs) - mu - band) / sigma),
                        100.0 * norm.cdf((np.log10(xs) - mu + band) / sigma),
                        color="#3b5b92", alpha=0.14, lw=0, zorder=2)
    for k, i in enumerate(order):
        ax_ssd.plot(ec50[i], 100.0 * emp[k], marker="o", ms=3.8,
                    mfc=colors[i] if shown[i] else "white",
                    mec=colors[i] if shown[i] else "#8a8a8a", mew=0.9,
                    zorder=5)
    ax_ssd.axhline(5.0, color=ACCENT, lw=0.8, ls=(0, (3, 2)), zorder=3)
    ax_ssd.plot([hc5, hc5], [0.0, 5.0], lw=0.8, ls=(0, (3, 2)), color=ACCENT,
                zorder=3)
    ax_ssd.plot(hc5, 5.0, marker="D", ms=3.6, color=ACCENT, zorder=6)
    ax_ssd.text(hc5 * 3.2, 6.5, f"$HC_5$ = {hc5:.1f} $\\mu$g/L",
                fontsize=6.4, color=ACCENT, ha="left", va="bottom")
    ax_ssd.set_xscale("log")
    ax_ssd.set_xlim(2.0, 3.0e3)
    ax_ssd.set_ylim(-4.0, 104.0)
    ax_ssd.set_yticks([0, 25, 50, 75, 100])
    ax_ssd.set_xlabel("$EC_{50}$ ($\\mu$g/L)", fontsize=8.0)
    ax_ssd.set_ylabel("Species affected (%)", fontsize=8.0)
    ax_ssd.tick_params(labelsize=6.8)
    ax_ssd.grid(color="#eeeeee", lw=0.6, zorder=0)
    style_axes(ax_ssd)
    ax_ssd.set_title(f"Species sensitivity distribution (n = {n})",
                     fontsize=8.5, fontweight="bold", loc="left", pad=4)

    # ---- (c) EC50 ranking --------------------------------------------------
    base = 2.0
    for k, i in enumerate(order):
        ax_rank.barh(k, ec50[i] - base, left=base, height=0.62,
                     color=colors[i], edgecolor="white", linewidth=0.5,
                     alpha=1.0 if shown[i] else 0.75, zorder=3)
        ax_rank.text(ec50[i] * 1.18, k, f"{ec50[i]:,.0f}", va="center",
                     ha="left", fontsize=6.2,
                     color=INK if shown[i] else MUTED)
    ax_rank.axvline(hc5, color=ACCENT, lw=1.0, ls=(0, (4, 2)), zorder=5)
    ax_rank.set_xscale("log")
    ax_rank.set_xlim(base, 3.0e3)
    ax_rank.set_ylim(-0.75, n - 0.25)
    ax_rank.set_yticks(np.arange(n))
    ax_rank.set_yticklabels([names[i] for i in order], fontsize=6.2)
    ax_rank.set_xlabel("$EC_{50}$ ($\\mu$g/L, log scale)", fontsize=8.0)
    ax_rank.grid(axis="x", color="#eeeeee", lw=0.6, zorder=0)
    style_axes(ax_rank)
    ax_rank.tick_params(axis="x", labelsize=6.8)
    ax_rank.tick_params(axis="y", length=0)
    ax_rank.spines["left"].set_visible(False)
    ax_rank.set_title("Sensitivity ranking, most sensitive first",
                      fontsize=8.5, fontweight="bold", loc="left", pad=4)

    panel_labels_inches(fig, [ax, ax_ssd, ax_rank], dx=-0.42, dy=0.09)

    span = shared_limits([ec50], log=True, pad=0.0)

    print(f"HC5={hc5:.2f} ug/L  mu={mu:.3f}  sigma={sigma:.3f}  n={n}")
    savefig(fig, "624_tox_dose_response_species.png")


if __name__ == "__main__":
    main()

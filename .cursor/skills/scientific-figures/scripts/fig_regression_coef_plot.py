"""Coefficient plot for three competing regression specifications.

Standardised betas with 95% confidence intervals, dodged vertically by
model, a zero reference rule and significance stars at the right end of each
interval.  The side panel carries the specification table (fixed effects,
R-squared, N), which is what a regression table would otherwise consume half
a page to say.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm

from pubstyle import set_pub_style, savefig, style_axes

VARIABLES = [
    "Log GDP per capita",
    "Human capital index",
    "Trade openness",
    "FDI inflow (% of GDP)",
    "Credit depth (% of GDP)",
    "Urbanisation rate",
    "Institutional quality",
    "Energy intensity",
]

MODELS = [
    dict(name="(1) Pooled OLS", color="#4c78a8",
         fe="none", r2=0.412, n=1240,
         coef=[0.42, 0.28, 0.15, 0.08, 0.19, -0.06, 0.33, -0.24],
         se=[0.061, 0.052, 0.048, 0.041, 0.058, 0.047, 0.070, 0.059]),
    dict(name="(2) Two-way FE", color="#c98b3a",
         fe="unit + year", r2=0.586, n=1240,
         coef=[0.38, 0.25, 0.11, 0.06, 0.14, -0.09, 0.30, -0.27],
         se=[0.068, 0.058, 0.052, 0.049, 0.061, 0.046, 0.072, 0.061]),
    dict(name="(3) IV / 2SLS", color="#3f8f7a",
         fe="unit + year", r2=0.551, n=1188,
         coef=[0.31, 0.22, 0.09, 0.02, 0.16, -0.12, 0.36, -0.31],
         se=[0.102, 0.079, 0.071, 0.063, 0.088, 0.068, 0.110, 0.093]),
]
OFFSETS = [0.26, 0.0, -0.26]


def stars(coef, se):
    p = 2.0 * (1.0 - norm.cdf(abs(coef) / se))
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def panel_coefficients(ax):
    y0 = np.arange(len(VARIABLES))[::-1]

    for i in y0[::2]:
        ax.axhspan(i - 0.5, i + 0.5, color="#f6f6f4", zorder=0)
    ax.axvline(0.0, color="#555555", lw=0.9, ls=(0, (4, 3)), zorder=2)

    for m, dy in zip(MODELS, OFFSETS):
        coef = np.asarray(m["coef"])
        se = np.asarray(m["se"])
        lo, hi = coef - 1.96 * se, coef + 1.96 * se
        yy = y0 + dy
        ax.hlines(yy, lo, hi, color=m["color"], lw=1.1, zorder=3)
        for y, a, b in zip(yy, lo, hi):
            ax.plot([a, a], [y - 0.07, y + 0.07], color=m["color"], lw=1.0,
                    zorder=3)
            ax.plot([b, b], [y - 0.07, y + 0.07], color=m["color"], lw=1.0,
                    zorder=3)
        ax.plot(coef, yy, "o", color=m["color"], ms=4.0, mec="white",
                mew=0.6, zorder=4)
        for y, c, s, h in zip(yy, coef, se, hi):
            tag = stars(c, s)
            if tag:
                ax.text(h + 0.012, y, tag, fontsize=6.4, color=m["color"],
                        va="center", ha="left", fontweight="bold", zorder=4)

    ax.set_yticks(y0)
    ax.set_yticklabels(VARIABLES, fontsize=8)
    ax.set_ylim(-0.6, len(VARIABLES) - 0.4)
    ax.set_xlim(-0.56, 0.70)
    ax.set_xticks(np.arange(-0.4, 0.61, 0.2))
    ax.set_xlabel("Standardised coefficient with 95% CI "
                  "(s.d. of outcome per s.d. of regressor)")
    ax.set_title("Determinants of regional productivity growth, 1998-2022",
                 fontsize=9.5, loc="left", pad=8, weight="bold")
    ax.grid(axis="x", lw=0.5, color="#ececec")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    style_axes(ax)


def panel_specs(ax):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.985, "Specification", fontsize=8, fontweight="bold",
            va="top", ha="left")

    rows = [("Fixed effects", "fe"), ("R$^2$", "r2"), ("Observations", "n")]
    block = 0.255
    for k, m in enumerate(MODELS):
        top = 0.885 - k * block
        ax.plot([0.03], [top - 0.022], "o", color=m["color"], ms=4.6,
                mec="white", mew=0.6, clip_on=False,
                transform=ax.transAxes)
        ax.text(0.13, top, m["name"], fontsize=7.4, fontweight="bold",
                color=m["color"], va="top", ha="left")
        for r, (lab, key) in enumerate(rows):
            val = m[key]
            txt = f"{val:,}" if key == "n" else (
                f"{val:.3f}" if key == "r2" else str(val))
            yy = top - 0.062 - r * 0.055
            ax.text(0.13, yy, lab, fontsize=6.6, color="#6f6f6f", va="top",
                    ha="left")
            ax.text(1.0, yy, txt, fontsize=6.6, color="#1a1a1a", va="top",
                    ha="right")
        ax.plot([0.0, 1.0], [top - 0.225, top - 0.225], color="#e2e2e2",
                lw=0.6, transform=ax.transAxes, clip_on=False)

    # Two lines: on one line the key ran 0.21 in past the right edge of the
    # canvas, which made the bbox_inches="tight" export expand instead of
    # clip (DESIGN_SYSTEM 12.11).
    ax.text(0.0, 0.885 - 3 * block - 0.02,
            "*** p < 0.01    ** p < 0.05\n* p < 0.10",
            fontsize=6.4, color="#5a5a5a", va="top", ha="left",
            linespacing=1.5)


def main():
    set_pub_style()
    fig = plt.figure(figsize=(7.4, 4.4))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 0.31], wspace=0.06,
                  left=0.185, right=0.975, top=0.90, bottom=0.155)
    ax = fig.add_subplot(gs[0, 0])
    axs = fig.add_subplot(gs[0, 1])
    panel_coefficients(ax)
    panel_specs(axs)

    # Wrapped: a single-line footer wider than the canvas makes the
    # bbox_inches="tight" export expand instead of clip (DESIGN_SYSTEM 12.11).
    savefig(fig, "109_regression_coef_plot.png")


if __name__ == "__main__":
    main()

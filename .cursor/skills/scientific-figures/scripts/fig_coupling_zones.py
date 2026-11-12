"""Coupling coordination trends with class bands + class-share donuts.

Top row: the coupling coordination degree over time for sites inside and
outside protected areas, drawn over ordered class bands.  Bottom row: the
share of survey years falling in each class -- computed directly from the
series above rather than hard-coded, so the two rows always agree.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

# ordered rust -> teal ramp so the class bands read as a scale, not categories
ZONES = [
    (0.0, 0.2, "#f4d3c9", "Severely uncoupled"),
    (0.2, 0.4, "#f9e6c9", "Moderately uncoupled"),
    (0.4, 0.6, "#eceedb", "Basically coupled"),
    (0.6, 0.8, "#d2e5d6", "Moderately coupled"),
    (0.8, 1.0, "#a9d1c4", "Highly coupled"),
]
C_LINE = "#1f6b57"
C_ERR = "#b5654a"


def smooth(years, y, n=250):
    xg = np.linspace(years[0], years[-1], n)
    try:
        from scipy.interpolate import PchipInterpolator
        return xg, PchipInterpolator(years, y)(xg)   # shape-preserving
    except Exception:                                # pragma: no cover
        return xg, np.interp(xg, years, y)


def classify(y):
    idx = np.clip((np.asarray(y) * 5).astype(int), 0, 4)
    return np.bincount(idx, minlength=5) / len(y) * 100.0


def trend_panel(ax, years, y, err, title, band_labels):
    ax.set_xlim(years[0] - 1, years[-1] + 1)
    ax.set_ylim(0, 1)
    for a, b, col, lab in ZONES:
        ax.axhspan(a, b, color=col, alpha=0.85, zorder=0, linewidth=0)
        if band_labels:
            ax.text(1.015, (a + b) / 2, lab, transform=ax.get_yaxis_transform(),
                    fontsize=7, va="center", ha="left", color="#4a4a4a")
    for edge in (0.2, 0.4, 0.6, 0.8):
        ax.axhline(edge, color="white", lw=0.8, zorder=1)

    xg, yg = smooth(years, y)
    ax.plot(xg, yg, color=C_LINE, lw=1.7, zorder=4)
    ax.errorbar(years, y, yerr=err, fmt="s", ms=4.2, color=C_LINE,
                mec="white", mew=0.6, ecolor=C_ERR, elinewidth=1.0,
                capsize=2.2, capthick=1.0, zorder=5, linestyle="none")

    ax.set_xticks(years)
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel("Coupling coordination degree (0-1)", fontsize=8)
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=4)
    ax.tick_params(labelsize=7)
    style_axes(ax)


def donut_panel(ax, shares, title, n_years):
    keep = [(s, z) for s, z in zip(shares, ZONES) if s > 0.05]
    sizes = [s for s, _ in keep]
    colors = [z[2] for _, z in keep]
    wedges, _, autotexts = ax.pie(
        sizes, colors=colors, startangle=90, counterclock=False,
        autopct=lambda v: f"{v:.0f}%", pctdistance=0.79,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.2),
        textprops=dict(fontsize=7.5, color="#333333"))
    for t in autotexts:
        t.set_fontweight("bold")
    ax.text(0, 0, f"{n_years}\nsurvey years", ha="center", va="center",
            fontsize=7.5, color="#555555", linespacing=1.3)
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=4)
    ax.set(aspect="equal")


def main():
    set_pub_style()
    rng = np.random.default_rng(2)
    years = np.arange(1985, 2021, 5)
    t = np.linspace(0, 1, len(years))

    inside = np.clip(0.58 + 0.24 * np.sin(3.4 * t + 0.4)
                     + rng.normal(0, 0.025, len(years)), 0.12, 0.95)
    outside = np.clip(0.55 + 0.20 * np.sin(3.1 * t + 0.2) - 0.06 * t
                      + rng.normal(0, 0.025, len(years)), 0.12, 0.95)
    err_in = rng.uniform(0.02, 0.05, len(years))
    err_out = rng.uniform(0.02, 0.05, len(years))

    # the donut row is deliberately shorter than the trend row so the two
    # class-share summaries read as subordinate to the two main time series
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2),
                             gridspec_kw={"height_ratios": [1.5, 0.85]})
    trend_panel(axes[0, 0], years, inside, err_in,
                "Trend within protected areas", band_labels=False)
    trend_panel(axes[0, 1], years, outside, err_out,
                "Trend outside protected areas", band_labels=True)
    # short titles: the donut axes are squared down, so a long centred title
    # would overhang the axes and run into the panel label
    donut_panel(axes[1, 0], classify(inside), "Class shares, within",
                len(years))
    donut_panel(axes[1, 1], classify(outside), "Class shares, outside",
                len(years))

    for ax, lab in zip(axes.ravel(), "abcd"):
        add_panel_label(ax, f"({lab})", x=-0.10, y=1.01, size=10)

    handles = [Patch(facecolor=z[2], edgecolor="white", label=z[3])
               for z in ZONES]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, 0.005), columnspacing=1.4,
               handlelength=1.4)
    fig.tight_layout(rect=(0, 0.045, 0.995, 1), h_pad=1.6, w_pad=3.4)
    savefig(fig, "08_coupling_zones.png")


if __name__ == "__main__":
    main()

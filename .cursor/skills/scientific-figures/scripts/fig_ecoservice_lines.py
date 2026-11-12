"""Ecosystem-service change inside versus outside a nature reserve.

One panel per service.  Each panel is split into two blocks -- land inside the
reserve (white background) and outside it (shaded) -- with the same seven
five-year periods on both sides, county trajectories as marker lines and the
reserve-wide mean as grey bars behind them.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

SERIES = {
    "Reserve-wide": ("#333333", "^"),
    "Antu County": ("#3d6b8e", "s"),
    "Fusong County": ("#c9a227", "o"),
    "Changbai County": ("#8c2f2f", "D"),
}
END_YEARS = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
GAP = 2.6                      # blank columns between the two blocks
NP = len(END_YEARS)
OFF = NP - 1 + GAP
C_BAND = "#f6efec"

SERVICES = [
    ("Water yield", "Water yield change (%)", 1.0),
    ("Carbon storage", "Carbon storage change (%)", 1.4),
    ("Habitat quality", "Habitat quality change (%)", 1.8),
    ("Soil conservation", "Soil conservation change (%)", 1.2),
    ("Comprehensive ES index", "Index change (%)", 0.9),
]


def trajectory(rng, scale, drift):
    """Common regional signal plus a county-specific deviation."""
    return np.cumsum(rng.normal(drift, scale, NP))


def panel(ax, rng, title, ylabel, scale, tag):
    x_in = np.arange(NP, dtype=float)
    x_out = x_in + OFF

    ax.axvspan(OFF - 1.0, OFF + NP - 0.4, color=C_BAND, zorder=0,
               linewidth=0)

    common_in = trajectory(rng, 1.1 * scale, 0.15 * scale)
    common_out = trajectory(rng, 2.2 * scale, -0.35 * scale)
    counties_in, counties_out = [], []
    for name, (col, mk) in SERIES.items():
        if name == "Reserve-wide":
            continue
        counties_in.append(common_in + trajectory(rng, 0.9 * scale, 0))
        counties_out.append(common_out + trajectory(rng, 1.9 * scale, 0))
    counties_in = np.array(counties_in)
    counties_out = np.array(counties_out)

    # grey / hatched bars = min-max envelope across the counties
    for xs, arr, kw in [
            (x_in, counties_in, dict(color="#e2e2e2")),
            (x_out, counties_out, dict(color="none", edgecolor="#c4c4c4",
                                       hatch="////", linewidth=0.45))]:
        lo, hi = arr.min(axis=0), arr.max(axis=0)
        ax.bar(xs, hi - lo, bottom=lo, width=0.62, zorder=1, **kw)

    order = [n for n in SERIES if n != "Reserve-wide"]
    for name, y_in, y_out in zip(order, counties_in, counties_out):
        col, mk = SERIES[name]
        ax.plot(x_in, y_in, marker=mk, color=col, ms=3.6, lw=1.0, zorder=5,
                mec="white", mew=0.4)
        ax.plot(x_out, y_out, marker=mk, color=col, ms=3.6, lw=1.0, zorder=5,
                mec="white", mew=0.4)
    col, mk = SERIES["Reserve-wide"]
    ax.plot(x_in, counties_in.mean(axis=0), marker=mk, color=col, ms=3.8,
            lw=1.6, zorder=6, mec="white", mew=0.4)
    ax.plot(x_out, counties_out.mean(axis=0), marker=mk, color=col, ms=3.8,
            lw=1.6, zorder=6, mec="white", mew=0.4)

    ax.axhline(0, color="#9a9a9a", lw=0.7, zorder=2)
    ax.set_xlim(-1.0, OFF + NP - 0.4)
    ax.set_xticks(list(x_in) + list(x_out))
    ax.set_xticklabels([str(y) for y in END_YEARS] * 2, fontsize=6.2,
                       rotation=45, ha="right")
    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.set_xlabel("Period ending in", fontsize=7.5, labelpad=1)
    ax.tick_params(labelsize=7, length=2.5)
    style_axes(ax)

    span = ax.get_ylim()
    top = span[1] + 0.02 * (span[1] - span[0])
    ax.text((NP - 1) / 2, top, "Inside the reserve", ha="center",
            va="bottom", fontsize=7, color="#555555")
    ax.text(OFF + (NP - 1) / 2, top, "Outside the reserve", ha="center",
            va="bottom", fontsize=7, color="#8a5b4a")
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=13)
    add_panel_label(ax, f"({tag})", x=-0.11, y=1.06, size=10)


def main():
    set_pub_style()
    rng = np.random.default_rng(6)
    fig, axes = plt.subplots(3, 2, figsize=(9.6, 9.0))
    for i, (ax, (title, ylabel, scale)) in enumerate(zip(axes.ravel(), SERVICES)):
        panel(ax, rng, title, ylabel, scale, chr(97 + i))

    legend_ax = axes.ravel()[5]
    legend_ax.axis("off")
    handles = [Line2D([], [], color=c, marker=m, ms=4.5, mec="white",
                      mew=0.4, lw=1.6 if n == "Reserve-wide" else 1.0,
                      label=n + (" (county mean)" if n == "Reserve-wide"
                                 else "")) for n, (c, m) in SERIES.items()]
    handles += [
        Patch(facecolor="#e2e2e2", label="County range, inside"),
        Patch(facecolor="white", edgecolor="#c4c4c4", hatch="////",
              label="County range, outside"),
        Patch(facecolor=C_BAND, label="Outside-reserve block"),
    ]
    legend_ax.legend(handles=handles, loc="center", fontsize=8, frameon=False,
                     labelspacing=0.8, handlelength=1.8,
                     title="Series and reference bars",
                     title_fontsize=8.5)
    fig.tight_layout(h_pad=2.6, w_pad=2.6)
    savefig(fig, "10_ecoservice_lines.png")


if __name__ == "__main__":
    main()

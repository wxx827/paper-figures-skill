"""Warming-stripes figure with quantitative companions.

Each of the two rows shows one region: a classic Ed Hawkins stripe band (one
bar per year, blue-to-red on a zero-centred scale), the underlying annual
anomaly series with a smoothed curve and two least-squares trends beneath it,
and a colour-matched distribution histogram on the right. Because global and
Arctic anomalies differ by a factor of two in amplitude, each row carries its
own compact colour scale next to the stripe band.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.ndimage import gaussian_filter1d
from pubstyle import set_pub_style, savefig, style_axes

# ColorBrewer RdBu, the palette used by the original warming stripes
STRIPES = LinearSegmentedColormap.from_list("stripes", [
    "#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6", "#9ecae1",
    "#c6dbef", "#deebf7", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a",
    "#ef3b2c", "#cb181d", "#a50f15", "#67000d"])
Y0, Y1 = 1850, 2024
BASE = (1961, 1991)
VOLCANO = [(1884, 0.16, "Krakatoa"), (1903, 0.11, None),
           (1964, 0.14, "Agung"), (1992, 0.15, "Pinatubo")]


def anomaly_series(rng, years, gain, noise, ar=0.30):
    """Smooth multi-stage warming plus volcanic dips and AR(1) variability."""
    t = years.astype(float)
    g = (0.18 * np.tanh((t - 1932) / 14.0)
         + 0.62 * np.tanh((t - 1997) / 18.0)
         + 0.22 * (t - Y0) / (Y1 - Y0))
    for yv, amp, _ in VOLCANO:
        g -= amp * np.exp(-((t - yv - 1.0) / 1.6) ** 2)
    e = rng.normal(size=t.size)
    r = np.zeros(t.size)
    for i in range(1, t.size):
        r[i] = ar * r[i - 1] + e[i]
    x = gain * g + noise * r / r.std()
    m = (years >= BASE[0]) & (years < BASE[1])
    return x - x[m].mean()


def draw_row(fig, cell, years, y, region, tags, show_x):
    vmax = np.ceil(np.abs(y).max() * 10) / 10
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    sub = cell.subgridspec(2, 2, width_ratios=[4.7, 1.05],
                           height_ratios=[0.80, 2.40],
                           wspace=0.055, hspace=0.09)
    ax_s = fig.add_subplot(sub[0, 0])
    ax_l = fig.add_subplot(sub[1, 0], sharex=ax_s)
    ax_h = fig.add_subplot(sub[1, 1], sharey=ax_l)

    # ---- stripe band ------------------------------------------------------
    ax_s.imshow(y[None, :], cmap=STRIPES, norm=norm, aspect="auto",
                interpolation="nearest",
                extent=[years[0] - 0.5, years[-1] + 0.5, 0, 1])
    ax_s.set_yticks([])
    ax_s.set_xticks([])
    ax_s.tick_params(labelbottom=False)
    for s in ax_s.spines.values():
        s.set_visible(False)
    ax_s.text(0.0, 1.24, region, transform=ax_s.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")
    ax_s.text(-0.062, 1.32, tags[0], transform=ax_s.transAxes, fontsize=11,
              fontweight="bold", va="bottom", ha="left")

    # compact per-row colour scale in the cell beside the stripe band
    pos = sub[0, 1].get_position(fig)
    cax = fig.add_axes([pos.x0 + 0.012, pos.y0 + 0.10 * pos.height,
                        pos.width * 0.80, 0.28 * pos.height])
    cb = fig.colorbar(ScalarMappable(cmap=STRIPES, norm=norm), cax=cax,
                      orientation="horizontal", ticks=[-vmax, 0, vmax])
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("top")
    cb.ax.set_xticklabels([f"{-vmax:g}", "0", f"+{vmax:g}"])
    cb.ax.tick_params(labelsize=6.4, length=2, pad=1.5)
    cb.set_label("anomaly ($\\degree$C)", fontsize=6.6, labelpad=2.5)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#555555")

    # ---- annual series + trends ------------------------------------------
    sm = gaussian_filter1d(y, 3.0)
    ax_l.axhline(0, color="#777777", lw=0.8, zorder=2)
    ax_l.fill_between(years, 0, sm, where=sm >= 0, color="#cb181d",
                      alpha=0.15, lw=0, interpolate=True)
    ax_l.fill_between(years, 0, sm, where=sm < 0, color="#2171b5",
                      alpha=0.15, lw=0, interpolate=True)
    ax_l.plot(years, y, color="#6a6a6a", lw=0.7, alpha=0.85, zorder=3)
    ax_l.scatter(years, y, s=6, c=y, cmap=STRIPES, norm=norm,
                 edgecolors="#3a3a3a", linewidths=0.25, zorder=4)
    ax_l.plot(years, sm, color="#1a1a1a", lw=1.8, zorder=5)

    fit = np.polyfit(years, y, 1)
    ax_l.plot(years, np.polyval(fit, years), color="#67000d", lw=1.2,
              ls=(0, (5, 2)), zorder=5.5)
    recent = years >= 1975
    fit2 = np.polyfit(years[recent], y[recent], 1)
    ax_l.plot(years[recent], np.polyval(fit2, years[recent]),
              color="#08306b", lw=1.4, ls=(0, (2, 1.6)), zorder=5.5)

    pad = 0.16 * np.ptp(y)
    ax_l.set_ylim(y.min() - pad, y.max() + 0.55 * pad)
    ax_l.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax_l.set_ylabel("Anomaly ($\\degree$C)", fontsize=8)
    style_axes(ax_l)

    # eruptions marked on the axis line, never over the data
    xtr = ax_l.get_xaxis_transform()
    for yv, _, name in VOLCANO:
        if name is None:
            continue
        ax_l.plot([yv + 1], [0.012], marker="^", ms=4.2, color="#4a5a6a",
                  transform=xtr, clip_on=False, zorder=6)
        ax_l.text(yv + 1, 0.052, name, transform=xtr, ha="center",
                  va="bottom", fontsize=6.2, color="#4a5a6a", zorder=6)

    if show_x:
        ax_l.set_xlabel("Year")
        ax_l.set_xticks(np.arange(1850, 2025, 25))
    else:
        ax_l.tick_params(labelbottom=False)

    ax_l.text(0.012, 0.96,
              f"1850-2024 trend {fit[0] * 100:+.2f} $\\degree$C century$^{{-1}}$\n"
              f"since 1975    {fit2[0] * 10:+.2f} $\\degree$C decade$^{{-1}}$",
              transform=ax_l.transAxes, ha="left", va="top", fontsize=7,
              color="#333333", linespacing=1.45, zorder=7,
              bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#dddddd",
                        lw=0.5, alpha=0.88))

    # ---- distribution -----------------------------------------------------
    edges = np.linspace(y.min() - 0.02, y.max() + 0.02, 19)
    cnt, _ = np.histogram(y, bins=edges)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    ax_h.barh(ctr, cnt, height=np.diff(edges) * 0.90,
              color=STRIPES(norm(ctr)), edgecolor="#8a8a8a", linewidth=0.35)
    ax_h.axhline(0, color="#777777", lw=0.8)
    med = float(np.median(y))
    ax_h.axhline(med, color="#1a1a1a", lw=1.1, ls=(0, (3, 1.6)))
    ax_h.set_xlim(0, cnt.max() * 1.34)
    ax_h.tick_params(labelleft=False, length=2.5)
    ax_h.locator_params(axis="x", nbins=3)
    ax_h.set_xlabel("Years", fontsize=7.5, labelpad=1)
    style_axes(ax_h)
    ax_h.text(0.045, 0.995, tags[1], transform=ax_h.transAxes, fontsize=11,
              fontweight="bold", va="top", ha="left", zorder=6,
              bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none",
                        alpha=0.75))
    ax_h.text(0.97, 0.975, f"median {med:+.2f} $\\degree$C\n"
              f"range {np.ptp(y):.2f} $\\degree$C",
              transform=ax_h.transAxes, ha="right", va="top", fontsize=6.6,
              color="#444444", linespacing=1.4)


def main():
    set_pub_style()
    rng = np.random.default_rng(82)
    years = np.arange(Y0, Y1 + 1)

    rows = [
        (anomaly_series(rng, years, 1.00, 0.115),
         "Global mean surface temperature", ("(a)", "(b)")),
        (anomaly_series(rng, years, 2.35, 0.420),
         "Arctic land and ocean, 64-90$\\degree$N", ("(c)", "(d)")),
    ]

    fig = plt.figure(figsize=(10.6, 7.2))
    gs = fig.add_gridspec(2, 1, left=0.072, right=0.985, top=0.875,
                          bottom=0.075, hspace=0.38)
    for i, (y, region, tags) in enumerate(rows):
        draw_row(fig, gs[i, 0], years, y, region, tags, show_x=(i == 1))

    handles = [
        plt.Line2D([], [], color="#1a1a1a", lw=1.8, label="9-yr smoothing"),
        plt.Line2D([], [], color="#67000d", lw=1.2, ls=(0, (5, 2)),
                   label="1850-2024 trend"),
        plt.Line2D([], [], color="#08306b", lw=1.4, ls=(0, (2, 1.6)),
                   label="post-1975 trend"),
        plt.Line2D([], [], color="#4a5a6a", lw=0, marker="^", ms=4.2,
                   label="major volcanic eruption"),
    ]
    savefig(fig, "82_climate_stripes.png")


if __name__ == "__main__":
    main()

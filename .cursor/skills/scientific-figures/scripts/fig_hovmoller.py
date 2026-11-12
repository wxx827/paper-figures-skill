"""Hovmoller diagram (time-latitude section) with marginal summary panels.

Two standardized anomaly fields are shown side by side. Each composite holds a
main time-latitude raster, a latitude-averaged time series on top and a
time-averaged meridional profile on the right. One diverging colorbar centred
on zero is shared by both variables.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from pubstyle import set_pub_style, savefig, style_axes

CMAP = "RdBu_r"
POS, NEG = "#b2182b", "#2166ac"
T0, T1 = 1980.0, 2021.0
NOISE_FRAC = 0.42


def red_noise(rng, n, smooth):
    """Smoothed white noise -> unit-variance low-frequency index."""
    x = gaussian_filter1d(rng.normal(size=n), smooth, mode="wrap")
    return (x - x.mean()) / x.std()


def make_field(rng, t, lat, kind):
    """Large-scale climate signal plus spatially coherent residual noise."""
    LAT = lat[:, None]
    T = (t - T0)[None, :]
    enso = red_noise(rng, t.size, 9.0)[None, :]

    if kind == "sst":
        sig = (1.60 * enso * np.exp(-(LAT / 14.0) ** 2)
               - 0.50 * enso * np.exp(-((np.abs(LAT) - 34.0) / 13.0) ** 2)
               + 0.70 * np.sin(2 * np.pi * (T / 9.0 - LAT / 95.0))
               * np.exp(-(LAT / 45.0) ** 2)
               + 0.030 * T * (0.40 + 1.00 * np.abs(LAT) / 60.0))
    else:
        sig = (-1.35 * enso * np.exp(-((LAT - 6.0) / 10.0) ** 2)
               + 1.00 * enso * np.exp(-((np.abs(LAT) - 38.0) / 11.0) ** 2)
               + 0.60 * np.cos(2 * np.pi * (T / 6.5 + LAT / 85.0))
               * np.exp(-(LAT / 50.0) ** 2)
               + 0.022 * T * np.tanh((np.abs(LAT) - 28.0) / 12.0)
               + 0.013 * T)

    noise = gaussian_filter(rng.normal(size=sig.shape), (3.0, 5.0))
    field = sig / sig.std() + NOISE_FRAC * noise / noise.std()
    return (field - field.mean()) / field.std(), enso.ravel()


def marginal_fill(ax, x, y, horizontal=False):
    """Zero-centred red/blue fill for a marginal curve."""
    if horizontal:
        ax.fill_betweenx(x, 0, y, where=y >= 0, color=POS, alpha=0.55, lw=0)
        ax.fill_betweenx(x, 0, y, where=y < 0, color=NEG, alpha=0.55, lw=0)
        ax.plot(y, x, color="#333333", lw=0.9)
        ax.axvline(0, color="#888888", lw=0.6, ls=(0, (3, 2)))
    else:
        ax.fill_between(x, 0, y, where=y >= 0, color=POS, alpha=0.55, lw=0)
        ax.fill_between(x, 0, y, where=y < 0, color=NEG, alpha=0.55, lw=0)
        ax.plot(x, y, color="#333333", lw=0.9)
        ax.axhline(0, color="#888888", lw=0.6, ls=(0, (3, 2)))


def nice_sym_ticks(v):
    """Three symmetric ticks spanning +-v with a readable step."""
    step = float(f"{v * 0.62:.1g}")
    return [-step, 0.0, step], max(v, step * 1.28)


def main():
    set_pub_style()
    rng = np.random.default_rng(79)

    nt = int((T1 - T0) * 12)
    t = T0 + np.arange(nt) / 12.0
    lat = np.linspace(-60, 60, 121)
    w = np.cos(np.deg2rad(lat))[:, None]

    panels = [
        ("sst", "Sea-surface temperature", "(a)"),
        ("pr", "Precipitation", "(b)"),
    ]

    fig = plt.figure(figsize=(11.2, 6.3))
    gs_out = fig.add_gridspec(1, 2, left=0.068, right=0.985, top=0.875,
                              bottom=0.185, wspace=0.20)
    norm = TwoSlopeNorm(vmin=-2.6, vcenter=0.0, vmax=2.6)
    im = None

    for k, (kind, title, tag) in enumerate(panels):
        sub = gs_out[0, k].subgridspec(2, 2, width_ratios=[4.4, 1.0],
                                       height_ratios=[1.0, 3.1],
                                       wspace=0.05, hspace=0.07)
        ax_main = fig.add_subplot(sub[1, 0])
        ax_top = fig.add_subplot(sub[0, 0], sharex=ax_main)
        ax_rgt = fig.add_subplot(sub[1, 1], sharey=ax_main)

        field, enso = make_field(rng, t, lat, kind)

        im = ax_main.imshow(field, origin="lower", aspect="auto", cmap=CMAP,
                            norm=norm, interpolation="bilinear",
                            extent=[t[0], t[-1], lat[0], lat[-1]])
        ax_main.axhline(0, color="#3a3a3a", lw=0.7, ls=(0, (4, 3)), alpha=0.85)
        ax_main.set_yticks([-60, -30, 0, 30, 60])
        ax_main.set_yticklabels(["60$\\degree$S", "30$\\degree$S", "EQ",
                                 "30$\\degree$N", "60$\\degree$N"])
        ax_main.set_xticks(np.arange(1980, 2021, 10))
        ax_main.set_xlabel("Year")
        if k == 0:
            ax_main.set_ylabel("Latitude")
        for s in ax_main.spines.values():
            s.set_visible(True)
            s.set_color("#333333")
            s.set_linewidth(0.8)
        ax_main.tick_params(length=3)

        # warm-event markers along the top edge of the raster
        strong = t[gaussian_filter1d(enso, 2) > 1.15]
        if strong.size:
            for seg in np.split(strong, np.where(np.diff(strong) > 0.5)[0] + 1):
                ax_main.plot([seg.mean()], [60], marker="v", ms=4.2,
                             color="#1a1a1a", clip_on=False, zorder=6)

        # --- top marginal: cosine-weighted latitude mean -------------------
        tmean = gaussian_filter1d((field * w).sum(0) / w.sum(), 2.0)
        marginal_fill(ax_top, t, tmean)
        ticks, lim = nice_sym_ticks(np.abs(tmean).max())
        ax_top.set_ylim(-lim, lim)
        ax_top.set_yticks(ticks)
        ax_top.tick_params(labelbottom=False, length=2.5)
        style_axes(ax_top)
        ax_top.set_ylabel("Zonal\nmean ($\\sigma$)", fontsize=7,
                          linespacing=1.15)
        slope = np.polyfit(t, tmean, 1)[0] * 10
        ax_top.text(0.012, 0.94, f"trend {slope:+.2f} $\\sigma$ decade$^{{-1}}$",
                    transform=ax_top.transAxes, ha="left", va="top",
                    fontsize=6.8, color="#333333")
        ax_top.set_title(title, fontsize=9.5, fontweight="bold", pad=6)
        ax_top.text(-0.085, 1.42, tag, transform=ax_top.transAxes,
                    fontsize=11, fontweight="bold", va="top")

        # --- right marginal: time-mean meridional profile ------------------
        prof = gaussian_filter1d(field.mean(1), 3.0)
        marginal_fill(ax_rgt, lat, prof, horizontal=True)
        ticks, lim = nice_sym_ticks(np.abs(prof).max())
        ax_rgt.set_xlim(-lim, lim)
        ax_rgt.set_xticks(ticks)
        ax_rgt.tick_params(labelleft=False, length=2.5)
        ax_rgt.set_xlabel("Time mean\n($\\sigma$)", fontsize=7,
                          linespacing=1.15)
        style_axes(ax_rgt)

    cax = fig.add_axes([0.345, 0.062, 0.29, 0.026])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("Standardized anomaly ($\\sigma$)", fontsize=8, labelpad=2)
    cb.ax.tick_params(labelsize=7, length=2.5)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    savefig(fig, "79_hovmoller.png")


if __name__ == "__main__":
    main()

"""2x3 spatial raster grid with a single shared diverging colorbar.

A multi-year gridded map figure.  The synthetic fields share one persistent
spatial pattern that intensifies over time, so the sequence reads as a trend
rather than six independent noise fields.  The north arrow and scale bar are
drawn once (all panels share the same extent) and every panel reports the
share of land above the safe boundary.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.ndimage import gaussian_filter
from pubstyle import set_pub_style, savefig, add_panel_label

CMAP = LinearSegmentedColormap.from_list(
    "bwr_soft", ["#4a74b4", "#8bb0d6", "#dbe7f0", "#f5f5ee", "#f4c98a",
                 "#e07b52", "#b5361f"])
YEARS = [1995, 2000, 2005, 2010, 2015, 2020]
NY, NX = 170, 190


def country_mask(rng):
    """Irregular landmass: an ellipse with a wobbly, noisy coastline."""
    yy, xx = np.mgrid[0:NY, 0:NX]
    cx, cy = NX * 0.5, NY * 0.5
    dx = (xx - cx) / (NX * 0.44)
    dy = (yy - cy) / (NY * 0.42)
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    wobble = (1.0 + 0.16 * np.sin(3 * th + 0.7) + 0.10 * np.sin(5 * th - 1.2)
              + 0.07 * np.sin(8 * th + 2.1))
    coast = gaussian_filter(rng.normal(0, 1, (NY, NX)), 7)
    coast = coast / (np.abs(coast).max() + 1e-9)
    return gaussian_filter(
        ((r < wobble * (0.80 + 0.16 * coast))).astype(float), 1.2) > 0.45


def fields(rng):
    """One persistent pattern plus a strengthening hotspot per year."""
    base = gaussian_filter(rng.normal(0, 1, (NY, NX)), 6)
    base = (base - base.mean()) / base.std()
    yy = np.linspace(-1, 1, NY)[:, None]
    xx = np.linspace(-1, 1, NX)[None, :]
    hot = np.exp(-(((xx - 0.18) ** 2 + (yy + 0.12) ** 2) / 0.06))
    hot2 = np.exp(-(((xx + 0.45) ** 2 + (yy - 0.35) ** 2) / 0.09))
    out = []
    for k in range(len(YEARS)):
        wobble = gaussian_filter(rng.normal(0, 1, (NY, NX)), 8) * 6
        out.append(base * 26 + hot * (30 + 34 * k) + hot2 * (10 + 16 * k)
                   + wobble - 34)
    return out


def scale_bar(ax):
    ax.plot([0.06, 0.34], [0.055, 0.055], transform=ax.transAxes,
            color="#333333", lw=1.6, solid_capstyle="butt", zorder=6)
    for xf in (0.06, 0.20, 0.34):
        ax.plot([xf, xf], [0.055, 0.082], transform=ax.transAxes,
                color="#333333", lw=1.0, zorder=6)
    for xf, lab in [(0.06, "0"), (0.20, "1000"), (0.34, "2000 km")]:
        ax.text(xf, 0.092, lab, transform=ax.transAxes, fontsize=6.0,
                ha="center", va="bottom", color="#333333")


def main():
    set_pub_style()
    rng = np.random.default_rng(201)
    mask = country_mask(rng)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    sl = (slice(max(rows[0] - 2, 0), min(rows[-1] + 3, NY)),
          slice(max(cols[0] - 2, 0), min(cols[-1] + 3, NX)))
    norm = TwoSlopeNorm(vmin=-60, vcenter=0, vmax=200)

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.6))
    im = None
    for k, (ax, data) in enumerate(zip(axes.ravel(), fields(rng))):
        land = data[mask]
        share = float((land > 0).mean() * 100)
        img = np.where(mask, data, np.nan)[sl]
        im = ax.imshow(img, cmap=CMAP, norm=norm, origin="lower",
                       interpolation="bilinear")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#cccccc")
            s.set_linewidth(0.6)

        ax.text(0.045, 0.945, str(YEARS[k]), transform=ax.transAxes,
                fontsize=8.5, fontweight="bold", color="white", va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="#2c3e50", ec="none"))
        ax.text(0.975, 0.045, f"{share:.0f}% of land above boundary",
                transform=ax.transAxes, fontsize=6.5, ha="right",
                va="bottom", color="#333333",
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#d5d5d5",
                          lw=0.5, alpha=0.9))
        add_panel_label(ax, f"({chr(97 + k)})", x=0.0, y=1.01, size=10)

        if k == 0:                       # all panels share one extent
            ax.annotate("N", xy=(0.945, 0.92), xytext=(0.945, 0.78),
                        xycoords="axes fraction", ha="center", fontsize=7,
                        arrowprops=dict(arrowstyle="-|>", color="#333333",
                                        lw=1.1))
            scale_bar(ax)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.15,
                        wspace=0.06, hspace=0.14)
    cax = fig.add_axes([0.30, 0.075, 0.40, 0.022])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both",
                      ticks=[-50, 0, 50, 100, 150, 200])
    cb.set_label("Exceedance of the safe nitrogen boundary "
                 "(kg N ha$^{-1}$ yr$^{-1}$)", fontsize=8, labelpad=2)
    cb.ax.tick_params(labelsize=7, length=2, width=0.6)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#666666")
    savefig(fig, "27_spatial_grid.png")


if __name__ == "__main__":
    main()

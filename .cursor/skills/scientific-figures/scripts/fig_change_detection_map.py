"""Before / after / difference change-detection triptych.

Two epochs of a gridded land-surface variable are masked to a synthetic
coastline and share one sequential colourbar; the difference panel uses a
symmetric diverging ramp with significant-change contours. Side panels give
the distribution of change and the area shares of gain, stability and loss.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from scipy.ndimage import gaussian_filter
from pubstyle import set_pub_style, savefig

SEQ = LinearSegmentedColormap.from_list(
    "veg", ["#f2eee2", "#dfe4c8", "#bcd2a6", "#8ab585", "#54936a",
            "#2b6b52", "#14432f"])
DIV = LinearSegmentedColormap.from_list(
    "div_change", ["#7b3a2f", "#b4694a", "#e0b088", "#f4f1ea", "#a9c8d8",
                   "#5a8ca8", "#25526e"])
GAIN, STABLE, LOSS = "#25526e", "#c9c4b4", "#7b3a2f"
NY, NX = 260, 300
THR = 6.0


def land_radius(theta):
    return (1.0 + 0.15 * np.cos(2 * theta + 1.9)
            + 0.10 * np.cos(3 * theta - 1.4)
            + 0.05 * np.cos(5 * theta + 0.6))


def grids():
    y = np.linspace(-1.25, 1.25, NY)
    x = np.linspace(-1.35, 1.35, NX)
    xx, yy = np.meshgrid(x, y)
    mask = np.hypot(xx, yy) < land_radius(np.arctan2(yy, xx))
    return xx, yy, mask


def smooth(rng, sigma=14):
    f = gaussian_filter(rng.normal(0, 1, (NY, NX)), sigma)
    return (f - f.mean()) / f.std()


def main():
    set_pub_style()
    rng = np.random.default_rng(88)
    xx, yy, mask = grids()
    t = np.linspace(0, 2 * np.pi, 400)
    coast = np.c_[np.cos(t), np.sin(t)] * land_radius(t)[:, None]

    base = 46 + 17 * smooth(rng, 16) + 12 * np.exp(
        -(((xx + 0.45) ** 2 + (yy - 0.35) ** 2) / 0.35))
    change = (-4.0 + 11.0 * smooth(rng, 20)
              + 24 * np.exp(-(((xx - 0.55) ** 2 + (yy + 0.30) ** 2) / 0.12))
              - 27 * np.exp(-(((xx + 0.30) ** 2 + (yy - 0.55) ** 2) / 0.10))
              - 18 * np.exp(-(((xx - 0.25) ** 2 + (yy + 0.75) ** 2) / 0.07))
              + 3.0 * smooth(rng, 5))
    t1 = np.clip(base, 2, 96)
    t2 = np.clip(base + change, 2, 96)
    diff = t2 - t1

    def m(a):
        return np.where(mask, a, np.nan)

    vmin, vmax = 5, 92
    dmax = 26

    fig = plt.figure(figsize=(11.2, 4.9))
    maps = []
    for k, (data, lab, title) in enumerate([
            (m(t1), "a", "Baseline (2005)"),
            (m(t2), "b", "Recent (2024)"),
            (m(diff), "c", "Change (2024 $-$ 2005)")]):
        ax = fig.add_axes([0.008 + 0.255 * k, 0.205, 0.245, 0.645])
        maps.append(ax)
        if k < 2:
            im = ax.imshow(data, cmap=SEQ, norm=Normalize(vmin, vmax),
                           origin="lower", extent=(-1.35, 1.35, -1.25, 1.25),
                           interpolation="bilinear")
            im_seq = im
        else:
            im_div = ax.imshow(data, cmap=DIV,
                               norm=TwoSlopeNorm(0, -dmax, dmax),
                               origin="lower",
                               extent=(-1.35, 1.35, -1.25, 1.25),
                               interpolation="bilinear")
            sig = np.where(mask, np.abs(diff), 0)
            ax.contour(np.linspace(-1.35, 1.35, NX),
                       np.linspace(-1.25, 1.25, NY), sig, levels=[THR],
                       colors="#333333", linewidths=0.5, alpha=0.75)
        ax.plot(coast[:, 0], coast[:, 1], color="#3a3a3a", lw=0.9)
        ax.set_aspect("equal")
        ax.set_xlim(coast[:, 0].min() - 0.06, coast[:, 0].max() + 0.06)
        ax.set_ylim(coast[:, 1].min() - 0.06, coast[:, 1].max() + 0.06)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.0, 1.005, "(%s)" % lab, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom")
        ax.text(0.105, 1.009, title, transform=ax.transAxes, fontsize=9.5,
                fontweight="bold", va="bottom", color="#2b4f42")
        stat = (np.nanmean(np.where(mask, t1, np.nan)) if k == 0 else
                np.nanmean(np.where(mask, t2, np.nan)) if k == 1 else
                np.nanmean(np.where(mask, diff, np.nan)))
        ax.text(0.985, 0.965, ("mean %+.1f" if k == 2 else "mean %.1f") % stat,
                transform=ax.transAxes, fontsize=7, ha="right", va="top",
                color="#333",
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#d5d0c4",
                          lw=0.5, alpha=0.9))
        if k == 0:
            ax.annotate("N", xy=(0.055, 0.90), xytext=(0.055, 0.79),
                        xycoords="axes fraction", ha="center", fontsize=7,
                        color="#333",
                        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1))
            ax.plot([0.03, 0.22], [0.03, 0.03], transform=ax.transAxes,
                    color="#333", lw=1.8, solid_capstyle="butt")
            ax.text(0.125, 0.048, "300 km", transform=ax.transAxes,
                    fontsize=6, ha="center", color="#333")

    cax1 = fig.add_axes([0.055, 0.125, 0.36, 0.024])
    cb1 = fig.colorbar(im_seq, cax=cax1, orientation="horizontal")
    cb1.set_label("canopy cover (%)", fontsize=7.5, labelpad=2)
    cb1.ax.tick_params(labelsize=6.5, length=2.5)
    cb1.outline.set_linewidth(0.5); cb1.outline.set_edgecolor("#777")

    cax2 = fig.add_axes([0.535, 0.125, 0.21, 0.024])
    cb2 = fig.colorbar(im_div, cax=cax2, orientation="horizontal",
                       extend="both")
    cb2.set_label("change in canopy cover (pp)", fontsize=7.5, labelpad=2)
    cb2.ax.tick_params(labelsize=6.5, length=2.5)
    cb2.outline.set_linewidth(0.5); cb2.outline.set_edgecolor("#777")

    # ---- change statistics
    d = diff[mask]
    gain = 100 * (d > THR).mean()
    loss = 100 * (d < -THR).mean()
    stable = 100 - gain - loss

    axh = fig.add_axes([0.795, 0.535, 0.185, 0.315])
    cnt, edges = np.histogram(d, bins=42, range=(-32, 32))
    ctr = 0.5 * (edges[:-1] + edges[1:])
    cols = [GAIN if c > THR else LOSS if c < -THR else STABLE for c in ctr]
    axh.bar(ctr, cnt / cnt.sum() * 100, width=edges[1] - edges[0],
            color=cols, edgecolor="white", linewidth=0.25)
    for xv in (-THR, THR):
        axh.axvline(xv, color="#333", lw=0.7, ls=(0, (3, 2)))
    axh.axvline(d.mean(), color="#b23a48", lw=1.1)
    axh.text(0.97, 0.95, "mean %+.1f pp" % d.mean(), transform=axh.transAxes,
             ha="right", va="top", fontsize=6.8, color="#b23a48",
             fontweight="bold")
    axh.set_xlabel("change (percentage points)", fontsize=7.5, labelpad=2)
    axh.set_ylabel("share of pixels (%)", fontsize=7.5, labelpad=2)
    axh.tick_params(labelsize=6.5, length=2.5)
    axh.set_xlim(-32, 32)
    for sp in ("top", "right"):
        axh.spines[sp].set_visible(False)
    axh.text(0.0, 1.03, "(d)", transform=axh.transAxes, fontsize=11,
             fontweight="bold", va="bottom")
    axh.text(0.135, 1.035, "Distribution of change", transform=axh.transAxes,
             fontsize=9.5, fontweight="bold", va="bottom", color="#2b4f42")

    axs = fig.add_axes([0.795, 0.205, 0.185, 0.155])
    left = 0.0
    for val, col, lab in [(loss, LOSS, "loss"), (stable, STABLE, "stable"),
                          (gain, GAIN, "gain")]:
        axs.barh([0], [val], left=left, height=0.5, color=col,
                 edgecolor="white", linewidth=0.8)
        axs.text(left + val / 2, 0, "%.0f%%" % val, ha="center", va="center",
                 fontsize=7, fontweight="bold",
                 color="white" if col != STABLE else "#3a3a3a")
        axs.text(left + val / 2, -0.42, lab, ha="center", va="top",
                 fontsize=6.6, color="#555")
        left += val
    axs.set_xlim(0, 100); axs.set_ylim(-0.75, 0.45)
    axs.set_xticks([]); axs.set_yticks([])
    for sp in axs.spines.values():
        sp.set_visible(False)
    axs.text(0.0, 1.10, "(e)", transform=axs.transAxes, fontsize=11,
             fontweight="bold", va="bottom")
    axs.text(0.135, 1.115, "Area shares ($\\pm$%.0f pp threshold)" % THR,
             transform=axs.transAxes, fontsize=9.5, fontweight="bold",
             va="bottom", color="#2b4f42")

    savefig(fig, "62_change_detection_map.png")


if __name__ == "__main__":
    main()

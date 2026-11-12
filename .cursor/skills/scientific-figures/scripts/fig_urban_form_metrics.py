"""Six urban-form indicators rendered on one shared city footprint.

Every panel is the same 12 x 12 km raster clipped to an identical
star-shaped built-up silhouette, so the six metrics can be read against
each other cell by cell.  Panel geometry is computed from inch offsets
rather than a grid spec: all six maps are exactly square and equal in
size, and each colour bar sits at the same width and distance below its
map.  Only three sequential hues are used across the figure - rust for
morphology, blue for intensity, green for amenity.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, gaussian_filter
from pubstyle import set_pub_style, savefig

RUST = ["#f6f4f0", "#f0dcc9", "#e0b291", "#cd8560", "#ae5a3c", "#7f3524"]
BLUE = ["#f4f6f7", "#d8e4ea", "#a9c4d4", "#7099b6", "#3f6d90", "#204864"]
GREEN = ["#f4f7f2", "#dbe8d5", "#b3d1a9", "#83b276", "#548c4c", "#2d5f2c"]
EPOCH = ["#7f3524", "#ae5a3c", "#cd8560", "#e0b291", "#f0dcc9"]
EPOCH_LAB = ["pre-80", "80s", "90s", "00s", "10s+"]
VOID = "#eeece7"
INK = "#2b2b2b"

N = 170
EXT = 12.0
FIGW, FIGH = 9.3, 7.15
SIDE = 2.75                       # panel side, inches
CB_Y = {1: 0.44, 0: 3.88}         # bar bottom per row, inches
CB_H, CB_GAP = 0.09, 0.13


def cmap_from(cols, name):
    return mpl.colors.LinearSegmentedColormap.from_list(name, cols)


def rect(col, row):
    """Panel and colour-bar rects for grid slot (col, row), in inches."""
    x0 = (0.30 + col * (SIDE + 0.28)) / FIGW
    w = SIDE / FIGW
    cb = CB_Y[row]
    y0 = (cb + CB_H + CB_GAP) / FIGH
    h = SIDE / FIGH
    return ([x0, y0, w, h],
            [x0 + 0.11 * w, cb / FIGH, 0.78 * w, CB_H / FIGH])


def build_city(rng):
    g = np.linspace(0, EXT, N)
    X, Y = np.meshgrid(g, g)

    def blob(cx, cy, s):
        return np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s ** 2)))

    U = (1.00 * blob(6.0, 6.1, 2.35) + 0.46 * blob(3.1, 8.5, 1.25)
         + 0.40 * blob(9.0, 3.8, 1.15) + 0.30 * blob(8.7, 8.4, 0.95)
         + 0.24 * blob(3.3, 3.1, 1.05))
    # radial corridors give the footprint its star shape
    U += 0.30 * np.exp(-((Y - 6.1) ** 2) / (2 * 0.85 ** 2)) * \
        np.exp(-((X - 6.0) ** 2) / (2 * 4.6 ** 2))
    U += 0.26 * np.exp(-((X - 6.0) ** 2) / (2 * 0.80 ** 2)) * \
        np.exp(-((Y - 6.1) ** 2) / (2 * 4.4 ** 2))
    d = (X - Y + 0.2) / np.sqrt(2)
    U += 0.22 * np.exp(-(d ** 2) / (2 * 0.75 ** 2)) * \
        np.exp(-(((X - 6) ** 2 + (Y - 6) ** 2) / (2 * 4.2 ** 2)))

    nz = gaussian_filter(rng.normal(size=(N, N)), 5.0)
    nz /= nz.std()
    U = np.clip(U * (1.0 + 0.30 * nz), 0.0, None)
    U /= U.max()
    mask = gaussian_filter(U, 1.2) > 0.085

    r = np.hypot(X - 6.0, Y - 6.1)
    fine = gaussian_filter(rng.normal(size=(N, N)), 2.4)
    fine /= fine.std()

    epoch = 4 - np.digitize(U, [0.085, 0.155, 0.275, 0.470])
    epoch = np.clip(epoch, 0, 4).astype(float)

    bdens = (4.0 + 46.0 * U ** 1.10) * np.exp(0.16 * fine)
    rdens = (2.2 + 15.5 * U ** 0.75) * np.exp(0.13 * fine)
    far = (0.25 + 5.4 * U ** 1.25) * np.exp(0.18 * fine)

    # green accessibility: walk minutes to the nearest park
    parks = []
    idx = np.argwhere(mask)
    rng.shuffle(idx)
    for i, j in idx:
        p = np.array([g[j], g[i]])
        if all(np.hypot(*(p - q)) > 1.55 for q in parks):
            parks.append(p)
        if len(parks) >= 15:
            break
    pk = np.zeros((N, N), bool)
    for px, py in parks:
        pk[int(py / EXT * (N - 1)), int(px / EXT * (N - 1))] = True
    dist = distance_transform_edt(~pk) * (EXT / (N - 1))
    walk = np.clip(dist / 4.5 * 60.0 * (1.0 + 0.10 * fine), 0.0, 26.0)

    mix = np.clip(0.22 + 0.66 * np.exp(-((r - 2.7) / 1.95) ** 2)
                  + 0.09 * fine, 0.0, 1.0)

    return mask, epoch, bdens, rdens, far, walk, mix, np.array(parks)


def main():
    set_pub_style()
    rng = np.random.default_rng(2024)
    mask, epoch, bdens, rdens, far, walk, mix, parks = build_city(rng)

    def clip(a):
        return np.ma.masked_where(~mask, a)

    panels = [
        (0, 0, "(a)", "Built-up expansion", clip(epoch), None, None,
         "decade of first development"),
        (1, 0, "(b)", "Building density", clip(bdens),
         cmap_from(RUST, "rust"), (0, 40), "gross floor area  (10$^3$ m$^2$/ha)"),
        (2, 0, "(c)", "Road density", clip(rdens),
         cmap_from(BLUE, "blue"), (0, 18), "street length  (km/km$^2$)"),
        (0, 1, "(d)", "Floor area ratio", clip(far),
         cmap_from(BLUE, "blue"), (0, 4), "plot ratio  (-)"),
        (1, 1, "(e)", "Green accessibility", clip(walk),
         cmap_from(GREEN, "green"), (0, 26),
         "walk to nearest park  (min)"),
        (2, 1, "(f)", "Land-use mix", clip(mix),
         cmap_from(GREEN, "green"), (0, 1), "entropy index  (0-1)"),
    ]

    fig = plt.figure(figsize=(FIGW, FIGH))
    for col, row, tag, title, data, cm, lim, cblab in panels:
        pr, cr = rect(col, row)
        ax = fig.add_axes(pr)
        cax = fig.add_axes(cr)
        ax.set_facecolor(VOID)

        if cm is None:                                   # categorical epochs
            cm = mpl.colors.ListedColormap(EPOCH)
            nrm = mpl.colors.BoundaryNorm(np.arange(-0.5, 5.1, 1.0), 5)
            im = ax.imshow(data, cmap=cm, norm=nrm, origin="lower",
                           extent=(0, EXT, 0, EXT), interpolation="nearest")
            cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                              ticks=np.arange(5))
            cb.ax.set_xticklabels(EPOCH_LAB, fontsize=6.0)
        else:
            im = ax.imshow(data, cmap=cm, vmin=lim[0], vmax=lim[1],
                           origin="lower", extent=(0, EXT, 0, EXT),
                           interpolation="bilinear")
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=6.4)
            cb.locator = mpl.ticker.MaxNLocator(nbins=5)
            cb.update_ticks()
        cb.set_label(cblab, fontsize=6.8, labelpad=2.0)
        cb.ax.tick_params(length=2.0, width=0.6, pad=1.5)
        cb.outline.set_linewidth(0.55)
        cb.outline.set_edgecolor("#666666")

        if tag == "(e)":
            ax.plot(parks[:, 0], parks[:, 1], ls="none", marker="o", ms=2.6,
                    mfc="white", mec="#2d5f2c", mew=0.7, zorder=5)
            ax.text(0.975, 0.030, "park centroid", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=6.0, color="#2d5f2c",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="none", alpha=0.85))

        ax.set_xlim(0, EXT); ax.set_ylim(0, EXT)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.7)
            sp.set_edgecolor("#bdb9b1")

        fig.text(pr[0], pr[1] + pr[3] + 0.075 / FIGH, tag, fontsize=10.0,
                 fontweight="bold", ha="left", va="baseline", color=INK)
        fig.text(pr[0] + 0.30 / FIGW, pr[1] + pr[3] + 0.075 / FIGH, title,
                 fontsize=8.4, fontweight="bold", ha="left", va="baseline",
                 color="#33474f")

        if tag == "(a)":
            ax.plot([0.65, 3.65], [0.72, 0.72], color=INK, lw=2.2,
                    solid_capstyle="butt", zorder=6)
            ax.plot([0.65, 2.15], [0.72, 0.72], color="white", lw=1.0,
                    solid_capstyle="butt", zorder=7)
            ax.text(2.15, 1.02, "3 km", fontsize=6.2, ha="center",
                    va="bottom", color=INK, zorder=7)
            ax.annotate("N", xy=(11.05, 11.15), xytext=(11.05, 10.10),
                        ha="center", fontsize=6.8, color=INK,
                        fontweight="bold", zorder=7,
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))

    fig.text(0.032, 0.055 / FIGH, "synthetic 12 x 12 km city, 70 m raster; "
             "every panel is clipped to the same built-up footprint "
             "(grey = non-urban)", ha="left", va="bottom", fontsize=6.5,
             color="#9a9a9a")
    savefig(fig, "210_urban_form_metrics.png")


if __name__ == "__main__":
    main()

"""Figure 1156 - Three depth slices through a trained radiance field.

Rows are three read-outs of ONE volume pair (sigma, c) built in ``field``:

    row 1  sigma(x)                     the density volume itself
    row 2  ||c(x)||                     the emitted-colour volume
    row 3  |grad sigma(x)|              np.gradient of row 1, same array

Columns are three z planes of that same volume, so a shape that grows between
columns really is growing in z.  The contour drawn on row 1 and repeated as a
reference on row 3 is one iso-level of sigma, so the ridge in row 3 has to sit
on it - that is the cross-check the caption points at.

Each row is normalised over its own three panels (the three quantities have
different units), which is why the shared colour bar is labelled per-row and
each row prints its true range in the left margin.  Every number in the figure
is an f-string over these arrays.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, get_cmap

INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"
HAIR = "#d5d5d5"
SEED = 1561  # original id; figure number is 1156

NXY, NZ = 132, 72
Z_SHOWN = (0.28, 0.46, 0.66)
ISO_FRAC = 0.32                  # iso-level as a fraction of max density


def _skin(d, w=0.022):
    """Solid inside the surface (d <= 0), soft shell of width w outside."""
    return np.exp(-(np.clip(d, 0.0, None) / w) ** 2)


def field():
    rng = np.random.default_rng(SEED)
    z, y, x = np.meshgrid(np.linspace(0, 1, NZ), np.linspace(0, 1, NXY),
                          np.linspace(0, 1, NXY), indexing="ij")

    sph = _skin(np.sqrt((x - 0.32) ** 2 + (y - 0.56) ** 2 +
                        (z - 0.38) ** 2) - 0.160)
    cube = _skin(np.maximum.reduce([np.abs(x - 0.705) - 0.120,
                                    np.abs(y - 0.430) - 0.105,
                                    np.abs(z - 0.600) - 0.160]))
    rod = _skin(np.maximum(np.sqrt((x - 0.575) ** 2 + (z - 0.300) ** 2)
                           - 0.052, np.abs(y - 0.430) - 0.300))
    floor = _skin(y - 0.095) * (0.82 + 0.30 * np.sin(11.0 * x + 2.4 * z))
    sigma = 24.0 * sph + 18.0 * cube + 14.0 * rod + 10.0 * floor
    sigma = gaussian_filter(sigma, 1.2) + 0.04

    junk = gaussian_filter(rng.standard_normal((NZ, NXY, NXY)), 3.2)
    junk = np.clip(0.5 + 0.5 * junk / (junk.std() + 1e-9), 0.0, 1.0)
    col = np.zeros(sigma.shape + (3,))
    col += np.array([0.20, 0.38, 0.56]) * sph[..., None] * \
        (0.62 + 0.72 * y)[..., None]
    col += np.array([0.26, 0.52, 0.49]) * cube[..., None] * \
        (0.72 + 0.46 * np.sin(46.0 * y))[..., None]
    col += np.array([0.44, 0.47, 0.54]) * rod[..., None]
    col += np.array([0.58, 0.63, 0.69]) * floor[..., None]
    # colour is unconstrained wherever nothing is ever sampled: low-amplitude
    # junk that the renderer never gets to see
    empty = np.clip(1.0 - (sph + cube + rod + floor), 0.0, 1.0)
    col += 0.26 * empty[..., None] * np.stack([junk, junk * 0.92,
                                               junk * 1.06], axis=-1)
    return sigma, np.clip(col, 0.0, 1.0)


def main():
    set_pub_style(base_font=8.5)
    sigma, col = field()
    dz = 1.0 / (NZ - 1)
    dxy = 1.0 / (NXY - 1)
    gmag = np.sqrt(sum(g ** 2 for g in np.gradient(sigma, dz, dxy, dxy)))
    cmag = np.linalg.norm(col, axis=-1)

    kz = [int(round(v * (NZ - 1))) for v in Z_SHOWN]
    iso = ISO_FRAC * float(sigma.max())
    rows = [
        ("density  $\\sigma$", [sigma[k] for k in kz], "$\\sigma$"),
        ("colour  $\\|\\mathbf{c}\\|$", [cmag[k] for k in kz],
         "$\\|\\mathbf{c}\\|$"),
        ("gradient  $|\\nabla\\sigma|$", [gmag[k] for k in kz],
         "$|\\nabla\\sigma|$"),
    ]

    fig = plt.figure(figsize=(8.4, 7.2))
    gs = fig.add_gridspec(3, 3, left=0.145, right=0.788, top=0.838,
                          bottom=0.062, wspace=0.055, hspace=0.075)
    cmap = get_cmap("cb_sequential_blue")

    for r, (rlab, panels, sym) in enumerate(rows):
        lo = float(min(p.min() for p in panels))
        hi = float(max(p.max() for p in panels))
        norm = mpl.colors.Normalize(lo, hi)
        for c in range(3):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(panels[c], origin="lower", cmap=cmap, norm=norm,
                      extent=(0, 1, 0, 1), interpolation="bilinear", zorder=1)
            ls = "-" if r != 1 else (0, (3.4, 2.2))
            for lw, colr in ((1.6, "white"), (0.7, "#0b3d63")):
                ax.contour(np.linspace(0, 1, NXY), np.linspace(0, 1, NXY),
                           sigma[kz[c]], levels=[iso], colors=[colr],
                           linewidths=lw, linestyles=[ls],
                           zorder=3 + (lw < 1.0))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#9aa7b2")
                s.set_linewidth(0.7)
            if r == 0:
                occ = 100.0 * (sigma[kz[c]] > iso).mean()
                stamp = f"{occ:.1f} % above iso"
                ax.set_title(f"$z$ = {Z_SHOWN[c]:.2f}", fontsize=8.6,
                             fontweight="bold", pad=4.0, color=INK)
            elif r == 1:
                m = sigma[kz[c]] <= iso
                stamp = f"empty-space mean {cmag[kz[c]][m].mean():.2f}"
            else:
                stamp = f"max {gmag[kz[c]].max():.0f}"
            ax.text(0.035, 0.035, stamp, transform=ax.transAxes, ha="left",
                    va="bottom", fontsize=6.4, color=SUB, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=HAIR,
                              lw=0.45, alpha=0.90))
            if c == 0:
                ax.set_ylabel(f"{rlab}\n{sym} $\\in$ [{lo:.1f}, {hi:.1f}]",
                              fontsize=8.0, labelpad=5.0, linespacing=1.7)
        if r == 2:
            ax.annotate("this ridge is the row-1 contour", xy=(0.735, 0.545),
                        xytext=(0.985, 0.905), textcoords="axes fraction",
                        fontsize=6.4, color="#0b3d63", ha="right", va="center",
                        zorder=7,
                        bbox=dict(boxstyle="round,pad=0.22", fc="white",
                                  ec="#b9c9d8", lw=0.5, alpha=0.94),
                        arrowprops=dict(arrowstyle="-", lw=0.7,
                                        color="#0b3d63", shrinkA=2,
                                        shrinkB=3))

    cax = fig.add_axes([0.812, 0.235, 0.019, 0.470])
    cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(0, 1), cmap),
                      cax=cax, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.set_yticklabels(["min", "", "mid", "", "max"])
    cb.ax.tick_params(labelsize=7.0, width=0.7, length=2.5, color="#333333")
    cb.set_label("value within its own row", fontsize=7.4, color=INK,
                 labelpad=4.0)
    fig.text(0.812, 0.838, "one ramp, three\nunits: each row is\nscaled over "
             "its own\nthree panels", fontsize=6.8, color=SUB, ha="left",
             va="top", linespacing=1.6)
    fig.text(0.812, 0.196, f"contour:\n$\\sigma$ = {iso:.1f}\n"
             f"({100 * ISO_FRAC:.0f} % of max)", fontsize=6.8,
             color="#0b3d63", ha="left", va="top", linespacing=1.6)

    e_lo = min(float(cmag[k][sigma[k] <= iso].mean()) for k in kz)
    e_hi = max(float(cmag[k][sigma[k] <= iso].mean()) for k in kz)
    savefig(fig, "1156_wm_nr_field_slices.png")


if __name__ == "__main__":
    main()

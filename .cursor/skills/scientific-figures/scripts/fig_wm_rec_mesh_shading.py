"""Four renderings of one reconstructed mesh under a single camera pose.

The same scanned vessel is shown as a wireframe, flat-shaded, smooth-shaded
and vertex-error-coloured surface.  Camera pose, axis range and the error
colour range are shared across all four panels; the ramp is keyed to the
colour bar on the right.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LightSource, Normalize

from pubstyle import set_pub_style, get_cmap, savefig

SEED = 20261525
ELEV, AZIM = 14.0, -58.0
ERR_CMAP = get_cmap("cb_sequential_blue")
SURF_COL = "#b6c4d1"
NC, NF = (34, 56), (110, 168)          # coarse / fine (t, theta) resolutions


# ------------------------------------------------------------------ geometry
def vessel(nt, nth):
    """Surface of revolution with an angular ripple: a scanned clay vessel."""
    t = np.linspace(0.0, 1.0, nt)
    th = np.linspace(0.0, 2.0 * np.pi, nth)
    T, TH = np.meshgrid(t, th, indexing="ij")
    prof = (0.30 + 0.44 * np.sin(np.pi * T) ** 1.25
            - 0.16 * np.exp(-((T - 0.74) / 0.10) ** 2)      # neck
            + 0.05 * np.exp(-((T - 0.20) / 0.09) ** 2))     # belly ring
    ripple = (1.0 + 0.085 * np.sin(3.0 * TH) * np.sin(np.pi * T)
              + 0.040 * np.cos(5.0 * TH + 1.2)
              - 0.070 * np.exp(-((T - 0.52) / 0.13) ** 2)
              * np.exp(-((np.cos(TH - 0.9) - 1.0) / 0.55) ** 2))  # dent
    R = prof * ripple
    return (R * np.cos(TH), R * np.sin(TH), -1.0 + 2.0 * T), (T, TH)


def error_field(T, TH, rng):
    """Per-vertex distance to the reference scan, in millimetres."""
    e = (0.16
         + 0.62 * np.exp(-((T - 0.78) / 0.11) ** 2)             # thin neck
         + 0.34 * np.exp(-((T - 0.02) / 0.06) ** 2)             # base rim
         + 0.30 * np.exp(-((T - 0.52) / 0.12) ** 2)
         * np.exp(-((np.cos(TH - 0.9) - 1.0) / 0.60) ** 2)      # dent
         + 0.10 * np.abs(np.sin(2.0 * TH + 1.4)) * T)
    e += 0.045 * rng.normal(0.0, 1.0, T.shape)
    return np.clip(e, 0.0, None)


def face_mean(a):
    """Vertex field -> per-quad field, matching plot_surface's facet grid."""
    return 0.25 * (a[:-1, :-1] + a[1:, :-1] + a[:-1, 1:] + a[1:, 1:])


def style3d(ax, lim=0.78, zlim=1.06):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-zlim, zlim)
    ax.set_box_aspect((1.0, 1.0, 1.62))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)


# -------------------------------------------------------------------- figure
def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    (Xc, Yc, Zc), _ = vessel(*NC)
    (Xf, Yf, Zf), (Tf, THf) = vessel(*NF)
    err = error_field(Tf, THf, rng)
    err_face = face_mean(err)
    vmin, vmax = 0.0, float(np.percentile(err_face, 99.0))
    norm = Normalize(vmin, vmax)
    ls = LightSource(azdeg=-40, altdeg=58)

    n_vert = Xc.size
    n_edge = (NC[0] - 1) * NC[1] + NC[0] * (NC[1] - 1)
    n_facet = (NC[0] - 1) * (NC[1] - 1) * 2
    n_fine = (NF[0] - 1) * (NF[1] - 1) * 2

    fig = plt.figure(figsize=(11.2, 5.8))
    PW = 0.212
    x0s = [0.008 + i * PW for i in range(4)]
    RECT = (-0.052, 0.088, 0.104, 0.830)

    axes = []
    for x0 in x0s:
        ax = fig.add_axes([x0 + RECT[0], RECT[1], PW + RECT[2], RECT[3]],
                          projection="3d")
        style3d(ax)
        axes.append(ax)

    # (a) wireframe
    axes[0].plot_wireframe(Xc, Yc, Zc, rstride=1, cstride=1,
                           color="#5f7183", linewidth=0.32, alpha=0.95)

    # (b) flat shaded: coarse facets, no interpolation across edges
    axes[1].plot_surface(Xc, Yc, Zc, color=SURF_COL, shade=True,
                         lightsource=ls, edgecolor="#8fa0b0", linewidth=0.18,
                         antialiased=False, rstride=1, cstride=1)

    # (c) smooth shaded: the same surface at scan resolution
    axes[2].plot_surface(Xf, Yf, Zf, color=SURF_COL, shade=True,
                         lightsource=ls, edgecolor="none", antialiased=True,
                         rstride=1, cstride=1)

    # (d) vertex error
    axes[3].plot_surface(Xf, Yf, Zf, facecolors=ERR_CMAP(norm(err_face)),
                         shade=True, lightsource=ls, edgecolor="none",
                         antialiased=True, rstride=1, cstride=1)

    titles = ["(a)  Wireframe", "(b)  Flat shaded", "(c)  Smooth shaded",
              "(d)  Vertex error"]
    notes = [f"control mesh\n{n_vert:,} vertices, {n_edge:,} edges",
             f"control mesh\n{n_facet:,} triangles, per-face normals",
             f"full scan\n{n_fine:,} triangles, per-vertex normals",
             f"full scan\nmean {err_face.mean():.2f} mm, "
             f"RMS {np.sqrt((err_face ** 2).mean()):.2f} mm"]
    for x0, ttl, note in zip(x0s, titles, notes):
        fig.text(x0 + PW / 2, 0.118, note, fontsize=7.0, ha="center",
                 va="top", color="#555555", linespacing=1.45)

    cax = fig.add_axes([0.898, 0.185, 0.014, 0.610])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=ERR_CMAP), cax=cax,
                      orientation="vertical")
    cb.set_label("distance to reference scan (mm)", fontsize=7.4,
                 color="#1a1a1a", labelpad=3.0)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.8, width=0.7, length=2.4, color="#333333")


    savefig(fig, "1120_wm_rec_mesh_shading.png")


if __name__ == "__main__":
    main()

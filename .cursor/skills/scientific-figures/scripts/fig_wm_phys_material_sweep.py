"""Figure 1144 - Material sweep: an elastic ring squeezed between two platens.

A 5 x 5 matrix over Young's modulus (rows) and Poisson ratio (columns).  Every
cell shows the same annular specimen - outer radius 25 mm, 55 % bore - after
the same platen load, so the only thing that changes between cells is the
material.

The kinematics are deliberately transparent: the platen closure is inversely
proportional to the modulus, the lateral response scales with nu / (1 - nu),
and a smooth two-sided clamp against the platens produces the flat contact
facets.  Displacement magnitude is then measured node by node between the
reference and deformed configurations, so every cell is shaded by a quantity
derived from the geometry actually drawn, and one colour normalisation covers
all twenty-five.

The compression percentage in each cell is measured from the deformed outer
boundary, not from the input parameter.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, get_cmap

ROUT, RIN = 1.0, 0.55           # normalised specimen radii
SCALE = 25.0                    # mm per unit -> 25 mm outer radius
LOAD = 0.62                     # platen closure x modulus (MPa)
SOFT = 0.030                    # smoothing of the platen clamp
EMOD = np.array([1.5, 2.6, 4.5, 7.8, 13.5])       # MPa
NU = np.array([0.15, 0.25, 0.35, 0.42, 0.48])
REF = (2, 2)                    # calibrated material, framed in red
ACCENT, INK, SUB, MUTED = "#b23a48", "#1a1a1a", "#555555", "#777777"
XLIM, YLIM = 1.46, 1.38


def ring_mesh(nr=11, nt=72):
    r = np.linspace(RIN, ROUT, nr)
    t = np.linspace(0.0, 2.0 * np.pi, nt, endpoint=False)
    Rg, Tg = np.meshgrid(r, t, indexing="ij")
    x, y = (Rg * np.cos(Tg)).ravel(), (Rg * np.sin(Tg)).ravel()
    idx = np.arange(nr * nt).reshape(nr, nt)
    tri = []
    for i in range(nr - 1):
        jn = np.r_[np.arange(1, nt), 0]
        a, b = idx[i], idx[i][jn]
        c, d = idx[i + 1][jn], idx[i + 1]
        tri += list(np.c_[a, b, c]) + list(np.c_[a, c, d])
    return x, y, np.array(tri), idx


def squeeze(x, y, e_mod, nu):
    """Platen compression: closure set by 1/E, lateral spread set by nu."""
    c = LOAD / e_mod
    beta = 0.55 * nu / (1.0 - nu) * 2.0
    h = ROUT * (1.0 - c)                          # platen half-gap
    yr = y * (1.0 - 0.85 * c)
    yd = (yr - SOFT * np.logaddexp(0.0, (yr - h) / SOFT)
          + SOFT * np.logaddexp(0.0, (-yr - h) / SOFT))
    return x * (1.0 + beta * c), yd, h


def main():
    set_pub_style(base_font=8)
    X, Y, tris, idx = ring_mesh()
    outer, inner = idx[-1], idx[0]

    cells = np.empty((len(EMOD), len(NU)), object)
    for i, e in enumerate(EMOD):
        for j, nu in enumerate(NU):
            xd, yd, h = squeeze(X, Y, e, nu)
            u = np.hypot(xd - X, yd - Y) * SCALE
            hgt = yd[outer].max() - yd[outer].min()
            wid = xd[outer].max() - xd[outer].min()
            cells[i, j] = dict(xd=xd, yd=yd, u=u, h=h,
                               comp=100.0 * (1.0 - hgt / (2.0 * ROUT)),
                               wide=100.0 * (wid / (2.0 * ROUT) - 1.0),
                               tri=mtri.Triangulation(xd, yd, tris))

    cmap = get_cmap("cb_sequential_blue")
    norm = mpl.colors.Normalize(
        0.0, max(c["u"].max() for c in cells.ravel()))

    fig = plt.figure(figsize=(8.4, 7.6))
    gs = fig.add_gridspec(len(EMOD), len(NU), left=0.128, right=0.852,
                          top=0.838, bottom=0.082, wspace=0.05, hspace=0.05)

    for i, e in enumerate(EMOD):
        for j, nu in enumerate(NU):
            ax = fig.add_subplot(gs[i, j])
            cell = cells[i, j]
            ref = (i, j) == REF
            ax.set_xlim(-XLIM, XLIM)
            ax.set_ylim(-YLIM, YLIM)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(1.4 if ref else 0.5)
                sp.set_color(ACCENT if ref else "#b8bcc2")

            for sgn in (-1.0, 1.0):
                ax.add_patch(Rectangle((-1.30, sgn * (cell["h"] + 0.02) -
                                        (0.13 if sgn > 0 else 0.0)),
                                       2.60, 0.13, facecolor="#a9aeb6",
                                       edgecolor="#7c828a", lw=0.4, zorder=2))
            ax.tripcolor(cell["tri"], cell["u"], cmap=cmap, norm=norm,
                         shading="gouraud", zorder=3, rasterized=True)
            for ring in (outer, inner):
                ax.plot(np.r_[cell["xd"][ring], cell["xd"][ring][0]],
                        np.r_[cell["yd"][ring], cell["yd"][ring][0]],
                        color="#20374f", lw=0.75, zorder=4)
            ax.text(0.965, 0.035, f"{cell['comp']:.0f}%",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=6.2, color=SUB, zorder=5)

            if i == 0:
                ax.set_title(f"$\\nu$ = {nu:.2f}", fontsize=8.2, pad=3.2,
                             color=INK)
            if j == 0:
                ax.set_ylabel(f"{e:.1f}", fontsize=8.2, labelpad=3.4,
                              color=INK)

    cax = fig.add_axes([0.874, 0.230, 0.0135, 0.460])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("displacement magnitude $|u|$ (mm)", fontsize=7.4)
    cb.ax.tick_params(labelsize=6.8, width=0.7, length=2.4)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    fig.text(0.030, 0.460, "Young's modulus $E$ (MPa)", fontsize=8.6,
             rotation=90, ha="center", va="center", color=INK)

    soft, stiff = cells[0, REF[1]], cells[-1, REF[1]]
    lo_nu, hi_nu = cells[0, 0], cells[0, -1]
    savefig(fig, "1144_wm_phys_material_sweep.png")


if __name__ == "__main__":
    main()

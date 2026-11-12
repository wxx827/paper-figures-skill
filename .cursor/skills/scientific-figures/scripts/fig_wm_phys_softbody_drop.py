"""Figure 1141 - Soft-body drop: a six-frame elastic deformation sequence.

Two rows of three frames of a hyperelastic *sphere* released above a rigid
floor: free fall, approach, first contact, maximum compression, recovery and
rebound.  Each frame is a triangular FE mesh drawn in its deformed
configuration and shaded by the maximum principal Green-Lagrange strain,
computed per element from the actual deformation gradient (linear triangles,
so ``F`` is exact).

The kinematics are a closed-form squash map rather than a solver, but they are
built to respect the one thing a viewer checks first: a neo-Hookean solid is
very nearly incompressible, so squashing it has to push material *sideways*.
The vertical stretch ratio lambda(Y) is graded over height - hardest against
the floor, mildest at the crown - and the radial stretch is tied to it by
1 / sqrt(lambda), the value that preserves the volume of an axisymmetric
layer.  The silhouette that falls out is a barrel with a bulged waist and a
necked-in contact patch, not a sphere with its bottom sliced off.  A soft-plus
against the floor then flattens the patch instead of leaving it tangent.

Strain is *derived* from that map, so every number printed on a frame is a
measurement of the geometry actually drawn.  All six frames share one colour
normalisation and one pair of axis limits; contact frames carry a Hertzian
ring of contact-force arrows scaled by the semi-elliptical pressure profile
over the measured patch.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Ellipse, Rectangle

from pubstyle import set_pub_style, savefig

R = 1.0                 # undeformed radius (normalised)
# The whole body compresses under an impact, hardest against the floor: the
# grading runs from a full share of the closure at the contact down to
# ``PSI_FLOOR`` of it at the crown.  Letting the crown come down too is what
# turns the silhouette from a sliced dome into a barrel - the widest point
# then sits near mid-height instead of right above the contact facet.
PSI_FLOOR, PSI_WID = 0.55, 1.20
PEN = 0.08              # how far the affine body is pressed through the floor
SOFT = 0.050 * R        # soft-plus width of the floor constraint
C_REF = 0.380           # compression of the deepest frame, sets arrow scale
FORCE = "#b23a48"       # the single accent colour (contact forces)
INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"
XLIM, YLO, YHI = 2.05 * R, -0.40 * R, 2.80 * R

# (phase, time in ms, centre height / R when airborne, compression c)
# Airborne frames scale uniformly; contact frames run the graded map below.
FRAMES = [("free fall",       0, 1.70,  0.000),
          ("approach",       28, 1.12,  0.000),
          ("first contact",  44, None,  0.100),
          ("max compression", 58, None, C_REF),
          ("recovering",     78, None,  0.150),
          ("rebound",       112, 1.42, -0.080)]


def disc_mesh(n_rings=15):
    """Concentric-ring node cloud on the unit disc, Delaunay triangulated."""
    pts = [(0.0, 0.0)]
    for k in range(1, n_rings + 1):
        r = k / n_rings
        m = max(6, int(round(2.0 * np.pi * r * n_rings)))
        a = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False) + 0.31 * k
        pts += list(np.c_[r * np.cos(a), r * np.sin(a)])
    p = np.array(pts)
    return p[:, 0], p[:, 1], mtri.Triangulation(p[:, 0], p[:, 1])


def _profile(c, n=1601):
    """Graded vertical stretch: hardest at the floor, mildest at the crown."""
    s = np.linspace(-1.0, 1.0, n)
    psi = PSI_FLOOR + (1.0 - PSI_FLOOR) * np.exp(-((s + 1.0) / PSI_WID) ** 2)
    lam = 1.0 - c * psi
    yl = np.r_[0.0, np.cumsum(0.5 * (lam[1:] + lam[:-1]) * np.diff(s))]
    return s, lam, yl


def deform(X, Y, yc, c, contact):
    """Map the reference disc into its deformed configuration.

    Radial stretch is ``1 / sqrt(lambda)`` in both cases, the incompressible
    response of an axisymmetric layer: squashing bulges the waist, stretching
    necks it in.
    """
    if contact:
        s, lam, yl = _profile(c)
        lamY = np.interp(Y, s, lam)
        yr = R * np.interp(Y, s, yl) - PEN * c * R
    else:
        lamY = np.full_like(Y, 1.0 - c)
        yr = yc + Y * R * lamY
    xr = X * R / np.sqrt(lamY)
    return xr, SOFT * np.logaddexp(0.0, yr / SOFT)     # flatten against y = 0


def principal_strain(tri, X, Y, xd, yd):
    """Max principal Green-Lagrange strain per linear triangle."""
    t = tri.triangles
    dx1, dy1 = X[t[:, 1]] - X[t[:, 0]], Y[t[:, 1]] - Y[t[:, 0]]
    dx2, dy2 = X[t[:, 2]] - X[t[:, 0]], Y[t[:, 2]] - Y[t[:, 0]]
    det = dx1 * dy2 - dx2 * dy1
    ex1, ey1 = xd[t[:, 1]] - xd[t[:, 0]], yd[t[:, 1]] - yd[t[:, 0]]
    ex2, ey2 = xd[t[:, 2]] - xd[t[:, 0]], yd[t[:, 2]] - yd[t[:, 0]]
    f11 = (ex1 * dy2 - ex2 * dy1) / det
    f12 = (ex2 * dx1 - ex1 * dx2) / det
    f21 = (ey1 * dy2 - ey2 * dy1) / det
    f22 = (ey2 * dx1 - ey1 * dx2) / det
    c11, c22 = f11 ** 2 + f21 ** 2, f12 ** 2 + f22 ** 2
    c12 = f11 * f12 + f21 * f22
    e11, e22, e12 = 0.5 * (c11 - 1.0), 0.5 * (c22 - 1.0), 0.5 * c12
    return 0.5 * (e11 + e22) + np.hypot(0.5 * (e11 - e22), e12)


def main():
    set_pub_style(base_font=8)
    X, Y, tri = disc_mesh()
    X, Y = X * R, Y * R

    states = []
    for (phase, t_ms, yc, c) in FRAMES:
        contact = yc is None
        xd, yd = deform(X / R, Y / R, yc, c, contact)
        if contact:
            yc = 0.5 * (yd.min() + yd.max())
        eps = principal_strain(tri, X, Y, xd, yd)
        # contact half-width and waist half-width, both measured on the
        # deformed boundary rather than assumed from the input parameter
        near = yd < 0.050 * R
        a = float(np.abs(xd[near]).max()) if near.any() else 0.0
        states.append(dict(phase=phase, t=t_ms, yc=yc, c=c, xd=xd, yd=yd,
                           eps=eps, a=a, contact=contact,
                           waist=float(np.abs(xd).max()),
                           height=float(yd.max() - yd.min()),
                           tri=mtri.Triangulation(xd, yd, tri.triangles)))

    norm = mpl.colors.Normalize(0.0, max(s["eps"].max() for s in states))
    cmap = mpl.colormaps["cividis"]
    peak = max(states, key=lambda s: s["eps"].max())
    k = int(np.argmax(peak["eps"]))
    ty = peak["yd"][peak["tri"].triangles[k]].mean()

    fig = plt.figure(figsize=(8.6, 5.4))
    gs = fig.add_gridspec(2, 3, left=0.038, right=0.885, top=0.838,
                          bottom=0.098, wspace=0.055, hspace=0.19)

    hull = np.argsort(np.arctan2(Y, X))
    hull = hull[np.hypot(X[hull], Y[hull]) > 0.985 * R]

    for j, s in enumerate(states):
        ax = fig.add_subplot(gs[j // 3, j % 3])
        ax.set_xlim(-XLIM, XLIM)
        ax.set_ylim(YLO, YHI)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.55)
            sp.set_color("#9a9a9a")

        ax.add_patch(Rectangle((-XLIM, YLO), 2 * XLIM, -YLO,
                               facecolor="#efefef", edgecolor="#c2c2c2",
                               hatch="////", lw=0.45, zorder=1))
        ax.axhline(0.0, color="#3a3a3a", lw=1.1, zorder=2)

        drop = max(s["yd"].min(), 0.0)
        ax.add_patch(Ellipse((0.0, -0.032 * R),
                             2.0 * max(s["a"], 0.55 * R) * (1.0 + 0.45 * drop),
                             0.10 * R, facecolor="#8a8a8a", edgecolor="none",
                             alpha=0.30 / (1.0 + 1.6 * drop), zorder=3))

        ax.tripcolor(s["tri"], facecolors=s["eps"], cmap=cmap, norm=norm,
                     shading="flat", zorder=4, rasterized=True)
        ax.triplot(s["tri"], color="#ffffff", lw=0.18, alpha=0.42, zorder=5)
        ax.plot(np.r_[s["xd"][hull], s["xd"][hull][0]],
                np.r_[s["yd"][hull], s["yd"][hull][0]],
                color="#12233a", lw=0.9, zorder=6)

        if s["contact"] and s["a"] > 0.05 * R:
            xa = np.linspace(-0.86 * s["a"], 0.86 * s["a"], 7)
            p = np.sqrt(np.clip(1.0 - (xa / s["a"]) ** 2, 0, 1))
            for xi, pi in zip(xa, p):
                ax.annotate("", xy=(xi, 0.02 * R + 0.72 * R * pi * s["c"] /
                                    C_REF), xytext=(xi, 0.02 * R), zorder=8,
                            arrowprops=dict(arrowstyle="-|>", lw=0.8,
                                            color=FORCE, mutation_scale=6.0,
                                            shrinkA=0, shrinkB=0))
            ax.plot([-s["a"], s["a"]], [0.0, 0.0], color=FORCE, lw=1.7,
                    solid_capstyle="butt", zorder=7)
        else:                                 # airborne: show the velocity
            up = s["c"] < 0.0
            ax.annotate("", xy=(1.66 * R, s["yc"] + (0.82 if up else -0.82) * R),
                        xytext=(1.66 * R, s["yc"] + (0.22 if up else -0.22) * R),
                        zorder=8,
                        arrowprops=dict(arrowstyle="-|>", lw=0.8,
                                        color="#8a8a8a", mutation_scale=6.5))
            ax.text(1.66 * R, s["yc"] + (0.95 if up else -0.98) * R,
                    "v", fontsize=6.6, color=MUTED, ha="center",
                    va="bottom" if up else "top", style="italic")

        ax.set_title(s["phase"], fontsize=8.2, fontweight="bold", color=INK,
                     pad=3.2)
        ax.set_xlabel(f"$t$ = {s['t']} ms      "
                      rf"$\varepsilon_1^{{\max}}$ = {s['eps'].max():.3f}",
                      fontsize=7.2, color=SUB, labelpad=2.8)

    cax = fig.add_axes([0.902, 0.205, 0.0125, 0.500])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("max principal strain $\\varepsilon_1$ (-)", fontsize=7.2)
    cb.ax.tick_params(labelsize=6.8, width=0.7, length=2.4)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    # both of these run past the 8.6 in canvas as single lines, and
    # `bbox_inches="tight"` widens the export instead of clipping them: the
    # footnote alone measured 10.2 in and dragged h/w from 0.628 down to 0.514.
    # The subtitle is broken by hand at the "|" - it carries mathtext, which
    # `textwrap` would happily cut through the middle of.
    savefig(fig, "1141_wm_phys_softbody_drop.png")


if __name__ == "__main__":
    main()

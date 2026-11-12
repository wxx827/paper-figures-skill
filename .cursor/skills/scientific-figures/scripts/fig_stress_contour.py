"""Finite-element style von Mises stress map of a plate with a central hole.

Panel (a): ``tricontourf`` of the von Mises stress on an unstructured,
hole-graded triangular mesh, drawn in the (exaggerated) deformed
configuration with the deformed element edges overlaid as hairlines and the
undeformed outline dashed behind.  The stress concentration is annotated and
a warm sequential colorbar in MPa sits on the right.

Panel (b): the hoop stress along the ligament above the hole - the linear
finite-element interpolation sampled from the mesh against the closed-form
solution - showing the Kt = 3 concentration.

Physics: exact Kirsch / Muskhelishvili solution for an infinite plate with a
circular hole under remote uniaxial tension.  Stresses come from the complex
potentials phi = S z/4 + S a^2/(2 z) and psi = -S z/2 - S a^2/(2 z)
+ S a^4/(2 z^3) (the coefficients follow from the traction-free condition on
|z| = a); displacements from 2 mu (u + i v) = kappa phi - z conj(phi')
- conj(psi).  Hence equilibrium and compatibility hold, the hole boundary
carries no traction and the peak hoop stress is exactly 3 sigma_0.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle, Rectangle
from pubstyle import set_pub_style, savefig, add_panel_label, get_cmap

A = 10.0          # hole radius [mm]
L = 30.0          # plate half-width [mm]
S0 = 100.0        # remote tension [MPa]
E = 2.10e5        # Young's modulus [MPa]
NU = 0.30
MU = E / (2.0 * (1.0 + NU))
KAPPA = (3.0 - NU) / (1.0 + NU)      # plane stress


def stresses(x, y):
    """Cartesian stress components [MPa] from the complex potentials."""
    z = x + 1j * y
    dphi = S0 / 4.0 - S0 * A ** 2 / (2.0 * z ** 2)
    ddphi = S0 * A ** 2 / z ** 3
    dpsi = -S0 / 2.0 + S0 * A ** 2 / (2.0 * z ** 2) \
        - 1.5 * S0 * A ** 4 / z ** 4
    tr = 4.0 * np.real(dphi)                    # sxx + syy
    comb = 2.0 * (np.conj(z) * ddphi + dpsi)    # syy - sxx + 2 i txy
    sxx = 0.5 * (tr - np.real(comb))
    syy = 0.5 * (tr + np.real(comb))
    txy = 0.5 * np.imag(comb)
    return sxx, syy, txy


def displacement(x, y):
    """Plane-stress displacement field [mm]."""
    z = x + 1j * y
    phi = S0 * z / 4.0 + S0 * A ** 2 / (2.0 * z)
    dphi = S0 / 4.0 - S0 * A ** 2 / (2.0 * z ** 2)
    psi = -S0 * z / 2.0 - S0 * A ** 2 / (2.0 * z) + S0 * A ** 4 / (2 * z ** 3)
    w = (KAPPA * phi - z * np.conj(dphi) - np.conj(psi)) / (2.0 * MU)
    return np.real(w), np.imag(w)


def von_mises(sxx, syy, txy):
    return np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * txy ** 2)


def build_mesh(rng):
    """Hole-graded unstructured node cloud + Delaunay triangulation."""
    def spacing(r):
        return np.clip(0.13 * A * (1.0 + 1.30 * (r - A) / A), 0.13 * A,
                       0.44 * A)

    pts = []
    nb = 96
    th = np.linspace(0, 2 * np.pi, nb, endpoint=False)
    pts += list(np.c_[A * np.cos(th), A * np.sin(th)])
    for k, rr in enumerate([1.14, 1.32, 1.55]):
        m = int(nb / (1 + 0.18 * k))
        t = np.linspace(0, 2 * np.pi, m, endpoint=False) + 0.03 * k
        pts += list(np.c_[A * rr * np.cos(t), A * rr * np.sin(t)])
    for s in np.linspace(-L, L, 21):
        pts += [(s, -L), (s, L), (-L, s), (L, s)]
    fixed = np.array(pts)

    cand = rng.uniform(-L, L, size=(34000, 2))
    r = np.hypot(cand[:, 0], cand[:, 1])
    cand = cand[r > A * 1.14]
    acc = list(fixed)
    arr = np.array(acc)
    for p in cand:
        d = np.hypot(arr[:, 0] - p[0], arr[:, 1] - p[1])
        if d.min() > spacing(np.hypot(*p)):
            acc.append(p)
            arr = np.vstack([arr, p])
    nodes = np.array(acc)

    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
    t = tri.triangles
    cx = nodes[t, 0].mean(axis=1)
    cy = nodes[t, 1].mean(axis=1)
    bad = np.hypot(cx, cy) < A * 1.001
    for i in range(3):
        j = (i + 1) % 3
        mx = 0.5 * (nodes[t[:, i], 0] + nodes[t[:, j], 0])
        my = 0.5 * (nodes[t[:, i], 1] + nodes[t[:, j], 1])
        bad |= np.hypot(mx, my) < A * 0.985
    tri.set_mask(bad)
    return nodes, tri


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(97)
    nodes, tri = build_mesh(rng)
    x, y = nodes[:, 0], nodes[:, 1]
    sxx, syy, txy = stresses(x, y)
    svm = von_mises(sxx, syy, txy)
    ux, uy = displacement(x, y)
    ux -= ux.mean()
    uy -= uy.mean()
    scale = 0.10 * L / np.abs(np.hypot(ux, uy)).max()
    xd, yd = x + scale * ux, y + scale * uy
    trid = mtri.Triangulation(xd, yd, tri.triangles)
    trid.set_mask(tri.mask)

    fig = plt.figure(figsize=(7.2, 3.72))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.58, 1.0], left=0.055,
                          right=0.965, top=0.905, bottom=0.145, wspace=0.52)
    ax = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[0, 1])

    lv = np.linspace(0, 300, 31)
    cf = ax.tricontourf(trid, svm, levels=lv, cmap=get_cmap("cb_sequential_warm"), extend="max")
    cf.set_edgecolor("face")
    ax.triplot(trid, color="#5f6870", lw=0.25, alpha=0.7, zorder=3)
    ax.add_patch(Rectangle((-L, -L), 2 * L, 2 * L, fill=False,
                           ec="#8a8a8a", lw=0.7, ls=(0, (4, 2)), zorder=2))
    ax.add_patch(Circle((0, 0), A, fill=False, ec="#8a8a8a", lw=0.7,
                        ls=(0, (4, 2)), zorder=2))

    k = int(np.argmax(svm))
    ax.plot(xd[k], yd[k], marker="o", ms=5.0, mfc="none", mec="#111418",
            mew=1.1, zorder=8)
    ax.annotate(r"$\sigma_{vM}^{max}$ = %.0f MPa" % svm[k] + "\n"
                r"$K_t$ = %.2f" % (svm.max() / S0),
                xy=(xd[k], yd[k]), xytext=(-31.5, 30.5), fontsize=6.6,
                fontweight="bold", color="#111418", ha="left", va="center",
                zorder=9,
                bbox=dict(boxstyle="round,pad=0.24", fc="white",
                          ec="#b9b9b9", lw=0.5, alpha=0.94),
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#111418",
                                shrinkA=2, shrinkB=5))
    ax.text(0, 0, "hole\n" + r"$a$ = 10 mm", ha="center", va="center",
            fontsize=6.2, color="#3a4049", zorder=8)

    for sgn in (-1, 1):
        for yy in (-20, 0, 20):
            ax.annotate("", xy=(sgn * (L + 10.0), yy),
                        xytext=(sgn * (L + 2.5), yy), annotation_clip=False,
                        arrowprops=dict(arrowstyle="-|>", lw=1.0,
                                        color="#3a4049"))
    ax.text(0.995, 0.99, r"remote tension  $\sigma_0$ = 100 MPa",
            transform=ax.transAxes, fontsize=6.5, color="#3a4049",
            ha="right", va="top")

    ax.set_xlim(-L - 12, L + 12)
    ax.set_ylim(-L - 4, L + 4)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ (mm)")
    ax.set_ylabel("$y$ (mm)")
    ax.set_xticks([-30, -15, 0, 15, 30])
    ax.set_yticks([-30, -15, 0, 15, 30])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#333333")
        s.set_linewidth(0.8)
    ax.tick_params(length=2.4)
    add_panel_label(ax, "(a)", x=-0.10, y=1.005, size=9.5)
    ax.set_title("deformed shape, %.0f$\\times$ magnification"
                 % round(scale, -1), fontsize=7.2, color="#3a4049", pad=3)

    cb = fig.colorbar(cf, ax=ax, fraction=0.040, pad=0.025, shrink=0.96,
                      extend="max", ticks=np.arange(0, 301, 50))
    cb.set_label(r"von Mises stress  $\sigma_{vM}$  (MPa)", fontsize=7.2)
    cb.ax.tick_params(labelsize=6.8, width=0.7, length=2.2)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    # ---- panel (b): ligament hoop stress ---------------------------------
    interp = mtri.LinearTriInterpolator(tri, sxx)
    yl = np.linspace(A, L, 400)
    fem = np.asarray(interp(np.zeros_like(yl), yl))
    exact = S0 * (1.0 + 0.5 * (A / yl) ** 2 + 1.5 * (A / yl) ** 4)

    axr.axhline(1.0, color="#c9c9c9", lw=0.7, zorder=0)
    axr.plot(yl / A, exact / S0, color="#2b3b4f", lw=1.5,
             label="Kirsch (exact)")
    axr.plot(yl[::14] / A, fem[::14] / S0, ls="none", marker="o", ms=3.0,
             mfc="#d9622b", mec="#7a3413", mew=0.5,
             label="linear FE mesh")
    axr.plot([1.0], [3.0], marker="*", ms=8.0, mfc="#e8b02a",
             mec="#12161b", mew=0.6, zorder=6)
    axr.annotate(r"$K_t$ = 3 at the hole edge", xy=(1.0, 3.0),
                 xytext=(1.55, 2.72), fontsize=6.5, color="#111418",
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color="#111418",
                                 shrinkA=1, shrinkB=4))
    axr.text(2.75, 1.13, "remote field", fontsize=6.3, color="#6b7280",
             ha="center", va="bottom")
    err = 100.0 * np.max(np.abs(fem - exact) / exact)
    axr.text(0.97, 0.53, "peak mesh error\n%.1f %%" % err,
             transform=axr.transAxes, fontsize=6.3, color="#7a3413",
             ha="right", va="center")
    axr.set_xlim(1.0, 3.0)
    axr.set_ylim(0.6, 3.35)
    axr.set_xlabel(r"distance along ligament  $y / a$")
    axr.set_ylabel(r"hoop stress  $\sigma_{xx} / \sigma_0$")
    axr.grid(True, lw=0.45, color="#e8e8e8")
    axr.legend(loc="upper right", fontsize=6.4, handlelength=1.7,
               borderpad=0.25, labelspacing=0.3)
    add_panel_label(axr, "(b)", x=-0.20, y=1.005, size=9.5)

    savefig(fig, "97_stress_contour.png")


if __name__ == "__main__":
    main()

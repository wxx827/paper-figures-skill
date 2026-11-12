"""Figure 1143 - Cloth drape: tension shading, fold curvature and convergence.

Panel (a) is a circular tablecloth settled over a round table, drawn as a
structured quad mesh in 3D and shaded by membrane tension.  The drape is a
closed-form rollover: the tangent angle of the skirt turns from horizontal to
vertical with an exponential relaxation past the rim, so the profile is
integrated rather than guessed, and a periodic radial modulation gathers the
skirt into lobes.

Tension follows from that geometry by a vertical force balance - the weight of
the material hanging below a given ring divided by the deformed circumference
it hangs from - so it is largest where the cloth leaves the table edge and
zero at the hem, and it dips wherever a lobe bulges outward.

Panels (b) and (c) are sized to 58 % of the main panel.  (b) is the curvature
of the hem measured on the very lobes visible in (a); (c) is the energy the
projective-dynamics solver still has to shed, on a log axis.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pubstyle import set_pub_style, savefig, get_cmap, add_panel_label

RC, RT = 1.40, 0.55        # cloth radius / table radius (m)
NF = 9                     # lobes gathered into the skirt
SIGMA_G = 4.2              # areal weight of the fabric (N/m^2)
ZF = -0.90                 # floor height (m)
INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"
ACCENT = "#b23a48"
ELEV, AZIM = 22, -58       # one view_init for every 3D panel in this figure


def drape(ns=60, nt=144):
    """Material polar grid -> deformed cloth, with the fold modulation."""
    s = np.linspace(0.0, RC, ns)
    th = np.linspace(0.0, 2.0 * np.pi, nt, endpoint=False)
    S, TH = np.meshgrid(s, th, indexing="ij")

    ag = np.linspace(0.0, RC - RT, 600)
    phi = 0.5 * np.pi * (1.0 - np.exp(-ag / 0.135))       # tangent angle
    ir = np.r_[0.0, np.cumsum(0.5 * (np.cos(phi[1:]) + np.cos(phi[:-1])) *
                              np.diff(ag))]
    iz = np.r_[0.0, np.cumsum(0.5 * (np.sin(phi[1:]) + np.sin(phi[:-1])) *
                              np.diff(ag))]

    a = np.clip(S - RT, 0.0, None)
    rho0 = np.where(S <= RT, S, RT + np.interp(a, ag, ir))
    z0 = np.where(S <= RT, 0.0, -np.interp(a, ag, iz))
    amp = 0.105 * (1.0 - np.exp(-a / 0.22))
    rho = rho0 + amp * np.cos(NF * TH + 0.35)
    z = z0 + 0.17 * amp * np.cos(NF * TH + 1.25)
    return s, th, S, TH, rho, z


def tension(S, rho):
    """Weight hanging below each ring / its deformed circumference (N/m)."""
    skirt = SIGMA_G * (RC ** 2 - S ** 2) / (2.0 * np.maximum(rho, 1e-6))
    rim = SIGMA_G * (RC ** 2 - RT ** 2) / (2.0 * RT)
    return np.where(S <= RT, rim * S / RT, skirt)


def hem_curvature(th, rho):
    """Signed curvature of each skirt ring as a planar polar curve (1/m)."""
    dt = th[1] - th[0]
    d1 = (np.roll(rho, -1, axis=1) - np.roll(rho, 1, axis=1)) / (2 * dt)
    d2 = (np.roll(rho, -1, axis=1) - 2 * rho +
          np.roll(rho, 1, axis=1)) / dt ** 2
    return ((rho ** 2 + 2 * d1 ** 2 - rho * d2) /
            np.power(rho ** 2 + d1 ** 2, 1.5))


def cylinder(r, z0, z1, n=60):
    t = np.linspace(0, 2 * np.pi, n)
    x, y = r * np.cos(t), r * np.sin(t)
    return (np.vstack([x, x]), np.vstack([y, y]),
            np.vstack([np.full(n, z0), np.full(n, z1)]))


def disc(r, z, n=80):
    t = np.linspace(0, 2 * np.pi, n)
    return np.c_[r * np.cos(t), r * np.sin(t), np.full(n, z)]


def main():
    set_pub_style(base_font=8)
    s, th, S, TH, rho, z = drape()
    T = tension(S, rho)
    kap = hem_curvature(th, rho)

    # close the surface in theta for plotting
    def wrap(a):
        return np.hstack([a, a[:, :1]])
    X, Y, Z = wrap(rho) * np.cos(wrap(TH)), wrap(rho) * np.sin(wrap(TH)), wrap(z)
    Tw = wrap(T)

    cmap = get_cmap("cb_sequential_blue")
    norm = mpl.colors.Normalize(0.0, T.max())

    # lobes actually present in the hem ring, counted from the array
    hem = rho[-1]
    lobes = int(((hem > np.roll(hem, 1)) & (hem > np.roll(hem, -1))).sum())

    skirt = S > RT
    kap_sk = kap[skirt]

    it = np.arange(0, 61)
    gap = 46.0 * np.exp(-it / 6.2) * (1.0 + 0.10 *
                                      np.sin(it / 1.7 + 0.4)) + 3.0e-4
    gap[0] = 46.0
    tol = 1.0e-2
    below = np.flatnonzero(gap < tol)
    conv = int(it[below[0]]) if below.size else int(it[-1])

    fig = plt.figure(figsize=(10.4, 5.6))
    ax3 = fig.add_axes([-0.058, -0.020, 0.595, 0.930], projection="3d")
    axb = fig.add_axes([0.665, 0.605, 0.310, 0.295])
    axc = fig.add_axes([0.665, 0.125, 0.310, 0.295])

    # ---- (a) draped cloth ------------------------------------------------
    rmax = float(np.abs(rho).max())
    ax3.add_collection3d(Poly3DCollection(
        [disc(1.02 * rmax, ZF + 0.004)], facecolor="#b4bac2",
        edgecolor="none", alpha=0.42, zorder=1))
    cx, cy, cz = cylinder(0.135, ZF + 0.01, -0.02)
    ax3.plot_surface(cx, cy, cz, color="#a9aeb6", linewidth=0, shade=False,
                     zorder=2)
    ax3.add_collection3d(Poly3DCollection(
        [disc(0.34, ZF + 0.014)], facecolor="#9ba1a9", edgecolor="none",
        zorder=2))

    ax3.plot_surface(X, Y, Z, facecolors=cmap(norm(Tw)), rstride=1, cstride=1,
                     linewidth=0, shade=False, antialiased=True, zorder=4)
    ax3.plot_wireframe(X[::4, ::6], Y[::4, ::6], Z[::4, ::6],
                       color="#2b3b4f", lw=0.30, alpha=0.60, zorder=5)

    rim = np.argmin(np.abs(s - RT))
    ax3.plot(X[rim], Y[rim], Z[rim], color=ACCENT, lw=1.2, zorder=6)
    ax3.text2D(0.055, 0.175, "red ring = table rim,\npeak tension "
               f"{T[rim].max():.2f} N/m", transform=ax3.transAxes,
               color=ACCENT, fontsize=6.9, ha="left", va="bottom",
               linespacing=1.5)

    ax3.set_xlim(-0.86, 0.86)
    ax3.set_ylim(-0.86, 0.86)
    ax3.set_zlim(ZF - 0.02, 0.22)
    ax3.set_box_aspect((1.0, 1.0, 0.66))
    ax3.view_init(elev=ELEV, azim=AZIM)
    ax3.set_axis_off()
    ax3.patch.set_alpha(0.0)
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.set_visible(False)

    cax = fig.add_axes([0.512, 0.215, 0.0092, 0.400])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("membrane tension (N/m)", fontsize=7.0)
    cb.ax.tick_params(labelsize=6.6, width=0.7, length=2.2)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    # ---- (b) fold curvature ---------------------------------------------
    lo, hi = np.percentile(kap_sk, [0.5, 99.5])
    axb.hist(kap_sk, bins=36, range=(lo, hi), color="#7fa8c9",
             edgecolor="#2e5f8a", lw=0.4, zorder=3)
    axb.axvline(np.median(kap_sk), color=ACCENT, lw=1.1, ls="--", zorder=4)
    axb.set_xlim(lo, hi)
    axb.text(0.975, 0.94, f"median {np.median(kap_sk):.2f} m$^{{-1}}$",
             transform=axb.transAxes, color=ACCENT, fontsize=6.8, ha="right",
             va="top", zorder=5,
             bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#e2c3c8",
                       lw=0.5, alpha=0.94))
    axb.set_xlabel("hem curvature $\\kappa$ (m$^{-1}$)", fontsize=8.0)
    axb.set_ylabel("mesh vertices", fontsize=8.0)
    axb.set_title(f"Curvature over the {lobes} lobes seen in (a)",
                  fontsize=8.6, fontweight="bold", pad=3.5)
    axb.tick_params(labelsize=7.0, length=2.4)
    axb.grid(True, axis="y", lw=0.45, color="#e8e8e8")
    axb.set_axisbelow(True)
    add_panel_label(axb, "(b)", x=-0.155, y=1.045)

    # ---- (c) solver convergence -----------------------------------------
    axc.semilogy(it, gap, color="#2e5f8a", lw=1.6, zorder=3)
    axc.plot(it[::6], gap[::6], ls="none", marker="o", ms=2.8,
             mfc="white", mec="#2e5f8a", mew=0.8, zorder=4)
    axc.axhline(tol, color="#808080", lw=0.8, ls=(0, (4, 2.4)), zorder=2)
    axc.text(1.5, tol * 1.6, f"tolerance {tol:g} J", fontsize=6.6,
             color=MUTED, ha="left", va="bottom")
    axc.axvline(conv, color=ACCENT, lw=1.0, zorder=3)
    axc.annotate(f"reached at\niteration {conv}", xy=(conv, gap[conv]),
                 xytext=(conv - 5.0, 4.0), fontsize=6.8, color=ACCENT,
                 ha="right", va="center", linespacing=1.5,
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=ACCENT,
                                 shrinkA=1, shrinkB=3))
    axc.set_xlim(0, 60)
    axc.set_ylim(1.0e-4, 1.4e2)
    axc.set_xlabel("projective-dynamics iteration", fontsize=8.0)
    axc.set_ylabel("energy above minimum (J)", fontsize=8.0)
    axc.set_title("Energy still to be shed", fontsize=8.6, fontweight="bold",
                  pad=3.5)
    axc.tick_params(labelsize=7.0, length=2.4)
    axc.grid(True, lw=0.45, color="#e8e8e8")
    axc.set_axisbelow(True)
    add_panel_label(axc, "(c)", x=-0.155, y=1.045)

    savefig(fig, "1143_wm_phys_cloth_drape.png")


if __name__ == "__main__":
    main()

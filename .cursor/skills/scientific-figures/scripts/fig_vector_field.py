"""2x2 vector-field snapshots of a diffusing counter-rotating vortex pair.

Background: vorticity field (diverging contourf, shared colorbar on the
right).  Foreground: quiver arrows whose length *and* darkness encode the
local speed (shared horizontal colorbar at the bottom).  Vortex cores and
the hyperbolic stagnation (saddle) points of the lab-frame flow are
annotated.

Physics: each vortex is a Lamb-Oseen core, u_theta = G/(2 pi r) *
[1 - exp(-r^2 / rc^2)] with rc(t) = sqrt(rc0^2 + 4 nu t), so circulation is
conserved, the field is divergence-free by construction and the pair
self-advects through mutual induction (integrated explicitly in time).
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from scipy.ndimage import minimum_filter
from scipy.optimize import fsolve
from pubstyle import set_pub_style, savefig, add_panel_label

GAMMA = 1.5          # circulation magnitude [m^2/s]
D0 = 0.55            # half separation of the pair [m]
RC0 = 0.30           # initial core radius [m]
NU = 2.5e-3          # kinematic viscosity [m^2/s]
U_INF = -0.10        # uniform stream [m/s]
TIMES = [0.0, 4.0, 8.0, 12.0]
XLIM, YLIM = (-2.6, 2.6), (-1.75, 1.75)

CMAP_W = LinearSegmentedColormap.from_list(
    "vort", ["#2f5f92", "#7ca4c8", "#c6d8e7", "#f5f4f0",
             "#f0d4b8", "#d4996d", "#a3512f"])
CMAP_A = LinearSegmentedColormap.from_list(
    "arrow", ["#aeb4bc", "#767d87", "#3c434c", "#12161b"])


def rasterize(artist):
    """Rasterise ``artist`` without matplotlib 3.10's spurious warning.

    ``ContourSet.draw`` overrides ``Collection.draw`` but does not re-apply
    ``@allow_rasterization``, so ``Artist.set_rasterized`` looks for that
    decorator, fails to find it and warns that rasterization "will be
    ignored".  It is not ignored: the override delegates to the decorated
    ``Collection.draw``, and an SVG export of a 42-level contourf really does
    come out as a single embedded image (verified on 3.10.8).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Rasterization of", UserWarning)
        artist.set_rasterized(True)


def core_radius(t):
    return np.sqrt(RC0 ** 2 + 4.0 * NU * t)


def velocity(pos, gam, rc, x, y, skip=None):
    """Lab-frame velocity of the vortex ensemble plus the uniform stream."""
    u = np.full(np.shape(x), U_INF, dtype=float)
    v = np.zeros(np.shape(x), dtype=float)
    for k, ((xc, yc), g) in enumerate(zip(pos, gam)):
        if skip is not None and k == skip:
            continue
        dx, dy = x - xc, y - yc
        r2 = dx * dx + dy * dy + 1e-12
        f = g / (2.0 * np.pi * r2) * (1.0 - np.exp(-r2 / rc ** 2))
        u = u - f * dy
        v = v + f * dx
    return u, v


def vorticity(pos, gam, rc, x, y):
    w = np.zeros(np.shape(x), dtype=float)
    for (xc, yc), g in zip(pos, gam):
        r2 = (x - xc) ** 2 + (y - yc) ** 2
        w += g / (np.pi * rc ** 2) * np.exp(-r2 / rc ** 2)
    return w


def advect(t_end, dt=0.05):
    """March the two cores with their mutually induced velocity."""
    pos = np.array([[-0.15, D0], [-0.15, -D0]])
    gam = np.array([GAMMA, -GAMMA])
    n = max(int(round(t_end / dt)), 0)
    for i in range(n):
        rc = core_radius(i * dt)
        new = pos.copy()
        for k in range(2):
            u, v = velocity(pos, gam, rc, pos[k, 0], pos[k, 1], skip=k)
            new[k] += dt * np.array([u, v])
        pos = new
    return pos, gam


def stagnation_points(pos, gam, rc):
    """Locate and classify zeros of the lab-frame velocity field."""
    gx = np.linspace(XLIM[0], XLIM[1], 241)
    gy = np.linspace(YLIM[0], YLIM[1], 161)
    X, Y = np.meshgrid(gx, gy)
    u, v = velocity(pos, gam, rc, X, Y)
    s = np.hypot(u, v)
    loc = (s == minimum_filter(s, size=5)) & (s < 0.05)
    saddles, centers = [], []

    def f(p):
        return np.array(velocity(pos, gam, rc, p[0], p[1])).ravel()

    for x0, y0 in zip(X[loc], Y[loc]):
        sol, _, ok, _ = fsolve(f, [x0, y0], full_output=True)
        if ok != 1 or not (XLIM[0] < sol[0] < XLIM[1]
                           and YLIM[0] < sol[1] < YLIM[1]):
            continue
        if any(np.hypot(sol[0] - a, sol[1] - b) < 0.08
               for a, b in saddles + centers):
            continue
        h = 1e-4
        j = np.empty((2, 2))
        j[:, 0] = (f(sol + [h, 0]) - f(sol - [h, 0])) / (2 * h)
        j[:, 1] = (f(sol + [0, h]) - f(sol - [0, h])) / (2 * h)
        ev = np.linalg.eigvals(j)
        if np.all(np.abs(ev.imag) < 1e-8) and ev.real[0] * ev.real[1] < 0:
            saddles.append(tuple(sol))
        else:
            centers.append(tuple(sol))
    return saddles, centers


def main():
    set_pub_style(base_font=8)
    gx = np.linspace(XLIM[0], XLIM[1], 260)
    gy = np.linspace(YLIM[0], YLIM[1], 180)
    X, Y = np.meshgrid(gx, gy)
    qx = np.linspace(XLIM[0] + 0.09, XLIM[1] - 0.09, 21)
    qy = np.linspace(YLIM[0] + 0.09, YLIM[1] - 0.09, 14)
    QX, QY = np.meshgrid(qx, qy)

    frames = []
    for t in TIMES:
        pos, gam = advect(t)
        rc = core_radius(t)
        w = vorticity(pos, gam, rc, X, Y)
        qu, qv = velocity(pos, gam, rc, QX, QY)
        frames.append(dict(t=t, pos=pos, gam=gam, rc=rc, w=w,
                           qu=qu, qv=qv, sp=stagnation_points(pos, gam, rc)))

    wmax = max(np.abs(fr["w"]).max() for fr in frames)
    wmax = np.ceil(wmax * 2) / 2
    smax = max(np.hypot(fr["qu"], fr["qv"]).max() for fr in frames)
    smax = np.ceil(smax * 10) / 10
    wnorm = TwoSlopeNorm(vmin=-wmax, vcenter=0.0, vmax=wmax)
    snorm = Normalize(vmin=0.0, vmax=smax)
    # even edge count -> odd band count -> one neutral band straddling zero
    levels = np.linspace(-wmax, wmax, 42)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True,
                             sharey=True)
    cf = None
    for k, (ax, fr) in enumerate(zip(axes.ravel(), frames)):
        cf = ax.contourf(X, Y, fr["w"], levels=levels, cmap=CMAP_W,
                         norm=wnorm, extend="both")
        rasterize(cf)
        ax.contour(X, Y, fr["w"], levels=[-wmax * 0.35, wmax * 0.35],
                   colors="#55606c", linewidths=0.4, alpha=0.8)
        spd = np.hypot(fr["qu"], fr["qv"])
        # compress the drawn length (~|u|^0.55) so the slow free stream stays
        # readable next to the fast core arrows; shade still encodes |u|
        g = spd ** 0.55 / np.maximum(spd, 1e-9)
        ax.quiver(QX, QY, fr["qu"] * g, fr["qv"] * g, spd, cmap=CMAP_A,
                  norm=snorm, scale=13.0, width=0.0044, headwidth=3.6,
                  headlength=4.2, headaxislength=3.5, pivot="mid", zorder=3)

        for (xc, yc), g in zip(fr["pos"], fr["gam"]):
            ax.plot(xc, yc, marker="o", ms=4.2, mfc="white",
                    mec="#111418", mew=0.9, zorder=6)
            ax.plot(xc, yc, marker="+" if g > 0 else "_", ms=2.6,
                    color="#111418", mew=0.9, zorder=7)
        for xs, ys in fr["sp"][0]:
            ax.plot(xs, ys, marker="x", ms=5.0, color="#8c1d1d", mew=1.4,
                    zorder=6)

        ax.text(0.025, 0.955, f"$t$ = {fr['t']:.0f} s",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
                fontweight="bold", color="#111418", zorder=8,
                bbox=dict(boxstyle="round,pad=0.22", fc="white",
                          ec="#c9c9c9", lw=0.5, alpha=0.9))
        ax.text(0.975, 0.955,
                r"$r_c$ = %.2f m,  $\omega_{max}$ = %.1f s$^{-1}$"
                % (fr["rc"], fr["w"].max()),
                transform=ax.transAxes, ha="right", va="top", fontsize=6.4,
                color="#3a4049", zorder=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="none", alpha=0.72))
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_aspect("equal")
        ax.tick_params(length=2.2)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color("#333333")
            s.set_linewidth(0.8)
        add_panel_label(ax, f"({'abcd'[k]})", x=-0.02, y=1.015, size=9.5)
        if k >= 2:
            ax.set_xlabel("$x$ (m)")
        if k % 2 == 0:
            ax.set_ylabel("$y$ (m)")

    ax0 = axes[0, 0]
    p0 = frames[0]["pos"]
    ax0.annotate("vortex core\n" + r"$\Gamma = -1.5$ m$^2$ s$^{-1}$",
                 xy=(p0[1, 0], p0[1, 1]), xytext=(-2.42, -1.42),
                 fontsize=6.3, color="#111418", ha="left", va="center",
                 zorder=9,
                 arrowprops=dict(arrowstyle="-", lw=0.6, color="#111418",
                                 shrinkA=1, shrinkB=5))
    if frames[0]["sp"][0]:
        xs, ys = min(frames[0]["sp"][0])
        ax0.annotate("saddle\n(stagnation)", xy=(xs, ys),
                     xytext=(-2.42, 0.60), fontsize=6.3,
                     color="#8c1d1d", ha="left", va="bottom", zorder=9,
                     arrowprops=dict(arrowstyle="-", lw=0.6, color="#8c1d1d",
                                     shrinkA=1, shrinkB=3))

    fig.subplots_adjust(left=0.065, right=0.865, top=0.955, bottom=0.155,
                        wspace=0.06, hspace=0.11)
    cax = fig.add_axes([0.885, 0.155, 0.020, 0.80])
    cb = fig.colorbar(cf, cax=cax, extend="both", ticks=np.arange(-5, 5.1, 2.5))
    cb.set_label(r"vorticity  $\omega_z$  (s$^{-1}$)", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.8, width=0.7, length=2.2)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    sm = plt.cm.ScalarMappable(cmap=CMAP_A, norm=snorm)
    cax2 = fig.add_axes([0.20, 0.055, 0.42, 0.024])
    cb2 = fig.colorbar(sm, cax=cax2, orientation="horizontal")
    cb2.set_label(r"arrow shade / length: speed $|\mathbf{u}|$ (m s$^{-1}$)",
                  fontsize=7.2, labelpad=1.5)
    cb2.ax.tick_params(labelsize=6.8, width=0.7, length=2.2)
    cb2.outline.set_linewidth(0.6)
    cb2.outline.set_edgecolor("#333333")

    savefig(fig, "95_vector_field.png")


if __name__ == "__main__":
    main()

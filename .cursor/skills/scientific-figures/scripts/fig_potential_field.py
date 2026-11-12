"""Artificial potential fields: attraction, repulsion, and the local minimum.

Khatib's formulation is evaluated analytically on a 0.05 m lattice:
a quadratic-to-conic attractive well plus the classic inverse-distance
repulsive barrier with influence radius rho0.  Panel (d) integrates the true
negative gradient from several starts; the run launched on the goal-obstacle
axis converges to a genuine force-balance point and stalls, which is the
textbook failure mode of the method.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from pubstyle import set_pub_style, savefig, add_panel_label, add_colorbar

W, H = 20.0, 14.0
GOAL = np.array([18.0, 7.0])
OBST = [(10.0, 7.0, 2.20), (5.5, 2.2, 1.30), (14.5, 11.8, 1.60),
        (14.5, 2.2, 1.50), (5.5, 11.8, 1.50)]
K_ATT, D_STAR = 2.0, 4.0
K_REP, RHO0 = 8.0, 1.40

OBS_C = "#3a3a3a"
S_C = "#1f8a4c"
G_C = "#b23a48"


def rho(x, y):
    """Distance to the nearest obstacle *surface* (clipped away from zero)."""
    d = np.full(np.broadcast(x, y).shape, np.inf)
    for cx, cy, r in OBST:
        d = np.minimum(d, np.hypot(x - cx, y - cy) - r)
    return np.maximum(d, 1e-3)


def u_att(x, y):
    d = np.hypot(x - GOAL[0], y - GOAL[1])
    return np.where(d <= D_STAR, 0.5 * K_ATT * d ** 2,
                    K_ATT * D_STAR * d - 0.5 * K_ATT * D_STAR ** 2)


def u_rep(x, y):
    r = rho(x, y)
    return np.where(r <= RHO0, 0.5 * K_REP * (1.0 / r - 1.0 / RHO0) ** 2, 0.0)


def grad_u(p):
    """Analytic gradient of the total potential at a single point."""
    x, y = p
    d = np.hypot(x - GOAL[0], y - GOAL[1])
    if d <= D_STAR:
        g = K_ATT * np.array([x - GOAL[0], y - GOAL[1]])
    else:
        g = K_ATT * D_STAR * np.array([x - GOAL[0], y - GOAL[1]]) / max(d, 1e-9)
    for cx, cy, r in OBST:
        dv = np.array([x - cx, y - cy])
        dn = np.hypot(*dv)
        rr = max(dn - r, 1e-3)
        if rr <= RHO0:
            g += -K_REP * (1.0 / rr - 1.0 / RHO0) / (rr ** 2) * (dv / dn)
    return g


def descend(p0, alpha=0.035, max_step=0.10, n=4000, tol=0.20):
    p = np.array(p0, float)
    path = [p.copy()]
    for _ in range(n):
        g = grad_u(p)
        s = -alpha * g
        m = np.hypot(*s)
        if m > max_step:
            s *= max_step / m
        p = p + s
        path.append(p.copy())
        if np.hypot(*(p - GOAL)) < tol:
            return np.array(path), "reached"
        if m < 2e-4:
            return np.array(path), "trapped"
    return np.array(path), "stalled"


def draw_obstacles(ax, ec="white"):
    for cx, cy, r in OBST:
        ax.add_patch(Circle((cx, cy), r, fc=OBS_C, ec=ec, lw=0.7, zorder=6))


def frame(ax, row, col):
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color("#333333")
    ax.set_xticks(np.arange(0, W + 1, 5))
    ax.set_yticks(np.arange(0, H + 1, 5))
    ax.tick_params(length=2.4, labelsize=6.6)
    if row == 1:
        ax.set_xlabel("x (m)", fontsize=7.6, labelpad=1.5)
    else:
        ax.set_xticklabels([])
    if col == 0:
        ax.set_ylabel("y (m)", fontsize=7.6, labelpad=1.5)
    else:
        ax.set_yticklabels([])


def main():
    set_pub_style(base_font=8)
    g = 0.05
    xs = np.arange(0, W + g, g)
    ys = np.arange(0, H + g, g)
    X, Y = np.meshgrid(xs, ys)
    Ua, Ur = u_att(X, Y), u_rep(X, Y)
    Ut = Ua + Ur
    inside = np.zeros_like(X, bool)
    for cx, cy, r in OBST:
        inside |= (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
    Utm = np.where(inside, np.nan, Ut)
    ext = [0, W, 0, H]

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.1))
    fig.subplots_adjust(left=0.055, right=0.945, top=0.855, bottom=0.075,
                        wspace=0.13, hspace=0.13)

    # (a) attractive well ---------------------------------------------------
    ax = axes[0, 0]
    lv = np.linspace(0, Ua.max(), 22)
    cf = ax.contourf(X, Y, Ua, levels=lv, cmap="Blues_r")
    ax.contour(X, Y, Ua, levels=lv[::2], colors="#2c5f8a", linewidths=0.4,
               alpha=0.8)
    st = 26
    gx, gy = np.gradient(Ua, g, g)
    ax.quiver(X[::st, ::st], Y[::st, ::st], -gx[::st, ::st], -gy[::st, ::st],
              color="#123f63", scale=95, width=0.0032, headwidth=3.6,
              alpha=0.85, zorder=5)
    ax.plot(*GOAL, marker="*", ms=14, mfc=G_C, mec="white", mew=0.9,
            ls="none", zorder=8)
    frame(ax, 0, 0)
    ax.set_title("Attractive potential $U_{att}$", fontsize=8.6,
                 fontweight="bold", pad=3)
    add_colorbar(fig, cf, ax, label="$U_{att}$ (a.u.)", fraction=0.036,
                 pad=0.02, aspect=16)
    ax.text(0.018, 0.035, "quadratic within $d^*$ = %.0f m,\nconic beyond "
            "($k_{att}$ = %.1f)" % (D_STAR, K_ATT), transform=ax.transAxes,
            ha="left", va="bottom", fontsize=6.4, linespacing=1.4, zorder=10,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#bbbbbb",
                      lw=0.6, alpha=0.94))

    # (b) repulsive barrier -------------------------------------------------
    ax = axes[0, 1]
    Urp = np.where(inside, np.nan, np.clip(Ur, 0, 30))
    cf = ax.contourf(X, Y, Urp, levels=np.linspace(0, 30, 22), cmap="Reds",
                     extend="max")
    ax.contour(X, Y, Urp, levels=[0.5, 2, 6, 14, 26], colors="#7f2704",
               linewidths=0.4, alpha=0.85)
    for cx, cy, r in OBST:
        ax.add_patch(Circle((cx, cy), r + RHO0, fc="none", ec="#7f2704",
                            lw=0.7, ls=(0, (2.5, 1.8)), zorder=5))
    draw_obstacles(ax)
    frame(ax, 0, 1)
    ax.set_title("Repulsive potential $U_{rep}$", fontsize=8.6,
                 fontweight="bold", pad=3)
    add_colorbar(fig, cf, ax, label="$U_{rep}$ (a.u., clipped at 30)",
                 fraction=0.036, pad=0.02, aspect=16)
    ax.text(0.018, 0.035, "$U_{rep} = \\frac{1}{2}k_{rep}"
            "(\\frac{1}{\\rho}-\\frac{1}{\\rho_0})^2$,  "
            "$\\rho_0$ = %.1f m\ndashed ring = influence boundary" % RHO0,
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.4,
            linespacing=1.5, zorder=10,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#bbbbbb",
                      lw=0.6, alpha=0.94))

    # (c) total field + force arrows ---------------------------------------
    ax = axes[1, 0]
    top = float(np.nanmax(Ua))
    cf = ax.contourf(X, Y, np.clip(Utm, 0, top), levels=24, cmap="viridis")
    ax.contour(X, Y, np.clip(Utm, 0, top), levels=14, colors="white",
               linewidths=0.28, alpha=0.45)
    gx, gy = np.gradient(np.where(inside, np.nan, Ut), g, g)
    st = 22
    U_, V_ = -gx[::st, ::st], -gy[::st, ::st]
    mag = np.hypot(U_, V_)
    with np.errstate(invalid="ignore"):
        U_, V_ = U_ / np.maximum(mag, 1e-9), V_ / np.maximum(mag, 1e-9)
    ax.quiver(X[::st, ::st], Y[::st, ::st], U_, V_, color="white", scale=42,
              width=0.0030, headwidth=3.6, alpha=0.9, zorder=5)
    draw_obstacles(ax)
    ax.plot(*GOAL, marker="*", ms=14, mfc=G_C, mec="white", mew=0.9,
            ls="none", zorder=8)
    frame(ax, 1, 0)
    ax.set_title("Total potential and unit force field", fontsize=8.6,
                 fontweight="bold", pad=3)
    add_colorbar(fig, cf, ax, label="$U_{tot}$ (a.u., clipped)",
                 fraction=0.036,
                 pad=0.02, aspect=16)

    # (d) gradient descent + trap ------------------------------------------
    ax = axes[1, 1]
    ax.contour(X, Y, np.clip(Utm, 0, top), levels=20, colors="#9aa3ab",
               linewidths=0.32, alpha=0.9)
    draw_obstacles(ax)
    starts = [(1.2, 4.2), (1.5, 12.4), (1.5, 7.0)]
    cols = [S_C, "#0072b2", G_C]
    labs = []
    for p0, c in zip(starts, cols):
        path, status = descend(p0)
        labs.append(status)
        ax.plot(path[:, 0], path[:, 1], color="white", lw=2.8,
                solid_capstyle="round", zorder=6)
        ax.plot(path[:, 0], path[:, 1], color=c, lw=1.7,
                solid_capstyle="round", zorder=7)
        ax.plot(*p0, marker="^", ms=8, mfc=c, mec="white", mew=0.9,
                ls="none", zorder=9)
        if status != "reached":
            ax.plot(*path[-1], marker="X", ms=8, mfc=G_C, mec="white",
                    mew=0.9, ls="none", zorder=10)
            ax.annotate("local minimum: attraction and\nrepulsion cancel, "
                        "robot stalls",
                        xy=tuple(path[-1]), xytext=(10.2, 2.7), fontsize=6.5,
                        color=G_C, fontweight="bold", ha="center",
                        arrowprops=dict(arrowstyle="->", color=G_C, lw=0.9,
                                        shrinkB=5), zorder=11)
    ax.plot(*GOAL, marker="*", ms=14, mfc=G_C, mec="white", mew=0.9,
            ls="none", zorder=9)
    frame(ax, 1, 1)
    ax.set_title("Steepest-descent execution from three starts",
                 fontsize=8.6, fontweight="bold", pad=3)
    ax.text(0.018, 0.035, "start A  %s\nstart B  %s\nstart C  %s"
            % tuple(labs), transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.4, linespacing=1.4, zorder=12,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#bbbbbb",
                      lw=0.6, alpha=0.94))
    for (px, py), nm in zip(starts, "ABC"):
        ax.text(px + 0.45, py + 0.45, nm, fontsize=7.0, fontweight="bold",
                color="#1a1a1a", zorder=11)

    for k, ax in enumerate(axes.ravel()):
        add_panel_label(ax, "(%s)" % "abcd"[k], x=-0.055, y=1.005, size=10)

    handles = [Line2D([], [], marker="^", ls="none", ms=6.5, mfc=S_C,
                      mec="white", label="start pose"),
               Line2D([], [], marker="*", ls="none", ms=10, mfc=G_C,
                      mec="white", label="goal G"),
               Line2D([], [], marker="X", ls="none", ms=6.5, mfc=G_C,
                      mec="white", label="stalled at local minimum"),
               Line2D([], [], color="#555555", lw=1.8,
                      label="gradient-descent path"),
               Line2D([], [], marker="o", ls="none", ms=6, mfc=OBS_C,
                      mec="none", label="obstacle")]
    savefig(fig, "168_potential_field.png")


if __name__ == "__main__":
    main()

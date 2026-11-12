"""Streamline map of separated flow past a circular cylinder.

Panel (a): ``streamplot`` of the velocity field with line width and colour
encoding the local speed, dashed pressure-coefficient iso-lines on top, the
solid body silhouette, the reverse-flow (recirculation) region shaded, and
the separation points marked on the wall.

Panel (b): the wall pressure distribution extracted from the same field,
compared with the inviscid solution, with the separation angle and the wake
pressure plateau highlighted.

Physics: the stream function is built as the potential-flow cylinder
solution plus a counter-rotating wake vortex pair with Milne-Thomson image
vortices, so the cylinder surface stays a streamline.  Velocities are taken
as u = d(psi)/dy, v = -d(psi)/dx, hence the field is divergence-free, and
Cp = 1 - (|u| / U)^2 follows from Bernoulli.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pubstyle import set_pub_style, savefig, add_panel_label, annotate_box

A = 1.0              # cylinder radius [m]
U = 1.0              # free-stream speed [m/s]
GAM = 4.60           # wake vortex circulation magnitude [m^2/s]
XV, YV = 1.40, 0.95  # wake vortex position [m]
RC = 0.45            # vortex core radius [m]
XLIM, YLIM = (-2.3, 5.2), (-2.05, 2.05)
CMAP = "cividis"


def psi(x, y):
    """Stream function: uniform + doublet + wake pair + circle images."""
    r2 = x ** 2 + y ** 2
    s = U * y * (1.0 - A ** 2 / np.maximum(r2, 1e-9))
    zs = [(XV, YV, -GAM), (XV, -YV, GAM)]
    for xv, yv, g in zs:
        d2 = (x - xv) ** 2 + (y - yv) ** 2
        s += -g / (4.0 * np.pi) * np.log(d2 + RC ** 2)
        # Milne-Thomson image: -g at a^2 / conj(z_v), +g at the centre
        m = A ** 2 / (xv ** 2 + yv ** 2)
        xi, yi = xv * m, yv * m
        di2 = (x - xi) ** 2 + (y - yi) ** 2
        s += g / (4.0 * np.pi) * np.log(di2 + (0.06) ** 2)
        s += -g / (4.0 * np.pi) * np.log(r2 + (0.06) ** 2)
    return s


BBOX = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.82)


def field(nx=430, ny=250):
    x = np.linspace(XLIM[0], XLIM[1], nx)
    y = np.linspace(YLIM[0], YLIM[1], ny)
    X, Y = np.meshgrid(x, y)
    P = psi(X, Y)
    u = np.gradient(P, y, axis=0)
    v = -np.gradient(P, x, axis=1)
    inside = X ** 2 + Y ** 2 < A ** 2
    u[inside] = np.nan
    v[inside] = np.nan
    return x, y, X, Y, P, u, v, inside


def wall_profile(n=721, rr=1.03):
    """Sample the field just outside the wall; theta from front stagnation."""
    th = np.linspace(0.0, np.pi, n)          # 0 = upstream stagnation
    phi = np.pi - th                          # polar angle
    x, y = rr * A * np.cos(phi), rr * A * np.sin(phi)
    h = 1e-4
    u = (psi(x, y + h) - psi(x, y - h)) / (2 * h)
    v = -(psi(x + h, y) - psi(x - h, y)) / (2 * h)
    # tangential component along increasing theta
    tx, ty = np.sin(phi), -np.cos(phi)
    ut = u * tx + v * ty
    cp = 1.0 - (u ** 2 + v ** 2) / U ** 2
    return np.degrees(th), ut, cp


def main():
    set_pub_style(base_font=8)
    x, y, X, Y, P, u, v, inside = field()
    spd = np.hypot(u, v)
    smax = 2.1
    cp = 1.0 - (spd / U) ** 2

    th, ut, cpw = wall_profile()
    sgn = np.sign(ut)
    idx = np.where((sgn[:-1] > 0) & (sgn[1:] <= 0) & (th[:-1] > 40))[0]
    th_sep = th[idx[0]] if idx.size else 128.0
    xs = np.linspace(A * 1.01, 6.0, 800)
    h = 1e-4
    uax = (psi(xs, h) - psi(xs, -h)) / (2 * h)
    kk = np.where((uax[:-1] < 0) & (uax[1:] >= 0))[0]
    x_re = xs[kk[0]] if kk.size else np.nan

    fig = plt.figure(figsize=(7.2, 3.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.42, 1.0],
                          left=0.055, right=0.975, top=0.90, bottom=0.235,
                          wspace=0.26)
    ax = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[0, 1])

    # reverse-flow (recirculation) region
    rev = np.where(np.nan_to_num(u, nan=1.0) < 0.0, 1.0, 0.0)
    ax.contourf(X, Y, rev, levels=[0.5, 1.5], colors=["#f3dccb"],
                alpha=0.85, zorder=0)

    lw = 0.25 + 1.55 * np.clip(np.nan_to_num(spd) / smax, 0, 1)
    st = ax.streamplot(x, y, u, v, color=np.nan_to_num(spd), cmap=CMAP,
                       norm=plt.Normalize(0, smax), linewidth=lw,
                       density=(1.55, 1.05), arrowsize=0.6,
                       arrowstyle="-|>", zorder=3)
    st.lines.set_rasterized(True)

    cl = ax.contour(X, Y, np.ma.masked_where(inside, cp),
                    levels=[-1.2, -0.6, 0.0],
                    colors="#8c2f2f", linewidths=0.6,
                    linestyles="--", zorder=4, alpha=0.85)
    ax.clabel(cl, fmt="%.1f", fontsize=6.0, inline=True, inline_spacing=1)

    ax.add_patch(Circle((0, 0), A, facecolor="#3b424b", edgecolor="#12161b",
                        lw=0.9, zorder=6))
    ax.text(0, 0, "cylinder", ha="center", va="center", color="white",
            fontsize=6.2, fontweight="bold", zorder=7)

    for sgn_y in (1, -1):
        ph = np.radians(180.0 - th_sep)
        ax.plot(A * np.cos(ph), sgn_y * A * np.sin(ph), marker="o", ms=4.0,
                mfc="#e8b02a", mec="#12161b", mew=0.8, zorder=8)
    ph_s = np.radians(180.0 - th_sep)
    ax.annotate(r"separation point,  $\theta_s$ = %.0f$^\circ$" % th_sep,
                xy=(A * np.cos(ph_s), A * np.sin(ph_s)),
                xytext=(1.30, 1.78), fontsize=6.6, color="#12161b",
                ha="left", va="center", zorder=9, bbox=BBOX,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#12161b",
                                shrinkA=1, shrinkB=4))
    ax.annotate("recirculating wake\n(reverse flow, $u < 0$)",
                xy=(1.65, -0.42), xytext=(2.70, -1.62), fontsize=6.6,
                color="#8a4b1e", ha="left", va="center", zorder=9, bbox=BBOX,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#8a4b1e",
                                shrinkA=1, shrinkB=3))
    ax.plot([x_re], [0.0], marker="s", ms=3.4, mfc="#8a4b1e",
            mec="#12161b", mew=0.7, zorder=8)
    ax.annotate("reattachment\n" r"$L_b/a$ = %.2f" % (x_re - A),
                xy=(x_re, 0.0), xytext=(3.05, 0.95), fontsize=6.4,
                color="#8a4b1e", ha="left", va="center", zorder=9, bbox=BBOX,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#8a4b1e",
                                shrinkA=1, shrinkB=3))
    ax.annotate("", xy=(-1.55, -1.62), xytext=(-2.15, -1.62), zorder=9,
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#33404d"))
    ax.text(-2.15, -1.86, r"$U_\infty$ = 1.0 m s$^{-1}$", fontsize=6.6,
            color="#33404d", ha="left", va="center", zorder=9, bbox=BBOX)

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_xlabel("$x / a$")
    ax.set_ylabel("$y / a$")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#333333")
        s.set_linewidth(0.8)
    ax.tick_params(length=2.4)
    add_panel_label(ax, "(a)", x=-0.055, y=1.01, size=9.5)

    cax = fig.add_axes([0.075, 0.075, 0.30, 0.030])
    cb = fig.colorbar(st.lines, cax=cax, orientation="horizontal",
                      extend="max")
    cb.set_label(r"speed $|\mathbf{u}|$ (m s$^{-1}$)", fontsize=7.0,
                 labelpad=1.5)
    cb.ax.tick_params(labelsize=6.6, width=0.7, length=2.2)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    fig.text(0.395, 0.088, "dashed red: $C_p$ iso-lines\n"
                           "line width $\\propto$ speed",
             fontsize=6.4, color="#3a4049", ha="left", va="center")

    # ---- panel (b): wall pressure ----------------------------------------
    ideal = 1.0 - 4.0 * np.sin(np.radians(th)) ** 2
    axr.axvspan(th_sep, 180, color="#f3dccb", alpha=0.85, zorder=0)
    axr.plot(th, ideal, color="#9aa3ad", lw=1.1, ls=(0, (4, 2)),
             label="inviscid (attached)")
    axr.plot(th, cpw, color="#1f5c8b", lw=1.6, label="with wake bubble")
    axr.axvline(th_sep, color="#e8b02a", lw=1.0, zorder=1)
    axr.axhline(0.0, color="#c9c9c9", lw=0.6, zorder=0)
    axr.plot([th_sep], [np.interp(th_sep, th, cpw)], marker="o", ms=4.2,
             mfc="#e8b02a", mec="#12161b", mew=0.8, zorder=5)
    axr.text(th_sep - 3, 0.86, r"$\theta_s$", color="#a5761a", fontsize=8,
             ha="right", va="center", fontweight="bold")
    axr.text(172, -1.55, "separated region", fontsize=6.3, rotation=90,
             color="#8a4b1e", ha="center", va="center")
    kmin = int(np.argmin(cpw))
    axr.annotate(r"suction peak" "\n" r"$C_{p,min}$ = %.2f" % cpw[kmin],
                 xy=(th[kmin], cpw[kmin]), xytext=(0.045, 0.115),
                 textcoords="axes fraction", fontsize=6.3, color="#1f5c8b",
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color="#1f5c8b",
                                 shrinkA=1, shrinkB=3))
    axr.set_xlim(0, 180)
    axr.set_ylim(-3.1, 1.45)
    axr.set_xticks([0, 45, 90, 135, 180])
    axr.set_xlabel(r"wall angle $\theta$ from stagnation ($^\circ$)")
    axr.set_ylabel(r"pressure coefficient  $C_p$")
    axr.grid(True, lw=0.45, color="#e8e8e8")
    axr.legend(loc="upper center", fontsize=6.2, handlelength=1.9,
               borderpad=0.25, labelspacing=0.25,
               bbox_to_anchor=(0.52, 1.03))
    add_panel_label(axr, "(b)", x=-0.22, y=1.01, size=9.5)

    savefig(fig, "96_streamline_flow.png")


if __name__ == "__main__":
    main()

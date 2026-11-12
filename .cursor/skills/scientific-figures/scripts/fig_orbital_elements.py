"""The six Keplerian elements, seen four ways.

(a) is the orbit in its own plane, with the two shaded sectors swept in equal
times illustrating the constant areal velocity; (b) places that ellipse in
three dimensions and marks the inclination, the longitude of the ascending
node and the argument of periapsis; (c) follows the secular oscillation of
two elements over 200 kyr; (d) is the a-e plane of a small-body population,
where mean-motion resonances with the perturber carve visible gaps.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pubstyle import set_pub_style, savefig, add_panel_label

NAVY = "#1f3355"
INDIGO = "#5b4b8a"
GOLD = "#c9962f"
CYAN = "#3f7f95"
RUST = "#a8523a"
GREY = "#8b9099"

A_ORB, ECC = 1.25, 0.62
INC = np.radians(32.0)
BIG_OM = np.radians(55.0)
SM_OM = np.radians(40.0)


def kepler_E(M, e, n=60):
    E = M.copy()
    for _ in range(n):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E


def orbit_xy(a, e, E):
    return a * (np.cos(E) - e), a * np.sqrt(1 - e ** 2) * np.sin(E)


def rot_z(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_x(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def slerp(u, v, n=80):
    ang = np.arccos(np.clip(np.dot(u, v), -1, 1))
    t = np.linspace(0, 1, n)[:, None]
    if ang < 1e-8:
        return np.tile(u, (n, 1))
    return (np.sin((1 - t) * ang) * u + np.sin(t * ang) * v) / np.sin(ang)


def panel_a(ax):
    E = np.linspace(0, 2 * np.pi, 600)
    x, y = orbit_xy(A_ORB, ECC, E)
    b = A_ORB * np.sqrt(1 - ECC ** 2)
    cx = -A_ORB * ECC

    for m0, col in [(0.0, GOLD), (np.pi, CYAN)]:
        Ms = np.linspace(m0 - 0.42, m0 + 0.42, 60)
        Es = kepler_E(Ms, ECC)
        sx, sy = orbit_xy(A_ORB, ECC, Es)
        ax.fill(np.r_[0, sx], np.r_[0, sy], color=col, alpha=0.28, lw=0,
                zorder=2)

    ax.plot(x, y, color=NAVY, lw=1.8, zorder=4)
    ax.plot([cx], [0], marker="+", ms=6, mew=1.0, color=GREY, ls="none",
            zorder=5)
    ax.plot([0], [0], marker="o", ms=9, mfc="#f5c542", mec="#7a5c12",
            mew=0.8, ls="none", zorder=7)
    ax.plot([-2 * A_ORB * ECC], [0], marker="x", ms=5, mew=0.9, color=GREY,
            ls="none", zorder=5)
    ax.plot([cx, A_ORB * (1 - ECC)], [0, 0], color=RUST, lw=1.0,
            ls=(0, (5, 2.4)), zorder=5)
    ax.plot([cx, cx], [0, b], color=INDIGO, lw=1.0, ls=(0, (5, 2.4)),
            zorder=5)
    ax.text((cx + A_ORB * (1 - ECC)) / 2, 0.055, "$a$", fontsize=8.4,
            color=RUST, ha="center", va="bottom", fontweight="bold")
    ax.text(cx - 0.055, b / 2, "$b$", fontsize=8.4, color=INDIGO,
            ha="right", va="center", fontweight="bold")

    nu = np.radians(58.0)
    Ev = 2 * np.arctan2(np.sqrt(1 - ECC) * np.sin(nu / 2),
                        np.sqrt(1 + ECC) * np.cos(nu / 2))
    px, py = orbit_xy(A_ORB, ECC, np.array([Ev]))
    ax.plot([0, px[0]], [0, py[0]], color=NAVY, lw=1.0, zorder=6)
    ax.plot(px, py, marker="o", ms=6.5, mfc=NAVY, mec="white", mew=0.8,
            ls="none", zorder=8)
    arc = np.linspace(0, nu, 60)
    ax.plot(0.30 * np.cos(arc), 0.30 * np.sin(arc), color=NAVY, lw=0.8,
            zorder=6)
    ax.text(0.36 * np.cos(nu / 2), 0.36 * np.sin(nu / 2), r"$\nu$",
            fontsize=8.0, color=NAVY, ha="center", va="center")
    ax.text(px[0] / 2 + 0.06, py[0] / 2 + 0.05, "$r$", fontsize=8.0,
            color=NAVY, ha="left", va="bottom")

    for xx, lab, ha in [(A_ORB * (1 - ECC), "periapsis\n$q = %.2f$ AU"
                         % (A_ORB * (1 - ECC)), "right"),
                        (-A_ORB * (1 + ECC), "apoapsis\n$Q = %.2f$ AU"
                         % (A_ORB * (1 + ECC)), "left")]:
        ax.plot([xx], [0], marker="o", ms=4.2, mfc=RUST, mec="white",
                mew=0.6, ls="none", zorder=8)
        ax.text(xx + (0.06 if ha == "left" else -0.06), -0.075, lab,
                fontsize=6.8, color=RUST, ha=ha, va="top", linespacing=1.35)

    ax.text(0.02, 0.965, "equal areas swept in equal times\n"
            "$a = %.2f$ AU,  $e = %.2f$" % (A_ORB, ECC),
            transform=ax.transAxes, ha="left", va="top", fontsize=6.9,
            linespacing=1.5, zorder=9,
            bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#cccccc",
                      lw=0.55, alpha=0.94))
    ax.set_xlim(-2.15, 1.05)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ in the orbital plane (AU)", fontsize=8.0, labelpad=2)
    ax.set_ylabel("$y$ (AU)", fontsize=8.0, labelpad=2)
    ax.tick_params(labelsize=6.8, length=2.4)
    ax.grid(True, lw=0.4, color="#eceef1")
    ax.set_axisbelow(True)
    ax.set_title("Orbit in its own plane", fontsize=8.8, fontweight="bold",
                 pad=3)


def panel_b(ax):
    R = rot_z(BIG_OM) @ rot_x(INC) @ rot_z(SM_OM)
    E = np.linspace(0, 2 * np.pi, 400)
    ox, oy = orbit_xy(A_ORB, ECC, E)
    pts = R @ np.vstack([ox, oy, np.zeros_like(ox)])

    lim = 1.85
    quad = [[(-lim, -lim, 0), (lim, -lim, 0), (lim, lim, 0), (-lim, lim, 0)]]
    ax.add_collection3d(Poly3DCollection(quad, facecolor="#eef2f6",
                                         edgecolor="#c8d2dc", lw=0.6,
                                         alpha=0.75, zorder=1))
    for g in np.linspace(-lim, lim, 9):
        ax.plot([-lim, lim], [g, g], [0, 0], color="#dbe2e9", lw=0.4,
                zorder=1)
        ax.plot([g, g], [-lim, lim], [0, 0], color="#dbe2e9", lw=0.4,
                zorder=1)

    ax.plot(pts[0], pts[1], np.zeros_like(pts[0]), color=GREY, lw=0.7,
            ls=(0, (2, 2.4)), zorder=2)
    ax.plot(pts[0], pts[1], pts[2], color=NAVY, lw=2.0, zorder=6)
    ax.plot([0], [0], [0], marker="o", ms=8, mfc="#f5c542", mec="#7a5c12",
            mew=0.8, zorder=8)

    node = np.array([np.cos(BIG_OM), np.sin(BIG_OM), 0.0])
    ax.plot([-1.7 * node[0], 1.7 * node[0]], [-1.7 * node[1], 1.7 * node[1]],
            [0, 0], color=RUST, lw=1.2, ls=(0, (6, 2.4)), zorder=5)
    ax.plot([0, 1.72], [0, 0], [0, 0], color="#5c636b", lw=1.0, zorder=5)
    ax.text(1.80, -0.28, 0, "reference direction", fontsize=6.4,
            color="#5c636b", ha="center", va="center")
    ax.text(-1.80 * node[0], -1.80 * node[1], -0.08, "line of nodes",
            fontsize=6.4, color=RUST, ha="center", va="top")
    ax.text(1.62 * node[0], 1.62 * node[1], 0.06, "ascending\nnode",
            fontsize=6.4, color=RUST, ha="center", va="bottom",
            linespacing=1.3)

    peri = R @ np.array([1.0, 0.0, 0.0])
    normal_ref = np.array([-np.sin(BIG_OM), np.cos(BIG_OM), 0.0])
    q_hat = R @ np.array([0.0, 1.0, 0.0])
    q_hat = q_hat / np.linalg.norm(q_hat)

    arcs = [(slerp(np.array([1.0, 0, 0]), node) * 0.52, GOLD,
             r"$\Omega$", 0.63),
            (slerp(normal_ref, q_hat) * 0.72, CYAN, "$i$", 0.84),
            (slerp(node, peri) * 0.94, INDIGO, r"$\omega$", 1.06)]
    for arc, col, lab, rlab in arcs:
        ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="white", lw=2.8,
                zorder=7)
        ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=col, lw=1.4, zorder=8)
        m = arc[len(arc) // 2] / np.linalg.norm(arc[len(arc) // 2]) * rlab
        ax.text(m[0], m[1], m[2], lab, fontsize=8.6, color=col,
                fontweight="bold", ha="center", va="center")

    ax.plot([peri[0] * A_ORB * (1 - ECC)], [peri[1] * A_ORB * (1 - ECC)],
            [peri[2] * A_ORB * (1 - ECC)], marker="o", ms=5, mfc=RUST,
            mec="white", mew=0.6, zorder=8)
    ax.quiver(0, 0, 0, 0, 0, 1.25, color=INDIGO, lw=0.9, arrow_length_ratio=0.1,
              zorder=6)
    ax.text(0.06, 0.06, 1.42, "$\\hat{z}$", fontsize=7.4, color=INDIGO)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-0.75, 1.45)
    ax.set_box_aspect((1, 1, 0.60), zoom=1.42)
    ax.view_init(elev=24, azim=-62)
    ax.set_axis_off()
    ax.set_title("Orientation in space:  $i$, $\\Omega$, $\\omega$",
                 fontsize=8.8, fontweight="bold", pad=-2)


def panel_c(ax, rng):
    t = np.linspace(0, 200, 2000)
    e = 0.28 + 0.11 * np.sin(2 * np.pi * t / 74.0) \
        + 0.020 * np.sin(2 * np.pi * t / 21.0 + 1.1)
    i = 18.0 - 7.5 * np.sin(2 * np.pi * t / 74.0) \
        + 1.3 * np.sin(2 * np.pi * t / 17.0 + 0.4)

    ax.axvspan(18.5, 92.5, color=GOLD, alpha=0.10, lw=0, zorder=0)
    ax.plot(t, e, color=CYAN, lw=1.6, zorder=4, label="eccentricity $e$")
    ax.set_xlim(0, 200)
    ax.set_ylim(0.10, 0.46)
    ax.set_xlabel("time (kyr)", fontsize=8.0, labelpad=2)
    ax.set_ylabel("eccentricity  $e$", fontsize=8.0, labelpad=2, color=CYAN)
    ax.tick_params(labelsize=6.8, length=2.4)
    ax.tick_params(axis="y", colors=CYAN)
    ax.grid(True, lw=0.4, color="#eceef1")
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    ax2.plot(t, i, color=INDIGO, lw=1.6, ls=(0, (6, 2.4)), zorder=4,
             label="inclination $i$")
    ax2.set_ylim(6.0, 30.0)
    ax2.set_ylabel("inclination  $i$ (deg)", fontsize=8.0, labelpad=3,
                   color=INDIGO)
    ax2.tick_params(labelsize=6.8, length=2.4, colors=INDIGO)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(INDIGO)
    ax2.spines["right"].set_linewidth(0.8)
    ax2.spines["right"].set_visible(True)

    ax.annotate("", xy=(92.5, 0.435), xytext=(18.5, 0.435),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color="#8a6b1f"))
    ax.text(55.5, 0.442, "secular cycle  74 kyr", ha="center", va="bottom",
            fontsize=6.8, color="#8a6b1f")
    ax.text(0.985, 0.045, "$e$ and $i$ oscillate in antiphase at constant\n"
            "$\\sqrt{1-e^2}\\cos i$;  $\\dot{\\omega} = 3.7$ deg kyr$^{-1}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8,
            linespacing=1.5, zorder=9,
            bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#cccccc",
                      lw=0.55, alpha=0.94))
    ax.set_title("Secular evolution of two elements", fontsize=8.8,
                 fontweight="bold", pad=3)
    return ax2


def panel_d(fig, ax, rng):
    res = [(2.502, "3:1"), (2.825, "5:2"), (2.958, "7:3"), (3.279, "2:1")]
    n = 2600
    a = rng.uniform(2.05, 3.45, n)
    e = np.abs(rng.normal(0.10, 0.075, n)) + 0.012
    inc = np.abs(rng.normal(0.0, 7.5, n)) + rng.random(n) * 2.0
    keep = np.ones(n, bool)
    for ar, _ in res:
        w = 0.030 if ar > 3.0 else 0.024
        d = np.abs(a - ar)
        keep &= ~((d < w) | ((d < 2.6 * w) & (rng.random(n) < 0.72)))
    keep &= e < 0.36
    a, e, inc = a[keep], e[keep], inc[keep]

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "inc", ["#cfe0ee", "#7fa8c6", "#46739c", "#2b4a7a", "#3c3468"])
    sc = ax.scatter(a, e, c=inc, cmap=cmap, vmin=0, vmax=25, s=3.2,
                    linewidths=0, alpha=0.85, zorder=3)
    for ar, lab in res:
        ax.axvline(ar, color=RUST, lw=0.9, ls=(0, (4, 2.4)), zorder=4)
        ax.text(ar, 0.007, lab, ha="center", va="bottom", fontsize=6.8,
                color=RUST, fontweight="bold", zorder=8,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                          alpha=0.9))
    q_lines = [(1.5, 2.09), (2.0, 2.42)]
    aa = np.linspace(2.0, 3.5, 200)
    for q, a_lab in q_lines:
        ax.plot(aa, 1 - q / aa, color=GREY, lw=0.8, ls=(0, (1.6, 2.0)),
                zorder=2, clip_on=True)
        ax.text(a_lab, 1 - q / a_lab, "$q = %.1f$ AU" % q, fontsize=6.2,
                color="#6f767e", ha="center", va="center", rotation=40,
                rotation_mode="anchor", zorder=6, clip_on=True,
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none",
                          alpha=0.85))
    ax.set_xlim(2.0, 3.5)
    ax.set_ylim(0, 0.40)
    ax.set_xlabel("semi-major axis  $a$ (AU)", fontsize=8.0, labelpad=2)
    ax.set_ylabel("eccentricity  $e$", fontsize=8.0, labelpad=2)
    ax.tick_params(labelsize=6.8, length=2.4)
    ax.grid(True, lw=0.4, color="#eceef1")
    ax.set_axisbelow(True)
    ax.set_title("Small-body population in the $a$-$e$ plane", fontsize=8.8,
                 fontweight="bold", pad=3)
    ax.text(0.985, 0.962, "dashed red: mean-motion\nresonances with the "
            "perturber", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.6, color=RUST, style="italic", zorder=8,
            linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#e0d0c8",
                      lw=0.5, alpha=0.94))

    ax.add_patch(mpl.patches.Rectangle(
        (0.028, 0.735), 0.345, 0.235, transform=ax.transAxes, fc="white",
        ec="#dddddd", lw=0.5, alpha=0.92, zorder=6))
    cax = ax.inset_axes([0.062, 0.885, 0.275, 0.040])
    cax.set_zorder(7)
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("inclination (deg)", fontsize=6.4, labelpad=2)
    cb.ax.tick_params(labelsize=6.0, width=0.5, length=1.8)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#666666")
    return sc


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(149)

    fig = plt.figure(figsize=(10.4, 8.0))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.20, left=0.062,
                          right=0.975, top=0.872, bottom=0.070)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], projection="3d")
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_a(ax_a)
    panel_b(ax_b)
    ax_c2 = panel_c(ax_c, rng)
    panel_d(fig, ax_d, rng)

    # panel letters stamped in figure space so the offset is identical for
    # the 2D and the 3D panel alike
    for ax_, lab in [(ax_a, "(a)"), (ax_c, "(c)"), (ax_d, "(d)")]:
        p = ax_.get_position()
        add_panel_label(ax_, lab, x=-0.030 / p.width,
                        y=1.0 + 0.026 / p.height, size=10)
    p = ax_b.get_position()
    fig.text(p.x0 - 0.030, p.y1 + 0.026, "(b)", ha="left", va="bottom",
             fontweight="bold", fontsize=10)

    handles = [Line2D([], [], color=NAVY, lw=1.8, label="orbit"),
               Line2D([], [], marker="o", ls="none", ms=6, mfc="#f5c542",
                      mec="#7a5c12", mew=0.7, label="central body (focus)"),
               Line2D([], [], marker="o", ls="none", ms=4.5, mfc=RUST,
                      mec="white", mew=0.6, label="periapsis"),
               Line2D([], [], color=RUST, lw=1.2, ls=(0, (6, 2.4)),
                      label="line of nodes / resonance"),
               Line2D([], [], color=GOLD, lw=4, alpha=0.4,
                      label="equal-area sector"),
               Line2D([], [], color=CYAN, lw=1.6, label="eccentricity $e$"),
               Line2D([], [], color=INDIGO, lw=1.6, ls=(0, (6, 2.4)),
                      label="inclination $i$")]
    savefig(fig, "241_orbital_elements.png")


if __name__ == "__main__":
    main()

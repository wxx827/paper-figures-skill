"""Synthetic schlieren of a diamond aerofoil in a Mach 2.2 tunnel.

Main panel: a grey-scale density-gradient image of the whole test section on an
equal-aspect axes, because a shock angle is only readable if the picture is not
stretched.  Every straight feature in it is drawn at an angle solved from the
oblique-shock relation rather than sketched: the leading-edge shocks, the
Prandtl-Meyer fan at the shoulders, the trailing-edge recompression, and the
irregular reflection at each wall with its Mach stem, triple point, reflected
shock and slip line.  The reflection is irregular because the deflection the
reflected shock would have to produce exceeds the maximum the local Mach number
allows - the check is made in the code, not assumed.

The companion panels are the two quantitative readings.  A horizontal cut
through the flow gives the pressure and Mach number steps as it crosses each
feature, computed with the same relations that positioned them; and the
theta-beta-Mach chart shows both the free-stream operating point on its weak
branch and the post-shock point that falls outside the curve, which is exactly
why the wall reflection has to grow a Mach stem.

Illustrative synthetic imagery: gradients are laid down as analytic bands.
"""
import textwrap

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.optimize import brentq

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from pubstyle import set_pub_style, savefig, style_axes, add_footer
from figkit import axes_inches, panel_labels_inches, halo_text

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
RED = "#b23a48"
GOLD = "#c9a227"

FIG_W, FIG_H = 10.6, 6.4
MAIN_RECT = (0.50, 1.16, 5.36, 3.60)
AUX_T_RECT = (7.28, 3.52, 3.02, 1.86)
AUX_B_RECT = (7.28, 0.82, 3.02, 1.86)

GAM = 1.4
M_INF = 2.20            # free-stream Mach number (dimensionless)
P_INF = 24.0            # kPa, free-stream static pressure
HALF_ANGLE = 16.0       # deg, semi-vertex angle of the diamond section
CHORD = 90.0            # mm
X_LE = 62.0             # mm, leading edge station
H_WALL = 115.0          # mm, tunnel half-height
X_MAX = 343.0           # mm, window width
H_STEM = 26.0           # mm, Mach stem height at the wall
Y_CUT = 26.0            # mm, station of the pressure / Mach traverse

NX, NY = 1120, 752


def theta_of_beta(beta, m):
    """Flow deflection produced by a shock at wave angle ``beta`` (rad)."""
    s = m ** 2 * np.sin(beta) ** 2 - 1.0
    return np.arctan(2.0 / np.tan(beta) * s
                     / (m ** 2 * (GAM + np.cos(2.0 * beta)) + 2.0))


def theta_max(m):
    """Largest deflection an attached oblique shock can turn (rad)."""
    b = np.linspace(np.arcsin(1.0 / m) + 1e-6, np.pi / 2 - 1e-6, 4000)
    return float(theta_of_beta(b, m).max())


def beta_of_theta(theta, m):
    """Weak-solution wave angle for a given deflection (rad); None if none."""
    b_lo = np.arcsin(1.0 / m)
    b = np.linspace(b_lo + 1e-7, np.pi / 2 - 1e-7, 4000)
    t = theta_of_beta(b, m)
    k = int(np.argmax(t))
    if theta > t[k]:
        return None
    return brentq(lambda x: theta_of_beta(x, m) - theta, b_lo + 1e-9, b[k])


def across_shock(m, beta, theta):
    """(M2, p2/p1) across an oblique shock of wave angle ``beta``."""
    mn1 = m * np.sin(beta)
    mn2 = np.sqrt((1.0 + 0.5 * (GAM - 1.0) * mn1 ** 2)
                  / (GAM * mn1 ** 2 - 0.5 * (GAM - 1.0)))
    pr = 1.0 + 2.0 * GAM / (GAM + 1.0) * (mn1 ** 2 - 1.0)
    return mn2 / np.sin(beta - theta), pr


def nu(m):
    """Prandtl-Meyer function (rad)."""
    r = np.sqrt((GAM + 1.0) / (GAM - 1.0))
    s = np.sqrt(max(m ** 2 - 1.0, 0.0))
    return r * np.arctan(s / r) - np.arctan(s)


def m_of_nu(target):
    return brentq(lambda m: nu(m) - target, 1.0 + 1e-9, 12.0)


def expand(m, turn):
    """(M2, p2/p1) through an isentropic expansion of ``turn`` radians."""
    m2 = m_of_nu(nu(m) + turn)
    pr = ((1.0 + 0.5 * (GAM - 1.0) * m ** 2)
          / (1.0 + 0.5 * (GAM - 1.0) * m2 ** 2)) ** (GAM / (GAM - 1.0))
    return m2, pr


def section():
    """Diamond aerofoil outline in mm and its shoulder / trailing-edge points."""
    t = 0.5 * CHORD * np.tan(np.deg2rad(HALF_ANGLE))
    xs = X_LE + np.array([0.0, 0.5 * CHORD, CHORD, 0.5 * CHORD])
    ys = np.array([0.0, t, 0.0, -t])
    return np.column_stack([xs, ys]), (xs[1], ys[1]), (xs[2], 0.0), t


def band(f, xx, yy, p0, p1, amp, sigma=1.9, taper=7.0):
    """Add a Gaussian density-gradient band along one straight segment."""
    x0, y0 = p0
    dx, dy = p1[0] - x0, p1[1] - y0
    ln = float(np.hypot(dx, dy))
    if ln < 1e-9:
        return
    ux, uy = dx / ln, dy / ln
    px, py = xx - x0, yy - y0
    t = px * ux + py * uy
    d = -px * uy + py * ux
    w = (1.0 / (1.0 + np.exp(-t / taper))) * (1.0 / (1.0 + np.exp((t - ln)
                                                                  / taper)))
    f += amp * np.exp(-(d / sigma) ** 2) * w


def geometry():
    """Solve every wave angle and station used by the picture and the cuts."""
    th = np.deg2rad(HALF_ANGLE)
    b1 = beta_of_theta(th, M_INF)
    m2, pr2 = across_shock(M_INF, b1, th)
    m3, pr3 = expand(m2, 2.0 * th)
    b_te = beta_of_theta(th, m3)
    m4, pr4 = across_shock(m3, b_te, th)

    poly, shoulder, te, thick = section()
    y_tp = H_WALL - H_STEM
    x_tp = X_LE + y_tp / np.tan(b1)
    x_foot = X_LE + H_WALL / np.tan(b1)

    th_max2 = theta_max(m2)
    irregular = th > th_max2
    th_r = min(th, 0.96 * th_max2)
    b_r = beta_of_theta(th_r, m2)
    m5, pr5 = across_shock(m2, b_r, th_r)
    ang_r = th - b_r                                  # absolute, radians
    return dict(b1=b1, m2=m2, pr2=pr2, m3=m3, pr3=pr3, b_te=b_te, m4=m4,
                pr4=pr4, poly=poly, shoulder=shoulder, te=te, thick=thick,
                y_tp=y_tp, x_tp=x_tp, x_foot=x_foot, th_max2=th_max2,
                irregular=irregular, b_r=b_r, th_r=th_r, m5=m5, pr5=pr5,
                ang_r=ang_r, th=th)


def schlieren(g, seed=88117):
    """Grey-scale density-gradient image of the whole window."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, X_MAX, NX)
    y = np.linspace(-H_WALL, H_WALL, NY)
    xx, yy = np.meshgrid(x, y)
    f = np.zeros_like(xx)

    for sg in (+1.0, -1.0):
        tp = (g["x_tp"], sg * g["y_tp"])
        band(f, xx, yy, (X_LE, 0.0), tp, -1.00, 1.9)          # incident shock
        # Mach stem: a short, nearly normal shock meeting the wall square.
        stem = [(g["x_tp"] - 5.0 * s ** 2, sg * (g["y_tp"] + H_STEM * s))
                for s in np.linspace(0.0, 1.0, 7)]
        for a, b in zip(stem[:-1], stem[1:]):
            band(f, xx, yy, a, b, -1.35, 2.4, 3.0)
        # reflected shock and the slip line that leaves the triple point
        er = (tp[0] + 150.0 * np.cos(g["ang_r"]),
              tp[1] + sg * 150.0 * np.sin(g["ang_r"]))
        band(f, xx, yy, tp, er, -0.62, 2.1)
        band(f, xx, yy, tp, (tp[0] + 145.0, tp[1] - sg * 14.0), -0.16, 3.0)

        # Prandtl-Meyer fan at the shoulder
        sh = (g["shoulder"][0], sg * g["shoulder"][1])
        a0 = g["th"] + np.arcsin(1.0 / g["m2"])
        a1 = np.arcsin(1.0 / g["m3"])
        for a in np.linspace(a0, a1, 11):
            band(f, xx, yy, sh, (sh[0] + 260.0 * np.cos(a),
                                 sh[1] + sg * 260.0 * np.sin(a)),
                 +0.20, 2.3, 10.0)

        # trailing-edge recompression
        te = g["te"]
        band(f, xx, yy, te, (te[0] + 210.0 * np.cos(g["b_te"]),
                             te[1] + sg * 210.0 * np.sin(g["b_te"])),
             -0.68, 2.0)
        # wall boundary layer
        band(f, xx, yy, (0.0, sg * H_WALL), (X_MAX, sg * H_WALL), -0.34, 4.5,
             3.0)

    # viscous wake behind the model
    band(f, xx, yy, g["te"], (X_MAX, -6.0), -0.30, 4.0, 8.0)

    f = f + gaussian_filter(rng.normal(0.0, 0.55, f.shape), 1.4) * 0.11
    f = gaussian_filter(f, 1.0)
    vign = 1.0 - 0.10 * (((xx - X_MAX / 2) / (X_MAX / 2)) ** 2
                         + ((yy) / H_WALL) ** 2)
    return x, y, f * vign


def traverse(g, n=1400):
    """Pressure and Mach number along the horizontal cut at ``Y_CUT``.

    The waves are met in the order the cut reaches them and each jump is
    evaluated at the *local* Mach number, so the trace cannot claim a jump the
    upstream state could not produce.
    """
    x = np.linspace(0.0, X_MAX, n)
    sh = g["shoulder"]
    a0 = g["th"] + np.arcsin(1.0 / g["m2"])
    a1 = np.arcsin(1.0 / g["m3"])
    ev = dict(shock=X_LE + Y_CUT / np.tan(g["b1"]),
              fan=(sh[0] + (Y_CUT - sh[1]) / np.tan(a0),
                   sh[0] + (Y_CUT - sh[1]) / np.tan(a1)),
              te=g["te"][0] + Y_CUT / np.tan(g["b_te"]),
              refl=g["x_tp"] + (g["y_tp"] - Y_CUT) / np.tan(-g["ang_r"]))
    plan = sorted([(ev["shock"], ev["shock"], "shock", g["th"]),
                   (ev["fan"][0], ev["fan"][1], "fan", 2.0 * g["th"]),
                   (ev["te"], ev["te"], "shock", g["th"]),
                   (ev["refl"], ev["refl"], "shock", g["th_r"])],
                  key=lambda e: e[0])

    p = np.ones_like(x)
    m = np.full_like(x, M_INF)
    m_cur, p_cur = M_INF, 1.0
    m_seq = [M_INF]
    for xa, xb, kind, par in plan:
        if kind == "shock":
            b = beta_of_theta(par, m_cur)
            m_new, pr = across_shock(m_cur, b, par)
            s = 0.5 * (1.0 + np.tanh((x - xa) / 1.1))
        else:
            m_new, pr = expand(m_cur, par)
            s = np.clip((x - xa) / (xb - xa), 0.0, 1.0)
        p = p + p_cur * (pr - 1.0) * s
        m = m + (m_new - m_cur) * s
        m_cur, p_cur = m_new, p_cur * pr
        m_seq.append(m_new)

    zones = [(0.0 if k == 0 else plan[k - 1][1],
              X_MAX if k == len(plan) else plan[k][0], m_seq[k])
             for k in range(len(plan) + 1)]
    return (x, gaussian_filter1d(p, 3.0) * P_INF, gaussian_filter1d(m, 3.0),
            ev, zones)


def check_layout(fig, foot, axes):
    """Warn if the footnote box touches an axis label, tick row or title."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for a in axes:
        ab = a.get_window_extent()
        arts = [a.xaxis.label, a.yaxis.label, a.title]
        arts += [t for t in a.get_xticklabels()
                 if ab.x0 - 3.0 <= t.get_window_extent().x0 <= ab.x1 + 3.0]
        for art in arts:
            if art.get_text() and fb.overlaps(art.get_window_extent()):
                print("WARNING: footnote overlaps:", art.get_text()[:44])


def lead(ax, text, xy, xytext, color="white"):
    """White leader-line label for use on the grey-scale image."""
    t = ax.annotate(text, xy=xy, xytext=xytext, fontsize=6.4, color=color,
                    ha="left" if xytext[0] >= xy[0] else "right",
                    va="center", linespacing=1.40, zorder=12,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.75,
                                    shrinkA=1, shrinkB=2))
    t.set_path_effects([pe.withStroke(linewidth=1.9, foreground="#1a1a1a")])
    t.arrow_patch.set_path_effects(
        [pe.withStroke(linewidth=1.8, foreground="#1a1a1a")])
    return t


def main():
    set_pub_style(base_font=8)
    g = geometry()
    x_img, y_img, img = schlieren(g)
    xt, pt, mt, ev, zones = traverse(g)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_c = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax_c, ax_b):
        a.set_axisbelow(True)
        a.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)
        style_axes(a)

    # ---- main: the schlieren window ----------------------------------------
    ax.set_aspect("equal")
    lim = float(np.percentile(np.abs(img), 99.0))
    ax.imshow(img, extent=(0.0, X_MAX, -H_WALL, H_WALL), origin="lower",
              cmap="gray", vmin=-lim, vmax=lim, interpolation="bilinear",
              zorder=2, rasterized=True)
    ax.add_patch(Polygon(g["poly"], closed=True, facecolor="#0d0d0d",
                         edgecolor="#e8e8e8", lw=0.8, zorder=6))
    ax.axhline(H_WALL, color="#dddddd", lw=1.0, zorder=7)
    ax.axhline(-H_WALL, color="#dddddd", lw=1.0, zorder=7)
    halo_text(ax, X_MAX - 6.0, Y_CUT - 6.0,
              f"traverse of panel (b), y = {Y_CUT:.0f} mm", ha="right",
              va="top", fontsize=6.3, color=GOLD, halo="#1a1a1a",
              halo_lw=2.0, zorder=9)

    b1d = np.rad2deg(g["b1"])
    lead(ax, f"leading-edge oblique shock\n$\\beta$ = {b1d:.1f} deg for "
         f"$\\theta$ = {HALF_ANGLE:.0f} deg",
         (X_LE + 46.0, -46.0 * np.tan(g["b1"])), (150.0, -96.0))
    lead(ax, "Prandtl-Meyer expansion fan\nturning "
         f"{2 * HALF_ANGLE:.0f} deg at the shoulder", (136.0, 42.0),
         (120.0, 68.0))
    lead(ax, f"Mach stem, {H_STEM:.0f} mm tall",
         (g["x_tp"] - 3.0, g["y_tp"] + 13.0), (g["x_tp"] - 40.0, H_WALL - 14.0))
    lead(ax, "triple point", (g["x_tp"] + 1.0, g["y_tp"] - 1.0),
         (g["x_tp"] + 12.0, g["y_tp"] - 30.0))
    lead(ax, "reflected shock", (g["x_tp"] + 52.0 * np.cos(g["ang_r"]),
                                 g["y_tp"] + 52.0 * np.sin(g["ang_r"])),
         (g["x_tp"] + 96.0, g["y_tp"] + 4.0))
    lead(ax, "slip line", (g["x_tp"] + 92.0, g["y_tp"] - 9.0),
         (g["x_tp"] + 116.0, g["y_tp"] - 40.0))
    lead(ax, "trailing-edge\nrecompression",
         (g["te"][0] + 40.0, -40.0 * np.tan(g["b_te"])), (252.0, -60.0))

    for x_a, x_b, m_z in zones:
        if x_b - x_a < 22.0:
            continue
        halo_text(ax, 0.5 * (x_a + x_b), Y_CUT + 7.0, f"Ma = {m_z:.2f}",
                  ha="center", va="bottom", fontsize=6.6, color="white",
                  halo="#1a1a1a", halo_lw=2.2, zorder=11)
        ax.plot([x_a, x_b], [Y_CUT, Y_CUT], color=GOLD, lw=2.2, alpha=0.9,
                solid_capstyle="butt", zorder=9)

    ax.set_xlim(0.0, X_MAX)
    ax.set_ylim(-H_WALL, H_WALL)
    ax.set_xticks(np.arange(0.0, 341.0, 80.0))
    ax.set_yticks(np.arange(-100.0, 101.0, 50.0))
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel("streamwise station x (mm)", fontsize=8.4, labelpad=1.5)
    ax.set_ylabel("y (mm)", fontsize=8.4)
    ax.set_title(f"Schlieren of a {2 * HALF_ANGLE:.0f} deg diamond section at "
                 f"$M_\\infty$ = {M_INF:.2f}, $p_\\infty$ = {P_INF:.0f} kPa",
                 fontsize=9.0, fontweight="bold", pad=5.0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # ---- aux (b): the traverse ---------------------------------------------
    ax_c.plot(xt, pt, color=RED, lw=1.8, zorder=6)
    ax_c.set_xlim(0.0, X_MAX)
    ax_c.set_ylim(0.0, pt.max() * 1.42)
    ax_c.tick_params(labelsize=6.8, axis="x")
    ax_c.tick_params(labelsize=6.8, axis="y", labelcolor=RED)
    ax_c.set_xlabel("streamwise station x (mm)", fontsize=7.5, labelpad=1.5)
    ax_c.set_ylabel("static pressure p (kPa)", fontsize=7.5, color=RED)
    ax_c.set_title(f"Steps along the cut at y = {Y_CUT:.0f} mm", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_m = ax_c.twinx()
    ax_m.plot(xt, mt, color=BLUE, lw=1.8, ls=(0, (4.4, 2.0)), zorder=5)
    ax_m.axhline(1.0, color=FAINT, lw=0.7, zorder=1)
    ax_m.set_ylim(0.0, mt.max() * 1.30)
    ax_m.set_ylabel("Mach number Ma (dimensionless)", fontsize=7.5,
                    color=BLUE)
    ax_m.tick_params(labelsize=6.8, labelcolor=BLUE)
    ax_m.spines["top"].set_visible(False)
    ax_m.grid(False)
    for xc, lab in ((ev["shock"], "LE shock"), (ev["te"], "TE shock"),
                    (ev["refl"], "reflected")):
        ax_c.axvline(xc, color=FAINT, lw=0.7, ls=(0, (2.4, 2.0)), zorder=2)
        ax_c.text(xc, 0.985, lab, transform=ax_c.get_xaxis_transform(),
                  rotation=90, ha="right", va="top", fontsize=6.0,
                  color=MUTED)
    ax_c.axvspan(*ev["fan"], color="#f3efe2", zorder=1)
    ax_c.text(np.mean(ev["fan"]), 0.030, "fan",
              transform=ax_c.get_xaxis_transform(), ha="center", va="bottom",
              fontsize=6.0, color=MUTED)
    ax_c.legend(handles=[Line2D([], [], color=RED, lw=1.8, label="p"),
                         Line2D([], [], color=BLUE, lw=1.8,
                                ls=(0, (4.4, 2.0)), label="Ma")],
                loc="upper left", frameon=False, fontsize=6.5,
                handlelength=1.7, labelspacing=0.26, borderaxespad=0.30)

    # ---- aux (c): theta-beta-Mach ------------------------------------------
    for m, col in ((1.49, GOLD), (1.8, BLUE), (M_INF, GREEN), (3.5, RED)):
        b = np.linspace(np.arcsin(1.0 / m) + 1e-6, np.pi / 2 - 1e-6, 1200)
        t = np.rad2deg(theta_of_beta(b, m))
        ax_b.plot(t, np.rad2deg(b), color=col, lw=1.6, zorder=5,
                  label=f"Ma = {m:.2f}")
        k = int(np.argmax(t))
        ax_b.plot([t[k]], [np.rad2deg(b[k])], marker="o", ms=2.8, color=col,
                  zorder=6)
    ax_b.plot([np.rad2deg(g["th"])], [np.rad2deg(g["b1"])], marker="o",
              ms=5.4, mfc="white", mec=GREEN, mew=1.5, zorder=9)
    ax_b.annotate(f"free stream: {HALF_ANGLE:.0f} deg is on the weak branch",
                  xy=(np.rad2deg(g["th"]), np.rad2deg(g["b1"])),
                  xytext=(19.0, 30.0), fontsize=6.3, color=GREEN, ha="left",
                  va="center", zorder=10,
                  arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=0.8,
                                  shrinkA=2, shrinkB=4))
    ax_b.axvline(np.rad2deg(g["th"]), color=FAINT, lw=0.8, ls=(0, (2.6, 2.0)),
                 zorder=2)
    ax_b.plot([np.rad2deg(g["th_max2"])], [0.0], marker="^", ms=5.0,
              color=GOLD, clip_on=False, zorder=9)
    ax_b.annotate(f"Ma = {g['m2']:.2f} can turn only "
                  f"{np.rad2deg(g['th_max2']):.1f} deg,\nso the wall "
                  "reflection must be irregular",
                  xy=(np.rad2deg(g["th_max2"]), 52.0), xytext=(17.5, 89.0),
                  fontsize=6.3, color=GOLD, ha="left", va="center",
                  linespacing=1.40, zorder=10,
                  arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=0.8,
                                  shrinkA=2, shrinkB=3))
    ax_b.set_xlim(0.0, 42.0)
    ax_b.set_ylim(0.0, 97.0)
    ax_b.tick_params(labelsize=6.8)
    ax_b.set_xlabel("flow deflection $\\theta$ (deg)", fontsize=7.5,
                    labelpad=1.5)
    ax_b.set_ylabel("shock wave angle $\\beta$ (deg)", fontsize=7.5)
    ax_b.set_title("Why the reflection grows a Mach stem", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_b.legend(loc="lower right", frameon=False, fontsize=6.3,
                handlelength=1.4, labelspacing=0.24, borderaxespad=0.30)

    panel_labels_inches(fig, [ax, ax_c, ax_b], dx=-0.50, dy=0.10)
    foot = add_footer(fig, "Synthetic data, figure 935")
    check_layout(fig, foot, [ax, ax_c, ax_b])
    savefig(fig, "935_wt_schlieren_shock.png")


if __name__ == "__main__":
    main()

"""Figure 1146 - Contact field of a braking tyre: pressure, friction cone, history.

Panel (a) is the contact patch of a rolling tyre seen from below: normal
pressure as a heat map, tangential traction as a vector field on top of it.
The traction follows the brush model - shear demand grows linearly with the
distance a tread element has travelled since entering the patch, and is capped
at mu times the local pressure - which is what produces the adhesion region at
the leading edge and the sliding region trailing it.  The dashed contour is
where the cap becomes active, drawn from the same arrays.

Panel (b) puts the resultant of that field into the friction cone; panel (c)
is the force history of the braking event.  The three panels are locked
together numerically: the resultant integrated over panel (a) is the vector
drawn in (b), and it is also the value the curves in (c) take at the marked
snapshot instant.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Wedge

from pubstyle import set_pub_style, savefig, get_cmap, add_panel_label
from figkit import safe_colorbar

A, B, NP = 105.0, 82.0, 3.2       # patch half-length / half-width (mm), shape
MU = 0.55
P0 = 205.0                        # peak tread pressure (kPa)
SLIPDIR = np.array([-1.0, -0.30])  # braking with a small slip angle
INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"
FN_C, FT_C, ACCENT = "#3b5b92", "#2e7d5b", "#b23a48"


def contact_field(n=340):
    x = np.linspace(-135.0, 135.0, n)
    y = np.linspace(-104.0, 104.0, int(round(n * 104.0 / 135.0)))
    Xg, Yg = np.meshgrid(x, y)

    shape = 1.0 - (np.abs(Xg / A) ** NP + np.abs(Yg / B) ** NP)
    taper = np.clip(shape, 0.0, 1.0) ** 0.45
    grooves = 1.0 - 0.95 * (np.exp(-((Yg - 30.0) / 6.0) ** 2) +
                            np.exp(-((Yg + 30.0) / 6.0) ** 2))
    ribs = (0.72 + 0.62 * np.exp(-((np.abs(Yg) - 54.0) / 27.0) ** 2))
    p = P0 * taper * grooves * ribs * np.exp(-((Xg + 22.0) / 78.0) ** 2)

    inside = shape > 0.0
    demand = 0.40 * (A - Xg)                  # brush model: grows with age
    cap = MU * p
    tau = np.where(inside, np.minimum(demand, cap), 0.0)
    slip = inside & (demand > cap)
    return x, y, Xg, Yg, np.where(inside, p, np.nan), tau, slip, inside


def main():
    set_pub_style(base_font=8.5)
    x, y, Xg, Yg, p, tau, slip, inside = contact_field()
    da = (x[1] - x[0]) * (y[1] - y[0])         # mm^2 per cell
    pv = np.nan_to_num(p)

    fn = 1.0e-3 * pv.sum() * da                # kPa*mm^2 -> N
    d = SLIPDIR / np.hypot(*SLIPDIR)
    fx = 1.0e-3 * (tau * d[0]).sum() * da
    fy = 1.0e-3 * (tau * d[1]).sum() * da
    ft = float(np.hypot(fx, fy))
    cop = (float((pv * Xg).sum() / pv.sum()), float((pv * Yg).sum() / pv.sum()))
    slip_frac = 100.0 * slip.sum() / inside.sum()

    # ---- force history, pinned to the snapshot above ---------------------
    t = np.linspace(0.0, 1.20, 481)
    tstar = 0.34
    fn_t = fn * (1.0 + 0.42 * (1.0 - np.exp(-t / 0.17))) / \
        (1.0 + 0.42 * (1.0 - np.exp(-tstar / 0.17)))
    dem_t = (1.0 - np.exp(-t / 0.13)) * (1.0 + 1.6 * t)
    ft_t = np.minimum(dem_t * ft / ((1.0 - np.exp(-tstar / 0.13)) *
                                    (1.0 + 1.6 * tstar)), MU * fn_t)
    locked = np.flatnonzero(ft_t >= MU * fn_t - 1e-9)
    t_lock = float(t[locked[0]]) if locked.size else float(t[-1])
    k = int(np.argmin(np.abs(t - tstar)))

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.04],
                          left=0.055, right=0.905, top=0.845, bottom=0.098,
                          wspace=0.62, hspace=0.42)
    ax = fig.add_subplot(gs[:, 0:2])
    axb = fig.add_subplot(gs[0, 2])
    axc = fig.add_subplot(gs[1, 2])

    # ---- (a) pressure + traction ----------------------------------------
    cmap = get_cmap("cb_sequential_blue")
    norm = mpl.colors.Normalize(0.0, np.nanmax(p))
    ax.set_facecolor("#f3f4f5")
    im = ax.imshow(np.ma.masked_invalid(p), origin="lower", cmap=cmap,
                   norm=norm, extent=(x[0], x[-1], y[0], y[-1]),
                   interpolation="bilinear", zorder=1)
    ax.contour(Xg, Yg, inside.astype(float), levels=[0.5], colors="#20374f",
               linewidths=1.1, zorder=4)
    ax.contour(Xg, Yg, slip.astype(float), levels=[0.5], colors=[ACCENT],
               linewidths=1.0, linestyles=[(0, (4.0, 2.2))], zorder=5)

    st = 19
    m = inside[::st, ::st] & (tau[::st, ::st] > 0.06 * tau.max())
    q = ax.quiver(Xg[::st, ::st][m], Yg[::st, ::st][m],
                  (tau[::st, ::st] * d[0])[m], (tau[::st, ::st] * d[1])[m],
                  color="#152233", scale=1.15e3, width=0.0036,
                  headwidth=3.6, headlength=4.2, zorder=6)
    ax.quiverkey(q, 0.845, 0.055, 40.0, "40 kPa shear", labelpos="E",
                 fontproperties=dict(size=6.8), color="#152233",
                 labelcolor=SUB, coordinates="axes")

    ax.plot(*cop, marker="o", ms=6.0, mfc="white", mec=ACCENT, mew=1.4,
            zorder=8)
    ax.annotate(f"centre of pressure ({cop[0]:.0f}, {cop[1]:.0f}) mm",
                xy=cop, xytext=(cop[0] - 12.0, cop[1] + 30.0), fontsize=6.9,
                color=ACCENT, ha="center", va="bottom", zorder=9,
                bbox=dict(boxstyle="round,pad=0.24", fc="white",
                          ec="#e2c3c8", lw=0.5, alpha=0.94),
                arrowprops=dict(arrowstyle="-", lw=0.7, color=ACCENT,
                                shrinkA=2, shrinkB=5))
    ax.annotate("", xy=(126.0, -95.0), xytext=(72.0, -95.0), zorder=9,
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color=INK,
                                mutation_scale=8))
    ax.text(99.0, -89.0, "rolling", fontsize=6.9, color=INK, ha="center",
            va="bottom")
    ax.text(-129.0, 95.0, f"adhesion {100 - slip_frac:.0f} %  |  sliding "
            f"{slip_frac:.0f} % of the patch", fontsize=7.0, color=SUB,
            ha="left", va="top")

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    ax.set_aspect("equal")
    ax.set_xlabel("longitudinal position (mm)")
    ax.set_ylabel("lateral position (mm)")
    ax.set_title("Sliding sets in wherever pressure is low: the grooves and "
                 "the trailing edge", fontsize=8.8, fontweight="bold", pad=4.0)
    ax.tick_params(length=2.6)
    add_panel_label(ax, "(a)", x=-0.075, y=1.015)
    safe_colorbar(fig, im, ax, label="normal pressure $p$ (kPa)", width=0.13,
                  pad=0.10, labelsize=7.2)

    # ---- (b) friction cone ----------------------------------------------
    phi = np.degrees(np.arctan(MU))
    xl, yl = 2.0 * MU * fn, 1.55 * fn
    axb.add_patch(Wedge((0.0, 0.0), 3.0 * yl, 90.0 - phi, 90.0 + phi,
                        facecolor="#dce6f0", edgecolor="none", zorder=1))
    for sgn in (-1.0, 1.0):
        axb.plot([0.0, sgn * MU * yl], [0.0, yl], color="#5b7ba8", lw=1.0,
                 zorder=2)
    axb.axvline(0.0, color="#8a8a8a", lw=0.8, ls=(0, (4, 2.4)), zorder=2)
    axb.axhline(0.0, color="#3a3a3a", lw=1.2, zorder=3)
    axb.annotate("", xy=(MU * fn, fn), xytext=(0.0, 0.0), zorder=5,
                 arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#5b7ba8",
                                 ls=(0, (3.4, 2.0)), mutation_scale=8))
    axb.annotate("", xy=(ft, fn), xytext=(0.0, 0.0), zorder=6,
                 arrowprops=dict(arrowstyle="-|>", lw=1.9, color=ACCENT,
                                 mutation_scale=10))
    # The cone edges and the dashed centre line run behind all four labels, so
    # each one carries a plate; without it the lines read as strike-throughs.
    plate = dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.88)
    axb.text(ft * 0.42, fn * 0.60, f"$F$ = {np.hypot(ft, fn):.0f} N",
             fontsize=7.2, color=ACCENT, ha="right", va="center", zorder=7,
             bbox=plate)
    axb.text(-0.96 * xl, 0.97 * yl, f"$\\mu$ = {MU:.2f}\n"
             f"half-angle {phi:.1f}$^\\circ$", fontsize=6.9, color=SUB,
             ha="left", va="top", linespacing=1.45, zorder=7, bbox=plate)
    # High on the right: the label is wide enough that down at the axis it
    # would reach back across the resultant arrow and its plate would dim it.
    axb.text(0.96 * xl, 0.88 * yl, f"cone edge $\\mu F_n$ = {MU * fn:.0f} N",
             fontsize=6.8, color="#476c96", ha="right", va="bottom", zorder=7,
             bbox=plate)
    axb.text(-0.96 * xl, 0.055 * yl,
             f"$F_t$ / $\\mu F_n$ = {ft / (MU * fn):.2f}", fontsize=7.2,
             color=INK, ha="left", va="bottom", zorder=7, bbox=plate)
    axb.set_xlim(-xl, xl)
    axb.set_ylim(-0.055 * yl, yl)
    axb.set_xticks([-2000, 0, 2000])
    axb.set_aspect("equal")
    axb.set_xlabel("tangential force $F_t$ (N)", fontsize=8.0)
    axb.set_ylabel("normal force $F_n$ (N)", fontsize=8.0)
    axb.set_title("Resultant of (a) sits inside the cone", fontsize=8.6,
                  fontweight="bold", pad=4.0)
    axb.tick_params(labelsize=7.0, length=2.4)
    add_panel_label(axb, "(b)", x=-0.235, y=1.02)

    # ---- (c) force history ----------------------------------------------
    axc.fill_between(t, 0.0, MU * fn_t, where=t >= t_lock, color=ACCENT,
                     alpha=0.12, lw=0, zorder=1)
    axc.plot(t, fn_t, color=FN_C, lw=1.6, zorder=4, label="normal $F_n$")
    axc.plot(t, MU * fn_t, color="#8a8a8a", lw=1.0, ls=(0, (4, 2.4)),
             zorder=3, label="friction bound $\\mu F_n$")
    axc.plot(t, ft_t, color=FT_C, lw=1.6, zorder=4, label="tangential $F_t$")
    axc.axvline(tstar, color=ACCENT, lw=1.0, zorder=5)
    axc.plot([tstar, tstar], [fn_t[k], ft_t[k]], ls="none", marker="o",
             ms=3.4, mfc="white", mec=ACCENT, mew=1.0, zorder=6)
    axc.annotate(f"snapshot in (a)\n$F_n$ = {fn_t[k]:.0f} N, $F_t$ = "
                 f"{ft_t[k]:.0f} N", xy=(tstar, fn_t[k]),
                 xytext=(0.700, fn * 0.755), fontsize=6.9,
                 color=ACCENT, ha="left", va="center", linespacing=1.45,
                 zorder=7,
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=ACCENT,
                                 shrinkA=1, shrinkB=4))
    axc.text(t_lock + 0.02, 0.10 * fn, f"full sliding\nfrom {t_lock:.2f} s",
             fontsize=6.9, color=ACCENT, ha="left", va="bottom",
             linespacing=1.45)
    axc.set_xlim(0.0, t[-1])
    axc.set_ylim(0.0, 1.42 * fn)
    axc.set_xlabel("time into the braking event (s)", fontsize=8.0)
    axc.set_ylabel("force (N)", fontsize=8.0)
    axc.set_title("Traction saturates once it meets $\\mu F_n$",
                  fontsize=8.6, fontweight="bold", pad=4.0)
    axc.tick_params(labelsize=7.0, length=2.4)
    axc.grid(True, lw=0.45, color="#ececec")
    axc.set_axisbelow(True)
    axc.legend(loc="upper left", fontsize=6.8, handlelength=1.7,
               labelspacing=0.3, borderaxespad=0.35)
    add_panel_label(axc, "(c)", x=-0.235, y=1.02)

    p_slip = float(np.nanmean(p[slip]))
    p_stick = float(np.nanmean(p[inside & ~slip]))
    savefig(fig, "1146_wm_phys_contact_field.png")


if __name__ == "__main__":
    main()

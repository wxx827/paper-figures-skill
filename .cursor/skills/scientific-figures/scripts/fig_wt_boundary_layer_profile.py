"""Law of the wall: hot-wire traverses through a flat-plate boundary layer.

Main panel: the signature plot of turbulent boundary-layer experiment, mean
velocity in wall units on a logarithmic wall distance.  Two straight lines fix
the frame - the viscous sublayer, where the profile has to be u+ = y+, and the
logarithmic law with the usual constants - and three traverses at increasing
Reynolds number climb from one to the other and then bow above the log line in
the wake region, further at each Reynolds number because the wake strength
grows with it.

The companion panels put the same data back into laboratory coordinates.  On
the left the outer-scaled profile is compared with the Blasius laminar
solution, which is integrated here rather than tabulated; on the right the
shape factor and momentum thickness are integrated from a family of profiles
along the plate, so the collapse of H from the laminar value to the turbulent
one is measured, not asserted.

Illustrative synthetic data: Spalding's inner law plus a Coles wake, sampled at
discrete wall distances with a little scatter added.
"""
import textwrap

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig, style_axes, add_footer
from figkit import axes_inches, panel_labels_inches

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
MAIN_RECT = (0.62, 0.86, 5.30, 4.42)
AUX_T_RECT = (7.32, 3.52, 3.00, 1.86)
AUX_B_RECT = (7.32, 0.82, 3.00, 1.86)

KAPPA = 0.41            # von Karman constant (dimensionless)
B_LOG = 5.0             # log-law intercept (dimensionless)
U_E = 25.0              # m/s, free-stream speed in the working section
NU = 1.50e-5            # m^2/s, kinematic viscosity of air at 20 degC
RE_X_TR = 5.0e5         # transition Reynolds number (dimensionless)

CASES = [dict(re_theta=1500, re_tau=540.0, pi=0.36, c=BLUE),
         dict(re_theta=4200, re_tau=1280.0, pi=0.48, c=GREEN),
         dict(re_theta=11000, re_tau=2890.0, pi=0.55, c=RED)]


def inner_law(u_plus):
    """Spalding's single formula for the sublayer, buffer and log regions."""
    k = KAPPA * u_plus
    return u_plus + np.exp(-KAPPA * B_LOG) * (
        np.exp(k) - 1.0 - k - k ** 2 / 2.0 - k ** 3 / 6.0)


def composite(re_tau, pi_wake, n=6000):
    """(y+, u+) through a boundary layer of thickness ``re_tau`` wall units."""
    up = np.linspace(1e-9, 32.0, n)
    yp = inner_law(up)
    keep = yp <= re_tau
    yp, up = yp[keep], up[keep]
    eta = np.clip(yp / re_tau, 0.0, 1.0)
    return yp, up + (2.0 * pi_wake / KAPPA) * np.sin(0.5 * np.pi * eta) ** 2


def sample(yp, up, re_tau, rng, n=24):
    """Discrete hot-wire stations, log-spaced, with a little scatter."""
    yq = np.logspace(np.log10(0.55), np.log10(0.97 * re_tau), n)
    uq = np.interp(np.log(yq), np.log(yp), up)
    return yq, uq * (1.0 + rng.normal(0.0, 0.009, n)) + rng.normal(0.0, 0.05, n)


def blasius(eta_max=9.0):
    """Integrate f''' + f f'' / 2 = 0; returns (eta, f') on a dense grid."""
    def rhs(_, s):
        return [s[1], s[2], -0.5 * s[0] * s[2]]
    sol = solve_ivp(rhs, (0.0, eta_max), [0.0, 0.0, 0.3320573362],
                    dense_output=True, rtol=1e-10, atol=1e-12)
    eta = np.linspace(0.0, eta_max, 900)
    return eta, sol.sol(eta)[1]


def blasius_shape(eta, fp):
    """Wall-normal coordinate scaled by delta_99 and the matching profile."""
    eta99 = float(np.interp(0.99, fp, eta))
    return eta / eta99, np.clip(fp, 0.0, 1.0)


def plate_growth(eta_b, fp_b, n=260):
    """Shape factor and momentum thickness along a transitioning flat plate."""
    yn_lam, u_lam = blasius_shape(eta_b, fp_b)
    yn = np.linspace(0.0, 1.0, 400)
    u_lam = np.interp(yn, yn_lam, u_lam)

    x = np.linspace(0.06, 2.20, n)
    re_x = U_E * x / NU
    x_tr = RE_X_TR * NU / U_E
    d_lam = 5.0 * x / np.sqrt(re_x)
    d_tur = 0.37 * x / re_x ** 0.2
    cf = 0.0592 / re_x ** 0.2
    re_tau = np.sqrt(cf / 2.0) * U_E * d_tur / NU

    h_arr, th_arr, dl_arr = [], [], []
    for i, xi in enumerate(x):
        w = 1.0 / (1.0 + np.exp(-(xi - x_tr) / 0.055))
        yp, up = composite(max(re_tau[i], 120.0), 0.50)
        u_tur = np.interp(yn * max(re_tau[i], 120.0), yp, up)
        u_tur = np.clip(u_tur / u_tur[-1], 0.0, 1.0)
        u = (1.0 - w) * u_lam + w * u_tur
        delta = (1.0 - w) * d_lam[i] + w * d_tur[i]
        d_star = delta * np.trapezoid(1.0 - u, yn)
        theta = delta * np.trapezoid(u * (1.0 - u), yn)
        h_arr.append(d_star / theta)
        th_arr.append(theta * 1e3)
        dl_arr.append(delta)
    return x, np.array(h_arr), np.array(th_arr), x_tr, yn, u_lam


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


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(20517)
    eta_b, fp_b = blasius()
    x, h_arr, th_arr, x_tr, yn, u_lam = plate_growth(eta_b, fp_b)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_p = axes_inches(fig, AUX_T_RECT)
    ax_h = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_p, ax_h):
        a.set_axisbelow(True)
        a.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)
        style_axes(a)

    # ---- main: wall units ---------------------------------------------------
    ax.set_xscale("log")
    ax.set_xlim(0.30, 6.0e3)
    ax.set_ylim(0.0, 30.0)
    ax.axvspan(0.30, 5.0, color="#eef2f7", zorder=0)
    ax.axvspan(30.0, 0.15 * CASES[-1]["re_tau"], color="#f4f1e6", zorder=0)

    yv = np.logspace(np.log10(0.30), np.log10(13.0), 60)
    ax.plot(yv, yv, color=INK, lw=1.3, ls=(0, (5.0, 2.4)), zorder=6)
    yl = np.logspace(np.log10(7.0), np.log10(6.0e3), 80)
    ax.plot(yl, np.log(yl) / KAPPA + B_LOG, color=INK, lw=1.3,
            ls=(0, (1.4, 1.8)), zorder=6)

    for cs in CASES:
        yp, up = composite(cs["re_tau"], cs["pi"])
        ax.plot(yp, up, color=cs["c"], lw=1.5, alpha=0.55, zorder=4)
        ys, us = sample(yp, up, cs["re_tau"], rng)
        ax.plot(ys, us, ls="none", marker="o", ms=3.6, mfc="white",
                mec=cs["c"], mew=1.15, zorder=7,
                label=f"$Re_\\theta$ = {cs['re_theta']:5d}, "
                      f"$Re_\\tau$ = {cs['re_tau']:.0f}")
        ax.plot([cs["re_tau"]], [up[-1]], marker="s", ms=3.6,
                color=cs["c"], zorder=8)

    ax.annotate("viscous sublayer\n$u^+ = y^+$", xy=(3.1, 3.1),
                xytext=(0.40, 7.4), fontsize=6.8, color=INK, ha="left",
                va="bottom", linespacing=1.4, zorder=9,
                arrowprops=dict(arrowstyle="-|>", color=FAINT, lw=0.8,
                                shrinkA=2, shrinkB=2))
    ax.text(12.0, 1.0, "buffer\nlayer", fontsize=6.6, color=MUTED,
            ha="center", va="bottom", linespacing=1.4, zorder=9)
    ax.text(150.0, 13.6, "logarithmic law\n"
            f"$u^+ = (1/{KAPPA})\\,\\ln y^+ + {B_LOG:.1f}$", fontsize=6.8,
            color=INK, ha="center", va="top", linespacing=1.4, zorder=9)
    ax.annotate("wake region: the profile\nlifts above the log line",
                xy=(1900.0, 26.4), xytext=(320.0, 28.4), fontsize=6.6,
                color=SUB, ha="center", va="top", linespacing=1.4, zorder=9,
                arrowprops=dict(arrowstyle="-|>", color=FAINT, lw=0.8,
                                shrinkA=2, shrinkB=2))
    ax.text(0.98, 0.030,
            "filled squares mark the boundary-layer edge, $y^+ = Re_\\tau$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            color=SUB, zorder=10)
    ax.tick_params(labelsize=7.2)
    ax.set_xlabel("wall distance in viscous units  $y^+ = y u_\\tau / \\nu$  "
                  "(dimensionless)", fontsize=8.4, labelpad=1.5)
    ax.set_ylabel("mean velocity  $u^+ = \\bar{u} / u_\\tau$  "
                  "(dimensionless)", fontsize=8.4)
    ax.set_title("Flat-plate boundary layer in wall units, "
                 f"$U_e$ = {U_E:.0f} m/s", fontsize=9.0, fontweight="bold",
                 pad=5.0)
    ax.legend(loc="upper left", frameon=False, fontsize=6.8,
              handlelength=1.4, labelspacing=0.32, borderaxespad=0.35)

    # ---- aux (b): outer coordinates ----------------------------------------
    yp, up = composite(CASES[1]["re_tau"], CASES[1]["pi"])
    u_tur = np.interp(yn * CASES[1]["re_tau"], yp, up)
    u_tur = np.clip(u_tur / u_tur[-1], 0.0, 1.0)
    ax_p.plot(u_lam, yn, color=GOLD, lw=1.8, zorder=5, label="laminar (Blasius)")
    ax_p.plot(u_tur, yn, color=GREEN, lw=1.8, zorder=5,
              label=f"turbulent ($Re_\\tau$ = {CASES[1]['re_tau']:.0f})")
    ax_p.fill_betweenx(yn, u_lam, u_tur, color="#eeeeee", zorder=2)
    ax_p.axhline(1.0, color=FAINT, lw=0.8, ls=(0, (3.0, 2.0)), zorder=1)
    ax_p.set_xlim(0.0, 1.06)
    ax_p.set_ylim(0.0, 1.18)
    ax_p.tick_params(labelsize=6.8)
    ax_p.set_xlabel("$\\bar{u}/U_e$ (dimensionless)", fontsize=7.5,
                    labelpad=1.5)
    ax_p.set_ylabel("$y/\\delta_{99}$ (dimensionless)", fontsize=7.5)
    ax_p.set_title("Turbulence packs the momentum against the wall",
                   fontsize=8.5, fontweight="bold", pad=4.0)
    h_lam = (np.trapezoid(1.0 - u_lam, yn)
             / np.trapezoid(u_lam * (1.0 - u_lam), yn))
    h_tur = (np.trapezoid(1.0 - u_tur, yn)
             / np.trapezoid(u_tur * (1.0 - u_tur), yn))
    ax_p.text(0.045, 0.955,
              f"integrated from these two curves:\n$H$ = {h_lam:.2f} laminar, "
              f"{h_tur:.2f} turbulent",
              transform=ax_p.transAxes, ha="left", va="top", fontsize=6.4,
              color=SUB, linespacing=1.45)
    ax_p.legend(loc="upper left", bbox_to_anchor=(0.030, 0.80), frameon=False,
                fontsize=6.5, handlelength=1.6, labelspacing=0.28,
                borderaxespad=0.0)

    # ---- aux (c): growth along the plate -----------------------------------
    ax_h.plot(x, h_arr, color=BLUE, lw=1.8, zorder=6)
    ax_h.axvline(x_tr, color=RED, lw=0.9, ls=(0, (3.0, 2.0)), zorder=3)
    ax_h.text(x_tr + 0.05, 0.955,
              f"transition\n$Re_x$ = {RE_X_TR:.0e}".replace("e+05", " x 10$^5$"),
              transform=ax_h.get_xaxis_transform(), ha="left", va="top",
              fontsize=6.4, color=RED, linespacing=1.4)
    ax_h.set_xlim(0.0, x.max())
    ax_h.set_ylim(1.15, 2.95)
    ax_h.tick_params(labelsize=6.8, labelcolor=BLUE, axis="y")
    ax_h.tick_params(labelsize=6.8, axis="x")
    ax_h.set_xlabel("distance from the leading edge x (m)", fontsize=7.5,
                    labelpad=1.5)
    ax_h.set_ylabel("shape factor $H = \\delta^*/\\theta$", fontsize=7.5,
                    color=BLUE)
    ax_h.set_title("The shape factor collapses at transition", fontsize=8.5,
                   fontweight="bold", pad=4.0)

    ax_t = ax_h.twinx()
    ax_t.plot(x, th_arr, color=GOLD, lw=1.8, ls=(0, (4.4, 2.0)), zorder=5)
    ax_t.set_ylim(0.0, th_arr.max() * 1.28)
    ax_t.set_ylabel("momentum thickness $\\theta$ (mm)", fontsize=7.5,
                    color=GOLD)
    ax_t.tick_params(labelsize=6.8, labelcolor=GOLD)
    ax_t.spines["top"].set_visible(False)
    ax_t.grid(False)
    ax_h.text(0.975, 0.055,
              f"$H$ falls from {h_arr[0]:.2f} to {h_arr[-1]:.2f};\n"
              f"$\\theta$ reaches {th_arr[-1]:.2f} mm at x = {x[-1]:.1f} m",
              transform=ax_h.transAxes, ha="right", va="bottom", fontsize=6.4,
              color=SUB, linespacing=1.45)
    ax_h.legend(handles=[Line2D([], [], color=BLUE, lw=1.8, label="$H$"),
                         Line2D([], [], color=GOLD, lw=1.8,
                                ls=(0, (4.4, 2.0)), label="$\\theta$")],
                loc="upper right", frameon=False, fontsize=6.5,
                handlelength=1.7, labelspacing=0.26, borderaxespad=0.25)

    panel_labels_inches(fig, [ax, ax_p, ax_h], dx=-0.56, dy=0.10)
    foot = add_footer(fig, "Synthetic data, seed=20517")
    check_layout(fig, foot, [ax, ax_p, ax_h])
    savefig(fig, "931_wt_boundary_layer_profile.png")


if __name__ == "__main__":
    main()

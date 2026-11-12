"""Peri-implant bone stress: a 2-D section, the interface profile, and a sweep.

One analytic stress kernel serves all three panels, so the contour map, the
interface profile extracted from it and the parameter sweep can never tell
different stories.  The kernel decays away from the implant surface, is
amplified inside the stiff cortical collar and is biased toward the loaded
side by the off-axis component of the bite force, which reproduces the
clinically familiar result: the peak sits in the crestal cortical bone at the
implant neck, not at the apex.  The quoted peak value and its coordinates are
read out of the plotted array with ``argmax``, and the sweep bars re-evaluate
the same kernel rather than restating stored numbers.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import axes_inches, panel_labels_inches, safe_colorbar

FW, FH = 10.4, 6.4
MAIN = (0.86, 0.86, 5.10, 4.56)
AUX_T = (7.12, 3.52, 3.00, 1.90)
AUX_B = (7.12, 0.86, 3.00, 1.76)

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HAIR = "#d5d5d5"
RED, BLUE, GOLD = "#b23a48", "#3b5b92", "#c9a227"
CORT_EDGE = "#8c6f42"       # cortical / cancellous interface rule
METAL, METAL_LT = "#3a3a3a", "#555555"
ENAMEL, ENAMEL_EDGE = "#f0ebe0", "#8a8a8a"
STRESS_CMAP = get_cmap("cb_sequential_warm")

XLIM, YLIM = (-11.8, 11.8), (-13.5, 7.6)
NX, NY = 660, 600           # one grid for every panel, so peaks agree exactly
CORT_T = 1.8                # cortical plate thickness, mm
L_IMP = 11.0                # intra-osseous implant length, mm
R_REF = 1.90                # reference implant radius, mm
F_REF = 200.0               # reference bite force, N
ANGLE = 30.0                # off-axis load angle of the main panel, deg

FOOT = ("Synthetic two-dimensional section through a bone-level implant; "
        "stress values come from an analytic kernel calibrated to the "
        "textbook pattern, not from a solved finite-element model. Cortical "
        "plate 1.8 mm thick over cancellous bone, implant 11.0 mm long, load "
        "applied at the crown. Panels (b) and (c) resample the same kernel, "
        "so all three panels share one scale. Illustrative only.")


def fit_limits(ax, xlim, ylim, rect):
    """Pad the shorter data axis so an equal-aspect panel fills its box."""
    want = rect[2] / rect[3]
    (x0, x1), (y0, y1) = xlim, ylim
    if (x1 - x0) / (y1 - y0) < want:
        e = (y1 - y0) * want - (x1 - x0)
        x0, x1 = x0 - e / 2.0, x1 + e / 2.0
    else:
        e = (x1 - x0) / want - (y1 - y0)
        y0, y1 = y0 - e / 2.0, y1 + e / 2.0
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")


def von_mises(X, Y, angle_deg=30.0, radius=R_REF, force=F_REF):
    """Equivalent stress in the bone, in MPa; NaN inside the implant and air."""
    th = np.deg2rad(angle_deg)
    dx = np.maximum(np.abs(X) - radius, 0.0)
    dy = np.maximum(-L_IMP - Y, 0.0)
    d = np.hypot(dx, dy)

    crest = np.exp(-((Y + 0.60) / 2.0) ** 2)            # crestal collar
    apex = np.exp(-(np.hypot(dx, Y + L_IMP) / 2.6) ** 2)
    side = 1.0 + 0.95 * np.sin(th) * np.tanh(X / 1.4)   # loaded side is worse
    cort = 1.0 + 1.65 * 0.5 * (1.0 + np.tanh((Y + CORT_T) / 0.32))
    slim = (R_REF / radius) ** 1.35                     # narrow = higher stress

    s = (force / F_REF) * slim * side * np.exp(-d / 1.55) * (
        52.0 * crest * cort + 17.0 * apex)
    s = np.where((np.abs(X) <= radius) & (Y >= -L_IMP), np.nan, s)
    s = np.where(Y > 0.0, np.nan, s)
    return s


def stress_grid(angle_deg=ANGLE, radius=R_REF):
    """The one sampling grid every panel uses, so their maxima are identical."""
    X, Y = np.meshgrid(np.linspace(XLIM[0], XLIM[1], NX),
                       np.linspace(YLIM[0], YLIM[1], NY))
    return X, Y, von_mises(X, Y, angle_deg=angle_deg, radius=radius)


def peak_stress(angle_deg, radius):
    """Maximum bone stress for one configuration, from the same kernel."""
    return float(np.nanmax(stress_grid(angle_deg, radius)[2]))


def implant_poly(radius, n_thread=13):
    """Half-outline of a threaded bone-level implant, mirrored about x = 0."""
    right = [(1.20, 3.00), (1.20, 0.90), (radius, 0.40)]
    y0, y1 = -0.30, -L_IMP + 1.5
    ys = np.linspace(y0, y1, n_thread + 1)
    for a, b in zip(ys[:-1], ys[1:]):
        right += [(radius, a), (radius - 0.34, 0.5 * (a + b)), (radius, b)]
    right += [(radius, y1), (radius * 0.62, -L_IMP + 0.30),
              (radius * 0.24, -L_IMP)]
    left = [(-x, y) for x, y in right[::-1]]
    return np.array(right + left)


def crown_poly(w=4.3, y0=3.0, y1=5.4, n_cusp=2):
    t = np.linspace(0.0, 1.0, 70)
    xs = np.linspace(-0.46 * w, 0.46 * w, 70)
    h = y1 - y0
    top = y0 + 0.82 * h + 0.18 * h * np.abs(np.sin(n_cusp * np.pi * t))
    pts = [(-0.30 * w, y0), (-0.50 * w, y0 + 0.35 * h)]
    pts += list(zip(xs, top))
    pts += [(0.50 * w, y0 + 0.35 * h), (0.30 * w, y0)]
    return np.array(pts)


def main():
    set_pub_style()
    fig = plt.figure(figsize=(FW, FH))

    X, Y, S = stress_grid()
    smax = float(np.nanmax(S))

    # ---- (a) sectional stress map ------------------------------------------
    ax = axes_inches(fig, MAIN)
    im = ax.imshow(S, origin="lower", cmap=STRESS_CMAP,
                   extent=[XLIM[0], XLIM[1], YLIM[0], YLIM[1]],
                   vmin=0.0, vmax=smax, interpolation="bilinear", zorder=2)
    ax.contour(X, Y, S, levels=np.linspace(14.0, smax * 0.92, 6),
               colors="#8c6f42", linewidths=0.35, alpha=0.55, zorder=3)

    ax.add_patch(Rectangle((XLIM[0], YLIM[0]), XLIM[1] - XLIM[0], -YLIM[0],
                           facecolor="none", edgecolor=CORT_EDGE, lw=0.8,
                           zorder=4))
    ax.plot(XLIM, [-CORT_T, -CORT_T], color=CORT_EDGE, lw=0.9,
            ls=(0, (3.0, 2.2)), zorder=5)
    ax.text(XLIM[0] + 0.4, -0.95, f"cortical plate, {CORT_T:.1f} mm",
            ha="left", va="center", fontsize=6.3, color=INK)
    ax.text(XLIM[0] + 0.4, -3.1, "cancellous bone", ha="left", va="center",
            fontsize=6.3, color=INK)

    ax.add_patch(Polygon(implant_poly(R_REF), closed=True, facecolor=METAL,
                         edgecolor="#1a1a1a", lw=0.5, zorder=6))
    ax.add_patch(Polygon([(-1.20, 0.40), (1.20, 0.40), (1.20, 3.00),
                          (-1.20, 3.00)], closed=True, facecolor=METAL_LT,
                         edgecolor="#1a1a1a", lw=0.5, zorder=6))
    ax.add_patch(Polygon(crown_poly(), closed=True, facecolor=ENAMEL,
                         edgecolor=ENAMEL_EDGE, lw=0.6, zorder=6))

    # Bite force, drawn at the angle actually used by the kernel.
    d = np.array([np.sin(np.deg2rad(ANGLE)), -np.cos(np.deg2rad(ANGLE))])
    tip = np.array([0.0, 5.30])
    tail = tip - 2.5 * d
    ax.annotate("", xy=tuple(tip), xytext=tuple(tail),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3,
                                mutation_scale=11), zorder=8)
    ax.plot([0.0, 0.0], [5.30, 7.30], color=MUTED, lw=0.6, ls=(0, (2.4, 2.0)),
            zorder=7)
    ax.text(tail[0] - 0.35, tail[1],
            f"F = {F_REF:.0f} N at {ANGLE:.0f} deg", ha="right", va="center",
            fontsize=6.8, color=INK)

    k = np.unravel_index(np.nanargmax(S), S.shape)
    px, py = float(X[k]), float(Y[k])
    ax.plot([px], [py], marker="o", ms=5.0, mfc="none", mec=RED, mew=1.2,
            zorder=9)
    ax.annotate(f"peak {smax:.0f} MPa in the crestal\ncortex, "
                f"{abs(py):.1f} mm below the crest",
                xy=(px, py), xytext=(px + 2.6, py + 4.2), fontsize=6.6,
                color=RED, ha="left", va="center", linespacing=1.4,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.9,
                                shrinkA=1.0, shrinkB=3.0), zorder=9)

    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fit_limits(ax, XLIM, YLIM, MAIN)
    ax.set_title("Sectional von Mises field around a bone-level implant",
                 fontsize=8.8, fontweight="bold", color=INK, pad=4)
    safe_colorbar(fig, im, ax, label="von Mises stress (MPa)", width=0.11,
                  pad=0.09, labelsize=6.6)

    # ---- (b) profile along the bone-implant interface ----------------------
    ax_b = axes_inches(fig, AUX_T)
    depth = np.linspace(0.05, L_IMP + 1.2, 240)
    prof = {}
    for sgn, col, name in ((+1.0, RED, "loaded side"),
                           (-1.0, BLUE, "opposite side")):
        xq = sgn * (R_REF + 0.15) * np.ones_like(depth)
        prof[name] = von_mises(xq, -depth)
        ax_b.plot(depth, prof[name], color=col, lw=1.4, label=name)
    ax_b.axvspan(0.0, CORT_T, color=GOLD, alpha=0.16, lw=0)
    ax_b.set_xlim(0, L_IMP + 1.2)
    ax_b.set_ylim(0, smax * 1.12)
    ax_b.text(CORT_T + 0.25, smax * 1.05, "cortical", ha="left", va="top",
              fontsize=6.2, color=SUB)
    top = prof["loaded side"]
    jm = int(np.argmax(top))
    ax_b.plot([depth[jm]], [top[jm]], marker="o", ms=4.0, mfc="none", mec=RED,
              mew=1.0)
    ax_b.text(depth[jm] + 1.5, top[jm], f"{top[jm]:.0f} MPa", ha="left",
              va="center", fontsize=6.3, color=RED)
    ax_b.set_xlabel("depth below bone crest (mm)", fontsize=8.0)
    ax_b.set_ylabel("von Mises stress (MPa)", fontsize=8.0)
    ax_b.tick_params(labelsize=6.6)
    ax_b.legend(loc="upper right", frameon=False, fontsize=6.5,
                handlelength=1.6)
    ax_b.set_title("Stress along the bone-implant interface", fontsize=8.5,
                   fontweight="bold", color=INK, pad=3)

    # ---- (c) peak stress across load angle and implant diameter ------------
    ax_c = axes_inches(fig, AUX_B)
    angles = [0.0, 15.0, ANGLE]
    diams = [2.0 * R_REF, 4.5]
    xg = np.arange(len(angles), dtype=float)
    for m, (dia, col) in enumerate(zip(diams, (GOLD, BLUE))):
        vals = [peak_stress(a, dia / 2.0) for a in angles]
        ax_c.bar(xg + (m - 0.5) * 0.34, vals, width=0.32, color=col,
                 edgecolor="white", linewidth=0.5,
                 label=f"{dia:.1f} mm diameter")
        for xi, v in zip(xg + (m - 0.5) * 0.34, vals):
            ax_c.text(xi, v + 2.0, f"{v:.0f}", ha="center", va="bottom",
                      fontsize=6.1, color=SUB)
    ax_c.grid(axis="y", color=HAIR, lw=0.6)
    ax_c.set_axisbelow(True)
    ax_c.set_xticks(xg)
    ax_c.set_xticklabels([f"{a:.0f} deg" for a in angles], fontsize=6.8)
    ax_c.set_xlabel("off-axis load angle (deg)", fontsize=8.0)
    ax_c.set_ylabel("peak stress (MPa)", fontsize=8.0)
    ax_c.tick_params(labelsize=6.6)
    ax_c.set_ylim(0, peak_stress(angles[-1], diams[0] / 2.0) * 1.30)
    ax_c.legend(loc="upper left", frameon=False, fontsize=6.4,
                handlelength=1.4)
    ax_c.set_title("Peak stress vs load angle and diameter", fontsize=8.5,
                   fontweight="bold", color=INK, pad=3)

    panel_labels_inches(fig, [ax, ax_b, ax_c], dx=-0.50, dy=0.10)
    savefig(fig, "818_dnt_implant_stress.png")


if __name__ == "__main__":
    main()

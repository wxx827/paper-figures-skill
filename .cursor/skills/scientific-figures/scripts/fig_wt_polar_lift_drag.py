"""Force-balance polars for an aerofoil section at three Reynolds numbers.

Main panel: the drag polar, lift coefficient against drag coefficient, with the
two constructions that a polar exists for.  The bucket floor gives the minimum
drag and the incidence that buys it, and the tangent drawn from the origin
touches the polar exactly at the best lift-to-drag ratio - both are located by
searching the plotted arrays, so the tangent cannot miss its own contact point.
An inset carries the lift curve itself, including the stall break that the
polar can only show as a fold.

The companion panels split the two questions a polar cannot answer on its own:
how far the best glide ratio is from the stall, and whether the section is
statically stable.  The pitching moment is referred to the balance centre at
0.40c, so its negative slope is the stability statement; the aerodynamic centre
printed on that panel is recovered from a straight-line fit of the plotted
moment against the plotted lift.

Illustrative synthetic data from a smooth lift model with a blended stall.
"""
import textwrap

import numpy as np

import matplotlib.pyplot as plt
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
MAIN_RECT = (0.66, 0.86, 5.26, 4.42)
AUX_T_RECT = (7.34, 3.52, 2.98, 1.86)
AUX_B_RECT = (7.34, 0.82, 2.98, 1.86)

ALPHA_0 = -2.1          # deg, zero-lift incidence of the cambered section
CL_SLOPE = 0.1015       # 1/deg, lift-curve slope (dimensionless per deg)
K_IND = 0.0088          # profile-drag curvature (dimensionless)
CL_DES = 0.26           # lift coefficient the drag bucket is centred on
CM_QC = -0.058          # quarter-chord moment in the linear range
X_REF = 0.40            # balance moment reference, fraction of chord

CASES = [dict(re=0.5e6, cd0=0.0094, a_s=11.4, harsh=0.115, c=BLUE),
         dict(re=1.0e6, cd0=0.0079, a_s=12.9, harsh=0.100, c=GREEN),
         dict(re=2.0e6, cd0=0.0068, a_s=14.4, harsh=0.088, c=RED)]


def polar(case, n=481):
    """Cl, Cd, Cm and the stall fraction over the tested incidence range."""
    a = np.linspace(-6.0, 17.0, n)
    stall = 1.0 / (1.0 + np.exp(-(a - case["a_s"]) / 1.25))
    cl_lin = CL_SLOPE * (a - ALPHA_0)
    cl_post = CL_SLOPE * (case["a_s"] - ALPHA_0) - 0.052 * (a - case["a_s"])
    cl = (1.0 - stall) * cl_lin + stall * cl_post
    cd = (case["cd0"] + K_IND * (cl - CL_DES) ** 2
          + case["harsh"] * stall ** 1.7)
    cm = CM_QC - (X_REF - 0.25) * cl - 0.085 * stall ** 1.3
    return a, cl, cd, cm


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
    data = [(cs, *polar(cs)) for cs in CASES]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_e = axes_inches(fig, AUX_T_RECT)
    ax_m = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_e, ax_m):
        a.set_axisbelow(True)
        a.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)
        style_axes(a)

    # ---- main: the drag polar ----------------------------------------------
    ax.axhline(0.0, color=FAINT, lw=0.7, zorder=1)
    ref = data[-1]
    for cs, a, cl, cd, _ in data:
        ax.plot(cd, cl, color=cs["c"], lw=1.8, zorder=5,
                label=f"$Re_c$ = {cs['re'] / 1e6:.1f} x 10$^6$")

    cs, a, cl, cd, cm = ref
    ld = cl / cd
    i_ld = int(np.argmax(ld))
    i_min = int(np.argmin(cd))
    slope = ld[i_ld]
    xt = np.array([0.0, cl.max() * 1.06 / slope])
    ax.plot(xt, slope * xt, color=INK, lw=1.0, ls=(0, (5.0, 2.4)), zorder=6)
    ax.plot([cd[i_ld]], [cl[i_ld]], marker="o", ms=5.4, mfc="white", mec=INK,
            mew=1.4, zorder=9)
    ax.annotate(f"tangent from the origin:\n$(C_l/C_d)_{{max}}$ = "
                f"{slope:.1f} at $C_l$ = {cl[i_ld]:.2f},\n"
                f"$\\alpha$ = {a[i_ld]:.1f} deg",
                xy=(cd[i_ld], cl[i_ld]), xytext=(0.0325, 1.30), fontsize=6.8,
                color=INK, ha="left", va="top", linespacing=1.45, zorder=10,
                arrowprops=dict(arrowstyle="-|>", color=FAINT, lw=0.9,
                                shrinkA=2, shrinkB=4))
    ax.plot([cd[i_min]], [cl[i_min]], marker="D", ms=4.6, mfc="white",
            mec=GOLD, mew=1.4, zorder=9)
    ax.annotate(f"drag bucket floor:\n$C_{{d,min}}$ = {cd[i_min]:.4f} at "
                f"$C_l$ = {cl[i_min]:.2f}",
                xy=(cd[i_min], cl[i_min]), xytext=(0.0150, -0.44),
                fontsize=6.8, color=GOLD, ha="left", va="bottom",
                linespacing=1.45, zorder=10,
                arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=0.9,
                                shrinkA=2, shrinkB=4))
    ax.plot([cd[int(np.argmax(cl))]], [cl.max()], marker="^", ms=5.0,
            color=RED, zorder=9)
    ax.text(cd[int(np.argmax(cl))], cl.max() + 0.045,
            f"$C_{{l,max}}$ = {cl.max():.2f}", fontsize=6.8, color=RED,
            ha="center", va="bottom", zorder=10)

    ax.set_xlim(0.0, 0.072)
    ax.set_ylim(-0.72, 1.78)
    ax.tick_params(labelsize=7.2)
    ax.set_xlabel("section drag coefficient $C_d$ (dimensionless)",
                  fontsize=8.4, labelpad=1.5)
    ax.set_ylabel("section lift coefficient $C_l$ (dimensionless)",
                  fontsize=8.4)
    ax.set_title("Drag polar of a cambered section, "
                 "$-6$ deg $\\leq \\alpha \\leq$ 17 deg", fontsize=9.0,
                 fontweight="bold", pad=5.0)
    ax.legend(loc="upper left", frameon=False, fontsize=6.9, handlelength=1.6,
              labelspacing=0.32, borderaxespad=0.35)

    # inset: the lift curve the polar folds away
    axi = ax.inset_axes([0.575, 0.055, 0.400, 0.395])
    axi.set_axisbelow(True)
    axi.grid(color=GRID, lw=0.5, alpha=0.75, zorder=0)
    style_axes(axi)
    axi.axhline(0.0, color=FAINT, lw=0.6, zorder=1)
    for cs2, a2, cl2, _, _ in data:
        axi.plot(a2, cl2, color=cs2["c"], lw=1.4, zorder=5)
        axi.plot([a2[int(np.argmax(cl2))]], [cl2.max()], marker="^", ms=3.4,
                 color=cs2["c"], zorder=6)
    axi.set_xlim(-6.0, 17.0)
    axi.set_ylim(-0.72, 1.78)
    axi.set_xticks([-5, 0, 5, 10, 15])
    axi.tick_params(labelsize=6.0, pad=1.5)
    axi.set_xlabel("incidence $\\alpha$ (deg)", fontsize=6.6, labelpad=1.0)
    axi.set_ylabel("$C_l$ (dimensionless)", fontsize=6.6, labelpad=1.5)
    axi.set_title("lift curve and the stall break", fontsize=6.9,
                  color=INK, pad=2.5)

    # ---- aux (b): glide ratio ----------------------------------------------
    for cs2, a2, cl2, cd2, _ in data:
        ld2 = cl2 / cd2
        ax_e.plot(a2, ld2, color=cs2["c"], lw=1.8, zorder=5)
        j = int(np.argmax(ld2))
        ax_e.plot([a2[j]], [ld2[j]], marker="o", ms=4.0, mfc="white",
                  mec=cs2["c"], mew=1.3, zorder=7)
    ax_e.axhline(0.0, color=FAINT, lw=0.7, zorder=1)
    ax_e.set_xlim(-6.0, 17.0)
    ld_all = np.concatenate([cl2 / cd2 for _, _, cl2, cd2, _ in data])
    ax_e.set_ylim(ld_all.min() - 8.0, ld_all.max() * 1.44)
    ax_e.tick_params(labelsize=6.8)
    ax_e.set_xlabel("incidence $\\alpha$ (deg)", fontsize=7.5, labelpad=1.5)
    ax_e.set_ylabel("$C_l/C_d$ (dimensionless)", fontsize=7.5)
    ax_e.set_title("One incidence flies furthest", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    peaks = [(cs2["re"], float((cl2 / cd2).max()),
              float(a2[int(np.argmax(cl2 / cd2))]))
             for cs2, a2, cl2, cd2, _ in data]
    ax_e.text(0.035, 0.94,
              "\n".join(f"$Re_c$ {r / 1e6:.1f} x 10$^6$: peak {p:.1f} at "
                        f"{aa:.1f} deg" for r, p, aa in peaks),
              transform=ax_e.transAxes, ha="left", va="top", fontsize=6.4,
              color=SUB, linespacing=1.45)

    # ---- aux (c): pitching moment and static stability ---------------------
    for cs2, a2, cl2, _, cm2 in data:
        ax_m.plot(a2, cm2, color=cs2["c"], lw=1.8, zorder=5)
    ax_m.axhline(0.0, color=FAINT, lw=0.7, zorder=1)
    cs2, a2, cl2, cd2, cm2 = data[-1]
    lin = (a2 > -4.0) & (a2 < 9.0)
    d_cm_da = float(np.polyfit(a2[lin], cm2[lin], 1)[0])
    d_cm_dcl = float(np.polyfit(cl2[lin], cm2[lin], 1)[0])
    x_ac = X_REF + d_cm_dcl
    ax_m.axvline(cs2["a_s"], color=FAINT, lw=0.9, ls=(0, (3.0, 2.0)), zorder=2)
    ax_m.text(cs2["a_s"] - 0.5, 0.955, "stall", rotation=90,
              transform=ax_m.get_xaxis_transform(), ha="right", va="top",
              fontsize=6.4, color=MUTED)
    ax_m.set_xlim(-6.0, 17.0)
    cm_all = np.concatenate([c for _, _, _, _, c in data])
    ax_m.set_ylim(cm_all.min() - 0.115, cm_all.max() + 0.030)
    ax_m.tick_params(labelsize=6.8)
    ax_m.set_xlabel("incidence $\\alpha$ (deg)", fontsize=7.5, labelpad=1.5)
    ax_m.set_ylabel(f"$C_m$ about {X_REF:.2f}c (dimensionless)", fontsize=7.5)
    ax_m.set_title("Negative slope means statically stable", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_m.text(0.035, 0.050,
              f"fit over $-4$ to 9 deg: d$C_m$/d$\\alpha$ = {d_cm_da:+.4f} "
              f"per deg\nd$C_m$/d$C_l$ = {d_cm_dcl:+.3f}, so the aerodynamic "
              f"centre is at x/c = {x_ac:.2f}",
              transform=ax_m.transAxes, ha="left", va="bottom", fontsize=6.4,
              color=SUB, linespacing=1.45)

    panel_labels_inches(fig, [ax, ax_e, ax_m], dx=-0.60, dy=0.10)
    foot = add_footer(fig, "Synthetic data, figure 932")
    check_layout(fig, foot, [ax, ax_e, ax_m])
    savefig(fig, "932_wt_polar_lift_drag.png")


if __name__ == "__main__":
    main()

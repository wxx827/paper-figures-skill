"""Surface pressure distribution over a cambered airfoil at cruise incidence.

The section is drawn to scale with pressure vectors normal to the skin (outward
where the flow is accelerated, inward where it is decelerated), and the same
distribution is replotted below on the conventional inverted ``Cp`` axis with
the stagnation point and the upper-surface transition marked.  Two subordinate
panels carry the integrated section coefficients: the lift curve and the drag
polar, both passing through the design point shown in the main panels.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

C_UP = "#3b5b92"      # upper surface, house blue
C_LO = "#2e7d5b"      # lower surface, house green
ACC = "#b23a48"       # single emphasis: design point / stagnation / transition
INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"

CHORD_M = 1.20        # reference chord (m)
ALPHA_DEG = 4.0       # design incidence
AR = 8.4              # wing aspect ratio used for the induced-drag polar
OSWALD = 0.86
CD0 = 0.0068
S_STAG = 0.0075       # stagnation station on the lower surface (x/c)
S_TRANS = 0.44        # upper-surface transition station (x/c)


def naca4(m, p, t, x):
    """Upper / lower coordinates of a NACA 4-digit section at stations ``x``."""
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                    + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    yc = np.where(x < p,
                  m / p ** 2 * (2 * p * x - x ** 2),
                  m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
    dyc = np.where(x < p,
                   2 * m / p ** 2 * (p - x),
                   2 * m / (1 - p) ** 2 * (p - x))
    th = np.arctan(dyc)
    return (x - yt * np.sin(th), yc + yt * np.cos(th),
            x + yt * np.sin(th), yc - yt * np.cos(th))


def cp_upper(s):
    """Suction-side pressure coefficient with a transition kink."""
    v = (1.665 - 0.735 * s) * (1.0 - np.exp(-s / 0.035))
    v = v * (1.0 - 0.040 / (1.0 + np.exp(-(s - S_TRANS) / 0.012)))
    return 1.0 - v ** 2


def cp_lower(s):
    """Pressure side: stagnation sits just aft of the leading edge."""
    v = (0.870 + 0.600 * s - 0.487 * s ** 2)
    v = v * (1.0 - np.exp(-np.abs(s - S_STAG) / 0.030))
    return 1.0 - v ** 2


def main():
    set_pub_style(base_font=9)

    beta = np.linspace(0.0, np.pi, 401)
    s = 0.5 * (1.0 - np.cos(beta))            # cosine-clustered chord stations
    xu, yu, xl, yl = naca4(0.02, 0.40, 0.12, s)
    cpu, cpl = cp_upper(s), cp_lower(s)

    # --- integrated section coefficients, all read off the curves above -----
    cl_design = float(np.trapezoid(cpl - cpu, s))
    cp_min = float(cpu.min())
    x_cpmin = float(s[int(np.argmin(cpu))])
    cp_stag = float(cpl.max())

    alpha = np.linspace(-6.0, 16.0, 300)
    a0 = 2.0 * np.pi * np.pi / 180.0          # thin-airfoil slope, per degree
    cl_lin = cl_design + a0 * (alpha - ALPHA_DEG)
    cl_max = 1.42
    cl = cl_max * np.tanh(cl_lin / cl_max)    # smooth approach to stall
    post = alpha > 12.5
    cl[post] = cl[post] - 0.055 * (alpha[post] - 12.5) ** 2
    cd = CD0 + cl ** 2 / (np.pi * AR * OSWALD) + 0.0021 * np.maximum(
        alpha - 9.0, 0.0) ** 1.7 / 100.0
    ld = cl / cd
    i_ld = int(np.argmax(ld))

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.56],
                          height_ratios=[0.86, 1.0], hspace=0.30, wspace=0.26)
    ax_f = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_l = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 1])

    # ================= (a) section with surface pressure vectors ===========
    ax_f.fill(np.concatenate([xu, xl[::-1]]), np.concatenate([yu, yl[::-1]]),
              facecolor="#e3e8f0", edgecolor=RULE, linewidth=1.0, zorder=4)

    scale = 0.115 / max(abs(cp_min), cp_stag)
    for xs, ys, cps, col in ((xu, yu, cpu, C_UP), (xl, yl, cpl, C_LO)):
        k = np.linspace(14, len(s) - 8, 20).astype(int)
        dx = np.gradient(xs)[k]
        dy = np.gradient(ys)[k]
        n = np.hypot(dx, dy)
        nx, ny = -dy / n, dx / n
        if col == C_LO:
            nx, ny = -nx, -ny                 # outward normal on both skins
        mag = -cps[k] * scale                 # suction pulls away from the skin
        keep = np.abs(mag) > 0.012            # sub-pixel arrows read as noise
        # A compression arrow points *at* the skin, so its tail is offset
        # outward by its own length; a suction arrow starts on the skin.
        off = np.where(mag < 0.0, -mag, 0.0)
        ax_f.quiver((xs[k] + nx * off)[keep], (ys[k] + ny * off)[keep],
                    (nx * mag)[keep], (ny * mag)[keep], angles="xy",
                    scale_units="xy", scale=1.0, width=0.0028,
                    headwidth=4.2, headlength=5.0, color=col, alpha=0.9,
                    zorder=6 if col == C_UP else 5)

    ax_f.plot(xl[np.argmax(cpl)], yl[np.argmax(cpl)], marker="o", ms=5.0,
              mfc="white", mec=ACC, mew=1.3, zorder=9)
    ax_f.annotate("stagnation point", xy=(xl[np.argmax(cpl)],
                                          yl[np.argmax(cpl)]),
                  xytext=(0.10, -0.235), fontsize=6.8, color=ACC,
                  ha="left", va="center",
                  arrowprops=dict(arrowstyle="-", color=ACC, lw=0.7,
                                  shrinkA=0.5, shrinkB=3.0,
                                  connectionstyle="arc3,rad=0.25"))
    j_tr = int(np.argmin(np.abs(s - S_TRANS)))
    ax_f.plot(xu[j_tr], yu[j_tr], marker="s", ms=4.2, mfc="white", mec=ACC,
              mew=1.2, zorder=9)
    ax_f.text(xu[j_tr] + 0.02, yu[j_tr] + 0.135, "transition", fontsize=6.8,
              color=ACC, ha="left", va="bottom")

    ax_f.annotate("", xy=(-0.015, 0.175), xytext=(-0.215, 0.175 - 0.200
                                                  * np.tan(np.deg2rad(
                                                      ALPHA_DEG))),
                  arrowprops=dict(arrowstyle="-|>", color=RULE, lw=1.0))
    ax_f.text(-0.215, 0.212, f"$V_\\infty$,  $\\alpha$ = {ALPHA_DEG:.1f}$^\\circ$",
              fontsize=7.0, color=RULE, ha="left", va="bottom")
    ax_f.plot([0.0, 1.0], [0.0, 0.0], color=MUTED, lw=0.7,
              ls=(0, (5, 3)), zorder=3)
    ax_f.text(1.02, 0.0, f"chord $c$ = {CHORD_M:.2f} m", fontsize=6.8,
              color=MUTED, ha="left", va="center")

    ax_f.set_xlim(-0.24, 1.32)
    ax_f.set_ylim(-0.31, 0.31)
    ax_f.set_aspect("equal")
    ax_f.axis("off")
    ax_f.set_title("NACA 2412 section, pressure vectors normal to the skin",
                   fontsize=8.8, fontweight="bold", pad=4)

    handles = [Line2D([], [], color=C_UP, lw=1.4, label="upper surface"),
               Line2D([], [], color=C_LO, lw=1.4, label="lower surface")]
    ax_f.legend(handles=handles, loc="lower right", fontsize=7.0,
                frameon=False, ncol=2, borderaxespad=0.1,
                bbox_to_anchor=(1.02, -0.06))

    # ================= (b) inverted Cp distribution ========================
    ax_c.plot(s, cpu, color=C_UP, lw=1.5, zorder=5, label="upper surface")
    ax_c.plot(s, cpl, color=C_LO, lw=1.5, zorder=5, label="lower surface")
    ax_c.fill_between(s, cpu, cpl, color=C_UP, alpha=0.09, lw=0, zorder=2)
    ax_c.axhline(0.0, color=MUTED, lw=0.7, ls=(0, (5, 3)), zorder=1)

    ax_c.plot(S_STAG, cp_stag, marker="o", ms=5.0, mfc="white", mec=ACC,
              mew=1.3, zorder=8)
    ax_c.annotate(f"stagnation, $C_p$ = {cp_stag:.2f}",
                  xy=(S_STAG, cp_stag), xytext=(0.12, cp_stag - 0.06),
                  fontsize=6.8, color=ACC, ha="left", va="center",
                  arrowprops=dict(arrowstyle="-|>", color=ACC, lw=0.7,
                                  shrinkA=1.0, shrinkB=3.0))
    ax_c.plot(S_TRANS, cp_upper(np.array([S_TRANS]))[0], marker="s", ms=4.2,
              mfc="white", mec=ACC, mew=1.2, zorder=8)
    ax_c.annotate(f"transition at $x/c$ = {S_TRANS:.2f}",
                  xy=(S_TRANS, cp_upper(np.array([S_TRANS]))[0]),
                  xytext=(0.52, -0.92), fontsize=6.8, color=ACC,
                  ha="left", va="center",
                  arrowprops=dict(arrowstyle="-|>", color=ACC, lw=0.7,
                                  shrinkA=1.0, shrinkB=3.0,
                                  connectionstyle="arc3,rad=-0.25"))
    ax_c.annotate(f"$C_{{p,\\min}}$ = {cp_min:.2f}",
                  xy=(x_cpmin, cp_min), xytext=(x_cpmin + 0.08, cp_min - 0.16),
                  fontsize=6.8, color=C_UP, ha="left", va="center",
                  arrowprops=dict(arrowstyle="-|>", color=C_UP, lw=0.7,
                                  shrinkA=1.0, shrinkB=2.0))

    ax_c.set_xlim(-0.02, 1.02)
    ax_c.set_ylim(1.28, cp_min - 0.42)        # inverted: suction points up
    ax_c.set_xlabel("chord station $x/c$ (-)")
    ax_c.set_ylabel("pressure coefficient $C_p$ (-)")
    ax_c.text(0.985, cp_min - 0.34,
              f"enclosed area gives $C_L$ = {cl_design:.3f}",
              fontsize=6.8, color=SUB, ha="right", va="top")
    ax_c.legend(loc="lower right", fontsize=7.0, frameon=False,
                borderaxespad=0.3)
    ax_c.grid(True, axis="y", lw=0.5, alpha=0.55)

    # ================= (c) lift curve (subordinate) ========================
    ax_l.plot(alpha, cl, color=C_UP, lw=1.4)
    ax_l.plot(ALPHA_DEG, cl_design, marker="o", ms=5.0, mfc="white", mec=ACC,
              mew=1.3, zorder=6)
    ax_l.axhline(0.0, color=MUTED, lw=0.6, ls=(0, (5, 3)))
    ax_l.annotate(f"design point\n$C_L$ = {cl_design:.2f}",
                  xy=(ALPHA_DEG, cl_design), xytext=(-5.4, 1.06),
                  fontsize=6.5, color=ACC, ha="left", va="center",
                  arrowprops=dict(arrowstyle="-|>", color=ACC, lw=0.7,
                                  shrinkA=1.0, shrinkB=3.0,
                                  connectionstyle="arc3,rad=0.22"))
    i_max = int(np.argmax(cl))
    ax_l.plot(alpha[i_max], cl[i_max], marker="v", ms=4.4, color=SUB,
              zorder=6)
    ax_l.text(alpha[i_max] - 0.5, cl[i_max] + 0.06,
              f"stall {alpha[i_max]:.1f}$^\\circ$", fontsize=6.5, color=SUB,
              ha="right", va="bottom")
    ax_l.set_xlabel("angle of attack $\\alpha$ (°)", fontsize=7.6)
    ax_l.set_ylabel("$C_L$ (-)", fontsize=7.6)
    ax_l.set_xlim(-6.5, 16.5)
    ax_l.set_ylim(-0.62, 1.72)
    ax_l.tick_params(labelsize=6.5)
    ax_l.set_title("Lift curve", fontsize=8.5, fontweight="bold", pad=3)

    # ================= (d) drag polar (subordinate) ========================
    ax_p.plot(cd * 1e4, cl, color=C_LO, lw=1.4)
    ax_p.plot(cd[i_ld] * 1e4, cl[i_ld], marker="D", ms=3.8, color=SUB,
              zorder=6)
    ax_p.plot(np.interp(ALPHA_DEG, alpha, cd) * 1e4, cl_design, marker="o",
              ms=5.0, mfc="white", mec=ACC, mew=1.3, zorder=7)
    ax_p.plot([0.0, cd[i_ld] * 1e4 * 1.25], [0.0, cl[i_ld] * 1.25],
              color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=2)
    ax_p.text(cd[i_ld] * 1e4 + 12.0, cl[i_ld] - 0.12,
              f"$(L/D)_{{max}}$ = {ld[i_ld]:.1f}", fontsize=6.5, color=SUB,
              ha="left", va="top")
    ax_p.text(np.interp(ALPHA_DEG, alpha, cd) * 1e4 - 14.0, cl_design + 0.10,
              f"$C_D$ = {np.interp(ALPHA_DEG, alpha, cd):.4f}", fontsize=6.5,
              color=ACC, ha="right", va="bottom")
    ax_p.set_xlabel("drag coefficient $C_D$ ($10^{-4}$)", fontsize=7.6)
    ax_p.set_ylabel("$C_L$ (-)", fontsize=7.6)
    ax_p.set_xlim(0.0, 420.0)
    ax_p.set_ylim(-0.62, 1.72)
    ax_p.tick_params(labelsize=6.5)
    ax_p.set_title("Drag polar", fontsize=8.5, fontweight="bold", pad=3)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95], h_pad=2.2, w_pad=2.2)
    panel_labels_inches(fig, [ax_f, ax_c, ax_l, ax_p], dx=-0.34, dy=0.06)
    savefig(fig, "504_aero_airfoil_pressure.png")


if __name__ == "__main__":
    main()

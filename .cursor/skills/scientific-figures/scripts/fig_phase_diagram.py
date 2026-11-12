"""Binary eutectic phase diagram with first- and second-order boundaries.

Panel (a): temperature-composition map of a model A-B alloy.  Single-phase
and two-phase fields are filled with muted colours and labelled in place;
first-order boundaries (liquidus, solidus, solvus, eutectic isotherm) are
solid, the second-order magnetic (Curie) boundary is dashed; the eutectic
three-phase invariant and the two congruent melting points are marked, and
synthetic DTA / EPMA measurements are overlaid with error bars.

Panel (b): lever-rule phase fractions along the dashed alloy section of
panel (a), sharing the temperature axis, showing the discontinuous jump at
the eutectic reaction.

Physics: the liquidus branches follow the ideal-solution relation
ln(a_i) = -dH_i / R (1/T - 1/T_i); the eutectic is their intersection; the
solidus uses constant partition coefficients and the solvus an Arrhenius
solubility.  Phase fractions come from the lever rule, so they are
consistent with the boundaries drawn in panel (a).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from pubstyle import set_pub_style, savefig, add_panel_label

R = 8.314
TA, TB = 1358.0, 1235.0        # melting points of A and B [K]
DHA, DHB = 13000.0, 11300.0    # enthalpies of fusion [J/mol]
KA, KB = 0.25, 0.20            # partition coefficients
Q = 18000.0                    # solvus activation energy [J/mol]
X0 = 0.40                      # alloy section composition
TBOT = 480.0                   # bottom of the diagram [K]

COL = {"L": "#c3d8ea", "a": "#c6ddc9", "b": "#d8cee6",
       "La": "#dbe7e6", "Lb": "#e2dfe9", "ab": "#f2e3cd"}
LINE = "#26374a"


def xl_a(t):
    return 1.0 - np.exp(-DHA / R * (1.0 / t - 1.0 / TA))


def xl_b(t):
    return np.exp(-DHB / R * (1.0 / t - 1.0 / TB))


def kelvin(t):
    return t - 273.15


def main():
    set_pub_style(base_font=8)
    te = brentq(lambda t: xl_a(t) - xl_b(t), 600.0, 1200.0)
    xe = xl_a(te)
    xsa_e, xsb_e = KA * xe, 1.0 - KB * (1.0 - xe)

    def xs_a(t):
        return KA * xl_a(t)

    def xs_b(t):
        return 1.0 - KB * (1.0 - xl_b(t))

    def solv_a(t):
        return xsa_e * np.exp(-Q / R * (1.0 / t - 1.0 / te))

    def solv_b(t):
        return 1.0 - (1.0 - xsb_e) * np.exp(-Q / R * (1.0 / t - 1.0 / te))

    def tc_of_x(x):                      # Curie temperature of the A phase
        return 700.0 - 1500.0 * x        # [K]

    t_cur = brentq(lambda t: tc_of_x(solv_a(t)) - t, TBOT, te)
    x_cur = solv_a(t_cur)

    fig = plt.figure(figsize=(7.2, 4.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.15, 1.0], left=0.075,
                          right=0.975, top=0.90, bottom=0.125, wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[0, 1], sharey=ax)

    ttop = 1420.0
    tg = np.linspace(te, ttop, 500)
    left = np.where(tg <= TA, xl_a(np.clip(tg, None, TA)), 0.0)
    right = np.where(tg <= TB, xl_b(np.clip(tg, None, TB)), 1.0)
    ax.fill_betweenx(kelvin(tg), left, right, color=COL["L"], lw=0)

    t1 = np.linspace(te, TA, 400)
    ax.fill_betweenx(kelvin(t1), xs_a(t1), xl_a(t1), color=COL["La"], lw=0)
    t2 = np.linspace(te, TB, 400)
    ax.fill_betweenx(kelvin(t2), xl_b(t2), xs_b(t2), color=COL["Lb"], lw=0)

    tlo = np.linspace(TBOT, te, 300)
    ax.fill_betweenx(kelvin(t1), 0.0, xs_a(t1), color=COL["a"], lw=0)
    ax.fill_betweenx(kelvin(tlo), 0.0, solv_a(tlo), color=COL["a"], lw=0)
    ax.fill_betweenx(kelvin(t2), xs_b(t2), 1.0, color=COL["b"], lw=0)
    ax.fill_betweenx(kelvin(tlo), solv_b(tlo), 1.0, color=COL["b"], lw=0)
    ax.fill_betweenx(kelvin(tlo), solv_a(tlo), solv_b(tlo), color=COL["ab"],
                     lw=0)

    for t, f in ((t1, xl_a), (t2, xl_b), (t1, xs_a), (t2, xs_b)):
        ax.plot(f(t), kelvin(t), color=LINE, lw=1.15, zorder=4)
    ax.plot(solv_a(tlo), kelvin(tlo), color=LINE, lw=1.15, zorder=4)
    ax.plot(solv_b(tlo), kelvin(tlo), color=LINE, lw=1.15, zorder=4)
    ax.plot([xsa_e, xsb_e], [kelvin(te)] * 2, color=LINE, lw=1.5, zorder=5)

    xc = np.linspace(0.0, x_cur, 100)
    ax.plot(xc, kelvin(tc_of_x(xc)), color="#a33b2b", lw=1.25,
            ls=(0, (5, 2.2)), zorder=5)
    ax.plot([x_cur, 1.0], [kelvin(t_cur)] * 2, color="#a33b2b", lw=1.25,
            ls=(0, (5, 2.2)), zorder=5)
    ax.text(0.62, kelvin(t_cur) - 16, "Curie line  $T_C$  (2nd order)",
            fontsize=6.5, color="#a33b2b", ha="center", va="top")

    # invariant points
    ax.plot([xe], [kelvin(te)], marker="o", ms=6.0, mfc="#d94f2b",
            mec="#2b1a12", mew=0.9, zorder=8)
    for xx, tt, lab in ((0.0, TA, "$T_A$"), (1.0, TB, "$T_B$")):
        ax.plot([xx], [kelvin(tt)], marker="o", ms=4.4, mfc="white",
                mec=LINE, mew=1.0, zorder=8, clip_on=False)
        ax.text(xx + (0.022 if xx == 0 else -0.022), kelvin(tt) + 6, lab,
                fontsize=7.0, color=LINE,
                ha="left" if xx == 0 else "right", va="bottom")
    ax.annotate("eutectic invariant\n"
                r"L $\rightarrow$ $\alpha$ + $\beta$" "\n"
                r"$x_E$ = %.2f,  $T_E$ = %.0f $^\circ$C" % (xe, kelvin(te)),
                xy=(xe, kelvin(te)), xytext=(0.615, 810), fontsize=6.5,
                color="#111418", ha="left", va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.26", fc="white",
                          ec="#b9b9b9", lw=0.5, alpha=0.95),
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#111418",
                                shrinkA=2, shrinkB=5))

    # synthetic measurements on the boundaries
    rng = np.random.default_rng(98)
    pts_t = np.concatenate([np.linspace(te + 40, TA - 40, 7),
                            np.linspace(te + 40, TB - 40, 6),
                            np.linspace(TBOT + 30, te - 30, 4),
                            np.linspace(TBOT + 30, te - 30, 4)])
    pts_x = np.concatenate([xl_a(np.linspace(te + 40, TA - 40, 7)),
                            xl_b(np.linspace(te + 40, TB - 40, 6)),
                            solv_a(np.linspace(TBOT + 30, te - 30, 4)),
                            solv_b(np.linspace(TBOT + 30, te - 30, 4))])
    err = rng.normal(0, 9.0, pts_t.size)
    ax.errorbar(pts_x, kelvin(pts_t + err), yerr=11.0, fmt="o", ms=3.0,
                mfc="white", mec="#111418", mew=0.7, ecolor="#111418",
                elinewidth=0.6, capsize=1.4, zorder=7,
                label="DTA / EPMA (this work)")
    lg = ax.legend(loc="upper right", fontsize=6.4, handlelength=1.2,
                   borderpad=0.35, frameon=True, framealpha=0.92)
    lg.get_frame().set_linewidth(0.5)

    lbl = [(0.56, 1000, "L"), (0.045, 900, r"$\alpha$"),
           (0.955, 900, r"$\beta$"), (0.245, 800, r"L + $\alpha$"),
           (0.80, 690, r"L + $\beta$"), (0.50, 450, r"$\alpha$ + $\beta$")]
    for xx, tt, s in lbl:
        ax.text(xx, tt, s, fontsize=9.0, fontweight="bold", color="#243040",
                ha="center", va="center", style="italic")

    ax.axvline(X0, color="#333333", lw=0.9, ls=(0, (3, 2)), zorder=6)
    ax.text(X0 - 0.014, 1128, "alloy section\n$x_B$ = %.2f" % X0,
            fontsize=6.4, color="#333333", ha="right", va="top", zorder=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(kelvin(TBOT), kelvin(ttop))
    ax.set_xlabel("composition  $x_B$  (mole fraction B)")
    ax.set_ylabel(r"temperature  $T$  ($^\circ$C)")
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#333333")
        s.set_linewidth(0.8)
    ax.tick_params(length=2.6)
    add_panel_label(ax, "(a)", x=-0.085, y=1.005, size=9.5)

    # ---- panel (b): lever-rule phase fractions ---------------------------
    tliq0 = brentq(lambda t: xl_a(t) - X0, te, TA)
    tt = np.linspace(TBOT, ttop, 900)
    fa = np.zeros_like(tt)
    fb = np.zeros_like(tt)
    for i, t in enumerate(tt):
        if t >= tliq0:
            continue
        if t >= te:
            xl, xs = xl_a(t), xs_a(t)
            fa[i] = (xl - X0) / (xl - xs)
        else:
            xa, xb = solv_a(t), solv_b(t)
            fa[i] = (xb - X0) / (xb - xa)
            fb[i] = 1.0 - fa[i]
    fl = 1.0 - fa - fb

    axr.fill_betweenx(kelvin(tt), 0, fa, color=COL["a"], lw=0)
    axr.fill_betweenx(kelvin(tt), fa, fa + fb, color=COL["b"], lw=0)
    axr.fill_betweenx(kelvin(tt), fa + fb, 1.0, color=COL["L"], lw=0)
    axr.plot(fa, kelvin(tt), color=LINE, lw=0.9)
    axr.plot(fa + fb, kelvin(tt), color=LINE, lw=0.9)
    axr.axhline(kelvin(te), color="#d94f2b", lw=1.0, ls=(0, (3, 2)))
    axr.axhline(kelvin(tliq0), color="#333333", lw=0.8, ls=(0, (2, 2)))
    axr.text(0.97, kelvin(tliq0) + 10, "liquidus  %.0f $^\\circ$C"
             % kelvin(tliq0), fontsize=6.2, color="#333333", ha="right",
             va="bottom")
    axr.text(0.03, kelvin(te) - 14, "eutectic reaction", fontsize=6.2,
             color="#a33b2b", ha="left", va="top")
    for xx, tt_, s, c in ((0.17, 700, r"$\alpha$", "#2f5d3a"),
                          (0.80, 380, r"$\beta$", "#4a3a6b"),
                          (0.72, 900, "L", "#27536f")):
        axr.text(xx, tt_, s, fontsize=8.5, fontweight="bold", color=c,
                 ha="center", va="center", style="italic")
    axr.set_xlim(0, 1)
    axr.set_xlabel("phase fraction")
    axr.set_xticks([0, 0.5, 1.0])
    plt.setp(axr.get_yticklabels(), visible=False)
    for s in axr.spines.values():
        s.set_visible(True)
        s.set_color("#333333")
        s.set_linewidth(0.8)
    axr.tick_params(length=2.6)
    add_panel_label(axr, "(b)", x=-0.06, y=1.005, size=9.5)

    savefig(fig, "98_phase_diagram.png")


if __name__ == "__main__":
    main()

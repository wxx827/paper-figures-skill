"""Intact stability: righting arms, the hydrostatic curves behind them, criteria.

The main panel is the righting-arm curve for three loading conditions.  Two
constructions live on it, and both are measured rather than declared: the
metacentric height is drawn as the tangent through the origin, which by
definition meets the one-radian ordinate at GM, and the range of stability is
the zero crossing of the plotted curve, found by interpolation.  The shaded
wedge is the 30 to 40 degree energy band that the criteria police, integrated
by trapezoid over the same array.

The upper companion panel is where the arms come from - displacement, centre
of buoyancy and metacentre against draft, with the loaded waterline marked;
KM minus KG at that draft is the metacentric height the main panel starts
from.  The lower panel scores each condition against six intact-stability
criteria as a ratio to the required value, so the pass line is at 1.0.

Illustrative synthetic data for a 180 m by 32 m hull; not a class submission.
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

RED = "#b23a48"
GOLD = "#c9a227"
GREEN = "#2e7d5b"
STEEL = "#5b6b7a"
BLUE = "#2e74a8"

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.06, 5.26, 4.34)
AUX_T = (7.58, 3.62, 2.94, 1.78)
AUX_B = (7.58, 1.06, 2.94, 1.72)

L_PP, B_MLD = 180.0, 32.0       # length and moulded breadth, m
RHO = 1.025                     # sea water density, t/m^3
CW_I = 0.60                     # waterplane inertia factor, dimensionless
PHI_FLOOD = 48.0                # downflooding angle, deg

# name, metacentric height GM (m), angle of vanishing stability (deg), colour
CONDITIONS = [dict(name="departure, full load", gm=1.45, phv=62.0, c=BLUE,
                   draft=11.4),
              dict(name="arrival, 10 % consumables", gm=1.10, phv=55.0,
                   c=GREEN, draft=10.2),
              dict(name="ballast, slack tanks", gm=0.78, phv=47.0, c=GOLD,
                   draft=7.6)]

# label, required value, unit, key
CRITERIA = [("area 0-30 deg", 0.055, "m rad", "a30"),
            ("area 0-40 deg", 0.090, "m rad", "a40"),
            ("area 30-40 deg", 0.030, "m rad", "a3040"),
            ("GZ at 30 deg", 0.200, "m", "gz30"),
            ("angle of max GZ", 25.0, "deg", "phmax"),
            ("range of stability", 30.0, "deg", "phv")]


def gz_curve(gm, phi_v_deg, phi_deg):
    """Righting arm (m) from a two-term Fourier form pinned to GM and phi_v."""
    pv = np.radians(phi_v_deg)
    c2 = gm / (4.0 * np.sin(pv) ** 2)
    c1 = gm - 3.0 * c2
    p = np.radians(phi_deg)
    return c1 * np.sin(p) + c2 * np.sin(3.0 * p)


def area(phi_deg, gz, lo, hi):
    """Area under the plotted GZ curve between two heel angles, m rad."""
    m = (phi_deg >= lo) & (phi_deg <= hi)
    return float(np.trapezoid(gz[m], np.radians(phi_deg[m])))


def hydrostatics(draft):
    """Displacement (t), KB (m), BM (m), KM (m) at the given draft."""
    cb = 0.700 + 0.020 * (draft - 8.0) / 4.0
    volume = cb * L_PP * B_MLD * draft
    disp = RHO * volume
    kb = draft * (0.833 - 0.166 * cb)
    inertia = CW_I * L_PP * B_MLD ** 3 / 12.0
    bm = inertia / volume
    return disp, kb, bm, kb + bm


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

    phi = np.linspace(0.0, 75.0, 1501)
    curves, scores = [], []
    for cd in CONDITIONS:
        gz = gz_curve(cd["gm"], cd["phv"], phi)
        i_max = int(np.argmax(gz))
        pos = gz > 0.0
        i_end = int(np.argmax(~pos & (phi > 10.0)))
        phi_v = float(np.interp(0.0, [gz[i_end], gz[i_end - 1]],
                                [phi[i_end], phi[i_end - 1]]))
        s = dict(a30=area(phi, gz, 0.0, 30.0),
                 a40=area(phi, gz, 0.0, 40.0),
                 a3040=area(phi, gz, 30.0, 40.0),
                 gz30=float(np.interp(30.0, phi, gz)),
                 phmax=float(phi[i_max]), gm=cd["gm"],
                 gzmax=float(gz[i_max]), phv=phi_v)
        curves.append(gz)
        scores.append(s)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN)
    ax_h = axes_inches(fig, AUX_T)
    ax_c = axes_inches(fig, AUX_B)
    for a in (ax, ax_h, ax_c):
        a.set_axisbelow(True)
        a.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)
        style_axes(a)

    # ---- main: the righting-arm curves -------------------------------------
    ax.axhline(0.0, color=FAINT, lw=0.7, zorder=1)
    for cd, gz, s in zip(CONDITIONS, curves, scores):
        ax.plot(phi, gz, color=cd["c"], lw=2.0, zorder=6,
                label="%s, GM = %.2f m" % (cd["name"], cd["gm"]))
        ax.plot([s["phmax"]], [s["gzmax"]], marker="o", ms=4.4, mfc="white",
                mec=cd["c"], mew=1.3, zorder=9)
        ax.text(s["phmax"] + 1.8, s["gzmax"] + 0.012,
                "%.3f m at %.1f deg" % (s["gzmax"], s["phmax"]),
                ha="left", va="bottom", fontsize=6.4, color=cd["c"],
                zorder=10)
        ax.plot([s["phv"]], [0.0], marker="v", ms=4.6, color=cd["c"],
                zorder=9)

    tan_cd = CONDITIONS[2]
    gz_ref, sref = curves[1], scores[1]
    ax.plot([0.0, np.degrees(1.0)], [0.0, tan_cd["gm"]], color=INK, lw=1.0,
            ls=(0, (5.0, 2.4)), zorder=7)
    ax.plot([np.degrees(1.0)], [tan_cd["gm"]], marker="s", ms=4.2,
            mfc="white", mec=INK, mew=1.3, zorder=9)
    ax.text(np.degrees(1.0) - 5.0, tan_cd["gm"] - 0.012,
            "tangent through the origin:\nGZ = GM = %.2f m at 1 rad "
            "(%.1f deg)" % (tan_cd["gm"], np.degrees(1.0)),
            ha="right", va="top", fontsize=6.7, color=INK, linespacing=1.45,
            zorder=11)

    band = (phi >= 30.0) & (phi <= 40.0)
    ax.fill_between(phi[band], 0.0, gz_ref[band], color=GREEN, alpha=0.22,
                    zorder=4)
    ax.annotate("30-40 deg energy band\n%.4f m rad by trapezoid"
                % sref["a3040"],
                xy=(35.0, float(np.interp(35.0, phi, gz_ref)) * 0.45),
                xytext=(0.305, 0.255), textcoords="axes fraction",
                fontsize=6.6, color=GREEN, ha="left", va="center",
                linespacing=1.45, zorder=11,
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=0.9,
                                shrinkA=2, shrinkB=3))

    ax.axvline(PHI_FLOOD, color=RED, lw=1.1, ls=(0, (4.0, 2.2)), zorder=5)
    ax.text(PHI_FLOOD - 1.0, 0.045, "downflooding angle %.0f deg" % PHI_FLOOD,
            transform=ax.get_xaxis_transform(), rotation=90, ha="right",
            va="bottom", fontsize=6.4, color=RED, zorder=6)

    ax.text(0.985, 0.985,
            "triangles: angle of vanishing stability, %.1f / %.1f / %.1f deg"
            % tuple(s["phv"] for s in scores), transform=ax.transAxes,
            ha="right", va="top", fontsize=6.5, color=SUB, zorder=11)

    ax.set_xlim(0.0, 75.0)
    ax.set_ylim(-0.17, 0.95)
    ax.tick_params(labelsize=7.2)
    ax.set_xlabel("heel angle $\\varphi$ (deg)", fontsize=8.4, labelpad=1.5)
    ax.set_ylabel("righting arm GZ (m)", fontsize=8.4)
    ax.set_title("Righting arms for three loading conditions",
                 fontsize=9.2, fontweight="bold", pad=5.0)
    ax.legend(loc="upper left", frameon=False, fontsize=6.9,
              handlelength=1.7, labelspacing=0.34, borderaxespad=0.35)

    # ---- aux (b): hydrostatic curves against draft -------------------------
    drafts = np.linspace(4.0, 12.5, 300)
    disp, kb, bm, km = hydrostatics(drafts)
    ax_h.plot(disp / 1000.0, drafts, color=BLUE, lw=1.9, zorder=6)
    ax_h.set_xlim(0.0, float(disp.max()) / 1000.0 * 1.06)
    ax_h.set_ylim(4.0, 12.5)
    ax_h.tick_params(labelsize=6.8)
    ax_h.set_xlabel("displacement $\\Delta$ (1000 t)", fontsize=7.5,
                    labelpad=1.5, color=BLUE)
    ax_h.tick_params(axis="x", labelcolor=BLUE)
    ax_h.set_ylabel("draft T (m, up positive)", fontsize=7.5)

    ax_h2 = ax_h.twiny()
    ax_h2.plot(km, drafts, color=STEEL, lw=1.7, zorder=6)
    ax_h2.plot(kb, drafts, color=GOLD, lw=1.7, ls=(0, (4.0, 2.0)), zorder=6)
    ax_h2.set_xlim(0.0, float(km.max()) * 1.10)
    ax_h2.set_ylim(4.0, 12.5)
    ax_h2.spines["right"].set_visible(False)
    ax_h2.tick_params(labelsize=6.8, length=3)
    ax_h2.set_xlabel("KM and KB above keel (m)", fontsize=7.5, labelpad=2.0)
    ax_h2.grid(False)
    ax_h2.text(km[0] - 0.7, 4.35, "KM", fontsize=6.6, color=STEEL,
               ha="right", va="bottom", zorder=8)
    ax_h2.text(kb[0] + 0.7, 4.35, "KB", fontsize=6.6, color=GOLD,
               ha="left", va="bottom", zorder=8)
    ax_h.text(float(disp[-1]) / 1000.0 * 0.72, 11.9, "$\\Delta$",
              fontsize=6.6, color=BLUE, ha="right", va="top", zorder=8)

    t_op = CONDITIONS[0]["draft"]
    d_op, kb_op, bm_op, km_op = hydrostatics(t_op)
    ax_h.axhline(t_op, color=RED, lw=1.0, ls=(0, (3.5, 2.0)), zorder=7)
    ax_h.text(0.030, t_op - 0.22,
              "loaded waterline T = %.1f m\n$\\Delta$ = %.0f t, KM = %.2f m"
              % (t_op, d_op, km_op),
              transform=ax_h.get_yaxis_transform(), ha="left", va="top",
              fontsize=6.2, color=RED, linespacing=1.4, zorder=8)

    # ---- aux (c): criteria scorecard ---------------------------------------
    n_c = len(CRITERIA)
    y_base = np.arange(n_c)[::-1]
    h_bar = 0.24
    for j, (cd, s) in enumerate(zip(CONDITIONS, scores)):
        ratio = [s[k] / req for _, req, _, k in CRITERIA]
        ax_c.barh(y_base + (1 - j) * h_bar, ratio, height=h_bar * 0.86,
                  color=cd["c"], edgecolor="white", linewidth=0.5, zorder=5)
        for yv, rv in zip(y_base + (1 - j) * h_bar, ratio):
            ax_c.text(min(rv, 3.35) + 0.06, yv, "%.2f" % rv, ha="left",
                      va="center", fontsize=6, color=SUB, zorder=7)
    ax_c.axvline(1.0, color=RED, lw=1.2, zorder=6)
    ax_c.text(1.06, n_c - 0.42, "required", fontsize=6.3, color=RED,
              ha="left", va="center", zorder=7)
    ax_c.set_yticks(y_base)
    ax_c.set_yticklabels(["%s  (%.3g %s)" % (lab, req, unit)
                          for lab, req, unit, _ in CRITERIA], fontsize=6.0)
    ax_c.set_xlim(0.0, 3.6)
    ax_c.set_ylim(-0.55, n_c - 0.35)
    ax_c.tick_params(labelsize=6.8)
    ax_c.set_xlabel("achieved / required (dimensionless)", fontsize=7.5,
                    labelpad=1.5)
    ax_c.grid(axis="y", visible=False)
    ax_c.set_title("Every condition clears the pass line", fontsize=8.5,
                   fontweight="bold", pad=4.0)

    panel_labels_inches(fig, [ax, ax_c], labels=["(a)", "(c)"], dx=-0.66,
                        dy=0.10)
    panel_labels_inches(fig, [ax_h], labels=["(b)"], dx=-0.66, dy=0.56)
    fig.text((AUX_T[0] + AUX_T[2] / 2.0) / FIG_W,
             (AUX_T[1] + AUX_T[3] + 0.46) / FIG_H,
             "Hydrostatic curves feed GM = KM $-$ KG", ha="center",
             va="bottom", fontsize=8.5, fontweight="bold")
    foot = add_footer(fig, "Synthetic data, figure 998")
    check_layout(fig, foot, [ax, ax_h, ax_c])
    savefig(fig, "998_shp_hydrostatics_gz.png")


if __name__ == "__main__":
    main()

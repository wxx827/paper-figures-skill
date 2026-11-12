"""Figure 412 - Hydrogeological cross-section with a pumping cone of depression.

Five units are drawn with the conventional ornament for their lithology, so the
section can be read without the key: coarse circles for sand and gravel, ruling
for the silty-clay aquitard, stipple for the confined sandstone, hachure for the
fractured limestone and crosses for the crystalline basement.

The water table, the flow arrows and the drawdown curve underneath are not three
separate drawings.  All three come from one head field - a regional gradient
towards the river minus a Thiem cone around the abstraction well - so the arrows
reverse exactly where the cone starts to dominate the regional gradient, and the
curve in panel (c) is that same cone plotted against the same distance axis.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle

from pubstyle import set_pub_style, savefig, lighten, darken, COLORS
from figkit import add_scale_bar, panel_label_inches, halo_text

X0, X1 = 0.0, 1200.0                # section length, m
Z0, Z1 = -70.0, 55.0                # elevation range, m a.s.l.
X_WELL, X_LOG = 760.0, 300.0        # abstraction well and the reference log
Q_WELL = 900.0                      # abstraction rate, m3/d
R_INFL, R_WELL = 420.0, 0.30        # radius of influence and well radius, m
H_UP, H_RIV, X_RIV = 38.0, 20.5, 1150.0

# Lithology ornament and hydraulic properties (domain convention,
# DESIGN_SYSTEM section 2.2).  K is in m/d; the millidarcy column is derived
# from it at run time, never typed in.
UNITS = [
    dict(key="sand", name="Alluvial sand and gravel (unconfined aquifer)",
         fc=lighten("#c9a227", 0.54), hatch="oo", k=12.0),
    dict(key="clay", name="Silty clay (aquitard)",
         fc=lighten("#2e7d5b", 0.72), hatch="---", k=3.0e-4),
    dict(key="sst", name="Sandstone (confined aquifer)",
         fc=lighten("#c0603f", 0.58), hatch="...", k=4.5),
    dict(key="lst", name="Fractured limestone (karst aquifer)",
         fc=lighten("#3b5b92", 0.68), hatch="\\\\", k=8.0),
    dict(key="base", name="Crystalline basement (aquiclude)",
         fc=lighten("#3a3a3a", 0.62), hatch="++", k=1.0e-5),
]
WATER = "#3d6b8e"
MU, RHO_G, DARCY = 1.002e-3, 9789.0, 9.869e-13     # Pa s, N/m3, m2 per darcy


def k_to_md(k_m_per_day):
    """Hydraulic conductivity (m/d) -> intrinsic permeability (mD)."""
    k_si = k_m_per_day / 86400.0                    # m/s
    return k_si * MU / RHO_G / DARCY * 1000.0


def fmt_md(md):
    """Millidarcy across seven decades, without rounding an aquitard to zero."""
    if md >= 100.0:
        return f"{md:,.0f}".replace(",", "\u2009")
    return f"{md:.2f}" if md >= 1.0 else f"{md:.3g}"


def surfaces(x):
    """Ground surface and the four unit contacts, all in m a.s.l."""
    zs = (44.0 - 20.0 * (x / X1) + 3.0 * np.sin(2 * np.pi * x / 520.0)
          + 1.4 * np.sin(2 * np.pi * x / 170.0 + 0.7))
    z1 = zs - (15.0 + 4.0 * np.sin(2 * np.pi * x / 700.0 + 0.3))
    z2 = z1 - (11.0 + 3.0 * np.sin(2 * np.pi * x / 430.0 + 1.1))
    z3 = z2 - (24.0 + 6.0 * np.sin(2 * np.pi * x / 900.0 + 0.4))
    z4 = z3 - (20.0 + 5.0 * np.sin(2 * np.pi * x / 610.0 + 2.2))
    return zs, z1, z2, z3, z4


def regional_head(x):
    """Water table with no abstraction: Dupuit parabola towards the river."""
    f = np.clip(x / X_RIV, 0.0, 1.0)
    return np.sqrt(H_UP ** 2 + (H_RIV ** 2 - H_UP ** 2) * f)


def drawdown(x, transmissivity):
    """Thiem cone about the well, zero beyond the radius of influence."""
    r = np.maximum(np.abs(x - X_WELL), R_WELL)
    s = Q_WELL / (2 * np.pi * transmissivity) * np.log(R_INFL / r)
    return np.clip(s, 0.0, None)


def main():
    set_pub_style(base_font=8, **{"hatch.linewidth": 0.45})
    x = np.linspace(X0, X1, 900)
    zs, z1, z2, z3, z4 = surfaces(x)

    k_sand = UNITS[0]["k"]
    b_sand = float(np.mean(zs - z1))                  # saturated thickness, m
    trans = k_sand * b_sand                           # transmissivity, m2/d
    h_reg = regional_head(x)
    s_x = drawdown(x, trans)
    h_wt = np.minimum(h_reg - s_x, zs - 0.6)

    # Confined head: above the aquitard, so the sandstone really is confined.
    h_conf = 34.0 - 9.0 * (x / X1) + 0.8 * np.sin(2 * np.pi * x / 640.0)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.235],
                          height_ratios=[1.0, 0.40],
                          left=0.058, right=0.988, top=0.868, bottom=0.150,
                          wspace=0.055, hspace=0.16)
    ax = fig.add_subplot(gs[0, 0])
    ax_k = fig.add_subplot(gs[0, 1], sharey=ax)
    ax_s = fig.add_subplot(gs[1, 0], sharex=ax)

    # ---- (a) the layer stack ----------------------------------------------
    tops = [zs, z1, z2, z3, z4]
    bots = [z1, z2, z3, z4, np.full_like(x, Z0)]
    for u, top, bot in zip(UNITS, tops, bots):
        ax.fill_between(x, bot, top, facecolor=u["fc"], hatch=u["hatch"],
                        edgecolor=darken(u["fc"], 0.45), linewidth=0.6,
                        zorder=2)

    # Saturated part of the unconfined aquifer, tinted but not hidden.
    ax.fill_between(x, z1, h_wt, where=h_wt > z1, facecolor=WATER, alpha=0.18,
                    lw=0.0, zorder=3)

    # River in the topographic low at the right-hand end.
    riv = x >= X_RIV - 55.0
    ax.fill_between(x[riv], zs[riv], np.maximum(zs[riv], H_RIV + 1.2),
                    facecolor=WATER, alpha=0.85, lw=0.0, zorder=5)
    halo_text(ax, X_RIV - 26.0, H_RIV + 5.0, "river", fontsize=6.6,
              ha="center", va="bottom", color="white", halo=darken(WATER, 0.4))

    # ---- water table, piezometric surface ----------------------------------
    ax.plot(x, h_wt, color=WATER, lw=1.5, zorder=7)
    ax.plot(x, h_conf, color=WATER, lw=1.2, ls=(0, (5, 2.5)), zorder=7)
    # Both surfaces are named in the shared key at the foot of the figure;
    # repeating the names on the section only crowds the flow arrows.
    ax.plot([70.0], [np.interp(70.0, x, h_wt)], marker="v", ms=5.0,
            mfc="white", mec=WATER, mew=1.0, zorder=8)

    # ---- flow arrows, direction taken from the head field itself -----------
    dh = np.gradient(h_wt, x)
    for xa in np.arange(60.0, 1130.0, 92.0):
        ha_ = np.interp(xa, x, h_wt)
        zb = np.interp(xa, x, z1)
        za = zb + 0.42 * (ha_ - zb)
        g = np.interp(xa, x, dh)
        if abs(g) < 1e-5:
            continue
        d = -np.sign(g) * 44.0
        ax.annotate("", xy=(xa + d, za), xytext=(xa - d * 0.15, za),
                    zorder=8,
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color=WATER,
                                    shrinkA=0, shrinkB=0, mutation_scale=8))
    # Downward leakage through the aquitard, the reason the cone is shallow.
    for xa in (430.0, 620.0, 900.0):
        zt, zb = np.interp(xa, x, z1), np.interp(xa, x, z2)
        ax.annotate("", xy=(xa, zb + 1.0), xytext=(xa, zt - 1.0), zorder=8,
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color=WATER,
                                    ls=(0, (2, 1.6)), shrinkA=0, shrinkB=0,
                                    mutation_scale=7))

    # ---- abstraction well ---------------------------------------------------
    zs_w = float(np.interp(X_WELL, x, zs))
    z1_w = float(np.interp(X_WELL, x, z1))
    ax.plot([X_WELL, X_WELL], [z1_w - 2.0, zs_w + 7.0], color=COLORS["ink"],
            lw=2.2, solid_capstyle="butt", zorder=9)
    ax.plot([X_WELL, X_WELL], [z1_w - 1.0, z1_w + 9.0], color="white", lw=1.0,
            zorder=10)
    ax.plot([X_WELL, X_WELL], [z1_w - 1.0, z1_w + 9.0], color=COLORS["ink"],
            lw=1.0, ls=(0, (1, 1)), zorder=11)
    ax.add_patch(Rectangle((X_WELL - 26.0, zs_w + 7.0), 52.0, 5.0,
                           facecolor=COLORS["ink"], edgecolor="none",
                           zorder=9))
    ax.annotate(f"abstraction well\nQ = {Q_WELL:.0f} m$^3$/d",
                xy=(X_WELL, zs_w + 12.0), xytext=(X_WELL - 175.0, Z1 - 4.5),
                fontsize=6.8, color=COLORS["ink"], ha="right", va="center",
                linespacing=1.35, zorder=12,
                arrowprops=dict(arrowstyle="->", lw=0.7,
                                color=COLORS["rule"], shrinkA=1, shrinkB=2))
    ax.text(X_WELL + 12.0, z1_w + 4.0, "screen", fontsize=6.2,
            color=COLORS["ink"], ha="left", va="center", zorder=12)

    for xr, tag in ((X_WELL - R_INFL, "R"), (X_WELL + R_INFL, None)):
        ax.plot([xr, xr], [Z0, np.interp(xr, x, zs)], color=COLORS["baseline"],
                lw=0.7, ls=(0, (4, 3)), zorder=6)
    ax.annotate("", xy=(X_WELL + R_INFL, Z1 - 13.0),
                xytext=(X_WELL - R_INFL, Z1 - 13.0), zorder=8,
                arrowprops=dict(arrowstyle="<|-|>", lw=0.8,
                                color=COLORS["baseline"], shrinkA=0,
                                shrinkB=0, mutation_scale=7))
    ax.text(X_WELL, Z1 - 11.5, f"radius of influence 2 x {R_INFL:.0f} m",
            fontsize=6.4, color=COLORS["sub"], ha="center", va="bottom")

    # ---- reference borehole for the K log ----------------------------------
    ax.plot([X_LOG, X_LOG], [np.interp(X_LOG, x, z4) - 6.0,
                             np.interp(X_LOG, x, zs) + 4.0],
            color=COLORS["rule"], lw=1.0, zorder=9)
    ax.text(X_LOG, np.interp(X_LOG, x, zs) + 6.0, "BH-1", fontsize=6.4,
            color=COLORS["rule"], ha="center", va="bottom", zorder=9)

    # ---- unit labels with their conductivity -------------------------------
    lab_x = [175.0, 560.0, 640.0, 250.0, 640.0]
    # The unconfined aquifer is the one unit whose middle is busy with the
    # water table and the flow arrows, so its label goes in the vadose zone.
    lab_y = [None, None, None, None, None]
    lab_y[0] = 0.5 * (float(np.interp(175.0, x, zs))
                      + float(np.interp(175.0, x, h_wt)))
    for u, top, bot, lx, ly in zip(UNITS, tops, bots, lab_x, lab_y):
        zt, zb = np.interp(lx, x, top), np.interp(lx, x, bot)
        halo_text(ax, lx, 0.5 * (zt + zb) if ly is None else ly,
                  f"K = {u['k']:g} m/d  ({fmt_md(k_to_md(u['k']))} mD)",
                  fontsize=6.2, ha="center", va="center", color="#1a1a1a",
                  halo="white", halo_lw=2.0, zorder=6)

    ax.set_xlim(X0, X1)
    ax.set_ylim(Z0, Z1)
    ax.set_ylabel("elevation (m a.s.l.)", fontsize=8.0)
    ax.tick_params(labelsize=7.0, labelbottom=False)
    ax.set_title("Hydrogeological section along the valley, looking north",
                 fontsize=8.8, fontweight="bold", pad=4.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    add_scale_bar(ax, 200.0, "200 m", loc=(0.035, 0.055), height=0.024,
                  fontsize=6.4, n_seg=2)

    # ---- (b) conductivity log at BH-1 --------------------------------------
    for u, top, bot in zip(UNITS, tops, bots):
        zt = float(np.interp(X_LOG, x, top))
        zb = max(float(np.interp(X_LOG, x, bot)), Z0)
        ax_k.barh(0.5 * (zt + zb), u["k"], height=zt - zb, left=2.0e-6,
                  facecolor=u["fc"], edgecolor=darken(u["fc"], 0.45),
                  linewidth=0.6, zorder=3)
    ax_k.set_xscale("log")
    ax_k.set_xlim(2.0e-6, 3.0e2)
    ax_k.set_xticks([1e-5, 1e-3, 1e-1, 1e1])
    ax_k.set_xlabel("hydraulic conductivity\nK at BH-1 (m/d)", fontsize=7.4,
                    linespacing=1.3)
    ax_k.tick_params(axis="x", labelsize=6.6)
    ax_k.tick_params(axis="y", labelleft=False, length=0)
    ax_k.grid(axis="x", linewidth=0.5, alpha=0.7)
    ax_k.set_title("K log", fontsize=8.8, fontweight="bold", pad=4.0)
    for s in ("top", "right", "left"):
        ax_k.spines[s].set_visible(False)

    # ---- (c) the cone of depression, on the section's own distance axis ----
    ax_s.fill_between(x, 0.0, s_x, color=lighten(WATER, 0.72), lw=0.0,
                      zorder=2)
    ax_s.plot(x, s_x, color=WATER, lw=1.3, zorder=3)
    ax_s.axhline(0.0, color=COLORS["baseline"], lw=0.8, zorder=4)
    for xr in (X_WELL - R_INFL, X_WELL + R_INFL):
        ax_s.axvline(xr, color=COLORS["baseline"], lw=0.7, ls=(0, (4, 3)),
                     zorder=4)
    s_max = float(drawdown(np.array([X_WELL]), trans)[0])
    ax_s.annotate(f"{s_max:.1f} m at the well face",
                  xy=(X_WELL, s_max), xytext=(X_WELL + 120.0, s_max * 0.86),
                  fontsize=6.6, color=darken(WATER, 0.25), ha="left",
                  va="center",
                  arrowprops=dict(arrowstyle="->", lw=0.7, color=WATER,
                                  shrinkA=0, shrinkB=2))
    ax_s.set_ylim(s_max * 1.22, -0.45)
    ax_s.set_xlabel("distance along section (m)", fontsize=8.0)
    ax_s.set_ylabel("drawdown (m)", fontsize=8.0)
    ax_s.tick_params(labelsize=7.0)
    ax_s.grid(axis="y", linewidth=0.5, alpha=0.65)
    ax_s.text(0.985, 0.72, f"Thiem cone, T = K b = {trans:.0f} m$^2$/d "
              f"(b = {b_sand:.1f} m)", transform=ax_s.transAxes, fontsize=6.4,
              color=COLORS["sub"], ha="right", va="top")
    for s in ("top", "right"):
        ax_s.spines[s].set_visible(False)

    # ---- one shared key ----------------------------------------------------
    handles = [Patch(facecolor=u["fc"], hatch=u["hatch"],
                     edgecolor=darken(u["fc"], 0.45), linewidth=0.6,
                     label=u["name"]) for u in UNITS]
    handles += [Line2D([0], [0], color=WATER, lw=1.5, label="water table"),
                Line2D([0], [0], color=WATER, lw=1.2, ls=(0, (5, 2.5)),
                       label="piezometric surface"),
                Line2D([0], [0], color=WATER, lw=1.0, marker=">", ms=3.5,
                       label="groundwater flow")]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.048), ncol=4, frameon=False,
               fontsize=6.8, handlelength=1.6, handleheight=1.15,
               columnspacing=1.4)

    panel_label_inches(fig, ax, "(a)", dx=-0.48, dy=0.09)
    panel_label_inches(fig, ax_k, "(b)", dx=-0.14, dy=0.09)
    panel_label_inches(fig, ax_s, "(c)", dx=-0.48, dy=0.09)

    bb = ax.get_position()
    fw, fh = fig.get_size_inches()
    ve = ((bb.height * fh) / (Z1 - Z0)) / ((bb.width * fw) / (X1 - X0))
    # Regional flow runs towards the river; the cone only turns it round on
    # the far side of the well.  Measure that stretch instead of asserting it.
    rev = x[(dh > 0.0) & (x > X_WELL)]
    l_rev = float(rev.max() - rev.min()) if rev.size else 0.0

    savefig(fig, "412_geo_hydrogeology.png")


if __name__ == "__main__":
    main()

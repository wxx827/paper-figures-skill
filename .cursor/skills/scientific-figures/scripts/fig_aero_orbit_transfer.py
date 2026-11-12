"""Two-impulse Hohmann transfer from a parking orbit to geostationary altitude.

The main panel is drawn in the equatorial plane of an Earth-centred inertial
frame at true scale, so the transfer ellipse really is an ellipse and the two
orbits really are that far apart.  Both burns are sized from the vis-viva
equation and the arrows are drawn tangentially at the point where they are
applied; the subordinate column carries the resulting delta-v budget and the
event timeline, both filled from the same numbers.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

C_PARK = "#3b5b92"     # house blue: parking orbit, nominal
C_TRAN = "#c9a227"     # house gold: transfer ellipse
C_TGT = "#2e7d5b"      # house green: target orbit
ACC = "#b23a48"        # house red: the two burns
INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"

MU = 398600.4418       # Earth gravitational parameter (km3/s2)
R_E = 6378.14          # equatorial radius (km)
H_PARK = 420.0         # parking-orbit altitude (km)
R_GEO = 42164.0        # geostationary radius (km)
DI_DEG = 0.0           # coplanar transfer; plane change is budgeted separately
DV_PLANE = 0.181       # residual inclination trim at apogee (km/s)
DV_MARGIN = 0.120      # navigation and finite-burn losses (km/s)


def vis_viva(r, a):
    """Orbital speed (km/s) at radius ``r`` on an orbit of semi-major axis ``a``."""
    return np.sqrt(MU * (2.0 / r - 1.0 / a))


def ellipse_xy(a, e, nu):
    """Perifocal coordinates (km) of a conic at true anomaly ``nu``."""
    r = a * (1.0 - e ** 2) / (1.0 + e * np.cos(nu))
    return r * np.cos(nu), r * np.sin(nu)


def main():
    set_pub_style(base_font=9)

    r1 = R_E + H_PARK
    r2 = R_GEO
    a_t = 0.5 * (r1 + r2)
    e_t = (r2 - r1) / (r2 + r1)

    v1 = vis_viva(r1, r1)
    v_peri = vis_viva(r1, a_t)
    v_apo = vis_viva(r2, a_t)
    v2 = vis_viva(r2, r2)
    dv1 = v_peri - v1
    dv2 = v2 - v_apo
    dv_total = dv1 + dv2 + DV_PLANE + DV_MARGIN
    t_transfer = np.pi * np.sqrt(a_t ** 3 / MU)          # seconds
    t_park = 2.0 * np.pi * np.sqrt(r1 ** 3 / MU)
    t_geo = 2.0 * np.pi * np.sqrt(r2 ** 3 / MU)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.55],
                          height_ratios=[1.0, 0.72], hspace=0.42, wspace=0.10)
    ax = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, 1])

    # ===================== (a) equatorial-plane geometry ===================
    nu = np.linspace(0.0, 2.0 * np.pi, 721)
    ax.add_patch(Circle((0.0, 0.0), R_E, facecolor="#c8d5e4",
                        edgecolor="#5f7c9e", lw=0.9, zorder=4))
    ax.text(0.0, 0.0, "Earth", fontsize=7.4, color="#2c4570", ha="center",
            va="center", zorder=5)

    ax.plot(r1 * np.cos(nu), r1 * np.sin(nu), color=C_PARK, lw=1.6, zorder=6)
    ax.plot(r2 * np.cos(nu), r2 * np.sin(nu), color=C_TGT, lw=1.6, zorder=6)

    nu_t = np.linspace(0.0, np.pi, 400)
    xt, yt = ellipse_xy(a_t, e_t, nu_t)
    ax.plot(xt, yt, color=C_TRAN, lw=2.0, zorder=7)
    xt2, yt2 = ellipse_xy(a_t, e_t, np.linspace(np.pi, 2.0 * np.pi, 400))
    ax.plot(xt2, yt2, color=C_TRAN, lw=1.0, ls=(0, (4, 3)), alpha=0.55,
            zorder=5)
    ax.fill(np.concatenate([xt, [0.0]]), np.concatenate([yt, [0.0]]),
            color=C_TRAN, alpha=0.07, lw=0, zorder=3)

    # Apsidal line, drawn outside the globe so it does not cross the label.
    ax.plot([-R_E, -r2], [0.0, 0.0], color=MUTED, lw=0.7, ls=(0, (5, 3)),
            zorder=5)
    ax.plot([r1], [0.0], marker="o", ms=6.0, mfc="white", mec=ACC, mew=1.6,
            zorder=10)
    ax.plot([-r2], [0.0], marker="o", ms=6.0, mfc="white", mec=ACC, mew=1.6,
            zorder=10)

    arrow_kw = dict(arrowstyle="-|>", color=ACC, lw=1.6, shrinkA=0.0,
                    shrinkB=0.0, mutation_scale=13)
    ax.annotate("", xy=(r1, 12500.0), xytext=(r1, 1800.0),
                arrowprops=arrow_kw, zorder=11)
    ax.annotate("", xy=(-r2, -11500.0), xytext=(-r2, -1800.0),
                arrowprops=arrow_kw, zorder=11)
    ax.text(r1 + 1600.0, 13600.0,
            f"burn 1  (perigee)\n$\\Delta v_1$ = {dv1:.3f} km/s\n"
            f"{v1:.3f} $\\rightarrow$ {v_peri:.3f} km/s",
            fontsize=7.0, color=ACC, ha="left", va="bottom", zorder=12)
    ax.text(-r2 + 1600.0, -13000.0,
            f"burn 2  (apogee)\n$\\Delta v_2$ = {dv2:.3f} km/s\n"
            f"{v_apo:.3f} $\\rightarrow$ {v2:.3f} km/s",
            fontsize=7.0, color=ACC, ha="left", va="top", zorder=12)

    ax.text(r1 + 2200.0, -3600.0, f"parking orbit\n{H_PARK:.0f} km circular",
            fontsize=7.0, color=C_PARK, ha="left", va="top", zorder=12)
    ax.text(0.0, r2 + 1400.0, "geostationary orbit,  42 164 km",
            fontsize=7.0, color=C_TGT, ha="center", va="bottom", zorder=12)
    ax.text(-9000.0, 21000.0,
            f"transfer ellipse\n$a$ = {a_t:,.0f} km,  $e$ = {e_t:.3f}"
            .replace(",", " "),
            fontsize=7.0, color="#8a6f18", ha="center", va="center", zorder=12)

    lim = 5.1e4
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("ECI $x$ (km)")
    ax.set_ylabel("ECI $y$ (km)")
    ax.set_xticks(np.arange(-40000, 40001, 20000))
    ax.set_yticks(np.arange(-40000, 40001, 20000))
    ax.grid(True, lw=0.5, alpha=0.45)
    ax.set_title("Equatorial plane, true scale", fontsize=9.0,
                 fontweight="bold", pad=4)

    # A 420 km parking orbit is indistinguishable from the limb at GEO scale,
    # so the departure geometry only exists in a magnified inset.
    axi = ax.inset_axes([0.605, 0.045, 0.375, 0.295])
    axi.set_zorder(8)
    axi.patch.set_visible(True)
    axi.add_patch(Circle((0.0, 0.0), R_E, facecolor="#c8d5e4",
                         edgecolor="#5f7c9e", lw=0.9, zorder=2))
    axi.plot(r1 * np.cos(nu), r1 * np.sin(nu), color=C_PARK, lw=1.8, zorder=4)
    axi.plot(xt, yt, color=C_TRAN, lw=2.0, zorder=5)
    axi.plot(xt2, yt2, color=C_TRAN, lw=1.0, ls=(0, (4, 3)), alpha=0.6,
             zorder=3)
    axi.plot([r1], [0.0], marker="o", ms=5.0, mfc="white", mec=ACC, mew=1.4,
             zorder=7)
    axi.annotate("", xy=(r1, 1750.0), xytext=(r1, 260.0),
                 arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.4,
                                 shrinkA=0.0, shrinkB=0.0, mutation_scale=9),
                 zorder=7)
    axi.plot([R_E, r1], [-620.0, -620.0], color=SUB, lw=0.7, zorder=6)
    axi.text(0.5 * (R_E + r1), -900.0, f"{H_PARK:.0f} km", fontsize=6.0,
             color=SUB, ha="center", va="top", zorder=7)
    axi.plot([4250.0, 5250.0], [1780.0, 1780.0], color=RULE, lw=1.4, zorder=7)
    axi.text(4750.0, 1900.0, "1000 km", fontsize=6.0, color=RULE,
             ha="center", va="bottom", zorder=7)
    axi.set_xlim(3900.0, 9300.0)
    axi.set_ylim(-2140.0, 2140.0)
    axi.set_aspect("equal")
    axi.set_xticks([])
    axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
        sp.set_color("#8a8a8a")
    axi.text(0.03, 0.03, "perigee detail", transform=axi.transAxes,
             fontsize=6.2, color=SUB, ha="left", va="bottom", zorder=7)
    ax.indicate_inset_zoom(axi, edgecolor="#8a8a8a", lw=0.7, alpha=0.9)

    # ===================== (b) delta-v budget (subordinate) ================
    items = [("burn 1, perigee raise", dv1, ACC),
             ("burn 2, apogee circularise", dv2, "#8c2f2f"),
             ("inclination trim", DV_PLANE, C_TGT),
             ("losses and margin", DV_MARGIN, MUTED)]
    left = 0.0
    for name, val, col in items:
        ax_b.barh([0], [val], left=left, height=0.52, color=col, alpha=0.92,
                  edgecolor="white", lw=0.8)
        if val > 0.30:
            ax_b.text(left + val / 2.0, 0.0, f"{val:.2f}", fontsize=6.6,
                      color="white", ha="center", va="center",
                      fontweight="bold")
        left += val
    ax_b.set_yticks([])
    ax_b.set_xlim(0.0, dv_total * 1.06)
    ax_b.set_ylim(-0.62, 2.15)
    ax_b.set_xlabel("$\\Delta v$ (km/s)", fontsize=7.6)
    ax_b.tick_params(labelsize=6.5)
    ax_b.text(dv_total, 0.34, f"total {dv_total:.3f} km/s", fontsize=7.2,
              color=INK, ha="right", va="bottom", fontweight="bold")

    y = 2.00
    for name, val, col in items:
        ax_b.plot([0.04], [y], marker="s", ms=4.2, color=col,
                  transform=ax_b.get_yaxis_transform(), clip_on=False)
        ax_b.text(0.085, y, f"{name}  ({val:.3f} km/s)", fontsize=6.4,
                  color=SUB, ha="left", va="center",
                  transform=ax_b.get_yaxis_transform())
        y -= 0.34
    ax_b.set_title("Delta-v budget", fontsize=8.5, fontweight="bold", pad=3)
    ax_b.spines["left"].set_visible(False)

    # ===================== (c) event timeline (subordinate) ================
    hours = t_transfer / 3600.0
    spans = [("GEO period", t_geo / 3600.0, C_TGT),
             ("transfer coast,\nburn 1 to burn 2", hours, C_TRAN),
             ("parking period", t_park / 3600.0, C_PARK)]
    ypos = np.arange(len(spans), dtype=float)
    for yv, (name, val, col) in zip(ypos, spans):
        ax_t.barh([yv], [val], height=0.46, color=col, alpha=0.88,
                  edgecolor="none")
        ax_t.text(val + 1.05, yv, f"{val:.2f} h", fontsize=6.6, color=col,
                  ha="left", va="center", fontweight="bold")
    ax_t.plot([0.0, hours], [1.0, 1.0], marker="o", ms=5.2, ls="none",
              mfc="white", mec=ACC, mew=1.4, zorder=6)
    ax_t.set_yticks(ypos)
    ax_t.set_yticklabels([s[0] for s in spans], fontsize=6.5)
    ax_t.set_ylim(-0.62, 2.62)
    ax_t.set_xlim(0.0, 29.0)
    ax_t.set_xlabel("duration (h)", fontsize=7.6)
    ax_t.tick_params(axis="x", labelsize=6.5)
    ax_t.tick_params(axis="y", length=0)
    ax_t.set_title("Transfer duration against orbital periods", fontsize=8.5,
                   fontweight="bold", pad=3)
    ax_t.spines["left"].set_visible(False)
    ax_t.grid(True, axis="x", lw=0.5, alpha=0.5)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95], h_pad=2.2, w_pad=2.4)
    panel_labels_inches(fig, [ax, ax_b, ax_t], dx=-0.42, dy=0.06)
    savefig(fig, "506_aero_orbit_transfer.png")


if __name__ == "__main__":
    main()

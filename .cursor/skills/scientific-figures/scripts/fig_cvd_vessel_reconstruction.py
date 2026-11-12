"""Coronary tree reconstruction with a lesion measured three ways.

The 3D panel is a real tapering tube model, not a set of equal-weight lines:
each centreline is split into segments whose *linewidth* is its lumen diameter
and whose colour comes from one shared diameter ramp, so calibre is legible
even where the branch is only a few pixels long.  Vessels are drawn after the
translucent myocardial shell and with a higher zorder, because mplot3d sorts
whole collections against each other by zorder rather than by depth - the
shell would otherwise swallow the vessels it is meant to place in context.

One lesion sits on the LAD.  Its severity is stated once, as a gaussian
narrowing of the reference taper, and everything else is derived: the diameter
profile in (b) is the same array plotted against arc length, and the pressure
ratio in (c) integrates Poiseuille resistance along it, so a wider or tighter
lesion moves the stenosis percentage, the lesion length and the distal FFR
together.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import panel_labels_inches, safe_colorbar, shared_norm

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRIDC = "#e2e2e2"
LESION = "#b23a48"
SHELL = "#9aa7b4"

A, B, C = 1.00, 0.92, 1.38          # myocardial shell semi-axes (model units)
R_EPI = 1.03                        # vessels ride just outside the epicardium
MM_PER_UNIT = 62.0                  # model unit -> millimetres
P_AORTA = 95.0                      # mean aortic pressure, mmHg
MU = 0.035                          # blood viscosity, poise (dyn s / cm^2)
RHO = 1.06                          # blood density, g / cm^3
Q_HYP = 2.9                         # hyperaemic proximal LAD flow, mL/s
DYN_PER_MMHG = 1333.22

# Lesion on the LAD: fractional narrowing, position along the vessel, width.
STEN_DEPTH, STEN_POS, STEN_SIG = 0.595, 0.36, 0.030

# name: (theta0, theta1, phi0, phi1, d_prox, d_dist, wobble)
MAIN = {
    "LAD": (50.0, 172.0, -10.0, 16.0, 3.70, 1.50, 0.16),
    "LCx": (62.0, 116.0, 32.0, 130.0, 3.20, 1.40, -0.20),
    "RCA": (56.0, 152.0, 206.0, 262.0, 3.40, 1.55, 0.22),
}
# name: (parent, u_branch, dtheta, dphi, d_prox, d_dist, wobble)
BRANCH = {
    "D1": ("LAD", 0.24, 30.0, -54.0, 2.00, 0.85, 0.10),
    "D2": ("LAD", 0.60, 24.0, -40.0, 1.55, 0.70, -0.10),
    "OM1": ("LCx", 0.42, 36.0, 26.0, 1.90, 0.80, 0.12),
    "PDA": ("RCA", 0.74, 30.0, -58.0, 1.80, 0.90, -0.14),
}


def surface_xyz(theta, phi, r=R_EPI):
    return np.column_stack([A * r * np.sin(theta) * np.cos(phi),
                            B * r * np.sin(theta) * np.sin(phi),
                            C * r * np.cos(theta)])


def centreline(t0, t1, p0, p1, wob, n=170):
    u = np.linspace(0.0, 1.0, n)
    t = np.deg2rad(t0 + (t1 - t0) * u)
    p = np.deg2rad(p0 + (p1 - p0) * u + 14.0 * wob * np.sin(np.pi * u))
    return u, surface_xyz(t, p), (t, p)


def taper(u, d0, d1):
    return d0 * (d1 / d0) ** u


def arc_mm(xyz):
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg))) * MM_PER_UNIT


def build_tree():
    vessels = {}
    for name, (t0, t1, p0, p1, d0, d1, wob) in MAIN.items():
        u, xyz, (t, p) = centreline(t0, t1, p0, p1, wob)
        d = taper(u, d0, d1)
        vessels[name] = dict(u=u, xyz=xyz, d=d, t=t, p=p, s=arc_mm(xyz))
    for name, (par, ub, dt, dp, d0, d1, wob) in BRANCH.items():
        pv = vessels[par]
        i = int(round(ub * (pv["u"].size - 1)))
        t0 = np.rad2deg(pv["t"][i])
        p0 = np.rad2deg(pv["p"][i])
        u, xyz, _ = centreline(t0, t0 + dt, p0, p0 + dp, wob, n=110)
        vessels[name] = dict(u=u, xyz=xyz, d=taper(u, d0, d1), s=arc_mm(xyz))
    return vessels


def apply_lesion(v):
    """Gaussian narrowing of the LAD taper; returns reference and lesion arrays."""
    ref = v["d"].copy()
    dia = ref * (1.0 - STEN_DEPTH *
                 np.exp(-0.5 * ((v["u"] - STEN_POS) / STEN_SIG) ** 2))
    v["d_ref"], v["d"] = ref, dia
    return ref, dia


def tube(ax, xyz, d, norm, cmap, lw_scale=2.05, zorder=5, override=None):
    segs = np.stack([xyz[:-1], xyz[1:]], axis=1)
    dm = 0.5 * (d[:-1] + d[1:])
    colours = cmap(norm(dm)) if override is None else override
    lc = Line3DCollection(segs, colors=colours, linewidths=dm * lw_scale,
                          capstyle="round", zorder=zorder)
    ax.add_collection3d(lc)
    return lc


def main():
    set_pub_style()
    vessels = build_tree()
    lad = vessels["LAD"]
    d_ref, d_len = apply_lesion(lad)

    # ---- lesion metrics, all read off the arrays ---------------------------
    i_min = int(np.argmin(lad["d"]))
    mld = float(lad["d"][i_min])
    ref_at = float(d_ref[i_min])
    sten_pct = (1.0 - mld / ref_at) * 100.0
    area_pct = (1.0 - (mld / ref_at) ** 2) * 100.0
    inside = lad["d"] < 0.90 * d_ref
    lesion_mm = float(lad["s"][inside].max() - lad["s"][inside].min())

    # Pressure loss in real units: Poiseuille friction integrated along the
    # lumen, plus the flow-separation loss that dominates a tight lesion.
    # Flow tapers with the *reference* calibre, because side branches - not
    # the lesion - decide how much blood the distal bed still needs.
    ds_cm = np.concatenate(([0.0], np.diff(lad["s"]))) / 10.0
    q = Q_HYP * (d_ref / d_ref[0]) ** 2
    r_cm = lad["d"] / 20.0
    visc = np.cumsum(8.0 * MU * q / (np.pi * r_cm ** 4) * ds_cm)
    v_act = q / (np.pi * r_cm ** 2)
    v_ref = q / (np.pi * (d_ref / 20.0) ** 2)
    sep = 0.5 * RHO * (v_act[i_min] - v_ref[i_min]) ** 2
    step = 1.0 / (1.0 + np.exp(-(lad["s"] - lad["s"][i_min] - 2.0) / 1.1))
    ffr = 1.0 - (visc + sep * step) / DYN_PER_MMHG / P_AORTA
    ffr_distal = float(ffr[-1])
    print(f"MLD {mld:.2f} mm, diameter stenosis {sten_pct:.0f}%, "
          f"distal FFR {ffr_distal:.2f}")

    cmap = get_cmap("cb_sequential_blue")
    norm = shared_norm([v["d"] for v in vessels.values()], vmin=0.6, vmax=3.8)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58], hspace=0.42,
                          wspace=0.10)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    ax_d = fig.add_subplot(gs[0, 1])
    ax_f = fig.add_subplot(gs[1, 1])

    # ---- (a) 3D reconstruction ---------------------------------------------
    ax3.set_axis_off()
    ax3.patch.set_alpha(0.0)
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.set_visible(False)
    ax3.set_proj_type("ortho")
    ax3.view_init(elev=16.0, azim=-64.0)
    ax3.set_box_aspect((1.0, 0.95, 1.30))
    ax3.set_xlim(-1.15, 1.15)
    ax3.set_ylim(-1.10, 1.10)
    ax3.set_zlim(-1.50, 1.05)

    tt = np.linspace(np.deg2rad(40.0), np.pi, 44)
    pp = np.linspace(0.0, 2.0 * np.pi, 70)
    ax3.plot_surface(A * np.outer(np.sin(tt), np.cos(pp)),
                     B * np.outer(np.sin(tt), np.sin(pp)),
                     C * np.outer(np.cos(tt), np.ones_like(pp)),
                     color=SHELL, alpha=0.13, linewidth=0.0, shade=False,
                     zorder=1)
    for ph in np.linspace(0.0, 2.0 * np.pi, 13)[:-1]:
        m = surface_xyz(tt, np.full_like(tt, ph), r=1.0)
        ax3.plot(m[:, 0], m[:, 1], m[:, 2], color=SHELL, lw=0.35, alpha=0.45,
                 zorder=2)
    ring = surface_xyz(np.full(80, np.deg2rad(40.0)),
                       np.linspace(0.0, 2.0 * np.pi, 80), r=1.0)
    ax3.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=SHELL, lw=0.8,
             alpha=0.8, zorder=2)

    for name, v in vessels.items():
        tube(ax3, v["xyz"], v["d"], norm, cmap,
             zorder=6 if name in MAIN else 5)

    # The lesion is the figure's single emphasis object.
    m = lad["d"] < 0.86 * d_ref
    idx = np.where(m)[0]
    lo, hi = idx.min(), idx.max() + 1
    tube(ax3, lad["xyz"][lo:hi], lad["d"][lo:hi], norm, cmap, zorder=8,
         override=LESION)
    pk = lad["xyz"][i_min]
    ax3.scatter([pk[0]], [pk[1]], [pk[2]], s=26, facecolor="none",
                edgecolor=LESION, linewidths=1.1, depthshade=False, zorder=9)

    for name, off in (("LAD", (0.10, -0.30, 0.10)), ("LCx", (-0.10, 0.30, 0.16)),
                      ("RCA", (0.18, 0.20, 0.14))):
        v = vessels[name]
        q = v["xyz"][int(0.18 * v["xyz"].shape[0])]
        ax3.text(q[0] + off[0], q[1] + off[1], q[2] + off[2], name,
                 fontsize=7.2, fontweight="bold", color=INK, ha="center",
                 zorder=10)
    for name in ("D1", "D2", "OM1", "PDA"):
        q = vessels[name]["xyz"][-1]
        ax3.text(q[0] * 1.16, q[1] * 1.16, q[2] - 0.06, name, fontsize=6.4,
                 color=MUTED, ha="center", zorder=10)
    ax3.text(pk[0] + 0.34, pk[1] - 0.52, pk[2] + 0.02,
             f"stenosis {sten_pct:.0f}% diameter", fontsize=7.0,
             fontweight="bold", color=LESION, ha="center", zorder=10)
    ax3.text2D(0.02, 0.985, "Coronary tree with a mid-LAD lesion",
               transform=ax3.transAxes, ha="left", va="top", fontsize=8.8,
               fontweight="bold", color=INK)
    ax3.text2D(0.02, 0.035, "translucent shell: left-ventricular myocardium\n"
               "tube width and colour: lumen diameter",
               transform=ax3.transAxes, ha="left", va="bottom", fontsize=6.4,
               color=MUTED, linespacing=1.4)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    safe_colorbar(fig, sm, ax3, label="lumen diameter (mm)", side="bottom",
                  width=0.11, pad=0.02, shrink=0.52)

    # ---- (b) diameter profile (auxiliary) ----------------------------------
    ax_d.plot(lad["s"], d_ref, color="#9a9a9a", lw=1.0, ls=(0, (4, 2.4)),
              label="reference taper", zorder=3)
    ax_d.fill_between(lad["s"], lad["d"], d_ref, where=lad["d"] < d_ref,
                      color=LESION, alpha=0.20, lw=0, zorder=2)
    ax_d.plot(lad["s"], lad["d"], color="#2e74a8", lw=1.5, zorder=4,
              label="measured lumen")
    ax_d.plot([lad["s"][i_min]], [mld], marker="o", ms=4.2, mfc=LESION,
              mec="white", mew=0.7, zorder=6)
    ax_d.annotate(f"MLD {mld:.2f} mm", xy=(lad["s"][i_min], mld),
                  xytext=(lad["s"][i_min] + 12.0, mld + 0.10), fontsize=6.6,
                  color=INK, va="bottom",
                  arrowprops=dict(arrowstyle="->", lw=0.7, color="#9a9a9a"))
    ax_d.set_xlabel("distance along LAD centreline (mm)", fontsize=7.4)
    ax_d.set_ylabel("lumen diameter (mm)", fontsize=7.4)
    ax_d.set_xlim(0.0, lad["s"][-1])
    ax_d.set_ylim(0.0, 4.2)
    ax_d.set_title("Calibre profile along the centreline", fontsize=8.5,
                   fontweight="bold", loc="left")
    ax_d.legend(loc="upper right", frameon=False, fontsize=6.5,
                handlelength=1.4)
    ax_d.grid(color=GRIDC, lw=0.5)
    ax_d.tick_params(labelsize=6.5)
    ax_d.text(0.02, 0.06, f"reference {ref_at:.2f} mm   |   "
              f"area stenosis {area_pct:.0f}%   |   lesion "
              f"{lesion_mm:.0f} mm", transform=ax_d.transAxes, fontsize=6.4,
              color=SUB)

    # ---- (c) pressure ratio (auxiliary) ------------------------------------
    f_lo = min(0.62, ffr_distal - 0.08)
    ax_f.axhspan(f_lo, 0.80, color="#f2d7da", alpha=0.55, lw=0, zorder=0)
    ax_f.axhline(0.80, color=LESION, lw=0.9, ls=(0, (4, 2.4)), zorder=3)
    ax_f.plot(lad["s"], ffr, color="#3b5b92", lw=1.6, zorder=4)
    ax_f.plot([lad["s"][-1]], [ffr_distal], marker="o", ms=4.2,
              mfc="#3b5b92", mec="white", mew=0.7, zorder=6)
    ax_f.text(lad["s"][-1] - 2.0, ffr_distal - 0.035,
              f"distal FFR {ffr_distal:.2f}", ha="right", va="top",
              fontsize=6.8, fontweight="bold", color=INK)
    ax_f.text(lad["s"][-1] - 2.0, 0.812, "ischaemic threshold FFR = 0.80",
              fontsize=6.4, color=LESION, va="bottom", ha="right")
    ax_f.set_xlabel("distance along LAD centreline (mm)", fontsize=7.4)
    ax_f.set_ylabel(r"pressure ratio $P_d/P_a$ (-)", fontsize=7.4)
    ax_f.set_xlim(0.0, lad["s"][-1])
    ax_f.set_ylim(f_lo, 1.03)
    ax_f.set_title("Translesional pressure loss", fontsize=8.5,
                   fontweight="bold", loc="left")
    ax_f.grid(color=GRIDC, lw=0.5)
    ax_f.tick_params(labelsize=6.5)
    ax_f.text(0.02, 0.08, f"Poiseuille loss integrated along the profile in "
              f"(b); $P_a$ = {P_AORTA:.0f} mmHg.",
              transform=ax_f.transAxes, fontsize=6.4, color=MUTED)

    fig.tight_layout(rect=[0.036, 0.02, 1, 0.95], w_pad=2.0)
    panel_labels_inches(fig, [ax3, ax_d, ax_f], dx=-0.34, dy=0.06)
    savefig(fig, "549_cvd_vessel_reconstruction.png")


if __name__ == "__main__":
    main()

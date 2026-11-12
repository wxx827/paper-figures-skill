"""Inlet total-pressure distortion measured on a rake at the engine face.

Main panel: the aerodynamic interface plane of an S-duct intake, looking
downstream, drawn on an equal-aspect axes because a distortion pattern is a
picture of where the low-pressure fluid sits and a stretched annulus would move
it.  Total-pressure recovery is contoured between the hub and the casing, the
forty rake probes that the contour is built from are drawn where they are, and
the sixty-degree sector that produces the worst sector average is outlined -
that sector is searched for in the field, not assumed to be at the bottom.

The companion panels are the two standard reductions.  The circumferential
traces show each rake ring unwrapped, which is how a low-pressure lobe is told
apart from a hub-to-tip gradient, and the descriptor sweep shows the classical
result that distortion is worst at low mass-flow ratio, with every point on it
recomputed from a fresh field.

Illustrative synthetic data: a duct-loss floor plus a swirling low-momentum
lobe whose depth and width grow as the mass-flow ratio falls.
"""
import textwrap

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge
from pubstyle import set_pub_style, savefig, style_axes, get_cmap, add_footer
from figkit import axes_inches, panel_labels_inches, safe_colorbar, halo_text

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
RED = "#b23a48"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"

FIG_W, FIG_H = 10.6, 6.4
MAIN_RECT = (0.54, 0.86, 5.10, 4.42)
AUX_T_RECT = (7.16, 3.52, 3.02, 1.86)
AUX_B_RECT = (7.16, 0.82, 3.02, 1.86)

R_HUB = 0.130           # m, spinner radius at the interface plane
R_TIP = 0.315           # m, casing radius at the interface plane
N_RAKE = 8              # equally spaced radial rakes
N_RING = 5              # probes per rake, on equal-area rings
MFR_REF = 0.78          # mass-flow ratio of the case shown in the main panel
SECTOR = 60.0           # deg, sector width of the DC descriptor
Q_FACE = 1.50e4         # Pa, face dynamic head used to normalise DC60
P_T0 = 1.01325e5        # Pa, free-stream total pressure


def ring_radii():
    """Equal-area ring radii, the standard rake layout (SAE ARP1420)."""
    f = (np.arange(N_RING) + 0.5) / N_RING
    return np.sqrt(R_HUB ** 2 + f * (R_TIP ** 2 - R_HUB ** 2))


def recovery(r, th, mfr):
    """Total-pressure recovery field: duct-loss floor plus a swirled lobe."""
    rn = (r - R_HUB) / (R_TIP - R_HUB)
    depth = 0.052 * (1.0 + 3.4 * (0.92 - mfr))
    width = np.deg2rad(46.0 + 40.0 * (0.92 - mfr))
    lobe_c = np.deg2rad(248.0) + 0.85 * rn          # swirl tilts the lobe
    d = np.arctan2(np.sin(th - lobe_c), np.cos(th - lobe_c))
    lobe = depth * np.exp(-(d / width) ** 2) * np.exp(-((rn - 0.62) / 0.46) ** 2)
    wall = 0.013 * (np.exp(-(rn / 0.16) ** 2) + np.exp(-((1.0 - rn) / 0.13) ** 2))
    tilt = 0.006 * np.cos(th - np.deg2rad(70.0)) * rn
    return 0.982 - lobe - wall + tilt


def descriptors(mfr, nr=140, nt=361):
    """DC60 and the peak circumferential intensity, measured off a field."""
    r = np.linspace(R_HUB, R_TIP, nr)
    th = np.linspace(0.0, 2.0 * np.pi, nt)
    R, T = np.meshgrid(r, th, indexing="ij")
    pr = recovery(R, T, mfr)
    face = float(np.average(pr, weights=np.broadcast_to(R, pr.shape)))

    half = int(round(0.5 * SECTOR / 360.0 * (nt - 1)))
    ring_avg = np.average(pr, axis=0, weights=r)
    roll = np.array([ring_avg[np.arange(i - half, i + half + 1) % (nt - 1)].mean()
                     for i in range(nt - 1)])
    k = int(np.argmin(roll))
    dc = (face - roll[k]) * P_T0 / Q_FACE
    ring_int = [(pr[i].mean() - pr[i].min()) / pr[i].mean() for i in range(nr)]
    return dc, float(np.rad2deg(th[k])), face, max(ring_int), r, th, pr


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
    dc60, th_worst, face, ring_max, r, th, pr = descriptors(MFR_REF)
    radii = ring_radii()
    rake_th = np.deg2rad(np.arange(N_RAKE) * 360.0 / N_RAKE)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_c = axes_inches(fig, AUX_T_RECT)
    ax_d = axes_inches(fig, AUX_B_RECT)
    for a in (ax_c, ax_d):
        a.set_axisbelow(True)
        a.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)
        style_axes(a)

    # ---- main: the interface plane -----------------------------------------
    ax.set_aspect("equal")
    R, T = np.meshgrid(r, th, indexing="ij")
    X, Y = R * np.cos(T), R * np.sin(T)
    levels = np.linspace(pr.min(), pr.max(), 26)
    im = ax.contourf(X, Y, pr, levels=levels,
                     cmap=get_cmap("cb_sequential_blue", reverse=True),
                     zorder=2)
    cs = ax.contour(X, Y, pr, levels=levels[::5], colors="#33333366",
                    linewidths=0.55, zorder=4)
    ax.clabel(cs, fmt="%.3f", fontsize=6, inline=True, inline_spacing=3)

    wedge = Wedge((0.0, 0.0), R_TIP, th_worst - SECTOR / 2.0,
                  th_worst + SECTOR / 2.0, width=R_TIP - R_HUB,
                  facecolor="none", edgecolor=RED, lw=1.5, zorder=7)
    ax.add_patch(wedge)
    ax.add_patch(Circle((0.0, 0.0), R_HUB, facecolor="#1a1a1a",
                        edgecolor="#1a1a1a", lw=0.8, zorder=8))
    ax.add_patch(Circle((0.0, 0.0), R_TIP, facecolor="none",
                        edgecolor="#5a5a5a", lw=1.3, zorder=8))
    ax.text(0.0, 0.0, "hub", color="white", fontsize=6.8, ha="center",
            va="center", zorder=9)

    for a_r in rake_th:
        ax.plot([R_HUB * np.cos(a_r), R_TIP * np.cos(a_r)],
                [R_HUB * np.sin(a_r), R_TIP * np.sin(a_r)],
                color="#ffffff", lw=0.7, alpha=0.55, zorder=5)
    pr_probe = recovery(*np.meshgrid(radii, rake_th, indexing="ij"), MFR_REF)
    px = np.outer(radii, np.cos(rake_th))
    py = np.outer(radii, np.sin(rake_th))
    ax.plot(px.ravel(), py.ravel(), ls="none", marker="o", ms=3.0,
            mfc="white", mec=INK, mew=0.8, zorder=9)
    i_lo = np.unravel_index(np.argmin(pr_probe), pr_probe.shape)
    ax.plot([px[i_lo]], [py[i_lo]], marker="o", ms=5.4, mfc=RED, mec="white",
            mew=1.0, zorder=10)
    halo_text(ax, px[i_lo], py[i_lo] - 0.028,
              f"worst probe {pr_probe.min():.3f}", ha="center", va="top",
              fontsize=6.3, color="white", halo="#1a1a1a", halo_lw=2.0,
              zorder=11)
    halo_text(ax, R_TIP * np.cos(np.deg2rad(th_worst)) * 1.10,
              R_TIP * np.sin(np.deg2rad(th_worst)) * 1.10,
              f"worst {SECTOR:.0f} deg sector\ncentred at "
              f"{th_worst:.0f} deg", ha="center", va="center", fontsize=6.4,
              color=RED, halo="white", halo_lw=2.4, zorder=11)

    lim = R_TIP * 1.30
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks([-0.3, -0.15, 0.0, 0.15, 0.3])
    ax.set_yticks([-0.3, -0.15, 0.0, 0.15, 0.3])
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel("y at the interface plane (m)", fontsize=8.4, labelpad=1.5)
    ax.set_ylabel("z at the interface plane (m)", fontsize=8.4)
    ax.set_title("Total-pressure recovery at the engine face, "
                 f"{N_RAKE} x {N_RING} rake, MFR = {MFR_REF:.2f}",
                 fontsize=9.0, fontweight="bold", pad=5.0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.012, 0.988,
            f"face-average recovery $P_{{t2}}/P_{{t0}}$ = {face:.4f}\n"
            f"DC({SECTOR:.0f}) = {dc60:.3f} (dimensionless), searched over "
            "all sector centres\nview looking downstream; open circles are "
            f"the {N_RAKE * N_RING} probe locations",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
            color=INK, linespacing=1.50, zorder=12,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.93))
    safe_colorbar(fig, im, ax, label="total-pressure recovery "
                  "$P_{t2}/P_{t0}$ (dimensionless)", width=0.13, pad=0.10,
                  shrink=0.84, labelsize=7.0,
                  ticks=np.round(np.linspace(pr.min(), pr.max(), 6), 3))

    # ---- aux (b): rings unwrapped ------------------------------------------
    th_deg = np.rad2deg(th)
    cols = (BLUE, GREEN, GOLD, PURPLE, RED)
    for j, (rr, col) in enumerate(zip(radii, cols)):
        pr_ring = recovery(np.full_like(th, rr), th, MFR_REF)
        ax_c.plot(th_deg, pr_ring, color=col, lw=1.5, zorder=5,
                  label=f"ring {j + 1}, r = {rr * 1e3:.0f} mm")
    ax_c.axhline(face, color=FAINT, lw=0.8, ls=(0, (3.0, 2.0)), zorder=2)
    ax_c.axvspan(th_worst - SECTOR / 2.0, th_worst + SECTOR / 2.0,
                 color="#f6e3e5", zorder=1)
    ax_c.set_xlim(0.0, 360.0)
    ax_c.set_ylim(pr.min() - 0.006, pr.max() + 0.024)
    ax_c.set_xticks(np.arange(0.0, 361.0, 90.0))
    ax_c.tick_params(labelsize=6.8)
    ax_c.set_xlabel("circumferential position $\\theta$ (deg)", fontsize=7.5,
                    labelpad=1.5)
    ax_c.set_ylabel("$P_{t2}/P_{t0}$ (dimensionless)", fontsize=7.5)
    ax_c.set_title("Every ring dips in the same sector", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_c.text(0.035, 0.055,
              f"shaded band is the worst {SECTOR:.0f} deg sector;\ndeepest "
              f"ring loses {100.0 * ring_max:.1f} % of its own mean",
              transform=ax_c.transAxes, ha="left", va="bottom", fontsize=6.4,
              color=SUB, linespacing=1.45)
    ax_c.legend(loc="upper left", frameon=False, fontsize=6.0, ncol=2,
                handlelength=1.3, columnspacing=1.0, labelspacing=0.22,
                borderaxespad=0.25)

    # ---- aux (c): descriptor sweep -----------------------------------------
    mfr = np.linspace(0.56, 0.94, 22)
    dcs, faces = [], []
    for m in mfr:
        d, _, f, _, _, _, _ = descriptors(m, nr=70, nt=181)
        dcs.append(d)
        faces.append(f)
    dcs = np.array(dcs)
    faces = np.array(faces)
    ax_d.plot(mfr, dcs, color=RED, lw=1.8, zorder=6)
    ax_d.plot([MFR_REF], [np.interp(MFR_REF, mfr, dcs)], marker="o", ms=4.6,
              mfc="white", mec=RED, mew=1.4, zorder=8)
    ax_d.axhline(0.30, color=INK, lw=0.9, ls=(0, (4.0, 2.2)), zorder=3)
    ax_d.text(0.945, 0.305, "engine acceptance limit DC(60) = 0.30",
              transform=ax_d.get_yaxis_transform(), ha="right", va="bottom",
              fontsize=6.3, color=INK)
    ax_d.set_xlim(mfr.min(), mfr.max())
    ax_d.set_ylim(0.0, max(dcs.max(), 0.30) * 1.52)
    ax_d.tick_params(labelsize=6.8, labelcolor=RED, axis="y")
    ax_d.tick_params(labelsize=6.8, axis="x")
    ax_d.set_xlabel("mass-flow ratio MFR (dimensionless)", fontsize=7.5,
                    labelpad=1.5)
    ax_d.set_ylabel("DC(60) (dimensionless)", fontsize=7.5, color=RED)
    ax_d.set_title("Throttling the intake makes it worse", fontsize=8.5,
                   fontweight="bold", pad=4.0)

    ax_t = ax_d.twinx()
    ax_t.plot(mfr, faces, color=BLUE, lw=1.8, ls=(0, (4.4, 2.0)), zorder=5)
    ax_t.set_ylabel("face-average recovery (dimensionless)", fontsize=7.5,
                    color=BLUE)
    ax_t.tick_params(labelsize=6.8, labelcolor=BLUE)
    ax_t.spines["top"].set_visible(False)
    ax_t.grid(False)
    cross = float(np.interp(0.30, dcs[::-1], mfr[::-1]))
    ax_d.text(0.035, 0.055,
              f"the limit is crossed at MFR = {cross:.2f};\nopen circle is "
              "the case mapped in panel (a)",
              transform=ax_d.transAxes, ha="left", va="bottom", fontsize=6.4,
              color=SUB, linespacing=1.45)
    ax_d.legend(handles=[Line2D([], [], color=RED, lw=1.8, label="DC(60)"),
                         Line2D([], [], color=BLUE, lw=1.8,
                                ls=(0, (4.4, 2.0)), label="recovery")],
                loc="upper right", frameon=False, fontsize=6.4,
                handlelength=1.7, labelspacing=0.26, borderaxespad=0.25)

    panel_labels_inches(fig, [ax, ax_c, ax_d], dx=-0.50, dy=0.10)
    foot = add_footer(fig, "Synthetic data, figure 933")
    check_layout(fig, foot, [ax, ax_c, ax_d])
    savefig(fig, "933_wt_inlet_distortion.png")


if __name__ == "__main__":
    main()

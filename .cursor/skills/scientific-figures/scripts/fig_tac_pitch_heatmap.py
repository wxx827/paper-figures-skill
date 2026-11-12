"""Single-player activity heat map on a full-size pitch.

One midfielder is followed for a whole match: a mean-reverting position track
is sampled at 1 Hz for 90 min and every number in the figure is read back out
of that one track.  The heat map is a Gaussian kernel density estimate of the
samples expressed as share of match time per 100 m2, the marginal curves are
the same samples binned along the length and across the width of the pitch,
the average-position cross is the track mean, the activity centroid is the
touch-weighted mean, and the third shares bin the touch subsample.

This is the *individual* counterpart to the team occupancy map: the question
here is not where the side played but how one player's zone is shaped, which
is why the marginal profiles matter more than a squad-wide summary.

Pitch geometry is the FIFA standard 105 x 68 m surface: 16.5 m penalty areas
40.32 m wide, 5.5 m goal areas 18.32 m wide, penalty spots at 11 m, a 9.15 m
centre circle and penalty arcs, and 1 m corner arcs.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Rectangle
from scipy.stats import gaussian_kde

from pubstyle import annotate_box, get_cmap, savefig, set_pub_style
from figkit import (axes_inches, halo_text, panel_labels_inches,
                    safe_colorbar, shared_norm)

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HOME = "#3b5b92"          # house blue - the player under analysis
ACCENT = "#b23a48"        # house red - the touch-weighted centroid
GOLD = "#c9a227"          # house gold - the across-pitch marginal
GRASS = "#f4f4f2"         # near-neutral playing surface, never saturated green
LINE = "#c2c2c2"          # pitch markings on that surface

PITCH_L, PITCH_W = 105.0, 68.0
MATCH_MIN = 90.0
HZ_SAMPLES = 5400         # 1 Hz for 90 min
N_TOUCH = 78              # touches taken by this player

PLAYER = "P. Nakamura"
SHIRT, ROLE, TEAM = 8, "RCM", "Team A"

THIRD = ["defensive third", "middle third", "attacking third"]


def draw_pitch(ax, color=LINE, lw=0.9, zorder=5):
    """Standard 105 x 68 m markings, drawn over whatever fills the surface."""
    ax.add_patch(Rectangle((0.0, 0.0), PITCH_L, PITCH_W, facecolor=GRASS,
                           edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0.0, 0.0), PITCH_L, PITCH_W, fill=False,
                           edgecolor=color, lw=lw, zorder=zorder))
    ax.plot([PITCH_L / 2.0, PITCH_L / 2.0], [0.0, PITCH_W], color=color,
            lw=lw, zorder=zorder, solid_capstyle="butt")
    ax.add_patch(Circle((PITCH_L / 2.0, PITCH_W / 2.0), 9.15, fill=False,
                        edgecolor=color, lw=lw, zorder=zorder))
    ax.add_patch(Circle((PITCH_L / 2.0, PITCH_W / 2.0), 0.40,
                        facecolor=color, edgecolor="none", zorder=zorder))

    arc_half = float(np.degrees(np.arccos((16.5 - 11.0) / 9.15)))
    for side in (0, 1):
        x_box = 0.0 if side == 0 else PITCH_L - 16.5
        x_six = 0.0 if side == 0 else PITCH_L - 5.5
        x_spot = 11.0 if side == 0 else PITCH_L - 11.0
        ax.add_patch(Rectangle((x_box, PITCH_W / 2.0 - 20.16), 16.5, 40.32,
                               fill=False, edgecolor=color, lw=lw,
                               zorder=zorder))
        ax.add_patch(Rectangle((x_six, PITCH_W / 2.0 - 9.16), 5.5, 18.32,
                               fill=False, edgecolor=color, lw=lw,
                               zorder=zorder))
        ax.add_patch(Circle((x_spot, PITCH_W / 2.0), 0.35, facecolor=color,
                            edgecolor="none", zorder=zorder))
        t1, t2 = ((-arc_half, arc_half) if side == 0
                  else (180.0 - arc_half, 180.0 + arc_half))
        ax.add_patch(Arc((x_spot, PITCH_W / 2.0), 2 * 9.15, 2 * 9.15,
                         theta1=t1, theta2=t2, edgecolor=color, lw=lw,
                         zorder=zorder))
        x_goal = -1.9 if side == 0 else PITCH_L
        ax.add_patch(Rectangle((x_goal, PITCH_W / 2.0 - 3.66), 1.9, 7.32,
                               fill=False, edgecolor=color, lw=lw,
                               zorder=zorder))

    for (cx, cy, t1, t2) in ((0.0, 0.0, 0.0, 90.0),
                             (0.0, PITCH_W, 270.0, 360.0),
                             (PITCH_L, 0.0, 90.0, 180.0),
                             (PITCH_L, PITCH_W, 180.0, 270.0)):
        ax.add_patch(Arc((cx, cy), 2.0, 2.0, theta1=t1, theta2=t2,
                         edgecolor=color, lw=lw, zorder=zorder))


def ou_track(rng, x0, y0, kx, ky, sx, sy, n):
    """Mean-reverting random walk clipped to the pitch; returns an (n, 2) path."""
    p = np.empty((n, 2))
    x, y = x0, y0
    ex = rng.normal(0.0, sx, n)
    ey = rng.normal(0.0, sy, n)
    for t in range(n):
        x = np.clip(x + kx * (x0 - x) + ex[t], 0.5, PITCH_L - 0.5)
        y = np.clip(y + ky * (y0 - y) + ey[t], 0.5, PITCH_W - 0.5)
        p[t] = (x, y)
    return p


def main():
    set_pub_style()
    rng = np.random.default_rng(864)

    track = ou_track(rng, 56.0, 42.0, 0.0026, 0.0050, 1.75, 1.55, HZ_SAMPLES)
    dist_km = float(np.hypot(*np.diff(track, axis=0).T).sum() / 1000.0)
    mean_pos = track.mean(axis=0)

    # touches: a forward-biased subsample of the same track, because a player
    # receives the ball where the team is trying to progress
    w = 0.30 + 0.70 * (track[:, 0] / PITCH_L) ** 1.8
    idx = rng.choice(HZ_SAMPLES, size=N_TOUCH, replace=False, p=w / w.sum())
    touch = track[idx]
    centroid = touch.mean(axis=0)

    # ---- kernel density of occupancy ---------------------------------------
    sub = track[rng.choice(HZ_SAMPLES, size=1600, replace=False)]
    kde = gaussian_kde(sub.T, bw_method=0.24)
    gx = np.linspace(0.0, PITCH_L, 150)
    gy = np.linspace(0.0, PITCH_W, 98)
    GX, GY = np.meshgrid(gx, gy)
    dens = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    # pdf in m^-2 -> per cent of match time inside each 100 m2 patch
    occ = 100.0 * 100.0 * dens
    field = np.where(occ < 0.06 * occ.max(), np.nan, occ)

    # ---- marginal profiles, same samples ------------------------------------
    ex = np.arange(0.0, PITCH_L + 1.0, 1.0)
    ey = np.arange(0.0, PITCH_W + 1.0, 1.0)
    hx, _ = np.histogram(track[:, 0], bins=ex)
    hy, _ = np.histogram(track[:, 1], bins=ey)
    px = 100.0 * hx / hx.sum()          # per cent of match time per metre
    py = 100.0 * hy / hy.sum()
    cx, cy = 0.5 * (ex[:-1] + ex[1:]), 0.5 * (ey[:-1] + ey[1:])

    # ---- third shares from the touch subsample ------------------------------
    ti = np.clip((touch[:, 0] / (PITCH_L / 3.0)).astype(int), 0, 2)
    share = 100.0 * np.bincount(ti, minlength=3) / len(ti)
    t_time = 100.0 * np.bincount(
        np.clip((track[:, 0] / (PITCH_L / 3.0)).astype(int), 0, 2),
        minlength=3) / HZ_SAMPLES

    # ---- layout -------------------------------------------------------------
    FW, FH = 11.0, 6.3
    fig = plt.figure(figsize=(FW, FH))
    ax_p = axes_inches(fig, (0.38, 1.05, 6.00, 4.278))
    ax_m = axes_inches(fig, (7.25, 3.30, 3.35, 2.00))
    ax_z = axes_inches(fig, (7.25, 0.95, 3.35, 1.60))

    cmap = get_cmap("cb_sequential_blue").copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    norm = shared_norm([field], robust=(1.0, 99.0))
    draw_pitch(ax_p, zorder=5)
    im = ax_p.imshow(field, origin="lower", extent=[0, PITCH_L, 0, PITCH_W],
                     cmap=cmap, norm=norm, alpha=0.86, zorder=2,
                     interpolation="bilinear")

    for i in (1, 2):
        ax_p.plot([i * PITCH_L / 3.0] * 2, [0.0, PITCH_W], color="#9a9a9a",
                  lw=0.7, ls=(0, (4, 3)), zorder=6)
    for i, name in enumerate(THIRD):
        halo_text(ax_p, (i + 0.5) * PITCH_L / 3.0, PITCH_W + 1.6,
                  f"{name} | {share[i]:.0f}% of touches", ha="center",
                  va="bottom", fontsize=6.2, color=SUB, halo="white",
                  halo_lw=1.8, zorder=9)

    ax_p.scatter(touch[:, 0], touch[:, 1], s=8.0, facecolor="white",
                 edgecolor=RULE, linewidth=0.5, alpha=0.85, zorder=7)

    ax_p.plot([mean_pos[0] - 4.2, mean_pos[0] + 4.2], [mean_pos[1]] * 2,
              color=HOME, lw=1.5, zorder=9, solid_capstyle="butt")
    ax_p.plot([mean_pos[0]] * 2, [mean_pos[1] - 4.2, mean_pos[1] + 4.2],
              color=HOME, lw=1.5, zorder=9, solid_capstyle="butt")
    ax_p.plot([mean_pos[0]], [mean_pos[1]], marker="o", ms=4.2, mfc="white",
              mec=HOME, mew=1.3, zorder=10)
    ax_p.plot([centroid[0]], [centroid[1]], marker="X", ms=8.0, mfc=ACCENT,
              mec="white", mew=1.0, zorder=10)
    halo_text(ax_p, centroid[0] + 2.4, centroid[1] - 2.6,
              f"touch centroid ({centroid[0]:.0f}, {centroid[1]:.0f}) m",
              ha="left", va="top", fontsize=6.2, color=ACCENT, halo="white",
              halo_lw=2.0, zorder=11)

    ax_p.annotate("", xy=(92.0, -3.4), xytext=(72.0, -3.4),
                  arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))
    ax_p.text(70.0, -3.4, "attacking direction", ha="right", va="center",
              fontsize=6.4, color=MUTED)

    ax_p.set_xlim(-6.0, 111.0)
    ax_p.set_ylim(-7.0, 76.0)
    ax_p.set_aspect("equal")
    ax_p.set_xticks([])
    ax_p.set_yticks([])
    for s in ax_p.spines.values():
        s.set_visible(False)
    ax_p.set_title(f"#{SHIRT} {PLAYER} ({ROLE}, {TEAM}) - 90 min occupancy",
                   fontsize=9.0, fontweight="bold", pad=5)

    annotate_box(
        ax_p,
        f"distance  {dist_km:.2f} km over {MATCH_MIN:.0f} min\n"
        f"mean position  ({mean_pos[0]:.1f}, {mean_pos[1]:.1f}) m\n"
        f"touches  {N_TOUCH} | peak {np.nanmax(field):.2f}% per 100 m$^2$",
        loc="lower left", fontsize=6.4, margin=0.016)

    safe_colorbar(fig, im, ax_p, label="share of match time (% per 100 m$^2$)",
                  side="bottom", width=0.10, pad=0.13, shrink=0.52,
                  labelsize=6.5)

    # ---- marginal profiles ---------------------------------------------------
    ax_m.fill_between(cx, 0.0, px, color=HOME, alpha=0.16, lw=0.0, zorder=2)
    ax_m.plot(cx, px, color=HOME, lw=1.3, zorder=4, label="length axis")
    ax_m.plot(cy, py, color=GOLD, lw=1.3, ls="--", zorder=4,
              label="width axis")
    ax_m.axvline(mean_pos[0], color=HOME, lw=0.8, ls=":", zorder=3)
    ax_m.axvline(mean_pos[1], color=GOLD, lw=0.8, ls=":", zorder=3)
    ax_m.set_xlim(0.0, PITCH_L)
    y_top = max(px.max(), py.max()) * 1.22
    ax_m.set_ylim(0.0, y_top)
    ax_m.text(mean_pos[1] + 1.4, y_top * 0.985, f"mean y {mean_pos[1]:.1f} m",
              ha="left", va="top", fontsize=6.2, color=GOLD)
    ax_m.text(mean_pos[0] + 1.4, y_top * 0.895, f"mean x {mean_pos[0]:.1f} m",
              ha="left", va="top", fontsize=6.2, color=HOME)
    ax_m.set_xlabel("distance along the pitch axis (m)", fontsize=8.0)
    ax_m.set_ylabel("time share (% per m)", fontsize=8.0)
    ax_m.tick_params(labelsize=7.0)
    ax_m.grid(axis="y", lw=0.5, color="#ececec", zorder=0)
    ax_m.spines["top"].set_visible(False)
    ax_m.spines["right"].set_visible(False)
    ax_m.legend(loc="upper left", frameon=False, fontsize=6.8,
                handlelength=1.9, borderaxespad=0.2)
    ax_m.set_title("Marginal activity profiles", fontsize=8.5,
                   fontweight="bold")

    # ---- third shares --------------------------------------------------------
    yy = np.arange(3)[::-1]
    ax_z.barh(yy + 0.19, share, height=0.34, color=HOME, edgecolor="white",
              lw=0.6, zorder=3, label="touches (%)")
    ax_z.barh(yy - 0.19, t_time, height=0.34, color="#9cc3dd",
              edgecolor="white", lw=0.6, zorder=3, label="time (%)")
    for y, a, b in zip(yy, share, t_time):
        ax_z.text(a + 0.9, y + 0.19, f"{a:.0f}", ha="left", va="center",
                  fontsize=6.2, color=SUB)
        ax_z.text(b + 0.9, y - 0.19, f"{b:.0f}", ha="left", va="center",
                  fontsize=6.2, color=SUB)
    ax_z.set_yticks(yy)
    ax_z.set_yticklabels([t.replace(" third", "\nthird") for t in THIRD],
                         fontsize=6.6)
    ax_z.set_xlim(0.0, max(share.max(), t_time.max()) * 1.28)
    ax_z.set_xlabel("share of the match (%)", fontsize=8.0)
    ax_z.tick_params(axis="x", labelsize=7.0)
    ax_z.grid(axis="x", lw=0.5, color="#ececec", zorder=0)
    ax_z.spines["top"].set_visible(False)
    ax_z.spines["right"].set_visible(False)
    ax_z.legend(loc="lower right", frameon=False, fontsize=6.8,
                handlelength=1.5, borderaxespad=0.2)
    ax_z.set_title("Where the touches happened", fontsize=8.5,
                   fontweight="bold")

    fig.legend(handles=[
        Line2D([0], [0], marker="P", color="none", markerfacecolor="none",
               markeredgecolor=HOME, markeredgewidth=1.2, markersize=7,
               label="average position (time-weighted)"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=ACCENT,
               markeredgecolor="white", markersize=7,
               label="activity centroid (touch-weighted)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=RULE, markersize=4.5, label="touch event"),
        Line2D([0], [0], color=LINE, lw=0.9,
               label="pitch markings (105 x 68 m)")],
        loc="lower left", bbox_to_anchor=(0.030, 0.050), ncol=4,
        frameon=False, fontsize=7.0, handlelength=1.8, columnspacing=1.9)

    panel_labels_inches(fig, [ax_p, ax_m, ax_z], dx=-0.26, dy=0.12)

    top = int(np.argmax(share))

    print(f"distance {dist_km:.2f} km  mean ({mean_pos[0]:.1f}, "
          f"{mean_pos[1]:.1f})  centroid ({centroid[0]:.1f}, "
          f"{centroid[1]:.1f})")
    print("touch share %:", share.round(1).tolist())
    print("time share  %:", t_time.round(1).tolist())
    print(f"peak occupancy {np.nanmax(field):.3f}% per 100 m2")
    savefig(fig, "864_tac_pitch_heatmap.png")


if __name__ == "__main__":
    main()

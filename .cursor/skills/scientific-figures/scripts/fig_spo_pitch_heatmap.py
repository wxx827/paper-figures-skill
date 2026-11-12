"""Match activity heat map on a full-size pitch.

Everything in this figure is read out of one simulated tracking file: eleven
Ornstein-Uhlenbeck position tracks sampled at 1 Hz for a 90 min match, clipped
to the playing surface.  The heat map is the smoothed 2-D histogram of those
samples, the average-position markers are the per-player track means, the
marker areas are the touch counts, the zone shares bin a touch subsample of
the same tracks into thirds and channels, and the distance bars are the summed
step lengths of the tracks.  Change the seed and every printed number moves.

Pitch geometry is the FIFA standard 105 x 68 m surface: 16.5 m penalty areas
40.32 m wide, 5.5 m goal areas 18.32 m wide, penalty spots at 11 m, a 9.15 m
centre circle and penalty arcs, and 1 m corner arcs.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Rectangle
from scipy.ndimage import gaussian_filter

from pubstyle import annotate_box, get_cmap, savefig, set_pub_style
from figkit import (halo_text, panel_labels_inches, safe_colorbar,
                    shared_norm)

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HOME = "#3b5b92"          # house blue - the team under analysis
GOLD = "#c9a227"          # house gold - opponent / secondary
LINE = "#b3b3b3"          # pitch markings on a white surface

PITCH_L, PITCH_W = 105.0, 68.0
MATCH_MIN = 90.0
HZ_SAMPLES = 5400         # 1 Hz for 90 min
N_TOUCH = 620             # team touches in a match

# shirt number, position label, anchor x, anchor y, kx, ky, sigma x, sigma y
SQUAD = [
    (1,  "GK", 12.0, 34.0, 0.0018, 0.0025, 0.75, 0.62),
    (2,  "RB", 25.0, 57.5, 0.0025, 0.0100, 1.75, 1.25),
    (5,  "RCB", 21.0, 42.0, 0.0035, 0.0090, 1.35, 1.15),
    (6,  "LCB", 21.0, 26.0, 0.0035, 0.0090, 1.35, 1.15),
    (3,  "LB", 25.0, 10.5, 0.0025, 0.0100, 1.75, 1.25),
    (4,  "DM", 42.0, 34.0, 0.0035, 0.0060, 1.55, 1.45),
    (8,  "RCM", 52.0, 46.0, 0.0030, 0.0060, 1.70, 1.50),
    (10, "LCM", 55.0, 24.0, 0.0030, 0.0060, 1.70, 1.50),
    (7,  "RW", 74.0, 58.0, 0.0030, 0.0090, 1.70, 1.30),
    (9,  "ST", 80.0, 34.0, 0.0035, 0.0060, 1.60, 1.55),
    (11, "LW", 74.0, 10.0, 0.0030, 0.0090, 1.70, 1.30),
]

THIRD = ["defensive", "middle", "attacking"]
CHANNEL = ["right", "centre", "left"]


def draw_pitch(ax, color=LINE, lw=1.0, zorder=5):
    """Standard 105 x 68 m markings; call after the heat map so lines stay up."""
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
        ax.add_patch(Circle((x_spot, PITCH_W / 2.0), 0.40, facecolor=color,
                            edgecolor="none", zorder=zorder))
        t1, t2 = ((-arc_half, arc_half) if side == 0
                  else (180.0 - arc_half, 180.0 + arc_half))
        ax.add_patch(Arc((x_spot, PITCH_W / 2.0), 2 * 9.15, 2 * 9.15,
                         theta1=t1, theta2=t2, edgecolor=color, lw=lw,
                         zorder=zorder))
        # goal frame, drawn just outside the goal line
        x_goal = -1.9 if side == 0 else PITCH_L
        ax.add_patch(Rectangle((x_goal, PITCH_W / 2.0 - 3.66), 1.9, 7.32,
                               fill=False, edgecolor=color, lw=lw,
                               zorder=zorder))

    for (cx, cy, t1, t2) in ((0.0, 0.0, 0.0, 90.0), (0.0, PITCH_W, 270.0, 360.0),
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
        x = np.clip(x + kx * (x0 - x) + ex[t], 0.4, PITCH_L - 0.4)
        y = np.clip(y + ky * (y0 - y) + ey[t], 0.4, PITCH_W - 0.4)
        p[t] = (x, y)
    return p


def main():
    set_pub_style()
    rng = np.random.default_rng(606)

    tracks, dist_km, means = [], [], []
    for num, pos, x0, y0, kx, ky, sx, sy in SQUAD:
        p = ou_track(rng, x0, y0, kx, ky, sx, sy, HZ_SAMPLES)
        tracks.append(p)
        dist_km.append(np.hypot(*np.diff(p, axis=0).T).sum() / 1000.0)
        means.append(p.mean(axis=0))
    dist_km = np.array(dist_km)
    means = np.array(means)

    allp = np.vstack(tracks)

    # touches: a subsample of the same tracks, biased forward because a team
    # plays the ball where it is trying to progress
    w = 0.35 + 0.65 * (allp[:, 0] / PITCH_L) ** 1.6
    idx = rng.choice(allp.shape[0], size=N_TOUCH, replace=False, p=w / w.sum())
    touch = allp[idx]
    owner = idx // HZ_SAMPLES
    touch_n = np.bincount(owner, minlength=len(SQUAD)).astype(float)

    # ---- occupancy field ---------------------------------------------------
    counts, _xe, _ye = np.histogram2d(
        allp[:, 0], allp[:, 1], bins=[np.arange(0.0, PITCH_L + 1.0, 1.0),
                                      np.arange(0.0, PITCH_W + 1.0, 1.0)])
    sm = gaussian_filter(counts, 3.4, mode="nearest")
    # minutes spent inside each 100 m^2 patch over the whole match
    occ = MATCH_MIN * 100.0 * sm / sm.sum()
    field = np.where(occ < 0.10 * occ.max(), np.nan, occ)

    # ---- zone shares -------------------------------------------------------
    ti = np.clip((touch[:, 0] / (PITCH_L / 3.0)).astype(int), 0, 2)
    ci = np.clip((touch[:, 1] / (PITCH_W / 3.0)).astype(int), 0, 2)
    zone = np.zeros((3, 3))
    for a, b in zip(ti, ci):
        zone[a, 2 - b] += 1.0
    zone = 100.0 * zone / zone.sum()

    # ---- layout ------------------------------------------------------------
    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.82],
                          left=0.030, right=0.960, top=0.850, bottom=0.150,
                          wspace=0.34, hspace=0.60)
    ax_p = fig.add_subplot(gs[:, 0])
    ax_z = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1])

    cmap = get_cmap("cb_sequential_blue")
    cmap = cmap.copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    norm = shared_norm([field], robust=(1, 97.0))
    im = ax_p.imshow(field.T, origin="lower", extent=[0, PITCH_L, 0, PITCH_W],
                     cmap=cmap, norm=norm, alpha=0.82, zorder=2,
                     interpolation="bilinear")
    draw_pitch(ax_p, zorder=5)

    smax = touch_n.max()
    for (num, pos, *_), (mx, my), tn in zip(SQUAD, means, touch_n):
        r = 1.55 + 1.55 * np.sqrt(tn / smax)
        ax_p.add_patch(Circle((mx, my), r, facecolor=HOME, edgecolor="white",
                              lw=1.0, zorder=7, alpha=0.95))
        ax_p.text(mx, my, str(num), ha="center", va="center", fontsize=6.4,
                  color="white", zorder=8)
        halo_text(ax_p, mx, my - r - 0.9, pos, ha="center", va="top",
                  fontsize=6.0, color=INK, halo="white", halo_lw=1.8,
                  zorder=8)

    ax_p.annotate("", xy=(84.0, 73.5), xytext=(64.0, 73.5),
                  arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))
    ax_p.text(62.0, 73.5, "attacking direction", ha="right", va="center",
              fontsize=6.4, color=MUTED)

    ax_p.set_xlim(-4.5, 109.5)
    ax_p.set_ylim(-4.0, 79.0)
    ax_p.set_aspect("equal")
    ax_p.set_xticks([])
    ax_p.set_yticks([])
    for s in ax_p.spines.values():
        s.set_visible(False)
    ax_p.set_title("Where the team spent the match", fontsize=9.0,
                   fontweight="bold", pad=6)

    annotate_box(
        ax_p,
        f"team distance  {dist_km.sum():.1f} km\n"
        f"per player  {dist_km.mean():.2f} km (max {dist_km.max():.2f})\n"
        f"touches  {N_TOUCH} | attacking third  {zone[2].sum():.0f}%",
        loc="upper left", fontsize=6.6, margin=0.014)

    safe_colorbar(fig, im, ax_p, label="occupancy (min per 100 m$^2$)",
                  side="bottom", width=0.10, pad=0.10, shrink=0.52,
                  labelsize=6.5)

    # ---- zone shares -------------------------------------------------------
    labels, vals = [], []
    for a in (2, 1, 0):
        for c in range(3):
            labels.append(f"{THIRD[a][:3]} {CHANNEL[c]}")
            vals.append(zone[a, c])
    yy = np.arange(len(vals))[::-1]
    cols = [HOME if v >= np.mean(vals) else "#9cc3dd" for v in vals]
    ax_z.barh(yy, vals, height=0.66, color=cols, edgecolor="white", lw=0.6,
              zorder=3)
    for y, v in zip(yy, vals):
        ax_z.text(v + 0.45, y, f"{v:.1f}", ha="left", va="center",
                  fontsize=6.2, color=SUB)
    ax_z.set_yticks(yy)
    ax_z.set_yticklabels(labels, fontsize=6.4)
    ax_z.set_xlim(0.0, max(vals) * 1.30)
    ax_z.set_xlabel("share of team touches (%)", fontsize=7.4)
    ax_z.tick_params(axis="x", labelsize=6.5)
    ax_z.grid(axis="x", lw=0.5, color="#ececec", zorder=0)
    ax_z.set_title("Touches by pitch zone", fontsize=8.5, fontweight="bold")

    # ---- distance covered --------------------------------------------------
    order = np.argsort(dist_km)
    yy2 = np.arange(len(order))
    ax_d.barh(yy2, dist_km[order], height=0.66, color="#5f9dc6",
              edgecolor="white", lw=0.6, zorder=3)
    ax_d.barh(yy2[-1], dist_km[order][-1], height=0.66, color=HOME,
              edgecolor="white", lw=0.6, zorder=4)
    ax_d.set_yticks(yy2)
    ax_d.set_yticklabels([f"#{SQUAD[i][0]} {SQUAD[i][1]}" for i in order],
                         fontsize=6.2)
    ax_d.axvline(dist_km.mean(), color="#808080", lw=0.9, ls="--", zorder=5)
    ax_d.text(dist_km.mean() + 0.18, 0.10, f"mean {dist_km.mean():.2f} km",
              ha="left", va="center", fontsize=6.2, color=SUB, zorder=6)
    ax_d.set_xlim(0.0, dist_km.max() * 1.14)
    ax_d.set_xlabel("distance covered (km)", fontsize=7.4)
    ax_d.tick_params(axis="x", labelsize=6.5)
    ax_d.grid(axis="x", lw=0.5, color="#ececec", zorder=0)
    ax_d.set_title("Work rate by player", fontsize=8.5, fontweight="bold")

    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=HOME,
               markeredgecolor="white", markersize=6.5,
               label="average position, area = touches"),
        Line2D([0], [0], color=LINE, lw=1.0, label="pitch markings (105 x 68 m)"),
        Line2D([0], [0], color="#808080", lw=0.9, ls="--",
               label="squad mean distance")],
        loc="lower center", bbox_to_anchor=(0.5, 0.086), ncol=3,
        frameon=False, fontsize=7.0, handlelength=1.8, columnspacing=2.0)

    panel_labels_inches(fig, [ax_p, ax_z, ax_d], dx=-0.24, dy=0.14)

    ta, ca = np.unravel_index(int(np.argmax(zone)), zone.shape)

    print(f"team distance {dist_km.sum():.1f} km  mean {dist_km.mean():.2f}  "
          f"max {dist_km.max():.2f} km")
    print("zone shares %:", zone.round(1).tolist())
    print(f"peak occupancy {np.nanmax(field):.2f} min per 100 m2")
    savefig(fig, "606_spo_pitch_heatmap.png")


if __name__ == "__main__":
    main()

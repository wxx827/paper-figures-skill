"""Hagerstrand space-time cube: five daily paths through one city.

The horizontal plane is the 6 x 6 km study area and the vertical axis is
clock time, so a sloped segment is travel and a vertical segment is a stay
episode.  Each path is echoed twice: greyed onto the floor, where it becomes
the ordinary spatial trace that a map would show, and onto the far wall,
where it becomes the x-versus-time profile that a time-space diagram would
show - the cube is exactly the object those two familiar views project from.
The viewpoint (elev 22, azim -58) was chosen so no path is hidden behind
another and both projection surfaces stay visible.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, CATEGORICAL, COLORS
from figkit import panel_labels_inches

W, H = 6.0, 6.0                      # city extent in km
T0, T1 = 6.0, 22.0                   # clock hours on the vertical axis
INK = COLORS["ink"]
SUB = COLORS["sub"]
MUTED = COLORS["muted"]
GHOST = "#b8b5ae"
FLOOR = "#f4f3f0"

PEOPLE = [
    ("P1 commuter", (0.9, 1.2), [(3.1, 3.2), (3.6, 4.4), (0.9, 1.2)],
     7.1, 18.4),
    ("P2 courier", (5.2, 0.9), [(4.2, 3.0), (1.6, 4.6), (4.9, 5.1),
                                (2.6, 2.4), (5.2, 0.9)], 6.6, 20.2),
    ("P3 student", (1.4, 5.0), [(2.7, 3.7), (4.6, 2.2), (1.4, 5.0)],
     8.2, 17.3),
    ("P4 shopper", (4.9, 4.8), [(3.2, 2.6), (1.2, 2.1), (4.9, 4.8)],
     9.6, 16.1),
    ("P5 shift worker", (2.2, 0.7), [(5.0, 3.9), (3.4, 5.2), (2.2, 0.7)],
     12.4, 21.5),
]
SPEED = [21.0, 26.0, 15.0, 18.0, 24.0]      # km/h between stops
STAY = [(6.4, 1.4), (0.9, 0.35), (3.2, 1.1), (1.1, 0.6), (5.0, 1.2)]


def make_path(rng, home, dests, t_start, v_kmh, stay_mu, stay_sd, m=16):
    """Alternating travel legs and stay episodes; returns x, y, t arrays."""
    cur = np.array(home, dtype=float)
    xs, ys, ts = [cur[0]], [cur[1]], [t_start]
    t = t_start
    for j, d in enumerate(dests):
        d = np.array(d, dtype=float)
        vec = d - cur
        dist = float(np.hypot(*vec))
        dur = dist / (v_kmh * rng.uniform(0.82, 1.18))
        nvec = np.array([-vec[1], vec[0]]) / max(dist, 1e-9)
        amp = rng.normal(0.0, 0.16) * dist
        for k in range(1, m + 1):
            f = k / m
            p = cur + vec * f + nvec * amp * np.sin(np.pi * f)
            xs.append(p[0])
            ys.append(p[1])
            ts.append(t + dur * f)
        t += dur
        if j < len(dests) - 1:
            hold = max(0.25, rng.normal(stay_mu, stay_sd))
            xs.append(d[0])
            ys.append(d[1])
            ts.append(t + hold)
            t += hold
        cur = d
    return np.array(xs), np.array(ys), np.array(ts)


def main():
    set_pub_style()
    rng = np.random.default_rng(483)

    paths = []
    for (name, home, dests, t0, t_end), v, (mu, sd) in zip(PEOPLE, SPEED,
                                                           STAY):
        x, y, t = make_path(rng, home, dests, t0, v, mu, sd)
        # Stretch the day onto its intended span; the travel / stay ratio is
        # preserved, and no artist can leave the box - 3D axes do not clip.
        t = t0 + (t - t[0]) * (t_end - t0) / (t[-1] - t[0])
        keep = (t >= T0) & (t <= T1)
        x, y, t = x[keep], y[keep], t[keep]
        paths.append((name, x, y, t))

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = fig.add_axes([0.010, 0.020, 0.735, 0.885], projection="3d")
    axr = fig.add_axes([0.800, 0.135, 0.170, 0.600])

    ax.view_init(elev=22, azim=-58)
    ax.set_box_aspect((1.0, 1.0, 0.94))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_zlim(T0, T1)

    for pane, col in ((ax.xaxis.pane, "#f7f6f4"), (ax.yaxis.pane, "#f7f6f4"),
                      (ax.zaxis.pane, FLOOR)):
        pane.set_facecolor(col)
        pane.set_edgecolor("#d8d5cf")
        pane.set_alpha(1.0)
    ax.grid(True)
    ax.xaxis._axinfo["grid"].update(color="#e2dfd9", linewidth=0.5)
    ax.yaxis._axinfo["grid"].update(color="#e2dfd9", linewidth=0.5)
    ax.zaxis._axinfo["grid"].update(color="#e2dfd9", linewidth=0.5)

    # vertical corner posts, so the volume reads as a box
    for cx, cy in [(0, 0), (W, 0), (0, H), (W, H)]:
        ax.plot([cx, cx], [cy, cy], [T0, T1], color="#cfccc5", lw=0.7,
                zorder=1)

    # ---------------------------------------------------------- projections
    for k, (_name, x, y, t) in enumerate(paths):
        ax.plot(x, y, np.full_like(t, T0), color=GHOST, lw=0.9, alpha=0.85,
                zorder=2)                                   # floor footprint
        ax.plot(x, np.full_like(x, H), t, color=CATEGORICAL[k], lw=0.8,
                alpha=0.42, zorder=2)                       # x-t wall profile

    # ------------------------------------------------------------ the paths
    for k, (_name, x, y, t) in enumerate(paths):
        ax.plot(x, y, t, color="white", lw=3.0, solid_capstyle="round",
                zorder=4)
        ax.plot(x, y, t, color=CATEGORICAL[k], lw=1.7, solid_capstyle="round",
                zorder=5)
        step = np.hypot(np.diff(x), np.diff(y))
        hold = np.flatnonzero(step < 1e-6)
        for i in hold:                                  # stay episode markers
            ax.plot([x[i]], [y[i]], [0.5 * (t[i] + t[i + 1])], marker="o",
                    ms=4.4, mfc="white", mec=CATEGORICAL[k], mew=1.2,
                    zorder=6)
        ax.plot([x[0]], [y[0]], [t[0]], marker="s", ms=3.4,
                color=CATEGORICAL[k], zorder=6)

    ax.set_xlabel("easting (km)", fontsize=7.6, labelpad=-4)
    ax.set_ylabel("northing (km)", fontsize=7.6, labelpad=-4)
    ax.set_zlabel("time of day (h)", fontsize=7.6, labelpad=-2)
    ax.set_xticks([0, 2, 4, 6])
    ax.set_yticks([0, 2, 4, 6])
    ax.set_zticks([6, 10, 14, 18, 22])
    ax.set_zticklabels(["06:00", "10:00", "14:00", "18:00", "22:00"])
    ax.tick_params(labelsize=6.6, pad=-1.5)

    handles = [Line2D([], [], color=CATEGORICAL[k], lw=1.8,
                      label=PEOPLE[k][0]) for k in range(len(PEOPLE))]
    handles.append(Line2D([], [], color="none", marker="o", ms=4.4,
                          mfc="white", mec="#555555", mew=1.2,
                          label="stay episode"))
    leg = ax.legend(handles=handles, loc="upper left",
                    bbox_to_anchor=(0.015, 0.975), fontsize=6.8,
                    labelspacing=0.45, handlelength=1.8, frameon=False)
    leg.set_zorder(9)
    fig.text(0.030, 0.615, "grey trace on the floor: spatial footprint\n"
             "tinted trace on the far wall: x-time profile",
             fontsize=6.5, color=SUB, ha="left", va="top", linespacing=1.45)

    # --------------------------------------------- (b) who is moving, by hour
    tb = np.arange(T0, T1 + 0.001, 0.5)
    mid = 0.5 * (tb[:-1] + tb[1:])
    moving = np.zeros(mid.size)
    staying = np.zeros(mid.size)
    for _name, x, y, t in paths:
        v = np.hypot(np.diff(x), np.diff(y)) / np.fmax(np.diff(t), 1e-6)
        tc = 0.5 * (t[:-1] + t[1:])
        for b in range(mid.size):
            m = (tc >= tb[b]) & (tc < tb[b + 1])
            if not m.any():
                continue
            if v[m].max() > 0.5:
                moving[b] += 1
            else:
                staying[b] += 1

    axr.barh(mid, moving, height=0.44, color="#3b5b92", edgecolor="white",
             linewidth=0.3, zorder=3, label="travelling")
    axr.barh(mid, staying, height=0.44, left=moving, color="#c2bfb8",
             edgecolor="white", linewidth=0.3, zorder=3, label="staying")
    axr.set_ylim(T0, T1)
    axr.set_xlim(0, len(paths))
    axr.set_yticks([6, 10, 14, 18, 22])
    axr.set_yticklabels(["06:00", "10:00", "14:00", "18:00", "22:00"],
                        fontsize=6.6)
    axr.set_xticks(np.arange(0, len(paths) + 1, 1))
    axr.set_xlabel("people (count)", fontsize=7.4, labelpad=2)
    axr.set_ylabel("time of day (h)", fontsize=7.4, labelpad=2)
    axr.set_title("Activity by half hour", fontsize=8.6, fontweight="bold",
                  color=INK, pad=4)
    axr.tick_params(labelsize=6.6, length=2.4)
    axr.grid(axis="x", color="#ececec", lw=0.55, zorder=0)
    axr.legend(loc="lower center", bbox_to_anchor=(0.5, -0.175), ncol=2,
               fontsize=6.5, frameon=False, handlelength=1.2,
               columnspacing=0.9, handletextpad=0.4)

    span = np.array([p[3][-1] - p[3][0] for p in paths])
    axr.text(0.0, -0.255, f"n = {len(paths)} paths, "
             f"{span.mean():.1f} h mean span", transform=axr.transAxes,
             fontsize=6.3, color=MUTED, ha="left", va="top")

    panel_labels_inches(fig, [axr], labels=["(b)"], dx=-0.34, dy=0.24)

    savefig(fig, "483_stm_spacetime_cube.png")


if __name__ == "__main__":
    main()

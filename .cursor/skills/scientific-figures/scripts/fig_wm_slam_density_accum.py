"""Occupancy density converging as range scans accumulate (5 / 18 / 24).

A 1440-beam range finder is ray-marched against an analytic floor plan from
poses spread along a ring patrol.  Hits are splatted with a 22 cm jitter
and fused with the usual inverse-sensor-model log-odds update, so the early map
is a smear of low-confidence blobs and only the cells seen from many poses ever
reach the deep end of the ladder.

All three panels share one axis range and one colour range (0 to 1); the single
narrow colourbar on the right therefore reads for the whole row.

Single-hue blue batch (BRIEF_WM_BLUE.md section 2): background light blue,
free space near white, occupied cells deep navy.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, add_panel_label

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"

# free (near white) -> unknown (light blue) -> occupied (deep navy)
DENSITY_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "wm_blue_density",
    [(0.00, BLUE[7]), (0.34, BLUE[6]), (0.50, BLUE[5]), (0.66, BLUE[4]),
     (0.82, BLUE[2]), (1.00, BLUE[0])], N=256)

W, H = 22.0, 15.0
RES = 0.10
T = 0.32
MAX_RANGE = 13.0
N_BEAMS = 1440
SIGMA_R = 0.035
JITTER = 0.22                  # per-hit registration error (m)
L_FREE, L_OCC, L_CLIP = -0.55, 1.15, 5.2
STAGES = (5, 18, 24)


def wall_rects():
    R = []

    def hwall(y, x0, x1, gaps=()):
        cuts = [x0]
        for g0, g1 in gaps:
            cuts += [g0, g1]
        cuts.append(x1)
        for a, b in zip(cuts[0::2], cuts[1::2]):
            if b - a > 1e-6:
                R.append((a, y - T / 2, b - a, T))

    def vwall(x, y0, y1, gaps=()):
        cuts = [y0]
        for g0, g1 in gaps:
            cuts += [g0, g1]
        cuts.append(y1)
        for a, b in zip(cuts[0::2], cuts[1::2]):
            if b - a > 1e-6:
                R.append((x - T / 2, a, T, b - a))

    hwall(T / 2, 0.0, W)
    hwall(H - T / 2, 0.0, W)
    vwall(T / 2, 0.0, H)
    vwall(W - T / 2, 0.0, H)
    # central block
    hwall(5.0, 6.0, 16.0, gaps=[(10.4, 11.6)])
    hwall(10.0, 6.0, 16.0)
    vwall(6.0, 5.0, 10.0)
    vwall(16.0, 5.0, 10.0)
    # side partitions
    vwall(6.0, T, 5.0, gaps=[(1.7, 2.7)])
    vwall(16.0, 10.0, H - T, gaps=[(11.9, 12.9)])
    hwall(3.4, 16.0, W - T, gaps=[(18.1, 19.1)])
    hwall(11.6, T, 6.0, gaps=[(2.4, 3.4)])
    return R


RECTS = wall_rects()

ROUTE = [(2.6, 2.0), (11.0, 2.4), (19.0, 2.0), (19.6, 6.6), (19.0, 12.6),
         (11.0, 12.6), (3.2, 12.9), (2.6, 7.6), (2.6, 3.0)]


def occupied(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.zeros(np.broadcast(x, y).shape, bool)
    for x0, y0, w, h in RECTS:
        m |= (x >= x0) & (x <= x0 + w) & (y >= y0) & (y <= y0 + h)
    return m | (x < 0) | (x > W) | (y < 0) | (y > H)


def poses(n=24):
    pts = np.asarray(ROUTE, float)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.linspace(0.0, cum[-1] * 0.995, n)
    return np.column_stack([np.interp(s, cum, pts[:, 0]),
                            np.interp(s, cum, pts[:, 1])])


def scan(p, rng, step=RES / 2.0):
    ang = np.linspace(0.0, 2 * np.pi, N_BEAMS, endpoint=False)
    t = np.arange(step, MAX_RANGE + step, step)
    X = p[0] + np.cos(ang)[:, None] * t[None, :]
    Y = p[1] + np.sin(ang)[:, None] * t[None, :]
    hit = occupied(X, Y)
    any_hit = hit.any(axis=1)
    first = np.where(any_hit, hit.argmax(axis=1), len(t) - 1)
    r = t[first] + np.where(any_hit, rng.normal(0, SIGMA_R, N_BEAMS), 0.0)
    return ang, np.clip(r, step, MAX_RANGE), any_hit


def fuse(P, rng):
    """Log-odds fusion of every scan in ``P``; returns the final belief."""
    nx, ny = int(round(W / RES)), int(round(H / RES))
    L = np.zeros((ny, nx))
    step = RES / 2.0
    t = np.arange(step, MAX_RANGE, step)
    for p in P:
        ang, r, hit = scan(p, rng, step)
        keep = t[None, :] < (r[:, None] - RES)
        fx = (p[0] + np.cos(ang)[:, None] * t[None, :])[keep]
        fy = (p[1] + np.sin(ang)[:, None] * t[None, :])[keep]
        flat = np.unique(np.clip((fy / RES).astype(int), 0, ny - 1) * nx +
                         np.clip((fx / RES).astype(int), 0, nx - 1))
        L.ravel()[flat] += L_FREE
        hx = p[0] + np.cos(ang[hit]) * r[hit] + rng.normal(0, JITTER, hit.sum())
        hy = p[1] + np.sin(ang[hit]) * r[hit] + rng.normal(0, JITTER, hit.sum())
        hi = np.clip((hy / RES).astype(int), 0, ny - 1)
        hj = np.clip((hx / RES).astype(int), 0, nx - 1)
        np.add.at(L, (hi, hj), L_OCC)
        np.clip(L, -L_CLIP, L_CLIP, out=L)
    return 1.0 - 1.0 / (1.0 + np.exp(L))


def main():
    set_pub_style(base_font=8)
    # each stage re-runs the whole survey with k poses spread over the ring,
    # so every panel covers the same area and only the sharpness changes
    stage_poses = {k: poses(k) for k in STAGES}
    snaps = {k: fuse(stage_poses[k], np.random.default_rng(21 + k))
             for k in STAGES}

    # one shared colour range for all three panels
    vmin = min(float(m.min()) for m in snaps.values())
    vmax = max(float(m.max()) for m in snaps.values())
    vmin, vmax = min(vmin, 0.0), max(vmax, 1.0)

    fig = plt.figure(figsize=(10.4, 3.6))
    left, gap, cbw = 0.028, 0.018, 0.013
    pw = (0.918 - left - 2 * gap) / 3.0
    ph = pw * (10.4 / 3.6) * (H / W)
    bottom = 0.100
    tags = "ABC"
    label_at = [(0.26, 0.42), (0.44, 0.82), (0.55, 0.28)]
    pose_pick = [-1, 3, 8]

    for i, k in enumerate(STAGES):
        P = stage_poses[k]
        ax = fig.add_axes([left + i * (pw + gap), bottom, pw, ph])
        ax.imshow(snaps[k], extent=[0, W, 0, H], origin="lower",
                  interpolation="nearest", cmap=DENSITY_CMAP,
                  vmin=vmin, vmax=vmax)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.9)
            sp.set_color(INK)
        rp = P[pose_pick[i]]
        ax.plot(*rp, marker="o", ms=5.2, mfc=BLUE[0], mec="white", mew=0.9,
                ls="none", zorder=6)
        lx, ly = label_at[i]
        ax.annotate("Robot position (%s)" % tags[i], xy=tuple(rp),
                    xycoords="data", xytext=(lx, ly), textcoords="axes fraction",
                    fontsize=6.8, fontweight="bold", color=INK,
                    ha="left", va="center", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.26", fc="white",
                              ec="#c3ccd4", lw=0.6, alpha=0.95),
                    arrowprops=dict(arrowstyle="-", color="#111417", lw=0.8,
                                    shrinkA=1.0, shrinkB=3.0))
        ax.plot(P[:, 0], P[:, 1], ls="none", marker="o", ms=1.9,
                mfc=BLUE[1], mec="white", mew=0.4, alpha=0.9, zorder=5)
        crisp = float((snaps[k] > 0.80).mean()) * 100.0
        ax.set_title("%d scans fused" % k, fontsize=8.4, pad=3.0,
                     color=INK, fontweight="bold")
        ax.text(0.018, 0.030, "crisp cells  %.1f%%" % crisp,
                transform=ax.transAxes, fontsize=6.2, color=INK, va="bottom",
                zorder=8,
                bbox=dict(boxstyle="round,pad=0.22", fc="white",
                          ec="#c3ccd4", lw=0.5, alpha=0.95))
        add_panel_label(ax, "(%s)" % "abc"[i], x=0.955, y=0.030, size=9.5)
        ax.texts[-1].set_bbox(dict(boxstyle="round,pad=0.20", fc="white",
                                   ec="#c3ccd4", lw=0.5, alpha=0.95))
        ax.texts[-1].set_ha("right")

    cax = fig.add_axes([0.945, bottom + ph * 0.24, cbw, ph * 0.52])
    cb = mpl.colorbar.ColorbarBase(
        cax, cmap=DENSITY_CMAP,
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
        orientation="vertical", ticks=[0.0, 0.5, 1.0])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(INK)
    cb.ax.set_yticklabels(["0  free", "0.5", "1  occupied"])
    cb.ax.tick_params(labelsize=6.2, width=0.6, length=2.0, color=INK)
    cb.set_label("P(occupied)", fontsize=6.6, color=INK, labelpad=2.0)

    savefig(fig, "1136_wm_slam_density_accum.png")


if __name__ == "__main__":
    main()

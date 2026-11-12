"""Pose-graph SLAM: drift, loop closure, graph topology and landmark spread.

Wheel odometry is corrupted with per-step Gaussian noise and integrated to
produce a drifting dead-reckoning track.  A textbook 2-D pose-graph is then
solved with Gauss-Newton on the SE(2) manifold (analytic Jacobians, first
pose anchored) both incrementally - re-optimising the graph available at
each checkpoint, which is what produces the step drop in panel (b) - and in
batch.  Panel (d) accumulates range-bearing information matrices to show the
landmark covariance shrinking as 1/k with the number of sightings.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle
from pubstyle import set_pub_style, savefig, add_panel_label

W, H = 20.0, 14.0
BLOCKS = [(4.2, 4.2, 3.8, 5.8), (12.0, 4.2, 3.8, 5.8)]
STEP = 0.26                 # distance between consecutive poses (m)
LAPS = 1.18
SIG_XY, SIG_TH = 0.014, 0.0070      # odometry noise per step (m, rad)
SIG_LC_XY, SIG_LC_TH = 0.030, 0.0080   # loop-closure constraint
SIG_RANGE, SIG_BEAR = 0.04, np.deg2rad(0.6)
SENSE_R = 6.0

OBS_C = "#3a3a3a"
GT_C = "#2b2b2b"
OD_C = "#d55e00"
SL_C = "#0072b2"
LC_C = "#b23a48"
LM_C = "#7b3294"


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def v2t(v):
    c, s = np.cos(v[2]), np.sin(v[2])
    return np.array([[c, -s, v[0]], [s, c, v[1]], [0.0, 0.0, 1.0]])


def t2v(T):
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])


def ground_truth():
    """Rounded-rectangle loop driven for 1.18 laps, uniform arc length."""
    x0, x1, y0, y1, r = 2.1, 17.9, 2.1, 11.9, 1.5
    seg = []
    seg.append(("l", (x0 + r, y0), (x1 - r, y0)))
    seg.append(("a", (x1 - r, y0 + r), (-np.pi / 2, 0.0)))
    seg.append(("l", (x1, y0 + r), (x1, y1 - r)))
    seg.append(("a", (x1 - r, y1 - r), (0.0, np.pi / 2)))
    seg.append(("l", (x1 - r, y1), (x0 + r, y1)))
    seg.append(("a", (x0 + r, y1 - r), (np.pi / 2, np.pi)))
    seg.append(("l", (x0, y1 - r), (x0, y0 + r)))
    seg.append(("a", (x0 + r, y0 + r), (np.pi, 1.5 * np.pi)))
    lens = []
    for kind, a, b in seg:
        lens.append(np.hypot(b[0] - a[0], b[1] - a[1]) if kind == "l"
                    else r * abs(b[1] - b[0]))
    per = sum(lens)
    s = np.arange(0.0, per * LAPS, STEP)
    P = np.zeros((len(s), 3))
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    for m, sv in enumerate(s):
        u = sv % per
        k = int(np.searchsorted(cum, u, "right") - 1)
        k = min(k, len(seg) - 1)
        loc = u - cum[k]
        kind, a, b = seg[k]
        if kind == "l":
            L = lens[k]
            t = loc / L
            px, py = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            th = np.arctan2(b[1] - a[1], b[0] - a[0])
        else:
            ang = b[0] + loc / r
            px, py = a[0] + r * np.cos(ang), a[1] + r * np.sin(ang)
            th = ang + np.pi / 2
        P[m] = (px, py, wrap(th))
    return P


def rel(a, b):
    return t2v(np.linalg.inv(v2t(a)) @ v2t(b))


def compose(a, d):
    return t2v(v2t(a) @ v2t(d))


def build_graph(gt, rng):
    n = len(gt)
    odo = np.zeros((n - 1, 3))
    for i in range(n - 1):
        odo[i] = (rel(gt[i], gt[i + 1]) +
                  rng.normal([0, 0, 0], [SIG_XY, SIG_XY, SIG_TH]))
    dead = np.zeros((n, 3))
    dead[0] = gt[0]
    for i in range(n - 1):
        dead[i + 1] = compose(dead[i], odo[i])

    # loop closure: revisit detection against poses at least 120 steps old
    loops = []
    for i in range(120, n, 2):
        d = np.hypot(*(gt[:i - 120, :2] - gt[i, :2]).T)
        if d.size and d.min() < 0.30:
            j = int(np.argmin(d))
            loops.append((j, i, rel(gt[j], gt[i]) +
                          rng.normal([0, 0, 0],
                                     [SIG_LC_XY, SIG_LC_XY, SIG_LC_TH])))
    return dead, odo, loops


def optimise(x0, sm, loops, n_pose, iters=10):
    x = x0[:n_pose].copy()
    om_o = np.diag([1 / SIG_XY ** 2, 1 / SIG_XY ** 2, 1 / SIG_TH ** 2])
    om_l = np.diag([1 / SIG_LC_XY ** 2, 1 / SIG_LC_XY ** 2,
                    1 / SIG_LC_TH ** 2])
    edges = [(i, i + 1, sm[i], om_o) for i in range(n_pose - 1)]
    edges += [(a, b, z, om_l) for a, b, z in loops
              if a < n_pose and b < n_pose]
    N = n_pose
    for _ in range(iters):
        Hm = np.zeros((3 * N, 3 * N))
        bv = np.zeros(3 * N)
        for i, j, z, om in edges:
            Xi, Xj, Z = v2t(x[i]), v2t(x[j]), v2t(z)
            e = t2v(np.linalg.inv(Z) @ np.linalg.inv(Xi) @ Xj)
            e[2] = wrap(e[2])
            ci, si = np.cos(x[i][2]), np.sin(x[i][2])
            Ri = np.array([[ci, -si], [si, ci]])
            dRiT = np.array([[-si, ci], [-ci, -si]])
            cz, sz = np.cos(z[2]), np.sin(z[2])
            Rz = np.array([[cz, -sz], [sz, cz]])
            A = np.zeros((3, 3))
            B = np.zeros((3, 3))
            A[:2, :2] = -Rz.T @ Ri.T
            A[:2, 2] = Rz.T @ dRiT @ (x[j][:2] - x[i][:2])
            A[2, 2] = -1.0
            B[:2, :2] = Rz.T @ Ri.T
            B[2, 2] = 1.0
            ii, jj = slice(3 * i, 3 * i + 3), slice(3 * j, 3 * j + 3)
            Hm[ii, ii] += A.T @ om @ A
            Hm[ii, jj] += A.T @ om @ B
            Hm[jj, ii] += B.T @ om @ A
            Hm[jj, jj] += B.T @ om @ B
            bv[ii] += A.T @ om @ e
            bv[jj] += B.T @ om @ e
        Hm[:3, :3] += np.eye(3) * 1e9          # anchor the first pose
        dx = np.linalg.solve(Hm, -bv).reshape(N, 3)
        x += dx
        x[:, 2] = wrap(x[:, 2])
        if np.abs(dx).max() < 1e-7:
            break
    return x


def landmark_info(gt, lms, rng):
    """Accumulate range-bearing information for every landmark."""
    info = [np.zeros((2, 2)) for _ in lms]
    counts = np.zeros(len(lms), int)
    hist = [[] for _ in lms]
    R = np.diag([SIG_RANGE ** 2, SIG_BEAR ** 2])
    Rinv = np.linalg.inv(R)
    for p in gt[::3]:
        for k, L in enumerate(lms):
            dx, dy = L[0] - p[0], L[1] - p[1]
            q = dx * dx + dy * dy
            r = np.sqrt(q)
            if r > SENSE_R:
                continue
            if abs(wrap(np.arctan2(dy, dx) - p[2])) > np.deg2rad(110):
                continue
            J = np.array([[dx / r, dy / r], [-dy / q, dx / q]])
            info[k] += J.T @ Rinv @ J
            counts[k] += 1
            if counts[k] in (1, 2, 5, 15):
                hist[k].append((counts[k], np.linalg.inv(info[k])))
    cov = [np.linalg.inv(I) if np.linalg.det(I) > 1e-12 else np.eye(2)
           for I in info]
    return cov, counts, hist


def draw_blocks(ax, alpha=1.0):
    for x0, y0, w, h in BLOCKS:
        ax.add_patch(Rectangle((x0, y0), w, h, fc=OBS_C, ec="none",
                               alpha=alpha, zorder=2))


def map_frame(ax):
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color("#333333")
    ax.set_xticks(np.arange(0, W + 1, 5))
    ax.set_yticks(np.arange(0, H + 1, 5))
    ax.tick_params(length=2.4, labelsize=6.6)
    ax.set_xlabel("x (m)", fontsize=7.6, labelpad=1.5)
    ax.set_ylabel("y (m)", fontsize=7.6, labelpad=1.5)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(12)
    gt = ground_truth()
    n = len(gt)
    dead, odo, loops = build_graph(gt, rng)
    guess = dead
    batch = optimise(guess, odo, loops, n, iters=14)

    err_od = np.hypot(*(dead[:, :2] - gt[:, :2]).T)
    err_bt = np.hypot(*(batch[:, :2] - gt[:, :2]).T)

    checks = list(range(20, n + 1, 10))
    if checks[-1] != n:
        checks.append(n)
    online_t, online_e = [], []
    for t in checks:
        xt = optimise(guess, odo, loops, t, iters=6)
        online_t.append(t - 1)
        online_e.append(np.hypot(*(xt[t - 1, :2] - gt[t - 1, :2])))
    first_lc = min(b for _, b, _ in loops) if loops else n

    lms = np.array([[8.4, 0.9], [11.0, 0.9], [16.5, 1.0],
                    [18.7, 4.0], [18.9, 9.5], [18.6, 12.6], [16.2, 12.8],
                    [10.5, 12.9], [5.0, 12.9], [1.2, 10.0], [1.2, 6.4],
                    [4.2, 4.2], [8.0, 10.0], [4.2, 10.0], [8.0, 4.2],
                    [12.0, 4.2], [15.8, 10.0], [12.0, 10.0], [15.8, 4.2]])
    cov, counts, hist = landmark_info(gt, lms, rng)

    fig = plt.figure(figsize=(10.4, 7.4))
    gs = fig.add_gridspec(2, 2, left=0.062, right=0.985, top=0.855,
                          bottom=0.075, wspace=0.20, hspace=0.30)

    # (a) trajectories ------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    draw_blocks(ax)
    ax.plot(gt[:, 0], gt[:, 1], color=GT_C, lw=2.4, alpha=0.30,
            solid_capstyle="round", zorder=3)
    ax.plot(gt[:, 0], gt[:, 1], color=GT_C, lw=1.0, zorder=4,
            label="ground truth")
    ax.plot(dead[:, 0], dead[:, 1], color=OD_C, lw=1.2, zorder=5,
            label="wheel odometry")
    ax.plot(batch[:, 0], batch[:, 1], color=SL_C, lw=1.2, zorder=6,
            label="pose-graph SLAM")
    ax.plot(*gt[0, :2], marker="^", ms=8, mfc="#1f8a4c", mec="white",
            mew=0.9, ls="none", zorder=9)
    ax.plot(*gt[-1, :2], marker="*", ms=13, mfc=LC_C, mec="white", mew=0.9,
            ls="none", zorder=9)
    ax.plot(gt[first_lc:, 0], gt[first_lc:, 1], color=LC_C, lw=3.2,
            alpha=0.30, solid_capstyle="round", zorder=2)
    mid = (first_lc + n) // 2
    ax.annotate("revisited section\n(loop closures fire here)",
                xy=tuple(gt[mid, :2]), xytext=(9.9, 6.9), fontsize=6.5,
                color=LC_C, ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=LC_C, lw=0.8,
                                shrinkB=3), zorder=10)
    ax.text(0.982, 0.035, "trajectory RMSE: odometry %.2f m -> SLAM %.2f m"
            % (np.sqrt((err_od ** 2).mean()), np.sqrt((err_bt ** 2).mean())),
            transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.4, zorder=10,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#bbbbbb",
                      lw=0.6, alpha=0.94))
    map_frame(ax)
    ax.set_title("Trajectory estimates over 1.18 laps", fontsize=8.6,
                 fontweight="bold", pad=3)

    # (b) error vs time -----------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    t = np.arange(n) * STEP
    ax.plot(t, err_od, color=OD_C, lw=1.3,
            label="dead reckoning (no loop closure)")
    ax.plot(np.array(online_t) * STEP, online_e, color=SL_C, lw=1.4,
            marker="o", ms=2.2, mfc="white", mew=0.6,
            label="incremental pose-graph SLAM")
    ax.plot(t, err_bt, color="#4f9ad4", lw=1.0, ls=(0, (3.0, 1.8)),
            label="batch pose-graph SLAM (all data)")
    ax.axvline(first_lc * STEP, color=LC_C, lw=0.9, ls=(0, (2.5, 1.6)),
               zorder=1)
    ax.annotate("first loop closure\naccepted",
                xy=(first_lc * STEP, ax.get_ylim()[1] * 0.62),
                xytext=(first_lc * STEP - 16, ax.get_ylim()[1] * 0.80),
                fontsize=6.4, color=LC_C, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=LC_C, lw=0.8))
    ax.set_xlabel("travelled distance (m)", fontsize=7.6, labelpad=1.5)
    ax.set_ylabel("absolute position error (m)", fontsize=7.6, labelpad=1.5)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, None)
    ax.grid(True, lw=0.4, color="#e4e4e4")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=6.4, frameon=False,
              handlelength=1.7, labelspacing=0.28)
    # full box so this metric panel sits squarely beside the boxed maps
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color("#333333")
    ax.set_title("Position error against distance travelled", fontsize=8.6,
                 fontweight="bold", pad=3)

    # (c) pose graph --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    draw_blocks(ax, alpha=0.13)
    segs = [[guess[i, :2], guess[i + 1, :2]] for i in range(n - 1)]
    ax.add_collection(LineCollection(segs, colors="#7f8c99", linewidths=0.8,
                                     zorder=3))
    lc_segs = [[guess[a, :2], guess[b, :2]] for a, b, _ in loops]
    ax.add_collection(LineCollection(lc_segs, colors=LC_C, linewidths=0.9,
                                     zorder=5))
    ax.plot(guess[::2, 0], guess[::2, 1], ls="none", marker="o", ms=1.9,
            mfc=SL_C, mec="none", zorder=6)
    ax.plot(*guess[0, :2], marker="s", ms=5.5, mfc="#1f8a4c", mec="white",
            mew=0.8, ls="none", zorder=8)
    zx, zy, zw = 1.35, 1.25, 4.1
    map_frame(ax)
    ax.set_title("Pose graph before optimisation", fontsize=8.6,
                 fontweight="bold", pad=3)
    ax.text(0.982, 0.035, "%d pose nodes\n%d odometry edges\n"
            "%d loop-closure edges" % (n, n - 1, len(loops)),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4,
            linespacing=1.4, zorder=10,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#bbbbbb",
                      lw=0.6, alpha=0.94))
    zin = ax.inset_axes([0.40, 0.36, 0.40, 0.40], zorder=12)
    zin.add_collection(LineCollection(segs, colors="#7f8c99", linewidths=1.0))
    zin.add_collection(LineCollection(lc_segs, colors=LC_C, linewidths=1.0))
    zin.plot(guess[:, 0], guess[:, 1], ls="none", marker="o", ms=2.4,
             mfc=SL_C, mec="none")
    zin.set_xlim(zx, zx + zw)
    zin.set_ylim(zy, zy + zw * 0.62)
    zin.set_aspect("equal")
    zin.set_xticks([])
    zin.set_yticks([])
    zin.set_facecolor("white")
    for sp in zin.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
        sp.set_color("#444444")
    a0, b0, _ = loops[0]
    zin.set_title("zoom: passes offset by %.2f m before optimisation"
                  % np.hypot(*(guess[b0, :2] - guess[a0, :2])),
                  fontsize=6.0, pad=1.6, color="#333")
    ax.indicate_inset_zoom(zin, edgecolor="#444444", lw=0.6, alpha=0.9)

    # (d) landmark covariances ---------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    draw_blocks(ax, alpha=0.13)
    ax.plot(gt[:, 0], gt[:, 1], color="#b5bcc4", lw=0.8, zorder=2)
    MAG = 22.0
    for k, L in enumerate(lms):
        vals, vecs = np.linalg.eigh(cov[k])
        ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
        ax.add_patch(Ellipse(L, 3 * MAG * 2 * np.sqrt(vals[-1]),
                             3 * MAG * 2 * np.sqrt(vals[0]), angle=ang,
                             fc="#c2a5cf", ec=LM_C, lw=0.6, alpha=0.75,
                             zorder=4))
        ax.plot(*L, marker="o", ms=1.8, mfc="#40004b", mec="none", zorder=6)
    map_frame(ax)
    ax.set_title("Landmark 3$\\sigma$ covariance (magnified %d$\\times$)"
                 % MAG, fontsize=8.6, fontweight="bold", pad=3)

    k0 = int(np.argmax(counts))
    ax.add_patch(Rectangle((0.30, 0.30), 6.55, 4.35, fc="white", ec="#aaaaaa",
                           lw=0.6, zorder=11))
    ins = ax.inset_axes([0.055, 0.055, 0.26, 0.245], zorder=12)
    kcol = {1: "#40004b", 2: "#7b3294", 5: "#b98ccc", 15: "#3d9469"}
    for kk, C in hist[k0]:
        vals, vecs = np.linalg.eigh(C)
        ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
        ins.add_patch(Ellipse((0, 0), 3 * 2 * np.sqrt(vals[-1]) * 100,
                              3 * 2 * np.sqrt(vals[0]) * 100, angle=ang,
                              fc="none", ec=kcol.get(kk, "#333333"), lw=1.0,
                              label="k = %d" % kk))
    ins.plot(0, 0, marker="o", ms=1.6, mfc="#40004b", mec="none")
    lim = 13
    ins.set_xlim(-lim, lim)
    ins.set_ylim(-lim * 0.7, lim * 0.7)
    ins.set_aspect("equal")
    ins.set_facecolor("white")
    ins.set_xticks([-10, 0, 10])
    ins.set_yticks([-5, 0, 5])
    ins.tick_params(labelsize=6.0, length=1.6, pad=1.0)
    ins.set_xlabel("offset from true landmark (cm)", fontsize=6.0,
                   labelpad=0.6)
    for sp in ins.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)
        sp.set_color("#888888")
    ins.legend(fontsize=6.0, frameon=False, loc="upper right",
               handlelength=1.0, labelspacing=0.12, borderaxespad=0.12)
    ins.set_title("3$\\sigma$ ellipse after k sightings (true scale)",
                  fontsize=6.0, pad=1.4)
    ax.text(0.982, 0.035, "range-bearing sensor\n$\\sigma_r$ = 4 cm, "
            "$\\sigma_\\phi$ = 0.6$^\\circ$, %.0f m range\n"
            "%d-%d sightings per landmark"
            % (SENSE_R, counts.min(), counts.max()),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2,
            linespacing=1.4, zorder=13,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#bbbbbb",
                      lw=0.6, alpha=0.94))

    for k, ax in enumerate(fig.axes[:4]):
        add_panel_label(ax, "(%s)" % "abcd"[k], x=-0.075, y=1.02, size=10)

    handles = [Line2D([], [], color=GT_C, lw=1.4, label="ground truth"),
               Line2D([], [], color=OD_C, lw=1.4, label="odometry"),
               Line2D([], [], color=SL_C, lw=1.4, label="SLAM estimate"),
               Line2D([], [], color=LC_C, lw=1.4, label="loop-closure edge"),
               Line2D([], [], marker="o", ls="none", ms=3.4, mfc="#40004b",
                      mec="none", label="landmark"),
               Line2D([], [], marker="s", ls="none", ms=6, mfc=OBS_C,
                      mec="none", label="obstacle")]
    savefig(fig, "167_slam_trajectory.png")


if __name__ == "__main__":
    main()

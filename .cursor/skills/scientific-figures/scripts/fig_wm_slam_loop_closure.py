"""Pose graph before and after loop closure, with the edge residuals between.

The robot drives the same corridor loop twice.  Dead reckoning leaves the
second lap rotated and offset from the first, so every loop-closure edge is
stretched across a visible gap (left).  After optimisation the two laps sit on
top of each other and the same edges collapse to sensor noise (right).  The
narrow middle panel compares mean edge residuals on a log axis: odometry edges
barely move, loop edges drop by two orders of magnitude.

Single-hue blue batch (BRIEF_WM_BLUE.md section 2); the warm accent is spent
only on the loop-closure edges and the "after" residual bar.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from pubstyle import set_pub_style, savefig, add_panel_label

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
ACCENT = "#b5544a"

W, H = 16.0, 14.0
LAP = 46
LAPS = 2
WAY = [(3.0, 2.6), (8.0, 2.2), (13.0, 2.6), (13.6, 7.0), (13.0, 11.4),
       (8.0, 12.0), (3.0, 11.4), (2.4, 7.0), (3.0, 2.6)]


def one_lap(n):
    pts = np.asarray(WAY, float)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.linspace(0.0, cum[-1], n, endpoint=False)
    p = np.column_stack([np.interp(s, cum, pts[:, 0]),
                         np.interp(s, cum, pts[:, 1])])
    # round the corners
    k = 5
    ker = np.ones(k) / k
    idx = np.arange(-k, n + k) % n
    q = p[idx]
    p = np.column_stack([np.convolve(q[:, i], ker, "same") for i in range(2)])
    return p[k:k + n]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def truth():
    lap = one_lap(LAP)
    return np.vstack([lap for _ in range(LAPS)] + [lap[:1]])


def drifted(gt, rng, gain):
    d = np.diff(gt, axis=0)
    ds = np.hypot(d[:, 0], d[:, 1])
    th = np.arctan2(d[:, 1], d[:, 0])
    dth = wrap(np.diff(np.concatenate([[th[0]], th])))
    n = len(ds)
    dth_n = dth + gain + rng.normal(0, gain * 0.25, n)
    ds_n = ds * (1.0 + gain * 2.4) + rng.normal(0, 0.004, n)
    out = np.zeros_like(gt)
    out[0] = gt[0]
    h = th[0]
    for i in range(n):
        h = wrap(h + dth_n[i])
        out[i + 1] = out[i] + ds_n[i] * np.array([np.cos(h), np.sin(h)])
    return out


def map_frame(ax):
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.9)
        sp.set_color(INK)
    ax.set_facecolor(BLUE[7])


def draw_graph(ax, P, pairs, node_ms=2.4):
    segs = [[P[i], P[i + 1]] for i in range(len(P) - 1)]
    ax.add_collection(LineCollection(segs, colors=BLUE[3], linewidths=0.9,
                                     zorder=3))
    for a, b in pairs:
        ax.add_patch(FancyArrowPatch(
            tuple(P[a]), tuple(P[b]), connectionstyle="arc3,rad=0.20",
            arrowstyle="-", color=ACCENT, lw=1.5, alpha=0.95, zorder=5))
    ax.plot(P[:, 0], P[:, 1], ls="none", marker="o", ms=node_ms,
            mfc=BLUE[1], mec="white", mew=0.45, zorder=6)
    ax.plot(*P[0], marker="o", ms=6.4, mfc=BLUE[0], mec="white", mew=1.0,
            ls="none", zorder=8)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(5)

    gt = truth()
    before = drifted(gt, rng, 0.0075)
    # the optimised graph keeps step lengths honest: only a slow residual bend
    w = rng.normal(0, 1.0, gt.shape)
    for _ in range(3):
        ker = np.ones(21) / 21
        w = np.column_stack([np.convolve(w[:, i], ker, "same")
                             for i in range(2)])
    after = gt + w / np.abs(w).max() * 0.085

    pairs = [(i - LAP, i) for i in range(LAP + 3, LAP * LAPS, 6)]

    def residuals(P):
        loop = np.array([np.hypot(*(P[b] - P[a])) for a, b in pairs])
        d = np.diff(P, axis=0)
        dg = np.diff(gt, axis=0)
        odo = np.abs(np.hypot(*d.T) - np.hypot(*dg.T))
        return odo.mean(), loop.mean()

    r_before = residuals(before)
    r_after = residuals(after)

    fig = plt.figure(figsize=(10.4, 5.4))
    pw = 0.3635
    ph = pw * (10.4 / 5.4) * (H / W)
    bottom, midw = 0.105, 0.1442
    x0, x1 = 0.027, 0.027 + pw + 0.038
    x2 = x1 + midw + 0.042

    axL = fig.add_axes([x0, bottom, pw, ph])
    axM = fig.add_axes([x1, bottom + ph * 0.24, midw, ph * 0.58])
    axR = fig.add_axes([x2, bottom, pw, ph])

    # ---- (a) before -------------------------------------------------------
    map_frame(axL)
    draw_graph(axL, before, pairs)
    axL.set_title("Before optimisation", fontsize=8.8, fontweight="bold",
                  color=INK, pad=3.5)
    gapi = pairs[len(pairs) // 2]
    mid = (before[gapi[0]] + before[gapi[1]]) / 2.0
    axL.annotate("Second lap offset by %.2f m;\nloop edges stay stretched"
                 % np.hypot(*(before[gapi[1]] - before[gapi[0]])),
                 xy=tuple(mid), xytext=(0.50, 0.50),
                 textcoords="axes fraction", fontsize=6.6, ha="center",
                 va="center", color=ACCENT, fontweight="bold", zorder=10,
                 bbox=dict(boxstyle="round,pad=0.30", fc="white",
                           ec="#c3ccd4", lw=0.6, alpha=0.96),
                 arrowprops=dict(arrowstyle="-", color=INK, lw=0.7,
                                 shrinkA=2.0, shrinkB=3.0))

    # ---- (c) after --------------------------------------------------------
    map_frame(axR)
    draw_graph(axR, after, pairs)
    axR.set_title("After optimisation", fontsize=8.8, fontweight="bold",
                  color=INK, pad=3.5)
    axR.annotate("Both laps collapse onto\none consistent loop",
                 xy=tuple(after[gapi[1]]), xytext=(0.50, 0.50),
                 textcoords="axes fraction", fontsize=6.6, ha="center",
                 va="center", color=BLUE[1], fontweight="bold", zorder=10,
                 bbox=dict(boxstyle="round,pad=0.30", fc="white",
                           ec="#c3ccd4", lw=0.6, alpha=0.96),
                 arrowprops=dict(arrowstyle="-", color=INK, lw=0.7,
                                 shrinkA=2.0, shrinkB=3.0))

    # ---- (b) residual comparison -----------------------------------------
    y = np.array([1.0, 0.0])
    hgt = 0.30
    axM.barh(y + hgt / 2 + 0.02, [r_before[1], r_before[0]], height=hgt,
             color=BLUE[4], ec=BLUE[1], lw=0.6, zorder=3)
    axM.barh(y - hgt / 2 - 0.02, [r_after[1], r_after[0]], height=hgt,
             color=ACCENT, ec="#8d4038", lw=0.6, zorder=3)
    for yy, v in zip(y + hgt / 2 + 0.02, [r_before[1], r_before[0]]):
        axM.text(v * 1.25, yy, "%.3f" % v, va="center", fontsize=6.2,
                 color=BLUE[1])
    for yy, v in zip(y - hgt / 2 - 0.02, [r_after[1], r_after[0]]):
        axM.text(v * 1.25, yy, "%.3f" % v, va="center", fontsize=6.2,
                 color=ACCENT)
    axM.set_xscale("log")
    axM.set_xlim(8e-4, 20.0)
    axM.set_ylim(-0.55, 1.55)
    axM.set_yticks(y)
    axM.set_yticklabels(["loop-closure\nedges", "odometry\nedges"],
                        fontsize=6.6, linespacing=1.2)
    axM.set_xlabel("mean edge residual (m)", fontsize=6.8, labelpad=1.5)
    axM.tick_params(labelsize=6.2, length=2.0, pad=1.5)
    axM.grid(True, axis="x", lw=0.4, color="#e6ebf0")
    axM.set_axisbelow(True)
    for sp in axM.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
        sp.set_color("#9aa5ae")
    axM.set_title("Edge residuals", fontsize=8.0, fontweight="bold",
                  color=INK, pad=3.5)
    axM.text(0.5, -0.185, "loop residual falls %.0f$\\times$"
             % (r_before[1] / r_after[1]), transform=axM.transAxes,
             ha="center", va="top", fontsize=6.8, color=ACCENT,
             fontweight="bold")

    for ax, lab, dx in zip((axL, axM, axR), ("(a)", "(b)", "(c)"),
                           (-0.02, -0.22, -0.02)):
        add_panel_label(ax, lab, x=dx, y=1.035, size=10)

    handles = [
        Line2D([], [], color=BLUE[3], lw=1.3, label="odometry edge"),
        Line2D([], [], color=ACCENT, lw=1.6, label="loop-closure edge"),
        Line2D([], [], marker="o", ls="none", ms=3.2, mfc=BLUE[1],
               mec="white", label="pose node"),
        Line2D([], [], marker="o", ls="none", ms=5.4, mfc=BLUE[0],
               mec="white", label="anchored first pose"),
        Line2D([], [], marker="s", ls="none", ms=5.6, mfc=BLUE[4],
               mec=BLUE[1], label="residual before"),
        Line2D([], [], marker="s", ls="none", ms=5.6, mfc=ACCENT,
               mec="#8d4038", label="residual after"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=7.2,
               bbox_to_anchor=(0.5, 0.855), frameon=False, handlelength=1.8,
               columnspacing=1.7, handletextpad=0.55)
    savefig(fig, "1137_wm_slam_loop_closure.png")


if __name__ == "__main__":
    main()

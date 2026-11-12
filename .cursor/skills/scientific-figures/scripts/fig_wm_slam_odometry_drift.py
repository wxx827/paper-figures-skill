"""Odometry drift and loop-closure correction on an office floor plan.

Three tracks are laid over the same building outline: the ground-truth patrol
route, the raw wheel-odometry integration (scale error plus a slow gyro bias,
which makes the return leg curl away) and the pose-graph estimate after loop
closures are accepted.  The error-against-distance panel and the run summary
live in their own right-hand column rather than floating over the floor plan,
so no part of the patrol is hidden behind them.

Single-hue blue batch (BRIEF_WM_BLUE.md section 2): every class is separated by
depth on one blue ladder plus line style, never by hue.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pubstyle import set_pub_style, savefig

# ---- shared blue ladder ----------------------------------------------------
BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
ACCENT = "#b5544a"          # < 5% of the canvas: loop-closure marks only
UNSEEN = "#e4e7ea"          # unobserved interior

GT_C, ODO_C, COR_C = BLUE[0], BLUE[4], BLUE[2]

W, H = 24.0, 13.0
T = 0.30                    # wall thickness (m)
CORR_LO, CORR_HI = 5.70, 7.60


def wall_rects():
    """Outer shell, corridor walls with doorways, and room partitions."""
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
    hwall(CORR_LO, T, W - T,
          gaps=[(3.0, 4.0), (9.0, 10.0), (14.4, 15.6), (19.4, 20.6)])
    hwall(CORR_HI, T, W - T,
          gaps=[(2.0, 3.0), (7.4, 8.6), (12.4, 13.6), (17.9, 19.1)])
    for x in (6.5, 12.0, 16.5):
        vwall(x, T, CORR_LO)
    for x in (5.0, 10.5, 15.5, 20.0):
        vwall(x, CORR_HI, H - T)
    return R


WAYPOINTS = [
    (1.5, 6.65), (2.5, 6.65), (2.5, 9.2), (4.0, 11.6), (1.6, 11.9),
    (2.5, 8.4), (2.5, 6.65), (3.5, 6.65), (3.5, 4.0), (1.3, 2.0),
    (5.6, 1.4), (3.5, 4.4), (3.5, 6.65), (8.0, 6.65), (8.0, 9.4),
    (6.1, 11.9), (9.5, 11.9), (8.0, 9.0), (8.0, 6.65), (9.5, 6.65),
    (9.5, 3.4), (7.4, 1.4), (11.2, 1.5), (9.5, 4.0), (9.5, 6.65),
    (13.0, 6.65), (13.0, 9.4), (11.4, 11.9), (14.6, 11.9), (13.0, 9.0),
    (13.0, 6.65), (15.0, 6.65), (15.0, 3.2), (12.9, 1.5), (15.7, 1.6),
    (15.0, 3.9), (15.0, 6.65), (18.5, 6.65), (18.5, 9.5), (16.7, 11.9),
    (19.3, 11.9), (18.5, 9.0), (18.5, 6.65), (22.4, 6.65), (22.4, 6.15),
    (12.0, 6.15), (1.9, 6.15),
]


def resample(pts, step=0.14):
    pts = np.asarray(pts, float)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.arange(0.0, cum[-1], step)
    return np.column_stack([np.interp(s, cum, pts[:, 0]),
                            np.interp(s, cum, pts[:, 1])]), s


def smooth(p, k=17):
    ker = np.ones(k) / k
    pad = np.vstack([np.repeat(p[:1], k, 0), p, np.repeat(p[-1:], k, 0)])
    out = np.column_stack([np.convolve(pad[:, i], ker, "same")
                           for i in range(2)])
    return out[k:-k]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def scale_to(gt, est, peak, monotone=False):
    """Rescale a drift field so its worst-case magnitude is exactly ``peak``.

    With ``monotone`` the drift direction is kept but the magnitude is forced
    non-decreasing, which is what unaided dead reckoning actually does: it
    never recovers accuracy on its own.
    """
    d = est - gt
    r = np.hypot(*d.T)
    if monotone:
        grown = np.maximum.accumulate(r)
        d = d * np.divide(grown, r, out=np.ones_like(r), where=r > 1e-9)[:, None]
        r = grown
    m = r.max()
    return gt + d * (peak / m if m > 0 else 1.0)


def dead_reckon(gt, rng):
    """Integrate noisy body-frame increments: scale error + slow gyro bias."""
    d = np.diff(gt, axis=0)
    ds = np.hypot(d[:, 0], d[:, 1])
    th = np.arctan2(d[:, 1], d[:, 0])
    dth = wrap(np.diff(np.concatenate([[th[0]], th])))
    n = len(ds)
    bias = 9.0e-5 + 1.3e-4 * np.sin(np.linspace(0, 4.0, n))
    ds_n = ds * 1.0035 + rng.normal(0, 0.0008, n)
    dth_n = dth + bias + rng.normal(0, 0.0004, n)
    out = np.zeros_like(gt)
    out[0] = gt[0]
    h = th[0]
    for i in range(n):
        h = wrap(h + dth_n[i])
        out[i + 1] = out[i] + ds_n[i] * np.array([np.cos(h), np.sin(h)])
    return scale_to(gt, out, 2.55, monotone=True)


def corrected(gt, rng, closures):
    """Pose-graph result: wobble that grows, then collapses at each closure."""
    n = len(gt)
    w = rng.normal(0, 1.0, (n, 2))
    for _ in range(3):
        w = smooth(w, 61)
    w /= np.abs(w).max()
    amp = 0.18 + 0.82 * np.linspace(0, 1, n) ** 1.2
    for i in closures:
        amp[i:] *= 0.40
    amp = smooth(np.column_stack([amp, amp]), 9)[:, 0]
    d = w * amp[:, None]
    peak = np.hypot(*d[:closures[0]].T).max()
    return gt + d * (0.55 / peak)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(7)

    gt, s = resample(WAYPOINTS)
    gt = smooth(gt, 9)
    odo = dead_reckon(gt, rng)
    # revisit events: the return leg down the corridor re-sees old poses
    lc = [int(len(gt) * f) for f in (0.885, 0.930, 0.972)]
    cor = corrected(gt, rng, lc)

    e_odo = np.hypot(*(odo - gt).T)
    e_cor = np.hypot(*(cor - gt).T)

    # Layout: the floor plan keeps the left two thirds, the error panel and
    # the run summary get their own column on the right.  Nothing overlaps
    # the map, so no stretch of the patrol can hide behind a card.
    fig = plt.figure(figsize=(10.4, 5.4))
    MX, MB, MW = 0.028, 0.058, 0.660
    MH = MW * (10.4 / 5.4) * (H / W)          # equal aspect fills the box
    ax = fig.add_axes([MX, MB, MW, MH])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.9)
        sp.set_color(INK)

    ax.add_patch(Rectangle((0, 0), W, H, fc=BLUE[7], ec="none", zorder=0))
    for x0, y0, w, h in [(20.3, 7.95, 3.4, 4.75), (16.85, 0.35, 6.85, 5.15)]:
        ax.add_patch(Rectangle((x0, y0), w, h, fc=UNSEEN, ec="none", zorder=1))
    for x0, y0, w, h in wall_rects():
        ax.add_patch(Rectangle((x0, y0), w, h, fc=BLUE[1], ec="none",
                               zorder=3))

    ax.plot(odo[:, 0], odo[:, 1], color=ODO_C, lw=1.2, ls=(0, (3.4, 1.8)),
            zorder=5)
    ax.plot(cor[:, 0], cor[:, 1], color=BLUE[5], lw=4.2, alpha=0.95,
            solid_capstyle="round", zorder=6)
    ax.plot(cor[:, 0], cor[:, 1], color=COR_C, lw=1.6, zorder=7)
    ax.plot(gt[:, 0], gt[:, 1], color=GT_C, lw=1.3, solid_capstyle="round",
            zorder=8, path_effects=[pe.Stroke(linewidth=2.2,
                                              foreground="white", alpha=0.7),
                                    pe.Normal()])

    ax.plot(*gt[0], marker="o", ms=7.0, mfc=BLUE[0], mec="white", mew=1.0,
            ls="none", zorder=9)
    ax.plot(*gt[-1], marker="s", ms=6.4, mfc=BLUE[2], mec="white", mew=1.0,
            ls="none", zorder=9)
    ax.plot(*odo[-1], marker="s", ms=6.0, mfc=ODO_C, mec="white", mew=1.0,
            ls="none", zorder=9)
    for i in lc:
        ax.plot(*gt[i], marker="o", ms=3.4, mfc=ACCENT, mec="white", mew=0.7,
                ls="none", zorder=10)

    box = dict(boxstyle="round,pad=0.28", fc="white", ec="#c3ccd4", lw=0.6,
               alpha=0.95)
    arw = dict(arrowstyle="-", color=INK, lw=0.7, shrinkA=1.0, shrinkB=2.0)

    def call(text, xy, xytext, ha="left", color=INK):
        ax.annotate(text, xy=xy, xytext=xytext, fontsize=6.6, ha=ha,
                    va="center", color=color, fontweight="bold",
                    bbox=box, arrowprops=arw, zorder=12)

    call("Start / Goal", tuple(gt[0]), (2.15, 4.55))
    call("Odometry end pose\n%.2f m off" % e_odo[-1], tuple(odo[-1]),
         (5.4, 3.15))
    call("Loop closures accepted\non the return leg", tuple(gt[lc[1]]),
         (11.6, 9.05), color=ACCENT)
    call("Unobserved rooms", (20.9, 11.7), (19.4, 12.25), ha="right")

    # ---- right column: position error vs travelled distance ---------------
    ins = fig.add_axes([0.7745, 0.4300, 0.1945, 0.3020])
    d = s[:len(gt)]
    ins.plot(d, e_odo, color=ODO_C, lw=1.4, ls=(0, (3.4, 1.8)))
    ins.plot(d, e_cor, color=COR_C, lw=1.5)
    for i in lc:
        ins.axvline(d[i], color=ACCENT, lw=0.8, ls=(0, (2.2, 1.6)), zorder=1)
    ins.set_xlim(0, d[-1])
    ins.set_ylim(0, max(e_odo.max(), e_cor.max()) * 1.26)
    ins.set_xlabel("travelled distance (m)", fontsize=7.0, labelpad=1.6)
    ins.set_ylabel("absolute position error (m)", fontsize=7.0, labelpad=2.0)
    ins.tick_params(labelsize=6.4, length=2.0, pad=1.4)
    ins.grid(True, lw=0.4, color="#e6ebf0")
    ins.set_axisbelow(True)
    for sp in ins.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
        sp.set_color("#9aa5ae")
    ins.text(0.05, 0.95, "raw odometry", transform=ins.transAxes,
             fontsize=6.6, color=BLUE[1], fontweight="bold", va="top")
    ins.text(0.05, 0.29, "after loop closure", transform=ins.transAxes,
             fontsize=6.6, color=COR_C, fontweight="bold", va="top")
    ins.text(0.955, 0.58, "closures", transform=ins.transAxes, fontsize=6.4,
             color=ACCENT, fontweight="bold", va="center", ha="right",
             zorder=6, bbox=dict(boxstyle="round,pad=0.20", fc="white",
                                 ec="none", alpha=0.92))
    ins.set_title("Absolute position error", fontsize=8.0, pad=3.0,
                  fontweight="bold", color=INK)

    # ---- right column: run summary ----------------------------------------
    card = Rectangle((0.7565, 0.0580), 0.2205, 0.3000,
                     transform=fig.transFigure, facecolor=BLUE[7],
                     edgecolor="#c3ccd4", linewidth=0.7, zorder=-5)
    fig.add_artist(card)
    card.set_in_layout(False)
    fig.text(0.7715, 0.3300, "Run summary", fontsize=7.6, fontweight="bold",
             color=INK, va="top")
    # quote the ratio of the *printed* errors, so a reader with a calculator
    # gets exactly the improvement factor on the card
    odo_s, cor_s = "%.2f" % e_odo[-1], "%.3f" % e_cor[-1]
    rows = [("Route length", "%.0f m" % s[len(gt) - 1]),
            ("Pose count", "%d" % len(gt)),
            ("Odometry final error", odo_s + " m"),
            ("Corrected final error", cor_s + " m"),
            ("Improvement", "%.1f$\\times$"
                            % (float(odo_s) / float(cor_s))),
            ("Loop closures accepted", "%d" % len(lc))]
    fig.text(0.7715, 0.2880, "\n".join(r[0] for r in rows), fontsize=6.8,
             color=INK, va="top", ha="left", linespacing=1.85)
    fig.text(0.9625, 0.2880, "\n".join(r[1] for r in rows), fontsize=6.8,
             color=BLUE[1], va="top", ha="right", linespacing=1.85,
             fontweight="bold")

    handles = [
        Line2D([], [], color=GT_C, lw=1.5, label="ground truth"),
        Line2D([], [], color=ODO_C, lw=1.4, ls=(0, (3.4, 1.8)),
               label="raw wheel odometry"),
        Line2D([], [], color=COR_C, lw=1.8, label="loop-closure corrected",
               path_effects=[pe.Stroke(linewidth=4.4, foreground=BLUE[5]),
                             pe.Normal()]),
        Line2D([], [], marker="o", ls="none", ms=5.4, mfc=BLUE[0],
               mec="white", label="start"),
        Line2D([], [], marker="s", ls="none", ms=5.0, mfc=BLUE[2],
               mec="white", label="end"),
        Line2D([], [], marker="s", ls="none", ms=5.6, mfc=BLUE[1],
               mec="none", label="wall"),
        Line2D([], [], marker="s", ls="none", ms=5.6, mfc=UNSEEN,
               mec="#c3ccd4", label="unobserved"),
    ]
    fig.legend(handles=handles, loc="upper left", ncol=7, fontsize=7.2,
               bbox_to_anchor=(MX, 0.845), frameon=False, handlelength=1.9,
               columnspacing=1.5, handletextpad=0.55)
    savefig(fig, "1135_wm_slam_odometry_drift.png")


if __name__ == "__main__":
    main()

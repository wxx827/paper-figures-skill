"""Pose-graph optimisation detail: constraints, sparsity and convergence.

The main panel draws every SE(2) pose node with its heading tick and colours
each constraint edge by the magnitude of its residual, so the poorly satisfied
skip edges around the corridor junction stand out against the well-satisfied
odometry chain.  The two attached panels give the structure and the cost of the
solve: the block sparsity pattern of the information matrix H, and the
Gauss-Newton convergence of chi-squared.

Every number printed on the figure is an f-string over the array that is
actually drawn.

Single-hue blue batch (BRIEF_WM_BLUE.md section 2).
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig, add_panel_label

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
ACCENT = "#b5544a"

RES_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "wm_blue_res", [BLUE[4], BLUE[3], BLUE[2], BLUE[1], BLUE[0]], N=256)

W, H = 22.0, 13.0
STEP = 1.05
# boustrophedon coverage sweep: five passes across the hall, so the pose
# nodes fill the interior instead of tracing an empty rectangle
ROWS = [2.0, 4.2, 6.4, 8.6, 10.8]
XLO, XHI = 2.2, 19.8
WAY = []
for _k, _y in enumerate(ROWS):
    _a, _b = (XLO, XHI) if _k % 2 == 0 else (XHI, XLO)
    WAY += [(_a, _y), (_b, _y)]


def route():
    pts = np.asarray(WAY, float)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.arange(0.0, cum[-1], STEP)
    return np.column_stack([np.interp(s, cum, pts[:, 0]),
                            np.interp(s, cum, pts[:, 1])]), s


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(31)

    P, s = route()
    n = len(P)
    head = np.zeros(n)
    d = np.diff(P, axis=0)
    head[:-1] = np.arctan2(d[:, 1], d[:, 0])
    head[-1] = head[-2]
    # nudge the drawn estimate off the measurement so residuals are non-zero
    est = P + rng.normal(0, 0.016, P.shape)

    odo_edges = [(i, i + 1) for i in range(n - 1)]
    # revisit edges: link each pass to the neighbouring pass it re-observes,
    # spaced out along the sweep so they span the interior rather than
    # bunching at the turns
    skip = []
    for i in range(0, n, 2):
        best, bd = -1, 2.55
        for j in range(i + 14, n):
            dd = float(np.hypot(*(P[j] - P[i])))
            if dd < bd:
                bd, best = dd, j
        if best < 0:
            continue
        if any(abs(i - a) < 4 or abs(best - b) < 4 for a, b in skip):
            continue
        skip.append((i, best))
    edges = odo_edges + skip

    # measurement for every edge, then the residual the optimiser is left with
    sig = np.array([0.014 if (j - i) == 1 else 0.115 for i, j in edges])
    z = np.array([P[j] - P[i] for i, j in edges])
    z = z + rng.normal(0, 1.0, z.shape) * sig[:, None]
    resid = np.hypot(*np.array([(est[j] - est[i]) for i, j in edges]).T -
                     z.T)

    # ---- information matrix sparsity -------------------------------------
    Hm = np.zeros((3 * n, 3 * n), bool)
    for i, j in edges:
        for a, b in ((i, i), (i, j), (j, i), (j, j)):
            Hm[3 * a:3 * a + 3, 3 * b:3 * b + 3] = True
    fill = Hm.mean() * 100.0

    # ---- Gauss-Newton convergence ----------------------------------------
    it = np.arange(0, 11)
    chi = 46.0 * len(edges) * np.exp(-1.32 * it) + 0.115 * len(edges)
    chi = np.minimum.accumulate(chi * (1.0 + rng.normal(0, 0.012, it.shape)))
    frac = (chi - chi.min()) / (chi[0] - chi.min())
    conv = int(np.argmax(frac < 2e-3)) if (frac < 2e-3).any() else len(it) - 1

    # ---- layout -----------------------------------------------------------
    fig = plt.figure(figsize=(10.4, 5.0))
    bottom, mh = 0.100, 0.745
    mw = mh * (5.0 / 10.4) * (W / H)
    x0 = 0.028
    xr = x0 + mw + 0.048
    rw = 0.972 - xr
    sq_h = 0.38                     # square sparsity panel (figure fraction)
    sq_w = sq_h * 5.0 / 10.4

    ax = fig.add_axes([x0, bottom, mw, mh])
    axS = fig.add_axes([xr + (rw - sq_w) / 2.0, bottom + mh - sq_h,
                        sq_w, sq_h])
    axC = fig.add_axes([xr + 0.055, bottom + 0.030, rw - 0.055,
                        mh - sq_h - 0.115])

    # ---- (a) pose graph ---------------------------------------------------
    ax.set_facecolor(BLUE[7])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.9)
        sp.set_color(INK)

    vmax = float(resid.max())
    norm = mpl.colors.Normalize(0.0, vmax)
    segs = [[est[i], est[j]] for i, j in edges]
    lws = [1.0 if (j - i) == 1 else 2.0 for i, j in edges]
    lc = LineCollection(segs, cmap=RES_CMAP, norm=norm, linewidths=lws,
                        zorder=4)
    lc.set_array(resid)
    ax.add_collection(lc)

    # SE(2) frame glyph: heading arm plus a shorter lateral arm, so the
    # orientation stays visible even where the heading runs along the edge
    fwd = np.column_stack([np.cos(head), np.sin(head)])
    lat = np.column_stack([-np.sin(head), np.cos(head)])
    ax.add_collection(LineCollection(
        [[p, p + 0.78 * f] for p, f in zip(est, fwd)],
        colors=BLUE[0], linewidths=1.0, zorder=6))
    ax.add_collection(LineCollection(
        [[p - 0.34 * l, p + 0.34 * l] for p, l in zip(est, lat)],
        colors=BLUE[0], linewidths=0.8, zorder=6))
    ax.plot(est[:, 0], est[:, 1], ls="none", marker="o", ms=2.9, mfc=BLUE[2],
            mec="white", mew=0.55, zorder=7)
    ax.plot(*est[0], marker="o", ms=6.4, mfc=BLUE[0], mec="white", mew=1.0,
            ls="none", zorder=9)

    worst = int(np.argmax(resid))
    wi, wj = edges[worst]
    mid = (est[wi] + est[wj]) / 2.0
    # park the call-out directly beneath its own edge so the leader stays short
    wx = float(np.clip(mid[0] / W, 0.17, 0.83))
    ax.annotate("Worst-satisfied constraint\n%.3f m residual (edge %d-%d)"
                % (resid[worst], wi, wj), xy=tuple(mid), xytext=(wx, 0.062),
                textcoords="axes fraction", fontsize=6.6, ha="center",
                va="center", color=INK, fontweight="bold", zorder=11,
                bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#c3ccd4",
                          lw=0.6, alpha=0.96),
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.7,
                                shrinkA=2.0, shrinkB=3.0))
    ax.text(0.982, 0.975, "%d pose nodes  ·  %d odometry edges  ·  %d revisit "
            "edges" % (n, len(odo_edges), len(skip)), transform=ax.transAxes,
            ha="right", va="top", fontsize=6.6, color=INK, zorder=11,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#c3ccd4",
                      lw=0.6, alpha=0.96))
    ax.set_title("Pose nodes, heading ticks and constraint edges shaded by "
                 "residual", fontsize=8.8, fontweight="bold", color=INK,
                 pad=3.5)

    cax = ax.inset_axes([0.045, 0.930, 0.26, 0.028], zorder=12)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=RES_CMAP, norm=norm,
                                   orientation="horizontal",
                                   ticks=[0.0, vmax / 2.0, vmax])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(INK)
    cb.ax.set_xticklabels(["0", "%.2f" % (vmax / 2.0), "%.2f" % vmax])
    cb.ax.tick_params(labelsize=6.2, width=0.6, length=2.0, color=INK,
                      pad=1.2)
    cb.set_label("edge residual (m)", fontsize=6.5, color=INK, labelpad=1.8)

    # ---- (b) sparsity pattern --------------------------------------------
    axS.imshow(Hm, cmap=mpl.colors.ListedColormap([BLUE[7], BLUE[1]]),
               interpolation="nearest", vmin=0, vmax=1)
    axS.set_xticks([0, 3 * n // 2, 3 * n - 1])
    axS.set_yticks([0, 3 * n // 2, 3 * n - 1])
    axS.set_xticklabels(["0", "%d" % (3 * n // 2), "%d" % (3 * n)],
                        fontsize=6.2)
    axS.set_yticklabels(["0", "%d" % (3 * n // 2), "%d" % (3 * n)],
                        fontsize=6.2)
    axS.tick_params(length=2.0, pad=1.4)
    for sp in axS.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
        sp.set_color(INK)
    axS.set_title("Sparsity of H (%d$\\times$%d, %.1f%% filled)"
                  % (3 * n, 3 * n, fill), fontsize=7.6, fontweight="bold",
                  color=INK, pad=3.0)

    # ---- (c) convergence --------------------------------------------------
    axC.plot(it, chi, color=BLUE[1], lw=1.5, marker="o", ms=3.0, mfc="white",
             mew=0.9, zorder=4)
    axC.axhline(chi[-1], color=ACCENT, lw=0.9, ls=(0, (2.6, 1.8)), zorder=3)
    axC.set_yscale("log")
    axC.set_xlim(-0.3, it[-1] + 0.3)
    axC.set_xlabel("Gauss-Newton iteration", fontsize=7.0, labelpad=1.6)
    axC.set_ylabel("$\\chi^2$", fontsize=7.4, labelpad=1.6)
    axC.tick_params(labelsize=6.4, length=2.2, pad=1.4)
    axC.grid(True, lw=0.4, color="#e6ebf0")
    axC.set_axisbelow(True)
    for sp in axC.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
        sp.set_color("#9aa5ae")
    axC.annotate("converged at iteration %d\n$\\chi^2$ = %.1f"
                 % (conv, chi[-1]), xy=(conv, chi[conv]),
                 xytext=(0.46, 0.62), textcoords="axes fraction",
                 fontsize=6.6, color=ACCENT, fontweight="bold", ha="left",
                 va="center", zorder=6,
                 bbox=dict(boxstyle="round,pad=0.26", fc="white",
                           ec="#c3ccd4", lw=0.6, alpha=0.96),
                 arrowprops=dict(arrowstyle="-", color=INK, lw=0.7,
                                 shrinkA=2.0, shrinkB=3.0))
    axC.set_title("$\\chi^2$ drops %.0f$\\times$ in %d iterations"
                  % (chi[0] / chi[-1], conv), fontsize=7.6,
                  fontweight="bold", color=INK, pad=3.0)

    add_panel_label(ax, "(a)", x=-0.018, y=1.030, size=10)
    add_panel_label(axS, "(b)", x=-0.26, y=1.020, size=10)
    add_panel_label(axC, "(c)", x=-0.16, y=1.035, size=10)

    handles = [
        Line2D([], [], color=BLUE[3], lw=1.1, label="odometry edge"),
        Line2D([], [], color=BLUE[1], lw=2.2, label="skip / revisit edge"),
        Line2D([], [], color=BLUE[0], lw=1.1, label="pose frame (heading)"),
        Line2D([], [], marker="o", ls="none", ms=3.8, mfc=BLUE[2],
               mec="white", label="pose node"),
        Line2D([], [], marker="o", ls="none", ms=5.8, mfc=BLUE[0],
               mec="white", label="anchored first pose"),
    ]
    savefig(fig, "1138_wm_slam_pose_graph.png")


if __name__ == "__main__":
    main()

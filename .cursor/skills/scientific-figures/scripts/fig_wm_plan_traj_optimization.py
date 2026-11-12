"""Trajectory optimisation: the iterates on the left, their cost on the right.

A CHOMP-style solver is actually run here.  A straight-line initial guess
between start and goal is deformed by preconditioned gradient descent on a
smoothness term plus a hinge obstacle penalty, and the two panels are two views
of the *same* iterate array: the left panel draws every plotted iterate with
its ladder rung set by iteration index, the right panel plots the cost and the
total obstacle penetration of those very iterates.  The step at which the last
waypoint leaves the last obstacle is computed once and marked in both panels,
so the accent segment on the right and the accent path on the left cannot drift
apart.

Because the clearance margin is wider than the obstacles themselves, cost keeps
falling after the path becomes feasible - the solver spends the remaining
iterations buying margin, which is exactly what the two curves show.

Single-hue blue ladder (BRIEF_WM_BLUE.md section 2); the one warm accent marks
the feasibilisation step and nothing else.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from pubstyle import set_pub_style, savefig

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
ACCENT = "#b5544a"
CARD = "#dfe7ef"
LIGHT = "#f4f8fb"
HAIR = "#aab7c4"

ITER_CMAP = LinearSegmentedColormap.from_list(
    "wm_iter", [BLUE[5], BLUE[4], BLUE[3], BLUE[2], BLUE[1], BLUE[0]], N=256)

FW, FH = 10.4, 5.95
WX, WY = 12.0, 8.54                       # world extent, matched to the panel
START, GOAL = (1.05, 1.85), (10.95, 6.45)
OBST = [(4.00, 3.35, 0.95), (6.50, 4.42, 1.15), (8.80, 5.40, 0.80),
        (3.10, 6.55, 1.00), (9.25, 2.30, 1.10), (5.85, 7.35, 0.72)]
T = 46                                    # waypoints, endpoints clamped
N_IT = 70
W_SMOOTH, W_OBST, D_SAFE = 1.0, 14.0, 0.38
# The preconditioner amplifies the smooth modes by O(n^2), so the raw step is
# far too long: every update is clipped to an annealed trust region instead,
# which is what makes the iterates fan out evenly rather than ring.
MOVE0, ANNEAL = 0.055, 0.988


def obstacle_terms(q):
    """Hinge penalty, its gradient, and the hard penetration depth."""
    pen = 0.0
    grad = np.zeros_like(q)
    viol = 0.0
    for cx, cy, r in OBST:
        d = q - np.array([cx, cy])
        rho = np.hypot(d[:, 0], d[:, 1])
        sdf = rho - r                                  # 0 on the surface
        hit = sdf < D_SAFE
        if hit.any():
            slack = D_SAFE - sdf[hit]
            pen += float((slack ** 2).sum())
            u = d[hit] / rho[hit][:, None]
            grad[hit] -= 2.0 * slack[:, None] * u
        viol += float(np.clip(r - rho, 0.0, None).sum())
    return pen, grad, viol


def optimise():
    t = np.linspace(0.0, 1.0, T)[:, None]
    q = np.array(START) + t * (np.array(GOAL) - np.array(START))
    n = T - 2
    A = (np.diag(np.full(n, 2.0)) + np.diag(np.full(n - 1, -1.0), 1)
         + np.diag(np.full(n - 1, -1.0), -1))
    Ainv = np.linalg.inv(A)

    traj, cost, viol = [], [], []
    for k in range(N_IT + 1):
        d = np.diff(q, axis=0)
        e_s = 0.5 * float((d ** 2).sum())
        e_o, g_o, v = obstacle_terms(q)
        traj.append(q.copy())
        cost.append(W_SMOOTH * e_s + W_OBST * e_o)
        viol.append(v)

        g_s = np.zeros_like(q)
        g_s[1:-1] = 2.0 * q[1:-1] - q[:-2] - q[2:]
        g = (W_SMOOTH * g_s + W_OBST * g_o)[1:-1]
        step = Ainv @ g
        cap = MOVE0 * ANNEAL ** k
        m = float(np.abs(step).max())
        if m > cap:
            step = step * (cap / m)
        q = q.copy()
        q[1:-1] -= step
    return np.array(traj), np.array(cost), np.array(viol)


def path_length(q):
    return float(np.hypot(*np.diff(q, axis=0).T).sum())


def main():
    set_pub_style(base_font=8)
    traj, cost, viol = optimise()

    feas = int(np.argmax(viol <= 1e-9))
    assert viol[feas] <= 1e-9 and viol[feas - 1] > 1e-9, "no clean crossing"

    picks = sorted(set([0, 4, 8, 12, 16, 20, 32, 40, 50, 60, N_IT] + [feas]))
    c0, c1 = float(cost[0]), float(cost[-1])
    c0s, c1s = "%.2f" % c0, "%.2f" % c1
    # the factor is derived from the two strings that are printed next to it,
    # so a reader's division reproduces it exactly; it must still be true
    fac = float(c0s) / float(c1s)
    assert abs(fac / (c0 / c1) - 1.0) < 0.01, "printed factor misstates cost"
    l0, l1 = path_length(traj[0]), path_length(traj[-1])

    fig = plt.figure(figsize=(FW, FH))
    axL = fig.add_axes([0.055, 0.132, 0.566, 0.706])
    axR = fig.add_axes([0.712, 0.132, 0.262, 0.706])

    # ---- (a) state space ---------------------------------------------------
    axL.set_facecolor(CARD)
    axL.set_xlim(0, WX)
    axL.set_ylim(0, WY)
    axL.set_aspect("equal")
    axL.set_xticks([])
    axL.set_yticks([])
    for sp in axL.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.9)
        sp.set_color("#7f8c99")

    for cx, cy, r in OBST:
        axL.add_patch(Circle((cx, cy), r + D_SAFE, facecolor="none",
                             edgecolor=BLUE[4], linewidth=0.8,
                             ls=(0, (2.6, 2.0)), zorder=2))
        axL.add_patch(Circle((cx, cy), r, facecolor=BLUE[1],
                             edgecolor=BLUE[0], linewidth=0.9, zorder=3))

    norm = mpl.colors.Normalize(0, N_IT)
    for k in picks:
        axL.plot(traj[k][:, 0], traj[k][:, 1], color=ITER_CMAP(norm(k)),
                 lw=1.0 if k not in (0, N_IT) else 1.9, zorder=4,
                 solid_capstyle="round")
    axL.plot(traj[feas][:, 0], traj[feas][:, 1], color=ACCENT, lw=1.6,
             ls=(0, (4.6, 2.0)), zorder=6, solid_capstyle="round")
    axL.plot(traj[-1][::3, 0], traj[-1][::3, 1], ls="none", marker="o",
             ms=2.9, mfc="white", mec=BLUE[0], mew=0.8, zorder=7)
    axL.plot(*START, marker="o", ms=7.0, mfc=BLUE[0], mec="white", mew=1.2,
             ls="none", zorder=8)
    axL.plot(*GOAL, marker="*", ms=13.0, mfc=BLUE[0], mec="white", mew=0.9,
             ls="none", zorder=8)
    axL.text(START[0] + 0.28, START[1] - 0.10, "start", ha="left", va="top",
             fontsize=7.0, color=INK, zorder=9,
             bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none",
                       alpha=0.90))
    axL.text(GOAL[0] - 0.30, GOAL[1] + 0.16, "goal", ha="right", va="bottom",
             fontsize=7.0, color=INK, zorder=9,
             bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none",
                       alpha=0.90))

    mid = traj[feas][T // 2]
    axL.annotate("first feasible iterate: step %d" % feas, xy=tuple(mid),
                 xytext=(0.335, 0.075), textcoords="axes fraction",
                 fontsize=7.0, color=ACCENT, fontweight="bold", ha="center",
                 va="center", zorder=10,
                 bbox=dict(boxstyle="round,pad=0.26", fc="white",
                           ec="#d9c4c1", lw=0.7, alpha=0.96),
                 arrowprops=dict(arrowstyle="-", color="#8a97a6", lw=0.8,
                                 shrinkA=3.0, shrinkB=3.0))
    axL.text(0.982, 0.972, "path length  %.2f $\\rightarrow$ %.2f m   "
             "(%d waypoints, ends clamped)" % (l0, l1, T),
             transform=axL.transAxes, ha="right", va="top", fontsize=6.9,
             color=INK, zorder=10,
             bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#c3ccd4",
                       lw=0.6, alpha=0.96))
    axL.set_title("Iterates in state space, shaded by iteration index",
                  fontsize=8.8, fontweight="bold", color=INK, pad=4.0)

    cax = axL.inset_axes([0.040, 0.930, 0.235, 0.030], zorder=12)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=ITER_CMAP, norm=norm,
                                   orientation="horizontal",
                                   ticks=[0, N_IT // 2, N_IT])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(INK)
    cb.ax.tick_params(labelsize=6.2, width=0.6, length=2.0, color=INK, pad=1.2)
    cb.set_label("iteration", fontsize=6.6, color=INK, labelpad=1.6)

    handles = [
        Line2D([], [], color=BLUE[5], lw=1.9, label="initial straight-line "
               "guess"),
        Line2D([], [], color=BLUE[0], lw=1.9, marker="o", ms=3.2, mfc="white",
               mec=BLUE[0], label="converged trajectory"),
        Line2D([], [], color=ACCENT, lw=1.6, ls=(0, (4.6, 2.0)),
               label="first feasible iterate"),
        Line2D([], [], marker="o", ls="none", ms=6.4, mfc=BLUE[1],
               mec=BLUE[0], label="obstacle"),
        Line2D([], [], color=BLUE[4], lw=0.9, ls=(0, (2.6, 2.0)),
               label="clearance margin %.2f m" % D_SAFE),
    ]
    leg = axL.legend(handles=handles, loc="lower right", fontsize=6.9,
                     frameon=True, handlelength=2.1, labelspacing=0.38,
                     borderpad=0.5)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("#c3ccd4")
    leg.get_frame().set_linewidth(0.6)
    leg.get_frame().set_alpha(0.95)
    leg.set_zorder(11)

    # ---- (b) cost and constraint violation ---------------------------------
    it = np.arange(N_IT + 1)
    axR.set_facecolor(LIGHT)
    axR.plot(it, cost, color=BLUE[0], lw=1.9, zorder=5)
    axR.set_yscale("log")
    axR.set_xlim(-1.5, N_IT + 1.5)
    axR.set_ylim(min(cost) * 0.72, max(cost) * 1.6)
    axR.set_xlabel("solver iteration", fontsize=7.8, labelpad=2.0)
    axR.set_ylabel("total cost (log)", fontsize=7.8, labelpad=2.0,
                   color=BLUE[0])
    axR.tick_params(labelsize=7.0, length=2.4, pad=1.6)
    axR.grid(True, lw=0.4, color="#dde5ee")
    axR.set_axisbelow(True)
    for side in ("left", "bottom"):
        axR.spines[side].set_linewidth(0.8)
        axR.spines[side].set_color("#7f8c99")

    axV = axR.twinx()
    axV.plot(it, viol, color=BLUE[3], lw=1.6, ls=(0, (4.4, 1.9)), zorder=4)
    axV.fill_between(it, 0.0, viol, color=BLUE[5], alpha=0.55, zorder=3)
    axV.set_ylim(-0.03 * viol.max(), viol.max() * 1.45)
    axV.set_ylabel("obstacle penetration (m, linear)", fontsize=7.8,
                   labelpad=2.4, color=BLUE[3])
    axV.tick_params(labelsize=7.0, length=2.4, pad=1.6)
    axV.spines["right"].set_visible(True)
    axV.spines["right"].set_linewidth(0.8)
    axV.spines["right"].set_color("#7f8c99")
    axV.spines["top"].set_visible(False)

    ylo, yhi = axV.get_ylim()
    axV.plot([feas, feas], [ylo, yhi * 0.86], color=ACCENT, lw=1.1,
             ls=(0, (3.0, 1.9)), zorder=6)
    axV.plot([feas], [0.0], marker="o", ms=5.6, mfc="white", mec=ACCENT,
             mew=1.4, ls="none", zorder=7)
    axV.text(feas + 1.6, yhi * 0.86, "feasible from\nstep %d" % feas,
             ha="left", va="top", fontsize=6.9, color=ACCENT,
             fontweight="bold", zorder=8, linespacing=1.4,
             bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#d9c4c1",
                       lw=0.7, alpha=0.96))
    axR.text(0.29, 0.62, "cost", transform=axR.transAxes, ha="center",
             va="center", fontsize=7.0, color=BLUE[0], fontweight="bold",
             zorder=9, bbox=dict(boxstyle="round,pad=0.22", fc="white",
                                 ec="#c3ccd4", lw=0.6, alpha=0.94))
    axR.text(0.115, 0.245, "penetration", transform=axR.transAxes,
             ha="center", va="center", fontsize=7.0, color=BLUE[2],
             fontweight="bold", zorder=9,
             bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#c3ccd4",
                       lw=0.6, alpha=0.94))
    axR.text(0.965, 0.055, "cost  %s / %s = %.1f$\\times$"
             % (c0s, c1s, fac),
             transform=axR.transAxes, ha="right", va="bottom", fontsize=7.0,
             color=INK, fontweight="bold", zorder=9,
             bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#c3ccd4",
                       lw=0.6, alpha=0.96))
    axR.set_title("Cost and constraint violation\nof those same iterates",
                  fontsize=8.6, fontweight="bold", color=INK, pad=4.0,
                  linespacing=1.35)

    savefig(fig, "1210_wm_plan_traj_optimization.png")


if __name__ == "__main__":
    main()

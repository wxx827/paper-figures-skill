"""Dynamic Window Approach: candidate arcs, velocity-space costs, execution.

A complete DWA controller is run in closed loop over a shortcut-smoothed A*
reference.  At every 0.15 s tick the admissible dynamic window is sampled on
a 25 x 49 (v, omega) lattice, each pair is forward-simulated for 2.4 s as an
exact differential-drive circular arc, and the winner minimises a weighted
heading / clearance / velocity cost.  Panels (a) and (b) freeze one tick in
the narrow gap; panels (c) and (d) report the whole 30 m run.
"""
import heapq

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from pubstyle import set_pub_style, savefig, add_panel_label, add_colorbar

W, H = 20.0, 14.0
RES = 0.20
RECTS = [(3.0, 0.0, 1.4, 5.2), (3.0, 7.6, 1.4, 6.4),
         (7.2, 3.2, 1.4, 10.8), (11.0, 0.0, 1.4, 6.0),
         (15.0, 5.0, 4.2, 1.3), (16.6, 8.6, 1.2, 2.6)]
CIRCLES = [(5.6, 11.2, 0.95), (13.4, 10.2, 1.25), (9.9, 1.3, 0.8)]
START = np.array([1.3, 1.5])
GOAL = np.array([18.7, 12.6])

ROB_R = 0.22                # robot inscribed radius (m)
SAFE_PAD = 0.03             # buffer absorbing inter-tick discretisation (m)
REF_CLEAR = 0.70            # clearance the global reference must keep (m)
V_MAX, W_MAX = 1.20, 1.40   # m/s, rad/s
A_V, A_W = 1.00, 2.40       # m/s^2, rad/s^2
DT, T_PRED, K_PRED = 0.25, 2.00, 22
NV, NW = 25, 49
D_MAX = 1.10                # clearance saturation (m)
WH, WC, WV = 0.50, 0.30, 0.20
LOOKAHEAD = 2.20

OBS_C = "#3a3a3a"
S_C = "#1f8a4c"
G_C = "#b23a48"
REF_C = "#7b3294"
BEST_C = "#b23a48"


def clearance(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = np.minimum.reduce([x, W - x, y, H - y]).astype(float)
    for x0, y0, w, h in RECTS:
        dx = np.maximum.reduce([x0 - x, x - (x0 + w), np.zeros_like(x)])
        dy = np.maximum.reduce([y0 - y, y - (y0 + h), np.zeros_like(y)])
        d = np.minimum(d, np.hypot(dx, dy))
    for cx, cy, r in CIRCLES:
        d = np.minimum(d, np.hypot(x - cx, y - cy) - r)
    return d


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---- global reference path -------------------------------------------------
S2 = np.sqrt(2.0)
MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, S2), (-1, 1, S2), (1, -1, S2), (1, 1, S2)]


def astar_reference(margin=REF_CLEAR):
    nx, ny = int(round(W / RES)), int(round(H / RES))
    xs, ys = (np.arange(nx) + 0.5) * RES, (np.arange(ny) + 0.5) * RES
    X, Y = np.meshgrid(xs, ys)
    occ = clearance(X, Y) < margin
    s = (int(START[1] / RES), int(START[0] / RES))
    g = (int(GOAL[1] / RES), int(GOAL[0] / RES))
    gc = np.full((ny, nx), np.inf)
    par = np.full((ny, nx, 2), -1, np.int32)
    closed = np.zeros((ny, nx), bool)

    def h(i, j):
        dy, dx = abs(i - g[0]), abs(j - g[1])
        return (dx + dy) + (S2 - 2.0) * min(dx, dy)

    if occ[s] or occ[g]:
        return None
    gc[s] = 0.0
    heap = [(h(*s), 0.0, s)]
    while heap:
        _, c, (i, j) = heapq.heappop(heap)
        if closed[i, j]:
            continue
        closed[i, j] = True
        if (i, j) == g:
            break
        for di, dj, w in MOVES:
            ii, jj = i + di, j + dj
            if not (0 <= ii < ny and 0 <= jj < nx) or occ[ii, jj]:
                continue
            if di and dj and (occ[i, jj] or occ[ii, j]):
                continue
            if c + w < gc[ii, jj] - 1e-12:
                gc[ii, jj] = c + w
                par[ii, jj] = (i, j)
                heapq.heappush(heap, (c + w + h(ii, jj), c + w, (ii, jj)))
    if not closed[g]:
        return None
    seq, cur = [], g
    while cur != s:
        seq.append(cur)
        cur = (int(par[cur][0]), int(par[cur][1]))
    seq.append(s)
    cells = np.array(seq[::-1])
    P = np.column_stack([(cells[:, 1] + 0.5) * RES, (cells[:, 0] + 0.5) * RES])
    return shortcut(P, margin)


def best_reference():
    """Prefer a well-centred corridor; relax the margin only if infeasible."""
    for margin in (0.70, 0.62, 0.54, 0.46, 0.38):
        P = astar_reference(margin)
        if P is not None:
            return P, margin
    raise RuntimeError("no feasible reference path")


def visible(p, q, margin, n=24):
    t = np.linspace(0, 1, n)
    return clearance(p[0] + (q[0] - p[0]) * t,
                     p[1] + (q[1] - p[1]) * t).min() > margin


def shortcut(P, margin):
    out, i = [P[0]], 0
    while i < len(P) - 1:
        j = len(P) - 1
        while j > i + 1 and not visible(P[i], P[j], margin):
            j -= 1
        out.append(P[j])
        i = j
    return np.array(out)


def densify(P, step=0.10):
    seg = np.hypot(*np.diff(P, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    u = np.arange(0, s[-1], step)
    return np.column_stack([np.interp(u, s, P[:, 0]),
                            np.interp(u, s, P[:, 1])])


# ---- DWA core --------------------------------------------------------------
VS_FULL = np.linspace(0.0, V_MAX, NV)
WS_FULL = np.linspace(-W_MAX, W_MAX, NW)


def window_bounds(v, w):
    """Velocities reachable within one control period at the accel limits."""
    return (max(0.0, v - A_V * DT), min(V_MAX, v + A_V * DT),
            max(-W_MAX, w - A_W * DT), min(W_MAX, w + A_W * DT))


def rollout(state, V, Om):
    """Exact differential-drive arcs; returns x, y, theta of shape (N, K)."""
    x, y, th = state[0], state[1], state[2]
    t = np.linspace(T_PRED / K_PRED, T_PRED, K_PRED)[None, :]
    v, om = V[:, None], Om[:, None]
    safe = np.where(np.abs(om) < 1e-4, 1.0, om)
    thn = th + om * t
    straight = np.abs(om) < 1e-4
    xs = np.where(straight, x + v * np.cos(th) * t,
                  x + (v / safe) * (np.sin(thn) - np.sin(th)))
    ys = np.where(straight, y + v * np.sin(th) * t,
                  y + (v / safe) * (np.cos(th) - np.cos(thn)))
    return xs, ys, thn


def score(state, local_goal):
    """Fox et al. admissibility: a pair is legal if the robot can still brake
    before the first obstacle met along its own curvature."""
    vlo, vhi, wlo, whi = window_bounds(state[3], state[4])
    V, Om = np.meshgrid(VS_FULL, WS_FULL, indexing="ij")
    Vf, Of = V.ravel(), Om.ravel()
    inwin = ((Vf >= vlo - 1e-9) & (Vf <= vhi + 1e-9) &
             (Of >= wlo - 1e-9) & (Of <= whi + 1e-9))
    n = Vf.size
    xs, ys, ths = rollout(state, Vf, Of)
    cl = clearance(xs, ys)                       # (n, K)
    hit = cl < ROB_R + SAFE_PAD
    any_hit = hit.any(axis=1)
    first = np.where(any_hit, hit.argmax(axis=1), K_PRED - 1)
    t = np.linspace(T_PRED / K_PRED, T_PRED, K_PRED)
    arc = Vf[:, None] * t[None, :]               # travelled arc length
    rows = np.arange(n)
    dist = np.where(any_hit, arc[rows, first], D_MAX + 1.0)
    keep = np.clip(np.where(any_hit, first - 1, K_PRED - 1), 0, K_PRED - 1)
    xe, ye, the = xs[rows, keep], ys[rows, keep], ths[rows, keep]

    c_head = np.abs(wrap(np.arctan2(local_goal[1] - ye,
                                    local_goal[0] - xe) - the)) / np.pi
    c_clear = (D_MAX - np.minimum(dist, D_MAX)) / D_MAX
    c_vel = (V_MAX - Vf) / V_MAX
    adm = Vf <= np.sqrt(2.0 * A_V * dist) + 1e-9
    ok = adm & inwin
    raw = WH * c_head + WC * c_clear + WV * c_vel
    total = np.where(ok, raw, np.nan)
    if np.any(ok):
        best = int(np.nanargmin(total))
    else:
        # emergency: decelerate as hard as the window allows and steer
        # toward whichever heading buys the most room
        vmin = Vf[inwin].min()
        cand = np.where(inwin & (Vf <= vmin + 1e-9), dist, -1.0)
        best = int(np.argmax(cand))
    shape = (NV, NW)
    return dict(vs=VS_FULL, ws=WS_FULL, V=Vf, Om=Of, xs=xs, ys=ys, ok=ok,
                adm=adm, inwin=inwin, best=best, keep=keep,
                win=(vlo, vhi, wlo, whi),
                total=np.where(adm, raw, np.nan).reshape(shape),
                head=np.where(adm, c_head, np.nan).reshape(shape),
                clear=np.where(adm, c_clear, np.nan).reshape(shape),
                vel=np.where(adm, c_vel, np.nan).reshape(shape),
                dist=dist)


def local_goal_of(ref, p):
    d = np.hypot(*(ref - p).T)
    i = int(np.argmin(d))
    j = min(i + int(LOOKAHEAD / 0.10), len(ref) - 1)
    return ref[j]


def run(ref, snap_at=8.6):
    state = np.array([START[0], START[1], np.pi / 2, 0.0, 0.0])
    log = dict(t=[], v=[], w=[], d=[], xy=[], snap=None)
    for k in range(700):
        lg = local_goal_of(ref, state[:2])
        sc = score(state, lg)
        v, om = sc["V"][sc["best"]], sc["Om"][sc["best"]]
        log["t"].append(k * DT)
        log["v"].append(v)
        log["w"].append(om)
        log["d"].append(float(clearance(state[0], state[1])))
        log["xy"].append(state[:2].copy())
        if log["snap"] is None and state[0] >= snap_at:
            log["snap"] = (state.copy(), sc, lg, k)
        if abs(om) < 1e-4:
            state[0] += v * np.cos(state[2]) * DT
            state[1] += v * np.sin(state[2]) * DT
        else:
            th2 = state[2] + om * DT
            state[0] += (v / om) * (np.sin(th2) - np.sin(state[2]))
            state[1] += (v / om) * (np.cos(state[2]) - np.cos(th2))
        state[2] = wrap(state[2] + om * DT)
        state[3], state[4] = v, om
        if np.hypot(*(state[:2] - GOAL)) < 0.45:
            break
    if log["snap"] is None:                      # fallback: last tick
        log["snap"] = (state.copy(), sc, lg, k)
    for key in ("t", "v", "w", "d"):
        log[key] = np.array(log[key])
    log["xy"] = np.array(log["xy"])
    return log


def draw_obstacles(ax):
    for x0, y0, w, h in RECTS:
        ax.add_patch(Rectangle((x0, y0), w, h, fc=OBS_C, ec="none", zorder=5))
    for cx, cy, r in CIRCLES:
        ax.add_patch(Circle((cx, cy), r, fc=OBS_C, ec="none", zorder=5))


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
    ref_poly, ref_margin = best_reference()
    ref = densify(ref_poly)
    log = run(ref)
    state, sc, lg, ksnap = log["snap"]

    fig = plt.figure(figsize=(10.6, 7.3))
    gs = fig.add_gridspec(2, 2, left=0.058, right=0.975, top=0.845,
                          bottom=0.075, wspace=0.20, hspace=0.32)

    # (a) candidate arcs ----------------------------------------------------
    ax_a = ax = fig.add_subplot(gs[0, 0])
    draw_obstacles(ax)
    ax.plot(ref[:, 0], ref[:, 1], color=REF_C, lw=1.1, ls=(0, (4.0, 2.0)),
            zorder=4, label="global reference")
    ax.plot(log["xy"][:, 0], log["xy"][:, 1], color="#9aa3ab", lw=1.0,
            zorder=4)
    tot = sc["total"].ravel()
    good = sc["ok"]
    norm = mpl.colors.Normalize(np.nanmin(tot[good]), np.nanmax(tot[good]))
    cmap = mpl.colormaps["viridis"]
    segs, cols = [], []
    for i in np.flatnonzero(good):
        segs.append(np.column_stack([np.r_[state[0], sc["xs"][i]],
                                     np.r_[state[1], sc["ys"][i]]]))
        cols.append(cmap(norm(tot[i])))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.5,
                                     alpha=0.9, zorder=6))
    bad = np.flatnonzero(sc["inwin"] & ~sc["adm"])
    for i in bad[::2]:
        ax.plot(np.r_[state[0], sc["xs"][i]], np.r_[state[1], sc["ys"][i]],
                color="#c9c9c9", lw=0.35, alpha=0.75, zorder=5.5)
    b = sc["best"]
    ax.plot(np.r_[state[0], sc["xs"][b]], np.r_[state[1], sc["ys"][b]],
            color="white", lw=3.0, zorder=7)
    ax.plot(np.r_[state[0], sc["xs"][b]], np.r_[state[1], sc["ys"][b]],
            color=BEST_C, lw=1.9, zorder=8, label="selected arc")
    ax.plot(*lg, marker="o", ms=4.5, mfc=REF_C, mec="white", mew=0.8,
            ls="none", zorder=9)
    ax.add_patch(Circle(state[:2], ROB_R, fc="#0072b2", ec="white", lw=0.7,
                        zorder=9))
    ax.annotate("", xy=(state[0] + 0.9 * np.cos(state[2]),
                        state[1] + 0.9 * np.sin(state[2])),
                xytext=(state[0], state[1]),
                arrowprops=dict(arrowstyle="-|>", color="#0072b2", lw=1.1,
                                mutation_scale=7), zorder=10)
    ax.plot(*START, marker="^", ms=8, mfc=S_C, mec="white", mew=0.9,
            ls="none", zorder=9)
    ax.plot(*GOAL, marker="*", ms=13, mfc=G_C, mec="white", mew=0.9,
            ls="none", zorder=9)
    map_frame(ax)
    ax.set_title("Dynamic window rollout at t = %.2f s" % (DT * ksnap),
                 fontsize=8.6, fontweight="bold", pad=3)
    add_colorbar(fig, mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax,
                 label="total cost (-)", fraction=0.036, pad=0.02, aspect=16)
    ax.text(0.018, 0.035, "%d of %d window pairs admissible\n"
            "chosen: v = %.2f m/s, $\\omega$ = %+.2f rad/s"
            % (int(good.sum()), int(sc["inwin"].sum()),
               sc["V"][b], sc["Om"][b]),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.3,
            linespacing=1.4, zorder=11,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#bbbbbb",
                      lw=0.6, alpha=0.94))

    # (b) cost terms in velocity space -------------------------------------
    sub = gs[0, 1].subgridspec(1, 3, wspace=0.14)
    ext = [-W_MAX, W_MAX, 0.0, V_MAX]
    fields = [("heading", sc["head"]), ("clearance", sc["clear"]),
              ("velocity", sc["vel"])]
    bi, bj = divmod(b, NW)
    vlo, vhi, wlo, whi = sc["win"]
    ax_b0 = None
    for m, (nm, F) in enumerate(fields):
        a = fig.add_subplot(sub[m])
        if m == 0:
            ax_b0 = a
        a.set_facecolor("#dcdcdc")
        im = a.imshow(F, extent=ext, origin="lower", aspect="auto",
                      cmap="magma_r", vmin=0, vmax=1, interpolation="nearest")
        a.add_patch(Rectangle((wlo, vlo), whi - wlo, vhi - vlo, fc="none",
                              ec="#3d6b8e", lw=1.0, ls=(0, (3.0, 1.6)),
                              zorder=4))
        a.plot(sc["ws"][bj], sc["vs"][bi], marker="o", ms=4.5, mfc="none",
               mec="#3d6b8e", mew=1.2, zorder=5)
        a.set_title(nm, fontsize=6.8, fontweight="bold", pad=1.8)
        a.set_xticks([-1, 0, 1])
        a.set_yticks([0, 0.4, 0.8, 1.2])
        a.tick_params(labelsize=6.0, length=1.8, pad=1.2)
        a.set_xlabel("$\\omega$ (rad/s)", fontsize=6.2, labelpad=0.8)
        if m == 0:
            a.set_ylabel("v (m/s)", fontsize=6.4, labelpad=1.2)
        else:
            a.set_yticklabels([])
        for s in a.spines.values():
            s.set_linewidth(0.7)
            s.set_color("#333333")
            s.set_visible(True)
        if m == 2:
            cb = add_colorbar(fig, im, a, label="normalised cost",
                              fraction=0.075, pad=0.05, aspect=14,
                              ticks=[0, 0.5, 1])
            cb.ax.tick_params(labelsize=6.0)
        if m == 1:
            a.text(0.5, -0.125, "grey = inadmissible (cannot brake before "
                   "the first obstacle);  dashed box = dynamic window;  "
                   "circle = selected pair", transform=a.transAxes,
                   ha="center", va="top", fontsize=6.0, color="#666")

    # (c) velocity commands -------------------------------------------------
    ax_c = ax = fig.add_subplot(gs[1, 0])
    ax.plot(log["t"], log["v"], color="#0072b2", lw=1.3,
            label="linear velocity $v$")
    ax.axhline(V_MAX, color="#0072b2", lw=0.7, ls=(0, (2.5, 1.6)), alpha=0.7)
    ax.text(log["t"][-1] * 0.995, V_MAX + 0.02, "$v_{max}$", fontsize=6.2,
            color="#0072b2", ha="right", va="bottom")
    ax.set_xlabel("time (s)", fontsize=7.6, labelpad=1.5)
    ax.set_ylabel("$v$ (m/s)", fontsize=7.6, labelpad=1.5, color="#0072b2")
    ax.set_ylim(-0.05, V_MAX * 1.22)
    ax.set_xlim(0, log["t"][-1])
    ax.tick_params(axis="y", labelcolor="#0072b2")
    ax.grid(True, lw=0.4, color="#e8e8e8")
    ax.set_axisbelow(True)
    ax2 = ax.twinx()
    ax2.plot(log["t"], log["w"], color="#d55e00", lw=1.1, alpha=0.9,
             label="angular velocity $\\omega$")
    ax2.axhline(0, color="#999999", lw=0.5)
    ax2.set_ylabel("$\\omega$ (rad/s)", fontsize=7.6, labelpad=1.5,
                   color="#d55e00")
    ax2.set_ylim(-W_MAX * 1.15, W_MAX * 1.15)
    ax2.tick_params(axis="y", labelcolor="#d55e00", labelsize=6.6)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.8)
    ax2.spines["top"].set_visible(False)
    ax.set_title("Commanded velocities along the run", fontsize=8.6,
                 fontweight="bold", pad=3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower center", ncol=2, fontsize=6.4,
              frameon=False, handlelength=1.6, bbox_to_anchor=(0.5, -0.015))
    ax.text(0.985, 0.965, "mean speed %.2f m/s, travel time %.1f s"
            % (log["v"].mean(), log["t"][-1]), transform=ax.transAxes,
            ha="right", va="top", fontsize=6.3, zorder=10,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#bbbbbb",
                      lw=0.6, alpha=0.94))

    # (d) obstacle distance -------------------------------------------------
    ax_d = ax = fig.add_subplot(gs[1, 1])
    ax.plot(log["t"], log["d"], color="#2e7d5b", lw=1.3,
            label="distance to nearest obstacle")
    ax.axhline(ROB_R, color=G_C, lw=1.0, ls=(0, (2.5, 1.6)))
    ax.axhline(ROB_R + SAFE_PAD, color="#c48b1a", lw=0.9,
               ls=(0, (1.6, 1.6)))
    ax.fill_between(log["t"], 0, ROB_R, color=G_C, alpha=0.10, lw=0)
    ax.text(log["t"][-1] * 0.02, ROB_R + SAFE_PAD + 0.04,
            "planner safety threshold %.2f m" % (ROB_R + SAFE_PAD),
            fontsize=6.2, color="#a3730f", va="bottom")
    ax.text(log["t"][-1] * 0.02, 0.045,
            "robot radius %.2f m (collision below)" % ROB_R, fontsize=6.2,
            color=G_C, va="bottom")
    kmin = int(np.argmin(log["d"]))
    ax.plot(log["t"][kmin], log["d"][kmin], marker="v", ms=5, mfc=G_C,
            mec="white", mew=0.7, zorder=6)
    ax.annotate("closest approach %.2f m\nat t = %.1f s"
                % (log["d"][kmin], log["t"][kmin]),
                xy=(log["t"][kmin], log["d"][kmin]),
                xytext=(log["t"][kmin] + 3.0, log["d"][kmin] + 0.55),
                fontsize=6.3, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.7))
    ax.set_xlabel("time (s)", fontsize=7.6, labelpad=1.5)
    ax.set_ylabel("clearance (m)", fontsize=7.6, labelpad=1.5)
    ax.set_xlim(0, log["t"][-1])
    ax.set_ylim(0, max(log["d"].max() * 1.15, 1.0))
    ax.grid(True, lw=0.4, color="#e8e8e8")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=6.4, frameon=False,
              handlelength=1.7)
    ax.set_title("Safety margin along the executed trajectory",
                 fontsize=8.6, fontweight="bold", pad=3)

    # the two time-series panels get a full box so they sit comfortably
    # next to the boxed map and heatmaps
    for a in (ax_c, ax_d):
        for s in a.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.8)
            s.set_color("#333333")
    for lab, a in zip(("a", "c", "d"), (ax_a, ax_c, ax_d)):
        add_panel_label(a, "(%s)" % lab, x=-0.075, y=1.02, size=10)
    add_panel_label(ax_b0, "(b)", x=-0.30, y=1.10, size=10)

    handles = [Line2D([], [], color=REF_C, lw=1.4, ls=(0, (4.0, 2.0)),
                      label="global A* reference"),
               Line2D([], [], color=BEST_C, lw=1.8, label="selected arc"),
               Line2D([], [], color="#9aa3ab", lw=1.4,
                      label="executed trajectory"),
               Line2D([], [], marker="o", ls="none", ms=5, mfc="#0072b2",
                      mec="white", label="robot footprint"),
               Line2D([], [], marker="s", ls="none", ms=6, mfc=OBS_C,
                      mec="none", label="obstacle")]
    savefig(fig, "171_local_planner_dwa.png")


if __name__ == "__main__":
    main()

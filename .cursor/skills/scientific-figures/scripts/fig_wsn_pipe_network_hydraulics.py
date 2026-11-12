"""Looped water distribution network solved for heads, not decorated with them.

A twenty-node looped grid is fed from one service reservoir through a single
trunk main.  Pipe flows follow the Hazen-Williams law and the nodal heads are
found by a damped Newton iteration on the continuity equations, so every
number on the page comes out of the same solution: the colour of a junction is
its residual pressure head, the width of a pipe is the discharge the solver
gave it, the arrow on a pipe is the sign of that discharge, and the orange
haloes mark the junctions that fall under the service standard.  Continuity is
asserted node by node and the trunk main delivers exactly the sum of demands.
The lower side panel re-solves the whole network for a sweep of demand
multipliers, which is why the deficient-node count is a curve and not a claim.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch

from pubstyle import set_pub_style, savefig, get_cmap

from figkit import axes_inches, panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
BLUE = "#3b5b92"
DEEP = "#134a76"
RED = "#b23a48"
ORANGE = "#c0603f"
PIPE = "#94a3b5"

SEED = 1002
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.02, 4.66, 4.46)
AUX_T = (7.42, 3.61, 2.70, 1.87)
AUX_B = (7.42, 1.02, 2.70, 1.82)

NCOL, NROW = 5, 4
DX, DY = 330.0, 460.0          # junction spacing, m
RES_HEAD = 62.0                # service reservoir water level, m above datum
P_MIN = 28.0                   # minimum service pressure, m H2O
C_HW = 120.0                   # Hazen-Williams roughness coefficient
NEXP = 1.852                   # Hazen-Williams flow exponent

# Pipes that the grid does not actually contain, so the network reads as a
# real skeleton rather than as graph paper.
MISSING = {((1, 1), (2, 1)), ((3, 2), (4, 2)), ((2, 0), (2, 1)),
           ((4, 1), (4, 2))}


def build_network():
    """Node table, pipe table and the demand vector, all in SI units."""
    rng = np.random.default_rng(SEED)
    coord, elev, name = [], [], []
    index = {}
    for j in range(NROW):
        for i in range(NCOL):
            index[(i, j)] = len(coord)
            coord.append((i * DX, j * DY))
            # Ground rises to the north-east; that corner is the weak zone.
            elev.append(18.0 + 0.0042 * i * DX + 0.0062 * j * DY
                        + float(rng.normal(0.0, 0.35)))
            name.append("J{:02d}".format(len(coord)))
    coord = np.array(coord, dtype=float)
    elev = np.array(elev, dtype=float)
    n = len(coord)

    demand = 0.006 + 0.011 * rng.random(n)            # m3/s per junction
    demand[index[(4, 3)]] += 0.020                    # industrial estate
    demand[index[(3, 3)]] += 0.011
    demand[index[(0, 0)]] += 0.008

    res = n                                           # fixed-head node id
    res_xy = (-0.78 * DX, 1.48 * DY)

    pipes = []                                        # (a, b, L, D)
    for j in range(NROW):
        for i in range(NCOL):
            for di, dj in ((1, 0), (0, 1)):
                a, b = (i, j), (i + di, j + dj)
                if b not in index or (a, b) in MISSING:
                    continue
                length = DX if di else DY
                far = 0.5 * (i + di * 0.5) / (NCOL - 1) + \
                      0.5 * (j + dj * 0.5) / (NROW - 1)
                dia = 0.400 if far < 0.30 else (0.350 if far < 0.52 else
                                                (0.300 if far < 0.74 else
                                                 0.250))
                pipes.append((index[a], index[b], length, dia))
    trunk_len = float(np.hypot(coord[index[(0, 1)], 0] - res_xy[0],
                               coord[index[(0, 1)], 1] - res_xy[1]))
    pipes.append((res, index[(0, 1)], trunk_len, 0.500))
    return coord, elev, name, demand, pipes, res, res_xy, index


def resistance(pipes):
    """Hazen-Williams pipe resistance K so that head loss is K * Q ** 1.852."""
    return np.array([10.67 * L / (C_HW ** NEXP * D ** 4.87)
                     for (_a, _b, L, D) in pipes])


def pipe_flows(head_full, A, B, K):
    """Signed discharge, positive from A to B (m3/s)."""
    dh = head_full[A] - head_full[B]
    mag = np.maximum(np.abs(dh), 1e-10)
    return np.sign(dh) * (mag / K) ** (1.0 / NEXP)


def solve_heads(demand, pipes, res, K, seed_head=None, iters=90):
    """Damped Newton on nodal continuity; returns heads and pipe flows.

    The iteration needs a *sloping* start: with every head identical the pipe
    flows are zero, the Jacobian is the zero matrix and the first solve fails.
    """
    n = len(demand)
    A = np.array([p[0] for p in pipes])
    B = np.array([p[1] for p in pipes])
    head = np.full(n + 1, RES_HEAD - 2.0) if seed_head is None else \
        seed_head.copy()
    head[:n] -= 0.05 * np.arange(n)
    head[res] = RES_HEAD

    for it in range(iters):
        q = pipe_flows(head, A, B, K)
        dh = np.abs(head[A] - head[B])
        g = np.maximum(np.abs(q) / (NEXP * np.maximum(dh, 1e-6)), 1e-7)

        f = -demand.astype(float).copy()
        np.add.at(f, B[B < n], q[B < n])
        np.add.at(f, A[A < n], -q[A < n])

        J = np.zeros((n, n))
        for p, (a, b) in enumerate(zip(A, B)):
            for node, sgn in ((a, -1.0), (b, +1.0)):
                if node >= n:
                    continue
                if a < n:
                    J[node, a] += sgn * g[p]
                if b < n:
                    J[node, b] -= sgn * g[p]
        step = np.linalg.solve(J, f)
        head[:n] -= (0.55 if it < 6 else 1.0) * step
        if np.max(np.abs(f)) < 1e-13:
            break

    q = pipe_flows(head, A, B, K)
    res_vec = -demand.astype(float).copy()
    np.add.at(res_vec, B[B < n], q[B < n])
    np.add.at(res_vec, A[A < n], -q[A < n])
    return head, q, res_vec


def sweep(demand, pipes, res, K, elev, mults):
    """Minimum pressure and deficient-node count against a demand multiplier."""
    pmin, ndef = [], []
    for m in mults:
        head, _q, _r = solve_heads(demand * m, pipes, res, K)
        p = head[:len(demand)] - elev
        pmin.append(float(p.min()))
        ndef.append(int(np.sum(p < P_MIN)))
    return np.array(pmin), np.array(ndef)


def fit_limits(ax, xs, ys, rect, pad=0.10):
    """Equal-aspect limits that fill the panel instead of letterboxing it."""
    x0, x1 = float(np.min(xs)), float(np.max(xs))
    y0, y1 = float(np.min(ys)), float(np.max(ys))
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    w, h = (x1 - x0) * (1.0 + 2 * pad), (y1 - y0) * (1.0 + 2 * pad)
    box = rect[2] / rect[3]
    if w / h < box:
        w = h * box
    else:
        h = w / box
    ax.set_xlim(cx - w / 2, cx + w / 2)
    ax.set_ylim(cy - h / 2, cy + h / 2)


def network_panel(fig, ax, coord, elev, demand, pipes, res, res_xy, q, press):
    xy = np.vstack([coord, np.array([res_xy])])
    fit_limits(ax, xy[:, 0], xy[:, 1], MAIN)
    ax.set_aspect("equal")

    low = press < P_MIN
    for k in np.flatnonzero(low):
        ax.add_patch(Circle(coord[k], 128.0, facecolor="#f2c9ae", alpha=0.55,
                            edgecolor="none", zorder=1))

    qmax = float(np.abs(q).max())
    for p, (a, b, L, D) in enumerate(pipes):
        xa, ya = xy[a]
        xb, yb = xy[b]
        lw = 0.9 + 3.6 * (abs(q[p]) / qmax) ** 0.55
        ax.plot([xa, xb], [ya, yb], color=PIPE, lw=lw,
                solid_capstyle="round", zorder=2)
        if q[p] < 0:
            xa, ya, xb, yb = xb, yb, xa, ya
        mx, my = 0.5 * (xa + xb), 0.5 * (ya + yb)
        ux, uy = xb - xa, yb - ya
        d = np.hypot(ux, uy)
        ux, uy = ux / d, uy / d
        ax.add_patch(FancyArrowPatch(
            (mx - 34 * ux, my - 34 * uy), (mx + 34 * ux, my + 34 * uy),
            arrowstyle="-|>", mutation_scale=7.0, lw=0.0, color="#5c6b7d",
            shrinkA=0, shrinkB=0, zorder=5))

    cmap = get_cmap("cb_sequential_blue")
    sizes = 26.0 + 780.0 * demand
    sc = ax.scatter(coord[:, 0], coord[:, 1], s=sizes, c=press, cmap=cmap,
                    vmin=press.min(), vmax=press.max(), edgecolor="white",
                    linewidth=0.8, zorder=6)
    ax.scatter(coord[low, 0], coord[low, 1], s=sizes[low] + 90.0,
               facecolor="none", edgecolor=RED, linewidth=1.15, zorder=7)

    ax.plot([res_xy[0]], [res_xy[1]], marker="s", ms=9.5, mfc=DEEP,
            mec="white", mew=1.0, zorder=8)
    ax.text(res_xy[0], res_xy[1] + 118.0,
            "service reservoir\n{:.0f} m water level".format(RES_HEAD),
            ha="center", va="bottom", fontsize=6.3, color=DEEP,
            linespacing=1.30, zorder=9)

    for k in range(len(coord)):
        ax.text(coord[k, 0] + 44.0, coord[k, 1] + 46.0,
                "J{:02d}".format(k + 1), fontsize=6.0, color=MUTED,
                ha="left", va="bottom", zorder=9)
    worst = int(np.argmin(press))
    ax.annotate("weakest junction J{:02d}\n{:.1f} m H2O, demand {:.0f} L/s"
                .format(worst + 1, press[worst], 1000.0 * demand[worst]),
                xy=coord[worst], xytext=(620.0, ax.get_ylim()[1] - 55.0),
                fontsize=6.5, color=RED, ha="left", va="top",
                linespacing=1.32, zorder=10,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.8,
                                shrinkA=2, shrinkB=7))
    trunk_q = abs(q[-1])
    ax.text(res_xy[0] + 20.0, res_xy[1] - 265.0,
            "trunk main DN500\n{:.0f} L/s, v = {:.2f} m/s".format(
                1000.0 * trunk_q, trunk_q / (np.pi * 0.25 * 0.5 ** 2)),
            ha="left", va="top", fontsize=6.3, color=SUB, linespacing=1.30,
            zorder=9)

    handles = [
        Line2D([], [], color=PIPE, lw=3.4, label="pipe, width = discharge"),
        Line2D([], [], color="#5c6b7d", lw=0.0, marker=">", ms=4.6,
               label="solved flow direction"),
        Line2D([], [], color="none", marker="o", mfc="#9cc3dd", mec="white",
               ms=6.5, label="junction, area = demand"),
        Patch(facecolor="#f2c9ae", edgecolor="none",
              label="pressure below {:.0f} m H2O".format(P_MIN)),
    ]
    leg = ax.legend(handles=handles, loc="lower right", fontsize=6.2,
                    frameon=True, facecolor="white", edgecolor="#dddddd",
                    framealpha=0.94, handlelength=1.5, borderaxespad=0.3,
                    labelspacing=0.28)
    leg.set_zorder(11)

    ax.set_xlabel("easting (m)")
    ax.set_ylabel("northing (m)")
    ax.set_title("Peak-hour solution of a 20-junction looped grid, "
                 "aspect ratio 1:1", fontsize=9.0, fontweight="bold",
                 color=INK, pad=6.0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    safe_colorbar(fig, sc, ax, label="residual pressure head (m H2O)",
                  width=0.11, pad=0.10, labelsize=6.4)


def pressure_panel(ax, press, demand):
    order = np.argsort(press)
    y = np.arange(len(press))
    vals = press[order]
    cols = [ORANGE if v < P_MIN else BLUE for v in vals]
    ax.barh(y, vals, height=0.72, color=cols, edgecolor="white",
            linewidth=0.5, zorder=4)
    ax.axvline(P_MIN, color=RED, lw=1.0, ls=(0, (4.5, 2.4)), zorder=6)
    ax.text(P_MIN + 0.5, 1.0, "service standard\n{:.0f} m H2O".format(P_MIN),
            ha="left", va="center", fontsize=6.1, color=RED, zorder=7,
            linespacing=1.30)
    ax.set_yticks(y[::3])
    ax.set_yticklabels(["J{:02d}".format(order[k] + 1) for k in y[::3]],
                       fontsize=6.2)
    ax.set_xlim(float(press.min()) - 2.2, float(press.max()) + 1.6)
    ax.set_ylim(-0.8, len(press) - 0.2)
    ax.set_xlabel("residual pressure head (m H2O)", fontsize=6.8)
    ax.grid(axis="x", lw=0.5, alpha=0.45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Junctions ranked by pressure, {:d} below standard"
                 .format(int(np.sum(press < P_MIN))), fontsize=8.0,
                 fontweight="bold", color=INK)


def sweep_panel(ax, mults, pmin, ndef):
    ax.plot(mults, pmin, color=BLUE, lw=1.7, zorder=5)
    ax.axhline(P_MIN, color=RED, lw=0.9, ls=(0, (4.5, 2.4)), zorder=4)
    cross = float(np.interp(P_MIN, pmin[::-1], mults[::-1]))
    ax.plot([cross], [P_MIN], marker="o", ms=5.0, mfc=RED, mec="white",
            mew=0.9, zorder=8)
    ax.annotate("standard lost at\ndemand factor {:.2f}".format(cross),
                xy=(cross, P_MIN), xytext=(cross - 0.16, P_MIN - 4.6),
                fontsize=6.1, color=RED, ha="left", va="bottom",
                linespacing=1.30, zorder=9,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7,
                                shrinkA=2, shrinkB=3))
    ax.set_xlim(mults[0], mults[-1])
    ax.set_ylim(float(pmin.min()) - 1.5, float(pmin.max()) + 2.0)
    ax.set_xlabel("demand multiplier on the peak-hour pattern (-)",
                  fontsize=6.8)
    ax.set_ylabel("minimum pressure (m H2O)", fontsize=6.6, color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)
    ax.grid(lw=0.5, alpha=0.45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax2 = ax.twinx()
    ax2.step(mults, ndef, where="post", color=ORANGE, lw=1.3, zorder=6)
    ax2.set_ylim(-0.5, max(ndef) + 1.5)
    ax2.set_ylabel("junctions below standard", fontsize=6.6, color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE, labelsize=6.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(ORANGE)
    ax.set_title("Network re-solved across the demand range", fontsize=8.0,
                 fontweight="bold", color=INK)
    return cross


def audit(fig, foot, axes):
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for ax in axes:
        ab = ax.get_window_extent()
        for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            tb = t.get_window_extent()
            if tb.width <= 0.0 or not t.get_text():
                continue
            if tb.x1 < ab.x0 - 2.0 or tb.x0 > ab.x1 + 2.0:
                continue
            if fb.overlaps(tb):
                print("OVERLAP with", repr(t.get_text()))


def main():
    set_pub_style()
    coord, elev, name, demand, pipes, res, res_xy, index = build_network()
    K = resistance(pipes)
    head, q, resid = solve_heads(demand, pipes, res, K)

    assert np.max(np.abs(resid)) < 1e-9, "nodal continuity is not satisfied"
    assert abs(abs(q[-1]) - demand.sum()) < 1e-9, "supply must equal demand"
    press = head[:len(demand)] - elev

    mults = np.linspace(0.40, 1.60, 49)
    pmin, ndef = sweep(demand, pipes, res, K, elev, mults)
    print("pressure range {:.1f} to {:.1f} m H2O, {:d} below standard"
          .format(press.min(), press.max(), int(np.sum(press < P_MIN))))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_net = axes_inches(fig, MAIN)
    ax_bar = axes_inches(fig, AUX_T)
    ax_swp = axes_inches(fig, AUX_B)

    network_panel(fig, ax_net, coord, elev, demand, pipes, res, res_xy, q,
                  press)
    pressure_panel(ax_bar, press, demand)
    cross = sweep_panel(ax_swp, mults, pmin, ndef)

    n_low = int(np.sum(press < P_MIN))


    panel_labels_inches(fig, [ax_net, ax_bar, ax_swp], dx=-0.62, dy=0.14)
    savefig(fig, "1002_wsn_pipe_network_hydraulics.png")


if __name__ == "__main__":
    main()

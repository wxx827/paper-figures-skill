"""Dynamic obstacle avoidance: four snapshots of an online replanning run.

The executed track comes from a rate-limited reactive guidance law
(goal attraction + moving-obstacle repulsion).  At every snapshot the
dashed curve is a fresh receding-horizon rollout that only uses the
obstacles already inside the sensing range, so plan and execution diverge
exactly where a new detection occurs.  The strip at the bottom marks the
replanning triggers.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, FancyArrow

from pubstyle import set_pub_style, savefig, add_panel_label, style_axes

LIM = 6.0                     # km
START, GOAL = np.array([0.4, 0.6]), np.array([5.6, 5.4])
VCRUISE = 26.0                # m/s
OMEGA = 0.16                  # rad/s, maximum turn rate
DT = 0.5                      # s
TEND = 360.0
RSENSE = 1.25                 # km, on-board sensing radius
RSAFE = 0.12                  # km, hard clearance around any obstacle

STATIC = [(2.55, 3.30, 0.48), (4.30, 1.70, 0.42)]
MOVING = [dict(p0=(5.30, 0.90), v=(-0.0092, 0.0075), r=0.34, tag="D1"),
          dict(p0=(0.90, 4.70), v=(0.0102, -0.0048), r=0.30, tag="D2"),
          dict(p0=(3.30, 5.90), v=(0.0016, -0.0104), r=0.27, tag="D3")]
SNAPS = [0.0, 100.0, 190.0, 300.0]
ALLTAGS = tuple(m["tag"] for m in MOVING)


def obstacle_state(t):
    return [(np.array(m["p0"]) + np.array(m["v"]) * t, np.array(m["v"]),
             m["r"], m["tag"]) for m in MOVING]


def rollout(p0, psi0, t0, known=ALLTAGS, tend=TEND):
    """Reactive guidance rollout using only the `known` moving obstacles."""
    n = int((tend - t0) / DT) + 1
    P = np.zeros((n, 2))
    P[0] = p0
    psi = float(psi0)
    hdg = np.zeros(n)
    hdg[0] = psi
    trig, active = [], {tag: False for tag in ALLTAGS}
    for k in range(1, n):
        t = t0 + (k - 1) * DT
        p = P[k - 1]
        to_goal = GOAL - p
        rep = to_goal / max(np.linalg.norm(to_goal), 1e-6)
        for cx, cy, r in STATIC:
            d = p - np.array([cx, cy])
            dn = np.linalg.norm(d)
            reach = r + RSAFE + 0.55
            if dn < reach:
                rep += 2.6 * (1.0 / max(dn - r, 0.05)
                              - 1.0 / (reach - r)) * d / dn
        for q, v, r, tag in obstacle_state(t):
            d = p - q
            dn = np.linalg.norm(d)
            seen = dn < RSENSE
            if seen and not active[tag]:
                trig.append((t, tag))
            active[tag] = seen
            if tag not in known:
                continue
            reach = r + RSAFE + 0.72
            if dn < reach:
                lat = np.array([-v[1], v[0]])
                lat /= max(np.linalg.norm(lat), 1e-9)
                lat *= 1.0 if np.dot(lat, d) >= 0 else -1.0
                w = 1.0 / max(dn - r, 0.05) - 1.0 / (reach - r)
                rep += 3.1 * w * d / dn + 1.5 * w * lat
        des = np.arctan2(rep[1], rep[0])
        err = (des - psi + np.pi) % (2 * np.pi) - np.pi
        psi += np.clip(err, -OMEGA * DT, OMEGA * DT)
        P[k] = p + (VCRUISE * DT / 1000.0) * np.array([np.cos(psi),
                                                       np.sin(psi)])
        hdg[k] = psi
    return np.arange(n) * DT + t0, P, hdg, trig


def min_clearance(t, P):
    worst = 9.9
    for k, tk in enumerate(t):
        for q, _, r, _ in obstacle_state(tk):
            worst = min(worst, np.linalg.norm(P[k] - q) - r)
        for cx, cy, r in STATIC:
            worst = min(worst, np.linalg.norm(P[k] - np.array([cx, cy])) - r)
    return worst


def draw_uav(ax, p, psi, color="#b23a48", size=0.24, zorder=12):
    body = np.array([[1.0, 0.0], [-0.55, 0.6], [-0.22, 0.0], [-0.55, -0.6]])
    R = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
    pts = (body * size) @ R.T + p
    ax.fill(pts[:, 0], pts[:, 1], facecolor=color, edgecolor="#5c1113",
            linewidth=0.6, zorder=zorder)


def main():
    set_pub_style(base_font=8)
    psi0 = float(np.arctan2(*(GOAL - START)[::-1]))
    t, P, hdg, trig = rollout(START, psi0, 0.0)
    worst = min_clearance(t, P)

    fig = plt.figure(figsize=(7.4, 4.0))
    L, R, TOPY, BOT = 0.052, 0.995, 0.845, 0.445
    w = (R - L) / 4.0
    axes = []
    for si, ts in enumerate(SNAPS):
        ax = fig.add_axes([L + si * w, BOT, w * 0.955, TOPY - BOT])
        axes.append(ax)
        k = int(round(ts / DT))
        seen = [tag for q, _, r, tag in obstacle_state(ts)
                if np.linalg.norm(P[k] - q) < RSENSE]
        _, Pp, _, _ = rollout(P[k], hdg[k], ts, known=tuple(seen))

        for cx, cy, r in STATIC:
            ax.add_patch(Circle((cx, cy), r, facecolor="#8c8c8c", alpha=0.5,
                                edgecolor="#4d4d4d", lw=0.7, zorder=3))
        ax.add_patch(Circle(P[k], RSENSE, facecolor="#1f6fb4", alpha=0.07,
                            edgecolor="#1f6fb4", lw=0.7, ls=(0, (3, 2)),
                            zorder=2))
        for q, v, r, tag in obstacle_state(ts):
            hot = tag in seen
            ax.add_patch(Circle(q, r, facecolor="#e07b1f",
                                alpha=0.55 if hot else 0.28,
                                edgecolor="#a3520d", lw=1.0 if hot else 0.6,
                                zorder=4))
            vv = v / np.linalg.norm(v) * 0.66
            ax.add_patch(FancyArrow(q[0], q[1], vv[0], vv[1], width=0.03,
                                    head_width=0.15, head_length=0.17,
                                    length_includes_head=True,
                                    facecolor="#a3520d", edgecolor="none",
                                    zorder=6))
            ax.text(q[0] + 0.30, q[1] + 0.30, tag, ha="left", va="bottom",
                    fontsize=6.0, color="#7a3d08", zorder=7,
                    fontweight="bold" if hot else "normal")
        ax.plot(Pp[:, 0], Pp[:, 1], color="#7b3294", lw=1.1, ls=(0, (3.5, 2)),
                zorder=6)
        ax.plot(P[:k + 1, 0], P[:k + 1, 1], color="#b23a48", lw=2.0, zorder=8,
                solid_capstyle="round")
        draw_uav(ax, P[k], hdg[k])
        ax.plot(*START, marker="^", ms=6.5, mfc="#2e7d5b", mec="#123d12",
                mew=0.8, ls="none", zorder=9)
        ax.plot(*GOAL, marker="*", ms=11, mfc="#b23a48", mec="#5c1113",
                mew=0.8, ls="none", zorder=9)
        lab = f"$t$ = {ts:.0f} s"
        ax.text(0.045, 0.962, lab, transform=ax.transAxes, ha="left",
                va="top", fontsize=7.0, fontweight="bold", zorder=13,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                          edgecolor="#bbbbbb", lw=0.6, alpha=0.94))
        if seen:
            ax.text(0.955, 0.962, "detected: " + ",".join(sorted(seen)),
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.0, color="#a3520d", zorder=13,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                              edgecolor="#e0c39a", lw=0.6, alpha=0.94))
        ax.set_xlim(0, LIM)
        ax.set_ylim(0, LIM)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(0, 7, 2))
        ax.set_yticks(np.arange(0, 7, 2))
        ax.grid(lw=0.3, alpha=0.4)
        ax.tick_params(labelsize=6.4)
        style_axes(ax)
        add_panel_label(ax, f"({'abcd'[si]})", x=-0.05, y=1.015, size=9.5)
        ax.set_xlabel("x (km)", fontsize=7, labelpad=1)
        if si == 0:
            ax.set_ylabel("y (km)", fontsize=7, labelpad=1)
        else:
            ax.tick_params(labelleft=False)

    # ---- replanning time axis ---------------------------------------------
    axt = fig.add_axes([L, 0.215, R - L, 0.085])
    axt.set_xlim(0, TEND)
    axt.set_ylim(0, 1)
    axt.plot([0, TEND], [0.34, 0.34], color="#333333", lw=1.0, zorder=2)
    for i, (tt, tag) in enumerate(trig):
        ytop = 0.60 if i % 2 else 0.85
        axt.plot([tt, tt], [0.34, ytop], color="#e07b1f", lw=1.0, zorder=3)
        axt.plot([tt], [ytop], marker="o", ms=3.2, color="#e07b1f", zorder=4)
        axt.text(tt, ytop + 0.04, f"replan \u00b7 {tag} @ {tt:.0f} s",
                 ha="center", va="bottom", fontsize=6.0, color="#a3520d")
    for si, ts in enumerate(SNAPS):
        axt.plot([ts], [0.34], marker="D", ms=4.2, mfc="#b23a48",
                 mec="white", mew=0.6, zorder=5)
        axt.text(ts, 0.20, f"({'abcd'[si]})", ha="center", va="top",
                 fontsize=6.4, fontweight="bold", color="#a11417")
    axt.set_yticks([])
    axt.set_xticks(np.arange(0, TEND + 1, 40))
    axt.set_xlabel("Mission time  t (s)", fontsize=7.5, labelpad=2)
    for s in ("top", "right", "left", "bottom"):
        axt.spines[s].set_visible(False)
    axt.tick_params(axis="x", length=0, pad=2, labelsize=6.4)

    handles = [
        Line2D([], [], color="#b23a48", lw=2.0, label="Executed track"),
        Line2D([], [], color="#7b3294", lw=1.1, ls=(0, (3.5, 2)),
               label="Receding-horizon plan from current state"),
        Patch(facecolor="#e07b1f", alpha=0.55, edgecolor="#a3520d", lw=0.9,
              label="Moving obstacle + velocity vector"),
        Patch(facecolor="#8c8c8c", alpha=0.55, edgecolor="#4d4d4d", lw=0.7,
              label="Static no-fly zone"),
        Patch(facecolor="#1f6fb4", alpha=0.14, edgecolor="#1f6fb4", lw=0.7,
              label=f"Sensing range ({RSENSE*1000:.0f} m)"),
        Line2D([], [], color="#2e7d5b", marker="^", ls="none", ms=6,
               mec="#123d12", mew=0.7, label="Start S"),
        Line2D([], [], color="#b23a48", marker="*", ls="none", ms=9,
               mec="#5c1113", mew=0.7, label="Goal G"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.012), frameon=False, fontsize=6.9,
               handlelength=1.8, columnspacing=1.4)

    print(f"min clearance = {worst*1000:.0f} m (bound {RSAFE*1000:.0f} m) "
          f"ok={worst >= RSAFE}")
    print("replan triggers:", [(round(a), b) for a, b in trig])
    print(f"path length = "
          f"{np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)):.2f} km, "
          f"direct = {np.linalg.norm(GOAL-START):.2f} km, "
          f"final gap = {np.linalg.norm(P[-1]-GOAL)*1000:.0f} m")
    savefig(fig, "158_uav_dynamic_avoid.png")


if __name__ == "__main__":
    main()

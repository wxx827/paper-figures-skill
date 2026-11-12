"""Cooperative multi-UAV mission: deconflicted tracks, separation, tasking.

(a) four deconflicted tracks with cooperative avoidance points,
(b) pairwise separation vs time against the safety bound,
(c) UAV-task assignment cost matrix with the optimal assignment,
(d) formation snapshots sampled along the mission.
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment

from pubstyle import (set_pub_style, savefig, add_panel_label, style_axes,
                      add_colorbar)

XMAX = YMAX = 10.0
TEND = 420.0                 # s
NT = 601
DSAFE = 150.0                # m, minimum admissible separation

NOFLY = [(3.0, 6.4, 0.90), (6.4, 3.2, 0.85), (6.6, 6.8, 0.80)]
UAV = {
    "UAV-1 (leader)": dict(c="#b23a48", lw=2.0,
                           wp=[(0.6, 0.8), (2.6, 2.4), (4.6, 4.6),
                               (5.4, 6.8), (6.4, 8.4), (9.0, 8.6)]),
    "UAV-2": dict(c="#1f6fb4", lw=1.5,
                  wp=[(0.8, 4.2), (1.4, 6.0), (2.0, 8.0), (4.0, 8.8),
                      (6.4, 9.0), (8.6, 9.4)]),
    "UAV-3": dict(c="#2e8b57", lw=1.5,
                  wp=[(1.2, 9.0), (1.8, 7.2), (3.4, 5.2), (5.2, 4.2),
                      (6.8, 4.4), (8.0, 3.2), (8.8, 2.0)]),
    "UAV-4": dict(c="#e07b1f", lw=1.5,
                  wp=[(4.0, 0.6), (3.0, 2.4), (2.2, 4.4), (1.4, 6.4),
                      (1.4, 8.2), (2.0, 9.2)]),
}
TASKS = ["Recon\nA", "Recon\nB", "Relay\nC", "Strike\nD", "Survey\nE",
         "Survey\nF"]


def spline_xy(wp, n=NT):
    wp = np.asarray(wp, float)
    tck, _ = splprep([wp[:, 0], wp[:, 1]], s=0, k=3)
    u = np.linspace(0, 1, n)
    xy = splev(u, tck)
    return np.column_stack([xy[0], xy[1]])


def deconflict(tracks, dsafe_km, zones=(), zmargin=0.06, iters=800, sigma=9.0):
    """Repulsion applied to a *smoothed correction field*.

    Smoothing the displacement instead of the absolute position keeps the
    nominal route shape and turns each conflict into a local, C1 avoidance
    bump that decays to zero at both ends of the leg.
    """
    P0 = [t.copy() for t in tracks]
    D = [np.zeros_like(t) for t in tracks]
    pairs = list(itertools.combinations(range(len(P0)), 2))
    for _ in range(iters):
        P = [P0[k] + D[k] for k in range(len(P0))]
        hit = False
        for i, j in pairs:
            d = np.linalg.norm(P[i] - P[j], axis=1)
            bad = d < dsafe_km
            if bad.any():
                hit = True
                v = P[i][bad] - P[j][bad]
                nv = np.linalg.norm(v, axis=1, keepdims=True)
                u = np.where(nv > 1e-9, v / np.maximum(nv, 1e-9),
                             np.array([1.0, 0.0]))
                push = 0.6 * (dsafe_km - d[bad])[:, None] * u
                D[i][bad] += push
                D[j][bad] -= push
        for k in range(len(P)):
            for cx, cy, r in zones:
                v = P[k] - np.array([cx, cy])
                d = np.linalg.norm(v, axis=1)
                bad = d < r + zmargin
                if bad.any():
                    hit = True
                    D[k][bad] += (((r + zmargin - d[bad]) / d[bad])[:, None]
                                  * v[bad])
        for k in range(len(D)):
            D[k][:, 0] = gaussian_filter1d(D[k][:, 0], sigma, mode="constant")
            D[k][:, 1] = gaussian_filter1d(D[k][:, 1], sigma, mode="constant")
            D[k][0] = D[k][-1] = 0.0
        if not hit:
            break
    return [P0[k] + D[k] for k in range(len(P0))]


def main():
    set_pub_style(base_font=8)
    names = list(UAV)
    raw = [spline_xy(UAV[n]["wp"]) for n in names]
    trk = deconflict(raw, DSAFE / 1000.0 * 1.06, zones=NOFLY)
    t = np.linspace(0.0, TEND, NT)

    pairs = list(itertools.combinations(range(len(trk)), 2))
    dist = {(i, j): np.linalg.norm(trk[i] - trk[j], axis=1) * 1000.0
            for i, j in pairs}
    dmin = min(d.min() for d in dist.values())

    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.975, top=0.925,
                          bottom=0.115, hspace=0.42, wspace=0.30,
                          width_ratios=[1.0, 1.06])

    # ---- (a) tracks --------------------------------------------------------
    axa = fig.add_subplot(gs[0, 0])
    for cx, cy, r in NOFLY:
        axa.add_patch(Circle((cx, cy), r, facecolor="#b23a48", alpha=0.15,
                             edgecolor="none", zorder=1))
        axa.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor="#b23a48",
                             lw=0.8, ls=(0, (3, 1.6)), zorder=2))
    for k, n in enumerate(names):
        st = UAV[n]
        axa.plot(trk[k][:, 0], trk[k][:, 1], color=st["c"], lw=st["lw"],
                 zorder=5, label=n)
        axa.plot(*trk[k][0], marker="^", ms=7, mfc=st["c"], mec="#222222",
                 mew=0.7, ls="none", zorder=7)
        axa.plot(*trk[k][-1], marker="*", ms=11, mfc=st["c"], mec="#222222",
                 mew=0.7, ls="none", zorder=7)
        dev = np.linalg.norm(trk[k] - raw[k], axis=1)
        if dev.max() > 0.015:
            i = int(np.argmax(dev))
            axa.plot(trk[k][i, 0], trk[k][i, 1], marker="o", ms=8,
                     mfc="none", mec="#333333", mew=0.9, ls="none", zorder=8)
    # annotate the globally closest encounter
    ci, cj = min(dist, key=lambda p: dist[p].min())
    it = int(np.argmin(dist[(ci, cj)]))
    mx = 0.5 * (trk[ci][it, 0] + trk[cj][it, 0])
    my = 0.5 * (trk[ci][it, 1] + trk[cj][it, 1])
    axa.annotate(f"closest approach\n{dist[(ci, cj)][it]:.0f} m @ "
                 f"t = {t[it]:.0f} s", xy=(mx, my), xycoords="data",
                 xytext=(0.47, 0.035), textcoords="axes fraction",
                 fontsize=6.3, ha="left", va="bottom", zorder=9,
                 arrowprops=dict(arrowstyle="->", lw=0.7, color="#333333",
                                 shrinkA=2, shrinkB=3),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor="#bbbbbb", lw=0.6, alpha=0.93))
    axa.set_xlim(0, XMAX)
    axa.set_ylim(0, YMAX)
    axa.set_aspect("equal")
    axa.set_xlabel("East  x (km)")
    axa.set_ylabel("North  y (km)")
    axa.set_title("Deconflicted cooperative tracks", fontsize=8.5)
    axa.grid(True, lw=0.4, alpha=0.5)
    style_axes(axa)
    add_panel_label(axa, "(a)", x=-0.17, y=1.02, size=10)

    # ---- (b) separation ----------------------------------------------------
    axb = fig.add_subplot(gs[0, 1])
    grey = ["#8c9aab", "#a8b4a0", "#c0a8b4", "#9aa8c0", "#b4b48c"]
    for k, (i, j) in enumerate(pairs):
        hot = (i, j) == (ci, cj)
        axb.plot(t, dist[(i, j)], lw=1.6 if hot else 0.9,
                 color="#b23a48" if hot else grey[k % len(grey)],
                 zorder=6 if hot else 3,
                 label=f"UAV-{i+1}\u2013UAV-{j+1}")
    axb.axhline(DSAFE, color="#a11417", lw=1.1, ls=(0, (4, 2)), zorder=5)
    axb.text(0.28, DSAFE * 1.14, f"minimum safe separation "
             f"$d_{{safe}}$ = {DSAFE:.0f} m",
             transform=axb.get_yaxis_transform(), ha="left", va="bottom",
             fontsize=6.3, color="#a11417")
    axb.axhspan(0, DSAFE, color="#b23a48", alpha=0.07, zorder=0)
    axb.plot([t[it]], [dist[(ci, cj)][it]], marker="v", ms=5, color="#b23a48",
             zorder=7)
    axb.set_yscale("log")
    axb.set_ylim(80, 60000)
    axb.set_xlim(0, TEND)
    axb.set_xlabel("Mission time  t (s)")
    axb.set_ylabel("Pairwise separation  d (m)")
    axb.set_title("Inter-UAV separation history", fontsize=8.5)
    axb.grid(True, which="both", lw=0.35, alpha=0.45)
    style_axes(axb)
    axb.legend(fontsize=6.0, ncol=2, loc="upper center", frameon=True,
               framealpha=0.9, edgecolor="#cccccc", handlelength=1.5,
               columnspacing=1.0, labelspacing=0.25,
               bbox_to_anchor=(0.5, 1.005)).get_frame().set_linewidth(0.5)
    add_panel_label(axb, "(b)", x=-0.17, y=1.02, size=10)

    # ---- (c) task allocation ----------------------------------------------
    axc = fig.add_subplot(gs[1, 0])
    rng = np.random.default_rng(11)
    C = np.round(rng.uniform(0.8, 9.5, size=(4, 6)), 1)
    C[0, 3] = 1.2
    C[1, 0] = 1.5
    C[2, 4] = 1.1
    C[3, 1] = 1.4
    row, col = linear_sum_assignment(C)
    total_cost = float(C[row, col].sum())
    cm = LinearSegmentedColormap.from_list(
        "alloc", ["#20603f", "#6bb08a", "#dfe8d8", "#f6d9a8", "#d1663a"])
    im = axc.imshow(C, cmap=cm, aspect="auto", vmin=0, vmax=10)
    for r in range(4):
        for c in range(6):
            axc.text(c, r, f"{C[r, c]:.1f}", ha="center", va="center",
                     fontsize=6.2, color="#1a1a1a" if 1.5 < C[r, c] < 8.5
                     else "white")
    for r, c in zip(row, col):
        axc.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                    edgecolor="#b23a48", lw=1.8, zorder=5))
    axc.set_xticks(range(6))
    axc.set_xticklabels(TASKS, fontsize=6.2)
    axc.set_yticks(range(4))
    axc.set_yticklabels([n.split(" ")[0] for n in names], fontsize=7)
    axc.set_title("Task allocation cost matrix", fontsize=8.5)
    axc.tick_params(length=0)
    for s in axc.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color("#333333")
    add_colorbar(fig, im, axc, label="assignment cost (\u2013)")
    add_panel_label(axc, "(c)", x=-0.17, y=1.02, size=10)

    # ---- (d) formation snapshots ------------------------------------------
    axd = fig.add_subplot(gs[1, 1])
    lead = trk[0]
    tang = np.gradient(lead, axis=0)
    psi = np.arctan2(tang[:, 1], tang[:, 0])
    cs, sn = np.cos(psi), np.sin(psi)
    offsets = [(0.0, 0.0), (-0.62, 0.52), (-0.62, -0.52), (-1.24, 0.0)]
    form = []
    for ox, oy in offsets:
        form.append(np.column_stack([lead[:, 0] + ox * cs - oy * sn,
                                     lead[:, 1] + ox * sn + oy * cs]))
    fmin = min(np.linalg.norm(form[i] - form[j], axis=1).min() * 1000.0
               for i, j in itertools.combinations(range(4), 2))
    snaps = np.linspace(0, NT - 1, 5).astype(int)
    fade = LinearSegmentedColormap.from_list(
        "fade", ["#cfd8e3", "#7f93ad", "#33475f"])
    for k in range(4):
        axd.plot(form[k][:, 0], form[k][:, 1], color=UAV[names[k]]["c"],
                 lw=0.8, alpha=0.4, zorder=2)
    for si, idx in enumerate(snaps):
        pts = np.array([form[k][idx] for k in range(4)])
        order = [0, 1, 3, 2, 0]
        col = fade(si / (len(snaps) - 1))
        axd.fill(pts[order, 0], pts[order, 1], facecolor=col, alpha=0.13,
                 edgecolor="none", zorder=3)
        axd.plot(pts[order, 0], pts[order, 1], color=col, lw=0.9,
                 ls=(0, (3, 1.5)), zorder=4)
        for k in range(4):
            axd.plot(pts[k, 0], pts[k, 1], marker="o", ms=4.0,
                     mfc=UAV[names[k]]["c"], mec="white", mew=0.6,
                     ls="none", zorder=6)
        off = np.array([-1.05, 0.85]) if si % 2 == 0 else np.array([0.75, -0.9])
        cen = pts.mean(axis=0) + off
        axd.text(cen[0], cen[1], f"t = {t[idx]:.0f} s", fontsize=6.0,
                 ha="center", va="center", color="#1a1a1a", zorder=7,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           edgecolor=col, lw=0.6, alpha=0.92))
    axd.text(0.97, 0.035, f"rigid diamond, edge \u2265 {fmin:.0f} m",
             transform=axd.transAxes, ha="right", va="bottom", fontsize=6.3,
             color="#333333",
             bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                       edgecolor="#bbbbbb", lw=0.6, alpha=0.92))
    axd.set_xlim(-0.6, XMAX)
    axd.set_ylim(-0.4, YMAX)
    axd.set_aspect("equal")
    axd.set_xlabel("East  x (km)")
    axd.set_ylabel("North  y (km)")
    axd.set_title("Leader\u2013follower formation snapshots", fontsize=8.5)
    axd.grid(True, lw=0.4, alpha=0.5)
    style_axes(axd)
    add_panel_label(axd, "(d)", x=-0.17, y=1.02, size=10)

    handles = [Line2D([], [], color=UAV[n]["c"], lw=UAV[n]["lw"], label=n)
               for n in names]
    handles += [
        Line2D([], [], color="#222222", marker="^", ls="none", ms=6,
               mfc="#bbbbbb", mew=0.7, label="Launch point"),
        Line2D([], [], color="#222222", marker="*", ls="none", ms=9,
               mfc="#bbbbbb", mew=0.7, label="Target"),
        Line2D([], [], color="#333333", marker="o", ls="none", ms=7,
               mfc="none", mew=0.9, label="Cooperative avoidance manoeuvre"),
        Patch(facecolor="none", edgecolor="#b23a48", lw=1.6,
              label=f"Hungarian optimum (total cost {total_cost:.1f})"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=4,
                     bbox_to_anchor=(0.5, -0.004), frameon=False,
                     fontsize=7.0, handlelength=1.8, columnspacing=1.5)
    leg.get_texts()[0].set_fontweight("bold")

    speeds = [np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)) * 1000.0
              / TEND for p in trk]
    zmin = min(np.linalg.norm(p - np.array([cx, cy]), axis=1).min() - r
               for p in trk for cx, cy, r in NOFLY) * 1000.0
    print(f"min pairwise separation = {dmin:.1f} m (bound {DSAFE:.0f} m)  "
          f"ok={dmin >= DSAFE};  min no-fly margin = {zmin:+.0f} m")
    print("cruise speeds (m/s): " +
          ", ".join(f"{n}={v:.1f}" for n, v in zip(names, speeds)))
    savefig(fig, "156_uav_multi_swarm.png")


if __name__ == "__main__":
    main()

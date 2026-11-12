"""Figure 1205 - Forward reachable sets against the safe set.

The second figure allowed red, and again red means one thing only: the safe
set boundary and the part of the tube that has left it.

Everything is measured on one ensemble.  ``ENSEMBLE`` rollouts start inside
X0 and are pushed through the same closed loop with bounded disturbance; the
convex hull of the cloud at three snapshots *is* the reachable set drawn in
(a), the shares quoted for each snapshot are cell counts of that hull against
the barrier ``h``, and panel (b) is ``h`` evaluated along the very same
rollouts.  The highlighted trajectory is picked by a rule - the largest final
offset - not by hand, and its first violation is one index into one array, so
the star in (a) and the crossing in (b) cannot drift apart.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from scipy.spatial import ConvexHull

from pubstyle import savefig, set_pub_style, style_axes

SEED = 1610  # original id; figure number is 1205
KP, KD = 0.30, 0.55            # closed-loop lane keeping
DT, N_STEP, ENSEMBLE = 0.03, 45, 6000
SNAP = (15, 30, 45)            # t+1, t+2, t+3 at 0.45 s spacing
W_MAX, W_HOLD = 1.45, 5        # disturbance bound, and how long it is held

# The two limits are equal so the whole level set fits in the frame; the
# cross term is what tilts it into something that looks like a real CBF.
E_LIM, ED_LIM, CROSS = 1.70, 1.70, 0.55

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#d5d5d5"
CARD, CARD_EDGE, GRID = "#e2e8ee", "#c3cdd6", "#ededed"
DEEP, MID, SOFT, PALE = "#0b3d63", "#2e6f9e", "#5f9dc6", "#cfe1ee"
RED, RED_FILL = "#b23a48", "#f0d2d7"
SAFE_BG = "#eff5fa"

X0_C = np.array([0.25, 0.62])
X0_R = np.array([0.12, 0.16])


def barrier(e, ed):
    """h >= 0 is the safe set; the cross term tilts it like a real CBF."""
    return (1.0 - (e / E_LIM) ** 2 - (ed / ED_LIM) ** 2
            - CROSS * e * ed / (E_LIM * ED_LIM))


def rollouts():
    rng = np.random.default_rng(SEED)
    th = rng.uniform(0, 2 * np.pi, ENSEMBLE)
    rr = np.sqrt(rng.uniform(0, 1, ENSEMBLE))
    x = np.stack([X0_C[0] + X0_R[0] * rr * np.cos(th),
                  X0_C[1] + X0_R[1] * rr * np.sin(th)], axis=1)
    traj = np.empty((N_STEP + 1, ENSEMBLE, 2))
    traj[0] = x
    # A disturbance redrawn every step averages itself away over 45 steps and
    # the tube barely grows; holding it for W_HOLD steps is both the standard
    # model and the reason the sets have any width at all.
    w = np.zeros(ENSEMBLE)
    for k in range(N_STEP):
        if k % W_HOLD == 0:
            w = rng.uniform(-W_MAX, W_MAX, ENSEMBLE)
        e, ed = traj[k, :, 0], traj[k, :, 1]
        traj[k + 1, :, 0] = e + DT * ed
        traj[k + 1, :, 1] = ed + DT * (-KP * e - KD * ed + w)
    return traj


def hull_of(points):
    h = ConvexHull(points)
    return points[h.vertices]


def outside_share(poly, gx, gy, hgrid):
    """Cell counts: inside the hull, and inside the hull with h < 0."""
    from matplotlib.path import Path
    X, Y = np.meshgrid(gx, gy)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    ins = Path(poly).contains_points(pts).reshape(X.shape)
    out = ins & (hgrid < 0.0)
    return int(ins.sum()), int(out.sum())


def main():
    set_pub_style(base_font=8.5)
    traj = rollouts()
    hs = barrier(traj[..., 0], traj[..., 1])

    # the highlighted rollout is chosen by a rule, not by eye
    pick = int(np.argmax(traj[-1, :, 0]))
    hv = hs[:, pick]
    first = int(np.argmax(hv < 0.0))
    assert bool((hv < 0.0).any()), "the highlighted rollout never leaves"
    t_first = first * DT

    gx = np.linspace(-2.25, 2.65, 561)
    gy = np.linspace(-2.15, 2.15, 501)
    HG = barrier(*np.meshgrid(gx, gy))

    polys, shares = [], []
    for k in SNAP:
        poly = hull_of(traj[k])
        nin, nout = outside_share(poly, gx, gy, HG)
        polys.append(poly)
        shares.append((nin, nout, 100.0 * nout / nin))
    assert shares[0][1] == 0 and shares[-1][1] > 0, \
        "the tube either starts unsafe or never becomes unsafe"

    fig = plt.figure(figsize=(10.4, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.56], left=0.056,
                          right=0.988, top=0.826, bottom=0.118, wspace=0.185)
    ax = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])

    # ---- (a) state-space projection ----------------------------------------
    style_axes(ax)
    ax.set_axisbelow(True)
    ax.grid(color=GRID, lw=0.7, zorder=0)
    # The grid spans the whole level set because the area counts need it, but
    # the view is cropped to where the tube actually is: at full extent the
    # crescent that matters is four pixels wide.
    ax.set_xlim(-0.95, 2.30)
    ax.set_ylim(-1.25, 1.75)

    ax.contourf(gx, gy, HG, levels=[0.0, 10.0], colors=[SAFE_BG], zorder=0.6)
    fills = [(DEEP, 0.88), (MID, 0.74), (SOFT, 0.58)]
    for poly, (col, al), k in zip(polys[::-1], fills[::-1], SNAP[::-1]):
        ax.add_patch(Polygon(poly, closed=True, facecolor=col, alpha=al,
                             edgecolor="white", lw=0.9,
                             zorder=1.0 + 0.1 * (N_STEP - k)))
    # The part of the t+3 tube that has left the safe set is painted over the
    # blue rather than tinted on top of it: red at 30 % alpha over mid blue
    # reads as mauve, which is a colour this batch does not own.
    from matplotlib.path import Path
    X, Y = np.meshgrid(gx, gy)
    ins = Path(polys[-1]).contains_points(
        np.stack([X.ravel(), Y.ravel()], axis=1)).reshape(X.shape)
    ax.contourf(gx, gy, np.where(ins & (HG < 0.0), 1.0, 0.0),
                levels=[0.5, 1.5], colors=[RED_FILL], zorder=2.6)
    ax.contour(gx, gy, np.where(ins & (HG < 0.0), 1.0, 0.0), levels=[0.5],
               colors=[RED], linewidths=0.7, zorder=2.7)
    ax.contour(gx, gy, HG, levels=[0.0], colors=["white"], linewidths=3.0,
               zorder=3)
    ax.contour(gx, gy, HG, levels=[0.0], colors=[RED], linewidths=1.7,
               zorder=3.2)

    ph = np.linspace(0, 2 * np.pi, 121)
    ax.add_patch(Polygon(np.stack([X0_C[0] + X0_R[0] * np.cos(ph),
                                   X0_C[1] + X0_R[1] * np.sin(ph)], axis=1),
                         closed=True, facecolor="white", edgecolor=INK,
                         lw=1.1, zorder=5))
    # The three sets are nested, so an in-plot label always ends up sitting on
    # a *larger* set than the one it names; the times ride in the legend keys
    # instead.

    ax.plot(traj[:, pick, 0], traj[:, pick, 1], color="white", lw=3.0,
            zorder=4)
    ax.plot(traj[:, pick, 0], traj[:, pick, 1], color=INK, lw=1.5, zorder=4.2)
    ax.plot([traj[0, pick, 0]], [traj[0, pick, 1]], marker="o", ms=4.6,
            mfc="white", mec=INK, mew=1.3, zorder=6)
    sx, sy = float(traj[first, pick, 0]), float(traj[first, pick, 1])
    ax.plot([sx, 1.62], [sy, 1.44], color=RED, lw=0.8, zorder=6)
    ax.plot([sx], [sy], marker="*", ms=13.0, mfc=RED, mec="white", mew=0.9,
            zorder=7)
    ax.text(1.66, 1.44, f"first violation\nt = {t_first:.2f} s", ha="left",
            va="center", fontsize=7.2, color=INK, linespacing=1.45, zorder=7,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=RED, lw=0.6,
                      alpha=0.96))
    ax.text(X0_C[0] - X0_R[0] - 0.07, X0_C[1], "$X_0$", ha="right",
            va="center", fontsize=8.6, color=INK, fontweight="bold", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                      alpha=0.93))
    ax.text(-0.45, -1.06, "safe set  $h(e, \\dot{e}) \\geq 0$", ha="left",
            va="center", fontsize=7.8, color=SUB, zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.3,
                      alpha=0.93))
    ax.set_xlabel("lateral offset  $e$  (m)", fontsize=8.4)
    ax.set_ylabel("lateral rate  $\\dot{e}$  (m/s)", fontsize=8.4)
    ax.tick_params(labelsize=7.2, length=2.4)
    bad = int(np.argmax([s[1] > 0 for s in shares]))
    ax.set_title(f"(a)  The tube first leaves the safe set at "
                 f"t+{SNAP[bad] // 15}; this rollout crosses at "
                 f"{t_first:.2f} s", fontsize=9.2, fontweight="bold", pad=4.0)
    handles = [Patch(facecolor=c, alpha=a, edgecolor="white",
                     label=f"reachable at t+{k // 15}   ({k * DT:.2f} s)")
               for (c, a), k in zip(fills, SNAP)]
    handles += [Line2D([], [], color=RED, lw=1.7, label="safe set boundary"),
                Patch(facecolor=RED_FILL, edgecolor=RED, lw=0.7,
                      label="tube outside the safe set"),
                Line2D([], [], color=INK, lw=1.5,
                       label="highlighted rollout"),
                Line2D([], [], color=RED, lw=0.0, marker="*", ms=9.0,
                       mfc=RED, mec="white", mew=0.7,
                       label="its first violation")]
    leg = ax.legend(handles=handles, loc="upper left",
                    bbox_to_anchor=(0.005, 0.995), frameon=True,
                    facecolor="white", edgecolor=HAIR, framealpha=0.95,
                    fontsize=7.1, handlelength=1.7, labelspacing=0.32,
                    borderpad=0.48)
    leg.get_frame().set_linewidth(0.5)
    leg.set_zorder(9)

    # ---- (b) the barrier along the same rollouts ---------------------------
    style_axes(axb)
    axb.set_axisbelow(True)
    axb.grid(color=GRID, lw=0.7, zorder=0)
    ts = np.arange(N_STEP + 1) * DT
    lo, hi = np.percentile(hs, [5.0, 95.0], axis=1)
    axb.fill_between(ts, lo, hi, color=SOFT, alpha=0.34, linewidth=0.0,
                     zorder=2, label="ensemble 5-95 %")
    axb.plot(ts, hs.min(axis=1), color=MID, lw=1.1, ls=(0, (4.0, 2.4)),
             zorder=3, label="ensemble minimum")
    axb.plot(ts, hv, color=INK, lw=1.6, zorder=4,
             label="highlighted rollout")
    axb.plot([ts[0], ts[-1]], [0.0, 0.0], color=RED, lw=1.4, zorder=5,
             label="$h = 0$, safe set boundary")
    axb.plot([t_first], [hv[first]], marker="*", ms=13.0, mfc=RED,
             mec="white", mew=0.9, zorder=7)
    axb.set_xlim(ts[0], ts[-1])
    ylo = float(min(hs.min(), hv.min())) - 0.08
    axb.set_ylim(ylo, 1.02)
    for k in SNAP:
        axb.plot([k * DT, k * DT], [ylo, 1.02], color=HAIR, lw=0.8,
                 ls=(0, (3.0, 2.6)), zorder=1)
        axb.text(k * DT, 1.00, f"t+{k // 15}", ha="center", va="top",
                 fontsize=6.9, color=MUTED, zorder=6,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.1,
                           alpha=0.93))
    axb.text(t_first - 0.03, hv[first] - 0.06, f"same star as (a),\n"
             f"t = {t_first:.2f} s", ha="right", va="top", fontsize=7.0,
             color=INK, linespacing=1.45, zorder=8,
             bbox=dict(boxstyle="round,pad=0.26", fc="white", ec=RED, lw=0.5,
                       alpha=0.96))
    axb.set_xlabel("time from $X_0$  (s)", fontsize=8.4)
    axb.set_ylabel("barrier value  $h$", fontsize=8.4)
    axb.tick_params(labelsize=7.2, length=2.4)
    axb.set_title("(b)  The same barrier, read along time", fontsize=9.2,
                  fontweight="bold", pad=4.0)
    leg2 = axb.legend(loc="lower left", bbox_to_anchor=(0.005, 0.012),
                      frameon=True, facecolor="white", edgecolor=HAIR,
                      framealpha=0.95, fontsize=6.9, handlelength=1.7,
                      labelspacing=0.30, borderpad=0.42)
    leg2.get_frame().set_linewidth(0.5)
    leg2.set_zorder(9)

    # ---- headline + footer --------------------------------------------------
    nin, nout, pc = shares[-1]
    frac_bad = 100.0 * float((hs[-1] < 0.0).mean())
    h_centre = float(barrier(*traj[-1].mean(axis=0)))
    assert h_centre > 0.0, "the headline claims a safe centre that is not"
    savefig(fig, "1205_wm_safe_reachable_set.png")


if __name__ == "__main__":
    main()

"""GPS trajectory clustering over a synthetic city street grid.

164 synthetic vehicle traces are drawn on a 6.6 x 6.0 km road network, each
one tinted by the corridor cluster it belongs to and stroked at low opacity
so the mainstream path of every cluster emerges from the overlay instead of
being asserted by an annotation.  The medoid-style centre trace of each
cluster is redrawn thick over a white casing.  The two auxiliary panels on
the right are held to ~55 % of the map width and carry the counts and mean
travel times, both computed from the same arrays that are plotted.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from scipy.interpolate import make_interp_spline

from pubstyle import set_pub_style, savefig, CATEGORICAL, COLORS
from figkit import add_scale_bar, panel_labels_inches

W, H = 6.6, 6.0                      # city extent in km
INK = COLORS["ink"]
SUB = COLORS["sub"]
MUTED = COLORS["muted"]
ROAD_MAJOR = "#b4b0a8"
ROAD_MINOR = "#cbc8c1"
LAND = "#fbfaf8"
WATER = "#ccd8de"
PARK = "#e7ede2"
FRAME = "#c9c6bf"

CLUSTER_NAMES = ["C1 west-east arterial", "C2 south-north radial",
                 "C3 cross-town diagonal", "C4 orbital ring",
                 "C5 CBD feeder"]
CLUSTER_SIZES = [44, 36, 28, 32, 24]
CLUSTER_SPEED = [33.0, 19.5, 30.0, 23.0, 14.5]   # km/h, free-running mean


def build_network(rng):
    """Jittered street grid plus a ring road; returns polylines and widths."""
    nx, ny = 16, 15
    gx = np.linspace(0.20, W - 0.20, nx)
    gy = np.linspace(0.22, H - 0.22, ny)
    X, Y = np.meshgrid(gx, gy)
    r = np.hypot(X - W / 2, Y - H / 2)
    jit = 0.020 + 0.048 * np.clip(r / 3.0, 0, 1) ** 1.5
    X = X + rng.normal(0, 1, X.shape) * jit
    Y = Y + rng.normal(0, 1, Y.shape) * jit

    def width(i, j):
        if i % 5 == 0 or j % 5 == 0:
            return 1.25
        if i % 2 == 0 or j % 2 == 0:
            return 0.70
        return 0.46

    segs, lws = [], []
    for i in range(ny):
        for j in range(nx - 1):
            w = width(i, j)
            if w < 0.5 and rng.random() < 0.24:
                continue
            segs.append(np.array([[X[i, j], Y[i, j]],
                                  [X[i, j + 1], Y[i, j + 1]]]))
            lws.append(w)
    for i in range(ny - 1):
        for j in range(nx):
            w = width(i, j)
            if w < 0.5 and rng.random() < 0.24:
                continue
            segs.append(np.array([[X[i, j], Y[i, j]],
                                  [X[i + 1, j], Y[i + 1, j]]]))
            lws.append(w)

    th = np.linspace(0, 2 * np.pi, 240)
    rr = 1.88 + 0.10 * np.cos(3 * th + 0.5) + 0.05 * np.cos(5 * th)
    ring = np.c_[W / 2 + rr * np.cos(th), H / 2 + rr * np.sin(th)]
    for a in range(0, 232, 8):
        segs.append(ring[a:a + 9])
        lws.append(1.30)
    return segs, np.array(lws)


def cluster_templates():
    """Five corridor skeletons, each a short list of control points."""
    ang = np.linspace(np.deg2rad(205), np.deg2rad(15), 6)
    ring = np.c_[W / 2 + 1.95 * np.cos(ang), H / 2 + 1.95 * np.sin(ang)]
    return [
        np.array([(0.30, 1.15), (1.60, 1.55), (3.05, 2.25),
                  (4.55, 2.45), (6.32, 2.10)]),
        np.array([(2.15, 0.28), (2.55, 1.60), (3.05, 2.95),
                  (3.40, 4.35), (3.62, 5.74)]),
        np.array([(0.45, 5.58), (1.75, 4.55), (3.00, 3.45),
                  (4.40, 1.95), (6.08, 0.72)]),
        ring,
        np.array([(6.10, 4.95), (5.05, 4.35), (4.05, 3.75),
                  (3.35, 3.15), (2.88, 2.72)]),
    ]


def sample_track(ctrl, rng, n=150):
    """One trace: jittered control points, cubic spline, small road wiggle."""
    w = np.ones(len(ctrl))
    w[0] = w[-1] = 0.42                       # endpoints stay near the anchor
    p = ctrl + rng.normal(0, 0.26, ctrl.shape) * w[:, None]
    t = np.linspace(0, 1, len(p))
    xy = make_interp_spline(t, p, k=3)(np.linspace(0, 1, n))
    wig = np.cumsum(rng.normal(0, 0.013, xy.shape), axis=0)
    xy = xy + 0.38 * (wig - wig.mean(axis=0))
    xy[:, 0] = np.clip(xy[:, 0], 0.08, W - 0.08)
    xy[:, 1] = np.clip(xy[:, 1], 0.08, H - 0.08)
    return xy


def main():
    set_pub_style()
    rng = np.random.default_rng(480)
    segs, lws = build_network(rng)

    tracks, centres, durations = [], [], []
    for k, (ctrl, n_tr, v0) in enumerate(
            zip(cluster_templates(), CLUSTER_SIZES, CLUSTER_SPEED)):
        grp = np.stack([sample_track(ctrl, rng) for _ in range(n_tr)])
        length = np.hypot(*np.diff(grp, axis=1).transpose(2, 0, 1)).sum(axis=1)
        speed = v0 * np.exp(rng.normal(0.0, 0.20, n_tr))
        tracks.append(grp)
        centres.append(grp.mean(axis=0))
        durations.append(60.0 * length / speed)

    fig = plt.figure(figsize=(10.4, 6.4))
    axm = fig.add_axes([0.032, 0.062, 0.578, 0.818])
    axn = fig.add_axes([0.660, 0.560, 0.312, 0.318])
    axd = fig.add_axes([0.660, 0.082, 0.312, 0.318])

    # ------------------------------------------------------------- base map
    axm.add_patch(Rectangle((0, 0), W, H, facecolor=LAND, edgecolor="none",
                            zorder=0))
    ry = np.linspace(-0.2, H + 0.2, 220)
    rx = 5.15 + 0.48 * np.sin(ry / 1.55) + 0.14 * np.sin(ry / 0.55)
    axm.plot(rx, ry, color=WATER, lw=7.0, solid_capstyle="round", zorder=1)
    for cx, cy, r0 in [(1.25, 1.40, 0.50), (2.55, 4.85, 0.40)]:
        th = np.linspace(0, 2 * np.pi, 60)
        rad = r0 * (1 + 0.16 * np.cos(3 * th + cx) + 0.08 * np.cos(5 * th))
        axm.add_patch(Polygon(np.c_[cx + rad * np.cos(th),
                                    cy + rad * np.sin(th)],
                              facecolor=PARK, edgecolor="#d5ddcd", lw=0.6,
                              zorder=1))
    axm.add_collection(LineCollection(
        segs, colors=np.where(lws > 0.5, ROAD_MAJOR, ROAD_MINOR),
        linewidths=lws, capstyle="round", zorder=2))

    # ----------------------------------------------------- clustered traces
    for k, grp in enumerate(tracks):
        axm.add_collection(LineCollection(
            list(grp), colors=CATEGORICAL[k], linewidths=0.62, alpha=0.26,
            capstyle="round", zorder=3))
    for k, ctr in enumerate(centres):
        axm.plot(ctr[:, 0], ctr[:, 1], color="white", lw=3.1,
                 solid_capstyle="round", zorder=4)
        axm.plot(ctr[:, 0], ctr[:, 1], color=CATEGORICAL[k], lw=1.75,
                 solid_capstyle="round", zorder=5)
        axm.plot([ctr[-1, 0]], [ctr[-1, 1]], marker="o", ms=4.0,
                 mfc="white", mec=CATEGORICAL[k], mew=1.2, zorder=6)

    axm.set_xlim(0, W)
    axm.set_ylim(0, H)
    axm.set_aspect("equal")
    axm.set_xticks([])
    axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
        sp.set_edgecolor(FRAME)

    add_scale_bar(axm, 1.0, "1 km", loc=(0.032, 0.042), height=0.014,
                  color="#333333", text_color="#1a1a1a", halo="white",
                  fontsize=6.2)
    axm.annotate("N", xy=(6.30, 5.70), xytext=(6.30, 5.24), ha="center",
                 fontsize=7.0, color=INK, fontweight="bold", zorder=9,
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))

    handles = [Line2D([], [], color=CATEGORICAL[k], lw=1.9,
                      label=f"{CLUSTER_NAMES[k]}  (n={CLUSTER_SIZES[k]})")
               for k in range(5)]
    leg = axm.legend(handles=handles, loc="upper left",
                     bbox_to_anchor=(0.012, 0.988), fontsize=6.6,
                     title="trajectory cluster", labelspacing=0.42,
                     handlelength=1.9, borderpad=0.45, frameon=True,
                     framealpha=0.92, edgecolor="#dcd9d2", facecolor="white")
    leg.get_title().set_fontsize(6.9)
    leg.get_title().set_fontweight("bold")
    leg.get_frame().set_linewidth(0.6)
    leg.set_zorder(10)

    axm.text(0.985, 0.020, f"{sum(CLUSTER_SIZES)} traces, 07:00-09:00",
             transform=axm.transAxes, ha="right", va="bottom", fontsize=6.3,
             color=MUTED, zorder=9)

    # ------------------------------------------------- (b) traces per cluster
    ypos = np.arange(5)[::-1]
    counts = np.array([len(g) for g in tracks])
    axn.barh(ypos, counts, height=0.62, color=CATEGORICAL[:5],
             edgecolor="white", linewidth=0.5, zorder=3)
    for y, c in zip(ypos, counts):
        axn.text(c + 1.2, y, f"{c}", va="center", ha="left", fontsize=6.6,
                 color=INK)
    axn.set_yticks(ypos)
    axn.set_yticklabels([f"C{i + 1}" for i in range(5)], fontsize=6.8)
    axn.set_xlim(0, counts.max() * 1.20)
    axn.set_xlabel("trajectories (trips)", fontsize=7.4, labelpad=2)
    axn.set_title("Cluster membership", fontsize=8.6, fontweight="bold",
                  color=INK, pad=4)
    axn.tick_params(labelsize=6.6, length=2.4)
    axn.grid(axis="x", color="#e9e7e2", lw=0.55, zorder=0)

    # ------------------------------------------------ (c) mean travel time
    means = np.array([d.mean() for d in durations])
    sds = np.array([d.std(ddof=1) for d in durations])
    axd.barh(ypos, means, height=0.62, xerr=sds, color=CATEGORICAL[:5],
             edgecolor="white", linewidth=0.5, zorder=3,
             error_kw=dict(ecolor="#555555", elinewidth=0.8, capsize=2.0,
                           capthick=0.8))
    for y, m in zip(ypos, means):
        axd.text(1.5, y, f"{m:.0f}", va="center", ha="left", fontsize=6.6,
                 color="white", fontweight="bold")
    axd.set_yticks(ypos)
    axd.set_yticklabels([f"C{i + 1}" for i in range(5)], fontsize=6.8)
    axd.set_xlim(0, (means + sds).max() * 1.12)
    axd.set_xlabel("mean travel time (min)", fontsize=7.4, labelpad=2)
    axd.set_title("Travel time by cluster", fontsize=8.6, fontweight="bold",
                  color=INK, pad=4)
    axd.tick_params(labelsize=6.6, length=2.4)
    axd.grid(axis="x", color="#e9e7e2", lw=0.55, zorder=0)
    axd.text(0.0, -0.235, "bars: cluster mean, whiskers: 1 s.d.",
             transform=axd.transAxes, fontsize=6.3, color=MUTED,
             ha="left", va="top")

    panel_labels_inches(fig, [axm, axn, axd], dx=-0.20, dy=0.30)

    savefig(fig, "480_stm_trajectory_cluster.png")


if __name__ == "__main__":
    main()

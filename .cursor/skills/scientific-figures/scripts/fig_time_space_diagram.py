"""Time-space diagram of a two-signal arterial with stop-and-go shockwaves.

Vehicle trajectories come from Newell's car-following rule (free-flow
speed, saturation headway, jam spacing) constrained by two traffic
signals running a common 80 s cycle with a progression offset.  Line
colour is the instantaneous speed on the same level-of-service ramp used
by the rest of the series, so queue formation and discharge waves are
readable directly from the slope *and* the hue.  The congested envelope
is contoured and the two shockwave fronts are called out.  The short
strip underneath is the discharge flow measured at the downstream stop
line, kept to 30 % of the main panel's height.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter
from pubstyle import set_pub_style, savefig

LOS_RAMP = ["#a8433a", "#d1834b", "#d9b96a", "#7ba05b", "#3f7d5f"]
GREEN = "#5f8f6a"
RED = "#b5563c"
INK = "#2b2b2b"
GREY = "#8c8c8c"

L = 3000.0            # corridor length, m
VF = 13.9             # free-flow speed, m/s  (50 km/h)
TAU = 1.9             # reaction / saturation headway, s
DELTA = 7.2           # jam spacing, m
DT = 0.5
T_END = 700.0         # simulated horizon
T0, T1 = 180.0, 660.0  # displayed window (8 min, warm-up discarded)
GONE = 1e5            # sentinel for vehicles past the downstream end
CYCLE, GREENT = 80.0, 38.0
SIGNALS = [(1100.0, 6.0, "A"), (2300.0, 12.0, "B")]   # x, offset, name


def is_green(t, offset):
    return ((np.asarray(t) - offset) % CYCLE) < GREENT


def simulate(rng):
    head = rng.gamma(shape=2.5, scale=1.28, size=420) + 1.00
    t_in = np.cumsum(head)
    t_in = t_in[t_in < T_END - 20.0]
    n = t_in.size
    steps = int(T_END / DT) + 1
    ktau = int(round(TAU / DT))
    x = np.full((n, steps), -1e4)
    for k in range(1, steps):
        t = k * DT
        prev = x[:, k - 1]
        born = t_in <= t
        prev = np.where(born & (prev < 0), 0.0, prev)
        cand = prev + VF * DT
        lead = np.empty(n)
        lead[0] = GONE
        lead[1:] = x[:-1, max(0, k - ktau)]
        cand = np.minimum(cand, np.where(lead < -1e3, GONE, lead - DELTA))
        for xs, off, _ in SIGNALS:
            if not is_green(t, off):
                # a vehicle at or upstream of the stop line may not cross
                cand = np.where(prev <= xs + 1e-6, np.minimum(cand, xs), cand)
        cand = np.maximum(cand, prev)
        cand = np.where(cand > L, GONE, cand)
        x[:, k] = np.where(born, cand, -1e4)
    return t_in, x


def main():
    set_pub_style()
    rng = np.random.default_rng(11)
    t_in, x = simulate(rng)
    n, steps = x.shape
    tgrid = np.arange(steps) * DT

    cmap = mpl.colors.LinearSegmentedColormap.from_list("los", LOS_RAMP)
    norm = mpl.colors.Normalize(0.0, 50.0)

    fig = plt.figure(figsize=(10.2, 6.15))
    axt = fig.add_axes([0.062, 0.325, 0.845, 0.585])
    axf = fig.add_axes([0.062, 0.085, 0.845, 0.170])
    cax = fig.add_axes([0.922, 0.325, 0.013, 0.585])

    # ---------------------------------------------------- (a) trajectories
    step = 4                                   # 2 s plotting resolution
    segs, spd = [], []
    for i in range(n):
        xi = x[i, ::step]
        ti = tgrid[::step]
        ok = (xi > -1e3) & (xi < L + 1) & (ti >= T0 - 2) & (ti <= T1 + 2)
        xi, ti = xi[ok], ti[ok]
        if xi.size < 3:
            continue
        pts = np.column_stack([(ti - T0) / 60.0, xi / 1000.0])
        segs.append(np.stack([pts[:-1], pts[1:]], axis=1))
        v = np.diff(xi) / np.diff(ti) * 3.6
        spd.append(v)
    segs = np.concatenate(segs)
    spd = np.concatenate(spd)
    lc = LineCollection(segs, array=spd, cmap=cmap, norm=norm, linewidths=0.55,
                        capstyle="round", zorder=3, alpha=0.95)
    axt.add_collection(lc)

    # ---- congested envelope (speed < 12 km/h) ---------------------------
    nx, nt = 150, 240
    xe = np.linspace(0, L, nx + 1)
    te = np.linspace(T0, T1, nt + 1)
    grid = np.zeros((nx, nt))
    v_all = np.diff(x, axis=1) / DT * 3.6
    xm = x[:, :-1]
    tm = tgrid[:-1]
    slow = ((v_all < 12.0) & (xm > -1e3) & (xm < L + 1)
            & (tm >= T0)[None, :] & (tm <= T1)[None, :])
    ix = np.clip((np.clip(xm, 0, L) / L * nx).astype(int), 0, nx - 1)
    it = np.clip(((tm - T0) / (T1 - T0) * nt).astype(int), 0, nt - 1)
    for i in range(n):
        s = slow[i]
        if s.any():
            np.add.at(grid, (ix[i][s], it[s]), 1.0)
    grid = gaussian_filter(grid, 1.6)
    TE, XE = np.meshgrid((0.5 * (te[:-1] + te[1:]) - T0) / 60.0,
                         0.5 * (xe[:-1] + xe[1:]) / 1000.0)
    axt.contourf(TE, XE, grid, levels=[0.55, 1e9], colors=["#a8433a"],
                 alpha=0.13, zorder=2)
    axt.contour(TE, XE, grid, levels=[0.55], colors=["#a8433a"],
                linewidths=0.8, alpha=0.55, zorder=4)

    # ---- signal indications ---------------------------------------------
    band = 0.070
    for xs, off, name in SIGNALS:
        t0 = off + CYCLE * np.floor((T0 - off) / CYCLE)
        while t0 < T1:
            for a, b, col in [(t0, t0 + GREENT, GREEN),
                              (t0 + GREENT, t0 + CYCLE, RED)]:
                a, b = max(a, T0), min(b, T1)
                if b > a:
                    axt.add_patch(Rectangle(
                        ((a - T0) / 60.0, xs / 1000.0 - band / 2),
                        (b - a) / 60.0, band, facecolor=col,
                        edgecolor="white", linewidth=0.45, zorder=6))
            t0 += CYCLE
        axt.text(0.045, xs / 1000.0 + band / 2 + 0.035, f"signal {name}",
                 ha="left", va="bottom", fontsize=7.2, fontweight="bold",
                 color=INK, zorder=7,
                 bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none",
                           alpha=0.86))

    axt.set_xlim(0, (T1 - T0) / 60.0)
    axt.set_ylim(0, L / 1000.0)
    axt.set_ylabel("distance along corridor  (km)", labelpad=3)
    axt.set_xticks(np.arange(0, 9))
    axt.set_xticklabels([])
    axt.set_yticks(np.arange(0, 3.1, 0.5))
    axt.grid(axis="x", color="#ececec", lw=0.5, zorder=0)
    axt.tick_params(length=2.6)
    for sp in ("top", "right"):
        axt.spines[sp].set_visible(False)

    # shockwave call-outs
    box = dict(boxstyle="round,pad=0.22", fc="white", ec="#e2d6d3", lw=0.6,
               alpha=0.92)
    axt.annotate("stopping wave\n(queue growth)", xy=(2.44, 0.88),
                 xytext=(0.90, 0.30), fontsize=7.0, color="#8a3a30",
                 fontweight="bold", ha="left", va="center", linespacing=1.25,
                 zorder=8, bbox=box,
                 arrowprops=dict(arrowstyle="-|>", lw=0.75, color="#a8433a",
                                 shrinkA=4, shrinkB=3,
                                 connectionstyle="arc3,rad=0.20"))
    axt.annotate("discharge wave\n(queue release)", xy=(3.24, 0.82),
                 xytext=(4.15, 0.30), fontsize=7.0, color="#8a3a30",
                 fontweight="bold", ha="left", va="center", linespacing=1.25,
                 zorder=8, bbox=box,
                 arrowprops=dict(arrowstyle="-|>", lw=0.75, color="#a8433a",
                                 shrinkA=4, shrinkB=3,
                                 connectionstyle="arc3,rad=-0.20"))
    axt.annotate("progression offset lets most\nof the platoon clear B "
                 "in one green", xy=(4.30, 2.30), xytext=(4.70, 2.72),
                 fontsize=7.0, color="#3d6b52", fontweight="bold", ha="left",
                 va="center", linespacing=1.25, zorder=8,
                 bbox=dict(boxstyle="round,pad=0.22", fc="white",
                           ec="#d6e0d6", lw=0.6, alpha=0.92),
                 arrowprops=dict(arrowstyle="-|>", lw=0.75, color="#5f8f6a",
                                 shrinkA=4, shrinkB=5,
                                 connectionstyle="arc3,rad=0.22"))

    cb = fig.colorbar(lc, cax=cax, ticks=[0, 10, 20, 30, 40, 50])
    cb.set_label("instantaneous speed  (km/h)", fontsize=7.2, labelpad=3)
    cb.ax.tick_params(labelsize=6.8, length=2.2, width=0.7)
    cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor("#555555")

    # ------------------------------------------------ (b) discharge flow
    xs_b = SIGNALS[1][0]
    cross = []
    for i in range(n):
        xi = x[i]
        idx = np.where((xi[:-1] < xs_b) & (xi[1:] >= xs_b)
                       & (xi[1:] < L + 1))[0]
        if idx.size:
            cross.append(tgrid[idx[0]])
    cross = np.array(cross)
    bw = 20.0
    edges = np.arange(T0, T1 + bw, bw)
    cnt, _ = np.histogram(cross, bins=edges)
    flow = cnt * 3600.0 / bw
    ctr = (0.5 * (edges[:-1] + edges[1:]) - T0) / 60.0

    off_b = SIGNALS[1][1]
    t0 = off_b + CYCLE * np.floor((T0 - off_b) / CYCLE)
    while t0 < T1:
        g0, g1 = max(t0, T0), min(t0 + GREENT, T1)
        if g1 > g0:
            axf.axvspan((g0 - T0) / 60.0, (g1 - T0) / 60.0, color=GREEN,
                        alpha=0.10, lw=0, zorder=0)
        t0 += CYCLE
    axf.bar(ctr, flow, width=bw / 60.0 * 0.88, color="#4a6f8a", alpha=0.85,
            edgecolor="white", linewidth=0.35, zorder=3)
    qbar = flow.mean()
    axf.axhline(qbar, color=INK, lw=0.9, ls=(0, (4, 2.4)), zorder=4)
    axf.text((T1 - T0) / 60.0 - 0.06, qbar + 90,
             f"mean {qbar:.0f} veh/h", ha="right", va="bottom", fontsize=6.8,
             fontweight="bold", color=INK, zorder=5)
    axf.set_xlim(0, (T1 - T0) / 60.0)
    axf.set_ylim(0, max(flow.max() * 1.32, 1000))
    axf.set_xticks(np.arange(0, 9))
    axf.set_xlabel("time  (min)", labelpad=2)
    axf.set_ylabel("discharge\n(veh/h)", labelpad=3, fontsize=7.6,
                   linespacing=1.25)
    axf.tick_params(length=2.6, labelsize=7.0)
    axf.grid(axis="y", color="#ececec", lw=0.5, zorder=0)
    for sp in ("top", "right"):
        axf.spines[sp].set_visible(False)
    axf.text(0.012, 0.93, "green at signal B", transform=axf.transAxes,
             fontsize=6.5, color="#4c7a5c", fontweight="bold", va="top")

    # ---- shared panel tags ----------------------------------------------
    for ax, tag, ttl in [(axt, "(a)", "Vehicle trajectories on a "
                          "signalised arterial"),
                         (axf, "(b)", "Stop-line discharge, 20 s bins")]:
        pos = ax.get_position()
        fig.text(pos.x0 - 0.46 / fig.get_figwidth(),
                 pos.y1 + 0.115 / fig.get_figheight(), tag, fontsize=10.5,
                 fontweight="bold", ha="left", va="baseline", color=INK)
        fig.text(pos.x0 - 0.10 / fig.get_figwidth(),
                 pos.y1 + 0.115 / fig.get_figheight(), ttl, fontsize=8.8,
                 fontweight="bold", ha="left", va="baseline", color="#33474f")

    fig.text(0.907, 0.910 + 0.115 / fig.get_figheight(),
             "Newell car-following, 80 s cycle / 38 s green, 12 s offset",
             ha="right", va="baseline", fontsize=6.6, color="#9a9a9a")
    savefig(fig, "206_time_space_diagram.png")


if __name__ == "__main__":
    main()

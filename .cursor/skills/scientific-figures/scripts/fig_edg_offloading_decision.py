"""Figure 642 - Where a task should run: tiers, decision map, Pareto front.

One cost model drives all three panels.  Panel (a) draws the device / edge /
cloud tiers with the compute and link capacities the model uses; panel (b)
evaluates that same model over a task-size by bandwidth grid and fills the
region each tier wins; panel (c) fixes one operating point and sweeps the
split of a batch across the three tiers, so every marker in the scatter is a
policy the model in (a) and (b) could actually be asked to choose.

Reference-gallery figure: the numbers are generated, the layout is the point.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from pubstyle import set_pub_style, savefig, COLORS
from figkit import (block, arrow, panel_card, panel_label_inches,
                    use_inch_units)

DEV = "#3b5b92"        # house blue   - terminal tier
EDG = "#2e7d5b"        # house green  - edge tier
CLD = "#8e6bbf"        # house purple - cloud tier
ACCENT = "#b23a48"     # house red    - the single emphasis (Pareto front)

# ---- one cost model, shared by every panel --------------------------------
F_DEV, F_EDG, F_CLD = 8.0, 320.0, 12000.0     # GFLOPS
P_DEV, P_TX, P_IDLE = 2.40, 0.72, 0.21        # W  (compute / radio / idle)
INTENSITY = 3.0                                # GFLOP per MB of input
BW_BACKHAUL = 2500.0                           # Mbps, edge -> cloud fibre
RTT_EDG, RTT_CLD = 4.0, 28.0                   # ms
BW_LO, BW_HI = 2.0, 1000.0                     # Mbps sweep of panel (b)
SZ_LO, SZ_HI = 0.05, 50.0                      # MB sweep of panel (b)
OP_SIZE, OP_BW = 1.5, 12.0                     # the operating point of (c)


def latency_ms(size_mb, bw_mbps):
    """Return (local, edge, cloud) end-to-end latency in ms."""
    up = 8.0 * size_mb / bw_mbps * 1e3
    t_loc = INTENSITY * size_mb / F_DEV * 1e3
    t_edg = up + INTENSITY * size_mb / F_EDG * 1e3 + RTT_EDG
    t_cld = (up + 8.0 * size_mb / BW_BACKHAUL * 1e3 +
             INTENSITY * size_mb / F_CLD * 1e3 + RTT_CLD)
    return t_loc, t_edg, t_cld


def energy_mj(size_mb, bw_mbps):
    """Device-side energy in mJ for the same three placements."""
    up = 8.0 * size_mb / bw_mbps
    t_loc, t_edg, t_cld = (v / 1e3 for v in latency_ms(size_mb, bw_mbps))
    return (P_DEV * t_loc * 1e3,
            (P_TX * up + P_IDLE * (t_edg - up)) * 1e3,
            (P_TX * up + P_IDLE * (t_cld - up)) * 1e3)


def pareto_mask(x, y):
    """True where no other point is better or equal on both axes."""
    keep = np.ones(x.size, dtype=bool)
    for i in range(x.size):
        keep[i] = not np.any((x <= x[i]) & (y <= y[i]) &
                             ((x < x[i]) | (y < y[i])))
    return keep


def tiers(fig, ax):
    w, h = use_inch_units(ax)
    y_dev, y_edg, y_cld = 0.145 * h, 0.470 * h, 0.815 * h

    for y, hh, lab in ((y_dev, 0.215 * h, "terminal tier"),
                       (y_edg, 0.195 * h, "edge tier"),
                       (y_cld, 0.190 * h, "cloud tier")):
        panel_card(ax, 0.50 * w, y, 0.96 * w, hh, fc="#f5f5f5", ec="#e2e2e2",
                   lw=0.7, zorder=1, rounding=0.06, label=lab, label_fs=6.2)

    devs = []
    for fx, name, k in ((0.185, "IoT sensor", 0.25),
                        (0.500, "smart camera", 1.00),
                        (0.815, "mobile AGV", 0.60)):
        devs.append(block(ax, fx * w, y_dev, 0.255 * w, 0.105 * h, name,
                          color=DEV, sub="%.1f GFLOPS" % (k * F_DEV),
                          fontsize=6.8, sub_color="#dfe6f2", zorder=4,
                          rounding=0.045))

    edges = []
    for fx, name in ((0.290, "edge node A"), (0.710, "edge node B")):
        edges.append(block(ax, fx * w, y_edg, 0.335 * w, 0.105 * h, name,
                           color=EDG, sub="%.0f GFLOPS / 32 GB" % F_EDG,
                           fontsize=6.8, sub_color="#dcece4", zorder=4,
                           rounding=0.045))

    cloud = block(ax, 0.50 * w, y_cld, 0.560 * w, 0.108 * h, "regional cloud",
                  color=CLD, sub="%.0f TFLOPS / elastic" % (F_CLD / 1e3),
                  fontsize=6.8, sub_color="#e8e0f4", zorder=4, rounding=0.045)

    for d, e in ((devs[0], edges[0]), (devs[1], edges[0]),
                 (devs[1], edges[1]), (devs[2], edges[1])):
        arrow(ax, d["t"], e["b"], color="#6b7d8f", lw=0.9, ms=8, gap=0.055,
              zorder=5)
    for e in edges:
        arrow(ax, e["t"], cloud["b"], color="#6b7d8f", lw=1.15, ms=9,
              gap=0.055, zorder=5)

    ax.text(0.015 * w, 0.5 * (y_dev + y_edg),
            "access uplink\n%.0f-%.0f Mbps\n%.0f ms RTT"
            % (BW_LO, BW_HI, RTT_EDG), ha="left", va="center", fontsize=6.2,
            color=COLORS["sub"], linespacing=1.4, zorder=7)
    ax.text(0.985 * w, 0.5 * (y_edg + y_cld), "backhaul\n%.1f Gbps\n%.0f ms"
            " RTT" % (BW_BACKHAUL / 1e3, RTT_CLD), ha="right", va="center",
            fontsize=6.2, color=COLORS["sub"], linespacing=1.4, zorder=7)
    ax.set_title("Three-tier offloading substrate", fontsize=8.8,
                 fontweight="bold", pad=4.0)


def main():
    set_pub_style()
    rng = np.random.default_rng(11)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.0, 0.58],
                          left=0.055, right=0.965, top=0.870, bottom=0.150,
                          wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    tiers(fig, ax_a)

    # ---- (b) which tier wins, over the size x bandwidth plane -------------
    sizes = np.logspace(np.log10(SZ_LO), np.log10(SZ_HI), 260)
    bws = np.logspace(np.log10(BW_LO), np.log10(BW_HI), 260)
    S, B = np.meshgrid(sizes, bws)
    stack = np.stack(latency_ms(S, B))
    winner = np.argmin(stack, axis=0).astype(float)
    best = np.min(stack, axis=0)

    tint = ListedColormap(["#c9d5e8", "#c4dccf", "#dbd0ec"])
    ax_b.contourf(S, B, winner, levels=[-0.5, 0.5, 1.5, 2.5], cmap=tint,
                  zorder=1)
    ax_b.contour(S, B, winner, levels=[0.5, 1.5], colors="#333333",
                 linewidths=0.9, zorder=3)
    cs = ax_b.contour(S, B, best, levels=[10, 30, 100, 300, 1000, 3000],
                      colors="#5a5a5a", linewidths=0.5,
                      linestyles="dashed", zorder=4)
    ax_b.clabel(cs, fmt="%.0f ms", fontsize=6.0, inline=True,
                inline_spacing=3)

    def loc(fs, fb):
        return (10 ** (np.log10(SZ_LO) + fs * np.log10(SZ_HI / SZ_LO)),
                10 ** (np.log10(BW_LO) + fb * np.log10(BW_HI / BW_LO)))

    for fs, fb, lab, col, ha in ((0.50, 0.055, "run locally", DEV, "center"),
                                 (0.045, 0.88, "offload to edge", EDG, "left"),
                                 (0.965, 0.88, "offload to cloud", CLD,
                                  "right")):
        x, y = loc(fs, fb)
        ax_b.text(x, y, lab, fontsize=7.2, fontweight="bold", color=col,
                  ha=ha, va="center", zorder=6)

    ax_b.plot([OP_SIZE], [OP_BW], marker="o", ms=5.4, mfc="white",
              mec=ACCENT, mew=1.4, zorder=7)
    ax_b.annotate("operating point of (c)", xy=(OP_SIZE, OP_BW),
                  xytext=(0.075, 4.2), fontsize=6.4, color=ACCENT,
                  ha="left", va="center", zorder=7,
                  arrowprops=dict(arrowstyle="-", lw=0.7, color=ACCENT))
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlim(sizes[0], sizes[-1])
    ax_b.set_ylim(bws[0], bws[-1])
    ax_b.set_xlabel("task input size (MB)", fontsize=8.0)
    ax_b.set_ylabel("available uplink bandwidth (Mbps)", fontsize=8.0)
    ax_b.tick_params(labelsize=7.0)
    for s in ("top", "right"):
        ax_b.spines[s].set_visible(False)
    ax_b.set_title("Decision regions and iso-latency contours", fontsize=8.8,
                   fontweight="bold", pad=4.0)

    # ---- (c) latency / energy trade-off at one operating point ------------
    weights = np.vstack([rng.dirichlet(np.full(3, 0.72), size=140),
                         np.eye(3)])                       # + the three pure
    dvfs = np.concatenate([rng.uniform(0.55, 1.0, size=140), np.ones(3)])
    lat_v = np.array(latency_ms(OP_SIZE, OP_BW))
    eng_v = np.array(energy_mj(OP_SIZE, OP_BW))
    # Lowering the device clock stretches local compute by 1/d but pays only
    # d times the energy, so the local corner is a curve rather than a point.
    lat = (weights[:, 0] * lat_v[0] / dvfs + weights[:, 1:] @ lat_v[1:])
    eng = (weights[:, 0] * eng_v[0] * dvfs + weights[:, 1:] @ eng_v[1:])
    front = pareto_mask(lat, eng)
    order = np.argsort(lat[front])

    ax_c.scatter(lat[~front], eng[~front], s=13, c="#b8c0cb",
                 edgecolors="white", linewidths=0.35, zorder=3,
                 label="dominated policy")
    ax_c.plot(lat[front][order], eng[front][order], ls=(0, (3.0, 1.8)),
              lw=1.25, color=ACCENT, zorder=4, label="Pareto front")
    ax_c.scatter(lat[front], eng[front], s=20, c=ACCENT, edgecolors="white",
                 linewidths=0.45, zorder=5)

    for k, (name, col, off, ha) in enumerate(
            (("all local", DEV, (5.0, -1.0), "left"),
             ("all edge", EDG, (-5.0, 5.5), "right"),
             ("all cloud", CLD, (-5.0, -8.0), "right"))):
        ax_c.plot([lat_v[k]], [eng_v[k]], marker="s", ms=5.0, mfc=col,
                  mec="white", mew=0.8, zorder=6)
        ax_c.annotate(name, xy=(lat_v[k], eng_v[k]), xytext=off,
                      textcoords="offset points", fontsize=6.4, color=col,
                      fontweight="bold", ha=ha, zorder=6)

    ax_c.margins(x=0.12, y=0.10)
    ax_c.set_xlabel("batch latency (ms)", fontsize=8.0)
    ax_c.set_ylabel("device energy (mJ)", fontsize=8.0)
    ax_c.tick_params(labelsize=6.8)
    ax_c.grid(lw=0.45, alpha=0.45)
    ax_c.set_axisbelow(True)
    for s in ("top", "right"):
        ax_c.spines[s].set_visible(False)
    ax_c.legend(fontsize=6.4, frameon=False, loc="upper right",
                handlelength=1.6, borderpad=0.2)
    ax_c.set_title("%d non-dominated policies" % int(front.sum()),
                   fontsize=8.5, fontweight="bold", pad=3.5)

    for a, lab, dx in ((ax_a, "(a)", -0.12), (ax_b, "(b)", -0.40),
                       (ax_c, "(c)", -0.40)):
        panel_label_inches(fig, a, lab, dx=dx, dy=0.09)

    saving = 1.0 - eng_v[1] / eng_v[0]
    slower = lat_v[1] / lat_v[0]

    savefig(fig, "642_edg_offloading_decision.png")


if __name__ == "__main__":
    main()

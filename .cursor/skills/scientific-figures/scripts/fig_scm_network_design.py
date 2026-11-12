"""Three-echelon supply-chain network design on a synthetic service region.

The map carries the decision: plants (squares), distribution centres
(triangles) and customer zones (dots), with filled markers for the sites the
siting model opened and hollow dashed markers for the candidates it rejected.
Flow arcs are width-scaled by shipped volume, so the trunk lanes read as
trunk lanes without a colour ramp.  The two auxiliary panels are held to
~58 % of the map width: the facility / transport cost trade-off that picks
the number of open DCs, and the throughput each open DC actually carries.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig
from figkit import add_scale_bar, panel_labels_inches

BLUE = "#3b5b92"        # plants / facility cost
GREEN = "#2e7d5b"       # distribution centres / chosen option
RED = "#b23a48"         # rejected candidates, emphasis
GOLD = "#c9a227"        # transport cost
INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
LAND = "#fbfaf8"

EXT_X, EXT_Y = 100.0, 88.0


def arc(p0, p1, rad=0.16, n=48):
    """Gently bowed connector between two map points."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    ctrl = 0.5 * (p0 + p1) + rad * np.array([-d[1], d[0]])
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1


def build(rng):
    plants = np.array([[12.0, 68.0], [16.0, 22.0], [78.0, 74.0]])
    plant_open = np.array([True, True, False])
    dcs = np.array([[34.0, 63.0], [30.0, 30.0], [58.0, 47.0],
                    [72.0, 22.0], [80.0, 60.0], [52.0, 76.0]])
    dc_open = np.array([True, True, True, True, False, False])

    cust = np.column_stack([rng.uniform(20.0, 95.0, 34),
                            rng.uniform(8.0, 82.0, 34)])
    demand = rng.gamma(3.6, 26.0, cust.shape[0]) + 45.0

    open_idx = np.where(dc_open)[0]
    d = np.linalg.norm(cust[:, None, :] - dcs[None, open_idx, :], axis=2)
    assign = open_idx[np.argmin(d, axis=1)]
    return plants, plant_open, dcs, dc_open, cust, demand, assign


def cost_curve(n_dc):
    """Facility, transport and total cost per week for n open DCs."""
    facility = 34.0 + 21.5 * n_dc
    transport = 46.0 + 268.0 / (n_dc ** 0.92)
    return facility, transport, facility + transport


def main():
    set_pub_style()
    rng = np.random.default_rng(558)
    plants, plant_open, dcs, dc_open, cust, demand, assign = build(rng)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.90],
                          wspace=0.20, hspace=0.42)
    axm = fig.add_subplot(gs[:, 0])
    axc = fig.add_subplot(gs[0, 1])
    axu = fig.add_subplot(gs[1, 1])

    # ------------------------------------------------------------------ map
    axm.set_facecolor(LAND)
    axm.set_xlim(0, EXT_X)
    axm.set_ylim(0, EXT_Y)
    axm.set_aspect("equal")
    axm.set_xticks([])
    axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
        sp.set_edgecolor("#c9c6bf")

    # secondary arcs: open DC -> customer zone, width by zone demand
    dmax = demand.max()
    for c, dem, k in zip(cust, demand, assign):
        p = arc(dcs[k], c, rad=0.10)
        axm.plot(p[:, 0], p[:, 1], color=GREEN, alpha=0.34,
                 lw=0.35 + 1.05 * dem / dmax, solid_capstyle="round",
                 zorder=2)

    # trunk arcs: open plant -> open DC, width by shipped volume
    dc_load = np.array([demand[assign == j].sum() for j in range(len(dcs))])
    trunk = []
    for j in np.where(dc_open)[0]:
        src = 0 if dcs[j][1] >= 45.0 else 1
        trunk.append((src, j, dc_load[j]))
    tmax = max(t[2] for t in trunk)
    for src, j, vol in trunk:
        p = arc(plants[src], dcs[j], rad=0.13)
        axm.plot(p[:, 0], p[:, 1], color=BLUE, alpha=0.80,
                 lw=0.9 + 3.3 * vol / tmax, solid_capstyle="round", zorder=3)
        q = p[len(p) // 2]
        axm.text(q[0], q[1] - 2.6, f"{vol:.0f}", fontsize=6.2,
                 color=BLUE, ha="center", va="top", zorder=6,
                 bbox=dict(boxstyle="round,pad=0.14", fc="white",
                           ec="none", alpha=0.88))

    axm.scatter(cust[:, 0], cust[:, 1], s=8.0 + 26.0 * demand / dmax,
                facecolor="#8a8a8a", edgecolor="white", linewidth=0.5,
                zorder=5)

    for p, is_open in zip(plants, plant_open):
        if is_open:
            axm.plot(p[0], p[1], marker="s", ms=10.5, mfc=BLUE, mec="white",
                     mew=1.0, zorder=7)
        else:
            axm.plot(p[0], p[1], marker="s", ms=10.5, mfc="none", mec=RED,
                     mew=1.0, zorder=7)
    for p, is_open in zip(dcs, dc_open):
        if is_open:
            axm.plot(p[0], p[1], marker="^", ms=10.0, mfc=GREEN, mec="white",
                     mew=1.0, zorder=7)
        else:
            axm.plot(p[0], p[1], marker="^", ms=10.0, mfc="none", mec=RED,
                     mew=1.0, zorder=7)

    for k, p in enumerate(plants):
        axm.text(p[0], p[1] - 4.2, f"P{k + 1}", fontsize=6.6, color=INK,
                 ha="center", va="top", zorder=8)
    for k, p in enumerate(dcs):
        axm.text(p[0] + 3.0, p[1] + 1.2, f"DC{k + 1}", fontsize=6.6,
                 color=INK if dc_open[k] else MUTED, ha="left", va="bottom",
                 zorder=8)

    add_scale_bar(axm, 20.0, "20 km", loc=(0.035, 0.045), height=0.017,
                  color="#3a3a3a", alt="white", fontsize=6.2,
                  text_color=INK, halo="white")

    handles = [
        Line2D([], [], marker="s", ms=7.0, mfc=BLUE, mec="white", ls="none",
               label="plant (open)"),
        Line2D([], [], marker="^", ms=7.0, mfc=GREEN, mec="white", ls="none",
               label="DC (open)"),
        Line2D([], [], marker="^", ms=7.0, mfc="none", mec=RED, ls="none",
               label="candidate (not opened)"),
        Line2D([], [], marker="o", ms=4.5, mfc="#8a8a8a", mec="white",
               ls="none", label="customer zone (size = demand)"),
        Line2D([], [], color=BLUE, lw=2.4, label="trunk flow (units/week)"),
        Line2D([], [], color=GREEN, lw=1.0, alpha=0.6,
               label="delivery flow (units/week)"),
    ]
    leg = axm.legend(handles=handles, loc="upper left",
                     bbox_to_anchor=(0.005, 0.995), fontsize=6.6,
                     labelspacing=0.42, handlelength=1.8, borderpad=0.45,
                     frameon=True, framealpha=0.92, facecolor="white",
                     edgecolor="#cccccc")
    leg.get_frame().set_linewidth(0.6)
    leg.set_zorder(10)
    axm.set_title("Opened sites and shipped volume", fontsize=9.0,
                  fontweight="bold", color=INK, pad=5)

    # ------------------------------------------------- (b) cost trade-off
    n = np.arange(1, 7)
    fac, tra, tot = cost_curve(n)
    best = int(n[np.argmin(tot)])
    axc.plot(n, fac, color=BLUE, lw=1.5, marker="o", ms=3.4,
             label="facility cost")
    axc.plot(n, tra, color=GOLD, lw=1.5, marker="s", ms=3.2,
             label="transport cost")
    axc.plot(n, tot, color=RED, lw=1.9, marker="D", ms=3.6,
             label="total cost")
    axc.axvline(best, color="#808080", lw=0.8, ls="--", zorder=1)
    axc.annotate(f"optimum: {best} DCs\n{tot.min():.0f} kUSD/week",
                 xy=(best, tot.min()), xytext=(best + 0.35, tot.min() - 88),
                 fontsize=6.6, color=INK, ha="left", va="bottom",
                 linespacing=1.25,
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=RULE,
                                 shrinkA=1, shrinkB=3))
    axc.set_xlabel("distribution centres opened (count)", fontsize=8.0)
    axc.set_ylabel("weekly cost (kUSD)", fontsize=8.0)
    axc.set_xticks(n)
    axc.set_ylim(0, 420)
    axc.tick_params(labelsize=6.8)
    axc.legend(fontsize=6.6, frameon=False, loc="upper right",
               labelspacing=0.35, handlelength=1.7)
    axc.grid(axis="y", color="#e6e4e0", lw=0.55)
    axc.set_axisbelow(True)
    for sp in ("top", "right"):
        axc.spines[sp].set_visible(False)
    axc.set_title("Cost trade-off over network size", fontsize=8.5,
                  fontweight="bold", color=INK, pad=4)

    # ------------------------------------------------- (c) DC utilisation
    oidx = np.where(dc_open)[0]
    load = dc_load[oidx]
    # Design capacities are sized to the assigned load at the target
    # utilisation band, then rounded up to the nearest 50-unit build step.
    target = np.array([0.78, 0.86, 0.81, 0.90])
    cap = np.ceil(load / target / 50.0) * 50.0
    names = [f"DC{j + 1}" for j in oidx]
    ypos = np.arange(len(oidx))[::-1]
    axu.barh(ypos, cap, height=0.56, color="#e2e0dc", edgecolor="white",
             linewidth=0.5, label="capacity", zorder=2)
    axu.barh(ypos, load, height=0.56, color=GREEN, edgecolor="white",
             linewidth=0.5, label="assigned load", zorder=3)
    for y, lo, ca in zip(ypos, load, cap):
        axu.text(ca + 40.0, y, f"{100.0 * lo / ca:.0f}%", fontsize=6.6,
                 color=INK, va="center", ha="left", zorder=4)
    axu.set_yticks(ypos)
    axu.set_yticklabels(names, fontsize=6.8)
    axu.set_xlim(0, cap.max() * 1.55)
    axu.set_xlabel("throughput (units/week)", fontsize=8.0)
    axu.tick_params(labelsize=6.8)
    axu.legend(fontsize=6.6, frameon=False, loc="lower right",
               labelspacing=0.32, handlelength=1.5)
    axu.grid(axis="x", color="#e6e4e0", lw=0.55)
    axu.set_axisbelow(True)
    for sp in ("top", "right"):
        axu.spines[sp].set_visible(False)
    axu.set_title("Utilisation of the open DCs", fontsize=8.5,
                  fontweight="bold", color=INK, pad=4)

    saving = 100.0 * (tot[0] - tot.min()) / tot[0]
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.6, w_pad=2.6)
    for ax in (axm, axc, axu):
        ax.apply_aspect()
    panel_labels_inches(fig, [axm, axc, axu], dx=-0.20, dy=0.16)
    savefig(fig, "558_scm_network_design.png")


if __name__ == "__main__":
    main()

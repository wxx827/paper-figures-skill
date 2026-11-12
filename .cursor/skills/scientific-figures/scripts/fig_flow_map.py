"""Origin-destination flow map with Bezier arcs over a synthetic country.

Node positions are regional centres sized by total throughput; curved
FancyArrowPatch arcs encode inter-regional flows generated from a gravity
model, with width proportional to volume and hue encoding the dominant
direction. A side panel ranks nodes by net balance (inflow - outflow).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Polygon
from pubstyle import set_pub_style, savefig

EAST = "#2f6d8c"      # net eastward / northward flows
WEST = "#bf6640"      # net westward / southward flows
NAMES = ["ARD", "BRN", "CTL", "DVN", "ELM", "FRT", "GLN",
         "HRT", "IVR", "JUN", "KLD", "LTN", "MRS", "NWP"]


def land_radius(theta):
    return (1.0 + 0.15 * np.cos(2 * theta + 0.3)
            + 0.10 * np.cos(3 * theta + 2.0)
            + 0.04 * np.cos(5 * theta - 1.1))


def poisson_nodes(rng, n, rmin=0.34):
    pts = []
    tries = 0
    while len(pts) < n and tries < 20000:
        tries += 1
        p = rng.uniform(-1.15, 1.15, size=2)
        if np.hypot(*p) > 0.86 * land_radius(np.arctan2(p[1], p[0])):
            continue
        if all(np.hypot(*(p - q)) > rmin for q in pts):
            pts.append(p)
    return np.array(pts)


def main():
    set_pub_style()
    rng = np.random.default_rng(23)
    xy = poisson_nodes(rng, len(NAMES))
    n = len(xy)
    t = np.linspace(0, 2 * np.pi, 500)
    coast = np.c_[np.cos(t), np.sin(t)] * land_radius(t)[:, None]

    mass = rng.lognormal(0.0, 0.55, n)
    mass = 1.0 + 6.0 * (mass - mass.min()) / np.ptp(mass)

    d = np.hypot(xy[:, None, 0] - xy[None, :, 0],
                 xy[:, None, 1] - xy[None, :, 1])
    np.fill_diagonal(d, np.inf)
    grav = (mass[:, None] * mass[None, :]) / d ** 1.6
    grav *= rng.lognormal(0, 0.45, (n, n))
    asym = rng.uniform(0.25, 0.75, (n, n))
    flow = grav * asym                       # directed i -> j

    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    pairs.sort(key=lambda p: flow[p], reverse=True)
    keep = pairs[:28]
    linked = {i for p in keep for i in p}
    for i in range(n):                      # keep every hub connected
        if i not in linked:
            keep.append(max(((i, j) for j in range(n) if j != i),
                            key=lambda p: flow[p]))
    fmax = max(flow[p] for p in keep)

    inflow = np.zeros(n); outflow = np.zeros(n)
    for i, j in keep:
        outflow[i] += flow[i, j]; inflow[j] += flow[i, j]
    total = inflow + outflow
    net = inflow - outflow

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = GridSpec(1, 2, width_ratios=[2.6, 1], left=0.012, right=0.965,
                  top=0.885, bottom=0.075, wspace=0.10)

    ax = fig.add_subplot(gs[0, 0])
    ax.add_patch(Polygon(coast, closed=True, facecolor="#f2f0ea",
                         edgecolor="#3a3a3a", lw=1.0, zorder=0))
    for i, j in sorted(keep, key=lambda p: flow[p]):
        w = flow[i, j] / fmax
        east = (xy[j, 0] - xy[i, 0]) >= 0
        col = EAST if east else WEST
        rad = 0.18 if east else -0.18
        arc = FancyArrowPatch(tuple(xy[i]), tuple(xy[j]),
                              connectionstyle="arc3,rad=%.2f" % rad,
                              arrowstyle="-|>", mutation_scale=6 + 7 * w,
                              lw=0.5 + 3.4 * w, color=col,
                              alpha=0.28 + 0.55 * w,
                              shrinkA=4 + 6 * mass[i] / mass.max(),
                              shrinkB=5 + 6 * mass[j] / mass.max(),
                              zorder=2, capstyle="round")
        ax.add_patch(arc)

    s = 28 + 340 * (total / total.max())
    ax.scatter(xy[:, 0], xy[:, 1], s=s, facecolor="#3f4a52",
               edgecolor="white", linewidth=0.9, zorder=4)
    for k, (x0, y0) in enumerate(xy):
        ax.text(x0, y0 - 0.085 - 0.00025 * s[k], NAMES[k], ha="center",
                va="top", fontsize=6.6, fontweight="bold", color="#22282d",
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none",
                          alpha=0.72))

    ax.set_aspect("equal")
    ax.set_xlim(coast[:, 0].min() - 0.08, coast[:, 0].max() + 0.08)
    ax.set_ylim(coast[:, 1].min() - 0.10, coast[:, 1].max() + 0.08)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.0, 1.005, "(a)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom")
    ax.text(0.055, 1.008, "Inter-regional commodity flows (leading corridors)",
            transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            va="bottom", color="#2f4a58")
    ax.annotate("N", xy=(0.955, 0.94), xytext=(0.955, 0.85),
                xycoords="axes fraction", ha="center", fontsize=7, color="#333",
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1))
    ax.plot([0.80, 0.955], [0.045, 0.045], transform=ax.transAxes,
            color="#333", lw=1.8, solid_capstyle="butt")
    ax.text(0.877, 0.062, "250 km", transform=ax.transAxes, fontsize=6,
            ha="center", color="#333")

    # ---- width / colour legend
    hl = [Line2D([], [], color=EAST, lw=0.7, label="0.2 (low)"),
          Line2D([], [], color=EAST, lw=2.0, label="0.6"),
          Line2D([], [], color=EAST, lw=3.6, label="1.0 (max)")]
    leg1 = ax.legend(handles=hl, loc="upper left", bbox_to_anchor=(0.005, 0.99),
                     fontsize=6.5, title="relative flow volume",
                     labelspacing=0.55, handlelength=2.1, borderpad=0.5,
                     frameon=True, framealpha=0.9, edgecolor="#d6d2c8")
    leg1.get_title().set_fontsize(6.8)
    leg1.get_title().set_fontweight("bold")
    ax.add_artist(leg1)
    node_h = [Line2D([], [], ls="none", marker="o", mfc="#3f4a52",
                     mec="white", mew=0.8, ms=ms, label=lab)
              for ms, lab in [(3.6, "low"), (6.4, "medium"), (9.2, "high")]]
    hd = ([Line2D([], [], color=EAST, lw=2.2, label="eastward"),
           Line2D([], [], color=WEST, lw=2.2, label="westward")] + node_h)
    leg2 = ax.legend(handles=hd, loc="lower left", bbox_to_anchor=(0.005, 0.02),
                     fontsize=6.5, title="direction  /  node throughput",
                     labelspacing=0.5, handlelength=2.1, borderpad=0.5,
                     frameon=True, framealpha=0.9, edgecolor="#d6d2c8")
    leg2.get_title().set_fontsize(6.8)
    leg2.get_title().set_fontweight("bold")
    ax.add_artist(leg2)

    # ---- net balance panel
    axb = fig.add_subplot(gs[0, 1])
    order = np.argsort(net)
    ypos = np.arange(n)
    cols = [EAST if net[i] > 0 else WEST for i in order]
    axb.barh(ypos, net[order] / total.max() * 100, color=cols, height=0.68,
             edgecolor="white", linewidth=0.4)
    axb.axvline(0, color="#333", lw=0.8)
    axb.set_yticks(ypos)
    axb.set_yticklabels([NAMES[i] for i in order], fontsize=6.6)
    axb.tick_params(length=2.5, labelsize=6.6)
    axb.tick_params(axis="y", length=0, pad=2)
    axb.set_xlabel("net balance (% of max throughput)", fontsize=7.5,
                   labelpad=2)
    axb.set_ylim(-0.7, n - 0.3)
    for sp in ("top", "right", "left"):
        axb.spines[sp].set_visible(False)
    axb.text(0.0, 1.012, "(b)", transform=axb.transAxes, fontsize=11,
             fontweight="bold", va="bottom")
    axb.text(0.135, 1.016, "Net inflow $-$ outflow", transform=axb.transAxes,
             fontsize=9.5, fontweight="bold", va="bottom", color="#2f4a58")
    axb.text(0.02, 0.985, "$\\leftarrow$ net exporter", transform=axb.transAxes,
             ha="left", va="top", fontsize=6.4, color=WEST, fontweight="bold")
    axb.text(0.98, 0.035, "net importer $\\rightarrow$",
             transform=axb.transAxes, ha="right", fontsize=6.4, color=EAST,
             fontweight="bold")

    savefig(fig, "59_flow_map.png")


if __name__ == "__main__":
    main()

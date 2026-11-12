"""Attack graph from an untrusted network entry point to a crown-jewel asset.

Nodes are hosts and services, sized by asset value and shaded by how exposed
they are to untrusted networks; directed edges are exploitable vulnerabilities
weighted by the attacker hours they cost.  Every simple path from the internet
to the payment database is enumerated, and the cheapest one is drawn in house
red - it is the single emphasis object of the figure.  The two side panels
rank the four cheapest paths by cost and by end-to-end success probability,
which do not agree: the fastest route is not the most reliable one.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig, get_cmap
from figkit import arrow, safe_colorbar, panel_labels_inches

RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
ACCENT = "#b23a48"      # house red: the shortest attack path only
BLUE = "#3b5b92"

# name -> (x, y, asset value 0-10, exposure 0-1, label side)
NODES = {
    "Internet": (0.30, 2.60, 1.0, 1.00, "left"),
    "VPN gateway": (1.60, 4.35, 4.0, 0.86, "above"),
    "Web server": (1.60, 2.60, 5.0, 0.92, "above"),
    "Mail relay": (1.60, 0.85, 3.5, 0.78, "below"),
    "App server": (2.95, 3.75, 6.5, 0.52, "above"),
    "Jump host": (2.95, 2.15, 6.0, 0.44, "below"),
    "File share": (2.95, 0.55, 5.0, 0.35, "below"),
    "Domain ctrl": (4.30, 3.15, 9.0, 0.20, "above"),
    "Backup vault": (4.30, 1.20, 7.5, 0.16, "below"),
    "Payment DB": (5.50, 2.20, 10.0, 0.08, "right"),
}

# (source, target, CVE, attacker hours, per-hop success probability)
EDGES = [
    ("Internet", "VPN gateway", "CVE-2024-3400", 6.0, 0.42),
    ("Internet", "Web server", "CVE-2023-44487", 3.5, 0.68),
    ("Internet", "Mail relay", "CVE-2023-23397", 5.0, 0.55),
    ("VPN gateway", "Jump host", "CVE-2024-21762", 4.0, 0.60),
    ("Web server", "App server", "CVE-2024-4577", 2.5, 0.72),
    ("Web server", "Jump host", "CVE-2022-1388", 5.5, 0.48),
    ("Mail relay", "File share", "CVE-2021-34527", 4.5, 0.51),
    ("App server", "Domain ctrl", "CVE-2021-42287", 7.0, 0.38),
    ("Jump host", "Domain ctrl", "CVE-2020-1472", 3.0, 0.66),
    ("Jump host", "Backup vault", "CVE-2023-27532", 6.5, 0.44),
    ("File share", "Backup vault", "CVE-2022-26134", 5.0, 0.40),
    ("Domain ctrl", "Payment DB", "CVE-2019-1040", 4.5, 0.58),
    ("Backup vault", "Payment DB", "CVE-2023-34362", 8.0, 0.33),
]

SRC, DST = "Internet", "Payment DB"
NODE_R = 0.21          # data-unit clearance kept between arrows and markers
# CVE captions sit off the edge normal; (side, position along the edge) is
# tuned per hop so no caption lands on a node label.
CVE_PLACE = {("Internet", "Web server"): (-1.0, 0.52),
             ("Web server", "Jump host"): (+1.0, 0.62),
             ("Jump host", "Domain ctrl"): (+1.0, 0.50),
             ("Domain ctrl", "Payment DB"): (+1.0, 0.46)}


def enumerate_paths():
    """All simple SRC -> DST paths with their cost and success probability."""
    out = []

    def walk(node, chain, hours, prob):
        if node == DST:
            out.append((list(chain), hours, prob))
            return
        for s, t, _cve, h, p in EDGES:
            if s == node and t not in chain:
                walk(t, chain + [t], hours + h, prob * p)

    walk(SRC, [SRC], 0.0, 1.0)
    out.sort(key=lambda r: r[1])
    return out


def label_offset(side):
    """Text anchor and offset for a node label placed on the given side."""
    return {"left": ((-0.30, 0.0), "right", "center"),
            "right": ((0.32, 0.0), "left", "center"),
            "above": ((0.0, 0.30), "center", "bottom"),
            "below": ((0.0, -0.30), "center", "top")}[side]


def main():
    set_pub_style()
    np.random.default_rng(576)          # fixed seed; the layout is scripted

    paths = enumerate_paths()
    best = paths[0]
    best_edges = set(zip(best[0][:-1], best[0][1:]))

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.60)
    ax = fig.add_subplot(gs[:, 0])
    ax_cost = fig.add_subplot(gs[0, 1])
    ax_prob = fig.add_subplot(gs[1, 1])

    # ---- (a) attack graph -------------------------------------------------
    cmap = get_cmap("cb_sequential_blue")
    for s, t, cve, h, p in EDGES:
        x0, y0, x1, y1 = NODES[s][0], NODES[s][1], NODES[t][0], NODES[t][1]
        hot = (s, t) in best_edges
        arrow(ax, (x0, y0), (x1, y1),
              color=ACCENT if hot else "#9a9a9a",
              lw=2.0 if hot else 0.85, ms=13 if hot else 9,
              alpha=1.0 if hot else 0.70, zorder=5 if hot else 3,
              gap=NODE_R)
        if hot:
            dx, dy = x1 - x0, y1 - y0
            d = float(np.hypot(dx, dy))
            px, py = -dy / d, dx / d              # unit normal to the edge
            sign, frac = CVE_PLACE[(s, t)]
            ax.text(x0 + frac * dx + 0.30 * sign * px,
                    y0 + frac * dy + 0.30 * sign * py,
                    f"{cve}\n{h:.1f} h", ha="center", va="center",
                    fontsize=6.0, color=ACCENT, linespacing=1.15, zorder=8)

    xs = np.array([v[0] for v in NODES.values()])
    ys = np.array([v[1] for v in NODES.values()])
    vals = np.array([v[2] for v in NODES.values()])
    expo = np.array([v[3] for v in NODES.values()])
    ax.scatter(xs, ys, s=70.0 + 26.0 * vals, c=expo, cmap=cmap,
               vmin=-0.25, vmax=1.0, edgecolors="white", linewidths=1.1,
               zorder=6)
    for name, (x, y, v, e, side) in NODES.items():
        (ox, oy), ha, va = label_offset(side)
        on_path = name in best[0]
        ax.text(x + ox, y + oy, f"{name}\nvalue {v:.0f}/10", ha=ha, va=va,
                fontsize=6.8, color=ACCENT if on_path else "#1a1a1a",
                fontweight="bold" if on_path else "normal",
                linespacing=1.20, zorder=8)

    ax.set_xlim(-0.95, 6.35)
    ax.set_ylim(-0.85, 5.55)
    ax.set_aspect("equal")
    ax.axis("off")

    handles = [
        Line2D([], [], color=ACCENT, lw=2.0,
               label=f"Cheapest attack path ({best[1]:.1f} h)"),
        Line2D([], [], color="#9a9a9a", lw=0.85,
               label="Other exploitable edge"),
        Line2D([], [], color=SUB, lw=0, marker="o", ms=3.4, mfc="#9cc3dd",
               mec="white", label="Asset value 4/10"),
        Line2D([], [], color=SUB, lw=0, marker="o", ms=6.2, mfc="#9cc3dd",
               mec="white", label="Asset value 10/10"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0),
              fontsize=6.8, frameon=False, ncol=2, handlelength=1.7,
              columnspacing=1.4, labelspacing=0.4)
    ax.set_title("Ten hosts, thirteen exploitable edges, one cheap way in",
                 fontsize=9.0, fontweight="bold", pad=4)

    # ---- (b) / (c) the four cheapest paths --------------------------------
    top = paths[:4]
    labels = [" > ".join(n.split()[0] for n in c[1:-1]) for c, _h, _p in top]
    y = np.arange(len(top))[::-1]
    cost = np.array([h for _c, h, _p in top])
    prob = np.array([100.0 * p for _c, _h, p in top])
    cols = [ACCENT if i == 0 else BLUE for i in range(len(top))]

    for axi, dat, unit, ttl, xlab in (
            (ax_cost, cost, "h", "Attacker effort per path",
             "attack cost (hours)"),
            (ax_prob, prob, "%", "End-to-end success probability",
             "success probability (%)")):
        axi.barh(y, dat, height=0.62, color=cols, edgecolor="white",
                 linewidth=0.7, zorder=3)
        for yi, d in zip(y, dat):
            axi.text(d + 0.02 * dat.max(), yi, f"{d:.1f} {unit}",
                     ha="left", va="center", fontsize=6.4, color=SUB)
        axi.set_yticks(y)
        axi.set_yticklabels([f"P{i + 1}  {lab}" for i, lab in
                             enumerate(labels)], fontsize=6.2)
        axi.set_xlim(0.0, dat.max() * 1.30)
        axi.tick_params(labelsize=6.5, length=2.4)
        axi.set_xlabel(xlab, fontsize=7.4)
        axi.set_title(ttl, fontsize=8.5, fontweight="bold")
        axi.grid(axis="x", lw=0.45, alpha=0.45)
        axi.set_axisbelow(True)

    slow = paths[-1]
    cut = min(h for c, h, _p in paths if best[0][2] not in c)

    fig.tight_layout(rect=[0, 0.045, 1, 0.94], w_pad=2.8, h_pad=2.8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-0.25, 1.0))
    safe_colorbar(fig, sm, ax, label="exposure to untrusted networks (-)",
                  side="bottom", width=0.11, pad=0.06, shrink=0.80,
                  ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    panel_labels_inches(fig, [ax, ax_cost, ax_prob], dx=-0.18, dy=0.06)
    savefig(fig, "576_sec_attack_graph.png")


if __name__ == "__main__":
    main()

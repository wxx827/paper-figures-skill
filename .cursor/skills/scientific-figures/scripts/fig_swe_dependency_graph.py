"""Figure 619 - Layered module dependency graph with a cycle called out.

One edge list drives all three panels.  Panel (a) lays the twelve modules on
four architectural tiers and draws every dependency downward, so the single
upward edge is visibly the odd one out; that edge closes a three-module cycle
which is stroked in house red together with the modules it passes through.
Panels (b) and (c) are derived from the same list: the fan-in / fan-out
counts per tier, and the instability-abstractness plane where instability is
computed from the degrees rather than assigned.

Reference-gallery figure: the numbers are generated, the layout is the point.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, get_palette, COLORS
from figkit import block, arrow, panel_card, panel_label_inches, use_inch_units

ACCENT = "#b23a48"          # house red - the cycle, and only the cycle
BLUE = "#3b5b92"
GREEN = "#2e7d5b"

TIERS = ["presentation", "business", "data access", "infrastructure"]

# name -> (tier index, KLOC, abstractness)
NODES = {
    "web.dashboard": (0, 18.4, 0.08),
    "web.admin": (0, 9.1, 0.05),
    "app.mobile": (0, 12.6, 0.10),
    "svc.catalog": (1, 14.2, 0.32),
    "svc.billing": (1, 21.5, 0.24),
    "svc.orders": (1, 26.8, 0.21),
    "repo.catalog": (2, 11.7, 0.46),
    "cache.session": (2, 6.3, 0.38),
    "repo.ledger": (2, 17.0, 0.41),
    "lib.config": (3, 4.2, 0.72),
    "lib.logging": (3, 3.4, 0.81),
    "lib.metrics": (3, 5.1, 0.66),
}
ORDER = {0: ["web.dashboard", "web.admin", "app.mobile"],
         1: ["svc.catalog", "svc.billing", "svc.orders"],
         2: ["repo.catalog", "cache.session", "repo.ledger"],
         3: ["lib.config", "lib.logging", "lib.metrics"]}

# (source, target, coupling strength)
EDGES = [
    ("web.dashboard", "svc.orders", 7), ("web.dashboard", "svc.catalog", 4),
    ("web.admin", "svc.billing", 5), ("web.admin", "svc.catalog", 2),
    ("app.mobile", "svc.orders", 6), ("app.mobile", "svc.catalog", 3),
    ("svc.catalog", "repo.catalog", 7), ("svc.catalog", "cache.session", 3),
    ("svc.billing", "repo.ledger", 6), ("svc.orders", "cache.session", 4),
    ("svc.orders", "repo.ledger", 8), ("svc.orders", "svc.billing", 5),
    ("repo.catalog", "lib.config", 3), ("repo.catalog", "lib.metrics", 3),
    ("cache.session", "lib.logging", 2), ("cache.session", "lib.metrics", 2),
    ("repo.ledger", "lib.config", 4), ("repo.ledger", "lib.logging", 4),
    ("repo.ledger", "svc.orders", 3),          # <- the layering violation
]
CYCLE = [("svc.orders", "svc.billing"), ("svc.billing", "repo.ledger"),
         ("repo.ledger", "svc.orders")]
CYCLE_NODES = ["svc.orders", "svc.billing", "repo.ledger"]


def degrees():
    fan_in = {k: 0 for k in NODES}
    fan_out = {k: 0 for k in NODES}
    for s, t, _ in EDGES:
        fan_out[s] += 1
        fan_in[t] += 1
    return fan_in, fan_out


def draw_graph(fig, ax, ramp):
    w, h = use_inch_units(ax)
    band_h = h / 4.8
    y_top = h - band_h / 2.0 - 0.015 * h
    y_bot = band_h / 2.0 + 0.015 * h
    xs = [0.215 * w, 0.455 * w, 0.695 * w]
    kmax = max(v[1] for v in NODES.values())

    boxes = {}
    for ti, tier in enumerate(TIERS):
        yc = y_top - ti * (y_top - y_bot) / 3.0
        panel_card(ax, 0.50 * w, yc, 0.96 * w, band_h, fc="#f5f5f5",
                   ec="#e0e0e0", lw=0.7, zorder=1, rounding=0.05)
        ax.text(0.035 * w, yc + band_h / 2 - 0.07, tier, ha="left", va="top",
                fontsize=6.8, color=COLORS["sub"], zorder=2)
        for name, x in zip(ORDER[ti], xs):
            kloc = NODES[name][1]
            sc = 0.62 + 0.38 * np.sqrt(kloc / kmax)
            bw, bh = 0.205 * w * sc, 0.40 * band_h * sc
            hot = name in CYCLE_NODES
            b = block(ax, x, yc, bw, bh, color=ramp[ti],
                      ec=ACCENT if hot else "white",
                      lw=1.7 if hot else 0.9, zorder=4, rounding=0.045)
            ax.text(x, yc + 0.085, name, ha="center", va="center",
                    fontsize=6.8, color="white" if ti else "#1a1a1a",
                    fontfamily="monospace", zorder=6)
            ax.text(x, yc - 0.095, "%.1f KLOC" % kloc, ha="center",
                    va="center", fontsize=6.0, zorder=6,
                    color="#eef4f9" if ti else "#3a3a3a")
            boxes[name] = b
    return boxes, w, h


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(619)
    ramp = get_palette("cb_sequential_blue", 7)[2:6]
    fan_in, fan_out = degrees()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58],
                          left=0.045, right=0.965, top=0.885, bottom=0.115,
                          wspace=0.30, hspace=0.46)
    ax = fig.add_subplot(gs[:, 0])
    ax_f = fig.add_subplot(gs[0, 1])
    ax_i = fig.add_subplot(gs[1, 1])

    # ---- (a) layered dependency graph -------------------------------------
    boxes, W, H = draw_graph(fig, ax, ramp)
    for s, t, c in EDGES:
        a, b = boxes[s], boxes[t]
        cyc = (s, t) in CYCLE
        col = ACCENT if cyc else "#6b7d8f"
        lw = (0.55 + 0.23 * c) * (1.55 if cyc else 1.0)
        if s == "repo.ledger" and t == "svc.orders":       # the back edge
            arrow(ax, a["r"], b["r"], color=col, lw=lw, rad=-0.85, ms=9,
                  gap=0.05, zorder=8)
        elif NODES[s][0] == NODES[t][0]:                   # same tier
            arrow(ax, a["l"], b["r"], color=col, lw=lw, rad=0.0, ms=9,
                  gap=0.05, zorder=8)
        else:
            arrow(ax, a["b"], b["t"], color=col, lw=lw, ms=9, gap=0.05,
                  alpha=0.85, zorder=3)

    ax.text(0.960 * W, 0.938 * H,
            "cycle\n%s" % "\n".join("-> " + n for n in CYCLE_NODES),
            ha="right", va="top", fontsize=6.4, color=ACCENT,
            fontfamily="monospace", linespacing=1.45, zorder=9)
    ax.set_title("%d modules, %d dependency edges, 1 cycle"
                 % (len(NODES), len(EDGES)), fontsize=8.8, fontweight="bold",
                 pad=4.0)

    # ---- (b) fan-in / fan-out per tier ------------------------------------
    fi = [sum(fan_in[n] for n in ORDER[t]) for t in range(4)]
    fo = [sum(fan_out[n] for n in ORDER[t]) for t in range(4)]
    yy = np.arange(4)[::-1]
    ax_f.barh(yy + 0.19, fi, height=0.34, color=BLUE, alpha=0.85,
              edgecolor="none", label="fan-in", zorder=3)
    ax_f.barh(yy - 0.19, fo, height=0.34, color=GREEN, alpha=0.85,
              edgecolor="none", label="fan-out", zorder=3)
    for y, a, b in zip(yy, fi, fo):
        ax_f.text(a + 0.35, y + 0.19, str(a), va="center", fontsize=6.2,
                  color=BLUE)
        ax_f.text(b + 0.35, y - 0.19, str(b), va="center", fontsize=6.2,
                  color=GREEN)
    ax_f.set_yticks(yy)
    ax_f.set_yticklabels(TIERS, fontsize=6.6)
    ax_f.set_xlabel("incident edges (-)", fontsize=7.4)
    ax_f.set_xlim(0, max(fi + fo) * 1.18)
    ax_f.tick_params(labelsize=6.6)
    ax_f.grid(axis="x", lw=0.45, alpha=0.55)
    ax_f.set_axisbelow(True)
    for s in ("top", "right"):
        ax_f.spines[s].set_visible(False)
    ax_f.legend(frameon=False, fontsize=6.4, loc="upper right",
                handlelength=1.2, handletextpad=0.4, borderpad=0.2)
    ax_f.set_title("Coupling budget per tier", fontsize=8.5,
                   fontweight="bold", pad=3.5)

    # ---- (c) instability versus abstractness ------------------------------
    names = list(NODES)
    inst = np.array([fan_out[n] / max(fan_in[n] + fan_out[n], 1)
                     for n in names])
    abst = np.array([NODES[n][2] for n in names])
    dist = np.abs(abst + inst - 1.0) / np.sqrt(2.0)
    hot = np.array([n in CYCLE_NODES for n in names])
    sz = np.array([NODES[n][1] for n in names])

    ax_i.plot([0, 1], [1, 0], color=COLORS["baseline"], lw=1.0,
              ls=(0, (4.5, 2.2)), zorder=2)
    ax_i.text(0.755, 0.225, "main sequence", fontsize=6.2,
              color=COLORS["muted"], rotation=-33, rotation_mode="anchor",
              ha="left", va="bottom", zorder=3)
    ax_i.scatter(inst[~hot], abst[~hot], s=2.4 * sz[~hot] + 6.0,
                 facecolor=BLUE, alpha=0.55, edgecolor="white", lw=0.5,
                 zorder=4)
    ax_i.scatter(inst[hot], abst[hot], s=2.4 * sz[hot] + 6.0,
                 facecolor=ACCENT, alpha=0.90, edgecolor="white", lw=0.5,
                 zorder=5)
    worst = int(np.argmax(dist))
    ax_i.annotate(names[worst], xy=(inst[worst], abst[worst]),
                  xytext=(inst[worst] - 0.14, abst[worst] + 0.20),
                  fontsize=6.2, color="#1a1a1a", fontfamily="monospace",
                  ha="center", zorder=6,
                  arrowprops=dict(arrowstyle="-", lw=0.6, color="#777777"))
    ax_i.set_xlim(-0.05, 1.05)
    ax_i.set_ylim(-0.05, 1.10)
    ax_i.set_xlabel("instability I = out / (in + out) (-)", fontsize=7.4)
    ax_i.set_ylabel("abstractness A (-)", fontsize=7.4)
    ax_i.tick_params(labelsize=6.6)
    ax_i.grid(True, lw=0.45, alpha=0.5)
    ax_i.set_axisbelow(True)
    for s in ("top", "right"):
        ax_i.spines[s].set_visible(False)
    ax_i.set_title("Mean distance from main sequence %.2f" % dist.mean(),
                   fontsize=8.5, fontweight="bold", pad=3.5)

    for a, lab in ((ax, "(a)"), (ax_f, "(b)"), (ax_i, "(c)")):
        panel_label_inches(fig, a, lab, dx=-0.34, dy=0.09)


    savefig(fig, "619_swe_dependency_graph.png")


if __name__ == "__main__":
    main()

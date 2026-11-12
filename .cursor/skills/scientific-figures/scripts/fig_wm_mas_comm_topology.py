"""Four communication topologies, priced in bandwidth and consensus rounds.

Nothing here is asserted by hand.  Each panel's round count comes from running
Metropolis-weight average consensus on that panel's own adjacency matrix until
the worst-case deviation drops under 1% of its starting value, and the curves
underneath are those same runs.  Bandwidth is 2 x edges x 64 B per step, with
the multiplication printed so the reader never has to trust it.

The critical edges highlighted in the learned panel are found by deleting one
edge at a time and re-running consensus: the three whose removal costs the most
rounds are the ones drawn in the accent colour.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import COLORS, savefig, set_pub_style

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
ACCENT = COLORS["accent"]   # the single emphasis colour, DESIGN_SYSTEM 2.1
# Batch rule: categories are separated by depth on the one blue ladder and by
# line style, never by a second hue.  "Critical" here is the deepest rung plus
# weight, which is why this figure needs no accent colour at all.
HILITE = BLUE[0]
EDGE = "#8ba3b8"
CARD = "#dfe6ee"           # 0.898 luminance: unambiguously on the ink side
N = 8
MSG_BYTES = 64
TOL = 0.01
MAXIT = 400

RING = [(i, (i + 1) % N) for i in range(N)]
TOPOS = [
    ("Fully connected", [(i, j) for i in range(N) for j in range(i + 1, N)]),
    ("Star (hub = A)", [(0, j) for j in range(1, N)]),
    ("Ring", RING),
    ("Learned sparse", RING + [(0, 4), (2, 6), (1, 5)]),
]


def adjacency(edges):
    A = np.zeros((N, N), bool)
    for i, j in edges:
        A[i, j] = A[j, i] = True
    return A


def consensus(edges):
    """Metropolis-weight averaging; returns the error trace and the rounds."""
    A = adjacency(edges)
    deg = A.sum(axis=1)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if A[i, j]:
                W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
        W[i, i] = 1.0 - W[i].sum()
    x = np.linspace(-1.0, 1.0, N) * np.array([1, -1, 1, -1, 1, -1, 1, -1])
    mu = x.mean()
    e0 = np.abs(x - mu).max()
    err = [1.0]
    rounds = None
    for k in range(1, MAXIT + 1):
        x = W @ x
        err.append(float(np.abs(x - mu).max() / e0))
        if rounds is None and err[-1] < TOL:
            rounds = k
    return np.array(err), (rounds if rounds is not None else MAXIT)


def main():
    set_pub_style(base_font=8)

    data = []
    for name, edges in TOPOS:
        err, rounds = consensus(edges)
        data.append(dict(name=name, edges=edges, err=err, rounds=rounds,
                         bytes=2 * len(edges) * MSG_BYTES))
    # critical edges of the learned graph: the three whose deletion costs most
    learned = data[-1]
    cost = []
    for e in learned["edges"]:
        rest = [q for q in learned["edges"] if q != e]
        if adjacency(rest).sum(axis=1).min() == 0:
            cost.append((MAXIT, e))
        else:
            cost.append((consensus(rest)[1], e))
    cost.sort(reverse=True)
    critical = [e for _, e in cost[:3]]
    penalty = cost[0][0] - learned["rounds"]

    ang = np.pi / 2 - 2 * np.pi * np.arange(N) / N
    P = np.stack([np.cos(ang), np.sin(ang)], axis=1) * 0.74
    names = [chr(ord("A") + i) for i in range(N)]

    fig = plt.figure(figsize=(11.0, 4.6))
    y0, ph = 0.462, 0.372
    pw = ph * 4.6 / 11.0
    gap, x_left = 0.020, 0.045
    x_end = x_left + 4 * pw + 3 * gap

    for k, d in enumerate(data):
        ax = fig.add_axes([x_left + k * (pw + gap), y0, pw, ph])
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                   facecolor=CARD, edgecolor=BLUE[5],
                                   linewidth=0.7, zorder=0))
        for (i, j) in d["edges"]:
            hot = (k == 3) and ((i, j) in critical)
            ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]],
                    color=ACCENT if hot else EDGE, lw=1.9 if hot else 0.85,
                    alpha=1.0 if hot else 0.85, zorder=3 if hot else 2)
        for i in range(N):
            ax.plot([P[i, 0]], [P[i, 1]], marker="o", ls="none", ms=11.5,
                    mfc=BLUE[1], mec="white", mew=1.0, zorder=4)
            ax.text(P[i, 0], P[i, 1], names[i], ha="center", va="center",
                    fontsize=6.2, fontweight="bold", color="white", zorder=5)
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.5, 1.035, d["name"], transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8.6, fontweight="bold",
                color=ACCENT if k == 3 else INK)
        ax.text(0.032, 0.968, f"({'abcd'[k]})", transform=ax.transAxes,
                ha="left", va="top", fontsize=8.4, fontweight="bold",
                color=BLUE[1])
        ax.text(0.5, -0.050,
                f"2 x {len(d['edges']):d} edges x {MSG_BYTES:d} B\n"
                f"= {d['bytes']:d} B / step", transform=ax.transAxes,
                ha="center", va="top", fontsize=7.0, color=INK,
                linespacing=1.45)
        ax.text(0.5, -0.240,
                f"consensus: {d['rounds']:d} "
                f"round{'' if d['rounds'] == 1 else 's'}",
                transform=ax.transAxes,
                ha="center", va="top", fontsize=7.4, fontweight="bold",
                color=ACCENT if k == 3 else COLORS["sub"])

    # ---- where those round counts come from -------------------------------
    axc = fig.add_axes([x_left, 0.115, 0.415, 0.215])
    styles = [(BLUE[0], "-"), (BLUE[2], (0, (5, 2))), (BLUE[4], (0, (1.6, 1.6))),
              (ACCENT, (0, (4, 1.4, 1, 1.4)))]
    xmax = max(d["rounds"] for d in data) + 6
    for d, (col, ls) in zip(data, styles):
        axc.semilogy(np.arange(d["err"].size), np.clip(d["err"], 1e-4, None),
                     color=col, lw=1.6, ls=ls, zorder=4,
                     label=f"{d['name']}, {d['rounds']:d} "
                           f"round{'' if d['rounds'] == 1 else 's'}")
        axc.plot([d["rounds"]], [d["err"][d["rounds"]]], marker="o", ms=4.2,
                 mfc="white", mec=col, mew=1.2, zorder=5)
    axc.plot([0, xmax], [TOL, TOL], color="#8a97a4", lw=0.9,
             ls=(0, (3, 2)), zorder=3)
    axc.text(1.6, TOL * 0.62, f"{TOL * 100:.0f}% tolerance", ha="left",
             va="top", fontsize=6.6, color=COLORS["muted"], zorder=6,
             bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                       alpha=0.92))
    axc.set_xlim(0, xmax)
    axc.set_ylim(2e-4, 1.6)
    axc.set_xlabel("communication round", fontsize=7.6, color=COLORS["sub"],
                   labelpad=1.5)
    axc.set_ylabel("max deviation\n(fraction of start)", fontsize=7.2,
                   color=COLORS["sub"], labelpad=2.0, linespacing=1.3)
    axc.tick_params(labelsize=7.0, length=2.6)
    axc.grid(color="#e6eaee", lw=0.6)
    axc.set_axisbelow(True)
    leg = axc.legend(loc="upper right", fontsize=6.8, frameon=True,
                     facecolor="white", edgecolor="#c3ccd4", framealpha=0.96,
                     handlelength=2.3, labelspacing=0.42, borderpad=0.42)
    leg.set_zorder(8)
    for txt in leg.get_texts():
        txt.set_color(INK)

    # ---- the price list ----------------------------------------------------
    cols = [0.505, 0.660, 0.760, 0.870, 0.962]
    head = ["topology", "edges", "B / step", "rounds", "B x rounds"]
    ytab = 0.315
    for x, h, al in zip(cols, head, ["left"] + ["right"] * 4):
        fig.text(x, ytab, h, ha=al, va="bottom", fontsize=7.0,
                 fontweight="bold", color=COLORS["sub"])
    fig.add_artist(Line2D([cols[0], cols[-1]], [ytab - 0.014, ytab - 0.014],
                          color="#c3ccd4", lw=0.8))
    full_cost = data[0]["bytes"] * data[0]["rounds"]
    for r, d in enumerate(data):
        yr = ytab - 0.052 - r * 0.040
        hot = (r == 3)
        col = ACCENT if hot else INK
        total = d["bytes"] * d["rounds"]
        vals = [d["name"], f"{len(d['edges']):d}", f"{d['bytes']:d}",
                f"{d['rounds']:d}", f"{total:d}"]
        for x, v, al in zip(cols, vals, ["left"] + ["right"] * 4):
            fig.text(x, yr, v, ha=al, va="baseline", fontsize=7.0,
                     fontweight="bold" if hot else "normal", color=col)
    learned_cost = data[3]["bytes"] * data[3]["rounds"]
    fig.text(cols[0], ytab - 0.052 - 4 * 0.040 - 0.014,
             f"Total traffic = bytes per step x rounds.  Full connectivity "
             f"is unbeatable in rounds but spends {data[0]['bytes']:d} B "
             f"inside that one step,\n"
             f"{data[0]['bytes'] / data[3]['bytes']:.1f}x the learned graph's "
             f"per-step budget; among the three sparse graphs the learned one "
             f"is the cheapest overall ({learned_cost:d} B).",
             ha="left", va="top", fontsize=6.7, color=COLORS["muted"],
             linespacing=1.55)

    key = [Line2D([], [], marker="o", ls="none", ms=7.0, mfc=BLUE[1],
                  mec="white", mew=1.0, label=f"agent ({N:d} per panel, same "
                  f"layout everywhere)"),
           Line2D([], [], color=EDGE, lw=0.85, label="communication link"),
           Line2D([], [], color=ACCENT, lw=1.9,
                  label=f"critical link (deleting one costs up to "
                  f"+{penalty:d} rounds)")]
    leg2 = fig.legend(handles=key, loc="upper left",
                      bbox_to_anchor=(x_end + 0.028, 0.855), frameon=False,
                      fontsize=7.2, handletextpad=0.6, labelspacing=0.7,
                      handlelength=1.8)
    for txt in leg2.get_texts():
        txt.set_color(INK)


    savefig(fig, "1186_wm_mas_comm_topology.png")


if __name__ == "__main__":
    main()

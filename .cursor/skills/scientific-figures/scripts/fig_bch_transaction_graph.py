"""Address-level transaction graph with one laundering chain traced through it.

The graph is built once: five hub addresses (three exchanges, two mining
pools) are pinned, a six-hop suspicious chain is threaded through the middle,
and the ordinary addresses are rejection-sampled around the hubs so that no
two discs touch.  Every edge is directed and its width scales with the
transferred amount, so the thick red path really is the largest single flow on
the canvas.

Both side panels read off that same graph rather than a second synthetic
draw: the degree CCDF is counted from the adjacency list and its slope comes
from a least-squares fit on the log-log tail, and the Lorenz curve, its Gini
coefficient and the top-decile share are all integrated from the balance
vector that also sets the disc radii.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

from pubstyle import set_pub_style, savefig, style_axes
from figkit import axes_inches, inch_axes, panel_labels_inches

W, H = 10.4, 6.4

EXCH = "#3b5b92"        # exchange custodial wallet (house blue)
POOL = "#c9a227"        # mining pool payout wallet (house gold)
REG = "#8a8a8a"         # ordinary user address (neutral)
SUSP = "#b23a48"        # flagged address on the traced chain (house red)

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

KIND_COLOR = {"exchange": EXCH, "pool": POOL, "regular": REG,
              "suspicious": SUSP}

HUBS = [(1.05, 3.85, "exchange"), (5.05, 3.95, "exchange"),
        (3.15, 1.05, "exchange"), (5.40, 2.00, "pool"), (0.90, 1.60, "pool")]
CHAIN = [(1.80, 2.90), (2.60, 2.28), (3.42, 2.82), (4.18, 2.18),
         (4.72, 3.00)]

XMIN, XMAX, YMIN, YMAX = 0.30, 5.90, 0.55, 4.62


def build_graph(rng):
    pos, kind = [], []
    for x, y, k in HUBS:
        pos.append((x, y))
        kind.append(k)
    for x, y in CHAIN:
        pos.append((x, y))
        kind.append("suspicious")

    n_reg, tries = 0, 0
    while n_reg < 30 and tries < 8000:
        tries += 1
        hx, hy, _ = HUBS[rng.integers(len(HUBS))]
        a = rng.uniform(0.0, 2.0 * np.pi)
        r = rng.uniform(0.60, 1.60)
        x, y = hx + r * np.cos(a), hy + r * np.sin(a)
        if not (XMIN <= x <= XMAX and YMIN <= y <= YMAX):
            continue
        if min(np.hypot(x - px, y - py) for px, py in pos) < 0.36:
            continue
        pos.append((x, y))
        kind.append("regular")
        n_reg += 1

    pos = np.asarray(pos)
    kind = np.asarray(kind)

    bal = np.empty(len(pos))
    for i, k in enumerate(kind):
        if k == "exchange":
            bal[i] = rng.uniform(260.0, 520.0)
        elif k == "pool":
            bal[i] = rng.uniform(120.0, 210.0)
        elif k == "suspicious":
            bal[i] = rng.uniform(9.0, 26.0)
        else:
            bal[i] = float(np.exp(rng.normal(0.6, 1.0)))

    hub_idx = list(range(len(HUBS)))
    chain_idx = list(range(len(HUBS), len(HUBS) + len(CHAIN)))
    reg_idx = list(range(len(HUBS) + len(CHAIN), len(pos)))

    edges = []                              # (src, dst, amount BTC)
    for i in reg_idx:
        d = [np.hypot(*(pos[i] - pos[h])) for h in hub_idx]
        h = hub_idx[int(np.argmin(d))]
        amt = float(np.exp(rng.normal(-0.2, 0.9)))
        edges.append((i, h, amt) if rng.random() < 0.55 else (h, i, amt))
        if rng.random() < 0.50:
            other = [j for j in reg_idx if j != i]
            dd = [np.hypot(*(pos[i] - pos[j])) for j in other]
            j = other[int(np.argmin(dd))]
            edges.append((i, j, float(np.exp(rng.normal(-0.9, 0.8)))))
    for a, b in ((0, 4), (1, 3), (2, 3), (0, 2)):
        edges.append((a, b, float(rng.uniform(6.0, 18.0))))

    path = [0] + chain_idx + [1]
    flow = float(rng.uniform(11.0, 15.0))
    trace = []
    for a, b in zip(path[:-1], path[1:]):
        flow *= rng.uniform(0.90, 0.98)     # a slice is skimmed at every hop
        trace.append((a, b, flow))
    return pos, kind, bal, edges, trace, path


def degree(n, edges, trace):
    deg = np.zeros(n, dtype=int)
    for a, b, _ in list(edges) + list(trace):
        deg[a] += 1
        deg[b] += 1
    return deg


def draw_network(ax, pos, kind, bal, edges, trace, path):
    wmax = max(a for *_, a in edges)
    for a, b, amt in edges:
        ax.add_patch(FancyArrowPatch(
            pos[a], pos[b], arrowstyle="-|>", mutation_scale=6.0,
            linewidth=0.35 + 1.05 * (amt / wmax) ** 0.5, color="#b3b3b3",
            alpha=0.62, connectionstyle="arc3,rad=0.13",
            shrinkA=5.0, shrinkB=6.0, zorder=2))

    for step, (a, b, amt) in enumerate(trace, start=1):
        ax.add_patch(FancyArrowPatch(
            pos[a], pos[b], arrowstyle="-|>", mutation_scale=11.0,
            linewidth=2.05, color=SUSP, alpha=0.95,
            connectionstyle="arc3,rad=0.10",
            shrinkA=8.0, shrinkB=9.0, zorder=5))
        mx, my = 0.5 * (pos[a] + pos[b])
        dx, dy = pos[b] - pos[a]
        d = max(float(np.hypot(dx, dy)), 1e-9)
        mx, my = mx - 0.20 * dy / d, my + 0.20 * dx / d
        ax.text(mx, my, f"{step}", ha="center", va="center",
                fontsize=6.4, color="white", zorder=7,
                bbox=dict(boxstyle="circle,pad=0.16", facecolor=SUSP,
                          edgecolor="white", linewidth=0.7))

    rmax = bal.max()
    for i, (x, y) in enumerate(pos):
        r = 0.052 + 0.085 * (bal[i] / rmax) ** 0.5
        ax.add_patch(Circle((x, y), r, facecolor=KIND_COLOR[kind[i]],
                            edgecolor="white", linewidth=0.8, zorder=6))
    for i, tag, dy in ((0, "EX-1", -0.30), (1, "EX-2", 0.30),
                       (2, "EX-3", -0.30)):
        ax.text(pos[i][0], pos[i][1] + dy, tag, ha="center",
                va="bottom" if dy > 0 else "top", fontsize=6.6, color=INK,
                zorder=8)
    ax.text(pos[path[1]][0] - 0.20, pos[path[1]][1] + 0.02, "entry hop",
            ha="right", va="center", fontsize=6.4, color=SUSP, zorder=8)
    ax.text(pos[path[-2]][0] + 0.22, pos[path[-2]][1] - 0.02, "cash-out hop",
            ha="left", va="center", fontsize=6.4, color=SUSP, zorder=8)


def legend_strip(ax, y):
    items = [(EXCH, "Exchange wallet"), (POOL, "Mining pool"),
             (REG, "Ordinary address"), (SUSP, "Flagged address")]
    x = 0.30
    for color, label in items:
        ax.add_patch(Circle((x, y), 0.075, facecolor=color,
                            edgecolor="white", linewidth=0.8, zorder=6))
        ax.text(x + 0.15, y, label, ha="left", va="center", fontsize=6.8,
                color=INK)
        x += 1.42
    ax.text(0.30, y - 0.30,
            "Disc area scales with wallet balance; edge width scales with "
            "transferred amount; the red route is the traced flow",
            ha="left", va="center", fontsize=6.5, color=MUTED)


def panel_degree(ax, deg):
    ks = np.arange(1, deg.max() + 1)
    ccdf = np.array([(deg >= k).mean() for k in ks])
    ok = ccdf > 0
    ks, ccdf = ks[ok], ccdf[ok]
    ax.loglog(ks, ccdf, marker="o", ms=3.4, lw=0.0, color=EXCH,
              mec="white", mew=0.5, zorder=4)
    fitm = ks >= 2
    slope, icpt = np.polyfit(np.log10(ks[fitm]), np.log10(ccdf[fitm]), 1)
    xf = np.linspace(np.log10(ks[fitm].min()), np.log10(ks.max()), 40)
    ax.plot(10.0 ** xf, 10.0 ** (icpt + slope * xf), lw=1.0, ls="--",
            color=SUSP, zorder=5,
            label=f"tail fit, slope {slope:.2f}")
    ax.set_xlabel("address degree $k$ (edges)", fontsize=8.0)
    ax.set_ylabel("$P(K \\geq k)$ (-)", fontsize=8.0)
    ax.set_title("Degree tail is scale free", fontsize=8.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=6.8, loc="lower left")
    ax.tick_params(labelsize=6.8)
    style_axes(ax)
    return slope


def panel_lorenz(ax, bal):
    b = np.sort(bal)
    cum = np.concatenate([[0.0], np.cumsum(b) / b.sum()]) * 100.0
    frac = np.linspace(0.0, 100.0, cum.size)
    gini = 1.0 - 2.0 * np.trapezoid(cum / 100.0, frac / 100.0)
    top10 = 100.0 * b[int(0.9 * b.size):].sum() / b.sum()

    ax.fill_between(frac, cum, frac, color=SUSP, alpha=0.16, lw=0.0)
    ax.plot([0, 100], [0, 100], lw=0.9, ls="--", color="#808080",
            label="perfect equality")
    ax.plot(frac, cum, lw=1.6, color=EXCH, label="observed balances")
    ax.plot([90, 90], [0, 100 - top10], lw=0.8, ls=":", color=RULE)
    ax.annotate(f"top 10% of addresses\nhold {top10:.0f}% of balance",
                xy=(90.0, 100.0 - top10), xytext=(19.0, 62.0),
                fontsize=6.6, color=INK, va="center",
                arrowprops=dict(arrowstyle="->", lw=0.7, color=RULE,
                                shrinkA=1.0, shrinkB=2.0))
    ax.text(0.045, 0.70, f"Gini = {gini:.2f}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=6.8, color=SUSP)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("addresses ranked by balance (%)", fontsize=8.0)
    ax.set_ylabel("cumulative balance (%)", fontsize=8.0)
    ax.set_title("Balance is highly concentrated", fontsize=8.5,
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    ax.tick_params(labelsize=6.8)
    style_axes(ax)
    return gini, top10


def main():
    set_pub_style()
    rng = np.random.default_rng(672)
    pos, kind, bal, edges, trace, path = build_graph(rng)
    deg = degree(len(pos), edges, trace)

    fig = plt.figure(figsize=(W, H))
    ax_net = inch_axes(fig, (0.10, 0.30, 6.30, 5.35))
    ax_deg = axes_inches(fig, (7.25, 3.42, 2.75, 2.10))
    ax_lor = axes_inches(fig, (7.25, 0.66, 2.75, 2.10))

    draw_network(ax_net, pos, kind, bal, edges, trace, path)
    legend_strip(ax_net, 0.62)
    slope = panel_degree(ax_deg, deg)
    gini, top10 = panel_lorenz(ax_lor, bal)

    moved = trace[0][2]
    delivered = trace[-1][2]
    skim = 100.0 * (1.0 - delivered / moved)


    panel_labels_inches(fig, [ax_net, ax_deg, ax_lor], dx=-0.02, dy=0.10)
    savefig(fig, "672_bch_transaction_graph.png")


if __name__ == "__main__":
    main()

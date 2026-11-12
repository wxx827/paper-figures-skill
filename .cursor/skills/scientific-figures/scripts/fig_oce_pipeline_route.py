"""Free-span assessment along a subsea pipeline route.

The main panel is a bathymetric section with the soil section filled in
underneath and the pipeline laid on it by a taut-string walk: from each
support the route takes the steepest line reachable within the maximum
allowed chord, so the free spans in red are found by the geometry rather than
drawn by hand.  The two smaller panels close the loop: every span measured in
the section is compared with the allowable span of the soil it crosses, and
the route alternatives are placed on a cost-risk plane with the Pareto set
marked.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
BLUE = "#3b5b92"        # water column, compliant spans
RED = "#b23a48"         # free spans and exceedances
WATER = "#dbe7f1"
PIPE = "#3a3a3a"

# Soil section: semantic colours, deliberately monotone in lightness.
SOIL = {"soft clay": "#d9d2c4", "sand waves": "#c9a227",
        "stiff clay": "#a89a80", "exposed rock": "#8a8f96"}
SEGMENTS = [(0.0, 3.1, "soft clay"), (3.1, 6.2, "sand waves"),
            (6.2, 9.0, "stiff clay"), (9.0, 12.0, "exposed rock")]
ALLOWABLE = {"soft clay": 52.0, "sand waves": 68.0, "stiff clay": 84.0,
             "exposed rock": 96.0}          # allowable free span, m

ROUTE_KM = 12.0
MAX_CHORD = 110.0       # longest chord the taut string may bridge, m
GAP_TOL = 0.40          # gap above the seabed counted as a free span, m
DIAM = 0.61             # pipeline outer diameter, m


DIPS = [(1.55, 0.030, 2.6), (2.70, 0.022, 1.6), (4.05, 0.040, 3.4),
        (5.30, 0.026, 2.1), (6.80, 0.045, 3.8), (8.15, 0.020, 1.4),
        (9.55, 0.038, 3.1), (10.90, 0.028, 2.3)]


def seabed(seed=17, n=4001):
    """Smooth regional slope with discrete scour hollows and light roughness."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, ROUTE_KM, n)
    z = -86.0 - 3.6 * x
    z += 7.0 * np.sin(2.0 * np.pi * x / 7.8) + 2.6 * np.sin(
        2.0 * np.pi * x / 2.9 + 0.8)
    for x0, w, d in DIPS:
        z -= d * np.exp(-((x - x0) / w) ** 2)
    rough = rng.normal(0.0, 1.0, n)
    ker = np.exp(-np.linspace(-3, 3, 41) ** 2)
    ker /= ker.sum()
    z += 0.55 * np.convolve(rough, ker, mode="same")
    return x, z


def taut_string(x, z, chord=MAX_CHORD):
    """Pipeline elevation: steepest reachable line from each support."""
    xm = x * 1000.0
    n = x.size
    pipe = np.empty(n)
    i, supports = 0, [0]
    while i < n - 1:
        j_end = int(np.searchsorted(xm, xm[i] + chord, side="right")) - 1
        j_end = max(j_end, i + 1)
        j_end = min(j_end, n - 1)
        j = np.arange(i + 1, j_end + 1)
        slope = (z[j] - z[i]) / (xm[j] - xm[i])
        nxt = int(j[np.argmax(slope)])
        pipe[i:nxt + 1] = z[i] + (xm[i:nxt + 1] - xm[i]) * slope.max()
        supports.append(nxt)
        i = nxt
    pipe[-1] = z[-1]
    return pipe, supports


def spans(x, z, pipe):
    gap = pipe - z
    free = gap > GAP_TOL
    out = []
    k = 0
    while k < x.size:
        if free[k]:
            j = k
            while j + 1 < x.size and free[j + 1]:
                j += 1
            length = (x[j] - x[k]) * 1000.0
            if length >= 20.0:
                mid = 0.5 * (x[k] + x[j])
                soil = next(s for a, b, s in SEGMENTS if a <= mid <= b)
                out.append(dict(i0=k, i1=j, km=mid, length=length,
                                soil=soil, gap=float(gap[k:j + 1].max())))
            k = j + 1
        else:
            k += 1
    return out


def routes(seed=3):
    rng = np.random.default_rng(seed)
    cost = np.array([182.0, 196.0, 205.0, 218.0, 232.0, 247.0, 263.0, 288.0])
    risk = 0.86 - 0.0021 * (cost - cost[0]) + rng.normal(0.0, 0.045, cost.size)
    risk = np.clip(risk, 0.12, 1.0)
    return cost, risk


def pareto(cost, risk):
    keep = []
    for i in range(cost.size):
        dominated = np.any((cost <= cost[i]) & (risk <= risk[i]) &
                           ((cost < cost[i]) | (risk < risk[i])))
        if not dominated:
            keep.append(i)
    return np.array(sorted(keep, key=lambda i: cost[i]))


def span_inset(ax, x, z, pipe, worst):
    """Magnified view of the governing span; a 2 m gap is invisible at 12 km."""
    a = x[worst["i0"]] - 0.36 * worst["length"] / 1000.0
    b = x[worst["i1"]] + 0.36 * worst["length"] / 1000.0
    m = (x >= a) & (x <= b)
    lo = z[m].min() - 1.1
    hi = max(pipe[m].max(), z[m].max()) + 1.1
    ax.add_patch(Rectangle((a, lo), b - a, hi - lo, facecolor="none",
                           edgecolor=RED, lw=1.0, zorder=8))

    axi = ax.inset_axes([0.395, 0.575, 0.315, 0.345])
    axi.set_zorder(9)
    axi.patch.set_visible(True)
    sl = slice(worst["i0"], worst["i1"] + 1)
    axi.fill_between((x[m] - a) * 1e3, lo, z[m], color=SOIL[worst["soil"]],
                     edgecolor="none", zorder=2)
    axi.plot((x[m] - a) * 1e3, z[m], color="#6f6f6f", lw=0.9, zorder=3)
    axi.fill_between((x[sl] - a) * 1e3, z[sl], pipe[sl], color=RED,
                     alpha=0.22, linewidth=0.0, zorder=4)
    axi.plot((x[m] - a) * 1e3, pipe[m], color=PIPE, lw=1.4, zorder=5)
    axi.plot((x[sl] - a) * 1e3, pipe[sl], color=RED, lw=2.2, zorder=6)
    axi.set_xlim(0.0, (b - a) * 1e3)
    axi.set_ylim(lo, hi)
    for s in axi.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.9)
        s.set_color(RED)
    axi.tick_params(labelsize=6.0, length=2.0)
    axi.set_xlabel("along route (m)", fontsize=6.2, labelpad=1.5)
    axi.set_ylabel("depth (m)", fontsize=6.2, labelpad=1.5)
    axi.set_title(f"governing span at KP {worst['km']:.1f}: "
                  f"{worst['length']:.0f} m long, {worst['gap']:.1f} m gap",
                  fontsize=6.5, color=RED, pad=2.5)
    return axi


def section_panel(ax, x, z, pipe, sp):
    top = z.max() + 26.0
    bottom = z.min() - 26.0
    ax.fill_between(x, z, top, color=WATER, edgecolor="none", zorder=1)
    for a, b, name in SEGMENTS:
        m = (x >= a) & (x <= b)
        ax.fill_between(x[m], bottom, z[m], color=SOIL[name], edgecolor="none",
                        zorder=2)
        ax.plot([b, b], [bottom, z[np.argmin(np.abs(x - b))]], color="white",
                lw=0.7, zorder=3)
    ax.plot(x, z, color="#6f6f6f", lw=0.9, zorder=4)
    ax.plot(x, pipe, color=PIPE, lw=1.6, zorder=6)
    for s in sp:
        sl = slice(s["i0"], s["i1"] + 1)
        ax.plot(x[sl], pipe[sl], color=RED, lw=2.6, solid_capstyle="round",
                zorder=7)
        ax.fill_between(x[sl], z[sl], pipe[sl], color=RED, alpha=0.20,
                        linewidth=0.0, zorder=5)

    span_inset(ax, x, z, pipe, max(sp, key=lambda s: s["length"]))
    ax.set_xlim(0.0, ROUTE_KM)
    ax.set_ylim(bottom, top)
    ax.set_xlabel("route kilometre post (km)")
    ax.set_ylabel("depth below sea level (m)")

    handles = [Line2D([], [], color=PIPE, lw=1.6, label="pipeline on seabed"),
               Line2D([], [], color=RED, lw=2.6, label="free span"),
               Line2D([], [], color="#6f6f6f", lw=0.9, label="seabed")]
    handles += [Patch(facecolor=SOIL[k], edgecolor="none", label=k)
                for k in SOIL]
    ax.legend(handles=handles, loc="lower left", fontsize=6.8, frameon=False,
              ncol=4, handlelength=1.5, borderaxespad=0.3,
              columnspacing=1.1)
    ax.set_title(f"{len(sp)} free spans over {ROUTE_KM:.0f} km of route",
                 fontsize=9.0, fontweight="bold", color=INK)


def span_panel(ax, sp):
    order = sorted(range(len(sp)), key=lambda i: sp[i]["km"])
    y = np.arange(len(order), dtype=float)
    length = np.array([sp[i]["length"] for i in order])
    allow = np.array([ALLOWABLE[sp[i]["soil"]] for i in order])
    over = length > allow

    ax.barh(y, length, height=0.62, color=np.where(over, RED, BLUE),
            edgecolor="white", linewidth=0.5, zorder=3)
    ax.plot(allow, y, ls="none", marker="|", ms=9.0, mew=1.4, color=RULE,
            zorder=5, label="allowable span")
    for yy, ln, al in zip(y, length, allow):
        ax.text(ln + 2.5, yy, f"{ln:.0f}", ha="left", va="center",
                fontsize=6.0, color=SUB)
    ax.set_yticks(y)
    ax.set_yticklabels([f"KP {sp[i]['km']:.1f}" for i in order], fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlim(0.0, max(length.max(), allow.max()) * 1.55)
    ax.set_xlabel("span length (m)", fontsize=7.4)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.7)
    handles = [Patch(facecolor=RED, edgecolor="none", label="over allowable"),
               Patch(facecolor=BLUE, edgecolor="none", label="acceptable"),
               Line2D([], [], ls="none", marker="|", ms=9.0, mew=1.4,
                      color=RULE, label="allowable")]
    ax.legend(handles=handles, fontsize=6.2, frameon=False, loc="lower right",
              handlelength=1.2, borderaxespad=0.2)
    ax.set_title(f"{int(over.sum())} spans need intervention", fontsize=8.2,
                 fontweight="bold", color=INK)
    return int(over.sum())


def tradeoff_panel(ax, cost, risk, front, pick):
    ax.plot(cost[front], risk[front], color=BLUE, lw=1.0, ls=(0, (4, 2.4)),
            zorder=3, label="Pareto set")
    ax.plot(cost, risk, ls="none", marker="o", ms=5.0, mfc="white", mec=RULE,
            mew=0.9, zorder=4)
    ax.plot(cost[front], risk[front], ls="none", marker="o", ms=5.0,
            mfc=BLUE, mec="white", mew=0.7, zorder=5)
    ax.plot([cost[pick]], [risk[pick]], ls="none", marker="o", ms=6.4,
            mfc=RED, mec="white", mew=0.8, zorder=6)
    ax.annotate(f"selected route\n{cost[pick]:.0f} M USD, risk "
                f"{risk[pick]:.2f}",
                xy=(cost[pick], risk[pick]),
                xytext=(cost[pick] + 8.0, risk[pick] + 0.18), ha="left",
                va="bottom", fontsize=6.5, color=RED, linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    ax.set_xlabel("installed cost (M USD)", fontsize=7.4)
    ax.set_ylabel("route risk index (-)", fontsize=7.4)
    ax.set_xlim(cost.min() - 14.0, cost.max() + 16.0)
    ax.set_ylim(0.0, 1.15)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right",
              handlelength=1.6, borderaxespad=0.2)
    ax.set_title("Route options, cost against risk", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    x, z = seabed()
    pipe, _ = taut_string(x, z)
    sp = spans(x, z, pipe)
    cost, risk = routes()
    front = pareto(cost, risk)
    pick = int(front[len(front) // 2])

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], left=0.070, right=0.985,
                          top=0.890, bottom=0.125, wspace=0.30, hspace=0.66)
    ax_sec = fig.add_subplot(gs[:, 0])
    ax_span = fig.add_subplot(gs[0, 1])
    ax_cr = fig.add_subplot(gs[1, 1])

    section_panel(ax_sec, x, z, pipe, sp)
    n_over = span_panel(ax_span, sp)
    tradeoff_panel(ax_cr, cost, risk, front, pick)

    bb = ax_sec.get_position()
    fw, fh = fig.get_size_inches()
    xs = np.ptp(ax_sec.get_xlim()) * 1000.0 / (bb.width * fw)
    ys = np.ptp(ax_sec.get_ylim()) / (bb.height * fh)
    ax_sec.text(0.985, 0.975, f"vertical exaggeration {xs / ys:.0f}x",
                transform=ax_sec.transAxes, ha="right", va="top",
                fontsize=6.8, color=MUTED)

    total = sum(s["length"] for s in sp)
    panel_labels_inches(fig, [ax_sec, ax_span, ax_cr], dx=-0.44, dy=0.12)
    savefig(fig, "659_oce_pipeline_route.png")


if __name__ == "__main__":
    main()

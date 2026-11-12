"""Evacuation flow on an office floor plate, and where the queue actually forms.

The main panel is the floor plan at equal aspect.  Each room sends its
occupants down the corridor to whichever exit is nearest along the walkable
path, and the corridor is drawn as two counter-flowing lanes whose stroke width
is the number of people crossing that segment.  The warm blobs are the queue
density that builds at each doorway once arrivals outrun the door's capacity,
so the bottleneck is visible as a shape rather than asserted in a caption.

Everything is computed: exit assignment is a shortest-path choice on the
corridor graph, discharge is the 1.3 persons/(m s) specific flow over an
effective width, arrivals combine a lognormal pre-movement distribution with
travel at walking speed, and the queue, RSET and the margin against ASET are
read out of the simulated cumulative curves.  Synthetic building, synthetic
occupants.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, despine, get_cmap
from figkit import axes_inches, panel_labels_inches, safe_colorbar, arrow

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
BASE = "#808080"
WALL = "#3a3a3a"
CORR = "#f0f0f0"

PLAN_W, PLAN_H = 30.0, 24.0        # m
CORR_Y0, CORR_Y1 = 11.0, 13.4      # main corridor, m
SPUR_X0, SPUR_X1 = 14.0, 16.4      # spur corridor, m
CY = 0.5 * (CORR_Y0 + CORR_Y1)
SX = 0.5 * (SPUR_X0 + SPUR_X1)
LANE = 0.55                        # m, lane offset from the corridor centre

DENSITY = 1.00                     # persons/m2, assembly-style loading
SPEED = 1.20                       # m/s, unimpeded walking speed
SPEC_FLOW = 1.30                   # persons/(m s)
BOUNDARY = 0.15                    # m, boundary layer lost on each door jamb
PRE_MOVE = 45.0                    # s, median pre-movement time
PRE_SIG = 0.28                     # lognormal shape of the pre-movement time
ASET = 240.0                       # s, design input from the smoke analysis
FAN_R = 5.0                        # m, radius of the queuing fan at a door
FLOW_WIN = 10.0                    # s, window the flow rate is averaged over

EXITS = {
    "E1": dict(name="west stair", xy=(1.0, CY), width=1.2, on="main",
               pos=1.0),
    "E2": dict(name="east stair", xy=(PLAN_W - 1.0, CY), width=1.5, on="main",
               pos=PLAN_W - 1.0),
    "E3": dict(name="south exit", xy=(SX, 1.0), width=2.2, on="spur",
               pos=1.0),
}

# name, rectangle (x0, y0, w, h), door ("main", x) or ("spur", y)
ROOMS = [
    ("N1", (1.0, CORR_Y1, 7.0, 23.0 - CORR_Y1), ("main", 4.5)),
    ("N2", (8.0, CORR_Y1, 7.0, 23.0 - CORR_Y1), ("main", 11.5)),
    ("N3", (15.0, CORR_Y1, 7.0, 23.0 - CORR_Y1), ("main", 18.5)),
    ("N4", (22.0, CORR_Y1, 7.0, 23.0 - CORR_Y1), ("main", 25.5)),
    ("S1", (1.0, 1.0, 6.0, CORR_Y0 - 1.0), ("main", 4.5)),
    ("S2", (7.5, 1.0, 6.5, CORR_Y0 - 1.0), ("spur", 6.0)),
    ("S3", (16.4, 1.0, 6.0, CORR_Y0 - 1.0), ("spur", 6.0)),
    ("S4", (23.0, 1.0, 6.0, CORR_Y0 - 1.0), ("main", 25.5)),
]


def eff_width(w):
    return max(w - 2.0 * BOUNDARY, 0.3)


def capacity(w):
    """Door discharge capacity in persons/s."""
    return SPEC_FLOW * eff_width(w)


def path_to(door, exit_id):
    """Walk distance from a door to an exit, and the polyline it follows."""
    kind, p = door
    ex = EXITS[exit_id]
    if kind == "main":
        if ex["on"] == "main":
            return abs(p - ex["pos"]), [("main", p, ex["pos"])]
        return (abs(p - SX) + (CY - ex["pos"]),
                [("main", p, SX), ("spur", CY, ex["pos"])])
    if ex["on"] == "spur":
        return abs(p - ex["pos"]), [("spur", p, ex["pos"])]
    return ((CY - p) + abs(SX - ex["pos"]),
            [("spur", p, CY), ("main", SX, ex["pos"])])


def assign(available):
    """Nearest-exit assignment plus the internal walk inside each room."""
    out = []
    for name, rect, door in ROOMS:
        x0, y0, w, h = rect
        n = int(round(DENSITY * w * h))
        best = min(available, key=lambda e: path_to(door, e)[0])
        dist, legs = path_to(door, best)
        # Internal travel: centroid to the door, plus a room-shape allowance.
        dx, dy = door
        d_door = (abs(x0 + w / 2.0 - dy) + h / 2.0 if dx == "main"
                  else abs(y0 + h / 2.0 - dy) + w / 2.0)
        out.append(dict(room=name, rect=rect, door=door, exit=best, n=n,
                        dist=dist + d_door, legs=legs))
    return out


def simulate(available, widths, rng, t):
    """Queue-served evacuation; returns per-exit cumulative and flow arrays."""
    rooms = assign(available)
    arrivals = {e: [] for e in available}
    for r in rooms:
        pre = PRE_MOVE * np.exp(PRE_SIG * rng.normal(size=r["n"]))
        arrivals[r["exit"]].append(pre + r["dist"] / SPEED)
    dt = t[1] - t[0]
    cum, flow, queue = {}, {}, {}
    for e in available:
        arr = np.sort(np.concatenate(arrivals[e])) if arrivals[e] else \
            np.array([0.0])
        cap = capacity(widths[e])
        served = 0.0
        c = np.empty_like(t)
        f = np.empty_like(t)
        q = np.empty_like(t)
        for i, ti in enumerate(t):
            waiting = float(np.searchsorted(arr, ti)) - served
            out = min(cap * dt, max(waiting, 0.0))
            served += out
            c[i], f[i], q[i] = served, out / dt, max(waiting - out, 0.0)
        cum[e], flow[e], queue[e] = c, f, q
    return rooms, cum, flow, queue


def rset_of(t, total_cum, n_total):
    return float(np.interp(n_total - 0.5, total_cum, t))


def smooth_flow(f, dt):
    """Boxcar mean over FLOW_WIN.

    People leave one at a time, so an instantaneous rate sampled every half
    second is a square wave between zero and one person per step.  Real exit
    surveys quote a rate averaged over a window, and so does this panel.
    """
    k = max(int(round(FLOW_WIN / dt)), 1)
    return np.convolve(f, np.ones(k) / k, mode="same")


def draw_plan(ax, rooms, queue, widths, t):
    ax.add_patch(Rectangle((0.0, 0.0), PLAN_W, PLAN_H, facecolor="white",
                           edgecolor=WALL, lw=1.6, zorder=1))
    ax.add_patch(Rectangle((1.0, CORR_Y0), PLAN_W - 2.0, CORR_Y1 - CORR_Y0,
                           facecolor=CORR, edgecolor="none", zorder=1))
    ax.add_patch(Rectangle((SPUR_X0, 1.0), SPUR_X1 - SPUR_X0, CORR_Y0 - 1.0,
                           facecolor=CORR, edgecolor="none", zorder=1))
    for r in rooms:
        x0, y0, w, h = r["rect"]
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor="white",
                               edgecolor="#8a8a8a", lw=0.7, zorder=2))
        ax.text(x0 + w / 2.0, y0 + h - 0.7, f"{r['room']}  {r['n']} persons",
                ha="center", va="top", fontsize=6.4, color=SUB, zorder=6)

    # Queue density field: a Gaussian reservoir at each door, only contoured
    # above 0.5 persons/m2 so the rest of the plan stays clean.
    gx = np.linspace(0.0, PLAN_W, 300)
    gy = np.linspace(0.0, PLAN_H, 240)
    gxx, gyy = np.meshgrid(gx, gy)
    dens = np.zeros_like(gxx)
    peaks = {}
    fan_area = 0.5 * np.pi * FAN_R ** 2
    for e, spec in EXITS.items():
        if e not in queue:
            continue
        pk = float(queue[e].max()) / fan_area
        peaks[e] = pk
        ex, ey = spec["xy"]
        sig = FAN_R / 1.7
        dens += pk * np.exp(-((gxx - ex) ** 2 + (gyy - ey) ** 2)
                            / (2.0 * sig ** 2))
    d_max = float(dens.max())
    levels = np.linspace(0.5, max(d_max, 0.75), 8)
    cf = ax.contourf(gxx, gyy, dens, levels=levels,
                     cmap=get_cmap("cb_sequential_warm"), alpha=0.72,
                     extend="max", zorder=3)
    cf.set_edgecolor("face")           # no hairline seams between bands

    # Corridor lanes: stroke width is the number of people on the segment.
    main_nodes = sorted({1.0, PLAN_W - 1.0, SX}
                        | {d[1] for _, _, d in ROOMS if d[0] == "main"})
    spur_nodes = sorted({1.0, CY}
                        | {d[1] for _, _, d in ROOMS if d[0] == "spur"})
    m_seg = list(zip(main_nodes[:-1], main_nodes[1:]))
    s_seg = list(zip(spur_nodes[:-1], spur_nodes[1:]))
    m_flow = {"+": np.zeros(len(m_seg)), "-": np.zeros(len(m_seg))}
    s_flow = {"+": np.zeros(len(s_seg)), "-": np.zeros(len(s_seg))}
    for r in rooms:
        for kind, a, b in r["legs"]:
            segs, store = (m_seg, m_flow) if kind == "main" else (s_seg, s_flow)
            key = "+" if b >= a else "-"
            lo, hi = min(a, b), max(a, b)
            for i, (s0, s1) in enumerate(segs):
                if s0 >= lo - 1e-6 and s1 <= hi + 1e-6:
                    store[key][i] += r["n"]

    n_max = max(m_flow["+"].max(), m_flow["-"].max(), s_flow["+"].max(),
                s_flow["-"].max(), 1.0)

    def lw_of(n):
        return 0.9 + 4.6 * (n / n_max)

    for i, (s0, s1) in enumerate(m_seg):
        if m_flow["+"][i] > 0:
            arrow(ax, (s0, CY + LANE), (s1, CY + LANE), color=BLUE,
                  lw=lw_of(m_flow["+"][i]), ms=8, zorder=5, alpha=0.9)
        if m_flow["-"][i] > 0:
            arrow(ax, (s1, CY - LANE), (s0, CY - LANE), color=BLUE,
                  lw=lw_of(m_flow["-"][i]), ms=8, zorder=5, alpha=0.9)
    for i, (s0, s1) in enumerate(s_seg):
        if s_flow["-"][i] > 0:
            arrow(ax, (SX + LANE, s1), (SX + LANE, s0), color=BLUE,
                  lw=lw_of(s_flow["-"][i]), ms=8, zorder=5, alpha=0.9)
        if s_flow["+"][i] > 0:
            arrow(ax, (SX - LANE, s0), (SX - LANE, s1), color=BLUE,
                  lw=lw_of(s_flow["+"][i]), ms=8, zorder=5, alpha=0.9)

    # Room stubs: centroid to the lane that carries that room away.
    for r in rooms:
        x0, y0, w, h = r["rect"]
        kind, p = r["door"]
        first = r["legs"][0]
        going = "+" if first[2] >= first[1] else "-"
        if kind == "main":
            tgt = (p, CY + (LANE if going == "+" else -LANE))
        else:
            tgt = (SX + (LANE if going == "-" else -LANE), p)
        arrow(ax, (x0 + w / 2.0, y0 + h / 2.0), tgt, color=BLUE, lw=0.8,
              ms=7, zorder=4, alpha=0.75, gap_b=0.10)

    return peaks, d_max, cf


def label_exits(ax, rooms, cum, widths, t, peaks):
    tallies = {}
    for r in rooms:
        tallies[r["exit"]] = tallies.get(r["exit"], 0) + r["n"]
    clear = {e: float(np.interp(tallies[e] - 0.5, cum[e], t)) for e in cum}
    worst = max(clear, key=lambda e: clear[e])
    for e, spec in EXITS.items():
        if e not in cum:
            continue
        ex, ey = spec["xy"]
        col = RED if e == worst else INK
        ax.plot([ex], [ey], marker="s", ms=6.0, mfc="white", mec=col,
                mew=1.4, zorder=8)
        txt = (f"{e} {spec['name']}, {widths[e]:.1f} m\n{tallies[e]} persons, "
               f"clears {clear[e]:.0f} s"
               + ("\nBOTTLENECK" if e == worst else ""))
        if e == "E1":
            xy, ha, va = (ex + 0.6, ey + 4.6), "left", "bottom"
        elif e == "E2":
            xy, ha, va = (ex - 0.6, ey + 4.6), "right", "bottom"
        else:
            xy, ha, va = (ex + 2.0, ey + 0.2), "left", "bottom"
        ax.annotate(txt, xy=(ex, ey), xytext=xy, ha=ha, va=va, fontsize=6.4,
                    color=col, linespacing=1.35, zorder=9,
                    bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                              edgecolor="#cccccc", lw=0.6, alpha=0.94),
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.7,
                                    shrinkA=2.0, shrinkB=3.0))
    return tallies, clear, worst


def cum_panel(ax, t, scenarios):
    for s in scenarios:
        ax.plot(t, s["total"], color=s["color"], lw=1.8, zorder=5)
        ax.plot([s["rset"]], [s["n"]], marker="o", ms=4.0, mfc="white",
                mec=s["color"], mew=1.1, zorder=7)
    ax.axvline(ASET, color=RED, lw=1.1, ls=(0, (4, 2.4)), zorder=4)
    ax.text(ASET - 5.0, 0.70, f"ASET {ASET:.0f} s",
            transform=ax.get_xaxis_transform(), ha="center", va="center",
            fontsize=6.4, color=RED, rotation=90)
    base = scenarios[0]
    y_m = base["n"] * 0.40
    ax.annotate("", xy=(base["rset"], y_m), xytext=(ASET, y_m),
                arrowprops=dict(arrowstyle="<|-|>", color=SUB, lw=0.8,
                                mutation_scale=7, shrinkA=0, shrinkB=0))
    ax.text(0.5 * (base["rset"] + ASET), y_m + base["n"] * 0.035,
            f"margin {ASET - base['rset']:.0f} s", ha="center", va="bottom",
            fontsize=6.5, color=SUB)
    ax.set_xlim(0.0, t[-1])
    ax.set_ylim(0.0, max(s["n"] for s in scenarios) * 1.12)
    ax.set_xlabel("time from alarm (s)", fontsize=7.6)
    ax.set_ylabel("occupants evacuated (persons)", fontsize=7.6)
    ax.tick_params(labelsize=6.8)
    despine(ax)
    ax.grid(axis="y", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend([plt.Line2D([], [], color=s["color"], lw=1.6)
               for s in scenarios],
              [f"{s['label']}: RSET {s['rset']:.0f} s" for s in scenarios],
              fontsize=6.4, frameon=False, loc="lower right",
              handlelength=1.7, labelspacing=0.3)
    ax.set_title("Only widening the bottleneck moves RSET", fontsize=8.4,
                 fontweight="bold", color=INK)


def flow_panel(ax, t, flow, widths):
    cols = {"E1": RED, "E2": BLUE, "E3": GREEN}
    dt = t[1] - t[0]
    for e in sorted(flow):
        cap = capacity(widths[e])
        ax.plot(t, smooth_flow(flow[e], dt), color=cols[e], lw=1.5, zorder=5,
                label=f"{e} ({widths[e]:.1f} m), cap {cap:.2f} persons/s")
        ax.axhline(cap, color=cols[e], lw=0.7, ls=(0, (1.6, 1.8)), zorder=4)
    ax.set_xlim(0.0, t[-1])
    ax.set_ylim(0.0, max(capacity(w) for w in widths.values()) * 1.42)
    ax.set_xlabel("time from alarm (s)", fontsize=7.6)
    ax.set_ylabel("exit flow rate (persons/s)", fontsize=7.6)
    ax.tick_params(labelsize=6.8)
    despine(ax)
    ax.grid(axis="y", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.3, frameon=False, loc="upper right",
              handlelength=1.7, labelspacing=0.3)
    ax.text(0.975, 0.80, f"flat tops sit on the dashed capacity\nlines; rates "
                         f"are {FLOW_WIN:.0f} s means",
            transform=ax.transAxes, ha="right", va="center", fontsize=6.4,
            color=SUB, linespacing=1.4)
    ax.set_title("Every door runs flat out at its own capacity",
                 fontsize=8.4, fontweight="bold", color=INK)


def main():
    set_pub_style()
    rng = np.random.default_rng(85403)
    t = np.arange(0.0, 400.0, 0.5)

    base_w = {e: EXITS[e]["width"] for e in EXITS}
    wide_w = dict(base_w, E1=2.2)

    runs = []
    for label, avail, widths, color in (
            ("all exits open", ["E1", "E2", "E3"], base_w, INK),
            ("east stair lost", ["E1", "E3"], base_w, GOLD),
            ("west stair widened to 2.2 m", ["E1", "E2", "E3"], wide_w, GREEN)):
        rooms, cum, flow, queue = simulate(avail, widths, rng, t)
        total = np.sum([cum[e] for e in cum], axis=0)
        n = sum(r["n"] for r in rooms)
        runs.append(dict(label=label, color=color, rooms=rooms, cum=cum,
                         flow=flow, queue=queue, widths=widths, total=total,
                         n=n, rset=rset_of(t, total, n)))

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_plan = axes_inches(fig, (0.80, 0.95, 5.00, 4.40))
    ax_cum = axes_inches(fig, (7.22, 3.45, 2.78, 1.95))
    ax_flow = axes_inches(fig, (7.22, 0.95, 2.78, 1.95))

    base = runs[0]
    ax_plan.set_xlim(-0.6, PLAN_W + 0.6)
    ax_plan.set_ylim(-0.6, PLAN_H + 0.6)
    ax_plan.set_aspect("equal")
    peaks, d_max, cf = draw_plan(ax_plan, base["rooms"], base["queue"],
                                 base["widths"], t)
    tallies, clear, worst = label_exits(ax_plan, base["rooms"], base["cum"],
                                        base["widths"], t, peaks)
    ax_plan.set_xlabel("plan east-west (m)", fontsize=8.0)
    ax_plan.set_ylabel("plan north-south (m)", fontsize=8.0)
    ax_plan.tick_params(labelsize=7.0)
    for s in ax_plan.spines.values():
        s.set_visible(False)
    ax_plan.set_title("Flow concentrates on the narrow west stair",
                      fontsize=9.0, fontweight="bold", color=INK)
    safe_colorbar(fig, cf, ax_plan,
                  label="queue density (persons/m$^{2}$)", width=0.12,
                  pad=0.10, shrink=0.80)

    cum_panel(ax_cum, t, runs)
    flow_panel(ax_flow, t, base["flow"], base["widths"])

    share = 100.0 * tallies[worst] / base["n"]
    panel_labels_inches(fig, [ax_plan, ax_cum, ax_flow], dx=-0.52, dy=0.14)
    savefig(fig, "854_fir_evacuation_flow.png")


if __name__ == "__main__":
    main()

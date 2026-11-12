"""Figure 1194 - Continuous batching: one timeline, and the sweep it came from.

``simulate`` is a small event-driven scheduler: four GPUs pull from one queue,
each holds at most ``B`` sequences, an iteration over ``b`` active sequences
costs ``T0 + T1 * b`` milliseconds, and a sequence that has held its slot for
``QUANTUM`` steps while others wait is preempted and requeued.

The same request stream is replayed at every batch limit, so the gantt in (a)
and the two curves in (b) are the *same* function of ``B`` - the marked point
on the curves is literally the run drawn above it.  Idle time is not measured
twice either: the bubbles drawn in (a) are asserted to add up to the idle term
of the utilisation printed in the footer.

Lane rows drive the canvas height, which is the DESIGN_SYSTEM section 1.2 E5
exception; the reference line at the top of (b) is drawn with explicit end
points, so no label can be struck through.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

from pubstyle import set_pub_style, savefig, style_axes

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#d5d5d5"
GRID, CARD, BUBBLE = "#ededed", "#e2e8ee", "#dde4ea"
# request length class -> fill, hatch, label colour
CLASS = [("short  < 40 steps", "#cfe1ee", "...", INK),
         ("medium  40-69 steps", "#5f9dc6", "", "white"),
         ("long  >= 70 steps", "#134a76", "", "white")]
SEED, NGPU, NREQ = 1599, 4, 30
T0, T1 = 8.0, 1.6          # ms of fixed and per-sequence iteration cost
QUANTUM = 26               # steps a sequence may hold a slot while others wait
SHOWN_B = 4
SWEEP = [1, 2, 4, 8, 16, 32]


def stream(seed=SEED):
    """One arrival stream, replayed at every batch limit."""
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(32.0, NREQ)
    arr = 20.0 + np.cumsum(gaps) - gaps[0]
    steps = rng.integers(16, 96, NREQ)
    return [(float(a), int(s)) for a, s in zip(arr, steps)]


def simulate(B, reqs):
    """Event-driven continuous batching over NGPU workers."""
    queue = [dict(rid=i, arr=a, left=s, steps=s, served=0)
             for i, (a, s) in enumerate(reqs)]
    queue.sort(key=lambda r: r["arr"])
    gpus = [dict(t=0.0, act=[], busy=0.0, gaps=[], slots=[None] * B)
            for _ in range(NGPU)]
    bars, preempts, lat = [], [], [None] * len(reqs)

    while True:
        live = [g for g in gpus if g["act"] or queue]
        if not live:
            break
        g = min(live, key=lambda z: z["t"])
        gi = gpus.index(g)
        t = g["t"]
        while len(g["act"]) < B and queue and queue[0]["arr"] <= t:
            r = queue.pop(0)
            r["slot"] = g["slots"].index(None)
            r["start"], r["gpu"], r["served"] = t, gi, 0
            g["slots"][r["slot"]] = r
            g["act"].append(r)
        if not g["act"]:
            nxt = queue[0]["arr"]
            if nxt > t:
                g["gaps"].append((t, nxt))
            g["t"] = max(t, nxt)
            continue
        dt = T0 + T1 * len(g["act"])
        for r in g["act"]:
            r["left"] -= 1
            r["served"] += 1
        g["busy"] += dt
        g["t"] = t + dt
        for r in list(g["act"]):
            if r["left"] <= 0:
                bars.append((gi, r["slot"], r["start"], g["t"], r["steps"],
                             False))
                lat[r["rid"]] = g["t"] - reqs[r["rid"]][0]
                g["act"].remove(r)
                g["slots"][r["slot"]] = None
        waiting = any(q["arr"] <= g["t"] for q in queue)
        if waiting and len(g["act"]) == B:
            cand = [r for r in g["act"] if r["served"] >= QUANTUM]
            if cand:
                r = max(cand, key=lambda z: z["served"])
                bars.append((gi, r["slot"], r["start"], g["t"], r["steps"],
                             True))
                preempts.append((gi, r["slot"], g["t"]))
                g["act"].remove(r)
                g["slots"][r["slot"]] = None
                r["arr"] = g["t"]
                queue.append(r)
                queue.sort(key=lambda z: z["arr"])

    span = max(g["t"] for g in gpus)
    bubbles = []
    for gi, g in enumerate(gpus):
        for (a, b) in g["gaps"]:
            bubbles.append((gi, a, b))
        if g["t"] < span:
            bubbles.append((gi, g["t"], span))
    busy = sum(g["busy"] for g in gpus)
    return dict(bars=bars, preempts=preempts, bubbles=bubbles, span=span,
                busy=busy, lat=np.array([v for v in lat], dtype=float),
                B=B)


def klass(steps):
    return 0 if steps < 40 else (1 if steps < 70 else 2)


def main():
    set_pub_style(base_font=8.5)
    reqs = stream()
    runs = {B: simulate(B, reqs) for B in SWEEP}
    run = runs[SHOWN_B]

    total = NGPU * run["span"]
    idle = sum(b - a for (_, a, b) in run["bubbles"])
    assert abs(total - run["busy"] - idle) < 1e-6, \
        "the drawn bubbles do not add up to the idle time in the footer"
    # Everything printed below is rounded once, in seconds, then reused - and
    # at three decimals, because a reader multiplying a 2-decimal span by four
    # lands 0.02 s away from a 2-decimal total.
    s_span = f"{run['span'] / 1000:.3f}"
    s_tot = f"{NGPU * float(s_span):.3f}"
    s_busy = f"{run['busy'] / 1000:.3f}"
    s_idle = f"{float(s_tot) - float(s_busy):.3f}"
    util = float(s_busy) / float(s_tot)
    thru = {B: 1000.0 * len(reqs) / r["span"] for B, r in runs.items()}
    p99 = {B: float(np.percentile(r["lat"], 99)) / 1000.0
           for B, r in runs.items()}
    # both optima are decided on the numbers that get printed on the curves
    shown_t = {B: f"{thru[B]:.1f}" for B in SWEEP}
    shown_p = {B: f"{p99[B]:.2f}" for B in SWEEP}
    sat = min(SWEEP, key=lambda B: (-float(shown_t[B]), B))
    knee = min(SWEEP, key=lambda B: (float(shown_p[B]), B))
    assert knee == sat == SHOWN_B, \
        "the drawn run is no longer the one both curves pick"
    last = SWEEP[-1]
    assert float(shown_p[last]) > float(shown_p[knee]), \
        "p99 no longer turns back up, so the headline is wrong"
    assert run["preempts"], "the drawn run has no preemption to label"
    assert run["bubbles"], "the drawn run has no bubble to label"

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.66], left=0.072,
                          right=0.986, top=0.815, bottom=0.150, hspace=0.42)
    axg = fig.add_subplot(gs[0, 0])
    axs = fig.add_subplot(gs[1, 0])
    xmax = run["span"] / 1000.0 * 1.008

    # ---- (a) the timeline ---------------------------------------------------
    style_axes(axg)
    axg.set_axisbelow(True)
    axg.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    lane_h, slot_h = 1.0, 0.94 / SHOWN_B
    for gi in range(NGPU):
        base = NGPU - 1 - gi
        axg.add_patch(Rectangle((0.0, base + 0.03), xmax, 0.94,
                                facecolor="#f7f9fb", edgecolor="none",
                                zorder=1))
    for (gi, a, b) in run["bubbles"]:
        base = NGPU - 1 - gi
        axg.add_patch(Rectangle((a / 1000.0, base + 0.03),
                                (b - a) / 1000.0, 0.94, facecolor=BUBBLE,
                                edgecolor="#b9c5ce", linewidth=0.5,
                                hatch="///", zorder=2))
    for (gi, slot, a, b, steps, cut) in run["bars"]:
        base = NGPU - 1 - gi
        y = base + 0.03 + slot * slot_h
        _, col, hat, _ = CLASS[klass(steps)]
        axg.add_patch(Rectangle((a / 1000.0, y + slot_h * 0.10),
                                (b - a) / 1000.0, slot_h * 0.80,
                                facecolor=col, edgecolor="white",
                                linewidth=0.6, hatch=hat, zorder=4))
    for (gi, slot, t) in run["preempts"]:
        base = NGPU - 1 - gi
        y = base + 0.03 + slot * slot_h
        # drawn twice so the tick survives on both the dark and the pale fills
        axg.plot([t / 1000.0, t / 1000.0],
                 [y + slot_h * 0.02, y + slot_h * 0.98], color="white",
                 lw=2.8, zorder=6, solid_capstyle="butt")
        axg.plot([t / 1000.0, t / 1000.0],
                 [y + slot_h * 0.02, y + slot_h * 0.98], color="#0b3d63",
                 lw=1.2, zorder=7, solid_capstyle="butt")
    axg.set_xlim(0.0, xmax)
    axg.set_ylim(0.0, NGPU)
    axg.set_yticks([NGPU - 0.5 - i for i in range(NGPU)])
    axg.set_yticklabels([f"GPU {i}" for i in range(NGPU)], fontsize=8.0)
    axg.set_xlabel("wall-clock time  (s)", fontsize=8.6)
    axg.tick_params(labelsize=7.4, length=2.4)
    axg.set_title(f"(a)  One run of the scheduler at batch limit B = "
                  f"{SHOWN_B}: {len(reqs)} requests, {NGPU} GPUs, "
                  f"{SHOWN_B} slots each", fontsize=9.4, fontweight="bold",
                  pad=5.0, loc="left")
    handles = [Patch(facecolor=c, edgecolor="white", hatch=h, label=n)
               for (n, c, h, _) in CLASS]
    handles += [Patch(facecolor=BUBBLE, edgecolor="#b9c5ce", hatch="///",
                      label="bubble (no work on that GPU)"),
                Line2D([], [], color="#0b3d63", lw=1.7,
                       label=f"preemption after {QUANTUM} steps")]
    leg = axg.legend(handles=handles, loc="lower left",
                     bbox_to_anchor=(0.0, 1.085), ncol=5, frameon=True,
                     facecolor="white", edgecolor=HAIR, framealpha=0.95,
                     fontsize=7.6, handlelength=1.5, columnspacing=1.4,
                     borderpad=0.42)
    leg.get_frame().set_linewidth(0.5)

    # ---- (b) the sweep the timeline came from -------------------------------
    style_axes(axs)
    axs.set_axisbelow(True)
    axs.grid(color=GRID, lw=0.7, zorder=0)
    xi = np.arange(len(SWEEP), dtype=float)
    tv = [thru[B] for B in SWEEP]
    axs.plot(xi, tv, color="#134a76", lw=1.8, marker="o", ms=5.6,
             mfc="white", mec="#134a76", mew=1.5, zorder=5)
    axs.set_xticks(xi)
    axs.set_xticklabels([str(B) for B in SWEEP], fontsize=7.8)
    axs.set_xlim(-0.32, len(SWEEP) - 0.68)
    axs.set_ylim(0.0, max(tv) * 1.26)
    axs.set_xlabel("batch limit  B  (sequences per GPU)", fontsize=8.6)
    axs.set_ylabel("throughput\n(requests / s)", fontsize=8.4,
                   linespacing=1.5, color="#134a76")
    axs.tick_params(labelsize=7.4, length=2.4)
    axs.tick_params(axis="y", colors="#134a76")
    axs.set_title("(b)  The same request stream replayed at six batch limits",
                  fontsize=9.4, fontweight="bold", pad=5.0, loc="left")

    axp = axs.twinx()
    axp.spines["top"].set_visible(False)
    pv = [p99[B] for B in SWEEP]
    axp.plot(xi, pv, color="#5f9dc6", lw=1.5, ls=(0, (4.0, 2.0)), marker="s",
             ms=5.0, mfc="white", mec="#5f9dc6", mew=1.4, zorder=5)
    axp.set_ylim(0.0, max(pv) * 1.26)
    axp.set_ylabel("p99 latency\n(s)", fontsize=8.4, linespacing=1.5,
                   color="#3d6b8e")
    axp.tick_params(labelsize=7.4, length=2.4, colors="#3d6b8e")

    k = SWEEP.index(SHOWN_B)
    axs.plot([xi[k], xi[k]], [0.0, tv[k]], color="#4d4d4d", lw=0.8,
             ls=(0, (2.4, 2.0)), zorder=4)
    axs.plot([xi[k]], [tv[k]], marker="o", ms=8.5, mfc="none", mec="#0b3d63",
             mew=1.6, zorder=7)
    axs.text(xi[k] + 0.20, tv[k] * 1.12, "the timeline in (a)", ha="left",
             va="bottom", fontsize=7.8, color="#0b3d63", fontweight="bold",
             zorder=8,
             bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=HAIR,
                       lw=0.6, alpha=0.96))
    for i, B in enumerate(SWEEP):
        axs.text(xi[i], tv[i] + max(tv) * 0.045, shown_t[B], ha="center",
                 va="bottom", fontsize=7.2, color="#134a76", zorder=6)
    ki = SWEEP.index(knee)
    axp.text(xi[ki], pv[ki] - max(pv) * 0.075,
             f"p99 floor  {shown_p[knee]} s", ha="center", va="top",
             fontsize=7.4, color="#3d6b8e", zorder=8,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.3,
                       alpha=0.94))
    axp.text(xi[-1], pv[-1] + max(pv) * 0.055, f"back to {shown_p[last]} s",
             ha="right", va="bottom", fontsize=7.4, color="#3d6b8e", zorder=8,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.3,
                       alpha=0.94))

    lines = [Line2D([], [], color="#134a76", lw=1.8, marker="o", ms=5.4,
                    mfc="white", mec="#134a76", mew=1.5,
                    label="throughput (left)"),
             Line2D([], [], color="#5f9dc6", lw=1.5, ls=(0, (4.0, 2.0)),
                    marker="s", ms=5.0, mfc="white", mec="#5f9dc6", mew=1.4,
                    label="p99 latency (right)")]
    leg2 = axs.legend(handles=lines, loc="lower right",
                      bbox_to_anchor=(0.988, 0.030), frameon=True,
                      facecolor="white", edgecolor=HAIR, framealpha=0.95,
                      fontsize=7.6, handlelength=2.0, labelspacing=0.32,
                      borderpad=0.42)
    leg2.get_frame().set_linewidth(0.5)
    leg2.set_zorder(9)

    # ---- headline and the arithmetic ---------------------------------------

    band = FancyBboxPatch((0.022, 0.012), 0.956, 0.058,
                          boxstyle="round,pad=0.004,rounding_size=0.008",
                          transform=fig.transFigure, facecolor=CARD,
                          edgecolor="#c3cdd6", linewidth=0.6, zorder=-5)
    fig.add_artist(band)
    band.set_in_layout(False)
    savefig(fig, "1194_wm_sys_batch_schedule.png")


if __name__ == "__main__":
    main()

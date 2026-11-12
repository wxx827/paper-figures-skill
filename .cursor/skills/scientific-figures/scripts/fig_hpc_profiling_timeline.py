"""Trace timeline of one solver step across 16 MPI ranks.

Each row is a rank and each block is a state: compute, point-to-point
communication, file I/O, or idle waiting. Because every phase ends in a
collective, the phase cannot close until its slowest rank arrives, so the
idle blocks are not scattered noise - they are the shadow cast by one block
per phase. Those blocks are outlined in house red and joined into the
critical path, and the stacked bars underneath convert the same trace into
the share of rank-time each phase spends in each state.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

COMPUTE = "#3b5b92"    # house blue
COMM = "#c9a227"       # house gold
IO = "#2e7d5b"         # house green
IDLE = "#b23a48"       # house red, drawn light because it is mostly area
CRIT = "#b23a48"       # the single emphasis object
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"

STATE_COLOR = {"compute": COMPUTE, "comm": COMM, "io": IO, "idle": IDLE}
STATE_ALPHA = {"compute": 1.0, "comm": 1.0, "io": 1.0, "idle": 0.30}
STATE_ORDER = ["compute", "comm", "io", "idle"]

NR = 16
COLL = 0.22            # ms spent inside the collective itself
PHASES = [("Read restart", "io", "read barrier"),
          ("Iteration 1", "solve", "allreduce 1"),
          ("Iteration 2", "solve", "allreduce 2"),
          ("Iteration 3", "solve", "allreduce 3"),
          ("Write fields", "io", "write barrier")]


def build_trace(rng):
    """Return the block list, the sync times and the per-phase spans."""
    blocks, syncs, spans, crit = [], [], [], []
    t0 = 0.0
    for name, kind, label in PHASES:
        if kind == "io":
            work = rng.uniform(1.05, 1.95, NR)
            parts = [[("io", w)] for w in work]
        else:
            comp = rng.uniform(2.10, 3.05, NR)
            comp[rng.integers(0, NR)] *= 1.22       # one straggler per phase
            halo = rng.uniform(0.26, 0.42, NR)
            parts = [[("compute", c), ("comm", h)] for c, h in zip(comp, halo)]
        dur = np.array([sum(d for _, d in p) for p in parts])
        span = float(dur.max())
        lead = int(np.argmax(dur))
        for r, p in enumerate(parts):
            t = t0
            for state, d in p:
                blocks.append(dict(rank=r, t0=t, dur=d, state=state,
                                   lead=(r == lead)))
                t += d
            if span - dur[r] > 1e-9:
                blocks.append(dict(rank=r, t0=t, dur=span - dur[r],
                                   state="idle", lead=False))
        syncs.append((t0 + span, label))
        for r in range(NR):
            blocks.append(dict(rank=r, t0=t0 + span, dur=COLL, state="comm",
                               lead=False))
        crit.append((name, t0, span, lead))
        spans.append((name, t0, span + COLL))
        t0 += span + COLL
    return blocks, syncs, spans, crit, t0


def main():
    set_pub_style()
    rng = np.random.default_rng(5681)
    blocks, syncs, spans, crit, makespan = build_trace(rng)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.42], hspace=0.42)
    ax = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    # ---- (a) timeline -----------------------------------------------------
    for b in blocks:
        ax.add_patch(Rectangle((b["t0"], b["rank"] - 0.38), b["dur"], 0.76,
                               facecolor=STATE_COLOR[b["state"]],
                               alpha=STATE_ALPHA[b["state"]],
                               edgecolor="none", zorder=2))
    for b in blocks:
        if b["lead"] and b["state"] in ("compute", "io"):
            ax.add_patch(Rectangle((b["t0"], b["rank"] - 0.38), b["dur"], 0.76,
                                   facecolor="none", edgecolor=CRIT,
                                   linewidth=1.3, zorder=6))

    span_tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for t, label in syncs:
        ax.plot([t, t], [-0.5, NR - 0.5], color=RULE, ls="--", lw=0.8,
                alpha=0.65, zorder=5)
        ax.text(t, 1.015, label, transform=span_tr, ha="center", va="bottom",
                fontsize=6.4, color=SUB)

    # critical path: the phase leader, joined rank to rank
    px, py = [], []
    for name, t0, span, lead in crit:
        px += [t0, t0 + span]
        py += [lead, lead]
    ax.plot(px, py, color=CRIT, lw=1.0, ls="-", alpha=0.75, zorder=7,
            solid_joinstyle="round")
    ax.plot(px, py, color=CRIT, lw=0.0, marker="o", ms=2.6, zorder=7)

    crit_len = sum(s for _, _, s, _ in crit)
    ax.annotate(f"critical path {crit_len:.2f} ms\n"
                f"({100.0 * crit_len / makespan:.0f} % of the "
                f"{makespan:.2f} ms step)",
                xy=(crit[-1][1] + crit[-1][2], crit[-1][3]),
                xytext=(1.002, 0.30), textcoords="axes fraction",
                ha="left", va="center", fontsize=6.8, color=CRIT,
                linespacing=1.35, annotation_clip=False,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=CRIT,
                                shrinkA=2.0, shrinkB=2.0))

    ax.set_xlim(-0.15, makespan + 0.15)
    ax.set_ylim(NR - 0.5, -0.5)
    ax.set_yticks(np.arange(0, NR, 2))
    ax.set_yticklabels([f"rank {r}" for r in range(0, NR, 2)])
    ax.tick_params(labelsize=7.0, length=2.6)
    ax.set_xlabel("wall-clock time (ms)", fontsize=8.0)
    ax.grid(axis="x", lw=0.45, alpha=0.40)
    ax.set_axisbelow(True)

    handles = [Patch(facecolor=STATE_COLOR[s], alpha=STATE_ALPHA[s],
                     label=s.capitalize() if s != "io" else "File I/O")
               for s in STATE_ORDER]
    handles.append(Line2D([], [], color=CRIT, lw=1.3, label="Critical path"))

    # ---- (b) state share per phase ---------------------------------------
    names = [s[0] for s in spans]
    share = np.zeros((len(spans), len(STATE_ORDER)))
    for k, (name, t0, span) in enumerate(spans):
        for b in blocks:
            if t0 - 1e-9 <= b["t0"] < t0 + span - 1e-9:
                share[k, STATE_ORDER.index(b["state"])] += b["dur"]
        share[k] *= 100.0 / (NR * span)

    y = np.arange(len(spans))
    left = np.zeros(len(spans))
    for j, s in enumerate(STATE_ORDER):
        ax_b.barh(y, share[:, j], left=left, height=0.60,
                  color=STATE_COLOR[s], alpha=STATE_ALPHA[s],
                  edgecolor="white", linewidth=0.6)
        for k in range(len(spans)):
            if share[k, j] >= 7.0:
                ax_b.text(left[k] + share[k, j] / 2.0, y[k],
                          f"{share[k, j]:.0f}", ha="center", va="center",
                          fontsize=6.2,
                          color="white" if s != "idle" else CRIT)
        left += share[:, j]

    ax_b.set_yticks(y)
    ax_b.set_yticklabels([f"{n}  ({s:.2f} ms)" for n, (_, _, s) in
                          zip(names, spans)], fontsize=6.8)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0.0, 100.0)
    ax_b.set_xticks([0, 20, 40, 60, 80, 100])
    ax_b.tick_params(labelsize=6.8, length=2.4)
    ax_b.set_xlabel("share of rank-time in the phase (%)", fontsize=7.4)
    ax_b.set_title("Where the rank-time goes, phase by phase", fontsize=8.5,
                   fontweight="bold")
    ax_b.grid(axis="x", lw=0.45, alpha=0.45)

    idle_all = sum(b["dur"] for b in blocks if b["state"] == "idle")
    comm_all = sum(b["dur"] for b in blocks if b["state"] == "comm")

    fig.tight_layout(rect=[0, 0.035, 0.88, 0.905], h_pad=2.6)
    panel_labels_inches(fig, [ax, ax_b], dx=-0.50, dy=0.08)
    savefig(fig, "568_hpc_profiling_timeline.png")


if __name__ == "__main__":
    main()

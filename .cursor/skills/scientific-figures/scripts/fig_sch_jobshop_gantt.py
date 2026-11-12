"""Job-shop schedule built by a non-delay SPT dispatcher, drawn as a Gantt.

The schedule is generated, not hand-placed: at each step the dispatcher looks
at the next unscheduled operation of every unfinished job, computes its
earliest feasible start as ``max(job ready time, machine free time)``, and
commits the operation with the smallest start, breaking ties by shortest
processing time.  Machine non-overlap and the routing precedence inside each
job therefore hold by construction, and the makespan, idle bands, machine
utilisation and job lateness are read back out of the resulting start/finish
table.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
GREY = "#808080"
RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"
JOB_COL = [BLUE, RED, GREEN, GOLD, PURPLE]
IDLE_FC = "#f0f0f0"

W, H = 10.4, 6.4
MACHINES = ["M1 turning", "M2 milling", "M3 drilling", "M4 grinding",
            "M5 inspect"]
# routing[job] = [(machine index, processing hours), ...] in required order
ROUTING = [
    [(0, 3.0), (1, 2.0), (3, 2.5), (4, 1.0)],
    [(1, 4.0), (0, 2.5), (2, 3.0), (4, 1.5)],
    [(2, 2.0), (3, 3.5), (1, 2.0), (0, 1.5)],
    [(0, 2.0), (2, 2.5), (4, 2.0), (3, 1.5)],
    [(3, 3.0), (1, 1.5), (2, 2.5), (4, 2.0)],
]
DUE = [11.0, 13.0, 12.5, 14.0, 12.0]     # promised finish, hours

FOOT = ("A non-delay dispatcher schedules the operation with the earliest "
        "feasible start at every step, breaking ties by shortest processing "
        "time, so machine non-overlap and the routing order inside each job "
        "hold by construction. Makespan, idle bands, utilisation and lateness "
        "are all read back from the resulting start and finish table rather "
        "than being drawn by hand.")


def schedule():
    n_job = len(ROUTING)
    n_mac = len(MACHINES)
    nxt = [0] * n_job
    ready = [0.0] * n_job
    free = [0.0] * n_mac
    ops = []
    total = sum(len(r) for r in ROUTING)
    while len(ops) < total:
        cand = []
        for j in range(n_job):
            if nxt[j] >= len(ROUTING[j]):
                continue
            m, p = ROUTING[j][nxt[j]]
            cand.append((max(ready[j], free[m]), p, j, m))
        start, p, j, m = min(cand)
        ops.append(dict(job=j, machine=m, start=start, finish=start + p,
                        step=nxt[j], dur=p))
        ready[j] = start + p
        free[m] = start + p
        nxt[j] += 1
    return ops


def check(ops):
    """Assert the two hard constraints; a violated Gantt is a lying figure."""
    for m in range(len(MACHINES)):
        row = sorted([o for o in ops if o["machine"] == m],
                     key=lambda o: o["start"])
        for a, b in zip(row, row[1:]):
            assert b["start"] >= a["finish"] - 1e-9, "machine overlap"
    for j in range(len(ROUTING)):
        row = sorted([o for o in ops if o["job"] == j], key=lambda o: o["step"])
        for a, b in zip(row, row[1:]):
            assert b["start"] >= a["finish"] - 1e-9, "precedence violated"


def idle_bands(ops, makespan):
    out = []
    for m in range(len(MACHINES)):
        row = sorted([o for o in ops if o["machine"] == m],
                     key=lambda o: o["start"])
        t = 0.0
        for o in row:
            if o["start"] > t + 1e-9:
                out.append((m, t, o["start"]))
            t = o["finish"]
        if t < makespan - 1e-9:
            out.append((m, t, makespan))
    return out


def draw_gantt(ax, ops, makespan, crit_job, bar_h=0.52):
    n_mac = len(MACHINES)
    ax.set_xlim(-0.35, makespan + 1.55)
    ax.set_ylim(-0.75, n_mac - 0.40)
    ax.set_axisbelow(True)

    for m, t0, t1 in idle_bands(ops, makespan):
        y = n_mac - 1 - m
        ax.add_patch(Rectangle((t0, y - bar_h / 2), t1 - t0, bar_h,
                               facecolor=IDLE_FC, edgecolor=HAIR,
                               linewidth=0.5, hatch="////", zorder=2))

    for o in ops:
        y = n_mac - 1 - o["machine"]
        col = JOB_COL[o["job"]]
        crit = o["job"] == crit_job
        ax.add_patch(Rectangle(
            (o["start"], y - bar_h / 2), o["dur"], bar_h,
            facecolor=lighten(col, 0.30 if crit else 0.52),
            edgecolor=col if not crit else INK,
            linewidth=1.5 if crit else 0.8, zorder=4 if crit else 3))
        ax.text(o["start"] + o["dur"] / 2.0, y, f"J{o['job'] + 1}",
                ha="center", va="center", fontsize=6.4,
                fontweight="bold" if crit else "normal",
                color=INK, zorder=6)

    for j in range(len(ROUTING)):
        row = sorted([o for o in ops if o["job"] == j], key=lambda o: o["step"])
        crit = j == crit_job
        for a, b in zip(row, row[1:]):
            ya = n_mac - 1 - a["machine"]
            yb = n_mac - 1 - b["machine"]
            ax.plot([a["finish"], b["start"], b["start"]],
                    [ya, ya, yb + np.sign(ya - yb) * bar_h / 2],
                    color=JOB_COL[j], lw=1.25 if crit else 0.65,
                    alpha=0.95 if crit else 0.55,
                    ls="-" if crit else (0, (2.4, 1.8)), zorder=5)

    ax.axvline(makespan, color=RED, lw=1.3, ls=(0, (4, 2.2)), zorder=7)
    ax.text(makespan + 0.18, n_mac - 0.55, f"makespan\n{makespan:.1f} h",
            ha="left", va="top", fontsize=6.8, color=RED, linespacing=1.35)

    ax.set_yticks(np.arange(n_mac))
    ax.set_yticklabels(MACHINES[::-1], fontsize=7.0)
    ax.set_xticks(np.arange(0.0, makespan + 1.0, 2.0))
    ax.set_xlabel("shop time (h from shift start)", fontsize=8.0)
    ax.set_title("Machine timeline with routing links between operations",
                 fontsize=9.0, fontweight="bold")
    ax.tick_params(axis="x", labelsize=7.2)
    ax.tick_params(axis="y", length=0)
    style_axes(ax)

    handles = [Patch(facecolor=lighten(JOB_COL[j], 0.52),
                     edgecolor=JOB_COL[j], label=f"job J{j + 1}")
               for j in range(len(ROUTING))]
    handles += [Patch(facecolor=IDLE_FC, edgecolor=HAIR, hatch="////",
                      label="machine idle"),
                Line2D([], [], color=JOB_COL[crit_job], lw=1.25,
                       label=f"critical job J{crit_job + 1}")]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.50, -0.155), fontsize=6.5, frameon=False,
              ncol=4, handlelength=1.5, columnspacing=1.5)


def draw_util(ax, ops, makespan):
    n_mac = len(MACHINES)
    busy = np.array([sum(o["dur"] for o in ops if o["machine"] == m)
                     for m in range(n_mac)])
    util = 100.0 * busy / makespan
    y = np.arange(n_mac)[::-1]
    ax.barh(y, util, height=0.55, color="#cfd8e4", edgecolor=BLUE,
            linewidth=0.8, zorder=3)
    ax.axvline(util.mean(), color=RED, lw=1.0, ls=(0, (4, 2.2)), zorder=4)
    ax.text(util.mean() + 1.5, n_mac - 0.35, f"mean {util.mean():.0f}%",
            ha="left", va="top", fontsize=6.2, color=RED)
    for yy, u, b in zip(y, util, busy):
        ax.text(u + 1.5, yy, f"{b:.1f} h", ha="left", va="center",
                fontsize=6.2, color=SUB)
    ax.set_yticks(y)
    ax.set_yticklabels([m.split()[0] for m in MACHINES], fontsize=6.5)
    ax.set_xlim(0.0, 118.0)
    ax.set_xlabel("machine utilisation (%)", fontsize=7.2)
    idlest = MACHINES[int(np.argmin(util))]
    ax.set_title(f"{idlest} idles most at {util.min():.0f}% busy",
                 fontsize=8.5, fontweight="bold")
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", length=0)
    style_axes(ax)


def draw_lateness(ax, ops):
    n_job = len(ROUTING)
    comp = np.array([max(o["finish"] for o in ops if o["job"] == j)
                     for j in range(n_job)])
    due = np.array(DUE)
    late = comp - due
    x = np.arange(n_job)
    cols = [RED if l > 0 else GREEN for l in late]
    ax.bar(x, comp, width=0.56, color=[lighten(JOB_COL[j], 0.52)
                                       for j in range(n_job)],
           edgecolor=[JOB_COL[j] for j in range(n_job)], linewidth=0.8,
           zorder=3)
    for xx, c, d, l in zip(x, comp, due, late):
        ax.plot([xx - 0.32, xx + 0.32], [d, d], color=INK, lw=1.2, zorder=5)
        ax.text(xx, c + 0.30, f"{l:+.1f}", ha="center", va="bottom",
                fontsize=6.2, color=RED if l > 0 else GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels([f"J{j + 1}" for j in range(n_job)], fontsize=6.5)
    ax.set_ylim(0.0, max(comp.max(), due.max()) * 1.22)
    ax.set_ylabel("completion time (h)", fontsize=7.2)
    n_late = int((late > 1e-9).sum())
    ax.set_title(f"{n_late:d} of {n_job:d} jobs finish late", fontsize=8.5,
                 fontweight="bold")
    ax.tick_params(axis="y", labelsize=6.5)
    ax.legend(handles=[Line2D([], [], color=INK, lw=1.2, label="due date"),
                       Line2D([], [], color="none",
                              label="labels are lateness (h)")],
              loc="upper left", fontsize=6.3, frameon=False, handlelength=1.3,
              labelspacing=0.3)
    style_axes(ax)


def main():
    set_pub_style()
    ops = schedule()
    check(ops)
    makespan = max(o["finish"] for o in ops)
    comp = [max(o["finish"] for o in ops if o["job"] == j)
            for j in range(len(ROUTING))]
    crit_job = int(np.argmax(comp))

    fig = plt.figure(figsize=(W, H))
    ax_g = axes_inches(fig, (1.00, 1.42, 5.20, 3.98))
    ax_top = axes_inches(fig, (7.12, 3.56, 2.98, 1.84))
    ax_bot = axes_inches(fig, (7.12, 1.16, 2.98, 1.84))

    draw_gantt(ax_g, ops, makespan, crit_job)
    draw_util(ax_top, ops, makespan)
    draw_lateness(ax_bot, ops)

    panel_labels_inches(fig, [ax_g, ax_top, ax_bot], dx=-0.76, dy=0.08)

    savefig(fig, "861_sch_jobshop_gantt.png")


if __name__ == "__main__":
    main()

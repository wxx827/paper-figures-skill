"""Construction programme: Gantt with critical path, float and actual progress.

One task table drives everything.  Each row carries a planned window, a total
float, an actual start and an actual duration; the bars, the slip labels, the
S-curve and the labour histogram are all evaluated from those four numbers, so
the schedule variance quoted in the title cannot disagree with the bars above
it.  The critical path is the only thing drawn in house red, because it is the
only thing on the sheet that has to be read first.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from pubstyle import set_pub_style, lighten, savefig
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
ACCENT = "#b23a48"
BLUE = "#3b5b92"
GREY = "#808080"
FLOAT = "#d5d5d5"

DATA_DATE = 168.0        # project day of the progress report
HORIZON = 292.0

# name, phase, start, duration, crew, critical, total float, slip, stretch
TASKS = [
    ("Site setup", "Enabling", 0, 14, 8, True, 0, 0, 1.00),
    ("Demolition", "Enabling", 6, 18, 12, False, 8, 2, 1.05),
    ("Piling", "Substructure", 14, 28, 16, True, 0, 1, 1.08),
    ("Pile caps", "Substructure", 40, 20, 14, True, 0, 3, 1.05),
    ("Basement slab", "Substructure", 58, 22, 18, True, 0, 4, 1.10),
    ("Ground floor deck", "Superstructure", 78, 18, 20, True, 0, 6, 1.06),
    ("Frame L1 to L3", "Superstructure", 94, 34, 26, True, 0, 7, 1.09),
    ("Core walls", "Superstructure", 100, 40, 12, False, 10, 4, 1.02),
    ("Frame L4 to L5", "Superstructure", 126, 26, 24, True, 0, 5, 1.04),
    ("Roof structure", "Superstructure", 150, 18, 14, False, 6, 3, 1.00),
    ("Facade units", "Envelope", 152, 40, 18, True, 0, 4, 1.00),
    ("Roofing", "Envelope", 168, 22, 10, False, 14, 0, 1.00),
    ("MEP risers", "Services", 140, 46, 22, False, 12, 3, 1.04),
    ("MEP terminals", "Services", 186, 44, 26, True, 0, 0, 1.00),
    ("Fit-out", "Services", 200, 52, 34, True, 0, 0, 1.00),
    ("Commissioning", "Services", 248, 28, 12, True, 0, 0, 1.00),
]
PHASES = ["Enabling", "Substructure", "Superstructure", "Envelope",
          "Services"]


def derive():
    """Planned and actual windows, progress and weights for every task."""
    rows = []
    for name, phase, s, d, crew, crit, flt, slip, stretch in TASKS:
        a_s = s + slip
        a_d = d * stretch
        done = float(np.clip((DATA_DATE - a_s) / a_d, 0.0, 1.0))
        rows.append(dict(name=name, phase=phase, start=float(s),
                         dur=float(d), crew=crew, crit=crit, flt=float(flt),
                         a_start=float(a_s), a_dur=float(a_d), done=done,
                         weight=float(d * crew),
                         slip_days=(a_s + a_d) - (s + d)))
    return rows


def curves(rows):
    days = np.arange(0.0, HORIZON)
    w = np.array([r["weight"] for r in rows])
    plan = np.zeros(days.size)
    act = np.zeros(days.size)
    crew_plan = np.zeros(days.size)
    crew_act = np.zeros(days.size)
    for r, wi in zip(rows, w):
        plan += wi * np.clip((days - r["start"]) / r["dur"], 0.0, 1.0)
        act += wi * np.clip((days - r["a_start"]) / r["a_dur"], 0.0, 1.0)
        crew_plan += r["crew"] * ((days >= r["start"]) &
                                  (days < r["start"] + r["dur"]))
        crew_act += r["crew"] * ((days >= r["a_start"]) &
                                 (days < r["a_start"] + r["a_dur"]))
    act = np.where(days <= DATA_DATE, act, np.nan)
    crew_act = np.where(days <= DATA_DATE, crew_act, np.nan)
    return days, 100.0 * plan / w.sum(), 100.0 * act / w.sum(), \
        crew_plan, crew_act


def gantt_panel(ax, rows):
    n = len(rows)
    for i, r in enumerate(rows):
        y = n - 1 - i
        base = ACCENT if r["crit"] else BLUE
        if r["flt"] > 0:
            ax.add_patch(Rectangle((r["start"] + r["dur"], y - 0.20),
                                   r["flt"], 0.40, facecolor=FLOAT,
                                   edgecolor="#bbbbbb", linewidth=0.4,
                                   hatch="////", zorder=2))
        ax.add_patch(Rectangle((r["start"], y - 0.30), r["dur"], 0.60,
                               facecolor=lighten(base, 0.62), edgecolor=base,
                               linewidth=0.7, zorder=3))
        if r["done"] > 0:
            end = r["a_start"] + r["a_dur"] * r["done"]
            ax.add_patch(Rectangle((r["a_start"], y - 0.16),
                                   r["a_dur"] * r["done"], 0.32,
                                   facecolor=base, edgecolor="none",
                                   zorder=4))
            late = r["slip_days"] >= 3.0
            ax.text(max(end, r["start"] + r["dur"] + r["flt"]) + 4.0, y,
                    f"{r['done'] * 100:.0f} %" +
                    (f", +{r['slip_days']:.0f} d" if late else ""),
                    ha="left", va="center", fontsize=6.0,
                    color=ACCENT if late else SUB, zorder=5)

    starts = {}
    for i, r in enumerate(rows):
        starts.setdefault(r["phase"], []).append(n - 1 - i)
    for k, phase in enumerate(PHASES):
        ys = starts[phase]
        if k % 2 == 0:
            ax.axhspan(min(ys) - 0.5, max(ys) + 0.5, facecolor="#f5f5f5",
                       edgecolor="none", zorder=0)
        ax.text(HORIZON - 4.0, max(ys), phase, ha="right", va="center",
                fontsize=6.4, color=MUTED, zorder=5)

    ax.axvline(DATA_DATE, color=INK, lw=1.0, ls=(0, (4.0, 2.4)), zorder=6)
    ax.text(DATA_DATE + 3.0, n + 0.05, f"data date, day {DATA_DATE:.0f}",
            ha="left", va="center", fontsize=6.6, color=INK, zorder=6)

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([r["name"] for r in rows][::-1], fontsize=6.6)
    ax.set_ylim(-1.15, n + 0.4)
    ax.set_xlim(-2.0, HORIZON)
    ax.set_xticks(np.arange(0, int(HORIZON) + 1, 30))
    ax.set_xlabel("project day (days)")
    ax.tick_params(labelsize=6.8)
    ax.grid(axis="x", lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    handles = [
        Patch(facecolor=lighten(ACCENT, 0.62), edgecolor=ACCENT, lw=0.7,
              label="critical path, planned"),
        Patch(facecolor=lighten(BLUE, 0.62), edgecolor=BLUE, lw=0.7,
              label="non-critical, planned"),
        Patch(facecolor=FLOAT, edgecolor="#bbbbbb", hatch="////",
              label="total float"),
        Patch(facecolor=BLUE, edgecolor="none", label="progress achieved")]
    ax.legend(handles=handles, loc="lower left", frameon=False,
              fontsize=6.6, ncol=2, handlelength=1.5, columnspacing=1.0,
              borderaxespad=0.2)
    ax.set_title("Programme against the reported progress", fontsize=9.0,
                 fontweight="bold", color=INK)


def scurve_panel(ax, days, plan, act):
    ax.plot(days, plan, color=GREY, lw=1.5, ls=(0, (4.0, 2.4)),
            label="planned")
    ax.plot(days, act, color=BLUE, lw=1.8, label="actual")
    i = int(DATA_DATE)
    gap = act[i] - plan[i]
    ax.axvline(DATA_DATE, color=INK, lw=0.8, ls=(0, (3.0, 2.2)))
    ax.plot([DATA_DATE, DATA_DATE], [act[i], plan[i]], color=ACCENT, lw=1.6)
    ax.annotate(f"{gap:+.1f} % of the work", xy=(DATA_DATE, (act[i] +
                                                             plan[i]) / 2),
                xytext=(DATA_DATE - 24.0, 78.0), ha="right", va="center",
                fontsize=6.6, color=ACCENT,
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.7))
    ax.set_xlim(0.0, HORIZON)
    ax.set_ylim(0.0, 104.0)
    ax.set_xlabel("project day (days)", fontsize=7.4)
    ax.set_ylabel("work complete (%)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left",
              handlelength=1.5, borderaxespad=0.2)
    ax.set_title("Cumulative S-curve", fontsize=8.2, fontweight="bold",
                 color=INK)
    return gap


def crew_panel(ax, days, crew_plan, crew_act):
    ax.fill_between(days, crew_plan, step="post", color=lighten(GREY, 0.66),
                    linewidth=0.0, label="planned")
    ax.step(days, crew_plan, where="post", color=GREY, lw=0.9)
    ax.step(days, crew_act, where="post", color=BLUE, lw=1.5, label="actual")
    ax.axvline(DATA_DATE, color=INK, lw=0.8, ls=(0, (3.0, 2.2)))
    peak = float(np.nanmax(crew_plan))
    ax.annotate(f"planned peak {peak:.0f} persons",
                xy=(float(days[int(np.nanargmax(crew_plan))]), peak),
                xytext=(HORIZON * 0.76, peak * 1.20), ha="center",
                va="bottom", fontsize=6.5, color=SUB,
                arrowprops=dict(arrowstyle="-", color=SUB, lw=0.7))
    ax.set_xlim(0.0, HORIZON)
    ax.set_ylim(0.0, max(peak, float(np.nanmax(crew_act))) * 1.22)
    ax.set_xlabel("project day (days)", fontsize=7.4)
    ax.set_ylabel("labour on site (persons)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", ncol=2,
              handlelength=1.5, borderaxespad=0.2)
    ax.set_title("Labour histogram", fontsize=8.2, fontweight="bold",
                 color=INK)


def main():
    set_pub_style()
    rows = derive()
    days, plan, act, crew_plan, crew_act = curves(rows)
    crit_slip = max(r["slip_days"] for r in rows if r["crit"])
    n_crit = sum(1 for r in rows if r["crit"])

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.30, hspace=0.62)
    ax_g = fig.add_subplot(gs[:, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    gantt_panel(ax_g, rows)
    gap = scurve_panel(ax_s, days, plan, act)
    crew_panel(ax_c, days, crew_plan, crew_act)

    fig.tight_layout(rect=[0, 0.040, 1, 0.945], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_g, ax_s, ax_c], dx=-0.72, dy=0.12)
    savefig(fig, "592_bim_construction_schedule.png")


if __name__ == "__main__":
    main()

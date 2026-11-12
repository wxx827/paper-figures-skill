"""Swimmer plot: per-patient treatment exposure, response and on-study events."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, style_axes

RESP_COLOR = {"CR": "#1f3f6e", "PR": "#4c78a8", "SD": "#adadad", "PD": "#b23a48"}
RESP_ORDER = ["CR", "PR", "SD", "PD"]
RESP_LONG = {"CR": "complete response", "PR": "partial response",
             "SD": "stable disease", "PD": "progressive disease"}

# dose level, n, P(CR, PR, SD, PD), exposure multiplier
COHORTS = [("0.3 mg/kg", 6, [0.00, 0.05, 0.35, 0.60], 0.68),
           ("1 mg/kg", 8, [0.00, 0.22, 0.42, 0.36], 0.86),
           ("3 mg/kg", 10, [0.10, 0.35, 0.35, 0.20], 1.00),
           ("10 mg/kg", 10, [0.18, 0.42, 0.30, 0.10], 1.14)]

DUR_RANGE = {"CR": (17.0, 30.0), "PR": (9.0, 24.0),
             "SD": (3.5, 13.0), "PD": (1.2, 6.5)}
ONGOING_P = {"CR": 0.75, "PR": 0.45, "SD": 0.14, "PD": 0.0}


def simulate(rng):
    """Build a cohort-grouped patient list, sorted by duration within cohort."""
    rows = []
    for dose, n, probs, mult in COHORTS:
        block = []
        for _ in range(n):
            resp = rng.choice(RESP_ORDER, p=probs)
            lo, hi = DUR_RANGE[resp]
            dur = float(rng.uniform(lo, hi)) * mult
            ongoing = bool(rng.random() < ONGOING_P[resp])
            onset = float(rng.uniform(1.4, 3.4)) if resp in ("CR", "PR") else None
            if onset is not None and onset > dur - 0.8:
                onset = max(0.9, dur - 1.2)
            if ongoing:
                prog, death = None, None
            else:
                prog = dur
                death = (dur + float(rng.uniform(0.8, 4.5))
                         if rng.random() < (0.55 if resp == "PD" else 0.18)
                         else None)
            block.append(dict(dose=dose, resp=resp, dur=dur, ongoing=ongoing,
                              onset=onset, prog=prog, death=death))
        block.sort(key=lambda r: r["dur"], reverse=True)
        rows.extend(block)
    return rows


def main():
    set_pub_style()
    rng = np.random.default_rng(20)
    rows = simulate(rng)
    n = len(rows)

    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    fig.subplots_adjust(left=0.115, right=0.700, top=0.895, bottom=0.105)

    # cohort bands ----------------------------------------------------------
    bounds, start = [], 0
    for dose, cnt, _, _ in COHORTS:
        bounds.append((dose, start, start + cnt))
        start += cnt

    for k, (dose, i0, i1) in enumerate(bounds):
        y0, y1 = n - i1 - 0.5, n - i0 - 0.5
        if k % 2 == 0:
            ax.axhspan(y0, y1, color="#f2f5f9", zorder=0)
        if k:
            ax.axhline(y1, color="#c9c9c9", lw=0.7, ls=(0, (4, 3)), zorder=1)

    for i, r in enumerate(rows):
        y = n - 1 - i
        c = RESP_COLOR[r["resp"]]
        ax.barh(y, r["dur"], height=0.60, color=c, edgecolor="white",
                linewidth=0.4, zorder=3)
        if r["death"] is not None:
            ax.plot([r["dur"], r["death"]], [y, y], lw=0.8, ls=(0, (2, 1.6)),
                    color="#8a8a8a", zorder=2)
            ax.plot(r["death"], y, marker="x", ms=4.2, mew=1.3, color="#1a1a1a",
                    zorder=6)
        if r["onset"] is not None:
            ax.plot(r["onset"], y, marker="^", ms=4.0, mfc="#ffffff",
                    mec="#12314f", mew=0.9, zorder=6)
        if r["prog"] is not None:
            ax.plot(r["prog"], y, marker="v", ms=4.2, mfc="#b23a48",
                    mec="white", mew=0.6, zorder=6)
        if r["ongoing"]:
            ax.plot(r["dur"] + 0.75, y, marker=">", ms=4.4, color=c,
                    mec="white", mew=0.4, zorder=6)

    xmax = 34.0
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.9, n - 0.1)
    ax.set_xticks(np.arange(0, 35, 4))
    ax.set_xlabel("Time from first dose (months)")
    ax.set_yticks([n - (i0 + i1) / 2.0 - 0.5 for _, i0, i1 in bounds])
    ax.set_yticklabels([d for d, _, _ in bounds], fontweight="bold",
                       fontsize=7.6)
    ax.set_ylabel("Dose cohort", labelpad=6)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e2e2e2", lw=0.6, zorder=0)
    style_axes(ax)
    ax.spines["left"].set_visible(False)

    # legends ---------------------------------------------------------------
    counts = {k: sum(r["resp"] == k for r in rows) for k in RESP_ORDER}
    resp_handles = [Patch(facecolor=RESP_COLOR[k], edgecolor="white",
                          label=f"{k} \u2014 {RESP_LONG[k]}  (n = {counts[k]})")
                    for k in RESP_ORDER]
    ev_handles = [
        Line2D([], [], ls="none", marker="^", ms=4.6, mfc="white",
               mec="#12314f", mew=0.9, label="First objective response"),
        Line2D([], [], ls="none", marker="v", ms=4.6, mfc="#b23a48",
               mec="white", mew=0.6, label="Disease progression"),
        Line2D([], [], ls="none", marker="x", ms=4.6, mew=1.3, color="#1a1a1a",
               label="Death"),
        Line2D([], [], ls="none", marker=">", ms=4.8, color="#4c78a8",
               mec="white", mew=0.4, label="Continuing on treatment"),
        Line2D([], [], ls=(0, (2, 1.6)), lw=0.8, color="#8a8a8a",
               label="Off-treatment follow-up"),
    ]
    leg1 = ax.legend(handles=resp_handles, title="Best overall response",
                     loc="upper left", bbox_to_anchor=(1.02, 1.005),
                     frameon=False, fontsize=6.6, handlelength=1.1,
                     handleheight=0.9, borderpad=0.2, labelspacing=0.45)
    leg1.get_title().set_fontsize(7.4)
    leg1.get_title().set_fontweight("bold")
    leg1._legend_box.align = "left"
    ax.add_artist(leg1)
    # add_artist() clips to the axes patch, which would drop the legend from
    # the tight bounding box on save.
    leg1.set_clip_on(False)
    leg1.set_clip_path(None)

    leg2 = ax.legend(handles=ev_handles, title="On-study events",
                     loc="upper left", bbox_to_anchor=(1.02, 0.70),
                     frameon=False, fontsize=6.6, handlelength=1.1,
                     borderpad=0.2, labelspacing=0.45)
    leg2.get_title().set_fontsize(7.4)
    leg2.get_title().set_fontweight("bold")
    leg2._legend_box.align = "left"

    med = float(np.median([r["dur"] for r in rows]))
    ongoing = sum(r["ongoing"] for r in rows)
    orr = counts["CR"] + counts["PR"]
    txt = (f"n = {n} patients\n"
           f"Median treatment duration\n  {med:.1f} months\n"
           f"Objective response rate\n  {orr}/{n} ({100 * orr / n:.0f}%)\n"
           f"Ongoing at data cut-off\n  {ongoing}/{n} ({100 * ongoing / n:.0f}%)")
    ax.text(1.02, 0.315, txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=6.6, color="#1a1a1a", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f6f6f6",
                      edgecolor="#d5d5d5", linewidth=0.6))


    savefig(fig, "113_swimmer_plot.png")


if __name__ == "__main__":
    main()

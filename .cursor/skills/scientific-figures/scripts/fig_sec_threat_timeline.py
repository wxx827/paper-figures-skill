"""One intrusion laid out along the kill chain, against the controls watching it.

The upper band is the campaign: one bar per kill-chain stage, shaded from the
early reconnaissance to the final actions on objectives, with a triangle at
each point where a control produced a usable signal.  The two background bands
split the timeline at the first of those signals, which is what turns the
figure into a statement about response rather than about the attacker: the
intrusion was seen less than half way through and still ran to completion.
The lower panel is the coverage audit - which stages each control could see at
all - and the combined row shows the gap the attacker walked through.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pubstyle import set_pub_style, savefig, get_palette
from figkit import panel_labels_inches

RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
ACCENT = "#b23a48"      # house red: undetected time and uncovered stages
GOLD = "#c9a227"
GREEN = "#2e7d5b"

# (stage, start h, duration h, detecting control or None, detection hour)
STAGES = [
    ("Reconnaissance", 0.0, 9.0, None, None),
    ("Weaponization", 7.5, 5.0, None, None),
    ("Delivery", 12.5, 1.6, None, None),
    ("Exploitation", 13.8, 2.2, None, None),
    ("Installation", 15.5, 3.6, "EDR", 18.6),
    ("Command and control", 18.0, 12.4, "Network IDS", 26.4),
    ("Actions on objectives", 29.5, 14.5, "DLP", 38.5),
]
CONTAINED = 44.0

# control -> the stages it can observe at all
COVERAGE = {
    "Email gateway": ["Delivery"],
    "EDR agent": ["Exploitation", "Installation", "Actions on objectives"],
    "Network IDS": ["Delivery", "Command and control"],
    "DLP": ["Actions on objectives"],
    "Identity / MFA": ["Exploitation"],
}


def merge(spans):
    """Union of (t0, t1) intervals, needed for the combined coverage row."""
    out = []
    for t0, t1 in sorted(spans):
        if out and t0 <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], t1)
        else:
            out.append([t0, t1])
    return [(a, b) for a, b in out]


def main():
    set_pub_style()
    np.random.default_rng(581)          # fixed seed; the timeline is scripted

    span = {s[0]: (s[1], s[1] + s[2]) for s in STAGES}
    t_start = min(s[1] for s in STAGES)
    t_end = max(s[1] + s[2] for s in STAGES)
    dets = [(s[4], s[3], s[0]) for s in STAGES if s[3] is not None]
    t_det = min(d[0] for d in dets)
    mttd = t_det - t_start
    mttr = CONTAINED - t_det

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.58], hspace=0.34,
                          left=0.135, right=0.795, top=0.885, bottom=0.105)
    ax = fig.add_subplot(gs[0])
    ax_c = fig.add_subplot(gs[1], sharex=ax)

    # ---- (a) kill-chain gantt --------------------------------------------
    for axi in (ax, ax_c):
        axi.axvspan(t_start, t_det, color=ACCENT, alpha=0.09, lw=0, zorder=0)
        axi.axvspan(t_det, CONTAINED, color=GOLD, alpha=0.10, lw=0, zorder=0)
        axi.axvline(t_det, color=ACCENT, lw=1.0, ls="--", zorder=2)
        axi.axvline(CONTAINED, color=GOLD, lw=1.0, ls="--", zorder=2)

    ramp = get_palette("cb_sequential_blue", 9)[2:9]
    y = np.arange(len(STAGES))
    for k, (name, t0, dur, ctrl, td) in enumerate(STAGES):
        ax.barh(k, dur, left=t0, height=0.56, color=ramp[k],
                edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(t0 + dur + 0.5, k, f"{dur:.1f} h", ha="left", va="center",
                fontsize=6.6, color=SUB, zorder=4)
        if td is not None:
            first = td == t_det
            col = ACCENT if first else GOLD
            ax.plot([td], [k - 0.36], marker="v", ms=6.0, color=col,
                    mec="white", mew=0.7, clip_on=False, zorder=6)
            ax.text(td + 0.6, k - 0.40, f"{ctrl} at {td:.1f} h", ha="left",
                    va="center", fontsize=6.2, color=col, zorder=6)

    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in STAGES], fontsize=7.4)
    ax.set_ylim(len(STAGES) - 0.45, -0.90)
    ax.set_ylabel("kill-chain stage", fontsize=8.0)
    ax.tick_params(labelsize=7.0, length=2.6, labelbottom=False)
    ax.grid(axis="x", lw=0.45, alpha=0.40)
    ax.set_axisbelow(True)
    ax.set_title("Stage durations, and the point at which each control spoke",
                 fontsize=9.0, fontweight="bold")

    ax.text(t_start + 0.4, -0.78, f"undetected: {mttd:.1f} h", ha="left",
            va="center", fontsize=6.8, color=ACCENT)
    ax.text(t_det + 0.5, -0.78, f"detected but not contained: {mttr:.1f} h",
            ha="left", va="center", fontsize=6.8, color=GOLD)

    # ---- (b) control coverage --------------------------------------------
    names = list(COVERAGE) + ["Combined coverage"]
    all_spans = []
    for i, ctrl in enumerate(COVERAGE):
        spans = [span[s] for s in COVERAGE[ctrl]]
        all_spans += spans
        for t0, t1 in spans:
            ax_c.barh(i, t1 - t0, left=t0, height=0.52, color=GREEN,
                      edgecolor="white", linewidth=0.7, zorder=3)
    union = merge(all_spans)
    covered = sum(b - a for a, b in union)
    row = len(COVERAGE)
    prev = t_start
    for a, b in union:
        if a > prev:
            ax_c.barh(row, a - prev, left=prev, height=0.52, color=ACCENT,
                      edgecolor="white", linewidth=0.7, zorder=3)
        ax_c.barh(row, b - a, left=a, height=0.52, color=GREEN,
                  edgecolor="white", linewidth=0.7, zorder=3)
        prev = b
    if prev < t_end:
        ax_c.barh(row, t_end - prev, left=prev, height=0.52, color=ACCENT,
                  edgecolor="white", linewidth=0.7, zorder=3)

    for _name, t0, _dur, _c, _td in STAGES:
        ax_c.axvline(t0, color="#cccccc", lw=0.5, ls=":", zorder=1)
    ax_c.set_yticks(np.arange(len(names)))
    ax_c.set_yticklabels(names, fontsize=7.0)
    ax_c.set_ylim(len(names) - 0.4, -0.6)
    ax_c.set_xlim(t_start - 0.8, CONTAINED + 1.2)
    ax_c.set_xticks(np.arange(0, 49, 6))
    ax_c.set_xlabel("time since first reconnaissance (hours)", fontsize=8.0)
    ax_c.set_ylabel("control", fontsize=8.0)
    ax_c.tick_params(labelsize=7.0, length=2.6)
    ax_c.grid(axis="x", lw=0.45, alpha=0.40)
    ax_c.set_axisbelow(True)
    ax_c.set_title("Which stages each control could observe at all",
                   fontsize=8.5, fontweight="bold")

    handles = [Patch(facecolor=ACCENT, alpha=0.30,
                     label="Undetected dwell window"),
               Patch(facecolor=GOLD, alpha=0.32, label="Response window"),
               Line2D([], [], lw=0, marker="v", ms=5.0, mfc=ACCENT,
                      mec="white", label="First reliable detection"),
               Line2D([], [], lw=0, marker="v", ms=5.0, mfc=GOLD,
                      mec="white", label="Later detection"),
               Patch(facecolor=GREEN, label="Stage observable"),
               Patch(facecolor=ACCENT, label="No control sees this stretch")]
    fig.legend(handles=handles, loc="upper left",
               bbox_to_anchor=(0.805, 0.855), ncol=1, fontsize=7.0,
               frameon=False, handlelength=1.6, labelspacing=0.55)

    card = dict(ha="left", va="top", fontsize=7.4, color=RULE,
                linespacing=1.5, transform=fig.transFigure,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5",
                          edgecolor="#cccccc", linewidth=0.7))
    fig.text(0.805, 0.40,
             f"MTTD  {mttd:.1f} h\n"
             f"first signal from {dets[0][1]}\n"
             f"at the {dets[0][2].lower()} stage", **card)
    fig.text(0.805, 0.235,
             f"MTTR  {mttr:.1f} h\n"
             f"detection to containment\n"
             f"at {CONTAINED:.0f} h", **card)


    panel_labels_inches(fig, [ax, ax_c], dx=-0.95, dy=0.07)
    savefig(fig, "581_sec_threat_timeline.png")


if __name__ == "__main__":
    main()

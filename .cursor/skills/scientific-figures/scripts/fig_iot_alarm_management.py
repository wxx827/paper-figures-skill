"""Figure 671 - Alarm flood analysis for one 12 h operating window.

Main panel: alarms per ten minutes, stacked by priority.  The EEMUA-191
manageable rate is drawn as a rule across the window and every run of bins
above it is banded, so a flood reads as an episode with a start and an end
rather than as a few tall bars.  Priority is stacked rather than plotted
separately because the question during a flood is what fraction of the burst
actually needed an operator, not how each class behaved.

Right column: (b) which tags produce the alarms, as a Pareto - the standard
first move in a rationalisation project; (c) how long acknowledgement took,
with the median and the shift target.

Tag counts in (b) are a multinomial draw from the same total the main panel
plots, and the response times in (c) are one per alarm in that total, so the
three panels always describe one window.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, annotate_box, lighten
from figkit import axes_inches, panel_label_inches

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#cccccc"
GRID, BASE, RULE = "#d9d9d9", "#808080", "#333333"

HIGH = "#b23a48"       # house red  - priority 1
MED = "#c9a227"        # house gold - priority 2
LOW = "#3d6b8e"        # grey-blue  - priority 3
GREEN = "#2e7d5b"

N_BIN = 72             # 10 min bins over a 12 h window
BIN_MIN = 10.0
EEMUA_MANAGE = 10.0    # alarms per 10 min above which the rate is a flood
EEMUA_TARGET = 1.0     # EEMUA-191 steady-state target, alarms per 10 min
ACK_TARGET = 5.0       # shift target for acknowledgement, minutes
TAGS = ["FIC-204 flow", "TI-118 temp", "PSV-31 relief", "LIC-402 level",
        "VIB-07 shaft", "AI-556 analyser", "MOV-12 valve", "PT-330 press"]
PRIOS = [("Priority 1", HIGH), ("Priority 2", MED), ("Priority 3", LOW)]


def runs_of(mask):
    """Contiguous True runs of a boolean array as (start, stop) index pairs."""
    out, i = [], 0
    while i < mask.size:
        if mask[i]:
            j = i
            while j + 1 < mask.size and mask[j + 1]:
                j += 1
            out.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return out


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(671)

    hours = (np.arange(N_BIN) + 0.5) * BIN_MIN / 60.0
    rate = np.full(N_BIN, 2.4)
    for start, stop, peak in ((20, 28, 24.0), (51, 57, 16.0)):
        k = np.arange(start, stop)
        rate[k] += peak * np.exp(-0.5 * ((k - (start + stop) / 2.0) / 2.2) ** 2)
    rate += 1.1 * np.sin(2 * np.pi * hours / 5.0) ** 2

    total = rng.poisson(rate)
    # Priority mix shifts toward the top class while the plant is upset.
    stress = np.clip((rate - 2.4) / 20.0, 0.0, 1.0)
    p_high = 0.10 + 0.26 * stress
    p_med = 0.28 + 0.06 * stress
    counts = np.zeros((3, N_BIN), dtype=int)
    for i in range(N_BIN):
        counts[:, i] = rng.multinomial(
            total[i], [p_high[i], p_med[i], 1.0 - p_high[i] - p_med[i]])

    flood = total > EEMUA_MANAGE
    n_total = int(total.sum())
    avg_rate = total.mean()
    flood_pct = 100.0 * flood.sum() / N_BIN

    weights = 1.0 / np.arange(1, len(TAGS) + 2) ** 1.15
    tag_counts = rng.multinomial(n_total, weights / weights.sum())
    order = np.argsort(tag_counts[:-1])[::-1]
    bar_counts = np.append(tag_counts[:-1][order], tag_counts[-1])
    bar_names = [TAGS[i] for i in order] + ["other tags"]
    cum = 100.0 * np.cumsum(bar_counts) / bar_counts.sum()

    slow = np.repeat(flood, total)[:n_total]
    ack = rng.lognormal(np.log(2.9), 0.60, n_total) + slow * \
        rng.lognormal(np.log(3.2), 0.55, n_total)
    ack_med, ack_p90 = np.percentile(ack, [50, 90])

    fig = plt.figure(figsize=(10.4, 6.4))

    # ---- (a) alarms per ten minutes ----------------------------------------
    axm = axes_inches(fig, (0.90, 0.76, 5.18, 4.66))
    for i0, i1 in runs_of(flood):
        axm.axvspan(hours[i0] - BIN_MIN / 120.0, hours[i1 - 1] + BIN_MIN / 120.0,
                    facecolor=lighten(HIGH, 0.88), edgecolor="none", zorder=0)

    bottom = np.zeros(N_BIN)
    for row, (name, col) in zip(counts, PRIOS):
        axm.bar(hours, row, bottom=bottom, width=0.145, color=col,
                edgecolor="none", zorder=3, label=name)
        bottom += row

    # Both reference rates are labelled in the legend, not on the plot: the
    # steady-state target sits at one alarm per bin, i.e. buried inside the
    # bars, where no in-axes label could be read.
    axm.axhline(EEMUA_MANAGE, color=RULE, lw=1.1, ls=(0, (5.0, 2.4)), zorder=5)
    axm.axhline(EEMUA_TARGET, color=GREEN, lw=1.0, ls=(0, (2.6, 2.0)),
                zorder=5)

    for i0, i1 in runs_of(flood):
        mid = 0.5 * (hours[i0] + hours[i1 - 1])
        axm.text(mid, total[i0:i1].max() + 1.4,
                 f"flood, {int((i1 - i0) * BIN_MIN)} min", ha="center",
                 va="bottom", fontsize=6.8, color=HIGH, fontweight="bold")

    axm.set_xlim(0.0, 12.0)
    axm.set_ylim(0.0, total.max() * 1.22)
    axm.set_xticks(np.arange(0, 13, 2))
    axm.set_xlabel("time into shift (hours)", fontsize=8.0)
    axm.set_ylabel("alarms per 10 min (alarms/10min)", fontsize=8.0)
    axm.tick_params(labelsize=7.0, length=2.6)
    axm.grid(axis="y", color=GRID, lw=0.6)
    annotate_box(axm, f"{n_total} alarms in {N_BIN * BIN_MIN / 60.0:.0f} h\n"
                      f"mean {avg_rate:.1f} per 10 min, "
                      f"{avg_rate / EEMUA_TARGET:.0f}$\\times$ target\n"
                      f"{flood_pct:.0f} % of the window in flood",
                 loc="upper left", fontsize=7.2)
    handles, labels = axm.get_legend_handles_labels()
    handles += [
        Line2D([], [], color=RULE, lw=1.1, ls=(0, (5.0, 2.4)),
               label=f"EEMUA manageable, {EEMUA_MANAGE:.0f} per 10 min"),
        Line2D([], [], color=GREEN, lw=1.0, ls=(0, (2.6, 2.0)),
               label=f"EEMUA steady state, {EEMUA_TARGET:.0f} per 10 min")]
    axm.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.005),
               ncol=3, fontsize=7.0, frameon=False, handlelength=1.9,
               columnspacing=1.5, handletextpad=0.45, labelspacing=0.3)
    axm.set_title(f"{len(runs_of(flood))} flood episodes in one shift",
                  fontsize=9.0, fontweight="bold", pad=32.0)
    for s in ("top", "right"):
        axm.spines[s].set_visible(False)

    # ---- (b) Pareto of alarm sources ---------------------------------------
    # Shifted 0.2 in left of the other figures in this set: the twin axis puts
    # a second tick column and label on the right, which would otherwise run
    # off the nominal canvas and let bbox_inches="tight" inflate it.
    axp = axes_inches(fig, (6.80, 3.36, 3.00, 2.06))
    x = np.arange(bar_counts.size)
    cols = [HIGH] + [lighten(LOW, f)
                     for f in np.linspace(0.0, 0.62, bar_counts.size - 1)]
    axp.bar(x, bar_counts, width=0.66, color=cols, edgecolor="white",
            linewidth=0.7, zorder=3)
    axp.set_xticks(x)
    axp.set_xticklabels(bar_names, rotation=32, ha="right")
    axp.set_ylabel("alarms in window (count)", fontsize=7.4)
    axp.set_ylim(0.0, bar_counts.max() * 1.15)
    axp.tick_params(labelsize=6.2, length=2.2)
    axp.grid(axis="y", color=GRID, lw=0.6)

    axq = axp.twinx()
    axq.plot(x, cum, lw=1.3, color=INK, marker="o", ms=2.8, mfc="white",
             mec=INK, mew=0.8, zorder=5)
    axq.axhline(80.0, color=GREEN, lw=0.9, ls=(0, (3.0, 2.0)), zorder=4)
    n80 = int(np.searchsorted(cum, 80.0) + 1)
    axq.text(x[-1], 82.0, f"{n80} tags reach 80 %", ha="right", va="bottom",
             fontsize=6.4, color=GREEN)
    axq.set_ylim(0.0, 108.0)
    axq.set_ylabel("cumulative share (%)", fontsize=7.4)
    axq.tick_params(labelsize=6.5, length=2.2)
    axq.spines["top"].set_visible(False)
    axq.spines["right"].set_visible(True)
    axp.set_title(f"Top tag alone is {100 * bar_counts[0] / n_total:.0f} % "
                  "of the load", fontsize=8.5, fontweight="bold", pad=3.0)
    for s in ("top",):
        axp.spines[s].set_visible(False)

    # ---- (c) acknowledgement time ------------------------------------------
    axa = axes_inches(fig, (6.80, 0.76, 3.00, 1.94))
    bins = np.linspace(0.0, 30.0, 46)
    axa.hist(np.clip(ack, 0.0, 29.9), bins=bins, color=lighten(LOW, 0.30),
             edgecolor="white", linewidth=0.5, zorder=3)
    axa.axvline(ack_med, color=HIGH, lw=1.2, zorder=5)
    axa.axvline(ACK_TARGET, color=GREEN, lw=1.0, ls=(0, (3.0, 2.0)), zorder=5)
    axa.axvline(ack_p90, color=BASE, lw=1.0, ls=(0, (2.2, 1.8)), zorder=5)
    axa.set_xlim(0.0, 30.0)
    axa.set_xlabel("time to acknowledge (min)", fontsize=7.4)
    axa.set_ylabel("alarms (count)", fontsize=7.4)
    axa.tick_params(labelsize=6.5, length=2.2)
    axa.grid(axis="y", color=GRID, lw=0.6)
    top = axa.get_ylim()[1]
    # Median and target land within a minute of each other, so the three
    # labels are staggered vertically rather than tucked beside their lines.
    axa.text(ack_med + 0.6, top * 0.99, f"median {ack_med:.1f} min",
             ha="left", va="top", fontsize=6.6, color=HIGH)
    axa.text(ack_med + 0.6, top * 0.86, f"target {ACK_TARGET:.0f} min",
             ha="left", va="top", fontsize=6.6, color=GREEN)
    axa.text(ack_p90 + 0.6, top * 0.70, f"p90 {ack_p90:.1f} min", ha="left",
             va="top", fontsize=6.6, color=SUB)
    axa.set_title(f"{100 * (ack > ACK_TARGET).mean():.0f} % acknowledged "
                  "late", fontsize=8.5, fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        axa.spines[s].set_visible(False)

    for ax, lab in ((axm, "(a)"), (axp, "(b)"), (axa, "(c)")):
        panel_label_inches(fig, ax, lab, dx=-0.66, dy=0.12)
    savefig(fig, "671_iot_alarm_management.png")


if __name__ == "__main__":
    main()

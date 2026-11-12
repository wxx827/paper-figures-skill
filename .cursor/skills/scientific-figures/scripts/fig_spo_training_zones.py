"""Five-zone training model: where a season's hours actually go.

The intensity axis is the whole figure.  Each zone is a coloured band on it,
carrying the physiology that defines the band and the two ventilatory
thresholds that bound it; the bars are one endurance athlete's annual hours in
each band, and the step line is the cumulative share of the season.  The two
subordinate panels answer the questions the main panel raises: which fuel each
band burns, and how the same five bands are re-weighted through a
periodisation cycle.  Fat, carbohydrate and protein keep the batch-wide
macronutrient colours.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import lighten, savefig, set_pub_style
from figkit import panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
BLUE, GREEN, RED, GOLD = "#3b5b92", "#2e7d5b", "#b23a48", "#c9a227"
FAT, CARB, PROTEIN = "#c0603f", "#c9a227", "#8e6bbf"   # batch-wide triad
SOLID = "#3a3a3a"

HR_MAX = 190.0          # bpm
VT1, VT2 = 78.0, 88.0   # % of HRmax
# name, label, physiology, lower %HRmax, upper %HRmax, annual hours, colour
ZONES = [
    ("Z1", "Recovery", "fat oxidation", 50.0, 60.0, 206.0,
     lighten(BLUE, 0.58)),
    ("Z2", "Endurance", "aerobic base", 60.0, 70.0, 178.0, BLUE),
    ("Z3", "Tempo", "below VT1", 70.0, 80.0, 44.0, GOLD),
    ("Z4", "Threshold", "VT1 to VT2", 80.0, 90.0, 52.0, lighten(RED, 0.34)),
    ("Z5", "$VO_{2max}$", "above VT2", 90.0, 100.0, 38.0, RED),
]
PHASES = [("General\nprep", [42., 40., 10., 5., 3.]),
          ("Specific\nprep", [38., 34., 12., 10., 6.]),
          ("Pre-\ncompetition", [40., 26., 8., 14., 12.]),
          ("Competition", [45., 22., 6., 12., 15.]),
          ("Transition", [62., 30., 5., 2., 1.])]


def main():
    set_pub_style()
    np.random.default_rng(665)          # fixed seed; the model is analytic

    lo = np.array([z[3] for z in ZONES])
    hi = np.array([z[4] for z in ZONES])
    mid = 0.5 * (lo + hi)
    hours = np.array([z[5] for z in ZONES])
    cols = [z[6] for z in ZONES]
    total = float(hours.sum())
    share = 100.0 * hours / total
    cum = np.cumsum(share)

    # substrate mix and energy turnover as functions of the zone midpoint
    fat_pc = 68.0 * np.exp(-0.5 * ((mid - 52.0) / 17.0) ** 2)
    pro_pc = 3.0 + 0.020 * (mid - 50.0)
    cho_pc = 100.0 - fat_pc - pro_pc
    kcal_min = 3.5 + 0.220 * (mid - 50.0)

    # ---- layout ------------------------------------------------------------
    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.94],
                          left=0.058, right=0.930, top=0.808, bottom=0.152,
                          wspace=0.34, hspace=0.62)
    ax_m = fig.add_subplot(gs[:, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 1])

    # ---- main: zones and annual hours -------------------------------------
    top = hours.max() * 1.46
    for (tag, name, phys, a, b, h, col), m in zip(ZONES, mid):
        ax_m.axvspan(a, b, color=col, alpha=0.15, zorder=0)
        ax_m.bar(m, h, width=(b - a) * 0.80, color=col, edgecolor="white",
                 lw=0.8, zorder=3)
        ax_m.text(m, h + 0.018 * top, f"{h:.0f} h", ha="center", va="bottom",
                  fontsize=6.8, color=SUB, zorder=5)
        ax_m.text(m, h / 2.0, f"{100.0 * h / total:.0f}%", ha="center",
                  va="center", fontsize=6.6, color="white", zorder=5)
        ax_m.text(m, 0.985, f"{tag}  {name}", transform=ax_m.
                  get_xaxis_transform(), ha="center", va="top", fontsize=7.0,
                  fontweight="bold", color=INK, zorder=6)
        ax_m.text(m, 0.945, phys, transform=ax_m.get_xaxis_transform(),
                  ha="center", va="top", fontsize=6.2, color=SUB, zorder=6)
    for x, name, col in ((VT1, "VT1", GREEN), (VT2, "VT2", RED)):
        ax_m.axvline(x, color=col, lw=1.0, ls=(0, (4, 2.5)), zorder=4)
        ax_m.text(x - 0.4, 0.885, name, transform=ax_m.get_xaxis_transform(),
                  ha="right", va="top", fontsize=6.6, fontweight="bold",
                  color=col, zorder=6)

    ax_m.set_xlim(50.0, 100.0)
    ax_m.set_ylim(0.0, top)
    ax_m.set_xticks(np.arange(50, 101, 5))
    ax_m.set_xlabel("intensity (% of $HR_{max}$)", fontsize=8.0)
    ax_m.set_ylabel("annual training time (h)", fontsize=8.0)
    ax_m.tick_params(labelsize=7.0)
    ax_m.grid(axis="y", lw=0.5, color="#f0f0f0", zorder=1)
    ax_m.set_title(f"Where {total:.0f} annual hours land on the intensity "
                   "axis", fontsize=9.0, fontweight="bold", pad=16)

    ax_c = ax_m.twinx()
    ax_c.step(np.concatenate([[50.0], hi]), np.concatenate([[0.0], cum]),
              where="post", color=SOLID, lw=1.2, zorder=6)
    ax_c.plot(hi, cum, ls="none", marker="o", ms=3.4, mfc="white",
              mec=SOLID, mew=0.9, zorder=7)
    ax_c.set_ylim(0.0, 146.0)
    ax_c.set_yticks([0, 25, 50, 75, 100])
    ax_c.set_ylabel("cumulative share of the season (%)", fontsize=7.8,
                    color=SOLID)
    ax_c.tick_params(axis="y", labelsize=6.8, colors=SOLID)
    ax_c.spines["right"].set_visible(True)
    ax_c.spines["right"].set_color("#cccccc")
    ax_c.spines["top"].set_visible(False)

    sec = ax_m.secondary_xaxis(
        "top", functions=(lambda p: p / 100.0 * HR_MAX,
                          lambda h: 100.0 * h / HR_MAX))
    sec.set_xlabel(f"heart rate (bpm, $HR_{{max}}$ = {HR_MAX:.0f})",
                   fontsize=7.4, labelpad=1.5)
    sec.tick_params(labelsize=6.6)

    # Thresholds fall inside bands, not on their edges, so the classical
    # three-zone split apportions each band by the share of its intensity
    # range that lies below VT1, between the thresholds, or above VT2.
    f_easy = np.clip((VT1 - lo) / (hi - lo), 0.0, 1.0)
    f_hard = np.clip((hi - VT2) / (hi - lo), 0.0, 1.0)
    easy = float((share * f_easy).sum())
    hard = float((share * f_hard).sum())
    moderate = 100.0 - easy - hard
    ax_m.text(0.985, 0.40, f"below VT1  {easy:.0f}%\n"
              f"VT1 - VT2  {moderate:.0f}%\n"
              f"above VT2  {hard:.0f}%", transform=ax_m.transAxes,
              ha="right", va="top", fontsize=6.6, color=SUB, linespacing=1.55,
              bbox=dict(boxstyle="round,pad=0.36", facecolor="white",
                        edgecolor="#cccccc", linewidth=0.6, alpha=0.94))

    # ---- aux 1: substrate mix ---------------------------------------------
    xx = np.arange(len(ZONES), dtype=float)
    bottom = np.zeros(len(ZONES))
    for row, col, name in ((fat_pc, FAT, "Fat"), (cho_pc, CARB,
                                                  "Carbohydrate"),
                           (pro_pc, PROTEIN, "Protein")):
        ax_s.bar(xx, row, bottom=bottom, width=0.64, color=col,
                 edgecolor="white", lw=0.7, zorder=3, label=name)
        for x, b, v in zip(xx, bottom, row):
            if v > 11.0:
                ax_s.text(x, b + v / 2.0, f"{v:.0f}", ha="center",
                          va="center", fontsize=6.0, color="white", zorder=5)
        bottom = bottom + row
    ax_s.set_xticks(xx)
    ax_s.set_xticklabels([z[0] for z in ZONES], fontsize=6.6)
    ax_s.set_ylim(0.0, 118.0)
    ax_s.set_yticks([0, 25, 50, 75, 100])
    ax_s.set_ylabel("energy from substrate (%)", fontsize=7.4)
    ax_s.set_xlabel("training zone (-)", fontsize=7.4)
    ax_s.tick_params(labelsize=6.5)
    ax_s.grid(axis="y", lw=0.5, color="#f2f2f2", zorder=0)
    ax_s.set_title("Fuel switches from fat to carbohydrate",
                   fontsize=8.5, fontweight="bold", pad=4)
    ax_s.legend(loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=3,
                frameon=False, fontsize=6.4, handlelength=1.3,
                columnspacing=1.1, borderaxespad=0.0)
    ax_sk = ax_s.twinx()
    ax_sk.plot(xx, kcal_min, color=SOLID, lw=1.2, marker="D", ms=3.4,
               mfc="white", mec=SOLID, mew=0.9, zorder=6)
    ax_sk.set_ylim(0.0, kcal_min.max() * 1.42)
    ax_sk.set_ylabel("energy turnover (kcal/min)", fontsize=7.4, color=SOLID)
    ax_sk.tick_params(axis="y", labelsize=6.5, colors=SOLID)
    ax_sk.spines["right"].set_visible(True)
    ax_sk.spines["right"].set_color("#cccccc")
    ax_sk.spines["top"].set_visible(False)

    # ---- aux 2: periodisation ---------------------------------------------
    px = np.arange(len(PHASES), dtype=float)
    alloc = np.array([p[1] for p in PHASES])
    bottom = np.zeros(len(PHASES))
    for j, (tag, _, _, _, _, _, col) in enumerate(ZONES):
        v = alloc[:, j]
        ax_p.bar(px, v, bottom=bottom, width=0.66, color=col,
                 edgecolor="white", lw=0.7, zorder=3)
        for x, b, vv in zip(px, bottom, v):
            if vv >= 9.0:
                ax_p.text(x, b + vv / 2.0, tag, ha="center", va="center",
                          fontsize=6.0, color="white", zorder=5)
        bottom = bottom + v
    intense = alloc[:, 3] + alloc[:, 4]
    ax_p.plot(px, intense, color=SOLID, lw=1.1, marker="o", ms=3.4,
              mfc="white", mec=SOLID, mew=0.9, zorder=6)
    ax_p.text(px[3], intense[3] + 5.0, "time in Z4 and Z5", ha="center",
              va="bottom", fontsize=6.2, color=SOLID, zorder=7)
    ax_p.set_xticks(px)
    ax_p.set_xticklabels([p[0] for p in PHASES], fontsize=6.2,
                         linespacing=1.25)
    ax_p.set_ylim(0.0, 108.0)
    ax_p.set_yticks([0, 25, 50, 75, 100])
    ax_p.set_ylabel("weekly training time (%)", fontsize=7.4)
    ax_p.set_xlabel("periodisation phase (-)", fontsize=7.4)
    ax_p.tick_params(labelsize=6.5)
    ax_p.grid(axis="y", lw=0.5, color="#f2f2f2", zorder=0)
    ax_p.set_title("Hard hours grow as the season closes in", fontsize=8.5,
                   fontweight="bold", pad=4)

    panel_labels_inches(fig, [ax_m, ax_s, ax_p], dx=-0.44, dy=0.10)

    print(f"total {total:.0f} h; share " +
          " ".join(f"{z[0]} {s:.1f}%" for z, s in zip(ZONES, share)))
    print(f"three-zone split: below VT1 {easy:.1f}%, moderate "
          f"{moderate:.1f}%, above VT2 {hard:.1f}%")
    print("fat share by zone: " +
          " ".join(f"{v:.0f}%" for v in fat_pc) +
          "   kcal/min: " + " ".join(f"{v:.1f}" for v in kcal_min))
    savefig(fig, "665_spo_training_zones.png")


if __name__ == "__main__":
    main()

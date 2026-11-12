"""Knee range of motion through a twelve-week rehabilitation programme.

Main panel: weekly active flexion range for a simulated cohort after knee
arthroplasty, drawn as one box per review week over the individual patient
trajectories that produced it.  The uninvolved-limb reference band is the grey
strip; the discharge target is the dashed rule.  Every trajectory is a
saturating recovery curve, so the boxes, the medians quoted on the panel and
the attainment counts are all read back out of the same array.

Auxiliary (both at 58 % of the main panel width): the cumulative share of the
cohort that has reached the target by each review, stepped at the week each
patient first crosses it; and the week-on-week median gain, which is what
turns the plateau after week eight into a number.

Synthetic cohort, illustrative values only; no real patient is depicted.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
GRID = "#d5d5d5"
BLUE = "#3b5b92"
RUST = "#c0603f"
ACCENT = "#b23a48"
TEAL = "#2e7d5b"
NORM = "#9a9a9a"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.86, 0.86, 5.54, 4.62)
AUX_T_RECT = (7.62, 3.48, 3.16, 2.00)
AUX_B_RECT = (7.62, 0.86, 3.16, 2.00)

N_PATIENTS = 24
WEEKS = np.arange(0, 13, 1)
TARGET = 120.0          # deg of active flexion wanted at discharge
NORM_MEAN = 137.0       # uninvolved limb, deg
NORM_SD = 5.5


def cohort(rng):
    """Saturating flexion recovery, one curve per simulated patient."""
    start = rng.normal(62.0, 8.5, N_PATIENTS)
    plateau = rng.normal(126.0, 11.0, N_PATIENTS)
    tau = rng.normal(3.4, 0.9, N_PATIENTS).clip(1.6, 6.5)
    w = WEEKS[None, :].astype(float)
    curves = (start[:, None]
              + (plateau[:, None] - start[:, None])
              * (1.0 - np.exp(-w / tau[:, None])))
    curves += rng.normal(0.0, 2.6, curves.shape)
    curves[:, 0] = start
    return np.maximum.accumulate(curves, axis=1).clip(35.0, 150.0)


def first_crossing(curves, level):
    """Week index at which each patient first reaches ``level``; -1 if never."""
    hit = curves >= level
    out = np.full(curves.shape[0], -1, dtype=int)
    for i in range(curves.shape[0]):
        idx = np.flatnonzero(hit[i])
        if idx.size:
            out[i] = int(idx[0])
    return out


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(978)

    rom = cohort(rng)
    cross = first_crossing(rom, TARGET)
    reached = cross >= 0
    med = np.median(rom, axis=0)
    gain = np.diff(med)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)
        a.grid(color=GRID, lw=0.5, alpha=0.6, zorder=0)

    # ---- main: weekly boxes over the individual trajectories ---------------
    ax.axhspan(NORM_MEAN - NORM_SD, NORM_MEAN + NORM_SD,
               color=lighten(NORM, 0.72), lw=0, zorder=1)
    ax.axhline(NORM_MEAN, color=NORM, lw=0.9, ls=(0, (4.0, 2.4)), zorder=2)
    ax.text(0.08, NORM_MEAN + NORM_SD + 1.4,
            f"uninvolved limb {NORM_MEAN:.0f} $\\pm$ {NORM_SD:.1f} deg "
            "(mean $\\pm$ 1 SD)", ha="left", va="bottom", fontsize=6.3,
            color=SUB)
    ax.axhline(TARGET, color=ACCENT, lw=1.1, ls=(0, (3.2, 2.2)), zorder=3)
    ax.text(0.08, TARGET - 1.8, f"discharge target {TARGET:.0f} deg",
            ha="left", va="top", fontsize=6.6, color=ACCENT)

    for i in range(N_PATIENTS):
        ax.plot(WEEKS, rom[i], color=BLUE if reached[i] else RUST, lw=0.55,
                alpha=0.34, zorder=2, solid_capstyle="round")

    bp = ax.boxplot([rom[:, k] for k in range(len(WEEKS))], positions=WEEKS,
                    widths=0.52, patch_artist=True, manage_ticks=False,
                    showfliers=True, zorder=5)
    for k, box in enumerate(bp["boxes"]):
        frac = k / (len(WEEKS) - 1.0)
        box.set(facecolor=lighten(BLUE, 0.78 - 0.36 * frac),
                edgecolor=BLUE, linewidth=0.85)
    for key, col, lw in (("whiskers", BLUE, 0.8), ("caps", BLUE, 0.8),
                         ("medians", INK, 1.3)):
        for art in bp[key]:
            art.set(color=col, linewidth=lw)
    for art in bp["fliers"]:
        art.set(marker="o", markersize=2.2, markerfacecolor="none",
                markeredgecolor=MUTED, markeredgewidth=0.5)

    ax.plot(WEEKS, med, color=INK, lw=1.5, zorder=6, solid_capstyle="round")
    ax.plot(WEEKS, med, ls="none", marker="o", ms=3.0, mfc="white", mec=INK,
            mew=0.9, zorder=7)

    n_hit = int(reached.sum())
    ax.text(0.08, 44.0,
            f"week 0 median {med[0]:.0f} deg   ->   week 12 median "
            f"{med[-1]:.0f} deg\n"
            f"{n_hit} of {N_PATIENTS} patients reach the target "
            f"({100.0 * n_hit / N_PATIENTS:.0f} %), "
            f"{N_PATIENTS - n_hit} do not",
            ha="left", va="bottom", fontsize=6.6, color=SUB, linespacing=1.5)

    handles = [
        Line2D([], [], color=BLUE, lw=1.1, label="reaches target by week 12"),
        Line2D([], [], color=RUST, lw=1.1, label="short of target at week 12"),
        Line2D([], [], color=INK, lw=1.5, marker="o", ms=3.0, mfc="white",
               mew=0.9, label="cohort median"),
    ]
    leg = ax.legend(handles=handles, loc="lower right", fontsize=6.5,
                    frameon=True, framealpha=0.94, borderpad=0.42,
                    handlelength=1.5, labelspacing=0.34)
    leg.get_frame().set(edgecolor=HAIR, linewidth=0.6)

    ax.set_xlim(-0.6, 12.6)
    ax.set_ylim(38.0, 152.0)
    ax.set_xticks(WEEKS)
    ax.set_yticks(np.arange(40, 151, 20))
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel("weeks since surgery (review visits)", fontsize=8.2,
                  labelpad=2.5)
    ax.set_ylabel("active knee flexion range of motion (deg)", fontsize=8.2,
                  labelpad=3.0)
    ax.set_title(f"Twelve-week flexion recovery, n={N_PATIENTS} patients",
                 fontsize=9.0, fontweight="bold", pad=5.0)

    # ---- aux (b): cumulative attainment ------------------------------------
    cum = np.array([100.0 * np.mean((cross >= 0) & (cross <= k))
                    for k in range(len(WEEKS))])
    ax_t.step(WEEKS, cum, where="post", color=BLUE, lw=1.5, zorder=4)
    ax_t.fill_between(WEEKS, 0.0, cum, step="post",
                      color=lighten(BLUE, 0.82), lw=0, zorder=2)
    hit_weeks = cross[reached]
    med_week = float(np.median(hit_weeks))
    ax_t.axvline(med_week, color=ACCENT, lw=1.0, ls=(0, (3.0, 2.2)), zorder=5)
    ax_t.text(med_week + 0.25, 16.0,
              f"median {med_week:.1f} weeks\nto target", ha="left",
              va="bottom", fontsize=6.2, color=ACCENT, linespacing=1.4)
    ax_t.axhline(100.0 * n_hit / N_PATIENTS, color=MUTED, lw=0.8,
                 ls=(0, (2.4, 2.0)), zorder=3)
    ax_t.text(12.4, 100.0 * n_hit / N_PATIENTS + 3.0,
              f"{100.0 * n_hit / N_PATIENTS:.0f} % by week 12", ha="right",
              va="bottom", fontsize=6.2, color=SUB)
    ax_t.set_xlim(-0.6, 12.6)
    ax_t.set_ylim(0.0, 108.0)
    ax_t.set_xticks(np.arange(0, 13, 2))
    ax_t.set_yticks([0, 25, 50, 75, 100])
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("weeks since surgery", fontsize=7.6, labelpad=1.5)
    ax_t.set_ylabel("cohort at target (%)", fontsize=7.6, labelpad=2.0)
    ax_t.set_title("When the target is met", fontsize=8.4,
                   fontweight="bold", pad=4.0)

    # ---- aux (c): weekly median gain ---------------------------------------
    xg = WEEKS[1:]
    plateau_lim = 2.0
    cols = [lighten(TEAL, 0.30) if g >= plateau_lim else lighten(MUTED, 0.42)
            for g in gain]
    ax_b.bar(xg, gain, width=0.66, color=cols, edgecolor="white",
             linewidth=0.7, zorder=3)
    ax_b.axhline(plateau_lim, color=ACCENT, lw=1.0, ls=(0, (3.0, 2.2)),
                 zorder=4)
    slow = xg[gain < plateau_lim]
    if slow.size:
        ax_b.axvspan(slow[0] - 0.5, 12.5, color=lighten(RUST, 0.88), lw=0,
                     zorder=1)
        ax_b.text(12.4, gain.max() * 0.86,
                  f"plateau from\nweek {slow[0]:d}", ha="right", va="top",
                  fontsize=6.2, color=RUST, linespacing=1.4)
    ax_b.text(2.3, gain.max() * 1.02,
              f"total median gain\n{med[-1] - med[0]:.0f} deg over 12 weeks",
              ha="left", va="top", fontsize=6.2, color=SUB, linespacing=1.4)
    ax_b.text(12.45, plateau_lim + 0.4, f"{plateau_lim:.0f} deg/week",
              ha="right", va="bottom", fontsize=6.1, color=ACCENT)
    ax_b.set_xlim(0.4, 12.6)
    ax_b.set_ylim(0.0, gain.max() * 1.16)
    ax_b.set_xticks(np.arange(2, 13, 2))
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("weeks since surgery", fontsize=7.6, labelpad=1.5)
    ax_b.set_ylabel("median gain (deg/week)", fontsize=7.6, labelpad=2.0)
    ax_b.set_title("The gain that dries up", fontsize=8.4,
                   fontweight="bold", pad=4.0)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.52, dy=0.10)
    savefig(fig, "978_reh_rom_progression.png")


if __name__ == "__main__":
    main()

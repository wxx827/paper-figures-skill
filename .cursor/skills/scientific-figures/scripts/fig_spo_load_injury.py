"""Training-load monitoring: acute-to-chronic workload ratio against injuries.

One athlete's season is simulated day by day.  Daily load is session RPE times
session duration, shaped by a weekly microcycle and a pre-season ramp; the
acute load is the trailing 7-day mean, the chronic load the trailing 28-day
mean, and their ratio is the ACWR.  Injuries are not scripted: each day the
athlete is exposed to a hazard that rises exponentially with that day's ACWR,
and a hit forces a rehabilitation block with a reduced-load return to play.
The risk panel plots the same hazard function against the ACWR distribution
the athlete actually experienced, so the two right-hand panels explain the
left-hand one rather than illustrating it.
"""
import textwrap

import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import annotate_box, get_cmap, savefig, set_pub_style
from figkit import panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
BLUE, LIGHT = "#3b5b92", "#9cc3dd"
GOLD, GREEN, RED = "#c9a227", "#2e7d5b", "#b23a48"

N_DAYS = 182
ACUTE_W, CHRONIC_W = 7, 28
SWEET = (0.80, 1.30)      # the commonly quoted ACWR sweet spot
DANGER = 1.50
REHAB = 13                # days out after an injury
HAZARD_BASE = 0.018       # daily injury probability at ACWR = 1.0
HAZARD_K = 3.0


def rolling_mean(x, w):
    """Trailing mean over ``w`` days, using whatever history exists."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.empty(x.size)
    for i in range(x.size):
        lo = max(0, i + 1 - w)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def hazard(acwr):
    """Daily injury probability as a function of the ACWR."""
    return np.clip(HAZARD_BASE * np.exp(HAZARD_K * (acwr - 1.0)), 0.0, 0.09)


def main():
    set_pub_style()
    rng = np.random.default_rng(611)

    # weekly microcycle: two hard days, a taper and a match day
    cycle = np.array([0.30, 1.28, 1.05, 0.55, 1.22, 0.45, 1.00])
    ramp = 0.62 + 0.38 * np.clip(np.arange(N_DAYS) / 46.0, 0.0, 1.0)
    # a warm-weather camp, a mid-season break and a congested fixture block:
    # the three places where a real ACWR trace leaves the sweet spot
    block = np.ones(N_DAYS)
    block[58:70] = 1.70
    block[70:76] = 0.55
    block[104:112] = 0.30
    block[138:148] = 1.55

    load = np.zeros(N_DAYS)
    acute = np.zeros(N_DAYS)
    chronic = np.zeros(N_DAYS)
    acwr = np.full(N_DAYS, np.nan)
    rehab_left = 0
    injuries = []

    for d in range(N_DAYS):
        if rehab_left > 0:
            frac = 1.0 - rehab_left / float(REHAB)
            base = 520.0 * ramp[d] * (0.12 + 0.72 * frac)
            rehab_left -= 1
        else:
            base = 520.0 * ramp[d] * cycle[d % 7] * block[d]
        load[d] = max(0.0, base * rng.normal(1.0, 0.14))
        acute[d] = rolling_mean(load[:d + 1], ACUTE_W)[-1]
        chronic[d] = rolling_mean(load[:d + 1], CHRONIC_W)[-1]
        if d >= CHRONIC_W and chronic[d] > 1.0:
            acwr[d] = acute[d] / chronic[d]
            if rehab_left == 0 and rng.random() < hazard(acwr[d]):
                injuries.append(d)
                rehab_left = REHAB

    day = np.arange(1, N_DAYS + 1)
    ok = np.isfinite(acwr)
    rpe = np.clip(load / 62.0 + rng.normal(0.0, 0.45, N_DAYS), 1.0, 10.0)
    sleep = np.clip(8.1 - 0.0017 * (load - load.mean())
                    + rng.normal(0.0, 0.42, N_DAYS), 4.0, 9.5)

    # ---- layout ------------------------------------------------------------
    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.30],
                          left=0.055, right=0.905, top=0.845, bottom=0.170,
                          wspace=0.30, hspace=0.30)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[1, 0], sharex=ax_l)
    gs_r = gs[:, 1].subgridspec(2, 1, height_ratios=[1.0, 0.86], hspace=0.60)
    ax_k = fig.add_subplot(gs_r[0])
    ax_h = fig.add_subplot(gs_r[1])

    # ---- load and ACWR -----------------------------------------------------
    ax_a = ax_l.twinx()
    ax_a.axhspan(DANGER, 2.2, color=RED, alpha=0.09, zorder=0)
    ax_a.axhspan(SWEET[1], DANGER, color=GOLD, alpha=0.09, zorder=0)
    ax_a.axhspan(*SWEET, color=GREEN, alpha=0.08, zorder=0)
    ax_a.axhspan(0.0, SWEET[0], color=GOLD, alpha=0.09, zorder=0)
    ax_a.plot(day[ok], acwr[ok], color=RED, lw=1.3, zorder=4)
    ax_a.axhline(DANGER, color=RED, lw=0.8, ls="--", zorder=3)
    ax_a.set_ylim(0.0, 2.2)
    ax_a.set_ylabel("acute : chronic workload ratio (-)", fontsize=7.8,
                    color=RED)
    ax_a.tick_params(axis="y", labelsize=6.8, colors=RED)
    ax_a.spines["right"].set_visible(True)
    ax_a.spines["right"].set_color(RED)
    ax_a.spines["top"].set_visible(False)
    for txt, yv, col in ((f"danger, ACWR > {DANGER:.1f}", DANGER + 0.05, RED),
                         (f"sweet spot {SWEET[0]:.1f} - {SWEET[1]:.1f}",
                          SWEET[0] + 0.04, GREEN)):
        t = ax_a.text(N_DAYS - 1.0, yv, txt, ha="right", va="bottom",
                      fontsize=6.2, color=col, zorder=6)
        t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])

    ax_l.set_zorder(ax_a.get_zorder() + 1)
    ax_l.patch.set_visible(False)
    ax_l.bar(day, load, width=0.85, color="#e2e2e2", edgecolor="none",
             zorder=2)
    ax_l.plot(day, chronic, color=LIGHT, lw=2.0, zorder=3)
    ax_l.plot(day, acute, color=BLUE, lw=1.5, zorder=4)

    for k, d0 in enumerate(injuries):
        ax_l.axvline(d0 + 1, color=RED, lw=0.9, ls=":", zorder=5)
        ax_a.plot([d0 + 1], [acwr[d0]], marker="v", ms=5.4, color=RED,
                  mec="white", mew=0.7, zorder=8)
        t = ax_a.text(d0 + 2.5, acwr[d0] + 0.05, f"I{k + 1}", ha="left",
                      va="bottom", fontsize=6.2, color=RED, zorder=8)
        t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])

    ax_l.set_ylim(0.0, load.max() * 1.22)
    ax_l.set_ylabel("training load (AU per day)", fontsize=7.8)
    ax_l.set_xlim(0.0, N_DAYS + 1.0)
    ax_l.tick_params(labelsize=6.8, labelbottom=False)
    ax_l.grid(axis="y", lw=0.5, color="#f0f0f0", zorder=0)
    ax_l.set_title("Workload, its ratio and the injuries that followed",
                   fontsize=9.0, fontweight="bold", pad=10)

    hi = float(np.mean(acwr[ok] > DANGER) * 100.0)
    annotate_box(ax_l,
                 f"season load  {load.sum() / 1000.0:.1f} kAU over "
                 f"{N_DAYS} d\n"
                 f"ACWR median  {np.nanmedian(acwr):.2f}, "
                 f"{hi:.0f}% of days above {DANGER:.1f}\n"
                 f"injuries  {len(injuries)} "
                 f"({len(injuries) * REHAB} d of modified training)",
                 loc="upper left", fontsize=6.6, margin=0.014)

    # ---- wellness strips ---------------------------------------------------
    edges = np.arange(0.5, N_DAYS + 1.5, 1.0)
    ax_s.pcolormesh(edges, [1.0, 2.0], rpe[None, :],
                    cmap=get_cmap("cb_sequential_warm"), vmin=1.0, vmax=10.0,
                    shading="flat", zorder=2)
    ax_s.pcolormesh(edges, [0.0, 1.0], sleep[None, :],
                    cmap=get_cmap("cb_sequential_blue"), vmin=4.0, vmax=9.5,
                    shading="flat", zorder=2)
    for d0 in injuries:
        ax_s.axvline(d0 + 1, color=RED, lw=0.9, ls=":", zorder=4)
    ax_s.set_ylim(0.0, 2.0)
    ax_s.set_yticks([0.5, 1.5])
    ax_s.set_yticklabels(["sleep\n(4-9.5 h)", "session RPE\n(1-10 AU)"],
                         fontsize=6.2)
    ax_s.tick_params(axis="y", length=0)
    ax_s.tick_params(axis="x", labelsize=6.8)
    ax_s.set_xlabel("day of pre-season and season (d)", fontsize=7.8)
    ax_s.set_xticks(np.arange(14, N_DAYS + 1, 14))
    for s in ax_s.spines.values():
        s.set_visible(False)
    ax_s.text(1.0, 2.10, "darker = higher", transform=ax_s.transAxes,
              ha="right", va="bottom", fontsize=6.2, color=MUTED)

    # ---- modelled risk against ACWR ---------------------------------------
    bins = np.array([0.0, 0.80, 1.00, 1.15, 1.30, 1.50, 2.20])
    lab = ["<0.8", "0.8-1.0", "1.0-1.15", "1.15-1.3", "1.3-1.5", ">1.5"]
    idx = np.clip(np.digitize(acwr[ok], bins) - 1, 0, len(lab) - 1)
    risk = np.array([100.0 * hazard(acwr[ok][idx == b]).mean()
                     if np.any(idx == b) else 0.0
                     for b in range(len(lab))])
    n_day = np.array([int(np.sum(idx == b)) for b in range(len(lab))])
    inj_bin = np.zeros(len(lab))
    for d0 in injuries:
        b = int(np.clip(np.digitize(acwr[d0], bins) - 1, 0, len(lab) - 1))
        inj_bin[b] += 1.0

    cols = [GOLD if i in (0,) else (RED if i >= 4 else GREEN)
            for i in range(len(lab))]
    xx = np.arange(len(lab))
    ax_k.bar(xx, risk, width=0.62, color=cols, edgecolor="white", lw=0.6,
             zorder=3)
    top = max(risk.max() * 1.55, 0.5)
    for x, r, nd, ib in zip(xx, risk, n_day, inj_bin):
        if r > 0.22 * top:
            ax_k.text(x, r * 0.5, f"{nd} d", ha="center", va="center",
                      fontsize=6.0, color="white", zorder=5)
        else:
            ax_k.text(x, r + 0.02 * top, f"{nd} d", ha="center", va="bottom",
                      fontsize=6.0, color=SUB, zorder=5)
        if ib:
            ax_k.plot([x] * int(ib),
                      [r + (0.11 + 0.05 * i) * top for i in range(int(ib))],
                      marker="v", ms=4.2, color=RED, ls="none", zorder=5)
    ax_k.set_xticks(xx)
    ax_k.set_xticklabels(lab, fontsize=6.2)
    ax_k.set_xlabel("acute : chronic workload ratio (-)", fontsize=7.4)
    ax_k.set_ylabel("injury risk (% per day)", fontsize=7.4)
    ax_k.set_ylim(0.0, top)
    ax_k.tick_params(axis="y", labelsize=6.5)
    ax_k.grid(axis="y", lw=0.5, color="#ececec", zorder=0)
    ax_k.set_title("Risk rises with the ratio", fontsize=8.5,
                   fontweight="bold")

    # ---- ACWR distribution -------------------------------------------------
    ax_h.axvspan(*SWEET, color=GREEN, alpha=0.10, zorder=0)
    ax_h.hist(acwr[ok], bins=np.arange(0.35, 2.05, 0.075), color=LIGHT,
              edgecolor="white", lw=0.5, zorder=3)
    ax_h.axvline(np.nanmedian(acwr), color=BLUE, lw=1.2, zorder=4)
    ax_h.axvline(DANGER, color=RED, lw=1.0, ls="--", zorder=4)
    ax_h.text(np.nanmedian(acwr) + 0.03, 0.94,
              f"median {np.nanmedian(acwr):.2f}",
              transform=ax_h.get_xaxis_transform(), ha="left", va="top",
              fontsize=6.2, color=BLUE)
    ax_h.set_xlim(0.35, 2.0)
    ax_h.set_xlabel("acute : chronic workload ratio (-)", fontsize=7.4)
    ax_h.set_ylabel("days (d)", fontsize=7.4)
    ax_h.tick_params(labelsize=6.5)
    ax_h.grid(axis="y", lw=0.5, color="#ececec", zorder=0)
    ax_h.set_title(f"{hi:.0f}% of days sat in the danger band", fontsize=8.5,
                   fontweight="bold")

    fig.legend(handles=[
        Line2D([0], [0], color=BLUE, lw=1.5,
               label=f"acute load ({ACUTE_W} d mean)"),
        Line2D([0], [0], color=LIGHT, lw=2.0,
               label=f"chronic load ({CHRONIC_W} d mean)"),
        Patch(facecolor="#e2e2e2", edgecolor="none", label="daily load"),
        Line2D([0], [0], color=RED, lw=1.3, label="ACWR"),
        Line2D([0], [0], color=RED, lw=0.9, ls=":", marker="v", ms=4.5,
               label="injury event")],
        loc="lower center", bbox_to_anchor=(0.5, 0.068), ncol=5,
        frameon=False, fontsize=7.0, handlelength=1.8, columnspacing=1.9)

    panel_labels_inches(fig, [ax_l, ax_s, ax_k, ax_h], dx=-0.40, dy=0.10)

    n_inj = len(injuries)
    plural = "injury" if n_inj == 1 else "injuries"
    if n_inj == 1:
        head = (f"The one injury struck at an ACWR of "
                f"{float(acwr[injuries[0]]):.2f}")
    elif n_inj:
        head = (f"{'Both' if n_inj == 2 else f'All {n_inj}'} injuries struck "
                f"at an ACWR of "
                f"{min(float(acwr[d]) for d in injuries):.2f} or higher")
    else:
        head = "A season spent inside the workload sweet spot"

    print(f"total load {load.sum() / 1000.0:.1f} kAU  ACWR median "
          f"{np.nanmedian(acwr):.2f}  days > {DANGER}: {hi:.1f}%")
    print(f"injuries on days {[d + 1 for d in injuries]}  "
          f"ACWR at injury {[round(float(acwr[d]), 2) for d in injuries]}")
    print("days per ACWR bin:", dict(zip(lab, n_day.tolist())))
    savefig(fig, "611_spo_load_injury.png")


if __name__ == "__main__":
    main()

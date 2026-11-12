"""Player ability radar against a league baseline, with percentile evidence.

A reference population of 240 same-position players is drawn once, with a
different mean and spread on each of the six ability axes.  The pale polygon
on the radar is that population's mean, the three player polygons are single
draws pushed by a per-player role offset, and the percentile bars on the right
count how much of the population each player actually beats - so the bars
cannot drift away from the radar they explain.  Overall ratings are the mean
of the six axes and the minutes panel is the same three players' season load.

All three polygons and the baseline share one radial scale (0-100), because
per-player autoscaling on a radar is the fastest way to make a weak profile
look strong.
"""
import textwrap

import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import annotate_box, savefig, set_pub_style
from figkit import panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
BLUE, GOLD, PURPLE = "#3b5b92", "#c9a227", "#8e6bbf"
BASE = "#b3b3b3"

AXES = ["attacking", "passing", "defending", "physical", "technical",
        "awareness"]
POP_MEAN = np.array([58.0, 62.0, 55.0, 60.0, 61.0, 57.0])
POP_SD = np.array([12.0, 10.5, 13.0, 9.5, 11.0, 12.5])
N_POP = 240

# name, colour, per-axis offset from the population mean, season minutes
PLAYERS = [
    ("A. Ruiz (10)", BLUE, np.array([22.0, 19.0, -4.0, 6.0, 21.0, 17.0]),
     2680),
    ("M. Kovac (8)", GOLD, np.array([6.0, 15.0, 14.0, 12.0, 8.0, 12.0]),
     2410),
    ("T. Ellis (23)", PURPLE, np.array([15.0, 4.0, -9.0, 17.0, 9.0, 2.0]),
     1265),
]


def main():
    set_pub_style()
    rng = np.random.default_rng(609)

    n_ax = len(AXES)
    pop = np.clip(rng.normal(POP_MEAN, POP_SD, size=(N_POP, n_ax)), 5.0, 99.0)
    league = pop.mean(axis=0)

    ratings, pcts = [], []
    for _, _, off, _ in PLAYERS:
        r = np.clip(POP_MEAN + off + rng.normal(0.0, 3.2, n_ax), 5.0, 99.0)
        ratings.append(r)
        pcts.append(100.0 * (pop < r).mean(axis=0))
    ratings = np.array(ratings)
    pcts = np.array(pcts)
    overall = ratings.mean(axis=1)
    minutes = np.array([p[3] for p in PLAYERS], dtype=float)

    # ---- layout ------------------------------------------------------------
    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.50],
                          left=0.045, right=0.965, top=0.815, bottom=0.185,
                          wspace=0.26, hspace=0.62)
    ax_r = fig.add_subplot(gs[:, 0], projection="polar")
    ax_p = fig.add_subplot(gs[0, 1])
    ax_m = fig.add_subplot(gs[1, 1])

    # ---- radar -------------------------------------------------------------
    th = np.linspace(0.0, 2.0 * np.pi, n_ax, endpoint=False)
    th_c = np.concatenate([th, th[:1]])
    ax_r.set_theta_offset(np.pi / 2.0)
    ax_r.set_theta_direction(-1)

    lg = np.concatenate([league, league[:1]])
    ax_r.fill(th_c, lg, color=BASE, alpha=0.30, lw=0.0, zorder=2)
    ax_r.plot(th_c, lg, color="#8a8a8a", lw=1.0, ls="--", zorder=3)

    for k, (name, col, _, _) in enumerate(PLAYERS):
        v = np.concatenate([ratings[k], ratings[k][:1]])
        lw = 2.0 if k == 0 else 1.2
        ax_r.plot(th_c, v, color=col, lw=lw, zorder=6 - k,
                  marker="o", ms=3.2 if k == 0 else 2.4, mew=0.0)
        ax_r.fill(th_c, v, color=col, alpha=0.16 if k == 0 else 0.07, lw=0.0,
                  zorder=4 - k)

    ax_r.set_ylim(0.0, 100.0)
    ax_r.set_yticks([20.0, 40.0, 60.0, 80.0, 100.0])
    ax_r.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=6.0,
                         color=SUB)
    ax_r.set_rlabel_position(90.0)
    for t in ax_r.get_yticklabels():
        t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
    ax_r.set_xticks(th)
    ax_r.set_xticklabels([a for a in AXES], fontsize=7.6, color=INK)
    ax_r.tick_params(axis="x", pad=4.0)
    ax_r.grid(color="#e0e0e0", lw=0.6)
    ax_r.spines["polar"].set_color("#cccccc")
    ax_r.spines["polar"].set_linewidth(0.8)
    ax_r.set_title("Six-axis profile against the league baseline",
                   fontsize=9.0, fontweight="bold", pad=9)

    annotate_box(ax_r,
                 "\n".join(f"{nm}   overall {ov:.1f}"
                           for (nm, _, _, _), ov in zip(PLAYERS, overall)) +
                 f"\nleague mean overall  {league.mean():.1f}",
                 loc="upper left", fontsize=6.6, margin=-0.02)

    # ---- percentile bars ---------------------------------------------------
    yy = np.arange(n_ax)[::-1]
    ax_p.barh(yy, np.full(n_ax, 100.0), height=0.62, color="#f0f0f0",
              edgecolor="none", zorder=1)
    ax_p.barh(yy, pcts[0], height=0.62, color=BLUE, edgecolor="white",
              lw=0.6, zorder=3)
    for k, (name, col, _, _) in enumerate(PLAYERS[1:], start=1):
        ax_p.scatter(pcts[k], yy, marker="|", s=90.0, c=col, linewidths=1.5,
                     zorder=5)
    ax_p.axvline(50.0, color="#808080", lw=0.8, ls="--", zorder=4)
    for y, v in zip(yy, pcts[0]):
        inside = v > 86.0
        ax_p.text(v - 2.0 if inside else v + 2.0, y, f"{v:.0f}",
                  ha="right" if inside else "left", va="center",
                  fontsize=6.2, color="white" if inside else SUB, zorder=6)
    ax_p.set_yticks(yy)
    ax_p.set_yticklabels(AXES, fontsize=6.6)
    ax_p.set_xlim(0.0, 108.0)
    ax_p.set_xticks([0, 25, 50, 75, 100])
    ax_p.set_xlabel("percentile among same-position players (%)",
                    fontsize=7.4)
    ax_p.tick_params(axis="x", labelsize=6.5)
    ax_p.grid(axis="x", lw=0.5, color="#ececec", zorder=0)
    ax_p.set_title(f"Where {PLAYERS[0][0]} ranks, n = {N_POP}", fontsize=8.5,
                   fontweight="bold")

    # ---- minutes -----------------------------------------------------------
    xx = np.arange(len(PLAYERS))
    ax_m.bar(xx, minutes, width=0.52,
             color=[p[1] for p in PLAYERS], edgecolor="white", lw=0.6,
             zorder=3)
    for x, m in zip(xx, minutes):
        ax_m.text(x, m + 60.0, f"{m:.0f}", ha="center", va="bottom",
                  fontsize=6.4, color=SUB)
    ax_m.set_xticks(xx)
    ax_m.set_xticklabels([p[0].split(" (")[0] for p in PLAYERS], fontsize=6.6)
    ax_m.set_ylim(0.0, minutes.max() * 1.24)
    ax_m.set_ylabel("season minutes (min)", fontsize=7.4)
    ax_m.tick_params(axis="y", labelsize=6.5)
    ax_m.grid(axis="y", lw=0.5, color="#ececec", zorder=0)
    ax_m.set_title("Sample behind each profile", fontsize=8.5,
                   fontweight="bold")

    handles = [Line2D([0], [0], color=col, lw=2.0 if k == 0 else 1.2,
                      marker="o", ms=3.4, mew=0.0, label=nm)
               for k, (nm, col, _, _) in enumerate(PLAYERS)]
    handles.append(Line2D([0], [0], color="#8a8a8a", lw=1.0, ls="--",
                          label=f"league mean (n = {N_POP})"))
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.080), ncol=4, frameon=False,
               fontsize=7.0, handlelength=1.8, columnspacing=2.0)

    panel_labels_inches(fig, [ax_r, ax_p, ax_m], dx=-0.30, dy=0.14)

    best = int(np.argmax(pcts[0]))
    worst = int(np.argmin(pcts[0]))
    tail = ("below the league median" if pcts[0][worst] < 50.0
            else "the clear weak axis")

    for (nm, _, _, _), r, p, ov in zip(PLAYERS, ratings, pcts, overall):
        print(f"{nm}: overall {ov:.1f}  ratings {r.round(1).tolist()}  "
              f"pct {p.round(0).tolist()}")
    print("league mean:", league.round(1).tolist())
    savefig(fig, "609_spo_player_radar.png")


if __name__ == "__main__":
    main()

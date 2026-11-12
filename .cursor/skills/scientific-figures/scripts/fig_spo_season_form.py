"""Season form: league position round by round, with the results behind it.

A whole 12-team, 34-round season is simulated once.  Each round the teams are
paired at random, the home and away scorelines are Poisson draws whose means
come from the two team strengths, and the table is then rebuilt from scratch
after every round on points, then goal difference, then goals scored.  The
ladder panel, the result ribbon, the goal-difference bars, the home-away split
and the cumulative points curve are five views of that one season, so no panel
can contradict another.

The position axis is inverted: first place is at the top, where a reader
expects to find it.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import annotate_box, savefig, set_pub_style
from figkit import panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
BLUE, GOLD, GREEN, RED = "#3b5b92", "#c9a227", "#2e7d5b", "#b23a48"
FAINT = "#d5d5d5"
DRAW = "#b3b3b3"

N_TEAMS, N_ROUNDS = 12, 34
OWN, RIV1, RIV2 = 0, 1, 2
NAMES = ["Harbour City", "Northgate", "Riverside", "Ashford", "Kingsmoor",
         "Beltham", "Cranwell", "Dunmore", "Eastvale", "Fairholt",
         "Greystone", "Holloway"]
CODES = ["HAR", "NOR", "RIV", "ASH", "KGM", "BEL", "CRA", "DUN", "EAS",
         "FAI", "GRE", "HOL"]
EURO_RANK = 4.5           # top four qualify
DROP_RANK = 9.5           # bottom three go down


def ordinal(n):
    n = int(n)
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def main():
    set_pub_style()
    rng = np.random.default_rng(610)

    strength = np.linspace(1.55, 0.80, N_TEAMS)
    strength = strength[rng.permutation(N_TEAMS)]
    strength[OWN] = 1.34

    pts = np.zeros((N_TEAMS, N_ROUNDS))
    gf = np.zeros((N_TEAMS, N_ROUNDS))
    ga = np.zeros((N_TEAMS, N_ROUNDS))
    at_home = np.zeros((N_TEAMS, N_ROUNDS), dtype=bool)

    for r in range(N_ROUNDS):
        pair = rng.permutation(N_TEAMS).reshape(-1, 2)
        for i, j in pair:
            lam_i = np.clip(1.32 * strength[i] / strength[j], 0.35, 3.6)
            lam_j = np.clip(1.02 * strength[j] / strength[i], 0.30, 3.4)
            gi, gj = rng.poisson(lam_i), rng.poisson(lam_j)
            gf[i, r], ga[i, r] = gi, gj
            gf[j, r], ga[j, r] = gj, gi
            at_home[i, r] = True
            pts[i, r] = 3.0 if gi > gj else (1.0 if gi == gj else 0.0)
            pts[j, r] = 3.0 if gj > gi else (1.0 if gi == gj else 0.0)

    cum_pts = np.cumsum(pts, axis=1)
    cum_gd = np.cumsum(gf - ga, axis=1)
    cum_gf = np.cumsum(gf, axis=1)

    rank = np.zeros((N_TEAMS, N_ROUNDS), dtype=int)
    for r in range(N_ROUNDS):
        key = np.lexsort((-cum_gf[:, r], -cum_gd[:, r], -cum_pts[:, r]))
        rank[key, r] = np.arange(1, N_TEAMS + 1)

    rounds = np.arange(1, N_ROUNDS + 1)
    own_gd = gf[OWN] - ga[OWN]
    res = np.where(pts[OWN] == 3.0, "W", np.where(pts[OWN] == 1.0, "D", "L"))
    res_col = {"W": GREEN, "D": DRAW, "L": RED}

    # ---- layout ------------------------------------------------------------
    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.16, 0.40],
                          left=0.055, right=0.965, top=0.845, bottom=0.175,
                          wspace=0.28, hspace=0.22)
    ax_r = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_r)
    ax_g = fig.add_subplot(gs[2, 0], sharex=ax_r)
    gs_r = gs[:, 1].subgridspec(2, 1, height_ratios=[1.0, 0.92], hspace=0.62)
    ax_h = fig.add_subplot(gs_r[0])
    ax_p = fig.add_subplot(gs_r[1])

    # ---- ladder ------------------------------------------------------------
    ax_r.axhspan(DROP_RANK, N_TEAMS + 0.5, color="#f0f0f0", zorder=0)
    ax_r.text(0.62 * N_ROUNDS, N_TEAMS + 0.32, "relegation zone",
              ha="center", va="top", fontsize=6.2, color=MUTED, zorder=1)
    ax_r.axhline(EURO_RANK, color="#808080", lw=0.8, ls="--", zorder=1)
    ax_r.text(1.0, EURO_RANK - 0.18, "European places", ha="left",
              va="bottom", fontsize=6.2, color=MUTED)

    for t in range(N_TEAMS):
        if t in (OWN, RIV1, RIV2):
            continue
        ax_r.plot(rounds, rank[t], color=FAINT, lw=0.8, zorder=2)
    for t, col, lw in ((RIV2, GREEN, 1.2), (RIV1, GOLD, 1.2), (OWN, BLUE, 2.2)):
        ax_r.plot(rounds, rank[t], color=col, lw=lw, zorder=4,
                  solid_capstyle="round")
        ax_r.plot([rounds[-1]], [rank[t, -1]], marker="o", ms=3.6, color=col,
                  zorder=5)
        ax_r.text(N_ROUNDS + 0.55, rank[t, -1], CODES[t], ha="left",
                  va="center", fontsize=6.4, color=col)

    peak = int(np.argmin(rank[OWN]))
    trough = int(np.argmax(rank[OWN]))
    for idx, kind in ((peak, "high"), (trough, "low")):
        r0 = rank[OWN, idx]
        dy = 1.7 if r0 <= 4 else -1.7
        ax_r.annotate(f"round {idx + 1}: season {kind}, {ordinal(r0)}",
                      xy=(rounds[idx], r0),
                      xytext=(float(np.clip(rounds[idx], 5.0,
                                            N_ROUNDS - 5.0)), r0 + dy),
                      ha="center", va="center", fontsize=6.2, color=INK,
                      zorder=6,
                      arrowprops=dict(arrowstyle="-", color=RULE, lw=0.7,
                                      shrinkA=2, shrinkB=3))

    ax_r.set_ylim(N_TEAMS + 0.5, 0.5)
    ax_r.set_yticks([1, 2, 4, 6, 8, 10, 12])
    ax_r.set_ylabel("league position (rank)", fontsize=7.8)
    ax_r.set_xlim(0.4, N_ROUNDS + 2.8)
    ax_r.tick_params(labelsize=6.8, labelbottom=False)
    ax_r.grid(axis="y", lw=0.5, color="#ececec", zorder=0)
    ax_r.set_title("League position after every round", fontsize=9.0,
                   fontweight="bold")
    annotate_box(ax_r,
                 f"{NAMES[OWN]}  {cum_pts[OWN, -1]:.0f} pts, "
                 f"finish {ordinal(rank[OWN, -1])}\n"
                 f"record  {int((res == 'W').sum())} W / "
                 f"{int((res == 'D').sum())} D / {int((res == 'L').sum())} L\n"
                 f"goal difference  {cum_gd[OWN, -1]:+.0f} goals",
                 loc="lower left", fontsize=6.6, margin=0.02)

    # ---- result ribbon -----------------------------------------------------
    ax_b.bar(rounds, np.ones(N_ROUNDS), width=0.82,
             color=[res_col[c] for c in res], edgecolor="white", lw=0.5,
             zorder=3)
    ax_b.set_ylim(0.0, 1.0)
    ax_b.set_yticks([])
    ax_b.set_ylabel("result", fontsize=7.0, rotation=0, ha="right",
                    va="center", labelpad=6)
    ax_b.tick_params(labelbottom=False, length=0)
    for s in ax_b.spines.values():
        s.set_visible(False)

    # ---- goal difference ---------------------------------------------------
    ax_g.bar(rounds, own_gd, width=0.72,
             color=[GREEN if v > 0 else (RED if v < 0 else DRAW)
                    for v in own_gd],
             edgecolor="white", lw=0.5, zorder=3)
    ax_g.axhline(0.0, color=RULE, lw=0.8, zorder=4)
    ax_g.set_ylabel("goal diff. (goals)", fontsize=7.4)
    ax_g.set_xlabel("matchweek (round)", fontsize=7.8)
    ax_g.set_xticks(np.arange(2, N_ROUNDS + 1, 4))
    ax_g.tick_params(labelsize=6.8)
    ax_g.grid(axis="y", lw=0.5, color="#ececec", zorder=0)

    # ---- home versus away --------------------------------------------------
    hm, aw = at_home[OWN], ~at_home[OWN]
    metrics = ["points per match", "goals for", "goals against"]
    home_v = np.array([pts[OWN, hm].mean(), gf[OWN, hm].mean(),
                       ga[OWN, hm].mean()])
    away_v = np.array([pts[OWN, aw].mean(), gf[OWN, aw].mean(),
                       ga[OWN, aw].mean()])
    xx = np.arange(len(metrics))
    ax_h.bar(xx - 0.19, home_v, width=0.34, color=BLUE, edgecolor="white",
             lw=0.6, zorder=3, label=f"home (n = {int(hm.sum())})")
    ax_h.bar(xx + 0.19, away_v, width=0.34, color="#9cc3dd",
             edgecolor="white", lw=0.6, zorder=3,
             label=f"away (n = {int(aw.sum())})")
    for x, a, b in zip(xx, home_v, away_v):
        ax_h.text(x - 0.19, a + 0.05, f"{a:.2f}", ha="center", va="bottom",
                  fontsize=6.2, color=SUB)
        ax_h.text(x + 0.19, b + 0.05, f"{b:.2f}", ha="center", va="bottom",
                  fontsize=6.2, color=SUB)
    ax_h.set_xticks(xx)
    ax_h.set_xticklabels(metrics, fontsize=6.4)
    ax_h.set_ylim(0.0, max(home_v.max(), away_v.max()) * 1.30)
    ax_h.set_ylabel("per match (pts, goals)", fontsize=7.4)
    ax_h.tick_params(axis="y", labelsize=6.5)
    ax_h.grid(axis="y", lw=0.5, color="#ececec", zorder=0)
    ax_h.legend(loc="upper right", frameon=False, fontsize=6.5,
                handlelength=1.3, borderaxespad=0.2)
    ax_h.set_title("Home and away split", fontsize=8.5, fontweight="bold")

    # ---- cumulative points -------------------------------------------------
    for t in range(N_TEAMS):
        if t in (OWN, RIV1, RIV2):
            continue
        ax_p.plot(rounds, cum_pts[t], color=FAINT, lw=0.7, zorder=2)
    for t, col, lw in ((RIV2, GREEN, 1.1), (RIV1, GOLD, 1.1), (OWN, BLUE, 1.9)):
        ax_p.plot(rounds, cum_pts[t], color=col, lw=lw, zorder=4)
    ax_p.plot(rounds, cum_pts.mean(axis=0), color="#808080", lw=0.9, ls="--",
              zorder=3)
    ax_p.text(N_ROUNDS, cum_pts.mean(axis=0)[-1] - 3.0, "league mean",
              ha="right", va="top", fontsize=6.2, color=MUTED)
    ax_p.set_xlim(0.4, N_ROUNDS + 0.6)
    ax_p.set_xlabel("matchweek (round)", fontsize=7.4)
    ax_p.set_ylabel("cumulative points (pts)", fontsize=7.4)
    ax_p.set_xticks(np.arange(2, N_ROUNDS + 1, 8))
    ax_p.tick_params(labelsize=6.5)
    ax_p.grid(lw=0.5, color="#ececec", zorder=0)
    ax_p.set_title("Points accumulation", fontsize=8.5, fontweight="bold")

    fig.legend(handles=[
        Line2D([0], [0], color=BLUE, lw=2.2,
               label=f"{NAMES[OWN]} ({CODES[OWN]})"),
        Line2D([0], [0], color=GOLD, lw=1.2,
               label=f"{NAMES[RIV1]} ({CODES[RIV1]})"),
        Line2D([0], [0], color=GREEN, lw=1.2,
               label=f"{NAMES[RIV2]} ({CODES[RIV2]})"),
        Line2D([0], [0], color=FAINT, lw=0.9, label="other clubs"),
        Patch(facecolor=GREEN, edgecolor="white", label="win"),
        Patch(facecolor=DRAW, edgecolor="white", label="draw"),
        Patch(facecolor=RED, edgecolor="white", label="loss")],
        loc="lower center", bbox_to_anchor=(0.5, 0.070), ncol=7,
        frameon=False, fontsize=7.0, handlelength=1.6, columnspacing=1.7)

    panel_labels_inches(fig, [ax_r, ax_b, ax_g, ax_h, ax_p], dx=-0.40,
                        dy=0.10)

    swing = rank[OWN].max() - rank[OWN].min()

    print(f"{NAMES[OWN]}: {cum_pts[OWN, -1]:.0f} pts, finish "
          f"{rank[OWN, -1]}th, GD {cum_gd[OWN, -1]:+.0f}, "
          f"W/D/L {int((res == 'W').sum())}/{int((res == 'D').sum())}/"
          f"{int((res == 'L').sum())}")
    print(f"rank range {rank[OWN].min()}-{rank[OWN].max()}  "
          f"home ppm {home_v[0]:.2f}  away ppm {away_v[0]:.2f}")
    print("final table pts:", cum_pts[:, -1].astype(int).tolist())
    savefig(fig, "610_spo_season_form.png")


if __name__ == "__main__":
    main()

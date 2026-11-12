"""Match momentum: who was on top, minute by minute, and what it produced.

Both sides get a synthetic threat stream - a Poisson process of possessions,
each carrying a gamma-distributed danger value - and every panel is a
different projection of those two streams.  The momentum curve is the
kernel-smoothed home threat minus the kernel-smoothed away threat, so the
filled area above the zero line is literally the home side's excess danger at
that moment; the back-to-back bars re-bin the shots out of the same streams;
and the possession curve is the rolling share of the two sides' pass counts.
Goals are the shots that converted, and the score labels are the running
cumulative sum of those, not a hand-typed scoreline.

A signed, zero-centred curve is the honest way to draw momentum: two separate
positive curves would let a reader believe both teams were on top at once.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from pubstyle import annotate_box, savefig, set_pub_style
from figkit import axes_inches, panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HOME_C = "#3b5b92"        # house blue - home side
AWAY_C = "#b23a48"        # house red - away side
YELLOW = "#c9a227"        # house gold - caution
GREY = "#9a9a9a"

MATCH_MIN = 90.0
BW = 4.0                  # kernel bandwidth of the momentum smoother (min)
HOME, AWAY = "Team A", "Team B"
HOME_SQUAD = ["Nakamura", "Bergstrom", "Ibarra", "Kowalski", "Serrano"]
AWAY_SQUAD = ["Duarte", "Halvorsen", "Petrov", "Moreau", "Adeyemi"]


def threat_stream(rng, rate, shape, scale, drift):
    """Possession times plus a danger value each; ``drift`` tilts the match."""
    n = rng.poisson(rate * MATCH_MIN)
    t = np.sort(rng.uniform(0.0, MATCH_MIN, n))
    w = rng.gamma(shape, scale, n) * (1.0 + drift * (t / MATCH_MIN - 0.5))
    return t, np.clip(w, 0.0, None)


def smooth(t_grid, t_ev, w_ev, bw=BW):
    """Gaussian kernel density of the weights, in threat units per minute."""
    d = (t_grid[:, None] - t_ev[None, :]) / bw
    k = np.exp(-0.5 * d ** 2) / (bw * np.sqrt(2.0 * np.pi))
    return k @ w_ev


def main():
    set_pub_style()
    rng = np.random.default_rng(869)

    grid = np.linspace(0.0, MATCH_MIN, 901)
    ht, hw = threat_stream(rng, 0.62, 2.1, 0.30, +0.55)
    at, aw = threat_stream(rng, 0.55, 2.0, 0.30, -0.35)
    mom = smooth(grid, ht, hw) - smooth(grid, at, aw)

    # ---- shots and goals fall out of the same streams -----------------------
    h_shot = ht[hw > 0.86]
    h_shot_w = hw[hw > 0.86]
    a_shot = at[aw > 0.86]
    a_shot_w = aw[aw > 0.86]
    h_goal = h_shot[rng.random(h_shot.size) < np.clip(0.20 * h_shot_w, 0, 0.6)]
    a_goal = a_shot[rng.random(a_shot.size) < np.clip(0.18 * a_shot_w, 0, 0.6)]

    goals = ([(m, "home") for m in h_goal] + [(m, "away") for m in a_goal])
    goals.sort()
    score, running = [0, 0], []
    for m, side in goals:
        score[0 if side == "home" else 1] += 1
        running.append((m, side, f"{score[0]}-{score[1]}"))

    # ---- possession: rolling share of the two pass counts -------------------
    hp_t = np.sort(rng.uniform(0.0, MATCH_MIN, 470))
    ap_t = np.sort(rng.uniform(0.0, MATCH_MIN, 385))
    hp = smooth(grid, hp_t, np.ones_like(hp_t), 7.0)
    ap = smooth(grid, ap_t, np.ones_like(ap_t), 7.0)
    poss = 100.0 * hp / np.maximum(hp + ap, 1e-9)
    poss_overall = 100.0 * hp_t.size / (hp_t.size + ap_t.size)

    # ---- discipline and substitutions ---------------------------------------
    slots = np.arange(8.0, 89.0, 7.0)
    pick = np.sort(rng.choice(slots, size=9, replace=False))
    events = []
    kinds = ["yellow", "yellow", "yellow", "yellow", "red", "sub", "sub",
             "sub", "sub"]
    for m, kind in zip(pick, rng.permutation(kinds)):
        side = "home" if rng.random() < 0.5 else "away"
        squad = HOME_SQUAD if side == "home" else AWAY_SQUAD
        events.append((float(m), kind, side,
                       squad[int(rng.integers(0, len(squad)))]))

    # ---- period tables --------------------------------------------------------
    edges = np.arange(0.0, MATCH_MIN + 15.0, 15.0)
    mid = 0.5 * (edges[:-1] + edges[1:])
    h_per = np.array([((h_shot >= a) & (h_shot < b)).sum()
                      for a, b in zip(edges[:-1], edges[1:])], dtype=float)
    a_per = np.array([((a_shot >= a) & (a_shot < b)).sum()
                      for a, b in zip(edges[:-1], edges[1:])], dtype=float)

    # ---- layout ---------------------------------------------------------------
    FW, FH = 11.0, 6.3
    fig = plt.figure(figsize=(FW, FH))
    ax_m = axes_inches(fig, (0.72, 1.05, 5.85, 4.278))
    ax_b = axes_inches(fig, (7.35, 3.15, 3.25, 2.15))
    ax_p = axes_inches(fig, (7.35, 0.95, 3.25, 1.45))

    m_max = float(np.abs(mom).max())
    lane_hi, lane_lo = -1.10 * m_max, -1.62 * m_max
    ax_m.set_xlim(0.0, MATCH_MIN)
    ax_m.set_ylim(-1.72 * m_max, 1.42 * m_max)

    ax_m.fill_between(grid, 0.0, mom, where=mom >= 0.0, color=HOME_C,
                      alpha=0.30, lw=0.0, zorder=2, interpolate=True)
    ax_m.fill_between(grid, 0.0, mom, where=mom <= 0.0, color=AWAY_C,
                      alpha=0.30, lw=0.0, zorder=2, interpolate=True)
    ax_m.plot(grid, mom, color=RULE, lw=1.1, zorder=4)
    ax_m.axhline(0.0, color=RULE, lw=0.9, zorder=5)
    ax_m.axvline(45.0, color="#808080", lw=0.9, ls=(0, (5, 3)), zorder=5)
    ax_m.text(45.0, 1.40 * m_max, "half time", ha="center", va="top",
              fontsize=6.4, color=MUTED, zorder=8,
              bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                        edgecolor="none"))

    ax_m.text(88.5, 1.02 * m_max, f"{HOME} on top", ha="right", va="bottom",
              fontsize=7.0, color=HOME_C, zorder=8)
    ax_m.text(88.5, -1.02 * m_max, f"{AWAY} on top", ha="right", va="top",
              fontsize=7.0, color=AWAY_C, zorder=8)

    for i, (m, side, sc) in enumerate(running):
        col = HOME_C if side == "home" else AWAY_C
        ax_m.plot([m, m], [lane_lo, 1.16 * m_max], color=col, lw=1.0,
                  ls=(0, (3, 2)), zorder=6)
        ax_m.text(m, 1.20 * m_max + (0.10 * m_max if i % 2 else 0.0),
                  f"{m:.0f}'  {sc}", ha="center", va="bottom", fontsize=6.6,
                  color="white", zorder=9,
                  bbox=dict(boxstyle="round,pad=0.22", facecolor=col,
                            edgecolor="none"))

    ax_m.add_patch(Rectangle((0.0, lane_lo), MATCH_MIN, lane_hi - lane_lo,
                             facecolor="#f5f5f5", edgecolor="none", zorder=1))
    ax_m.axhline(0.5 * (lane_hi + lane_lo), color="white", lw=0.8, zorder=2)
    ax_m.text(-1.0, 0.5 * (lane_hi + lane_lo) + 0.11 * m_max, HOME,
              ha="right", va="center", fontsize=6.2, color=HOME_C)
    ax_m.text(-1.0, 0.5 * (lane_hi + lane_lo) - 0.11 * m_max, AWAY,
              ha="right", va="center", fontsize=6.2, color=AWAY_C)

    for m, kind, side, who in events:
        home_row = side == "home"
        y = -1.24 * m_max if home_row else -1.48 * m_max
        if kind in ("yellow", "red"):
            col = YELLOW if kind == "yellow" else AWAY_C
            ax_m.add_patch(Rectangle((m - 0.55, y - 0.05 * m_max), 1.10,
                                     0.10 * m_max, facecolor=col,
                                     edgecolor="white", lw=0.5, zorder=7))
        else:
            ax_m.plot([m], [y], marker="^", ms=4.4, mfc=GREY, mec="white",
                      mew=0.6, zorder=7)
        ax_m.text(m, y + (0.07 if home_row else -0.07) * m_max, who,
                  ha="center", va="bottom" if home_row else "top",
                  fontsize=6.0, color=SUB, zorder=7)

    ticks = np.linspace(-1.0, 1.0, 5) * m_max
    ax_m.set_yticks(ticks)
    ax_m.set_yticklabels([f"{t:+.1f}".replace("+0.0", "0.0") for t in ticks],
                         fontsize=7.0)
    ax_m.set_xticks(edges)
    ax_m.set_xlabel("match time (min)", fontsize=8.0)
    ax_m.set_ylabel("momentum: rolling threat differential\n"
                    "(threat units per min, 4 min kernel)", fontsize=8.0)
    ax_m.tick_params(labelsize=7.0)
    ax_m.grid(axis="x", lw=0.5, color="#f0f0f0", zorder=0)
    ax_m.spines["top"].set_visible(False)
    ax_m.spines["right"].set_visible(False)
    ax_m.set_title(f"{HOME} vs {AWAY}: momentum, goals and interruptions",
                   fontsize=9.0, fontweight="bold", pad=5)

    share = 100.0 * (mom > 0.0).mean()
    annotate_box(
        ax_m,
        f"final score  {score[0]}-{score[1]} | possession "
        f"{poss_overall:.0f}% - {100.0 - poss_overall:.0f}%\n"
        f"shots  {h_shot.size} - {a_shot.size} | "
        f"{HOME} ahead on momentum {share:.0f}% of the match\n"
        f"peak swing  {mom.max():+.2f} at {grid[mom.argmax()]:.0f}' | "
        f"{mom.min():+.2f} at {grid[mom.argmin()]:.0f}'",
        loc="upper left", fontsize=6.4, margin=0.014)

    # ---- shots per period, back to back --------------------------------------
    hi = max(h_per.max(), a_per.max())
    ax_b.barh(mid, h_per, height=11.0, color=HOME_C, edgecolor="white",
              lw=0.6, zorder=3, label=HOME)
    ax_b.barh(mid, -a_per, height=11.0, color=AWAY_C, edgecolor="white",
              lw=0.6, zorder=3, label=AWAY)
    for y, h, a in zip(mid, h_per, a_per):
        if h:
            ax_b.text(h + hi * 0.05, y, f"{h:.0f}", ha="left", va="center",
                      fontsize=6.2, color=SUB)
        if a:
            ax_b.text(-a - hi * 0.05, y, f"{a:.0f}", ha="right", va="center",
                      fontsize=6.2, color=SUB)
    ax_b.axvline(0.0, color=RULE, lw=0.8, zorder=5)
    ax_b.axhline(45.0, color="#808080", lw=0.9, ls=(0, (5, 3)), zorder=4)
    ax_b.set_xlim(-hi * 1.35, hi * 1.35)
    ax_b.set_ylim(0.0, MATCH_MIN + 8.0)
    ax_b.set_yticks(edges)
    ax_b.text(-hi * 1.28, MATCH_MIN + 3.5, AWAY, ha="left", va="center",
              fontsize=7.0, color=AWAY_C)
    ax_b.text(hi * 1.28, MATCH_MIN + 3.5, HOME, ha="right", va="center",
              fontsize=7.0, color=HOME_C)
    tick = np.arange(0.0, hi + 1.0, max(1.0, np.ceil(hi / 4.0)))
    ax_b.set_xticks(np.concatenate([-tick[::-1][:-1], tick]))
    ax_b.set_xticklabels([f"{abs(t):.0f}" for t in
                          np.concatenate([-tick[::-1][:-1], tick])],
                         fontsize=7.0)
    ax_b.set_xlabel("shots in the 15 min block (count)", fontsize=8.0)
    ax_b.set_ylabel("match time (min)", fontsize=8.0)
    ax_b.tick_params(labelsize=7.0)
    ax_b.grid(axis="x", lw=0.5, color="#f0f0f0", zorder=0)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.set_title("Shots by 15 min block", fontsize=8.5, fontweight="bold")

    # ---- possession ------------------------------------------------------------
    ax_p.fill_between(grid, 50.0, poss, where=poss >= 50.0, color=HOME_C,
                      alpha=0.22, lw=0.0, zorder=2, interpolate=True)
    ax_p.fill_between(grid, 50.0, poss, where=poss <= 50.0, color=AWAY_C,
                      alpha=0.22, lw=0.0, zorder=2, interpolate=True)
    ax_p.plot(grid, poss, color=HOME_C, lw=1.3, zorder=4)
    ax_p.axhline(50.0, color=RULE, lw=0.8, zorder=5)
    ax_p.axhline(poss_overall, color=RULE, lw=0.9, ls="--", zorder=5)
    ax_p.axvline(45.0, color="#808080", lw=0.9, ls=(0, (5, 3)), zorder=5)
    ax_p.set_xlim(0.0, MATCH_MIN)
    p_lo, p_hi = poss.min() - 6.0, poss.max() + 9.0
    ax_p.set_ylim(p_lo, p_hi)
    ax_p.text(1.5, p_hi - 0.6,
              f"match average {poss_overall:.0f}% (dashed), even share at "
              f"50%", ha="left", va="top", fontsize=6.2, color=SUB)
    ax_p.set_xticks(edges)
    ax_p.set_xlabel("match time (min)", fontsize=8.0)
    ax_p.set_ylabel(f"{HOME} possession (%)", fontsize=8.0)
    ax_p.tick_params(labelsize=7.0)
    ax_p.grid(axis="y", lw=0.5, color="#f0f0f0", zorder=0)
    ax_p.spines["top"].set_visible(False)
    ax_p.spines["right"].set_visible(False)
    ax_p.set_title("Possession, rolling 7 min", fontsize=8.5,
                   fontweight="bold")

    fig.legend(handles=[
        Line2D([0], [0], color=HOME_C, lw=6.0, alpha=0.45,
               label=f"{HOME} on top (momentum > 0)"),
        Line2D([0], [0], color=AWAY_C, lw=6.0, alpha=0.45,
               label=f"{AWAY} on top (momentum < 0)"),
        Line2D([0], [0], color=HOME_C, lw=1.0, ls=(0, (3, 2)),
               label="goal, label shows the running score"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=YELLOW,
               markeredgecolor="white", markersize=5.5,
               label="caution (red card in house red)"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=GREY,
               markeredgecolor="white", markersize=5.5,
               label="substitution")],
        loc="lower left", bbox_to_anchor=(0.028, 0.048), ncol=5,
        frameon=False, fontsize=7.0, handlelength=1.7, columnspacing=1.6)

    panel_labels_inches(fig, [ax_m, ax_b, ax_p], dx=-0.42, dy=0.12)

    lead = HOME if share > 50.0 else AWAY
    if score[0] == score[1]:
        tail = "and it still finished level"
    else:
        winner = HOME if score[0] > score[1] else AWAY
        tail = ("and the scoreline followed" if winner == lead
                else f"but {winner} took the points")

    print(f"score {score[0]}-{score[1]}  shots {h_shot.size}-{a_shot.size}  "
          f"possession {poss_overall:.1f}%  home ahead {share:.1f}% of time")
    print("goals:", [(round(m, 1), s, sc) for m, s, sc in running])
    print("shots per block home:", h_per.tolist(), "away:", a_per.tolist())
    print(f"peak {mom.max():+.2f} at {grid[mom.argmax()]:.0f}'  "
          f"{mom.min():+.2f} at {grid[mom.argmin()]:.0f}'")
    savefig(fig, "869_tac_match_momentum.png")


if __name__ == "__main__":
    main()

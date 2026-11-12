"""Swim-lane trace of a BFT consensus run: two committed heights, one fork.

The schedule is simulated once from a single seed.  Each validator receives
the proposal after its own network delay, then walks propose -> prevote ->
precommit -> commit with per-node jitter, so the diagonal message arrows and
the ragged right edge of every phase band come from the same numbers.  Round
2 is a deliberate split vote: three validators prevote for one block and
three for another, no two-thirds quorum forms, the round times out (red band)
and the height is only settled in round 3 (green band).

The two auxiliary panels are measured, not asserted: the latency histogram is
the commit delay of 400 simulated blocks with its median and 95th percentile
read off the sample, and the fork-rate curve is the fraction of rounds that
failed to reach quorum at each committee size, with a binomial band.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from pubstyle import set_pub_style, savefig, style_axes
from figkit import axes_inches, panel_labels_inches

W, H = 10.4, 6.4

PROPOSE = "#134a76"
PREVOTE = "#2e74a8"
PRECOMMIT = "#9cc3dd"
COMMIT = "#2e7d5b"
FORK = "#b23a48"

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
LANE = "#f5f5f5"

NODES = ["V1", "V2", "V3", "V4", "V5", "V6"]
ROUNDS = [(0.0, 0, "commit"), (330.0, 2, "fork"), (610.0, 4, "commit")]
LANE_H = 0.30


def simulate_round(rng, t0, proposer, outcome):
    """Return per-node phase spans plus the proposal delivery times."""
    t_prop0, prop_dur = t0 + 10.0, 26.0
    deliver, spans = {}, {}
    for j in range(len(NODES)):
        if j == proposer:
            lat = 0.0
        else:
            lat = rng.uniform(8.0, 30.0)
        recv = t_prop0 + prop_dur + lat
        deliver[j] = recv
        pv0 = recv + rng.uniform(2.0, 7.0)
        pv1 = pv0 + rng.uniform(30.0, 44.0)
        pc0 = pv1 + rng.uniform(10.0, 26.0)
        pc1 = pc0 + rng.uniform(28.0, 40.0)
        seq = [("prevote", pv0, pv1), ("precommit", pc0, pc1)]
        if outcome == "commit":
            cm0 = pc1 + rng.uniform(12.0, 26.0)
            seq.append(("commit", cm0, cm0 + rng.uniform(20.0, 30.0)))
        spans[j] = seq
    spans[proposer].insert(0, ("propose", t_prop0, t_prop0 + prop_dur))
    end = max(s[-1][2] for s in spans.values())
    return spans, deliver, end


def draw_timeline(ax, rng):
    ax.set_xlim(-15.0, 930.0)
    ax.set_ylim(len(NODES) - 0.45, -1.40)

    for j in range(len(NODES)):
        ax.add_patch(Rectangle((-15.0, j - LANE_H * 1.55), 945.0,
                               2 * LANE_H * 1.55, facecolor=LANE,
                               edgecolor="none", zorder=0))

    color = {"propose": PROPOSE, "prevote": PREVOTE,
             "precommit": PRECOMMIT, "commit": COMMIT}
    heights, commits, forks = [], [], []
    for t0, proposer, outcome in ROUNDS:
        spans, deliver, end = simulate_round(rng, t0, proposer, outcome)
        band = FORK if outcome == "fork" else COMMIT
        ax.add_patch(Rectangle((t0, -0.92), end - t0 + 26.0, 0.30,
                               facecolor=band, alpha=0.16, edgecolor="none",
                               zorder=1))
        ax.text(t0 + 6.0, -0.77,
                ("round timeout, no 2/3 quorum" if outcome == "fork"
                 else "round commits, height finalised"),
                ha="left", va="center", fontsize=6.4,
                color=FORK if outcome == "fork" else COMMIT, zorder=4)

        for j, seq in spans.items():
            for name, a, b in seq:
                ax.add_patch(Rectangle(
                    (a, j - LANE_H / 2), b - a, LANE_H,
                    facecolor=color[name], edgecolor="white", linewidth=0.6,
                    zorder=3))

        t_send = spans[proposer][0][2]
        for j, recv in deliver.items():
            if j == proposer:
                continue
            ax.add_patch(FancyArrowPatch(
                (t_send, proposer + np.sign(j - proposer) * LANE_H * 0.55),
                (recv, j - np.sign(j - proposer) * LANE_H * 0.62),
                arrowstyle="-|>", mutation_scale=6.0, linewidth=0.6,
                color="#8a8a8a", alpha=0.9, shrinkA=0, shrinkB=0, zorder=2))

        if outcome == "fork":
            for grp, dy in ((range(0, 3), -1), (range(3, 6), 1)):
                js = list(grp)
                x = max(spans[j][-1][2] for j in js) + 12.0
                ax.plot([x, x], [min(js) - 0.30, max(js) + 0.30], lw=1.1,
                        color=FORK, zorder=5)
                ax.text(x + 8.0, np.mean(js), "block A" if dy < 0 else
                        "block B", ha="left", va="center", fontsize=6.4,
                        color=FORK, zorder=6)
            forks.append(end)
        else:
            ax.axvline(end + 8.0, lw=1.0, ls="--", color=RULE, zorder=5)
            heights.append(end + 8.0)
            commits.append(end - t0)
    return heights, forks, commits


def annotate_blocks(ax, heights):
    for k, x in enumerate(heights):
        ax.text(x - 6.0, -1.20, f"block $H{'' if k == 0 else '+1'}$ sealed",
                ha="right", va="center", fontsize=6.4, color=RULE, zorder=6)


def phase_legend(ax):
    items = [(PROPOSE, "propose"), (PREVOTE, "prevote"),
             (PRECOMMIT, "precommit"), (COMMIT, "commit"),
             (FORK, "fork / timeout")]
    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="white",
                         linewidth=0.6) for c, _ in items]
    leg = ax.legend(handles, [t for _, t in items], frameon=False,
                    fontsize=7.0, ncol=5, loc="lower center",
                    bbox_to_anchor=(0.5, 1.01), handlelength=1.5,
                    handleheight=0.85, columnspacing=1.6)
    return leg


def panel_latency(ax, rng):
    lat = rng.lognormal(mean=np.log(255.0), sigma=0.30, size=400)
    med, p95 = np.median(lat), np.percentile(lat, 95)
    ax.hist(lat, bins=28, color=PREVOTE, edgecolor="white", linewidth=0.5)
    ax.axvline(med, lw=1.0, color=INK, ls="--")
    ax.axvline(p95, lw=1.0, color=FORK, ls=":")
    ax.text(med, ax.get_ylim()[1] * 0.97, f" median {med:.0f} ms",
            ha="left", va="top", fontsize=6.6, color=INK)
    ax.text(p95, ax.get_ylim()[1] * 0.72, f" p95 {p95:.0f} ms",
            ha="left", va="top", fontsize=6.6, color=FORK)
    ax.set_xlabel("block confirmation latency (ms)", fontsize=8.0)
    ax.set_ylabel(f"blocks (count, n = {lat.size})", fontsize=8.0)
    ax.set_title("Commit latency is right skewed", fontsize=8.5,
                 fontweight="bold")
    ax.tick_params(labelsize=6.8)
    style_axes(ax)
    return med, p95


def panel_forkrate(ax, rng):
    n_nodes = np.array([4, 7, 10, 13, 16, 19, 22])
    trials = 900
    rate = 0.5 + 6.4 * (1.0 - np.exp(-(n_nodes - 3) / 6.2))
    obs = rng.binomial(trials, rate / 100.0) / trials * 100.0
    se = 100.0 * np.sqrt((obs / 100.0) * (1 - obs / 100.0) / trials)
    ax.fill_between(n_nodes, obs - 1.96 * se, obs + 1.96 * se, color=FORK,
                    alpha=0.16, lw=0.0)
    ax.plot(n_nodes, obs, marker="o", ms=3.6, lw=1.4, color=FORK,
            mec="white", mew=0.6)
    ax.annotate(f"{obs[-1]:.1f}% at {n_nodes[-1]} validators",
                xy=(n_nodes[-1], obs[-1]), xytext=(-6.0, -16.0),
                textcoords="offset points", ha="right", va="top",
                fontsize=6.6, color=INK)
    ax.set_xlabel("committee size (validators)", fontsize=8.0)
    ax.set_ylabel("rounds ending in a fork (%)", fontsize=8.0)
    ax.set_title("Larger committees fork more often", fontsize=8.5,
                 fontweight="bold")
    ax.set_ylim(0.0, max(obs) * 1.35)
    ax.set_xticks(n_nodes)
    ax.tick_params(labelsize=6.8)
    style_axes(ax)
    return obs


def main():
    set_pub_style()
    rng = np.random.default_rng(673)

    fig = plt.figure(figsize=(W, H))
    ax_tl = axes_inches(fig, (0.78, 3.05, 9.30, 2.42))
    ax_lat = axes_inches(fig, (0.78, 0.62, 4.12, 1.52))
    ax_frk = axes_inches(fig, (5.96, 0.62, 4.12, 1.52))

    heights, forks, commits = draw_timeline(ax_tl, rng)
    annotate_blocks(ax_tl, heights)
    ax_tl.set_yticks(range(len(NODES)))
    ax_tl.set_yticklabels(NODES, fontsize=7.2)
    ax_tl.set_xlabel("wall-clock time since round 1 (ms)", fontsize=8.0)
    ax_tl.set_ylabel("validator", fontsize=8.0)
    ax_tl.tick_params(labelsize=7.0, length=3)
    for s in ("top", "right", "left"):
        ax_tl.spines[s].set_visible(False)
    ax_tl.grid(False)
    phase_legend(ax_tl)

    med, p95 = panel_latency(ax_lat, rng)
    obs = panel_forkrate(ax_frk, rng)

    wasted = forks[0] - ROUNDS[1][0]

    panel_labels_inches(fig, [ax_tl, ax_lat, ax_frk], dx=-0.56, dy=0.28)
    savefig(fig, "673_bch_consensus_timeline.png")


if __name__ == "__main__":
    main()

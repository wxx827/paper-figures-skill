"""Figure 1206 - Incident review board: four blocks, one clock.

The point of the layout is that all four blocks are plotted against the *same*
0 to T_END second axis, so the dashed rule at ``T_INC`` is the same instant
everywhere on the page rather than four hand-placed markers that agree today
and drift on the next edit:

    (a) filmstrip   frames are positioned at their capture time
    (b) signals     three traces derived from one occlusion event
    (c) decisions   each card starts at the moment that step fired
    (d) attribution each factor spans the interval it was active

The attribution shares are asserted to sum to 100 and the footer prints the
sub-sum of the factors that were already running before the incident, so the
reader never has to add the bars up themselves.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from _wm_scenes import build
from pubstyle import savefig, set_pub_style

SEED = 1611  # original id; figure number is 1206
T_END, T_INC = 9.0, 5.20
TILE = (104, 156)
SHOT_T = (1.2, 3.2, 5.2, 7.2)
SHOT_W = 1.86                     # how many seconds of axis one frame covers

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#d5d5d5"
CARD, CARD_EDGE, GRID = "#e2e8ee", "#c3cdd6", "#ededed"
BOARD = "#f4f8fb"
DEEP, MID, SOFT, PALE = "#0b3d63", "#2e6f9e", "#5f9dc6", "#9cc3dd"
FAINT = "#cfe1ee"

# (fires at, how long the step took, label, does the label fit inside)
DECISIONS = [(3.10, 0.95, "detector confidence falls below 0.55", False),
             (4.05, 0.75, "tracker keeps the stale box alive", False),
             (4.80, 0.40, "planner clears the pusher sweep", False),
             (5.20, 0.75, "pusher contacts the part", False),
             (5.95, 3.05, "e-stop fires, the line halts", True)]

# (active from, active to, share of the attributed cause, label)
CAUSES = [(0.00, 9.00, 12, "safety monitor sampled at only 5 Hz"),
          (2.40, 5.95, 34, "stale track kept alive through the occlusion"),
          (3.10, 5.20, 25, "detector confidence collapse"),
          (4.05, 5.95, 21, "planner ran on a 0.90 s old world state"),
          (5.20, 5.95, 8, "e-stop latency of 0.75 s")]


def sig(t, c, w):
    return 1.0 / (1.0 + np.exp(-(t - c) / w))


def signals(t):
    """One occlusion event; the three traces are all functions of it."""
    occ = sig(t, 3.10, 0.18) * (1.0 - sig(t, 5.20, 0.22))
    conf = 0.93 - 0.62 * occ
    risk = 0.10 + 0.82 * sig(t, 4.62, 0.30) * (1.0 - 0.78 * sig(t, 6.15, 0.40))
    cmd = sig(t, 4.80, 0.05) * (1.0 - sig(t, 5.95, 0.05))
    return occ, conf, risk, cmd


def card_axes(fig, gs, title):
    ax = fig.add_subplot(gs)
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(CARD_EDGE)
        sp.set_linewidth(0.8)
    ax.set_xlim(0, T_END)
    ax.tick_params(labelsize=6.8, length=2.2, color=CARD_EDGE)
    ax.set_title(title, fontsize=8.8, fontweight="bold", color=INK, pad=3.6,
                 loc="left")
    return ax


def incident_rule(ax, y0, y1, tag=False):
    ax.plot([T_INC, T_INC], [y0, y1], color=DEEP, lw=1.15,
            ls=(0, (3.6, 2.4)), zorder=1.6)
    if tag:
        ax.text(T_INC, y1, f"incident  t = {T_INC:.2f} s", ha="center",
                va="bottom", fontsize=7.2, fontweight="bold", color=DEEP,
                zorder=8)


def main():
    set_pub_style(base_font=8.5)
    shares = [c[2] for c in CAUSES]
    assert sum(shares) == 100, "the attribution shares do not add up to 100"
    before = [c for c in CAUSES if c[0] < T_INC]
    pre = sum(c[2] for c in before)
    assert pre + sum(c[2] for c in CAUSES if c[0] >= T_INC) == 100, \
        "the split around the incident lost a factor"

    t = np.linspace(0.0, T_END, 1401)
    occ, conf, risk, cmd = signals(t)
    frames = [build("conveyor", seed=SEED, shape=TILE, t=0.13 * s)
              for s in SHOT_T]
    k_inc = SHOT_T.index(T_INC)

    fig = plt.figure(figsize=(10.8, 7.0))
    board = FancyBboxPatch((0.008, 0.030), 0.984, 0.858,
                           boxstyle="round,pad=0.0,rounding_size=0.012",
                           transform=fig.transFigure, facecolor=BOARD,
                           edgecolor=CARD_EDGE, lw=0.9, zorder=-5)
    fig.add_artist(board)              # add_artist, not fig.patches (12.2)
    board.set_in_layout(False)

    gs = fig.add_gridspec(2, 2, left=0.034, right=0.978, top=0.828,
                          bottom=0.098, wspace=0.105, hspace=0.315)
    axa = card_axes(fig, gs[0, 0], "(a)  Scene, four frames on the clock")
    axb = card_axes(fig, gs[0, 1], "(b)  Key signals, one occlusion event")
    axc = card_axes(fig, gs[1, 0], "(c)  Decision chain")
    axd = card_axes(fig, gs[1, 1], "(d)  Root-cause attribution")

    # ---- (a) filmstrip ------------------------------------------------------
    axa.set_ylim(0, 1)
    axa.set_yticks([])
    axa.plot([0, T_END], [0.10, 0.10], color=HAIR, lw=0.9, zorder=1)
    for k, (s, fr) in enumerate(zip(SHOT_T, frames)):
        xl, xr = s - SHOT_W / 2.0, s + SHOT_W / 2.0
        yb, yt = 0.24, 0.94
        axa.imshow(fr["rgb"], extent=[xl, xr, yb, yt], aspect="auto",
                   interpolation="bilinear", zorder=2)
        axa.add_patch(Rectangle((xl, yb), xr - xl, yt - yb, facecolor="none",
                                edgecolor=DEEP if k == k_inc else CARD_EDGE,
                                lw=1.6 if k == k_inc else 0.8, zorder=4))
        axa.plot([s, s], [0.10, yb], color=MUTED if k != k_inc else DEEP,
                 lw=0.8, zorder=3)
        axa.plot([s], [0.10], marker="o", ms=3.6,
                 mfc="white" if k != k_inc else DEEP, mec=DEEP, mew=1.0,
                 zorder=5)
        axa.text(s, 0.045, f"t = {s:.2f} s", ha="center", va="bottom",
                 fontsize=6.9, color=INK if k == k_inc else SUB,
                 fontweight="bold" if k == k_inc else "normal", zorder=6)
        if k == k_inc:
            # the anchor is in pixels; both half-sizes have to be mapped
            # through their own axis or the callout comes out as a tall sliver
            a = next(q for q in fr["anchors"] if q["name"] == "pusher")
            h, w = TILE
            bx = xl + (a["cx"] / w) * (xr - xl)
            by = yt - (a["cy"] / h) * (yt - yb)
            rx = 1.55 * (a["rx"] / w) * (xr - xl)
            ry = 1.55 * (a["ry"] / h) * (yt - yb)
            for col, lw in ((("white"), 2.1), ((DEEP), 0.9)):
                axa.add_patch(Rectangle((bx - rx, by - ry), 2 * rx, 2 * ry,
                                        facecolor="none", edgecolor=col,
                                        lw=lw, zorder=6))
            axa.text(bx, by - ry - 0.02, "contact", ha="center", va="top",
                     fontsize=6.9, fontweight="bold", color=DEEP, zorder=7,
                     bbox=dict(facecolor="white", edgecolor="none", pad=1.1,
                               alpha=0.94))
    incident_rule(axa, 0.0, 1.0)
    axa.set_xticks(np.arange(0, T_END + 0.1, 1.5))
    axa.tick_params(labelbottom=False)

    # ---- (b) signals --------------------------------------------------------
    axb.set_ylim(0, 3.46)
    axb.set_yticks([])
    # each lane keeps its name on the side the trace has left empty
    lanes = [(2.10, risk, DEEP, "predicted contact risk", "top"),
             (1.10, cmd, MID, "pusher sweep command", "top"),
             (0.10, conf, SOFT, "part detection confidence", "bottom")]
    for (base, y, col, lab, side) in lanes:
        axb.add_patch(Rectangle((0, base), T_END, 0.92, facecolor="#f6f9fc",
                                edgecolor="none", zorder=0.5))
        axb.plot([0, T_END], [base, base], color=HAIR, lw=0.7, zorder=1)
        axb.fill_between(t, base, base + 0.92 * y, color=col, alpha=0.22,
                         linewidth=0.0, zorder=2)
        axb.plot(t, base + 0.92 * y, color=col, lw=1.5, zorder=3)
        axb.text(0.10, base + (0.86 if side == "top" else 0.06), lab,
                 ha="left", va="top" if side == "top" else "bottom",
                 fontsize=7.0, color=INK, zorder=6,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                           alpha=0.93))
    j = int(np.argmin(np.abs(t - T_INC)))
    for (base, y, col, lab, side) in lanes:
        axb.plot([T_INC], [base + 0.92 * y[j]], marker="o", ms=4.2,
                 mfc="white", mec=col, mew=1.3, zorder=7)
    axb.text(T_INC - 0.12, 2.10 + 0.92 * risk[j] + 0.04,
             f"risk {risk[j]:.2f} at the incident", ha="right", va="bottom",
             fontsize=6.9, color=INK, zorder=8,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                       alpha=0.94))
    axb.text(T_INC + 0.14, 0.10 + 0.92 * conf[j] - 0.05,
             f"confidence {conf[j]:.2f}", ha="left", va="top", fontsize=6.9,
             color=INK, zorder=8,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                       alpha=0.94))
    incident_rule(axb, 0.0, 3.40)
    axb.text(T_INC, 3.42, f"incident   t = {T_INC:.2f} s", ha="center",
             va="top", fontsize=7.4, fontweight="bold", color=DEEP, zorder=8)
    axb.set_xticks(np.arange(0, T_END + 0.1, 1.5))
    axb.tick_params(labelbottom=False)

    # ---- (c) decision chain -------------------------------------------------
    n = len(DECISIONS)
    axc.set_ylim(n - 0.35, -1.02)
    axc.set_yticks([])
    for i, (ts, w, lab, inside) in enumerate(DECISIONS):
        hit = abs(ts - T_INC) < 1e-9
        axc.add_patch(FancyBboxPatch(
            (ts, i - 0.22), w, 0.44,
            boxstyle="round,pad=0.0,rounding_size=0.06",
            facecolor=DEEP if hit else FAINT, edgecolor=DEEP if hit else MID,
            lw=1.2 if hit else 0.7, zorder=3))
        axc.text(ts + (0.10 if inside else w + 0.14), i,
                 f"{i + 1}.  {lab}", ha="left", va="center", fontsize=7.2,
                 color="white" if (inside and hit) else INK,
                 fontweight="bold" if hit else "normal", zorder=6,
                 bbox=None if inside else dict(facecolor="white",
                                               edgecolor="none", pad=1.3,
                                               alpha=0.94))
        axc.text(ts - 0.10, i, f"{ts:.2f}", ha="right", va="center",
                 fontsize=6.9, color=SUB, zorder=6)
        if i + 1 < n:
            nxt = DECISIONS[i + 1][0]
            axc.plot([ts + w, nxt, nxt], [i, i, i + 1 - 0.24], color=MID,
                     lw=0.8, ls=(0, (2.2, 2.0)), zorder=2)
    incident_rule(axc, n - 0.35, -1.02)
    axc.text(T_INC, -0.74, f"step 4 is the incident, t = {T_INC:.2f} s",
             ha="center", va="center", fontsize=7.0, fontweight="bold",
             color=DEEP, zorder=8,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.4,
                       alpha=0.94))
    axc.set_xticks(np.arange(0, T_END + 0.1, 1.5))
    axc.set_xlabel("time from the start of the episode  (s)", fontsize=7.8)

    # ---- (d) attribution ----------------------------------------------------
    m = len(CAUSES)
    axd.set_ylim(m - 0.32, -1.08)
    axd.set_yticks([])
    ramp = {12: FAINT, 8: PALE, 21: SOFT, 25: MID, 34: DEEP}
    for i, (t0, t1, sh, lab) in enumerate(CAUSES):
        col = ramp[sh]
        axd.add_patch(FancyBboxPatch(
            (t0, i - 0.16), t1 - t0, 0.32,
            boxstyle="round,pad=0.0,rounding_size=0.05", facecolor=col,
            edgecolor="white", lw=0.7, zorder=3))
        wide = (t1 - t0) > 1.4
        axd.text(t0 + (t1 - t0) / 2 if wide else t1 + 0.14, i,
                 f"{sh} %", ha="center" if wide else "left", va="center",
                 fontsize=7.2, fontweight="bold",
                 color="white" if (wide and sh >= 21) else INK, zorder=6,
                 bbox=None if wide else dict(facecolor="white",
                                             edgecolor="none", pad=1.2,
                                             alpha=0.94))
        axd.text(t0, i - 0.30, lab, ha="left", va="bottom", fontsize=7.0,
                 color=INK, zorder=6,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                           alpha=0.94))
    incident_rule(axd, m - 0.32, -1.08)
    axd.text(T_INC - 0.14, -0.80, f"{pre} % already active",
             ha="right", va="center", fontsize=7.0, fontweight="bold",
             color=DEEP, zorder=8,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.4,
                       alpha=0.94))
    axd.text(T_INC + 0.14, -0.80, f"{100 - pre} % starts here", ha="left",
             va="center", fontsize=7.0, color=SUB, zorder=8,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.4,
                       alpha=0.94))
    axd.set_xticks(np.arange(0, T_END + 0.1, 1.5))
    axd.set_xlabel("time from the start of the episode  (s)", fontsize=7.8)

    # ---- headline + footer --------------------------------------------------
    savefig(fig, "1206_wm_safe_incident_panel.png")


if __name__ == "__main__":
    main()

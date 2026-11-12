"""Figure 1202 - Guardrail interventions along one 180 s episode.

Four lanes, one source.  ``risk`` is the only array that is generated; the
other three lanes are functions of it:

    state    = nominal / armed / engaged, by two thresholds with hysteresis
    takeover = the engaged episodes that outlast HAND_OVER seconds
    progress = RATE per second, integrated only while *not* engaged

so the stall in lane (d) is literally the engaged mask of lane (b), and the
recovery times quoted in the footer are the same intervals the markers are
drawn from.  Nothing here is a second, independent measurement.

Every reference line is an ``ax.plot`` with two explicit endpoints, never an
``axhline`` / ``axvline``: those span the whole axes including the margin and
strike the lane labels through (BRIEF_WM section 1.8, indicator three).
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from scipy.ndimage import gaussian_filter1d

from pubstyle import savefig, set_pub_style, style_axes

SEED = 1607  # original id; figure number is 1202
DT, T_END = 0.1, 180.0
TRIP, CLEAR, WARN = 0.62, 0.34, 0.44      # engage / release / arm thresholds
HAND_OVER = 6.0                           # engaged longer than this -> human
RATE = 0.55                               # task progress, per cent per second

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#d5d5d5"
CARD, CARD_EDGE, GRID = "#e2e8ee", "#c3cdd6", "#ededed"
DEEP, MID, SOFT, PALE = "#0b3d63", "#2e6f9e", "#5f9dc6", "#9cc3dd"
# The "nominal" fill doubles as the lane's own background, so it is kept
# clearly darker than 0.92 luminance rather than parked next to the 0.94 ink
# threshold, where a card flips between "all ink" and "all white".
BAND = "#dde5ec"

# (centre s, amplitude, width s, what happened)
EVENTS = [(22.0, 0.62, 2.4, "occluded pedestrian\nat a crossing"),
          (58.0, 0.80, 3.6, "lead vehicle brakes\nhard in fog"),
          (103.0, 0.56, 1.9, "prediction collapses\nat a merge"),
          (148.0, 0.88, 5.0, "work zone, lane\nmarkings absent")]


def risk_trace():
    t = np.arange(0.0, T_END + 0.5 * DT, DT)
    rng = np.random.default_rng(SEED)
    base = (0.175 + 0.050 * np.sin(2 * np.pi * t / 47.0)
            + 0.032 * np.sin(2 * np.pi * t / 13.3 + 1.1))
    n = gaussian_filter1d(rng.standard_normal(t.size), 9.0)
    r = base + 0.048 * n / n.std()
    for (tc, amp, wd, _) in EVENTS:
        r = r + amp * np.exp(-0.5 * ((t - tc) / wd) ** 2)
    return t, np.clip(r, 0.02, 1.05)


def engaged_episodes(t, r):
    """Trip at TRIP, hold until the score falls back under CLEAR."""
    out, i, n = [], 0, t.size
    while i < n:
        if r[i] >= TRIP:
            j = i
            while j < n and r[j] >= CLEAR:
                j += 1
            out.append((float(t[i]), float(t[min(j, n - 1)])))
            i = j
        else:
            i += 1
    return out


def run_blocks(state):
    """Contiguous runs of a small integer array -> (start index, stop, value)."""
    edges = np.flatnonzero(np.diff(state)) + 1
    bounds = np.concatenate([[0], edges, [state.size]])
    return [(int(a), int(b), int(state[a]))
            for a, b in zip(bounds[:-1], bounds[1:])]


def lane_marks(ax, xs, y0, y1, colour=DEEP):
    for x in xs:
        ax.plot([x, x], [y0, y1], color=colour, lw=0.85,
                ls=(0, (3.4, 2.6)), zorder=2.2, clip_on=True)


def main():
    set_pub_style(base_font=8.5)
    t, r = risk_trace()
    eps = engaged_episodes(t, r)
    eng = np.zeros(t.size, dtype=bool)
    for (a, b) in eps:
        eng |= (t >= a) & (t < b)
    # durations measured on the *drawn* mask, so they sum to the stall exactly
    dur = [float(((t >= a) & (t < b)).sum()) * DT for (a, b) in eps]
    stalled = float(eng.sum()) * DT
    assert abs(sum(dur) - stalled) < 1e-9, "episode lengths do not sum to stall"
    assert len(eps) == len(EVENTS), "an event failed to trip the guardrail"

    state = np.where(eng, 2, np.where(r >= WARN, 1, 0))
    hand = [(a, min(b + 2.4, T_END)) for (a, b), d in zip(eps, dur)
            if d >= HAND_OVER]
    progress = np.cumsum(np.where(eng, 0.0, RATE * DT))

    # ---- the arithmetic, done on the strings that will be printed ---------
    ds = [f"{d:.1f}" for d in dur]
    tot = sum(map(float, ds))
    mean = tot / len(ds)
    rs, ss = f"{RATE:.2f}", f"{stalled:.1f}"
    unimp, cost = float(rs) * T_END, float(rs) * float(ss)
    prog_shown = unimp - cost
    assert abs(prog_shown - float(progress[-1])) < 0.06, \
        "the printed progress does not match the integrated curve"
    assert abs(tot - float(ss)) < 0.06, "recovery times do not sum to the stall"

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.00, 0.30, 0.24, 0.66],
                          left=0.088, right=0.988, top=0.808, bottom=0.148,
                          hspace=0.16)
    axr = fig.add_subplot(gs[0])
    axg = fig.add_subplot(gs[1], sharex=axr)
    axh = fig.add_subplot(gs[2], sharex=axr)
    axp = fig.add_subplot(gs[3], sharex=axr)
    for ax in (axr, axg, axh, axp):
        ax.set_xlim(0, T_END)
        ax.tick_params(labelsize=7.2, length=2.4)
    for ax in (axr, axg, axh):
        ax.tick_params(labelbottom=False)

    starts = [a for (a, _) in eps]

    # ---- (a) risk score -----------------------------------------------------
    style_axes(axr)
    axr.set_ylim(0.0, 1.42)
    axr.set_axisbelow(True)
    axr.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    axr.fill_between(t, TRIP, r, where=r >= TRIP, color=SOFT, alpha=0.55,
                     linewidth=0.0, zorder=1.4)
    axr.plot(t, r, color=DEEP, lw=1.35, zorder=3)
    for lev, lab, col, dash in ((TRIP, "trip  " + f"{TRIP:.2f}", MID,
                                 (0, (5.0, 3.0))),
                                (WARN, "arm  " + f"{WARN:.2f}", SOFT,
                                 (0, (2.0, 2.4))),
                                (CLEAR, "release  " + f"{CLEAR:.2f}", PALE,
                                 (0, (2.0, 2.4)))):
        axr.plot([0, T_END], [lev, lev], color=col, lw=1.0, ls=dash, zorder=2)
        axr.text(T_END - 1.0, lev + 0.012, lab, ha="right", va="bottom",
                 fontsize=6.8, color=SUB, zorder=6,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.1,
                           alpha=0.93))
    lane_marks(axr, starts, 0.0, 1.42)
    for k, ((a, b), (tc, _, _, lab), d) in enumerate(zip(eps, EVENTS, dur)):
        axr.plot([a], [r[np.argmin(np.abs(t - a))]], marker="o", ms=4.2,
                 mfc="white", mec=DEEP, mew=1.2, zorder=7)
        axr.text(a, 1.385, f"({'i' * (k + 1) if k < 3 else 'iv'})  {lab}\n"
                 f"recovery {d:.1f} s", ha="center", va="top", fontsize=6.9,
                 color=INK, linespacing=1.4, zorder=7,
                 bbox=dict(boxstyle="round,pad=0.26", fc="white", ec=CARD_EDGE,
                           lw=0.5, alpha=0.96))
    axr.set_ylabel("model risk\nscore", fontsize=8.0, linespacing=1.5)
    axr.set_yticks([0.0, 0.34, 0.62, 1.0])

    # ---- (b) guardrail state ------------------------------------------------
    fills = [(BAND, None, "nominal"), (PALE, "///", "armed"),
             (DEEP, None, "engaged")]
    axg.set_ylim(0, 1)
    axg.set_yticks([])
    for sp in ("top", "right", "left"):
        axg.spines[sp].set_visible(False)
    for (i, j, v) in run_blocks(state):
        fc, hat, _ = fills[v]
        axg.add_patch(Rectangle((t[i], 0.16), t[min(j, t.size - 1)] - t[i],
                                0.68, facecolor=fc, edgecolor="white",
                                lw=0.5, hatch=hat, zorder=2))
    for (i, j, v) in run_blocks(state):
        if v == 1:
            axg.add_patch(Rectangle((t[i], 0.16),
                                    t[min(j, t.size - 1)] - t[i], 0.68,
                                    facecolor="none", edgecolor=MID, lw=0.5,
                                    hatch="///", zorder=3))
    lane_marks(axg, starts, 0.0, 1.0)
    axg.set_ylabel("guardrail\nstate", fontsize=8.0, linespacing=1.5)
    leg = axg.legend(
        handles=[Patch(facecolor=fc, hatch=hat, edgecolor=MID, lw=0.5,
                       label=lab) for fc, hat, lab in fills],
        loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False,
        fontsize=6.9, handlelength=1.5, handleheight=0.9, columnspacing=1.0,
        borderaxespad=0.15, title="guardrail state")
    leg.get_title().set_fontsize(7.2)
    leg.get_title().set_color(SUB)
    leg.set_zorder(8)

    # ---- (c) human takeover -------------------------------------------------
    axh.set_ylim(0, 1)
    axh.set_yticks([])
    for sp in ("top", "right", "left"):
        axh.spines[sp].set_visible(False)
    axh.add_patch(Rectangle((0, 0.20), T_END, 0.60, facecolor=BAND,
                            edgecolor="none", zorder=1))
    axh.plot([0, T_END], [0.5, 0.5], color=HAIR, lw=0.8, zorder=2)
    for (a, b) in hand:
        axh.add_patch(Rectangle((a, 0.20), b - a, 0.60, facecolor=SOFT,
                                edgecolor=MID, lw=0.7, hatch="xxx", zorder=3))
        axh.text(0.5 * (a + b), 0.94, f"driver holds the wheel "
                 f"{b - a:.1f} s", ha="center", va="bottom", fontsize=6.8,
                 color=INK, zorder=6,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.1,
                           alpha=0.93))
    lane_marks(axh, starts, 0.0, 1.0)
    axh.set_ylabel("human\ntakeover", fontsize=8.0, linespacing=1.5)

    # ---- (d) task progress --------------------------------------------------
    style_axes(axp)
    axp.set_axisbelow(True)
    axp.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    axp.set_ylim(0, 108)
    for (a, b) in eps:
        axp.add_patch(Rectangle((a, 0), b - a, 108, facecolor=CARD,
                                edgecolor="none", zorder=1))
    axp.plot([0, T_END], [0, unimp], color=PALE, lw=1.2, ls=(0, (5.0, 3.0)),
             zorder=3)
    axp.plot(t, progress, color=MID, lw=1.8, zorder=4)
    lane_marks(axp, starts, 0, 108)
    axp.text(116.0, float(rs) * 116.0 + 2.0,
             f"unimpeded  {rs} %/s $\\rightarrow$ {unimp:.2f} %", ha="left",
             va="bottom", fontsize=7.0, color=SUB, rotation=13.0,
             rotation_mode="anchor", zorder=6)
    axp.text(T_END - 2.0, prog_shown - 4.0, f"achieved  {prog_shown:.2f} %",
             ha="right", va="top", fontsize=7.4, color=DEEP,
             fontweight="bold", zorder=6,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                       alpha=0.93))
    axp.plot([T_END - 0.9, T_END - 0.9], [prog_shown, unimp], color=DEEP,
             lw=1.1, zorder=5)
    for lev in (prog_shown, unimp):                 # bracket serifs
        axp.plot([T_END - 2.6, T_END - 0.9], [lev, lev], color=DEEP, lw=1.1,
                 zorder=5)
    axp.text(T_END - 3.4, 0.5 * (prog_shown + unimp), f"{cost:.2f} % lost",
             ha="right", va="center", fontsize=7.0, color=INK, zorder=6,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                       alpha=0.93))
    axp.set_ylabel("task progress\n(%)", fontsize=8.0, linespacing=1.5)
    axp.set_xlabel("episode time  (s)", fontsize=8.4)
    axp.set_xticks(np.arange(0, T_END + 1, 20))

    # ---- headline + footer arithmetic --------------------------------------
    savefig(fig, "1202_wm_safe_guardrail_timeline.png")


if __name__ == "__main__":
    main()

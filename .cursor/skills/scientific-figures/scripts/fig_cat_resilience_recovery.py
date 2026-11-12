"""Figure 725 - Lifeline restoration after a damaging event.

Five lifelines lose functionality at t = 0 and recover along a stretched
exponential.  Three of them cannot start until the system they depend on has
been brought back far enough - communications wait for 40 % power, water for
60 % power, healthcare for 70 % water - so their restart times are computed
from the curves rather than assumed, and the delay each dependency costs is
measured off the same arrays.

Panel (b) collapses the five lifelines into one community-functionality curve
and re-runs it under three preparedness plans; the shaded resilience triangle
is integrated, so the percentages quoted are the actual area reductions.
Panel (c) is the restoration schedule underneath panel (a), with the chain of
tasks that sets the completion date drawn in the accent colour.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, lighten, darken, COLORS
from figkit import axes_inches, panel_label_inches

FW, FH = 10.4, 6.4
MAIN_X, MAIN_W, MAIN_TOP = 0.68, 5.30, 5.50
AUX_X, AUX_W, AUX_GAP = 7.05, 2.95, 0.66

ACCENT = "#b23a48"
GREEN = "#2e7d5b"
T_END = 70.0                         # days plotted after the event
T_PRE = 7.0                          # days of pre-event baseline

# name, residual functionality (%), recovery time constant (d),
# (system it waits for, threshold %), mobilisation lag (d)
SYSTEMS = [
    ("Power", 18.0, 7.0, None, 0.0),
    ("Transport", 42.0, 12.0, None, 1.0),
    ("Communications", 25.0, 5.0, ("Power", 40.0), 1.5),
    ("Water supply", 30.0, 9.0, ("Power", 60.0), 1.5),
    ("Healthcare", 55.0, 16.0, ("Water supply", 70.0), 1.5),
]
SYS_COLORS = [darken(GREEN, 0.42), darken(GREEN, 0.16), GREEN,
              lighten(GREEN, 0.30), lighten(GREEN, 0.52)]

PLANS = [("Baseline plan", 0.0, 1.00), ("Hardened assets", 15.0, 1.00),
         ("Rapid response", 0.0, 0.62)]
PLAN_COLORS = [darken(ACCENT, 0.32), ACCENT, lighten(ACCENT, 0.38)]

# label, start (d), duration (d), on the critical path
TASKS = [
    ("Debris clearance", 0.0, 4.0, True),
    ("Substation repair", 4.0, 6.5, True),
    ("Feeder energising", 10.5, 3.5, True),
    ("Pump restart", 14.0, 5.0, True),
    ("Main flushing", 19.0, 4.5, True),
    ("Hospital re-open", 23.5, 8.0, True),
    ("Road resurfacing", 4.0, 15.0, False),
]


def recover(t, q0, tau, start):
    """Stretched-exponential restoration from ``q0`` back to 100 %."""
    u = np.clip((t - start) / tau, 0.0, None)
    return q0 + (100.0 - q0) * (1.0 - np.exp(-u ** 1.6))


def crossing(t, q, level):
    """First time the curve reaches ``level``; NaN if it never does."""
    hit = np.nonzero(q >= level)[0]
    if hit.size == 0:
        return float("nan")
    i = hit[0]
    if i == 0:
        return float(t[0])
    return float(np.interp(level, [q[i - 1], q[i]], [t[i - 1], t[i]]))


def build(t, q_boost=0.0, tau_scale=1.0):
    """All five lifelines, resolving the dependency chain in order.

    Thresholds are searched on the post-event part only; the pre-event
    baseline sits at 100 % and would otherwise satisfy every threshold at
    the left edge of the axis.
    """
    post = t >= 0.0
    out, starts = {}, {}
    for name, q0, tau, dep, lag in SYSTEMS:
        start = lag
        if dep is not None:
            start = crossing(t[post], out[dep[0]][post], dep[1]) + lag
        starts[name] = start
        q0e = min(q0 + q_boost, 92.0)
        out[name] = np.where(t < 0.0, 100.0,
                             recover(t, q0e, tau * tau_scale, start))
    return out, starts


def main():
    set_pub_style(base_font=8)
    t = np.linspace(-T_PRE, T_END, 1600)

    base, starts = build(t)
    agg = np.mean([base[n] for n, *_ in SYSTEMS], axis=0)

    post = t >= 0.0
    plans = []
    for (label, boost, scale), col in zip(PLANS, PLAN_COLORS):
        curves = build(t, boost, scale)[0]
        a = np.mean([curves[n] for n, *_ in SYSTEMS], axis=0)
        loss = float(np.trapezoid(100.0 - a[post], t[post]))
        plans.append(dict(label=label, curve=a, loss=loss, color=col))
    base_loss = plans[0]["loss"]

    # ---- figure -------------------------------------------------------------
    fig = plt.figure(figsize=(FW, FH))
    main_h = 4.70
    main_y = MAIN_TOP - main_h
    ax = axes_inches(fig, (MAIN_X, main_y, MAIN_W, main_h))
    aux_h = (main_h - AUX_GAP) / 2.0
    ax_r = axes_inches(fig, (AUX_X, main_y + aux_h + AUX_GAP, AUX_W, aux_h))
    ax_g = axes_inches(fig, (AUX_X, main_y, AUX_W, aux_h))

    # ---- (a) lifeline restoration ------------------------------------------
    ax.fill_between(t[post], agg[post], 100.0, facecolor=ACCENT, alpha=0.10,
                    lw=0.0, zorder=1)
    ax.plot(t, agg, color="#333333", lw=1.2, ls=(0, (4.2, 2.0)), zorder=6,
            label="community mean")
    ax.axhline(100.0, color="#808080", lw=0.8, zorder=2)
    ax.axvline(0.0, color="#1a1a1a", lw=1.2, zorder=5)
    ax.text(0.6, 4.0, "event", fontsize=6.6, color="#1a1a1a", ha="left",
            va="bottom", rotation=90)

    for (name, q0, tau, dep, lag), col in zip(SYSTEMS, SYS_COLORS):
        q = base[name]
        t90 = crossing(t[post], q[post], 90.0)
        ax.plot(t, q, color=col, lw=1.8, zorder=5,
                label=f"{name}   90 % at day {t90:.0f}")
        ax.plot([t90], [90.0], marker="o", ms=3.8, mfc="white", mec=col,
                mew=1.2, zorder=7)

    ax.axhline(90.0, color="#808080", lw=0.7, ls=(0, (3.0, 2.2)), zorder=3)
    ax.text(T_END * 0.995, 90.6, "90 % service restored", fontsize=6.3,
            color=COLORS["muted"], ha="right", va="bottom")

    # dependency delays, measured off the curves that are drawn
    for name, ytxt in (("Water supply", 30.0), ("Healthcare", 55.0)):
        dep = dict((s[0], s[3]) for s in SYSTEMS)[name]
        d = starts[name]
        ax.annotate("", xy=(d, ytxt), xytext=(0.0, ytxt),
                    arrowprops=dict(arrowstyle="<|-|>", color=ACCENT,
                                    lw=0.9, shrinkA=0, shrinkB=0,
                                    mutation_scale=7), zorder=8)
        ax.text(d + 1.4, ytxt,
                f"{d:.1f} d waiting on {dep[1]:.0f} % {dep[0].lower()}",
                fontsize=6.3, color=ACCENT, ha="left", va="center",
                zorder=8)

    ax.set_xlim(-T_PRE, T_END)
    ax.set_ylim(0.0, 108.0)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("time since the event (days)", fontsize=8.0)
    ax.set_ylabel("system functionality $Q(t)$ (%)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.grid(axis="y", lw=0.5, alpha=0.42)
    ax.set_title("Restoration of five interdependent lifelines",
                 fontsize=8.8, fontweight="bold", pad=4.0)
    tri = float(np.trapezoid(100.0 - agg[post], t[post]))
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=ACCENT, alpha=0.16, linewidth=0.0))
    labels.append(f"loss of resilience   {tri:,.0f} %$\\cdot$d")
    ax.legend(handles, labels, loc="lower right", frameon=True,
              framealpha=0.93, facecolor="white", edgecolor="#cccccc",
              borderpad=0.45, fontsize=6.6, handlelength=1.9,
              labelspacing=0.32)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ---- (b) resilience triangle by plan ------------------------------------
    for p in plans:
        ax_r.fill_between(t[post], p["curve"][post], 100.0,
                          facecolor=p["color"], alpha=0.16, lw=0.0, zorder=2)
        cut = 1.0 - p["loss"] / base_loss
        tag = "reference" if abs(cut) < 1e-9 else f"$-${cut:.0%} area"
        ax_r.plot(t, p["curve"], color=p["color"], lw=1.6, zorder=5,
                  label=f"{p['label']}   {p['loss']:,.0f} %$\\cdot$d  ({tag})")
    ax_r.axhline(100.0, color="#808080", lw=0.8, zorder=3)
    ax_r.axvline(0.0, color="#1a1a1a", lw=1.0, zorder=4)
    ax_r.set_xlim(-T_PRE, T_END)
    ax_r.set_ylim(30.0, 108.0)
    ax_r.set_xlabel("time since the event (days)", fontsize=7.4)
    ax_r.set_ylabel("community $Q(t)$ (%)", fontsize=7.2)
    ax_r.tick_params(labelsize=6.6)
    ax_r.grid(axis="y", lw=0.5, alpha=0.42)
    ax_r.set_title("Resilience triangle under three plans", fontsize=8.5,
                   fontweight="bold", pad=3.5)
    ax_r.legend(loc="lower right", frameon=False, fontsize=6.2,
                handlelength=1.6, labelspacing=0.28)
    for sp in ("top", "right"):
        ax_r.spines[sp].set_visible(False)

    # ---- (c) restoration schedule and critical path -------------------------
    crit_end = max(s + d for _, s, d, c in TASKS if c)
    y = np.arange(len(TASKS))[::-1].astype(float)
    for yy, (label, s, d, c) in zip(y, TASKS):
        ax_g.barh(yy, d, left=s, height=0.52,
                  facecolor=ACCENT if c else "#b3b3b3", edgecolor="white",
                  linewidth=0.5, zorder=4)
        if not c:
            ax_g.plot([s + d, crit_end], [yy, yy], color="#b3b3b3", lw=0.8,
                      ls=(0, (2.0, 1.8)), zorder=3)
            ax_g.text(0.5 * (s + d + crit_end), yy - 0.30,
                      f"slack {crit_end - s - d:.1f} d", fontsize=6.0,
                      color=COLORS["muted"], ha="center", va="bottom",
                      zorder=5)
        else:
            ax_g.text(s + d + 0.7, yy, f"{d:.1f} d", fontsize=6.0,
                      color=COLORS["sub"], ha="left", va="center", zorder=5)
    for (a_, sa, da, ca), (b_, sb, db, cb), ya, yb in zip(
            TASKS[:-2], TASKS[1:-1], y[:-2], y[1:-1]):
        if ca and cb:
            ax_g.annotate("", xy=(sb, yb + 0.26), xytext=(sa + da, ya - 0.26),
                          arrowprops=dict(arrowstyle="-|>",
                                          color=darken(ACCENT, 0.30), lw=0.7,
                                          shrinkA=0, shrinkB=0,
                                          mutation_scale=5), zorder=6)
    ax_g.axvline(crit_end, color=darken(ACCENT, 0.30), lw=0.9,
                 ls=(0, (2.6, 1.8)), zorder=5)
    ax_g.set_yticks(y)
    ax_g.set_yticklabels([t_[0] for t_ in TASKS], fontsize=6.5)
    ax_g.set_xlim(0.0, crit_end * 1.24)
    ax_g.set_ylim(-0.7, len(TASKS) - 0.3)
    ax_g.set_xlabel("days since the event", fontsize=7.4)
    ax_g.tick_params(labelsize=6.6)
    ax_g.grid(axis="x", lw=0.5, alpha=0.42)
    ax_g.set_title("Restoration schedule and critical path", fontsize=8.5,
                   fontweight="bold", pad=3.5)
    ax_g.text(0.985, 0.97, f"critical path {crit_end:.1f} d",
              transform=ax_g.transAxes, fontsize=6.4,
              color=darken(ACCENT, 0.30), ha="right", va="top")
    for sp in ("top", "right"):
        ax_g.spines[sp].set_visible(False)

    for a, lab in ((ax, "(a)"), (ax_r, "(b)"), (ax_g, "(c)")):
        panel_label_inches(fig, a, lab, dx=-0.54, dy=0.09)

    t90_all = max(crossing(t[post], base[n][post], 90.0) for n, *_ in SYSTEMS)
    best = min(plans[1:], key=lambda p: p["loss"])

    savefig(fig, "725_cat_resilience_recovery.png")


if __name__ == "__main__":
    main()

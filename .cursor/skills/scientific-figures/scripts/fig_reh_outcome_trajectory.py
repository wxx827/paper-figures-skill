"""Functional independence recovery over six months of inpatient rehab.

Main panel: a Barthel-style 0-100 activities-of-daily-living score at
fortnightly reviews for three admission-severity strata, each drawn as a mean
with its 95 % confidence ribbon over the individual patient trajectories.
Severity is an ordered quantity so the strata take a sequential ramp rather
than three unrelated hues.  The plateau marker on each curve is the first
review at which the mean gain falls below one point per week, read off the
same means that are plotted.

Auxiliary (both at 58 % of the main panel width): those weekly gain rates
against the one-point-per-week rule, which is what dates each plateau; and the
proportional-recovery view, plotting the outcome at week 24 against the
admission score with an ordinary least-squares fit and the 70 %-of-headroom
line the fit is compared against.

Synthetic rehabilitation cohort, illustrative values only; no real patient.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, lighten, get_cmap
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

WEEKS = np.arange(0, 25, 2)               # fortnightly reviews
CEILING = 100.0                           # points, top of the scale
HEADROOM = 0.70                           # proportional-recovery fraction
PLATEAU_RULE = 1.0                        # points/week
# stratum name, n, admission score mean, admission SD, time constant weeks
STRATA = [("mild, 55-75 at admission", 22, 64.0, 5.0, 3.4),
          ("moderate, 30-50 at admission", 24, 40.0, 5.5, 5.2),
          ("severe, 5-25 at admission", 20, 16.0, 5.0, 7.6)]
THRESHOLDS = [(60.0, "assisted living"), (85.0, "independent at home")]


def stratum_curves(rng, n, s0_mean, s0_sd, tau_mean):
    """One saturating recovery trajectory per patient, in scale points."""
    s0 = np.clip(rng.normal(s0_mean, s0_sd, n), 2.0, 78.0)
    prop = np.clip(rng.normal(HEADROOM, 0.13, n), 0.18, 0.98)
    tau = np.clip(rng.normal(tau_mean, 1.5, n), 1.8, 13.0)
    plateau = s0 + prop * (CEILING - s0)
    w = WEEKS[None, :].astype(float)
    y = s0[:, None] + (plateau[:, None] - s0[:, None]) * (
        1.0 - np.exp(-w / tau[:, None]))
    y += rng.normal(0.0, 2.4, y.shape)
    y[:, 0] = s0
    return np.clip(np.maximum.accumulate(y, axis=1), 0.0, CEILING)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(981)

    cmap = get_cmap("cb_sequential_blue")
    cols = [cmap(v) for v in (0.42, 0.64, 0.90)]

    data = [stratum_curves(rng, n, m, s, t) for _, n, m, s, t in STRATA]
    means = [d.mean(axis=0) for d in data]
    cis = [1.96 * d.std(axis=0, ddof=1) / np.sqrt(d.shape[0]) for d in data]
    gains = [np.diff(m) / np.diff(WEEKS) for m in means]

    plateau_wk = []
    for g in gains:
        idx = np.flatnonzero(g < PLATEAU_RULE)
        plateau_wk.append(int(WEEKS[idx[0] + 1]) if idx.size else -1)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)
        a.grid(color=GRID, lw=0.5, alpha=0.6, zorder=0)

    # ---- main: trajectories by admission severity --------------------------
    for lvl, lab in THRESHOLDS:
        ax.axhline(lvl, color=NORM, lw=0.9, ls=(0, (4.0, 2.4)), zorder=2)
        ax.text(0.1, lvl + 1.0, lab, ha="left", va="bottom", fontsize=6.3,
                color=SUB, zorder=8,
                bbox=dict(boxstyle="square,pad=0.16", facecolor="white",
                          edgecolor="none", alpha=0.82))

    for k, ((name, n, _, _, _), d, m, ci, col) in enumerate(
            zip(STRATA, data, means, cis, cols)):
        for row in d:
            ax.plot(WEEKS, row, color=col, lw=0.5, alpha=0.28, zorder=3)
        ax.fill_between(WEEKS, m - ci, m + ci, color=col, alpha=0.30, lw=0,
                        zorder=4)
        ax.plot(WEEKS, m, color=col, lw=2.0, zorder=6, solid_capstyle="round")
        ax.plot(WEEKS, m, ls="none", marker="o", ms=3.0, mfc="white", mec=col,
                mew=1.0, zorder=7)
        pw = plateau_wk[k]
        if pw > 0:
            ym = float(np.interp(pw, WEEKS, m))
            ax.plot([pw], [ym], marker="v", ms=6.0, mfc=ACCENT, mec="white",
                    mew=0.7, zorder=9)
            ax.text(pw, ym + 3.2, f"plateau week {pw}", ha="center",
                    va="bottom", fontsize=6.2, color=ACCENT)
        ax.text(25.1, m[-1], f"{m[-1]:.0f} pts", ha="left", va="center",
                fontsize=6.4, color=col, fontweight="bold")

    handles = [Line2D([], [], color=c, lw=2.0,
                      label=f"{name} (n={n})")
               for (name, n, _, _, _), c in zip(STRATA, cols)]
    handles.append(Line2D([], [], color=ACCENT, lw=0, marker="v", ms=5.0,
                          label="mean gain drops below 1 point/week"))
    leg = ax.legend(handles=handles, loc="lower right", fontsize=6.5,
                    frameon=True, framealpha=0.94, borderpad=0.42,
                    handlelength=1.6, labelspacing=0.32)
    leg.get_frame().set(edgecolor=HAIR, linewidth=0.6)

    ax.set_xlim(-0.8, 27.4)
    ax.set_ylim(0.0, 104.0)
    ax.set_xticks(WEEKS)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel("weeks since admission (fortnightly review)", fontsize=8.2,
                  labelpad=2.5)
    ax.set_ylabel("activities-of-daily-living score (points, 0-100)",
                  fontsize=8.2, labelpad=3.0)
    ax.set_title(f"Six-month recovery by admission severity, "
                 f"n={sum(s[1] for s in STRATA)} patients, "
                 "mean with 95 % CI", fontsize=9.0, fontweight="bold", pad=5.0)

    # ---- aux (b): weekly gain rate -----------------------------------------
    mid = 0.5 * (WEEKS[1:] + WEEKS[:-1])
    for g, col, (name, *_ ) in zip(gains, cols, STRATA):
        ax_t.plot(mid, g, color=col, lw=1.6, marker="o", ms=3.0, mfc="white",
                  mew=0.9, zorder=5)
    ax_t.axhline(PLATEAU_RULE, color=ACCENT, lw=1.1, ls=(0, (3.0, 2.2)),
                 zorder=4)
    ax_t.text(0.0, PLATEAU_RULE - 0.16, f"{PLATEAU_RULE:.0f} point/week "
              "plateau rule", ha="left", va="top", fontsize=6.2, color=ACCENT)
    for pw, col in zip(plateau_wk, cols):
        if pw > 0:
            ax_t.axvline(pw, color=col, lw=0.8, ls=(0, (2.2, 2.0)), zorder=3)
    ax_t.set_xlim(-0.8, 24.8)
    ax_t.set_ylim(0.0, max(g.max() for g in gains) * 1.14)
    ax_t.set_xticks(np.arange(0, 25, 4))
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("weeks since admission", fontsize=7.6, labelpad=1.5)
    ax_t.set_ylabel("mean gain (points/week)", fontsize=7.6, labelpad=2.0)
    ax_t.set_title("The rate that dates each plateau", fontsize=8.4,
                   fontweight="bold", pad=4.0)

    # ---- aux (c): proportional recovery ------------------------------------
    x0 = np.concatenate([d[:, 0] for d in data])
    x24 = np.concatenate([d[:, -1] for d in data])
    grp = np.concatenate([np.full(d.shape[0], i) for i, d in enumerate(data)])
    b, a = np.polyfit(x0, x24, 1)
    pred = a + b * x0
    r2 = 1.0 - np.sum((x24 - pred) ** 2) / np.sum((x24 - x24.mean()) ** 2)
    xs = np.linspace(0.0, 80.0, 60)
    for i, col in enumerate(cols):
        sel = grp == i
        ax_b.plot(x0[sel], x24[sel], ls="none", marker="o", ms=3.0, mfc=col,
                  mec="white", mew=0.5, alpha=0.9, zorder=4)
    ax_b.plot(xs, xs + HEADROOM * (CEILING - xs), color=ACCENT, lw=1.2,
              ls=(0, (3.4, 2.2)), zorder=5)
    ax_b.plot(xs, a + b * xs, color=INK, lw=1.5, zorder=6)
    ax_b.plot(xs, xs, color=NORM, lw=0.9, ls=(0, (2.2, 2.0)), zorder=3)
    ax_b.text(2.0, 99.0, f"OLS  y = {a:.1f} + {b:.2f} x,  "
              f"$R^2$ = {r2:.2f}", ha="left", va="top", fontsize=6.3,
              color=INK)
    ax_b.text(2.0, 91.0, f"{100.0 * HEADROOM:.0f} % of headroom rule",
              ha="left", va="top", fontsize=6.3, color=ACCENT)
    ax_b.text(78.0, 12.0, "no change", ha="right", va="bottom", fontsize=6.2,
              color=MUTED)
    ax_b.set_xlim(0.0, 80.0)
    ax_b.set_ylim(0.0, 104.0)
    ax_b.set_xticks(np.arange(0, 81, 20))
    ax_b.set_yticks(np.arange(0, 101, 25))
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("admission score (points)", fontsize=7.6, labelpad=1.5)
    ax_b.set_ylabel("week 24 score (points)", fontsize=7.6, labelpad=2.0)
    ax_b.set_title("Outcome is mostly set on admission", fontsize=8.4,
                   fontweight="bold", pad=4.0)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.52, dy=0.10)
    savefig(fig, "981_reh_outcome_trajectory.png")


if __name__ == "__main__":
    main()

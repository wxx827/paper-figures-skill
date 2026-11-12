"""Twenty-five budget rounds seen as a shifting share of one envelope.

Main panel: the five programme areas of a fictional national budget stacked
as shares of the total, so the panel answers "what did the money move
towards" rather than "did the money grow".  Each band is labelled at its
right-hand end with the change in percentage points since the first year,
and the two areas that moved most are pulled out.  The envelope itself is
the dark line on the right-hand axis in billions, which is what stops a
falling share from being read as a falling budget.

Auxiliary (both at 58 % of the main panel width): spending per head against
the outcome index for each area, drawn as a trajectory through the years so
that the slope is the return on an extra dollar; and the budget execution
rate, the share of each appropriation actually spent.

The shares are normalised to sum to exactly 100 % in every year, so the
stack is airtight; the percentage-point changes are differences between the
plotted end points, per-head spending is the plotted total times the plotted
share divided by the population series, and each return slope is a
least-squares fit through the points shown.  Fictional country, illustrative
values only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, style_axes, lighten, darken
from figkit import axes_inches, panel_labels_inches, halo_text

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"
ACCENT = "#b23a48"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.92, 0.86, 5.46, 4.62)
AUX_T_RECT = (7.70, 3.48, 3.08, 2.00)
AUX_B_RECT = (7.70, 0.86, 3.08, 2.00)

YEAR0, YEAR1 = 2000, 2024
# stacked bottom to top; (name, colour, start share, end share, curvature)
AREAS = [
    ("Social protection", "#3b5b92", 30.0, 33.5, 0.35),
    ("Health", "#2e7d5b", 19.0, 27.5, 0.55),
    ("Education", "#c9a227", 24.0, 19.0, 0.30),
    ("Transport", "#8e6bbf", 19.5, 12.5, 0.60),
    ("Environment", "#c0603f", 7.5, 7.5, 0.50),
]
# outcome index gain per area: (level, response to cumulative spend)
OUTCOME = [(48.0, 7.2), (52.0, 11.2), (55.0, 5.0), (44.0, 9.4), (38.0, 13.4)]
EXECUTION = [97.4, 94.1, 96.8, 86.3, 78.5]


def build_budget(seed=927):
    """Shares (%), total envelope (B USD) and population (millions)."""
    rng = np.random.default_rng(seed)
    yr = np.arange(YEAR0, YEAR1 + 1)
    t = (yr - YEAR0) / (YEAR1 - YEAR0)

    raw = []
    for _n, _c, s0, s1, curve in AREAS:
        path = s0 + (s1 - s0) * (t ** (1.0 / (1.0 + curve)))
        path = path * np.exp(rng.normal(0.0, 0.021, t.size)).cumprod() ** 0.30
        raw.append(path)
    raw = np.array(raw)
    share = 100.0 * raw / raw.sum(axis=0)          # airtight by construction

    total = 214.0 * np.exp(0.052 * (yr - YEAR0))
    total *= 1.0 - 0.075 * np.exp(-((yr - 2009) / 1.6) ** 2)
    total *= 1.0 - 0.058 * np.exp(-((yr - 2020) / 1.2) ** 2)
    total *= np.exp(rng.normal(0.0, 0.008, yr.size))
    pop = 33.4 * np.exp(0.0068 * (yr - YEAR0))
    return yr, share, total, pop


def main():
    set_pub_style(base_font=8)
    yr, share, total, pop = build_budget()
    assert np.allclose(share.sum(axis=0), 100.0)

    names = [a[0] for a in AREAS]
    cols = [a[1] for a in AREAS]
    delta = share[:, -1] - share[:, 0]
    riser = int(np.argmax(delta))
    faller = int(np.argmin(delta))

    # per head in USD: B USD -> USD, share in %, population in millions
    per_cap = (total[None, :] * 1e9) * (share / 100.0) / (pop[None, :] * 1e6)
    cum = np.cumsum(per_cap, axis=1)
    rng = np.random.default_rng(9271)
    outcome = np.array([
        lvl + resp * np.log1p(cum[i] / cum[i, 0])
        + rng.normal(0.0, 0.55, yr.size)
        for i, (lvl, resp) in enumerate(OUTCOME)])
    ret = np.array([np.polyfit(per_cap[i], outcome[i], 1)[0] * 100.0
                    for i in range(len(AREAS))])

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_e = axes_inches(fig, AUX_T_RECT)
    ax_x = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_e, ax_x):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: stacked shares plus the envelope on the right axis -----------
    ax.stackplot(yr, share, colors=[lighten(c, 0.24) for c in cols],
                 edgecolor="white", linewidth=0.8, zorder=3)
    cum_sh = np.cumsum(share, axis=0)
    for i, c in enumerate(cols):
        ax.plot(yr, cum_sh[i], color=darken(c, 0.12), lw=0.9, zorder=4)

    mid = cum_sh - share / 2.0
    for i, name in enumerate(names):
        hot = i in (riser, faller)
        ax.text(YEAR1 + 0.9, mid[i, -1] + 1.2, name, ha="left", va="center",
                fontsize=7.2, color=ACCENT if hot else darken(cols[i], 0.22),
                fontweight="bold" if hot else "normal", zorder=6)
        ax.text(YEAR1 + 0.9, mid[i, -1] - 2.1,
                f"{share[i, -1]:.1f} %  ({delta[i]:+.1f} pp)", ha="left",
                va="center", fontsize=6.3, color=ACCENT if hot else MUTED,
                zorder=6)

    for i, verb in ((riser, "largest gain"), (faller, "largest fall")):
        halo_text(ax, YEAR0 + 1.1, mid[i, 4],
                  f"{names[i]}: {delta[i]:+.1f} pp, the {verb}",
                  color=ACCENT, halo="white", halo_lw=2.6, fontsize=6.9,
                  fontweight="bold", ha="left", va="center", zorder=10)
        ax.annotate("", xy=(YEAR0 + 15.0, mid[i, 15]),
                    xytext=(YEAR0 + 11.9, mid[i, 4]),
                    arrowprops=dict(arrowstyle="-|>", lw=0.85, color=ACCENT,
                                    connectionstyle="arc3,rad=0.16",
                                    shrinkA=1, shrinkB=1), zorder=10)

    ax.set_xlim(YEAR0, YEAR1 + 9.2)
    ax.set_ylim(0.0, 124.0)
    ax.set_xticks(np.arange(YEAR0, YEAR1 + 1, 4))
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.spines["left"].set_bounds(0.0, 100.0)
    ax.spines["bottom"].set_bounds(YEAR0, YEAR1)
    ax.set_xlabel("budget year", fontsize=8.5)
    ax.set_ylabel("share of the total budget (%)", fontsize=8.5)
    ax.set_title("Programme shares of the Aldrune national budget",
                 fontsize=9.0, fontweight="bold", pad=5.0)

    ax2 = ax.twinx()
    ax2.plot(yr, total, color=INK, lw=1.9, zorder=8, solid_capstyle="round")
    ax2.plot(yr[::4], total[::4], ls="none", marker="o", ms=3.4, mfc="white",
             mec=INK, mew=1.1, zorder=9)
    ax2.set_xlim(YEAR0, YEAR1 + 9.2)
    ax2.set_ylim(0.0, 1300.0)
    ax2.set_yticks([0, 200, 400, 600, 800])
    ax2.set_ylabel("total budget envelope (B USD)", fontsize=8.5,
                   color=INK, labelpad=2.0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_bounds(0.0, 800.0)
    ax2.tick_params(length=3)
    ax2.set_zorder(5)
    ax2.patch.set_visible(False)

    # headroom strip above the airtight stack, for the envelope note
    ax.plot([YEAR0 + 0.4, YEAR0 + 1.9], [109.0, 109.0], color=INK, lw=1.9,
            solid_capstyle="round", zorder=10)
    ax.plot([YEAR0 + 1.15], [109.0], marker="o", ms=3.4, mfc="white",
            mec=INK, mew=1.1, zorder=11)
    ax.text(YEAR0 + 2.4, 109.0,
            f"total budget envelope, right axis: {total[0]:.0f} -> "
            f"{total[-1]:.0f} B USD over the period "
            f"({total[-1] / total[0]:.1f}x)",
            ha="left", va="center", fontsize=6.9, color=INK, zorder=10)

    # ---- aux (b): what a dollar per head buys -------------------------------
    ax_e.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    # limits come from the data, otherwise the end labels walk off the canvas
    x_hi = float(per_cap.max()) * 1.46
    for i, name in enumerate(names):
        ax_e.plot(per_cap[i], outcome[i], color=lighten(cols[i], 0.18),
                  lw=1.3, zorder=4)
        ax_e.plot([per_cap[i, -1]], [outcome[i, -1]], marker="o", ms=4.2,
                  mfc=cols[i], mec="white", mew=0.7, zorder=6)
        ax_e.text(per_cap[i, -1] + 0.016 * x_hi, outcome[i, -1],
                  f"{name}  {ret[i]:+.2f}", ha="left", va="center",
                  fontsize=6.1, color=darken(cols[i], 0.18), zorder=6)
    ax_e.set_xlim(0.0, x_hi)
    ax_e.set_ylim(float(outcome.min()) - 5.0, float(outcome.max()) + 8.0)
    ax_e.tick_params(labelsize=6.6)
    ax_e.set_xlabel("programme spending (USD per capita per year)",
                    fontsize=7.6, labelpad=1.5)
    ax_e.set_ylabel("outcome index (0-100)", fontsize=7.6)
    ax_e.set_title("The same dollar buys different outcomes", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_e.text(0.975, 0.045,
              f"trajectories run {YEAR0} to {YEAR1}; the number after\n"
              f"each name is index points gained per extra\n"
              f"100 USD per capita spent in that area",
              transform=ax_e.transAxes, ha="right", va="bottom", fontsize=6.0,
              color=SUB, linespacing=1.5)

    # ---- aux (c): execution rate --------------------------------------------
    ax_x.grid(axis="x", color=GRID, lw=0.5, alpha=0.7, zorder=0)
    exe = np.array(EXECUTION)
    idx = np.argsort(exe)[::-1]
    for k, i in enumerate(idx):
        col = ACCENT if exe[i] < 90.0 else cols[i]
        ax_x.barh(k, exe[i], height=0.60, color=lighten(col, 0.34),
                  edgecolor=col, linewidth=0.85, zorder=3)
        ax_x.text(exe[i] - 1.4, k, f"{exe[i]:.1f} %", ha="right", va="center",
                  fontsize=6.3, color="white", fontweight="bold", zorder=5)
        ax_x.text(exe[i] + 1.2, k,
                  f"{(100.0 - exe[i]) / 100.0 * total[-1] * share[i, -1] / 100.0:.1f} "
                  f"B USD unspent", ha="left", va="center", fontsize=6.0,
                  color=SUB, zorder=5)
    ax_x.axvline(100.0, color=FAINT, lw=0.9, ls=(0, (3.0, 2.2)), zorder=4)
    ax_x.set_yticks(range(len(AREAS)))
    ax_x.set_yticklabels([names[i] for i in idx], fontsize=6.6)
    ax_x.set_ylim(-0.7, len(AREAS) + 0.35)
    ax_x.invert_yaxis()
    ax_x.set_xlim(60.0, 118.0)
    ax_x.set_xticks([60, 70, 80, 90, 100])
    ax_x.tick_params(axis="x", labelsize=6.6)
    ax_x.set_xlabel("budget executed in the final year "
                    "(% of appropriation)", fontsize=7.6, labelpad=1.5)
    ax_x.set_title("Appropriated is not the same as spent", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_x.text(117.0, len(AREAS) + 0.22,
              "dashed rule is full execution of the appropriation",
              ha="right", va="bottom", fontsize=6.0, color=SUB)

    panel_labels_inches(fig, [ax, ax_e, ax_x], dx=-0.66, dy=0.12)
    savefig(fig, "927_pcy_budget_allocation.png")


if __name__ == "__main__":
    main()

"""Cohort retention: the triangle, the curve it averages to, and the payback.

Main panel: the classic retention triangle -- acquisition month down the side,
lifecycle month across the top, colour and number both showing the share of the
cohort still active.  Cells past the observation window are left blank rather
than imputed, so the reader can see exactly how thin the evidence gets on the
right.  The two narrow right-hand panels turn the triangle into the two numbers
a growth team acts on: the retention curve by acquisition source, and the
cumulative LTV against the cost of acquiring the customer in the first place.

``gen_cohorts`` produces the triangle and the per-source curves; the M1 / M3 /
M12 call-outs, the worst cohort and the payback month are all read back out.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, style_axes, get_cmap
from figkit import panel_labels_inches, safe_colorbar

GREEN = "#2e7d5b"
BLUE = "#3b5b92"
GOLD = "#c9a227"
RED = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRID = "#ededed"

N_COHORT = 12
N_LIFE = 13                      # M0 .. M12
FIRST_MONTH = "2025"
SOURCES = ["Referral", "Organic", "Paid"]
SCOL = {"Referral": GOLD, "Organic": GREEN, "Paid": BLUE}
# source -> (asymptotic floor, decay constant in months, mix weight)
SPEC = {"Referral": (0.45, 3.2, 0.15),
        "Organic": (0.36, 2.6, 0.40),
        "Paid": (0.17, 1.8, 0.45)}
ARPU = 32.0                      # USD per active customer per month
CAC = 148.0                      # USD per acquired customer
PROMO = 5                        # the discount-led cohort


def source_curve(floor, tau, m):
    """Retention of a source at lifecycle month m, anchored at 1.0 in M0."""
    return np.where(m == 0, 1.0, floor + (1.0 - floor) * np.exp(-m / tau))


def gen_cohorts(seed=641):
    """Retention triangle in per cent plus the blended per-source curves."""
    rng = np.random.default_rng(seed)
    m = np.arange(N_LIFE)
    curves = {s: source_curve(SPEC[s][0], SPEC[s][1], m) for s in SOURCES}
    blend = sum(SPEC[s][2] * curves[s] for s in SOURCES)

    tri = np.full((N_COHORT, N_LIFE), np.nan)
    for i in range(N_COHORT):
        q = 1.0 + 0.10 * np.cos(2 * np.pi * i / N_COHORT)
        if i == PROMO:
            q *= 0.80                       # discount cohort, weaker intent
        row = blend * q * np.exp(rng.normal(0.0, 0.030, N_LIFE))
        row[0] = 1.0
        tri[i, :N_LIFE - i] = np.clip(row[:N_LIFE - i], 0.02, 1.0)
    return tri * 100.0, curves, blend


def main():
    set_pub_style(base_font=8)
    tri, curves, blend = gen_cohorts()
    months = [f"{FIRST_MONTH}-{k:02d}" for k in range(1, N_COHORT + 1)]
    life = np.arange(N_LIFE)

    obs = ~np.isnan(tri)
    avg = np.nanmean(tri, axis=0)
    n_obs = obs.sum(axis=0)
    m1, m3, m12 = avg[1], avg[3], avg[12]
    worst = int(np.nanargmin(tri[:, 3]))

    ltv = np.cumsum(blend * ARPU)
    ltv_hi = np.cumsum(curves["Referral"] * ARPU)
    ltv_lo = np.cumsum(curves["Paid"] * ARPU)
    payback = np.nonzero(ltv >= CAC)[0]
    payback = int(payback[0]) if payback.size else -1

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          left=0.076, right=0.972, top=0.878, bottom=0.132,
                          hspace=0.50, wspace=0.300)
    ax = fig.add_subplot(gs[:, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    ax_l = fig.add_subplot(gs[1, 1])

    for a in (ax_r, ax_l):
        a.set_axisbelow(True)
        a.grid(color=GRID, lw=0.7, zorder=0)
        style_axes(a)

    # ---- main: the retention triangle --------------------------------------
    cmap = get_cmap("cb_sequential_green").copy()
    cmap.set_bad("#f4f4f4")
    im = ax.imshow(tri, cmap=cmap, aspect="auto", vmin=0.0, vmax=100.0,
                   zorder=2)
    for i in range(N_COHORT):
        for j in range(N_LIFE):
            if obs[i, j]:
                v = tri[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=6.0, zorder=6,
                        color="white" if v > 52 else INK)

    ax.add_patch(Rectangle((-0.5, worst - 0.5), N_LIFE - worst, 1.0,
                           facecolor="none", edgecolor=RED, linewidth=1.5,
                           zorder=7))
    ax.annotate(f"{months[worst]} discount cohort:\n"
                f"M3 retention {tri[worst, 3]:.0f}% against "
                f"{np.nanmean(np.delete(tri[:, 3], worst)):.0f}% elsewhere",
                xy=(N_LIFE - worst - 0.6, worst), xytext=(9.8, worst + 2.5),
                ha="center", va="center", fontsize=6.8, color=RED,
                linespacing=1.45, zorder=9,
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=RED,
                                shrinkA=2, shrinkB=3),
                bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                          edgecolor="#e0c3c8", linewidth=0.7, alpha=0.95))

    ax.set_xticks(life)
    ax.set_xticklabels([f"M{j}" for j in life], fontsize=6.8)
    ax.set_yticks(np.arange(N_COHORT))
    ax.set_yticklabels(months, fontsize=6.8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    ax.set_xlabel("Lifecycle month since acquisition (months)", fontsize=9.0)
    ax.set_ylabel("Acquisition cohort (month)", fontsize=9.0)
    ax.set_title("Retention triangle, blank where the window has not closed",
                 fontsize=9.0, fontweight="bold", pad=5.0)
    safe_colorbar(fig, im, ax, label="customers still active (%)",
                  width=0.12, pad=0.09)

    # ---- aux (b): retention curve by source --------------------------------
    for s in SOURCES:
        ax_r.plot(life, curves[s] * 100.0, color=SCOL[s], lw=1.6, zorder=5)
    ax_r.plot(life, avg, color="#4d4d4d", lw=1.9, ls=(0, (3.4, 2.0)),
              zorder=6)
    ax_r.set_xlim(-0.3, N_LIFE + 0.9)
    ax_r.set_ylim(0, 105)
    ax_r.set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax_r.set_yticks([0, 25, 50, 75, 100])
    ax_r.tick_params(labelsize=6.8)
    ax_r.set_xlabel("lifecycle month (months)", fontsize=8.0, labelpad=1.5)
    ax_r.set_ylabel("retention (%)", fontsize=8.0)
    ax_r.set_title("Curves flatten, but at very different floors",
                   fontsize=8.5, fontweight="bold", pad=4.0)
    for j, lab in ((1, "M1"), (3, "M3"), (12, "M12")):
        ax_r.plot([j], [avg[j]], marker="o", ms=4.0, mfc="white",
                  mec="#4d4d4d", mew=1.2, zorder=8)
        ax_r.annotate(f"{lab} {avg[j]:.0f}%", xy=(j, avg[j]),
                      xytext=(4, 7), textcoords="offset points", ha="left",
                      va="bottom", fontsize=6.5, color="#333333")
    ax_r.legend(handles=[Line2D([], [], color=SCOL[s], lw=1.6, label=s)
                         for s in SOURCES]
                + [Line2D([], [], color="#4d4d4d", lw=1.9, ls=(0, (3.4, 2.0)),
                          label="cohort average")],
                loc="upper right", frameon=False, fontsize=6.5,
                handlelength=1.9, labelspacing=0.32, borderpad=0.25)

    # ---- aux (c): cumulative LTV against CAC -------------------------------
    ax_l.fill_between(life, ltv_lo, ltv_hi, color="#cfe1ee", alpha=0.75,
                      lw=0, zorder=3)
    ax_l.plot(life, ltv, color=BLUE, lw=1.9, zorder=6)
    ax_l.axhline(CAC, color=RED, lw=1.3, ls=(0, (3.2, 2.0)), zorder=5)
    ax_l.set_xlim(-0.3, N_LIFE - 0.7)
    ax_l.set_ylim(0, max(ltv_hi.max(), CAC) * 1.16)
    ax_l.set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax_l.tick_params(labelsize=6.8)
    ax_l.set_xlabel("lifecycle month (months)", fontsize=8.0, labelpad=1.5)
    ax_l.set_ylabel("cumulative LTV (USD)", fontsize=8.0)
    ax_l.set_title("Payback lands inside the first year", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_l.text(0.2, CAC + 6, f"CAC = {CAC:,.0f} USD", ha="left",
              va="bottom", fontsize=6.5, color=RED)
    if payback > 0:
        ax_l.plot([payback], [ltv[payback]], marker="o", ms=4.6, mfc="white",
                  mec=BLUE, mew=1.4, zorder=8)
        ax_l.annotate(f"payback at M{payback}", xy=(payback, ltv[payback]),
                      xytext=(-7, 9), textcoords="offset points",
                      ha="right", va="bottom", fontsize=6.5, color=BLUE)
    ax_l.text(0.985, 0.055,
              f"LTV/CAC = {ltv[-1] / CAC:.1f} by M{N_LIFE - 1}",
              transform=ax_l.transAxes, ha="right", va="bottom",
              fontsize=6.5, color=SUB)
    ax_l.legend(handles=[
        Line2D([], [], color=BLUE, lw=1.9, label="blended cohort"),
        Line2D([], [], color="#cfe1ee", lw=4.0, label="paid to referral range"),
        Line2D([], [], color=RED, lw=1.3, ls=(0, (3.2, 2.0)),
               label="acquisition cost")],
        loc="upper left", frameon=False, fontsize=6.4, handlelength=1.8,
        labelspacing=0.3, borderpad=0.25)

    panel_labels_inches(fig, [ax, ax_r, ax_l], dx=-0.50, dy=0.06)

    savefig(fig, "641_mkt_cohort_retention.png")


if __name__ == "__main__":
    main()

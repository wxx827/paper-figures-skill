"""Regulatory impact assessment: when a rule that costs first starts to pay.

Main panel: cumulative discounted compliance cost against cumulative
discounted benefit over a twenty-year appraisal window.  Cost is front
loaded and flattens once the capital programme is done; benefit ramps in
behind it.  The shaded wedge between the two curves is the running deficit
and then the running surplus, and the point where they cross is the payback
year, read off by interpolation rather than asserted.  The net present value
curve is drawn against the same money axis, and a family of thin NPV curves
shows what a different discount rate would have done to that conclusion.

Auxiliary (both at 58 % of the main panel width): the waterfall that takes
the three benefit streams and the three cost streams to the net present
value; and a tornado of the assumptions, each swung over a plausible range
with everything else held at base.

The whole figure comes out of one appraisal function: flows are integrated
numerically against a continuous discount factor, the payback year is the
zero crossing of the plotted NPV curve, the waterfall bars are the present
values of the same streams, and every tornado bar is a re-run of the same
function.  Fictional rule, illustrative values only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, lighten, darken
from figkit import axes_inches, panel_labels_inches

DEEP = "#2b4470"
BLUE = "#3b5b92"
TEAL = "#2e7d5b"
RUST = "#c0603f"
ACCENT = "#b23a48"
GOLD = "#c9a227"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (1.00, 0.86, 5.42, 4.62)
AUX_T_RECT = (7.72, 3.48, 3.06, 2.00)
AUX_B_RECT = (7.72, 0.86, 3.06, 2.00)

HORIZON = 20.0
DT = 0.02
BASE_RATE = 0.04
RATES = [0.02, 0.04, 0.06, 0.08]

# (label, annual rate M USD, decay time constant yr or None for flat)
COSTS = [("Capital outlay", 430.0, 2.2),
         ("Ongoing compliance", 42.0, None),
         ("Administrative burden", 13.0, None)]
# (label, plateau rate M USD, ramp time constant yr, lag yr)
BENEFITS = [("Avoided damages", 175.0, 4.5, 1.5),
            ("Health co-benefits", 76.0, 6.0, 2.5),
            ("Productivity gain", 42.0, 8.0, 4.0)]
TAKE_UP = 0.90


def flows(t, capital=1.0, ongoing=1.0, ramp=1.0, damages=1.0, take_up=TAKE_UP):
    """Annual cost and benefit flows in M USD, with multiplier knobs."""
    scale = {"Capital outlay": capital, "Ongoing compliance": ongoing,
             "Administrative burden": ongoing}
    cost = {}
    for name, rate, tau in COSTS:
        cost[name] = (rate * scale[name] * (np.exp(-t / tau) if tau
                                            else np.ones_like(t)))
    ben = {}
    for name, rate, tau, lag in BENEFITS:
        amp = rate * (damages if name == "Avoided damages" else 1.0)
        ben[name] = (take_up * amp
                     * (1.0 - np.exp(-np.maximum(t - lag, 0.0) / (tau * ramp))))
    return cost, ben


def appraise(rate=BASE_RATE, **knobs):
    """Time grid, cumulative discounted cost / benefit / NPV, in M USD."""
    t = np.arange(0.0, HORIZON + DT, DT)
    cost, ben = flows(t, **knobs)
    disc = np.exp(-rate * t)
    c_tot = sum(cost.values()) * disc
    b_tot = sum(ben.values()) * disc
    cum_c = np.cumsum(c_tot) * DT
    cum_b = np.cumsum(b_tot) * DT
    pv = {k: float(np.sum(v * disc) * DT) for k, v in cost.items()}
    pv.update({k: float(np.sum(v * disc) * DT) for k, v in ben.items()})
    return t, cum_c, cum_b, cum_b - cum_c, pv


def payback(t, npv):
    """First upward zero crossing of the NPV curve, linearly interpolated."""
    hit = np.flatnonzero((npv[:-1] < 0.0) & (npv[1:] >= 0.0))
    if hit.size == 0:
        return np.nan
    i = int(hit[0])
    f = -npv[i] / (npv[i + 1] - npv[i])
    return float(t[i] + f * (t[i + 1] - t[i]))


def waterfall(ax, labels, values, colors, total_label, total_color):
    """Bars that chain from zero to the running total, plus a total bar."""
    run = 0.0
    for k, (v, c) in enumerate(zip(values, colors)):
        lo = min(run, run + v)
        ax.bar(k, abs(v), bottom=lo, width=0.64, color=lighten(c, 0.30),
               edgecolor=c, linewidth=0.85, zorder=4)
        ax.text(k, max(run, run + v) + 62.0, f"{v:+,.0f}", ha="center",
                va="bottom", fontsize=6.0, color=darken(c, 0.15), zorder=6)
        if k + 1 < len(values):
            ax.plot([k + 0.32, k + 1 - 0.32], [run + v, run + v], lw=0.7,
                    color="#b8b8b8", ls=(0, (2.4, 1.8)), zorder=3)
        run += v
    k = len(values)
    ax.plot([k - 0.68, k - 0.32], [run, run], lw=0.7, color="#b8b8b8",
            ls=(0, (2.4, 1.8)), zorder=3)
    ax.bar(k, run, width=0.64, color=lighten(total_color, 0.24),
           edgecolor=total_color, linewidth=1.0, zorder=4)
    ax.text(k, run + 62.0, f"{run:+,.0f}", ha="center", va="bottom",
            fontsize=6.4, fontweight="bold", color=total_color, zorder=6)
    return labels + [total_label], run


def main():
    set_pub_style(base_font=8)
    t, cum_c, cum_b, npv, pv = appraise()
    pb = payback(t, npv)
    npv_end = float(npv[-1])
    bcr = float(cum_b[-1] / cum_c[-1])

    fam = {}
    for r in RATES:
        _t, cc, bb, nn, _p = appraise(rate=r)
        fam[r] = (nn, payback(_t, nn))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_w = axes_inches(fig, AUX_T_RECT)
    ax_s = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_w, ax_s):
        a.set_axisbelow(True)
        style_axes(a)
        a.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)

    # ---- main: cumulative cost, benefit and NPV -----------------------------
    ax.axhline(0.0, color=FAINT, lw=0.9, zorder=2)
    ax.fill_between(t, cum_c, cum_b, where=cum_c >= cum_b,
                    facecolor=lighten(RUST, 0.76), linewidth=0.0, zorder=1)
    ax.fill_between(t, cum_c, cum_b, where=cum_b > cum_c,
                    facecolor=lighten(TEAL, 0.76), linewidth=0.0, zorder=1)
    ax.plot(t, cum_c, color=RUST, lw=2.0, zorder=6, solid_capstyle="round")
    ax.plot(t, cum_b, color=TEAL, lw=2.0, zorder=6, solid_capstyle="round")

    for r in RATES:
        if r == BASE_RATE:
            continue
        ax.plot(t, fam[r][0], color=lighten(DEEP, 0.52), lw=0.95, zorder=4)
        ax.text(HORIZON + 0.25, fam[r][0][-1], f"{100 * r:.0f} %", ha="left",
                va="center", fontsize=6.0, color=lighten(DEEP, 0.30))
    ax.plot(t, npv, color=DEEP, lw=2.0, zorder=7, solid_capstyle="round")

    ax.plot([pb, pb], [-980.0, 0.0], color=ACCENT, lw=1.0, ls=(0, (3.0, 2.2)),
            zorder=5)
    ax.plot([pb], [0.0], marker="o", ms=5.0, mfc="white", mec=ACCENT, mew=1.5,
            zorder=9)
    ax.plot([pb], [float(np.interp(pb, t, cum_c))], marker="o", ms=4.4,
            mfc="white", mec=ACCENT, mew=1.3, zorder=9)

    ax.text(HORIZON + 0.25, cum_c[-1], "cumulative cost", ha="left",
            va="center", fontsize=7.0, color=darken(RUST, 0.12))
    ax.text(HORIZON + 0.25, cum_b[-1], "cumulative benefit", ha="left",
            va="center", fontsize=7.0, color=darken(TEAL, 0.12))
    ax.text(HORIZON + 0.25, npv[-1] + 95.0, f"net present value\n"
            f"at {100 * BASE_RATE:.0f} % discount", ha="left", va="center",
            fontsize=7.0, color=DEEP, linespacing=1.4)

    ax.annotate(f"payback in year {pb:.1f}, where the cumulative\n"
                f"curves cross and the NPV turns positive",
                xy=(pb, 0.0), xytext=(pb + 0.8, -1000.0), fontsize=6.9,
                color=ACCENT, ha="left", va="bottom", linespacing=1.5,
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=ACCENT,
                                connectionstyle="arc3,rad=-0.22",
                                shrinkA=2, shrinkB=4))
    ax.text(0.55, 2320.0,
            f"NPV at year {HORIZON:.0f} = {npv_end:+,.0f} M USD   |   "
            f"benefit-cost ratio {bcr:.2f}   |   "
            f"present-value cost {cum_c[-1]:,.0f} M USD, "
            f"benefit {cum_b[-1]:,.0f} M USD",
            ha="left", va="bottom", fontsize=6.9, color=SUB)

    ax.set_xlim(0.0, HORIZON + 5.6)
    ax.set_ylim(-1050.0, 2560.0)
    ax.set_xticks(np.arange(0, 21, 4))
    ax.set_yticks([-1000, -500, 0, 500, 1000, 1500, 2000])
    ax.spines["bottom"].set_bounds(0.0, HORIZON)
    ax.set_xlabel("years since the rule takes effect", fontsize=8.5)
    ax.set_ylabel("cumulative present value (M USD)", fontsize=8.5)
    ax.set_title("Appraisal of the Aldrune emissions reporting rule, "
                 "20-year window", fontsize=9.0, fontweight="bold", pad=5.0)

    hh = [Line2D([], [], color=lighten(DEEP, 0.52), lw=0.95,
                 label="NPV at 2 / 6 / 8 % discount"),
          Line2D([], [], color=ACCENT, lw=1.0, ls=(0, (3.0, 2.2)),
                 label="payback year")]
    ax.legend(handles=hh, loc="upper left", frameon=False, fontsize=6.6,
              handlelength=2.1, labelspacing=0.32, borderpad=0.15,
              bbox_to_anchor=(0.012, 0.885))

    # ---- aux (b): waterfall to the net present value ------------------------
    b_names = [b[0] for b in BENEFITS]
    c_names = [c[0] for c in COSTS]
    vals = [pv[n] for n in b_names] + [-pv[n] for n in c_names]
    cols = [TEAL] * 3 + [RUST] * 3
    labs, run = waterfall(ax_w, b_names + c_names, vals, cols,
                          "Net present value", DEEP)
    assert abs(run - npv_end) < 1.0
    ax_w.axhline(0.0, color=FAINT, lw=0.9, zorder=2)
    ax_w.set_xticks(range(len(labs)))
    ax_w.set_xticklabels(["Avoided\ndamages", "Health\nco-benefits",
                          "Productivity\ngain", "Capital\noutlay",
                          "Ongoing\ncompliance", "Admin\nburden",
                          "Net present\nvalue"], fontsize=6)
    ax_w.set_xlim(-0.68, len(labs) - 0.32)
    ax_w.set_ylim(-260.0, 2620.0)
    ax_w.set_yticks([0, 1000, 2000])
    ax_w.tick_params(axis="y", labelsize=6.6)
    ax_w.tick_params(axis="x", length=0, pad=1.5)
    ax_w.set_ylabel("present value (M USD)", fontsize=7.6)
    ax_w.set_title("From gross benefit to net present value", fontsize=8.4,
                   fontweight="bold", pad=4.0)

    # ---- aux (c): tornado over the appraisal assumptions --------------------
    swings = [
        ("Discount rate  2 - 8 %",
         dict(rate=0.02), dict(rate=0.08)),
        ("Damage valuation  -35 / +35 %",
         dict(damages=0.65), dict(damages=1.35)),
        ("Capital outlay  -30 / +30 %",
         dict(capital=0.70), dict(capital=1.30)),
        ("Benefit ramp  fast / slow",
         dict(ramp=0.60), dict(ramp=1.60)),
        ("Programme take-up  100 / 75 %",
         dict(take_up=1.00), dict(take_up=0.75)),
        ("Ongoing compliance  -25 / +25 %",
         dict(ongoing=0.75), dict(ongoing=1.25)),
    ]
    rows = []
    for label, kw_hi, kw_lo in swings:
        a = appraise(**kw_hi)[3][-1]
        b = appraise(**kw_lo)[3][-1]
        rows.append((label, float(min(a, b)), float(max(a, b))))
    rows.sort(key=lambda r: r[2] - r[1])

    for k, (label, lo, hi) in enumerate(rows):
        ax_s.barh(k, npv_end - lo, left=lo, height=0.58,
                  color=lighten(RUST, 0.40), edgecolor=RUST, linewidth=0.8,
                  zorder=4)
        ax_s.barh(k, hi - npv_end, left=npv_end, height=0.58,
                  color=lighten(TEAL, 0.40), edgecolor=TEAL, linewidth=0.8,
                  zorder=4)
        ax_s.text(lo - 40.0, k, f"{lo:+,.0f}", ha="right", va="center",
                  fontsize=6, color=darken(RUST, 0.12), zorder=6)
        ax_s.text(hi + 40.0, k, f"{hi:+,.0f}", ha="left", va="center",
                  fontsize=6, color=darken(TEAL, 0.12), zorder=6)
    ax_s.axvline(0.0, color=FAINT, lw=0.9, zorder=3)
    ax_s.axvline(npv_end, color=DEEP, lw=1.1, ls=(0, (3.0, 2.2)), zorder=5)
    ax_s.set_yticks(range(len(rows)))
    ax_s.set_yticklabels([r[0] for r in rows], fontsize=6.0)
    ax_s.set_ylim(-0.7, len(rows) + 0.35)
    lo_all = min(r[1] for r in rows)
    hi_all = max(r[2] for r in rows)
    pad = 0.30 * (hi_all - lo_all)
    ax_s.set_xlim(lo_all - pad, hi_all + pad)
    ax_s.tick_params(axis="x", labelsize=6.6)
    ax_s.set_xlabel("net present value at year 20 (M USD)", fontsize=7.6,
                    labelpad=1.5)
    ax_s.set_title("What could overturn the conclusion", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_s.text(hi_all + pad * 0.9, len(rows) + 0.22,
              f"dashed rule is the base case, {npv_end:+,.0f} M USD; "
              f"only the top two swings can push it below zero",
              ha="right", va="bottom", fontsize=6, color=SUB)

    panel_labels_inches(fig, [ax, ax_w, ax_s], dx=-0.74, dy=0.12)
    savefig(fig, "928_pcy_regulatory_impact.png")


if __name__ == "__main__":
    main()

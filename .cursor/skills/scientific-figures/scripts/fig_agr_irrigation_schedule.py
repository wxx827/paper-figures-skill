"""Figure 398 - Season-long irrigation schedule for an irrigated wheat block.

Three stacked panels on one time axis: the root-zone water content against the
field-capacity / refill / wilting-point references, the water applied (rising
irrigation bars, hanging rainfall bars), and crop water demand against what the
crop actually transpired.  Everything comes out of one daily bucket balance, so
the shaded stress windows in all three panels are the same days.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, add_panel_label

SEED = 3181
NDAY = 151

FC, WP, P_DEPL = 0.320, 0.140, 0.55        # m3/m3, m3/m3, depletion fraction
THETA_REFILL = FC - P_DEPL * (FC - WP)

# Unified growth-stage palette for the whole agriculture batch.
STAGES = [("Sowing", 0, 8, "#8a6b4f"),
          ("Emergence", 8, 30, "#9ccfae"),
          ("Jointing", 30, 62, "#6bbf8a"),
          ("Heading", 62, 88, "#2e7d5b"),
          ("Grain filling", 88, 125, "#c9a227"),
          ("Maturity", 125, NDAY - 1, "#c0603f")]

C_THETA = "#3b5b92"
C_IRR = "#2e7d5b"
C_RAIN = "#3d6b8e"
C_DEMAND = "#c0603f"
C_ACTUAL = "#2e7d5b"
C_STRESS = "#b23a48"
PUMP_OUT = (79, 99)                        # pump failure window, days


def runs(mask, min_len=2):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        out.append((start, len(mask)))
    return out


def kc_curve(d):
    return np.interp(d, [0, 8, 26, 66, 106, 130, 150],
                     [0.28, 0.30, 0.34, 1.16, 1.14, 0.72, 0.38])


def simulate(rng):
    d = np.arange(NDAY)
    et0 = np.clip(3.2 + 2.7 * np.sin(np.pi * (d + 18) / 205.0)
                  + 0.55 * rng.standard_normal(NDAY), 1.4, 8.4)
    etc = kc_curve(d) * et0

    rain = np.zeros(NDAY)
    wet = rng.random(NDAY) < 0.11
    rain[wet] = rng.exponential(7.6, wet.sum())
    rain[rain < 0.6] = 0.0

    zr = 0.25 + 0.75 * np.clip(d / 70.0, 0.0, 1.0)
    taw = (FC - WP) * zr * 1000.0
    raw = P_DEPL * taw

    depl = np.zeros(NDAY)
    theta = np.zeros(NDAY)
    irr = np.zeros(NDAY)
    eta = np.zeros(NDAY)
    ks = np.zeros(NDAY)
    dep = 18.0
    for i in range(NDAY):
        ks[i] = 1.0 if dep <= raw[i] else max(
            0.0, (taw[i] - dep) / max(taw[i] - raw[i], 1e-6))
        eta[i] = etc[i] * ks[i]
        if (dep > raw[i] and not (PUMP_OUT[0] <= i < PUMP_OUT[1])
                and i > 12 and i < 132):
            irr[i] = min(dep, 34.0)
        dep = float(np.clip(dep - rain[i] - irr[i] + eta[i], 0.0, taw[i]))
        depl[i] = dep
        theta[i] = FC - dep / (zr[i] * 1000.0)
    return dict(d=d, et0=et0, etc=etc, eta=eta, ks=ks, rain=rain, irr=irr,
                theta=theta, taw=taw, raw=raw)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)
    s = simulate(rng)
    d = s["d"]
    stress = runs(s["ks"] < 0.995, min_len=3)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.115, 1.0, 0.62, 0.72],
                          left=0.075, right=0.905, top=0.885, bottom=0.105,
                          hspace=0.18)
    ax_stg = fig.add_subplot(gs[0])
    ax_th = fig.add_subplot(gs[1], sharex=ax_stg)
    ax_w = fig.add_subplot(gs[2], sharex=ax_stg)
    ax_et = fig.add_subplot(gs[3], sharex=ax_stg)

    # ---- growth-stage strip ------------------------------------------------
    for name, d0, d1, col in STAGES:
        ax_stg.axvspan(d0, d1, facecolor=col, edgecolor="white", lw=0.7)
        if d1 - d0 > 14:
            ax_stg.text(0.5 * (d0 + d1), 0.5, name, ha="center", va="center",
                        fontsize=6.4,
                        color="white" if col in ("#8a6b4f", "#2e7d5b")
                        else "#1a1a1a")
    ax_stg.set_yticks([])
    ax_stg.set_ylim(0, 1)
    for sp in ax_stg.spines.values():
        sp.set_visible(False)
    ax_stg.tick_params(axis="x", length=0)
    ax_stg.text(-0.012, 0.5, "stage", transform=ax_stg.transAxes, ha="right",
                va="center", fontsize=7.4, color="#555555")

    # ---- (a) root-zone water content --------------------------------------
    th = 100.0 * s["theta"]
    ax_th.axhspan(100 * THETA_REFILL, 100 * FC, color="#a4d7b8", alpha=0.35,
                  zorder=0)
    ax_th.axhspan(100 * WP, 100 * THETA_REFILL, color="#c9a227", alpha=0.20,
                  zorder=0)
    ax_th.axhspan(9.5, 100 * WP, color=C_STRESS, alpha=0.13, zorder=0)
    for y, lab, col in [(100 * FC, "field capacity", "#1f6a47"),
                        (100 * THETA_REFILL, "refill point (55% depletion)",
                         "#8a6b1f"),
                        (100 * WP, "permanent wilting point", "#8c2f2f")]:
        ax_th.axhline(y, color=col, lw=0.9, ls=(0, (4.0, 2.2)), zorder=2)
        ax_th.text(1.8, y + 0.28, f"{lab}  {y:.1f}%", ha="left",
                   va="bottom", fontsize=6.4, color=col)
    ax_th.plot(d, th, color=C_THETA, lw=1.7, zorder=4)
    ax_th.set_ylim(9.5, 35.0)
    ax_th.set_ylabel("root-zone water\ncontent (% vol.)", fontsize=8.0)

    # ---- (b) water applied -------------------------------------------------
    ax_w.bar(d, s["irr"], width=1.5, color=C_IRR, zorder=3)
    ax_w.set_ylim(0, 92.0)
    ax_w.set_yticks([0, 15, 30])
    ax_w.set_ylabel("irrigation\n(mm)", fontsize=8.0, color=C_IRR)
    ax_w.tick_params(axis="y", colors=C_IRR)

    ax_r = ax_w.twinx()
    ax_r.bar(d, s["rain"], width=1.5, color=C_RAIN, zorder=3)
    ax_r.set_ylim(92.0, 0)
    ax_r.set_yticks([0, 15, 30])
    ax_r.set_ylabel("rainfall\n(mm)", fontsize=8.0, color=C_RAIN, rotation=270,
                    labelpad=22, va="center")
    ax_r.tick_params(axis="y", colors=C_RAIN)
    ax_r.spines["right"].set_visible(True)
    ax_r.spines["right"].set_color(C_RAIN)
    ax_r.spines["top"].set_visible(False)

    n_irr = int((s["irr"] > 0).sum())
    ax_w.text(0.012, 0.50,
              f"{n_irr} irrigations, {s['irr'].sum():.0f} mm applied   |   "
              f"rainfall {s['rain'].sum():.0f} mm",
              transform=ax_w.transAxes, ha="left", va="center", fontsize=6.6,
              color="#555555")

    # ---- (c) crop water demand vs actual ----------------------------------
    ax_et.fill_between(d, s["eta"], s["etc"], color=C_STRESS, alpha=0.30,
                       lw=0, zorder=2)
    ax_et.plot(d, s["etc"], color=C_DEMAND, lw=1.5, zorder=4)
    ax_et.plot(d, s["eta"], color=C_ACTUAL, lw=1.4, zorder=5)
    ax_et.set_ylim(0, max(s["etc"]) * 1.30)
    ax_et.set_ylabel("water flux\n(mm/d)", fontsize=8.0)
    ax_et.set_xlabel("days after sowing (d)", fontsize=8.0)

    gap = float((s["etc"] - s["eta"]).sum())
    ax_et.text(0.012, 0.94,
               f"seasonal ET$_c$ {s['etc'].sum():.0f} mm, "
               f"ET$_a$ {s['eta'].sum():.0f} mm  \u2192  "
               f"{gap:.0f} mm unmet ({gap / s['etc'].sum():.0%})",
               transform=ax_et.transAxes, ha="left", va="top", fontsize=6.6,
               color="#555555")

    # ---- shared stress windows --------------------------------------------
    for a, b in stress:
        for ax in (ax_stg, ax_th, ax_w, ax_et):
            ax.axvspan(a, b, color=C_STRESS, alpha=0.09, lw=0, zorder=0)
    if stress:
        a, b = max(stress, key=lambda t: t[1] - t[0])
        ax_th.annotate(f"pump outage,\n{b - a} d of water stress",
                       xy=(0.5 * (a + b), th[(a + b) // 2]),
                       xytext=(0.5 * (a + b) - 30, 10.5), fontsize=6.8,
                       color=C_STRESS, ha="center", va="bottom",
                       linespacing=1.4,
                       arrowprops=dict(arrowstyle="-|>", lw=0.8,
                                       color=C_STRESS, shrinkA=2, shrinkB=3))

    for ax in (ax_th, ax_w, ax_et):
        ax.set_xlim(0, NDAY - 1)
        ax.tick_params(labelsize=7.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", lw=0.5, alpha=0.45)
    ax_w.spines["right"].set_visible(False)
    for ax in (ax_stg, ax_th, ax_w):
        plt.setp(ax.get_xticklabels(), visible=False)
    ax_et.set_xticks(np.arange(0, NDAY, 25))

    add_panel_label(ax_th, "(a)", x=-0.062, y=1.02)
    add_panel_label(ax_w, "(b)", x=-0.062, y=1.02)
    add_panel_label(ax_et, "(c)", x=-0.062, y=1.02)

    handles = [Line2D([0], [0], color=C_THETA, lw=1.7,
                      label="root-zone water content"),
               Patch(facecolor=C_IRR, label="irrigation applied"),
               Patch(facecolor=C_RAIN, label="rainfall"),
               Line2D([0], [0], color=C_DEMAND, lw=1.5,
                      label="crop demand ET$_c$"),
               Line2D([0], [0], color=C_ACTUAL, lw=1.4,
                      label="actual ET$_a$"),
               Patch(facecolor=C_STRESS, alpha=0.30,
                     label="unmet demand / stress window")]


    savefig(fig, "398_agr_irrigation_schedule.png")


if __name__ == "__main__":
    main()

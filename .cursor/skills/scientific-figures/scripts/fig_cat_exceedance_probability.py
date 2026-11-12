"""Figure 722 - Exceedance probability curves for a catastrophe portfolio.

A Poisson-lognormal event set is simulated for 200 000 synthetic years: the
event count each year is Poisson, each event draws a ground-up loss from a
heavy-tailed lognormal, and the year is summarised twice - by its largest
single event (occurrence exceedance probability) and by its total
(aggregate exceedance probability).  Panel (a) plots both empirical curves on
log-log axes and reads the 10, 50, 100 and 250-year return periods straight
off them.

Panel (b) is the severity distribution behind those curves, with a generalised
Pareto tail fitted above the 90th percentile of event loss.  Panel (c) turns
the same simulation into the numbers an underwriter quotes: probable maximum
loss at two return periods and tail value at risk at two confidence levels,
all against the average annual loss.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto

from pubstyle import set_pub_style, savefig, lighten, darken, COLORS
from figkit import axes_inches, panel_label_inches

FW, FH = 10.4, 6.4
MAIN_X, MAIN_W, MAIN_TOP = 0.68, 5.30, 5.50
AUX_X, AUX_W, AUX_GAP = 7.05, 2.95, 0.66

ACCENT = "#b23a48"
OEP_COL = darken(ACCENT, 0.42)
AEP_COL = ACCENT

N_YEARS = 200_000
LAMBDA = 3.2                        # events per year
LN_MU, LN_SIG = 2.55, 1.58          # lognormal severity in log(USD mn)
RETURN_PERIODS = [10.0, 50.0, 100.0, 250.0]
TAIL_Q = 0.90                       # GPD threshold quantile


def ep_curve(losses):
    """Empirical exceedance probability: sorted losses and their P(L > x)."""
    s = np.sort(losses)[::-1]
    p = (np.arange(1, s.size + 1)) / float(s.size)
    keep = s > 0.0
    return s[keep], p[keep]


def at_return_period(s, p, period):
    """Loss whose annual exceedance probability equals 1 / period."""
    # ``p`` is ascending and ``s`` descending; np.interp only needs the
    # abscissa to increase, so feed it log(p) and let log(s) fall.
    return float(np.exp(np.interp(np.log(1.0 / period), np.log(p),
                                  np.log(s))))


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(7220)

    counts = rng.poisson(LAMBDA, N_YEARS)
    total = int(counts.sum())
    sev = rng.lognormal(LN_MU, LN_SIG, total)          # USD mn per event
    edge = np.concatenate([[0], np.cumsum(counts)])
    oep = np.zeros(N_YEARS)
    aep = np.zeros(N_YEARS)
    for i in range(N_YEARS):
        a, b = edge[i], edge[i + 1]
        if b > a:
            chunk = sev[a:b]
            oep[i] = chunk.max()
            aep[i] = chunk.sum()

    so, po = ep_curve(oep)
    sa, pa = ep_curve(aep)
    aal = float(aep.mean())

    rp_oep = [at_return_period(so, po, T) for T in RETURN_PERIODS]
    rp_aep = [at_return_period(sa, pa, T) for T in RETURN_PERIODS]

    tvar = {}
    for q in (0.99, 0.996):
        var = float(np.quantile(aep, q))
        tvar[q] = float(aep[aep >= var].mean())

    # ---- figure -------------------------------------------------------------
    fig = plt.figure(figsize=(FW, FH))
    main_h = 4.70
    main_y = MAIN_TOP - main_h
    ax = axes_inches(fig, (MAIN_X, main_y, MAIN_W, main_h))
    aux_h = (main_h - AUX_GAP) / 2.0
    ax_t = axes_inches(fig, (AUX_X, main_y + aux_h + AUX_GAP, AUX_W, aux_h))
    ax_m = axes_inches(fig, (AUX_X, main_y, AUX_W, aux_h))

    # ---- (a) the EP curves --------------------------------------------------
    xlo, xhi = 1.0, 3.0e4
    ylo, yhi = 6.0e-5, 1.0
    for T in RETURN_PERIODS:
        ax.axhline(1.0 / T, color="#cccccc", lw=0.7, ls=(0, (3.2, 2.2)),
                   zorder=1)
        ax.text(xhi * 0.94, 1.0 / T * 1.09, f"{T:.0f} yr", fontsize=6.4,
                color=COLORS["muted"], ha="right", va="bottom")

    idx = np.unique(np.geomspace(1, so.size, 1400).astype(int) - 1)
    ax.plot(sa[idx], pa[idx], color=AEP_COL, lw=1.9, zorder=5,
            label="AEP - annual aggregate loss")
    ax.plot(so[idx], po[idx], color=OEP_COL, lw=1.9, ls=(0, (5.0, 2.0)),
            zorder=5, label="OEP - largest event in the year")
    for T, lo_, la_ in zip(RETURN_PERIODS, rp_oep, rp_aep):
        ax.plot([lo_], [1.0 / T], marker="o", ms=4.0, mfc="white",
                mec=OEP_COL, mew=1.3, zorder=7)
        ax.plot([la_], [1.0 / T], marker="o", ms=4.0, mfc="white",
                mec=AEP_COL, mew=1.3, zorder=7)
    ax.axvline(aal, color="#808080", lw=0.9, ls=(0, (1.8, 1.8)), zorder=2)
    ax.text(aal * 1.10, ylo * 1.35, f"AAL = {aal:,.0f} USD mn", fontsize=6.4,
            color=COLORS["sub"], ha="left", va="bottom", rotation=90)

    rows = ["  T (yr)      OEP      AEP"]
    rows += [f"  {T:6.0f}  {lo_:7,.0f}  {la_:7,.0f}"
             for T, lo_, la_ in zip(RETURN_PERIODS, rp_oep, rp_aep)]
    ax.text(0.035, 0.055, "\n".join(rows), transform=ax.transAxes,
            fontsize=6.4, family="DejaVu Sans Mono", color="#1a1a1a",
            ha="left", va="bottom", linespacing=1.5, zorder=8,
            bbox=dict(boxstyle="round,pad=0.42", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.94))
    ax.text(0.035, 0.238, "loss at return period, USD mn",
            transform=ax.transAxes, fontsize=6.3, color=COLORS["sub"],
            ha="left", va="bottom", zorder=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["1", "10", "100", "1 000", "10 000"])
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    ax.set_yticklabels(["0.0001", "0.001", "0.01", "0.1", "1"])
    ax.set_xlabel("loss (USD mn)", fontsize=8.0)
    ax.set_ylabel("annual exceedance probability $P_{exc}$ (-)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.grid(which="major", lw=0.5, alpha=0.42)
    ax.set_title("Occurrence and aggregate exceedance probability",
                 fontsize=8.8, fontweight="bold", pad=4.0)
    ax.legend(loc="upper right", frameon=False, fontsize=6.8,
              handlelength=2.0, labelspacing=0.32)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ---- (b) severity tail fit ---------------------------------------------
    u = float(np.quantile(sev, TAIL_Q))
    exc = sev[sev > u] - u
    xi, _, beta_s = genpareto.fit(exc, floc=0.0)
    ss, ps = ep_curve(sev)
    jdx = np.unique(np.geomspace(1, ss.size, 700).astype(int) - 1)
    ax_t.plot(ss[jdx], ps[jdx], color="#8a8a8a", lw=1.2, zorder=3,
              label="empirical event severity")
    xt = np.geomspace(u, ss.max() * 1.2, 200)
    ax_t.plot(xt, (1.0 - TAIL_Q) * genpareto.sf(xt - u, xi, scale=beta_s),
              color=ACCENT, lw=1.6, zorder=5,
              label=f"GPD tail, $\\xi$ = {xi:.2f}")
    ax_t.axvline(u, color="#333333", lw=0.8, ls=(0, (2.6, 1.8)), zorder=4)
    ax_t.text(u * 1.15, 0.55, f"threshold\n{u:,.0f} USD mn", fontsize=6.2,
              color="#333333", ha="left", va="top")
    ax_t.set_xscale("log")
    ax_t.set_yscale("log")
    ax_t.set_xlim(xlo, xhi)               # same money axis as panel (a)
    ax_t.set_ylim(2.0e-6, 1.4)
    ax_t.set_xticks([1, 10, 100, 1000, 10000])
    ax_t.set_xlabel("event loss (USD mn)", fontsize=7.4)
    ax_t.set_ylabel("$P(L > x)$ (-)", fontsize=7.2)
    ax_t.tick_params(labelsize=6.6)
    ax_t.grid(which="major", lw=0.5, alpha=0.42)
    ax_t.legend(loc="lower left", frameon=False, fontsize=6.3,
                handlelength=1.7, labelspacing=0.26)
    ax_t.set_title("Severity distribution and its tail", fontsize=8.5,
                   fontweight="bold", pad=3.5)
    for sp in ("top", "right"):
        ax_t.spines[sp].set_visible(False)

    # ---- (c) PML and TVaR metrics ------------------------------------------
    metrics = [
        ("$PML_{100}$ (OEP)", rp_oep[2]),
        ("$PML_{250}$ (OEP)", rp_oep[3]),
        ("AEP 100 yr", rp_aep[2]),
        ("AEP 250 yr", rp_aep[3]),
        ("$TVaR_{99\\%}$", tvar[0.99]),
        ("$TVaR_{99.6\\%}$", tvar[0.996]),
    ]
    metrics.sort(key=lambda m: m[1])
    vals = np.array([m[1] for m in metrics])
    shades = [lighten(ACCENT, f) for f in
              np.linspace(0.58, 0.0, len(metrics))]
    y = np.arange(len(metrics), dtype=float)
    ax_m.barh(y, vals, height=0.62, color=shades, edgecolor="white",
              linewidth=0.6, zorder=3)
    for yy, v in zip(y, vals):
        ax_m.text(v + vals.max() * 0.018, yy, f"{v:,.0f}", fontsize=6.4,
                  color="#1a1a1a", ha="left", va="center", zorder=4)
    ax_m.axvline(aal, color="#808080", lw=1.0, ls=(0, (2.4, 1.8)), zorder=5)
    ax_m.text(0.98, 0.03, f"dashed line: AAL {aal:,.0f} USD mn",
              transform=ax_m.transAxes, fontsize=6.3, color=COLORS["sub"],
              ha="right", va="bottom")
    ax_m.set_yticks(y)
    ax_m.set_yticklabels([m[0] for m in metrics], fontsize=6.8)
    ax_m.set_xlim(0.0, vals.max() * 1.24)
    ax_m.set_ylim(-0.68, len(metrics) - 0.32)
    ax_m.set_xlabel("loss (USD mn)", fontsize=7.4)
    ax_m.tick_params(labelsize=6.6)
    ax_m.grid(axis="x", lw=0.5, alpha=0.42)
    ax_m.set_title("Capital metrics from the same simulation", fontsize=8.5,
                   fontweight="bold", pad=3.5)
    for sp in ("top", "right"):
        ax_m.spines[sp].set_visible(False)

    for a, lab in ((ax, "(a)"), (ax_t, "(b)"), (ax_m, "(c)")):
        panel_label_inches(fig, a, lab, dx=-0.54, dy=0.09)


    savefig(fig, "722_cat_exceedance_probability.png")


if __name__ == "__main__":
    main()

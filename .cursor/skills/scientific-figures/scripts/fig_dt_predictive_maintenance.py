"""Figure 668 - Predictive maintenance from a fleet health index.

Main panel: five units of the same machine class degrade along a power law.
Each trace is solid where it has been measured and dashed where the twin
extrapolates it; the worst unit carries a forecast interval that widens with
horizon, and the remaining useful life is the gap between today and that
interval's crossing of the failure threshold.

Right column: (b) how quickly the RUL estimate settles as more of the life is
observed - the same fit is re-run at every possible decision date over thirty
fleet histories, so the convergence is measured rather than drawn; (c) what the
resulting policy costs against the two conventional alternatives.

The forecast interval comes from the covariance of the log-log degradation fit,
propagated by sampling, so the band, the point estimate and the printed RUL are
all one calculation (DESIGN_SYSTEM section 13.2).
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, lighten, darken
from figkit import axes_inches, panel_label_inches

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#cccccc"
GRID, BASE = "#d9d9d9", "#808080"

HEALTHY = "#3b5b92"    # house blue - units with headroom
CRIT = "#b23a48"       # house red  - the single emphasis: the worst unit
GOLD = "#c9a227"
GREEN = "#2e7d5b"

UNITS = ["U-02", "U-05", "U-11", "U-14", "U-19"]
LIFE = np.array([5900.0, 6800.0, 7700.0, 8500.0, 9400.0])   # true failure, h
T_NOW = 4200.0         # fleet inspection date, operating hours
THRESH = 30.0          # health index at which a unit is declared failed
H0 = 100.0             # health index of a new unit
EXPO = 1.70            # degradation exponent
MIN_DROP = 8.0         # a fit needs this much observed degradation to be sane

POLICIES = ["Run to failure", "Calendar-based", "Predictive (twin)"]
COMPONENTS = ["Unplanned downtime", "Spare parts", "Labour"]
COST_KUSD = np.array([[430.0, 175.0, 92.0],
                      [128.0, 262.0, 178.0],
                      [58.0, 171.0, 124.0]])


def health(t, life):
    """Power-law health index; equals THRESH exactly at ``life``."""
    return H0 - (H0 - THRESH) * (np.maximum(t, 1.0) / life) ** EXPO


def fit_line(t, hi):
    """Least squares in log-log space; ``None`` if the record cannot carry it.

    Very early records have barely left 100 %, so ``log(H0 - hi)`` is dominated
    by measurement noise and the recovered exponent runs away.  Requiring a
    minimum observed drop and a physically plausible exponent turns those cases
    into an honest "no estimate" instead of a spike worth 10^22 hours.
    """
    ok = (hi < H0 - MIN_DROP) & (t > 1.0)
    if ok.sum() < 8:
        return None
    y = np.log(np.maximum(H0 - hi[ok], 1e-3))
    x = np.log(t[ok])
    p, c = np.polyfit(x, y, 1)
    if not (0.6 <= p <= 3.5):
        return None
    return float(p), float(c)


def life_of(p, c):
    return float(np.exp((np.log(H0 - THRESH) - c) / p))


def fit_life(t, hi):
    r = fit_line(t, hi)
    if r is None:
        return np.nan
    life = life_of(r[0], r[1])
    return life if (t.max() < life < 40000.0) else np.nan


def movavg(a, k):
    """Moving average that keeps the endpoints, unlike ``mode='same'``."""
    if k <= 1:
        return a
    pad = k // 2
    return np.convolve(np.pad(a, pad, mode="edge"),
                       np.ones(k) / k, mode="valid")[:a.size]


def make_record(rng, life, hours):
    hi = health(hours, life) + rng.normal(0.0, 0.55, hours.size)
    walk = np.cumsum(rng.normal(0.0, 0.055, hours.size))
    return hi + walk - walk.mean()


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(668)

    hours = np.arange(0.0, 9800.0, 25.0)
    obs_m = hours <= T_NOW
    shades = [lighten(HEALTHY, f) for f in (0.10, 0.28, 0.44, 0.58)]

    records = [make_record(rng, life, hours) for life in LIFE]
    fits = np.array([fit_life(hours[obs_m], hi[obs_m]) for hi in records])

    worst = int(np.nanargmin(fits))
    colors = [CRIT if i == worst else shades[min(i, 3)]
              for i in range(len(UNITS))]

    # ---- convergence study, also the source of the forecast interval -------
    # Re-running the same estimator over many independent histories gives a
    # *calibrated* spread: at any decision date the empirical error quantiles
    # say how wrong this estimator actually is.  That is a far more honest
    # band than the least-squares covariance, which treats a random-walk
    # degradation signal as independent samples and comes out ten times too
    # confident.
    decide = np.arange(1800.0, 5601.0, 200.0)
    n_rep = 6
    histories = [(life, make_record(rng, life, hours))
                 for life in LIFE for _ in range(n_rep)]

    signed = {}
    for name, k in (("physics twin only", 1), ("twin + residual learner", 11)):
        err = np.full((len(histories), decide.size), np.nan)
        for i, (life, hi) in enumerate(histories):
            hs = movavg(hi, k)
            for j, tp in enumerate(decide):
                m = (hours <= tp) & (hours > 0.0)
                lf = fit_life(hours[m], hs[m])
                if np.isfinite(lf):
                    err[i, j] = lf - life
        signed[name] = err

    j_now = int(np.argmin(np.abs(decide - T_NOW)))
    e_now = signed["physics twin only"][:, j_now]
    q10, q90 = np.nanpercentile(e_now, [10, 90])
    base_life = fits[worst]
    lo_life, hi_life = base_life - q90, base_life - q10
    rul = base_life - T_NOW
    rul_lo, rul_hi = lo_life - T_NOW, hi_life - T_NOW

    fig = plt.figure(figsize=(10.4, 6.4))

    # ---- (a) fleet degradation and the forecast ----------------------------
    axm = axes_inches(fig, (0.86, 0.76, 5.22, 4.66))
    axm.axvspan(T_NOW, hours[-1], facecolor="#f5f5f5", edgecolor="none",
                zorder=0)
    tb = np.linspace(T_NOW, hi_life, 300)
    axm.fill_between(tb, health(tb, lo_life), health(tb, hi_life),
                     facecolor=lighten(CRIT, 0.76), edgecolor="none",
                     zorder=1, label="80 % forecast interval, worst unit")

    for i, (name, hi, col, lf) in enumerate(zip(UNITS, records, colors, fits)):
        axm.plot(hours[obs_m], hi[obs_m], lw=1.7 if i == worst else 1.1,
                 color=col, zorder=5 if i == worst else 3)
        # Stop each forecast at its own threshold crossing: past that the
        # health index has no meaning and the lines would just cross-hatch.
        tf = np.linspace(T_NOW, lf, 200)
        axm.plot(tf, health(tf, lf), lw=1.5 if i == worst else 1.0,
                 color=col, ls=(0, (4.0, 2.2)),
                 zorder=5 if i == worst else 3)
        axm.plot([T_NOW], [hi[obs_m][-1]], marker="o", ms=4.0, mfc="white",
                 mec=col, mew=1.1, zorder=6)
        axm.text(lf, THRESH + 2.2, name, ha="center", va="bottom",
                 fontsize=6.8, color=darken(col, 0.18), zorder=7,
                 fontweight="bold" if i == worst else "normal")

    axm.axhline(THRESH, color=CRIT, lw=1.1, ls=(0, (5.0, 2.4)), zorder=4)
    axm.text(180.0, THRESH + 1.8, f"failure threshold {THRESH:.0f} %",
             ha="left", va="bottom", fontsize=6.8, color=CRIT)
    axm.axvline(T_NOW, color=BASE, lw=1.0, ls=(0, (2.6, 2.0)), zorder=4)
    axm.text(T_NOW - 140.0, 21.0, "today", ha="right", va="bottom",
             fontsize=6.8, color=SUB)

    y_r = THRESH - 6.0
    axm.annotate("", xy=(base_life, y_r), xytext=(T_NOW, y_r),
                 arrowprops=dict(arrowstyle="<|-|>", lw=1.0, color=CRIT,
                                 shrinkA=0, shrinkB=0), zorder=7)
    axm.text((T_NOW + base_life) / 2.0, y_r - 2.4,
             f"RUL {rul:.0f} h  [{rul_lo:.0f}, {rul_hi:.0f}]", ha="center",
             va="top", fontsize=7.0, color=CRIT, fontweight="bold")

    axm.set_xlim(0.0, hours[-1])
    axm.set_ylim(16.0, 104.0)
    axm.set_xticks(np.arange(0, 10000, 2000))
    axm.set_xlabel("operating hours since overhaul (hours)", fontsize=8.0)
    axm.set_ylabel("health index (%)", fontsize=8.0)
    axm.tick_params(labelsize=7.0, length=2.6)
    axm.grid(axis="y", color=GRID, lw=0.6)
    axm.plot([], [], lw=1.4, color=HEALTHY, label="measured health index")
    axm.plot([], [], lw=1.2, color=HEALTHY, ls=(0, (4.0, 2.2)),
             label="twin extrapolation")
    axm.legend(loc="upper right", fontsize=7.0, frameon=False,
               handlelength=2.0, labelspacing=0.3)
    axm.set_title(f"Fleet of {len(UNITS)} units, one inspection date",
                  fontsize=9.0, fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        axm.spines[s].set_visible(False)

    # ---- (b) how fast the RUL estimate settles -----------------------------
    axc = axes_inches(fig, (7.02, 3.36, 3.02, 2.06))
    for (name, raw), col in zip(signed.items(), (BASE, GREEN)):
        err = np.abs(raw)
        med = np.nanmedian(err, axis=0)
        q1, q3 = np.nanpercentile(err, [25, 75], axis=0)
        axc.fill_between(decide, q1, q3, facecolor=lighten(col, 0.76),
                         edgecolor="none", zorder=2)
        axc.plot(decide, med, lw=1.5, color=col, zorder=3, label=name)
    axc.axvline(T_NOW, color=BASE, lw=1.0, ls=(0, (2.6, 2.0)), zorder=4)
    axc.text(T_NOW - 90.0, 55.0, "today", ha="right", va="bottom",
             fontsize=6.4, color=SUB)
    tol = 250.0
    axc.axhline(tol, color=CRIT, lw=0.9, ls=(0, (3.0, 2.0)), zorder=4)
    axc.text(decide[-1], tol * 1.16, f"{tol:.0f} h tolerance", ha="right",
             va="bottom", fontsize=6.4, color=CRIT)
    axc.set_xlim(decide[0], decide[-1])
    # The error falls through two orders of magnitude; on a linear axis the
    # early blow-up would flatten the whole converged tail into the baseline.
    axc.set_yscale("log")
    axc.set_ylim(40.0, 14000.0)
    axc.set_xlabel("decision date (operating hours)", fontsize=7.4)
    axc.set_ylabel("absolute RUL error (hours)", fontsize=7.4)
    axc.tick_params(labelsize=6.5, length=2.2)
    axc.grid(axis="y", color=GRID, lw=0.6)
    axc.legend(loc="upper right", fontsize=6.5, frameon=False,
               handlelength=1.4, labelspacing=0.25)
    axc.set_title(f"RUL error, median and IQR over "
                  f"{len(histories)} histories", fontsize=8.5,
                  fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)

    # ---- (c) what each maintenance policy costs ----------------------------
    axb = axes_inches(fig, (7.02, 0.76, 3.02, 1.94))
    cost = COST_KUSD * np.exp(rng.normal(0.0, 0.035, COST_KUSD.shape))
    totals = cost.sum(axis=1)
    ypos = np.arange(len(POLICIES))[::-1]
    left = np.zeros(len(POLICIES))
    for k, (comp, col) in enumerate(zip(COMPONENTS, [CRIT, HEALTHY, GOLD])):
        axb.barh(ypos, cost[:, k], left=left, height=0.56, color=col,
                 edgecolor="white", linewidth=0.7, label=comp, zorder=3)
        left += cost[:, k]
    for y, tot in zip(ypos, totals):
        axb.text(tot + 12.0, y, f"{tot:.0f}", ha="left", va="center",
                 fontsize=6.8, color=INK)
    saving = 100.0 * (1.0 - totals[2] / totals[:2].min())
    axb.text(totals.max() * 1.16, 0.52,
             f"{saving:.0f} % below the best alternative", ha="right",
             va="center", fontsize=6.6, color=GREEN)
    axb.set_yticks(ypos)
    axb.set_yticklabels(POLICIES)
    axb.set_xlim(0.0, totals.max() * 1.20)
    axb.set_xlabel("annual cost (k$ per unit per year)", fontsize=7.4)
    axb.tick_params(labelsize=6.5, length=2.2)
    axb.grid(axis="x", color=GRID, lw=0.6)
    axb.legend(loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=3,
               fontsize=6.4, frameon=False, handlelength=1.1,
               columnspacing=0.9, handletextpad=0.4)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)

    for ax, lab in ((axm, "(a)"), (axc, "(b)"), (axb, "(c)")):
        panel_label_inches(fig, ax, lab, dx=-0.62, dy=0.12)
    savefig(fig, "668_dt_predictive_maintenance.png")


if __name__ == "__main__":
    main()

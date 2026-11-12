"""Ten-year cardiovascular risk nomogram with its own validation panels.

A nomogram is a printed calculator, so the geometry *is* the model: each
predictor ruler is its regression coefficient rendered as length, which is why
age is the longest ruler and diabetes the shortest.  Points are scaled so the
widest predictor spans exactly 100, the total-points ruler is linear, and the
risk ruler underneath it is not - it is the logistic link, which is what makes
the same 20 points worth far more risk at the right-hand end than at the left.

The worked example is a real traverse of the model: the dashed path reads each
value up to the Points ruler, the points are summed in code, and the risk
quoted at the bottom comes from evaluating the same logistic function at that
total.  The two auxiliary panels are computed from a simulated cohort scored
by the same coefficients, so the calibration curve, the Brier score and every
C-index bar move together if a coefficient changes.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

HALO = [pe.withStroke(linewidth=2.0, foreground="white")]

INK = "#1a1a1a"
AXIS_C = "#333333"
SUB = "#555555"
MUTED = "#777777"
GRIDC = "#e2e2e2"
PATH_C = "#b23a48"          # the worked example, the single emphasis object
RISK_C = "#3b5b92"

BETA = {"age": 0.062, "sbp": 0.018, "chol": 0.240, "smoke": 0.620,
        "diab": 0.750}
XMIN = {"age": 40.0, "sbp": 100.0, "chol": 3.5, "smoke": 0.0, "diab": 0.0}
XMAX = {"age": 80.0, "sbp": 190.0, "chol": 8.0, "smoke": 1.0, "diab": 1.0}

SPAN = {k: BETA[k] * (XMAX[k] - XMIN[k]) for k in BETA}
SCALE = 100.0 / max(SPAN.values())          # points per unit linear predictor
LP_MIN = sum(BETA[k] * XMIN[k] for k in BETA)
LP_REF = 9.72                               # intercept: 1% risk at zero points
TP_MAX = 270.0

PATIENT = [("Age (years)", "age", 64.0, "64 y"),
           ("Systolic BP (mmHg)", "sbp", 152.0, "152 mmHg"),
           ("Total cholesterol (mmol/L)", "chol", 6.4, "6.4 mmol/L"),
           ("Current smoker", "smoke", 1.0, "smoker"),
           ("Diabetes", "diab", 0.0, "no diabetes")]

ROWS = [("Points", 10.0), ("Age (years)", 8.8),
        ("Systolic BP (mmHg)", 7.6), ("Total cholesterol (mmol/L)", 6.4),
        ("Current smoker", 5.2), ("Diabetes", 4.0),
        ("Total points", 2.2), ("10-year CVD risk", 0.85)]
YROW = dict(ROWS)

BANDS = [(0.0, 0.05, "#cfe0d6", "low"), (0.05, 0.075, "#eee0be", "borderline"),
         (0.075, 0.20, "#f0d3c4", "intermediate"),
         (0.20, 0.95, "#f2d7da", "high")]


def pts(key, value):
    return (np.asarray(value, dtype=float) - XMIN[key]) * BETA[key] * SCALE


def risk(tp):
    lp = LP_MIN + np.asarray(tp, dtype=float) / SCALE
    return 1.0 / (1.0 + np.exp(-(lp - LP_REF)))


def tp_for_risk(p):
    return (LP_REF + np.log(p / (1.0 - p)) - LP_MIN) * SCALE


def scale_row(ax, y, u, labels, lw=1.1, tick=0.16, fs=6.2, color=AXIS_C):
    ax.plot([min(u), max(u)], [y, y], lw=lw, color=color,
            solid_capstyle="butt", zorder=3)
    for uu, lab in zip(u, labels):
        ax.plot([uu, uu], [y, y + tick], lw=0.8, color=color, zorder=3)
        if lab is not None:
            ax.text(uu, y + tick + 0.05, lab, ha="center", va="bottom",
                    fontsize=fs, color=INK, zorder=6, path_effects=HALO)


def cohort(rng, n=2600):
    """Synthetic cohort scored by the same coefficients as the nomogram."""
    x = {"age": np.clip(rng.normal(58.0, 9.5, n), 40, 80),
         "sbp": np.clip(rng.normal(133.0, 17.0, n), 100, 190),
         "chol": np.clip(rng.normal(5.3, 0.9, n), 3.5, 8.0),
         "smoke": (rng.random(n) < 0.28).astype(float),
         "diab": (rng.random(n) < 0.16).astype(float)}
    lp = sum(BETA[k] * x[k] for k in BETA)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(lp - LP_REF)))).astype(int)
    # The model is fitted, not omniscient: scoring noise is what makes the
    # calibration slope differ from one and the C-index fall short of truth.
    lp_hat = lp + rng.normal(0.0, 0.28, n)
    return x, y, 1.0 / (1.0 + np.exp(-(lp_hat - LP_REF))), lp


def auc(y, score):
    r = rankdata(score)
    n1 = float(y.sum())
    n0 = float(y.size - n1)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def wilson(k, n, z=1.96):
    ph = k / n
    d = 1.0 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    set_pub_style()
    rng = np.random.default_rng(9151)
    x, y, p_hat, lp_true = cohort(rng)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.55],
                          height_ratios=[1.0, 0.80], hspace=0.46, wspace=0.16)
    ax = fig.add_subplot(gs[:, 0])
    ax_cal = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    # ---- (a) the nomogram ---------------------------------------------------
    ax.set_xlim(-0.42, 1.08)
    ax.set_ylim(-1.35, 11.0)
    ax.axis("off")
    for name, yy in ROWS:
        ax.text(-0.045, yy, name, ha="right", va="center", fontsize=7.2,
                fontweight="bold" if name in ("Points", "Total points",
                                              "10-year CVD risk") else "normal",
                color=INK)

    p = np.arange(0, 101, 10)
    scale_row(ax, YROW["Points"], p / 100.0, [f"{v:d}" for v in p], lw=1.3)

    age = np.arange(40, 81, 5)
    scale_row(ax, YROW["Age (years)"], pts("age", age) / 100.0,
              [f"{v:d}" for v in age])
    sbp = np.arange(100, 191, 10)
    scale_row(ax, YROW["Systolic BP (mmHg)"], pts("sbp", sbp) / 100.0,
              [f"{v:d}" for v in sbp])
    chol = np.arange(3.5, 8.01, 0.5)
    scale_row(ax, YROW["Total cholesterol (mmol/L)"], pts("chol", chol) / 100.0,
              [f"{v:.1f}" for v in chol])
    for row, key in (("Current smoker", "smoke"), ("Diabetes", "diab")):
        scale_row(ax, YROW[row], pts(key, np.array([0.0, 1.0])) / 100.0,
                  ["no", "yes"])

    ax.plot([-0.42, 1.05], [3.22, 3.22], lw=0.7, color="#cfcfcf")
    ax.text(1.05, 3.32, "scale changes below", ha="right", va="bottom",
            fontsize=6.0, style="italic", color="#8a8a8a")

    tp = np.arange(0, 271, 30)
    scale_row(ax, YROW["Total points"], tp / TP_MAX, [f"{v:d}" for v in tp],
              lw=1.3)

    ticks = [0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70]
    u_r, lab_r = [], []
    for pr in ticks:
        t = tp_for_risk(pr)
        if 0.0 <= t <= TP_MAX:
            u_r.append(t / TP_MAX)
            lab_r.append(f"{pr * 100:g}")
    scale_row(ax, YROW["10-year CVD risk"], np.array(u_r), lab_r,
              color=RISK_C, fs=6.2)

    y_band = YROW["10-year CVD risk"] - 0.36
    for lo, hi, colour, lab in BANDS:
        u0 = max(tp_for_risk(max(lo, 1e-4)) / TP_MAX, 0.0)
        u1 = min(tp_for_risk(min(hi, 0.95)) / TP_MAX, 1.0)
        if u1 <= u0:
            continue
        ax.add_patch(Rectangle((u0, y_band), u1 - u0, 0.20, facecolor=colour,
                               edgecolor="white", lw=0.6, zorder=3))
        if u1 - u0 > 0.10:          # a 2.5-point band cannot hold a caption
            ax.text((u0 + u1) / 2.0, y_band + 0.10, lab, ha="center",
                    va="center", fontsize=6.0, color=SUB, zorder=5)
    ax.text(-0.045, y_band + 0.10, "Risk category", ha="right", va="center",
            fontsize=6.6, color=SUB)

    total = 0.0
    for row, key, val, _ in PATIENT:
        pv = float(pts(key, val))
        total += pv
        u = pv / 100.0
        ax.add_patch(FancyArrowPatch(
            (u, YROW[row] + 0.16), (u, YROW["Points"] - 0.02),
            arrowstyle="-|>", mutation_scale=7, lw=0.85, color=PATH_C,
            linestyle=(0, (3.2, 2.2)), shrinkA=0, shrinkB=0, alpha=0.85,
            zorder=4))
        ax.plot([u], [YROW[row]], "o", ms=3.6, mfc=PATH_C, mec="white",
                mew=0.6, zorder=6)
        ax.text(u + 0.012, YROW[row] - 0.20, f"{pv:.0f} pts", ha="left",
                va="top", fontsize=6.2, color=PATH_C, zorder=6,
                path_effects=HALO)

    ut = total / TP_MAX
    p_risk = float(risk(total))
    ax.plot([ut], [YROW["Total points"]], "o", ms=4.2, mfc=PATH_C, mec="white",
            mew=0.7, zorder=6)
    ax.add_patch(FancyArrowPatch(
        (ut, YROW["Total points"] - 0.05), (ut, YROW["10-year CVD risk"] + 0.02),
        arrowstyle="-|>", mutation_scale=8, lw=0.95, color=PATH_C,
        linestyle=(0, (3.2, 2.2)), shrinkA=0, shrinkB=0, zorder=4))
    ax.plot([ut], [YROW["10-year CVD risk"]], "o", ms=4.0, mfc=PATH_C,
            mec="white", mew=0.6, zorder=6)
    ax.text(ut + 0.014, YROW["Total points"] - 0.20, f"{total:.0f} points",
            ha="left", va="top", fontsize=7.0, fontweight="bold",
            color=PATH_C, zorder=6, path_effects=HALO)
    ax.text(ut, y_band - 0.10, f"{p_risk * 100:.1f}% at 10 years",
            ha="center", va="top", fontsize=7.4, fontweight="bold",
            color=PATH_C, zorder=7, path_effects=HALO)

    band_lab = next(lab for lo, hi, _, lab in BANDS if lo <= p_risk < hi)
    foot = [(-0.62, PATH_C, "bold", 6.6,
             "Worked example (dashed path):  " +
             ",  ".join(t for *_, t in PATIENT)),
            (-0.96, PATH_C, "bold", 6.6,
             f"\u2192  {total:.0f} total points  \u2192  {p_risk * 100:.1f}% "
             f"ten-year risk  \u2192  {band_lab} risk category"),
            (-1.28, SUB, "normal", 6.3,
             "Read each predictor up to the Points ruler, sum the points, "
             "then read the total down to the risk ruler.")]
    for yy, col, wt, fs, txt in foot:
        ax.text(-0.42, yy, txt, ha="left", va="center", fontsize=fs,
                color=col, fontweight=wt)
    ax.set_title("Ten-year cardiovascular risk nomogram", fontsize=8.8,
                 fontweight="bold", loc="left", x=-0.005, pad=8)

    # ---- (b) calibration (auxiliary) ---------------------------------------
    edges = np.quantile(p_hat, np.linspace(0.0, 1.0, 11))
    edges[-1] += 1e-9
    idx = np.digitize(p_hat, edges[1:-1])
    xm, ym, lo, hi = [], [], [], []
    for b in range(10):
        m = idx == b
        xm.append(p_hat[m].mean())
        ym.append(y[m].mean())
        a, c = wilson(float(y[m].sum()), float(m.sum()))
        lo.append(a)
        hi.append(c)
    xm, ym = np.array(xm), np.array(ym)
    lim = max(xm.max(), np.max(hi)) * 1.08

    logit = np.log(p_hat / (1.0 - p_hat)).reshape(-1, 1)
    slope = float(LogisticRegression(penalty=None, max_iter=400)
                  .fit(logit, y).coef_[0, 0])
    brier = float(np.mean((p_hat - y) ** 2))

    ax_cal.plot([0, lim], [0, lim], color="#9a9a9a", lw=0.9, ls=(0, (4, 2.4)),
                zorder=2)
    ax_cal.errorbar(xm, ym, yerr=[ym - np.array(lo), np.array(hi) - ym],
                    fmt="o", ms=4.0, mfc=RISK_C, mec="white", mew=0.6,
                    ecolor="#9cc3dd", elinewidth=1.0, capsize=2.0, zorder=4)
    ax_cal.plot(xm, ym, color=RISK_C, lw=1.2, zorder=3)
    ax_cal.set_xlim(0, lim)
    ax_cal.set_ylim(0, lim)
    ax_cal.set_aspect("equal")
    ax_cal.set_xlabel("predicted 10-year risk (-)", fontsize=7.4)
    ax_cal.set_ylabel("observed event rate (-)", fontsize=7.4)
    ax_cal.set_title("Calibration by risk decile", fontsize=8.5,
                     fontweight="bold", loc="left")
    ax_cal.grid(color=GRIDC, lw=0.5)
    ax_cal.tick_params(labelsize=6.5)
    ax_cal.text(0.03, 0.97, f"n = {y.size:,}, {y.sum():,} events\n"
                f"calibration slope {slope:.2f}\nBrier score {brier:.3f}",
                transform=ax_cal.transAxes, ha="left", va="top", fontsize=6.5,
                color=SUB, linespacing=1.45)

    # ---- (c) discrimination (auxiliary) ------------------------------------
    models = [("Age", ["age"]), ("Age + SBP", ["age", "sbp"]),
              ("Full model", list(BETA))]
    names, cs, err = [], [], [[], []]
    for label, keys in models:
        s = sum(BETA[k] * x[k] for k in keys)
        c = auc(y, s)
        boot = []
        for _ in range(200):
            j = rng.integers(0, y.size, y.size)
            if y[j].sum() in (0, y.size):
                continue
            boot.append(auc(y[j], s[j]))
        q = np.percentile(boot, [2.5, 97.5])
        names.append(label)
        cs.append(c)
        err[0].append(c - q[0])
        err[1].append(q[1] - c)
    ypos = np.arange(len(names))[::-1]
    colours = ["#9cc3dd", "#5f9dc6", RISK_C]
    # Dot-and-whisker, not bars: the x axis has to start near 0.5 to resolve
    # the differences, and a truncated bar would exaggerate every one of them.
    ax_c.errorbar(cs, ypos, xerr=err, fmt="none", ecolor=AXIS_C,
                  elinewidth=0.9, capsize=2.6, zorder=4)
    for yy, c, colour in zip(ypos, cs, colours):
        ax_c.plot([c], [yy], marker="o", ms=6.0, mfc=colour, mec="white",
                  mew=0.9, zorder=5)
    for yy, c, e in zip(ypos, cs, err[1]):
        ax_c.text(c + e + 0.010, yy, f"{c:.3f}", va="center", fontsize=6.6,
                  color=INK, zorder=6)
    ax_c.axvline(0.5, color="#808080", lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax_c.set_yticks(ypos)
    ax_c.set_yticklabels(names, fontsize=6.8)
    ax_c.set_ylim(-0.6, len(names) - 0.4)
    ax_c.set_xlim(0.47, 0.80)
    ax_c.set_xlabel("C-index (-)", fontsize=7.4)
    ax_c.set_title("Discrimination, 200 bootstrap resamples", fontsize=8.5,
                   fontweight="bold", loc="left")
    ax_c.grid(axis="x", color=GRIDC, lw=0.5)
    ax_c.tick_params(labelsize=6.5)
    ax_c.tick_params(axis="y", length=0)
    ax_c.spines["left"].set_visible(False)
    ax_c.text(0.985, 0.02, "dashed line: chance (0.5)   |   whiskers: 95% CI",
              transform=ax_c.transAxes, ha="right", va="bottom", fontsize=6.4,
              color=MUTED)

    fig.tight_layout(rect=[0.02, 0.0, 1, 0.94], w_pad=2.4)
    panel_labels_inches(fig, [ax, ax_cal, ax_c], dx=-0.34, dy=0.06)
    savefig(fig, "551_cvd_risk_nomogram.png")


if __name__ == "__main__":
    main()

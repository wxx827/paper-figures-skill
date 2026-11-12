"""Cox-model nomogram: predictor point scales, total points and survival axes."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch

from pubstyle import set_pub_style, savefig

HALO = [pe.withStroke(linewidth=2.0, foreground="white")]

INK = "#333333"
AXIS_C = "#2f2f2f"
PATH_C = "#b23a48"

# ---- underlying Cox model -------------------------------------------------
# beta per unit; the widest predictor range is normalised to 100 points.
BETA = {"age": 0.012, "size": 0.085, "nodes": 0.055,
        "grade": 0.200, "er": 0.260, "ki67": 0.005}
XMIN = {"age": 30, "size": 0.5, "nodes": 0, "grade": 1, "er": 0, "ki67": 10}
XMAX = {"age": 85, "size": 8.0, "nodes": 12, "grade": 3, "er": 1, "ki67": 90}

SPAN = {k: BETA[k] * (XMAX[k] - XMIN[k]) for k in BETA}
SCALE = 100.0 / max(SPAN.values())            # points per unit of linear predictor
LP_MIN = sum(BETA[k] * XMIN[k] for k in BETA)
LP_REF = 1.40                                  # centring value of the model
TP_MAX = 470.0
S0 = {3: 0.88, 5: 0.78}                        # baseline survival at t years

ROWS = [
    ("Points",             10.0),
    ("Age (years)",         9.0),
    ("Tumour size (cm)",    8.0),
    ("Positive nodes (n)",  7.0),
    ("Histological grade",  6.0),
    ("ER status",           5.0),
    ("Ki-67 index (%)",     4.0),
    ("Total points",        2.3),
    ("3-year survival",     1.2),
    ("5-year survival",     0.1),
]
YROW = dict(ROWS)


def pts(key, value):
    return (value - XMIN[key]) * BETA[key] * SCALE


def surv(tp, years):
    lp = LP_MIN + tp / SCALE
    return S0[years] ** np.exp(lp - LP_REF)


def tp_for_surv(s, years):
    lp = LP_REF + np.log(np.log(s) / np.log(S0[years]))
    return (lp - LP_MIN) * SCALE


def scale_row(ax, y, u_positions, labels, lw=1.1, tick=0.17, fs=6.2,
              rot=0, color=AXIS_C):
    """Draw one nomogram ruler: baseline, upward ticks, labels above."""
    u0, u1 = min(u_positions), max(u_positions)
    ax.plot([u0, u1], [y, y], lw=lw, color=color, solid_capstyle="butt",
            zorder=3)
    for u, lab in zip(u_positions, labels):
        ax.plot([u, u], [y, y + tick], lw=0.8, color=color, zorder=3)
        if lab is not None:
            ax.text(u, y + tick + 0.06, lab, ha="center", va="bottom",
                    fontsize=fs, color="#1a1a1a", rotation=rot, zorder=6,
                    path_effects=HALO)


def main():
    set_pub_style()
    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    ax.set_xlim(-0.30, 1.05)
    ax.set_ylim(-3.6, 11.2)
    ax.axis("off")

    for name, y in ROWS:
        weight = "bold" if name in ("Points", "Total points") else "normal"
        ax.text(-0.025, y, name, ha="right", va="center", fontsize=7.4,
                fontweight=weight, color="#111111")

    # ---- points ruler ------------------------------------------------------
    p = np.arange(0, 101, 10)
    scale_row(ax, YROW["Points"], p / 100.0, [f"{v:d}" for v in p], lw=1.3)

    # ---- predictor rulers --------------------------------------------------
    age = np.arange(30, 86, 5)
    scale_row(ax, YROW["Age (years)"], pts("age", age) / 100.0,
              [f"{v:d}" for v in age])

    size = np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8])
    scale_row(ax, YROW["Tumour size (cm)"], pts("size", size) / 100.0,
              [f"{v:g}" for v in size])

    nod = np.arange(0, 13, 1)
    scale_row(ax, YROW["Positive nodes (n)"], pts("nodes", nod) / 100.0,
              [f"{v:d}" for v in nod])

    grade = np.array([1, 2, 3])
    scale_row(ax, YROW["Histological grade"], pts("grade", grade) / 100.0,
              ["I", "II", "III"])

    er = np.array([0, 1])
    scale_row(ax, YROW["ER status"], pts("er", er) / 100.0,
              ["Positive", "Negative"])

    ki = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    scale_row(ax, YROW["Ki-67 index (%)"], pts("ki67", ki) / 100.0,
              [f"{v:d}" for v in ki])

    # ---- scale break -------------------------------------------------------
    ax.plot([-0.30, 1.02], [3.25, 3.25], lw=0.7, color="#cfcfcf", ls="-")
    ax.text(1.02, 3.36, "scale changes below", ha="right", va="bottom",
            fontsize=6.0, style="italic", color="#8a8a8a")

    # ---- total points + survival ------------------------------------------
    tp = np.arange(0, 451, 50)
    scale_row(ax, YROW["Total points"], tp / TP_MAX, [f"{v:d}" for v in tp],
              lw=1.3)
    for v in np.arange(25, 451, 50):
        ax.plot([v / TP_MAX] * 2, [YROW["Total points"]] * 2 +
                np.array([0, 0.09]), lw=0.7, color=AXIS_C, zorder=3)

    for years, ticks in ((3, [0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]),
                         (5, [0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])):
        row = f"{years}-year survival"
        xs, labs = [], []
        for s in ticks:
            t = tp_for_surv(s, years)
            if 0 <= t <= TP_MAX:
                xs.append(t / TP_MAX)
                labs.append(f"{s:g}")
        scale_row(ax, YROW[row], np.array(xs), labs, color="#2f5d8a")

    # ---- worked example ----------------------------------------------------
    patient = [("Age (years)", "age", 62), ("Tumour size (cm)", "size", 4.5),
               ("Positive nodes (n)", "nodes", 2),
               ("Histological grade", "grade", 2), ("ER status", "er", 0),
               ("Ki-67 index (%)", "ki67", 35)]
    total = 0.0
    for row, key, val in patient:
        pv = pts(key, val)
        total += pv
        u = pv / 100.0
        ax.plot([u, u], [YROW[row], YROW["Points"]], lw=0.8, ls=(0, (3, 2)),
                color=PATH_C, alpha=0.75, zorder=2)
        ax.plot([u], [YROW[row]], "o", ms=3.4, mfc=PATH_C, mec="white",
                mew=0.6, zorder=5)
    ax.plot([total / TP_MAX], [YROW["Total points"]], "o", ms=4.2, mfc=PATH_C,
            mec="white", mew=0.7, zorder=5)

    ut = total / TP_MAX
    ax.add_patch(FancyArrowPatch(
        (ut, YROW["Total points"]), (ut, YROW["5-year survival"] - 0.02),
        arrowstyle="-|>", mutation_scale=8, lw=0.9, color=PATH_C,
        linestyle="--", shrinkA=0, shrinkB=0, zorder=4))
    for years in (3, 5):
        row = f"{years}-year survival"
        s = surv(total, years)
        ax.plot([ut], [YROW[row]], "o", ms=3.8, mfc=PATH_C, mec="white",
                mew=0.6, zorder=5)
        ax.text(ut + 0.018, YROW[row] - 0.30, f"{s * 100:.1f}%", ha="left",
                va="top", fontsize=7.0, fontweight="bold", color=PATH_C,
                zorder=6, path_effects=HALO)
    ax.text(ut + 0.012, YROW["Total points"] - 0.28, f"{total:.0f} points",
            ha="left", va="top", fontsize=7.0, fontweight="bold",
            color=PATH_C, zorder=6, path_effects=HALO)

    foot = [
        (-1.35, PATH_C, "bold", 6.5,
         "Worked example (dashed path):  age 62 y, tumour 4.5 cm, 2 positive "
         "nodes, grade II, ER-positive, Ki-67 35%"),
        (-2.05, PATH_C, "bold", 6.5,
         f"\u2192  {total:.0f} total points  \u2192  predicted 3-year survival "
         f"{surv(total, 3) * 100:.1f}%, 5-year survival "
         f"{surv(total, 5) * 100:.1f}%"),
        (-2.80, "#555555", "normal", 6.3,
         "Read each predictor upward to the Points ruler, sum the points, then "
         "read the total downward to the survival axes."),
        (-3.35, "#555555", "normal", 6.3,
         "Model: Cox proportional hazards (n = 1,142; 318 deaths); "
         "optimism-corrected C-index 0.74 (95% CI 0.71\u20130.77)."),
    ]
    for y, col, wt, fs, txt in foot:
        ax.text(-0.30, y, txt, ha="left", va="center", fontsize=fs,
                color=col, fontweight=wt)

    ax.set_title("Nomogram predicting 3- and 5-year overall survival after "
                 "primary resection", fontsize=10.0, fontweight="bold",
                 pad=10, loc="left", x=-0.005)

    fig.tight_layout()
    savefig(fig, "112_nomogram.png")


if __name__ == "__main__":
    main()

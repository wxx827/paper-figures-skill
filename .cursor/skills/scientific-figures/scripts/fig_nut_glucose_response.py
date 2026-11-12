"""Postprandial glycaemia: four 50 g carbohydrate test meals compared.

Each meal is described by two design parameters only - the height and the
timing of its glucose excursion.  Everything the figure claims is then measured
off the simulated capillary samples: the incremental area under the curve is
trapezoidal over the standard sampling grid, the glycaemic index is that iAUC
expressed as a percentage of the glucose reference, the peak and the
return-to-baseline time are read from the interpolating curve, and the
glycaemic load of a real portion is the computed index times the available
carbohydrate in that portion.  Insulin follows the same excursion with a
delay and an amplitude proportional to the glucose iAUC.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

from pubstyle import annotate_box, savefig, set_pub_style
from figkit import panel_labels_inches, shared_limits

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
BLUE, GREEN, RED, GOLD = "#3b5b92", "#2e7d5b", "#b23a48", "#c9a227"

BASE = 5.0                 # fasting plasma glucose (mmol/L)
FAST_LO, FAST_HI = 3.9, 5.6
THRESH = 7.8               # 2 h postprandial threshold (mmol/L)
MGDL = 18.016              # mmol/L -> mg/dL
SAMPLE_T = np.array([0., 15., 30., 45., 60., 90., 120., 150., 180.])
FASTING_INSULIN = 6.0      # mIU/L

# name, excursion amplitude (mmol/L), time to peak (min), late-dip fraction,
# available carbohydrate in one realistic portion (g), colour
MEALS = [
    ("Glucose drink", 3.55, 31.0, 0.26, 25.0, RED),
    ("White bread", 1.95, 38.0, 0.16, 30.0, GOLD),
    ("Basmati rice", 1.04, 47.0, 0.05, 45.0, BLUE),
    ("Lentils", 0.46, 58.0, 0.00, 20.0, GREEN),
]

# Reference foods for the index-versus-load panel: name, index, portion carb (g)
REFERENCE = [
    ("Baked potato", 85, 30), ("Cornflakes", 81, 26), ("Rice cakes", 82, 21),
    ("Watermelon", 76, 11), ("Rye bread", 71, 24), ("Sweet potato", 63, 25),
    ("Oats", 55, 27), ("Banana", 51, 24), ("Pasta", 49, 40),
    ("Milk", 39, 12), ("Apple", 36, 15), ("Chickpeas", 28, 30),
]
LOW_GI, HIGH_GI = 55.0, 70.0


def excursion(t, amp, tp, dip, k=2.2):
    """Rise-and-fall glucose excursion with an optional reactive undershoot."""
    x = np.maximum(np.asarray(t, dtype=float), 1e-6) / tp
    y = amp * x ** k * np.exp(k * (1.0 - x))
    if dip:
        y = y - dip * amp * np.exp(-0.5 * ((t - 2.5 * tp) / (0.95 * tp)) ** 2)
    return y


def iauc(t, y, base):
    """Incremental area under the curve, positive excursion only."""
    return float(np.trapezoid(np.clip(y - base, 0.0, None), t))


def gi_class(g):
    return GREEN if g <= LOW_GI else (GOLD if g < HIGH_GI else RED)


def main():
    set_pub_style()
    rng = np.random.default_rng(661)

    t_dense = np.linspace(0.0, 180.0, 361)
    meals = []
    for name, amp, tp, dip, portion, col in MEALS:
        y = BASE + excursion(SAMPLE_T, amp, tp, dip)
        y = y + rng.normal(0.0, 0.055, SAMPLE_T.size)
        y[0] = BASE + rng.normal(0.0, 0.03)
        curve = PchipInterpolator(SAMPLE_T, y)(t_dense)
        area = iauc(SAMPLE_T, y, y[0])
        j = int(np.argmax(curve))
        after = np.where((t_dense > t_dense[j]) & (curve <= y[0] + 0.16))[0]
        meals.append(dict(name=name, colour=col, sample=y, curve=curve,
                          base=float(y[0]), iauc=area, peak=float(curve[j]),
                          t_peak=float(t_dense[j]), portion=portion,
                          t_back=float(t_dense[after[0]]) if after.size
                          else np.nan))
    ref_iauc = meals[0]["iauc"]
    for m in meals:
        m["gi"] = 100.0 * m["iauc"] / ref_iauc
        m["gl"] = m["gi"] * m["portion"] / 100.0
        m["ins_amp"] = 0.36 * m["iauc"]

    # insulin: same shape, later peak, amplitude set by the glucose iAUC
    for m, (_, _, tp, _, _, _) in zip(meals, MEALS):
        yi = FASTING_INSULIN + excursion(SAMPLE_T, m["ins_amp"], tp + 11.0,
                                         0.10, k=1.9)
        yi = np.clip(yi + rng.normal(0.0, 0.9, SAMPLE_T.size), 1.0, None)
        yi[0] = FASTING_INSULIN + rng.normal(0.0, 0.35)
        m["ins_sample"] = yi
        m["ins_curve"] = PchipInterpolator(SAMPLE_T, yi)(t_dense)
        m["ins_iauc"] = iauc(SAMPLE_T, yi, yi[0])

    # ---- layout ------------------------------------------------------------
    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.88],
                          left=0.058, right=0.958, top=0.840, bottom=0.160,
                          wspace=0.30, hspace=0.46)
    ax_m = fig.add_subplot(gs[:, 0])
    ax_g = fig.add_subplot(gs[0, 1])
    ax_i = fig.add_subplot(gs[1, 1])

    # ---- main: glucose curves ---------------------------------------------
    ax_m.axhspan(FAST_LO, FAST_HI, color=BLUE, alpha=0.09, zorder=0)
    ax_m.axhline(THRESH, color=RED, lw=0.9, ls=(0, (4, 2.5)), zorder=2)
    ax_m.text(178.0, THRESH + 0.06,
              f"2 h postprandial threshold {THRESH:.1f} mmol/L", ha="right",
              va="bottom", fontsize=6.2, color=RED, zorder=5)
    ax_m.text(2.0, FAST_LO + 0.08,
              f"fasting reference {FAST_LO:.1f}-{FAST_HI:.1f} mmol/L",
              ha="left", va="bottom", fontsize=6.2, color=BLUE, zorder=5)

    for k, m in enumerate(meals):
        ax_m.fill_between(t_dense, m["base"], m["curve"],
                          where=m["curve"] > m["base"], color=m["colour"],
                          alpha=0.13, lw=0.0, zorder=2)
        ax_m.plot(t_dense, m["curve"], color=m["colour"], lw=1.6, zorder=4,
                  label=f"{m['name']}   GI {m['gi']:.0f}   "
                        f"iAUC {m['iauc']:.0f}")
        ax_m.plot(SAMPLE_T, m["sample"], ls="none", marker="o", ms=2.9,
                  mfc="white", mec=m["colour"], mew=0.8, zorder=5)
        ax_m.plot([m["t_peak"]], [m["peak"]], marker="v", ms=4.6,
                  color=m["colour"], mec="white", mew=0.6, zorder=6)
        ax_m.text(m["t_peak"], m["peak"] + 0.14,
                  f"{m['peak']:.1f} at {m['t_peak']:.0f} min", ha="center",
                  va="bottom", fontsize=6.0, color=m["colour"], zorder=6)
        if np.isfinite(m["t_back"]):
            ax_m.plot([m["t_back"]], [m["base"]], marker="^", ms=4.4,
                      color=m["colour"], mec="white", mew=0.5, zorder=6)
            ax_m.text(m["t_back"], m["base"] - 0.11 - 0.17 * (k % 2),
                      f"{m['t_back']:.0f} min", ha="center", va="top",
                      fontsize=6.0, color=m["colour"], zorder=6)

    lo, hi = shared_limits([m["curve"] for m in meals], pad=0.05)
    ax_m.set_ylim(min(lo, 3.7), hi + 0.55)
    ax_m.set_xlim(-4.0, 182.0)
    ax_m.set_xticks(np.arange(0, 181, 30))
    ax_m.set_xlabel("time after the 50 g carbohydrate test meal (min)",
                    fontsize=8.0)
    ax_m.set_ylabel("capillary glucose (mmol/L)", fontsize=8.0)
    ax_m.tick_params(labelsize=7.0)
    ax_m.grid(axis="y", lw=0.5, color="#f0f0f0", zorder=0)
    ax_m.set_title("Same carbohydrate dose, very different excursions",
                   fontsize=9.0, fontweight="bold", pad=7)
    ax_m.text(0.985, 0.015, "triangles mark the return to the fasting level",
              transform=ax_m.transAxes, ha="right", va="bottom",
              fontsize=6.2, color=MUTED)

    ax_mg = ax_m.twinx()
    ax_mg.set_ylim(*[v * MGDL for v in ax_m.get_ylim()])
    ax_mg.set_ylabel("capillary glucose (mg/dL)", fontsize=8.0, color=SUB)
    ax_mg.tick_params(axis="y", labelsize=7.0, colors=SUB)
    ax_mg.spines["right"].set_visible(True)
    ax_mg.spines["right"].set_color("#cccccc")
    ax_mg.spines["top"].set_visible(False)

    leg = ax_m.legend(loc="upper right", frameon=False, fontsize=6.8,
                      handlelength=1.7, labelspacing=0.36,
                      borderaxespad=0.25,
                      title="meal, index and iAUC (mmol/L min)")
    leg.get_title().set_fontsize(6.8)
    leg.get_title().set_color(SUB)

    # ---- aux 1: index versus load -----------------------------------------
    ax_g.axvspan(0.0, LOW_GI, color=GREEN, alpha=0.07, zorder=0)
    ax_g.axvspan(LOW_GI, HIGH_GI, color=GOLD, alpha=0.08, zorder=0)
    ax_g.axvspan(HIGH_GI, 112.0, color=RED, alpha=0.07, zorder=0)
    for xx, txt, col in ((LOW_GI / 2.0, "low", GREEN),
                         ((LOW_GI + HIGH_GI) / 2.0, "medium", GOLD),
                         ((HIGH_GI + 112.0) / 2.0, "high", RED)):
        ax_g.text(xx, 0.975, txt, transform=ax_g.get_xaxis_transform(),
                  ha="center", va="top", fontsize=6.2, color=col)

    r_gi = np.array([r[1] for r in REFERENCE], dtype=float)
    r_gl = np.array([r[1] * r[2] / 100.0 for r in REFERENCE])
    ax_g.scatter(r_gi, r_gl, s=17, c=[gi_class(g) for g in r_gi],
                 alpha=0.55, edgecolors="white", linewidths=0.5, zorder=3)
    for m in meals:
        ax_g.scatter([m["gi"]], [m["gl"]], s=44, c=m["colour"],
                     edgecolors=INK, linewidths=0.7, zorder=5)
        ax_g.annotate(m["name"], (m["gi"], m["gl"]),
                      textcoords="offset points", xytext=(0, 7),
                      ha="center", va="bottom", fontsize=6.0,
                      color=m["colour"], zorder=6)
    ax_g.axhline(20.0, color="#808080", lw=0.8, ls=":", zorder=2)
    ax_g.text(3.0, 20.4, "high load, GL > 20 g", ha="left", va="bottom",
              fontsize=6.0, color=MUTED)
    ax_g.set_xlim(0.0, 112.0)
    ax_g.set_ylim(0.0, max(r_gl.max(), max(m["gl"] for m in meals)) * 1.30)
    ax_g.set_xlabel("glycaemic index (% of glucose reference)", fontsize=7.4)
    ax_g.set_ylabel("glycaemic load (g per portion)", fontsize=7.4)
    ax_g.tick_params(labelsize=6.5)
    ax_g.grid(lw=0.5, color="#f2f2f2", zorder=0)
    ax_g.set_title("Index ranks the food, load ranks the portion",
                   fontsize=8.5, fontweight="bold", pad=5)

    # ---- aux 2: insulin ----------------------------------------------------
    ax_i.axhline(FASTING_INSULIN, color="#808080", lw=0.8, ls=":", zorder=2)
    for m in meals:
        ax_i.plot(t_dense, m["ins_curve"], color=m["colour"], lw=1.4,
                  zorder=4)
        ax_i.plot(SAMPLE_T, m["ins_sample"], ls="none", marker="o", ms=2.5,
                  mfc="white", mec=m["colour"], mew=0.7, zorder=5)
    ax_i.set_xlim(-4.0, 182.0)
    ax_i.set_xticks(np.arange(0, 181, 30))
    ax_i.set_ylim(0.0, max(m["ins_curve"].max() for m in meals) * 1.34)
    ax_i.set_xlabel("time after the test meal (min)", fontsize=7.4)
    ax_i.set_ylabel("plasma insulin (mIU/L)", fontsize=7.4)
    ax_i.tick_params(labelsize=6.5)
    ax_i.grid(axis="y", lw=0.5, color="#f2f2f2", zorder=0)
    ax_i.set_title("Insulin tracks the glucose area", fontsize=8.5,
                   fontweight="bold", pad=5)
    annotate_box(ax_i, "insulin iAUC (mIU/L min)\n" + "\n".join(
        f"{m['name']:<14s}{m['ins_iauc']:6.0f}" for m in meals),
        loc="upper right", fontsize=6.0, margin=0.018)

    ratio = meals[0]["iauc"] / meals[-1]["iauc"]
    panel_labels_inches(fig, [ax_m, ax_g, ax_i], dx=-0.42, dy=0.10)

    for m in meals:
        print(f"{m['name']:<14s} GI {m['gi']:5.1f}  iAUC {m['iauc']:6.1f}  "
              f"peak {m['peak']:.2f} mmol/L at {m['t_peak']:.0f} min  "
              f"back {m['t_back']:.0f} min  GL {m['gl']:.1f} g")
    savefig(fig, "661_nut_glucose_response.png")


if __name__ == "__main__":
    main()

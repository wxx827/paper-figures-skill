"""Average bioequivalence: three metrics, three confidence intervals, one verdict.

A 2x2 crossover of 24 subjects is simulated at the level of the concentration
profiles, and nothing else on the page is entered by hand.  Cmax is the largest
sampled concentration, AUC0-t is a trapezoid over the sampling schedule and
AUC0-inf adds Clast/lambda_z with lambda_z taken from a log-linear fit to the
terminal points, all per subject and per period.  The 90 % confidence intervals
in the main panel come from a paired t interval on the log ratios, so the bar
that crosses the 80 % boundary is the same metric whose individual ratios are
visibly the most scattered, and the equivalence verdict printed beside each bar
is decided by the plotted interval rather than asserted.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
RED = "#b23a48"
ORANGE = "#c0603f"
GREY = "#808080"
HAIR = "#cccccc"

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.94, 1.02, 4.62, 4.48)
AUX_T = (7.50, 3.61, 2.68, 1.89)
AUX_B = (7.50, 1.02, 2.68, 1.82)

N_SUBJ = 24
TIMES = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0,
                  12.0, 16.0, 24.0])
BE_LO, BE_HI = 80.0, 125.0        # regulatory equivalence interval, per cent
ALPHA = 0.10                      # two one-sided tests at 5 %, i.e. a 90 % CI

KA0, KE0, VF0, DOSE = 1.45, 0.098, 42.0, 100.0     # 1/h, 1/h, L, mg
KA_T_FACTOR = 0.50                # the test tablet releases more slowly
F_T_FACTOR = 0.985
SD_KE_BETWEEN, SD_V_BETWEEN = 0.24, 0.26
SD_KA_WITHIN, SD_F_WITHIN = 0.62, 0.105
CV_ASSAY = 0.10

METRICS = [("$C_{max}$ (ng/mL)", "cmax"),
           ("$AUC_{0-t}$ ($\\mu g{\\cdot}h/mL$)", "auc_t"),
           ("$AUC_{0-\\infty}$ ($\\mu g{\\cdot}h/mL$)", "auc_inf")]


def profile(t, ka, ke, vf, dose):
    """One-compartment oral concentration in ng/mL for a dose in mg."""
    t = np.asarray(t, dtype=float)
    amt = dose * 1e6 / vf                      # mg -> ng, L -> mL
    return amt * ka / (ka - ke) * (np.exp(-ke * t) - np.exp(-ka * t)) / 1000.0


def simulate(rng):
    """Concentration matrices (subjects x times) for reference and test."""
    ke = KE0 * np.exp(rng.normal(0.0, SD_KE_BETWEEN, N_SUBJ))
    vf = VF0 * np.exp(rng.normal(0.0, SD_V_BETWEEN, N_SUBJ))
    out = {}
    for arm, ka_f, f_f in (("R", 1.0, 1.0), ("T", KA_T_FACTOR, F_T_FACTOR)):
        ka = KA0 * ka_f * np.exp(rng.normal(0.0, SD_KA_WITHIN, N_SUBJ))
        rel = f_f * np.exp(rng.normal(0.0, SD_F_WITHIN, N_SUBJ))
        conc = np.empty((N_SUBJ, TIMES.size))
        for i in range(N_SUBJ):
            clean = profile(TIMES, ka[i], ke[i], vf[i], DOSE * rel[i])
            conc[i] = np.clip(clean * rng.normal(1.0, CV_ASSAY, TIMES.size),
                              0.0, None)
        out[arm] = conc
    return out


def exposure(conc):
    """Cmax, AUC0-t and AUC0-inf per subject, straight off the curves."""
    cmax = conc.max(axis=1)
    auc_t = np.trapezoid(conc, TIMES, axis=1)
    auc_inf = np.empty(N_SUBJ)
    for i in range(N_SUBJ):
        tail = slice(-4, None)
        slope = np.polyfit(TIMES[tail], np.log(conc[i, tail]), 1)[0]
        lam = max(-slope, 1e-3)
        auc_inf[i] = auc_t[i] + conc[i, -1] / lam
    return dict(cmax=cmax, auc_t=auc_t / 1000.0, auc_inf=auc_inf / 1000.0)


def be_stats(test, ref):
    """Point estimate, 90 % CI and intra-subject CV from the log ratios."""
    d = np.log(test) - np.log(ref)
    n = d.size
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    tcrit = float(stats.t.ppf(1.0 - ALPHA / 2.0, n - 1))
    half = tcrit * sd / np.sqrt(n)
    sw = sd / np.sqrt(2.0)                    # crossover: differences pool two
    return dict(pe=100.0 * np.exp(mean),
                lo=100.0 * np.exp(mean - half),
                hi=100.0 * np.exp(mean + half),
                cv=100.0 * np.sqrt(np.exp(sw ** 2) - 1.0),
                ratios=100.0 * np.exp(d))


def sample_size(cv_pct, gmr=0.95, power=0.80):
    """Total subjects for a 2x2 crossover, normal-theory approximation."""
    s2 = np.log(1.0 + (cv_pct / 100.0) ** 2)
    delta = np.log(BE_HI / 100.0) - abs(np.log(gmr))
    n = 8.0
    for _ in range(60):
        df = max(n - 2.0, 1.0)
        ta = stats.t.ppf(1.0 - ALPHA / 2.0, df)
        tb = stats.t.ppf(power, df)
        n_new = 2.0 * (ta + tb) ** 2 * s2 / delta ** 2
        n_new = max(np.ceil(n_new / 2.0) * 2.0, 4.0)
        if abs(n_new - n) < 0.5:
            n = n_new
            break
        n = n_new
    return float(n)


def main_panel(ax, res, rng):
    ax.axvspan(BE_LO, BE_HI, color="#eef3f0", zorder=0)
    for v, lab in ((BE_LO, "80 %"), (BE_HI, "125 %")):
        ax.axvline(v, color=GREEN, lw=1.15, ls=(0, (5.0, 2.2)), zorder=4)
        ax.text(v, 2.58, lab, ha="center", va="bottom", fontsize=6.5,
                color=GREEN, fontweight="bold", zorder=7)
    ax.axvline(100.0, color=GREY, lw=0.8, ls=(0, (1.6, 1.8)), zorder=3)
    ax.text(101.0, -0.62, "no difference", ha="left", va="bottom",
            fontsize=6.1, color=MUTED, zorder=6)
    ax.text(0.5 * (BE_LO + BE_HI), 2.78,
            "regulatory equivalence interval", ha="center", va="bottom",
            fontsize=6.8, color=GREEN, zorder=7)

    y = np.arange(len(METRICS))[::-1]
    for yi, (label, key) in zip(y, METRICS):
        s = res[key]
        ok = s["lo"] >= BE_LO and s["hi"] <= BE_HI
        col = BLUE if ok else ORANGE
        jit = rng.uniform(-0.155, 0.155, N_SUBJ)
        ax.plot(np.clip(s["ratios"], 52.0, 178.0), yi + 0.30 + jit, ls="none",
                marker="o", ms=2.9, mfc="white", mec=lighten(col, 0.30),
                mew=0.7, alpha=0.90, zorder=5)
        ax.plot([s["lo"], s["hi"]], [yi, yi], color=col, lw=3.4,
                solid_capstyle="butt", zorder=6)
        for v in (s["lo"], s["hi"]):
            ax.plot([v, v], [yi - 0.115, yi + 0.115], color=col, lw=1.5,
                    zorder=7)
        ax.plot([s["pe"]], [yi], marker="D", ms=6.0, mfc="white", mec=col,
                mew=1.4, zorder=8)
        ax.text(0.5 * (s["lo"] + s["hi"]), yi - 0.215,
                "{:.1f}   90 % CI {:.1f} - {:.1f}".format(s["pe"], s["lo"],
                                                          s["hi"]),
                ha="center", va="top", fontsize=6.5, color=col,
                fontweight="bold", zorder=8)
        ax.text(176.0, yi + 0.055,
                "equivalent" if ok else "not equivalent", ha="right",
                va="bottom", fontsize=7.0, color=col, fontweight="bold",
                zorder=8)
        ax.text(176.0, yi - 0.075,
                "intra-subject CV {:.1f} %".format(s["cv"]), ha="right",
                va="top", fontsize=6.2, color=SUB, zorder=8)

    ax.set_yticks(y)
    ax.set_yticklabels([m[0] for m in METRICS], fontsize=8.2)
    ax.set_xlim(52.0, 178.0)
    ax.set_ylim(-0.80, 3.02)
    ax.set_xticks([60, 80, 100, 125, 150, 175])
    ax.set_xlabel("test / reference geometric mean ratio (%)")
    ax.grid(axis="x", lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [
        Line2D([], [], color=BLUE, lw=3.4, label="90 % CI, equivalent"),
        Line2D([], [], color=ORANGE, lw=3.4, label="90 % CI, not equivalent"),
        Line2D([], [], color=SUB, lw=0.0, marker="D", ms=5.0, mfc="white",
               mec=SUB, mew=1.2, label="geometric mean ratio"),
        Line2D([], [], color=GREY, lw=0.0, marker="o", ms=3.0, mfc="white",
               mec=GREY, mew=0.7,
               label="individual subject ratio ($n$ = {:d})".format(N_SUBJ)),
        Patch(facecolor="#eef3f0", edgecolor=GREEN, lw=1.0,
              label="{:.0f} - {:.0f} % acceptance band".format(BE_LO, BE_HI)),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=6.6, frameon=False,
              bbox_to_anchor=(0.004, 0.004), handlelength=1.7,
              borderaxespad=0.0, labelspacing=0.30, ncol=2, columnspacing=1.4)
    ax.set_title("Two-way crossover, {:d} subjects, single 100 mg dose"
                 .format(N_SUBJ), fontsize=9.0, fontweight="bold", color=INK,
                 pad=6.0)


def power_panel(ax, res):
    cv = np.linspace(8.0, 42.0, 90)
    for power, col, ls in ((0.80, BLUE, "-"), (0.90, GREEN, (0, (4.0, 2.0)))):
        n = np.array([sample_size(c, power=power) for c in cv])
        ax.plot(cv, n, color=col, lw=1.5, ls=ls,
                label="{:.0f} % power".format(100.0 * power))
    for key, col in (("cmax", ORANGE), ("auc_t", BLUE), ("auc_inf", BLUE)):
        c = res[key]["cv"]
        ax.plot([c], [sample_size(c, power=0.80)], marker="o", ms=4.4,
                mfc=col, mec="white", mew=0.8, zorder=6)
    c_max = res["cmax"]["cv"]
    ax.annotate("$C_{{max}}$  $n$ = {:.0f}".format(sample_size(c_max)),
                xy=(c_max, sample_size(c_max)), xytext=(6.0, -1.0),
                textcoords="offset points", ha="left", va="center",
                fontsize=6.2, color=ORANGE, fontweight="bold", zorder=7)
    c_auc = 0.5 * (res["auc_t"]["cv"] + res["auc_inf"]["cv"])
    ax.annotate("both $AUC$ metrics  $n$ = {:.0f}".format(sample_size(c_auc)),
                xy=(c_auc, sample_size(c_auc)), xytext=(6.0, 8.0),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=6.2, color=BLUE, fontweight="bold", zorder=7,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6))
    ax.axhline(N_SUBJ, color=GREY, lw=0.9, ls=(0, (1.8, 1.8)), zorder=3)
    ax.text(41.4, N_SUBJ - 2.2, "this study, {:d} subjects".format(N_SUBJ),
            ha="right", va="top", fontsize=6.1, color=SUB, zorder=6)
    ax.set_xlim(8.0, 42.0)
    ax.set_ylim(0.0, 96.0)
    ax.set_xlabel("intra-subject CV (%)", fontsize=7.2)
    ax.set_ylabel("subjects required", fontsize=7.2)
    ax.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", fontsize=6.1, frameon=False, handlelength=1.5,
              borderaxespad=0.25, labelspacing=0.24)
    ax.set_title("Sample size at GMR 0.95", fontsize=8.0, fontweight="bold",
                 color=INK)


def profile_panel(ax, conc):
    for arm, col, name in (("R", BLUE, "reference"), ("T", ORANGE, "test")):
        m = conc[arm].mean(axis=0)
        sd = conc[arm].std(axis=0, ddof=1)
        ax.fill_between(TIMES, m - sd, m + sd, color=lighten(col, 0.80),
                        lw=0.0, zorder=2)
        ax.plot(TIMES, m, color=col, lw=1.5, zorder=4)
        ax.plot(TIMES, m, ls="none", marker="o", ms=2.8, mfc="white",
                mec=col, mew=0.8, zorder=5)
        j = int(np.argmax(m))
        ax.plot([TIMES[j]], [m[j]], marker="v", ms=5.0, mfc=col, mec="white",
                mew=0.8, zorder=6)
        ax.plot([], [], color=col, lw=1.5,
                label="{:s}   $C_{{max}}$ {:.0f} ng/mL at {:g} h".format(
                    name, m[j], TIMES[j]))
    ax.set_xlim(-0.6, 24.6)
    ax.set_ylim(0.0, None)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlabel("time after dose (h)", fontsize=7.2)
    ax.set_ylabel("concentration (ng/mL)", fontsize=7.2)
    ax.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", fontsize=6.1, frameon=False,
              handlelength=1.4, borderaxespad=0.25, labelspacing=0.24)
    ax.set_title("Mean profiles $\\pm$ 1 SD", fontsize=8.0, fontweight="bold",
                 color=INK)


def audit(fig, foot, axes):
    """Warn if the footnote box overlaps an in-view tick or axis label."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for ax in axes:
        ab = ax.get_window_extent()
        for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            tb = t.get_window_extent()
            if tb.width <= 0.0 or not t.get_text():
                continue
            if tb.x1 < ab.x0 - 2.0 or tb.x0 > ab.x1 + 2.0:
                continue
            if fb.overlaps(tb):
                print("OVERLAP with", repr(t.get_text()))


def main():
    set_pub_style()
    rng = np.random.default_rng(9441)

    conc = simulate(rng)
    exp_r, exp_t = exposure(conc["R"]), exposure(conc["T"])
    res = {key: be_stats(exp_t[key], exp_r[key]) for _lab, key in METRICS}
    for lab, key in METRICS:
        s = res[key]
        print("{:22s} PE {:6.1f}  CI {:6.1f}-{:6.1f}  CV {:5.1f}".format(
            lab, s["pe"], s["lo"], s["hi"], s["cv"]))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_pow = axes_inches(fig, AUX_T)
    ax_pro = axes_inches(fig, AUX_B)

    main_panel(ax_main, res, rng)
    power_panel(ax_pow, res)
    profile_panel(ax_pro, conc)

    fails = [lab for lab, key in METRICS
             if not (res[key]["lo"] >= BE_LO and res[key]["hi"] <= BE_HI)]
    cm, at = res["cmax"], res["auc_t"]


    panel_labels_inches(fig, [ax_main, ax_pow, ax_pro], dx=-0.68, dy=0.14)
    savefig(fig, "944_pha_bioequivalence.png")


if __name__ == "__main__":
    main()

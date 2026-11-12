"""Radionuclide decay and the internal dose that follows from it.

Four therapeutic nuclides decay on a semi-logarithmic ordinate, where the
slope of each straight line *is* its half-life; the quoted half-lives are
recovered by a log-linear fit to the plotted arrays rather than copied from a
table.  Over them sits the measured kidney time-activity curve for one of
those nuclides: uptake, biological clearance and physical decay together, so
its tail is visibly steeper than the physical line it started from.

The two panels on the right follow the same chain to its end - the area under
each organ curve is the cumulated activity, and the cumulated activity times
an organ S value is the absorbed dose per unit administered activity.

Synthetic illustration for a visual reference library - not patient data.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, despine
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
HAIR = "#cccccc"

FIGW, FIGH = 10.4, 6.4
A0 = 1000.0                 # administered activity (MBq)
T_END = 400.0               # end of the plotted follow-up (h)
LN2 = float(np.log(2.0))

# name, physical half-life (h), colour
NUCLIDES = [
    ("Sm-153", 46.50, "#2e7d5b"),
    ("Y-90", 64.05, "#3b5b92"),
    ("Lu-177", 159.53, "#c9a227"),
    ("I-131", 192.48, "#8e6bbf"),
]
CARRIER = "Lu-177"
ORGAN_C = "#b23a48"

# organ, uptake fraction (-), uptake half-time (h), biological half-life (h),
# S value (mGy per MBq h)
ORGANS = [
    ("Kidneys", 0.140, 3.0, 100.0, 0.62),
    ("Liver", 0.105, 5.0, 250.0, 0.21),
    ("Spleen", 0.035, 4.0, 180.0, 0.55),
    ("Tumour", 0.085, 8.0, 520.0, 0.85),
    ("Red marrow", 0.040, 2.0, 45.0, 0.15),
]
MAIN_ORGAN = "Kidneys"


def organ_tac(t, frac, t_uptake, t_bio, t_phys):
    """Uptake, biological clearance and physical decay in one curve (MBq)."""
    k_u = LN2 / t_uptake
    k_e = LN2 / t_bio + LN2 / t_phys
    return (A0 * frac * k_u / (k_u - k_e) *
            (np.exp(-k_e * t) - np.exp(-k_u * t)))


def half_life_from_curve(t, a, lo, hi):
    """Half-life (h) recovered by a log-linear fit over ``[lo, hi]``."""
    sel = (t >= lo) & (t <= hi) & (a > 0.0)
    slope = np.polyfit(t[sel], np.log(a[sel]), 1)[0]
    return -LN2 / slope


def main():
    set_pub_style()
    t = np.linspace(0.0, T_END, 1601)
    t_long = np.linspace(0.0, 4000.0, 40001)      # for the time integrals

    fig = plt.figure(figsize=(FIGW, FIGH))
    ax = axes_inches(fig, (0.90, 1.02, 5.22, 4.34))
    ax_cum = axes_inches(fig, (7.14, 3.70, 2.86, 1.74))
    ax_dose = axes_inches(fig, (7.14, 0.95, 2.86, 1.74))
    for a in (ax, ax_cum, ax_dose):
        despine(a)

    # ---- main panel: physical decay of four nuclides ----------------------
    t_phys_carrier = dict((n, h) for n, h, _ in NUCLIDES)[CARRIER]
    for name, t_half, col in NUCLIDES:
        a = A0 * np.exp(-LN2 / t_half * t)
        fitted = half_life_from_curve(t, a, 0.0, T_END)
        ax.plot(t, a, color=col, lw=1.5, zorder=4,
                label=f"{name}, $T_{{1/2}}$ = {fitted:.0f} h")

    # ---- the organ curve on top of its own physical line ------------------
    name, frac, t_up, t_bio, s_val = [o for o in ORGANS
                                      if o[0] == MAIN_ORGAN][0]
    tac = organ_tac(t, frac, t_up, t_bio, t_phys_carrier)
    phys_only = A0 * frac * np.exp(-LN2 / t_phys_carrier * t)
    t_eff_theory = 1.0 / (1.0 / t_bio + 1.0 / t_phys_carrier)
    t_eff = half_life_from_curve(t, tac, 120.0, T_END)

    ax.plot(t, phys_only, color=ORGAN_C, lw=0.9, ls=(0, (1.6, 1.8)),
            alpha=0.85, zorder=5,
            label=f"{MAIN_ORGAN}, physical decay only")
    ax.plot(t, tac, color=ORGAN_C, lw=2.0, zorder=6,
            label=f"{MAIN_ORGAN} in vivo, {CARRIER}")
    ax.annotate(f"effective $T_{{1/2}}$ = {t_eff:.0f} h  <  "
                f"biological {t_bio:.0f} h  <  physical "
                f"{t_phys_carrier:.0f} h",
                xy=(340.0, float(np.interp(340.0, t, tac))),
                xytext=(42.0, 1.45), fontsize=6.6, color=ORGAN_C, ha="left",
                va="center", linespacing=1.6,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=ORGAN_C))
    i_pk = int(np.argmax(tac))
    ax.plot([t[i_pk]], [tac[i_pk]], marker="o", ms=4.0, mfc="white",
            mec=ORGAN_C, mew=1.2, zorder=7)
    ax.annotate(f"peak {tac[i_pk]:.0f} MBq at {t[i_pk]:.0f} h",
                xy=(t[i_pk], tac[i_pk]), xytext=(14.0, 8.0),
                textcoords="offset points", fontsize=6.4, color=ORGAN_C,
                ha="left", va="bottom")

    ax.set_yscale("log")
    ax.set_xlim(0.0, T_END)
    ax.set_ylim(1.0, 3000.0)
    ax.set_xticks(np.arange(0, 401, 50))
    ax.set_xlabel("time after administration (h)", fontsize=8.0)
    ax.set_ylabel("activity (MBq, log scale)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.tick_params(axis="y", which="minor", length=1.8, color=HAIR)
    ax.grid(True, which="major", color="#ececec", lw=0.5, zorder=0)
    ax.grid(True, which="minor", axis="y", color="#f5f5f5", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.985, 0.975, f"administered activity {A0:.0f} MBq",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
            color=SUB)
    leg = ax.legend(loc="lower left", ncol=3, frameon=False,
                    bbox_to_anchor=(-0.005, 1.005), fontsize=6.8,
                    handlelength=1.7, columnspacing=1.3,
                    handletextpad=0.45, labelspacing=0.35)
    for h in leg.legend_handles:
        h.set_linewidth(1.7)

    # ---- aux 1 and 2: cumulated activity and absorbed dose ----------------
    names = [o[0] for o in ORGANS]
    cum, dose = [], []
    for nm, fr, tu, tb, sv in ORGANS:
        curve = organ_tac(t_long, fr, tu, tb, t_phys_carrier)
        a_tilde = float(np.trapezoid(curve, t_long))       # MBq h
        cum.append(a_tilde)
        dose.append(a_tilde * sv / A0)                     # mGy per MBq
    order = np.argsort(cum)[::-1]
    names = [names[i] for i in order]
    cum = [cum[i] for i in order]
    dose = [dose[i] for i in order]
    cols = [ORGAN_C if n == MAIN_ORGAN else "#3b5b92" for n in names]
    pos = np.arange(len(names), dtype=float)

    ax_cum.barh(pos, cum, height=0.62, color=cols, edgecolor="white",
                linewidth=0.5, zorder=3)
    for i, v in enumerate(cum):
        ax_cum.text(v + 0.02 * max(cum), i, f"{v / 1000.0:.1f}",
                    va="center", ha="left", fontsize=6.2, color=SUB)
    ax_cum.set_yticks(pos)
    ax_cum.set_yticklabels(names, fontsize=6.4)
    ax_cum.invert_yaxis()
    ax_cum.set_xlim(0.0, 1.20 * max(cum))
    ax_cum.set_xticks(np.arange(0, int(1.2 * max(cum)) + 1, 5000))
    ax_cum.set_xticklabels([f"{v // 1000:.0f}"
                            for v in np.arange(0, int(1.2 * max(cum)) + 1,
                                               5000)], fontsize=6.5)
    ax_cum.tick_params(axis="y", length=0, pad=2.0)
    ax_cum.set_xlabel("cumulated activity (10$^3$ MBq h)", fontsize=7.2)
    ax_cum.set_title("Area under each organ curve", fontsize=8.5,
                     fontweight="bold", color=INK, pad=4.0)

    ax_dose.barh(pos, dose, height=0.62, color=cols, edgecolor="white",
                 linewidth=0.5, zorder=3)
    for i, v in enumerate(dose):
        ax_dose.text(v + 0.02 * max(dose), i, f"{v:.2f}", va="center",
                     ha="left", fontsize=6.2, color=SUB)
    ax_dose.set_yticks(pos)
    ax_dose.set_yticklabels(names, fontsize=6.4)
    ax_dose.invert_yaxis()
    ax_dose.set_xlim(0.0, 1.20 * max(dose))
    ax_dose.tick_params(axis="x", labelsize=6.5)
    ax_dose.tick_params(axis="y", length=0, pad=2.0)
    ax_dose.set_xlabel("absorbed dose (mGy per MBq administered)",
                       fontsize=7.2)
    ax_dose.set_title("Cumulated activity times the organ S value",
                      fontsize=8.5, fontweight="bold", color=INK, pad=4.0)



    panel_labels_inches(fig, [ax], labels=["(a)"], dx=-0.64, dy=0.50)
    panel_labels_inches(fig, [ax_cum, ax_dose], labels=["(b)", "(c)"],
                        dx=-0.64, dy=0.22)
    savefig(fig, "875_nmd_dosimetry_decay.png")


if __name__ == "__main__":
    main()

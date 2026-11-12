"""Target heat-flux footprint across a detachment scan, and its two limits.

Each profile is an Eich footprint - an exponential decay of width lambda_q
convolved with a Gaussian of width S, evaluated as q(s) = (q0/2) exp((S /
2 lambda_q)^2 - s / (lambda_q f_x)) erfc(S / 2 lambda_q - s / (S f_x)) + q_bg,
then given a little diagnostic noise.  The widths quoted on the figure are not
the widths that generated the curves: every profile is re-fitted with the same
five-parameter Eich form by least squares, so lambda_q, S and the peak load
are recovered the way an experiment would recover them, and the recovery error
of the attached case is printed in the footnote as a check.

Raising the radiated power fraction is the only knob turned across the family.
It broadens S, shifts the peak away from the strike point and drops q_peak, so
the second panel - peak load against radiated fraction, with the engineering
limit crossed at a fraction read off that curve - is a summary of the main
panel rather than new data.  The third panel closes the loop on the material:
a semi-infinite tungsten slab under each peak load, T(t) = T0 + 2 q sqrt(t /
pi k rho c), against the recrystallisation temperature.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.special import erfcx

from pubstyle import set_pub_style, savefig, lighten, get_cmap
from figkit import axes_inches, panel_labels_inches

RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3b5b92"
PURPLE = "#8e6bbf"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

S_TARGET = np.linspace(-9.0, 33.0, 900)      # mm along the target
F_X = 4.6                                    # flux expansion at the target
Q_BG = 0.32                                  # MW/m^2 background
LIMIT = 10.0                                 # MW/m^2 engineering limit

#      f_rad,  q0 (MW/m^2), lambda_q (mm), S (mm), label
CASES = [(0.20, 44.0, 1.05, 0.55, "attached"),
         (0.45, 30.0, 1.15, 1.15, "partial detachment"),
         (0.65, 18.5, 1.30, 2.20, "onset of detachment"),
         (0.80, 9.4, 1.50, 3.60, "detached"),
         (0.90, 4.2, 1.75, 5.40, "fully detached")]

# tungsten monoblock, semi-infinite slab response
W_K, W_RHO, W_CP = 130.0, 19250.0, 135.0     # W/m/K, kg/m^3, J/kg/K
T_BASE = 200.0                               # degC coolant-side base
T_RECRYST = 1300.0                           # degC
T_MELT = 3422.0
T_PULSE = np.linspace(0.0, 10.0, 400)        # s

FW, FH = 10.4, 6.4
MAIN = (0.86, 1.06, 4.86, 4.14)
AUX1 = (6.62, 3.30, 2.80, 1.90)
AUX2 = (6.62, 1.06, 2.80, 1.52)


# ------------------------------------------------------------------ physics
def eich(s, q0, lam, sw, s0, qbg):
    """Eich footprint: exponential decay convolved with a Gaussian.

    Written through the scaled complementary error function.  The textbook
    form multiplies exp((S/2L)^2 - s/(L f_x)) by erfc(...), and on the
    private-flux side that product is a genuine overflow times a genuine
    underflow: the profile silently collapses to zero and a least-squares fit
    of it then returns widths that are wrong by tens of percent.  Factoring
    the Gaussian out, exp(...) erfc(z) = erfcx(z) exp(-(s / S f_x)^2), keeps
    every term finite until z itself is far negative, where erfc has saturated
    at 2 and the plain exponential branch is exact.
    """
    s = np.asarray(s, dtype=float)
    lam = max(abs(float(lam)), 1e-3)
    sw = max(abs(float(sw)), 1e-3)
    a = sw / (2.0 * lam)
    u = (s - s0) / (sw * F_X)
    z = a - u
    out = np.empty_like(s)
    near = z > -20.0
    out[near] = 0.5 * q0 * erfcx(z[near]) * np.exp(-u[near] ** 2)
    far = ~near
    out[far] = q0 * np.exp(a ** 2 - (s[far] - s0) / (lam * F_X))
    return out + qbg


def make_family(rng):
    out = []
    for frad, q0, lam, sw, name in CASES:
        clean = eich(S_TARGET, q0, lam, sw, 0.0, Q_BG)
        noisy = clean + rng.normal(0.0, 0.012 * clean.max() + 0.05,
                                   S_TARGET.size)
        p0 = [clean.max(), 1.0, 1.0, 0.0, Q_BG]
        try:
            fit, _ = curve_fit(eich, S_TARGET, noisy, p0=p0, maxfev=40000)
        except RuntimeError:
            fit = np.array(p0)
        model = eich(S_TARGET, *fit)
        k = int(np.argmax(model))
        out.append(dict(frad=frad, name=name, clean=clean, noisy=noisy,
                        model=model, q0=abs(fit[0]), lam=abs(fit[1]),
                        sw=abs(fit[2]), s0=fit[3], qbg=fit[4],
                        peak=float(model[k]), s_peak=float(S_TARGET[k]),
                        true_lam=lam, true_s=sw,
                        wet=float(np.trapezoid(model - fit[4], S_TARGET) /
                                  max(model[k] - fit[4], 1e-9))))
    return out


def slab_temperature(q_mw):
    """Semi-infinite slab surface temperature under a constant heat flux."""
    return T_BASE + 2.0 * (q_mw * 1e6) * np.sqrt(
        T_PULSE / (np.pi * W_K * W_RHO * W_CP))


# ------------------------------------------------------------------ panels
def main_panel(ax, fam):
    cmap = get_cmap("cb_sequential_warm")
    handles = []
    for i, c in enumerate(fam):
        col = cmap(0.88 - 0.60 * i / (len(fam) - 1))
        c["color"] = col
        ax.plot(S_TARGET, c["noisy"], color=col, lw=0.0, marker=".", ms=0.9,
                alpha=0.32, zorder=3)
        ax.plot(S_TARGET, c["model"], color=col, lw=1.8, zorder=5)
        handles.append(Line2D(
            [], [], color=col, lw=1.8,
            label="f$_{{rad}}$ = {:.2f},  {:s}   (q$_{{peak}}$ = "
                  "{:.1f} MW/m$^2$)".format(c["frad"], c["name"], c["peak"])))

    ax.axhline(LIMIT, color=RED, lw=1.2, ls=(0, (5.0, 2.6)), zorder=6)
    ax.text(S_TARGET[-1] - 0.4, LIMIT + 0.9, "steady-state engineering limit "
            "{:.0f} MW/m$^2$".format(LIMIT), ha="right", va="bottom",
            fontsize=6.4, color=RED, zorder=7)
    ax.axvline(0.0, color=SUB, lw=0.8, ls=(0, (3.0, 2.2)), zorder=2)
    ax.text(-0.5, 46.5, "strike point", ha="right", va="top", fontsize=6.2,
            color=SUB, rotation=90.0)

    a, d = fam[0], fam[-1]
    ax.plot([a["s_peak"]], [a["peak"]], marker="v", ms=4.6, color=INK,
            zorder=8)
    ax.annotate("attached peak {:.1f} MW/m$^2$\nat s = {:+.2f} mm".format(
        a["peak"], a["s_peak"]), xy=(a["s_peak"], a["peak"]),
        xytext=(3.6, a["peak"] * 1.005), ha="left", va="top", fontsize=6.4,
        color=INK, linespacing=1.3, zorder=9,
        arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))

    y = a["peak"] * 0.66
    lam_end = a["s0"] + a["lam"] * F_X
    ax.annotate("", xy=(a["s0"], y), xytext=(lam_end, y), zorder=9,
                arrowprops=dict(arrowstyle="<|-|>", color=BLUE, lw=0.9,
                                mutation_scale=7))
    ax.text(lam_end + 0.6, y, "fitted $\\lambda_q f_x$ = {:.2f} mm\n"
            "($\\lambda_q$ = {:.2f} mm at the midplane, f$_x$ = "
            "{:.1f})".format(a["lam"] * F_X, a["lam"], F_X), ha="left",
            va="center", fontsize=6.3, color=BLUE, linespacing=1.3, zorder=9,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="none", alpha=0.88))

    y2 = d["peak"] * 1.35
    ax.annotate("", xy=(d["s0"] - d["sw"], y2),
                xytext=(d["s0"] + d["sw"], y2), zorder=9,
                arrowprops=dict(arrowstyle="<|-|>", color=PURPLE, lw=0.9,
                                mutation_scale=7))
    ax.annotate("Gaussian spreading of the fully detached case,\n"
                "fitted S = {:.2f} mm against {:.2f} mm attached".format(
                    d["sw"], a["sw"]),
                xy=(d["s0"] + d["sw"], y2), xytext=(11.0, 6.6), ha="left",
                va="bottom", fontsize=6.3, color=PURPLE, linespacing=1.3,
                zorder=9, bbox=dict(boxstyle="round,pad=0.22",
                                    facecolor="white", edgecolor="none",
                                    alpha=0.88),
                arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.7))

    ax.annotate("radiating the power away broadens S by {:.1f}x\n"
                "and cuts the peak load by {:.0f} %".format(
                    d["sw"] / a["sw"], 100.0 * (1.0 - d["peak"] / a["peak"])),
                xy=(d["s_peak"] + 3.4, d["peak"] * 1.05), xytext=(16.5, 20.5),
                ha="left", va="bottom", fontsize=6.4, color=ORANGE,
                linespacing=1.35, zorder=9,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.7))

    ax.set_xlim(S_TARGET[0], S_TARGET[-1])
    ax.set_ylim(0.0, 48.0)
    ax.set_xlabel("distance along the target from the strike point s (mm)")
    ax.set_ylabel("target heat-flux density q$_\\perp$ (MW/m$^2$)")
    ax.tick_params(labelsize=6.9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", lw=0.5, alpha=0.45)
    ax.set_title("Divertor target footprint across a detachment scan",
                 fontsize=9.0, fontweight="bold", color=INK, pad=5.0)
    handles.append(Line2D([], [], color=SUB, lw=0.0, marker=".", ms=3.0,
                          alpha=0.5, label="synthetic IR measurement, "
                                           "solid line is the Eich fit"))
    ax.legend(handles=handles, loc="upper right", fontsize=6.3,
              frameon=False, bbox_to_anchor=(0.998, 0.998), handlelength=1.8,
              labelspacing=0.32, borderaxespad=0.0)


def rollover_panel(ax, fam):
    fr = np.array([c["frad"] for c in fam])
    pk = np.array([c["peak"] for c in fam])
    sw = np.array([c["sw"] for c in fam])
    grid = np.linspace(fr.min(), fr.max(), 300)
    curve = np.exp(np.interp(grid, fr, np.log(pk)))

    ax.plot(grid, curve, color=ORANGE, lw=1.7, zorder=5)
    ax.plot(fr, pk, marker="o", ms=5.0, mfc="white", mec=ORANGE, mew=1.3,
            ls="none", zorder=6)
    ax.axhline(LIMIT, color=RED, lw=1.0, ls=(0, (4.4, 2.4)), zorder=4)
    f_cross = float(np.interp(-LIMIT, -curve, grid))
    ax.plot([f_cross], [LIMIT], marker="X", ms=6.0, color=RED, zorder=7)
    ax.annotate("limit met at f$_{{rad}}$ = {:.2f}".format(f_cross),
                xy=(f_cross, LIMIT), xytext=(f_cross - 0.03, LIMIT * 2.6),
                ha="right", va="bottom", fontsize=6.3, color=RED, zorder=8,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))

    axs = ax.twinx()
    axs.plot(fr, sw, color=PURPLE, lw=1.2, ls=(0, (4.0, 2.2)), marker="s",
             ms=3.4, zorder=5)
    axs.set_ylabel("fitted S (mm)", fontsize=7.2, color=PURPLE)
    axs.tick_params(axis="y", colors=PURPLE, labelsize=6.6)
    axs.set_ylim(0.0, sw.max() * 1.30)
    axs.spines["right"].set_color(PURPLE)
    axs.spines["top"].set_visible(False)
    axs.spines["left"].set_visible(False)

    ax.set_yscale("log")
    ax.set_xlim(0.15, 0.95)
    ax.set_ylim(2.5, 80.0)
    ax.set_xlabel("radiated power fraction f$_{rad}$ (-)", fontsize=7.4)
    ax.set_ylabel("peak load q$_{peak}$ (MW/m$^2$)", fontsize=7.4,
                  color=ORANGE)
    ax.tick_params(axis="y", colors=ORANGE, labelsize=6.6)
    ax.tick_params(axis="x", labelsize=6.6)
    ax.set_yticks([3, 10, 30])
    ax.set_yticklabels(["3", "10", "30"])
    ax.spines["left"].set_color(ORANGE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", lw=0.5, alpha=0.45)
    ax.set_title("Peak load rolls over as radiation rises", fontsize=8.2,
                 fontweight="bold", color=INK)
    return f_cross


def slab_panel(ax, fam):
    cmap = get_cmap("cb_sequential_warm")
    for i, c in enumerate(fam):
        col = cmap(0.88 - 0.60 * i / (len(fam) - 1))
        t = slab_temperature(c["peak"])
        ax.plot(T_PULSE, t, color=col, lw=1.5, zorder=5)
        ax.text(9.85, min(t[-1], 3820.0), "{:.1f}".format(c["peak"]),
                ha="left", va="center", fontsize=6, color=col,
                clip_on=False)
    ax.axhline(T_RECRYST, color=RED, lw=1.0, ls=(0, (4.4, 2.4)), zorder=6)
    ax.text(0.15, T_RECRYST + 110.0, "W recrystallisation "
            "{:.0f} $^\\circ$C".format(T_RECRYST), ha="left", va="bottom",
            fontsize=6.1, color=RED)
    ax.axhline(T_MELT, color="#4d4d4d", lw=1.0, ls=(0, (2.6, 2.0)), zorder=6)
    ax.text(0.15, T_MELT + 110.0, "W melting {:.0f} $^\\circ$C".format(
        T_MELT), ha="left", va="bottom", fontsize=6.1, color="#4d4d4d")
    t_att = slab_temperature(fam[0]["peak"])
    t_cross = float(np.interp(T_RECRYST, t_att, T_PULSE))
    ax.plot([t_cross], [T_RECRYST], marker="X", ms=5.4, color=RED, zorder=8)
    ax.text(2.6, T_RECRYST - 170.0,
            "attached case reaches it in {:.2f} s".format(t_cross),
            ha="left", va="top", fontsize=6.1, color=RED, linespacing=1.25,
            zorder=9, bbox=dict(boxstyle="round,pad=0.20",
                                facecolor="white", edgecolor="none",
                                alpha=0.88))
    ax.text(9.85, 120.0, "q$_{peak}$\n(MW/m$^2$)", ha="left", va="bottom",
            fontsize=6, color=SUB, linespacing=1.2, clip_on=False)

    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 4100.0)
    ax.set_xlabel("time under load t (s)", fontsize=7.4)
    ax.set_ylabel("surface temperature ($^\\circ$C)", fontsize=7.4)
    ax.tick_params(labelsize=6.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", lw=0.5, alpha=0.45)
    ax.set_title("What the tungsten surface does about it", fontsize=8.2,
                 fontweight="bold", color=INK)
    return t_cross


def main():
    set_pub_style()
    rng = np.random.default_rng(921)
    fam = make_family(rng)

    fig = plt.figure(figsize=(FW, FH))
    ax_m = axes_inches(fig, MAIN)
    ax_r = axes_inches(fig, AUX1)
    ax_s = axes_inches(fig, AUX2)
    main_panel(ax_m, fam)
    f_cross = rollover_panel(ax_r, fam)
    t_cross = slab_panel(ax_s, fam)

    a, d = fam[0], fam[-1]
    err_lam = 100.0 * abs(a["lam"] - a["true_lam"]) / a["true_lam"]
    err_s = 100.0 * abs(a["sw"] - a["true_s"]) / a["true_s"]
    panel_labels_inches(fig, [ax_m, ax_r, ax_s], dx=-0.62, dy=0.13)
    savefig(fig, "921_fus_divertor_heatflux.png")


if __name__ == "__main__":
    main()

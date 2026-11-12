"""Maillard browning as a consecutive reaction, integrated rather than drawn.

One five-species ODE system - sugar plus amino acid condensing to an Amadori
intermediate, the intermediate partitioning into melanoidin and HMF, and
asparagine reacting with the same intermediate pool to acrylamide that then
degrades - is solved with ``solve_ivp`` at every temperature on the page.  The
bell shape of the intermediate is therefore not drawn by hand: it is what a
consecutive A -> B -> C system does, and it is the reason browning lags sugar
loss.  Instrumental colour on the second axis is a fixed transform of the
melanoidin trace from the same solution, so L* cannot fall before pigment
forms.  Sweeping the same integrator over twelve bake temperatures with
Arrhenius rate constants gives the browning-rate panel, whose slope recovers
an activation energy, and the acrylamide panel, whose maximum near 180 C
appears because formation and thermal loss have different activation
energies - again a consequence of the model, not an inserted peak.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

GREEN = "#2e7d5b"        # reducing sugar
BLUE = "#3b5b92"         # free amino acid
GOLD = "#c9a227"         # Amadori intermediate
BROWN = "#8c4a12"        # melanoidin
OCHRE = "#c0603f"        # HMF
RED = "#b23a48"          # acrylamide
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

R_GAS = 8.314
T_REF = 423.15                   # K, 150 C reference for every rate constant
#      k_ref (units below)  Ea (J/mol)
K_COND = (0.00018, 100000.0)     # L/(mmol min), sugar + amine -> Amadori
K_MEL = (0.0580, 125000.0)       # 1/min, Amadori -> melanoidin
K_HMF = (0.0220, 140000.0)       # 1/min, Amadori -> HMF
K_ACR = (0.00060, 155000.0)      # L/(mmol min), Amadori + Asn -> acrylamide
K_LOSS = (0.00120, 200000.0)     # 1/min, acrylamide elimination

S0, A0, N0 = 100.0, 80.0, 12.0   # mmol/L glucose, lysine, asparagine
BAKE_MIN = 60.0
T_MAIN_C = 150.0
T_SWEEP_C = np.linspace(120.0, 210.0, 13)
ACR_MW = 71.08                   # g/mol, for the mg/kg conversion
SOLIDS = 0.32                    # kg dry matter per L of dough slurry

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 4.62, 4.48)
AUX_T = (7.44, 3.61, 2.68, 1.89)
AUX_B = (7.44, 1.02, 2.68, 1.82)


def arrh(pair, temp_c):
    k_ref, ea = pair
    return k_ref * np.exp(-ea / R_GAS * (1.0 / (temp_c + 273.15) -
                                         1.0 / T_REF))


def solve(temp_c, t_end=BAKE_MIN, n=600):
    """Integrate the five-species scheme at one temperature."""
    k1 = arrh(K_COND, temp_c)
    k2 = arrh(K_MEL, temp_c)
    k3 = arrh(K_HMF, temp_c)
    k4 = arrh(K_ACR, temp_c)
    k5 = arrh(K_LOSS, temp_c)

    def rhs(_t, y):
        s, a, i, _m, _h, n_asn, acr = y
        cond = k1 * s * a
        acryl = k4 * i * n_asn
        return [-cond, -cond, cond - k2 * i - k3 * i, k2 * i, k3 * i,
                -acryl, acryl - k5 * acr]

    t = np.linspace(0.0, t_end, n)
    sol = solve_ivp(rhs, (0.0, t_end), [S0, A0, 0.0, 0.0, 0.0, N0, 0.0],
                    t_eval=t, rtol=1e-8, atol=1e-10, method="LSODA")
    return t, sol.y


def colour_from(mel):
    """L* a* b* as one fixed transform of the melanoidin trace."""
    f = mel / (0.30 * S0)
    lab_l = 92.0 - 46.0 * (1.0 - np.exp(-2.1 * f))
    lab_a = -1.5 + 21.0 * f * np.exp(-0.55 * f)
    lab_b = 14.0 + 44.0 * f * np.exp(-0.95 * f)
    return lab_l, lab_a, lab_b


def main_panel(ax, t, y, rng):
    s, a, i, m, h = y[0], y[1], y[2], y[3], y[4]
    ax_c = ax.twinx()
    ax.set_zorder(2)
    ax.patch.set_visible(False)
    ax_c.set_zorder(1)

    lab_l, lab_a, lab_b = colour_from(m)
    for arr, col, lab in ((lab_l, MUTED, "$L^*$"), (lab_a, RED, "$a^*$"),
                          (lab_b, GOLD, "$b^*$")):
        ax_c.plot(t, arr, color=col, lw=1.0, ls=(0, (4.5, 2.0)), zorder=2)
        ax_c.text(BAKE_MIN - 0.6, arr[-1], lab, ha="right", va="bottom",
                  fontsize=7.0, color=col, zorder=3)

    ax.fill_between(t, 0.0, i, color=lighten(GOLD, 0.80), lw=0.0, zorder=3)
    for arr, col, lab in ((s, GREEN, "reducing sugar"),
                          (a, BLUE, "free amino acid"),
                          (i, GOLD, "Amadori intermediate"),
                          (m, BROWN, "melanoidin pigment"),
                          (h, OCHRE, "HMF")):
        ax.plot(t, arr, color=col, lw=1.7, zorder=6, label=lab)

    t_s = np.linspace(5.0, BAKE_MIN, 8)
    for arr, col in ((s, GREEN), (m, BROWN)):
        obs = np.interp(t_s, t, arr) * rng.normal(1.0, 0.045, t_s.size)
        ax.plot(t_s, obs, marker="o", ms=3.4, mfc="white", mec=col, mew=0.9,
                ls="none", zorder=7)

    j = int(np.argmax(i))
    ax.annotate("intermediate maximum\n{:.1f} mmol/L at {:.1f} min".format(
        i[j], t[j]), xy=(t[j], i[j]), xytext=(t[j] + 5.0, i[j] + 13.0),
        ha="left", va="bottom", fontsize=6.7, color=GOLD, linespacing=1.30,
        zorder=9, arrowprops=dict(arrowstyle="-", color=GOLD, lw=0.8))
    ax.annotate("", xy=(t[j], 0.0), xytext=(t[j], i[j]), zorder=8,
                arrowprops=dict(arrowstyle="-", color=GOLD, lw=0.7,
                                linestyle=(0, (2.0, 1.8))))
    k = int(np.argmin(np.abs(m - 0.5 * m[-1])))
    ax.annotate("pigment half-formed\nat {:.1f} min".format(t[k]),
                xy=(t[k], m[k]), xytext=(t[k] + 5.0, m[k] + 15.0),
                ha="left", va="bottom", fontsize=6.7, color=BROWN,
                linespacing=1.30, zorder=9,
                arrowprops=dict(arrowstyle="-", color=BROWN, lw=0.8))

    ax.set_xlim(0.0, BAKE_MIN)
    ax.set_ylim(0.0, 1.20 * S0)
    ax_c.set_ylim(-12.0, 118.0)
    ax.set_xlabel("bake time at {:.0f} $^\\circ$C (min)".format(T_MAIN_C))
    ax.set_ylabel("concentration (mmol/L)")
    ax_c.set_ylabel("CIE $L^*$, $a^*$, $b^*$ (-)", color=SUB)
    ax_c.tick_params(axis="y", colors=SUB, labelsize=6.8)
    ax_c.spines["right"].set_color(HAIR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["left"].set_visible(False)
    ax.grid(lw=0.5, alpha=0.45)
    handles = ax.get_legend_handles_labels()[0] + [
        Line2D([], [], color=SUB, lw=1.0, ls=(0, (4.5, 2.0)),
               label="CIE colour, right axis"),
        Line2D([], [], color=SUB, lw=0.0, marker="o", ms=3.6, mfc="white",
               mec=SUB, mew=0.9, label="sampled assay, 4.5 % scatter")]
    ax.legend(handles=handles, loc="upper right", fontsize=6.8, frameon=False,
              bbox_to_anchor=(0.995, 0.995), handlelength=1.7,
              borderaxespad=0.0, labelspacing=0.30)
    ax.set_title("Consecutive kinetics: the intermediate peaks, then the "
                 "pigment arrives", fontsize=9.0, fontweight="bold",
                 color=INK, pad=6.0)


def rate_panel(ax, temps, rate_max, ea_fit):
    inv = 1000.0 / (temps + 273.15)
    ax.plot(inv, rate_max, color=BROWN, lw=1.4, marker="o", ms=3.6,
            mfc="white", mec=BROWN, mew=0.9, zorder=5)
    coef = np.polyfit(inv, np.log(rate_max), 1)
    grid = np.linspace(inv.min(), inv.max(), 40)
    ax.plot(grid, np.exp(coef[1] + coef[0] * grid), color=INK, lw=0.9,
            ls=(0, (4.0, 2.0)), zorder=4)
    ax.text(0.035, 0.055,
            "Arrhenius slope over 120-210 $^\\circ$C\n"
            "$E_a$ = {:.0f} kJ/mol".format(ea_fit / 1000.0),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.6,
            color=INK, linespacing=1.40, zorder=7)
    for tc in (140.0, 180.0):
        x = 1000.0 / (tc + 273.15)
        ax.annotate("{:.0f} $^\\circ$C".format(tc),
                    xy=(x, np.interp(x, inv[::-1], rate_max[::-1])),
                    xytext=(0.0, 7.0), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.0, color=SUB,
                    zorder=7)
    ax.set_yscale("log")
    ax.set_xlabel("$10^3/T$ (1/K)", fontsize=7.4)
    ax.set_ylabel("max browning rate\n(mmol/(L min))", fontsize=7.4)
    ax.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Browning rate against temperature", fontsize=8.0,
                 fontweight="bold", color=INK)


def acrylamide_panel(ax, temps, acr_mg):
    ax.fill_between(temps, 0.0, acr_mg, color=lighten(RED, 0.84), lw=0.0,
                    zorder=2)
    ax.plot(temps, acr_mg, color=RED, lw=1.6, zorder=5)
    j = int(np.argmax(acr_mg))
    ax.plot([temps[j]], [acr_mg[j]], marker="o", ms=4.6, mfc=RED,
            mec="white", mew=0.8, zorder=6)
    ax.annotate("maximum {:.0f} $\\mu$g/kg\nat {:.0f} $^\\circ$C".format(
        acr_mg[j], temps[j]), xy=(temps[j], acr_mg[j]), xytext=(0.0, 6.0),
        textcoords="offset points", ha="center", va="bottom", fontsize=6.4,
        color=RED, linespacing=1.30, zorder=7)
    ax.axhline(500.0, color=SUB, lw=0.9, ls=(0, (3.5, 2.0)), zorder=4)
    ax.text(temps[0] + 1.0, 520.0, "500 $\\mu$g/kg benchmark", ha="left",
            va="bottom", fontsize=6.0, color=SUB, zorder=6)
    ax.set_xlim(temps[0], temps[-1])
    ax.set_ylim(0.0, float(acr_mg.max()) * 1.30)
    ax.set_xlabel("bake temperature ($^\\circ$C)", fontsize=7.4)
    ax.set_ylabel("acrylamide at 60 min\n($\\mu$g/kg dry matter)",
                  fontsize=7.4)
    ax.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Hotter is not always more acrylamide", fontsize=8.0,
                 fontweight="bold", color=INK)


def audit(fig, foot, axes):
    """Report any in-view tick or axis label the footnote would sit on."""
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
    rng = np.random.default_rng(915)

    t, y = solve(T_MAIN_C)
    rate_max, acr_mg = [], []
    for tc in T_SWEEP_C:
        tt, yy = solve(tc)
        rate_max.append(float(np.gradient(yy[3], tt).max()))
        acr_mg.append(1e3 * ACR_MW * yy[6][-1] / (1000.0 * SOLIDS))
    rate_max = np.array(rate_max)
    acr_mg = np.array(acr_mg)

    inv = 1000.0 / (T_SWEEP_C + 273.15)
    slope = np.polyfit(inv, np.log(rate_max), 1)[0]
    ea_fit = -slope * R_GAS * 1000.0

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_rate = axes_inches(fig, AUX_T)
    ax_acr = axes_inches(fig, AUX_B)

    main_panel(ax_main, t, y, rng)
    rate_panel(ax_rate, T_SWEEP_C, rate_max, ea_fit)
    acrylamide_panel(ax_acr, T_SWEEP_C, acr_mg)

    j = int(np.argmax(y[2]))
    lab_l, _a, _b = colour_from(y[3])
    jm = int(np.argmax(acr_mg))
    panel_labels_inches(fig, [ax_main, ax_rate, ax_acr], dx=-0.62, dy=0.14)
    savefig(fig, "915_fud_maillard_kinetics.png")


if __name__ == "__main__":
    main()

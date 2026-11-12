"""Cyclic voltammetry of a quasi-reversible one-electron couple.

(a) Main panel: the voltammogram family, one closed loop per scan rate,
    coloured by scan rate with a matching colour bar.  Nothing here is drawn
    by hand: each loop comes from an explicit finite-difference solution of
    semi-infinite linear diffusion with a Butler-Volmer boundary condition, so
    the duck shape, the peak positions and the growth of the peak separation
    with scan rate all fall out of the simulation.
(b) Auxiliary panel: the Randles-Sevcik test - peak current against the square
    root of scan rate - with the least-squares line and its statistics.
(c) Auxiliary panel: anodic-to-cathodic peak separation against scan rate,
    against the 59 mV that a fully reversible couple would show.

Anodic current is positive, cathodic negative (IUPAC).  Illustrative values.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches, safe_colorbar

FARADAY = 96485.0        # C mol-1
RGAS = 8.314             # J mol-1 K-1
TEMP = 298.15            # K
FRT = FARADAY / (RGAS * TEMP)

D_COEF = 1.0e-5          # diffusion coefficient, cm2 s-1
C_BULK = 1.0e-6          # bulk concentration, mol cm-3 (1 mM)
K0 = 1.2e-2              # standard rate constant, cm s-1
ALPHA = 0.5              # transfer coefficient
E_HIGH, E_LOW = 0.35, -0.35     # switching potentials, V vs E0'
E_FORMAL = 0.0           # formal potential, V vs RHE (shifted below)
E_OFFSET = 0.42          # places the couple on a plausible RHE scale, V

RATES = np.array([10.0, 25.0, 50.0, 100.0, 200.0, 400.0])   # mV s-1
LAMBDA = 0.40            # explicit-scheme stability number
N_STEP = 5000            # time steps per full cycle

REV = "#b23a48"          # house red: the reversible reference


def simulate(v_mv):
    """One closed voltammogram; returns potential (V) and current (mA cm-2)."""
    v = v_mv / 1000.0                       # V s-1
    t_total = 2.0 * (E_HIGH - E_LOW) / v
    dt = t_total / N_STEP
    dx = np.sqrt(D_COEF * dt / LAMBDA)
    n_x = int(6.0 * np.sqrt(D_COEF * t_total) / dx) + 2

    c_o = np.full(n_x, C_BULK)
    c_r = np.zeros(n_x)
    d_dx = D_COEF / dx

    e_rec = np.empty(N_STEP)
    j_rec = np.empty(N_STEP)
    for k in range(N_STEP):
        t = (k + 0.5) * dt
        e = (E_HIGH - v * t if t <= t_total / 2.0
             else E_LOW + v * (t - t_total / 2.0))

        c_o[1:-1] += LAMBDA * (c_o[2:] - 2.0 * c_o[1:-1] + c_o[:-2])
        c_r[1:-1] += LAMBDA * (c_r[2:] - 2.0 * c_r[1:-1] + c_r[:-2])

        k_f = K0 * np.exp(-ALPHA * FRT * (e - E_FORMAL))        # reduction
        k_b = K0 * np.exp((1.0 - ALPHA) * FRT * (e - E_FORMAL))  # oxidation
        surf = c_o[1] + c_r[1]                  # equal D: the sum has no flux
        c_o[0] = (d_dx * c_o[1] + k_b * surf) / (d_dx + k_f + k_b)
        c_r[0] = surf - c_o[0]

        flux = k_f * c_o[0] - k_b * c_r[0]      # mol cm-2 s-1, reduction > 0
        e_rec[k] = e + E_OFFSET
        j_rec[k] = -1000.0 * FARADAY * flux     # mA cm-2, anodic positive
    return e_rec, j_rec


def main():
    set_pub_style(base_font=9)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.85, 1.02, 4.95, 4.62))
    axs = axes_inches(fig, (7.32, 3.74, 2.88, 1.90))
    axb = axes_inches(fig, (7.32, 1.02, 2.88, 1.90))

    cmap = mpl.colormaps["viridis"]
    norm = mpl.colors.LogNorm(vmin=RATES.min(), vmax=RATES.max())

    runs = []
    for v in RATES:
        e, j = simulate(v)
        half = N_STEP // 2
        k_c = int(np.argmin(j[:half]))          # cathodic peak, forward scan
        k_a = int(np.argmax(j[half:])) + half   # anodic peak, reverse scan
        runs.append(dict(v=v, e=e, j=j, ep_c=e[k_c], jp_c=j[k_c],
                         ep_a=e[k_a], jp_a=j[k_a],
                         d_ep=1000.0 * (e[k_a] - e[k_c])))

    span = float(np.max([np.abs(r["j"]).max() for r in runs]))

    # ---------------- (a) the voltammogram family -------------------------
    ax.axhline(0.0, color="#cccccc", lw=0.7, zorder=1)
    for r in runs:
        ax.plot(r["e"], r["j"], color=cmap(norm(r["v"])), lw=1.25, zorder=3)

    fast, slow = runs[-1], runs[0]
    ax.plot([fast["ep_c"]], [fast["jp_c"]], marker="o", ms=4.6, mfc="white",
            mec=REV, mew=1.1, zorder=6)
    ax.plot([fast["ep_a"]], [fast["jp_a"]], marker="o", ms=4.6, mfc="white",
            mec=REV, mew=1.1, zorder=6)
    ax.annotate("anodic peak\n$E_{pa}$ = %.3f V,  $j_{pa}$ = %+.2f mA cm$^{-2}$"
                % (fast["ep_a"], fast["jp_a"]),
                xy=(fast["ep_a"], fast["jp_a"]),
                xytext=(E_LOW + E_OFFSET + 0.02, 0.82 * span),
                fontsize=6.8, color=REV, ha="left", va="center",
                linespacing=1.40,
                arrowprops=dict(arrowstyle="-|>", color=REV, lw=0.8,
                                mutation_scale=8, shrinkB=3.0))
    ax.annotate("cathodic peak\n$E_{pc}$ = %.3f V,  $j_{pc}$ = %+.2f mA cm$^{-2}$"
                % (fast["ep_c"], fast["jp_c"]),
                xy=(fast["ep_c"], fast["jp_c"]),
                xytext=(E_HIGH + E_OFFSET - 0.01, -0.80 * span),
                fontsize=6.8, color=REV, ha="right", va="center",
                linespacing=1.40,
                arrowprops=dict(arrowstyle="-|>", color=REV, lw=0.8,
                                mutation_scale=8, shrinkB=3.0))
    y_sep = 1.12 * span
    ax.annotate("", xy=(fast["ep_a"], y_sep), xytext=(fast["ep_c"], y_sep),
                arrowprops=dict(arrowstyle="<->", color="#555555", lw=0.8,
                                mutation_scale=7, shrinkA=0, shrinkB=0))
    ax.text((fast["ep_a"] + fast["ep_c"]) / 2.0, y_sep + 0.02 * span,
            "$\\Delta E_p$ = %.0f mV at %.0f mV s$^{-1}$"
            % (fast["d_ep"], fast["v"]), fontsize=6.6, color="#555555",
            ha="center", va="bottom")
    ax.text(0.025, 0.965, "scan starts at the positive limit and returns;\n"
            "loop area grows with scan rate", transform=ax.transAxes,
            fontsize=6.6, color="#777777", ha="left", va="top",
            linespacing=1.45)

    ax.set_xlim(E_LOW + E_OFFSET - 0.03, E_HIGH + E_OFFSET + 0.03)
    ax.set_ylim(-1.32 * span, 1.38 * span)
    ax.set_xlabel("Potential $E$ (V vs RHE)")
    ax.set_ylabel("Current density $j$ (mA cm$^{-2}$)")
    ax.tick_params(labelsize=7.2)
    ax.grid(color="#f2f2f2", lw=0.5, zorder=0)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = safe_colorbar(fig, sm, ax, label="scan rate $\\nu$ (mV s$^{-1}$)",
                       width=0.12, pad=0.09, ticks=RATES)
    cb.ax.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(
        ["%.0f" % v for v in RATES]))
    cb.ax.minorticks_off()

    # ---------------- (b) Randles-Sevcik ----------------------------------
    root_v = np.sqrt(RATES / 1000.0)
    jp_c = np.array([-r["jp_c"] for r in runs])
    jp_a = np.array([r["jp_a"] for r in runs])
    slope, icept = np.polyfit(root_v, jp_c, 1)
    pred = slope * root_v + icept
    r2 = 1.0 - np.sum((jp_c - pred) ** 2) / np.sum((jp_c - jp_c.mean()) ** 2)

    xf = np.linspace(0.0, root_v.max() * 1.10, 40)
    axs.plot(xf, slope * xf + icept, color="#555555", lw=1.0,
             ls=(0, (3.4, 2.4)), zorder=3)
    axs.plot(root_v, jp_a, ls="none", marker="s", ms=3.8, mfc="white",
             mec="#3b5b92", mew=0.9, zorder=5, label="anodic $|j_{pa}|$")
    for x, y, v in zip(root_v, jp_c, RATES):
        axs.plot([x], [y], marker="o", ms=4.4, mfc=cmap(norm(v)), mec="white",
                 mew=0.6, zorder=6)
    axs.plot([], [], ls="none", marker="o", ms=4.4, mfc=cmap(norm(RATES[3])),
             mec="white", label="cathodic $|j_{pc}|$")
    axs.text(0.035, 0.955, "$|j_p|$ = %.2f $\\nu^{1/2}$ + %.3f\n"
             "$R^2$ = %.4f,  $n$ = %d" % (slope, icept, r2, root_v.size),
             transform=axs.transAxes, fontsize=6.4, color="#333333",
             ha="left", va="top", linespacing=1.45)
    axs.set_xlim(0.0, root_v.max() * 1.12)
    axs.set_ylim(0.0, max(jp_a.max(), jp_c.max()) * 1.30)
    axs.set_xlabel("$\\nu^{1/2}$ ((V s$^{-1}$)$^{1/2}$)", fontsize=8,
                   labelpad=1.5)
    axs.set_ylabel("$|j_p|$ (mA cm$^{-2}$)", fontsize=8, labelpad=1.5)
    axs.tick_params(labelsize=6.5)
    axs.grid(color="#ededed", lw=0.5, zorder=0)
    axs.legend(loc="lower right", fontsize=6.4, handlelength=1.2,
               labelspacing=0.30, borderaxespad=0.35)

    # ---------------- (c) peak separation ---------------------------------
    d_ep = np.array([r["d_ep"] for r in runs])
    axb.axhline(2.303 * 1000.0 / FRT, color=REV, lw=1.0, ls=(0, (3.0, 2.2)),
                zorder=4)
    axb.text(0.035, 0.955, "dashed: reversible limit %.0f mV"
             % (2.303 * 1000.0 / FRT), transform=axb.transAxes, fontsize=6.3,
             color=REV, ha="left", va="top")
    axb.plot(RATES, d_ep, color="#555555", lw=1.0, zorder=3)
    for v, d in zip(RATES, d_ep):
        axb.plot([v], [d], marker="o", ms=4.6, mfc=cmap(norm(v)), mec="white",
                 mew=0.6, zorder=5)
    axb.annotate("+%.0f mV over the decade\nfrom %.0f to %.0f mV s$^{-1}$"
                 % (d_ep[-1] - d_ep[1], RATES[1], RATES[-1]),
                 xy=(RATES[-1], d_ep[-1]), xytext=(RATES[0] * 1.02,
                                                   d_ep.max() * 1.20),
                 fontsize=6.3, color="#555555", ha="left", va="top",
                 linespacing=1.40,
                 arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.7,
                                 mutation_scale=7, shrinkB=3.0))
    axb.set_xscale("log")
    axb.set_xlim(RATES.min() * 0.75, RATES.max() * 1.35)
    axb.set_ylim(0.0, d_ep.max() * 1.35)
    axb.set_xticks(RATES)
    axb.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(
        ["%.0f" % v for v in RATES]))
    axb.minorticks_off()
    axb.set_xlabel("Scan rate $\\nu$ (mV s$^{-1}$)", fontsize=8, labelpad=1.5)
    axb.set_ylabel("$\\Delta E_p$ (mV)", fontsize=8, labelpad=1.5)
    axb.tick_params(labelsize=6.5)
    axb.grid(axis="y", color="#ededed", lw=0.5, zorder=0)

    panel_labels_inches(fig, [ax, axs, axb], dx=-0.36, dy=0.10)

    savefig(fig, "764_ec_cv_curves.png")


if __name__ == "__main__":
    main()

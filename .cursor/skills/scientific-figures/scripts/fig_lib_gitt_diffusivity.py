"""GITT of an NMC811 cathode: pulse train, one zoomed pulse, and D_Li(SOC).

(a) Cell voltage against experiment time for a discharge GITT.  Current
    pulses are the grey bars on a twin axis; the OCV envelope is the voltage
    at the end of each rest.
(b) One pulse at mid-SOC, with the IR drop and the rest used for D_Li
    marked on the same trace.
(c) Lithium diffusivity against SOC from the Weppner-Huggins intercept of
    each rest.  Values are computed from the plotted pulses.

Illustrative synthetic transients.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
RED = "#b23a48"
GOLD = "#c9a227"
INK = "#1a1a1a"
MUTED = "#777777"
SEED = 1215
N_PULSE = 18
T_PULSE = 600.0
T_REST = 3600.0
I_PULSE = 0.12
L_CM = 2.5e-4  # characteristic diffusion length, cm


def ocv(soc):
    x = np.clip(soc, 0.05, 0.95)
    return (3.55 + 0.70 * x
            + 0.14 * np.tanh((x - 0.25) / 0.06)
            + 0.10 * np.tanh((x - 0.62) / 0.08))


def d_true(soc):
    return 1.6e-10 * np.exp(-((soc - 0.45) / 0.28) ** 2) + 3.5e-11


def main():
    set_pub_style(base_font=9)
    rng = np.random.default_rng(SEED)
    socs = np.linspace(0.92, 0.18, N_PULSE)
    t_all, v_all, i_all = [], [], []
    d_fit = []
    t0 = 0.0
    traces = []
    for k, soc in enumerate(socs):
        t_p = np.linspace(0, T_PULSE, 80)
        ir = 0.018 + 0.006 * (1.0 - soc)
        dV = abs(ocv(soc) - ocv(soc - 0.04))
        pol = 0.028 * np.sqrt(8.0e-11 / d_true(soc))
        v_p = ocv(soc) - ir - pol * np.sqrt(np.clip(t_p / T_PULSE, 0, 1))
        t_r = np.linspace(0, T_REST, 120)
        v_inf = ocv(soc - 0.04)
        v_r = v_p[-1] + (v_inf - v_p[-1]) * (1.0 - np.exp(-t_r / 780.0))
        v_r = v_r + rng.normal(0, 0.0004, v_r.size)
        t_seg = np.concatenate([t0 + t_p, t0 + T_PULSE + t_r])
        v_seg = np.concatenate([v_p, v_r]) + rng.normal(0, 0.0003, 80 + 120)
        i_seg = np.concatenate([np.full(80, I_PULSE), np.zeros(120)])
        t_all.append(t_seg)
        v_all.append(v_seg)
        i_all.append(i_seg)
        traces.append((t_p, v_p, t_r, v_r, ir, soc))
        dE_t = abs(v_p[8] - v_p[-1])
        d_li = ((4.0 / np.pi) * (L_CM ** 2 / T_PULSE)
                * (dV / max(dE_t, 1e-4)) ** 2)
        d_fit.append(d_li)
        t0 += T_PULSE + T_REST

    t = np.concatenate(t_all) / 3600.0
    v = np.concatenate(v_all)
    i = np.concatenate(i_all)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.85, 1.02, 5.35, 4.62))
    axb = axes_inches(fig, (7.30, 3.74, 2.90, 1.90))
    axc = axes_inches(fig, (7.30, 1.02, 2.90, 1.90))

    ax.plot(t, v, color=BLUE, lw=1.15, zorder=4)
    axi = ax.twinx()
    axi.fill_between(t, 0, i, color=GOLD, alpha=0.28, lw=0, zorder=1)
    axi.set_ylabel("Current (C-rate)", color=GOLD)
    axi.tick_params(axis="y", labelsize=7.2, colors=GOLD)
    axi.spines["top"].set_visible(False)
    axi.spines["right"].set_color(GOLD)
    axi.set_ylim(0, 0.42)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Cell voltage (V)", color=BLUE)
    ax.tick_params(axis="y", labelsize=7.2, colors=BLUE)
    ax.tick_params(axis="x", labelsize=7.2)
    ax.spines["left"].set_color(BLUE)
    ax.spines["top"].set_visible(False)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(3.35, 4.45)
    ax.set_title("Discharge GITT, 18 pulses", fontsize=9.0, fontweight="bold",
                 pad=6)
    ocv_end = [tr[3][-1] for tr in traces]
    t_end = np.arange(N_PULSE) * (T_PULSE + T_REST) / 3600.0 + (T_PULSE + T_REST) / 3600.0
    ax.plot(t_end, ocv_end, color=RED, ls="none", marker="o", ms=3.4,
            zorder=5, label="rest OCV")
    ax.legend(loc="upper right", frameon=False, fontsize=7.0)

    k = 8
    tp, vp, tr, vr, ir, soc_k = traces[k]
    tt = np.concatenate([tp, T_PULSE + tr]) / 60.0
    vv = np.concatenate([vp, vr])
    axb.plot(tt, vv, color=BLUE, lw=1.4)
    axb.axvline(T_PULSE / 60.0, color=MUTED, lw=0.7, ls=(0, (2.4, 1.6)))
    axb.annotate("IR = {:.0f} mV".format(1e3 * ir),
                 xy=(0.4, vp[0] - 0.4 * ir), fontsize=6.4, color=RED,
                 ha="left")
    axb.set_xlabel("Time (min)", fontsize=8, labelpad=1.5)
    axb.set_ylabel("Voltage (V)", fontsize=8, labelpad=1.5)
    axb.tick_params(labelsize=6.5)
    axb.set_title("Pulse at SOC = {:.0f} %".format(100 * soc_k),
                  fontsize=8.5, fontweight="bold", pad=4)
    for side in ("top", "right"):
        axb.spines[side].set_visible(False)

    axc.semilogy(100 * socs, d_fit, color=BLUE, marker="o", ms=4.0, lw=1.4,
                 mfc="white", mew=0.8)
    axc.set_xlabel("State of charge (%)", fontsize=8, labelpad=1.5)
    axc.set_ylabel("$D_{\\mathrm{Li}}$ (cm$^{2}$ s$^{-1}$)", fontsize=8,
                   labelpad=1.5)
    axc.tick_params(labelsize=6.5)
    axc.grid(color="#ededed", lw=0.5, which="both", zorder=0)
    for side in ("top", "right"):
        axc.spines[side].set_visible(False)
    axc.set_title("Weppner-Huggins $D_{\\mathrm{Li}}$", fontsize=8.5,
                  fontweight="bold", pad=4)
    axc.text(0.04, 0.08, "median = {:.1e} cm$^{{2}}$/s".format(np.median(d_fit)),
             transform=axc.transAxes, fontsize=6.4, color=INK, ha="left")

    panel_labels_inches(fig, [ax, axb, axc], dx=-0.34, dy=0.10)
    savefig(fig, "1215_lib_gitt_diffusivity.png")


if __name__ == "__main__":
    main()

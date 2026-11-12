"""Incremental capacity and differential voltage of an NMC811/graphite cell.

(a) Charge voltage against capacity at beginning of life and after 500
    cycles.  The lost capacity is the horizontal gap at 4.2 V, read from
    the two arrays.
(b) Incremental capacity dQ/dV of the same two charge half-cycles, with
    the principal graphite and NMC peaks labelled at the voltages where
    the plotted curves actually peak.
(c) Differential voltage dV/dQ against capacity, the complementary view
    used to separate electrode slippage from loss of active material.

Illustrative synthetic voltage curves.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
RED = "#b23a48"
INK = "#1a1a1a"
MUTED = "#777777"
SEED = 1218


def v_charge(q, qmax, shift=0.0, lam_loss=0.0):
    """Smooth NMC811/graphite charge curve, V vs capacity in A h."""
    x = np.clip(q / qmax, 1e-4, 0.999)
    v = (3.48 + 0.55 * x
         + 0.10 * np.tanh((x - 0.08 - shift) / 0.025)
         + 0.07 * np.tanh((x - 0.22 - shift) / 0.03)
         + 0.09 * np.tanh((x - 0.48 - shift) / 0.04)
         + 0.16 * np.tanh((x - 0.88) / 0.035)
         - lam_loss * x)
    return v


def smooth(y, w=9):
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


def main():
    set_pub_style(base_font=9)
    rng = np.random.default_rng(SEED)
    q_bol = np.linspace(0.02, 4.82, 900)
    q_eol = np.linspace(0.02, 4.18, 900)
    v_bol_s = v_charge(q_bol, 4.85)
    v_eol_s = v_charge(q_eol, 4.22, shift=0.035, lam_loss=0.04)
    v_bol = v_bol_s + rng.normal(0, 0.0008, q_bol.size)
    v_eol = v_eol_s + rng.normal(0, 0.0008, q_eol.size)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.85, 1.02, 5.35, 4.62))
    axb = axes_inches(fig, (7.30, 3.74, 2.90, 1.90))
    axc = axes_inches(fig, (7.30, 1.02, 2.90, 1.90))

    ax.plot(q_bol, v_bol, color=BLUE, lw=1.6, label="fresh")
    ax.plot(q_eol, v_eol, color=RED, lw=1.6, label="500 cycles")
    dq = q_bol[-1] - q_eol[-1]
    ax.annotate("{:.2f} A h lost".format(dq),
                xy=(q_eol[-1], 4.20), xytext=(q_eol[-1] - 1.15, 4.32),
                fontsize=7.0, color=RED, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.8,
                                mutation_scale=8, shrinkB=3.0))
    ax.set_xlabel("Capacity (A h)")
    ax.set_ylabel("Cell voltage (V)")
    ax.set_xlim(0, 5.1)
    ax.set_ylim(3.35, 4.42)
    ax.legend(loc="upper left", frameon=False, fontsize=7.2)
    ax.tick_params(labelsize=7.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Charge half-cycle, C/10, 25 $^\\circ$C", fontsize=9.0,
                 fontweight="bold", pad=6)

    ica_b = np.clip(smooth(np.gradient(q_bol, v_bol_s), 21), 0.0, None)
    ica_e = np.clip(smooth(np.gradient(q_eol, v_eol_s), 21), 0.0, None)
    axb.plot(v_bol_s, ica_b, color=BLUE, lw=1.3)
    axb.plot(v_eol_s, ica_e, color=RED, lw=1.3)
    k_pk = int(np.argmax(ica_b[(v_bol_s > 3.5) & (v_bol_s < 3.8)]))
    v_win = v_bol_s[(v_bol_s > 3.5) & (v_bol_s < 3.8)]
    axb.axvline(v_win[k_pk], color=MUTED, lw=0.6, ls=(0, (2.2, 1.6)))
    axb.set_xlabel("Voltage (V)", fontsize=8, labelpad=1.5)
    axb.set_ylabel("d$Q$/d$V$ (A h V$^{-1}$)", fontsize=8, labelpad=1.5)
    axb.set_xlim(3.40, 4.25)
    axb.set_ylim(0.0, None)
    axb.tick_params(labelsize=6.5)
    axb.grid(color="#ededed", lw=0.5, zorder=0)
    for side in ("top", "right"):
        axb.spines[side].set_visible(False)
    axb.set_title("Incremental capacity", fontsize=8.5, fontweight="bold",
                  pad=4)

    dva_b = np.clip(smooth(np.gradient(v_bol_s, q_bol), 21), 0.0, None)
    dva_e = np.clip(smooth(np.gradient(v_eol_s, q_eol), 21), 0.0, None)
    axc.plot(q_bol, dva_b, color=BLUE, lw=1.3)
    axc.plot(q_eol, dva_e, color=RED, lw=1.3)
    axc.set_xlabel("Capacity (A h)", fontsize=8, labelpad=1.5)
    axc.set_ylabel("d$V$/d$Q$ (V A$^{-1}$ h$^{-1}$)", fontsize=8, labelpad=1.5)
    axc.set_xlim(0, 5.1)
    axc.tick_params(labelsize=6.5)
    axc.grid(color="#ededed", lw=0.5, zorder=0)
    for side in ("top", "right"):
        axc.spines[side].set_visible(False)
    axc.set_title("Differential voltage", fontsize=8.5, fontweight="bold",
                  pad=4)

    panel_labels_inches(fig, [ax, axb, axc], dx=-0.34, dy=0.10)
    savefig(fig, "1218_lib_dva_ica.png")


if __name__ == "__main__":
    main()

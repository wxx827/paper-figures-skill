"""Where the fidelity goes: channel budget, code comparison, syndrome record.

Left, the three dominant channels are decayed separately and then combined,
so the reader can see which one sets the usable circuit duration; the two
quoted crossing times are interpolated from the combined curve rather than
typed in.  Middle, the logical error of three surface-code distances is
accumulated over depth against the unencoded reference on a log axis, and the
depth at which each one hits the one-percent budget is read off the same
arrays.  Right, the detection-event record that a decoder actually consumes.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, get_cmap, get_palette, savefig
from figkit import panel_labels_inches, safe_colorbar

RED = "#b23a48"         # the combined budget: the one emphasis of the figure
BLUE = "#3b5b92"
PURPLE = "#8e6bbf"
GOLD = "#c9a227"
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"

T1, T2, T_DEP = 72.0, 41.0, 95.0        # microseconds
P_PHYS = 1.0e-3                          # physical two-qubit error per layer
P_TH = 1.0e-2                            # surface-code threshold
A_SUP = 0.03                             # sub-threshold suppression prefactor
DISTANCES = (3, 5, 7)
N_ROUNDS, N_STAB = 24, 8
WARM = get_cmap("cb_sequential_warm")


def channels(t):
    """Each channel keeps a coherence factor; the budget multiplies them."""
    c1, c2, cd = np.exp(-t / T1), np.exp(-t / T2), np.exp(-t / T_DEP)
    return (0.5 + 0.5 * c1, 0.5 + 0.5 * c2, 0.25 + 0.75 * cd,
            0.25 + 0.75 * c1 * c2 * cd)


def logical_error(depth):
    """Per-round logical error below threshold, accumulated over depth."""
    out = {d: 1.0 - (1.0 - A_SUP * (P_PHYS / P_TH) ** ((d + 1) / 2.0)) ** depth
           for d in DISTANCES}
    out["phys"] = 1.0 - (1.0 - P_PHYS) ** depth
    return out


def syndrome(seed=21):
    rng = np.random.default_rng(seed)
    base = rng.random((N_STAB, N_ROUNDS)) * 0.045 + 0.015
    # A defect chain: two neighbouring stabilisers fire together for a while.
    base[3, 8:15] += 0.16
    base[4, 8:15] += 0.13
    base[6, 17:21] += 0.10
    return np.clip(base, 0.0, 1.0)


def fidelity_panel(ax):
    t = np.linspace(0.0, 80.0, 600)
    f_t1, f_t2, f_dp, comb = channels(t)
    for y, color, name in [(f_dp, PURPLE, "depolarising"),
                           (f_t1, BLUE, r"$T_1$ relaxation"),
                           (f_t2, GOLD, r"$T_2$ dephasing")]:
        ax.plot(t, y * 100.0, color=color, lw=1.3, label=name)
    ax.plot(t, comb * 100.0, color=RED, lw=2.0, label="combined budget",
            zorder=5)
    ax.axhline(25.0, color=GREY, lw=0.9, ls=(0, (4, 2.6)), zorder=2)
    ax.text(1.5, 26.2, "fully mixed floor", ha="left", va="bottom",
            fontsize=6.6, color=GREY)

    t90 = float(np.interp(90.0, comb[::-1] * 100.0, t[::-1]))
    t50 = float(np.interp(50.0, comb[::-1] * 100.0, t[::-1]))
    for level, tx in [(90.0, t90), (50.0, t50)]:
        ax.plot([tx], [level], marker="o", ms=3.6, color=RED, zorder=6)
    ax.annotate(f"50 % at {t50:.0f} $\\mu$s", xy=(t50, 50.0),
                xytext=(t50 + 5.0, 62.0), fontsize=6.8, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))

    ax.set_xlabel(r"circuit duration $t$ ($\mu$s)")
    ax.set_ylabel("state fidelity $F$ (%)")
    ax.set_xlim(0.0, 80.0)
    ax.set_ylim(20.0, 102.0)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.8, frameon=False, loc="upper right",
              handlelength=1.5, labelspacing=0.3)
    ax.set_title(f"90 % window lasts {t90:.1f} $\\mu$s", fontsize=8.8,
                 fontweight="bold", color=INK)


def code_panel(ax):
    depth = np.logspace(0.0, 4.6, 600)
    curves = logical_error(depth)
    shades = get_palette("cb_sequential_blue", 6)[2:5][::-1]

    ax.loglog(depth, curves["phys"], color=GREY, lw=1.1, ls=(0, (4, 2.6)),
              label="unencoded")
    for d, color in zip(DISTANCES, shades):
        ax.loglog(depth, curves[d], color=color, lw=1.5, label=f"$d$ = {d}")
    ax.axhline(1.0e-2, color=RED, lw=0.9, ls=(0, (2.4, 2.0)), zorder=2)
    ax.text(1.2, 1.2e-2, "1 % failure budget", fontsize=6.6, color=RED,
            ha="left", va="bottom")

    reach = {k: float(np.interp(1.0e-2, v, depth)) for k, v in curves.items()}
    for k in list(DISTANCES) + ["phys"]:
        ax.plot([reach[k]], [1.0e-2], marker="o", ms=3.0, color=RED,
                zorder=6)
    d_top = DISTANCES[-1]
    ax.text(reach[d_top] * 1.35, 1.5e-2,
            f"$d$ = {d_top}:  {reach[d_top]:.0f} layers", ha="left",
            va="bottom", fontsize=6.8, color=INK)
    gain = reach[d_top] / reach["phys"]

    ax.set_xlabel("circuit depth (layers)")
    ax.set_ylabel("logical error rate (-)")
    ax.set_xlim(1.0, 4.0e4)
    ax.set_ylim(1.0e-7, 1.0)
    ax.grid(which="both", lw=0.4, alpha=0.55)
    ax.legend(fontsize=6.8, frameon=False, loc="upper left",
              handlelength=1.5, labelspacing=0.3)
    ax.set_title(f"Distance {d_top} buys {gain:.0f}x depth", fontsize=8.8,
                 fontweight="bold", color=INK)


def syndrome_panel(fig, ax, s):
    labels = ([rf"$Z_{i + 1}$" for i in range(4)] +
              [rf"$X_{i + 1}$" for i in range(4)])
    im = ax.imshow(s * 100.0, cmap=WARM, aspect="auto", origin="lower",
                   vmin=0.0, vmax=float(s.max() * 100.0))
    ax.set_yticks(np.arange(N_STAB))
    ax.set_yticklabels(labels, fontsize=6.4)
    ax.set_xticks(np.arange(0, N_ROUNDS, 4))
    ax.tick_params(labelsize=6.4, length=2.0)
    ax.set_xlabel("measurement round (-)", fontsize=7.4)
    ax.set_ylabel("stabiliser", fontsize=7.4)

    row = int(np.argmax(s.mean(axis=1)))
    fired = np.flatnonzero(s[row] > 0.10)
    start = int(fired.min()) if fired.size else 0
    ax.set_title(f"Defect chain from round {start}", fontsize=8.2,
                 fontweight="bold", color=INK)
    safe_colorbar(fig, im, ax, label="detection event rate (%)", width=0.09,
                  pad=0.06, labelsize=6.4)


def main():
    set_pub_style()
    s = syndrome()

    # Single row of three panels: h/w = 0.44 is exception E8 in DESIGN_SYSTEM
    # section 1.2 - squeezing a three-column comparison into the 0.55 band
    # turns each panel into a vertical sliver.
    fig = plt.figure(figsize=(10.4, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.94, 0.74], wspace=0.34)
    ax_f = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[0, 2])

    fidelity_panel(ax_f)
    code_panel(ax_c)
    syndrome_panel(fig, ax_s, s)

    fig.tight_layout(rect=[0, 0.045, 1, 0.935], w_pad=2.6)
    panel_labels_inches(fig, [ax_f, ax_c, ax_s], dx=-0.42, dy=0.12)
    savefig(fig, "496_qc_noise_error.png")


if __name__ == "__main__":
    main()

"""Digital filter characterisation: magnitude, group delay, poles/zeros, IR."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from scipy import signal

from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

FS = 500.0          # Hz
FC = 80.0           # passband-edge frequency, Hz
RP = 0.5            # passband ripple, dB
RS = 55.0           # stopband attenuation, dB
ORDERS = [4, 6, 8]
COLORS = ["#7fb2d4", "#3579a8", "#123f5e"]


def design(order):
    return signal.ellip(order, RP, RS, FC / (FS / 2), btype="low",
                        output="sos")


def main():
    set_pub_style(base_font=8)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    fig.subplots_adjust(left=0.088, right=0.975, top=0.915, bottom=0.150,
                        wspace=0.30, hspace=0.42)
    ax_mag, ax_gd = axes[0]
    ax_pz, ax_ir = axes[1]

    filters = [(n, design(n)) for n in ORDERS]

    # ---- (a) magnitude response -------------------------------------------
    for (n, sos), col in zip(filters, COLORS):
        w, h = signal.sosfreqz(sos, worN=8192, fs=FS)
        ax_mag.plot(w, 20 * np.log10(np.abs(h) + 1e-12), color=col, lw=1.4,
                    label=f"order {n}")
    ax_mag.axvline(FC, color="#b23a48", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax_mag.axhline(-RS, color="#8a8a8a", lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax_mag.axhspan(-RP, 0, color="#c9b458", alpha=0.30, lw=0, zorder=0)
    ax_mag.set_xlim(0, FS / 2)
    ax_mag.set_ylim(-95, 12)
    ax_mag.set_xlabel("Frequency (Hz)")
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.set_yticks([0, -20, -40, -60, -80])
    ax_mag.text(FC + 4, 5, f"$f_c$ = {FC:.0f} Hz", fontsize=6.8,
                color="#b23a48", fontweight="bold", va="center")
    ax_mag.text(6, -RS + 3.0, f"$A_s$ = {RS:.0f} dB", fontsize=6.8,
                color="#666666", ha="left", va="bottom")
    style_axes(ax_mag)
    add_panel_label(ax_mag, "(a)", x=-0.135, y=1.02, size=10)

    # passband-ripple inset
    axi = ax_mag.inset_axes([0.53, 0.60, 0.44, 0.32])
    for (n, sos), col in zip(filters, COLORS):
        w, h = signal.sosfreqz(sos, worN=8192, fs=FS)
        axi.plot(w, 20 * np.log10(np.abs(h) + 1e-12), color=col, lw=1.0)
    axi.axhspan(-RP, 0, color="#c9b458", alpha=0.30, lw=0)
    axi.set_xlim(0, FC * 1.02)
    axi.set_ylim(-0.85, 0.25)
    axi.set_yticks([0, -0.5])
    axi.set_xticks([0, FC / 2, FC])
    axi.tick_params(labelsize=6.0, length=1.8, pad=1.2)
    axi.set_title(f"passband ripple {RP} dB", fontsize=6.2, pad=2.0,
                  color="#8a7420")
    for s in axi.spines.values():
        s.set_linewidth(0.5)
        s.set_visible(True)
        s.set_edgecolor("#999999")

    # ---- (b) group delay ---------------------------------------------------
    gds = []
    for (n, sos), col in zip(filters, COLORS):
        b, a = signal.sos2tf(sos)
        w, gd = signal.group_delay((b, a), w=4096, fs=FS)
        gd = gd / FS * 1e3
        gds.append((w, gd, col))
        ax_gd.plot(w, gd, color=col, lw=1.4, label=f"order {n}")
    ax_gd.axvline(FC, color="#b23a48", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax_gd.set_xlim(0, FS / 2)
    ax_gd.set_ylim(0, None)
    ax_gd.set_xlabel("Frequency (Hz)")
    ax_gd.set_ylabel("Group delay (ms)")
    ax_gd.text(FC + 5, ax_gd.get_ylim()[1] * 0.97, "phase distortion\n"
               "at the band edge", fontsize=6.6, color="#b23a48", va="top",
               linespacing=1.3)
    style_axes(ax_gd)
    add_panel_label(ax_gd, "(b)", x=-0.135, y=1.02, size=10)

    axg = ax_gd.inset_axes([0.47, 0.40, 0.50, 0.38])
    pb_max = 0.0
    for w, gd, col in gds:
        axg.plot(w, gd, color=col, lw=1.0)
        pb_max = max(pb_max, gd[w <= FC * 0.92].max())
    axg.set_xlim(0, FC * 0.92)
    axg.set_ylim(0, pb_max * 1.18)
    axg.set_xticks([0, FC / 2])
    axg.tick_params(labelsize=6.0, length=1.8, pad=1.2)
    axg.set_title("passband detail", fontsize=6.2, pad=2.0, color="#666666")
    for s in axg.spines.values():
        s.set_linewidth(0.5)
        s.set_visible(True)
        s.set_edgecolor("#999999")

    # ---- (c) pole-zero plot ------------------------------------------------
    ax_pz.add_patch(Circle((0, 0), 1.0, fill=False, ec="#9a9a9a", lw=0.8,
                           ls=(0, (4, 3)), zorder=1))
    ax_pz.axhline(0, color="#dcdcdc", lw=0.6, zorder=0)
    ax_pz.axvline(0, color="#dcdcdc", lw=0.6, zorder=0)
    for (n, sos), col in zip(filters, COLORS):
        z, p, _ = signal.sos2zpk(sos)
        ax_pz.plot(z.real, z.imag, "o", ms=5.0, mfc="none", mec=col, mew=1.0,
                   ls="none", zorder=3)
        ax_pz.plot(p.real, p.imag, "x", ms=4.6, mec=col, mew=1.2, ls="none",
                   zorder=4)
    ax_pz.set_xlim(-1.32, 1.32)
    ax_pz.set_ylim(-1.20, 1.20)
    ax_pz.set_aspect("equal")
    ax_pz.set_xlabel("Real part")
    ax_pz.set_ylabel("Imaginary part")
    ax_pz.text(-1.26, 1.10, "unit circle", fontsize=6.6, color="#8a8a8a",
               va="top")
    style_axes(ax_pz)
    pz_handles = [
        Line2D([], [], marker="o", ls="none", ms=5, mfc="none", mec="#333333",
               mew=1.0, label="zeros"),
        Line2D([], [], marker="x", ls="none", ms=4.6, mec="#333333", mew=1.2,
               label="poles"),
    ]
    ax_pz.legend(handles=pz_handles, loc="upper right", fontsize=6.8,
                 handletextpad=0.4, labelspacing=0.25, borderaxespad=0.15)
    add_panel_label(ax_pz, "(c)", x=-0.135, y=1.02, size=10)

    # ---- (d) impulse and step response -------------------------------------
    nsamp = 110
    imp = signal.unit_impulse(nsamp)
    tt = np.arange(nsamp) / FS * 1e3
    ax_step = ax_ir.twinx()
    for (n, sos), col in zip(filters, COLORS):
        ax_ir.plot(tt, signal.sosfilt(sos, imp), color=col, lw=1.2)
        ax_step.plot(tt, signal.sosfilt(sos, np.ones(nsamp)), color=col,
                     lw=1.0, ls=(0, (3, 2)), alpha=0.85)
    ax_ir.axhline(0, color="#bbbbbb", lw=0.6, zorder=0)
    ax_step.axhline(1, color="#bbbbbb", lw=0.6, ls=(0, (1, 2)), zorder=0)
    ax_ir.set_xlim(0, tt[-1])
    ax_step.set_ylim(-0.12, 1.42)
    ax_ir.set_xlabel("Time (ms)")
    ax_ir.set_ylabel("Impulse response  $h[n]$")
    ax_step.set_ylabel("Step response", color="#666666")
    ax_step.tick_params(axis="y", colors="#666666", labelsize=7)
    ax_step.spines["top"].set_visible(False)
    ax_step.spines["right"].set_color("#666666")
    ax_step.spines["right"].set_linewidth(0.8)
    style_axes(ax_ir)
    ax_ir.spines["right"].set_visible(False)
    ir_handles = [
        Line2D([], [], color="#333333", lw=1.2, label="impulse (left axis)"),
        Line2D([], [], color="#333333", lw=1.0, ls=(0, (3, 2)),
               label="step (right axis)"),
    ]
    ax_ir.legend(handles=ir_handles, loc="upper right", fontsize=6.8,
                 handlelength=1.6, labelspacing=0.25, borderaxespad=0.15)
    add_panel_label(ax_ir, "(d)", x=-0.135, y=1.02, size=10)

    handles = [Line2D([], [], color=c, lw=1.6, label=f"elliptic, order {n}")
               for n, c in zip(ORDERS, COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=7.4, handlelength=1.6, columnspacing=2.0,
               bbox_to_anchor=(0.53, 0.038))

    savefig(fig, "141_filter_response.png")


if __name__ == "__main__":
    main()

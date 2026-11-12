"""MOSFET characterisation triptych: semi-log transfer curves with extracted
threshold voltage and subthreshold swing, the output-curve family split into
linear and saturation regions, and the high- / low-frequency C-V pair."""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import (set_pub_style, savefig, style_axes, get_palette,
                      annotate_box)
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
RED = "#b23a48"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
BASE = "#808080"

VTH0 = 0.42          # long-channel threshold, V
DIBL = 0.060         # threshold roll-off with drain bias, V/V
NKT = 0.0745         # subthreshold ideality x thermal voltage, V
ALPHA = 2.20         # above-threshold current exponent
AMP = 1.38e-3        # current prefactor, A/um per V^ALPHA
IFLOOR = 4.0e-13     # junction leakage floor, A/um
EPS0 = 8.854e-12
COX = 24.0           # gate capacitance density, fF/um^2
VFB = -0.35          # flat-band voltage, V


def vth_of(vd):
    return VTH0 - DIBL * vd


def veff(vg, vd):
    """Smoothed overdrive; exponential below threshold, linear above."""
    return NKT * np.log1p(np.exp((vg - vth_of(vd)) / NKT))


def id_transfer(vg, vd):
    """Drain current per unit width, A/um, including GIDL and the leakage floor."""
    core = AMP * veff(vg, vd) ** ALPHA
    gidl = 3.0e-13 * 10.0 ** (-(vg + 0.05) / 0.30) * (vd / 0.9)
    return core + gidl + IFLOOR


def id_output(vd, vg):
    """Soft-knee output current, A/um: linear rise then channel-length modulation."""
    ov = veff(vg, vd)
    isat = AMP * ov ** ALPHA
    knee = np.maximum(ov, 1e-6)
    return isat * (1.0 - np.exp(-vd / (0.62 * knee))) * (1.0 + 0.09 * vd)


def extract(vg, idd):
    """Constant-current V_th and the steepest subthreshold swing, from the curve.

    The swing is read off the *rising* branch only: the GIDL shoulder at
    negative gate bias has the opposite slope and would otherwise win the
    minimum and report a negative swing.
    """
    lg = np.log10(idd)
    vth = float(np.interp(np.log10(1.0e-6), lg, vg))
    slope = np.gradient(lg, vg)                      # decades per volt
    band = (vg > vg[int(np.argmin(idd))]) & (idd > 3e-12) & (idd < 1e-8)
    return vth, 1000.0 / float(np.max(slope[band]))


def transfer_panel(ax, vg):
    curves = {0.05: BLUE, 0.90: RED}
    for vd, col in curves.items():
        idd = id_transfer(vg, vd)
        ax.semilogy(vg, idd, color=col, lw=1.6, zorder=4,
                    label=f"$V_{{ds}}$ = {vd:.2f} V")
    idd_sat = id_transfer(vg, 0.90)
    vth, ss = extract(vg, idd_sat)

    ax.axvline(vth, color=BASE, lw=0.9, ls=(0, (4, 2.5)), zorder=3)
    ax.axhline(1.0e-6, color=BASE, lw=0.7, ls=(0, (1.6, 2.0)), zorder=3)
    ax.text(vth + 0.04, 1.2e-11, f"$V_{{th}}$ = {vth:.3f} V\n"
            f"at 1 \u00b5A/\u00b5m", ha="left", va="center", fontsize=6.6,
            color=INK, linespacing=1.45)

    # subthreshold tangent, drawn with exactly the swing that was measured
    v0 = vth - 0.22
    i0 = float(np.interp(v0, vg, idd_sat))
    vt = np.array([v0 - 0.16, v0 + 0.16])
    ax.plot(vt, i0 * 10.0 ** ((vt - v0) / (ss / 1000.0)), color=GREEN, lw=1.0,
            zorder=5)
    ax.annotate(f"SS = {ss:.0f} mV/dec", xy=(v0 - 0.13, i0 * 0.26),
                xytext=(0.04, 0.55), textcoords="axes fraction", fontsize=6.8,
                color=GREEN, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.7))

    ion = float(id_transfer(np.array([1.10]), 0.90)[0])
    ioff = float(id_transfer(np.array([0.0]), 0.90)[0])
    annotate_box(ax, f"$I_{{on}}$ = {ion * 1e6:.0f} \u00b5A/\u00b5m\n"
                 f"$I_{{off}}$ = {ioff * 1e12:.0f} pA/\u00b5m\n"
                 f"ratio = {ion / ioff:.1e}", loc="lower right",
                 fontsize=6.4, linespacing=1.5)

    ax.set_xlim(vg[0], vg[-1])
    ax.set_ylim(1e-13, 3e-3)
    ax.set_xlabel("Gate voltage  $V_{gs}$ (V)", fontsize=8.0)
    ax.set_ylabel("Drain current  $I_d$ (A/\u00b5m)", fontsize=8.0)
    ax.set_title("Transfer characteristic", fontsize=8.8, fontweight="bold",
                 pad=4)
    ax.legend(loc="upper left", fontsize=7.0, frameon=False,
              handlelength=1.4, borderaxespad=0.25)
    ax.tick_params(labelsize=7.0)
    ax.yaxis.grid(True, color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)
    return vth, ss


def output_panel(ax, vd):
    vgs = np.array([0.55, 0.70, 0.85, 1.00, 1.15])
    cols = get_palette("cb_sequential_blue", len(vgs) + 2)[2:]
    knee_x, knee_y = [], []
    for vg, col in zip(vgs, cols):
        idd = id_output(vd, vg) * 1e6
        ax.plot(vd, idd, color=col, lw=1.6, zorder=4)
        ax.annotate(f"{vg:.2f} V", xy=(vd[-1], idd[-1]),
                    xytext=(-2, 1.6), textcoords="offset points",
                    ha="right", va="bottom", fontsize=6.4, color=col)
        vk = vg - VTH0                      # one fixed-point pass for DIBL
        vk = vg - vth_of(vk)
        knee_x.append(vk)
        knee_y.append(float(id_output(np.array([vk]), vg)[0]) * 1e6)
    ax.plot(knee_x, knee_y, color=RED, lw=1.0, ls=(0, (4, 2.2)), zorder=6)
    ax.plot(knee_x, knee_y, "o", ms=3.0, mfc="white", mec=RED, mew=0.9,
            zorder=7)
    ax.annotate("$V_{ds} = V_{gs} - V_{th}$", xy=(knee_x[2], knee_y[2]),
                xytext=(0.10, 0.72), textcoords="axes fraction",
                fontsize=6.8, color=RED, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    ax.text(0.045, 0.975, "linear", transform=ax.transAxes, ha="left",
            va="top", fontsize=6.8, color=SUB)
    ax.text(0.72, 0.945, "saturation", transform=ax.transAxes, ha="center",
            va="top", fontsize=6.8, color=SUB)
    ax.text(0.045, 0.905, "$V_{gs}$ step 0.15 V", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.6, color=MUTED)

    ax.set_xlim(0, vd[-1])
    ax.set_ylim(0, id_output(np.array([vd[-1]]), vgs[-1])[0] * 1e6 * 1.14)
    ax.set_xlabel("Drain voltage  $V_{ds}$ (V)", fontsize=8.0)
    ax.set_ylabel("Drain current  $I_d$ (\u00b5A/\u00b5m)", fontsize=8.0)
    ax.set_title("Output characteristic", fontsize=8.8, fontweight="bold",
                 pad=4)
    ax.tick_params(labelsize=7.0)
    ax.grid(True, color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)


def cv_panel(ax, vg, vth):
    cmin = 0.28 * COX
    c_hf = cmin + (COX - cmin) / (1.0 + np.exp((vg - VFB) / 0.16))
    c_lf = c_hf + (COX - cmin) / (1.0 + np.exp(-(vg - vth) / 0.10))
    ax.plot(vg, c_lf, color=GOLD, lw=1.6, zorder=4, label="quasi-static, 1 kHz")
    ax.plot(vg, c_hf, color=BLUE, lw=1.6, zorder=5, label="high frequency, 1 MHz")
    ax.axhline(COX, color=BASE, lw=0.9, ls=(0, (4, 2.5)), zorder=3)
    ax.text(vg[0] + 0.04, COX - 1.3, f"$C_{{ox}}$ = {COX:.0f} fF/\u00b5m$^2$",
            ha="left", va="top", fontsize=6.6, color=SUB)
    for x, lab, col in ((VFB, "$V_{fb}$", GREEN), (vth, "$V_{th}$", RED)):
        ax.axvline(x, color=col, lw=0.8, ls=(0, (2.4, 2.0)), zorder=3)
        ax.text(x - 0.035, cmin * 1.15, f"{lab} = {x:.2f} V", rotation=90,
                ha="right", va="bottom", fontsize=6.4, color=col)
    for xc, lab in (((vg[0] + VFB) / 2, "accumulation"),
                    ((VFB + vth) / 2, "depletion"),
                    ((vth + vg[-1]) / 2, "inversion")):
        ax.text(xc, COX * 0.055, lab, ha="center", va="bottom", fontsize=6.5,
                color=MUTED)

    eot = 3.9 * EPS0 / (COX * 1e-15 / 1e-12) * 1e9
    annotate_box(ax, f"EOT = {eot:.2f} nm\n$C_{{min}}/C_{{ox}}$ = "
                 f"{cmin / COX:.2f}", loc="upper right", fontsize=6.4,
                 linespacing=1.5)
    ax.set_xlim(vg[0], vg[-1])
    ax.set_ylim(0, COX * 1.22)
    ax.set_xlabel("Gate voltage  $V_{gs}$ (V)", fontsize=8.0)
    ax.set_ylabel("Gate capacitance  $C_{gg}$ (fF/\u00b5m$^2$)", fontsize=8.0)
    ax.set_title("Capacitance-voltage", fontsize=8.8, fontweight="bold", pad=4)
    ax.legend(loc="upper left", fontsize=6.8, frameon=False, handlelength=1.4,
              borderaxespad=0.25)
    ax.tick_params(labelsize=7.0)
    ax.yaxis.grid(True, color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)


def main():
    set_pub_style()
    vg = np.linspace(-0.40, 1.20, 601)
    vd = np.linspace(0.0, 1.20, 481)
    vg_cv = np.linspace(-1.20, 1.20, 601)

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.0))
    vth, ss = transfer_panel(axes[0], vg)
    output_panel(axes[1], vd)
    cv_panel(axes[2], vg_cv, vth)

    fig.tight_layout(rect=[0, 0.035, 1, 0.935], w_pad=2.6)
    panel_labels_inches(fig, axes, dx=-0.42, dy=0.08)
    savefig(fig, "499_semi_device_iv.png")


if __name__ == "__main__":
    main()

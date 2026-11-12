"""Deep brain stimulation: the amplitude / pulse-width therapeutic window.

Main panel: symptom improvement over the stimulation amplitude x pulse-width
plane as a filled warm field, with the side-effect probability of the
neighbouring capsule drawn on top as blue iso-lines and the therapeutic window
(improvement at or above 50 per cent while side effects stay at or below 20
per cent) outlined; the programmed operating point is the grid cell with the
largest improvement-minus-penalty score.  Auxiliary panels (both 58 per cent
of the main panel width): the strength-duration thresholds of the two
structures with the window between them, and a frequency sweep of benefit,
side effect and average power at the programmed amplitude.

Everything plotted comes from one pair of strength-duration curves,
I_th(PW) = I_rh (1 + chronaxie / PW), one for the therapeutic target and one
for the side-effect structure, pushed through a logistic recruitment law.  The
quoted window area is the share of plotted grid cells inside the outlined
region, the window ratio in (b) is the quotient of the two threshold curves at
each pulse width, and the volume of tissue activated is the sphere implied by
the charge per pulse at the operating point.  Synthetic illustration, not a
patient programming record.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pubstyle import set_pub_style, savefig, get_palette
from figkit import axes_inches, panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3b5b92"
TEAL = "#2a7f7a"
GREY = "#b3b3b3"
BOX = dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="none",
           alpha=0.86)

PW = np.linspace(30.0, 200.0, 205)        # pulse width, microseconds
AMP = np.linspace(0.20, 4.50, 197)        # amplitude, mA
I_RH_T, CHRON_T = 0.62, 55.0              # target: rheobase mA, chronaxie us
I_RH_S, CHRON_S = 0.78, 130.0             # side-effect structure
B_MAX = 76.0                              # ceiling of symptom improvement, %
BEN_MIN, SIDE_MAX = 50.0, 20.0            # window definition, %
R_TISSUE = 1.2e3                          # electrode-tissue impedance, ohm
F_SHOW = 130.0                            # programmed rate, Hz
PW_SHOW = 60.0                            # pulse width the clinic starts from
FREQ = np.linspace(10.0, 250.0, 241)

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.14, 3.16, 1.86)


def threshold(pw, i_rh, chron):
    """Lapicque strength-duration threshold in mA for a pulse width in us."""
    return i_rh * (1.0 + chron / pw)


def recruit(amp, i_th, ceiling, sharp):
    """Logistic recruitment of a structure above its threshold."""
    return ceiling / (1.0 + np.exp(-(amp - i_th) / (sharp * i_th)))


def fields():
    """Benefit and side-effect percentages on the pulse-width x amplitude grid."""
    pw, amp = np.meshgrid(PW, AMP)
    ben = recruit(amp, threshold(pw, I_RH_T, CHRON_T), B_MAX, 0.22)
    side = recruit(amp, threshold(pw, I_RH_S, CHRON_S), 100.0, 0.16)
    return pw, amp, ben, side


def vta_volume(amp_ma, pw_us):
    """Volume of tissue activated, mm3, from the sphere the charge fills."""
    r = 1.05 * np.sqrt(amp_ma * pw_us / 60.0)
    return 4.0 / 3.0 * np.pi * r ** 3, r


def avg_power(amp_ma, pw_us, freq_hz):
    """Mean stimulation power in mW for a rectangular pulse train."""
    return (amp_ma * 1e-3) ** 2 * R_TISSUE * (pw_us * 1e-6) * freq_hz * 1e3


def window_panel(fig, ax, pw, amp, ben, side, op):
    # The top of the warm ramp is nearly black, and the benefit field spends
    # most of the plane on its ceiling; truncating the ramp keeps the blue
    # iso-lines and the hatching readable over the saturated corner.
    warm = mcolors.LinearSegmentedColormap.from_list(
        "warm_cut", get_palette("cb_sequential_warm", 16)[:12])
    cf = ax.contourf(pw, amp, ben, levels=np.linspace(0.0, B_MAX, 14),
                     cmap=warm, zorder=1)
    lvl = [20.0, 50.0, 80.0]
    cs = ax.contour(pw, amp, side, levels=lvl, colors=BLUE,
                    linewidths=[0.9, 1.3, 0.9],
                    linestyles=[(0, (4.0, 2.0)), "-", (0, (1.6, 1.8))],
                    zorder=4)
    # Hand-placed so the labels keep clear of the callout and its leader.
    spots = []
    for p, pw_lab in zip(lvl, (185.0, 155.0, 120.0)):
        i_th = threshold(pw_lab, I_RH_S, CHRON_S)
        spots.append((pw_lab, i_th * (1.0 + 0.16 * np.log(p / (100.0 - p)))))
    labs = ax.clabel(cs, fmt="side effect %d%%", fontsize=6.0, inline=True,
                     inline_spacing=6, colors=BLUE, manual=spots)
    for t in labs:
        t.set_bbox(BOX)

    win = ((ben >= BEN_MIN) & (side <= SIDE_MAX)).astype(float)
    ax.contour(pw, amp, win, levels=[0.5], colors=[INK], linewidths=1.5,
               zorder=5)
    ax.contourf(pw, amp, win, levels=[0.5, 1.5], colors=["none"],
                hatches=["///"], zorder=3)

    ax.plot(PW, threshold(PW, I_RH_T, CHRON_T), color=RED, lw=1.2,
            ls=(0, (5.0, 2.0)), zorder=6)
    ax.text(PW[-1] - 3.0, threshold(PW[-1], I_RH_T, CHRON_T) - 0.06,
            "target threshold", fontsize=6.2, color=RED, ha="right",
            va="top", bbox=BOX, zorder=8)

    ax.plot([op["pw"]], [op["amp"]], marker="o", ms=6.5, mfc="white",
            mec=INK, mew=1.4, zorder=9)
    ax.annotate("programmed setting\n{:.0f} $\\mu$s, {:.2f} mA at {:.0f} Hz\n"
                "{:.0f} % improvement, {:.0f} % side-effect risk\n"
                "charge {:.0f} nC per pulse, VTA {:.0f} mm$^3$"
                .format(op["pw"], op["amp"], F_SHOW, op["ben"], op["side"],
                        op["charge"], op["vta"]),
                xy=(op["pw"], op["amp"]),
                xytext=(op["pw"] + 34.0, op["amp"] + 1.02), fontsize=6.4,
                color=INK, ha="left", va="center", linespacing=1.45,
                bbox=BOX, zorder=10,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8,
                                shrinkA=2, shrinkB=4))
    ax.text(PW[0] + 4.0, AMP[0] + 0.12,
            "hatched: {:.0f} % of the plane clears both criteria"
            .format(op["area"]), fontsize=6.4, color=SUB, ha="left",
            va="bottom", bbox=BOX, zorder=8)

    ax.set_xlim(PW[0], PW[-1])
    ax.set_ylim(AMP[0], AMP[-1])
    ax.set_xlabel("pulse width ($\\mu$s)")
    ax.set_ylabel("stimulation amplitude (mA, cathodic monopolar)")
    ax.tick_params(width=0.8, length=3.0)
    for side_name in ("top", "right"):
        ax.spines[side_name].set_visible(False)
    cb = safe_colorbar(fig, cf, ax, width=0.11, pad=0.09, labelsize=6.2,
                       ticks=[0.0, 20.0, 40.0, 60.0, B_MAX])
    cb.ax.set_ylabel("symptom improvement (%, UPDRS-III style score)",
                     fontsize=6.2)
    ax.set_title("Short pulses buy amplitude headroom before the capsule "
                 "answers", fontsize=9.0, fontweight="bold", color=INK,
                 pad=6.0)


def strength_duration_panel(ax, op):
    th_t = threshold(PW, I_RH_T, CHRON_T)
    th_s = threshold(PW, I_RH_S, CHRON_S)
    ax.fill_between(PW, th_t, th_s, facecolor=ORANGE, alpha=0.16,
                    edgecolor="none", zorder=2)
    ax.plot(PW, th_t, color=RED, lw=1.5, zorder=5, label="target structure")
    ax.plot(PW, th_s, color=BLUE, lw=1.5, zorder=5,
            label="side-effect structure")
    ax.plot([op["pw"]], [op["amp"]], marker="o", ms=4.6, mfc="white",
            mec=INK, mew=1.1, zorder=7)
    ax.set_xlim(PW[0], PW[-1])
    ax.set_ylim(0.0, 4.3)
    ax.set_xlabel("pulse width ($\\mu$s)", fontsize=6.8)
    ax.set_ylabel("threshold amplitude (mA)", fontsize=6.8)
    ax.tick_params(labelsize=6.0, width=0.7, length=2.5)
    ax.grid(lw=0.5, alpha=0.28)
    for s in ("top",):
        ax.spines[s].set_visible(False)

    ratio = th_s / th_t
    ax2 = ax.twinx()
    ax2.plot(PW, ratio, color=TEAL, lw=1.2, ls=(0, (3.5, 2.0)), zorder=6)
    ax2.set_ylim(1.0, 3.0)
    ax2.set_ylabel("window ratio (side / target)", fontsize=6.5, color=TEAL)
    ax2.tick_params(labelsize=6.0, width=0.7, length=2.5, colors=TEAL)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(TEAL)
    ax2.text(PW[-1] - 4.0, ratio[-1] + 0.09,
             "ratio falls from {:.2f} to {:.2f}".format(ratio[0], ratio[-1]),
             fontsize=6, color=TEAL, ha="right", va="bottom", bbox=BOX,
             zorder=8)
    ax.legend(loc="upper right", fontsize=6, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.4, borderaxespad=0.3, labelspacing=0.22)
    ax.set_title("Strength-duration thresholds, {:.0f} to {:.0f} $\\mu$s"
                 .format(PW[0], PW[-1]), fontsize=8.0, fontweight="bold",
                 color=INK)


def frequency_panel(ax, op):
    gain = 1.0 / (1.0 + np.exp(-(FREQ - 58.0) / 13.0))
    ben_f = op["ben"] * gain
    side_f = op["side"] + 26.0 * (FREQ / FREQ[-1]) ** 2.6
    pwr = avg_power(op["amp"], op["pw"], FREQ)
    ax.plot(FREQ, ben_f, color=RED, lw=1.5, zorder=6, label="improvement")
    ax.plot(FREQ, side_f, color=BLUE, lw=1.4, ls=(0, (4.0, 2.0)), zorder=6,
            label="side-effect risk")
    ax.axvline(F_SHOW, color=GREY, lw=0.9, ls=(0, (2.0, 2.0)), zorder=3)
    ax.text(F_SHOW + 5.0, 4.0, "{:.0f} Hz".format(F_SHOW), fontsize=6.0,
            color=SUB, ha="left", va="bottom", bbox=BOX, zorder=8)
    ax.set_xlim(FREQ[0], FREQ[-1])
    ax.set_ylim(0.0, 92.0)
    ax.set_xlabel("stimulation rate (Hz)", fontsize=6.8)
    ax.set_ylabel("outcome (%)", fontsize=6.8)
    ax.tick_params(labelsize=6.0, width=0.7, length=2.5)
    ax.grid(lw=0.5, alpha=0.28)
    ax.spines["top"].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(FREQ, pwr * 1e3, color=MUTED, lw=1.1, zorder=4)
    ax2.set_ylim(0.0, float(pwr[-1] * 1e3) * 1.25)
    ax2.set_ylabel("mean power ($\\mu$W)", fontsize=6.5, color=MUTED)
    ax2.tick_params(labelsize=6.0, width=0.7, length=2.5, colors=MUTED)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(MUTED)
    k = int(np.argmin(np.abs(FREQ - F_SHOW)))
    ax2.text(FREQ[0] + 6.0, pwr[-1] * 1e3 * 1.16,
             "{:.0f} $\\mu$W drawn at the programmed rate"
             .format(pwr[k] * 1e3), fontsize=6, color=MUTED, ha="left",
             va="top", bbox=BOX, zorder=8)
    ax.legend(loc="center right", fontsize=6, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.4, borderaxespad=0.4, labelspacing=0.22)
    ax.set_title("Rate sweep at {:.2f} mA and {:.0f} $\\mu$s"
                 .format(op["amp"], op["pw"]), fontsize=8.0,
                 fontweight="bold", color=INK)


def audit(fig, foot, axes):
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
    np.random.default_rng(4711)
    pw, amp, ben, side = fields()

    # The clinic fixes the pulse width first, then titrates the amplitude, so
    # the operating point is the best cell of one column rather than of the
    # whole plane (whose optimum always sits on the short-pulse edge).
    score = ben - 1.6 * side
    i = int(np.argmin(np.abs(PW - PW_SHOW)))
    j = int(np.argmax(score[:, i]))
    op_pw, op_amp = float(pw[j, i]), float(amp[j, i])
    vol, rad = vta_volume(op_amp, op_pw)
    op = dict(pw=op_pw, amp=op_amp, ben=float(ben[j, i]),
              side=float(side[j, i]), charge=op_amp * op_pw, vta=float(vol),
              radius=float(rad),
              area=100.0 * float(((ben >= BEN_MIN) &
                                  (side <= SIDE_MAX)).mean()))

    th_t = threshold(PW, I_RH_T, CHRON_T)
    th_s = threshold(PW, I_RH_S, CHRON_S)
    ratio = th_s / th_t
    print("operating point {:.0f} us {:.2f} mA -> {:.1f} % benefit, "
          "{:.1f} % side effect".format(op["pw"], op["amp"], op["ben"],
                                        op["side"]))
    print("window covers {:.1f} % of the plotted plane; VTA radius {:.2f} mm, "
          "volume {:.1f} mm3".format(op["area"], op["radius"], op["vta"]))
    print("window ratio {:.2f} at {:.0f} us falls to {:.2f} at {:.0f} us"
          .format(ratio[0], PW[0], ratio[-1], PW[-1]))
    print("mean power at {:.0f} Hz: {:.1f} uW"
          .format(F_SHOW, avg_power(op["amp"], op["pw"], F_SHOW) * 1e3))
    assert ratio[0] > ratio[-1], "short pulses widen the window"
    assert op["side"] <= SIDE_MAX, "the programmed point sits in the window"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_w = axes_inches(fig, MAIN)
    ax_s = axes_inches(fig, AUX_T)
    ax_f = axes_inches(fig, AUX_B)

    window_panel(fig, ax_w, pw, amp, ben, side, op)
    strength_duration_panel(ax_s, op)
    frequency_panel(ax_f, op)



    panel_labels_inches(fig, [ax_w, ax_s, ax_f], dx=-0.62, dy=0.14)
    savefig(fig, "1056_bci_dbs_therapeutic_window.png")


if __name__ == "__main__":
    main()

"""Seakeeping response amplitude operators for a 180 m hull.

The main panel is the roll operator swept over wave frequency for five wave
headings, the family that decides whether a ship can hold course in a beam
sea.  The resonance is not assumed: for every heading the script takes the
argmax of the curve it has just drawn and converts that frequency into a
natural period, so the marked peaks and the quoted period always agree with
the ink.  A small polar inset cuts the same surface the other way, giving the
roll response at resonance as a function of heading.

The companion panels carry the two vertical-plane operators in head seas -
heave in metres per metre of wave amplitude, pitch in degrees per metre - and
the phase lead of each motion over the wave, which is what tells a naval
architect whether the bow is going down into the crest or riding over it.

Illustrative synthetic data from linear second-order response models.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, style_axes, add_footer
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

RED = "#b23a48"
GOLD = "#c9a227"
STEEL = "#5b6b7a"
BLUE = "#2e74a8"
HEAD = ["#bcd4e6", "#8fb8d4", "#5f9dc6", "#2e74a8", "#134a76"]

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.06, 5.26, 4.34)
AUX_T = (7.50, 3.62, 3.02, 1.78)
AUX_B = (7.50, 1.06, 3.02, 1.72)

G = 9.81
L_PP = 180.0            # length between perpendiculars, m
T_DRAFT = 10.5          # mean draft, m

W_N3, Z_3 = 0.615, 0.34     # heave natural frequency (rad/s), damping ratio
W_N5, Z_5 = 0.660, 0.30     # pitch
W_N4, Z_4 = 0.420, 0.058    # roll, lightly damped
K_SLOPE = 0.76              # effective wave-slope coefficient, dimensionless

HEADINGS = [30.0, 60.0, 90.0, 120.0, 150.0]     # deg from following seas


def magnification(w, wn, zeta):
    """Dimensionless dynamic magnification of a second-order oscillator."""
    r = w / wn
    return 1.0 / np.sqrt((1.0 - r ** 2) ** 2 + (2.0 * zeta * r) ** 2)


def phase_lag(w, wn, zeta):
    """Phase lag of the response behind the excitation, deg."""
    r = w / wn
    return np.degrees(np.arctan2(2.0 * zeta * r, 1.0 - r ** 2))


def roll_rao(w, mu_deg):
    """Roll amplitude per unit wave amplitude, deg/m."""
    k = w ** 2 / G
    exc = K_SLOPE * k * abs(np.sin(np.radians(mu_deg)))
    return np.degrees(exc) * magnification(w, W_N4, Z_4)


def heave_rao(w):
    """Heave amplitude per unit wave amplitude, m/m."""
    k = w ** 2 / G
    return np.exp(-k * T_DRAFT * 0.62) * magnification(w, W_N3, Z_3)


def pitch_rao(w):
    """Pitch amplitude per unit wave amplitude, deg/m."""
    k = w ** 2 / G
    lobe = np.abs(np.sinc(k * L_PP / (2.0 * np.pi)))     # Froude-Krylov lobe
    return (np.degrees(k) * 1.42 * lobe * np.exp(-k * T_DRAFT * 0.90)
            * magnification(w, W_N5, Z_5))


def check_layout(fig, foot, axes):
    """Warn if the footnote box touches an axis label, tick row or title."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for a in axes:
        ab = a.get_window_extent()
        arts = [a.xaxis.label, a.yaxis.label, a.title]
        arts += [t for t in a.get_xticklabels()
                 if ab.x0 - 3.0 <= t.get_window_extent().x0 <= ab.x1 + 3.0]
        for art in arts:
            if art.get_text() and fb.overlaps(art.get_window_extent()):
                print("WARNING: footnote overlaps:", art.get_text()[:44])


def main():
    set_pub_style(base_font=8)

    w = np.linspace(0.16, 1.45, 1300)
    rolls = [roll_rao(w, mu) for mu in HEADINGS]
    heave = heave_rao(w)
    pitch = pitch_rao(w)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN)
    ax_v = axes_inches(fig, AUX_T)
    ax_p = axes_inches(fig, AUX_B)
    for a in (ax, ax_v, ax_p):
        a.set_axisbelow(True)
        a.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)
        style_axes(a)

    # ---- main: the roll family ---------------------------------------------
    peaks = []
    for mu, r, col in zip(HEADINGS, rolls, HEAD):
        ax.plot(w, r, color=col, lw=1.9, zorder=6,
                label="heading $\\mu$ = %.0f deg" % mu)
        i = int(np.argmax(r))
        peaks.append((mu, float(w[i]), float(r[i])))
        ax.plot([w[i]], [r[i]], marker="o", ms=4.0, mfc="white", mec=col,
                mew=1.2, zorder=8)

    mu_b, w_b, r_b = max(peaks, key=lambda p: p[2])
    t_n = 2.0 * np.pi / w_b
    ax.axvline(w_b, color=FAINT, lw=0.9, ls=(0, (3.0, 2.0)), zorder=2)
    ax.annotate("roll resonance from the plotted maximum:\n"
                "%.3f rad/s, $T_\\phi$ = %.1f s, %.2f deg/m ($\\mu$ = "
                "%.0f deg)" % (w_b, t_n, r_b, mu_b),
                xy=(w_b, r_b), xytext=(0.190, 0.985),
                textcoords="axes fraction", fontsize=6.8, color=INK,
                ha="left", va="top", linespacing=1.45, zorder=10,
                arrowprops=dict(arrowstyle="-|>", color=FAINT, lw=0.9,
                                shrinkA=2, shrinkB=4))

    limit = 6.0
    ax.axhline(limit, color=RED, lw=1.0, ls=(0, (5.0, 2.4)), zorder=3)
    ax.text(1.425, limit + 0.16, "crew-comfort limit %.0f deg/m" % limit,
            fontsize=6.4, color=RED, ha="right", va="bottom", zorder=6)
    over = w[(rolls[2] > limit)]
    ax.axvspan(float(over.min()), float(over.max()), color=RED, alpha=0.075,
               zorder=1)
    ax.annotate("beam sea above the limit over\n%.2f to %.2f rad/s "
                "(%.1f to %.1f s encounter)"
                % (over.min(), over.max(), 2.0 * np.pi / over.max(),
                   2.0 * np.pi / over.min()),
                xy=(float(over.max()), limit), xytext=(0.315, 0.505),
                textcoords="axes fraction", ha="left", va="bottom",
                fontsize=6.3, color=RED, linespacing=1.45, zorder=7,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.8,
                                shrinkA=2, shrinkB=3))

    ax.set_xlim(0.16, 1.45)
    ax.set_ylim(0.0, max(p[2] for p in peaks) * 1.20)
    ax.tick_params(labelsize=7.2)
    ax.set_xlabel("wave frequency $\\omega$ (rad/s)", fontsize=8.4,
                  labelpad=1.5)
    ax.set_ylabel("roll response $|\\phi|/\\zeta_a$ (deg/m)", fontsize=8.4)
    ax.set_title("Roll operator across five wave headings",
                 fontsize=9.2, fontweight="bold", pad=5.0)
    ax.legend(loc="upper right", frameon=False, fontsize=6.9,
              handlelength=1.7, labelspacing=0.34, borderaxespad=0.35)

    # polar inset: the same surface cut at resonance
    axr = ax.inset_axes([0.560, 0.130, 0.385, 0.450], projection="polar")
    mus = np.linspace(0.0, 360.0, 361)
    ring = roll_rao(np.full_like(mus, w_b), mus)
    axr.plot(np.radians(mus), ring, color=BLUE, lw=1.6, zorder=5)
    axr.fill(np.radians(mus), ring, color=BLUE, alpha=0.16, zorder=4)
    axr.set_theta_zero_location("S")
    axr.set_theta_direction(1)
    axr.set_rlabel_position(148.0)
    axr.set_ylim(0.0, float(ring.max()) * 1.10)
    axr.set_yticks([2.0, 4.0, 6.0, 8.0])
    axr.set_yticklabels(["2", "4", "6", "8 deg/m"], fontsize=6,
                        color=MUTED)
    axr.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    axr.set_xticklabels(["following", "", "beam", "", "head", "", "beam", ""],
                        fontsize=6.0, color=SUB)
    axr.tick_params(pad=0.6)
    axr.grid(color=GRID, lw=0.5, alpha=0.9)
    axr.set_title("roll at resonance vs heading", fontsize=6.6, color=INK,
                  pad=3.0)

    # ---- aux (b): vertical-plane operators in head seas --------------------
    ax_v.plot(w, heave, color=STEEL, lw=1.9, zorder=6)
    ax_v.set_ylim(0.0, float(heave.max()) * 1.30)
    ax_v.set_xlim(0.16, 1.45)
    ax_v.tick_params(labelsize=6.8, colors=INK)
    ax_v.set_xlabel("wave frequency $\\omega$ (rad/s)", fontsize=7.5,
                    labelpad=1.5)
    ax_v.set_ylabel("heave $|z|/\\zeta_a$ (m/m)", fontsize=7.5, color=STEEL)
    ax_v.tick_params(axis="y", labelcolor=STEEL)

    ax_v2 = ax_v.twinx()
    ax_v2.plot(w, pitch, color=GOLD, lw=1.9, zorder=6)
    ax_v2.set_ylim(0.0, float(pitch.max()) * 1.30)
    ax_v2.spines["top"].set_visible(False)
    ax_v2.tick_params(labelsize=6.8, labelcolor=GOLD, length=3)
    ax_v2.set_ylabel("pitch $|\\theta|/\\zeta_a$ (deg/m)", fontsize=7.5,
                     color=GOLD)
    ax_v2.grid(False)

    ih, ip = int(np.argmax(heave)), int(np.argmax(pitch))
    ax_v.plot([w[ih]], [heave[ih]], marker="o", ms=4.0, mfc="white",
              mec=STEEL, mew=1.3, zorder=8)
    ax_v2.plot([w[ip]], [pitch[ip]], marker="o", ms=4.0, mfc="white",
               mec=GOLD, mew=1.3, zorder=8)
    ax_v.text(0.035, 0.955,
              "heave peak %.2f m/m at %.2f rad/s\npitch peak %.2f deg/m at "
              "%.2f rad/s" % (heave[ih], w[ih], pitch[ip], w[ip]),
              transform=ax_v.transAxes, ha="left", va="top", fontsize=6.3,
              color=SUB, linespacing=1.45, zorder=9)
    ax_v.set_title("Heave and pitch in head seas", fontsize=8.5,
                   fontweight="bold", pad=4.0)

    # ---- aux (c): phase ----------------------------------------------------
    for wn, z, col, lab in ((W_N3, Z_3, STEEL, "heave"),
                            (W_N5, Z_5, GOLD, "pitch"),
                            (W_N4, Z_4, BLUE, "roll")):
        ax_p.plot(w, phase_lag(w, wn, z), color=col, lw=1.8, zorder=6,
                  label=lab)
        ax_p.plot([wn], [90.0], marker="o", ms=3.6, mfc="white", mec=col,
                  mew=1.2, zorder=8)
    ax_p.axhline(90.0, color=FAINT, lw=0.8, ls=(0, (3.0, 2.0)), zorder=2)
    ax_p.text(1.425, 93.0, "quadrature at resonance", fontsize=6.3,
              color=MUTED, ha="right", va="bottom")
    ax_p.set_xlim(0.16, 1.45)
    ax_p.set_ylim(0.0, 190.0)
    ax_p.set_yticks([0, 45, 90, 135, 180])
    ax_p.tick_params(labelsize=6.8)
    ax_p.set_xlabel("wave frequency $\\omega$ (rad/s)", fontsize=7.5,
                    labelpad=1.5)
    ax_p.set_ylabel("phase lag (deg)", fontsize=7.5)
    ax_p.set_title("Phase crosses 90 deg at each resonance", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_p.legend(loc="lower right", frameon=False, fontsize=6.6,
                handlelength=1.5, labelspacing=0.28, borderaxespad=0.35)

    panel_labels_inches(fig, [ax, ax_v, ax_p], dx=-0.64, dy=0.10)
    foot = add_footer(fig, "Synthetic data, figure 997")
    check_layout(fig, foot, [ax, ax_v, ax_p])
    savefig(fig, "997_shp_motion_rao.png")


if __name__ == "__main__":
    main()

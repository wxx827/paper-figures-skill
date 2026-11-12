"""Trans-tibial prosthetic alignment and the knee kinematics it produces.

Main panel: sagittal knee angle over one normalised gait cycle for five
anterior-posterior foot placements of the same prosthesis, drawn against the
able-bodied reference band and the sound limb.  The alignment sweep is a
sequential quantity so it gets a sequential ramp; the sound limb keeps the
house blue and the reference band the house grey.  Loading-response flexion
on the prosthetic side sits below the band at every setting, which is the
point of the panel.

Auxiliary (both at 58 % of the main panel width): the root-mean-square
deviation from the reference band, in standard-deviation units, computed
separately over stance and over swing so their disagreeing optima are visible;
and a dumbbell chart moving five curve-derived gait metrics from the delivery
alignment to the optimised one, scored against the same reference.

Synthetic alignment study, illustrative values only; no real patient.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, lighten, get_cmap
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
GRID = "#d5d5d5"
BLUE = "#3b5b92"
RUST = "#c0603f"
ACCENT = "#b23a48"
TEAL = "#2e7d5b"
NORM = "#9a9a9a"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.86, 0.86, 5.54, 4.62)
AUX_T_RECT = (7.62, 3.48, 3.16, 2.00)
AUX_B_RECT = (7.62, 0.86, 3.16, 2.00)

CYCLE = np.linspace(0.0, 100.0, 501)      # % gait cycle
TOE_OFF = 62.0                            # % gait cycle, sound-limb stance end
SHIFTS = np.array([-8.0, -4.0, 0.0, 4.0, 8.0])   # mm, foot anterior positive
DELIVERY = -8.0                           # mm, alignment the socket arrived at
# metric name, reference SD, unit
METRICS = [("knee angle at initial contact", 4.0, "deg"),
           ("peak loading-response flexion", 4.5, "deg"),
           ("peak swing flexion", 6.5, "deg"),
           ("timing of peak swing flexion", 3.0, "% gait cycle"),
           ("total knee excursion", 7.0, "deg")]


def norm_mean(p):
    """Able-bodied sagittal knee angle, deg, over the normalised cycle."""
    return (4.5
            + 15.0 * np.exp(-0.5 * ((p - 16.0) / 8.0) ** 2)
            + 58.0 * np.exp(-0.5 * ((p - 72.0) / 11.0) ** 2))


def norm_sd(p):
    """Between-subject spread of that reference, deg."""
    return 3.4 + 4.2 * np.exp(-0.5 * ((p - 72.0) / 15.0) ** 2)


def prosthetic(p, shift):
    """Prosthetic-side knee angle for an anterior-posterior foot shift, mm.

    A posterior shift buys back some loading-response flexion but leaves the
    knee flexed through mid stance; an anterior shift trades that for a
    hyperextension thrust.  The competing terms are what give the stance
    deviation curve an interior optimum.
    """
    st_amp = 6.2 - 0.42 * shift
    sw_amp = 54.0 + 0.55 * shift
    sw_time = 72.0 + 0.34 * shift
    return (4.0 + 0.10 * shift
            + st_amp * np.exp(-0.5 * ((p - 18.0) / 8.5) ** 2)
            - 0.34 * shift * np.exp(-0.5 * ((p - 45.0) / 9.0) ** 2)
            + sw_amp * np.exp(-0.5 * ((p - sw_time) / 11.5) ** 2))


def sound_side(p, rng):
    """Sound limb: the reference with a little compensatory drift."""
    y = norm_mean(p) + 2.4 * np.exp(-0.5 * ((p - 74.0) / 12.0) ** 2)
    n = rng.normal(0.0, 1.0, p.size)
    k = np.exp(-0.5 * np.linspace(-3.0, 3.0, 81) ** 2)
    return y + 1.1 * np.convolve(n, k / k.sum(), mode="same")


def rms_z(curve, mask):
    """RMS distance from the reference, in reference standard deviations."""
    d = (curve - norm_mean(CYCLE)) / norm_sd(CYCLE)
    return float(np.sqrt(np.mean(d[mask] ** 2)))


def metrics_of(curve):
    """The five curve-derived numbers scored in the dumbbell panel."""
    stance = CYCLE <= TOE_OFF
    swing = ~stance
    i_sw = int(np.argmax(np.where(swing, curve, -np.inf)))
    return np.array([curve[0],
                     float(np.max(curve[stance])),
                     float(curve[i_sw]),
                     float(CYCLE[i_sw]),
                     float(curve.max() - curve.min())])


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(980)

    stance = CYCLE <= TOE_OFF
    swing = ~stance
    curves = {s: prosthetic(CYCLE, s) for s in SHIFTS}
    sound = sound_side(CYCLE, rng)

    sweep = np.linspace(-10.0, 10.0, 81)
    z_st = np.array([rms_z(prosthetic(CYCLE, s), stance) for s in sweep])
    z_sw = np.array([rms_z(prosthetic(CYCLE, s), swing) for s in sweep])
    z_all = np.array([rms_z(prosthetic(CYCLE, s), np.ones_like(CYCLE, bool))
                      for s in sweep])
    best = float(sweep[int(np.argmin(z_all))])

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)
        a.grid(color=GRID, lw=0.5, alpha=0.6, zorder=0)

    # ---- main: knee angle over the gait cycle ------------------------------
    lo = norm_mean(CYCLE) - norm_sd(CYCLE)
    hi = norm_mean(CYCLE) + norm_sd(CYCLE)
    ax.fill_between(CYCLE, lo, hi, color=lighten(NORM, 0.62), lw=0, zorder=1)
    ax.plot(CYCLE, norm_mean(CYCLE), color=NORM, lw=1.0, ls=(0, (4.0, 2.4)),
            zorder=3)
    ax.axvspan(0.0, TOE_OFF, color="#fbfbfb", lw=0, zorder=0)
    ax.axvline(TOE_OFF, color=SUB, lw=0.8, ls=(0, (2.6, 2.2)), zorder=4)

    cmap = get_cmap("cb_sequential_warm")
    cols = [cmap(v) for v in np.linspace(0.34, 0.96, len(SHIFTS))]
    for s, col in zip(SHIFTS, cols):
        lw = 1.9 if s == 0.0 else 1.2
        ax.plot(CYCLE, curves[s], color=col, lw=lw, zorder=6,
                solid_capstyle="round")
    ax.plot(CYCLE, sound, color=BLUE, lw=1.6, zorder=5, solid_capstyle="round")

    out = curves[0.0] < lo
    ax.fill_between(CYCLE, curves[0.0], lo, where=out,
                    color=lighten(ACCENT, 0.84), lw=0, zorder=2)
    frac_out = 100.0 * np.mean(out[stance])
    ax.annotate(f"prosthetic loading response stays\nbelow the reference band "
                f"for {frac_out:.0f} % of stance",
                xy=(19.0, 0.5 * (float(np.interp(19.0, CYCLE, lo))
                                 + float(np.interp(19.0, CYCLE, curves[0.0])))),
                xytext=(28.0, 40.0), ha="left", va="bottom", fontsize=6.4,
                color=ACCENT, linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8,
                                shrinkA=1.0, shrinkB=1.0))

    ax.text(TOE_OFF / 2.0, 76.0, "stance", ha="center", va="center",
            fontsize=7.4, color=SUB)
    ax.text((TOE_OFF + 100.0) / 2.0, 76.0, "swing", ha="center", va="center",
            fontsize=7.4, color=SUB)
    ax.text(TOE_OFF - 1.2, 71.0, "toe off", ha="right", va="center",
            fontsize=6.3, color=SUB)

    handles = [Line2D([], [], color=c, lw=1.9 if s == 0 else 1.2,
                      label=f"{s:+.0f} mm" if s else "0 mm (nominal)")
               for s, c in zip(SHIFTS, cols)]
    handles += [
        Line2D([], [], color=BLUE, lw=1.6, label="sound limb"),
        Line2D([], [], color=NORM, lw=5.0, alpha=0.45,
               label="able-bodied mean $\\pm$ 1 SD, n=28"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", fontsize=6.4, ncol=2,
                    frameon=True, framealpha=0.94, borderpad=0.42,
                    columnspacing=1.0, handlelength=1.6, labelspacing=0.30,
                    title="prosthetic foot shift, anterior positive",
                    title_fontsize=6.6)
    leg.get_frame().set(edgecolor=HAIR, linewidth=0.6)

    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(-4.0, 84.0)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 81, 20))
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel("% gait cycle (0 = prosthetic initial contact)",
                  fontsize=8.2, labelpad=2.5)
    ax.set_ylabel("knee flexion angle (deg)", fontsize=8.2, labelpad=3.0)
    ax.set_title("Sagittal knee kinematics across a 16 mm alignment sweep",
                 fontsize=9.0, fontweight="bold", pad=5.0)

    # ---- aux (b): deviation against alignment ------------------------------
    ax_t.plot(sweep, z_st, color=RUST, lw=1.6, zorder=5)
    ax_t.plot(sweep, z_sw, color=BLUE, lw=1.6, zorder=5)
    ax_t.plot(sweep, z_all, color=INK, lw=1.1, ls=(0, (3.6, 2.2)), zorder=6)
    for arr, col, lab, va, dy in ((z_st, RUST, "stance", "bottom", 0.17),
                                  (z_sw, BLUE, "swing", "top", -0.12)):
        i = int(np.argmin(arr))
        ax_t.plot([sweep[i]], [arr[i]], marker="o", ms=4.0, mfc="white",
                  mec=col, mew=1.2, zorder=7)
        ax_t.text(float(np.clip(sweep[i], -5.0, 6.4)), arr[i] + dy,
                  f"{lab} best {sweep[i]:+.1f} mm", ha="center", va=va,
                  fontsize=6.2, color=col)
    ax_t.axvline(best, color=ACCENT, lw=1.0, ls=(0, (3.0, 2.2)), zorder=4)
    ax_t.text(best + 0.4, z_all.max() * 0.98,
              f"whole-cycle optimum\n{best:+.1f} mm", ha="left", va="top",
              fontsize=6.2, color=ACCENT, linespacing=1.4)
    ax_t.axvline(DELIVERY, color=MUTED, lw=0.9, ls=(0, (2.2, 2.0)), zorder=4)
    ax_t.text(DELIVERY + 0.4, 0.16, "as delivered", ha="left", va="bottom",
              fontsize=6.2, color=MUTED)
    ax_t.set_xlim(-10.4, 10.4)
    ax_t.set_ylim(0.0, max(z_st.max(), z_sw.max()) * 1.16)
    ax_t.set_xticks(np.arange(-10, 11, 5))
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("foot anterior-posterior shift (mm)", fontsize=7.6,
                    labelpad=1.5)
    ax_t.set_ylabel("RMS deviation (SD)", fontsize=7.6, labelpad=2.0)
    ax_t.set_title("Stance and swing want different alignments",
                   fontsize=8.4, fontweight="bold", pad=4.0)

    # ---- aux (c): metric-by-metric move ------------------------------------
    ref = metrics_of(norm_mean(CYCLE))
    sd = np.array([m[1] for m in METRICS])
    z_before = (metrics_of(prosthetic(CYCLE, DELIVERY)) - ref) / sd
    z_after = (metrics_of(prosthetic(CYCLE, best)) - ref) / sd
    ypos = np.arange(len(METRICS))[::-1]
    ax_b.axvspan(-2.0, 2.0, color=lighten(NORM, 0.80), lw=0, zorder=1)
    ax_b.axvline(0.0, color=SUB, lw=0.9, zorder=3)
    for y, b, a in zip(ypos, z_before, z_after):
        ax_b.plot([b, a], [y, y], color=HAIR, lw=2.4,
                  solid_capstyle="round", zorder=3)
        ax_b.plot([b], [y], marker="o", ms=4.4, mfc=lighten(MUTED, 0.30),
                  mec="white", mew=0.8, zorder=5)
        ax_b.plot([a], [y], marker="o", ms=4.8,
                  mfc=TEAL if abs(a) <= 2.0 else RUST, mec="white", mew=0.8,
                  zorder=6)
    ax_b.text(-2.0, 4.72, "acceptable |z| $\\leq$ 2", ha="left", va="top",
              fontsize=6.2, color=SUB)
    improved = int(np.sum(np.abs(z_after) < np.abs(z_before)))
    ax_b.text(4.05, -0.76, f"{improved} of {len(METRICS)} metrics move toward "
              "the reference", ha="right", va="bottom", fontsize=6.2,
              color=SUB)
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels([m[0] for m in METRICS], fontsize=6.3)
    ax_b.set_ylim(-0.85, 4.85)
    ax_b.set_xlim(-4.2, 4.2)
    ax_b.set_xticks(np.arange(-4, 5, 2))
    ax_b.tick_params(axis="x", labelsize=6.6)
    ax_b.grid(axis="y", visible=False)
    ax_b.set_xlabel("z score against the able-bodied reference",
                    fontsize=7.6, labelpad=1.5)
    ax_b.set_title(f"Delivery {DELIVERY:+.0f} mm  ->  optimised "
                   f"{best:+.1f} mm", fontsize=8.4, fontweight="bold", pad=4.0)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.52, dy=0.10)
    savefig(fig, "980_reh_prosthetic_alignment.png")


if __name__ == "__main__":
    main()

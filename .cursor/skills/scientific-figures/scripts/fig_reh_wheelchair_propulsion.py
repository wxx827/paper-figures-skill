"""Handrim propulsion mechanics across one push cycle.

Main panel: resultant handrim force with its tangential and radial components
over a normalised push cycle, the push phase shaded and the recovery left
open.  The gap between the resultant and its tangential component is the force
that never turns the wheel, so it is filled rather than left to be inferred;
the fraction of effective force quoted on the panel is the ratio of the two
integrals over exactly that shaded interval.

Auxiliary (both at 58 % of the main panel width): the resultant shoulder joint
force over the same cycle for five rear-axle positions, which is an ordered
quantity and so gets a sequential ramp; and the trade-off those five setups
trace out, each point being the mean effective fraction and the peak shoulder
load taken from the curves above it.

Synthetic propulsion model, illustrative values only; no real user depicted.
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

CYCLE = np.linspace(0.0, 100.0, 601)      # % push cycle
AXLE = np.array([0.0, 20.0, 40.0, 60.0, 80.0])   # mm rearward of reference
BODY_MASS = 72.0                          # kg
SHOULDER_LIMIT = 2.20                     # N/kg, load the fitting keeps under
REFERENCE = 40.0                          # mm, the setup drawn in panel (a)


def setup(d):
    """Cycle geometry and gains for a rear-axle offset of ``d`` mm."""
    return dict(push_end=44.0 - 0.10 * d,
                contact=92.0 - 0.32 * d,
                f_peak=96.0 + 0.22 * d,
                fef_max=0.78 + 0.0016 * d,
                sh_gain=1.15 + 0.0060 * d,
                freq=48.0 + 0.18 * d)


def forces(d):
    """Resultant, tangential and radial handrim force, in N, over the cycle."""
    s = setup(d)
    u = np.clip(CYCLE / s["push_end"], 0.0, 1.0)
    shape = np.where(CYCLE <= s["push_end"],
                     np.sin(np.pi * np.clip(u, 0.0, 1.0)) ** 1.55, 0.0)
    f_tot = s["f_peak"] * shape
    # effective direction improves through the push and falls away at release;
    # the profile is normalised so fef_max really is the peak it reaches
    prof = (0.62 + 0.55 * u - 0.34 * u ** 2) / 0.8423
    fef = np.clip(s["fef_max"] * prof, 0.0, 0.985)
    f_tan = f_tot * fef
    f_rad = np.sqrt(np.maximum(f_tot ** 2 - f_tan ** 2, 0.0))
    return f_tot, f_tan, f_rad


def shoulder_force(d):
    """Resultant shoulder joint force, N/kg, driven by the same rim force."""
    s = setup(d)
    f_tot, _, _ = forces(d)
    postural = 0.16 * BODY_MASS * np.exp(
        -0.5 * ((CYCLE - 74.0) / 15.0) ** 2)      # recovery-phase arm carriage
    return (s["sh_gain"] * f_tot + postural) / BODY_MASS


def cycle_time(d):
    """Seconds per push cycle, from the push frequency of that setup."""
    return 60.0 / setup(d)["freq"]


def summarise(d):
    """Effective fraction, tangential impulse and peak shoulder load."""
    s = setup(d)
    f_tot, f_tan, _ = forces(d)
    push = CYCLE <= s["push_end"]
    t = CYCLE / 100.0 * cycle_time(d)
    fef = 100.0 * np.trapezoid(f_tan[push], t[push]) / np.trapezoid(
        f_tot[push], t[push])
    impulse = float(np.trapezoid(f_tan[push], t[push]))
    return fef, impulse, float(shoulder_force(d).max())


def main():
    set_pub_style(base_font=8)
    np.random.default_rng(982)

    cmap = get_cmap("cb_sequential_warm")
    cols = [cmap(v) for v in np.linspace(0.32, 0.95, len(AXLE))]

    f_tot, f_tan, f_rad = forces(REFERENCE)
    s_ref = setup(REFERENCE)
    push = CYCLE <= s_ref["push_end"]
    fef_ref, imp_ref, sh_ref = summarise(REFERENCE)

    stats = [summarise(d) for d in AXLE]
    fefs = np.array([v[0] for v in stats])
    peaks = np.array([v[2] for v in stats])
    ok = peaks <= SHOULDER_LIMIT
    pick = int(np.argmax(np.where(ok, fefs, -np.inf)))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)
        a.grid(color=GRID, lw=0.5, alpha=0.6, zorder=0)

    # ---- main: force components over the push cycle ------------------------
    ax.axvspan(0.0, s_ref["push_end"], color="#f7f7f7", lw=0, zorder=0)
    ax.axvline(s_ref["push_end"], color=SUB, lw=0.8, ls=(0, (2.6, 2.2)),
               zorder=4)
    ax.fill_between(CYCLE, f_tan, f_tot, color=lighten(RUST, 0.82), lw=0,
                    zorder=2)
    ax.plot(CYCLE, f_tot, color=BLUE, lw=2.0, zorder=6)
    ax.plot(CYCLE, f_tan, color=TEAL, lw=1.7, zorder=6)
    ax.plot(CYCLE, f_rad, color=RUST, lw=1.3, ls=(0, (4.0, 2.2)), zorder=5)

    i_tot = int(np.argmax(f_tot))
    i_tan = int(np.argmax(f_tan))
    for i, arr, col, lab in ((i_tot, f_tot, BLUE, "resultant"),
                             (i_tan, f_tan, TEAL, "tangential")):
        ax.plot([CYCLE[i]], [arr[i]], marker="o", ms=3.8, mfc="white",
                mec=col, mew=1.1, zorder=8)
    ax.text(CYCLE[i_tot] + 2.2, f_tot[i_tot],
            f"peak resultant {f_tot[i_tot]:.0f} N at "
            f"{CYCLE[i_tot]:.0f} % of the cycle", ha="left", va="center",
            fontsize=6.4, color=BLUE)
    ax.text(CYCLE[i_tan] + 2.2, f_tan[i_tan] - 4.0,
            f"peak tangential {f_tan[i_tan]:.0f} N", ha="left", va="top",
            fontsize=6.4, color=TEAL)
    ax.text(CYCLE[i_tot], 0.5 * (f_tot[i_tot] + f_tan[i_tot]),
            "force that never turns the wheel", ha="center", va="center",
            fontsize=6.4, color=RUST, zorder=7)

    ax.text(s_ref["push_end"] / 2.0, 113.0, "push phase", ha="center",
            va="center", fontsize=7.4, color=SUB)
    ax.text((s_ref["push_end"] + 100.0) / 2.0, 113.0,
            "recovery phase (hand off the rim)", ha="center", va="center",
            fontsize=7.4, color=SUB)
    ax.text(s_ref["push_end"] + 1.0, 104.0, "hand release", ha="left",
            va="top", fontsize=6.3, color=SUB)
    ax.text(99.0, 26.0,
            f"rear axle {REFERENCE:.0f} mm rearward\n"
            f"contact angle {s_ref['contact']:.0f} deg\n"
            f"{s_ref['freq']:.0f} pushes/min, {cycle_time(REFERENCE):.2f} s "
            "per cycle\n"
            f"effective force fraction {fef_ref:.0f} %\n"
            f"tangential impulse {imp_ref:.1f} N$\\cdot$s per cycle",
            ha="right", va="bottom", fontsize=6.5, color=SUB, linespacing=1.6)

    handles = [
        Line2D([], [], color=BLUE, lw=2.0, label="resultant rim force"),
        Line2D([], [], color=TEAL, lw=1.7, label="tangential (propulsive)"),
        Line2D([], [], color=RUST, lw=1.3, ls=(0, (4.0, 2.2)),
               label="radial (into the rim)"),
    ]
    leg = ax.legend(handles=handles, loc="center right", fontsize=6.6,
                    frameon=True, framealpha=0.94, borderpad=0.42,
                    handlelength=1.7, labelspacing=0.32)
    leg.get_frame().set(edgecolor=HAIR, linewidth=0.6)

    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 122.0)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 111, 20))
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel("% push cycle (0 = hand contact with the rim)",
                  fontsize=8.2, labelpad=2.5)
    ax.set_ylabel("handrim force (N)", fontsize=8.2, labelpad=3.0)
    ax.set_title("One propulsion cycle at 1.1 m/s on a level surface",
                 fontsize=9.0, fontweight="bold", pad=5.0)

    # ---- aux (b): shoulder load across axle positions ----------------------
    for d, col in zip(AXLE, cols):
        sh = shoulder_force(d)
        ax_t.plot(CYCLE, sh, color=col, lw=1.5, zorder=5)
        i = int(np.argmax(sh))
        ax_t.plot([CYCLE[i]], [sh[i]], marker="o", ms=3.0, mfc="white",
                  mec=col, mew=0.9, zorder=6)
    ax_t.axhline(SHOULDER_LIMIT, color=ACCENT, lw=1.1, ls=(0, (3.0, 2.2)),
                 zorder=4)
    ax_t.text(99.0, SHOULDER_LIMIT + 0.05,
              f"fitting limit {SHOULDER_LIMIT:.2f} N/kg", ha="right",
              va="bottom", fontsize=6.2, color=ACCENT)
    ax_t.text(2.0, peaks.max() * 1.05, "rear axle 0 mm (forward)  ->  80 mm "
              "(rearward)", ha="left", va="top", fontsize=6.2, color=SUB)
    ax_t.set_xlim(0.0, 100.0)
    ax_t.set_ylim(0.0, peaks.max() * 1.16)
    ax_t.set_xticks(np.arange(0, 101, 20))
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("% push cycle", fontsize=7.6, labelpad=1.5)
    ax_t.set_ylabel("shoulder force (N/kg)", fontsize=7.6, labelpad=2.0)
    ax_t.set_title(f"Shoulder load, {BODY_MASS:.0f} kg user, n=5 setups",
                   fontsize=8.4, fontweight="bold", pad=4.0)

    # ---- aux (c): the trade-off the five setups trace ----------------------
    ax_b.axhspan(0.0, SHOULDER_LIMIT, color=lighten(TEAL, 0.88), lw=0,
                 zorder=1)
    ax_b.plot(fefs, peaks, color=HAIR, lw=1.4, zorder=3)
    for k, (x, y, col) in enumerate(zip(fefs, peaks, cols)):
        ax_b.plot([x], [y], marker="o", ms=6.0, mfc=col, mec="white", mew=0.9,
                  zorder=5)
        ax_b.text(x, y + 0.078, f"{AXLE[k]:.0f}", ha="center", va="bottom",
                  fontsize=6.0, color=SUB)
    ax_b.axhline(SHOULDER_LIMIT, color=ACCENT, lw=1.0, ls=(0, (3.0, 2.2)),
                 zorder=4)
    ax_b.plot([fefs[pick]], [peaks[pick]], marker="o", ms=10.0,
              mfc="none", mec=ACCENT, mew=1.3, zorder=6)
    ax_b.text(fefs[pick] + 0.45, peaks[pick] - 0.08,
              f"best effective fraction\nunder the limit: "
              f"{AXLE[pick]:.0f} mm", ha="left", va="top", fontsize=6.2,
              color=ACCENT, linespacing=1.4)
    ax_b.text(fefs.min() - 0.2, peaks.max(),
              f"{fefs.max() - fefs.min():.1f} points of effective fraction "
              f"cost\n{peaks.max() - peaks.min():.2f} N/kg at the shoulder",
              ha="left", va="top", fontsize=6.2, color=SUB, linespacing=1.4)
    ax_b.set_xlim(fefs.min() - 1.4, fefs.max() + 1.0)
    ax_b.set_ylim(peaks.min() - 0.22, peaks.max() + 0.16)
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("mean fraction of effective force (%)", fontsize=7.6,
                    labelpad=1.5)
    ax_b.set_ylabel("peak shoulder force (N/kg)", fontsize=7.6, labelpad=2.0)
    ax_b.set_title("Labels are the rear-axle offset in mm", fontsize=8.4,
                   fontweight="bold", pad=4.0)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.52, dy=0.10)
    savefig(fig, "982_reh_wheelchair_propulsion.png")


if __name__ == "__main__":
    main()

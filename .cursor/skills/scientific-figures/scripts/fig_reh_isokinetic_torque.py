"""Isokinetic knee testing: torque against joint angle for both limbs.

Main panel: extensor torque above the zero rule and flexor torque below it, so
one signed axis carries the whole test.  Solid curves are concentric, dashed
curves eccentric, blue is the uninvolved limb and red-orange the involved one.
Peak torque and the angle at which it occurs are marked on the curves, and
both numbers are the argmax of the plotted array rather than a stated value.

Auxiliary (both at 58 % of the main panel width): the limb symmetry index for
the four contraction modes, computed as involved over uninvolved peak torque;
and the conventional and functional hamstring-to-quadriceps ratios across
three test speeds, each a quotient of two peaks from the same torque model.

Synthetic dynamometer traces, illustrative values only; no real patient.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, lighten
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

ANGLE = np.linspace(12.0, 95.0, 420)      # knee flexion angle, deg
SPEEDS = np.array([60.0, 180.0, 300.0])   # deg/s
BODY_MASS = 74.0                          # kg, for the N.m/kg annotation
LSI_LIMIT = 90.0                          # % below which return to sport waits

# muscle group -> (peak at 60 deg/s concentric, optimum angle, low-side
# width, high-side width) for the uninvolved limb
GROUPS = {
    "extensor": (232.0, 63.0, 26.0, 30.0),
    "flexor": (128.0, 33.0, 15.0, 34.0),
}
# involved-limb strength retained, by group
RETAINED = {"extensor": 0.66, "flexor": 0.87}


def bell(theta, tmax, opt, w_lo, w_hi):
    """Asymmetric bell in joint angle: the length-tension curve of the group."""
    w = np.where(theta < opt, w_lo, w_hi)
    return tmax * np.exp(-0.5 * ((theta - opt) / w) ** 2)


# Hill-type velocity constant, deg/s: the extensors shed concentric torque
# faster with speed than the flexors do, which is what tilts the ratio panel.
V0 = {"extensor": 240.0, "flexor": 420.0}


def speed_factor(speed, mode, group):
    """Concentric torque falls with test speed, eccentric torque rises."""
    if mode == "con":
        v0 = V0[group]
        return (1.0 + 60.0 / v0) / (1.0 + speed / v0)
    return 1.26 - 0.10 * np.exp(-(speed - 60.0) / 160.0)


def torque(group, limb, mode, speed, rng=None):
    """One torque-angle trace, in N.m, on the shared ANGLE grid."""
    tmax, opt, w_lo, w_hi = GROUPS[group]
    tmax = tmax * speed_factor(speed, mode, group)
    if limb == "involved":
        tmax *= RETAINED[group]
        opt += 4.5 if group == "extensor" else 2.0
    y = bell(ANGLE, tmax, opt, w_lo, w_hi)
    if rng is not None:
        ripple = rng.normal(0.0, 1.0, ANGLE.size)
        k = np.exp(-0.5 * (np.linspace(-3.0, 3.0, 41)) ** 2)
        ripple = np.convolve(ripple, k / k.sum(), mode="same")
        y = y * (1.0 + 0.022 * ripple)
    return np.maximum(y, 0.0)


def peak_of(y):
    """Peak torque and the angle it occurs at, straight off the array."""
    i = int(np.argmax(y))
    return float(y[i]), float(ANGLE[i])


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(979)

    traces = {}
    for group in GROUPS:
        for limb in ("uninvolved", "involved"):
            for mode in ("con", "ecc"):
                for sp in SPEEDS:
                    traces[(group, limb, mode, sp)] = torque(
                        group, limb, mode, sp, rng)
    peaks = {k: peak_of(v) for k, v in traces.items()}

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)
        a.grid(color=GRID, lw=0.5, alpha=0.6, zorder=0)

    # ---- main: signed torque-angle curves at 60 deg/s ----------------------
    sp0 = SPEEDS[0]
    ax.axhline(0.0, color=INK, lw=0.9, zorder=4)
    for group, sign in (("extensor", +1.0), ("flexor", -1.0)):
        for limb, col in (("uninvolved", BLUE), ("involved", RUST)):
            for mode, ls, lw in (("con", "-", 1.7), ("ecc", (0, (4.0, 2.2)),
                                                     1.4)):
                y = sign * traces[(group, limb, mode, sp0)]
                ax.plot(ANGLE, y, color=col, ls=ls, lw=lw, zorder=6,
                        solid_capstyle="round")
                pk, ang = peaks[(group, limb, mode, sp0)]
                ax.plot([ang], [sign * pk], ls="none", marker="o", ms=3.4,
                        mfc="white", mec=col, mew=1.0, zorder=8)

    ax.fill_between(ANGLE, traces[("extensor", "involved", "con", sp0)],
                    traces[("extensor", "uninvolved", "con", sp0)],
                    color=lighten(RUST, 0.86), lw=0, zorder=2)

    def peak_block(group, sign, y0, title):
        """Peak torque and its angle, listed clear of the curves."""
        ax.text(13.5, y0, title, ha="left", va="top", fontsize=7.2,
                fontweight="bold", color=SUB)
        for k, (limb, col) in enumerate((("uninvolved", BLUE),
                                         ("involved", RUST))):
            pk, ang = peaks[(group, limb, "con", sp0)]
            ax.text(13.5, y0 - 16.0 - 15.0 * k,
                    f"{limb} concentric peak {pk:.0f} N$\\cdot$m at "
                    f"{ang:.0f} deg", ha="left", va="top", fontsize=6.3,
                    color=col)
        un = peaks[(group, "uninvolved", "con", sp0)][0]
        inv = peaks[(group, "involved", "con", sp0)][0]
        ax.text(13.5, y0 - 46.0,
                f"deficit {un - inv:.0f} N$\\cdot$m "
                f"({100.0 * (1.0 - inv / un):.0f} % of the uninvolved peak)",
                ha="left", va="top", fontsize=6.3, color=MUTED)
        return un, inv

    ext_un, ext_in = peak_block("extensor", +1.0, 296.0,
                                "knee extensors (quadriceps)")
    peak_block("flexor", -1.0, -176.0, "knee flexors (hamstrings)")
    ax.text(94.0, 16.0, f"peak extensor torque {ext_un / BODY_MASS:.2f} "
            f"N$\\cdot$m/kg on the uninvolved limb ({BODY_MASS:.0f} kg)",
            ha="right", va="bottom", fontsize=6.2, color=MUTED)

    handles = [
        Line2D([], [], color=BLUE, lw=1.7, label="uninvolved limb"),
        Line2D([], [], color=RUST, lw=1.7, label="involved limb"),
        Line2D([], [], color=SUB, lw=1.7, label="concentric"),
        Line2D([], [], color=SUB, lw=1.4, ls=(0, (4.0, 2.2)),
               label="eccentric"),
    ]
    leg = ax.legend(handles=handles, loc="lower right", fontsize=6.6, ncol=2,
                    frameon=True, framealpha=0.94, borderpad=0.42,
                    columnspacing=1.1, handlelength=1.6, labelspacing=0.32)
    leg.get_frame().set(edgecolor=HAIR, linewidth=0.6)

    ax.set_xlim(11.0, 96.0)
    ax.set_ylim(-260.0, 300.0)
    ax.set_xticks(np.arange(20, 96, 10))
    ax.set_yticks(np.arange(-240, 301, 60))
    ax.set_yticklabels([f"{abs(v):.0f}" for v in np.arange(-240, 301, 60)])
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel("knee flexion angle (deg, 0 = full extension)",
                  fontsize=8.2, labelpad=2.5)
    ax.set_ylabel("torque magnitude (N$\\cdot$m)", fontsize=8.2, labelpad=3.0)
    ax.set_title(f"Torque-angle curves at {sp0:.0f} deg/s, n=1 test session",
                 fontsize=9.0, fontweight="bold", pad=5.0)

    # ---- aux (b): limb symmetry index --------------------------------------
    modes = [("extensor", "con", "extensor concentric"),
             ("extensor", "ecc", "extensor eccentric"),
             ("flexor", "con", "flexor concentric"),
             ("flexor", "ecc", "flexor eccentric")]
    lsi = np.array([100.0 * peaks[(g, "involved", m, sp0)][0]
                    / peaks[(g, "uninvolved", m, sp0)][0]
                    for g, m, _ in modes])
    ypos = np.arange(len(modes))[::-1]
    cols = [lighten(TEAL, 0.24) if v >= LSI_LIMIT else lighten(RUST, 0.24)
            for v in lsi]
    ax_t.barh(ypos, lsi, height=0.58, color=cols, edgecolor="white",
              linewidth=0.7, zorder=3)
    ax_t.axvline(LSI_LIMIT, color=ACCENT, lw=1.1, ls=(0, (3.0, 2.2)), zorder=5)
    ax_t.text(LSI_LIMIT - 1.5, 3.55, f"{LSI_LIMIT:.0f} % clearance limit",
              ha="right", va="top", fontsize=6.2, color=ACCENT)
    for y, v in zip(ypos, lsi):
        ax_t.text(v + 1.2, y, f"{v:.0f} %", ha="left", va="center",
                  fontsize=6.4, color=INK)
    ax_t.set_yticks(ypos)
    ax_t.set_yticklabels([lab for _, _, lab in modes], fontsize=6.4)
    ax_t.set_ylim(-0.62, 3.72)
    ax_t.set_xlim(0.0, 116.0)
    ax_t.set_xticks([0, 25, 50, 75, 100])
    ax_t.tick_params(axis="x", labelsize=6.6)
    ax_t.set_xlabel("limb symmetry index (%)", fontsize=7.6, labelpad=1.5)
    ax_t.grid(axis="y", visible=False)
    ax_t.set_title("Involved over uninvolved peak torque", fontsize=8.4,
                   fontweight="bold", pad=4.0)

    # ---- aux (c): hamstring-to-quadriceps ratios ---------------------------
    conv = {}
    func = {}
    for limb in ("uninvolved", "involved"):
        conv[limb] = np.array([peaks[("flexor", limb, "con", s)][0]
                               / peaks[("extensor", limb, "con", s)][0]
                               for s in SPEEDS])
        func[limb] = np.array([peaks[("flexor", limb, "ecc", s)][0]
                               / peaks[("extensor", limb, "con", s)][0]
                               for s in SPEEDS])
    ax_b.axhspan(0.50, 0.62, color=lighten(NORM, 0.74), lw=0, zorder=1)
    ax_b.text(38.0, 0.40, "conventional reference band 0.50 - 0.62",
              ha="left", va="bottom", fontsize=6.1, color=SUB)
    for limb, col in (("uninvolved", BLUE), ("involved", RUST)):
        ax_b.plot(SPEEDS, conv[limb], color=col, lw=1.5, marker="o", ms=3.6,
                  mfc="white", mew=1.0, zorder=5)
        ax_b.plot(SPEEDS, func[limb], color=col, lw=1.3,
                  ls=(0, (4.0, 2.2)), marker="s", ms=3.4, mfc="white",
                  mew=1.0, zorder=5)
    ax_b.text(SPEEDS[-1] + 8.0, func["uninvolved"][-1], "functional\n"
              "(ecc flexor / con extensor)", ha="left", va="center",
              fontsize=6.1, color=SUB, linespacing=1.35)
    ax_b.text(SPEEDS[-1] + 8.0, conv["uninvolved"][-1], "conventional\n"
              "(con / con)", ha="left", va="center", fontsize=6.1, color=SUB,
              linespacing=1.35)
    hi = max(func["involved"].max(), func["uninvolved"].max())
    ax_b.set_xlim(30.0, 452.0)
    ax_b.set_ylim(0.34, hi * 1.14)
    ax_b.set_xticks(SPEEDS)
    ax_b.set_yticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("angular velocity (deg/s)", fontsize=7.6, labelpad=1.5)
    ax_b.set_ylabel("hamstring : quadriceps", fontsize=7.6, labelpad=2.0)
    ax_b.set_title("Ratios climb with test speed", fontsize=8.4,
                   fontweight="bold", pad=4.0)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.52, dy=0.10)
    savefig(fig, "979_reh_isokinetic_torque.png")


if __name__ == "__main__":
    main()

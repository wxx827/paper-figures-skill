"""Sagittal joint kinematics of the lower limb over one normalised gait cycle.

Hip, knee and ankle share one gait-cycle axis and one angle scale, so the
flattening of the patient group's knee excursion cannot be hidden by per-panel
autoscaling.  The phase ribbon on top segments the cycle the way a gait lab
reports it, and the spatio-temporal panel on the right is plotted relative to
the control mean because the six parameters do not share a unit.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pubstyle import set_pub_style, savefig, style_axes, COLORS, lighten
from figkit import shared_limits, panel_labels_inches

CONTROL = "#3b5b92"          # house blue  - healthy reference group
PATIENT = "#b23a48"          # house red   - single emphasis, the patient group
GOLD = "#c9a227"

N_POINT = 201
TOE_OFF = {"Control": 62.0, "Patient": 68.5}

# (name, start %, end %) - the seven functional sub-phases of the cycle
PHASES = [("Loading", 0.0, 12.0), ("Mid stance", 12.0, 31.0),
          ("Terminal stance", 31.0, 50.0), ("Pre-swing", 50.0, 62.0),
          ("Initial swing", 62.0, 75.0), ("Mid swing", 75.0, 87.0),
          ("Terminal swing", 87.0, 100.0)]

# Spatio-temporal parameters: (name, unit, control, patient)
SPATIOTEMPORAL = [
    ("Walking speed", "m/s", 1.32, 0.86),
    ("Cadence", "steps/min", 112.0, 96.0),
    ("Stride length", "m", 1.41, 1.07),
    ("Stance duration", "% cycle", 62.0, 68.5),
    ("Double support", "% cycle", 21.0, 29.5),
    ("Step width", "m", 0.11, 0.17),
]


def _bump(p, centre, width, amp):
    """Cyclic Gaussian bump on the 0-100 % gait axis."""
    d = (p - centre + 50.0) % 100.0 - 50.0
    return amp * np.exp(-0.5 * (d / width) ** 2)


def _smooth_noise(rng, n, span=31):
    """Seeded, wrapped, smooth 0-1 profile used to shape the SD bands."""
    x = rng.normal(0.0, 1.0, n)
    k = np.exp(-0.5 * np.linspace(-3.0, 3.0, span) ** 2)
    k /= k.sum()
    pad = span // 2
    s = np.convolve(np.r_[x[-pad:], x, x[:pad]], k, mode="same")[pad:-pad]
    return (s - s.min()) / (np.ptp(s) + 1e-9)


def joint_curves(p, group):
    """Mean sagittal angles (deg, flexion positive) for one group."""
    if group == "Control":
        hip = 10.0 + 20.0 * np.cos(2 * np.pi * p / 100.0) + \
            3.0 * np.cos(4 * np.pi * p / 100.0 + 0.8)
        knee = 2.0 + _bump(p, 16.0, 8.0, 16.0) + _bump(p, 72.0, 12.0, 58.0)
        ankle = 2.0 + _bump(p, 45.0, 14.0, 12.0) - _bump(p, 8.0, 7.0, 7.0) \
            - _bump(p, 64.0, 9.0, 22.0)
    else:
        hip = 15.0 + 15.5 * np.cos(2 * np.pi * (p - 3.0) / 100.0) + \
            2.0 * np.cos(4 * np.pi * p / 100.0 + 0.8)
        knee = 8.0 + _bump(p, 20.0, 9.0, 9.0) + _bump(p, 77.0, 15.0, 33.0)
        ankle = -1.0 + _bump(p, 48.0, 16.0, 7.0) - _bump(p, 10.0, 8.0, 4.0) \
            - _bump(p, 70.0, 11.0, 9.0)
    return {"Hip": hip, "Knee": knee, "Ankle": ankle}


def build(rng):
    p = np.linspace(0.0, 100.0, N_POINT)
    data = {}
    for group, base, scale in (("Control", 1.9, 2.6), ("Patient", 3.1, 4.4)):
        mean = joint_curves(p, group)
        data[group] = {j: (m, base + scale * _smooth_noise(rng, p.size))
                       for j, m in mean.items()}
    return p, data


def phase_ribbon(ax):
    for i, (name, x0, x1) in enumerate(PHASES):
        stance = x1 <= 62.0
        base, k = (CONTROL, i) if stance else (GOLD, i - 4)
        amount = (0.86 - 0.07 * k) if stance else (0.84 - 0.09 * k)
        ax.add_patch(Rectangle((x0, 0.0), x1 - x0, 1.0,
                               facecolor=lighten(base, amount),
                               edgecolor="white", linewidth=0.9))
        ax.text((x0 + x1) / 2.0, 0.5, name, ha="center", va="center",
                fontsize=6.0, color=COLORS["ink"])
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(31.0, 1.30, "Stance", ha="center", va="bottom", fontsize=7.0,
            color=COLORS["sub"])
    ax.text(81.0, 1.30, "Swing", ha="center", va="bottom", fontsize=7.0,
            color=COLORS["sub"])


def main():
    set_pub_style()
    rng = np.random.default_rng(510)
    p, data = build(rng)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.0, 0.55],
                          height_ratios=[0.30, 1.0, 1.0, 1.0],
                          wspace=0.30, hspace=0.30, left=0.065, right=0.985,
                          top=0.855, bottom=0.095)
    ax_band = fig.add_subplot(gs[0, 0])
    ax_hip = fig.add_subplot(gs[1, 0])
    ax_knee = fig.add_subplot(gs[2, 0], sharex=ax_hip)
    ax_ankle = fig.add_subplot(gs[3, 0], sharex=ax_hip)
    ax_bar = fig.add_subplot(gs[1:, 1])
    joint_axes = {"Hip": ax_hip, "Knee": ax_knee, "Ankle": ax_ankle}

    phase_ribbon(ax_band)

    # ---- (a)-(c) three joints, one shared angle scale ---------------------
    envelopes = []
    for group in data:
        for m, sd in data[group].values():
            envelopes += [m - sd, m + sd]
    lo, hi = shared_limits(envelopes, pad=0.06)

    titles = {"Hip": "Hip flexion / extension",
              "Knee": "Knee flexion",
              "Ankle": "Ankle dorsiflexion / plantarflexion"}
    for joint, ax in joint_axes.items():
        ax.axhline(0.0, color=COLORS["baseline"], lw=0.7, ls=(0, (3.0, 2.4)),
                   zorder=1)
        for group, colour in (("Control", CONTROL), ("Patient", PATIENT)):
            m, sd = data[group][joint]
            ax.fill_between(p, m - sd, m + sd, color=colour, alpha=0.18, lw=0,
                            zorder=2)
            ax.plot(p, m, color=colour, lw=1.5, zorder=4)
        for group, colour in (("Control", CONTROL), ("Patient", PATIENT)):
            ax.axvline(TOE_OFF[group], color=colour, lw=0.9,
                       ls=(0, (4.0, 2.4)), alpha=0.85, zorder=3)
        ax.set_ylim(lo, hi)
        ax.set_xlim(0.0, 100.0)
        ax.set_ylabel(f"{joint} angle (\u00b0)", fontsize=8.0)
        ax.tick_params(labelsize=7.0)
        ax.set_title(titles[joint], fontsize=8.5, fontweight="bold")
        style_axes(ax)
    ax_hip.tick_params(labelbottom=False)
    ax_knee.tick_params(labelbottom=False)
    ax_ankle.set_xlabel("Gait cycle (%)", fontsize=8.0)

    for group, colour in (("Control", CONTROL), ("Patient", PATIENT)):
        ax_hip.text(TOE_OFF[group] + 1.0, hi - 3.0, f"toe-off {group.lower()}",
                    ha="left", va="top", fontsize=6.2, color=colour,
                    rotation=90)

    peak = {g: float(data[g]["Knee"][0].max()) for g in data}
    loss = 100.0 * (1.0 - peak["Patient"] / peak["Control"])
    ax_knee.annotate(f"peak swing flexion {peak['Patient']:.0f}\u00b0 vs "
                     f"{peak['Control']:.0f}\u00b0 ({loss:.0f} % lower)",
                     xy=(float(p[np.argmax(data['Patient']['Knee'][0])]),
                         peak["Patient"]),
                     xytext=(1.5, hi - 3.0), ha="left", va="top",
                     fontsize=6.6, color=COLORS["ink"],
                     arrowprops=dict(arrowstyle="-", lw=0.7,
                                     color=COLORS["muted"],
                                     connectionstyle="arc3,rad=-0.18"))

    # ---- (d) spatio-temporal parameters, relative to the control mean ------
    names = [r[0] for r in SPATIOTEMPORAL]
    y = np.arange(len(names))[::-1]
    ctrl = np.array([r[2] for r in SPATIOTEMPORAL])
    pat = np.array([r[3] for r in SPATIOTEMPORAL])
    rel = 100.0 * pat / ctrl
    h = 0.34
    ax_bar.barh(y + h / 2, np.full_like(ctrl, 100.0), height=h,
                color=CONTROL, alpha=0.85, edgecolor="white", linewidth=0.6)
    ax_bar.barh(y - h / 2, rel, height=h, color=PATIENT, alpha=0.9,
                edgecolor="white", linewidth=0.6)
    ax_bar.axvline(100.0, color=COLORS["baseline"], lw=0.8, ls=(0, (3.0, 2.4)))
    for yi, v in zip(y, rel):
        ax_bar.text(v + 2.0, yi - h / 2, f"{v:.0f} %", ha="left", va="center",
                    fontsize=6.2, color=PATIENT)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(
        [f"{r[0]}\n{r[2]:g} vs {r[3]:g} {r[1]}" for r in SPATIOTEMPORAL],
        fontsize=6.2, linespacing=1.35)
    ax_bar.set_xlim(0.0, 178.0)
    ax_bar.set_xlabel("Patient value relative to control mean (%)",
                      fontsize=7.5)
    ax_bar.tick_params(axis="x", labelsize=6.5)
    ax_bar.set_title("Spatio-temporal parameters", fontsize=8.5,
                     fontweight="bold")
    style_axes(ax_bar)

    handles = [Line2D([], [], color=CONTROL, lw=1.6, label="Control (n = 18)"),
               Line2D([], [], color=PATIENT, lw=1.6, label="Patient (n = 14)"),
               Line2D([], [], color=COLORS["muted"], lw=6, alpha=0.30,
                      label="\u00b11 SD"),
               Line2D([], [], color=COLORS["muted"], lw=0.9,
                      ls=(0, (4.0, 2.4)), label="toe-off")]

    panel_labels_inches(fig, [ax_hip, ax_knee, ax_ankle, ax_bar],
                        dx=-0.46, dy=0.10)
    savefig(fig, "510_bio_gait_cycle.png")


if __name__ == "__main__":
    main()

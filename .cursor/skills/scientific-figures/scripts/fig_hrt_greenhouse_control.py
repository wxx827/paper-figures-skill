"""A day of greenhouse climate control, simulated one minute at a time.

An explicit Euler integration runs the air temperature against an outside
course and a solar gain, while a deadband controller decides at every step
whether to heat, do nothing, or open the vents proportionally.  Everything
reported is then counted on those same arrays: the hours spent in each control
state, the share of the day inside the deadband, the mean absolute deviation
from the setpoint, and the carbon dioxide actually injected, which the
enrichment loop in the upper side panel withholds whenever the vents are open
far enough to blow it straight out of the roof.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3d6b8e"
GOLD = "#c9a227"
GREEN = "#2e7d5b"

SEED = 1036
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 4.62, 4.48)
AUX_T = (7.44, 3.61, 2.68, 1.89)
AUX_B = (7.44, 1.02, 2.68, 1.82)

DT = 1.0 / 60.0                 # integration step, h
NSTEP = int(24.0 / DT) + 1
DEADBAND = 1.0                  # half width of the tolerated band, degrees C
SP_NIGHT, SP_DAY = 17.0, 21.0
VENT_GAIN = 140.0               # per cent of aperture per degree of overshoot
HEAT_GAIN = 3.0                 # fraction of the boiler per degree of deficit
CO2_SP = 900.0                  # enrichment target, ppm
CO2_OUT = 420.0                 # outside concentration, ppm
VENT_BLOCK = 25.0               # vents wider than this stop the injection, %

U_ENV = 1.60                    # envelope loss, degrees C per h per degree C
U_VENT = 0.060                  # extra loss per h per degree C per % aperture
A_SOL = 0.030                   # solar gain, degrees C per h per W m-2
A_LAMP = 1.60                   # lamp gain, degrees C per h
Q_HEAT = 26.0                   # heating capacity, degrees C per h at full


def drivers(rng):
    """Outside temperature, global radiation and the setpoint schedule."""
    t = np.arange(NSTEP) * DT
    t_out = 11.5 + 5.6 * np.cos(2.0 * np.pi * (t - 15.0) / 24.0)
    t_out += rng.normal(0.0, 0.09, t.size).cumsum() * 0.02
    day = np.clip(np.sin(np.pi * (t - 6.2) / 11.9), 0.0, None)
    cloud = 1.0 - 0.32 * (np.sin(2.0 * np.pi * t / 3.1) ** 2) * (day > 0)
    rad = 760.0 * day ** 1.15 * cloud
    sp = np.where((t >= 6.5) & (t < 18.5), SP_DAY, SP_NIGHT)
    ramp = np.clip((t - 5.5) / 1.0, 0.0, 1.0) * ((t >= 5.5) & (t < 6.5))
    sp = np.where((t >= 5.5) & (t < 6.5),
                  SP_NIGHT + (SP_DAY - SP_NIGHT) * ramp, sp)
    lamp = ((t >= 4.0) & (t < 9.0) | (t >= 15.5) & (t < 20.0)) & (rad < 260.0)
    return t, t_out, rad, sp, lamp.astype(float)


def simulate(t, t_out, rad, sp, lamp):
    """Deadband controller on an Euler integration of the air temperature."""
    n = t.size
    t_in = np.empty(n)
    vent = np.zeros(n)
    heat = np.zeros(n)
    co2 = np.empty(n)
    inject = np.zeros(n)
    t_in[0], co2[0] = SP_NIGHT, CO2_OUT + 60.0
    for i in range(n - 1):
        err = t_in[i] - sp[i]
        if err > DEADBAND:                      # vent, never heat at once
            vent[i] = float(np.clip((err - DEADBAND) * VENT_GAIN, 0.0, 100.0))
            heat[i] = 0.0
        else:
            vent[i] = 0.0
            heat[i] = float(np.clip(-err * HEAT_GAIN, 0.0, 1.0))
        rate = (A_SOL * rad[i] + A_LAMP * lamp[i] + Q_HEAT * heat[i]
                - (U_ENV + U_VENT * vent[i]) * (t_in[i] - t_out[i]))
        t_in[i + 1] = t_in[i] + rate * DT

        uptake = 0.16 * (rad[i] + 210.0 * lamp[i]) ** 0.55
        open_ok = vent[i] < VENT_BLOCK and (rad[i] > 40.0 or lamp[i] > 0.5)
        want = np.clip((CO2_SP - co2[i]) * 0.9, 0.0, 260.0)
        inject[i] = want if open_ok else 0.0
        leak = (0.55 + 0.085 * vent[i]) * (co2[i] - CO2_OUT)
        co2[i + 1] = co2[i] + (inject[i] - leak - uptake) * DT
    vent[-1], heat[-1], inject[-1] = vent[-2], heat[-2], inject[-2]
    return t_in, vent, heat, co2, inject


def widest_gap(t, mask):
    """Centre of the longest stretch where the state is off, for a label."""
    off = ~np.asarray(mask, dtype=bool)
    best, run, start = (0, 0.5 * (t[0] + t[-1])), 0, 0
    for i, v in enumerate(off):
        if v:
            run = run + 1 if run else 1
            start = i - run + 1
            if run > best[0]:
                best = (run, 0.5 * (t[start] + t[i]))
        else:
            run = 0
    return best[1]


def main_panel(ax, t, t_in, t_out, sp, vent, heat, lamp):
    ax.axvspan(0.0, 6.2, color="#eef0f4", zorder=1)
    ax.axvspan(18.1, 24.0, color="#eef0f4", zorder=1)
    ax.text(3.1, 3.3, "night", ha="center", va="bottom", fontsize=6.3,
            color=MUTED, zorder=4)
    ax.text(21.0, 3.3, "night", ha="center", va="bottom", fontsize=6.3,
            color=MUTED, zorder=4)

    ax.fill_between(t, sp - DEADBAND, sp + DEADBAND, color="#dfe6df",
                    alpha=0.85, zorder=2, lw=0)
    ax.plot(t, sp, color=INK, lw=1.1, ls=(0, (4.5, 2.4)), zorder=6)
    ax.plot(t, t_out, color=BLUE, lw=1.3, zorder=5)
    ax.plot(t, t_in, color=ORANGE, lw=2.0, zorder=8)

    out = np.abs(t_in - sp) > DEADBAND
    ax.fill_between(t, 2.0, 31.0, where=out, color="#f6e2e2", alpha=0.55,
                    zorder=1, lw=0)

    for arr, y, col, name in ((heat > 0.02, 29.8, RED, "heating"),
                              (vent > 1.0, 28.8, BLUE, "vents open"),
                              (lamp > 0.5, 27.8, GOLD, "lamps on")):
        ax.fill_between(t, y - 0.28, y + 0.28, where=arr, color=col,
                        alpha=0.85, zorder=7, lw=0)
        ax.text(widest_gap(t, arr), y, name, ha="center", va="center",
                fontsize=6.1, color=col, zorder=8)

    axr = ax.twinx()
    axr.fill_between(t, 0.0, vent, color="#cfe0ea", alpha=0.85, zorder=3,
                     lw=0)
    axr.plot(t, vent, color=BLUE, lw=0.8, alpha=0.9, zorder=4)
    axr.set_ylim(0.0, 170.0)
    axr.set_yticks([0, 25, 50, 75, 100])
    axr.set_ylabel("vent aperture (%)", fontsize=7.6, color=BLUE)
    axr.tick_params(axis="y", labelsize=7.0, colors=BLUE)
    axr.spines["top"].set_visible(False)
    axr.spines["right"].set_color(BLUE)
    axr.set_zorder(1)
    ax.set_zorder(2)
    ax.patch.set_visible(False)

    i_hot = int(np.argmax(t_in))
    ax.annotate("peak {:.1f} " .format(t_in[i_hot]) + u"\u00b0" +
                "C at {:.1f} h with the vents at {:.0f} %"
                .format(t[i_hot], vent[i_hot]), xy=(t[i_hot], t_in[i_hot]),
                xytext=(t[i_hot] + 1.1, 26.2), fontsize=6.4, color=ORANGE,
                ha="left", va="center", zorder=10,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8,
                                shrinkA=2, shrinkB=4))
    i_cold = int(np.argmin(t_in[t < 6.0]))
    ax.annotate("heating holds the night setpoint to within {:.1f} "
                .format(abs(t_in[i_cold] - sp[i_cold])) + u"\u00b0" + "C",
                xy=(t[i_cold], t_in[i_cold]), xytext=(1.0, 12.6),
                fontsize=6.4, color=RED, ha="left", va="center", zorder=10,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.8, shrinkA=2,
                                shrinkB=4))

    handles = [Line2D([], [], color=ORANGE, lw=2.0, label="greenhouse air"),
               Line2D([], [], color=BLUE, lw=1.3, label="outside air"),
               Line2D([], [], color=INK, lw=1.1, ls=(0, (4.5, 2.4)),
                      label="setpoint"),
               Patch(facecolor="#dfe6df", edgecolor="none",
                     label="deadband $\\pm$ {:.1f} ".format(DEADBAND) +
                     u"\u00b0" + "C"),
               Patch(facecolor="#f6e2e2", edgecolor="none",
                     label="outside the deadband"),
               Patch(facecolor="#cfe0ea", edgecolor=BLUE, lw=0.6,
                     label="vent aperture")]
    ax.legend(handles=handles, loc="upper left", fontsize=6.2, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.6, ncol=2, columnspacing=0.9, borderaxespad=0.4,
              labelspacing=0.24, bbox_to_anchor=(0.012, 0.795))

    ax.set_xlim(0.0, 24.0)
    ax.set_ylim(2.0, 31.0)
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_xlabel("time of day (h)")
    ax.set_ylabel("air temperature (" + u"\u00b0" + "C)")
    ax.grid(axis="y", lw=0.5, alpha=0.32)
    ax.set_title("One spring day under a deadband climate controller",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)


def co2_panel(ax, t, co2, inject, vent):
    ax.fill_between(t, CO2_OUT, co2, color="#dcece2", alpha=0.9, zorder=2,
                    lw=0)
    ax.plot(t, co2, color=GREEN, lw=1.6, zorder=6)
    ax.axhline(CO2_SP, color=INK, lw=0.9, ls=(0, (4.0, 2.4)), zorder=5)
    ax.axhline(CO2_OUT, color=MUTED, lw=0.8, zorder=4)
    ax.text(0.4, CO2_SP + 18.0, "enrichment target {:.0f} ppm"
            .format(CO2_SP), ha="left", va="bottom", fontsize=6.1, color=INK,
            zorder=8)
    ax.text(23.6, CO2_OUT + 14.0, "outside {:.0f} ppm".format(CO2_OUT),
            ha="right", va="bottom", fontsize=6.0, color=MUTED, zorder=8)

    blocked = (vent >= VENT_BLOCK) & (co2 < CO2_SP - 20.0)
    ax.fill_between(t, 330.0, 1080.0, where=blocked, color="#f3e3e3",
                    alpha=0.75, zorder=1, lw=0)
    if blocked.any():
        ax.text(float(t[blocked][len(t[blocked]) // 2]) + 0.4, 1055.0,
                "injection off,\nvents past {:.0f} %".format(VENT_BLOCK),
                ha="left", va="top", fontsize=6.0, color="#8c3f3f",
                linespacing=1.3, zorder=8)

    ax.set_xlim(0.0, 24.0)
    ax.set_ylim(330.0, 1080.0)
    ax.set_xticks(np.arange(0, 25, 6))
    ax.set_xlabel("time of day (h)", fontsize=6.8)
    ax.set_ylabel("CO$_2$ (ppm)", fontsize=6.8)
    ax.grid(lw=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Enrichment competes with ventilation", fontsize=8.0,
                 fontweight="bold", color=INK)


def state_panel(ax, hours, share_in, mad):
    names = [n for (n, _h, _c) in hours]
    vals = [h for (_n, h, _c) in hours]
    cols = [c for (_n, _h, c) in hours]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, height=0.6, color=cols, edgecolor="white", linewidth=0.7,
            zorder=4)
    for yi, v in zip(y, vals):
        ax.text(v + 0.16, yi, "{:.1f} h  ({:.0f} % of the day)"
                .format(v, 100.0 * v / 24.0), ha="left", va="center",
                fontsize=6.0, color=SUB, zorder=6)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.4)
    ax.set_xlim(0.0, max(vals) * 1.85)
    ax.set_ylim(-0.62, len(names) - 0.38)
    ax.set_xlabel("time in each control state (h)", fontsize=6.8)
    ax.grid(axis="x", lw=0.5, alpha=0.48)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(max(vals) * 1.82, -0.52,
            "{:.0f} % of the day inside the deadband\nmean absolute "
            "deviation {:.2f} ".format(share_in, mad) + u"\u00b0" + "C",
            ha="right", va="bottom", fontsize=6.1, color=INK,
            linespacing=1.32, zorder=7)
    ax.set_title("What the controller actually did", fontsize=8.0,
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
    rng = np.random.default_rng(SEED)

    t, t_out, rad, sp, lamp = drivers(rng)
    t_in, vent, heat, co2, inject = simulate(t, t_out, rad, sp, lamp)

    heating = heat > 0.02
    venting = vent > 1.0
    passive = ~heating & ~venting
    assert not np.any(heating & venting), "heating and venting must not overlap"
    hours = [("Heating", float(heating.sum()) * DT, RED),
             ("Passive", float(passive.sum()) * DT, "#cbd5cb"),
             ("Venting", float(venting.sum()) * DT, BLUE)]
    assert abs(sum(h for (_n, h, _c) in hours) - t.size * DT) < 1e-6, \
        "the three control states must partition the day"

    inside = np.abs(t_in - sp) <= DEADBAND
    share_in = 100.0 * inside.mean()
    mad = float(np.abs(t_in - sp).mean())
    co2_kg = float(inject.sum() * DT) * 1.8e-3      # g m-2 over the day

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_co2 = axes_inches(fig, AUX_T)
    ax_st = axes_inches(fig, AUX_B)

    main_panel(ax_main, t, t_in, t_out, sp, vent, heat, lamp)
    co2_panel(ax_co2, t, co2, inject, vent)
    state_panel(ax_st, hours, share_in, mad)



    panel_labels_inches(fig, [ax_main, ax_co2, ax_st], dx=-0.62, dy=0.14)
    savefig(fig, "1036_hrt_greenhouse_control.png")


if __name__ == "__main__":
    main()

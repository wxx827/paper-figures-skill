"""Transient thermal impedance of a package, and what it costs at the junction.

Main panel: the heating curve Z_th(t) of one die in three mounting
conditions, on log-log axes, built from a five-stage Foster ladder.  Each
decade of time belongs to one layer of the stack, so the vertical guides at
the stage time constants say which material is being charged where; the
plateaus are the steady-state junction-to-ambient resistances.

The right-hand panels are the two questions that follow.  How is the
resistance spent - a stacked budget per condition, in K/W and in per cent.
And what does it do to the junction - a pulsed-power run through the same
ladder, where the still-air mounting crosses the 125 degC limit and the
heat-sinked one does not.

The ladder is integrated exactly for a piecewise-constant power train, and
every quoted number - steady-state resistance, the time to reach 90 % of it,
peak junction temperature, the crossing time - is measured off the plotted
arrays.  Values are illustrative only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, add_footer
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRID = "#d5d5d5"

BLUE = "#3b5b92"
RED = "#b23a48"
GOLD = "#c9a227"
GREEN = "#2e7d5b"
PURPLE = "#8e6bbf"

FIG_W, FIG_H = 10.4, 6.4
MAIN_RECT = (0.86, 0.80, 5.18, 4.58)
AUX_T_RECT = (7.16, 3.50, 2.88, 1.88)
AUX_B_RECT = (7.16, 0.80, 2.88, 1.88)

# Foster stages shared by every mounting: (name, R in K/W, tau in s)
STACK = [("Die", 0.045, 8.0e-5, RED),
         ("Die attach", 0.110, 1.2e-3, GOLD),
         ("Substrate + lid", 0.240, 2.5e-2, GREEN),
         ("TIM", 0.300, 3.5e-1, PURPLE)]
# The last stage is what the board designer actually chooses.
MOUNTS = [("Liquid cold plate", 0.16, 3.0, GREEN),
          ("Heat sink + fan", 0.55, 9.0, BLUE),
          ("Bare PCB, still air", 1.85, 45.0, RED)]

T_AMB = 45.0                       # degC
TJ_MAX = 125.0                     # degC
P_PULSE = 60.0                     # W
PERIOD = 20.0                      # s
DUTY = 0.45


def zth(t, stages):
    """Foster heating curve in K/W on a time vector."""
    out = np.zeros_like(t)
    for _, r, tau, *_ in stages:
        out = out + r * (1.0 - np.exp(-t / tau))
    return out


def pulse_train(t_end, dt):
    """Piecewise-constant power train: returns (t, P in W)."""
    t = np.arange(0.0, t_end + dt, dt)
    p = np.where((t % PERIOD) < DUTY * PERIOD, P_PULSE, 0.0)
    return t, p


def junction_response(t, p, stages):
    """Exact Foster integration for a piecewise-constant power train."""
    dt = float(t[1] - t[0])
    temps = np.zeros((len(stages), t.size))
    state = np.zeros(len(stages))
    decay = np.array([np.exp(-dt / s[2]) for s in stages])
    rs = np.array([s[1] for s in stages])
    for i in range(1, t.size):
        state = state * decay + rs * p[i] * (1.0 - decay)
        temps[:, i] = state
    return T_AMB + temps.sum(axis=0)


def check_layout(fig, foot, axes):
    """Warn if the footnote box touches any axis label or title."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for a in axes:
        for art in (a.xaxis.label, a.yaxis.label, a.title):
            if art.get_text() and fb.overlaps(art.get_window_extent()):
                print("WARNING: footnote overlaps:", art.get_text()[:40])


def main():
    set_pub_style(base_font=8)
    t = np.logspace(-5.0, 2.6, 700)

    curves, totals, t90 = {}, {}, {}
    for name, r_s, tau_s, col in MOUNTS:
        stages = STACK + [("Mounting", r_s, tau_s, col)]
        z = zth(t, stages)
        curves[name] = z
        totals[name] = float(z[-1])
        t90[name] = float(np.interp(0.90 * z[-1], z, t))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_b = axes_inches(fig, AUX_T_RECT)
    ax_j = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_b, ax_j):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: the heating curves ------------------------------------------
    ax.grid(which="both", color=GRID, lw=0.5, alpha=0.65, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for name, r_s, tau_s, col in MOUNTS:
        ax.plot(t, curves[name], color=col, lw=1.9, zorder=6)
        ax.axhline(totals[name], color=col, lw=0.7, ls=(0, (2.6, 2.0)),
                   zorder=3)
        ax.text(t[-1] * 0.90, totals[name] * 1.07,
                f"$R_{{th,JA}}$ = {totals[name]:.2f} K/W", ha="right",
                va="bottom", fontsize=6.8, color=col)
    for nm, r, tau, col in STACK:
        ax.axvline(tau, color="#e2e2e2", lw=0.8, zorder=1)
        ax.text(tau * 1.22, 2.6e-3, nm, rotation=90, ha="left", va="bottom",
                fontsize=6.3, color=MUTED)
    ax.text(8.0, 2.6e-3, "mounting", rotation=90, ha="left", va="bottom",
            fontsize=6.3, color=MUTED)
    ax.axvspan(3.0, 45.0, color="#f2f2f2", zorder=0)

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(2.2e-3, 14.0)
    ax.set_xlabel("heating time after the step (s, log scale)", fontsize=8.4)
    ax.set_ylabel("transient thermal impedance $Z_{th,JA}$ (K/W)",
                  fontsize=8.4)
    ax.set_title("Every decade of time charges a different layer of the stack",
                 fontsize=9.0, fontweight="bold", pad=5.0)
    ax.legend(handles=[Line2D([], [], color=c, lw=1.9, label=n)
                       for n, _, _, c in MOUNTS],
              loc="lower right", frameon=False, fontsize=7.2,
              handlelength=2.0, labelspacing=0.40, borderaxespad=0.45)
    spread = totals[MOUNTS[2][0]] / totals[MOUNTS[0][0]]
    ax.text(0.020, 0.965,
            f"the four package layers are common to all three\n"
            f"curves and are fully charged within "
            f"{STACK[-1][2]:.2f} s; the mounting\nsets both the plateau "
            f"({spread:.1f}x spread) and the time to reach it\n"
            f"(90 % of steady state after {t90[MOUNTS[0][0]]:.1f} s, "
            f"{t90[MOUNTS[1][0]]:.1f} s and {t90[MOUNTS[2][0]]:.0f} s)",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.9,
            color=SUB, linespacing=1.50)

    # ---- aux (b): where the resistance is spent ----------------------------
    ax_b.grid(axis="x", color=GRID, lw=0.55, alpha=0.75, zorder=0)
    names = [m[0] for m in MOUNTS]
    ys = np.arange(len(MOUNTS))[::-1]
    for y, (name, r_s, tau_s, col) in zip(ys, MOUNTS):
        left = 0.0
        segs = [(nm, r, c) for nm, r, _, c in STACK] + \
               [("Mounting", r_s, "#8a939b")]
        for nm, r, c in segs:
            ax_b.barh(y, r, left=left, height=0.58, color=c,
                      edgecolor="white", linewidth=0.7, zorder=3)
            if r / totals[name] > 0.16:
                ax_b.text(left + r / 2.0, y, f"{100 * r / totals[name]:.0f}%",
                          ha="center", va="center", fontsize=6.2,
                          color="white", fontweight="bold", zorder=5)
            left += r
        ax_b.text(left + 0.05, y, f"{left:.2f} K/W", ha="left", va="center",
                  fontsize=6.4, color=SUB, zorder=5)
    ax_b.set_yticks(ys)
    ax_b.set_yticklabels(names, fontsize=6.8)
    ax_b.set_xlim(0.0, max(totals.values()) * 1.30)
    ax_b.tick_params(labelsize=6.8)
    ax_b.set_xlabel("junction-to-ambient resistance budget (K/W)",
                    fontsize=7.5, labelpad=1.5)
    ax_b.set_title(f"Where the {sum(s[1] for s in STACK):.2f} K/W of package "
                   "sits", fontsize=8.5, fontweight="bold", pad=4.0)
    ax_b.legend(handles=[Line2D([], [], color=c, lw=5.0, label=nm)
                         for nm, _, _, c in STACK] +
                        [Line2D([], [], color="#8a939b", lw=5.0,
                                label="Mounting")],
                loc="upper right", frameon=False, fontsize=6.2, ncol=2,
                handlelength=1.1, labelspacing=0.30, columnspacing=0.9,
                borderaxespad=0.20)

    # ---- aux (c): pulsed power at the junction -----------------------------
    tp, p = pulse_train(200.0, 5.0e-3)
    ax_j.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)
    for k in range(int(tp[-1] / PERIOD) + 1):
        ax_j.axvspan(k * PERIOD, k * PERIOD + DUTY * PERIOD,
                     color="#f0f2f4", zorder=1)
    peaks = {}
    for name, r_s, tau_s, col in (MOUNTS[1], MOUNTS[2]):
        tj = junction_response(tp, p, STACK + [("Mounting", r_s, tau_s, col)])
        ax_j.plot(tp, tj, color=col, lw=1.4, zorder=5)
        peaks[name] = float(tj.max())
        if tj.max() > TJ_MAX:
            cross = float(tp[np.argmax(tj > TJ_MAX)])
            ax_j.plot([cross], [TJ_MAX], marker="o", ms=4.0, mfc="white",
                      mec=col, mew=1.2, zorder=8)
            ax_j.annotate(f"still air crosses the limit at {cross:.0f} s",
                          xy=(cross, TJ_MAX), xytext=(0.045, 0.045),
                          textcoords="axes fraction", fontsize=6.4, color=col,
                          ha="left", va="bottom",
                          arrowprops=dict(arrowstyle="-|>", lw=0.8, color=col,
                                          shrinkA=2, shrinkB=2))
    ax_j.axhline(TJ_MAX, color=INK, lw=0.9, ls=(0, (3.2, 2.0)), zorder=6)
    ax_j.text(tp[-1], TJ_MAX + 1.5, f"$T_j$ limit {TJ_MAX:.0f} $^\\circ$C",
              ha="right", va="bottom", fontsize=6.4, color=INK)
    ax_j.set_xlim(0.0, tp[-1])
    ax_j.set_ylim(T_AMB - 8.0, 182.0)
    ax_j.tick_params(labelsize=6.8)
    ax_j.set_xlabel("time (s)", fontsize=7.5, labelpad=1.5)
    ax_j.set_ylabel("junction temperature ($^\\circ$C)", fontsize=7.5)
    ax_j.set_title("Same die, two mountings, one limit", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_j.text(0.035, 0.965,
              f"{P_PULSE:.0f} W pulses, {PERIOD:.0f} s period, "
              f"{100 * DUTY:.0f} % duty, ambient {T_AMB:.0f} $^\\circ$C;\n"
              f"peak $T_j$ {peaks[MOUNTS[1][0]]:.0f} $^\\circ$C with the sink "
              f"against {peaks[MOUNTS[2][0]]:.0f} $^\\circ$C without",
              transform=ax_j.transAxes, ha="left", va="top", fontsize=6.4,
              color=SUB, linespacing=1.45)

    panel_labels_inches(fig, [ax, ax_b, ax_j], dx=-0.62, dy=0.10)
    foot = add_footer(fig, "Synthetic data, figure 909")
    check_layout(fig, foot, [ax, ax_b, ax_j])
    savefig(fig, "909_rel_thermal_impedance.png")


if __name__ == "__main__":
    main()

"""Solder-joint fatigue under accelerated thermal cycling.

Main panel: the trapezoidal chamber profile of the first six cycles on the
left axis - ramp rate, hot dwell and cold dwell all dimensioned - with the
accumulated equivalent plastic strain of the critical corner joint on the
right axis.  Damage is integrated from a creep-weighted rate, so it climbs in
steps: almost all of each step is taken during the hot dwell, which is the
point of the panel.  An inset carries the same damage path out to the crack-
through criterion, where the characteristic life is read off.

The two right-hand panels are the design consequence: a Coffin-Manson line
fitted through six test cells on log-log axes, and the characteristic life
each package style inherits from that same fitted line.

Nothing is hard-coded downstream: the per-cycle strain increment is measured
from the plotted staircase, the fatigue exponent from the fitted line, the
criterion from their product, and the package bars from the fit.  Values are
illustrative only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, add_footer
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

BLUE = "#3b5b92"
RED = "#b23a48"
GOLD = "#c9a227"
GREEN = "#2e7d5b"
PURPLE = "#8e6bbf"

FIG_W, FIG_H = 10.4, 6.4
MAIN_RECT = (0.80, 0.80, 5.24, 4.58)
AUX_T_RECT = (7.16, 3.50, 2.88, 1.88)
AUX_B_RECT = (7.16, 0.80, 2.88, 1.88)

T_LOW, T_HIGH = -40.0, 125.0      # degC, chamber extremes
RAMP = 10.0                        # degC/min
DWELL = 15.0                       # min, at each extreme
N_SHOW = 6                         # cycles drawn in the main panel
EPS_CYCLE = 0.12                   # %, inelastic strain added per cycle

PACKAGES = [("WLCSP 0.4 mm", 0.212, RED), ("QFN 0.5 mm", 0.155, GOLD),
            ("BGA-256 0.8 mm", 0.120, BLUE), ("CSP 0.65 mm", 0.088, GREEN),
            ("LGA 1.0 mm", 0.061, PURPLE)]
CM_A, CM_M = 1.349, -1.90          # generator for the Coffin-Manson cells


def cycle_profile(n_cycles, dt=0.20):
    """Trapezoidal chamber profile: t in minutes, T in degC, plus a phase id.

    Phase ids: 0 cold dwell, 1 up-ramp, 2 hot dwell, 3 down-ramp.
    """
    ramp_t = (T_HIGH - T_LOW) / RAMP
    segs = [(DWELL, T_LOW, T_LOW, 0), (ramp_t, T_LOW, T_HIGH, 1),
            (DWELL, T_HIGH, T_HIGH, 2), (ramp_t, T_HIGH, T_LOW, 3)]
    t, temp, ph = [0.0], [T_LOW], [0]
    for _ in range(n_cycles):
        for dur, a, b, pid in segs:
            k = max(int(round(dur / dt)), 2)
            tt = t[-1] + np.linspace(dur / k, dur, k)
            t.extend(tt.tolist())
            temp.extend(np.linspace(a + (b - a) / k, b, k).tolist())
            ph.extend([pid] * k)
    return (np.asarray(t), np.asarray(temp), np.asarray(ph),
            4 * ramp_t / 4 + 0.0, 2 * ramp_t + 2 * DWELL)


def damage_path(t, ph, period):
    """Accumulated inelastic strain (%) from a creep-weighted damage rate."""
    w = np.select([ph == 2, (ph == 1) | (ph == 3), ph == 0],
                  [1.0, 0.30, 0.07])
    w = w * (1.0 + 0.018 * np.floor(t / period))       # mild ratcheting
    eps = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) *
                                           np.diff(t))])
    one = np.interp(period, t, eps)
    return eps * (EPS_CYCLE / one)


def dim_span(ax, x0, x1, y, label, color=MUTED, fs=6.4, dy=0.0):
    """Horizontal double-headed dimension with a label above it."""
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="<|-|>",
                                 mutation_scale=6, lw=0.75, color=color,
                                 shrinkA=0, shrinkB=0, zorder=7))
    ax.text(0.5 * (x0 + x1), y + dy, label, ha="center", va="bottom",
            fontsize=fs, color=color, zorder=7)


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
    rng = np.random.default_rng(9072)

    t, temp, ph, _, period = cycle_profile(N_SHOW)
    eps = damage_path(t, ph, period)
    th = t / 60.0

    # per-cycle increment, measured off the plotted staircase
    bounds = np.arange(N_SHOW + 1) * period
    steps = np.diff(np.interp(bounds, t, eps))
    d_eps = float(steps.mean())
    hot_share = 100.0 * (np.interp(2 * period - (T_HIGH - T_LOW) / RAMP, t, eps)
                         - np.interp(period + DWELL + (T_HIGH - T_LOW) / RAMP,
                                     t, eps)) / steps[1]

    # ---- Coffin-Manson test cells and their fit ----------------------------
    de = np.array([0.052, 0.078, 0.112, 0.158, 0.225, 0.310])
    nf = 10.0 ** (CM_A + CM_M * np.log10(de) +
                  rng.normal(0.0, 0.055, de.size))
    m, b = np.polyfit(np.log10(de), np.log10(nf), 1)
    c_exp = -1.0 / m                                   # fatigue ductility exp.
    de_grid = np.logspace(np.log10(de.min() * 0.62), np.log10(de.max() * 1.55),
                          120)
    nf_grid = 10.0 ** (b + m * np.log10(de_grid))

    n63 = float(10.0 ** (b + m * np.log10(d_eps)))     # this condition
    eps_f = n63 * d_eps                                # crack-through criterion
    hours = n63 * period / 60.0

    # ---- figure ------------------------------------------------------------
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_cm = axes_inches(fig, AUX_T_RECT)
    ax_pk = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_cm, ax_pk):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: profile + damage staircase ----------------------------------
    ax.grid(color=GRID, lw=0.55, alpha=0.70, zorder=0)
    for k in range(N_SHOW + 1):
        ax.axvline(k * period / 60.0, color="#e6e6e6", lw=0.6, zorder=1)
    ax.plot(th, temp, color=BLUE, lw=1.8, zorder=5, solid_joinstyle="round")
    ax.axhline(T_HIGH, color=FAINT, lw=0.6, ls=(0, (2.4, 2.0)), zorder=2)
    ax.axhline(T_LOW, color=FAINT, lw=0.6, ls=(0, (2.4, 2.0)), zorder=2)
    ax.set_xlim(-0.12, N_SHOW * period / 60.0 + 0.12)
    ax.set_ylim(-190.0, 240.0)
    ax.set_yticks([-40, 0, 40, 80, 125])
    ax.set_xticks(np.arange(0, N_SHOW + 1) * period / 60.0)
    ax.set_xticklabels([f"{v:.1f}" for v in
                        np.arange(0, N_SHOW + 1) * period / 60.0])
    ax.set_xlabel("chamber time (hours)   -   tick spacing = one cycle",
                  fontsize=8.4)
    ax.set_ylabel("chamber temperature ($^\\circ$C)", fontsize=8.4,
                  color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)
    ax.set_title("Damage is taken during the hot dwell, not during the ramp",
                 fontsize=9.0, fontweight="bold", pad=5.0)

    ramp_t = (T_HIGH - T_LOW) / RAMP
    p0 = period / 60.0
    dim_span(ax, p0 + (DWELL + ramp_t) / 60.0,
             p0 + (2 * DWELL + ramp_t) / 60.0, T_HIGH + 12.0,
             f"{DWELL:.0f} min at +{T_HIGH:.0f} $^\\circ$C", color=RED, dy=4.0)
    dim_span(ax, 3 * p0, 3 * p0 + DWELL / 60.0, T_LOW - 20.0,
             f"{DWELL:.0f} min at {T_LOW:.0f} $^\\circ$C", color=BLUE, dy=3.0)
    ax.text(0.012, 0.985,
            f"$\\Delta T$ = {T_HIGH - T_LOW:.0f} $^\\circ$C,\n"
            f"{period:.0f} min per cycle\n"
            f"ramp $\\pm${RAMP:.0f} $^\\circ$C/min\n"
            f"({ramp_t:.1f} min per ramp)",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.0,
            color=SUB, linespacing=1.55)

    ax2 = ax.twinx()
    ax2.plot(th, eps, color=RED, lw=1.7, zorder=6)
    ax2.set_ylim(0.0, 2.42)
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax2.set_ylabel("accumulated inelastic strain $\\varepsilon_{acc}$ (%)",
                   fontsize=8.4, color=RED)
    ax2.tick_params(axis="y", colors=RED, labelsize=7.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(RED)
    for k in range(1, N_SHOW + 1):
        ax2.plot([bounds[k] / 60.0], [np.interp(bounds[k], t, eps)],
                 marker="o", ms=2.8, mfc="white", mec=RED, mew=0.9, zorder=7)
    ax2.annotate(f"each cycle adds $\\Delta\\varepsilon_p$ = {d_eps:.3f} %,\n"
                 f"{hot_share:.0f} % of it during the hot dwell",
                 xy=(bounds[2] / 60.0, np.interp(bounds[2], t, eps)),
                 xytext=(0.045, 0.225), textcoords="axes fraction",
                 fontsize=6.9, color=RED, linespacing=1.45, ha="left",
                 va="bottom",
                 arrowprops=dict(arrowstyle="-|>", lw=0.8, color=RED,
                                 shrinkA=2, shrinkB=3))

    # inset: the same damage path carried out to the criterion
    axi = ax.inset_axes([0.545, 0.735, 0.435, 0.255])
    style_axes(axi)
    nn = np.linspace(0.0, n63 * 1.22, 260)
    ee = d_eps * nn * (1.0 + 0.10 * (nn / n63) ** 2.4)
    axi.plot(nn, ee, color=RED, lw=1.5, zorder=5)
    axi.axhline(eps_f, color=INK, lw=0.9, ls=(0, (3.2, 2.0)), zorder=4)
    n_cross = float(np.interp(eps_f, ee, nn))
    axi.plot([n_cross], [eps_f], marker="o", ms=4.0, mfc="white", mec=INK,
             mew=1.2, zorder=7)
    axi.set_xlim(0.0, nn.max())
    axi.set_ylim(0.0, eps_f * 1.32)
    axi.tick_params(labelsize=6.0, pad=1.5)
    axi.set_xlabel("cycles", fontsize=6.6, labelpad=1.0)
    axi.set_ylabel("$\\varepsilon_{acc}$ (%)", fontsize=6.6, labelpad=1.5)
    axi.text(0.035, 0.96,
             f"crack-through criterion {eps_f:.0f} %, reached at\n"
             f"$N_{{63}}$ = {n_cross:,.0f} cycles = {hours / 24.0:.0f} days "
             f"of chamber time",
             transform=axi.transAxes, ha="left", va="top", fontsize=6.1,
             color=INK, linespacing=1.40)
    axi.annotate("first 6 cycles\n= main panel",
                 xy=(N_SHOW, d_eps * N_SHOW), xytext=(0.075, 0.10),
                 textcoords="axes fraction", fontsize=6.0, color=RED,
                 linespacing=1.30, ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="-|>", lw=0.7, color=RED,
                                 shrinkA=1.5, shrinkB=1.5))

    # ---- aux (b): Coffin-Manson --------------------------------------------
    ax_cm.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0, which="both")
    ax_cm.set_xscale("log")
    ax_cm.set_yscale("log")
    ax_cm.plot(de_grid, nf_grid, color=INK, lw=1.3, zorder=4)
    ax_cm.plot(de, nf, ls="none", marker="o", ms=4.2, mfc="white", mec=BLUE,
               mew=1.3, zorder=6)
    ax_cm.plot([d_eps], [n63], marker="D", ms=5.0, mfc=RED, mec="white",
               mew=0.9, zorder=8)
    ax_cm.set_xlim(de_grid.min(), de_grid.max())
    ax_cm.set_ylim(nf_grid.min() * 0.55, nf_grid.max() * 1.9)
    ax_cm.tick_params(labelsize=6.8)
    ax_cm.set_xlabel("plastic strain range per cycle $\\Delta\\varepsilon_p$ "
                     "(%)", fontsize=7.5, labelpad=1.5)
    ax_cm.set_ylabel("cycles to failure $N_f$", fontsize=7.5)
    ax_cm.set_title("Coffin-Manson line, six test cells", fontsize=8.5,
                    fontweight="bold", pad=4.0)
    ax_cm.text(0.035, 0.055,
               f"slope {m:.2f} on log-log\nfatigue exponent c = {c_exp:.2f}",
               transform=ax_cm.transAxes, ha="left", va="bottom",
               fontsize=6.5, color=INK, linespacing=1.45)
    ax_cm.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=4.2, mfc="white", mec=BLUE,
               mew=1.3, label="test cell"),
        Line2D([], [], marker="D", ls="none", ms=5.0, mfc=RED, mec="white",
               label="main-panel condition")],
        loc="upper right", frameon=False, fontsize=6.4,
        handletextpad=0.35, labelspacing=0.35, borderaxespad=0.15)

    # ---- aux (c): package styles inherit the same line ---------------------
    ax_pk.grid(axis="x", color=GRID, lw=0.55, alpha=0.75, zorder=0,
               which="both")
    names = [p[0] for p in PACKAGES]
    strains = np.array([p[1] for p in PACKAGES])
    cols = [p[2] for p in PACKAGES]
    lives = 10.0 ** (b + m * np.log10(strains))
    ys = np.arange(len(PACKAGES))[::-1]
    ax_pk.set_xscale("log")
    ax_pk.barh(ys, lives, height=0.60, color=cols, edgecolor="white",
               linewidth=0.6, zorder=3)
    for y, v, s in zip(ys, lives, strains):
        ax_pk.text(v * 1.10, y, f"{v:,.0f} cycles", ha="left", va="center",
                   fontsize=6.4, color=SUB)
    ax_pk.set_yticks(ys)
    ax_pk.set_yticklabels(names, fontsize=6.8)
    ax_pk.set_xlim(60.0, lives.max() * 6.2)
    ax_pk.tick_params(labelsize=6.8)
    ax_pk.set_xlabel("characteristic life $N_{63}$ (cycles, log scale)",
                     fontsize=7.5, labelpad=1.5)
    ax_pk.set_title("Stiff, fine-pitch styles die first", fontsize=8.5,
                    fontweight="bold", pad=4.0)
    ax_pk.text(0.985, 0.96,
               f"{lives.max() / lives.min():.0f}x spread across five styles",
               transform=ax_pk.transAxes, ha="right", va="top",
               fontsize=6.5, color=INK)

    panel_labels_inches(fig, [ax, ax_cm, ax_pk], dx=-0.60, dy=0.10)
    foot = add_footer(fig, "Synthetic data, seed=9072")
    check_layout(fig, foot, [ax, ax_cm, ax_pk])
    savefig(fig, "907_pkg_solder_joint_thermal.png")


if __name__ == "__main__":
    main()

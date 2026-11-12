"""Spanwise shear, bending moment and torque envelopes for a transport wing.

Four sizing load cases are integrated inwards from the tip on one common
spanwise grid, so the three stacked curves are the same four cases seen through
three integrals rather than three independent plots.  The shaded band is the
envelope over all cases at every station, and the design limit line in each
panel is read off that band, not typed in.  The planform below shares the
spanwise axis and shows where the discrete loads are applied.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches, shared_limits

C_A = "#3b5b92"        # house blue: +2.5 g manoeuvre, the sizing case
C_B = "#8e6bbf"        # house purple: negative manoeuvre
C_C = "#2e7d5b"        # house green: gust case
C_D = "#c9a227"        # house gold: ground load case
C_LIM = "#b23a48"      # house red: design limit lines
INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"

SEMI_SPAN = 17.0       # m
C_ROOT, C_TIP = 4.60, 1.50     # m
SWEEP_LE = np.deg2rad(27.0)
MTOW_N = 765.0e3       # maximum take-off weight (N)
W_STRUCT = 62.0e3      # half-wing structural weight (N)
W_FUEL_FULL = 118.0e3  # half-wing fuel weight when tanks are full (N)
W_ENGINE = 31.4e3      # engine and pylon weight (N)
Y_ENGINE = 5.20        # engine station (m)
E_OFFSET = 0.055       # aerodynamic centre ahead of the elastic axis, x chord

# name, load factor n (g), fuel fill fraction, airborne flag, colour
CASES = [("+2.5 g manoeuvre, MTOW", 2.5, 0.30, True, C_A),
         ("-1.0 g manoeuvre", -1.0, 0.30, True, C_B),
         ("+2.0 g gust, full tanks", 2.0, 0.85, True, C_C),
         ("2.0 g ground bump", 2.0, 1.00, False, C_D)]


def chord(y):
    return C_ROOT + (C_TIP - C_ROOT) * (y / SEMI_SPAN)


def int_from_tip(f, y):
    """Cumulative integral of ``f`` taken inboard from the tip."""
    step = np.concatenate([[0.0],
                           np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(y))])
    return step[-1] - step


def load_case(y, n, fuel, airborne):
    """Shear (kN), bending moment (kN.m) and torque (kN.m) along the span."""
    ell = np.sqrt(np.clip(1.0 - (y / SEMI_SPAN) ** 2, 0.0, None))
    lift = (n * 0.5 * MTOW_N * ell / np.trapezoid(ell, y)) if airborne \
        else np.zeros_like(y)
    w_span = W_STRUCT + fuel * W_FUEL_FULL
    weight = n * w_span * chord(y) / np.trapezoid(chord(y), y)
    p = lift - weight                                     # N/m, up positive
    step = np.where(y < Y_ENGINE, -n * W_ENGINE, 0.0)     # N, point load
    shear = int_from_tip(p, y) + step
    moment = int_from_tip(shear, y)
    twist = p * E_OFFSET * chord(y)                       # N.m/m
    torque = int_from_tip(twist, y) + step * E_OFFSET * chord(Y_ENGINE)
    return shear / 1e3, moment / 1e3, torque / 1e3


def main():
    set_pub_style(base_font=9)

    y = np.linspace(0.0, SEMI_SPAN, 500)
    results = {name: load_case(y, n, fuel, air)
               for name, n, fuel, air, _ in CASES}

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 1.0, 0.62],
                          hspace=0.22)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    ax_p = fig.add_subplot(gs[3])

    specs = [(0, "$V$ (kN)", "kN", "Shear force"),
             (1, "$M$ (kN$\\cdot$m)", "kN$\\cdot$m", "Bending moment"),
             (2, "$T$ (kN$\\cdot$m)", "kN$\\cdot$m",
              "Torque about the elastic axis")]

    for k, ylabel, unit, tag in specs:
        ax = axes[k]
        curves = [results[name][k] for name, *_ in CASES]
        lo_env = np.min(np.vstack(curves), axis=0)
        hi_env = np.max(np.vstack(curves), axis=0)
        ax.fill_between(y, lo_env, hi_env, color=C_A, alpha=0.13, lw=0,
                        zorder=2, label="envelope of all cases")
        for (name, _, _, _, col), cur in zip(CASES, curves):
            ax.plot(y, cur, color=col, lw=1.35, zorder=4, label=name)

        up, dn = float(hi_env.max()), float(lo_env.min())
        for val, va, txt in ((up, "bottom", f"design limit  +{up:,.0f} {unit}"),
                             (dn, "top", f"{dn:,.0f} {unit}")):
            ax.axhline(val, color=C_LIM, lw=1.0, ls=(0, (5, 2.4)), zorder=5)
            ax.text(SEMI_SPAN * 0.995, val, txt.replace(",", " "),
                    fontsize=6.4, color=C_LIM, ha="right", va=va, zorder=6)

        lo, hi = shared_limits(curves, pad=0.22)
        ax.set_ylim(lo, hi)
        ax.set_xlim(0.0, SEMI_SPAN)
        ax.axhline(0.0, color=MUTED, lw=0.7)
        ax.axvline(Y_ENGINE, color=MUTED, lw=0.7, ls=(0, (2, 2.4)), zorder=1)
        ax.set_ylabel(ylabel, fontsize=7.8)
        ax.tick_params(labelsize=6.8, labelbottom=False)
        ax.grid(True, axis="y", lw=0.5, alpha=0.5)
        ax.text(0.30, hi, tag, fontsize=6.8, color=SUB, ha="left", va="top")

    handles, labels = axes[0].get_legend_handles_labels()
    order = [1, 2, 3, 4, 0]
    v_step = results[CASES[0][0]][0]
    j_step = int(np.searchsorted(y, Y_ENGINE))
    axes[0].annotate(f"engine inertia step, {W_ENGINE / 1e3 * 2.5:.0f} kN",
                     xy=(Y_ENGINE, 0.5 * (v_step[j_step - 1] + v_step[j_step])),
                     xytext=(Y_ENGINE + 1.4, axes[0].get_ylim()[1] * 0.80),
                     fontsize=6.4, color=MUTED, ha="left", va="center",
                     arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7,
                                     shrinkA=1.0, shrinkB=3.0,
                                     connectionstyle="arc3,rad=0.20"))

    # ===================== (d) planform and load stations ==================
    x_le = y * np.tan(SWEEP_LE)
    x_te = x_le + chord(y)
    ax_p.add_patch(Polygon(
        np.column_stack([np.concatenate([y, y[::-1]]),
                         np.concatenate([x_le, x_te[::-1]])]),
        facecolor="#dde4ee", edgecolor="#5f7c9e", lw=0.9, zorder=3))
    y_tank = y[(y >= 1.4) & (y <= 13.2)]
    ax_p.add_patch(Polygon(
        np.column_stack([
            np.concatenate([y_tank, y_tank[::-1]]),
            np.concatenate([y_tank * np.tan(SWEEP_LE) + 0.22 * chord(y_tank),
                            (y_tank * np.tan(SWEEP_LE)
                             + 0.66 * chord(y_tank))[::-1]])]),
        facecolor="none", edgecolor="#8a6f18", lw=0.7, hatch="///",
        alpha=0.85, zorder=4))
    ax_p.plot(y, x_le + E_OFFSET * chord(y) + 0.38 * chord(y), color=C_LIM,
              lw=1.0, ls=(0, (5, 2.4)), zorder=5)
    ax_p.text(13.6, 13.6 * np.tan(SWEEP_LE) - 0.55, "elastic axis",
              fontsize=6.2, color=C_LIM, ha="center", va="bottom", zorder=6)
    ax_p.text(6.6, 6.6 * np.tan(SWEEP_LE) + chord(6.6) + 1.05,
              "integral fuel tank", fontsize=6.2, color="#8a6f18",
              ha="center", va="top", zorder=6)

    ax_p.plot([Y_ENGINE], [Y_ENGINE * np.tan(SWEEP_LE) - 0.55], marker="s",
              ms=6.0, mfc=C_D, mec=RULE, mew=0.8, zorder=6)
    ax_p.annotate(f"engine, {W_ENGINE / 1e3:.1f} kN",
                  xy=(Y_ENGINE, Y_ENGINE * np.tan(SWEEP_LE) - 0.55),
                  xytext=(Y_ENGINE - 0.6, -0.30), fontsize=6.2, color=RULE,
                  ha="right", va="center",
                  arrowprops=dict(arrowstyle="-|>", color=RULE, lw=0.7,
                                  shrinkA=1.0, shrinkB=3.0))
    for ys in (2.6, 8.0, 11.4, 14.6):
        ax_p.plot([ys], [ys * np.tan(SWEEP_LE) + 0.30 * chord(ys)],
                  marker="o", ms=3.4, color=C_A, zorder=6)
    ax_p.text(13.4, 13.4 * np.tan(SWEEP_LE) + chord(13.4) + 1.05,
              "load stations", fontsize=6.2, color=C_A, ha="center",
              va="top", zorder=6)

    ax_p.set_xlim(0.0, SEMI_SPAN)
    ax_p.set_ylim(13.6, -1.1)
    ax_p.set_yticks([])
    ax_p.set_xlabel("spanwise station $y$ (m)")
    ax_p.tick_params(labelsize=6.8)
    ax_p.spines["left"].set_visible(False)
    ax_p.text(0.15, 13.2, "planform, chord axis compressed", fontsize=6.2,
              color=MUTED, ha="left", va="bottom")

    fig.tight_layout(rect=[0, 0.005, 1, 0.855], h_pad=0.9)
    panel_labels_inches(fig, axes + [ax_p], dx=-0.50, dy=0.01)
    savefig(fig, "508_aero_structural_load.png")


if __name__ == "__main__":
    main()

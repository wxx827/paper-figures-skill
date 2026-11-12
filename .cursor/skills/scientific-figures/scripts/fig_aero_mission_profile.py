"""Mission profile with the fuel account that goes with it.

The altitude trace, the fuel-flow steps, the remaining-fuel curve and the
donut all come from one piecewise-constant flow model integrated along the
same range grid, so the phase shares in the donut are the areas under the
steps in the panel above it.  The mission deliberately includes a go-around
and a diversion, which is where the reserve that sizes the tank is spent.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from pubstyle import set_pub_style, savefig, darken
from figkit import panel_labels_inches

C_CLIMB = "#9cc3dd"    # sequential blue ramp for the nominal mission
C_CRUISE = "#3b5b92"
C_DESC = "#6f96bd"
C_GO = "#b23a48"       # house red: the contingency phase
C_DIV = "#c9a227"      # house gold: diversion
C_FUEL = "#2e7d5b"     # house green: remaining fuel
INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"

FUEL_BLOCK = 15000.0   # fuel on board at brake release (kg)
FUEL_RESERVE = 2800.0  # final reserve that may not be planned into the trip

# range (km), altitude (km) waypoints of the flown trajectory
WAYPOINTS = np.array([
    [0, 0.00], [8, 0.90], [15, 1.60], [60, 5.20], [130, 8.60], [250, 11.30],
    [1500, 11.80], [2850, 12.20], [2950, 8.00], [3030, 3.00], [3070, 0.15],
    [3080, 0.60], [3110, 1.60], [3160, 4.00], [3200, 6.10], [3380, 6.10],
    [3450, 3.00], [3520, 0.00]], dtype=float)

# name, range start, range end, fuel flow (kg/km), colour, label rotation
PHASES = [("take-off and climb", 0.0, 250.0, 6.90, C_CLIMB, 90),
          ("cruise", 250.0, 2850.0, 2.75, C_CRUISE, 0),
          ("descent", 2850.0, 3070.0, 1.15, C_DESC, 90),
          ("go-around", 3070.0, 3160.0, 9.00, C_GO, 90),
          ("diversion and landing", 3160.0, 3520.0, 3.40, C_DIV, 90)]


def main():
    set_pub_style(base_font=9)

    rng = np.linspace(0.0, WAYPOINTS[-1, 0], 1400)
    alt = PchipInterpolator(WAYPOINTS[:, 0], WAYPOINTS[:, 1])(rng)

    flow = np.zeros_like(rng)
    for _, r0, r1, f, _, _ in PHASES:
        flow[(rng >= r0) & (rng <= r1)] = f
    burned = np.concatenate([[0.0], np.cumsum(
        0.5 * (flow[1:] + flow[:-1]) * np.diff(rng))])
    remaining = FUEL_BLOCK - burned
    share = np.array([f * (r1 - r0) for _, r0, r1, f, _, _ in PHASES])

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.62], hspace=0.16, wspace=0.16)
    ax = fig.add_subplot(gs[0, 0])
    ax_f = fig.add_subplot(gs[1, 0], sharex=ax)
    ax_d = fig.add_subplot(gs[:, 1])

    # ===================== (a) altitude against range ======================
    for name, r0, r1, _, col, rot in PHASES:
        ax.axvspan(r0, r1, color=col, alpha=0.20, lw=0, zorder=1)
        ax.axvline(r1, color="white", lw=0.9, zorder=2)
        xm = 0.5 * (r0 + r1)
        if rot:
            # rotation_mode="anchor" with ha="right" hangs the label downward
            # from the top of the panel, which is the only way a 90 km wide
            # band can carry a name without spilling over the title.
            ax.text(xm, 14.35, name, fontsize=6.6, color=SUB, rotation=90,
                    ha="right", va="center", rotation_mode="anchor", zorder=6)
        else:
            ax.text(xm, 14.30, name, fontsize=6.6, color=SUB, ha="center",
                    va="top", zorder=6)
    ax.plot(rng, alt, color=RULE, lw=1.8, zorder=5)
    ax.fill_between(rng, 0.0, alt, color=RULE, alpha=0.05, lw=0, zorder=1)

    i_toc = int(np.argmin(np.abs(rng - 250.0)))
    i_tod = int(np.argmin(np.abs(rng - 2850.0)))
    for i, lab, dy in ((i_toc, "top of climb", 0.9), (i_tod, "top of descent",
                                                      0.9)):
        ax.plot(rng[i], alt[i], marker="o", ms=5.0, mfc="white", mec=RULE,
                mew=1.3, zorder=7)
        ax.annotate(f"{lab}\n{alt[i]:.1f} km", xy=(rng[i], alt[i]),
                    xytext=(rng[i] + (240.0 if i == i_toc else -240.0),
                            alt[i] - 2.6), fontsize=6.6, color=RULE,
                    ha="left" if i == i_toc else "right", va="center",
                    arrowprops=dict(arrowstyle="-|>", color=RULE, lw=0.7,
                                    shrinkA=1.0, shrinkB=3.0))
    ax.annotate(f"go-around to {alt[np.argmin(np.abs(rng - 3110.0))]:.1f} km",
                xy=(3110.0, alt[np.argmin(np.abs(rng - 3110.0))]),
                xytext=(2760.0, 5.2), fontsize=6.6, color=C_GO, ha="right",
                va="center",
                arrowprops=dict(arrowstyle="-|>", color=C_GO, lw=0.8,
                                shrinkA=1.0, shrinkB=3.0,
                                connectionstyle="arc3,rad=-0.25"))

    ax.set_ylim(0.0, 14.6)
    ax.set_xlim(0.0, WAYPOINTS[-1, 0])
    ax.set_ylabel("pressure altitude (km)")
    ax.tick_params(labelbottom=False)
    ax.grid(True, axis="y", lw=0.5, alpha=0.5)
    ax.set_title("Altitude profile, including a go-around and a diversion",
                 fontsize=9.0, fontweight="bold", pad=4)

    # ===================== (b) fuel flow and fuel remaining ================
    ax_f.plot(rng, flow, color=C_CRUISE, lw=1.5, zorder=5)
    ax_f.fill_between(rng, 0.0, flow, color=C_CRUISE, alpha=0.14, lw=0,
                      zorder=2)
    for _, r0, r1, f, col, _ in PHASES:
        ax_f.axvspan(r0, r1, color=col, alpha=0.10, lw=0, zorder=1)
    ax_f.set_ylim(0.0, 11.5)
    ax_f.set_ylabel("fuel flow (kg/km)", color=darken(C_CRUISE, 0.28))
    ax_f.tick_params(axis="y", colors=darken(C_CRUISE, 0.28))
    ax_f.set_xlabel("range flown (km)")
    ax_f.grid(True, axis="y", lw=0.5, alpha=0.4)

    ax_r = ax_f.twinx()
    ax_r.spines["right"].set_visible(True)
    ax_r.plot(rng, remaining, color=C_FUEL, lw=1.7, zorder=6)
    ax_r.axhline(FUEL_RESERVE, color=C_GO, lw=1.0, ls=(0, (5, 2.4)),
                 zorder=5)
    ax_r.text(WAYPOINTS[-1, 0] * 0.985, FUEL_RESERVE + 260.0,
              f"final reserve {FUEL_RESERVE:,.0f} kg".replace(",", " "),
              fontsize=6.5, color=C_GO, ha="right", va="bottom")
    ax_r.plot(rng[-1], remaining[-1], marker="o", ms=5.0, mfc="white",
              mec=C_FUEL, mew=1.4, zorder=7)
    ax_r.annotate(f"landing fuel {remaining[-1]:,.0f} kg".replace(",", " "),
                  xy=(rng[-1], remaining[-1]), xytext=(2980.0, 8600.0),
                  fontsize=6.6, color=C_FUEL, ha="right", va="center",
                  arrowprops=dict(arrowstyle="-|>", color=C_FUEL, lw=0.8,
                                  shrinkA=1.0, shrinkB=3.0,
                                  connectionstyle="arc3,rad=-0.22"))
    ax_r.set_ylim(0.0, 16200.0)
    ax_r.set_ylabel("fuel remaining (kg)", color=darken(C_FUEL, 0.28))
    ax_r.tick_params(axis="y", colors=darken(C_FUEL, 0.28), labelsize=8)

    # ===================== (c) fuel share by phase (subordinate) ===========
    frac = share / share.sum()
    cols = [p[4] for p in PHASES]
    wedges, _ = ax_d.pie(frac, colors=cols, startangle=90, counterclock=False,
                         radius=1.0,
                         wedgeprops=dict(width=0.40, edgecolor="white",
                                         linewidth=1.0))
    cum = np.concatenate([[0.0], np.cumsum(frac)])
    for i, w in enumerate(wedges):
        if frac[i] < 0.10:
            continue
        a = np.deg2rad(90.0 - 360.0 * 0.5 * (cum[i] + cum[i + 1]))
        ax_d.text(0.80 * np.cos(a), 0.80 * np.sin(a),
                  f"{frac[i] * 100:.1f} %", fontsize=6.6, color="white",
                  fontweight="bold", ha="center", va="center")
    ax_d.text(0.0, 0.10, f"{share.sum():,.0f} kg".replace(",", " "),
              fontsize=10.5, fontweight="bold", color=INK, ha="center",
              va="center")
    ax_d.text(0.0, -0.16, "trip fuel burned", fontsize=6.8, color=SUB,
              ha="center", va="center")
    ax_d.legend(wedges,
                [f"{PHASES[i][0]}   {share[i]:,.0f} kg "
                 f"({frac[i] * 100:.1f} %)".replace(",", " ")
                 for i in range(len(PHASES))],
                loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1,
                fontsize=6.6, frameon=False, handlelength=1.1,
                labelspacing=0.5, borderaxespad=0.0)
    ax_d.set_xlim(-1.28, 1.28)
    ax_d.set_ylim(-1.28, 1.28)
    ax_d.set_aspect("equal")
    ax_d.set_title("Fuel share by phase", fontsize=8.5, fontweight="bold",
                   pad=3)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95], h_pad=1.2, w_pad=2.0)
    # The donut keeps a nominal box far wider than the circle it draws, so its
    # letter has to be placed against the box the aspect actually produced.
    ax_d.apply_aspect()
    panel_labels_inches(fig, [ax, ax_f, ax_d], dx=-0.44, dy=0.06)
    savefig(fig, "509_aero_mission_profile.png")


if __name__ == "__main__":
    main()

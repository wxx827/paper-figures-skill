"""Pressing intensity by pitch zone: recoveries, PPDA and what followed.

Two synthetic event streams drive the figure.  The opponent's completed passes
are scattered with a build-up bias toward their own goal, and the defending
team's ball recoveries are scattered with a pressing bias toward the same end;
both are then binned into the same 5 x 3 grid.  Cell colour is the recovery
count, the number printed under it is PPDA - opponent passes divided by
defensive actions in that cell, so a *low* number marks an intense press - and
the arrows are the recoveries that turned into a shot-directed attack inside
5 s.  The period panel re-bins the same two streams by time and the height
panel re-bins the recoveries along the pitch, so no panel can drift from the
grid it explains.

Pitch geometry is the FIFA standard 105 x 68 m surface; the team under
analysis defends the left-hand goal and presses toward the right.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle

from pubstyle import annotate_box, get_cmap, savefig, set_pub_style
from figkit import (axes_inches, halo_text, panel_labels_inches,
                    safe_colorbar, shared_norm)

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HOME = "#3b5b92"          # house blue - recovery density
ACCENT = "#b23a48"        # house red - transition arrows
GOLD = "#c9a227"          # house gold - press trigger line
GRASS = "#f4f4f2"
LINE = "#b0b0b0"

PITCH_L, PITCH_W = 105.0, 68.0
NX, NY = 5, 3             # zones along the pitch, channels across it
MATCH_MIN = 90.0
N_OPP_PASS = 430
N_RECOVERY = 74
TRIGGER_X = 68.0          # high-press trigger line
TEAM, OPP = "Team A", "Team B"

BLOCK = ["own box", "defensive", "middle", "advanced", "high press"]
CHANNEL = ["right", "centre", "left"]   # index 0 is the low-y touchline


def draw_pitch(ax, color=LINE, lw=0.9, zorder=6):
    """Standard 105 x 68 m markings, drawn over the zone grid."""
    ax.add_patch(Rectangle((0.0, 0.0), PITCH_L, PITCH_W, facecolor=GRASS,
                           edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0.0, 0.0), PITCH_L, PITCH_W, fill=False,
                           edgecolor=color, lw=lw, zorder=zorder))
    ax.plot([PITCH_L / 2.0] * 2, [0.0, PITCH_W], color=color, lw=lw,
            zorder=zorder, solid_capstyle="butt")
    ax.add_patch(Circle((PITCH_L / 2.0, PITCH_W / 2.0), 9.15, fill=False,
                        edgecolor=color, lw=lw, zorder=zorder))
    arc_half = float(np.degrees(np.arccos((16.5 - 11.0) / 9.15)))
    for side in (0, 1):
        x_box = 0.0 if side == 0 else PITCH_L - 16.5
        x_six = 0.0 if side == 0 else PITCH_L - 5.5
        x_spot = 11.0 if side == 0 else PITCH_L - 11.0
        ax.add_patch(Rectangle((x_box, PITCH_W / 2.0 - 20.16), 16.5, 40.32,
                               fill=False, edgecolor=color, lw=lw,
                               zorder=zorder))
        ax.add_patch(Rectangle((x_six, PITCH_W / 2.0 - 9.16), 5.5, 18.32,
                               fill=False, edgecolor=color, lw=lw,
                               zorder=zorder))
        ax.add_patch(Circle((x_spot, PITCH_W / 2.0), 0.35, facecolor=color,
                            edgecolor="none", zorder=zorder))
        t1, t2 = ((-arc_half, arc_half) if side == 0
                  else (180.0 - arc_half, 180.0 + arc_half))
        ax.add_patch(Arc((x_spot, PITCH_W / 2.0), 2 * 9.15, 2 * 9.15,
                         theta1=t1, theta2=t2, edgecolor=color, lw=lw,
                         zorder=zorder))
        x_goal = -1.9 if side == 0 else PITCH_L
        ax.add_patch(Rectangle((x_goal, PITCH_W / 2.0 - 3.66), 1.9, 7.32,
                               fill=False, edgecolor=color, lw=lw,
                               zorder=zorder))


def main():
    set_pub_style()
    rng = np.random.default_rng(867)

    # ---- opponent passes: build-up biased toward their own goal (x = 105) ---
    px = np.clip(PITCH_L - rng.gamma(2.6, 15.5, N_OPP_PASS), 2.0,
                 PITCH_L - 2.0)
    py = np.clip(rng.normal(PITCH_W / 2.0, 17.0, N_OPP_PASS), 1.5,
                 PITCH_W - 1.5)
    p_min = rng.uniform(0.0, MATCH_MIN, N_OPP_PASS)

    # ---- recoveries: pressing bias toward the same end, fading with fatigue
    rx = np.clip(PITCH_L - rng.gamma(3.1, 12.0, N_RECOVERY), 2.0,
                 PITCH_L - 2.0)
    ry = np.clip(rng.normal(PITCH_W / 2.0, 16.0, N_RECOVERY), 1.5,
                 PITCH_W - 1.5)
    r_min = np.sort(rng.beta(0.86, 1.06, N_RECOVERY) * MATCH_MIN)
    # late recoveries drop deeper: the press cannot hold its height for 90 min
    rx = np.clip(rx - 0.20 * r_min, 2.0, PITCH_L - 2.0)

    ex = np.linspace(0.0, PITCH_L, NX + 1)
    ey = np.linspace(0.0, PITCH_W, NY + 1)
    rec_grid, _, _ = np.histogram2d(rx, ry, bins=[ex, ey])
    pass_grid, _, _ = np.histogram2d(px, py, bins=[ex, ey])
    ppda = pass_grid / np.maximum(rec_grid, 1.0)
    ppda = np.where(rec_grid > 0, ppda, np.nan)
    team_ppda = pass_grid[3:].sum() / max(rec_grid[3:].sum(), 1.0)

    # ---- recoveries that became an attack inside 5 s -----------------------
    p_trans = np.clip(0.10 + 0.42 * (rx / PITCH_L) ** 2.2, 0.0, 0.85)
    trans = rng.random(N_RECOVERY) < p_trans
    tx = np.clip(rx[trans] + rng.uniform(9.0, 24.0, trans.sum()), 0.0,
                 PITCH_L - 1.0)
    ty = ry[trans] + 0.45 * (PITCH_W / 2.0 - ry[trans]) \
        + rng.normal(0.0, 4.0, trans.sum())

    # ---- period series -------------------------------------------------------
    edges = np.arange(0.0, MATCH_MIN + 15.0, 15.0)
    ppda_t, rec_t, mid_t = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        npass = ((p_min >= a) & (p_min < b)).sum()
        nrec = ((r_min >= a) & (r_min < b)).sum()
        ppda_t.append(npass / max(nrec, 1))
        rec_t.append(nrec)
        mid_t.append(0.5 * (a + b))
    ppda_t = np.array(ppda_t, dtype=float)
    mid_t = np.array(mid_t)
    slope, icept = np.polyfit(mid_t, ppda_t, 1)

    # ---- recovery height distribution ---------------------------------------
    hb = np.linspace(0.0, PITCH_L, 15)
    hist, _ = np.histogram(rx, bins=hb)
    hc = 0.5 * (hb[:-1] + hb[1:])
    mean_h = rx.mean()

    # ---- layout ---------------------------------------------------------------
    FW, FH = 11.0, 6.3
    fig = plt.figure(figsize=(FW, FH))
    ax_p = axes_inches(fig, (0.38, 1.05, 6.00, 4.278))
    ax_t = axes_inches(fig, (7.25, 3.15, 3.35, 2.15))
    ax_h = axes_inches(fig, (7.25, 0.95, 3.35, 1.45))

    cmap = get_cmap("cb_sequential_blue")
    norm = shared_norm([rec_grid])
    im = ax_p.imshow(rec_grid.T, origin="lower", cmap=cmap, norm=norm,
                     extent=[0, PITCH_L, 0, PITCH_W], alpha=0.88,
                     interpolation="nearest", zorder=1)
    for x in ex:
        ax_p.plot([x, x], [0.0, PITCH_W], color="white", lw=1.1, zorder=4)
    for y in ey:
        ax_p.plot([0.0, PITCH_L], [y, y], color="white", lw=1.1, zorder=4)
    draw_pitch(ax_p, zorder=6)

    for i in range(NX):
        for j in range(NY):
            cx = 0.5 * (ex[i] + ex[i + 1])
            cy = 0.5 * (ey[j] + ey[j + 1])
            dark = norm(rec_grid[i, j]) > 0.52
            col, halo = ("white", "#2b3f63") if dark else (INK, "white")
            halo_text(ax_p, cx, cy + 1.9, f"{rec_grid[i, j]:.0f} recoveries",
                      ha="center", va="center", fontsize=6.4, color=col,
                      halo=halo, halo_lw=1.7, fontweight="bold", zorder=8)
            lab = ("PPDA n/a" if not np.isfinite(ppda[i, j])
                   else f"PPDA {ppda[i, j]:.1f}")
            halo_text(ax_p, cx, cy - 2.2, lab, ha="center", va="center",
                      fontsize=6.2, color=col, halo=halo, halo_lw=1.7,
                      zorder=8)

    for x0, y0, x1, y1 in zip(rx[trans], ry[trans], tx, ty):
        ax_p.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6.0,
            connectionstyle="arc3,rad=0.16", shrinkA=0, shrinkB=0,
            color=ACCENT, lw=0.9, alpha=0.85, zorder=9))
    ax_p.plot(rx[trans], ry[trans], marker="o", ls="none", ms=2.6,
              mfc=ACCENT, mec="none", zorder=10)

    ax_p.plot([TRIGGER_X] * 2, [0.0, PITCH_W], color=GOLD, lw=1.4,
              ls=(0, (6, 3)), zorder=11)
    halo_text(ax_p, TRIGGER_X, PITCH_W + 1.4,
              f"press trigger line  x = {TRIGGER_X:.0f} m", ha="center",
              va="bottom", fontsize=6.4, color=GOLD, halo="white",
              halo_lw=2.0, zorder=12)

    for i in range(NX):
        halo_text(ax_p, 0.5 * (ex[i] + ex[i + 1]), -2.4, BLOCK[i],
                  ha="center", va="top", fontsize=6.2, color=SUB,
                  halo="white", halo_lw=1.8, zorder=9)
    for j in range(NY):
        ax_p.text(-2.2, 0.5 * (ey[j] + ey[j + 1]), CHANNEL[j], ha="right",
                  va="center", fontsize=6.2, color=SUB, rotation=90,
                  zorder=9)

    ax_p.annotate("", xy=(96.0, -6.6), xytext=(76.0, -6.6),
                  arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))
    ax_p.text(74.0, -6.6, "pressing direction", ha="right", va="center",
              fontsize=6.4, color=MUTED)

    ax_p.set_xlim(-7.0, 111.0)
    ax_p.set_ylim(-9.5, 76.0)
    ax_p.set_aspect("equal")
    ax_p.set_xticks([])
    ax_p.set_yticks([])
    for s in ax_p.spines.values():
        s.set_visible(False)
    ax_p.set_title(f"{TEAM} pressing grid against {OPP}", fontsize=9.0,
                   fontweight="bold", pad=5)

    annotate_box(
        ax_p,
        f"{N_RECOVERY} recoveries | {N_OPP_PASS} opponent passes\n"
        f"PPDA past the trigger line  {team_ppda:.1f}\n"
        f"{int(trans.sum())} recoveries became an attack inside 5 s "
        f"({100.0 * trans.mean():.0f}%)",
        loc="upper left", fontsize=6.4, margin=0.014)

    safe_colorbar(fig, im, ax_p, label="ball recoveries in the zone (count)",
                  side="bottom", width=0.10, pad=0.16, shrink=0.52,
                  labelsize=6.5)

    # ---- PPDA by period -------------------------------------------------------
    ax_t.plot(mid_t, ppda_t, marker="o", ms=4.6, color=HOME, lw=1.4,
              mfc="white", mec=HOME, mew=1.1, zorder=4,
              label="PPDA per 15 min block")
    ax_t.plot(mid_t, slope * mid_t + icept, color=ACCENT, lw=1.1, ls="--",
              zorder=3, label=f"trend {slope * 30.0:+.1f} PPDA per 30 min")
    ax_t.set_xlim(0.0, MATCH_MIN)
    y_top = ppda_t.max() * 1.35
    ax_t.set_ylim(0.0, y_top)
    ax_t.axvline(45.0, color="#808080", lw=0.9, ls=":", zorder=2)
    ax_t.text(44.4, y_top * 0.985, "half time", ha="right", va="top",
              fontsize=6.2, color=MUTED)
    for x, v, k in zip(mid_t, ppda_t, rec_t):
        ax_t.annotate(f"{v:.1f}", xy=(x, v), xytext=(0, 6),
                      textcoords="offset points", ha="center", va="bottom",
                      fontsize=6.2, color=SUB)
        ax_t.annotate(f"{k}", xy=(x, 0.0), xytext=(0, 3),
                      textcoords="offset points", ha="center", va="bottom",
                      fontsize=6.0, color=MUTED)
    ax_t.set_xticks(edges)
    ax_t.set_xlabel("match time (min)   |   number under the axis = recoveries",
                    fontsize=7.2)
    ax_t.set_ylabel("PPDA (opponent passes\nper defensive action)",
                    fontsize=8.0)
    ax_t.tick_params(labelsize=7.0)
    ax_t.grid(axis="y", lw=0.5, color="#ececec", zorder=0)
    ax_t.spines["top"].set_visible(False)
    ax_t.spines["right"].set_visible(False)
    ax_t.legend(loc="upper left", bbox_to_anchor=(0.0, 0.90), frameon=False,
                fontsize=6.8, handlelength=1.9, borderaxespad=0.2)
    ax_t.set_title("The press loosens as the match wears on", fontsize=8.5,
                   fontweight="bold")

    # ---- recovery height ------------------------------------------------------
    ax_h.bar(hc, hist, width=(hb[1] - hb[0]) * 0.88, color="#9cc3dd",
             edgecolor="white", lw=0.5, zorder=3)
    ax_h.bar(hc[hc > TRIGGER_X], hist[hc > TRIGGER_X],
             width=(hb[1] - hb[0]) * 0.88, color=HOME, edgecolor="white",
             lw=0.5, zorder=4, label="past the trigger line")
    ax_h.axvline(mean_h, color=ACCENT, lw=1.1, ls="--", zorder=5)
    ax_h.axvline(TRIGGER_X, color=GOLD, lw=1.1, ls=(0, (6, 3)), zorder=5)
    ax_h.set_xlim(0.0, PITCH_L)
    ax_h.set_ylim(0.0, hist.max() * 1.28)
    ax_h.text(mean_h - 1.8, hist.max() * 1.26,
              f"mean recovery height {mean_h:.0f} m", ha="right", va="top",
              fontsize=6.3, color=ACCENT)
    ax_h.set_xticks([0, 17.5, 35, 52.5, 70, 87.5, 105])
    ax_h.set_xlabel("distance from own goal line (m)", fontsize=8.0)
    ax_h.set_ylabel("recoveries", fontsize=8.0)
    ax_h.tick_params(labelsize=7.0)
    ax_h.grid(axis="y", lw=0.5, color="#ececec", zorder=0)
    ax_h.spines["top"].set_visible(False)
    ax_h.spines["right"].set_visible(False)
    ax_h.legend(loc="upper right", frameon=False, fontsize=6.8,
                handlelength=1.4, borderaxespad=0.2)
    ax_h.set_title("How high the ball was won", fontsize=8.5,
                   fontweight="bold")

    fig.legend(handles=[
        Line2D([0], [0], color=GOLD, lw=1.4, ls=(0, (6, 3)),
               label=f"press trigger line (x = {TRIGGER_X:.0f} m)"),
        Line2D([0], [0], color=ACCENT, lw=1.0,
               label="recovery converted to an attack inside 5 s"),
        Line2D([0], [0], color=LINE, lw=0.9,
               label="pitch markings (105 x 68 m), 5 x 3 zone grid"),
        Line2D([0], [0], color="none", label="low PPDA = more intense press")],
        loc="lower left", bbox_to_anchor=(0.028, 0.050), ncol=4,
        frameon=False, fontsize=7.0, handlelength=1.8, columnspacing=1.7)

    panel_labels_inches(fig, [ax_p, ax_t, ax_h], dx=-0.32, dy=0.12)

    hot = int(np.argmin(np.where(rec_grid.sum(axis=1) > 0,
                                 pass_grid.sum(axis=1)
                                 / np.maximum(rec_grid.sum(axis=1), 1.0),
                                 1e9)))

    print(f"recoveries {N_RECOVERY}  opp passes {N_OPP_PASS}  "
          f"PPDA high {team_ppda:.2f}  transitions {int(trans.sum())}")
    print("recoveries per zone (x by y):", rec_grid.astype(int).tolist())
    print("PPDA per period:", ppda_t.round(2).tolist())
    print(f"mean recovery height {mean_h:.1f} m  trend "
          f"{slope * 30.0:+.2f} PPDA per 30 min")
    savefig(fig, "867_tac_pressing_zones.png")


if __name__ == "__main__":
    main()

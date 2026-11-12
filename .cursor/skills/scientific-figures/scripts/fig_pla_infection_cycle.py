"""Infection cycle: what the pathogen needs at each step, and how often.

The ring is the disease cycle drawn as the six things that actually have to
happen, each with the condition that gates it and the time it takes.  It is
not a decoration: the "cycles per season" printed in the middle is the end
point of the accrual curve in panel (b), which advances every day by one over
the current cycle length, so favourable weather shortens the loop and the ring
count rises.  The two control windows are picked out of that same daily rate
as the ten-day stretches where the most cycles are gained.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon

from pubstyle import set_pub_style, savefig, darken, lighten
from figkit import axes_inches, inch_axes, panel_label_inches, arrow

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
SOFT = "#f5f5f5"
GREEN = "#2e7d5b"
LEAF = "#6bba8e"
PALE = "#bfe3cd"
GOLD = "#c9a227"
RED = "#b23a48"
DEEP_RED = "#8c2f2f"
EARTH = "#c0603f"
GREY = "#808080"

W, H = 10.4, 6.4
PATHOGEN = "Magnaporthe oryzae"
DISEASE = "rice blast"
SEASON = 140                    # days after crop emergence
WINDOW = 10                     # control window width, days
DEG = r"$^{\circ}$C"

# (step, gating condition, duration text, silhouette key)
STEPS = [
    ("Overwintering inoculum", "infected residue and seed", "5 - 6 months",
     "residue"),
    ("Primary infection", "6 - 10 h leaf wetness, 20 - 28 %s" % DEG,
     "12 - 24 h", "entry"),
    ("Latent period", "no visible symptom yet", "5 - 7 d at 25 %s" % DEG,
     "latent"),
    ("Symptom expression", "spindle lesions on leaf", "2 - 3 d", "lesion"),
    ("Sporulation", "> 90 % RH through the night", "1 - 2 d", "spores"),
    ("Secondary dispersal", "wind and rain splash", "hours", "dispersal"),
]


# ------------------------------------------------------------- primitives
def leaf_frame(rot):
    th = np.radians(rot)
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def draw_leaf(ax, x, y, s, fc, rot=26.0, zorder=6, alpha=1.0):
    u = np.linspace(0.0, 1.0, 70)
    hw = 0.30 * s * np.sin(np.pi * u) ** 0.75
    xs = (u - 0.5) * s
    pts = np.vstack([np.column_stack([xs, hw]),
                     np.column_stack([xs[::-1], -hw[::-1]])])
    rot_m = leaf_frame(rot)
    org = np.array([x, y])
    ax.add_patch(Polygon(pts @ rot_m.T + org, closed=True, facecolor=fc,
                         edgecolor="white", lw=0.8, zorder=zorder,
                         alpha=alpha))
    mid = np.column_stack([xs, np.zeros_like(xs)]) @ rot_m.T + org
    ax.plot(mid[:, 0], mid[:, 1], color="white", lw=0.6, zorder=zorder + 1,
            alpha=alpha)
    return rot_m, org


def on_leaf(rot_m, org, a, b):
    return tuple(np.array([a, b]) @ rot_m.T + org)


def sym_residue(ax, x, y, s):
    for i, (dx, dy) in enumerate([(-0.16, -0.12), (0.10, -0.02),
                                  (-0.04, 0.12)]):
        ax.add_patch(FancyBboxPatch(
            (x + dx * s - 0.20 * s, y + dy * s - 0.045 * s), 0.40 * s,
            0.09 * s, boxstyle="round,pad=0,rounding_size=0.03",
            facecolor=lighten(EARTH, 0.12 * i), edgecolor="white", lw=0.6,
            mutation_aspect=1.0, zorder=6 + i))
    ax.add_patch(Circle((x + 0.20 * s, y + 0.20 * s), 0.075 * s,
                        facecolor="#3a3a3a", edgecolor="white", lw=0.5,
                        zorder=10))


def sym_entry(ax, x, y, s):
    rot_m, org = draw_leaf(ax, x, y + 0.04 * s, 0.86 * s, LEAF, rot=20.0)
    tip = on_leaf(rot_m, org, 0.02 * s, 0.06 * s)
    src = (x - 0.30 * s, y + 0.34 * s)
    ax.add_patch(Circle(src, 0.055 * s, facecolor=GOLD, edgecolor="white",
                        lw=0.5, zorder=9))
    arrow(ax, src, tip, color=GOLD, lw=0.9, ms=6.5, zorder=9, gap_b=0.03)


def sym_latent(ax, x, y, s):
    rot_m, org = draw_leaf(ax, x, y, 0.86 * s, PALE, rot=20.0)
    for a in (-0.16, 0.04, 0.20):
        c = on_leaf(rot_m, org, a * s, 0.0)
        ax.add_patch(Ellipse(c, 0.15 * s, 0.09 * s, angle=20.0,
                             facecolor="none", edgecolor=GREEN, lw=0.7,
                             ls=(0, (1.4, 1.2)), zorder=9))


def sym_lesion(ax, x, y, s):
    rot_m, org = draw_leaf(ax, x, y, 0.86 * s, LEAF, rot=20.0)
    for a, b, w in [(-0.18, 0.03, 0.17), (0.02, -0.04, 0.21),
                    (0.22, 0.05, 0.14)]:
        c = on_leaf(rot_m, org, a * s, b * s)
        ax.add_patch(Ellipse(c, w * s, 0.085 * s, angle=20.0,
                             facecolor=DEEP_RED, edgecolor="white", lw=0.5,
                             zorder=9))


def sym_spores(ax, x, y, s, rng):
    rot_m, org = draw_leaf(ax, x, y - 0.06 * s, 0.86 * s, LEAF, rot=20.0)
    c = on_leaf(rot_m, org, 0.0, 0.0)
    ax.add_patch(Ellipse(c, 0.24 * s, 0.10 * s, angle=20.0,
                         facecolor=DEEP_RED, edgecolor="white", lw=0.5,
                         zorder=9))
    for ang in np.linspace(35.0, 145.0, 6):
        r = 0.22 * s + 0.05 * s * rng.random()
        ax.add_patch(Circle((c[0] + r * np.cos(np.radians(ang)),
                             c[1] + r * np.sin(np.radians(ang))),
                            0.035 * s, facecolor=GOLD, edgecolor="none",
                            zorder=10))


def sym_dispersal(ax, x, y, s):
    rot_m, org = draw_leaf(ax, x + 0.20 * s, y - 0.10 * s, 0.62 * s, LEAF,
                           rot=18.0)
    for dy, rad in [(0.30, 0.10), (0.16, 0.0), (0.02, -0.10)]:
        p0 = (x - 0.36 * s, y + dy * s)
        ax.add_patch(Circle(p0, 0.05 * s, facecolor=GOLD, edgecolor="white",
                            lw=0.5, zorder=9))
        arrow(ax, p0, on_leaf(rot_m, org, -0.14 * s, 0.02 * s), color=GOLD,
              lw=0.8, ms=6.0, rad=rad, zorder=9, gap_a=0.06, gap_b=0.04)


SYMBOLS = {"residue": sym_residue, "entry": sym_entry, "latent": sym_latent,
           "lesion": sym_lesion, "spores": sym_spores,
           "dispersal": sym_dispersal}


def arc_arrow(ax, cx, cy, r, a0, a1, color=RULE, lw=1.05, ls="-", ms=9.0,
              zorder=3):
    a = np.radians(np.linspace(a0, a1, 80))
    x, y = cx + r * np.cos(a), cy + r * np.sin(a)
    ax.plot(x[:-1], y[:-1], color=color, lw=lw, ls=ls, zorder=zorder,
            solid_capstyle="round")
    arrow(ax, (x[-2], y[-2]), (x[-1], y[-1]), color=color, lw=lw, ms=ms,
          zorder=zorder)


# ------------------------------------------------------------------- cycle
def season_model(rng):
    """Daily cycle-length model; returns rate, cumulative cycles, weather."""
    d = np.arange(0.0, SEASON + 1.0)
    temp = 21.0 + 6.5 * np.sin(2.0 * np.pi * (d - 22.0) / 190.0)
    wet = np.clip(0.30 + 0.52 * np.sin(2.0 * np.pi * (d - 26.0) / 62.0) +
                  rng.normal(0.0, 0.09, d.size), 0.0, 1.0)
    t_fav = np.exp(-0.5 * ((temp - 25.5) / 4.6) ** 2)
    fav = np.clip(t_fav * wet, 0.02, 1.0)
    cycle_len = 6.0 + 16.0 * (1.0 - fav)          # days per infection cycle
    rate = 1.0 / cycle_len
    return d, fav, rate, np.cumsum(rate)


def control_windows(rate, n=2):
    roll = np.convolve(rate, np.ones(WINDOW), mode="valid")
    picks, used = [], np.zeros(roll.size, dtype=bool)
    for _ in range(n):
        cand = np.where(used, -1.0, roll)
        i = int(np.argmax(cand))
        picks.append(i)
        used[max(i - WINDOW - 4, 0):i + WINDOW + 4] = True
    return sorted(picks)


# ------------------------------------------------------------------- panels
def ring_panel(ax, w, h, rng, n_cycles):
    cx, cy = w / 2.0, h / 2.0
    radius, node = 1.50, 0.42
    n = len(STEPS)
    step = 360.0 / n
    gap = np.degrees((node + 0.075) / radius)

    ax.add_patch(Circle((cx, cy), 0.84, facecolor=SOFT, edgecolor=HAIR,
                        lw=0.7, zorder=1))

    for i in range(n):
        last = i == n - 1
        arc_arrow(ax, cx, cy, radius, 90.0 - i * step - gap,
                  90.0 - (i + 1) * step + gap,
                  color=GREY if last else RULE,
                  ls=(0, (3.4, 2.0)) if last else "-")
    # Re-infection shortcut: the short way over the top, skipping survival.
    arc_arrow(ax, cx, cy, 1.04, 90.0 + step - gap * 0.7,
              90.0 - step + gap * 0.7, color=RED, lw=1.1, ls=(0, (2.6, 1.8)))

    for i, (name, cond, dur, key) in enumerate(STEPS):
        a = np.radians(90.0 - i * step)
        x, y = cx + radius * np.cos(a), cy + radius * np.sin(a)
        ax.add_patch(Circle((x, y), node, facecolor="white", edgecolor=HAIR,
                            lw=0.8, zorder=4))
        if key == "spores":
            SYMBOLS[key](ax, x, y, 0.74, rng)
        else:
            SYMBOLS[key](ax, x, y, 0.74)

        lx = cx + (radius + node + 0.16) * np.cos(a)
        ly = cy + (radius + node + 0.16) * np.sin(a)
        ca, sa = np.cos(a), np.sin(a)
        ha = "left" if ca > 0.30 else ("right" if ca < -0.30 else "center")
        dy = 0.13 if sa > 0.30 else (-0.13 if sa < -0.30 else 0.0)
        ax.text(lx, ly + dy + 0.155, name, ha=ha, va="center", fontsize=7.2,
                color=INK, fontweight="bold", zorder=9)
        ax.text(lx, ly + dy, cond, ha=ha, va="center", fontsize=6.4,
                color=SUB, zorder=9)
        ax.text(lx, ly + dy - 0.155, dur, ha=ha, va="center", fontsize=6.4,
                color=MUTED, zorder=9)

    ax.text(cx, cy + 0.38, PATHOGEN, ha="center", va="center", fontsize=8.2,
            fontstyle="italic", color=INK, zorder=9)
    ax.text(cx, cy + 0.19, DISEASE, ha="center", va="center", fontsize=7.4,
            color=RULE, zorder=9)
    ax.text(cx, cy - 0.06, "%.1f cycles" % n_cycles, ha="center",
            va="center", fontsize=8.6, fontweight="bold", color=DEEP_RED,
            zorder=9)
    ax.text(cx, cy - 0.26, "per %d-day season" % SEASON, ha="center",
            va="center", fontsize=6.8, color=SUB, zorder=9)
    ax.text(cx, cy - 0.50, "grey dash = survival", ha="center", va="center",
            fontsize=6.2, color=MUTED, zorder=9)
    ax.text(cx, cy - 0.64, "red dash = re-infection", ha="center",
            va="center", fontsize=6.2, color=RED, zorder=9)


def accrual_panel(ax, d, fav, rate, cum, picks):
    ax2 = ax.twinx()
    ax2.fill_between(d, 0.0, fav, color=GREY, alpha=0.16, linewidth=0.0,
                     zorder=0)
    ax2.plot(d, fav, color=GREY, lw=0.8, zorder=1, label="weather favourability")
    ax2.set_ylim(0.0, 3.1)
    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.set_ylabel("favourability index (-)", color=GREY, fontsize=7.4)
    ax2.tick_params(axis="y", colors=GREY, labelsize=6.5)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(HAIR)

    for j, i0 in enumerate(picks):
        ax.axvspan(d[i0], d[i0 + WINDOW], facecolor=GREEN, alpha=0.16,
                   edgecolor="none", zorder=1)
        ax.text(d[i0] + WINDOW / 2.0, cum[-1] * 0.025, "window %d" % (j + 1),
                ha="center", va="bottom", fontsize=6.4, color=GREEN,
                rotation=90.0, zorder=6)

    ax.plot(d, cum, color=RED, lw=1.9, zorder=5, label="cumulative cycles")
    whole = np.arange(1.0, np.floor(cum[-1]) + 1.0)
    idx = [int(np.searchsorted(cum, v)) for v in whole]
    ax.plot(d[idx], cum[idx], ls="none", marker="o", ms=3.4, mfc="white",
            mew=1.0, mec=RED, zorder=6, label="completed cycle")
    for v, i in zip(whole, idx):
        ax.plot([d[i], d[i]], [0.0, cum[i]], color=RED, lw=0.5, alpha=0.35,
                zorder=2)

    ax.text(d[-1] - 2.0, cum[-1] * 1.015, "%.1f cycles by day %d"
            % (cum[-1], SEASON), ha="right", va="bottom", fontsize=6.8,
            color=DEEP_RED)
    ax.set_xlim(0.0, SEASON)
    ax.set_ylim(0.0, cum[-1] * 1.10)
    ax.set_xlabel("days after crop emergence (d)", fontsize=7.6)
    ax.set_ylabel("cumulative infection cycles (cycles)", color=RED,
                  fontsize=7.6)
    ax.tick_params(axis="y", colors=RED, labelsize=6.8)
    ax.tick_params(axis="x", labelsize=6.8)
    ax.spines["left"].set_color(RED)
    ax.grid(axis="y", lw=0.5, alpha=0.6)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.6, frameon=False, loc="upper left",
              bbox_to_anchor=(0.015, 1.015), handlelength=1.5)
    ax.set_title("Two ten-day windows carry %.0f %% of the season"
                 % (100.0 * sum(np.sum(rate[i:i + WINDOW]) for i in picks) /
                    cum[-1]), fontsize=8.6, fontweight="bold", color=INK)


def main():
    set_pub_style()
    rng = np.random.default_rng(7421)
    d, fav, rate, cum = season_model(rng)
    picks = control_windows(rate)

    fig = plt.figure(figsize=(W, H))
    ax_ring = inch_axes(fig, (0.22, 0.55, 5.70, 5.05))
    ax_acc = axes_inches(fig, (6.42, 1.28, 3.38, 3.22))

    ring_panel(ax_ring, 5.70, 5.05, rng, float(cum[-1]))
    accrual_panel(ax_acc, d, fav, rate, cum, picks)

    fig.text(0.016, 6.14 / H,
             "Six steps close the loop in about %.0f days, so the pathogen "
             "completes %.1f infection cycles in one season"
             % (1.0 / np.max(rate), cum[-1]),
             ha="left", va="center", fontsize=11.0, fontweight="bold",
             color=INK)
    fig.text(0.016, 0.085 / H, textwrap.fill(
             "The cycle count in the ring is the end point of the accrual "
             "curve in (b): each day adds one over the current cycle length, "
             "which runs from %.0f d when the weather is fully favourable to "
             "%.0f d when it is not. Control windows are the two %d-day "
             "stretches with the largest gain."
             % (1.0 / np.max(rate), 1.0 / np.min(rate), WINDOW), 195),
             ha="left", va="bottom", fontsize=6.5, color=MUTED,
             linespacing=1.5)

    panel_label_inches(fig, ax_ring, "(a)", dx=0.06, dy=-0.14)
    panel_label_inches(fig, ax_acc, "(b)", dx=-0.66, dy=0.18)

    savefig(fig, "742_pla_infection_cycle.png")


if __name__ == "__main__":
    main()

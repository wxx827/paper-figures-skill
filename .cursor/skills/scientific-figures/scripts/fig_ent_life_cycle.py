"""Insect life cycle: one thermal budget explains every stage duration.

The ring is the conclusion - seven stages, drawn as silhouettes, joined by the
arrows a generation actually follows.  Nothing on it is typed twice: each stage
carries a thermal constant in degree-days, and the day count printed beside it
is that constant divided by the reference thermal excess, so the ring, the
rate curve in panel (b) and the stacked durations in panel (c) are three views
of the same seven numbers.  Panel (b) adds the part a linear degree-day model
cannot express - development collapses above the optimum - and panel (c) shows
what that costs in calendar time at three field temperatures.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon

from pubstyle import set_pub_style, savefig, get_palette, darken, lighten
from figkit import axes_inches, inch_axes, panel_label_inches, arrow

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
SOFT = "#f5f5f5"
GOLD = "#c9a227"
RED = "#b23a48"
BLUE = "#3b5b92"
GREY = "#808080"

W, H = 10.4, 6.4
SPECIES = "Ostrinia furnacalis"
T_BASE = 11.2               # lower developmental threshold, deg C
T_OPT_SET = 30.5            # onset of high-temperature decline, deg C
T_MAX = 36.0                # upper developmental limit, deg C
REF_T = 24.0                # reference rearing temperature, deg C
FIELD_T = (18.0, 24.0, 30.0)

# (stage, thermal constant in deg C.d, silhouette kind, instar index)
STAGES = [
    ("Egg", 42.0, "egg", 0),
    ("Instar I", 28.0, "larva", 1),
    ("Instar II", 24.0, "larva", 2),
    ("Instar III", 30.0, "larva", 3),
    ("Instar IV", 46.0, "larva", 4),
    ("Pupa", 118.0, "pupa", 0),
    ("Adult", 36.0, "adult", 0),
]

DEG = r"$^{\circ}$C"
DEGD = r"$^{\circ}$C$\cdot$d"


def stage_colors():
    blues = get_palette("cb_sequential_blue", 8)[2:7]
    return blues + [GOLD, RED]


def duration(k, temp):
    """Calendar days a stage needs at ``temp`` under the degree-day model."""
    return k / max(temp - T_BASE, 1e-6)


def dev_rate(temp):
    """Whole-generation development rate, 1/d, with a high-temperature drop."""
    temp = np.asarray(temp, dtype=float)
    ktot = sum(s[1] for s in STAGES)
    lin = np.clip(temp - T_BASE, 0.0, None) / ktot
    drop = np.where(temp <= T_OPT_SET, 1.0,
                    np.clip(1.0 - ((temp - T_OPT_SET) /
                                   (T_MAX - T_OPT_SET)) ** 2, 0.0, 1.0))
    return lin * drop


# ---------------------------------------------------------------- silhouettes
def draw_egg(ax, x, y, s, col):
    ax.add_patch(Ellipse((x, y), 0.44 * s, 0.60 * s, facecolor=col,
                         edgecolor="white", lw=0.8, zorder=6))
    ax.add_patch(Ellipse((x - 0.08 * s, y + 0.12 * s), 0.14 * s, 0.19 * s,
                         facecolor=lighten(col, 0.50), edgecolor="none",
                         zorder=7))


def draw_larva(ax, x, y, s, col, instar):
    n = 4 + instar
    span = (0.54 + 0.085 * instar) * s
    xs = np.linspace(-span / 2.0, span / 2.0, n)
    ys = 0.09 * s * np.sin(np.linspace(0.0, np.pi, n))
    rmax = (0.110 + 0.017 * instar) * s
    for i, (dx, dy) in enumerate(zip(xs, ys)):
        f = 1.0 - 0.30 * abs(i - (n - 1) / 2.0) / ((n - 1) / 2.0)
        ax.add_patch(Circle((x + dx, y + dy), rmax * f, facecolor=col,
                            edgecolor="white", lw=0.55, zorder=6))
    ax.add_patch(Circle((x + xs[-1] + 0.34 * rmax, y + ys[-1]), rmax * 0.84,
                        facecolor=darken(col, 0.30), edgecolor="white",
                        lw=0.55, zorder=7))


def draw_pupa(ax, x, y, s, col):
    u = np.linspace(-1.0, 1.0, 90)
    hw = (0.30 * s * np.sqrt(np.clip(1.0 - u ** 2, 0.0, None)) *
          (1.0 - 0.32 * u))
    yy = y + 0.46 * s * u
    pts = np.vstack([np.column_stack([x + hw, yy]),
                     np.column_stack([x - hw[::-1], yy[::-1]])])
    ax.add_patch(Polygon(pts, closed=True, facecolor=col, edgecolor="white",
                         lw=0.8, zorder=6))
    for uu in (-0.40, -0.14, 0.14):
        hwi = 0.30 * s * np.sqrt(1.0 - uu ** 2) * (1.0 - 0.32 * uu)
        ax.plot([x - 0.84 * hwi, x + 0.84 * hwi],
                [y + 0.46 * s * uu] * 2, color=darken(col, 0.30), lw=0.55,
                zorder=7, solid_capstyle="round")


def draw_adult(ax, x, y, s, col):
    wing = lighten(col, 0.50)
    for sgn in (-1.0, 1.0):
        ax.add_patch(Ellipse((x + sgn * 0.24 * s, y + 0.02 * s), 0.52 * s,
                             0.30 * s, angle=sgn * 26.0, facecolor=wing,
                             edgecolor="white", lw=0.7, zorder=6))
    ax.add_patch(Ellipse((x, y), 0.16 * s, 0.62 * s, facecolor=col,
                         edgecolor="white", lw=0.7, zorder=7))
    for sgn in (-1.0, 1.0):
        ax.plot([x, x + sgn * 0.15 * s], [y + 0.28 * s, y + 0.46 * s],
                color=darken(col, 0.25), lw=0.7, zorder=8,
                solid_capstyle="round")


def draw_stage(ax, kind, x, y, s, col, instar):
    if kind == "egg":
        draw_egg(ax, x, y, s, col)
    elif kind == "larva":
        draw_larva(ax, x, y, s, col, instar)
    elif kind == "pupa":
        draw_pupa(ax, x, y, s, col)
    else:
        draw_adult(ax, x, y, s, col)


# ------------------------------------------------------------------ the ring
def ring_panel(ax, w, h, cols, days):
    cx, cy = w / 2.0, h / 2.0
    radius, node = 1.47, 0.390
    n = len(STAGES)
    step = 360.0 / n
    gap = np.degrees((node + 0.070) / radius)

    ax.add_patch(Circle((cx, cy), 0.98, facecolor=SOFT, edgecolor=HAIR,
                        lw=0.7, zorder=1))

    for i in range(n):
        a0 = 90.0 - i * step - gap
        a1 = 90.0 - (i + 1) * step + gap
        theta = np.radians(step - 2.0 * gap)
        p0 = (cx + radius * np.cos(np.radians(a0)),
              cy + radius * np.sin(np.radians(a0)))
        p1 = (cx + radius * np.cos(np.radians(a1)),
              cy + radius * np.sin(np.radians(a1)))
        arrow(ax, p0, p1, color=RULE if i < n - 1 else GREY,
              rad=-np.tan(theta / 4.0), lw=1.05, ms=8.5, zorder=3,
              ls="-" if i < n - 1 else (0, (3.4, 2.0)))

    for i, ((name, k, kind, instar), col) in enumerate(zip(STAGES, cols)):
        a = np.radians(90.0 - i * step)
        x, y = cx + radius * np.cos(a), cy + radius * np.sin(a)
        ax.add_patch(Circle((x, y), node, facecolor="white", edgecolor=HAIR,
                            lw=0.8, zorder=4))
        draw_stage(ax, kind, x, y, 0.62, col, instar)

        lx = cx + (radius + node + 0.15) * np.cos(a)
        ly = cy + (radius + node + 0.15) * np.sin(a)
        ca = np.cos(a)
        ha = "left" if ca > 0.30 else ("right" if ca < -0.30 else "center")
        dy = 0.10 if np.sin(a) > 0.30 else (-0.10 if np.sin(a) < -0.30 else 0.0)
        ax.text(lx, ly + dy + 0.085, name, ha=ha, va="center", fontsize=7.2,
                color=INK, fontweight="bold", zorder=8)
        ax.text(lx, ly + dy - 0.085,
                "%.1f d  |  %.0f %s" % (days[i], k, DEGD), ha=ha, va="center",
                fontsize=6.4, color=SUB, zorder=8)

    total_d, total_k = float(np.sum(days)), sum(s[1] for s in STAGES)
    ax.text(cx, cy + 0.30, SPECIES, ha="center", va="center", fontsize=8.2,
            fontstyle="italic", color=INK, zorder=8)
    ax.text(cx, cy + 0.08, "%.1f d at %.0f %s" % (total_d, REF_T, DEG),
            ha="center", va="center", fontsize=7.4, color=RULE, zorder=8)
    ax.text(cx, cy - 0.12, "%.0f %s per generation" % (total_k, DEGD),
            ha="center", va="center", fontsize=7.0, color=SUB, zorder=8)
    ax.text(cx, cy - 0.36, "$T_{base}$ = %.1f %s" % (T_BASE, DEG),
            ha="center", va="center", fontsize=6.8, color=MUTED, zorder=8)
    ax.text(cx, cy - 0.62, "dashed arrow = oviposition", ha="center",
            va="center", fontsize=6.2, color=MUTED, zorder=8)
    return total_d


# ----------------------------------------------------------------- rate panel
def rate_panel(ax, rng):
    grid = np.linspace(T_BASE - 2.0, T_MAX + 0.6, 500)
    rate = dev_rate(grid)
    t_opt = float(grid[int(np.argmax(rate))])
    r_opt = float(np.max(rate))

    lin_t = np.linspace(T_BASE, T_OPT_SET + 3.0, 60)
    ax.plot(lin_t, np.clip(lin_t - T_BASE, 0.0, None) /
            sum(s[1] for s in STAGES), color=GREY, lw=1.0,
            ls=(0, (4.0, 2.2)), zorder=3, label="degree-day model")
    ax.plot(grid, rate, color=BLUE, lw=1.9, zorder=5, label="observed rate")

    obs_t = np.array([13.0, 16.0, 19.0, 22.0, 25.0, 28.0, 31.0, 33.0, 35.0])
    obs_r = dev_rate(obs_t) * (1.0 + rng.normal(0.0, 0.055, obs_t.size))
    ax.plot(obs_t, obs_r, ls="none", marker="o", ms=3.4, mfc="white", mew=1.0,
            mec=BLUE, zorder=6)

    ax.axvline(T_BASE, color=RULE, lw=0.8, ls=(0, (1.6, 1.8)), zorder=2)
    ax.axvline(t_opt, color=RED, lw=0.9, ls=(0, (1.6, 1.8)), zorder=2)
    ax.plot([t_opt], [r_opt], marker="v", ms=4.2, color=RED, zorder=7)
    ax.text(T_BASE + 0.5, r_opt * 0.94, "$T_{base}$\n%.1f %s" % (T_BASE, DEG),
            ha="left", va="top", fontsize=6.4, color=RULE, linespacing=1.4)
    ax.text(t_opt - 0.6, r_opt * 1.02, "$T_{opt}$ %.1f %s" % (t_opt, DEG),
            ha="right", va="bottom", fontsize=6.6, color=RED)

    ax.set_xlabel("temperature (%s)" % DEG, fontsize=7.4)
    ax.set_ylabel("development rate (1/d)", fontsize=7.4)
    ax.set_xlim(T_BASE - 2.5, T_MAX + 1.0)
    ax.set_ylim(0.0, r_opt * 1.24)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.6, frameon=False, loc="upper left",
              bbox_to_anchor=(0.015, 1.02), handlelength=1.5)
    ax.set_title("Rate is linear only below $T_{opt}$", fontsize=8.4,
                 fontweight="bold", color=INK)
    return t_opt


def stack_panel(ax, cols):
    ypos = np.arange(len(FIELD_T), dtype=float)
    handles = []
    for j, temp in enumerate(FIELD_T):
        left = 0.0
        for (name, k, _, _), col in zip(STAGES, cols):
            d = duration(k, temp)
            b = ax.barh(ypos[j], d, left=left, height=0.56, color=col,
                        edgecolor="white", lw=0.7, zorder=3)
            if j == 0:
                handles.append((b[0], name))
            left += d
        ax.text(left + 1.0, ypos[j], "%.1f d" % left, ha="left", va="center",
                fontsize=6.6, color=INK)

    totals = [sum(duration(k, t) for _, k, _, _ in STAGES) for t in FIELD_T]
    ax.set_yticks(ypos)
    ax.set_yticklabels(["%.0f %s" % (t, DEG) for t in FIELD_T], fontsize=6.8)
    ax.set_xlabel("cumulative stage duration (days)", fontsize=7.4)
    ax.set_xlim(0.0, max(totals) * 1.16)
    ax.set_ylim(-0.62, len(FIELD_T) - 0.38)
    ax.invert_yaxis()
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.7)
    ax.set_title("A 6 %s lift halves the generation" % DEG, fontsize=8.4,
                 fontweight="bold", color=INK)
    return handles, totals


def main():
    set_pub_style()
    rng = np.random.default_rng(7381)
    cols = stage_colors()
    days = [duration(k, REF_T) for _, k, _, _ in STAGES]

    fig = plt.figure(figsize=(W, H))
    ax_ring = inch_axes(fig, (0.16, 0.80, 5.60, 4.86))
    ax_rate = axes_inches(fig, (6.90, 3.42, 2.95, 1.92))
    ax_stk = axes_inches(fig, (6.90, 0.98, 2.95, 1.62))

    total_d = ring_panel(ax_ring, 5.60, 4.86, cols, days)
    t_opt = rate_panel(ax_rate, rng)
    handles, totals = stack_panel(ax_stk, cols)

    fig.legend([h for h, _ in handles], [n for _, n in handles],
               loc="lower center", bbox_to_anchor=(0.5, 0.038), ncol=7,
               frameon=False, fontsize=6.8, handlelength=1.3,
               columnspacing=1.1, handletextpad=0.45)

    fig.text(0.016, 6.14 / H,
             "Seven stages, one thermal budget: %.0f %s completes a "
             "generation in %.1f d at %.0f %s"
             % (sum(s[1] for s in STAGES), DEGD, total_d, REF_T, DEG),
             ha="left", va="center", fontsize=11.0, fontweight="bold",
             color=INK)
    fig.text(0.016, 0.085 / H,
             "Stage days are the stage thermal constant divided by "
             "(T - $T_{base}$); the same constants give %.1f d at %.0f %s "
             "and %.1f d at %.0f %s. $T_{opt}$ = %.1f %s is read off the "
             "rate curve." % (totals[0], FIELD_T[0], DEG, totals[-1],
                              FIELD_T[-1], DEG, t_opt, DEG),
             ha="left", va="bottom", fontsize=6.5, color=MUTED)

    panel_label_inches(fig, ax_ring, "(a)", dx=0.02, dy=-0.10)
    panel_label_inches(fig, ax_rate, "(b)", dx=-0.62, dy=0.16)
    panel_label_inches(fig, ax_stk, "(c)", dx=-0.62, dy=0.16)

    savefig(fig, "738_ent_life_cycle.png")


if __name__ == "__main__":
    main()

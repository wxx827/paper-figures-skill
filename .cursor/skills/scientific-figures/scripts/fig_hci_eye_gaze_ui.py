"""Gaze overlay on a UI wireframe: heat map, scan path and per-AOI timings.

One recorded trial drives everything.  A visit list walks the participant
through ten areas of interest; inside each visit the individual fixations are
scattered around the AOI centre with sampled durations, and that single
fixation table is then used three ways - as the kernel-density heat map, as
the numbered scan path between visit centroids, and as the two bar panels.
So the AOI that looks hottest is by construction the one with the longest
dwell, and the time-to-first-fixation bars are cumulative times read off the
same sequence.

The wireframe itself is drawn in device pixels on an equal-aspect axes, so
the AOI rectangles keep the proportions of a real 1280 x 800 layout.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from pubstyle import set_pub_style, savefig, style_axes
from figkit import axes_inches, panel_labels_inches, safe_colorbar

W, H = 10.4, 6.4
PX, PY = 1280.0, 800.0

RED = "#b23a48"         # gaze heat, task-critical target
BLUE = "#3b5b92"        # scan path, primary UI chrome
GREEN = "#2e7d5b"       # task completion
INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
WIRE = "#e8e8e8"
WIRE_EDGE = "#b3b3b3"

# name -> (x, y, w, h) in device pixels, origin at the lower-left corner
AOI = {
    "Logo": (26, 726, 150, 52),
    "Nav bar": (200, 720, 470, 64),
    "Search": (700, 726, 330, 52),
    "Account": (1058, 726, 196, 52),
    "Sidebar": (26, 90, 232, 604),
    "Headline": (296, 596, 660, 84),
    "Hero image": (296, 356, 660, 216),
    "Body copy": (296, 150, 660, 182),
    "Recommend card": (992, 356, 262, 324),
    "CTA button": (992, 176, 262, 96),
}

# (AOI, number of fixations in this visit) in temporal order
VISITS = [("Logo", 2), ("Nav bar", 3), ("Search", 4), ("Account", 2),
          ("Headline", 6), ("Body copy", 9), ("Sidebar", 7),
          ("Hero image", 5), ("Recommend card", 6), ("Body copy", 4),
          ("CTA button", 5)]


def sample_trial(rng):
    fx, fy, fd, faoi = [], [], [], []
    for name, n in VISITS:
        x, y, w, h = AOI[name]
        for _ in range(n):
            fx.append(x + w * (0.5 + rng.normal(0.0, 0.17)))
            fy.append(y + h * (0.5 + rng.normal(0.0, 0.19)))
            fd.append(float(rng.uniform(0.13, 0.42)))     # seconds
            faoi.append(name)
    return (np.array(fx), np.array(fy), np.array(fd), np.array(faoi))


def aoi_stats(fx, fd, faoi):
    start = np.concatenate([[0.0], np.cumsum(fd)[:-1]])
    names = list(AOI)
    first = {n: float(start[faoi == n].min()) if (faoi == n).any() else np.nan
             for n in names}
    dwell = {n: float(fd[faoi == n].sum()) for n in names}
    return first, dwell, start


def heat_field(fx, fy, fd, nx=200, ny=125, sigma=46.0):
    gx = np.linspace(0.0, PX, nx)
    gy = np.linspace(0.0, PY, ny)
    X, Y = np.meshgrid(gx, gy)
    Z = np.zeros_like(X)
    for x, y, d in zip(fx, fy, fd):
        Z += d * np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2.0 * sigma ** 2))
    return Z / Z.max()


def draw_wireframe(ax):
    ax.add_patch(Rectangle((0, 0), PX, PY, facecolor="white",
                           edgecolor=WIRE_EDGE, linewidth=0.9, zorder=1))
    ax.add_patch(Rectangle((0, 700), PX, 100, facecolor="#f0f0f0",
                           edgecolor="none", zorder=1))
    for name, (x, y, w, h) in AOI.items():
        fc = BLUE if name == "CTA button" else WIRE
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
            facecolor=fc, edgecolor=WIRE_EDGE, linewidth=0.7,
            mutation_aspect=1.0, zorder=2))
        ax.text(x + 8, y + h - 8, name, ha="left", va="top", fontsize=6.4,
                color="white" if name == "CTA button" else MUTED, zorder=4)
    ax.text(*np.add(AOI["CTA button"][:2], (16, 18)), "Start free trial",
            ha="left", va="center", fontsize=7.2, color="white", zorder=5)

    for i in range(6):                       # wireframe text ruling
        y = 292 - i * 26
        ax.plot([312, 312 + (600 if i < 5 else 340)], [y, y], lw=2.4,
                color="#dcdcdc", solid_capstyle="round", zorder=3)
    for i in range(7):
        y = 640 - i * 74
        ax.plot([44, 44 + (188 if i % 3 else 120)], [y, y], lw=2.4,
                color="#dcdcdc", solid_capstyle="round", zorder=3)
    ax.plot([320, 900], [628, 628], lw=5.0, color="#d0d0d0",
            solid_capstyle="round", zorder=3)


def draw_gaze(ax, fx, fy, fd, faoi, Z):
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "gaze_heat", [(1.0, 1.0, 1.0, 0.0),
                      (*mpl.colors.to_rgb("#c9a227"), 0.30),
                      (*mpl.colors.to_rgb(RED), 0.78)])
    im = ax.imshow(Z, extent=(0.0, PX, 0.0, PY), origin="lower", cmap=cmap,
                   vmin=0.0, vmax=1.0, interpolation="bilinear", zorder=6)
    ax.scatter(fx, fy, s=3.0, color=RULE, alpha=0.40, lw=0.0, zorder=7)

    cx, cy, k = [], [], 0
    for name, n in VISITS:
        cx.append(fx[k:k + n].mean())
        cy.append(fy[k:k + n].mean())
        k += n
    ax.plot(cx, cy, lw=1.0, color=BLUE, alpha=0.85, zorder=8)
    for i, (x, y) in enumerate(zip(cx, cy), start=1):
        last = i == len(cx)
        ax.scatter([x], [y], s=118.0, color=GREEN if last else BLUE,
                   edgecolors="white", linewidths=0.9, zorder=9)
        ax.text(x, y, f"{i}", ha="center", va="center", fontsize=6.4,
                color="white", zorder=10)
    ax.annotate("task completed:\nclick on the trial button",
                xy=(cx[-1], cy[-1]), xytext=(cx[-1] - 250, cy[-1] - 118),
                ha="center", va="top", fontsize=6.6, color=GREEN, zorder=11,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=GREEN,
                                shrinkA=1.0, shrinkB=6.0))
    return im


def bar_panel(ax, names, vals, color, title, xlabel, note_fmt):
    y = np.arange(len(names))
    ax.barh(y, vals, height=0.66, color=color, edgecolor="white",
            linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlim(0.0, max(vals) * 1.30)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.025, yi, note_fmt.format(v), va="center",
                ha="left", fontsize=6.2, color=SUB)
    ax.set_xlabel(xlabel, fontsize=8.0)
    ax.set_title(title, fontsize=8.5, fontweight="bold")
    ax.tick_params(labelsize=6.6)
    style_axes(ax)


def main():
    set_pub_style()
    rng = np.random.default_rng(675)
    fx, fy, fd, faoi = sample_trial(rng)
    first, dwell, start = aoi_stats(fx, fd, faoi)
    Z = heat_field(fx, fy, fd)

    fig = plt.figure(figsize=(W, H))
    ax_ui = axes_inches(fig, (0.42, 1.24, 6.10, 3.81))
    ax_ff = axes_inches(fig, (8.02, 3.30, 2.04, 1.94))
    ax_dw = axes_inches(fig, (8.02, 0.76, 2.04, 1.94))

    ax_ui.set_xlim(0.0, PX)
    ax_ui.set_ylim(0.0, PY)
    ax_ui.set_aspect("equal")
    ax_ui.axis("off")
    draw_wireframe(ax_ui)
    im = draw_gaze(ax_ui, fx, fy, fd, faoi, Z)
    ax_ui.set_title("Gaze never reaches the call to action until the very "
                    "last visit", fontsize=9.0, fontweight="bold")
    safe_colorbar(fig, im, ax_ui, label="fixation density (-)",
                  side="bottom", width=0.12, pad=0.16, shrink=0.55)

    order = sorted(AOI, key=lambda n: -dwell[n])
    bar_panel(ax_ff, order, [first[n] for n in order], BLUE,
              "Time to first fixation", "time from trial start (s)",
              "{:.1f} s")
    bar_panel(ax_dw, order, [dwell[n] for n in order], RED,
              "Total dwell time", "summed fixation time (s)", "{:.1f} s")

    total = float(fd.sum())
    cta_first = first["CTA button"]

    panel_labels_inches(fig, [ax_ui, ax_ff, ax_dw], dx=-0.34, dy=0.28)
    savefig(fig, "675_hci_eye_gaze_ui.png")


if __name__ == "__main__":
    main()

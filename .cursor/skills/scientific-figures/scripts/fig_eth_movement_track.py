"""GPS relocation tracks of four collared individuals in a patchy landscape.

The main panel is the claim: movement is not a plain random walk but a
two-mode process - long bouts of short, strongly correlated steps inside a
habitat patch, punctuated by rare near-ballistic relocations between patches.
Every annotated number (path length, net displacement, straightness) is
measured from the plotted coordinate arrays.  The two subordinate panels carry
the evidence for the two modes separately: the step-length survival curve on
log axes, where a heavy-tailed model separates from an exponential one, and
the turning-angle rose, whose concentration at zero is what "correlated" means.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.interpolate import CubicSpline

from pubstyle import set_pub_style, savefig, annotate_box, lighten
from figkit import axes_inches, panel_labels_inches, safe_colorbar, add_scale_bar

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"

N_STEP = 520              # relocations per individual
DT_MIN = 15.0             # fix interval, minutes
IDS = ["F01", "F02", "M03", "M04"]

PATCHES = [(0.00, 0.00, 0.62), (1.85, 0.72, 0.50), (0.95, 1.85, 0.55),
           (-1.40, 1.30, 0.44), (1.10, -1.45, 0.48)]


def blob(rng, cx, cy, r, k=9):
    """Smooth closed outline of a habitat patch, in kilometres."""
    a = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    rr = r * (1.0 + 0.20 * rng.normal(size=k))
    cs = CubicSpline(np.append(a, 2.0 * np.pi), np.append(rr, rr[0]),
                     bc_type="periodic")
    t = np.linspace(0.0, 2.0 * np.pi, 220)
    return cx + cs(t) * np.cos(t), cy + cs(t) * np.sin(t)


def wrapped_cauchy(rng, rho, n=1):
    u = rng.uniform(0.0, 1.0, n)
    v = (1.0 - rho) / (1.0 + rho)
    ang = 2.0 * np.arctan(v * np.tan(np.pi * (u - 0.5)))
    return ang if n > 1 else float(ang[0])


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def walk(rng, start, n=N_STEP):
    """Correlated random walk with rare long inter-patch relocations."""
    centres = np.array([[p[0], p[1]] for p in PATCHES])
    pos = np.array(start, dtype=float)
    heading = rng.uniform(-np.pi, np.pi)
    xs, ys = [pos[0]], [pos[1]]
    steps, turns = [], []
    for _ in range(n):
        if rng.random() < 0.030:                      # foraging relocation
            length = 240.0 + 240.0 * rng.pareto(1.35)
            turn = float(rng.uniform(-np.pi, np.pi))
        else:                                          # within-patch step
            length = float(rng.gamma(2.5, 24.0))
            turn = wrapped_cauchy(rng, 0.72)
        heading = wrap(heading + turn)
        d = centres - pos
        dist = np.hypot(d[:, 0], d[:, 1])
        j = int(np.argmin(dist))
        pull = 0.035 + 0.16 * float(np.clip(dist[j] / 0.55 - 1.0, 0.0, 1.0))
        heading = wrap(heading + pull * wrap(np.arctan2(d[j, 1], d[j, 0])
                                             - heading))
        pos = pos + (length / 1000.0) * np.array([np.cos(heading),
                                                  np.sin(heading)])
        xs.append(pos[0])
        ys.append(pos[1])
        steps.append(length)
        turns.append(turn)
    return (np.array(xs), np.array(ys), np.array(steps),
            np.degrees(np.array(turns)))


def frame_to_box(ax, xs, ys, box_w, box_h, pad=0.07):
    """Pad the data extent to the panel aspect so equal-aspect fills the box."""
    x0, x1 = float(np.min(xs)), float(np.max(xs))
    y0, y1 = float(np.min(ys)), float(np.max(ys))
    mx, my = pad * (x1 - x0), pad * (y1 - y0)
    x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my, y1 + my
    want, have = box_w / box_h, (x1 - x0) / (y1 - y0)
    if have < want:
        e = ((y1 - y0) * want - (x1 - x0)) / 2.0
        x0, x1 = x0 - e, x1 + e
    else:
        e = ((x1 - x0) / want - (y1 - y0)) / 2.0
        y0, y1 = y0 - e, y1 + e
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    return x1 - x0


def north_arrow(ax, x=0.955, y=0.855, size=0.075):
    ax.annotate("", xy=(x, y + size), xytext=(x, y), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=RULE, lw=1.0,
                                mutation_scale=8, shrinkA=0, shrinkB=0))
    ax.text(x, y + size + 0.014, "N", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=7.0, color=RULE)


def main_panel(fig, ax, tracks, box):
    rng = np.random.default_rng(4)
    for cx, cy, r in PATCHES:
        bx, by = blob(rng, cx, cy, r)
        ax.fill(bx, by, facecolor=lighten(GREEN, 0.86),
                edgecolor=lighten(GREEN, 0.55), linewidth=0.7, zorder=0)

    hours = N_STEP * DT_MIN / 60.0
    norm = mpl.colors.Normalize(0.0, hours)
    t = np.arange(N_STEP + 1) * DT_MIN / 60.0
    for x, y, _, _ in tracks:
        pts = np.column_stack([x, y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", norm=norm, linewidth=0.85,
                            alpha=0.95, zorder=3, capstyle="round")
        lc.set_array(t[:-1])
        ax.add_collection(lc)

    rows = []
    for name, (x, y, s, _) in zip(IDS, tracks):
        path = float(s.sum()) / 1000.0
        net = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        rows.append(f"{name}  {path:6.1f}   {net:5.2f}   {net / path:5.3f}")
        ax.plot([x[0]], [y[0]], marker="o", ms=5.2, mfc="white", mec=RULE,
                mew=1.0, zorder=6)
        ax.plot([x[-1]], [y[-1]], marker="s", ms=5.0, mfc=RED, mec="white",
                mew=0.9, zorder=6)
        ax.text(x[0], y[0] + 0.085, name, ha="center", va="bottom",
                fontsize=6.6, color=INK, zorder=7)

    ax.set_xlabel("easting (km)")
    ax.set_ylabel("northing (km)")
    ax.set_aspect("equal")
    ax.grid(lw=0.45, alpha=0.55)
    ax.set_title("Patch residence broken by rare long relocations",
                 fontsize=9.0, fontweight="bold", color=INK)
    annotate_box(ax, "ID     path     net      SI\n"
                     "       (km)    (km)     (-)\n" + "\n".join(rows),
                 loc="upper left", fontsize=6.4, fontfamily="monospace",
                 color=SUB)
    add_scale_bar(ax, 1.0, "1 km", loc=(0.035, 0.055), height=0.016,
                  text_color=INK, halo="white", fontsize=6.4)
    north_arrow(ax)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap="viridis")
    safe_colorbar(fig, sm, ax, label="time since collaring (h)", width=0.10,
                  pad=0.07, labelsize=6.6)
    ax.plot([], [], marker="o", ms=4.6, mfc="white", mec=RULE, mew=1.0, lw=0,
            label="first fix")
    ax.plot([], [], marker="s", ms=4.4, mfc=RED, mec="white", mew=0.8, lw=0,
            label="last fix")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right",
              handletextpad=0.4, borderaxespad=0.6)


def step_panel(ax, steps):
    s = np.sort(steps)
    n = s.size
    ccdf = 1.0 - np.arange(n) / n
    xmin = float(np.percentile(s, 80.0))
    tail = s[s >= xmin]
    alpha = 1.0 + tail.size / float(np.sum(np.log(tail / xmin)))
    p_tail = tail.size / float(n)

    ax.loglog(s, ccdf, lw=0.0, marker="o", ms=1.9, mfc=lighten(RULE, 0.45),
              mec="none", label=f"observed (n={n})")
    grid = np.logspace(np.log10(s[0]), np.log10(s[-1] * 1.1), 200)
    ax.loglog(grid, np.exp(-grid / float(s.mean())), color=BLUE, lw=1.3,
              ls=(0, (4, 2.2)), label="exponential")
    g2 = grid[grid >= xmin]
    ax.loglog(g2, p_tail * (g2 / xmin) ** (1.0 - alpha), color=RED, lw=1.5,
              label=f"power law, $\\alpha$ = {alpha:.2f}")
    ax.axvline(xmin, color=MUTED, lw=0.8, ls=(0, (2.2, 2.0)))
    ax.text(xmin * 1.12, 3.5e-4, f"$l_{{min}}$ = {xmin:.0f} m", fontsize=6.2,
            color=MUTED, ha="left", va="bottom")

    ax.set_xlabel("step length $l$ (m)", fontsize=7.4)
    ax.set_ylabel("P(L > $l$) (-)", fontsize=7.4)
    ax.set_ylim(2.5e-4, 1.6)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.45, alpha=0.6, which="major")
    ax.legend(fontsize=6.4, frameon=False, loc="lower left",
              handlelength=1.5, borderaxespad=0.25)
    ax.set_title("Heavy tail beyond the patch scale", fontsize=8.2,
                 fontweight="bold", color=INK)
    return alpha


def rose_panel(ax, turns):
    edges = np.deg2rad(np.arange(-180.0, 181.0, 15.0))
    cnt, _ = np.histogram(np.deg2rad(turns), bins=edges)
    frac = 100.0 * cnt / cnt.sum()
    centres = 0.5 * (edges[:-1] + edges[1:])
    cmap = mpl.colormaps["magma"]
    cols = [cmap(0.18 + 0.62 * v / frac.max()) for v in frac]
    ax.bar(centres, frac, width=np.deg2rad(14.0), bottom=0.0, color=cols,
           edgecolor="white", linewidth=0.35, zorder=3)

    r = float(np.abs(np.mean(np.exp(1j * np.deg2rad(turns)))))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0.0, 360.0, 45.0),
                      labels=["0", "45", "90", "135", "180", "-135", "-90",
                              "-45"], fontsize=6.2)
    ax.set_rlabel_position(112.0)
    ax.set_ylim(0.0, frac.max() * 1.12)
    ax.tick_params(labelsize=6.0)
    ax.grid(lw=0.45, alpha=0.6)
    ax.set_title("Turning angles (deg), 15 deg bins", fontsize=8.2,
                 fontweight="bold", color=INK, pad=7.0)
    ax.text(0.5, -0.16, f"mean vector length r = {r:.2f},  bars in % of "
            f"n={turns.size}", transform=ax.transAxes, ha="center",
            va="top", fontsize=6.4, color=SUB)
    return r


def main():
    set_pub_style()
    rng = np.random.default_rng(20250813)
    tracks = [walk(rng, s) for s in [(0.05, -0.10), (1.80, 0.60),
                                     (0.90, 1.75), (-1.30, 1.20)]]
    steps = np.concatenate([t[2] for t in tracks])
    turns = np.concatenate([t[3] for t in tracks])

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (0.80, 0.80, 5.05, 4.62))
    ax_step = axes_inches(fig, (6.98, 3.62, 2.90, 1.80))
    ax_rose = axes_inches(fig, (6.98, 0.66, 2.90, 1.74), projection="polar")

    xs = np.concatenate([t[0] for t in tracks])
    ys = np.concatenate([t[1] for t in tracks])
    frame_to_box(ax_main, xs, ys, 5.05, 4.62)
    main_panel(fig, ax_main, tracks, 5.05)
    alpha = step_panel(ax_step, steps)
    r = rose_panel(ax_rose, turns)

    panel_labels_inches(fig, [ax_main, ax_step, ax_rose], dx=-0.46, dy=0.14)
    savefig(fig, "774_eth_movement_track.png")


if __name__ == "__main__":
    main()

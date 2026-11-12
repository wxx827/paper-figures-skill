"""Urban road-network speed map with a graded level-of-service legend.

A synthetic 6 x 6 km street grid (freeway, arterial, collector, local) is
drawn as slightly bent polylines whose stroke width encodes the road class
and whose colour encodes the peak-hour operating speed on a five-step
level-of-service ramp.  Severely congested links get a dark casing so the
bottlenecks pop out.  The two auxiliary panels on the right are held to
~43 % of the map width and grouped on a light grey card so the map keeps
the visual lead.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon
from pubstyle import set_pub_style, savefig

# ---- shared transport palette --------------------------------------------
LOS_COLORS = ["#a8433a", "#d1834b", "#d9b96a", "#7ba05b", "#3f7d5f"]
LOS_EDGES = [0.0, 15.0, 25.0, 35.0, 45.0, 90.0]
LOS_NAMES = ["severe (< 15)", "congested (15-25)", "unstable (25-35)",
             "stable (35-45)", "free flow (> 45)"]
INK = "#2b2b2b"
GREY = "#8c8c8c"
CARD = "#f4f3f0"
WATER = "#ccd8de"

EXTENT = 6.0
CLASS_LW = {"freeway": 3.0, "arterial": 1.95, "collector": 1.15,
            "local": 0.58}
CLASS_V0 = {"freeway": 82.0, "arterial": 54.0, "collector": 39.0,
            "local": 27.0}
CLASS_SENS = {"freeway": 0.84, "arterial": 0.80, "collector": 0.64,
              "local": 0.46}


def bend(p0, p1, amp, rng, n=9):
    """Turn a straight edge into a gently bowed polyline."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = np.linspace(0.0, 1.0, n)
    seg = p0 + (p1 - p0) * t[:, None]
    d = p1 - p0
    L = np.hypot(*d)
    if L < 1e-9:
        return seg
    nvec = np.array([-d[1], d[0]]) / L
    a = rng.normal(0.0, amp) * L
    seg = seg + nvec * (a * np.sin(np.pi * t))[:, None]
    return seg


def congestion_field(x, y):
    """0 (free) .. 1 (gridlock) surface over the city."""
    c = np.exp(-(((x - 3.0) ** 2 + (y - 3.05) ** 2) / (2 * 1.45 ** 2)))
    c += 0.62 * np.exp(-(((x - 4.55) ** 2 + (y - 1.65) ** 2) / (2 * 0.72 ** 2)))
    c += 0.55 * np.exp(-(((x - 1.55) ** 2 + (y - 4.45) ** 2) / (2 * 0.78 ** 2)))
    c += 0.35 * np.exp(-(((x - 4.9) ** 2 + (y - 4.7) ** 2) / (2 * 0.62 ** 2)))
    return np.clip(c, 0.0, 1.12)


def build_network(rng):
    nx, ny = 13, 11
    gx = np.linspace(0.28, EXTENT - 0.28, nx)
    gy = np.linspace(0.30, EXTENT - 0.30, ny)
    X, Y = np.meshgrid(gx, gy)
    r = np.hypot(X - 3.0, Y - 3.0)
    jit = 0.030 + 0.052 * np.clip(r / 3.2, 0, 1) ** 1.6
    X = X + rng.normal(0, 1, X.shape) * jit
    Y = Y + rng.normal(0, 1, Y.shape) * jit

    edges = []

    def klass(i, j):
        if i % 5 == 0 or j % 6 == 0:
            return "arterial"
        if i % 2 == 0 or j % 2 == 0:
            return "collector"
        return "local"

    for i in range(ny):
        for j in range(nx - 1):
            k = klass(i, j)
            if k == "local" and rng.random() < 0.30:
                continue
            edges.append(((X[i, j], Y[i, j]), (X[i, j + 1], Y[i, j + 1]), k))
    for i in range(ny - 1):
        for j in range(nx):
            k = klass(i, j)
            if k == "local" and rng.random() < 0.30:
                continue
            edges.append(((X[i, j], Y[i, j]), (X[i + 1, j], Y[i + 1, j]), k))

    polys, classes = [], []
    for p0, p1, k in edges:
        amp = {"arterial": 0.010, "collector": 0.030, "local": 0.055}[k]
        polys.append(bend(p0, p1, amp, rng))
        classes.append(k)

    # ---- ring road -------------------------------------------------------
    th = np.linspace(0, 2 * np.pi, 260)
    rr = 2.06 + 0.09 * np.cos(3 * th + 0.6) + 0.05 * np.cos(5 * th)
    ring = np.c_[3.0 + rr * np.cos(th), 3.02 + rr * np.sin(th)]
    for a in range(0, 260 - 10, 10):
        polys.append(ring[a:a + 11])
        classes.append("arterial")

    # ---- diagonal freeway ------------------------------------------------
    s = np.linspace(0, 1, 200)
    fx = 0.12 + 5.78 * s
    fy = 0.72 + 4.35 * s + 0.55 * np.sin(np.pi * s) - 0.25 * np.sin(2 * np.pi * s)
    free = np.c_[fx, fy]
    for a in range(0, 200 - 8, 8):
        polys.append(free[a:a + 9])
        classes.append("freeway")

    # ---- speeds ----------------------------------------------------------
    speeds, lengths = [], []
    for poly, k in zip(polys, classes):
        mid = poly.mean(axis=0)
        c = congestion_field(mid[0], mid[1])
        v0 = CLASS_V0[k]
        v = v0 * (1.0 - CLASS_SENS[k] * c * 0.95)
        v *= np.exp(rng.normal(0.0, 0.135))
        speeds.append(np.clip(v, 5.5, 88.0))
        lengths.append(np.hypot(*np.diff(poly, axis=0).T).sum())
    return polys, np.array(classes), np.array(speeds), np.array(lengths)


def main():
    set_pub_style()
    rng = np.random.default_rng(19)
    polys, classes, speeds, lengths = build_network(rng)

    cmap = mpl.colors.ListedColormap(LOS_COLORS)
    norm = mpl.colors.BoundaryNorm(LOS_EDGES, cmap.N)
    cols = cmap(norm(speeds))
    lws = np.array([CLASS_LW[k] for k in classes])

    # Explicit rects: the map is locked square (5.4 x 5.4 in) and the two
    # auxiliary panels together span 2.8 in -> 52 % of the map width.
    fig = plt.figure(figsize=(9.0, 6.2))
    axm = fig.add_axes([0.012, 0.070, 0.600, 0.871])
    axh = fig.add_axes([0.668, 0.578, 0.307, 0.347])
    axl = fig.add_axes([0.668, 0.100, 0.307, 0.292])

    # ---- grouping card behind the two auxiliary panels -------------------
    card = FancyBboxPatch((0.638, 0.052), 0.352, 0.905,
                          boxstyle="round,pad=0.0,rounding_size=0.012",
                          transform=fig.transFigure, facecolor=CARD,
                          edgecolor="#e4e2dc", lw=0.8, zorder=-5)
    fig.patches.append(card)

    # ---------------------------------------------------------------- map
    axm.set_facecolor("#fbfaf8")
    axm.add_patch(Rectangle((0, 0), EXTENT, EXTENT, facecolor="#fbfaf8",
                            edgecolor="none", zorder=0))
    # river
    ry = np.linspace(-0.2, EXTENT + 0.2, 200)
    rx = 4.62 + 0.55 * np.sin(ry / 1.55) + 0.16 * np.sin(ry / 0.6)
    axm.plot(rx, ry, color=WATER, lw=7.5, solid_capstyle="round", zorder=1,
             alpha=0.95)
    axm.plot(rx, ry, color="#b9c8d0", lw=0.6, zorder=1, alpha=0.9)
    # two parks
    for cx, cy, rr0 in [(1.30, 1.42, 0.52), (2.62, 4.86, 0.42)]:
        th = np.linspace(0, 2 * np.pi, 60)
        rad = rr0 * (1 + 0.16 * np.cos(3 * th + cx) + 0.09 * np.cos(5 * th))
        axm.add_patch(Polygon(np.c_[cx + rad * np.cos(th),
                                    cy + rad * np.sin(th)],
                              facecolor="#e7ede2", edgecolor="#d5ddcd",
                              lw=0.6, zorder=1))

    # casing for the worst links, then the coloured strokes
    bad = speeds < LOS_EDGES[1]
    if bad.any():
        axm.add_collection(LineCollection(
            [p for p, b in zip(polys, bad) if b],
            colors="#5e1f1c", linewidths=lws[bad] * 1.55 + 1.5,
            capstyle="round", zorder=2, alpha=0.30))
    axm.add_collection(LineCollection(
        polys, colors=cols, linewidths=np.where(bad, lws * 1.5, lws),
        capstyle="round", joinstyle="round", zorder=3))

    axm.set_xlim(-0.02, EXTENT + 0.02)
    axm.set_ylim(-0.02, EXTENT + 0.02)
    axm.set_aspect("equal")
    axm.set_xticks([]); axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.7); sp.set_edgecolor("#c9c6bf")

    # bottleneck call-outs
    for (bx, by), lab, (dx, dy, ha) in [
            ((4.55, 1.65), "interchange\nbottleneck", (-0.70, -0.75, "right")),
            ((1.55, 4.45), "CBD fringe\nqueueing", (0.62, -0.82, "left"))]:
        axm.annotate(lab, xy=(bx, by), xytext=(bx + dx, by + dy),
                     fontsize=6.8, color="#7a2f2a", fontweight="bold",
                     ha=ha, va="top", linespacing=1.2, zorder=7,
                     bbox=dict(boxstyle="round,pad=0.22", fc="white",
                               ec="#e0cfcb", lw=0.6, alpha=0.90),
                     arrowprops=dict(arrowstyle="-", lw=0.7, color="#a8433a",
                                     shrinkA=2, shrinkB=6,
                                     connectionstyle="arc3,rad=-0.25"))

    # scale bar + north arrow
    axm.add_patch(Rectangle((0.16, 0.09), 1.32, 0.42, facecolor="white",
                            edgecolor="none", alpha=0.86, zorder=7))
    axm.plot([0.30, 1.30], [0.20, 0.20], color=INK, lw=2.4,
             solid_capstyle="butt", zorder=8)
    axm.plot([0.30, 0.80], [0.20, 0.20], color="white", lw=1.1,
             solid_capstyle="butt", zorder=9)
    for xv, lab in [(0.30, "0"), (1.30, "1 km")]:
        axm.text(xv, 0.30, lab, fontsize=6.4, ha="center", va="bottom",
                 color=INK, zorder=9)
    axm.annotate("N", xy=(5.66, 5.62), xytext=(5.66, 5.20),
                 ha="center", fontsize=7.2, color=INK, fontweight="bold",
                 zorder=9,
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))

    # road-class width legend, inside the map
    hcl = [Line2D([], [], color="#6f7a80", lw=CLASS_LW[k], label=lab)
           for k, lab in [("freeway", "freeway"), ("arterial", "arterial"),
                          ("collector", "collector"), ("local", "local")]]
    leg = axm.legend(handles=hcl, loc="lower right",
                     bbox_to_anchor=(0.995, 0.012), fontsize=6.5,
                     title="road class", labelspacing=0.45, handlelength=2.0,
                     borderpad=0.45, frameon=True, framealpha=0.92,
                     edgecolor="#dcd9d2", facecolor="white")
    leg.get_title().set_fontsize(6.8)
    leg.get_title().set_fontweight("bold")
    leg.set_zorder(10)
    leg.get_frame().set_linewidth(0.6)
    leg.set_clip_on(False)

    # -------------------------------------------------- (b) speed histogram
    bins = np.arange(0, 91, 3.0)
    cnt, _ = np.histogram(speeds, bins=bins, weights=lengths)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    bcol = cmap(norm(ctr))
    axh.bar(ctr, cnt, width=2.7, color=bcol, edgecolor="white", linewidth=0.35,
            zorder=3)
    for e in LOS_EDGES[1:-1]:
        axh.axvline(e, color="#ffffff", lw=1.1, zorder=4)
        axh.axvline(e, color="#b9b5ac", lw=0.6, ls=(0, (2.4, 2.0)), zorder=4)
    vbar = float(np.average(speeds, weights=lengths))
    axh.axvline(vbar, color=INK, lw=1.0, zorder=5)
    axh.text(vbar + 1.6, axh.get_ylim()[1] * 0.94,
             f"network mean\n{vbar:.1f} km/h", fontsize=6.6, color=INK,
             fontweight="bold", ha="left", va="top", linespacing=1.25,
             zorder=6)
    axh.set_xlim(0, 90)
    axh.set_xlabel("peak-hour link speed  (km/h)", labelpad=2, fontsize=7.6)
    axh.set_ylabel("network length  (km)", labelpad=2, fontsize=7.6)
    axh.set_facecolor("white")
    axh.tick_params(labelsize=6.8, length=2.4)
    axh.grid(axis="y", color="#e9e7e2", lw=0.55, zorder=0)
    for sp in ("top", "right"):
        axh.spines[sp].set_visible(False)

    # -------------------------------------------------- (c) graded legend
    axl.set_facecolor("white")
    axl.set_xlim(0, 1); axl.set_ylim(0, 1)
    axl.set_xticks([]); axl.set_yticks([])
    for sp in axl.spines.values():
        sp.set_visible(False)
    idx = np.clip(np.digitize(speeds, LOS_EDGES[1:-1]), 0, 4)
    share = np.array([lengths[idx == i].sum() for i in range(5)])
    share = 100.0 * share / share.sum()
    ys = np.linspace(0.85, 0.22, 5)
    x0, xb, wmax = 0.035, 0.475, 0.36
    for i in range(5):
        y = ys[4 - i]
        axl.add_patch(Rectangle((x0, y - 0.045), 0.062, 0.090,
                                facecolor=LOS_COLORS[i], edgecolor="white",
                                lw=0.6, transform=axl.transAxes))
        axl.text(x0 + 0.085, y, LOS_NAMES[i], fontsize=6.9, va="center",
                 ha="left", color=INK, transform=axl.transAxes)
        axl.add_patch(Rectangle((xb, y - 0.038),
                                wmax * share[i] / share.max(), 0.076,
                                facecolor=LOS_COLORS[i], alpha=0.55,
                                edgecolor="none", transform=axl.transAxes))
        axl.text(xb + wmax * share[i] / share.max() + 0.018, y,
                 f"{share[i]:.0f}%", fontsize=6.9, va="center", ha="left",
                 color=INK, fontweight="bold", transform=axl.transAxes)
    axl.text(x0, 0.985, "level of service", fontsize=7.2,
             fontweight="bold", va="top", ha="left", color="#33474f",
             transform=axl.transAxes)
    axl.text(xb + 0.13, 0.985, "% of length", fontsize=7.2, va="top",
             ha="left", color="#6d7378", transform=axl.transAxes)
    axl.text(x0, 0.015, "%.0f km of link-level speed observations, "
             "17:30-18:30" % lengths.sum(), fontsize=6.3, color="#9a9a9a",
             va="bottom", ha="left", transform=axl.transAxes)

    # ---- panel tags / titles at one constant inch offset -----------------
    for ax, tag, ttl in [(axm, "(a)", "Peak-hour network speed"),
                         (axh, "(b)", "Speed distribution"),
                         (axl, "(c)", "Level-of-service breakdown")]:
        pos = ax.get_position()
        fig.text(pos.x0 - 0.02 / fig.get_figwidth(),
                 pos.y1 + 0.115 / fig.get_figheight(), tag, fontsize=10.5,
                 fontweight="bold", ha="left", va="baseline", color=INK)
        fig.text(pos.x0 + 0.34 / fig.get_figwidth(),
                 pos.y1 + 0.115 / fig.get_figheight(), ttl, fontsize=8.8,
                 fontweight="bold", ha="left", va="baseline", color="#33474f")

    savefig(fig, "204_network_speed_map.png")


if __name__ == "__main__":
    main()

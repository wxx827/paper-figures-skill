"""Origin-destination travel demand: log-scaled matrix plus a chord diagram.

Twelve traffic analysis zones are nested in four districts.  Panel (a) is
the full 12 x 12 daily OD matrix on a logarithmic colour scale, with
district ribbons on both margins and white rules separating the blocks.
Panel (b) folds the 28 heaviest inter-zonal relations into a chord
diagram: arc length is the zone's displayed throughput and ribbon width
is the flow, coloured by the origin district.  Both panels share one
district colour key.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from pubstyle import set_pub_style, savefig

# ---- shared transport palette --------------------------------------------
DISTRICTS = ["Core", "North", "East", "South"]
DCOLORS = ["#4a6f8a", "#6f9464", "#cf9a4e", "#a86a5e"]
ZONES = ["C1", "C2", "C3", "N1", "N2", "N3",
         "E1", "E2", "E3", "S1", "S2", "S3"]
GROUP = np.repeat(np.arange(4), 3)
SEQ = ["#f8f6f2", "#dfe6e8", "#b3c7d1", "#7ea3b8", "#4c7796", "#2a5471",
       "#14324a"]
INK = "#2b2b2b"

XY = np.array([
    [0.00, 0.10], [0.55, -0.35], [-0.50, -0.20],        # Core
    [-0.30, 2.05], [0.70, 2.35], [1.55, 1.70],          # North
    [2.45, 0.35], [2.60, -0.85], [1.70, -1.35],         # East
    [-1.10, -1.85], [-2.10, -1.05], [-2.35, 0.35],      # South
])


def bezier(p0, p1, p2, n=56):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def make_od(rng):
    n = len(ZONES)
    prod = np.array([7.6, 6.3, 5.7, 5.2, 4.6, 4.0,
                     5.1, 4.4, 3.8, 4.7, 4.1, 3.6])
    attr = np.array([6.6, 5.6, 5.0, 4.4, 4.0, 3.6,
                     5.2, 4.4, 3.8, 4.2, 3.7, 3.3])
    d = np.hypot(XY[:, None, 0] - XY[None, :, 0],
                 XY[:, None, 1] - XY[None, :, 1])
    np.fill_diagonal(d, 0.62)
    m = 320.0 * prod[:, None] * attr[None, :] / (0.55 + d) ** 1.85
    same = GROUP[:, None] == GROUP[None, :]
    m = m * np.where(same, 2.9, 1.0)
    m = m * rng.lognormal(0.0, 0.30, (n, n))
    np.fill_diagonal(m, np.diag(m) * 2.1)
    return np.round(m)


TAG_Y = 0.962          # one shared baseline for every panel tag


def panel_tag(fig, x0, tag, title):
    fig.text(x0, TAG_Y, tag, fontsize=10.5, fontweight="bold", ha="left",
             va="baseline", color=INK)
    fig.text(x0 + 0.36 / fig.get_figwidth(), TAG_Y, title, fontsize=8.8,
             fontweight="bold", ha="left", va="baseline", color="#33474f")


def main():
    set_pub_style()
    rng = np.random.default_rng(41)
    od = make_od(rng)
    n = len(ZONES)

    cmap = mpl.colors.LinearSegmentedColormap.from_list("odblue", SEQ)
    norm = mpl.colors.LogNorm(vmin=120, vmax=od.max())

    fig = plt.figure(figsize=(10.6, 5.9))
    axm = fig.add_axes([0.072, 0.145, 0.395, 0.710])
    cax = fig.add_axes([0.110, 0.062, 0.320, 0.026])
    axc = fig.add_axes([0.495, 0.090, 0.490, 0.830])

    # ------------------------------------------------------------- (a) matrix
    im = axm.imshow(od, cmap=cmap, norm=norm, origin="upper",
                    extent=(0, n, n, 0), zorder=2, interpolation="nearest")
    for b in (3, 6, 9):
        axm.axhline(b, color="white", lw=1.8, zorder=4)
        axm.axvline(b, color="white", lw=1.8, zorder=4)
    axm.plot([0, n], [0, n], color="white", lw=0.0)

    # district ribbons on both margins, names set inside the bands
    b0, b1 = 0.49, 0.95            # data units clear of the tick labels
    for g in range(4):
        i0 = 3 * g
        axm.add_patch(Rectangle((i0 + 0.06, -b1), 2.88, b1 - b0,
                                facecolor=DCOLORS[g], edgecolor="none",
                                clip_on=False, zorder=5))
        axm.add_patch(Rectangle((-b1, i0 + 0.06), b1 - b0, 2.88,
                                facecolor=DCOLORS[g], edgecolor="none",
                                clip_on=False, zorder=5))
        axm.text(i0 + 1.5, -(b0 + b1) / 2, DISTRICTS[g], ha="center",
                 va="center", fontsize=6.6, fontweight="bold", color="white",
                 clip_on=False, zorder=6)
        axm.text(-(b0 + b1) / 2, i0 + 1.5, DISTRICTS[g], ha="center",
                 va="center", rotation=90, fontsize=6.6, fontweight="bold",
                 color="white", clip_on=False, zorder=6)
    axm.text(6.0, -1.34, "destination zone  $\\rightarrow$", ha="center",
             va="bottom", fontsize=7.8, color="#1a1a1a", clip_on=False,
             zorder=6)
    axm.text(-1.34, 6.0, "$\\leftarrow$  origin zone", ha="center",
             va="top", rotation=90, fontsize=7.8, color="#1a1a1a",
             clip_on=False, zorder=6)

    # the three heaviest relations get an inline read-out
    flat = [(od[i, j], i, j) for i in range(n) for j in range(n)]
    flat.sort(reverse=True)
    for v, i, j in flat[:3]:
        axm.text(j + 0.5, i + 0.5, f"{v/1000:.1f}k", ha="center", va="center",
                 fontsize=6.0, color="white", fontweight="bold", zorder=6)

    axm.set_xticks(np.arange(n) + 0.5)
    axm.set_yticks(np.arange(n) + 0.5)
    axm.set_xticklabels(ZONES, fontsize=6.4)
    axm.set_yticklabels(ZONES, fontsize=6.4)
    axm.tick_params(length=0, pad=1.8)
    axm.xaxis.set_ticks_position("top")
    axm.set_xlim(0, n); axm.set_ylim(n, 0)
    axm.set_aspect("equal")
    for sp in axm.spines.values():
        sp.set_linewidth(0.7); sp.set_edgecolor("#b9b5ac")
        sp.set_visible(True)

    cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                      ticks=[200, 500, 1000, 2000, 5000, 10000, 20000])
    cb.ax.set_xticklabels(["200", "500", "1k", "2k", "5k", "10k", "20k"])
    cb.set_label("daily trips  (trips/day, log scale)", fontsize=7.2,
                 labelpad=2.5)
    cb.ax.tick_params(labelsize=6.6, length=2.2, width=0.7)
    cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor("#555555")

    # -------------------------------------------------------------- (b) chord
    # every zone contributes its three strongest outbound relations, so the
    # ring stays legible instead of collapsing onto the core
    keep = []
    for i in range(n):
        row = sorted(((od[i, j], i, j) for j in range(n) if j != i),
                     reverse=True)
        keep.extend(row[:3])
    keep = sorted(set(keep), reverse=True)
    tot = np.zeros(n)
    for v, i, j in keep:
        tot[i] += v; tot[j] += v
    live = tot > 0

    gap_in, gap_out = 0.016, 0.075
    gaps = np.array([gap_out if (k == 0 or GROUP[k] != GROUP[k - 1])
                     else gap_in for k in range(n)])
    usable = 2 * np.pi - gaps.sum()
    angles, cur = [], np.pi / 2 + 0.16
    for k in range(n):
        cur -= gaps[k]
        span = tot[k] / tot.sum() * usable if live[k] else 0.0
        angles.append((cur, cur - span))
        cur -= span

    R = 1.0
    for k in range(n):
        a0, a1 = angles[k]
        if a0 - a1 < 1e-6:
            continue
        th = np.linspace(a0, a1, 40)
        axc.plot(1.045 * np.cos(th), 1.045 * np.sin(th),
                 color=DCOLORS[GROUP[k]], lw=6.0, solid_capstyle="butt",
                 zorder=5)
        mid = (a0 + a1) / 2
        cx, sy = np.cos(mid), np.sin(mid)
        axc.text(1.115 * cx, 1.115 * sy, ZONES[k],
                 ha="left" if cx > 0.10 else "right" if cx < -0.10 else "center",
                 va="center" if abs(sy) < 0.88 else ("bottom" if sy > 0 else "top"),
                 fontsize=7.4, fontweight="bold", color=DCOLORS[GROUP[k]],
                 zorder=8)

    offs = [a[0] for a in angles]
    per = [(angles[k][0] - angles[k][1]) / max(tot[k], 1e-9) for k in range(n)]
    for v, i, j in sorted(keep, reverse=True):
        wi, wj = v * per[i], v * per[j]
        ai0, ai1 = offs[i], offs[i] - wi
        offs[i] -= wi
        aj0, aj1 = offs[j], offs[j] - wj
        offs[j] -= wj
        p = {k: R * np.array([np.cos(a), np.sin(a)]) for k, a in
             [("i0", ai0), ("i1", ai1), ("j0", aj0), ("j1", aj1)]}
        c1 = bezier(p["i1"], np.array([0.0, 0.0]), p["j0"])
        c2 = bezier(p["j1"], np.array([0.0, 0.0]), p["i0"])
        arc_j = np.linspace(aj0, aj1, 14)
        arc_i = np.linspace(ai1, ai0, 14)
        poly = np.vstack([c1,
                          R * np.column_stack([np.cos(arc_j), np.sin(arc_j)]),
                          c2,
                          R * np.column_stack([np.cos(arc_i), np.sin(arc_i)])])
        axc.add_patch(Polygon(poly, closed=True, facecolor=DCOLORS[GROUP[i]],
                              alpha=0.40, lw=0.35, edgecolor="white",
                              zorder=2))

    # outer district band
    for g in range(4):
        ids = [k for k in range(n) if GROUP[k] == g and live[k]]
        if not ids:
            continue
        a0 = angles[ids[0]][0] + gap_in * 0.4
        a1 = angles[ids[-1]][1] - gap_in * 0.4
        th = np.linspace(a0, a1, 60)
        axc.plot(1.245 * np.cos(th), 1.245 * np.sin(th), color=DCOLORS[g],
                 lw=2.4, solid_capstyle="butt", alpha=0.80, zorder=5)
        mid = 0.5 * (a0 + a1)
        cx, sy = np.cos(mid), np.sin(mid)
        axc.text(1.345 * cx, 1.345 * sy, DISTRICTS[g],
                 ha="left" if cx > 0.12 else "right" if cx < -0.12 else "center",
                 va="center" if abs(sy) < 0.90 else ("bottom" if sy > 0 else "top"),
                 fontsize=8.2, fontweight="bold", color=DCOLORS[g], zorder=8)

    axc.set_aspect("equal")
    axc.axis("off")
    axc.set_xlim(-1.66, 1.66)
    axc.set_ylim(-1.52, 1.52)

    panel_tag(fig, 0.058, "(a)", "Zone-to-zone demand matrix")
    panel_tag(fig, 0.520, "(b)", "Dominant OD corridors")

    # ---- shared district key --------------------------------------------
    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none")
               for c in DCOLORS]
    leg = fig.legend(handles, DISTRICTS, loc="lower center",
                     bbox_to_anchor=(0.74, 0.008), ncol=4, frameon=False,
                     fontsize=7.4, handlelength=1.1, handleheight=0.85,
                     columnspacing=1.4, title="district")
    leg.get_title().set_fontsize(7.4)
    leg.get_title().set_fontweight("bold")
    leg.set_clip_on(False)

    savefig(fig, "205_od_matrix_chord.png")


if __name__ == "__main__":
    main()

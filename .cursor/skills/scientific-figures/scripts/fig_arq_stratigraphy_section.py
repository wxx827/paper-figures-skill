"""Trench section drawn at 1:1, with the finds and the dates read off it.

The section is the primary record and the two side panels hang off its depth
axis, so a reader never has to translate between three different vertical
scales.  Layer boundaries are smoothed noise rather than straight lines
because a real interface is not a ruled edge, and the two negative features -
a bell-shaped pit and a narrow intrusive drain - are drawn on top of the
deposits with dashed cuts, which is how a section conveys that something was
dug through something else.

Nothing in the side panels is asserted independently of the drawing.  Find
densities come from a raster classification of the same polygons the section
is filled with, minus the area lost to the two intrusions, so a thin layer
with a handful of sherds cannot pretend to be rich.  The age-depth line is a
least-squares fit through the four stratified determinations only; the date
from the pit fill is plotted as an open marker and excluded, because a date
from an intrusive feature does not belong on the accumulation curve.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.path import Path
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter1d

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, halo_text, panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HAIR = "#cccccc"

FW, FH = 10.4, 6.4
# Section box is 5.50 x 3.00 in and the drawn window is 5.50 x 3.00 m, so the
# drawing is 1:1 and set_aspect("equal") cannot shrink the axes box.
X0, X1 = -0.15, 5.35
Y0, Y1 = -0.42, 2.58

# (code, name, mean thickness m, soil colour, hatch)
LAYERS = [
    ("101", "topsoil",        0.26, "#8d8069", "..."),
    ("102", "colluvium",      0.32, "#c3b69c", "///"),
    ("103", "occupation I",   0.30, "#a98c63", "xxx"),
    ("104", "ash lens",       0.15, "#c9c2b6", "---"),
    ("105", "occupation II",  0.36, "#9d8055", "ooo"),
    ("106", "sterile clay",   0.42, "#9aa38d", "|||"),
    ("107", "natural gravel", 0.60, "#8b8b84", "\\\\\\"),
]
PIT_FILL, PIT_EDGE = "#bfae93", "#5c4a33"
CUT_FILL = "#cfc7b8"

FIND_CLASSES = [
    ("pottery",  "o", "#b23a48"),
    ("bone",     "^", "#3b5b92"),
    ("lithic",   "D", "#2e7d5b"),
    ("charcoal", "*", "#3a3a3a"),
]
# finds per unit, in LAYERS order; the pit fill is counted separately
FIND_N = [(9, 3, 2, 1), (6, 2, 2, 1), (17, 7, 5, 4), (4, 2, 1, 5),
          (21, 9, 6, 6), (2, 1, 1, 0), (0, 0, 0, 0)]
PIT_N = (8, 4, 2, 3)

# (label, depth m, median cal BC as a negative year, 95 % half range, layer)
DATES = [
    ("C14-1", 0.55, -512.0, 55.0, "103"),
    ("C14-2", 0.79, -627.0, 48.0, "104"),
    ("C14-3", 1.06, -688.0, 62.0, "105"),
    ("C14-4", 1.42, -851.0, 70.0, "106"),
]
PIT_DATE = ("C14-5", 1.02, -372.0, 58.0)

XG = np.linspace(X0, X1, 420)


def rough(rng, amp, wave, smooth=9.0):
    """Hand-drawn looking undulation: smoothed noise plus one long wave."""
    n = gaussian_filter1d(rng.normal(0.0, 1.0, XG.size), smooth)
    n /= max(np.abs(n).max(), 1e-9)
    return amp * n + 0.45 * amp * np.sin(2 * np.pi * XG / wave +
                                         rng.uniform(0.0, 6.28))


def build_boundaries(rng):
    b = 0.11 + rough(rng, 0.030, 3.1)
    bounds = [b]
    for _, _, t, _, _ in LAYERS:
        step = t * (1.0 + 0.22 * rough(rng, 1.0, rng.uniform(1.8, 4.4)))
        b = b + np.clip(step, 0.06, None)
        bounds.append(b)
    bounds[-1] = np.maximum(bounds[-1], Y1 + 0.05)   # basal unit runs off-frame
    return bounds


def pit_polygon():
    """Bell-shaped pit: wide mouth, tapering to a rounded base."""
    t = np.linspace(0.0, 1.0, 90)
    top, bot, cx, hw = 0.40, 1.62, 3.58, 0.62
    y = top + (bot - top) * t
    w = hw * np.sqrt(np.clip(1.0 - t ** 2.4, 0.0, 1.0)) * (1.0 + 0.05 *
                                                           np.sin(9 * t))
    xs = np.concatenate([cx - w, (cx + w)[::-1]])
    ys = np.concatenate([y, y[::-1]])
    return np.column_stack([xs, ys])


def cut_polygon():
    """Narrow intrusive drain, cut from the modern surface."""
    y = np.linspace(0.04, 0.98, 60)
    cx = 1.16 + 0.06 * np.sin(2 * np.pi * y / 1.5)
    w = 0.135 * (1.0 - 0.35 * (y / 1.02) ** 2)
    xs = np.concatenate([cx - w, (cx + w)[::-1]])
    ys = np.concatenate([y, y[::-1]])
    return np.column_stack([xs, ys])


def main():
    set_pub_style()
    rng = np.random.default_rng(20260813)

    bounds = build_boundaries(rng)
    pit, cut = pit_polygon(), cut_polygon()
    pit_path, cut_path = Path(pit), Path(cut)

    def intruded(pts):
        return pit_path.contains_points(pts) | cut_path.contains_points(pts)

    # ---- areas: raster the same polygons the section is filled with -------
    gx = np.linspace(0.0, 5.2, 300)
    gy = np.linspace(0.02, Y1, 240)
    GX, GY = np.meshgrid(gx, gy)
    cell = (gx[1] - gx[0]) * (gy[1] - gy[0])
    free = ~intruded(np.column_stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    area = []
    for k in range(len(LAYERS)):
        up = np.interp(gx, XG, bounds[k])[None, :]
        lo = np.interp(gx, XG, bounds[k + 1])[None, :]
        area.append(float(np.count_nonzero((GY >= up) & (GY < lo) & free) * cell))
    area = np.array(area)
    pit_area = float(np.count_nonzero(
        pit_path.contains_points(np.column_stack([GX.ravel(), GY.ravel()]))) *
        cell)

    # ---- finds: rejection-sample inside each layer, never in an intrusion -
    finds = {c: [[], []] for c, _, _ in FIND_CLASSES}
    for k, counts in enumerate(FIND_N):
        up, lo = bounds[k], bounds[k + 1]
        for (cls, _, _), n in zip(FIND_CLASSES, counts):
            got = 0
            while got < n:
                x = rng.uniform(0.08, 5.12)
                d0 = np.interp(x, XG, up) + 0.02
                d1 = min(np.interp(x, XG, lo) - 0.02, Y1 - 0.03)
                if d1 <= d0:
                    continue
                y = rng.uniform(d0, d1)
                if intruded(np.array([[x, y]]))[0]:
                    continue
                finds[cls][0].append(x)
                finds[cls][1].append(y)
                got += 1
    pit_pts = {c: [[], []] for c, _, _ in FIND_CLASSES}
    for (cls, _, _), n in zip(FIND_CLASSES, PIT_N):
        got = 0
        while got < n:
            x, y = rng.uniform(2.95, 4.2), rng.uniform(0.5, 1.55)
            if not pit_path.contains_points(np.array([[x, y]]))[0]:
                continue
            pit_pts[cls][0].append(x)
            pit_pts[cls][1].append(y)
            got += 1

    dens = np.array([[n / a for n in cnt] for cnt, a in zip(FIND_N, area)])
    pit_dens = sum(PIT_N) / pit_area

    # ---- age-depth model from the stratified dates only -------------------
    dd = np.array([d[1] for d in DATES])
    yy = np.array([d[2] for d in DATES])
    slope, icept = np.polyfit(dd, yy, 1)
    resid = yy - (slope * dd + icept)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    rate = 1.0e4 / abs(slope)          # cm of deposit per century

    fig = plt.figure(figsize=(FW, FH))
    ax_s = axes_inches(fig, (0.78, 1.95, 5.50, 3.00))
    ax_d = axes_inches(fig, (6.65, 1.95, 1.45, 3.00))
    ax_a = axes_inches(fig, (8.32, 1.95, 1.62, 3.00))

    # ---- (a) the section --------------------------------------------------
    for k, (code, nm, _, col, hat) in enumerate(LAYERS):
        ax_s.fill_between(XG, bounds[k], bounds[k + 1], facecolor=col,
                          edgecolor="#6f6656", hatch=hat, linewidth=0.0,
                          zorder=2)
        ax_s.plot(XG, bounds[k], color="#4f4739", lw=0.9, zorder=4)
    ax_s.plot(XG, bounds[0], color="#3a3a3a", lw=1.2, zorder=5)

    ax_s.add_patch(Polygon(cut, closed=True, facecolor=CUT_FILL,
                           edgecolor=PIT_EDGE, lw=1.0, ls=(0, (3.5, 1.8)),
                           hatch="..", zorder=6))
    ax_s.add_patch(Polygon(pit, closed=True, facecolor=PIT_FILL,
                           edgecolor=PIT_EDGE, lw=1.2, ls=(0, (3.5, 1.8)),
                           hatch="..", zorder=6))

    for cls, mk, col in FIND_CLASSES:
        ax_s.plot(finds[cls][0], finds[cls][1], ls="none", marker=mk,
                  ms=3.1 if mk != "*" else 4.4, color=col, mec="white",
                  mew=0.35, zorder=8)
        ax_s.plot(pit_pts[cls][0], pit_pts[cls][1], ls="none", marker=mk,
                  ms=3.1 if mk != "*" else 4.4, color=col, mec="white",
                  mew=0.35, zorder=8)

    col_x = 4.72
    col_y = np.arange(0.18, 2.35, 0.20)
    ax_s.plot(np.full_like(col_y, col_x), col_y, ls="none", marker="s", ms=2.6,
              mfc="white", mec="#3a3a3a", mew=0.7, zorder=9)
    halo_text(ax_s, 5.10, 2.44, "bulk sampling column, 0.20 m steps",
              ha="right", va="bottom", fontsize=6.2, color=INK, halo="white",
              halo_lw=2.0)

    for lab, dep, med, _, _ in DATES:
        ax_s.plot([0.30], [dep], marker="*", ms=7.0, color="#c9a227",
                  mec="#6b5410", mew=0.5, zorder=9)
        halo_text(ax_s, 0.42, dep, lab, ha="left", va="center", fontsize=6.2,
                  color=INK, halo="white", halo_lw=2.0)
    ax_s.plot([3.58], [PIT_DATE[1]], marker="*", ms=7.0, color="white",
              mec="#6b5410", mew=0.9, zorder=9)
    halo_text(ax_s, 3.70, PIT_DATE[1], PIT_DATE[0], ha="left", va="center",
              fontsize=6.2, color=INK, halo="white", halo_lw=2.0)

    for k, (code, nm, _, _, _) in enumerate(LAYERS):
        # Thin lens goes on the right, clear of the pit; the rest sit in the
        # band between the drain and the pit mouth.
        x_lab = 4.34 if code == "104" else 2.05
        x_mid = x_lab + 0.45
        mid = 0.5 * (np.interp(x_mid, XG, bounds[k]) +
                     np.interp(x_mid, XG, bounds[k + 1]))
        halo_text(ax_s, x_lab, mid, f"SU {code}  {nm}", ha="left", va="center",
                  fontsize=6.6, color=INK, halo="white", halo_lw=2.2)
    halo_text(ax_s, 3.58, 0.30, "SU 108 pit fill", ha="center", va="bottom",
              fontsize=6.6, color=INK, halo="white", halo_lw=2.2)
    halo_text(ax_s, 4.28, 1.62, "cut [109] truncates SU 102-106", ha="right",
              va="top", fontsize=6.2, color=PIT_EDGE, halo="white",
              halo_lw=2.0)
    halo_text(ax_s, 1.34, 0.86, "SU 110 field drain,\nintrusive", ha="left",
              va="top", fontsize=6.2, color=PIT_EDGE, halo="white",
              halo_lw=2.0, linespacing=1.2)

    ax_s.axhline(0.0, color=RULE, lw=0.8, ls=(0, (5, 2.5)), zorder=7)
    ax_s.text(5.33, -0.03, "site datum 0.00 m", ha="right", va="bottom",
              fontsize=6.4, color=SUB)
    ax_s.plot([0.20, 1.20], [2.42, 2.42], color="#2b2b2b", lw=2.4,
              solid_capstyle="butt", zorder=9)
    ax_s.plot([0.70, 1.20], [2.42, 2.42], color="white", lw=1.6,
              solid_capstyle="butt", zorder=10)
    halo_text(ax_s, 0.70, 2.36, "1 m (1:1, no vertical exaggeration)",
              ha="center", va="bottom", fontsize=6.2, color=INK, halo="white",
              halo_lw=2.0)

    ax_s.set_xlim(X0, X1)
    ax_s.set_ylim(Y1, Y0)
    ax_s.set_aspect("equal")
    ax_s.set_xlabel("distance along section, west to east (m)", fontsize=8.0)
    ax_s.set_ylabel("depth below datum (m, positive down)", fontsize=8.0)
    ax_s.set_yticks(np.arange(0.0, 2.51, 0.5))
    ax_s.tick_params(labelsize=7.0, length=2.6)
    ax_s.set_title("East section of Trench 4, 5.2 m exposed",
                   fontsize=9.0, fontweight="bold", loc="left")
    ax_s.legend(handles=[Line2D([0], [0], ls="none", marker=mk,
                                ms=3.4 if mk != "*" else 4.8, color=c,
                                mec="white", mew=0.35, label=n)
                         for n, mk, c in FIND_CLASSES] +
                [Line2D([0], [0], ls="none", marker="s", ms=3.0, mfc="white",
                        mec="#3a3a3a", label="bulk sample"),
                 Line2D([0], [0], ls="none", marker="*", ms=6.0,
                        color="#c9a227", mec="#6b5410", label="AMS sample")],
                loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=6,
                fontsize=6.4, frameon=False, handletextpad=0.35,
                columnspacing=1.0, borderpad=0.1)

    # ---- (b) find density, sharing the depth axis -------------------------
    mids = np.array([0.5 * (bounds[k].mean() + bounds[k + 1].mean())
                     for k in range(len(LAYERS))])
    hts = np.array([(bounds[k + 1] - bounds[k]).mean() * 0.74
                    for k in range(len(LAYERS))])
    hts[-1] = min(hts[-1], 0.42)
    left = np.zeros(len(LAYERS))
    for j, (cls, _, col) in enumerate(FIND_CLASSES):
        ax_d.barh(mids, dens[:, j], height=hts, left=left, color=col,
                  edgecolor="white", linewidth=0.4, zorder=3)
        left = left + dens[:, j]
    for k, (code, _, _, _, _) in enumerate(LAYERS):
        if left[k] > 0:
            ax_d.text(left[k] + 0.25, mids[k], f"n={sum(FIND_N[k])}",
                      ha="left", va="center", fontsize=6.0, color=SUB)
    ax_d.set_ylim(Y1, Y0)
    ax_d.set_xlim(0.0, left.max() * 1.34)
    ax_d.set_yticks(np.arange(0.0, 2.51, 0.5))
    ax_d.set_yticklabels([])
    ax_d.tick_params(labelsize=7.0, length=2.4)
    ax_d.set_xlabel("finds per m$^2$ of section", fontsize=8.0)
    ax_d.set_title("Find density", fontsize=8.5, fontweight="bold", loc="left")
    ax_d.grid(axis="x", lw=0.5, color=HAIR, alpha=0.8)
    ax_d.set_axisbelow(True)
    for s in ("top", "right"):
        ax_d.spines[s].set_visible(False)

    # ---- (c) age-depth model ---------------------------------------------
    dgrid = np.linspace(0.42, 1.58, 60)
    fit = slope * dgrid + icept
    ax_a.fill_betweenx(dgrid, fit - rms, fit + rms, color="#3b5b92",
                       alpha=0.16, lw=0, zorder=2)
    ax_a.plot(fit, dgrid, color="#3b5b92", lw=1.2, zorder=3)
    for lab, dep, med, half, _ in DATES:
        ax_a.plot([med - half, med + half], [dep, dep], color=RULE, lw=1.1,
                  solid_capstyle="butt", zorder=4)
        ax_a.plot([med], [dep], marker="o", ms=3.6, color="#c9a227",
                  mec="#6b5410", mew=0.6, zorder=5)
        ax_a.text(med, dep - 0.07, lab, ha="center", va="bottom",
                  fontsize=6.0, color=SUB)
    lab, dep, med, half = PIT_DATE
    ax_a.plot([med - half, med + half], [dep, dep], color=PIT_EDGE, lw=1.1,
              ls=(0, (2.5, 1.5)), zorder=4)
    ax_a.plot([med], [dep], marker="o", ms=3.6, mfc="white", mec=PIT_EDGE,
              mew=0.9, zorder=5)
    ax_a.text(med, dep - 0.07, lab, ha="center", va="bottom", fontsize=6.0,
              color=PIT_EDGE)
    ax_a.text(0.03, 0.985, "open marker: intrusive,\nnot used in the fit",
              transform=ax_a.transAxes, ha="left", va="top", fontsize=6.2,
              color=PIT_EDGE, linespacing=1.3)
    ax_a.set_ylim(Y1, Y0)
    ax_a.set_yticks(np.arange(0.0, 2.51, 0.5))
    ax_a.set_yticklabels([])
    ax_a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{-v:.0f}"))
    ax_a.tick_params(labelsize=7.0, length=2.4)
    ax_a.set_xlabel("modelled age (cal BC)", fontsize=8.0)
    ax_a.set_title("Age-depth", fontsize=8.5, fontweight="bold", loc="left")
    ax_a.grid(axis="x", lw=0.5, color=HAIR, alpha=0.8)
    ax_a.set_axisbelow(True)
    for s in ("top", "right"):
        ax_a.spines[s].set_visible(False)
    ax_a.text(0.97, 0.035, f"{rate:.0f} cm per century\nRMS {rms:.0f} yr",
              transform=ax_a.transAxes, ha="right", va="bottom", fontsize=6.4,
              color=INK, linespacing=1.3)

    fig.legend(handles=[Patch(facecolor=c, edgecolor="#6f6656", hatch=h,
                              label=f"SU {code} {nm}")
                        for code, nm, _, c, h in LAYERS] +
               [Patch(facecolor=PIT_FILL, edgecolor=PIT_EDGE, hatch="..",
                      ls=(0, (3, 2)), label="SU 108 pit fill / cut [109]")],
               loc="lower center", bbox_to_anchor=(0.5, 0.118), ncol=8,
               fontsize=6.6, frameon=False, handlelength=1.6, handleheight=1.1,
               columnspacing=1.1, handletextpad=0.45)

    panel_labels_inches(fig, [ax_s, ax_d, ax_a], dx=-0.34, dy=0.16)

    for k, (code, nm, _, _, _) in enumerate(LAYERS):
        print(f"SU {code} {nm:15s} area {area[k]:5.2f} m2  "
              f"n={sum(FIND_N[k]):3d}  {dens[k].sum():5.2f} per m2")
    print(f"pit fill area {pit_area:.2f} m2  n={sum(PIT_N)}  "
          f"{pit_dens:.2f} per m2")
    print(f"age-depth slope {slope:.1f} yr/m -> {rate:.1f} cm/century, "
          f"RMS {rms:.1f} yr")
    savefig(fig, "889_arq_stratigraphy_section.png")


if __name__ == "__main__":
    main()

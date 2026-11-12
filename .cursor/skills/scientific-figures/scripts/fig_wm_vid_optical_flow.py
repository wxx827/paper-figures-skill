"""Figure 1165 - Optical flow between two dash-camera frames.

ONE array drives the whole page.  ``flow_field`` returns a single ``(H, W, 2)``
displacement field; frame t+1 is frame t *warped by that field*, so the pair of
photographs, the colour-coded map, the quiver overlay, the magnitude histogram
and the direction rose are five projections of the same numbers.  Change the
field and every panel, every printed statistic and the wheel legend move
together.

Colour encoding.  The classic optical-flow wheel sweeps the full hue circle and
is therefore half warm, which this batch does not allow (warm area budget 8 %).
The house encoding here is a *cool cyclic* two-channel scheme:

    hue        <- direction, cycling through four cool anchors
                  (teal-cyan -> indigo -> violet -> sea-green -> teal-cyan)
    lightness  <- magnitude, white at zero, fully saturated at the 99th pct

All four anchors sit between hue 160 and 270 degrees, and RGB interpolation
between neighbours never leaves the convex hull of two cool colours, so the map
stays inside the budget while remaining genuinely periodic in direction.  The
disc in (g) is drawn with the same function that colours (c).
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from scipy.ndimage import gaussian_filter, map_coordinates

from pubstyle import set_pub_style, savefig, add_panel_label
from figkit import image_axes
from _wm_scenes import build, grey

INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"
# Card grey is deliberately below 0.92 relative luminance, not the 0.94 house
# value: at 0.94 a backing card flips between "all ink" and "all white"
# depending on which threshold the audit uses.  Below 0.92 it always counts.
HAIR, CARD, BAND = "#d5d5d5", "#e6e9ec", "#dfe9f2"
DEEP = "#134a76"
SEED = 1570  # original id; figure number is 1165
H, W = 150, 224

# cool cyclic direction anchors at 0 / 90 / 180 / 270 degrees, then wrapping
DIR_ANCHORS = ["#2a8fa8", "#3a5fa8", "#7a56a8", "#3f9a7a", "#2a8fa8"]
DIR_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "cool_cyclic_dir", DIR_ANCHORS, N=512)


def flow_field():
    """Forward camera translation plus one independently moving vehicle."""
    rng = np.random.default_rng(SEED)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    # radial expansion about the focus of expansion, scaled by ground distance
    fx, fy = 0.505 * W, 0.462 * H
    dx, dy = xx - fx, yy - fy
    r = np.hypot(dx, dy) + 1e-6
    depth = 1.0 / (0.16 + 2.6 * np.clip((fy - yy) / H + 0.55, 0.04, 1.0))
    ex = 0.085 * dx / r * r ** 0.86 * depth * 0.62
    ey = 0.085 * dy / r * r ** 0.86 * depth * 0.62
    # a slight yaw: everything slides left as the car turns into the bend
    ex = ex - 0.55
    # a vehicle in the right lane pulling away: its image motion points back
    # towards the focus of expansion, i.e. against the radial field
    obj = np.exp(-(((xx - 0.720 * W) / (0.082 * W)) ** 2
                   + ((yy - 0.715 * H) / (0.072 * H)) ** 2))
    ex = ex - 9.2 * obj
    ey = ey - 5.4 * obj
    u = gaussian_filter(ex, 2.2) + 0.10 * gaussian_filter(
        rng.standard_normal((H, W)), 6.0)
    v = gaussian_filter(ey, 2.2) + 0.10 * gaussian_filter(
        rng.standard_normal((H, W)), 6.0)
    return np.dstack([u, v]), (fx, fy), (0.720 * W, 0.715 * H)


def warp(img, flow):
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    return np.clip(np.dstack([
        map_coordinates(img[..., c], [yy + flow[..., 1], xx + flow[..., 0]],
                        order=1, mode="nearest") for c in range(3)]), 0, 1)


def flow_rgb(u, v, msat):
    """Cool cyclic hue for direction, lightness for magnitude."""
    ang = (np.arctan2(v, u) / (2 * np.pi)) % 1.0
    mag = np.clip(np.hypot(u, v) / max(msat, 1e-9), 0, 1)
    base = DIR_CMAP(ang)[..., :3]
    out = 1.0 - (1.0 - base) * mag[..., None] ** 0.58
    return np.clip(out * (1.0 - 0.16 * mag[..., None]), 0, 1)


def main():
    set_pub_style(base_font=8.5)
    flow, foe, car = flow_field()
    frame0 = build("road", seed=SEED, shape=(H, W), t=0.0)["rgb"]
    frame1 = warp(frame0, -flow)                 # pull-back warp: I1(x) = I0(x-w)

    u, v = flow[..., 0], flow[..., 1]
    mag = np.hypot(u, v)
    ang = np.degrees(np.arctan2(v, u)) % 360.0
    msat = float(np.percentile(mag, 90.0))      # colour code saturates here
    med = float(np.median(mag))
    p95 = float(np.percentile(mag, 95.0))
    fast = float((mag > p95).mean())

    # Pixels whose flow points back towards the focus of expansion cannot be
    # explained by ego-motion.  The same boolean mask is contoured in (c) and
    # in (d), so the two panels are annotating one region, not two guesses.
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    dxr, dyr = xx - foe[0], yy - foe[1]
    radial = (u * dxr + v * dyr) / (np.hypot(dxr, dyr) + 1e-6)
    against = (radial < 0.0) & (mag > med)
    a_frac = float(against.mean())

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[0.66, 1.28],
                          left=0.045, right=0.982, top=0.855, bottom=0.118,
                          wspace=0.16, hspace=0.30)

    # ---- (a) (b) the two input frames -------------------------------------
    for k, (im, ttl) in enumerate(((frame0, "Frame $I_t$"),
                                   (frame1, "Frame $I_{t+1}$"))):
        ax = fig.add_subplot(gs[0, k])
        image_axes(ax, color="#9a9a9a", lw=0.6)
        ax.imshow(im, interpolation="bilinear")
        ax.set_title(ttl, fontsize=8.6, pad=3.0, color=INK)
        add_panel_label(ax, f"({'ab'[k]})", x=-0.045, y=1.00, size=9.5)

    # ---- (c) colour-coded flow --------------------------------------------
    axc = fig.add_subplot(gs[0, 2])
    image_axes(axc, color="#9a9a9a", lw=0.6)
    axc.imshow(flow_rgb(u, v, msat), interpolation="bilinear")
    axc.contour(against.astype(float), levels=[0.5], colors=["#0b3d63"],
                linewidths=0.9, linestyles=[(0, (3, 2))], zorder=5)
    axc.plot(*foe, marker="+", ms=7, mew=1.1, color="#20374f", zorder=5)
    axc.text(foe[0] + 6, foe[1] - 5, "focus of\nexpansion", fontsize=6.6,
             color="#20374f", ha="left", va="bottom", linespacing=1.35,
             zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                                 alpha=0.92))
    axc.set_title("Flow $\\mathbf{w}$, cool cyclic code", fontsize=8.6,
                  pad=3.0, color=INK)
    add_panel_label(axc, "(c)", x=-0.045, y=1.00, size=9.5)

    # ---- (d) quiver over a desaturated frame ------------------------------
    axd = fig.add_subplot(gs[0, 3])
    image_axes(axd, color="#9a9a9a", lw=0.6)
    axd.imshow(grey(frame0, 0.36, 0.62), cmap="gray", vmin=0, vmax=1,
               interpolation="bilinear")
    step = 13
    ys, xs = np.mgrid[7:H:step, 7:W:step]
    qu, qv = u[ys, xs], v[ys, xs]
    qcol = DIR_CMAP(((np.arctan2(qv, qu) / (2 * np.pi)) % 1.0).ravel())
    axd.quiver(xs, ys, qu, qv, angles="xy", scale_units="xy", scale=0.62,
               width=0.0062, headwidth=3.4, headlength=3.8, headaxislength=3.2,
               color=qcol, alpha=0.95, zorder=4)
    axd.contour(against.astype(float), levels=[0.5], colors=["#0b3d63"],
                linewidths=1.0, linestyles=[(0, (3, 2))], zorder=6)
    axd.annotate(f"flows against the\nexpansion: {100 * a_frac:.1f} % of px",
                 xy=(car[0] - 21.0, car[1] - 12.0),
                 xytext=(0.035, 0.155), textcoords="axes fraction",
                 fontsize=6.6, color="#0b3d63", ha="left", va="center",
                 linespacing=1.35, zorder=7,
                 bbox=dict(boxstyle="round,pad=0.22", fc="white",
                           ec="#b9c9d8", lw=0.55, alpha=0.94),
                 arrowprops=dict(arrowstyle="-", lw=0.7, color="#0b3d63",
                                 shrinkA=3, shrinkB=2))
    axd.set_title(f"Vectors, 1 of every {step} px", fontsize=8.6, pad=3.0,
                  color=INK)
    add_panel_label(axd, "(d)", x=-0.045, y=1.00, size=9.5)
    axd.set_xlim(-0.5, W - 0.5)
    axd.set_ylim(H - 0.5, -0.5)

    # ---- (e) magnitude histogram ------------------------------------------
    axe = fig.add_subplot(gs[1, :2])
    axe.set_facecolor(CARD)
    edges = np.linspace(0.0, float(mag.max()), 49)
    cnt, _ = np.histogram(mag.ravel(), bins=edges)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    frac = 100.0 * cnt / mag.size
    axe.bar(ctr, frac, width=0.92 * (edges[1] - edges[0]), color="#5f9dc6",
            edgecolor="white", linewidth=0.35, zorder=3)
    axe.bar(ctr[ctr > p95], frac[ctr > p95],
            width=0.92 * (edges[1] - edges[0]), color=DEEP,
            edgecolor="white", linewidth=0.35, zorder=4)
    ylo, ytop = 4.0e-3, 42.0
    axe.add_patch(Rectangle((p95, ylo), float(mag.max()) - p95, ytop,
                            facecolor=BAND, edgecolor="none", zorder=2))
    axe.plot([msat, msat], [ylo, 0.55], color="#2e74a8", lw=0.9,
             ls=(0, (2, 2)), zorder=5)
    axe.text(msat + 0.14, 0.62, f"colour code\nsaturates {msat:.2f} px",
             fontsize=6.4, color="#2e74a8", ha="left", va="bottom",
             linespacing=1.4, zorder=6,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.4,
                       alpha=0.92))
    axe.plot([med, med], [ylo, 12.0], color="#0b3d63", lw=1.0,
             ls=(0, (4, 2)), zorder=5)
    axe.text(med + 0.16, 13.5, f"median {med:.2f} px", fontsize=7.0,
             color="#0b3d63", ha="left", va="bottom", zorder=6,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.4,
                       alpha=0.92))
    axe.annotate(f"shaded: the fastest {100 * fast:.0f} % of pixels "
                 f"(above {p95:.2f} px).\nThey are near-field road surface at "
                 f"the bottom edge,\nnot the moving vehicle - that one is a "
                 f"direction outlier",
                 xy=(0.5 * (p95 + float(mag.max())), 0.07),
                 xytext=(0.985, 0.845), textcoords="axes fraction",
                 fontsize=6.9, color=DEEP, ha="right", va="center",
                 linespacing=1.5, zorder=7,
                 bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#b9c9d8",
                           lw=0.6, alpha=0.94),
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=DEEP,
                                 shrinkA=3, shrinkB=2))
    axe.set_yscale("log")
    axe.set_xlim(0.0, float(mag.max()) * 1.02)
    axe.set_ylim(ylo, ytop)
    axe.set_xlabel("flow magnitude $\\|\\mathbf{w}\\|$  (px / frame)",
                   fontsize=8.0)
    axe.set_ylabel("share of pixels (%, log)", fontsize=8.0)
    axe.tick_params(labelsize=7.0, length=2.4)
    axe.grid(True, axis="y", lw=0.45, color="#e2e2e2")
    axe.set_axisbelow(True)
    axe.set_title("Magnitude distribution over all "
                  f"{H * W / 1000:.1f} k pixels of (c)", fontsize=8.6,
                  pad=3.0, color=INK)
    add_panel_label(axe, "(e)", x=-0.048, y=1.00, size=9.5)

    # ---- (f) direction rose ------------------------------------------------
    axf = fig.add_subplot(gs[1, 2], projection="polar")
    nb = 24
    bedge = np.linspace(0, 2 * np.pi, nb + 1)
    wsum, _ = np.histogram(np.radians(ang).ravel(), bins=bedge,
                           weights=mag.ravel())
    wsum = wsum / wsum.sum() * 100.0
    bctr = 0.5 * (bedge[:-1] + bedge[1:])
    axf.bar(bctr, wsum, width=2 * np.pi / nb * 0.92, bottom=0.0,
            color=DIR_CMAP(bctr / (2 * np.pi)), edgecolor="white",
            linewidth=0.4, zorder=3)
    axf.set_theta_zero_location("E")
    axf.set_theta_direction(-1)
    axf.set_thetagrids([0, 90, 180, 270],
                       ["right", "down", "left", "up"], fontsize=6.4)
    axf.set_rlabel_position(-42)
    step_r = 4.0 if wsum.max() > 9.0 else 2.0
    axf.set_rticks(np.arange(step_r, wsum.max() + step_r, step_r))
    axf.tick_params(labelsize=6.0, pad=0.5)
    axf.grid(True, lw=0.45, color="#dcdcdc")
    for lab in axf.get_yticklabels():          # radial ticks sit on the grid
        lab.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.8,
                          alpha=0.92))
    axf.set_title("Direction rose,\nmagnitude-weighted", fontsize=8.0,
                  pad=5.0, color=INK, linespacing=1.35)
    add_panel_label(axf, "(f)", x=-0.13, y=0.98, size=9.5)

    # ---- (g) the wheel legend ---------------------------------------------
    axg = fig.add_subplot(gs[1, 3])
    n = 221
    gy, gx = np.mgrid[0:n, 0:n].astype(float)
    gx = (gx - (n - 1) / 2) / ((n - 1) / 2)
    gy = (gy - (n - 1) / 2) / ((n - 1) / 2)
    rr = np.hypot(gx, gy)
    wheel = flow_rgb(gx * msat, gy * msat, msat)
    wheel[rr > 1.0] = 1.0
    axg.imshow(wheel, extent=(-1, 1, 1, -1), interpolation="bilinear",
               zorder=2)
    axg.add_patch(Circle((0, 0), 1.0, facecolor="none", edgecolor="#9a9a9a",
                         lw=0.7, zorder=4))
    for rad in (0.5,):
        axg.add_patch(Circle((0, 0), rad, facecolor="none", edgecolor="#7a7a7a",
                             lw=0.5, ls=(0, (2, 2)), zorder=4))
    for a, lab in ((0, "right"), (90, "down"), (180, "left"), (270, "up")):
        th = np.radians(a)
        axg.text(1.30 * np.cos(th), 1.30 * np.sin(th), lab, fontsize=6.6,
                 color=SUB, ha="center", va="center", zorder=5)
    for rad, lab in ((0.50, f"{0.5 * msat:.2f}"), (1.19, f"{msat:.2f} px")):
        axg.text(rad * np.cos(np.radians(-45)), rad * np.sin(np.radians(-45)),
                 lab, fontsize=6.2, color="#20374f", ha="center", va="center",
                 zorder=6, bbox=dict(facecolor="white", edgecolor="none",
                                     pad=1.0, alpha=0.94))
    axg.set_xlim(-1.62, 1.62)
    axg.set_ylim(1.62, -1.62)
    axg.set_aspect("equal")
    axg.set_xticks([])
    axg.set_yticks([])
    for s in axg.spines.values():
        s.set_visible(False)
    axg.set_title("Encoding: hue = direction,\nlightness = magnitude",
                  fontsize=8.0, pad=5.0, color=INK, linespacing=1.35)
    add_panel_label(axg, "(g)", x=-0.05, y=0.98, size=9.5)

    # ---- headline ----------------------------------------------------------
    savefig(fig, "1165_wm_vid_optical_flow.png")


if __name__ == "__main__":
    main()

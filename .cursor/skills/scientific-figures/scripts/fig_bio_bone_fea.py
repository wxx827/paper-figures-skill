"""Finite-element von Mises stress in a proximal femur under single-leg stance.

The mesh is genuinely unstructured: nodes are scattered inside an implicit
femur outline and triangulated, so the cloud is drawn with ``tricontourf``
rather than on a raster grid.  The stress model is a cortical-shell field -
bending about the diaphysis plus a concentration on the inferior neck - which
is why the colour piles up on the outer wall and the cancellous core stays
cold.  Everything quoted in the figure (peak stress, its location, the safety
factors, the profile curves) is read back out of that same nodal field.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.lines import Line2D
from pubstyle import (set_pub_style, savefig, style_axes, get_cmap, COLORS,
                      annotate_box)
from figkit import safe_colorbar, panel_labels_inches, arrow, add_scale_bar

MEDIAL = "#b23a48"       # house red   - compressive medial cortex (the peak)
LATERAL = "#3b5b92"      # house blue  - tensile lateral cortex
YIELD_MPA = 120.0        # cortical bone yield stress
SF_TARGET = 1.5
LOAD_N, ABDUCTOR_N, BODY_N = 2400.0, 1150.0, 750.0
DEFLECT_MM = 2.4         # true peak deflection
DEFORM_SCALE = 6.0       # displayed deformation magnification
CORTEX_MM = 9.0          # cortical wall thickness used by the stress model


def _seg_dist(x, y, p0, p1):
    px, py = p1[0] - p0[0], p1[1] - p0[1]
    t = np.clip(((x - p0[0]) * px + (y - p0[1]) * py) / (px * px + py * py),
                0.0, 1.0)
    return np.hypot(x - (p0[0] + t * px), y - (p0[1] + t * py))


def _axis(y):
    """Diaphyseal centre line and half-width (mm) at height ``y``."""
    xc = -6.0 * np.sin(np.pi * np.clip(y, 0.0, 400.0) / 400.0)
    w = 15.0 + 10.0 * np.exp(-((y - 370.0) / 45.0) ** 2)
    return xc, w


def femur_field(x, y):
    """Implicit femur: positive inside, in millimetres from the surface."""
    xc, w = _axis(y)
    f_shaft = np.minimum(np.minimum(w - np.abs(x - xc), y - 22.0), 404.0 - y)
    f_head = 23.0 - np.hypot(x + 48.0, y - 432.0)
    f_neck = 15.0 - _seg_dist(x, y, (-42.0, 424.0), (2.0, 392.0))
    f_troch = 17.0 * (1.0 - np.hypot((x - 20.0) / 17.0, (y - 398.0) / 27.0))
    f_cond = 20.0 * (1.0 - np.hypot((x - 2.0) / 36.0, (y - 18.0) / 21.0))
    return np.maximum.reduce([f_shaft, f_head, f_neck, f_troch, f_cond])


def von_mises(x, y, f=None):
    """Cortical-shell bending field with an inferior-neck concentration (MPa)."""
    f = femur_field(x, y) if f is None else f
    xc, w = _axis(y)
    edge = np.clip(1.0 - f / CORTEX_MM, 0.0, 1.0)
    bend = np.exp(-((y - 150.0) / 165.0) ** 2)
    side = np.clip((xc - x) / np.maximum(w, 1.0), -1.2, 1.2)
    neck = np.exp(-_seg_dist(x, y, (-30.0, 414.0), (-8.0, 396.0)) / 17.0)
    return (4.0 + 58.0 * edge * (0.28 + 0.72 * bend) * (1.0 + 0.42 * side)
            + 60.0 * edge * neck)


def deflection(y):
    """Cantilever shape function scaled to ``DEFLECT_MM`` at the head."""
    L = 460.0
    yy = np.clip(y, 0.0, L)
    return -DEFLECT_MM * (yy ** 2 * (3.0 * L - yy)) / (2.0 * L ** 3)


def outline(n=340):
    """Ordered boundary polygon of the implicit shape, resampled evenly."""
    gx, gy = np.meshgrid(np.linspace(-90.0, 50.0, 340),
                         np.linspace(-14.0, 464.0, 1000))
    tmp = plt.figure()
    cs = tmp.add_subplot(111).contour(gx, gy, femur_field(gx, gy),
                                      levels=[0.0])
    try:
        segs = cs.allsegs[0]
    except AttributeError:
        segs = [p.vertices for p in cs.get_paths()]
    poly = max(segs, key=len)
    plt.close(tmp)
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(poly, axis=0).T))]
    s = np.linspace(0.0, d[-1], n)
    return np.column_stack([np.interp(s, d, poly[:, 0]),
                            np.interp(s, d, poly[:, 1])])


def mesh(rng, poly, step, jitter, max_edge):
    """Unstructured triangulation of the femur; returns (tri, x, y)."""
    xs = np.arange(-80.0, 45.0, step)
    ys = np.arange(-8.0, 460.0, step)
    gx, gy = np.meshgrid(xs, ys)
    gx = gx + rng.uniform(-jitter, jitter, gx.shape)
    gy = gy + rng.uniform(-jitter, jitter, gy.shape)
    keep = femur_field(gx, gy) > 0.35 * step
    x = np.r_[poly[:, 0], gx[keep]]
    y = np.r_[poly[:, 1], gy[keep]]
    tri = mtri.Triangulation(x + deflection(y) * DEFORM_SCALE, y)
    t = tri.triangles
    cx, cy = x[t].mean(axis=1), y[t].mean(axis=1)
    edge = np.max([np.hypot(x[t[:, i]] - x[t[:, j]], y[t[:, i]] - y[t[:, j]])
                   for i, j in ((0, 1), (1, 2), (2, 0))], axis=0)
    tri.set_mask((femur_field(cx, cy) < 0.6) | (edge > max_edge))
    return tri, x, y


def main():
    set_pub_style()
    rng = np.random.default_rng(512)
    poly = outline()
    tri, nx, ny = mesh(rng, poly, step=3.0, jitter=1.05, max_edge=13.0)
    coarse, cx_u, cy_u = mesh(rng, poly[::7], step=11.0, jitter=2.2,
                              max_edge=26.0)
    sig = von_mises(nx, ny)
    px, py = poly[:, 0] + deflection(poly[:, 1]) * DEFORM_SCALE, poly[:, 1]

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58], wspace=0.30, hspace=0.42,
                          left=0.035, right=0.955, top=0.885, bottom=0.100)
    ax_m = fig.add_subplot(gs[:, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[1, 1])

    # ---- (a) stress cloud on the deformed mesh -----------------------------
    cf = ax_m.tricontourf(tri, sig, levels=14,
                          cmap=get_cmap("cb_sequential_warm"), zorder=2)
    cf.set_edgecolor("face")
    ax_m.triplot(coarse, color="#3a3a3a", lw=0.22, alpha=0.35, zorder=3)
    ax_m.plot(poly[:, 0], poly[:, 1], color=COLORS["muted"], lw=0.8,
              ls=(0, (3.2, 2.4)), zorder=4)
    ax_m.plot(px, py, color=COLORS["rule"], lw=0.9, zorder=5)

    i_max = int(np.argmax(sig))
    x_max = nx[i_max] + deflection(ny[i_max]) * DEFORM_SCALE
    ax_m.plot(x_max, ny[i_max], marker="o", ms=5.0, mfc="none",
              mec=COLORS["ink"], mew=1.0, zorder=7)
    ax_m.annotate(f"peak {sig[i_max]:.0f} MPa",
                  xy=(x_max, ny[i_max]), xytext=(72.0, ny[i_max] - 20.0),
                  ha="left", va="center", fontsize=6.8, color=COLORS["ink"],
                  arrowprops=dict(arrowstyle="-", lw=0.7,
                                  color=COLORS["rule"]))

    # load and boundary conditions
    arrow(ax_m, (-108.0, 530.0), (-52.0, 438.0), color=COLORS["ink"], lw=1.5,
          ms=13, zorder=8)
    ax_m.text(-112.0, 534.0, f"joint reaction {LOAD_N:.0f} N "
              f"({LOAD_N / BODY_N:.1f} \u00d7 BW)", ha="left", va="bottom",
              fontsize=6.8, color=COLORS["ink"])
    arrow(ax_m, (96.0, 482.0), (32.0, 421.0), color=COLORS["ink"], lw=1.3,
          ms=12, zorder=8)
    ax_m.text(100.0, 484.0, f"abductor\n{ABDUCTOR_N:.0f} N", ha="left",
              va="bottom", fontsize=6.8, color=COLORS["ink"], linespacing=1.35)

    ax_m.plot([-46.0, 52.0], [-9.0, -9.0], color=COLORS["rule"], lw=1.2,
              zorder=8)
    for x0 in np.linspace(-44.0, 44.0, 9):
        ax_m.plot([x0, x0 - 9.0], [-9.0, -20.0], color=COLORS["rule"],
                  lw=0.8, zorder=8)
    ax_m.text(60.0, -14.0, "fixed distal support\n(all DOF constrained)",
              ha="left", va="center", fontsize=6.8, color=COLORS["ink"],
              linespacing=1.35)

    ax_m.set_xlim(-120.0, 212.0)
    ax_m.set_ylim(-48.0, 560.0)
    ax_m.set_aspect("equal")
    ax_m.set_xticks([])
    ax_m.set_yticks([])
    for sp in ax_m.spines.values():
        sp.set_visible(False)
    add_scale_bar(ax_m, 50.0, "50 mm", loc=(0.02, 0.028), height=0.010,
                  color=COLORS["rule"], alt="white", text_color=COLORS["ink"],
                  halo="white", fontsize=6.2)
    annotate_box(ax_m, f"E = 17 GPa, \u03bd = 0.30\n"
                 f"yield {YIELD_MPA:.0f} MPa\n"
                 f"{nx.size:,} nodes\n{(~tri.mask).sum():,} elements",
                 loc="center right", fontsize=6.4, margin=0.015)
    ax_m.set_title("Stress on the deformed mesh", fontsize=8.5,
                   fontweight="bold")
    safe_colorbar(fig, cf, ax_m, label="von Mises stress (MPa)", side="left",
                  width=0.11, pad=0.08, shrink=0.80)

    # ---- (b) stress profile down the two cortices --------------------------
    yy = np.linspace(38.0, 352.0, 220)
    xc, w = _axis(yy)
    s_med = von_mises(xc - w + 2.5, yy)
    s_lat = von_mises(xc + w - 2.5, yy)
    ax_p.plot(yy, s_med, color=MEDIAL, lw=1.5, label="medial cortex")
    ax_p.plot(yy, s_lat, color=LATERAL, lw=1.5, label="lateral cortex")
    ax_p.fill_between(yy, s_lat, s_med, color=MEDIAL, alpha=0.10, lw=0)
    j = int(np.argmax(s_med))
    ax_p.annotate(f"{s_med[j]:.0f} MPa at {yy[j]:.0f} mm",
                  xy=(yy[j], s_med[j]), xytext=(yy[j] + 12.0, s_med[j] + 9.0),
                  ha="left", va="bottom", fontsize=6.6, color=COLORS["ink"],
                  arrowprops=dict(arrowstyle="-", lw=0.7,
                                  color=COLORS["muted"]))
    ax_p.set_xlim(38.0, 352.0)
    ax_p.set_ylim(0.0, max(s_med.max(), s_lat.max()) * 1.30)
    ax_p.set_xlabel("Distance from distal support (mm)", fontsize=7.5)
    ax_p.set_ylabel("von Mises stress (MPa)", fontsize=7.5)
    ax_p.tick_params(labelsize=6.5)
    ax_p.legend(loc="upper left", frameon=False, fontsize=6.8, ncol=2,
                handlelength=1.3)
    ax_p.set_title("Stress along the diaphyseal cortices", fontsize=8.5,
                   fontweight="bold")
    style_axes(ax_p)

    # ---- (c) safety factor per anatomical region ---------------------------
    head = np.hypot(nx + 48.0, ny - 432.0) < 23.0
    regions = [("Femoral head", head),
               ("Neck / trochanter", (~head) & (ny > 385.0)),
               ("Proximal shaft", (ny > 260.0) & (ny <= 385.0)),
               ("Mid shaft", (ny > 120.0) & (ny <= 260.0)),
               ("Distal shaft", (ny > 45.0) & (ny <= 120.0)),
               ("Condyles", ny <= 45.0)]
    names = [r[0] for r in regions]
    sf = np.array([YIELD_MPA / sig[m].max() for _, m in regions])
    y_pos = np.arange(len(names))[::-1]
    colours = [MEDIAL if v < SF_TARGET else "#8a8a8a" for v in sf]
    ax_s.barh(y_pos, sf, height=0.62, color=colours, edgecolor="white",
              linewidth=0.6)
    ax_s.axvline(1.0, color=COLORS["rule"], lw=0.9)
    ax_s.axvline(SF_TARGET, color=COLORS["baseline"], lw=0.8,
                 ls=(0, (3.0, 2.4)))
    ax_s.text(1.0, len(names) - 0.30, "yield", ha="center", va="bottom",
              fontsize=6.2, color=COLORS["rule"])
    ax_s.text(SF_TARGET, len(names) - 0.30, f"target {SF_TARGET:.1f}",
              ha="center", va="bottom", fontsize=6.2, color=COLORS["sub"])
    for yi, v in zip(y_pos, sf):
        ax_s.text(v + 0.06, yi, f"{v:.2f}", ha="left", va="center",
                  fontsize=6.4, color=COLORS["ink"])
    ax_s.set_yticks(y_pos)
    ax_s.set_yticklabels(names, fontsize=6.6)
    ax_s.set_xlim(0.0, max(3.0, sf.max() * 1.22))
    ax_s.set_ylim(-0.7, len(names) - 0.05)
    ax_s.set_xlabel("Safety factor, yield / peak stress (-)", fontsize=7.5)
    ax_s.tick_params(axis="x", labelsize=6.5)
    ax_s.set_title("Regional safety factor", fontsize=8.5, fontweight="bold")
    style_axes(ax_s)

    handles = [Line2D([], [], color=COLORS["rule"], lw=0.9,
                      label="deformed surface"),
               Line2D([], [], color=COLORS["muted"], lw=0.8,
                      ls=(0, (3.2, 2.4)), label="undeformed surface")]
    ax_m.legend(handles=handles, loc="lower left", frameon=False,
                fontsize=6.8, bbox_to_anchor=(0.50, 0.125))

    panel_labels_inches(fig, [ax_m, ax_p, ax_s], dx=-0.10, dy=0.10)
    worst = names[int(np.argmin(sf))]
    savefig(fig, "512_bio_bone_fea.png")


if __name__ == "__main__":
    main()

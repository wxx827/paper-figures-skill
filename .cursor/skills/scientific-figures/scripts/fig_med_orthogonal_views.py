"""Figure 382 - Orthogonal reformats of one volume, plus a surface rendering.

The three planes are cut out of a single 3-D phantom, so the cross hairs really
do meet at one voxel; each plane owns a colour and draws the *other* two planes
as lines in their colours, which is the convention every PACS viewer uses and
the only reason a reader can fuse three 2-D pictures into one 3-D location.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.ndimage import gaussian_filter, rotate

from pubstyle import set_pub_style, savefig, COLORS
from figkit import image_axes, axes_inches, corner_stamp, panel_labels_inches

NV = 128                              # isotropic volume, NV^3 voxels
MM_VOX = 220.0 / NV
AX_C, COR_C, SAG_C = "#3b5b92", "#2e7d5b", "#c9a227"
TUM_C = "#b23a48"

# cross-hair voxel, chosen to sit on the lesion
CUT = dict(z=70, y=52, x=84)


def _sm3(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def head_volume(n=NV, seed=8):
    """T1-like head volume plus the lesion mask, both (z, y, x) ordered."""
    rng = np.random.default_rng(seed)
    g = (np.arange(n) - (n - 1) / 2) / (n / 2)
    z = g[:, None, None]              # +z is inferior (row 0 is superior)
    y = g[None, :, None]              # +y is posterior
    x = g[None, None, :]              # +x is patient left

    rr = np.sqrt((x / 0.74) ** 2 + (y / 0.86) ** 2 + (z / 0.82) ** 2)
    vol = np.full((n, n, n), 0.02)
    vol = np.where(rr <= 1.000, 0.38, vol)                 # scalp
    vol = np.where(rr <= 0.955, 0.10, vol)                 # skull
    vol = np.where(rr <= 0.895, 0.14, vol)                 # subarachnoid CSF
    brain = rr <= 0.870
    vol = np.where(brain, 0.48, vol)                       # grey matter

    rw = np.sqrt((x / 0.56) ** 2 + ((y - 0.02) / 0.66) ** 2
                 + ((z - 0.04) / 0.60) ** 2)
    vol = np.where(brain & (rw <= 1.0), 0.72, vol)         # white matter

    g1, g2 = _sm3(rng, n, 5.0), _sm3(rng, n, 10.0)
    sulci = (np.sin(7.0 * g1 + 4.0 * g2) > 0.55) & brain & (rr > 0.56)
    vol = np.where(sulci, 0.15, vol)

    for sx in (-1, 1):                                     # lateral ventricles
        v = np.sqrt(((x - sx * 0.12) / 0.075) ** 2 + ((y + 0.02) / 0.34) ** 2
                    + ((z - 0.06) / 0.20) ** 2)
        vol = np.where(v <= 1.0, 0.07, vol)
    vol = np.where((np.abs(x) < 0.014) & brain, 0.22, vol)  # falx

    lz, ly, lx = (CUT["z"] - (n - 1) / 2) / (n / 2), \
                 (CUT["y"] - (n - 1) / 2) / (n / 2), \
                 (CUT["x"] - (n - 1) / 2) / (n / 2)
    dl = np.sqrt(((x - lx) / 0.135) ** 2 + ((y - ly) / 0.135) ** 2
                 + ((z - lz) / 0.130) ** 2)
    lesion = (dl <= 1.0) & brain
    vol = np.where(lesion, 0.90, vol)
    vol = np.where((dl > 1.0) & (dl <= 1.55) & brain, 0.60, vol)   # edema

    vol = gaussian_filter(vol, 0.9)
    vol = vol + 0.010 * rng.standard_normal((n, n, n))
    return np.clip(vol, 0, 1), lesion, rr <= 1.0


def crosshair(ax, cx, cy, vcolor, hcolor, n):
    ax.axvline(cx, color=vcolor, lw=0.8, alpha=0.95, zorder=5)
    ax.axhline(cy, color=hcolor, lw=0.8, alpha=0.95, zorder=5)
    ax.plot([cx], [cy], marker="+", ms=9, mew=1.1, color="white", zorder=6)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)


def depth_render(mask):
    """Depth-shaded first-hit surface of a boolean volume, viewed along y."""
    hit = mask.any(axis=1)
    depth = np.where(hit, np.argmax(mask, axis=1).astype(float), np.nan)
    filled = np.nan_to_num(depth, nan=float(np.nanmax(depth)))
    gz, gx = np.gradient(gaussian_filter(filled, 1.2))
    norm = np.sqrt(gx ** 2 + gz ** 2 + 1.0)
    shade = (0.40 * (-gx) + 0.55 * (-gz) + 1.0) / norm
    shade = (shade - np.nanmin(shade)) / (np.ptp(shade) + 1e-9)
    return np.where(hit, 0.20 + 0.75 * shade, np.nan), depth


def main():
    set_pub_style(base_font=8)
    vol, lesion, head = head_volume()

    axial = vol[CUT["z"], :, :]
    coronal = vol[:, CUT["y"], :]
    sagittal = vol[:, :, CUT["x"]]

    fig = plt.figure(figsize=(10.4, 6.4))
    tile, gap = 2.62, 0.30
    x0 = (10.4 - (3 * tile + 2 * gap)) / 2.0
    top_y, bot_y = 3.28, 0.42
    ax_a = axes_inches(fig, (x0, top_y, tile, tile))
    ax_c = axes_inches(fig, (x0 + tile + gap, top_y, tile, tile))
    ax_s = axes_inches(fig, (x0 + 2 * (tile + gap), top_y, tile, tile))
    ax_v = axes_inches(fig, (x0, bot_y, tile, tile))
    ax_l = axes_inches(fig, (x0 + tile + gap, bot_y, tile, tile))
    ax_i = axes_inches(fig, (x0 + 2 * (tile + gap), bot_y, tile, tile))

    grey = dict(cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")

    # -- (a) axial: columns are x (sagittal), rows are y (coronal) -----------
    image_axes(ax_a, color=AX_C, lw=1.1)
    ax_a.imshow(axial, **grey)
    crosshair(ax_a, CUT["x"], CUT["y"], SAG_C, COR_C, NV)
    ax_a.set_title("Axial", fontsize=9.0, fontweight="bold", color=AX_C,
                   pad=3.5)
    corner_stamp(ax_a, f"z = {CUT['z']}  ({(CUT['z'] - NV / 2) * MM_VOX:+.0f} "
                       "mm)", loc="lower left", fontsize=6.6)

    # -- (b) coronal: columns are x, rows are z ------------------------------
    image_axes(ax_c, color=COR_C, lw=1.1)
    ax_c.imshow(coronal, **grey)
    crosshair(ax_c, CUT["x"], CUT["z"], SAG_C, AX_C, NV)
    ax_c.set_title("Coronal", fontsize=9.0, fontweight="bold", color=COR_C,
                   pad=3.5)
    corner_stamp(ax_c, f"y = {CUT['y']}  ({(CUT['y'] - NV / 2) * MM_VOX:+.0f} "
                       "mm)", loc="lower left", fontsize=6.6)

    # -- (c) sagittal: columns are y, rows are z -----------------------------
    image_axes(ax_s, color=SAG_C, lw=1.1)
    ax_s.imshow(sagittal, **grey)
    crosshair(ax_s, CUT["y"], CUT["z"], COR_C, AX_C, NV)
    ax_s.set_title("Sagittal", fontsize=9.0, fontweight="bold", color=SAG_C,
                   pad=3.5)
    corner_stamp(ax_s, f"x = {CUT['x']}  ({(CUT['x'] - NV / 2) * MM_VOX:+.0f} "
                       "mm)", loc="lower left", fontsize=6.6)

    # -- (d) oblique surface rendering ---------------------------------------
    rot = dict(axes=(1, 2), reshape=False, order=1, mode="constant")
    head_r = rotate(head.astype(float), 34.0, **rot) > 0.5
    brain_r = rotate((vol > 0.30).astype(float), 34.0, **rot) > 0.5
    les_r = rotate(lesion.astype(float), 34.0, **rot) > 0.5
    shade, depth = depth_render(head_r)

    image_axes(ax_v, color="#d5d5d5", lw=0.6)
    ax_v.imshow(shade, cmap="gray", vmin=0.0, vmax=1.0,
                interpolation="bilinear")
    ax_v.contour(np.nan_to_num(depth, nan=NV), levels=np.arange(28, 90, 9),
                 colors=["#9a9a9a"], linewidths=0.45, alpha=0.75, zorder=3)
    ax_v.contour(brain_r.any(axis=1).astype(float), levels=[0.5],
                 colors=[AX_C], linewidths=1.0, zorder=4)
    ax_v.contour(les_r.any(axis=1).astype(float), levels=[0.5],
                 colors=[TUM_C], linewidths=1.4, zorder=5)
    ax_v.set_xlim(-0.5, NV - 0.5)
    ax_v.set_ylim(NV - 0.5, -0.5)
    ax_v.set_title("Surface rendering  (34 deg oblique)", fontsize=9.0,
                   fontweight="bold", pad=3.5)
    corner_stamp(ax_v, "iso-depth contours every 9 voxels", loc="lower left",
                 fontsize=6.4)

    # -- (e) where the three planes sit inside the volume --------------------
    ax_l.set_xlim(-0.10, 1.30)
    ax_l.set_ylim(-0.14, 1.58)
    ax_l.set_aspect("equal")
    ax_l.axis("off")

    def iso(px, py, pz):
        return (0.50 + 0.52 * (px - py), 0.30 + 0.30 * (px + py) + 0.72 * pz)

    corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
               (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    proj = [iso(*c) for c in corners]
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
                 (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        ax_l.plot([proj[a][0], proj[b][0]], [proj[a][1], proj[b][1]],
                  color="#bbbbbb", lw=0.7, zorder=1)

    fz = 1.0 - CUT["z"] / NV                 # z index 0 is superior
    fy, fx = CUT["y"] / NV, CUT["x"] / NV
    planes = [
        ("Axial", AX_C, [(0, 0, fz), (1, 0, fz), (1, 1, fz), (0, 1, fz)]),
        ("Coronal", COR_C, [(0, fy, 0), (1, fy, 0), (1, fy, 1), (0, fy, 1)]),
        ("Sagittal", SAG_C, [(fx, 0, 0), (fx, 1, 0), (fx, 1, 1), (fx, 0, 1)]),
    ]
    for name, colour, quad in planes:
        pts = [iso(*p) for p in quad]
        ax_l.add_patch(Polygon(pts, closed=True, facecolor=colour, alpha=0.26,
                               edgecolor=colour, linewidth=1.1, zorder=3))
    ax_l.plot(*iso(fx, fy, fz), marker="+", ms=10, mew=1.4, color=TUM_C,
              zorder=6)
    for name, colour, quad in planes:
        pts = np.array([iso(*p) for p in quad])
        ax_l.text(pts[:, 0].mean(), pts[:, 1].max() + 0.03, name,
                  fontsize=6.8, color=colour, ha="center", va="bottom",
                  zorder=7)
    ax_l.text(*iso(0.5, 0.5, -0.16), "R", fontsize=7.0, ha="center",
              color=COLORS["muted"])
    ax_l.set_title("Plane locator", fontsize=9.0, fontweight="bold", pad=3.5)

    # -- (f) the header a viewer would show ----------------------------------
    ax_i.axis("off")
    ax_i.add_patch(plt.Rectangle((0.0, 0.0), 1.0, 1.0, transform=ax_i.transAxes,
                                 facecolor="#f5f5f5", edgecolor="#d5d5d5",
                                 linewidth=0.7))
    mm = [(k, (CUT[k] - NV / 2) * MM_VOX) for k in ("x", "y", "z")]
    lines = [
        ("series", "3D T1w MPRAGE, isotropic"),
        ("matrix", f"{NV} x {NV} x {NV} voxels"),
        ("voxel size", f"{MM_VOX:.2f} x {MM_VOX:.2f} x {MM_VOX:.2f} mm"),
        ("field of view", f"{NV * MM_VOX:.0f} mm cubed"),
        ("cross hair", "  ".join(f"{k} {v:+.0f} mm" for k, v in mm)),
        ("lesion volume", f"{lesion.sum() * MM_VOX ** 3 / 1000.0:.1f} cm3"),
        ("window", "W 1.00 / L 0.50 (a.u.)"),
    ]
    for i, (key, val) in enumerate(lines):
        yv = 0.90 - 0.125 * i
        ax_i.text(0.06, yv, key, fontsize=6.8, color=COLORS["muted"],
                  ha="left", va="center", transform=ax_i.transAxes)
        ax_i.text(0.42, yv, val, fontsize=6.8, color=COLORS["ink"], ha="left",
                  va="center", transform=ax_i.transAxes)
    ax_i.set_title("Acquisition and cursor read-out", fontsize=9.0,
                   fontweight="bold", pad=3.5)

    for a in (ax_a, ax_c, ax_s, ax_v, ax_l, ax_i):
        a.apply_aspect()
    panel_labels_inches(fig, [ax_a, ax_c, ax_s, ax_v, ax_l, ax_i], dx=-0.22,
                        dy=0.12)


    savefig(fig, "382_med_orthogonal_views.png")


if __name__ == "__main__":
    main()

"""Neural radiance field: novel-view renders, depth maps and camera orbit.

Left: the spherical capture trajectory with the four evaluated poses picked
out.  Right: a 2 x 5 grid whose top row holds the RGB renders (four novel
views plus the held-out ground truth) and whose bottom row holds the matching
depth maps, with per-view PSNR underneath.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.cm import ScalarMappable

from pubstyle import set_pub_style, savefig

SEED = 8842
NU, NV = 58, 104
VIEWS = [(18.0, -72.0), (26.0, -22.0), (34.0, 34.0), (14.0, 86.0)]
VCOL = ["#1f6fb4", "#2e8b57", "#c9701f", "#8e4bbf"]
PSNR = [30.42, 31.85, 28.97, 30.11]
LIGHT = np.array([0.40, 0.34, 0.85])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def blob():
    u = np.linspace(0.04, np.pi - 0.04, NU)
    v = np.linspace(0.0, 2.0 * np.pi, NV, endpoint=False)
    U, V = np.meshgrid(u, v, indexing="ij")
    R = (1.0 + 0.22 * np.sin(2.0 * U) * np.cos(3.0 * V)
         + 0.12 * np.sin(4.0 * U) + 0.08 * np.cos(5.0 * V - 0.6))
    return U, V, R


def verts_of(U, V, R):
    return np.stack([R * np.sin(U) * np.cos(V), R * np.sin(U) * np.sin(V),
                     R * np.cos(U) * 0.92], axis=-1).reshape(-1, 3)


def tri_index():
    t = []
    for i in range(NU - 1):
        for j in range(NV):
            j2 = (j + 1) % NV
            a, b = i * NV + j, i * NV + j2
            c, d = (i + 1) * NV + j, (i + 1) * NV + j2
            t += [(a, c, b), (b, c, d)]
    return np.array(t)


def texture(cent):
    """Two-tone painted pattern so the object has recognisable appearance."""
    r = np.linalg.norm(cent, axis=1) + 1e-9
    th = np.arccos(np.clip(cent[:, 2] / r, -1, 1))
    ph = np.arctan2(cent[:, 1], cent[:, 0])
    w = 0.5 + 0.5 * np.tanh(2.4 * np.sin(3.0 * ph) * np.sin(2.2 * th))
    c1, c2 = np.array(to_rgb("#d9573f")), np.array(to_rgb("#e8dcc4"))
    col = c1 * w[:, None] + c2 * (1 - w[:, None])
    band = (np.sin(6.0 * th) > 0.86)[:, None]
    return np.where(band, col * 0.68, col)


def shade(verts, tris, cols):
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    n = np.cross(b - a, c - a)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    lam = 0.42 + 0.58 * np.clip(n @ LIGHT, 0.0, 1.0)
    spec = 0.22 * np.clip(n @ LIGHT, 0.0, 1.0) ** 18
    return np.clip(cols * lam[:, None] + spec[:, None], 0.0, 1.0)


def view_dir(elev, azim):
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def clean3d(ax, elev, azim, lim):
    ax.patch.set_alpha(0.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)
    U, V, R = blob()
    tris = tri_index()
    Vgt = verts_of(U, V, R)
    LIM = 1.02 * np.abs(Vgt).max()

    dcm = LinearSegmentedColormap.from_list(
        "d", ["#fdf3b4", "#f2a03c", "#c8452f", "#7b2a6b", "#241a52"])

    fig = plt.figure(figsize=(8.0, 3.62))
    GX, GW, GAP = 0.232, 0.1495, 0.0040
    RY = [0.500, 0.108]
    RH = 0.375

    cols_gt = texture(Vgt[tris].mean(axis=1))
    panels = list(zip(VIEWS, VCOL)) + [((26.0, -22.0), "#444444")]

    for k, ((elev, azim), _) in enumerate(panels):
        gt_col = k == 4
        if gt_col:
            Vk, cbase, floats = Vgt, cols_gt, 0
        else:
            Rk = R + rng.normal(0.0, 0.011, R.shape)
            Vk = verts_of(U, V, Rk)
            cbase = texture(Vk[tris].mean(axis=1))
            cbase = 0.80 * cbase + 0.20 * cbase.mean(axis=0)   # slight haze
            floats = 26
        x0 = GX + k * (GW + GAP)

        ax = fig.add_axes([x0 - 0.021, RY[0] - 0.075, GW + 0.042, RH + 0.150],
                          projection="3d")
        s = ax.plot_trisurf(Vk[:, 0], Vk[:, 1], Vk[:, 2], triangles=tris,
                            linewidth=0, antialiased=True, shade=False)
        s.set_fc(shade(Vk, tris, cbase))
        if floats:                              # NeRF-style floaters
            d = view_dir(elev, azim)
            p = rng.normal(0, 0.75, (floats, 3))
            p += d * (LIM * 0.9)
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=2.4, c="#b9c6d4",
                       alpha=0.55, marker=".", linewidths=0, depthshade=False)
        clean3d(ax, elev, azim, LIM)

        axd = fig.add_axes([x0 - 0.021, RY[1] - 0.075, GW + 0.042, RH + 0.150],
                           projection="3d")
        sd = axd.plot_trisurf(Vk[:, 0], Vk[:, 1], Vk[:, 2], triangles=tris,
                              linewidth=0, antialiased=True, shade=False)
        d = view_dir(elev, azim)
        dep = Vk[tris].mean(axis=1) @ d
        sd.set_fc(dcm(Normalize(-LIM * 0.75, LIM * 0.95)(dep)))
        clean3d(axd, elev, azim, LIM)

        head = "GT (held out)" if gt_col else f"View {k + 1}"
        txt = "reference" if gt_col else f"PSNR {PSNR[k]:.2f} dB"

    fig.text(GX - 0.012, RY[0] + RH / 2, "RGB render", fontsize=7.6,
             rotation=90, ha="right", va="center", color="#444444")
    fig.text(GX - 0.012, RY[1] + RH / 2, "Depth", fontsize=7.6, rotation=90,
             ha="right", va="center", color="#444444")

    # ---------------- accessory: capture trajectory ------------------------
    fig.patches.append(plt.Rectangle(
        (0.008, 0.030), 0.198, 0.880, transform=fig.transFigure,
        facecolor="#f4f4f5", edgecolor="#dedede", linewidth=0.6, zorder=-1))
    axt = fig.add_axes([-0.014, 0.395, 0.242, 0.470], projection="3d")
    _orbit(axt, rng)
    fig.text(0.107, 0.352, "72 training poses on a hemisphere\n"
             "r = 3.2 m   |   4 held-out views", fontsize=6.2,
             color="#666666", ha="center", va="center", linespacing=1.45)

    cax = fig.add_axes([0.038, 0.205, 0.138, 0.015])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=dcm), cax=cax,
                      orientation="horizontal", ticks=[0, 1])
    cb.ax.set_xticklabels(["near", "far"], fontsize=6.0)
    cb.set_label("normalised depth (-)", fontsize=7.0, color="#1a1a1a",
                 labelpad=1.5)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=7.0, width=0.7, length=2.5, color="#333333")
    cb.ax.tick_params(width=0.6, length=1.6, pad=1.2)

    fig.text(0.107, 0.090, "Mean PSNR 30.34 dB\nSSIM 0.947  |  LPIPS 0.061",
             fontsize=6.4, color="#1a1a1a", ha="center", va="center",
             linespacing=1.45)


    savefig(fig, "289_nerf_rendering.png")


def _orbit(ax, rng):
    r = 3.2
    for el in (12.0, 30.0, 48.0):
        a = np.linspace(0, 2 * np.pi, 160)
        e = np.deg2rad(el)
        ax.plot(r * np.cos(e) * np.cos(a), r * np.cos(e) * np.sin(a),
                np.full_like(a, r * np.sin(e)), color="#c8c8c8", lw=0.5)
    for el in (12.0, 30.0, 48.0):
        a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        e = np.deg2rad(el)
        ax.scatter(r * np.cos(e) * np.cos(a), r * np.cos(e) * np.sin(a),
                   np.full_like(a, r * np.sin(e)), s=3.0, c="#9aa0a6",
                   marker="o", linewidths=0, depthshade=False)
    # the object at the centre
    u = np.linspace(0, np.pi, 22)
    v = np.linspace(0, 2 * np.pi, 40)
    Uu, Vv = np.meshgrid(u, v)
    rr = 0.85 + 0.16 * np.sin(2 * Uu) * np.cos(3 * Vv)
    ax.plot_surface(rr * np.sin(Uu) * np.cos(Vv), rr * np.sin(Uu) * np.sin(Vv),
                    rr * np.cos(Uu), color="#d9573f", alpha=0.85,
                    linewidth=0, antialiased=True, shade=True)
    for (elev, azim), col in zip(VIEWS, VCOL):
        d = view_dir(elev, azim) * r
        ax.plot([0, d[0]], [0, d[1]], [0, d[2]], color=col, lw=0.7,
                ls=(0, (3, 2)), alpha=0.8)
        ax.scatter([d[0]], [d[1]], [d[2]], s=26, c=col, marker="^",
                   edgecolors="white", linewidths=0.5, depthshade=False)
    ax.patch.set_alpha(0.0)
    ax.set_xlim(-3.35, 3.35)
    ax.set_ylim(-3.35, 3.35)
    ax.set_zlim(-1.4, 3.3)
    ax.set_box_aspect((1, 1, 0.70))
    ax.view_init(elev=22, azim=-58)
    ax.set_axis_off()


if __name__ == "__main__":
    main()

"""Figure 392 - Rolling-texture pole figures, (100) / (110) / (111).

Orientations are drawn from a five-component model (cube, brass, copper, S and
a random background), then projected with the equal-area Lambert map so that a
random texture really does give a flat disc.  All three pole figures share one
set of contour levels and one colour bar, so their intensities are comparable;
the component fractions on the right are counted from the same orientations.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation

from pubstyle import (set_pub_style, savefig, get_cmap, style_axes, COLORS,
                      CATEGORICAL)
from figkit import panel_labels_inches

# Bunge Euler angles (phi1, Phi, phi2) of the ideal fcc rolling components.
COMPONENTS = [
    ("Cube", (0.0, 0.0, 0.0), 0.16, CATEGORICAL[0]),
    ("Brass", (35.0, 45.0, 0.0), 0.26, CATEGORICAL[1]),
    ("Copper", (90.0, 35.0, 45.0), 0.21, CATEGORICAL[2]),
    ("S", (59.0, 37.0, 63.0), 0.15, CATEGORICAL[3]),
    ("Random", None, 0.22, COLORS["baseline"]),
]
IDEALS = ["{001}<100>", "{110}<112>", "{112}<111>", "{123}<634>", "-"]
SPREAD_DEG = 9.0
N_GRAINS = 4000
POLES = [((1, 0, 0), "(100)"), ((1, 1, 0), "(110)"), ((1, 1, 1), "(111)")]
GRID = 220


def symmetry_equivalents(hkl):
    """Unique unit vectors of a cubic {hkl} family."""
    out = set()
    h = np.array(hkl, float)
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1),
                 (2, 1, 0)):
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    v = np.array([sx * h[perm[0]], sy * h[perm[1]],
                                  sz * h[perm[2]]])
                    out.add(tuple(np.round(v / np.linalg.norm(v), 9)))
    return np.array(sorted(out))


def sample_orientations(seed=17):
    rng = np.random.default_rng(seed)
    frac = np.array([c[2] for c in COMPONENTS])
    frac = frac / frac.sum()
    counts = np.round(frac * N_GRAINS).astype(int)
    mats, tags = [], []
    for k, (_, euler, _, _) in enumerate(COMPONENTS):
        n = counts[k]
        if euler is None:
            q = rng.normal(size=(n, 4))
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            r = Rotation.from_quat(q)
        else:
            base = Rotation.from_euler("zxz", euler, degrees=True)
            ax = rng.normal(size=(n, 3))
            ax /= np.linalg.norm(ax, axis=1, keepdims=True)
            ang = np.radians(np.abs(rng.normal(0, SPREAD_DEG, n)))
            r = Rotation.from_rotvec(ax * ang[:, None]) * base
        mats.append(r.as_matrix())
        tags.append(np.full(n, k))
    return np.concatenate(mats), np.concatenate(tags), counts


def pole_density(mats, hkl):
    """Equal-area density map in multiples of a random distribution."""
    hs = symmetry_equivalents(hkl)
    v = np.einsum("nji,mj->nmi", mats, hs).reshape(-1, 3)
    v[v[:, 2] < 0] *= -1.0                       # poles are axes
    r = np.sqrt(np.clip(1.0 - v[:, 2], 0.0, 2.0))
    phi = np.arctan2(v[:, 1], v[:, 0])
    x, y = r * np.cos(phi), r * np.sin(phi)

    edges = np.linspace(-1.02, 1.02, GRID + 1)
    hist, _, _ = np.histogram2d(y, x, bins=[edges, edges])
    dens = gaussian_filter(hist, 5.0, mode="nearest")
    cen = 0.5 * (edges[1:] + edges[:-1])
    X, Y = np.meshgrid(cen, cen)
    disc = X ** 2 + Y ** 2 <= 1.0
    dens = dens / dens[disc].mean()
    return X, Y, np.where(disc, dens, np.nan)


def frame_pole(ax, label):
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color=COLORS["rule"], lw=0.9)
    for a in (0.0, np.pi / 2):
        ax.plot([-np.cos(a), np.cos(a)], [-np.sin(a), np.sin(a)],
                color=COLORS["rule"], lw=0.5, alpha=0.6)
    ax.text(0.0, 1.06, "RD", ha="center", va="bottom", fontsize=6.8,
            color=COLORS["sub"])
    ax.text(1.07, 0.0, "TD", ha="left", va="center", fontsize=6.8,
            color=COLORS["sub"])
    ax.set_xlim(-1.28, 1.28)
    ax.set_ylim(-1.20, 1.22)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(label, fontsize=8.5, fontweight="bold", pad=2.0)


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    mats, tags, counts = sample_orientations()
    fields = [pole_density(mats, hkl) for hkl, _ in POLES]
    vmax = max(np.nanmax(f[2]) for f in fields)
    levels = np.linspace(0.0, np.ceil(vmax * 2) / 2.0, 12)
    cmap = get_cmap("cb_sequential_blue")

    fig = plt.figure(figsize=(10.4, 4.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.82],
                          wspace=0.24, left=0.020, right=0.975, top=0.855,
                          bottom=0.180)
    axp = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_c = fig.add_subplot(gs[0, 3])

    cs = None
    for i, (ax, (X, Y, dens)) in enumerate(zip(axp, fields)):
        cs = ax.contourf(X, Y, dens, levels=levels, cmap=cmap, extend="max")
        cs.set_edgecolor("face")
        ax.contour(X, Y, dens, levels=levels[2::3], colors=[COLORS["rule"]],
                   linewidths=0.4, alpha=0.7)
        frame_pole(ax, POLES[i][1])
        ax.text(-1.20, -1.12, f"max {np.nanmax(dens):.1f} m.r.d.",
                fontsize=6.6, color=COLORS["ink"], ha="left", va="bottom")

    for ax in axp:
        ax.apply_aspect()
    p0, p2 = axp[0].get_position(), axp[2].get_position()
    cax = fig.add_axes([p0.x0 + 0.02, p0.y0 - 0.075,
                        p2.x1 - p0.x0 - 0.04, 0.030])
    cb = fig.colorbar(cs, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(COLORS["rule"])
    cb.ax.tick_params(labelsize=6.8, width=0.7, length=2.5,
                      color=COLORS["rule"])
    cb.set_ticks(np.arange(0, levels[-1] + 0.1, 2.0))
    cb.set_label("orientation density (multiples of random distribution)",
                 fontsize=7.0, color=COLORS["ink"])

    # ---- (d) component fractions -----------------------------------------
    vol = 100.0 * counts / counts.sum()
    names = [c[0] for c in COMPONENTS]
    cols = [c[3] for c in COMPONENTS]
    y = np.arange(len(names))[::-1]
    ax_c.barh(y, vol, color=cols, edgecolor=COLORS["rule"], linewidth=0.5,
              height=0.62)
    for yi, v, ideal in zip(y, vol, IDEALS):
        ax_c.text(v + 0.9, yi, f"{v:.1f} %  {ideal}", va="center", ha="left",
                  fontsize=6.4, color=COLORS["rule"])
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(names, fontsize=7.0)
    ax_c.set_xlim(0, vol.max() * 1.95)
    ax_c.set_xlabel("volume fraction (%)", fontsize=7.5)
    ax_c.tick_params(labelsize=6.6, length=2.5)
    ax_c.grid(axis="x", linewidth=0.5, alpha=0.7)
    style_axes(ax_c)
    ax_c.set_title("Texture components", fontsize=8.5, fontweight="bold",
                   pad=2.0)
    panel_labels_inches(fig, axp + [ax_c], dx=-0.14, dy=0.05)

    savefig(fig, "392_mat_pole_figure.png")


if __name__ == "__main__":
    main()

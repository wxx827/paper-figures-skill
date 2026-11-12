"""Figure 400 - UAV field-phenotyping pipeline, three trial blocks x five stages.

Each stage sits in its own dashed rounded frame -- Flight, Orthomosaic, Plot
segmentation, Trait extraction, Genotype ranking -- with the final ranking
column emphasised.  Every panel in a row is derived from the same synthetic
micro-plot trial, so the plot polygons, the trait scatter and the ranking are
genuinely consistent with the orthomosaic to their left.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, get_cmap

SEED = 8402
N = 190                     # orthomosaic side, px
NC, NR = 8, 6               # micro-plots across / down
PW, PH = 19.0, 25.0         # plot pitch, px
ALLEY = 4.0                 # alley width, px
N_GENO = 12

STAGES = ["Flight", "Orthomosaic", "Plot segmentation", "Trait extraction",
          "Genotype ranking"]
BLOCKS = [("Block A", 12.0), ("Block B", -21.0), ("Block C", 5.0)]

NDVI_CMAP = get_cmap("cb_sequential_green")
SOIL = "#b09a7c"
ACCENT = "#b23a48"
RULE = "#333333"
NDVI_LO, NDVI_HI = 0.28, 0.88


def _noise(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def rot_fwd(x, y, cx, cy, th):
    u = (x - cx) * np.cos(th) + (y - cy) * np.sin(th) + N / 2.0
    v = -(x - cx) * np.sin(th) + (y - cy) * np.cos(th) + N / 2.0
    return u, v


def rot_back(u, v, cx, cy, th):
    uu, vv = u - N / 2.0, v - N / 2.0
    return (cx + uu * np.cos(th) - vv * np.sin(th),
            cy + uu * np.sin(th) + vv * np.cos(th))


def build_block(rng, ang, geno_eff):
    """Trial layout -> (rgb ortho, plot records, geometry)."""
    th = np.deg2rad(ang)
    cx = cy = N / 2.0
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    u, v = rot_fwd(xx, yy, cx, cy, th)

    u0 = N / 2.0 - NC * PW / 2.0
    v0 = N / 2.0 - NR * PH / 2.0
    pi_ = np.floor((u - u0) / PW).astype(int)
    pj = np.floor((v - v0) / PH).astype(int)
    inside = (pi_ >= 0) & (pi_ < NC) & (pj >= 0) & (pj < NR)
    alley = (np.mod(u - u0, PW) < ALLEY) | (np.mod(v - v0, PH) < ALLEY)
    canopy = inside & ~alley

    # Genotype assignment: every genotype gets four replicate plots.
    order = rng.permutation(NC * NR)
    geno = np.empty(NC * NR, int)
    geno[order] = np.repeat(np.arange(N_GENO), (NC * NR) // N_GENO)

    field_grad = 0.055 * _noise(rng, N, 34)
    pid = np.where(inside, pj * NC + pi_, -1)

    recs = []
    ndvi_px = np.zeros((N, N))
    for k in range(NC * NR):
        m = (pid == k) & canopy
        if not m.any():
            continue
        base = 0.615 + geno_eff[geno[k]] + float(field_grad[m].mean())
        base += rng.normal(0.0, 0.022)
        nd = np.clip(base + 0.035 * _noise(rng, N, 1.6), NDVI_LO, NDVI_HI)
        ndvi_px[m] = nd[m]
        h = np.clip(0.42 + 2.05 * (base - 0.60) + rng.normal(0, 0.035),
                    0.30, 1.30)
        recs.append(dict(k=k, geno=geno[k], ndvi=float(nd[m].mean()),
                         height=float(h)))

    tex = _noise(rng, N, 1.8)
    rgb = np.zeros((N, N, 3))
    soil = np.array(plt.matplotlib.colors.to_rgb(SOIL))
    rgb[...] = soil * (0.92 + 0.16 * tex)[..., None]
    veg = NDVI_CMAP((np.clip(ndvi_px, NDVI_LO, NDVI_HI) - NDVI_LO)
                    / (NDVI_HI - NDVI_LO))[..., :3]
    rgb = np.where(canopy[..., None], veg * (0.90 + 0.18 * tex)[..., None],
                   rgb)
    vig = 1.0 - 0.09 * (((xx / N - 0.5) ** 2 + (yy / N - 0.5) ** 2) / 0.5)
    rgb = np.clip(rgb * vig[..., None], 0.0, 1.0)
    return rgb, recs, dict(th=th, cx=cx, cy=cy, u0=u0, v0=v0)


def plot_corners(k, g):
    i, j = k % NC, k // NC
    us = g["u0"] + i * PW + ALLEY
    ue = g["u0"] + (i + 1) * PW
    vs = g["v0"] + j * PH + ALLEY
    ve = g["v0"] + (j + 1) * PH
    pts = [(us, vs), (ue, vs), (ue, ve), (us, ve)]
    return np.array([rot_back(uu, vv, g["cx"], g["cy"], g["th"])
                     for uu, vv in pts])


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def mini(ax):
    ax.tick_params(labelsize=6.0, width=0.6, length=2.0, pad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.7)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)
    geno_eff = np.sort(rng.normal(0.0, 0.062, N_GENO))[::-1]
    geno_eff = geno_eff[rng.permutation(N_GENO)]

    FW, FH = 12.6, 7.6
    L, R = 0.052, 0.010
    gapx, gapy = 0.019, 0.024
    top, bot = 0.868, 0.088
    cw = (1.0 - L - R - 4 * gapx) / 5.0
    rh = (top - bot - 2 * gapy) / 3.0

    fig = plt.figure(figsize=(FW, FH))

    def cell(r, c):
        return (L + c * (cw + gapx), top - (r + 1) * rh - r * gapy, cw, rh)

    def img_axes(r, c):
        x0, y0, w, h = cell(r, c)
        return fig.add_axes([x0 + 0.007, y0 + 0.036, w - 0.014, h - 0.052])

    def chart_axes(r, c):
        x0, y0, w, h = cell(r, c)
        return fig.add_axes([x0 + 0.030, y0 + 0.052, w - 0.042, h - 0.082])

    rank_pool = []
    for r, (bname, ang) in enumerate(BLOCKS):
        rgb, recs, g = build_block(rng, ang, geno_eff)
        nd = np.array([x["ndvi"] for x in recs])
        ht = np.array([x["height"] for x in recs])
        gi = np.array([x["geno"] for x in recs])

        # ---- stage 1: flight plan ----------------------------------------
        ax = img_axes(r, 0)
        ax.imshow(np.ones((N, N, 3)) * 0.985, interpolation="nearest")
        outline = np.array([rot_back(u, v, g["cx"], g["cy"], g["th"])
                            for u, v in [(g["u0"] - 7, g["v0"] - 7),
                                         (g["u0"] + NC * PW + 7, g["v0"] - 7),
                                         (g["u0"] + NC * PW + 7,
                                          g["v0"] + NR * PH + 7),
                                         (g["u0"] - 7, g["v0"] + NR * PH + 7)]])
        ax.add_patch(Polygon(outline, closed=True, facecolor="#e8ecdf",
                             edgecolor="#7d8a6a", lw=0.9))
        n_line = 7
        vs = np.linspace(g["v0"] - 2, g["v0"] + NR * PH + 2, n_line)
        path = []
        for i, vv in enumerate(vs):
            ends = [g["u0"] - 4, g["u0"] + NC * PW + 4]
            if i % 2:
                ends = ends[::-1]
            path += [rot_back(e, vv, g["cx"], g["cy"], g["th"]) for e in ends]
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], color="#3b5b92", lw=0.9,
                solid_joinstyle="round", zorder=4)
        ax.plot(path[::2, 0], path[::2, 1], marker="o", ms=1.8, ls="none",
                color="#3b5b92", zorder=5)
        ax.plot([path[0, 0]], [path[0, 1]], marker="X", ms=5.5, mfc=ACCENT,
                mec="white", mew=0.7, zorder=6)
        ax.text(0.5, -0.055, f"{n_line} lines  |  75 m AGL  |  80/70% overlap",
                transform=ax.transAxes, ha="center", va="top", fontsize=6.2,
                color="#555555")
        ax.set_xlim(0, N)
        ax.set_ylim(N, 0)
        ax.set_aspect("equal")
        clean(ax)

        # ---- stage 2: orthomosaic ----------------------------------------
        ax = img_axes(r, 1)
        ax.imshow(rgb, interpolation="bilinear")
        ax.set_aspect("equal")
        clean(ax)
        ax.text(0.5, -0.055, "2.1 cm GSD  |  RGB + NIR",
                transform=ax.transAxes, ha="center", va="top", fontsize=6.2,
                color="#555555")

        # ---- stage 3: plot segmentation ----------------------------------
        ax = img_axes(r, 2)
        ax.imshow(rgb * 0.30 + 0.68, interpolation="bilinear")
        for rec in recs:
            c = plot_corners(rec["k"], g)
            ax.add_patch(Polygon(
                c, closed=True,
                facecolor=NDVI_CMAP((rec["ndvi"] - NDVI_LO)
                                    / (NDVI_HI - NDVI_LO)),
                edgecolor=ACCENT, lw=0.45, zorder=4))
        ax.set_xlim(0, N)
        ax.set_ylim(N, 0)
        ax.set_aspect("equal")
        clean(ax)
        ax.text(0.5, -0.055, f"{len(recs)} micro-plots delineated",
                transform=ax.transAxes, ha="center", va="top", fontsize=6.2,
                color="#555555")

        # ---- stage 4: trait extraction -----------------------------------
        ax = chart_axes(r, 3)
        ax.scatter(nd, ht, s=14, c=nd, cmap=NDVI_CMAP, vmin=NDVI_LO,
                   vmax=NDVI_HI, edgecolor=RULE, linewidth=0.35, zorder=4)
        b, a = np.polyfit(nd, ht, 1)
        xs = np.linspace(nd.min(), nd.max(), 20)
        ax.plot(xs, a + b * xs, color=RULE, lw=0.9, ls=(0, (3.0, 1.8)),
                zorder=5)
        rr = float(np.corrcoef(nd, ht)[0, 1])
        ax.text(0.03, 0.96, f"r = {rr:.2f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=6.2, color="#555555")
        ax.set_xlabel("plot NDVI (-)", fontsize=6.6, labelpad=1.5)
        ax.set_ylabel("canopy height (m)", fontsize=6.6, labelpad=1.5)
        ax.set_xlim(0.44, 0.86)
        ax.set_ylim(0.28, 1.12)
        mini(ax)

        # ---- stage 5: genotype ranking -----------------------------------
        ax = chart_axes(r, 4)
        gm = np.array([nd[gi == k].mean() for k in range(N_GENO)])
        gs = np.array([nd[gi == k].std() for k in range(N_GENO)])
        best = np.argsort(gm)[::-1][:6]
        cols = [ACCENT if i == 0 else "#2e7d5b" for i in range(6)]
        ypos = np.arange(6)
        ax.barh(ypos, gm[best], xerr=gs[best], height=0.62, color=cols,
                edgecolor="white", linewidth=0.5,
                error_kw=dict(ecolor=RULE, lw=0.6, capsize=1.6), zorder=3)
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"G{k + 1:02d}" for k in best], fontsize=6.0)
        ax.invert_yaxis()
        ax.set_xlim(0.50, 0.86)
        ax.set_xticks([0.55, 0.65, 0.75, 0.85])
        ax.set_xlabel("mean plot NDVI (-)", fontsize=6.6, labelpad=1.5)
        mini(ax)
        ax.grid(axis="x", lw=0.4, alpha=0.5)
        rank_pool.append(int(best[0]))

        x0, y0, _, _ = cell(r, 0)
        fig.text(L - 0.011, y0 + rh / 2.0, f"{bname}\n{len(recs)} plots",
                 rotation=90, fontsize=7.4, va="center", ha="right",
                 color="#333333", linespacing=1.5)

    # ---- stage frames, headers and arrows ---------------------------------
    for c in range(5):
        last = c == 4
        ec = ACCENT if last else "#9a9a9a"
        for r in range(3):
            x0, y0, w, h = cell(r, c)
            fig.patches.append(FancyBboxPatch(
                (x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.010",
                transform=fig.transFigure, facecolor="none", edgecolor=ec,
                linewidth=1.05 if last else 0.85,
                linestyle=(0, (3.2, 2.0)), zorder=6))
            if c < 4:
                xa = x0 + w
                fig.patches.append(FancyArrowPatch(
                    (xa + 0.0032, y0 + h / 2), (xa + gapx - 0.0032,
                                                y0 + h / 2),
                    transform=fig.transFigure, arrowstyle="-|>",
                    mutation_scale=10.0, linewidth=1.9, color="#5b6470",
                    zorder=7))
        x0, _, w, _ = cell(0, c)

    cax = fig.add_axes([L + 0.028, 0.038, 0.145, 0.014])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(NDVI_LO, NDVI_HI),
                                     cmap=NDVI_CMAP), cax=cax,
                      orientation="horizontal", ticks=[0.3, 0.55, 0.85])
    cb.set_label("plot NDVI (-)", fontsize=7.0, color="#1a1a1a", labelpad=1.2)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(RULE)
    cb.ax.tick_params(labelsize=6.6, width=0.7, length=2.5, color=RULE)

    winner = np.bincount(rank_pool).argmax()

    savefig(fig, "400_agr_uav_phenotyping.png")


if __name__ == "__main__":
    main()

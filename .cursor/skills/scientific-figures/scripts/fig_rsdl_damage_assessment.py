"""Figure 336 - Post-disaster building damage assessment.

Bi-temporal very-high-resolution pair, the change response derived from it,
a pixel-level damage grade map and the building-level grading that a paper
would actually report, plus the per-grade building counts.  One red zoom
window, at the same place in every image panel, sits over the worst-hit
block.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Patch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from scipy.ndimage import gaussian_filter, label as cc_label, center_of_mass

from pubstyle import set_pub_style, PALETTES, savefig, add_panel_label, \
    style_axes

ZOOM_RED = "#b23a48"
GSD = 0.5
DAMAGE = ["No damage", "Minor damage", "Major damage", "Destroyed"]
DAMAGE_COLORS = ["#9db98a", "#f4c088", "#cc722f", "#8c2f2f"]

DISTRICTS = [
    (14.0, 34.0, 22.0, 128.0, 6.4, 0.16),
    (-38.0, 29.0, 17.0, 104.0, 5.8, 0.20),
    (62.0, 52.0, 44.0, 150.0, 7.0, 0.34),
]


def _noise(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _fbm(rng, n, scales=(3, 8, 20), weights=(0.5, 0.32, 0.18)):
    out = sum(w * _noise(rng, n, s) for s, w in zip(scales, weights))
    return out / (out.std() + 1e-9)


def _rot(xx, yy, ang):
    th = np.deg2rad(ang)
    return xx * np.cos(th) + yy * np.sin(th), -xx * np.sin(th) + yy * np.cos(th)


def urban_layout(n, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    anchors = rng.uniform(0.12, 0.88, size=(3, 2)) * n
    dists = np.stack([np.hypot(xx - a[0], yy - a[1]) + 14 * _noise(rng, n, 26)
                      for a in anchors])
    reg = np.argmin(dists, axis=0)

    build = np.zeros((n, n), bool)
    road = np.zeros((n, n), bool)
    bid = np.zeros((n, n), int)
    for r, (ang, cu, pv, cv, wr, skip) in enumerate(DISTRICTS):
        here = reg == r
        if not here.any():
            continue
        jit = 1.8 * _noise(rng, n, 16)
        u, v = _rot(xx, yy, ang + 3.0 * rng.standard_normal())
        u, v = u + jit, v + jit
        fu, fv = np.mod(u, cu), np.mod(v, pv)
        street = (fu < wr) | (np.mod(v, cv) < wr + 1.6)
        iu = np.floor(u / cu).astype(int)
        iv = np.floor(v / pv).astype(int)
        key = np.abs(iu * 7919 + iv * 131 + r * 104729) % 4096
        lut_skip, lut_d, lut_l = (rng.random(4096) for _ in range(3))
        setback = 2.2
        depth = wr + setback + (cu - wr - 2 * setback) * (0.58 + 0.40 * lut_d[key])
        gap = 2.0 + 3.2 * lut_l[key]
        inside = (fu > wr + setback) & (fu < depth) & (fv > gap) & (fv < pv - gap)
        b = here & inside & (lut_skip[key] > skip) & ~street
        build |= b
        road |= here & street
        bid = np.where(b, iu * 7919 + iv * 131 + r * 104729, bid)
    return build, road, bid, rng


def render(build, road, bid, rubble, seed, n):
    """Turn masks into an RGB scene; ``rubble`` marks collapsed roof pixels."""
    rng = np.random.default_rng(seed)
    veg = gaussian_filter((_fbm(rng, n, (4, 11, 26)) > 0.72).astype(float),
                          1.3) > 0.45
    veg &= ~build & ~road
    soil = (_noise(rng, n, 8) > 0.85) & ~build & ~road & ~veg

    bands = np.zeros((3, n, n))
    base = {"grass": [0.088, 0.132, 0.086], "soil": [0.215, 0.198, 0.170],
            "forest": [0.038, 0.070, 0.042], "road": [0.130, 0.134, 0.140]}
    for k in range(3):
        bands[k] = base["grass"][k]
        bands[k][soil] = base["soil"][k]
        bands[k][veg] = base["forest"][k]
        bands[k][road] = base["road"][k]

    key_b = np.abs(bid) % 4096
    lut_roof, lut_hue = rng.random(4096), rng.random(4096)
    bright = 0.13 + 0.26 * lut_roof[key_b]
    hue = lut_hue[key_b]
    terra, metal = hue > 0.70, hue < 0.14
    roofs = (bright * (1.0 + 0.42 * terra - 0.10 * metal),
             bright * (1.0 - 0.14 * terra - 0.02 * metal),
             bright * (1.0 - 0.34 * terra + 0.08 * metal))
    for k, rf in enumerate(roofs):
        bands[k] = np.where(build, rf, bands[k])

    intact = build & ~rubble
    sh = np.zeros((n, n), bool)
    for d in range(1, 5):
        sh |= np.roll(np.roll(intact, d, axis=0), d, axis=1)
    sh &= ~build
    for k in range(3):
        bands[k] = np.where(sh, bands[k] * 0.42 + 0.012, bands[k])

    # Speckle is always drawn so the pre/post random streams stay aligned and
    # the change response is genuinely zero outside the damaged roofs.
    speck = 0.5 + 0.5 * np.clip(_noise(rng, n, 0.7), -2, 2) / 2.0
    for k, tint in enumerate((0.215, 0.196, 0.172)):
        bands[k] = np.where(rubble, tint * (0.55 + 0.85 * speck), bands[k])

    rng2 = np.random.default_rng(seed + 1000)
    tex = 1 + 0.12 * _noise(rng2, n, 1.0) + 0.06 * _noise(rng2, n, 3.0)
    illum = 1 + 0.05 * _noise(rng2, n, 30)
    for k in range(3):
        bands[k] = (bands[k] * tex * illum
                    + 0.0035 * rng2.standard_normal((n, n)))
    return bands


def to_rgb_img(bands, lo, hi):
    return np.dstack([np.clip((bands[i] - lo) / max(hi - lo, 1e-9), 0, 1) ** g
                      for i, g in zip((0, 1, 2), (0.78, 0.78, 0.84))])


def image_axes(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")


def add_zoom(ax, disp, box, loc=(0.605, 0.020, 0.375, 0.375), cmap=None,
             vmin=None, vmax=None):
    x0, y0, w, h = box
    ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), w, h, fill=False,
                           edgecolor=ZOOM_RED, linewidth=1.0, zorder=7))
    ins = ax.inset_axes(loc)
    ins.set_zorder(8)
    ins.imshow(disp[y0:y0 + h, x0:x0 + w], interpolation="nearest",
               cmap=cmap, vmin=vmin, vmax=vmax)
    ins.set_xticks([])
    ins.set_yticks([])
    for s in ins.spines.values():
        s.set_visible(True)
        s.set_linewidth(1.1)
        s.set_color(ZOOM_RED)
    return ins


def add_scale_bar(ax, n, length_m=50.0, label="50 m"):
    px = length_m / GSD
    x0, y0 = 0.032 * n, 0.935 * n
    h = 0.016 * n
    for i in range(2):
        ax.add_patch(Rectangle((x0 + i * px / 2, y0), px / 2, h, zorder=6,
                               facecolor="white" if i else "#111111",
                               edgecolor="#111111", linewidth=0.5))
    ax.text(x0 + px / 2, y0 - 0.012 * n, label, ha="center", va="bottom",
            fontsize=6.0, color="white", zorder=7,
            path_effects=[pe.withStroke(linewidth=1.7, foreground="#1a1a1a")])


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    n = 208
    build, road, bid, _ = urban_layout(n, seed=31)
    inst, k = cc_label(build)
    cen = np.array(center_of_mass(build, inst, np.arange(1, k + 1)))
    rng = np.random.default_rng(6)

    # --- damage severity follows a tornado-like corridor -------------------
    t = np.linspace(0, 1, 200)
    trk_x = 0.10 * n + 0.85 * n * t
    trk_y = 0.72 * n - 0.42 * n * t + 0.06 * n * np.sin(4.5 * t)
    d = np.min(np.hypot(cen[:, 1][:, None] - trk_x[None, :],
                        cen[:, 0][:, None] - trk_y[None, :]), axis=1)
    sev = np.clip(1.25 * np.exp(-0.5 * (d / (0.135 * n)) ** 2)
                  + 0.16 * rng.standard_normal(k), 0, 1.4)
    level = np.digitize(sev, [0.20, 0.46, 0.78])

    lvl_map = np.zeros(k + 1, int)
    lvl_map[1:] = level
    lvl_img = lvl_map[inst]

    rubble = np.zeros((n, n), bool)
    half = _noise(rng, n, 5.0) > 0.0
    rubble |= build & (lvl_img == 3)
    rubble |= build & (lvl_img == 2) & half

    pre_b = render(build, road, bid, np.zeros((n, n), bool), 12, n)
    post_b = render(build, road, bid, rubble, 12, n)
    debris = gaussian_filter((lvl_img >= 2).astype(float), 3.0)
    for c in range(3):
        post_b[c] = post_b[c] * (1 - 0.22 * debris) + 0.055 * debris

    lo, hi = np.percentile(np.concatenate([pre_b.ravel(), post_b.ravel()]),
                           [1.2, 99.0])
    pre = to_rgb_img(pre_b, lo, hi)
    post = to_rgb_img(post_b, lo, hi)
    change = gaussian_filter(np.abs(post - pre).mean(axis=2), 1.0)

    # --- predicted grades (a few off-by-one confusions) --------------------
    pred = level.copy()
    flip = rng.random(k)
    step = np.where(rng.random(k) > 0.5, 1, -1)
    pred = np.where(flip < 0.13, np.clip(level + step, 0, 3), level)
    oa = float(np.mean(pred == level))
    pred_map = np.zeros(k + 1, int)
    pred_map[1:] = pred
    pred_img = pred_map[inst]
    print(f"  buildings = {k}   grade OA = {oa:.3f}")

    lut = np.array([(1.0, 1.0, 1.0)] + [to_rgb(c) for c in DAMAGE_COLORS])

    def grade_overlay(rgbimg, lvl, alpha=0.60):
        out = rgbimg.copy()
        m = inst > 0
        cols = lut[np.clip(lvl, 0, 3) + 1]
        out[m] = (1 - alpha) * rgbimg[m] + alpha * cols[m]
        return np.clip(out, 0, 1)

    def grade_solid(lvl, base):
        grey = base.mean(axis=2, keepdims=True) * 0.55 + 0.40
        out = np.repeat(grey, 3, axis=2)
        m = inst > 0
        out[m] = lut[np.clip(lvl, 0, 3) + 1][m]
        return np.clip(out, 0, 1)

    cmap_ch = LinearSegmentedColormap.from_list(
        "ch", PALETTES["cb_sequential_warm"])

    box = (108, 42, 60, 60)
    fig = plt.figure(figsize=(9.6, 6.9))
    gs = fig.add_gridspec(2, 3, wspace=0.035, hspace=0.135,
                          left=0.045, right=0.985, top=0.876, bottom=0.108)

    panels = [
        ("Pre-disaster", pre, None),
        ("Post-disaster", post, None),
        ("Change response", change, cmap_ch),
        ("Pixel-level grade", grade_overlay(post, lvl_img), None),
        ("Building-level grade", grade_solid(pred_img, post), None),
    ]
    axes = []
    for i, (title, img, cm) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        axes.append(ax)
        image_axes(ax)
        ax.patch.set_visible(False)
        if cm is None:
            ax.imshow(img, interpolation="nearest")
            add_zoom(ax, img, box)
        else:
            im = ax.imshow(img, cmap=cm, vmin=0.0, vmax=0.55,
                           interpolation="nearest")
            add_zoom(ax, img, box, cmap=cm, vmin=0.0, vmax=0.55)
            ax.add_patch(FancyBboxPatch(
                (0.030, 0.800), 0.545, 0.176, transform=ax.transAxes,
                boxstyle="round,pad=0.008,rounding_size=0.02",
                facecolor="white", edgecolor="#8a8a8a", linewidth=0.5,
                alpha=0.93, zorder=8))
            cax = ax.inset_axes([0.062, 0.868, 0.462, 0.032])
            cax.set_zorder(9)
            cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                              ticks=[0.0, 0.25, 0.5])
            cb.outline.set_linewidth(0.5)
            cb.outline.set_edgecolor("#333333")
            cb.ax.tick_params(labelsize=6.0, width=0.5, length=1.8, pad=1.0,
                              color="#333333")
            ax.text(0.062, 0.944, "change magnitude (-)",
                    transform=ax.transAxes, fontsize=6.0, ha="left",
                    va="center", color="#1a1a1a", zorder=9,
                    fontweight="bold")
        ax.set_title(title, fontsize=8.0, fontweight="bold", pad=3.4)
        add_panel_label(ax, f"({'abcde'[i]})", x=0.0, y=1.030, size=9.5)
        if i == 0:
            add_scale_bar(ax, n)

    # ---- (f) per-grade building counts ------------------------------------
    ax_b = fig.add_subplot(gs[1, 2])
    ref_ct = np.array([(level == i).sum() for i in range(4)])
    prd_ct = np.array([(pred == i).sum() for i in range(4)])
    x = np.arange(4)
    ax_b.bar(x - 0.19, ref_ct, 0.34, color=DAMAGE_COLORS,
             edgecolor="#333333", linewidth=0.5, label="Reference")
    ax_b.bar(x + 0.19, prd_ct, 0.34, color=DAMAGE_COLORS, alpha=0.55,
             edgecolor="#333333", linewidth=0.5, hatch="///",
             label="Predicted")
    for xi, a, b in zip(x, ref_ct, prd_ct):
        ax_b.text(xi - 0.19, a + 0.8, str(a), ha="center", va="bottom",
                  fontsize=6.4, color="#333333")
        ax_b.text(xi + 0.19, b + 0.8, str(b), ha="center", va="bottom",
                  fontsize=6.4, color="#666666")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(["None", "Minor", "Major", "Destroyed"],
                         fontsize=7.0)
    ax_b.set_ylim(0, ref_ct.max() * 1.42)
    ax_b.set_ylabel("building count (-)", fontsize=7.4, labelpad=2)
    ax_b.tick_params(labelsize=6.8, length=2, pad=1.5)
    ax_b.grid(axis="y", linewidth=0.4, alpha=0.6)
    ax_b.legend(frameon=False, fontsize=6.6, loc="upper right",
                handlelength=1.3, borderaxespad=0.2)
    style_axes(ax_b)
    ax_b.set_title("Grade distribution", fontsize=8.0, fontweight="bold",
                   pad=3.4)
    add_panel_label(ax_b, "(f)", x=-0.16, y=1.030, size=9.5)
    ax_b.text(0.02, 0.985, f"grade OA {oa:.3f}\n{k} buildings",
              transform=ax_b.transAxes, ha="left", va="top", fontsize=6.6,
              color="#1a1a1a",
              bbox=dict(boxstyle="round,pad=0.32", facecolor="#f2f5f8",
                        edgecolor="#b9c6d1", linewidth=0.6))

    handles = [Patch(facecolor=c, edgecolor="#333333", linewidth=0.4,
                     label=nm) for c, nm in zip(DAMAGE_COLORS, DAMAGE)]
    handles.append(Patch(facecolor="none", edgecolor=ZOOM_RED, linewidth=1.0,
                         label="Zoom window (same in all panels)"))
    leg = fig.legend(handles=handles, loc="lower center", ncol=5,
                     frameon=False, fontsize=7.0, handlelength=1.3,
                     handleheight=1.0, columnspacing=1.2,
                     bbox_to_anchor=(0.52, 0.006))
    leg.set_clip_on(False)


    savefig(fig, "336_rsdl_damage_assessment.png")


if __name__ == "__main__":
    main()

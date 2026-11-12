"""Figure 543 - Spatial distribution of a nuclear IHC biomarker.

Panel (a) is a 40x immunohistochemistry field: DAB-positive nuclei in brown,
negative nuclei in haematoxylin blue, composited through the Beer-Lambert law
with the standard DAB and haematoxylin optical-density vectors.  Panel (b)
maps positive-cell density over the whole section and marks the hot spot.
The three subordinate panels underneath and beside them carry the evidence:
how density falls from the tumour centre outward, how the per-nucleus DAB
optical density separates positive from negative, and how positivity differs
between tumour core, invasive front and peritumoural stroma.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import (gaussian_filter, distance_transform_edt,
                           uniform_filter)

from pubstyle import set_pub_style, savefig, get_cmap, style_axes, annotate_box
from figkit import (image_axes, add_scale_bar, halo_text, safe_colorbar,
                    panel_labels_inches)

OD_H = np.array([0.650, 0.704, 0.286])
OD_DAB = np.array([0.268, 0.570, 0.776])
GLASS = np.array([0.988, 0.982, 0.988])

ACCENT = "#b23a48"          # hot spot marker - the single emphasis object
POS = "#8c4a12"             # DAB brown, for the positive series
NEG = "#3b5b92"             # haematoxylin blue, for the negative series
MPP_SEC = 10.0              # section overview, micrometres per pixel
MPP_IHC = 0.25              # 40x field
ZONES = ["Tumour core", "Mid tumour", "Invasive front", "Peritumoural"]
ZONE_EDGES = [-0.90, -0.26, -0.08, 0.08, 0.55]      # signed distance, mm


def _noise(rng, shape, sigma):
    a = gaussian_filter(rng.standard_normal(shape), sigma, mode="reflect")
    return a / (a.std() + 1e-9)


def _fbm(rng, shape, scales, weights):
    out = sum(w * _noise(rng, shape, s) for s, w in zip(scales, weights))
    return out / (out.std() + 1e-9)


def _blob(shape, fy, fx, ry, rx):
    hh, ww = shape
    yy, xx = np.mgrid[0:hh, 0:ww].astype(float)
    return np.exp(-0.5 * (((yy - fy * hh) / (ry * hh)) ** 2
                          + ((xx - fx * ww) / (rx * ww)) ** 2))


def ihc_rgb(ch, cd, desat=0.08):
    od = (np.clip(ch, 0, None)[..., None] * OD_H
          + np.clip(cd, 0, None)[..., None] * OD_DAB)
    rgb = np.exp(-od)
    lum = rgb @ np.array([0.30, 0.59, 0.11])
    return np.clip(rgb * (1.0 - desat) + lum[..., None] * desat, 0, 1)


# ---------------------------------------------------------------------------
def build_section(shape=(560, 520), seed=29):
    rng = np.random.default_rng(seed)
    hh, ww = shape
    yy, xx = np.mgrid[0:hh, 0:ww].astype(float)
    cy, cx = 0.50 * hh, 0.50 * ww
    th = np.arctan2(yy - cy, xx - cx)
    rr = (np.hypot((yy - cy) / (0.440 * hh), (xx - cx) / (0.435 * ww))
          + 0.030 * _noise(rng, shape, 6.0))
    edge = (1.0 + 0.130 * np.sin(3 * th + 2.0) + 0.090 * np.sin(5 * th - 0.8)
            + 0.055 * np.sin(9 * th + 1.2))
    tissue = np.clip((edge - rr) / 0.045, 0, 1)

    f1 = _fbm(rng, shape, (14, 30, 52), (0.44, 0.34, 0.22))
    nests = (1.15 * _blob(shape, 0.42, 0.44, 0.215, 0.230)
             + 0.82 * _blob(shape, 0.74, 0.64, 0.130, 0.140))
    p_tum = np.clip((nests + 0.22 * f1 - 0.48) / 0.26, 0, 1) * tissue
    p_tum = gaussian_filter(p_tum, 3.0)

    # Proliferation is highest at the advancing edge of the nest and falls
    # toward the hypoxic centre - the pattern the profile panel exists to show.
    tum = p_tum > 0.5
    d_in = distance_transform_edt(tum) * MPP_SEC / 1000.0        # mm inside
    d_out = distance_transform_edt(~tum) * MPP_SEC / 1000.0      # mm outside
    signed = d_out - d_in                                        # <0 = inside
    rimness = np.exp(-0.5 * ((signed + 0.09) / 0.20) ** 2)
    expr = np.clip(0.07 + 0.88 * rimness * p_tum
                   + 0.10 * p_tum * (0.5 + 0.5 * f1)
                   + 0.05 * _fbm(rng, shape, (10, 22), (.6, .4)), 0, 1)
    expr = gaussian_filter(expr, 4.0) * tissue

    dens_all = np.clip(1100 + 3600 * p_tum, 100, 9000)           # cells/mm2
    dens_pos = gaussian_filter(dens_all * expr, 6.0) * tissue
    return dict(tissue=tissue, p_tum=p_tum, expr=expr, signed=signed,
                dens_all=dens_all, dens_pos=dens_pos)


def ihc_field(sec, shape, y0, x0, seed=61):
    """40x IHC tile sampled from the section fields at ``(y0, x0)``."""
    rng = np.random.default_rng(seed)
    hh, ww = shape
    fac = MPP_SEC / MPP_IHC
    ys = (y0 + np.arange(hh) / fac).astype(int)
    xs = (x0 + np.arange(ww) / fac).astype(int)
    take = lambda a: a[np.clip(ys, 0, a.shape[0] - 1)][
        :, np.clip(xs, 0, a.shape[1] - 1)]
    p_t = np.clip(take(sec["p_tum"])
                  + 0.18 * _fbm(rng, shape, (3, 9), (.6, .4)), 0, 1)
    expr = np.clip(take(sec["expr"])
                   + 0.10 * _fbm(rng, shape, (4, 12), (.6, .4)), 0, 1)

    cd = 0.06 + 0.10 * (1.0 - p_t)                    # faint cytoplasmic haze
    ch = 0.05 + 0.04 * p_t
    cd += 0.05 * _fbm(rng, shape, (2, 7), (.6, .4)) ** 2

    dens = np.clip(1100 + 3600 * p_t, 100, 9000) * (MPP_IHC / 1000.0) ** 2
    r_px = np.clip(1.95 + 1.55 * p_t, 1.1, None) / MPP_IHC
    peak = float(dens.max())
    n_try = int(np.clip(26.0 * peak * hh * ww, 300, 90000))
    yy = rng.integers(0, hh, n_try)
    xx = rng.integers(0, ww, n_try)
    keep = rng.random(n_try) < dens[yy, xx] / peak
    occ = np.zeros(shape, bool)
    n_pos = n_neg = 0
    od_pos, od_neg = [], []
    for y, x in zip(yy[keep], xx[keep]):
        r = r_px[y, x] * rng.uniform(0.76, 1.30)
        rr = int(max(1, round(0.94 * r)))
        ya, yb = max(0, y - rr), min(hh, y + rr + 1)
        xa, xb = max(0, x - rr), min(ww, x + rr + 1)
        if occ[ya:yb, xa:xb].any():
            continue
        occ[ya:yb, xa:xb] = True
        e = rng.uniform(1.0, 1.45)
        a, b = r * e, r / e
        th = rng.uniform(0, np.pi)
        rad = int(np.ceil(1.9 * max(a, b))) + 2
        ya, yb = max(0, y - rad), min(hh, y + rad + 1)
        xa, xb = max(0, x - rad), min(ww, x + rad + 1)
        gy, gx = np.mgrid[ya:yb, xa:xb]
        ct, st = np.cos(th), np.sin(th)
        u = ((gx - x) * ct + (gy - y) * st) / max(a, 1e-6)
        v = (-(gx - x) * st + (gy - y) * ct) / max(b, 1e-6)
        ang = np.arctan2(v, u)
        ph = rng.uniform(0, 6.28)
        d = np.hypot(u, v) / (1.0 + 0.14 * np.sin(3 * ang + ph)
                              + 0.08 * np.sin(5 * ang - ph))
        core = np.clip(1.18 - d * d, 0, 1) ** 0.55
        # A nucleus is scored positive if its own DAB optical density clears
        # the assay threshold; the histogram panel plots exactly this quantity.
        od = np.clip(0.06 + 1.95 * expr[y, x] * (0.22 + 0.78 * p_t[y, x])
                     * rng.uniform(0.10, 1.25)
                     + 0.05 * rng.standard_normal(), 0.02, 2.2)
        if od >= 0.55:
            cd[ya:yb, xa:xb] += od * core
            ch[ya:yb, xa:xb] += 0.22 * core
            n_pos += 1
            od_pos.append(od)
        else:
            ch[ya:yb, xa:xb] += (0.95 + 0.35 * rng.random()) * core
            cd[ya:yb, xa:xb] += 0.10 * core
            n_neg += 1
            od_neg.append(od)
    rgb = ihc_rgb(ch, cd)
    rgb = np.dstack([gaussian_filter(rgb[..., c], 0.5) for c in range(3)])
    rgb = np.clip(rgb + 0.007 * rng.standard_normal(rgb.shape), 0, 1)
    return rgb, n_pos, n_neg, np.array(od_pos), np.array(od_neg)


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    shape = (560, 520)
    sec = build_section(shape, seed=29)
    tis = sec["tissue"] > 0.5

    hot = gaussian_filter(sec["dens_pos"], 8.0)
    hy, hx = np.unravel_index(np.argmax(hot * tis), hot.shape)

    # Sample the shown field where tumour meets stroma and expression is
    # mid-range, so the tile actually contains both positive and negative
    # nuclei; a crop from the hot spot alone would be uniformly brown.
    tile = (420, 380)
    bh, bw = tile[0] * MPP_IHC / MPP_SEC, tile[1] * MPP_IHC / MPP_SEC
    em = uniform_filter(sec["expr"], size=int(bh))
    pm = uniform_filter(sec["p_tum"], size=int(bh))
    cost = np.abs(em - 0.46) + np.abs(pm - 0.55)
    cost[:int(bh) + 2, :] = np.inf
    cost[-int(bh) - 2:, :] = np.inf
    cost[:, :int(bw) + 2] = np.inf
    cost[:, -int(bw) - 2:] = np.inf
    fy, fx = np.unravel_index(np.argmin(cost), cost.shape)
    y0, x0 = int(fy - bh / 2), int(fx - bw / 2)
    rgb, n_pos, n_neg, od_pos, od_neg = ihc_field(sec, tile, y0, x0, seed=61)
    pos_frac = 100.0 * n_pos / max(n_pos + n_neg, 1)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.62],
                          height_ratios=[1.0, 0.46], wspace=0.38, hspace=0.42,
                          left=0.050, right=0.950, top=0.872, bottom=0.082)
    ax_ihc = fig.add_subplot(gs[0, 0])
    ax_den = fig.add_subplot(gs[0, 1])
    ax_pro = fig.add_subplot(gs[0, 2])
    ax_od = fig.add_subplot(gs[1, 0])
    ax_zn = fig.add_subplot(gs[1, 1:])

    # ---- (a) IHC field ----------------------------------------------------
    image_axes(ax_ihc, color="#bdbdbd", lw=0.6)
    ax_ihc.imshow(rgb, interpolation="bilinear", zorder=1)
    add_scale_bar(ax_ihc, 25.0 / MPP_IHC, "25 µm", loc=(0.035, 0.045),
                  height=0.020, fontsize=6.2)
    halo_text(ax_ihc, 0.968, 0.958,
              f"{n_pos} positive / {n_pos + n_neg} nuclei  =  {pos_frac:.0f} %",
              transform=ax_ihc.transAxes, ha="right", va="top", fontsize=6.6,
              color="#1a1a1a", halo="white", halo_lw=2.4)
    ax_ihc.set_title("IHC at the invasive front  (40x)", fontsize=8.5,
                     fontweight="bold", pad=3.6)

    # ---- (b) density map --------------------------------------------------
    image_axes(ax_den, color="#bdbdbd", lw=0.6)
    field = np.where(tis, sec["dens_pos"], np.nan)
    im = ax_den.imshow(field, cmap=get_cmap("cb_sequential_warm"),
                       vmin=0.0, vmax=float(np.nanmax(field)),
                       interpolation="bilinear", zorder=1)
    ax_den.contour(gaussian_filter(sec["p_tum"], 5.0), levels=[0.5],
                   colors=["#3a3a3a"], linewidths=0.7, linestyles="--",
                   zorder=4)
    ax_den.add_patch(Circle((hx, hy), 26, facecolor="none", edgecolor=ACCENT,
                            lw=1.3, zorder=6))
    halo_text(ax_den, float(np.clip(hx, 0.22 * shape[1], 0.78 * shape[1])),
              hy - 34, f"hot spot {hot[hy, hx]:,.0f} cells/mm²", ha="center",
              va="bottom", fontsize=6.4, color="white", halo=ACCENT, zorder=7)
    # The 40x field is only ~0.1 mm across, i.e. 10 px at this scale, so the
    # true-size box is ringed to make it findable without exaggerating it.
    ax_den.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), bw, bh,
                                   facecolor="none", edgecolor="#c9a227",
                                   lw=1.0, zorder=7))
    ax_den.add_patch(Circle((x0 + bw / 2, y0 + bh / 2), 19, facecolor="none",
                            edgecolor="#c9a227", lw=1.1, zorder=6))
    halo_text(ax_den, x0 + bw / 2, y0 + bh / 2 + 25, "field (a)",
              ha="center", va="top", fontsize=6.2, color="white",
              halo="#8c6a14", zorder=7)
    add_scale_bar(ax_den, 1000.0 / MPP_SEC, "1 mm", loc=(0.035, 0.045),
                  height=0.016, fontsize=6.2)
    ax_den.set_title("Positive-cell density over the section",
                     fontsize=8.5, fontweight="bold", pad=3.6)
    ax_den.apply_aspect()
    safe_colorbar(fig, im, ax_den, label="positive nuclei (cells/mm²)",
                  side="bottom", width=0.10, pad=0.06, labelsize=6.4)

    # ---- (c) centre-to-edge profile ---------------------------------------
    s = sec["signed"]
    bins = np.arange(-1.6, 0.85, 0.06)
    mid = 0.5 * (bins[:-1] + bins[1:])
    med, lo, hi = [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        v = sec["dens_pos"][tis & (s >= a) & (s < b)]
        if v.size < 12:
            med.append(np.nan)
            lo.append(np.nan)
            hi.append(np.nan)
        else:
            med.append(np.median(v))
            lo.append(np.percentile(v, 25))
            hi.append(np.percentile(v, 75))
    med, lo, hi = np.array(med), np.array(lo), np.array(hi)
    ax_pro.fill_between(mid, lo, hi, color=POS, alpha=0.20, linewidth=0.0)
    ax_pro.plot(mid, med, color=POS, lw=1.5)
    ax_pro.axvline(0.0, color="#808080", lw=0.8, ls="--")
    peak_x = mid[np.nanargmax(med)]
    ax_pro.plot([peak_x], [np.nanmax(med)], marker="o", ms=4.0, mfc="white",
                mec=ACCENT, mew=1.2, zorder=6)
    ax_pro.text(0.04, 0.96, f"peak {peak_x * 1000:+.0f} µm from the edge",
                transform=ax_pro.transAxes, fontsize=6.4, color="#333333",
                ha="left", va="top")
    ax_pro.text(0.03, 0.02, "tumour edge", fontsize=6.2, color="#555555",
                rotation=90, ha="left", va="bottom",
                transform=ax_pro.get_xaxis_transform())
    ax_pro.set_xlabel("signed distance to tumour edge (mm)", fontsize=7.2,
                      labelpad=1.8)
    ax_pro.set_ylabel("density (cells/mm²)", fontsize=7.2, labelpad=2.0)
    ax_pro.set_xlim(bins[0], bins[-1])
    ax_pro.tick_params(labelsize=6.5, length=2.2, pad=1.6)
    ax_pro.grid(linewidth=0.4, alpha=0.6)
    style_axes(ax_pro)
    ax_pro.set_title("Centre to edge", fontsize=8.2, fontweight="bold",
                     pad=3.0)

    # ---- (d) per-nucleus DAB optical density ------------------------------
    edges = np.linspace(0.0, 2.0, 44)
    ax_od.hist(od_neg, bins=edges, color=NEG, alpha=0.80, linewidth=0.0,
               label=f"negative (n={od_neg.size})")
    ax_od.hist(od_pos, bins=edges, color=POS, alpha=0.85, linewidth=0.0,
               label=f"positive (n={od_pos.size})")
    ax_od.axvline(0.55, color="#333333", lw=0.9, ls="--")
    ax_od.text(0.575, 0.94, "threshold 0.55", fontsize=6.2, color="#333333",
               ha="left", va="top", transform=ax_od.get_xaxis_transform())
    ax_od.set_xlabel("per-nucleus DAB optical density (-)", fontsize=7.2,
                     labelpad=1.8)
    ax_od.set_ylabel("nuclei", fontsize=7.2, labelpad=2.0)
    ax_od.tick_params(labelsize=6.5, length=2.2, pad=1.6)
    ax_od.legend(frameon=False, fontsize=6.4, loc="upper right",
                 handlelength=1.1, borderaxespad=0.2)
    ax_od.grid(axis="y", linewidth=0.4, alpha=0.6)
    style_axes(ax_od)
    ax_od.set_title("Positivity call", fontsize=8.2, fontweight="bold",
                    pad=3.0)

    # ---- (e) positivity by zone -------------------------------------------
    frac, err = [], []
    for a, b in zip(ZONE_EDGES[:-1], ZONE_EDGES[1:]):
        m = tis & (s >= a) & (s < b)
        v = 100.0 * sec["dens_pos"][m] / np.maximum(sec["dens_all"][m], 1)
        frac.append(float(v.mean()) if v.size else 0.0)
        err.append(float(v.std()) if v.size > 1 else 0.0)
    xs = np.arange(len(ZONES))
    ax_zn.bar(xs, frac, yerr=err, width=0.56, color=POS, alpha=0.88,
              edgecolor="#3a3a3a", linewidth=0.5,
              error_kw=dict(elinewidth=0.8, capsize=2.5, ecolor="#555555"))
    for x, f, e in zip(xs, frac, err):
        ax_zn.text(x, f + e + max(frac) * 0.045, f"{f:.0f} %", ha="center",
                   va="bottom", fontsize=6.4, color="#333333")
    ax_zn.set_xticks(xs)
    ax_zn.set_xticklabels(ZONES, fontsize=6.6)
    ax_zn.set_ylabel("positive fraction (%)", fontsize=7.2, labelpad=2.0)
    ax_zn.set_ylim(0, max(np.array(frac) + np.array(err)) * 1.30)
    ax_zn.tick_params(labelsize=6.5, length=2.2, pad=1.6)
    ax_zn.grid(axis="y", linewidth=0.4, alpha=0.6)
    style_axes(ax_zn)
    ax_zn.set_title("Positivity by zone  (mean ± sd over tiles)",
                    fontsize=8.2, fontweight="bold", pad=3.0)

    for ax in (ax_ihc, ax_den):
        ax.apply_aspect()
    panel_labels_inches(fig, [ax_ihc, ax_den, ax_pro, ax_od, ax_zn],
                        dx=-0.30, dy=0.09)

    savefig(fig, "543_path_biomarker_spatial.png")


if __name__ == "__main__":
    main()

"""Figure 544 - Histological grade, from well-formed glands to solid sheets.

Six representative fields (three grades x two cases) are synthesised from the
same generator with only the architectural parameters changed: grade 1 has
regular round glands with open lumina, grade 2 has fused and cribriform
glands, grade 3 has solid sheets and larger, more pleomorphic nuclei.  Each
tile carries the *same* red magnifier box so the architectural difference,
not the crop, is what changes down the row.

The subordinate strip underneath compares the AI grade with the reporting
pathologist: confusion matrix, per-grade agreement and the kappa statistics,
all computed from the same simulated read.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.ndimage import gaussian_filter, distance_transform_edt

from pubstyle import set_pub_style, savefig, get_cmap, style_axes
from figkit import image_axes, add_zoom_inset, halo_text, ZOOM

OD_H = np.array([0.650, 0.704, 0.286])
OD_E = np.array([0.092, 0.954, 0.283])

MPP = 0.5                                    # micrometres per pixel (20x)
TH, TW = 520, 822                            # tile = 260 x 411 micrometres
ZOOM_BOX = (96, 74, 210, 133)                # identical in all six tiles
ACCENT = "#b23a48"
GRADE_TITLES = ["Grade 1  ·  well differentiated",
                "Grade 2  ·  moderately differentiated",
                "Grade 3  ·  poorly differentiated"]
GRADES = ["G1", "G2", "G3"]

# Architecture per grade: gland spacing and radius in micrometres, lumen as a
# fraction of gland radius, boundary wobble, nuclear radius and hyperchromasia.
ARCH = {
    1: dict(spacing=104.0, radius=40.0, lumen=0.52, wob=0.09, jit=7.0,
            r_nuc=2.7, spread=0.14, amp=1.00, solid=0.0),
    2: dict(spacing=82.0, radius=37.0, lumen=0.28, wob=0.30, jit=17.0,
            r_nuc=3.4, spread=0.28, amp=1.14, solid=0.18),
    3: dict(spacing=76.0, radius=33.0, lumen=0.06, wob=0.42, jit=26.0,
            r_nuc=5.1, spread=0.42, amp=1.30, solid=0.62),
}


def _noise(rng, shape, sigma):
    a = gaussian_filter(rng.standard_normal(shape), sigma, mode="reflect")
    return a / (a.std() + 1e-9)


def _fbm(rng, shape, scales, weights):
    out = sum(w * _noise(rng, shape, s) for s, w in zip(scales, weights))
    return out / (out.std() + 1e-9)


def he_rgb(ch, ce, desat=0.10):
    od = (np.clip(ch, 0, None)[..., None] * OD_H
          + np.clip(ce, 0, None)[..., None] * OD_E)
    rgb = np.exp(-od)
    lum = rgb @ np.array([0.30, 0.59, 0.11])
    return np.clip(rgb * (1.0 - desat) + lum[..., None] * desat, 0, 1)


def architecture(shape, grade, rng):
    """Lumen / epithelium / stroma masks for one grade."""
    hh, ww = shape
    a = ARCH[grade]
    lumen = np.zeros(shape)
    epi = np.zeros(shape)
    step = a["spacing"] / MPP
    for gy in np.arange(0.35 * step, hh + step, step):
        for gx in np.arange(0.35 * step, ww + step, step):
            cy = gy + a["jit"] / MPP * rng.standard_normal()
            cx = gx + a["jit"] / MPP * rng.standard_normal()
            rad = a["radius"] / MPP * rng.uniform(0.78, 1.26)
            box = int(np.ceil(1.7 * rad)) + 2
            y0, y1 = max(0, int(cy) - box), min(hh, int(cy) + box + 1)
            x0, x1 = max(0, int(cx) - box), min(ww, int(cx) + box + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            dy, dx = yy - cy, xx - cx
            ang = np.arctan2(dy, dx)
            ph = rng.uniform(0, 6.28)
            ecc = 1.0 + a["wob"] * rng.uniform(0.0, 1.4)
            wob = (1.0 + a["wob"] * np.sin(3 * ang + ph)
                   + 0.65 * a["wob"] * np.sin(5 * ang - 1.7 * ph))
            d = np.hypot(dy / ecc, dx * ecc) / (rad * wob)
            epi[y0:y1, x0:x1] = np.maximum(epi[y0:y1, x0:x1],
                                           (d <= 1.0).astype(float))
            lf = a["lumen"] * rng.uniform(0.7, 1.25)
            lumen[y0:y1, x0:x1] = np.maximum(lumen[y0:y1, x0:x1],
                                             (d <= lf).astype(float))
    if a["solid"] > 0:                       # sheets that swallow the glands
        f = _fbm(rng, shape, (18, 40), (0.6, 0.4))
        sheet = (f > (0.90 - 1.7 * a["solid"])).astype(float)
        sheet = gaussian_filter(sheet, 3.0) > 0.45
        epi = np.maximum(epi, sheet.astype(float))
        lumen = lumen * (1.0 - 0.85 * sheet)
    epi = np.clip(epi - lumen, 0, 1)
    return gaussian_filter(lumen, 1.4), gaussian_filter(epi, 1.4)


def grade_tile(grade, seed):
    """One H&E field for the requested grade."""
    shape = (TH, TW)
    rng = np.random.default_rng(seed)
    a = ARCH[grade]
    lumen, epi = architecture(shape, grade, rng)
    stroma = np.clip(1.0 - lumen - epi, 0, 1)

    ce = 0.30 + 0.52 * stroma + 0.20 * epi - 0.26 * lumen
    ce *= 1.0 + 0.22 * _fbm(rng, shape, (1.2, 4.0, 11.0), (.42, .34, .24))
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    fib = np.sin((0.83 * xx + 0.56 * yy) / 5.0 + 2.4 * _noise(rng, shape, 8.0))
    ce += 0.11 * fib * stroma
    ch = 0.06 + 0.05 * epi

    # Nuclei sit in the epithelium, aligned tangentially to the nearest lumen.
    dist = gaussian_filter(distance_transform_edt(lumen < 0.5), 3.0)
    gy, gx = np.gradient(dist)
    ori = np.arctan2(gy, gx) + np.pi / 2.0

    dens = (900 + 8200 * epi) * (1.0 - 0.9 * lumen)
    dens_px = dens * (MPP / 1000.0) ** 2
    r_px = (a["r_nuc"] * (0.65 + 0.55 * epi)) / MPP
    peak = float(dens_px.max())
    n_try = int(np.clip(26.0 * peak * shape[0] * shape[1], 400, 120000))
    ys = rng.integers(0, shape[0], n_try)
    xs = rng.integers(0, shape[1], n_try)
    keep = rng.random(n_try) < dens_px[ys, xs] / peak
    occ = np.zeros(shape, bool)
    areas = []
    for y, x in zip(ys[keep], xs[keep]):
        r = r_px[y, x] * (1.0 + a["spread"] * rng.standard_normal())
        r = float(np.clip(r, 1.2, 3.2 * r_px[y, x]))
        rr = int(max(1, round(0.94 * r)))
        y0, y1 = max(0, y - rr), min(shape[0], y + rr + 1)
        x0, x1 = max(0, x - rr), min(shape[1], x + rr + 1)
        if occ[y0:y1, x0:x1].any():
            continue
        occ[y0:y1, x0:x1] = True
        in_epi = epi[y, x] > 0.4
        e = rng.uniform(1.05, 1.55 if in_epi else 2.4)
        aa, bb = r * e, r / e
        th = (ori[y, x] + 0.35 * rng.standard_normal() if in_epi
              else rng.uniform(0, np.pi))
        rad = int(np.ceil(1.9 * max(aa, bb))) + 2
        y0, y1 = max(0, y - rad), min(shape[0], y + rad + 1)
        x0, x1 = max(0, x - rad), min(shape[1], x + rad + 1)
        gyy, gxx = np.mgrid[y0:y1, x0:x1]
        ct, st = np.cos(th), np.sin(th)
        u = ((gxx - x) * ct + (gyy - y) * st) / max(aa, 1e-6)
        v = (-(gxx - x) * st + (gyy - y) * ct) / max(bb, 1e-6)
        ang = np.arctan2(v, u)
        ph = rng.uniform(0, 6.28)
        d = np.hypot(u, v) / (1.0 + 0.14 * np.sin(3 * ang + ph)
                              + 0.08 * np.sin(5 * ang - ph))
        ch[y0:y1, x0:x1] += (a["amp"] * (0.85 + 0.35 * epi[y, x])
                             * np.clip(1.18 - d * d, 0, 1) ** 0.55)
        ce[y0:y1, x0:x1] += 0.12 * np.clip(1.9 - d * d, 0, 1) ** 1.6
        areas.append(np.pi * aa * bb * MPP ** 2)
    ch *= 1.0 + 0.28 * _fbm(rng, shape, (0.7, 2.0), (.6, .4))
    rgb = he_rgb(ch, ce)
    rgb = np.dstack([gaussian_filter(rgb[..., c], 0.45) for c in range(3)])
    rgb = np.clip(rgb + 0.007 * rng.standard_normal(rgb.shape), 0, 1)
    gland_frac = 100.0 * float((lumen > 0.5).mean())
    return rgb, np.array(areas), gland_frac


# ---------------------------------------------------------------------------
def simulate_read(n=180, seed=13):
    """Pathologist grade vs AI grade; grade 2 is where readers disagree."""
    rng = np.random.default_rng(seed)
    truth = rng.choice([0, 1, 2], size=n, p=[0.30, 0.44, 0.26])
    conf = np.array([[0.88, 0.11, 0.01],
                     [0.13, 0.74, 0.13],
                     [0.01, 0.15, 0.84]])
    ai = np.array([rng.choice(3, p=conf[t]) for t in truth])
    cm = np.zeros((3, 3), int)
    for t, p in zip(truth, ai):
        cm[t, p] += 1
    return cm


def kappas(cm):
    n = cm.sum()
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    k = (po - pe) / (1 - pe)
    i, j = np.indices(cm.shape)
    w = ((i - j) / 2.0) ** 2
    exp = np.outer(cm.sum(1), cm.sum(0)) / n
    kw = 1.0 - (w * cm).sum() / max((w * exp).sum(), 1e-9)
    within1 = (cm[np.abs(i - j) <= 1].sum()) / n
    return po, k, kw, within1


def main():
    set_pub_style(base_font=8)
    tiles, areas, gfrac = {}, {}, {}
    for r, seed0 in enumerate((410, 940)):
        for c, g in enumerate((1, 2, 3)):
            rgb, ar, gf = grade_tile(g, seed0 + 31 * c)
            tiles[(r, c)] = rgb
            areas[(r, c)] = ar
            gfrac[(r, c)] = gf

    cm = simulate_read()
    po, k, kw, within1 = kappas(cm)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.72], wspace=0.045,
                          hspace=0.15, left=0.052, right=0.988, top=0.876,
                          bottom=0.062)
    axes = np.empty((2, 3), object)
    for r in range(2):
        for c in range(3):
            ax = fig.add_subplot(gs[r, c])
            axes[r, c] = ax
            image_axes(ax, color="#b0b0b0", lw=0.55)
            ax.imshow(tiles[(r, c)], interpolation="bilinear", zorder=1)
            add_zoom_inset(ax, tiles[(r, c)], ZOOM_BOX,
                           (0.615, 0.030, 0.360, 0.360), color=ZOOM,
                           box_lw=1.0, border_lw=1.2,
                           interpolation="bilinear")
            if r == 0:
                ax.set_title(GRADE_TITLES[c], fontsize=8.4, pad=3.4,
                             color="#1a1a1a")
            if c == 0:
                ax.set_ylabel(f"Case {'AB'[r]}", fontsize=8.0, labelpad=3.4,
                              color="#1a1a1a")
            halo_text(ax, 0.028, 0.955,
                      f"{GRADES[c]}   lumen {gfrac[(r, c)]:.0f} %   "
                      f"mean nucleus {areas[(r, c)].mean():.0f} µm²",
                      transform=ax.transAxes, ha="left", va="top",
                      fontsize=6.2, color="#1a1a1a", halo="white",
                      halo_lw=2.3)

    # ---- subordinate strip: reader agreement ------------------------------
    sub = gs[2, :].subgridspec(1, 3, width_ratios=[0.52, 1.14, 1.14],
                               wspace=0.34)
    ax_cm = fig.add_subplot(sub[0, 0])
    ax_ag = fig.add_subplot(sub[0, 1])
    ax_card = fig.add_subplot(sub[0, 2])
    ax_card.axis("off")

    row = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax_cm.imshow(row, cmap=get_cmap("cb_sequential_blue"), vmin=0, vmax=1,
                 interpolation="nearest")
    for i in range(3):
        for j in range(3):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                       fontsize=6.4, fontweight="bold" if i == j else "normal",
                       color="white" if row[i, j] > 0.55 else "#333333")
    ax_cm.set_xticks(range(3))
    ax_cm.set_yticks(range(3))
    ax_cm.set_xticklabels(GRADES, fontsize=6.6)
    ax_cm.set_yticklabels(GRADES, fontsize=6.6)
    ax_cm.set_xlabel("AI grade", fontsize=7.2, labelpad=1.6)
    ax_cm.set_ylabel("Pathologist", fontsize=7.2, labelpad=1.6)
    ax_cm.tick_params(length=1.8, pad=1.4)
    for s in ax_cm.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#333333")
    ax_cm.set_title(f"Confusion matrix  (n = {cm.sum()})", fontsize=8.0,
                    fontweight="bold", pad=3.0)

    agree = 100.0 * np.diag(cm) / np.maximum(cm.sum(1), 1)
    ax_ag.bar(np.arange(3), agree, width=0.56, color="#3b5b92", alpha=0.88,
              edgecolor="#3a3a3a", linewidth=0.5)
    for x, v in zip(np.arange(3), agree):
        ax_ag.text(x, v + 2.5, f"{v:.0f} %", ha="center", va="bottom",
                   fontsize=6.4, color="#333333")
    ax_ag.axhline(100.0 * po, color=ACCENT, lw=1.0, ls="--")
    ax_ag.text(2.44, 101.0, f"overall {100 * po:.0f} %", ha="right",
               va="bottom", fontsize=6.2, color=ACCENT)
    ax_ag.set_xticks(range(3))
    ax_ag.set_xticklabels(GRADES, fontsize=6.6)
    ax_ag.set_ylim(0, 118)
    ax_ag.set_ylabel("exact agreement (%)", fontsize=7.2, labelpad=2.0)
    ax_ag.tick_params(labelsize=6.5, length=2.2, pad=1.5)
    ax_ag.grid(axis="y", linewidth=0.4, alpha=0.6)
    style_axes(ax_ag)
    ax_ag.set_title("Agreement per pathologist grade", fontsize=8.0,
                    fontweight="bold", pad=3.0)

    chips = [("Cohen κ", f"{k:.3f}"), ("Quadratic κ", f"{kw:.3f}"),
             ("Within one grade", f"{100 * within1:.0f} %")]
    for i, (name, val) in enumerate(chips):
        x0 = 0.015 + i * 0.332
        ax_card.add_patch(FancyBboxPatch(
            (x0, 0.10), 0.305, 0.74, transform=ax_card.transAxes,
            boxstyle="round,pad=0.006,rounding_size=0.04",
            facecolor="#f2f5f8", edgecolor="#b9c6d1", linewidth=0.6,
            clip_on=False))
        ax_card.text(x0 + 0.152, 0.60, val, transform=ax_card.transAxes,
                     ha="center", va="center", fontsize=10.5,
                     fontweight="bold", color="#1b3f5c")
        ax_card.text(x0 + 0.152, 0.26, name, transform=ax_card.transAxes,
                     ha="center", va="center", fontsize=6.4, color="#4a5b68")
    ax_card.set_title("Reader concordance", fontsize=8.0, fontweight="bold",
                      pad=3.0)

    savefig(fig, "544_path_grading_compare.png")


if __name__ == "__main__":
    main()

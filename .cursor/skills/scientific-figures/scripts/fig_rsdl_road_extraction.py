"""Figure 334 - Road extraction with an explicit connectivity read-out.

Two rural tiles (rows) x six views: image, reference centreline mask, three
networks and the vectorised centreline graph recovered by our model.  Roads
are thin, so the story is topological: the baselines shred the network into
fragments while ours keeps it connected.  A subordinate row (60% height)
carries APLS, break density and the gap-length distribution.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter, distance_transform_edt

from pubstyle import set_pub_style, savefig, style_axes

LC = {"water": "#3b7ea1", "crop": "#8fbc4a", "forest": "#1d6b3f",
      "grass": "#d8d878", "building": "#b23a48", "soil": "#d9a86c",
      "road": "#6f6f6f"}
ROAD_HL = "#c9a227"        # road overlay colour (grey roads need contrast)
# house red is already the built-up land-cover class here, so the zoom chrome
# takes house purple instead
ZOOM_RED = "#8e6bbf"
GSD = 1.5                  # m per pixel


def _noise(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _fbm(rng, n, scales=(3, 8, 20), weights=(0.5, 0.32, 0.18)):
    out = sum(w * _noise(rng, n, s) for s, w in zip(scales, weights))
    return out / (out.std() + 1e-9)


def _polyline_dist(n, pts):
    m = np.zeros((n, n), bool)
    pts = np.asarray(pts, float)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        k = int(max(abs(x1 - x0), abs(y1 - y0)) * 2) + 2
        ii = np.clip(np.round(np.linspace(y0, y1, k)).astype(int), 0, n - 1)
        jj = np.clip(np.round(np.linspace(x0, x1, k)).astype(int), 0, n - 1)
        m[ii, jj] = True
    return distance_transform_edt(~m)


# ---------------------------------------------------------------------------
# Synthetic rural scene with a road network
# ---------------------------------------------------------------------------
def road_network(n, rng):
    """List of (polyline, half-width) covering arterials, links and tracks."""
    t = np.linspace(0, n - 1, 360)
    lines = []
    ya = (rng.uniform(0.20, 0.32) * n
          + rng.uniform(0.05, 0.10) * n * np.sin(2 * np.pi * t / n
                                                 * rng.uniform(0.7, 1.0)
                                                 + rng.uniform(0, 6))
          + 0.028 * n * np.sin(2 * np.pi * t / n * 2.4 + rng.uniform(0, 6)))
    lines.append((np.c_[t, ya], 2.2))
    yb = (rng.uniform(0.72, 0.86) * n
          + rng.uniform(0.04, 0.08) * n * np.sin(2 * np.pi * t / n
                                                 * rng.uniform(0.9, 1.3)
                                                 + rng.uniform(0, 6)))
    lines.append((np.c_[t, yb], 1.9))
    xc = (rng.uniform(0.55, 0.72) * n
          + rng.uniform(0.05, 0.10) * n * np.sin(2 * np.pi * t / n
                                                 * rng.uniform(0.8, 1.1)
                                                 + rng.uniform(0, 6)))
    lines.append((np.c_[xc, t], 2.0))

    s = np.linspace(0, 1, 140)
    for f in rng.uniform(0.08, 0.94, size=3):
        x0 = f * n
        y0 = np.interp(x0, t, ya)
        y1 = np.interp(x0, t, yb)
        cx = x0 + rng.uniform(-0.06, 0.06) * n * np.sin(np.pi * s)
        lines.append((np.c_[cx, y0 + (y1 - y0) * s], 1.3))
    for _ in range(2):
        y0 = rng.uniform(0.36, 0.66) * n
        cy = y0 + rng.uniform(-0.06, 0.06) * n * np.sin(np.pi * s)
        cx = np.linspace(rng.uniform(0.02, 0.12) * n,
                         rng.uniform(0.50, 0.66) * n, 140)
        lines.append((np.c_[cx, cy], 1.0))
    return lines


def rural_scene(n=224, seed=5):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)

    # --- farmland parcels --------------------------------------------------
    anchors = rng.uniform(0.1, 0.9, size=(3, 2)) * n
    dists = np.stack([np.hypot(xx - a[0], yy - a[1]) + 13 * _noise(rng, n, 24)
                      for a in anchors])
    reg = np.argmin(dists, axis=0)
    fid = np.zeros((n, n), int)
    for r, (ang, cu, cv) in enumerate([(19, 30, 40), (-27, 34, 26),
                                       (44, 26, 33)]):
        th = np.deg2rad(ang)
        u = xx * np.cos(th) + yy * np.sin(th) + 4.0 * _noise(rng, n, 15)
        v = -xx * np.sin(th) + yy * np.cos(th) + 4.0 * _noise(rng, n, 15)
        f = (np.floor(u / cu).astype(int) * 211
             + np.floor(v / cv).astype(int) * 37 + r * 1013)
        fid = np.where(reg == r, f, fid)
    key = np.abs(fid) % 4096
    vig = 0.30 + 0.66 * rng.random(4096)[key]
    kind = rng.random(4096)[key]

    forest = gaussian_filter((_fbm(rng, n, (5, 13, 28)) > 0.80).astype(float),
                             1.6) > 0.44
    river = _polyline_dist(n, np.c_[
        0.16 * n + 0.07 * n * np.sin(np.linspace(0, 5, 200)),
        np.linspace(0, n - 1, 200)]) <= 2.4

    lines = road_network(n, rng)
    road = np.zeros((n, n), bool)
    for pl, w in lines:
        road |= _polyline_dist(n, pl) <= w

    # --- reflectance: linear canopy / soil mixing keeps hues natural -------
    veg_spec = (0.040, 0.078, 0.038)
    soil_spec = (0.212, 0.192, 0.162)
    fc = np.clip(vig, 0.04, 1.0)
    bands = np.stack([fc * veg_spec[k] + (1 - fc) * soil_spec[k]
                      for k in range(3)])
    bare = kind < 0.13
    for k, val in enumerate((0.235, 0.205, 0.172)):
        bands[k] = np.where(bare, val, bands[k])
    for k, val in enumerate((0.036, 0.068, 0.040)):
        bands[k] = np.where(forest, val, bands[k])
    for k, val in enumerate((0.030, 0.048, 0.058)):
        bands[k] = np.where(river, val, bands[k])
    for k, val in enumerate((0.146, 0.148, 0.152)):
        bands[k] = np.where(road, val, bands[k])

    amp = np.where(forest, 0.26, np.where(road, 0.05, 0.09))
    tex = 1 + amp * (0.6 * _noise(rng, n, 1.0) + 0.4 * _noise(rng, n, 3.0))
    illum = 1 + 0.05 * _noise(rng, n, 32)
    for k in range(3):
        bands[k] = bands[k] * tex * illum + 0.003 * rng.standard_normal((n, n))

    p0, p1 = np.percentile(bands, [1.2, 99.0])
    rgb = np.dstack([np.clip((bands[i] - p0) / max(p1 - p0, 1e-9), 0, 1) ** g
                     for i, g in zip((0, 1, 2), (0.80, 0.80, 0.86))])
    return rgb, road, lines


# ---------------------------------------------------------------------------
# Network simulation and topology metrics
# ---------------------------------------------------------------------------
def predict_road(gt, seed, sigma, thr, disp, gap_rate, gap_sigma, fp):
    n = gt.shape[0]
    rng = np.random.default_rng(seed)
    field = gaussian_filter(gt.astype(float), sigma)
    yy, xx = np.mgrid[0:n, 0:n]
    dy = disp * _noise(rng, n, 8.0)
    dx = disp * _noise(rng, n, 8.0)
    sy = np.clip(np.round(yy + dy).astype(int), 0, n - 1)
    sx = np.clip(np.round(xx + dx).astype(int), 0, n - 1)
    pred = field[sy, sx] > thr

    g = _noise(rng, n, gap_sigma)
    pred &= ~(g > np.quantile(g, 1.0 - gap_rate))
    if fp > 0:
        b = _noise(rng, n, 2.0)
        halo = gaussian_filter(gt.astype(float), 5.0) > 0.02
        pred |= halo & (b > np.quantile(b, 1.0 - fp))
    return pred


def trace_centreline(pred, lines, n, tol=2.0):
    """Sample each reference polyline and split it into detected runs."""
    hit = distance_transform_edt(~pred) <= tol
    runs, total, det, gaps, marks = [], 0, 0, [], []
    for pl, _w in lines:
        p = np.asarray(pl, float)
        seg = np.hypot(*(np.diff(p, axis=0).T))
        cum = np.concatenate([[0], np.cumsum(seg)])
        m = int(cum[-1] // 1.5) + 2
        ss = np.linspace(0, cum[-1], m)
        px = np.interp(ss, cum, p[:, 0])
        py = np.interp(ss, cum, p[:, 1])
        ii = np.clip(np.round(py).astype(int), 0, n - 1)
        jj = np.clip(np.round(px).astype(int), 0, n - 1)
        on = hit[ii, jj]
        total += m
        det += int(on.sum())
        k = 0
        while k < m:
            if on[k]:
                j = k
                while j + 1 < m and on[j + 1]:
                    j += 1
                if j - k >= 2:
                    runs.append(np.c_[px[k:j + 1], py[k:j + 1]])
                k = j + 1
            else:
                j = k
                while j + 1 < m and not on[j + 1]:
                    j += 1
                if j - k >= 1:
                    gaps.append((j - k + 1) * 1.5 * GSD)
                    mid = (k + j) // 2
                    marks.append((px[mid], py[mid]))
                k = j + 1
    length_km = total * 1.5 * GSD / 1000.0
    frac = det / max(total, 1)
    bpk = len(gaps) / max(length_km, 1e-6)
    apls = 0.90 * frac * np.exp(-0.035 * bpk)
    return runs, gaps, frac, apls, length_km, marks


def iou_f1(gt, pr):
    inter = np.count_nonzero(gt & pr)
    union = np.count_nonzero(gt | pr)
    prec = inter / max(np.count_nonzero(pr), 1)
    rec = inter / max(np.count_nonzero(gt), 1)
    return inter / max(union, 1), 2 * prec * rec / max(prec + rec, 1e-9)


# ---------------------------------------------------------------------------
def image_axes(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")


def overlay(rgb, mask, color, alpha=0.82):
    col = np.array(plt.matplotlib.colors.to_rgb(color))
    out = rgb.copy()
    out[mask] = (1 - alpha) * rgb[mask] + alpha * col
    return np.clip(out, 0, 1)


def roi_rect(ax, box, color=ZOOM_RED, lw=0.9):
    x0, y0, w, h = box
    ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), w, h, fill=False,
                           edgecolor=color, linewidth=lw, zorder=7))


def add_zoom(ax, disp, box, loc=(0.585, 0.020, 0.398, 0.398),
             color=ZOOM_RED):
    x0, y0, w, h = box
    roi_rect(ax, box, color)
    ins = ax.inset_axes(loc)
    ins.set_zorder(8)
    ins.imshow(disp[y0:y0 + h, x0:x0 + w], interpolation="nearest")
    ins.set_xticks([])
    ins.set_yticks([])
    for s in ins.spines.values():
        s.set_visible(True)
        s.set_linewidth(1.0)
        s.set_color(color)
    return ins


def pick_roi(gt, n, size, inset_loc):
    """Densest road window that does not sit under the corner inset."""
    ix0 = inset_loc[0] * n
    iy0 = (1.0 - inset_loc[1] - inset_loc[3]) * n
    best, box = -1.0, (0, 0, size, size)
    for y0 in range(4, n - size - 4, 6):
        for x0 in range(4, n - size - 4, 6):
            if x0 + size > ix0 and y0 + size > iy0:
                continue
            v = gt[y0:y0 + size, x0:x0 + size].sum()
            if v > best:
                best, box = v, (x0, y0, size, size)
    return box


def stamp(ax, text, weight="normal"):
    ax.text(0.028, 0.030, text, transform=ax.transAxes, ha="left",
            va="bottom", fontsize=6.4, color="white", fontweight=weight,
            zorder=9,
            path_effects=[pe.withStroke(linewidth=1.7, foreground="#1a1a1a")])


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    n = 224
    zoom_loc = (0.585, 0.020, 0.398, 0.398)
    tiles = [5, 19]

    methods = [
        ("U-Net", dict(sigma=0.98, thr=0.40, disp=0.95, gap_rate=0.130,
                       gap_sigma=3.2, fp=0.014)),
        ("D-LinkNet", dict(sigma=0.86, thr=0.41, disp=0.72, gap_rate=0.098,
                           gap_sigma=3.5, fp=0.009)),
        ("Ours (TopoRoad)", dict(sigma=0.72, thr=0.42, disp=0.42,
                                 gap_rate=0.028, gap_sigma=5.2, fp=0.004)),
    ]

    # Tile 2 is the scene where D-LinkNet keeps slightly fewer breaks.
    tweak = {(2, 1): dict(gap_rate=0.058)}

    samples = []
    for k, sd in enumerate(tiles):
        rgb, road, lines = rural_scene(n, seed=sd)
        preds = []
        for j, (_nm, kw) in enumerate(methods):
            kw = {**kw, **tweak.get((j, k), {})}
            preds.append(predict_road(road, seed=700 + 13 * j + k, **kw))
        traces = [trace_centreline(p, lines, n) for p in preds]
        box = pick_roi(road, n, 52, zoom_loc)
        samples.append(dict(rgb=rgb, gt=road, lines=lines, preds=preds,
                            traces=traces, box=box))

    pooled, apls, breaks, allgaps = [], [], [], []
    for j in range(len(methods)):
        g = np.concatenate([s["gt"].ravel() for s in samples])
        p = np.concatenate([s["preds"][j].ravel() for s in samples])
        pooled.append(iou_f1(g, p))
        a = np.mean([s["traces"][j][3] for s in samples])
        bpk = sum(len(s["traces"][j][1]) for s in samples) / \
            sum(s["traces"][j][4] for s in samples)
        apls.append(a)
        breaks.append(bpk)
        allgaps.append(np.concatenate([np.asarray(s["traces"][j][1])
                                       for s in samples]))
    for (nm, _), (i, f), a, b in zip(methods, pooled, apls, breaks):
        print(f"  {nm:18s} IoU={i:.3f} F1={f:.3f} APLS={a:.3f} "
              f"breaks/km={b:.2f}")

    cols = ["Image", "Reference", "U-Net", "D-LinkNet", "Ours (TopoRoad)",
            "Centreline graph"]
    fig = plt.figure(figsize=(10.4, 6.3))
    gs_t = fig.add_gridspec(2, 6, left=0.038, right=0.988, top=0.882,
                            bottom=0.360, wspace=0.028, hspace=0.030)
    gs_b = fig.add_gridspec(1, 3, left=0.070, right=0.978, top=0.278,
                            bottom=0.085, wspace=0.30)
    axes = np.empty((2, 6), object)

    for r, s in enumerate(samples):
        for c in range(6):
            ax = fig.add_subplot(gs_t[r, c])
            axes[r, c] = ax
            image_axes(ax)
            ax.patch.set_visible(False)
            if c == 0:
                disp = s["rgb"]
            elif c == 1:
                disp = overlay(s["rgb"], s["gt"], ROAD_HL)
            elif c <= 4:
                disp = overlay(s["rgb"], s["preds"][c - 2], ROAD_HL)
            else:
                disp = None

            if disp is not None:
                ax.imshow(disp, interpolation="nearest")
                add_zoom(ax, disp, s["box"])
            else:
                ax.imshow(np.ones((n, n, 3)) * 0.988, interpolation="nearest")
                for pl, w in s["lines"]:
                    ax.plot(pl[:, 0], pl[:, 1], color="#b9b9b9",
                            lw=1.2 + 0.8 * w, solid_capstyle="round",
                            zorder=2)
                for run in s["traces"][2][0]:
                    ax.plot(run[:, 0], run[:, 1], color="#2e7d5b", lw=1.3,
                            solid_capstyle="round", zorder=3)
                mk = np.asarray(s["traces"][2][5]).reshape(-1, 2)
                if mk.size:
                    ax.plot(mk[:, 0], mk[:, 1], ls="none", marker="x",
                            ms=3.4, mew=0.9, color=ZOOM_RED, zorder=5)
                for pl, _w in s["lines"]:
                    ax.plot(pl[[0, -1], 0], pl[[0, -1], 1], ls="none",
                            marker="o", ms=2.2, mfc="#3b5b92",
                            mec="white", mew=0.4, zorder=4)
                ax.set_xlim(-0.5, n - 0.5)
                ax.set_ylim(n - 0.5, -0.5)
                roi_rect(ax, s["box"])

            if c == 0:
                ax.set_ylabel(f"Tile {r + 1}", fontsize=7.0,
                              fontweight="bold", labelpad=3)
            if 2 <= c <= 4:
                i_t, _f = iou_f1(s["gt"], s["preds"][c - 2])
                nb = len(s["traces"][c - 2][1])
                best = c == 4
                stamp(ax, f"IoU {i_t:.3f}   {nb} break"
                          f"{'' if nb == 1 else 's'}",
                      weight="bold" if best else "normal")
            if r == 0:
                ax.set_title(f"({'abcdef'[c]})  {cols[c]}", fontsize=7.6,
                             fontweight="bold", pad=3.4,
                             color="#7a1f2b" if c == 4 else "#1a1a1a")

    p_top = axes[0, 4].get_position()
    p_bot = axes[1, 4].get_position()
    fig.patches.append(Rectangle(
        (p_top.x0 - 0.006, p_bot.y0 - 0.020), p_top.width + 0.012,
        (p_top.y1 + 0.042) - (p_bot.y0 - 0.020),
        transform=fig.transFigure, facecolor="#f2efe9", edgecolor="none",
        zorder=-1))

    # ---- subordinate metric row -------------------------------------------
    names = [m[0].replace(" (TopoRoad)", "") for m in methods]
    bar_c = ["#9aa7b1", "#7d8f9e", "#b23a48"]

    ax1 = fig.add_subplot(gs_b[0, 0])
    groups = ["IoU", "F1", "APLS"]
    gx = np.arange(3)
    vals = np.array([[pooled[j][0], pooled[j][1], apls[j]] for j in range(3)])
    for j, (nm, cc) in enumerate(zip(names, bar_c)):
        off = (j - 1) * 0.26
        ax1.bar(gx + off, vals[j], 0.24, color=cc, edgecolor="#333333",
                linewidth=0.5, label=nm)
        for xi, v in zip(gx + off, vals[j]):
            ax1.text(xi, v + 0.018, f"{v:.2f}", ha="center", va="bottom",
                     fontsize=6.0, color="#333333")
    ax1.set_xticks(gx)
    ax1.set_xticklabels(groups, fontsize=7.0)
    ax1.set_ylim(0, 1.06)
    ax1.set_ylabel("score (-)", fontsize=7.2, labelpad=2)
    ax1.tick_params(labelsize=6.5, length=2, pad=1.5)
    ax1.grid(axis="y", linewidth=0.4, alpha=0.6)
    ax1.legend(frameon=False, fontsize=6.4, ncol=3, loc="upper center",
               handlelength=1.1, columnspacing=0.9, borderaxespad=0.05,
               bbox_to_anchor=(0.5, 1.03))
    style_axes(ax1)
    ax1.set_title("(g)  Pixel and topology scores", fontsize=7.6,
                  fontweight="bold", pad=3.0)

    ax2 = fig.add_subplot(gs_b[0, 1])
    x = np.arange(3)
    ax2.bar(x, breaks, 0.52, color=bar_c, edgecolor="#333333", linewidth=0.5)
    for xi, v in zip(x, breaks):
        ax2.text(xi, v + 0.04, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=6.4, color="#333333")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=6.6)
    ax2.set_ylim(0, max(breaks) * 1.28)
    ax2.set_ylabel(r"breaks per km (km$^{-1}$)", fontsize=7.2, labelpad=2)
    ax2.tick_params(labelsize=6.5, length=2, pad=1.5)
    ax2.grid(axis="y", linewidth=0.4, alpha=0.6)
    style_axes(ax2)
    ax2.set_title("(h)  Centreline discontinuities", fontsize=7.6,
                  fontweight="bold", pad=3.0)

    ax3 = fig.add_subplot(gs_b[0, 2])
    edges = np.linspace(0, 60, 13)
    for nm, gaps, cc in zip(names, allgaps, bar_c):
        if gaps.size == 0:
            continue
        h, _ = np.histogram(np.clip(gaps, 0, 59), bins=edges)
        ax3.step(edges[:-1], h, where="post", color=cc, lw=1.3, label=nm)
        ax3.fill_between(edges[:-1], 0, h, step="post", color=cc, alpha=0.16)
    ax3.set_xlabel("gap length (m)", fontsize=7.2, labelpad=1.5)
    ax3.set_ylabel("count (-)", fontsize=7.2, labelpad=2)
    ax3.tick_params(labelsize=6.5, length=2, pad=1.5)
    ax3.grid(linewidth=0.4, alpha=0.6)
    ax3.legend(frameon=False, fontsize=6.4, loc="upper right",
               handlelength=1.2)
    style_axes(ax3)
    ax3.set_title("(i)  Gap-length distribution", fontsize=7.6,
                  fontweight="bold", pad=3.0)

    handles = [Patch(facecolor=ROAD_HL, edgecolor="none",
                     label="Road mask"),
               Line2D([], [], color="#2e7d5b", lw=1.4,
                      label="Recovered centreline"),
               Line2D([], [], color="#c8c8c8", lw=1.4,
                      label="Reference centreline"),
               Patch(facecolor="none", edgecolor=ZOOM_RED, linewidth=1.0,
                     label="Zoom window (identical per row)")]
    leg = fig.legend(handles=handles, loc="lower center", ncol=4,
                     frameon=False, fontsize=6.8, handlelength=1.4,
                     bbox_to_anchor=(0.52, 0.004))
    leg.set_clip_on(False)


    savefig(fig, "334_rsdl_road_extraction.png")


if __name__ == "__main__":
    main()

"""Figure 274 - Per-head self-attention maps for one query patch.

A single street scene is tokenised into a 16 x 16 patch grid; the red square
marks the query patch.  The twelve panels below show where each head of the
layer sends that query's attention.  Maps are drawn as gain over the uniform
1/256 baseline, which lets every head sit on one honest colour scale.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig

HEAT = "magma"
ALPHA = 0.55
QUERY = "#b23a48"
P = 16                      # patches per side
N = 128                     # pixels per side
VMAX = 4.5                  # gain over uniform attention


def _noise(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _fbm(rng, n, scales=(1.0, 3.0, 8.0), weights=(0.45, 0.33, 0.22)):
    out = sum(w * _noise(rng, n, s) for s, w in zip(scales, weights))
    return out / (out.std() + 1e-9)


def _soft(mask, sigma=0.8):
    return np.clip(gaussian_filter(mask.astype(float), sigma), 0, 1)


def _paint(img, mask, value, texture):
    return img * (1 - mask) + mask * np.clip(value + texture, 0.02, 0.99)


def street_scene(seed=7, n=N):
    """Layered geometry: sky, hills, buildings, trees, road and a car."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    v = yy / (n - 1.0)
    tex = 0.030 * _fbm(rng, n, (0.9, 2.5, 6.0))

    img = 0.92 - 0.34 * np.clip(v / 0.62, 0, 1) ** 0.8      # sky gradient
    img += 0.05 * _fbm(rng, n, (6, 16, 30))

    hills = 0.55 * n + 0.05 * n * np.sin(xx / 21.0) + 0.03 * n * np.sin(xx / 7.0)
    img = _paint(img, _soft(yy > hills, 1.2), 0.52, tex * 1.4)
    img = _paint(img, _soft(yy > 0.72 * n, 1.0), 0.33, tex * 1.0)   # road

    for x0, w, top, val in ((0.03, 0.135, 0.315, 0.62), (0.175, 0.10, 0.400, 0.47),
                            (0.285, 0.155, 0.255, 0.70), (0.455, 0.095, 0.430, 0.55)):
        blk = ((xx > x0 * n) & (xx < (x0 + w) * n)
               & (yy > top * n) & (yy < 0.735 * n))
        img = _paint(img, _soft(blk, 0.6), val, tex * 0.8)
        for wy in np.arange(top + 0.045, 0.70, 0.062):          # window rows
            for wx in np.arange(x0 + 0.022, x0 + w - 0.02, 0.038):
                win = ((xx > wx * n) & (xx < (wx + 0.020) * n)
                       & (yy > wy * n) & (yy < (wy + 0.030) * n))
                img = _paint(img, _soft(win, 0.35),
                             0.20 + 0.55 * rng.random(), tex * 0.3)

    for cx, cy, rr in ((0.635, 0.520, 0.085), (0.755, 0.560, 0.062),
                       (0.905, 0.495, 0.075)):                  # tree crowns
        crown = ((xx - cx * n) ** 2 + (yy - cy * n) ** 2) < (rr * n) ** 2
        img = _paint(img, _soft(crown, 0.9), 0.30, tex * 2.6)
        trunk = ((np.abs(xx - cx * n) < 0.012 * n)
                 & (yy > cy * n) & (yy < 0.735 * n))
        img = _paint(img, _soft(trunk, 0.5), 0.24, tex * 0.6)

    car_c = (0.415, 0.815)                                       # car on road
    body = (((xx - car_c[0] * n) / (0.145 * n)) ** 2
            + ((yy - car_c[1] * n) / (0.048 * n)) ** 2) < 1
    body |= ((np.abs(xx - car_c[0] * n) < 0.135 * n)
             & (np.abs(yy - car_c[1] * n) < 0.040 * n))
    cabin = (((xx - car_c[0] * n) / (0.078 * n)) ** 2
             + ((yy - (car_c[1] - 0.038) * n) / (0.040 * n)) ** 2) < 1
    img = _paint(img, _soft(body | cabin, 0.6), 0.14, tex * 0.9)
    img = _paint(img, _soft(cabin & (yy < (car_c[1] - 0.040) * n), 0.5),
                 0.72, tex * 0.4)
    for wx in (-0.082, 0.082):
        wheel = (((xx - (car_c[0] + wx) * n) / (0.030 * n)) ** 2
                 + ((yy - (car_c[1] + 0.042) * n) / (0.028 * n)) ** 2) < 1
        img = _paint(img, _soft(wheel, 0.5), 0.06, tex * 0.3)

    img += 0.010 * rng.standard_normal((n, n))
    lo, hi = np.percentile(img, [1.0, 99.0])
    return np.clip((img - lo) / (hi - lo), 0, 1), car_c


def build_heads(img, car_c, seed=5):
    """Twelve qualitatively different attention patterns on the patch grid."""
    rng = np.random.default_rng(seed)
    pr, pc = np.mgrid[0:P, 0:P].astype(float)
    qc = int(car_c[0] * P)
    qr = int(car_c[1] * P)
    d2 = (pr - qr) ** 2 + (pc - qc) ** 2

    coarse = img.reshape(P, N // P, P, N // P).mean(axis=(1, 3))
    gy, gx = np.gradient(gaussian_filter(coarse, 0.6))
    edges = np.hypot(gy, gx)
    edges = edges / (edges.max() + 1e-9)
    obj = np.exp(-(((pc - qc) / 2.4) ** 2 + ((pr - qr) / 1.3) ** 2))

    heads = [
        ("local", np.exp(-d2 / (2 * 0.85 ** 2))),
        ("neighbourhood", np.exp(-d2 / (2 * 2.4 ** 2))),
        ("row band", np.exp(-((pr - qr) ** 2) / (2 * 0.95 ** 2)) + 0.05),
        ("column band", np.exp(-((pc - qc) ** 2) / (2 * 0.95 ** 2)) + 0.05),
        ("object", obj + 0.10),
        ("contours", 0.06 + edges ** 1.5),
        ("background", 0.10 + (1.0 - obj) * (0.4 + 0.6 * (pr < 0.55 * P))),
        ("diffuse", 0.30 + 1.35 * _noise(rng, P, 2.6).clip(-1.6, 1.6) / 3.2
         + 0.55),
        ("attn. sinks", 0.10 + np.exp(-((pr - 0) ** 2 + (pc - 0) ** 2) / 2.0)
         + np.exp(-((pr - 0) ** 2 + (pc - P + 1) ** 2) / 2.0)
         + np.exp(-((pr - P + 1) ** 2 + (pc - 0) ** 2) / 2.0)
         + np.exp(-((pr - P + 1) ** 2 + (pc - P + 1) ** 2) / 2.0)),
        ("symmetric", np.exp(-d2 / (2 * 1.5 ** 2))
         + 0.85 * np.exp(-(((pr - qr) ** 2 + (pc - (P - 1 - qc)) ** 2))
                         / (2 * 1.7 ** 2)) + 0.06),
        ("texture", 0.08 + gaussian_filter(np.abs(np.gradient(coarse)[1]), 0.4)
         / (np.abs(np.gradient(coarse)[1]).max() + 1e-9)),
        ("diagonal", np.exp(-(((pr - pc) - (qr - qc)) ** 2) / (2 * 1.1 ** 2))
         + 0.06),
    ]
    out = []
    for name, m in heads:
        m = np.clip(m, 1e-6, None)
        m = m / m.sum() * m.size          # gain over the uniform baseline
        out.append((name, m))
    return out, (qr, qc)


def image_axes(ax, color="#c9c9c9", lw=0.5):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(lw)
        s.set_color(color)


def main():
    set_pub_style(base_font=8)
    img, car_c = street_scene()
    heads, (qr, qc) = build_heads(img, car_c)
    norm = Normalize(vmin=0.0, vmax=VMAX)
    step = N / P

    fig = plt.figure(figsize=(7.2, 7.3))

    ax0 = fig.add_axes([0.075, 0.690, 0.2545, 0.2511])
    image_axes(ax0, color="#8a8a8a", lw=0.7)
    ax0.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="bilinear",
               extent=[0, N, N, 0])
    for k in range(1, P):
        ax0.axhline(k * step, color="white", lw=0.25, alpha=0.30)
        ax0.axvline(k * step, color="white", lw=0.25, alpha=0.30)
    ax0.add_patch(Rectangle((qc * step, qr * step), step, step, fill=False,
                            edgecolor=QUERY, linewidth=1.6, zorder=5))
    ax0.set_xlim(0, N)
    ax0.set_ylim(N, 0)
    ax0.set_title("Input  (16 x 16 patches)", fontsize=7.8, pad=3.6)
    ax0.text(0.5, -0.045, f"query patch (row {qr}, col {qc})",
             transform=ax0.transAxes, ha="center", va="top", fontsize=6.6,
             color=QUERY, fontweight="bold")

    tx, ty = 0.352, 0.940
    fig.text(tx, ty, "Multi-head self-attention, block 8 of 12",
             fontsize=8.4, fontweight="bold", ha="left", va="top")
    body = [
        "Each panel below shows the attention that the single red query",
        "patch sends to all 256 keys, for one of the 12 heads.",
        "",
        "Values are plotted as gain over the uniform 1/256 baseline, so a",
        "value of 1.0 means 'no preference' and 4.5 means 'this key gets",
        "4.5x its fair share'.  All heads share this one colour scale, and",
        "each map is overlaid on the input at 55% opacity.",
        "",
        "Heads specialise: some stay inside the query object, some sweep a",
        "whole row or column, some lock onto contours, and some spend their",
        "budget on the corner tokens that act as attention sinks.",
    ]
    for i, line in enumerate(body):
        fig.text(tx, ty - 0.030 - i * 0.0205, line, fontsize=6.7,
                 color="#4a4a4a", ha="left", va="top")

    gs = fig.add_gridspec(3, 4, left=0.075, right=0.895, top=0.6300,
                          bottom=0.0253, wspace=0.040, hspace=0.040)
    for i, (name, m) in enumerate(heads):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        image_axes(ax)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="bilinear",
                  extent=[0, N, N, 0])
        ax.imshow(m, cmap=HEAT, norm=norm, alpha=ALPHA,
                  interpolation="nearest", extent=[0, N, N, 0])
        ax.add_patch(Rectangle((qc * step, qr * step), step, step, fill=False,
                               edgecolor=QUERY, linewidth=0.9, zorder=5))
        ax.set_xlim(0, N)
        ax.set_ylim(N, 0)
        ax.text(0.035, 0.035, f"Head {i + 1}  {name}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=6.1, zorder=6,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                          edgecolor="#b8b8b8", linewidth=0.4, alpha=0.90))

    cax = fig.add_axes([0.908, 0.0253, 0.014, 0.6047])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=HEAT), cax=cax,
                      ticks=[0, 1, 2, 3, 4], extend="max")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=7.0, width=0.7, length=2.5, color="#333333")
    cb.set_label("attention gain over uniform (x)", fontsize=7.0,
                 color="#1a1a1a", labelpad=3.5)


    savefig(fig, "274_attention_maps.png")


if __name__ == "__main__":
    main()

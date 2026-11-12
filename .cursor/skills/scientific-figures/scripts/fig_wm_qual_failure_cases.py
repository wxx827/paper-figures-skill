"""Figure 1091 - Failure gallery: nine ways the world model breaks.

A 3 x 3 contact sheet, and deliberately nine *different* scenes: a failure
atlas whose value is coverage undermines itself if every cell is the same
street.  Cells are drawn from the shared library's five kinds (road, tabletop,
indoor, aerial, shelf) with different seeds, horizons, exposures and white
balances, so no two tiles read as the same clip.

Each tile carries a severity chip in the top-right corner (three tiers), a
dashed red ellipse around the offending region and a one-line caption
underneath.  The faults are local operators (erase, ghost, melt, warp, band,
tint) applied on top of an object the scene generator actually placed, so the
callout always encircles real content.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.ndimage import gaussian_filter, map_coordinates

from pubstyle import set_pub_style, savefig
from figkit import image_axes
from _wm_scenes import build

H, W = 112, 204
CALLOUT = "#d1495b"
# three severity tiers: (chip fill, chip ink)
SEV = {"high": ("#f5dade", "#a5344a"),
       "med": ("#f8e5d1", "#b4592f"),
       "low": ("#f1eedc", "#7d6f28")}


def soft_mask(cx, cy, rx, ry):
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    d = ((xx - cx) / max(rx, 1e-6)) ** 2 + ((yy - cy) / max(ry, 1e-6)) ** 2
    return gaussian_filter(np.clip(1.25 - d, 0, 1), 2.0)[..., None]


def pick(anchors, name=None, idx=0):
    """The object a fault will act on, in pixels: (cx, cy, rx, ry)."""
    pool = [a for a in anchors if name is None or a["name"] == name] or anchors
    a = pool[min(idx, len(pool) - 1)]
    return a["cx"], a["cy"], max(a["rx"], 4.0), max(a["ry"], 4.0)


def blur_plate(img, sig=(6.0, 9.0)):
    return np.dstack([gaussian_filter(img[..., c], sig) for c in range(3)])


# ---------------------------------------------------------------------------
# Fault operators.  Each returns (image, ellipse geometry to call out).
# ---------------------------------------------------------------------------
def f_erase(img, anc, rng):
    cx, cy, rx, ry = pick(anc, "vehicle", 2)
    m = soft_mask(cx, cy, 1.05 * rx, 1.15 * ry)
    return img * (1 - m) + blur_plate(img, (7.0, 11.0)) * m, \
        (cx, cy, 2.9 * rx, 3.1 * ry)


def f_dup(img, anc, rng):
    cx, cy, rx, ry = pick(anc, "mug")
    out = img.copy()
    for dx in (-2.1 * rx, 2.1 * rx):
        m = soft_mask(cx + dx, cy, 0.95 * rx, 1.05 * ry)
        out = out * (1 - 0.88 * m) + np.roll(img, -int(dx), axis=1) * 0.88 * m
    return out, (cx, cy, 7.0 * rx, 3.0 * ry)


def f_float(img, anc, rng):
    # the middle shelf, near the centre of the frame: a book lifted off the
    # top shelf is invisible at this tile size
    pool = [a for a in anc if a["name"] == "book"]
    pool.sort(key=lambda a: abs(a["cx"] - 0.46 * W) + abs(a["cy"] - 0.52 * H))
    a = pool[0]
    cx, cy, rx, ry = a["cx"], a["cy"], max(a["rx"], 4.0), max(a["ry"], 4.0)
    lift = int(2.3 * ry)
    m = soft_mask(cx, cy, 1.15 * rx, 1.10 * ry)
    out = img * (1 - m) + blur_plate(img, (4.0, 3.0)) * m
    m2 = soft_mask(cx, cy - lift, 1.05 * rx, 1.00 * ry)
    return out * (1 - m2) + np.roll(img, -lift, axis=0) * m2, \
        (cx, cy - 0.5 * lift, 4.2 * rx, 3.4 * ry)


def f_melt(img, anc, rng):
    cx, cy = 0.40 * W, 0.68 * H
    m = soft_mask(cx, cy, 0.28 * W, 0.20 * H)
    k = np.dstack([gaussian_filter(img[..., c], (1.0, 14.0)) for c in range(3)])
    return img * (1 - m) + k * m, (cx, cy, 0.60 * W, 0.44 * H)


def f_warp(img, anc, rng):
    cx, cy, rx, ry = pick(anc, "window")
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    m = soft_mask(cx, cy, 0.95 * rx, 0.95 * ry)[..., 0]
    d = gaussian_filter(rng.standard_normal((H, W)), 7.0)
    d = d / (d.std() + 1e-9) * m * 2.1          # a bend, not a hole
    out = np.dstack([map_coordinates(img[..., c], [yy + 0.35 * d, xx + d],
                                     order=1, mode="nearest") for c in range(3)])
    return out, (cx, cy, 2.3 * rx, 2.1 * ry)


def f_ghost(img, anc, rng):
    cx, cy, rx, ry = pick(anc, "vehicle", 1)
    m = soft_mask(cx + 1.3 * rx, cy, 0.95 * rx, 1.0 * ry)
    return img * (1 - 0.74 * m) + np.roll(img, -int(1.3 * rx), axis=1) * 0.74 * m, \
        (cx + 0.8 * rx, cy, 3.6 * rx, 2.8 * ry)


def f_band(img, anc, rng):
    yy = np.mgrid[0:H, 0:W][0].astype(float)
    cx, cy = 0.52 * W, 0.52 * H
    m = soft_mask(cx, cy, 0.44 * W, 0.30 * H)
    return np.clip(img + 0.13 * np.sin(yy / 1.6)[..., None] * m, 0, 1), \
        (cx, cy, 0.90 * W, 0.62 * H)


def f_tint(img, anc, rng):
    out = np.clip(img + np.array([-0.10, -0.02, 0.13]), 0, 1)
    lum = out @ np.array([0.30, 0.59, 0.11])
    return out * 0.72 + lum[..., None] * 0.28, (0.74 * W, 0.30 * H,
                                                0.38 * W, 0.46 * H)


def f_smooth(img, anc, rng):
    cx, cy = 0.50 * W, 0.50 * H
    m = soft_mask(cx, cy, 0.40 * W, 0.24 * H)
    k = np.dstack([gaussian_filter(img[..., c], 3.4) for c in range(3)])
    return img * (1 - m) + k * m, (cx, cy, 0.80 * W, 0.52 * H)


# (fault, scene kind, seed, scene params, camera overrides, name, tier, story)
CASES = [
    (f_erase, "road", 2201, dict(horizon=0.455), {},
     "Object vanishing", "high",
     "The lead vehicle dissolves into the road two steps after a partial "
     "occlusion, with nothing in front of it."),
    (f_dup, "tabletop", 2202, dict(), {},
     "Identity duplication", "high",
     "The mug is emitted three times across the desk; all copies share a "
     "single contact shadow."),
    (f_float, "shelf", 2203, dict(shelves=3, plan=5), {},
     "Physics violation", "high",
     "A volume detaches from the shelf and hovers half a book-height above "
     "its own gap."),
    (f_melt, "aerial", 2204, dict(plan=11, pan=26.0), {},
     "Texture melting", "med",
     "Field boundaries smear into horizontal streaks once the parcel edges "
     "leave the context window."),
    (f_warp, "indoor", 2205, dict(horizon=0.54), {},
     "Geometry drift", "med",
     "The window frame bends: straight vertical mullions acquire a "
     "low-frequency wobble of about 3 px."),
    (f_ghost, "road", 2206, dict(horizon=0.415, vpoint=0.58, block=16),
     dict(exposure=0.94, wb=(1.02, 1.00, 0.99)),
     "Occlusion amnesia", "med",
     "After a two-step occlusion the car reappears displaced by half a lane, "
     "ghosting over the original."),
    (f_band, "tabletop", 2207, dict(horizon=0.42),
     dict(exposure=0.95, wb=(0.98, 1.00, 1.05), contrast=1.06),
     "Temporal flicker", "med",
     "Horizontal banding oscillates frame to frame; the amplitude grows with "
     "the classifier-free guidance scale."),
    (f_tint, "aerial", 2208, dict(plan=23, pan=52.0),
     dict(exposure=1.10, wb=(1.04, 1.00, 0.95)),
     "Colour drift", "low",
     "Global white balance creeps towards blue and loses saturation over a "
     "48-step rollout."),
    (f_smooth, "shelf", 2209, dict(shelves=4, plan=31),
     dict(exposure=1.06, wb=(1.00, 1.00, 1.02)),
     "Detail collapse", "low",
     "Spine lettering and shelf edges are washed out; high-frequency energy "
     "drops by 38 percent."),
]


def main():
    set_pub_style(base_font=8)

    fig = plt.figure(figsize=(9.6, 7.0))
    gs = fig.add_gridspec(3, 3, wspace=0.045, hspace=0.245,
                          left=0.030, right=0.988, top=0.892, bottom=0.052)

    for k, (fault, kind, seed, prm, cam, name, sev, story) in enumerate(CASES):
        sc = build(kind, seed=seed, shape=(H, W), cam=cam or None, **prm)
        out, (ex, ey, ew, eh) = fault(sc["rgb"], sc["anchors"],
                                      np.random.default_rng(70 + k))

        ax = fig.add_subplot(gs[k // 3, k % 3])
        image_axes(ax, color="#9a9a9a", lw=0.55)
        ax.set_aspect("auto")
        ax.imshow(np.clip(out, 0, 1), interpolation="bilinear", aspect="auto")
        ax.add_patch(Ellipse((ex, ey), ew, eh, facecolor="none",
                             edgecolor=CALLOUT, lw=1.0, ls=(0, (3.2, 2.0)),
                             zorder=6))

        fill, ink = SEV[sev]
        ax.text(0.977, 0.945, f"{name}  ·  {sev}", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.1, color=ink, zorder=9,
                bbox=dict(boxstyle="round,pad=0.30", facecolor=fill,
                          edgecolor=ink, linewidth=0.55))
        ax.text(0.023, 0.945, f"({'abcdefghi'[k]})", transform=ax.transAxes,
                ha="left", va="top", fontsize=6.6, fontweight="bold",
                color="#ffffff", zorder=9,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#1a1a1a",
                          edgecolor="none"))
        ax.set_xlabel(textwrap.fill(story, 58), fontsize=6.1, color="#666666",
                      labelpad=2.8, linespacing=1.34)

    savefig(fig, "1091_wm_qual_failure_cases.png")


if __name__ == "__main__":
    main()

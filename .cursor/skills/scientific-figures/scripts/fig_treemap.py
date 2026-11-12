"""Treemap: nested rectangles sized by value (squarified layout).

Self-contained squarified treemap (no squarify dependency).  Values are sorted
before tiling -- the squarify algorithm assumes descending input, and feeding
it unsorted values silently mismatches labels and rectangles.  Panel (a) is
land-cover area, panel (b) carbon stock by sub-region.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pubstyle import set_pub_style, savefig, add_panel_label

COVER = [
    ("Forest", 38.6, "#1f6b4d"),
    ("Cropland", 26.4, "#c9a227"),
    ("Grassland", 15.2, "#7cb342"),
    ("Urban", 8.1, "#a4503f"),
    ("Wetland", 5.3, "#4fa5a0"),
    ("Water", 3.6, "#3d6b8e"),
    ("Barren", 2.8, "#b9b2a6"),
]
REGIONS = [
    ("Antu N", 28.4), ("Antu S", 24.9), ("Fusong E", 22.1),
    ("Fusong W", 19.6), ("Changbai N", 17.8), ("Changbai S", 15.2),
    ("Baishan", 13.4), ("Jingyu", 11.1), ("Linjiang", 9.3),
    ("Songjiang", 7.2), ("Erdao", 5.6), ("Songshan", 3.9),
]
REGION_RAMP = ["#0b3d2e", "#1f6b4d", "#2e8b63", "#5aa87e", "#8cc4a2"]


def squarify(values, x, y, w, h):
    """Rectangles for descending-sorted ``values`` inside the given box."""
    values = np.asarray(values, dtype=float)
    values = values / values.sum() * (w * h)
    rects = []
    _sq(list(values), [], x, y, w, h, rects)
    return rects


def _worst(row, length):
    total = sum(row)
    s2 = length ** 2
    r2 = total ** 2
    return max(s2 * max(row) / r2, r2 / (s2 * min(row)))


def _sq(vals, row, x, y, w, h, rects):
    if not vals:
        if row:
            _layout(row, x, y, w, h, rects)
        return
    length = min(w, h)
    if not row:
        row = [vals.pop(0)]
    if vals and _worst(row, length) >= _worst(row + [vals[0]], length):
        row.append(vals.pop(0))
        _sq(vals, row, x, y, w, h, rects)
    else:
        nx, ny, nw, nh = _layout(row, x, y, w, h, rects)
        _sq(vals, [], nx, ny, nw, nh, rects)


def _layout(row, x, y, w, h, rects):
    total = sum(row)
    if w >= h:
        rw = total / h
        oy = y
        for v in row:
            rects.append((x, oy, rw, v / rw))
            oy += v / rw
        return x + rw, y, w - rw, h
    rh = total / w
    ox = x
    for v in row:
        rects.append((ox, y, v / rh, rh))
        ox += v / rh
    return x, y + rh, w, h - rh


def text_color(hexcol):
    r, g, b = plt.matplotlib.colors.to_rgb(hexcol)
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.58 else "#2b2b2b"


def draw(ax, labels, values, colors, unit):
    order = np.argsort(values)[::-1]
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    colors = [colors[i] for i in order]
    total = sum(values)
    rects = squarify(values, 0, 0, 100, 100)

    for (x, y, w, h), lab, v, col in zip(rects, labels, values, colors):
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=col,
                                       edgecolor="white", lw=1.4))
        tc = text_color(col)
        share = f"{v / total * 100:.0f}%"
        full = (f"{lab}\n{v:.1f}%" if unit == "%"
                else f"{lab}\n{v:.1f} {unit}\n{share}")
        roomy = w > 13 and h > 13
        if roomy:
            ax.text(x + w / 2, y + h / 2, full, ha="center", va="center",
                    fontsize=7.2, color=tc, fontweight="bold",
                    linespacing=1.35)
        elif w > 7 and h > 7:
            ax.text(x + w / 2, y + h / 2, f"{lab}\n{share}", ha="center",
                    va="center", fontsize=6.0, color=tc, linespacing=1.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")
    return total


def main():
    set_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.4))

    labs = [c[0] for c in COVER]
    vals = [c[1] for c in COVER]
    cols = [c[2] for c in COVER]
    t0 = draw(axes[0], labs, vals, cols, "%")
    axes[0].set_title(f"Land-cover area share (total {t0:.0f}%)",
                      fontsize=9, fontweight="bold", pad=6)

    rlabs = [r[0] for r in REGIONS]
    rvals = [r[1] for r in REGIONS]
    ramp = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "reg", REGION_RAMP)
    rcols = [plt.matplotlib.colors.to_hex(ramp(1 - i / (len(REGIONS) - 1)))
             for i in np.argsort(np.argsort(rvals)[::-1])]
    t1 = draw(axes[1], rlabs, rvals, rcols, "Tg C")
    axes[1].set_title(f"Carbon stock by sub-region (total {t1:.0f} Tg C)",
                      fontsize=9, fontweight="bold", pad=6)

    for ax, lab in zip(axes, "ab"):
        add_panel_label(ax, f"({lab})", x=-0.02, y=1.0, size=10)
    fig.tight_layout(rect=(0, 0.045, 1, 1), w_pad=2.5)
    savefig(fig, "23_treemap.png")


if __name__ == "__main__":
    main()

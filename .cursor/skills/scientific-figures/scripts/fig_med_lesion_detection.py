"""Figure 377 - Brain metastasis detection on a contrast-enhanced T1w slice.

The main panel is the read: every box on it is a decision (true positive,
false positive, missed).  The two subordinate panels are the evidence - a FROC
curve whose marked operating point is the one drawn on the main panel, and the
same detections re-sliced by lesion diameter.  All three are computed from one
lesion table, so no number on the figure can drift away from the boxes.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, COLORS
from figkit import image_axes, axes_inches, panel_labels_inches, halo_text

N = 256
MM_PX = 220.0 / N                 # field of view / matrix
TP_C, FP_C, FN_C = "#2e7d5b", "#b23a48", "#c9a227"
CURVE_C = "#3b5b92"


# ---------------------------------------------------------------------------
# Synthetic brain phantom (layered ellipses + gyral folding, as in figure 253)
# ---------------------------------------------------------------------------
def _sm(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _sym(a):
    return 0.5 * (a + a[:, ::-1])


def _grid(n):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    return (xx - (n - 1) / 2) / (n / 2), (yy - (n - 1) / 2) / (n / 2)


def _rad(x, y, cx, cy, rx, ry, ang=0.0):
    t = np.deg2rad(ang)
    xr = (x - cx) * np.cos(t) + (y - cy) * np.sin(t)
    yr = -(x - cx) * np.sin(t) + (y - cy) * np.cos(t)
    return np.hypot(xr / rx, yr / ry)


def base_slice(seed=9):
    """Contrast-enhanced T1w axial slice plus the brain mask."""
    rng = np.random.default_rng(seed)
    x, y = _grid(N)
    wob = 1.0 + 0.015 * _sym(_sm(rng, N, 9))
    rr = _rad(x, y, 0.0, 0.0, 0.80, 0.92) * wob

    img = np.full((N, N), 0.02)
    img = np.where(rr <= 1.000, 0.50, img)                     # enhancing scalp
    img = np.where(rr <= 0.962, 0.08, img)                     # skull
    img = np.where(rr <= 0.900, 0.12, img)                     # CSF
    brain = rr <= 0.875
    img = np.where(brain, 0.40 + 0.030 * _sym(_sm(rng, N, 3)), img)
    rw = _rad(x, y, 0.0, 0.03, 0.62, 0.74) * wob
    img = np.where(brain & (rw <= 1.0), 0.58 + 0.025 * _sm(rng, N, 5), img)

    g1, g2 = _sym(_sm(rng, N, 7.0)), _sym(_sm(rng, N, 15.0))
    sulci = (np.sin(8.0 * g1 + 4.2 * g2) > 0.62) & brain & (rr > 0.58)
    img = np.where(sulci, 0.13, img)
    for vx, vy, vrx, vry, va in [(0.115, -0.02, 0.055, 0.22, -13),
                                 (-0.115, -0.02, 0.055, 0.22, 13),
                                 (0.0, 0.10, 0.030, 0.10, 0)]:
        img = np.where(_rad(x, y, vx, vy, vrx, vry, va) * wob <= 1.0, 0.07, img)
    img = np.where((np.abs(x) < 0.010) & brain, 0.55, img)     # falx enhances
    return img, brain, rng


# a fixed diameter ladder so every size stratum in panel (c) is populated
DIAMETERS = [3.1, 3.8, 4.6, 5.4, 6.6, 7.9, 8.8, 10.4, 12.1, 13.7, 15.2, 17.0,
             18.6, 20.5]
CUTOFF_MM = 7.5                   # detector's effective size threshold


def plant_lesions(img, brain, rng):
    """Sprinkle enhancing metastases; return the image and the lesion table."""
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    ys, xs = np.nonzero(brain)
    lesions = []
    order = rng.permutation(len(DIAMETERS))
    for idx in order:
        for _try in range(80):
            i = rng.integers(len(xs))
            cx, cy = float(xs[i]), float(ys[i])
            if all(np.hypot(cx - l["x"], cy - l["y"]) > 30 for l in lesions):
                break
        d_mm = DIAMETERS[idx]
        r_px = 0.5 * d_mm / MM_PX
        dist = np.hypot(xx - cx, yy - cy)
        soft = 1.0 / (1.0 + np.exp((dist - r_px) / 1.1))
        img = np.maximum(img, (0.86 + 0.10 * rng.random()) * soft)
        # detectability is size-driven, with borderline lesions going either way
        hit = bool(d_mm + rng.normal(0.0, 2.2) > CUTOFF_MM)
        lesions.append(dict(x=cx, y=cy, d=d_mm, r=r_px, hit=hit))
    img = gaussian_filter(img, 0.8) + 0.012 * rng.standard_normal((N, N))
    return np.clip(img, 0, 1.0), lesions


def spurious(rng, brain, lesions, n_fp=4):
    ys, xs = np.nonzero(brain)
    out = []
    while len(out) < n_fp:
        i = rng.integers(len(xs))
        cx, cy = float(xs[i]), float(ys[i])
        if (all(np.hypot(cx - l["x"], cy - l["y"]) > 34 for l in lesions)
                and all(np.hypot(cx - a, cy - b) > 34 for a, b, _ in out)):
            out.append((cx, cy, float(rng.uniform(4.0, 9.0))))
    return out


def box(ax, cx, cy, side, color, ls="-", lw=1.15):
    ax.add_patch(Rectangle((cx - side / 2 - 0.5, cy - side / 2 - 0.5), side,
                           side, facecolor="none", edgecolor=color,
                           linewidth=lw, linestyle=ls, zorder=6))


def main():
    set_pub_style(base_font=8)
    img, brain, rng = base_slice()
    img, lesions = plant_lesions(img, brain, rng)
    fps = spurious(rng, brain, lesions)

    n_tp = sum(l["hit"] for l in lesions)
    n_fn = len(lesions) - n_tp
    n_fp = len(fps)
    sens = n_tp / len(lesions)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.42, 0.46, 5.05, 5.05))
    image_axes(ax, color="#d5d5d5", lw=0.6)
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")

    placed = []
    for l in lesions:
        side = max(2.6 * l["r"], 13.0)
        box(ax, l["x"], l["y"], side, TP_C if l["hit"] else FN_C,
            ls="-" if l["hit"] else (0, (3, 2)))
        if l["hit"]:                 # only the misses are annotated: less
            continue                 # clutter, and it is the misses that carry
        tx, ty, ha, va = (l["x"] + side / 2 + 2, l["y"] - side / 2 - 2,
                          "left", "bottom")                  # the message
        if any(np.hypot(tx - a, ty - b) < 34 for a, b in placed):
            tx, ty, ha, va = (l["x"] - side / 2 - 2, l["y"] + side / 2 + 2,
                              "right", "top")
        placed.append((tx, ty))
        halo_text(ax, tx, ty, f"{l['d']:.1f} mm", ha=ha, va=va, fontsize=6.4,
                  color="#f2e6bf")
    for cx, cy, r in fps:
        box(ax, cx, cy, max(2.6 * r, 13.0), FP_C)

    handles = [Line2D([], [], color=TP_C, lw=1.4, label=f"true positive "
                      f"(n = {n_tp})"),
               Line2D([], [], color=FN_C, lw=1.4, ls=(0, (3, 2)),
                      label=f"missed lesion (n = {n_fn})"),
               Line2D([], [], color=FP_C, lw=1.4,
                      label=f"false positive (n = {n_fp})")]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.012), ncol=3, frameon=False,
              fontsize=7.2, handlelength=1.8, columnspacing=1.6)
    ax.set_title("Contrast-enhanced T1w, single slice of the screening study",
                 fontsize=9.0, fontweight="bold", pad=4.0)

    # -- subordinate panel 1: FROC, marked at the operating point above ------
    ax1 = axes_inches(fig, (6.42, 3.52, 3.52, 1.98))
    f = np.linspace(0.02, 8.0, 240)
    s_max, s_base = 0.955, 0.855
    k = -np.log(1.0 - sens / s_max) / n_fp
    ax1.plot(f, s_max * (1.0 - np.exp(-k * f)), color=CURVE_C, lw=1.6,
             label="ours")
    ax1.plot(f, s_base * (1.0 - np.exp(-0.45 * k * f)),
             color=COLORS["baseline"], lw=1.3, ls="--", label="baseline CAD")
    ax1.plot([n_fp], [sens], marker="o", ms=5.0, mfc="white", mec=CURVE_C,
             mew=1.4, zorder=5)
    ax1.annotate(f"operating point\n{n_fp} FP / scan, sens. {sens:.2f}",
                 xy=(n_fp, sens), xytext=(4.30, 0.155),
                 fontsize=6.6, color=COLORS["sub"], ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="-", lw=0.7,
                                 color=COLORS["baseline"]))
    ax1.set_xlim(0, 8.0)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlabel("false positives per scan (-)", fontsize=7.5)
    ax1.set_ylabel("sensitivity (-)", fontsize=7.5)
    ax1.tick_params(labelsize=6.5)
    ax1.grid(True, lw=0.5, alpha=0.7)
    ax1.legend(loc="upper left", frameon=False, fontsize=6.8)
    ax1.set_title("FROC", fontsize=8.5, fontweight="bold", pad=3.0)

    # -- subordinate panel 2: the same detections, stratified by size --------
    ax2 = axes_inches(fig, (6.42, 0.62, 3.52, 1.98))
    edges = [0.0, 5.0, 10.0, 15.0, 25.0]
    names = ["< 5", "5 - 10", "10 - 15", "> 15"]
    rate, tot = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        grp = [l for l in lesions if lo <= l["d"] < hi]
        tot.append(len(grp))
        rate.append(np.mean([l["hit"] for l in grp]) if grp else np.nan)
    xs = np.arange(len(names))
    ax2.bar(xs, rate, width=0.62, color=CURVE_C, edgecolor="white",
            linewidth=0.7)
    for xi, r, t in zip(xs, rate, tot):
        if np.isfinite(r):
            ax2.text(xi, r + 0.035, f"{r:.0%}", ha="center", va="bottom",
                     fontsize=6.6, color=COLORS["ink"])
        ax2.text(xi, -0.085, f"n = {t}", ha="center", va="top", fontsize=6.2,
                 color=COLORS["muted"])
    ax2.axhline(sens, color=FP_C, lw=0.9, ls=":", zorder=1)
    ax2.text(0.5, sens + 0.02, f"overall {sens:.0%}", ha="center",
             va="bottom", fontsize=6.4, color=FP_C)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(names)
    ax2.set_ylim(0, 1.12)
    ax2.set_xlabel("lesion diameter (mm)", fontsize=7.5, labelpad=9.0)
    ax2.set_ylabel("detection rate (-)", fontsize=7.5)
    ax2.tick_params(labelsize=6.5)
    ax2.grid(True, axis="y", lw=0.5, alpha=0.7)
    ax2.set_title("Detection rate by lesion size", fontsize=8.5,
                  fontweight="bold", pad=3.0)

    panel_labels_inches(fig, [ax, ax1, ax2], dx=-0.30, dy=0.16)

    savefig(fig, "377_med_lesion_detection.png")


if __name__ == "__main__":
    main()

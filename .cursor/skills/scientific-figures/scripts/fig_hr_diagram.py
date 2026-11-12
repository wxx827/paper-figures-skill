"""Hertzsprung-Russell diagram of a synthetic stellar population.

Stars are drawn from a power-law mass function on the main sequence plus
hand-placed giant, supergiant and white-dwarf populations, then coloured by
effective temperature with a black-body-like ramp.  Both astronomical
conventions are respected: the temperature axis increases to the left and the
luminosity axis is logarithmic.  Dashed diagonals are loci of constant radius,
L = 4*pi*R^2*sigma*Teff^4, so the vertical distance between a star and the
R = 1 Rsun line reads directly as a radius ratio.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.spatial import ConvexHull
from pubstyle import set_pub_style, savefig

# ---- shared astronomy palette (identical across figures 233-242) ----------
NAVY = "#1f3355"
INDIGO = "#5b4b8a"
GOLD = "#c9962f"
CYAN = "#3f7f95"
RUST = "#a8523a"
GREY = "#8b9099"

# black-body-like ramp: cool red -> gold -> white -> blue
BB_COLORS = ["#7a2a1e", "#ab4a27", "#d18b39", "#eed6a5", "#f6f4ee",
             "#cfe0f1", "#94b8dd", "#5d86bc"]

TSUN = 5772.0
XLIM = (4.66, 3.37)          # log10 Teff, inverted (hot on the left)
YLIM = (-5.1, 6.4)           # log10 L / Lsun


def sample_population(rng):
    """Return (log10 Teff, log10 L/Lsun) for four stellar groups."""
    groups = {}

    # main sequence: power-law masses, Teff and L scaling with mass
    a, mlo, mhi = 1.65, 0.11, 22.0
    u = rng.random(2400)
    m = ((mhi ** (1 - a) - mlo ** (1 - a)) * u + mlo ** (1 - a)) ** (1 / (1 - a))
    lt = np.log10(TSUN * m ** 0.53) + rng.normal(0, 0.012, m.size)
    ll = 3.55 * np.log10(m) + rng.normal(0, 0.10, m.size)
    ll -= 0.55 * np.clip(np.log10(m) - 0.9, 0, None)      # gentle turn-off
    groups["ms"] = (lt, ll)

    # red-giant branch plus a red clump overdensity
    n = 260
    ll_g = rng.uniform(0.9, 3.5, n)
    lt_g = np.log10(4950) - 0.070 * (ll_g - 0.9) + rng.normal(0, 0.014, n)
    nc = 110
    ll_c = rng.normal(1.72, 0.11, nc)
    lt_c = np.log10(4820) + rng.normal(0, 0.010, nc)
    groups["giant"] = (np.concatenate([lt_g, lt_c]),
                       np.concatenate([ll_g, ll_c]))

    # supergiants: a broad, nearly horizontal band at high luminosity
    n = 46
    lt_s = rng.uniform(np.log10(3500), np.log10(19000), n)
    ll_s = 4.75 + 0.45 * (lt_s - np.log10(3500)) + rng.normal(0, 0.16, n)
    groups["sg"] = (lt_s, ll_s)

    # white dwarfs: R ~ 0.013 Rsun, so they track a low isoradius line
    n = 170
    lt_w = rng.uniform(np.log10(5200), np.log10(38000), n)
    ll_w = 2 * np.log10(0.0135) + 4 * (lt_w - np.log10(TSUN))
    ll_w += rng.normal(0, 0.13, n)
    groups["wd"] = (lt_w, ll_w)
    return groups


def region_outline(x, y, xr, yr, pad=0.022, smooth=9, n=260):
    """Smoothed convex hull, padded by a constant screen-fraction margin."""
    p = np.column_stack([(np.asarray(x) - xr[0]) / (xr[1] - xr[0]),
                         (np.asarray(y) - yr[0]) / (yr[1] - yr[0])])
    v = p[ConvexHull(p).vertices]
    d = v - v.mean(0)
    v = v + pad * d / np.maximum(np.hypot(*d.T), 1e-9)[:, None]
    v = np.vstack([v, v[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(v, axis=0).T))])
    t = np.linspace(0, s[-1], n, endpoint=False)
    q = np.column_stack([np.interp(t, s, v[:, 0]), np.interp(t, s, v[:, 1])])
    k = np.ones(smooth) / smooth
    for _ in range(3):
        q = np.column_stack([
            np.convolve(np.r_[q[-smooth:, i], q[:, i], q[:smooth, i]], k,
                        "same")[smooth:-smooth] for i in (0, 1)])
    q = np.vstack([q, q[:1]])
    return (q[:, 0] * (xr[1] - xr[0]) + xr[0],
            q[:, 1] * (yr[1] - yr[0]) + yr[0])


def data_angle(ax, dx, dy):
    """Screen angle (deg) of a data-space direction."""
    p0 = ax.transData.transform((0.0, 0.0))
    p1 = ax.transData.transform((dx, dy))
    return float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))


SPECTRAL = [("O", 4.480, 4.663, "#7fa7d4"), ("B", 4.000, 4.480, "#a7c4e4"),
            ("A", 3.875, 4.000, "#d8e3f0"), ("F", 3.778, 3.875, "#f4f1e6"),
            ("G", 3.716, 3.778, "#f2dfa8"), ("K", 3.568, 3.716, "#e0a663"),
            ("M", 3.370, 3.568, "#c1653f")]


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(11)
    g = sample_population(rng)

    cmap = mpl.colors.LinearSegmentedColormap.from_list("bb", BB_COLORS)
    norm = mpl.colors.Normalize(np.log10(2900), np.log10(33000))

    fig = plt.figure(figsize=(8.8, 8.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.055, 1.0, 0.30],
                          hspace=0.085, left=0.088, right=0.855,
                          top=0.905, bottom=0.072)
    ax_sp = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])
    ax_h = fig.add_subplot(gs[2])

    # ---- isoradius grid ---------------------------------------------------
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    xs = np.linspace(XLIM[1], XLIM[0], 60)
    for r in [0.01, 0.1, 1.0, 10.0, 100.0]:
        ys = 2 * np.log10(r) + 4 * (xs - np.log10(TSUN))
        ax.plot(xs, ys, ls=(0, (5, 3)), lw=0.7, color="#b9bfc8", zorder=1)

    # ---- stars ------------------------------------------------------------
    order = [("ms", 5.2, 0.80), ("giant", 12.0, 0.88),
             ("sg", 30.0, 0.95), ("wd", 8.0, 0.85)]
    for key, size, alpha in order:
        lt, ll = g[key]
        ax.scatter(lt, ll, s=size, c=cmap(norm(lt)), alpha=alpha,
                   linewidths=0.0, zorder=3, rasterized=False)

    # the Sun
    ax.plot(np.log10(TSUN), 0.0, marker="o", ms=8.5, mfc="#f5c542",
            mec="#3d3018", mew=0.9, ls="none", zorder=6)
    ax.annotate(r"Sun  $T_{\rm eff}=5772$ K", xy=(np.log10(TSUN), 0.0),
                xytext=(3.845, -1.35), fontsize=7.4, color="#3d3018",
                ha="left", va="center", zorder=7,
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#7a6a3a",
                                shrinkA=0, shrinkB=5))

    # ---- region call-outs -------------------------------------------------
    regions = [
        ("ms", "Main sequence\ncore H burning", (4.36, -1.05), NAVY, 0.020,
         None),
        ("giant", "Red-giant branch\n+ red clump", (3.78, 3.55), RUST, 0.026,
         None),
        ("sg", "Supergiants", (4.55, 5.90), GOLD, 0.024, (4.27, 5.12)),
        ("wd", "White dwarfs\n$R\\approx0.013\\,R_\\odot$", (4.51, -4.30),
         INDIGO, 0.022, (4.10, -2.90)),
    ]
    for key, label, tp, col, exp, tgt in regions:
        lt, ll = g[key]
        ox, oy = region_outline(lt, ll, XLIM, YLIM, pad=exp)
        ax.plot(ox, oy, ls=(0, (4, 2.6)), lw=1.0, color=col, alpha=0.85,
                zorder=4)
        cx, cy = tgt if tgt else (float(np.mean(lt)), float(np.mean(ll)))
        ax.annotate(label, xy=(cx, cy), xytext=tp, fontsize=7.8, color=col,
                    fontweight="bold", ha="center", va="center",
                    linespacing=1.45, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.32", fc="white",
                              ec=col, lw=0.6, alpha=0.90),
                    arrowprops=dict(arrowstyle="-", lw=0.7, color=col,
                                    alpha=0.7, shrinkA=2, shrinkB=6,
                                    connectionstyle="arc3,rad=0.15"))

    # isoradius labels, rotated along their own slope
    ang = data_angle(ax, -0.1, -0.4)
    for r, ytxt in [(0.01, -3.9), (0.1, -1.9), (1.0, 0.15),
                    (10.0, 2.2), (100.0, 4.25)]:
        xt = np.log10(TSUN) + (ytxt - 2 * np.log10(r)) / 4.0
        lab = (r"$%g\,R_\odot$" % r) if r != 1 else r"$1\,R_\odot$"
        ax.text(xt, ytxt, lab, fontsize=6.9, color="#7d848e", rotation=ang,
                rotation_mode="anchor", ha="center", va="center", zorder=2,
                bbox=dict(boxstyle="round,pad=0.13", fc="white", ec="none",
                          alpha=0.85))

    ax.set_ylabel(r"Luminosity  $L/L_\odot$", fontsize=9, labelpad=3)
    ax.set_yticks(np.arange(-5, 7, 1))
    ax.set_yticklabels([r"$10^{%d}$" % v for v in range(-5, 7)])
    tt = [30000, 20000, 10000, 7000, 5000, 4000, 3000]
    ax.set_xticks([np.log10(v) for v in tt])
    ax.set_xticklabels([])
    ax.tick_params(labelsize=7.4, length=3)
    ax.grid(True, lw=0.4, color="#eceef1")
    ax.set_axisbelow(True)

    handles = [Line2D([], [], ls=(0, (5, 3)), lw=0.8, color="#b9bfc8",
                      label="constant radius"),
               Line2D([], [], ls=(0, (4, 2.6)), lw=1.0, color=GREY,
                      label="population envelope"),
               Line2D([], [], marker="o", ls="none", ms=6, mfc="#f5c542",
                      mec="#3d3018", mew=0.8, label="the Sun")]
    leg = ax.legend(handles=handles, loc="lower right", fontsize=7.2,
                    frameon=True, handlelength=1.9,
                    bbox_to_anchor=(0.995, 0.045))
    leg.get_frame().set_linewidth(0.6)
    leg.get_frame().set_edgecolor("#cccccc")

    # ---- spectral-class strip (top) ---------------------------------------
    ax_sp.set_xlim(*XLIM)
    ax_sp.set_ylim(0, 1)
    for name, lo, hi, col in SPECTRAL:
        ax_sp.add_patch(Rectangle((lo, 0), hi - lo, 1, fc=col, ec="white",
                                  lw=0.8))
        ax_sp.text((lo + hi) / 2, 0.44, name, ha="center", va="center",
                   fontsize=7.6, fontweight="bold",
                   color="#2b2b2b" if name in "AFG" else "#1a1a1a")
    ax_sp.set_xticks([])
    ax_sp.set_yticks([])
    for s in ax_sp.spines.values():
        s.set_visible(False)
    ax_sp.text(1.008, 0.44, "spectral class", transform=ax_sp.transAxes,
               ha="left", va="center", fontsize=7.0, color="#666")

    # ---- demoted strip: counts per temperature bin ------------------------
    all_lt = np.concatenate([g[k][0] for k in g])
    bins = np.linspace(XLIM[1], XLIM[0], 46)
    ax_h.set_facecolor("#f4f5f7")
    ax_h.hist(all_lt, bins=bins, color=NAVY, alpha=0.55, lw=0.0)
    ax_h.hist(all_lt, bins=bins, histtype="step", color=NAVY, lw=0.8)
    ax_h.set_xlim(*XLIM)
    ax_h.set_xticks([np.log10(v) for v in tt])
    ax_h.set_xticklabels(["%d" % v for v in tt])
    ax_h.set_xlabel(r"Effective temperature  $T_{\rm eff}$ (K)"
                    "    (hotter to the left)", fontsize=9, labelpad=2.5)
    ax_h.set_ylabel("stars / bin", fontsize=7.4, labelpad=2)
    ax_h.tick_params(labelsize=6.6, length=2.2)
    ax_h.grid(True, axis="y", lw=0.4, color="#e3e5e8")
    ax_h.set_axisbelow(True)
    for s in ax_h.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#9aa0a6")
    ax_h.set_title("Supporting statistic  |  temperature distribution of the "
                   "same %d stars" % all_lt.size, fontsize=7.2,
                   style="italic", color="#555555", pad=2.5)

    # ---- colourbar --------------------------------------------------------
    pos = ax.get_position()
    cax = fig.add_axes([0.876, pos.y0 + 0.20 * pos.height, 0.017,
                        0.58 * pos.height])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(r"$T_{\rm eff}$ (K)", fontsize=7.6, labelpad=4)
    ct = [3000, 5000, 8000, 15000, 30000]
    cb.set_ticks([np.log10(v) for v in ct])
    cb.set_ticklabels(["%d" % v for v in ct])
    cb.ax.tick_params(labelsize=6.8, width=0.6, length=2.4)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    savefig(fig, "233_hr_diagram.png")


if __name__ == "__main__":
    main()

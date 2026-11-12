"""Field spectral signature library for five land-cover classes.

Main panel: 400-2500 nm reflectance means with +/-1 sigma envelopes, grey
masks over the two strong atmospheric water-vapour windows, and markers for
the chlorophyll absorption trough and the vegetation red edge.

Auxiliary panels (58 % of the main panel): broad-band mean reflectance per
class, and the pairwise class separability matrix derived from those same
band statistics.

Reflectance curves are interpolated from hand-placed anchor points and given
a smooth per-class spread, so the shapes are textbook-correct while the
numbers are synthetic - this is a visual reference figure, not a data product.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

from pubstyle import set_pub_style, savefig, style_axes, get_cmap
from figkit import axes_inches, panel_labels_inches, safe_colorbar

# Semantic land-cover colours, all taken from the house categorical order.
CLASSES = [
    ("Vegetation", "Veg", "#2e7d5b"),
    ("Bare soil", "Soil", "#c0603f"),
    ("Water", "Water", "#3b5b92"),
    ("Built-up", "Built", "#b23a48"),
    ("Snow", "Snow", "#8e6bbf"),
]

# (wavelength nm, reflectance fraction) anchor points per class.
ANCHORS = {
    "Vegetation": [
        (400, 0.030), (450, 0.036), (500, 0.046), (550, 0.088), (600, 0.056),
        (670, 0.031), (700, 0.078), (720, 0.235), (750, 0.425), (800, 0.470),
        (900, 0.482), (1000, 0.470), (1100, 0.452), (1200, 0.428),
        (1300, 0.398), (1450, 0.175), (1600, 0.292), (1700, 0.268),
        (1800, 0.198), (1950, 0.076), (2100, 0.182), (2200, 0.164),
        (2300, 0.131), (2400, 0.101), (2500, 0.086),
    ],
    "Bare soil": [
        (400, 0.074), (500, 0.106), (600, 0.146), (700, 0.190), (800, 0.231),
        (900, 0.256), (1000, 0.276), (1100, 0.291), (1200, 0.301),
        (1300, 0.312), (1450, 0.232), (1600, 0.346), (1700, 0.351),
        (1800, 0.331), (1950, 0.221), (2100, 0.332), (2200, 0.286),
        (2300, 0.301), (2400, 0.281), (2500, 0.266),
    ],
    "Water": [
        (400, 0.046), (450, 0.051), (500, 0.045), (550, 0.038), (600, 0.026),
        (700, 0.013), (800, 0.008), (900, 0.006), (1100, 0.005),
        (1300, 0.004), (1600, 0.003), (2000, 0.003), (2500, 0.002),
    ],
    "Built-up": [
        (400, 0.116), (500, 0.136), (600, 0.156), (700, 0.171), (800, 0.186),
        (900, 0.196), (1000, 0.206), (1100, 0.216), (1200, 0.221),
        (1300, 0.226), (1450, 0.181), (1600, 0.246), (1700, 0.246),
        (1800, 0.236), (1950, 0.176), (2100, 0.236), (2200, 0.231),
        (2300, 0.221), (2400, 0.211), (2500, 0.201),
    ],
    "Snow": [
        (400, 0.958), (500, 0.972), (600, 0.960), (700, 0.928), (800, 0.868),
        (900, 0.758), (1000, 0.618), (1100, 0.421), (1200, 0.330),
        (1300, 0.181), (1450, 0.046), (1600, 0.076), (1700, 0.055),
        (1800, 0.031), (1950, 0.011), (2100, 0.031), (2200, 0.028),
        (2300, 0.021), (2400, 0.016), (2500, 0.013),
    ],
}

# Strong atmospheric water-vapour absorption windows (nm).
VAPOUR = [(1338, 1452), (1798, 1962)]

# Broad bands of a typical multispectral sensor (nm).
BANDS = [
    ("Blue", 450, 515), ("Green", 525, 600), ("Red", 630, 690),
    ("NIR", 770, 900), ("SWIR", 1560, 1750),
]


def smooth_noise(rng, n, sigma):
    """Unit-variance noise with a slow correlation length."""
    z = gaussian_filter1d(rng.normal(size=n), sigma, mode="nearest")
    return z / (z.std() + 1e-12)


def build_library(rng, wl, n_spec):
    """Draw an ensemble of field spectra per class; return its mean and sd."""
    lib = {}
    for name, _, _ in CLASSES:
        a = np.asarray(ANCHORS[name], dtype=float)
        base = PchipInterpolator(a[:, 0], a[:, 1])(wl)
        base = np.clip(gaussian_filter1d(base, 2.0, mode="nearest"), 0.0, 1.0)
        ens = np.empty((n_spec, wl.size))
        for k in range(n_spec):
            # multiplicative brightness drift plus a small additive floor
            ens[k] = (base * (1.0 + 0.10 * smooth_noise(rng, wl.size, 26.0))
                      + 0.008 * smooth_noise(rng, wl.size, 10.0))
        ens = np.clip(ens, 0.0, 1.0)
        lib[name] = (100.0 * ens.mean(axis=0), 100.0 * ens.std(axis=0))
    return lib


def band_stats(wl, lib):
    """Mean and pooled spread of every class inside every broad band."""
    mu = np.zeros((len(CLASSES), len(BANDS)))
    sd = np.zeros_like(mu)
    for i, (name, _, _) in enumerate(CLASSES):
        m, s = lib[name]
        for j, (_, lo, hi) in enumerate(BANDS):
            sel = (wl >= lo) & (wl <= hi)
            mu[i, j] = m[sel].mean()
            sd[i, j] = np.sqrt((s[sel] ** 2).mean())
    return mu, sd


def separability(mu, sd):
    """Normalised spectral distance between every pair of classes."""
    n = mu.shape[0]
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            num = (mu[i] - mu[j]) ** 2
            den = sd[i] ** 2 + sd[j] ** 2
            d[i, j] = np.sqrt((num / den).mean())
    return d


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(756)

    n_spec = 48
    wl = np.linspace(400.0, 2500.0, 421)
    lib = build_library(rng, wl, n_spec)
    mu, sd = band_stats(wl, lib)
    sep = separability(mu, sd)

    FW, FH = 10.4, 6.4
    fig = plt.figure(figsize=(FW, FH))

    ax = axes_inches(fig, (0.78, 1.05, 5.25, 4.45))
    ax_bar = axes_inches(fig, (6.86, 3.75, 3.04, 1.75))
    ax_sep = axes_inches(fig, (7.14, 1.05, 1.80, 1.80))

    # ---- (a) full-range spectral signatures ------------------------------
    for lo, hi in VAPOUR:
        ax.axvspan(lo, hi, facecolor="#cccccc", alpha=0.62, lw=0, zorder=0)
        ax.text((lo + hi) / 2.0, 99.0, "H$_2$O", ha="center", va="top",
                fontsize=6.2, color="#555555", rotation=90, zorder=1)

    for name, _, col in CLASSES:
        m, s = lib[name]
        ax.fill_between(wl, m - s, m + s, color=col, alpha=0.20, lw=0,
                        zorder=2)
        ax.plot(wl, m, color=col, lw=1.5, label=name, zorder=3)

    ax.set_xlim(400, 2500)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance (%)")
    ax.set_title("Laboratory-grade signatures over the reflective spectrum",
                 fontsize=9.0, fontweight="bold")
    style_axes(ax)
    ax.grid(axis="y", lw=0.5, alpha=0.45)

    sec = ax.secondary_xaxis("top", functions=(lambda v: v / 1000.0,
                                               lambda v: v * 1000.0))
    sec.set_xlabel("Wavelength ($\\mu$m)", fontsize=7.6, labelpad=2.0)
    sec.tick_params(labelsize=6.8, width=0.7, length=2.5)

    veg = lib["Vegetation"][0]
    i670 = int(np.argmin(np.abs(wl - 670.0)))
    ax.annotate("Chlorophyll trough\n"
                + "%.0f nm, %.1f %%" % (wl[i670], veg[i670]),
                xy=(wl[i670], veg[i670]), xytext=(810, 15.0),
                fontsize=6.4, color="#1a5c40", ha="left", va="center",
                linespacing=1.3,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#1a5c40",
                                shrinkA=1.0, shrinkB=1.5))

    # Red edge: steepest rise of the vegetation curve between 680 and 780 nm.
    win = (wl >= 680) & (wl <= 780)
    slope = np.gradient(veg, wl)
    i_re = int(np.arange(wl.size)[win][np.argmax(slope[win])])
    ax.axvspan(690, 760, facecolor="#2e7d5b", alpha=0.10, lw=0, zorder=1)
    ax.annotate("Red edge\ninflection %.0f nm" % wl[i_re],
                xy=(wl[i_re], veg[i_re]), xytext=(810, 58.0),
                fontsize=6.4, color="#1a5c40", ha="left", va="center",
                linespacing=1.3,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#1a5c40",
                                shrinkA=1.0, shrinkB=1.5))

    handles = [Line2D([], [], color=c, lw=1.6) for _, _, c in CLASSES]
    handles.append(Line2D([], [], color="#cccccc", lw=5.0, alpha=0.8))
    labels = [n for n, _, _ in CLASSES] + ["Vapour window"]
    ax.legend(handles, labels, loc="upper right", fontsize=7.0,
              frameon=False, labelspacing=0.30, handlelength=1.5,
              bbox_to_anchor=(0.995, 1.0))

    ax.text(0.014, 0.965, "shaded band = $\\pm$1$\\sigma$ of n = %d field "
            "spectra per class" % n_spec, transform=ax.transAxes, ha="left",
            va="top", fontsize=6.4, color="#777777")

    # ---- (b) broad-band means --------------------------------------------
    nb = len(BANDS)
    xb = np.arange(nb, dtype=float)
    w = 0.15
    for i, (name, _, col) in enumerate(CLASSES):
        off = (i - (len(CLASSES) - 1) / 2.0) * w
        ax_bar.bar(xb + off, mu[i], width=w * 0.92, color=col,
                   edgecolor="white", linewidth=0.35, zorder=3)
        ax_bar.errorbar(xb + off, mu[i], yerr=sd[i], fmt="none",
                        ecolor="#555555", elinewidth=0.55, capsize=1.1,
                        capthick=0.55, zorder=4)
    ax_bar.set_xticks(xb)
    ax_bar.set_xticklabels(["%s\n%d-%d" % (n, lo, hi)
                            for n, lo, hi in BANDS], fontsize=6.2)
    ax_bar.set_ylim(0, 105)
    ax_bar.set_ylabel("Band mean (%)", fontsize=7.4)
    ax_bar.set_xlabel("Wavelength range (nm)", fontsize=7.4, labelpad=0.5)
    ax_bar.set_title("Mean reflectance per broad band", fontsize=8.5,
                     fontweight="bold")
    ax_bar.tick_params(labelsize=6.6)
    style_axes(ax_bar)
    ax_bar.grid(axis="y", lw=0.5, alpha=0.45)

    # ---- (c) class separability matrix ------------------------------------
    im = ax_sep.imshow(sep, cmap=get_cmap("cb_sequential_blue"),
                       vmin=0.0, vmax=float(sep.max()),
                       interpolation="nearest")
    ax_sep.set_aspect("equal")
    short = [s for _, s, _ in CLASSES]
    ax_sep.set_xticks(range(len(short)))
    ax_sep.set_yticks(range(len(short)))
    ax_sep.set_xticklabels(short, fontsize=6.3, rotation=38, ha="right")
    ax_sep.set_yticklabels(short, fontsize=6.3)
    ax_sep.tick_params(width=0.7, length=2.0)
    for s in ax_sep.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    hi = float(sep.max())
    for i in range(len(short)):
        for j in range(len(short)):
            if i == j:
                ax_sep.text(j, i, "-", ha="center", va="center",
                            fontsize=6.3, color="#777777")
                continue
            ax_sep.text(j, i, "%.1f" % sep[i, j], ha="center", va="center",
                        fontsize=6.0,
                        color="white" if sep[i, j] > 0.55 * hi else "#1a1a1a")
    ax_sep.set_title("Pairwise separability", fontsize=8.5, fontweight="bold")
    safe_colorbar(fig, im, ax_sep, label="distance (-)", width=0.10,
                  pad=0.07, labelsize=6.4)

    worst = np.unravel_index(np.argmin(sep + 1e6 * np.eye(len(short))),
                             sep.shape)
    ax_sep.text(0.5, -0.32,
                "least separable pair: %s vs %s (%.1f)"
                % (short[worst[0]], short[worst[1]], sep[worst]),
                transform=ax_sep.transAxes, ha="center", va="top",
                fontsize=6.2, color="#777777")

    panel_labels_inches(fig, [ax, ax_bar, ax_sep], dx=-0.42, dy=0.16)


    savefig(fig, "756_rs_spectral_signature.png")


if __name__ == "__main__":
    main()

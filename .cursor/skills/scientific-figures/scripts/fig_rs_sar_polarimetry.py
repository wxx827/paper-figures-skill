"""Polarimetric SAR decomposition and H-alpha classification.

Top row: Pauli RGB composite (|HH-VV| red, |HV| green, |HH+VV| blue) and the
Cloude-Pottier entropy-alpha plane with its nine canonical zones and the
feasible-region boundary curves.
Bottom row: multi-polarisation backscatter by land cover and the
surface / double-bounce / volume power partition of the same classes.

Scattering mechanisms are consistent across panels: water and bare soil are
surface dominated (blue), forest is volume dominated (green), built-up areas
are double-bounce dominated (red).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, PALETTES, savefig, style_axes, add_panel_label

N = 220                      # 10 m pixels -> 2.2 km square
LOOKS = 6

CLASSES = ["Water", "Bare soil", "Cropland", "Forest", "Built-up"]
CCOL = ["#2166ac", "#c9a227", "#6bbf8a", "#0b3d2e", "#8c2f2f"]
# mean linear power of the three Pauli channels per class (arbitrary units)
PAULI_MEAN = {
    "Water": (0.004, 0.002, 0.060),
    "Bare soil": (0.045, 0.020, 0.230),
    "Cropland": (0.070, 0.120, 0.180),
    "Forest": (0.110, 0.330, 0.130),
    "Built-up": (0.620, 0.150, 0.170),
}
# backscatter coefficients (dB): HH, VV, HV
SIGMA0 = {
    "Water": (-22.5, -20.8, -30.5),
    "Bare soil": (-11.2, -9.6, -21.8),
    "Cropland": (-9.4, -8.1, -16.2),
    "Forest": (-7.6, -8.3, -12.9),
    "Built-up": (-3.1, -5.2, -10.8),
}
SIGMA0_SD = {
    "Water": (2.4, 2.2, 2.6),
    "Bare soil": (1.7, 1.6, 1.9),
    "Cropland": (1.5, 1.4, 1.5),
    "Forest": (1.1, 1.1, 1.0),
    "Built-up": (2.6, 2.5, 2.3),
}
# Freeman-Durden style power partition: surface / double / volume
PARTITION = {
    "Water": (0.93, 0.02, 0.05),
    "Bare soil": (0.76, 0.08, 0.16),
    "Cropland": (0.44, 0.11, 0.45),
    "Forest": (0.14, 0.11, 0.75),
    "Built-up": (0.19, 0.63, 0.18),
}
# H-alpha cluster centres (H, alpha deg) and spread
HA_CENTRE = {
    "Water": (0.22, 12.0, 0.07, 5.0),
    "Bare soil": (0.42, 24.0, 0.09, 6.0),
    "Cropland": (0.68, 38.0, 0.09, 7.0),
    "Forest": (0.88, 47.0, 0.05, 6.0),
    "Built-up": (0.58, 61.0, 0.11, 7.5),
}


def norm01(a):
    a = np.asarray(a, float)
    return (a - a.min()) / (a.max() - a.min() + 1e-12)


def fractal_noise(rng, shape, sigmas=(18, 7, 2.5), weights=(1.0, 0.5, 0.22)):
    out = np.zeros(shape)
    for s, w in zip(sigmas, weights):
        out += w * gaussian_filter(rng.normal(size=shape), s, mode="reflect")
    return norm01(out)


def build_classmap(rng):
    yy, xx = np.mgrid[0:N, 0:N]
    elev = fractal_noise(rng, (N, N), (22, 9, 3))

    river = np.abs(xx - (52 + 22 * np.sin(yy / 34.0) + 7 * np.sin(yy / 11.0))) < 4.0
    pond = ((xx - 178.0) ** 2 / 15.0 ** 2 + (yy - 40.0) ** 2 / 9.0 ** 2) < 1.0
    water = river | pond

    urban = np.exp(-(((xx - 145.0) / 34.0) ** 2 + ((yy - 150.0) / 27.0) ** 2))
    urban = (urban + 0.42 * (fractal_noise(rng, (N, N), (6, 2)) - 0.45)) > 0.44
    # street grid inside the built-up area
    streets = ((xx % 26) < 3) | ((yy % 24) < 3)
    urban &= ~(streets & (fractal_noise(rng, (N, N), (30,), (1.0,)) > 0.2))

    forest = (elev > np.quantile(elev, 0.72)) & (~urban) & (~water)

    block = (yy // 14) * 512 + (xx // 18)
    kind = rng.choice([1, 2], size=block.max() + 1, p=[0.35, 0.65])
    lc = kind[block]
    lc[forest] = 3
    lc[urban] = 4
    lc[water] = 0
    return lc


def pauli_rgb(rng, lc):
    """Speckled Pauli composite; multi-look gamma statistics."""
    chan = np.zeros((N, N, 3))
    for i, name in enumerate(CLASSES):
        m = lc == i
        if not m.any():
            continue
        mu = PAULI_MEAN[name]
        for c in range(3):
            chan[..., c][m] = mu[c]
    # spatial texture then speckle
    tex = 0.55 + 0.9 * fractal_noise(rng, (N, N), (7, 2.5))[..., None]
    chan = chan * tex
    speckle = rng.gamma(LOOKS, 1.0 / LOOKS, size=chan.shape)
    chan = chan * speckle

    db = 10.0 * np.log10(np.clip(chan, 1e-5, None))
    out = np.zeros_like(db)
    for c in range(3):
        lo, hi = np.percentile(db[..., c], [3, 98])
        out[..., c] = np.clip((db[..., c] - lo) / (hi - lo), 0, 1)
    return out ** 0.95


def feasible_curves(n=400):
    """Cloude-Pottier feasible region for the eigenvalue set [1, m, m].

    Entropy H(m) runs from 0 (m = 0) to 1 (m = 1); the two extreme
    eigenvector configurations give the upper and lower alpha bounds, which
    converge on the random-volume point (H = 1, alpha = 60 deg).
    """
    m = np.linspace(0.0, 1.0, n)
    p1 = 1.0 / (1.0 + 2.0 * m)
    p2 = m / (1.0 + 2.0 * m)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(p1 * np.log(p1)
              + 2 * np.where(p2 > 0, p2 * np.log(np.maximum(p2, 1e-300)), 0.0)
              ) / np.log(3)
    h[0] = 0.0
    upper = 90.0 * (1.0 + m) / (1.0 + 2.0 * m)
    lower = 90.0 * (2.0 * m) / (1.0 + 2.0 * m)
    return h, upper, lower


def frame_raster(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_edgecolor("#333333")


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(20240188)

    lc = build_classmap(rng)
    rgb = pauli_rgb(rng, lc)

    fig = plt.figure(figsize=(7.3, 5.6))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.14],
                  height_ratios=[1.0, 0.78], hspace=0.42, wspace=0.26)

    # ---- (a) Pauli RGB ----------------------------------------------------
    axa = fig.add_subplot(gs[0, 0])
    axa.imshow(rgb, origin="lower", interpolation="bilinear")
    frame_raster(axa)
    axa.set_title("Pauli RGB composite", fontsize=8.4)
    add_panel_label(axa, "(a)", x=-0.035, y=1.02, size=10)
    handles = [
        Patch(fc="#b23a48", ec="none", label="R  |HH\u2212VV|  double bounce"),
        Patch(fc="#2e7d5b", ec="none", label="G  |HV|  volume"),
        Patch(fc="#3b5b92", ec="none", label="B  |HH+VV|  surface"),
    ]
    lg = axa.legend(handles=handles, loc="upper center",
                    bbox_to_anchor=(0.5, -0.02), ncol=1, fontsize=6.2,
                    frameon=False, handlelength=1.0, handleheight=0.9,
                    labelspacing=0.25, borderaxespad=0.0)
    lg.set_clip_on(False)

    # ---- (b) H-alpha plane ------------------------------------------------
    axb = fig.add_subplot(gs[0, 1])
    hh, up_c, lo_c = feasible_curves()
    upper = np.interp(np.linspace(0, 1, 400), hh, up_c)
    lower = np.interp(np.linspace(0, 1, 400), hh, lo_c)
    hh_g = np.linspace(0, 1, 400)
    axb.fill_between(hh_g, upper, 90, color="#ececec", lw=0, zorder=1)
    axb.fill_between(hh_g, 0, lower, color="#ececec", lw=0, zorder=1)
    axb.plot(hh, up_c, color="#333333", lw=0.9, zorder=4)
    axb.plot(hh, lo_c, color="#333333", lw=0.9, zorder=4)
    axb.text(0.50, 86, "unfeasible", fontsize=6.2, color="#777777",
             style="italic", zorder=5, ha="center")
    axb.text(0.30, 3.0, "unfeasible", fontsize=6.2, color="#777777",
             style="italic", zorder=5, ha="center")

    # zone boundaries, clipped to the feasible lens
    for hv in (0.5, 0.9):
        axb.plot([hv, hv], [np.interp(hv, hh, lo_c), np.interp(hv, hh, up_c)],
                 color="#666666", lw=0.7, zorder=4)
    for (x0, x1), avals in (((0.0, 0.5), (42.5, 47.5)),
                            ((0.5, 0.9), (40.0, 50.0)),
                            ((0.9, 1.0), (40.0, 55.0))):
        for av in avals:
            xs = np.linspace(x0, x1, 60)
            keep = ((av <= np.interp(xs, hh, up_c))
                    & (av >= np.interp(xs, hh, lo_c)))
            if keep.any():
                axb.plot(xs[keep], np.full(keep.sum(), av), color="#666666",
                         lw=0.7, zorder=4)
    zlab = {"Z9": (0.115, 30), "Z8": (0.115, 45.0), "Z7": (0.115, 62),
            "Z6": (0.565, 19), "Z5": (0.545, 45.0), "Z4": (0.585, 66),
            "Z2": (0.955, 52), "Z1": (0.945, 62)}
    for lab, (zx, zy) in zlab.items():
        axb.text(zx, zy, lab, fontsize=6.0, color="#8a8a8a", ha="center",
                 va="center", zorder=5, fontweight="bold")
    axb.text(0.985, 2.5, "Z3 outside feasible region", fontsize=6.0,
             color="#8a8a8a", ha="right", va="bottom", style="italic",
             zorder=5)

    npts = 900
    for i, name in enumerate(CLASSES):
        frac = (lc == i).mean()
        k = max(60, int(npts * frac * 2.2))
        hc, ac, hs, as_ = HA_CENTRE[name]
        hx = np.clip(rng.normal(hc, hs, k), 0.02, 0.985)
        ay = np.clip(rng.normal(ac, as_, k), 1.0, 88.0)
        ay = np.clip(ay, np.interp(hx, hh, lo_c) + 1.0,
                     np.interp(hx, hh, up_c) - 1.0)
        axb.scatter(hx, ay, s=4.0, color=CCOL[i], alpha=0.42, linewidth=0,
                    zorder=3, label=name)
        axb.plot(hc, ac, marker="o", ms=5.0, mfc=CCOL[i], mec="white",
                 mew=0.9, zorder=6)
    axb.set_xlim(0, 1)
    axb.set_ylim(0, 90)
    axb.set_xlabel("Entropy $H$ (\u2013)", labelpad=1.5)
    axb.set_ylabel("Mean scattering angle $\\bar{\\alpha}$ (deg)")
    axb.set_title("Cloude\u2013Pottier H\u2013$\\bar{\\alpha}$ plane",
                  fontsize=8.4)
    axb.set_yticks(np.arange(0, 91, 15))
    style_axes(axb)
    lgb = axb.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5,
                     fontsize=6.2, frameon=False, handletextpad=0.25,
                     columnspacing=0.8, borderaxespad=0.0,
                     markerscale=2.4)
    lgb.set_clip_on(False)
    for h in lgb.legend_handles:
        h.set_alpha(0.95)
    add_panel_label(axb, "(b)", x=-0.14, y=1.02, size=10)

    # ---- (c) backscatter by polarisation ----------------------------------
    axc = fig.add_subplot(gs[1, 0])
    axc.set_facecolor("#f7f7f4")
    pols = ["HH", "VV", "HV"]
    pcol = ["#3b5b92", "#7aa0c4", "#c0603f"]
    xpos = np.arange(len(CLASSES))
    w = 0.26
    floor = -36.0
    for j, (p, c) in enumerate(zip(pols, pcol)):
        vals = np.array([SIGMA0[n][j] for n in CLASSES])
        errs = [SIGMA0_SD[n][j] for n in CLASSES]
        axc.bar(xpos + (j - 1) * w, vals - floor, bottom=floor, width=w,
                color=c, edgecolor="white", linewidth=0.4, label=p, zorder=3)
        axc.errorbar(xpos + (j - 1) * w, vals, yerr=errs, fmt="none",
                     ecolor="#444444", elinewidth=0.6, capsize=1.6, zorder=4)
    axc.set_xticks(xpos)
    axc.set_xticklabels(CLASSES, fontsize=6.6, rotation=18, ha="right")
    axc.set_ylabel("$\\sigma^0$ (dB)")
    axc.set_ylim(floor, 1.5)
    axc.set_yticks(np.arange(-35, 1, 5))
    axc.set_title("Backscatter by polarisation", fontsize=8.4)
    style_axes(axc)
    axc.grid(axis="y", lw=0.5, alpha=0.55)
    axc.legend(loc="upper left", fontsize=6.4, ncol=3, frameon=False,
               handlelength=1.1, columnspacing=0.8, borderaxespad=0.25)
    add_panel_label(axc, "(c)", x=-0.16, y=1.03, size=10)

    # ---- (d) power partition ----------------------------------------------
    axd = fig.add_subplot(gs[1, 1])
    axd.set_facecolor("#f7f7f4")
    mech = ["Surface", "Double bounce", "Volume"]
    mcol = ["#3b5b92", "#b23a48", "#2e7d5b"]
    bottom = np.zeros(len(CLASSES))
    for j, (mname, c) in enumerate(zip(mech, mcol)):
        vals = np.array([PARTITION[n][j] for n in CLASSES]) * 100
        axd.barh(xpos, vals, left=bottom, height=0.6, color=c,
                 edgecolor="white", linewidth=0.5, label=mname, zorder=3)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 11:
                axd.text(b + v / 2, xi, f"{v:.0f}", ha="center", va="center",
                         fontsize=6.0, color="white", fontweight="bold",
                         zorder=4)
        bottom += vals
    axd.set_yticks(xpos)
    axd.set_yticklabels(CLASSES, fontsize=6.6)
    axd.set_xlim(0, 100)
    axd.set_xlabel("Share of total scattered power (%)", labelpad=1.5)
    axd.set_title("Three-component decomposition", fontsize=8.4)
    axd.invert_yaxis()
    style_axes(axd)
    axd.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3,
               fontsize=6.4, frameon=False, handlelength=1.1,
               columnspacing=1.0, borderaxespad=0.0)
    add_panel_label(axd, "(d)", x=-0.16, y=1.03, size=10)

    fig.subplots_adjust(top=0.905, bottom=0.115, left=0.055, right=0.985)
    savefig(fig, "188_rs_sar_polarimetry.png")


if __name__ == "__main__":
    main()

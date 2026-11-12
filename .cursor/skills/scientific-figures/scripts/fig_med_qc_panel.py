"""Figure 383 - Weekly scanner QC: four metric maps, one trend, four verdicts.

Every number on this page comes out of the same synthetic phantom acquisition:
the four maps are the spatial distribution of each metric, the trend panel plots
each metric as a fraction of *its own* action limit so that four incompatible
units share one honest axis, and the verdict cards restate the measured value
against the limit.  The last point of every trend line is the value drawn in
the map above it.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from scipy.ndimage import gaussian_filter, uniform_filter

from pubstyle import (set_pub_style, savefig, get_cmap, style_axes, COLORS,
                      CATEGORICAL)
from figkit import (image_axes, safe_colorbar, corner_stamp, add_scale_bar,
                    panel_labels_inches, shared_norm)

N = 192                       # phantom image side, px
FOV_MM = 200.0                # square field of view
MM_PX = FOV_MM / N
SEED = 17
N_SESSIONS = 26               # weekly QC sessions on record

# Action limits (ACR-style): two metrics have a floor, two have a ceiling.
LIMITS = {
    "SNR": dict(value=90.0, kind="min", unit="(-)"),
    "Uniformity": dict(value=87.5, kind="min", unit="(%)"),
    "Geometric error": dict(value=2.0, kind="max", unit="(mm)"),
    "Ghosting": dict(value=2.5, kind="max", unit="(%)"),
}
PASS_C, FAIL_C = "#2e7d5b", "#b23a48"


# ---------------------------------------------------------------------------
# Synthetic phantom acquisition
# ---------------------------------------------------------------------------
def _sm(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _grid(n):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    return (xx - (n - 1) / 2) / (n / 2), (yy - (n - 1) / 2) / (n / 2)


def acquire(n=N, seed=SEED):
    """One uniform-cylinder acquisition: image, true signal, masks."""
    rng = np.random.default_rng(seed)
    x, y = _grid(n)
    r = np.hypot(x, y)
    disc = r <= 0.78

    # Transmit inhomogeneity plus a receive roll-off toward the edge.
    shade = 1.0 + 0.052 * _sm(rng, n, 24) - 0.038 * (r / 0.78) ** 2
    sig = gaussian_filter(np.where(disc, 1000.0 * shade, 0.0), 1.1)

    # Parallel-imaging g-factor: the noise floor is worst in the middle.
    sd = 7.2 * (1.0 + 0.55 * np.exp(-(r / 0.40) ** 2))
    img = sig + sd * rng.standard_normal((n, n))

    # N/2 ghost along the phase-encode (vertical) direction.
    ghost = 0.020 * np.roll(sig, n // 2, axis=0)
    return img + ghost, sig, disc, ghost


def snr_map(img, disc, box=9):
    """Local mean / local standard deviation, the two-region SNR estimate."""
    m = uniform_filter(img, box)
    v = np.maximum(uniform_filter(img * img, box) - m * m, 1e-9)
    return np.ma.masked_where(~disc, m / np.sqrt(v))


def uniformity_map(img, roi, sigma=4.0):
    """Per-cent departure of the smoothed signal from the ROI mean.

    The smoothing is normalised by the ROI itself; a plain ``gaussian_filter``
    would drag the zero background inward and manufacture a 40 % roll-off ring
    that has nothing to do with the scanner.
    """
    w = roi.astype(float)
    s = gaussian_filter(img * w, sigma) / np.maximum(gaussian_filter(w, sigma),
                                                     1e-6)
    ref = float(s[roi].mean())
    return np.ma.masked_where(~roi, 100.0 * (s - ref) / ref), ref


def piu(u_map):
    """Integral uniformity from the 1st / 99th percentile of the same map."""
    v = np.asarray(u_map[~u_map.mask]).ravel()
    lo, hi = np.percentile(v, [1.0, 99.0])
    smin, smax = 1.0 + lo / 100.0, 1.0 + hi / 100.0
    return 100.0 * (1.0 - (smax - smin) / (smax + smin))


def distortion(n=N):
    """Gradient-nonlinearity displacement field in mm, and its magnitude."""
    x, y = _grid(n)
    r2 = x ** 2 + y ** 2
    ux = 2.55 * x * r2 + 0.36 * x * y
    uy = 2.55 * y * r2 - 0.28 * x * y + 0.22 * y * np.abs(y)
    return ux, uy, np.hypot(ux, uy)


def ghost_pct(img, sig, ghost, disc, r):
    """Background signal as a per-cent of the phantom mean, plus ghost ROIs.

    The background stops at ``r > 0.84`` rather than at the phantom edge: the
    display smoothing bleeds phantom signal a few pixels outward, and that
    bleed - not the ghost - would otherwise dominate the statistic.
    """
    ref = float(sig[disc].mean())
    bg = r > 0.90
    pct = np.ma.masked_where(~bg, 100.0 * gaussian_filter(img, 2.5) / ref)
    groi = (ghost > 0.35 * ghost.max()) & bg
    return pct, bg, groi, 100.0 * float(img[groi].mean()) / ref


def trend(rng, endpoint, n=N_SESSIONS, drift=0.030, wobble=0.022):
    """Weekly history whose final session is exactly the measured value."""
    w = 1.0 + np.cumsum(drift * rng.standard_normal(n)) / np.sqrt(n)
    w = w + wobble * rng.standard_normal(n)
    return endpoint * w / w[-1]


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED + 1)

    img, sig, disc, ghost = acquire()
    x, y = _grid(N)
    r = np.hypot(x, y)
    inner = r <= 0.70                # ACR-style large ROI inside the phantom

    snr = snr_map(img, disc)
    unif, ref = uniformity_map(img, inner)
    ux, uy, dmag = distortion()
    gpct, bg, groi, ghost_val = ghost_pct(img, sig, ghost, disc, r)

    measured = {
        "SNR": float(np.asarray(snr)[inner].mean()),
        "Uniformity": float(piu(unif)),
        "Geometric error": float(dmag[inner].max()),
        "Ghosting": ghost_val,
    }

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.58],
                          left=0.042, right=0.986, top=0.878, bottom=0.100,
                          wspace=0.235, hspace=0.50)
    tiles = [fig.add_subplot(gs[0, j]) for j in range(4)]
    ax_tr = fig.add_subplot(gs[1, 0:3])
    ax_cd = fig.add_subplot(gs[1, 3])

    for ax in tiles:
        image_axes(ax, color=COLORS["hairline"], lw=0.6)

    # ---- (a) SNR ----------------------------------------------------------
    ax = tiles[0]
    im = ax.imshow(snr, cmap=get_cmap("cb_sequential_blue"),
                   vmin=np.percentile(np.asarray(snr[~snr.mask]), 1),
                   vmax=np.percentile(np.asarray(snr[~snr.mask]), 99),
                   interpolation="bilinear")
    ax.add_patch(Circle(((N - 1) / 2, (N - 1) / 2), 0.70 * N / 2,
                        facecolor="none", edgecolor=COLORS["rule"], lw=0.8,
                        ls=(0, (3, 2)), zorder=6))
    ax.set_title("Signal-to-noise ratio", fontsize=8.6, fontweight="bold",
                 pad=3.5)
    safe_colorbar(fig, im, ax, side="bottom", width=0.085, pad=0.050,
                  shrink=0.88, label="local SNR (-)")
    corner_stamp(ax, f"ROI mean {measured['SNR']:.0f}  |  9 x 9 px window",
                 loc="lower left", fontsize=6.2)
    add_scale_bar(ax, 50.0 / MM_PX, "50 mm", loc=(0.035, 0.115), height=0.020,
                  fontsize=6.0)

    # ---- (b) uniformity ---------------------------------------------------
    ax = tiles[1]
    nrm = shared_norm([np.asarray(unif[~unif.mask])], robust=(0.5, 99.5),
                      symmetric=True)
    im = ax.imshow(unif, cmap=get_cmap("cb_diverging"), norm=nrm,
                   interpolation="bilinear")
    ax.contour(np.where(inner, unif.filled(0.0), np.nan),
               levels=[-2.0, 0.0, 2.0], colors=[COLORS["rule"]],
               linewidths=0.45, alpha=0.7)
    ax.add_patch(Circle(((N - 1) / 2, (N - 1) / 2), 0.78 * N / 2,
                        facecolor="none", edgecolor=COLORS["grid"], lw=0.7,
                        zorder=6))
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(N - 0.5, -0.5)
    ax.set_title("Signal uniformity", fontsize=8.6, fontweight="bold", pad=3.5)
    safe_colorbar(fig, im, ax, side="bottom", width=0.085, pad=0.050,
                  shrink=0.88, label="departure from mean (%)")
    corner_stamp(ax, f"PIU {measured['Uniformity']:.1f} %  |  contours at "
                     "0, +/-2 %", loc="lower left", fontsize=6.2)

    # ---- (c) geometric accuracy -------------------------------------------
    ax = tiles[2]
    dm = np.ma.masked_where(~disc, dmag)
    im = ax.imshow(dm, cmap=get_cmap("cb_sequential_warm"), vmin=0.0,
                   vmax=LIMITS["Geometric error"]["value"],
                   interpolation="bilinear")
    step = 16
    sl = (slice(step // 2, None, step), slice(step // 2, None, step))
    keep = disc[sl]
    ax.quiver(np.mgrid[0:N, 0:N][1][sl][keep], np.mgrid[0:N, 0:N][0][sl][keep],
              (ux[sl] / MM_PX)[keep], (-uy[sl] / MM_PX)[keep],
              color=COLORS["rule"], angles="xy", scale_units="xy", scale=0.14,
              width=0.006, headwidth=3.4, headlength=4.0, zorder=5)
    ax.add_patch(Circle(((N - 1) / 2, (N - 1) / 2), 0.70 * N / 2,
                        facecolor="none", edgecolor=CATEGORICAL[0], lw=0.8,
                        ls=(0, (3, 2)), zorder=6))
    ax.set_title("Geometric accuracy", fontsize=8.6, fontweight="bold",
                 pad=3.5)
    safe_colorbar(fig, im, ax, side="bottom", width=0.085, pad=0.050,
                  shrink=0.88, label="rod displacement (mm)")
    corner_stamp(ax, f"peak {measured['Geometric error']:.2f} mm inside the "
                     "dashed ROI", loc="lower left", fontsize=6.2)

    # ---- (d) ghosting / artefact ------------------------------------------
    ax = tiles[3]
    gv = np.asarray(gpct[~gpct.mask])
    gcmap = get_cmap("cb_sequential_purple")
    gcmap.set_bad(COLORS["solid"])
    im = ax.imshow(gpct, cmap=gcmap, vmin=0.0,
                   vmax=float(np.percentile(gv, 99.0)),
                   interpolation="bilinear")
    ax.text((N - 1) / 2, (N - 1) / 2, "phantom\nmasked out", ha="center",
            va="center", fontsize=6.2, color="#d5d5d5", zorder=5,
            linespacing=1.3)
    rows = np.where(groi.any(axis=1))[0]
    for seg in np.split(rows, np.where(np.diff(rows) > 1)[0] + 1):
        cols = np.where(groi[seg].any(axis=0))[0]
        ax.add_patch(Rectangle((cols[0] - 0.5, seg[0] - 0.5),
                               cols[-1] - cols[0] + 1, seg[-1] - seg[0] + 1,
                               facecolor="none", edgecolor=CATEGORICAL[0],
                               lw=0.9, zorder=6))
    ax.set_title("N/2 ghost detection", fontsize=8.6, fontweight="bold",
                 pad=3.5)
    safe_colorbar(fig, im, ax, side="bottom", width=0.085, pad=0.050,
                  shrink=0.88, label="background signal (%)")
    corner_stamp(ax, f"ghost ROI mean {measured['Ghosting']:.2f} % of "
                     "phantom", loc="lower left", fontsize=6.2)

    # ---- (e) trend, in units of each metric's own action limit -------------
    weeks = np.arange(1, N_SESSIONS + 1)
    series, util = {}, {}
    for k, spec in LIMITS.items():
        s = trend(rng, measured[k])
        series[k] = s
        util[k] = (spec["value"] / s if spec["kind"] == "min"
                   else s / spec["value"])

    for i, (k, u) in enumerate(util.items()):
        ax_tr.plot(weeks, u, color=CATEGORICAL[i], lw=1.3, marker="o", ms=2.4,
                   mew=0.0, zorder=3,
                   label=f"{k}  {measured[k]:.2f} {LIMITS[k]['unit']}")
    top = max(1.55, max(u.max() for u in util.values()) * 1.35)
    ax_tr.axhline(1.0, color=FAIL_C, lw=1.0, ls="--", zorder=2)
    ax_tr.axhspan(1.0, top, facecolor=FAIL_C, alpha=0.06, zorder=0)
    ax_tr.text(N_SESSIONS + 0.4, 1.015, "action limit", fontsize=6.4,
               color=FAIL_C, va="bottom", ha="right")
    ax_tr.set_xlim(0.4, N_SESSIONS + 0.6)
    ax_tr.set_ylim(0.0, top)
    ax_tr.set_xlabel("weekly QC session (-)", fontsize=8.0)
    ax_tr.set_ylabel("fraction of action limit (-)", fontsize=8.0)
    ax_tr.tick_params(labelsize=6.8)
    ax_tr.grid(axis="y", linewidth=0.5, alpha=0.7)
    ax_tr.set_title(f"Tolerance utilisation over {N_SESSIONS} sessions",
                    fontsize=8.6, fontweight="bold", pad=3.5)
    ax_tr.legend(loc="upper left", ncol=4, frameon=False, fontsize=6.5,
                 handlelength=1.4, columnspacing=1.0, borderaxespad=0.25)
    style_axes(ax_tr)

    # ---- (f) verdict cards -------------------------------------------------
    ax_cd.set_xlim(0, 1)
    ax_cd.set_ylim(0, 1)
    ax_cd.axis("off")
    ax_cd.set_title("Session verdict", fontsize=8.6, fontweight="bold",
                    pad=3.5)
    n_fail = 0
    for i, (k, spec) in enumerate(LIMITS.items()):
        ok = (measured[k] >= spec["value"] if spec["kind"] == "min"
              else measured[k] <= spec["value"])
        n_fail += (not ok)
        col = PASS_C if ok else FAIL_C
        y0 = 0.795 - 0.235 * i
        ax_cd.add_patch(FancyBboxPatch(
            (0.02, y0), 0.96, 0.195,
            boxstyle="round,pad=0,rounding_size=0.02", facecolor="#f5f5f5",
            edgecolor=COLORS["grid"], linewidth=0.6, mutation_aspect=0.4))
        ax_cd.add_patch(Rectangle((0.02, y0), 0.030, 0.195, facecolor=col,
                                  edgecolor="none"))
        ax_cd.text(0.085, y0 + 0.135, k, fontsize=7.0, color=COLORS["ink"],
                   ha="left", va="center")
        ax_cd.text(0.085, y0 + 0.052,
                   f"{measured[k]:.2f} {LIMITS[k]['unit']}   "
                   f"({'>=' if spec['kind'] == 'min' else '<='} "
                   f"{spec['value']:.1f})",
                   fontsize=6.4, color=COLORS["muted"], ha="left",
                   va="center")
        ax_cd.text(0.955, y0 + 0.096, "PASS" if ok else "FAIL", fontsize=7.4,
                   fontweight="bold", color=col, ha="right", va="center")
    ax_cd.text(0.5, 0.005,
               "all four metrics within tolerance" if n_fail == 0 else
               f"{n_fail} metric(s) out of tolerance", fontsize=6.6,
               color=PASS_C if n_fail == 0 else FAIL_C, ha="center",
               va="bottom")

    tight = max(util, key=lambda k: util[k][-1])
    panel_labels_inches(fig, tiles + [ax_tr, ax_cd], dx=-0.20, dy=0.09)

    savefig(fig, "383_med_qc_panel.png")


if __name__ == "__main__":
    main()

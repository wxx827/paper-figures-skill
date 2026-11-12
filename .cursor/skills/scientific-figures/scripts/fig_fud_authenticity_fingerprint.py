"""Authenticity screening: three origins, one adulterant, and the wavenumbers
that actually separate them.

Ninety-six mid-infrared spectra are synthesised from a shared set of bands
whose heights differ by origin, plus a sunflower-oil adulterant profile mixed
in at four levels.  Everything downstream is fitted on that matrix rather than
drawn: the scores panel is a real ``sklearn`` PCA of the standardised spectra
with per-origin 95 % ellipses taken from each group's own covariance
eigenvectors, and the importance panel is the textbook VIP statistic of a
two-component PLS-DA of authentic against adulterated, so the bands it marks
are the bands the discriminant actually leaned on.  The shaded regions in all
three panels are the same two wavenumber windows, which is the point of the
figure: the highlighted region on the stacked fingerprint is where the VIP
curve crosses one, not a region chosen by eye.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, halo_text, panel_labels_inches

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
RED = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

WN_LO, WN_HI, N_VAR = 800.0, 1800.0, 260
N_PER_ORIGIN = 26
N_FAKE = 18
ADULTERATION = np.array([0.05, 0.10, 0.20, 0.35])

#           centre  width  base height, per-origin multiplier
BANDS = [(1745.0, 11.0, 0.95, (1.00, 1.02, 0.98)),      # ester C=O
         (1655.0, 15.0, 0.30, (1.00, 1.24, 0.86)),      # cis C=C
         (1465.0, 12.0, 0.42, (1.00, 0.97, 1.03)),      # CH2 bend
         (1418.0, 8.0, 0.14, (1.00, 1.45, 0.72)),       # =CH bend
         (1377.0, 8.0, 0.18, (1.00, 1.05, 0.95)),       # CH3 bend
         (1238.0, 16.0, 0.34, (1.00, 0.90, 1.12)),      # C-O stretch
         (1163.0, 14.0, 0.72, (1.00, 1.04, 0.96)),      # C-O-C
         (1098.0, 15.0, 0.40, (1.00, 0.94, 1.08)),
         (1032.0, 13.0, 0.16, (1.00, 1.10, 0.90)),
         (966.0, 9.0, 0.09, (1.00, 0.85, 1.30)),        # trans C=C
         (870.0, 12.0, 0.11, (1.00, 1.00, 1.00))]
# Sunflower oil differs mostly in the linoleic bands
FAKE_MULT = {1655.0: 1.55, 1418.0: 1.90, 966.0: 2.30, 1238.0: 0.78,
             1098.0: 1.34}
ORIGINS = ["origin A (PDO hill)", "origin B (coastal)", "origin C (inland)"]
ORIGIN_COLOUR = [BLUE, GREEN, GOLD]

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 4.62, 4.48)
AUX_T = (7.44, 3.61, 2.68, 1.89)
AUX_B = (7.44, 1.02, 2.68, 1.82)


def spectrum(wn, origin, fake_frac, rng):
    """One absorbance spectrum with a sloping baseline and band noise."""
    y = 0.045 + 0.030 * (WN_HI - wn) / (WN_HI - WN_LO)
    for centre, width, height, mult in BANDS:
        h = height * mult[origin] * rng.normal(1.0, 0.035)
        h *= (1.0 - fake_frac) + fake_frac * FAKE_MULT.get(centre, 1.0)
        y = y + h * np.exp(-0.5 * ((wn - centre) / width) ** 2)
    return y + rng.normal(0.0, 0.0035, wn.size)


def build(rng):
    wn = np.linspace(WN_LO, WN_HI, N_VAR)
    rows, group, frac = [], [], []
    for o in range(3):
        for _ in range(N_PER_ORIGIN):
            rows.append(spectrum(wn, o, 0.0, rng))
            group.append(o)
            frac.append(0.0)
    for j in range(N_FAKE):
        o = j % 3
        f = float(ADULTERATION[j % ADULTERATION.size])
        rows.append(spectrum(wn, o, f, rng))
        group.append(3)
        frac.append(f)
    return wn, np.array(rows), np.array(group), np.array(frac)


def vip_scores(pls, x):
    """Variable importance in projection of a fitted PLS model."""
    t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    ssy = (t ** 2).sum(axis=0) * (q ** 2).sum(axis=0)
    wn2 = (w / np.linalg.norm(w, axis=0)) ** 2
    return np.sqrt(x.shape[1] * (wn2 @ ssy) / ssy.sum())


def windows(wn, vip, floor=1.0, min_width=14.0):
    """Contiguous wavenumber ranges where VIP stays above ``floor``."""
    hot = vip > floor
    out, start = [], None
    for i, flag in enumerate(hot):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if wn[i - 1] - wn[start] >= min_width:
                out.append((wn[start], wn[i - 1]))
            start = None
    if start is not None and wn[-1] - wn[start] >= min_width:
        out.append((wn[start], wn[-1]))
    return out


def main_panel(ax, wn, x, group, frac, bands):
    # The abscissa is inverted, so WN_HI is the *left* edge: a label anchored
    # there needs ha="left" or it grows out of the panel.
    tag = dict(boxstyle="round,pad=0.18", facecolor="white",
               edgecolor="none", alpha=0.88)
    offset = 0.0
    step = 0.42
    for o in range(3):
        idx = np.flatnonzero(group == o)[:3]
        for j, k in enumerate(idx):
            ax.plot(wn, x[k] + offset, color=ORIGIN_COLOUR[o], lw=1.0,
                    alpha=1.0 if j == 0 else 0.55, zorder=5)
        ax.text(WN_LO + 10.0, offset + 0.30, ORIGINS[o], ha="right",
                va="bottom", fontsize=6.6, color=ORIGIN_COLOUR[o],
                fontweight="bold", zorder=7, bbox=tag)
        offset += step * 3.0
    fake = np.flatnonzero(group == 3)
    order = [int(fake[np.flatnonzero(frac[fake] == lvl)[0]])
             for lvl in ADULTERATION[::-1][:3]]
    for j, k in enumerate(order):
        ax.plot(wn, x[k] + offset, color=RED, lw=1.0,
                alpha=1.0 if j == 0 else 0.55, zorder=6)
    ax.text(WN_LO + 10.0, offset + 0.30, "adulterated, {:s} %".format(
        " / ".join("{:.0f}".format(100.0 * frac[k]) for k in order)),
        ha="right", va="bottom", fontsize=6.6, color=RED, fontweight="bold",
        zorder=7, bbox=tag)

    top = offset + 1.28
    for lo, hi in bands:
        ax.axvspan(lo, hi, color=lighten(RED, 0.85), lw=0.0, zorder=1)
        ax.text(0.5 * (lo + hi), top, "{:.0f}-{:.0f}".format(lo, hi),
                ha="center", va="bottom", fontsize=6, color=RED, zorder=8)
    ax.text(WN_HI - 10.0, top + 0.12,
            "shaded: the two strongest VIP > 1 windows of panel (c)",
            ha="left",
            va="bottom", fontsize=6.4, color=RED, zorder=8)

    ax.set_xlim(WN_HI, WN_LO)
    ax.set_ylim(-0.10, top + 0.36)
    ax.set_yticks([])
    ax.set_xlabel("wavenumber (cm$^{-1}$)")
    ax.set_ylabel("absorbance (a.u.), offset-stacked by group")
    ax.grid(axis="x", lw=0.5, alpha=0.45)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Mid-infrared fingerprints, three declared origins over one "
                 "adulterant", fontsize=9.0, fontweight="bold", color=INK,
                 pad=6.0)


def score_panel(ax, scores, group, frac, var):
    for o in range(3):
        m = group == o
        ax.plot(scores[m, 0], scores[m, 1], marker="o", ms=3.2,
                mfc=lighten(ORIGIN_COLOUR[o], 0.35), mec="white", mew=0.4,
                ls="none", zorder=4)
        pts = scores[m][:, :2]
        cen = pts.mean(axis=0)
        val, vec = np.linalg.eigh(np.cov(pts, rowvar=False))
        ang = np.degrees(np.arctan2(vec[1, -1], vec[0, -1]))
        ax.add_patch(Ellipse(cen, *(2.0 * np.sqrt(5.991 * val[::-1])),
                             angle=ang, facecolor="none",
                             edgecolor=ORIGIN_COLOUR[o], lw=1.0,
                             ls=(0, (4.0, 1.8)), zorder=5))
        halo_text(ax, cen[0], cen[1], "ABC"[o], ha="center", va="center",
                  fontsize=7.6, color=ORIGIN_COLOUR[o], halo="white",
                  halo_lw=2.0, fontweight="bold", zorder=8)
    m = group == 3
    sz = 10.0 + 42.0 * frac[m] / frac[m].max()
    ax.scatter(scores[m, 0], scores[m, 1], s=sz, marker="D", facecolor=RED,
               edgecolor="white", linewidth=0.5, zorder=6)
    ax.set_xlabel("PC1 ({:.1f} % of variance)".format(100.0 * var[0]),
                  fontsize=7.4)
    ax.set_ylabel("PC2 ({:.1f} %)".format(100.0 * var[1]), fontsize=7.4)
    ax.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [Line2D([], [], color=RED, lw=0.0, marker="D", ms=4.0,
                      mfc=RED, mec="white",
                      label="adulterated, marker size $\\propto$ level"),
               Line2D([], [], color=SUB, lw=1.0, ls=(0, (4.0, 1.8)),
                      label="95 % ellipse per origin")]
    ax.legend(handles=handles, loc="lower left", fontsize=6, frameon=False,
              handlelength=1.4, borderaxespad=0.2, labelspacing=0.22)
    ax.set_title("Adulterated lots leave their own ellipse", fontsize=8.0,
                 fontweight="bold", color=INK)


def vip_panel(ax, wn, vip, bands):
    for lo, hi in bands:
        ax.axvspan(lo, hi, color=lighten(RED, 0.85), lw=0.0, zorder=1)
    ax.fill_between(wn, 0.0, vip, color=lighten(BLUE, 0.80), lw=0.0, zorder=2)
    ax.plot(wn, vip, color=BLUE, lw=1.1, zorder=4)
    ax.axhline(1.0, color=RED, lw=1.0, ls=(0, (3.5, 2.0)), zorder=5)
    ax.text(WN_HI - 12.0, 1.06, "VIP = 1", ha="left", va="bottom",
            fontsize=6.0, color=RED, zorder=6,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                      edgecolor="none", alpha=0.90))
    j = int(np.argmax(vip))
    ax.annotate("{:.0f} cm$^{{-1}}$, VIP {:.2f}".format(wn[j], vip[j]),
                xy=(wn[j], vip[j]), xytext=(0.0, 5.0),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=6.2, color=INK, zorder=7)
    ax.set_xlim(WN_HI, WN_LO)
    ax.set_ylim(0.0, float(vip.max()) * 1.26)
    ax.set_xlabel("wavenumber (cm$^{-1}$)", fontsize=7.4)
    ax.set_ylabel("PLS-DA VIP (-)", fontsize=7.4)
    ax.grid(axis="y", lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Which wavenumbers carry the call", fontsize=8.0,
                 fontweight="bold", color=INK)


def audit(fig, foot, axes):
    """Report any in-view tick or axis label the footnote would sit on."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for ax in axes:
        ab = ax.get_window_extent()
        for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            tb = t.get_window_extent()
            if tb.width <= 0.0 or not t.get_text():
                continue
            if tb.x1 < ab.x0 - 2.0 or tb.x0 > ab.x1 + 2.0:
                continue
            if fb.overlaps(tb):
                print("OVERLAP with", repr(t.get_text()))


def main():
    set_pub_style()
    rng = np.random.default_rng(917)
    wn, x, group, frac = build(rng)

    # Mean-centred, not autoscaled: standardising 260 wavenumbers gives the
    # flat baseline regions the same weight as the bands, and the VIP curve
    # then reports pure noise at unit height.
    xc = x - x.mean(axis=0)
    pca = PCA(n_components=4).fit(xc)
    scores = pca.transform(xc)

    y = (group == 3).astype(float)
    pls = PLSRegression(n_components=2, scale=False).fit(xc, y)
    vip = vip_scores(pls, xc)
    bands = sorted(windows(wn, vip),
                   key=lambda b: -np.max(vip[(wn >= b[0]) & (wn <= b[1])]))[:2]
    bands.sort()

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_sc = axes_inches(fig, AUX_T)
    ax_vip = axes_inches(fig, AUX_B)

    main_panel(ax_main, wn, x, group, frac, bands)
    score_panel(ax_sc, scores, group, frac, pca.explained_variance_ratio_)
    vip_panel(ax_vip, wn, vip, bands)

    j = int(np.argmax(vip))
    panel_labels_inches(fig, [ax_main, ax_sc, ax_vip], dx=-0.62, dy=0.14)
    savefig(fig, "917_fud_authenticity_fingerprint.png")


if __name__ == "__main__":
    main()

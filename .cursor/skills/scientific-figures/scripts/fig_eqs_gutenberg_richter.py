"""Frequency-magnitude statistics of two earthquake catalogues.

Nothing on this page is a drawn straight line.  Magnitudes are drawn from a
truncated Gutenberg-Richter exponential and then thinned by a magnitude
dependent detection probability, which is what bends the incremental curve
over below completeness.  The completeness magnitude is read back off that
bend with the maximum-curvature rule, the b value is the Aki-Utsu maximum
likelihood estimate of the surviving sample with the Shi and Bolt standard
error, and the quoted a value is whatever makes the fitted line pass through
the observed count at Mc.  The upper side panel repeats the estimate for every
possible cutoff so the reader can see the plateau the rule is aiming at; the
lower panel bootstraps the same estimator to show the two regions separate by
more than their scatter.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import erf

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
DEEP = "#22406e"
PALE = "#a8bcd8"
RED = "#b23a48"
ROSE = "#d79aa2"
GOLD = "#c9a227"
GREY = "#8a8a8a"

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.02, 4.66, 4.46)
AUX_T = (7.42, 3.61, 2.70, 1.87)
AUX_B = (7.42, 1.02, 2.70, 1.82)

DM = 0.1                     # magnitude bin width, Mw
M_MIN, M_MAX = 1.8, 7.8      # generating range, Mw
LOG10E = np.log10(np.e)

REGIONS = [
    dict(key="A", name="Arc trench segment", n=16000, b=0.87,
         mu_det=3.00, sd_det=0.18, color=DEEP, fill=PALE, marker="o"),
    dict(key="B", name="Inland crustal belt", n=11000, b=1.18,
         mu_det=2.50, sd_det=0.14, color=RED, fill=ROSE, marker="s"),
]


def draw_catalogue(rng, n, b, mu_det, sd_det):
    """Truncated GR magnitudes, reported to 0.1 Mw and thinned by detection.

    The rounding is not cosmetic.  A real bulletin quotes magnitude on a 0.1
    grid, and the ``Mc - dM/2`` term in the Aki-Utsu estimator exists exactly
    to undo that discretisation; feeding it unrounded magnitudes would bias
    every b value on the page low by about ten per cent.
    """
    beta = b * np.log(10.0)
    u = rng.uniform(0.0, 1.0, n)
    span = 1.0 - np.exp(-beta * (M_MAX - M_MIN))
    m = M_MIN - np.log1p(-u * span) / beta
    m = np.round(m / DM) * DM
    p = 0.5 * (1.0 + erf((m - mu_det) / (np.sqrt(2.0) * sd_det)))
    return np.sort(m[rng.uniform(0.0, 1.0, n) < p])


def binned(mags):
    """Bin centres, incremental counts and cumulative counts N(>= M)."""
    edges = np.arange(M_MIN - DM / 2.0, M_MAX + DM, DM)
    inc, _ = np.histogram(mags, bins=edges)
    centres = edges[:-1] + DM / 2.0
    cum = np.cumsum(inc[::-1])[::-1]
    return centres, inc, cum


def mc_maximum_curvature(centres, inc):
    """Bin of peak incremental count plus the usual 0.2 Mw correction."""
    return float(np.round((centres[int(np.argmax(inc))] + 0.2) / DM) * DM)


def aki_b(mags, mc):
    """Maximum-likelihood b and its Shi and Bolt standard error."""
    s = mags[mags >= mc - 1e-9]
    n = s.size
    mean = float(s.mean())
    b = LOG10E / max(mean - (mc - DM / 2.0), 1e-9)
    var = float(((s - mean) ** 2).sum()) / (n * (n - 1))
    return b, n, 2.30 * b * b * np.sqrt(var)


def b_vs_cutoff(mags, cuts, n_min=60):
    """b and its error for every trial cutoff that keeps enough events."""
    out_b, out_s, out_c = [], [], []
    for c in cuts:
        if (mags >= c).sum() < n_min:
            continue
        b, _, s = aki_b(mags, c)
        out_b.append(b)
        out_s.append(s)
        out_c.append(c)
    return np.array(out_c), np.array(out_b), np.array(out_s)


def mc_stability(mags, mc0, window=8):
    """Smallest cutoff whose b already equals the mean of the next eight bins.

    The maximum-curvature bin is a starting guess only; it habitually sits
    inside the roll-off and biases b low, so the cutoff is walked upward until
    the estimate stops drifting by more than its own standard error.
    """
    cuts = np.round(np.arange(mc0 - 0.2, mc0 + 2.01, DM), 2)
    c, b, s = b_vs_cutoff(mags, cuts, n_min=120)
    for i in range(max(c.size - window, 0)):
        if abs(b[i:i + window + 1].mean() - b[i]) <= s[i]:
            return float(c[i])
    return float(mc0)


def bootstrap_b(rng, mags, mc, n_rep=900):
    s = mags[mags >= mc - 1e-9]
    idx = rng.integers(0, s.size, size=(n_rep, s.size))
    mean = s[idx].mean(axis=1)
    return LOG10E / np.maximum(mean - (mc - DM / 2.0), 1e-9)


def main_panel(ax, cats):
    for c in cats:
        ax.plot(c["centres"], np.where(c["cum"] > 0, c["cum"], np.nan),
                ls="none", marker=c["marker"], ms=3.6, mfc=c["color"],
                mec="white", mew=0.4, zorder=6)
        ax.plot(c["centres"], np.where(c["inc"] > 0, c["inc"], np.nan),
                ls="none", marker=c["marker"], ms=3.2, mfc="none",
                mec=c["fill"], mew=0.9, zorder=5)
        mfit = np.linspace(c["mc"], M_MAX, 120)
        ax.plot(mfit, 10.0 ** (c["a"] - c["b"] * mfit), color=c["color"],
                lw=1.5, zorder=7)
        ax.axvline(c["mc"], color=c["color"], lw=0.9, ls=(0, (3.6, 2.4)),
                   zorder=4)
        ax.plot([c["mc"]], [c["n_above"]], marker="*", ms=8.5,
                mfc=GOLD, mec=c["color"], mew=0.7, zorder=8)

    top = max(c["cum"].max() for c in cats)
    ax.set_yscale("log")
    ax.set_xlim(M_MIN - 0.05, M_MAX + 0.15)
    ax.set_ylim(0.6, top * 4.2)
    ax.set_xlabel("moment magnitude Mw (dimensionless)")
    ax.set_ylabel("number of events (log scale)")
    ax.tick_params(labelsize=6.8)
    ax.grid(which="both", lw=0.45, alpha=0.32)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Frequency-magnitude distributions, 30 yr catalogues",
                 fontsize=9.0, fontweight="bold", color=INK, pad=5.0)

    lines = []
    for c in cats:
        lines.append("{:s}: b = {:.2f} +/- {:.2f}, Mc = {:.1f} Mw, "
                     "a = {:.2f}, N(>=Mc) = {:d}"
                     .format(c["name"], c["b"], c["berr"], c["mc"], c["a"],
                             int(c["n_above"])))
    lines.append("the roll-off left of each dashed line is the network,")
    lines.append("not the earth, so no fit is drawn through it")
    ax.text(0.975, 0.955, "\n".join(lines), transform=ax.transAxes,
            ha="right", va="top", fontsize=6.6, color=INK, linespacing=1.55,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor="#dddddd", linewidth=0.6, alpha=0.94),
            zorder=10)

    handles = [
        Line2D([], [], ls="none", marker="o", ms=3.6, mfc=DEEP, mec="white",
               label="cumulative N(>= Mw)"),
        Line2D([], [], ls="none", marker="o", ms=3.2, mfc="none", mec=PALE,
               mew=0.9, label="incremental N per 0.1 Mw bin"),
        Line2D([], [], color=SUB, lw=1.5, label="Aki-Utsu maximum-likelihood fit"),
        Line2D([], [], color=SUB, lw=0.9, ls=(0, (3.6, 2.4)),
               label="completeness magnitude Mc"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=6.3, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.7, labelspacing=0.28, borderaxespad=0.35)


def stability_panel(ax, cats):
    for c in cats:
        cut, b, s = c["cut"], c["bcut"], c["scut"]
        ax.fill_between(cut, b - s, b + s, facecolor=c["fill"], alpha=0.45,
                        edgecolor="none", zorder=3)
        ax.plot(cut, b, color=c["color"], lw=1.3, zorder=5,
                label=c["name"])
        ax.plot([c["mc"]], [c["b"]], marker="*", ms=8.0, mfc=GOLD,
                mec=c["color"], mew=0.7, zorder=7)
        ax.axhline(c["b"], color=c["color"], lw=0.7, ls=(0, (2.6, 2.2)),
                   alpha=0.7, zorder=4)
    ax.set_xlim(2.0, 5.0)
    ax.set_ylim(0.55, 1.65)
    ax.set_xlabel("trial cutoff magnitude (Mw)", fontsize=6.9)
    ax.set_ylabel("b value (dimensionless)", fontsize=6.9)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.45, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.968, 0.06, "star = adopted Mc", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.1, color=MUTED)
    ax.legend(loc="upper right", fontsize=6.1, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.4, labelspacing=0.22, borderaxespad=0.3)
    ax.set_title("b stabilises once the cutoff clears Mc", fontsize=8.0,
                 fontweight="bold", color=INK, pad=3.0)


def bootstrap_panel(ax, cats):
    edges = np.linspace(0.60, 1.45, 52)
    top = 0.0
    for c in cats:
        n, _, _ = ax.hist(c["boot"], bins=edges, color=c["fill"],
                          edgecolor=c["color"], linewidth=0.6, alpha=0.72,
                          zorder=4)
        ax.axvline(c["b"], color=c["color"], lw=1.2, zorder=6)
        top = max(top, float(n.max()))
    lo_a, hi_a = np.percentile(cats[0]["boot"], [2.5, 97.5])
    lo_b, hi_b = np.percentile(cats[1]["boot"], [2.5, 97.5])
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0.0, top * 1.95)
    ax.set_xlabel("bootstrap b value (dimensionless)", fontsize=6.9)
    ax.set_ylabel("resamples (count)", fontsize=6.9)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.45, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.03, 0.955,
            "900 resamples above each Mc\n"
            "arc 95 %: {:.2f} to {:.2f}\ncrust 95 %: {:.2f} to {:.2f}"
            .format(lo_a, hi_a, lo_b, hi_b),
            transform=ax.transAxes, ha="left", va="top", fontsize=6.2,
            color=SUB, linespacing=1.45, zorder=8)
    ax.set_title("The two b values do not overlap", fontsize=8.0,
                 fontweight="bold", color=INK, pad=3.0)


def audit(fig, foot, axes):
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
    rng = np.random.default_rng(20460813)

    cats = []
    for spec in REGIONS:
        mags = draw_catalogue(rng, spec["n"], spec["b"], spec["mu_det"],
                              spec["sd_det"])
        centres, inc, cum = binned(mags)
        mc0 = mc_maximum_curvature(centres, inc)
        mc = mc_stability(mags, mc0)
        b, n_above, berr = aki_b(mags, mc)
        a = np.log10(n_above) + b * mc
        cut, bcut, scut = b_vs_cutoff(mags, np.arange(2.0, 5.01, 0.1))
        boot = bootstrap_b(rng, mags, mc)
        cats.append(dict(spec, mags=mags, centres=centres, inc=inc, cum=cum,
                         mc0=mc0, mc=mc, b=b, berr=berr, a=a, n_above=n_above,
                         cut=cut, bcut=bcut, scut=scut, boot=boot))

    assert cats[0]["b"] < cats[1]["b"], "arc should be the shallower slope"
    assert all(2.4 < c["mc"] < 4.2 for c in cats), "Mc off the plotted range"
    assert all(c["n_above"] > 400 for c in cats), "too few events above Mc"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_stab = axes_inches(fig, AUX_T)
    ax_boot = axes_inches(fig, AUX_B)

    main_panel(ax_main, cats)
    stability_panel(ax_stab, cats)
    bootstrap_panel(ax_boot, cats)



    panel_labels_inches(fig, [ax_main, ax_stab, ax_boot], dx=-0.60, dy=0.12)
    savefig(fig, "1050_eqs_gutenberg_richter.png")


if __name__ == "__main__":
    main()

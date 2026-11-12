"""Carbon-oxygen crossplot used to screen carbonates for diagenetic overprint.

The main panel is the screening test itself: every sample plotted in
delta-13C / delta-18O space, coloured by stratigraphic unit, each unit wrapped
in its 95 % confidence ellipse.  A primary marine field is drawn in the
depleted-resistant corner, and the fitted covariation line running away from
it is the diagnostic that matters - strong C-O covariation is the classical
fingerprint of meteoric alteration.

The two smaller panels carry the statistics behind the screen: the delta
distributions unit by unit on one shared per-mil scale, and the covariation
coefficient with its 95 % interval against the usual rejection threshold.
Every coefficient, slope and interval is computed from the plotted points.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from scipy import stats

from pubstyle import set_pub_style, sig_stars, savefig
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
RED = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
SOLID = "#3a3a3a"

PERMIL = "\u2030"
R_THRESHOLD = 0.70          # conventional C-O covariation rejection level
PRIMARY = (-4.2, -1.9, 0.8, 3.1)   # o_lo, o_hi, c_lo, c_hi of the marine field

UNITS = [
    ("Unit A, packstone", BLUE, (-3.05, 2.15), (0.58, 0.52), 0.12, 62),
    ("Unit B, wackestone", GREEN, (-4.55, 1.15), (0.82, 0.70), 0.34, 58),
    ("Unit C, altered", GOLD, (-6.55, -0.95), (1.10, 1.35), 0.78, 55),
    ("Unit D, sparry", RED, (-8.60, -3.45), (1.30, 1.85), 0.89, 48),
]


def samples(seed=44):
    """One bivariate cloud per stratigraphic unit."""
    rng = np.random.default_rng(seed)
    out = []
    for name, color, (mo, mc), (so, sc), rho, n in UNITS:
        cov = [[so ** 2, rho * so * sc], [rho * so * sc, sc ** 2]]
        o, c = rng.multivariate_normal([mo, mc], cov, size=n).T
        out.append(dict(name=name, color=color, o=o, c=c))
    return out


def conf_ellipse(ax, x, y, color, n_std=2.4477):
    """95 % coverage ellipse (chi-square, two degrees of freedom)."""
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2.0 * n_std * np.sqrt(np.maximum(vals, 1e-12))
    ax.add_patch(Ellipse((x.mean(), y.mean()), w, h, angle=ang,
                         facecolor="none", edgecolor=color, lw=1.1,
                         ls=(0, (4.0, 2.2)), zorder=4))


def panel_cross(ax, data):
    o_all = np.concatenate([d["o"] for d in data])
    c_all = np.concatenate([d["c"] for d in data])

    o_lo, o_hi, c_lo, c_hi = PRIMARY
    ax.add_patch(Rectangle((o_lo, c_lo), o_hi - o_lo, c_hi - c_lo,
                           facecolor="#f0f0f0", edgecolor=GREY, lw=0.7,
                           ls=(0, (3.0, 2.2)), zorder=1))
    ax.text(0.5 * (o_lo + o_hi), c_hi + 0.18, "primary marine field",
            ha="center", va="bottom", fontsize=6.8, color=SUB)

    for d in data:
        ax.scatter(d["o"], d["c"], s=13, color=d["color"], alpha=0.72,
                   linewidths=0.0, zorder=3, label=d["name"])
        conf_ellipse(ax, d["o"], d["c"], d["color"])

    slope, icpt = np.polyfit(o_all, c_all, 1)
    r, p = stats.pearsonr(o_all, c_all)
    xs = np.linspace(o_all.min() - 0.5, o_all.max() + 0.5, 50)
    ax.plot(xs, slope * xs + icpt, color=SOLID, lw=1.1, ls=(0, (5.0, 2.4)),
            zorder=5)
    tip = o_all.min() - 0.1
    ax.annotate("meteoric\ndiagenesis", xy=(tip, slope * tip + icpt),
                xytext=(o_all.min() - 1.0, c_all.min() + 2.3),
                ha="left", va="center", fontsize=7.2, color=SOLID,
                linespacing=1.3,
                arrowprops=dict(arrowstyle="-|>", color=SOLID, lw=0.9,
                                shrinkA=3, shrinkB=2))

    ax.set_xlabel("$\\delta^{18}O$ (%s VPDB)" % PERMIL)
    ax.set_ylabel("$\\delta^{13}C$ (%s VPDB)" % PERMIL)
    ax.set_xlim(o_all.min() - 1.5, o_all.max() + 1.0)
    ax.set_ylim(c_all.min() - 1.2, c_all.max() + 1.3)
    ax.grid(lw=0.5, alpha=0.65)
    ax.legend(fontsize=7.0, frameon=False, loc="upper left",
              handletextpad=0.4, borderaxespad=0.25, scatterpoints=1)
    ax.text(0.985, 0.03,
            "all samples: $\\delta^{13}C$ = %.2f $\\delta^{18}O$ %+.2f,  "
            "r = %.2f %s,  n = %d"
            % (slope, icpt, r, sig_stars(p), o_all.size),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0,
            color=SOLID)
    ax.set_title("Two units fall off the primary field along one trend",
                 fontsize=9.0, fontweight="bold", color=INK)
    return slope, r


def panel_boxes(ax, data):
    """Both delta values, unit by unit, on one shared per-mil scale."""
    vals, colors, pos, ticks, labels = [], [], [], [], []
    for i, d in enumerate(data):
        base = i * 1.0
        vals += [d["o"], d["c"]]
        colors += [d["color"], d["color"]]
        pos += [base - 0.17, base + 0.17]
        ticks.append(base)
        labels.append(d["name"].split(",")[0].replace("Unit ", ""))

    bp = ax.boxplot(vals, positions=pos, widths=0.28, patch_artist=True,
                    showfliers=False, medianprops=dict(color="white", lw=1.0),
                    whiskerprops=dict(color=SOLID, lw=0.7),
                    capprops=dict(color=SOLID, lw=0.7))
    for patch, col, i in zip(bp["boxes"], colors, range(len(colors))):
        patch.set_facecolor(col)
        patch.set_alpha(0.95 if i % 2 == 0 else 0.45)
        patch.set_edgecolor(SOLID)
        patch.set_linewidth(0.6)

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_xlabel("stratigraphic unit (-)", fontsize=7.6)
    ax.set_ylabel("delta value (%s VPDB)" % PERMIL, fontsize=7.6)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.65)
    ax.axhline(0.0, color=GREY, lw=0.7, ls=(0, (2.4, 2.0)))
    ax.text(0.02, 0.03, "solid box $\\delta^{18}O$,  pale box $\\delta^{13}C$",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.4,
            color=SUB)
    ax.set_title("Distributions by unit", fontsize=8.5, fontweight="bold",
                 color=INK)


def panel_covariation(ax, data):
    """Per-unit C-O correlation with a Fisher-z 95 % interval."""
    ys = np.arange(len(data))[::-1]
    for y, d in zip(ys, data):
        r, p = stats.pearsonr(d["o"], d["c"])
        n = d["o"].size
        se = 1.0 / np.sqrt(n - 3)
        zt = np.arctanh(r)
        lo, hi = np.tanh(zt - 1.96 * se), np.tanh(zt + 1.96 * se)
        ax.plot([lo, hi], [y, y], color=d["color"], lw=1.6,
                solid_capstyle="round", zorder=3)
        ax.plot([r], [y], marker="o", ms=5.0, color=d["color"], mfc="white",
                mew=1.3, zorder=4)
        ax.text(1.03, y, "%s  n = %d" % (sig_stars(p), n), ha="left",
                va="center", fontsize=6.4, color=SUB)

    ax.axvline(R_THRESHOLD, color=RED, lw=0.9, ls=(0, (3.4, 2.2)), zorder=2)
    ax.text(R_THRESHOLD - 0.03, len(data) - 0.42,
            "rejection threshold r = %.2f" % R_THRESHOLD, ha="right",
            va="center", fontsize=6.4, color=RED)
    ax.axvline(0.0, color=GREY, lw=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([d["name"].split(",")[0] for d in data], fontsize=6.8)
    ax.set_ylim(-0.6, len(data) - 0.15)
    ax.set_xlim(-0.35, 1.30)
    ax.set_xticks([-0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("$\\delta^{13}C$ against $\\delta^{18}O$ correlation (-)",
                  fontsize=7.6)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.65)
    ax.set_title("Covariation screening", fontsize=8.5, fontweight="bold",
                 color=INK)


def main():
    set_pub_style()
    data = samples()

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (0.92, 0.82, 5.05, 4.62))
    ax_box = axes_inches(fig, (7.10, 3.28, 2.85, 2.16))
    ax_cov = axes_inches(fig, (7.10, 0.86, 2.85, 1.72))

    slope, r_all = panel_cross(ax_main, data)
    panel_boxes(ax_box, data)
    panel_covariation(ax_cov, data)

    flagged = [d["name"].split(",")[0] for d in data
               if stats.pearsonr(d["o"], d["c"])[0] > R_THRESHOLD]

    panel_labels_inches(fig, [ax_main, ax_box, ax_cov], dx=-0.62, dy=0.10)
    savefig(fig, "753_iso_delta_crossplot.png")


if __name__ == "__main__":
    main()

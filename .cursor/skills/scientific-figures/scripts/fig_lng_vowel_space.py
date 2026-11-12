"""Acoustic vowel space of two speaker generations in the F1-F2 plane.

The main panel is the phonetician's canonical chart: both axes run backwards so
that the origin sits at the top right and the vowel clouds arrange themselves
into the familiar quadrilateral that mirrors tongue position in the mouth.
Each vowel carries its own token cloud, a two-standard-deviation ellipse and an
ASCII label at the cloud centroid; the two speaker groups are separated by
colour and by ellipse line style.  The supporting panels quantify what the main
panel shows qualitatively: the convex-hull area spanned by the five centroids
(measured with scipy, not asserted) and the vowel durations that shrink along
with the space.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull

from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
OLD = "#3b5b92"          # house blue  - older speakers
YOUNG = "#b23a48"        # house red   - younger speakers

FW, FH = 10.4, 6.4
MAIN = (0.82, 0.84, 5.02, 4.56)
AUX_X, AUX_W, AUX_H = 7.08, 2.94, 1.98
AUX_TOP = (AUX_X, 3.42, AUX_W, AUX_H)
AUX_BOT = (AUX_X, 0.84, AUX_W, AUX_H)

N_TOK = 42
# label, F1 (Hz), F2 (Hz), duration (ms) for the older generation
OLDER = [("i", 305, 2290, 118),
         ("e", 445, 2010, 132),
         ("a", 760, 1330, 168),
         ("o", 515, 960, 150),
         ("u", 345, 830, 124)]
# the younger generation shows /u/ fronting and a lowered, retracted /a/
SHIFT = {"i": (10, -110, -14), "e": (-10, -60, -16), "a": (-75, 40, -22),
         "o": (-10, 40, -18), "u": (-12, 260, -12)}


def tokens(seed=804):
    """Synthetic formant clouds; returns {group: {vowel: (F2, F1, dur)}}."""
    rng = np.random.default_rng(seed)
    out = {"older": {}, "younger": {}}
    for lab, f1, f2, dur in OLDER:
        d1, d2, dd = SHIFT[lab]
        for grp, (m1, m2, md) in (("older", (f1, f2, dur)),
                                  ("younger", (f1 + d1, f2 + d2, dur + dd))):
            s1 = 0.075 * m1
            s2 = 0.065 * m2
            rho = -0.35 if lab in ("i", "e") else 0.30
            cov = [[s2 ** 2, rho * s1 * s2], [rho * s1 * s2, s1 ** 2]]
            xy = rng.multivariate_normal([m2, m1], cov, size=N_TOK)
            dr = rng.normal(md, 0.13 * md, size=N_TOK)
            out[grp][lab] = (xy[:, 0], xy[:, 1], dr)
    return out


def conf_ellipse(ax, x, y, color, ls, n_std=2.0):
    cov = np.cov(x, y)
    pear = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell = Ellipse((0.0, 0.0), 2.0 * np.sqrt(1.0 + pear),
                  2.0 * np.sqrt(1.0 - pear), facecolor=color, alpha=0.11,
                  edgecolor=color, linewidth=1.05, linestyle=ls, zorder=3)
    tr = (mtransforms.Affine2D()
          .rotate_deg(45.0)
          .scale(n_std * np.sqrt(cov[0, 0]), n_std * np.sqrt(cov[1, 1]))
          .translate(x.mean(), y.mean()))
    ell.set_transform(tr + ax.transData)
    ax.add_patch(ell)


def main_panel(ax, data):
    hulls = {}
    for grp, color, ls, mk in (("older", OLD, "-", "o"),
                               ("younger", YOUNG, (0, (3.4, 2.0)), "^")):
        cent = []
        for lab, _, _, _ in OLDER:
            f2, f1, _ = data[grp][lab]
            ax.scatter(f2, f1, s=6.5, color=color, alpha=0.34,
                       marker=mk, linewidths=0.0, zorder=2)
            conf_ellipse(ax, f2, f1, color, ls)
            cent.append((f2.mean(), f1.mean()))
        cent = np.asarray(cent)
        hull = ConvexHull(cent)
        loop = np.append(hull.vertices, hull.vertices[0])
        ax.plot(cent[loop, 0], cent[loop, 1], color=color, lw=0.9,
                ls=(0, (1.4, 1.8)), alpha=0.85, zorder=4)
        for (lab, _, _, _), (cx, cy) in zip(OLDER, cent):
            t = ax.text(cx, cy, lab, ha="center", va="center", fontsize=9.5,
                        fontweight="bold", color=color, zorder=6,
                        fontstyle="italic")
            t.set_path_effects([pe.withStroke(linewidth=2.4,
                                              foreground="white")])
        hulls[grp] = hull.volume

    ax.set_xlim(2760, 600)
    ax.set_ylim(1030, 215)
    ax.set_xlabel("second formant  F2 (Hz)")
    ax.set_ylabel("first formant  F1 (Hz)")
    ax.set_xticks(np.arange(700, 2601, 300))
    ax.set_yticks(np.arange(300, 1001, 100))
    ax.grid(lw=0.5, alpha=0.55)
    ax.set_axisbelow(True)

    ax.annotate("front", xy=(2720, 243), ha="left", va="center",
                fontsize=6.8, color=MUTED)
    ax.annotate("back", xy=(640, 243), ha="right", va="center",
                fontsize=6.8, color=MUTED)
    ax.annotate("close", xy=(628, 300), ha="center", va="top", rotation=90,
                fontsize=6.8, color=MUTED)
    ax.annotate("open", xy=(628, 1000), ha="center", va="bottom", rotation=90,
                fontsize=6.8, color=MUTED)

    fu = data["older"]["u"][0].mean(), data["older"]["u"][1].mean()
    yu = data["younger"]["u"][0].mean(), data["younger"]["u"][1].mean()
    ax.annotate("", xy=(yu[0], yu[1]), xytext=(fu[0], fu[1]),
                arrowprops=dict(arrowstyle="-|>", color=SUB, lw=0.9,
                                shrinkA=6, shrinkB=6))
    ax.text(0.5 * (fu[0] + yu[0]), 0.5 * (fu[1] + yu[1]) - 42,
            "/u/ fronting  +%d Hz F2" % round(yu[0] - fu[0]), ha="center",
            va="bottom", fontsize=7.0, color=SUB)

    handles = [Line2D([], [], color=OLD, marker="o", ms=3.6, lw=1.05,
                      label="older speakers (n=%d)" % N_TOK),
               Line2D([], [], color=YOUNG, marker="^", ms=3.6, lw=1.05,
                      ls=(0, (3.4, 2.0)),
                      label="younger speakers (n=%d)" % N_TOK)]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=7.2,
              handlelength=2.2, borderaxespad=0.35)
    ax.set_title("Both axes run backwards: the chart is a map of the mouth",
                 fontsize=9.0, fontweight="bold", color=INK)
    return hulls


def area_panel(ax, hulls):
    keys = ["older", "younger"]
    vals = [hulls[k] / 1e5 for k in keys]
    ax.bar([0, 1], vals, width=0.52, color=[OLD, YOUNG], alpha=0.88,
           edgecolor="white", linewidth=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03 * max(vals), "%.2f" % v, ha="center", va="bottom",
                fontsize=6.8, color=INK)
    drop = 100.0 * (1.0 - vals[1] / vals[0])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["older", "younger"], fontsize=6.8)
    ax.set_ylim(0, max(vals) * 1.28)
    ax.set_ylabel(r"hull area ($10^5$ Hz$^2$)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.text(0.5, 0.90, "space contracts %.1f %%" % drop,
            transform=ax.transAxes, ha="center", va="top", fontsize=6.8,
            color=SUB)
    ax.set_title("Convex-hull area of the five centroids", fontsize=8.2,
                 fontweight="bold", color=INK)
    return drop


def duration_panel(ax, data):
    labs = [v[0] for v in OLDER]
    xs = np.arange(len(labs), dtype=float)
    for k, (grp, color) in enumerate((("older", OLD), ("younger", YOUNG))):
        mu = [data[grp][l][2].mean() for l in labs]
        sd = [data[grp][l][2].std() for l in labs]
        ax.bar(xs + (k - 0.5) * 0.34, mu, width=0.32, color=color, alpha=0.88,
               edgecolor="white", linewidth=0.7, yerr=sd, capsize=1.8,
               error_kw=dict(elinewidth=0.7, ecolor=GREY))
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, fontsize=7.2, fontstyle="italic")
    ax.set_xlabel("vowel", fontsize=7.4)
    ax.set_ylabel("duration (ms)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    all_mu = np.array([[data[g][l][2].mean() for l in labs]
                       for g in ("older", "younger")])
    ax.set_ylim(0, all_mu.max() * 1.42)
    ax.set_title("Vowel duration by generation", fontsize=8.2,
                 fontweight="bold", color=INK)
    return 100.0 * (1.0 - all_mu[1].mean() / all_mu[0].mean())


def main():
    set_pub_style()
    data = tokens()

    fig = plt.figure(figsize=(FW, FH))
    ax_main = fig.add_axes([MAIN[0] / FW, MAIN[1] / FH,
                            MAIN[2] / FW, MAIN[3] / FH])
    ax_area = fig.add_axes([AUX_TOP[0] / FW, AUX_TOP[1] / FH,
                            AUX_TOP[2] / FW, AUX_TOP[3] / FH])
    ax_dur = fig.add_axes([AUX_BOT[0] / FW, AUX_BOT[1] / FH,
                           AUX_BOT[2] / FW, AUX_BOT[3] / FH])

    hulls = main_panel(ax_main, data)
    drop = area_panel(ax_area, hulls)
    dur_drop = duration_panel(ax_dur, data)

    panel_labels_inches(fig, [ax_main, ax_area, ax_dur], dx=-0.52, dy=0.10)
    savefig(fig, "804_lng_vowel_space.png")


if __name__ == "__main__":
    main()

"""Ceramic typology: drawn profile sequence against its morphometric space.

The plate on the left is the primary evidence and is drawn the way an
excavation report draws it - section on the left of the centre line, exterior
elevation on the right, one row per ware type and one column per stage of the
sequence.  Every profile is a parametric curve built from the same six
measurements that feed the ordination, so the shapes and the scatter cannot
drift apart: change a vessel and its point moves.

The right-hand panels are supporting evidence.  The ordination shows that the
four ware groups occupy separate but touching regions of shape space, and the
span bars put each type on the calendar so the reader can see that the
morphological drift is also a chronological one.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import PchipInterpolator
from scipy.spatial import ConvexHull

from pubstyle import set_pub_style, savefig, darken, lighten
from figkit import axes_inches, panel_labels_inches, shared_limits

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HAIR = "#cccccc"

PERIOD_COLOR = {
    "Neolithic": "#8e6bbf",
    "Bronze Age": "#3b5b92",
    "Iron Age": "#2e7d5b",
    "Roman": "#c9a227",
    "Medieval": "#b23a48",
}

FW, FH = 10.4, 6.4
SCALE = 0.024                    # drawing inches per centimetre of vessel
WALL_DRAW = 0.031                # section wall thickness on paper, inches

# code, name, period, calendar span (negative = BC), neck factor
TYPES = [
    ("I", "Cordoned urn", "Bronze Age", (-1400, -820), 1.06),
    ("II", "Barrel jar", "Iron Age", (-760, -60), 1.00),
    ("III", "Carinated bowl", "Roman", (40, 400), 0.98),
    ("IV", "Glazed jug", "Medieval", (1060, 1450), 0.84),
]

# per type: height cm, max body diameter cm, base diameter cm, height of the
# maximum as a fraction of total, wall thickness mm - stage 1 value and the
# increment applied at each later stage.  Rim diameter is stored as a fraction
# of the maximum diameter and is the parameter the sequence is built around.
ARCHETYPE = {
    "I":   dict(h=(26.0, -0.9), dmax=(22.0, 0.35), dbase=(9.5, 0.30),
                hmax=(0.62, -0.020), wall=(9.4, -0.32), rimf=(0.62, 0.062)),
    "II":  dict(h=(23.0, -0.5), dmax=(19.5, 0.55), dbase=(10.5, 0.20),
                hmax=(0.50, 0.014), wall=(8.2, -0.26), rimf=(0.70, 0.048)),
    "III": dict(h=(11.5, 0.35), dmax=(24.0, 0.70), dbase=(8.0, 0.35),
                hmax=(0.58, 0.022), wall=(5.4, -0.18), rimf=(0.86, 0.030)),
    "IV":  dict(h=(24.5, 0.60), dmax=(17.0, -0.30), dbase=(9.0, -0.15),
                hmax=(0.42, 0.016), wall=(6.1, -0.20), rimf=(0.55, 0.055)),
}
N_STAGE = 4
FEATURES = ["height", "max diameter", "base diameter", "shoulder height",
            "wall thickness", "rim aperture"]


def archetype(code, stage):
    """The six measurements of one type-stage form."""
    a = ARCHETYPE[code]
    return {k: v0 + stage * dv for k, (v0, dv) in a.items()}


def profile(m, neck_f, n=260):
    """Radius as a function of height, in centimetres, for one form."""
    rb, rmax = 0.5 * m["dbase"], 0.5 * m["dmax"]
    rrim = 0.5 * m["dmax"] * m["rimf"]
    rneck = rrim * neck_f
    hm = float(np.clip(m["hmax"], 0.22, 0.72))
    h4 = min(hm + 0.52 * (1.0 - hm), 0.80)
    hs = np.array([0.0, 0.05, 0.55 * hm, hm, h4, 0.90, 1.0])
    rs = np.array([rb, rb * 1.06, rb + 0.72 * (rmax - rb), rmax,
                   rmax - 0.52 * (rmax - rneck), rneck, rrim])
    f = PchipInterpolator(hs, rs)
    t = np.linspace(0.0, 1.0, n)
    return t * m["h"], f(t)


def draw_vessel(ax, cx, y0, m, neck_f, color):
    """Archaeological convention: section left of the axis, elevation right."""
    z_cm, r_cm = profile(m, neck_f)
    z, r = z_cm * SCALE, r_cm * SCALE
    tb = 0.036                                    # base slab on paper
    t = WALL_DRAW

    ax.fill_betweenx(y0 + z, cx - r, cx + r, color=lighten(color, 0.80),
                     lw=0.0, zorder=2)
    ax.plot(cx + r, y0 + z, color=RULE, lw=0.85, zorder=4)
    ax.plot([cx - r[-1], cx + r[-1]], [y0 + z[-1]] * 2, color=RULE, lw=0.85,
            zorder=4)
    ax.plot([cx, cx], [y0 - 0.02, y0 + z[-1] + 0.03], color="#9a9a9a",
            lw=0.5, ls=(0, (2.4, 2.0)), zorder=3)

    keep = z >= tb
    inner = np.minimum(cx - r[keep] + t, cx)
    poly_x = np.concatenate([cx - r, inner[::-1], [cx, cx]])
    poly_y = np.concatenate([y0 + z, (y0 + z[keep])[::-1],
                             [y0 + tb, y0]])
    ax.fill(poly_x, poly_y, facecolor=darken(color, 0.45), edgecolor=RULE,
            linewidth=0.6, zorder=5)
    return float(r_cm.max()), float(r_cm[-1])


def draw_plate(ax, w, h):
    ax.set_xlim(0.0, w)
    ax.set_ylim(0.0, h)
    ax.set_aspect("equal")
    ax.axis("off")

    band = (h - 0.34) / len(TYPES)
    col_w = w / N_STAGE
    ratios = {}
    for i, (code, name, period, span, nf) in enumerate(TYPES):
        col = PERIOD_COLOR[period]
        y_row = h - 0.34 - (i + 1) * band
        if i:
            ax.plot([0.02, w - 0.02], [y_row + band] * 2, color=HAIR,
                    lw=0.6, zorder=1)
        ax.text(0.03, y_row + 0.845, f"Type {code}  {name}  -  {period}",
                ha="left", va="bottom", fontsize=6.9, fontweight="bold",
                color=col, zorder=6)
        for s in range(N_STAGE):
            m = archetype(code, s)
            cx = (s + 0.5) * col_w
            rmax, rrim = draw_vessel(ax, cx, y_row + 0.19, m, nf, col)
            ratios[(code, s)] = rrim / rmax
            ax.text(cx, y_row + 0.175, f"{code}.{s + 1}", ha="center",
                    va="top", fontsize=6.4, color=INK, fontweight="bold")
            ax.text(cx, y_row + 0.092,
                    f"h {m['h']:.0f} cm  rim {m['dmax'] * m['rimf']:.0f} cm",
                    ha="center", va="top", fontsize=6.0, color=MUTED)

    # scale bar in the reserved top strip: the drawing is metric
    bar = 10.0 * SCALE
    x0, yb = 0.03, h - 0.21
    for k in range(2):
        ax.add_patch(Rectangle((x0 + k * bar / 2.0, yb), bar / 2.0, 0.038,
                               facecolor="#3a3a3a" if k % 2 == 0 else "white",
                               edgecolor="#3a3a3a", linewidth=0.5, zorder=6))
    ax.text(x0 + bar + 0.05, yb + 0.019, "10 cm", ha="left", va="center",
            fontsize=6.2, color=SUB)
    ax.text(w - 0.03, yb + 0.019,
            "left of centre line: section   |   right: exterior elevation",
            ha="right", va="center", fontsize=6.2, color=MUTED)
    ax.set_title("Ware sequence plate, Trench 4 assemblage", fontsize=8.5,
                 fontweight="bold")
    return ratios


def morphometrics(rng, n_per=32):
    rows, labels = [], []
    for code, name, period, span, nf in TYPES:
        for s in range(N_STAGE):
            m = archetype(code, s)
            base = np.array([m["h"], m["dmax"], m["dbase"], m["hmax"] * 100.0,
                             m["wall"], m["rimf"] * 100.0])
            k = n_per // N_STAGE
            noise = rng.normal(0.0, 1.0, (k, 6)) * np.array(
                [1.5, 1.3, 0.9, 3.2, 0.45, 3.4])
            rows.append(base + noise)
            labels += [code] * k
    return np.vstack(rows), np.array(labels)


def pca(X):
    Xc = (X - X.mean(0)) / X.std(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = U * S
    evr = S ** 2 / np.sum(S ** 2)
    return scores, evr, Vt


def draw_pca(ax, scores, labels, evr):
    for code, name, period, span, nf in TYPES:
        col = PERIOD_COLOR[period]
        m = labels == code
        p = scores[m][:, :2]
        ax.scatter(p[:, 0], p[:, 1], s=11, color=col, alpha=0.72,
                   edgecolors="white", linewidths=0.35, zorder=4,
                   label=f"Type {code}")
        hull = ConvexHull(p)
        loop = np.append(hull.vertices, hull.vertices[0])
        ax.fill(p[loop, 0], p[loop, 1], facecolor=col, alpha=0.11, lw=0.0,
                zorder=2)
        ax.plot(p[loop, 0], p[loop, 1], color=col, lw=0.8, alpha=0.75,
                zorder=3)
    ax.axhline(0.0, color=HAIR, lw=0.6, zorder=1)
    ax.axvline(0.0, color=HAIR, lw=0.6, zorder=1)
    ax.set_xlabel(f"PC1 ({evr[0]:.1%} of shape variance)", fontsize=8.0)
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.set_title("Morphometric ordination of 128 measured sherds",
                 fontsize=8.5, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7.0, loc="upper left", handletextpad=0.3,
              borderpad=0.2, labelspacing=0.28)


def year_fmt(v, _pos=None):
    v = int(round(v))
    if v < 0:
        return f"{-v} BC"
    return f"AD {v}"


def draw_spans(ax):
    for i, (code, name, period, span, nf) in enumerate(TYPES):
        y = len(TYPES) - 1 - i
        col = PERIOD_COLOR[period]
        a, b = span
        ax.barh(y, b - a, left=a, height=0.52, color=col, alpha=0.42,
                edgecolor=darken(col, 0.30), linewidth=0.7, zorder=3)
        mids = a + (np.arange(N_STAGE) + 0.5) * (b - a) / N_STAGE
        ax.plot(mids, [y] * N_STAGE, marker="|", ms=7.5, mew=1.0,
                color=darken(col, 0.35), ls="none", zorder=5)
        ax.text(a - 28, y, f"Type {code}", ha="right", va="center",
                fontsize=6.6, color=INK)
    ax.set_yticks([])
    ax.set_ylim(-0.7, len(TYPES) - 0.3)
    ends = [v for _c, _n, _p, sp, _f in TYPES for v in sp]
    ax.set_xlim(min(ends) - 420, max(ends) + 190)
    ax.xaxis.set_major_formatter(FuncFormatter(year_fmt))
    ax.set_xticks([-1500, -1000, -500, 0, 500, 1000, 1500])
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel("Calendar age (cal BC / cal AD, yr)", fontsize=7.6)
    ax.set_title("Currency of each ware type", fontsize=8.0,
                 fontweight="bold")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def main():
    set_pub_style()
    rng = np.random.default_rng(631)

    fig = plt.figure(figsize=(FW, FH))
    ax_p = axes_inches(fig, (0.44, 0.96, 5.30, 4.20))
    ax_o = axes_inches(fig, (6.46, 2.78, 3.52, 2.38))
    ax_t = axes_inches(fig, (6.46, 0.88, 3.52, 1.18))

    ratios = draw_plate(ax_p, 5.30, 4.20)
    X, labels = morphometrics(rng)
    scores, evr, _ = pca(X)
    draw_pca(ax_o, scores, labels, evr)
    draw_spans(ax_t)

    # one padded range for both ordination axes, so the hulls are not
    # distorted relative to one another
    lim = shared_limits([scores[:, 0], scores[:, 1]], pad=0.09)
    ax_o.set_xlim(*lim)
    ax_o.set_ylim(*lim)

    panel_labels_inches(fig, [ax_p, ax_o, ax_t], dx=-0.30, dy=0.20)

    r1 = np.mean([ratios[(c, 0)] for c, *_ in TYPES])
    r4 = np.mean([ratios[(c, N_STAGE - 1)] for c, *_ in TYPES])
    walls = [archetype(c, 0)["wall"] - archetype(c, N_STAGE - 1)["wall"]
             for c, *_ in TYPES]


    print(f"PC1 {evr[0]:.3f}  PC2 {evr[1]:.3f}  n={X.shape[0]}")
    print(f"rim ratio stage1 {r1:.3f} -> stage4 {r4:.3f}")
    savefig(fig, "631_arc_artifact_typology.png")


if __name__ == "__main__":
    main()

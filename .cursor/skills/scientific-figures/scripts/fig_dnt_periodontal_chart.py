"""Full-mouth periodontal chart: unrolled arches with six probing strips.

The chart is the signature figure of the discipline, so the layout follows the
clinical convention exactly: two rows of FDI-numbered crowns, and around each
arch a buccal and a lingual/palatal strip carrying three probing sites per
tooth.  Every strip is drawn from the same synthetic site table, so the
recession line, the shaded attachment-loss band (which spans exactly
PD + recession) and the red deep-site markers can never disagree with one
another.  All quoted statistics - the fraction of sites deeper than 4 mm, the
per-quadrant mean attachment loss, the bleeding index - are reduced from that
same table at run time.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

FW, FH = 10.4, 6.4
MAIN = (0.86, 0.86, 5.58, 4.56)
AUX_T = (7.00, 3.52, 3.12, 1.90)
AUX_B = (7.00, 0.86, 3.12, 1.76)

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
GRID, HAIR = "#cccccc", "#d5d5d5"
RED, BLUE, GREEN, GOLD = "#b23a48", "#3b5b92", "#2e7d5b", "#c9a227"
TISSUE = "#c0603f"          # gingival margin / recession trace
ENAMEL = "#f0ebe0"          # crown fill
EDGE = "#8a8a8a"            # crown outline

# Geometry of the unrolled chart, in chart units.  S converts millimetres of
# probing depth into chart units so that both arches use one ruler.
S = 0.22
XT = np.arange(16) + 0.5
OFF = np.array([-0.30, 0.0, 0.30])
PD_MAX = 7

MX_B_BASE, MX_P_BASE = 12.45, 9.55
MN_L_BASE, MN_B_BASE = 6.30, 3.50
MX_CROWN = (11.62, 10.22)   # (cervical, occlusal): maxillary crowns point down
MN_CROWN = (4.20, 5.60)     # mandibular crowns point up

UPPER = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
MISSING = {18, 28, 46, 38}

FOOT = ("Synthetic full-mouth chart, six sites per tooth (mesio-, mid- and "
        "disto- on the buccal and lingual/palatal aspects), FDI tooth "
        "notation. Shaded band spans clinical attachment level = probing "
        "depth + recession; red markers flag sites deeper than the 4 mm "
        "threshold and open triangles mark bleeding on probing. Illustrative "
        "data only - not a real patient record.")


def fit_limits(ax, xlim, ylim, rect):
    """Pad the shorter data axis so an equal-aspect panel fills its box."""
    want = rect[2] / rect[3]
    (x0, x1), (y0, y1) = xlim, ylim
    if (x1 - x0) / (y1 - y0) < want:
        e = (y1 - y0) * want - (x1 - x0)
        x0, x1 = x0 - e / 2.0, x1 + e / 2.0
    else:
        e = (x1 - x0) / want - (y1 - y0)
        y0, y1 = y0 - e / 2.0, y1 + e / 2.0
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")


def crown_poly(cx, y_cerv, y_occl, w, n_cusp):
    """Simplified facial crown outline; cusps are bumps on the occlusal edge."""
    s = np.sign(y_occl - y_cerv)
    h = abs(y_occl - y_cerv)
    t = np.linspace(0.0, 1.0, 80)
    xs = np.linspace(-0.46 * w, 0.46 * w, 80)
    occl = y_cerv + s * (0.84 * h + 0.16 * h * np.abs(np.sin(n_cusp * np.pi * t)))
    head = [(-0.36 * w, y_cerv), (-0.50 * w, y_cerv + s * 0.40 * h)]
    tail = [(0.50 * w, y_cerv + s * 0.40 * h), (0.36 * w, y_cerv)]
    pts = np.vstack([head, np.column_stack([xs, occl]), tail])
    pts[:, 0] += cx
    return pts


def tooth_shape(code):
    """(crown width, cusp count) from the FDI tooth-type digit."""
    kind = code % 10
    if kind in (1, 2):
        return 0.56, 1
    if kind == 3:
        return 0.62, 1
    if kind in (4, 5):
        return 0.70, 2
    return 0.84, 3


def make_sites(rng, tilt):
    """Probing depth, recession and bleeding for 16 teeth x 2 aspects x 3."""
    n = len(XT)
    post = np.abs(np.arange(n) - 7.5) / 7.5
    pd = np.zeros((n, 2, 3))
    rec = np.zeros((n, 2, 3))
    for i in range(n):
        base = 2.1 + 2.4 * post[i] ** 1.6 + tilt * (i / n)
        for a in range(2):
            v = base + rng.normal(0.0, 0.55, 3) + (0.45 if a == 1 else 0.0)
            pd[i, a] = np.clip(np.round(v), 1, PD_MAX)
            r = rng.normal(0.5 + 1.0 * post[i], 0.7, 3)
            rec[i, a] = np.clip(np.round(r), 0, 3)
    bop = rng.random(pd.shape) < np.clip((pd - 2.0) / 7.0, 0.0, 0.9)
    return pd, rec, bop


def draw_strip(ax, base, sign, pd, rec, bop, present):
    """One probing strip: threshold rule, CAL band, depth trace, BOP flags."""
    xs = (XT[:, None] + OFF[None, :]).ravel()
    d = pd.ravel().copy()
    r = rec.ravel().copy()
    keep = np.repeat(present, 3)
    d[~keep] = np.nan
    r[~keep] = np.nan

    ax.axhline(base, color=HAIR, lw=0.6, zorder=1)
    ax.plot([XT[0] - 0.5, XT[-1] + 0.5], [base + sign * 4.0 * S] * 2,
            color=RED, lw=0.6, ls=(0, (2.2, 2.0)), alpha=0.75, zorder=1)

    y_pd = base + sign * d * S
    y_rec = base - sign * r * S
    ax.fill_between(xs, y_rec, y_pd, color=RED, alpha=0.14, lw=0, zorder=2)
    ax.plot(xs, y_rec, color=TISSUE, lw=0.8, zorder=3)
    ax.plot(xs, y_pd, color=BLUE, lw=0.9, zorder=4)

    deep = d > 4.0
    ax.plot(xs[~deep & keep], y_pd[~deep & keep], ls="none", marker="o",
            ms=1.9, mfc=BLUE, mec="none", zorder=5)
    ax.plot(xs[deep], y_pd[deep], ls="none", marker="o", ms=3.1, mfc=RED,
            mec="white", mew=0.4, zorder=6)

    b = bop.ravel() & keep
    ax.plot(xs[b], y_pd[b] + sign * 0.20, ls="none", marker="^", ms=2.6,
            mfc="none", mec=RED, mew=0.6, zorder=6)


def draw_arch(ax, codes, crown, base_out, base_in, sign_out, pd, rec, bop):
    present = np.array([c not in MISSING for c in codes])
    y_cerv, y_occl = crown
    for i, code in enumerate(codes):
        w, cusps = tooth_shape(code)
        if present[i]:
            ax.add_patch(Polygon(crown_poly(XT[i], y_cerv, y_occl, w, cusps),
                                 closed=True, facecolor=ENAMEL,
                                 edgecolor=EDGE, lw=0.55, zorder=3))
        else:
            k = 0.30
            ym = 0.5 * (y_cerv + y_occl)
            for sx in (-1, 1):
                ax.plot([XT[i] - sx * k, XT[i] + sx * k], [ym - k, ym + k],
                        color=MUTED, lw=1.0, zorder=3)
    draw_strip(ax, base_out, sign_out, pd[:, 0], rec[:, 0], bop[:, 0], present)
    draw_strip(ax, base_in, -sign_out, pd[:, 1], rec[:, 1], bop[:, 1], present)
    return present


def main():
    set_pub_style()
    rng = np.random.default_rng(816)

    pd_u, rec_u, bop_u = make_sites(rng, tilt=0.35)
    pd_l, rec_l, bop_l = make_sites(rng, tilt=-0.25)

    fig = plt.figure(figsize=(FW, FH))
    ax = axes_inches(fig, MAIN)
    ax.axis("off")

    up = draw_arch(ax, UPPER, MX_CROWN, MX_B_BASE, MX_P_BASE, +1,
                   pd_u, rec_u, bop_u)
    lo = draw_arch(ax, LOWER, MN_CROWN, MN_B_BASE, MN_L_BASE, -1,
                   pd_l, rec_l, bop_l)

    # Tooth numbers, arch/aspect captions and the midline split.
    for codes, y, present in ((UPPER, 14.30, up), (LOWER, 1.65, lo)):
        for i, code in enumerate(codes):
            ax.text(XT[i], y, str(code), ha="center", va="center",
                    fontsize=6.0,
                    color=MUTED if not present[i] else RULE)
    ax.plot([8.0, 8.0], [1.30, 14.55], color=HAIR, lw=0.7,
            ls=(0, (3, 2.5)), zorder=0)
    for x, lab, y in ((4.0, "Q1  upper right", 14.85),
                      (12.0, "Q2  upper left", 14.85),
                      (4.0, "Q4  lower right", 1.15),
                      (12.0, "Q3  lower left", 1.15)):
        ax.text(x, y, lab, ha="center", va="center", fontsize=6.2, color=SUB)

    for y, lab in ((MX_B_BASE + 0.34, "Buccal"), (MX_P_BASE - 0.34, "Palatal"),
                   (MN_L_BASE + 0.34, "Lingual"), (MN_B_BASE - 0.34, "Buccal")):
        ax.text(-0.20, y, lab, ha="right", va="center", fontsize=6.2,
                color=SUB)
    for y, lab in ((11.00, "Maxillary"), (4.90, "Mandibular")):
        ax.text(-1.62, y, lab, ha="center", va="center", fontsize=7.0,
                color=INK, fontweight="bold", rotation=90)

    # A 4 mm calibration bracket makes the vertical ruler explicit.
    xr = 16.05
    ax.plot([xr, xr], [MX_B_BASE, MX_B_BASE + 4 * S], color=RULE, lw=0.9)
    for yy in (MX_B_BASE, MX_B_BASE + 4 * S):
        ax.plot([xr - 0.14, xr + 0.14], [yy, yy], color=RULE, lw=0.9)
    ax.text(xr, MX_B_BASE + 4 * S + 0.14, "4 mm", ha="center", va="bottom",
            fontsize=6.2, color=RULE)

    handles = [Line2D([], [], color=BLUE, lw=0.9, marker="o", ms=2.4,
                      label="probing depth"),
               Line2D([], [], color=RED, lw=0, marker="o", ms=3.1,
                      label="site > 4 mm"),
               Line2D([], [], color=TISSUE, lw=0.8, label="gingival margin"),
               Patch(facecolor=RED, alpha=0.14, label="attachment loss"),
               Line2D([], [], color=RED, lw=0, marker="^", ms=2.8,
                      mfc="none", mew=0.6, label="bleeding on probing"),
               Line2D([], [], color=MUTED, lw=0, marker="x", ms=3.6, mew=1.0,
                      label="missing tooth")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.005),
              ncol=3, frameon=False, fontsize=6.4, handlelength=1.5,
              columnspacing=1.4, labelspacing=0.3)

    fit_limits(ax, (-1.95, 16.55), (1.00, 16.30), MAIN)
    ax.set_title("Unrolled arch chart, six sites per tooth", fontsize=8.8,
                 fontweight="bold", color=INK, pad=4)

    # ---- statistics, all reduced from the arrays plotted above -------------
    keep_u = np.repeat(np.array([c not in MISSING for c in UPPER]), 6)
    keep_l = np.repeat(np.array([c not in MISSING for c in LOWER]), 6)
    pd_all = np.concatenate([pd_u.reshape(16, 6).ravel()[keep_u],
                             pd_l.reshape(16, 6).ravel()[keep_l]])
    rec_all = np.concatenate([rec_u.reshape(16, 6).ravel()[keep_u],
                              rec_l.reshape(16, 6).ravel()[keep_l]])
    bop_all = np.concatenate([bop_u.reshape(16, 6).ravel()[keep_u],
                              bop_l.reshape(16, 6).ravel()[keep_l]])
    cal_all = pd_all + rec_all
    frac4 = float(np.mean(pd_all > 4.0)) * 100.0
    frac6 = float(np.mean(pd_all > 6.0)) * 100.0
    bleed = float(np.mean(bop_all)) * 100.0

    ax_b = axes_inches(fig, AUX_T)
    edges = np.arange(0.5, PD_MAX + 1.6, 1.0)
    counts, _ = np.histogram(pd_all, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    cols = [RED if c > 4.0 else BLUE for c in centres]
    ax_b.bar(centres, counts, width=0.82, color=cols, edgecolor="white",
             linewidth=0.5)
    for x, c in zip(centres, counts):
        if c:
            ax_b.text(x, c + 0.6, str(int(c)), ha="center", va="bottom",
                      fontsize=6.0, color=SUB)
    ax_b.set_ylim(0, counts.max() * 1.30)
    for thr, lab in ((4.5, "4 mm"), (6.5, "6 mm")):
        ax_b.axvline(thr, color=RULE, lw=0.7, ls=(0, (2.5, 2.0)))
        ax_b.text(thr + 0.10, ax_b.get_ylim()[1] * 0.30, lab, ha="left",
                  va="bottom", fontsize=6.2, color=RULE)
    ax_b.set_xlabel("probing depth (mm)", fontsize=8.0)
    ax_b.set_ylabel("sites (n)", fontsize=8.0)
    ax_b.set_xticks(np.arange(1, PD_MAX + 1))
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_ylim(0, counts.max() * 1.30)
    ax_b.set_title(f"Site depth distribution (n={pd_all.size})", fontsize=8.5,
                   fontweight="bold", color=INK, pad=3)
    ax_b.text(0.98, 0.92,
              f"> 4 mm: {frac4:.1f} %\n> 6 mm: {frac6:.1f} %\n"
              f"BOP: {bleed:.1f} %",
              transform=ax_b.transAxes, ha="right", va="top", fontsize=6.5,
              color=INK, linespacing=1.5)

    ax_c = axes_inches(fig, AUX_B)
    quad_names = ["Q1\nUR", "Q2\nUL", "Q3\nLL", "Q4\nLR"]
    quad_cal, quad_err = [], []
    for codes, pdq, recq, half in ((UPPER, pd_u, rec_u, 0), (UPPER, pd_u, rec_u, 1),
                                   (LOWER, pd_l, rec_l, 1), (LOWER, pd_l, rec_l, 0)):
        sl = slice(0, 8) if half == 0 else slice(8, 16)
        ok = np.array([c not in MISSING for c in codes[sl]])
        v = (pdq[sl] + recq[sl]).reshape(8, 6)[ok].ravel()
        quad_cal.append(v.mean())
        quad_err.append(v.std(ddof=1) / np.sqrt(v.size))
    quad_cal = np.array(quad_cal)
    quad_err = np.array(quad_err)
    worst = int(np.argmax(quad_cal))
    cols = [RED if i == worst else BLUE for i in range(4)]
    ax_c.bar(np.arange(4), quad_cal, yerr=quad_err, width=0.62, color=cols,
             edgecolor="white", linewidth=0.5,
             error_kw=dict(ecolor=RULE, elinewidth=0.7, capsize=2.0))
    for i, v in enumerate(quad_cal):
        ax_c.text(i, v + quad_err[i] + 0.10, f"{v:.1f}", ha="center",
                  va="bottom", fontsize=6.2, color=SUB)
    ax_c.axhline(float(cal_all.mean()), color=RULE, lw=0.7,
                 ls=(0, (2.5, 2.0)))
    ax_c.set_xlim(-0.85, 3.5)
    ax_c.text(-0.80, float(cal_all.mean()) + 0.06, "mean", ha="left",
              va="bottom", fontsize=6.0, color=RULE)
    ax_c.set_xticks(np.arange(4))
    ax_c.set_xticklabels(quad_names, fontsize=6.6)
    ax_c.set_ylabel("mean CAL (mm)", fontsize=8.0)
    ax_c.tick_params(labelsize=6.6)
    ax_c.set_ylim(0, quad_cal.max() * 1.32)
    ax_c.set_title("Attachment loss by quadrant", fontsize=8.5,
                   fontweight="bold", color=INK, pad=3)

    panel_labels_inches(fig, [ax, ax_b, ax_c], dx=-0.46, dy=0.10)
    savefig(fig, "816_dnt_periodontal_chart.png")


if __name__ == "__main__":
    main()

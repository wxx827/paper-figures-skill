"""Excavation section read against a Bayesian radiocarbon sequence.

The trench drawing and the dating panel share one downward depth axis, which
is the whole argument of the figure: a calibrated probability density only
means something once it is pinned to the context it came from.  Cultural
deposits carry both an approximate soil colour and a hatch ornament, because
the ornament survives thumbnailing while the colour does not, and the two
negative features are drawn as dashed cuts so that intrusive material is
visibly not part of the layer it sits in.

Every date is simulated once and then used three times: the pale outline is
the unmodelled calibrated density, the filled curve is that same density after
the stratigraphic order constraint, and the bar beneath it is the 95.4 % HPD
interval measured off the constrained curve.  All densities share a single
vertical scale, so a tight date really does look tighter than a broad one.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

from pubstyle import set_pub_style, savefig, darken
from figkit import axes_inches, halo_text, panel_labels_inches, shared_limits

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HAIR = "#cccccc"

# Shared archaeological period colours (house qualitative order).  The same
# period keeps the same colour in 631 and 632.
PERIOD_COLOR = {
    "Neolithic": "#8e6bbf",
    "Bronze Age": "#3b5b92",
    "Iron Age": "#2e7d5b",
    "Roman": "#c9a227",
    "Medieval": "#b23a48",
}

FW, FH = 10.4, 6.4
ZMAX = 2.60                      # trench depth below datum, m

# name, top (m), base (m), period, approximate soil face colour, hatch,
# short name for the cramped in-section caption
LAYERS = [
    ("Plough soil", 0.00, 0.30, None, "#a89877", "..", "Plough soil"),
    ("Medieval occupation", 0.30, 0.72, "Medieval", "#6d5334", "//",
     "Medieval occ."),
    ("Burnt destruction", 0.72, 0.92, "Medieval", "#5c4030", "xx",
     "Burnt destr."),
    ("Roman occupation", 0.92, 1.50, "Roman", "#8c7a52", "--", "Roman occ."),
    ("Iron Age midden", 1.50, 2.10, "Iron Age", "#6f6a4a", "\\\\",
     "Iron Age midden"),
    ("Natural subsoil", 2.10, 2.60, None, "#9aa1a8", "++", "Natural"),
]
# modelled age of the top and base contact of each layer, cal BP
LAYER_AGE = [(120.0, 560.0), (560.0, 790.0), (790.0, 870.0),
             (870.0, 1760.0), (1760.0, 2360.0), (2360.0, 2760.0)]

# depth (m), layer index, laboratory code, dated material
SAMPLES = [
    (0.42, 1, "OxA-4118", "charcoal"),
    (0.82, 2, "OxA-4120", "charred grain"),
    (1.20, 3, "OxA-4133", "animal bone"),
    (1.62, 4, "OxA-4147", "charcoal"),
    (2.00, 4, "OxA-4152", "charred grain"),
]

XGRID = np.linspace(200.0, 3000.0, 1400)
CURVE_H = 0.185                  # tallest density, in metres of depth


def cal_year(bp):
    """cal BP -> 'cal AD 1400' / 'cal BC 410' string."""
    y = 1950.0 - bp
    return f"cal AD {y:.0f}" if y > 0 else f"cal BC {-y:.0f}"


def span_label(bp_hi, bp_lo):
    """Age span written the way a report writes it, oldest first."""
    a, b = 1950.0 - bp_hi, 1950.0 - bp_lo
    if a <= 0 and b <= 0:
        return f"cal BC {-a:.0f}-{-b:.0f}"
    if a > 0 and b > 0:
        return f"cal AD {a:.0f}-{b:.0f}"
    return f"cal BC {-a:.0f} - cal AD {b:.0f}"


def calibrated_density(rng, true_bp):
    """Wiggly multi-modal calibrated density, the shape a plateau produces."""
    d = np.zeros_like(XGRID)
    n_mode = int(rng.integers(2, 4))
    for k in range(n_mode):
        mu = true_bp + rng.normal(0.0, 42.0)
        sd = rng.uniform(26.0, 62.0)
        w = rng.uniform(0.45, 1.0) if k else 1.0
        d += w * np.exp(-0.5 * ((XGRID - mu) / sd) ** 2)
    return d


def constrain(d, age_top, age_base, soft=16.0):
    """Apply the stratigraphic order prior as a soft window."""
    lo = 1.0 / (1.0 + np.exp(-(XGRID - age_top) / soft))
    hi = 1.0 / (1.0 + np.exp((XGRID - age_base) / soft))
    return d * lo * hi


def hpd_interval(x, d, mass=0.954):
    """Highest posterior density interval measured off the drawn curve."""
    dx = np.gradient(x)
    p = d / np.sum(d * dx)
    order = np.argsort(p)[::-1]
    csum = np.cumsum(p[order] * dx[order])
    k = int(np.searchsorted(csum, mass)) + 1
    sel = order[:min(k, order.size)]
    return float(x[sel].min()), float(x[sel].max())


def weighted_median(x, d):
    c = np.cumsum(d) / np.sum(d)
    return float(np.interp(0.5, c, x))


def build_dates(rng):
    out = []
    for depth, li, code, material in SAMPLES:
        top, base = LAYER_AGE[li]
        frac = (depth - LAYERS[li][1]) / (LAYERS[li][2] - LAYERS[li][1])
        true_bp = top + frac * (base - top) + rng.normal(0.0, 22.0)
        raw = calibrated_density(rng, true_bp)
        mod = constrain(raw, top, base)
        lo, hi = hpd_interval(XGRID, mod)
        out.append(dict(depth=depth, layer=li, code=code, material=material,
                        raw=raw, mod=mod, lo=lo, hi=hi,
                        med=weighted_median(XGRID, mod)))
    return out


def draw_section(ax, rng, dates):
    w = ZMAX * (1.95 / 4.10)                     # equal aspect fills the slot
    ax.set_xlim(0.0, w)
    ax.set_ylim(ZMAX, 0.0)
    ax.set_aspect("equal")

    for name, top, base, period, fc, hatch, short in LAYERS:
        ax.add_patch(Rectangle((0.0, top), w, base - top, facecolor=fc,
                               edgecolor=darken(fc, 0.42), linewidth=0.7,
                               hatch=hatch, zorder=2))
    # contacts drawn as slightly irregular lines, the way a section really cuts
    xs = np.linspace(0.0, w, 60)
    for _, top, base, *_ in LAYERS[1:]:
        wob = 0.012 * np.sin(xs * 9.0 + rng.uniform(0, 6.0))
        ax.plot(xs, top + wob, color=RULE, lw=0.8, zorder=4)

    # negative features, kept on the right half so the captions stay legible
    pit = np.array([[0.60, 0.30], [0.58, 0.62], [0.63, 1.02], [0.76, 1.24],
                    [0.90, 1.06], [0.94, 0.66], [0.92, 0.30]]) * [w, 1.0]
    ax.add_patch(Polygon(pit, closed=True, facecolor="#7f6a4f",
                         edgecolor="#2b2b2b", linewidth=1.0,
                         linestyle=(0, (3.5, 1.8)), hatch="oo", zorder=5))
    post = np.array([[0.62, 1.56], [0.615, 1.90], [0.68, 1.98], [0.735, 1.88],
                     [0.73, 1.56]]) * [w, 1.0]
    ax.add_patch(Polygon(post, closed=True, facecolor="#6a6152",
                         edgecolor="#2b2b2b", linewidth=1.0,
                         linestyle=(0, (3.5, 1.8)), zorder=5))
    halo_text(ax, 0.76 * w, 1.31, "F.12 pit", ha="center", va="top",
              fontsize=6.4, color="white", halo="#2b2b2b", zorder=8)
    halo_text(ax, 0.80 * w, 1.78, "F.7 post", ha="left", va="center",
              fontsize=6.4, color="white", halo="#2b2b2b", zorder=8)

    # layer name + modelled span, stacked inside each unit
    for (name, top, base, period, fc, hatch, short), (a_top, a_base) in zip(
            LAYERS, LAYER_AGE):
        ym = 0.5 * (top + base)
        halo_text(ax, 0.030 * w, ym - 0.045, short, ha="left", va="center",
                  fontsize=6.6, fontweight="bold", color="white",
                  halo="#1a1a1a", halo_lw=2.0, zorder=9)
        halo_text(ax, 0.030 * w, ym + 0.055, span_label(a_base, a_top),
                  ha="left", va="center", fontsize=6.0, color="#f0f0f0",
                  halo="#1a1a1a", halo_lw=1.8, zorder=9)

    # sample positions echoed on the section
    for d in dates:
        ax.plot([0.965 * w], [d["depth"]], marker="<", ms=4.2,
                mfc=PERIOD_COLOR[LAYERS[d["layer"]][3]], mec="white",
                mew=0.6, zorder=10, clip_on=False)

    ax.set_yticks(np.arange(0.0, ZMAX + 0.01, 0.5))
    ax.set_ylabel("Depth below site datum (m)", fontsize=8.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlabel("Section width (m)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)
    ax.set_title("Trench 4, east section", fontsize=8.5, fontweight="bold")


def draw_dates(ax, dates, scale):
    for d in dates:
        col = PERIOD_COLOR[LAYERS[d["layer"]][3]]
        z0 = d["depth"]
        ax.plot(XGRID, z0 - d["raw"] * scale, color="#9a9a9a", lw=0.7,
                zorder=3)
        ax.fill_between(XGRID, z0, z0 - d["mod"] * scale, color=col,
                        alpha=0.42, lw=0.0, zorder=4)
        ax.plot(XGRID, z0 - d["mod"] * scale, color=darken(col, 0.25),
                lw=1.0, zorder=5)
        ax.add_patch(Rectangle((d["lo"], z0), d["hi"] - d["lo"], 0.062,
                               facecolor=col, edgecolor=darken(col, 0.35),
                               linewidth=0.6, zorder=6))
        # white halo: neighbouring curves reach under these captions
        top = z0 - float(d["raw"].max()) * scale
        halo_text(ax, d["med"], top - 0.115, f"{d['code']}  {d['material']}",
                  ha="center", va="bottom", fontsize=6.2, color=INK,
                  halo="white", halo_lw=2.2, zorder=8)
        halo_text(ax, d["med"], top - 0.028,
                  f"{span_label(d['hi'], d['lo'])} (95.4%)", ha="center",
                  va="bottom", fontsize=6.0, color=SUB, halo="white",
                  halo_lw=2.2, zorder=8)

    for _, top, base, *_ in LAYERS[1:]:
        ax.axhline(top, color=HAIR, lw=0.55, ls=(0, (3, 2)), zorder=1)
    ax.set_ylim(ZMAX, 0.0)
    ax.set_yticks(np.arange(0.0, ZMAX + 0.01, 0.5))
    ax.tick_params(labelleft=False, labelsize=7.0)
    ax.set_xlabel("Calibrated age (cal BP)", fontsize=8.0)
    ax.set_title("Sequence-modelled radiocarbon dates", fontsize=8.5,
                 fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_model(ax, dates, rng):
    zb = np.array([0.0] + [b for _, _, b, *_ in LAYERS])
    ab = np.array([LAYER_AGE[0][0]] + [a for _, a in LAYER_AGE])
    sb = 24.0 + 0.035 * ab
    zz = np.linspace(0.0, ZMAX, 300)
    aa = np.interp(zz, zb, ab)
    ss = np.interp(zz, zb, sb)
    ax.fill_betweenx(zz, aa - 2.0 * ss, aa + 2.0 * ss, color="#3b5b92",
                     alpha=0.18, lw=0.0, zorder=2)
    ax.plot(aa, zz, color="#3b5b92", lw=1.3, zorder=4)
    for z, a in zip(zb, ab):
        ax.plot([a], [z], marker="_", ms=7.0, color="#3b5b92", mew=1.1,
                zorder=5)
    for d in dates:
        col = PERIOD_COLOR[LAYERS[d["layer"]][3]]
        ax.plot([d["lo"], d["hi"]], [d["depth"]] * 2, color=col, lw=1.6,
                solid_capstyle="butt", zorder=6)
        ax.plot([d["med"]], [d["depth"]], marker="o", ms=3.4, mfc="white",
                mec=darken(col, 0.25), mew=0.9, zorder=7)
    ax.set_ylim(ZMAX, 0.0)
    ax.set_yticks(np.arange(0.0, ZMAX + 0.01, 0.5))
    ax.tick_params(labelleft=False, labelsize=7.0)
    ax.set_xlabel("Modelled age (cal BP)", fontsize=8.0)
    ax.set_title("Age-depth model", fontsize=8.5, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    set_pub_style(**{"hatch.linewidth": 0.42})
    rng = np.random.default_rng(630)
    dates = build_dates(rng)
    scale = CURVE_H / max(float(d["raw"].max()) for d in dates)

    fig = plt.figure(figsize=(FW, FH))
    ax_s = axes_inches(fig, (0.62, 1.10, 1.95, 4.10))
    ax_d = axes_inches(fig, (3.19, 1.10, 4.20, 4.10))
    ax_m = axes_inches(fig, (7.94, 1.10, 2.32, 4.10))

    draw_section(ax_s, rng, dates)
    draw_dates(ax_d, dates, scale)
    draw_model(ax_m, dates, rng)

    # one age range for both age panels, covering every drawn curve
    support = [XGRID[d["raw"] > 0.01 * d["raw"].max()] for d in dates]
    support.append(np.array([a for p in LAYER_AGE for a in p]))
    lo, hi = shared_limits(support, pad=0.06)
    ax_d.set_xlim(lo, hi)
    ax_m.set_xlim(lo, hi)

    handles = [Rectangle((0, 0), 1, 1, facecolor=fc, hatch=h,
                         edgecolor=darken(fc, 0.42), linewidth=0.6,
                         label=n)
               for n, _, _, _, fc, h, _s in LAYERS]
    handles.append(Line2D([0], [0], color="#2b2b2b", lw=1.0,
                          ls=(0, (3.5, 1.8)), label="feature cut"))
    handles.append(Line2D([0], [0], color="#9a9a9a", lw=0.8,
                          label="unmodelled density"))
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.082), ncol=8, frameon=False,
               fontsize=6.8, handlelength=1.5, handleheight=1.1,
               columnspacing=1.25)

    panel_labels_inches(fig, [ax_s, ax_d, ax_m], dx=-0.30, dy=0.22)

    tight = min(dates, key=lambda d: d["hi"] - d["lo"])
    broad = max(dates, key=lambda d: d["hi"] - d["lo"])
    shrink = 1.0 - ((tight["hi"] - tight["lo"]) /
                    (hpd_interval(XGRID, tight["raw"])[1] -
                     hpd_interval(XGRID, tight["raw"])[0]))


    for d in dates:
        print(f"{d['code']} z={d['depth']:.2f} m  "
              f"{d['lo']:.0f}-{d['hi']:.0f} cal BP  "
              f"{span_label(d['hi'], d['lo'])}")
    savefig(fig, "630_arc_stratigraphy_dating.png")


if __name__ == "__main__":
    main()

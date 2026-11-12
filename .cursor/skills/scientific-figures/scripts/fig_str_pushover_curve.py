"""Static pushover of a seven-storey frame in three retrofit configurations.

The main panel is the base shear against roof displacement for each scheme,
laid over the performance bands the assessment is graded against: immediate
occupancy, life safety and collapse prevention, drawn at 0.5 / 1.5 / 2.5 %
roof drift.  The bare frame also carries its equal-energy bilinear
idealization and the order in which its plastic hinges open, so the reader can
see that stiffening the structure buys strength at the cost of the very
ductility the CP check depends on.

Nothing on the figure is typed in.  The yield point comes from the FEMA
equal-area construction solved as a quadratic against the integral of the
plotted curve, the peak is its arg-max, the ultimate point is the post-peak
crossing of 0.8 V_max, hinge displacements are multiples of the derived yield
displacement read back off the curve, and the drift profiles are normalised so
the storey drifts sum exactly to the roof displacement.  Synthetic backbones.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, despine
from figkit import axes_inches, panel_labels_inches, shared_limits

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
BASE = "#808080"

N_STOREY = 7
H_STOREY = 3000.0                 # mm
H_TOTAL = N_STOREY * H_STOREY     # mm
D_MAX = 700.0                     # mm, pushover target displacement
Y_TOP = 7600.0                    # kN; the band above holds the legend
BAND_TOP = 5700.0                 # kN, top of the shaded performance bands

# Performance levels as roof drift ratios (ASCE 41 style).
LEVELS = [("IO", 0.005, "#f5f5f5"), ("LS", 0.015, "#ececec"),
          ("CP", 0.025, "#e2e2e2")]

SCHEMES = [
    dict(name="bare moment frame", v_max=2600.0, d_ref=95.0, d_u=520.0,
         q=6.0, color=BLUE),
    dict(name="frame + buckling-restrained braces", v_max=4200.0, d_ref=55.0,
         d_u=430.0, q=5.0, color=GREEN),
    dict(name="frame + RC shear wall", v_max=5800.0, d_ref=38.0, d_u=330.0,
         q=4.5, color=GOLD),
]

# Hinge order for the bare frame, as multiples of the derived yield drift.
HINGES = [("beam, level 2", 0.62, "beam"), ("beam, level 3", 0.78, "beam"),
          ("beam, level 1", 0.96, "beam"), ("beam, level 4", 1.22, "beam"),
          ("beam, level 5", 1.62, "beam"), ("column base", 2.15, "column"),
          ("beam, level 6", 2.75, "beam"), ("column, level 2", 3.55, "column")]


def backbone(d, spec):
    """Hardening-then-softening pushover backbone, kN."""
    rise = 1.0 - np.exp(-d / spec["d_ref"])
    soften = 1.0 / (1.0 + (d / spec["d_u"]) ** spec["q"])
    return spec["v_max"] * rise * soften


def characterise(d, v):
    """Peak, 0.8 V_max ultimate point, and the equal-energy bilinear."""
    i_pk = int(np.argmax(v))
    d_pk, v_pk = float(d[i_pk]), float(v[i_pk])
    tail = slice(i_pk, None)
    d_u = float(np.interp(-0.8 * v_pk, -v[tail], d[tail]))
    # Secant stiffness at 0.6 V_max on the rising branch, kN/mm.
    d_60 = float(np.interp(0.6 * v_pk, v[:i_pk + 1], d[:i_pk + 1]))
    k_e = 0.6 * v_pk / d_60
    m = d <= d_u
    area = float(np.trapezoid(v[m], d[m]))                  # kN mm
    # Elastic-perfectly-plastic idealisation with the same area to d_u:
    # Vy^2 / (2 Ke) - Vy d_u + area = 0.
    disc = max((k_e * d_u) ** 2 - 2.0 * k_e * area, 0.0)
    v_y = k_e * d_u - np.sqrt(disc)
    d_y = v_y / k_e
    return dict(d_pk=d_pk, v_pk=v_pk, d_u=d_u, k_e=k_e, v_y=float(v_y),
                d_y=float(d_y), mu=float(d_u / d_y), area=area)


def drift_profile(roof, cp_roof):
    """Storey drift ratios (%) that sum exactly to the roof displacement."""
    lam = float(np.clip(roof / cp_roof, 0.0, 1.0))
    st = np.arange(N_STOREY)
    soft = np.exp(-((st - 1.4) ** 2) / 2.8)
    shape = (1.0 - lam) * np.ones(N_STOREY) + lam * (0.30 + 2.1 * soft)
    delta = shape / shape.sum() * roof                      # mm per storey
    return delta / H_STOREY * 100.0


def main_panel(ax, d, runs, prim, hinges):
    x_prev = 0.0
    for tag, ratio, col in LEVELS:
        x = ratio * H_TOTAL
        ax.axvspan(x_prev, x, ymin=0.0, ymax=BAND_TOP / Y_TOP, color=col,
                   lw=0, zorder=0)
        ax.text(0.5 * (x_prev + x), BAND_TOP * 0.965, tag, ha="center",
                va="top", fontsize=7.2, color=SUB, fontweight="bold",
                zorder=2)
        ax.text(0.5 * (x_prev + x), BAND_TOP * 0.915,
                f"{ratio * 100:.1f} % drift", ha="center", va="top",
                fontsize=6.2, color=MUTED, zorder=2)
        ax.axvline(x, color="#cccccc", lw=0.7, zorder=1)
        x_prev = x
    ax.text(0.5 * (x_prev + D_MAX), BAND_TOP * 0.965, "beyond CP",
            ha="center", va="top", fontsize=7.2, color=SUB,
            fontweight="bold", zorder=2)

    handles, labels = [], []
    for r in runs:
        ln, = ax.plot(d, r["v"], color=r["color"], lw=2.0, zorder=6)
        c = r["char"]
        ax.plot([c["d_pk"]], [c["v_pk"]], marker="o", ms=4.6, mfc="white",
                mec=r["color"], mew=1.3, zorder=8)
        ax.plot([c["d_u"]], [0.8 * c["v_pk"]], marker="v", ms=4.6,
                mfc="white", mec=r["color"], mew=1.3, zorder=8)
        handles.append(ln)
        labels.append(f"{r['name']}: $V_{{max}}$ {c['v_pk']:.0f} kN at "
                      f"{c['d_pk']:.0f} mm, $\\Delta_u$ {c['d_u']:.0f} mm")

    c = prim["char"]
    bl, = ax.plot([0.0, c["d_y"], c["d_u"]], [0.0, c["v_y"], c["v_y"]],
                  color=RED, lw=1.4, ls=(0, (5.0, 2.4)), zorder=7)
    ax.plot([c["d_y"], c["d_u"]], [c["v_y"], c["v_y"]], marker="s", ms=3.8,
            mfc="white", mec=RED, mew=1.1, ls="none", zorder=9)
    ax.annotate(f"equal-energy bilinear\n$V_y$ = {c['v_y']:.0f} kN, "
                f"$\\Delta_y$ = {c['d_y']:.0f} mm\n"
                f"$K_e$ = {c['k_e']:.1f} kN/mm",
                xy=(c["d_y"], c["v_y"]), xytext=(c["d_y"] + 88.0,
                                                 c["v_y"] * 0.44),
                ha="left", va="center", fontsize=6.5, color=RED,
                linespacing=1.4, zorder=10,
                bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                          edgecolor="#cccccc", lw=0.6, alpha=0.94),
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7,
                                shrinkA=3.0, shrinkB=3.0))

    hd = np.array([h["d"] for h in hinges])
    hv = np.array([h["v"] for h in hinges])
    hm, = ax.plot(hd, hv, marker="D", ms=3.6, mfc=prim["color"], mec="white",
                  mew=0.8, ls="none", zorder=10)
    for k in (0, 5, len(hinges) - 1):
        h = hinges[k]
        ax.annotate(f"{k + 1}. {h['label']}", xy=(h["d"], h["v"]),
                    xytext=(h["d"] + 26.0, h["v"] - 620.0 - 190.0 * (k == 0)),
                    ha="left", va="center", fontsize=6.3, color=prim["color"],
                    zorder=10,
                    arrowprops=dict(arrowstyle="-", color=prim["color"],
                                    lw=0.6, shrinkA=1.0, shrinkB=2.5))

    ax.set_xlim(0.0, D_MAX)
    ax.set_ylim(0.0, Y_TOP)
    ax.set_yticks(np.arange(0.0, 6001.0, 1000.0))
    ax.set_xlabel("roof displacement (mm)", fontsize=8.0)
    ax.set_ylabel("base shear (kN)", fontsize=8.0)
    ax.tick_params(labelsize=7.2)
    despine(ax)
    ax.set_axisbelow(True)
    ax.set_title("Stiffer schemes gain strength and give up displacement "
                 "capacity", fontsize=9.0, fontweight="bold", color=INK)

    ax2 = ax.secondary_xaxis(
        "top", functions=(lambda x: x / H_TOTAL * 100.0,
                          lambda r: r * H_TOTAL / 100.0))
    ax2.set_xlabel("roof drift ratio (%)", fontsize=7.4, labelpad=2.0)
    ax2.tick_params(labelsize=6.8)

    handles += [bl, hm,
                plt.Line2D([], [], marker="o", ls="none", ms=3.8, mfc="white",
                           mec=RULE, mew=1.1),
                plt.Line2D([], [], marker="v", ls="none", ms=3.8, mfc="white",
                           mec=RULE, mew=1.1)]
    labels += ["equal-energy bilinear (bare frame)",
               "plastic hinge formation", "peak strength",
               "ultimate point (0.8 $V_{max}$)"]
    leg = ax.legend(handles, labels, fontsize=6.6, frameon=False, ncol=2,
                    loc="upper left", bbox_to_anchor=(0.0, 1.0),
                    handlelength=1.9, columnspacing=1.4, labelspacing=0.34)
    leg.set_zorder(11)


def drift_panel(ax, prim):
    cp_roof = LEVELS[-1][1] * H_TOTAL
    shades = ["#9cc3dd", "#2e74a8", "#08263f"]
    profiles = []
    for (tag, ratio, _), col in zip(LEVELS, shades):
        roof = ratio * H_TOTAL
        pr = drift_profile(roof, cp_roof)
        profiles.append(pr)
        ax.step(np.append(pr, pr[-1]), np.arange(1, N_STOREY + 2) - 0.5,
                where="pre", color=col, lw=1.6, zorder=5,
                label=f"{tag}, roof {ratio * 100:.1f} %")
        ax.plot([pr.max()], [np.argmax(pr) + 1], marker="o", ms=3.2,
                mfc="white", mec=col, mew=1.0, zorder=6)
    lo, hi = shared_limits(profiles, pad=0.08, lo=0.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0.5, N_STOREY + 0.5)
    ax.set_yticks(np.arange(1, N_STOREY + 1))
    ax.set_xlabel("interstorey drift ratio (%)", fontsize=7.6)
    ax.set_ylabel("storey", fontsize=7.6)
    ax.tick_params(labelsize=6.8)
    despine(ax)
    ax.grid(axis="x", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.4, frameon=False, loc="upper right",
              handlelength=1.6, labelspacing=0.3)
    peak = int(np.argmax(profiles[-1])) + 1
    ax.set_title(f"Drift migrates into storey {peak} as the frame softens",
                 fontsize=8.4, fontweight="bold", color=INK)
    return profiles


def hinge_panel(ax, hinges):
    y = np.arange(len(hinges))
    cols = {"beam": BLUE, "column": RED}
    for tag, ratio, _ in LEVELS:
        ax.axvline(ratio * H_TOTAL, color="#cccccc", lw=0.7, zorder=1)
        ax.text(ratio * H_TOTAL, 0.972, tag, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6.2, color=MUTED, zorder=2)
    for i, h in enumerate(hinges):
        ax.plot([0.0, h["d"]], [i, i], color=cols[h["kind"]], lw=0.9,
                alpha=0.55, zorder=4)
        ax.plot([h["d"]], [i], marker="o", ms=4.2, mfc=cols[h["kind"]],
                mec="white", mew=0.8, zorder=6)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{i + 1}. {h['label']}" for i, h in enumerate(hinges)],
                       fontsize=6.3)
    ax.invert_yaxis()
    ax.set_xlim(0.0, D_MAX)
    ax.set_xlabel("roof displacement at hinge formation (mm)", fontsize=7.6)
    ax.tick_params(axis="x", labelsize=6.8)
    despine(ax)
    ax.grid(axis="x", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend([plt.Line2D([], [], marker="o", ls="none", ms=4.0,
                          mfc=cols[k], mec="white", mew=0.8)
               for k in ("beam", "column")], ["beam hinge", "column hinge"],
              fontsize=6.4, frameon=False, loc="lower right",
              handlelength=1.0, labelspacing=0.3)
    ax.set_title("Beams hinge first, columns last", fontsize=8.4,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    rng = np.random.default_rng(85504)
    d = np.linspace(0.0, D_MAX, 1401)

    runs = []
    for s in SCHEMES:
        spec = dict(s, v_max=s["v_max"] * (1.0 + 0.015 * rng.normal()))
        v = backbone(d, spec)
        runs.append(dict(spec, v=v, char=characterise(d, v)))

    prim = runs[0]
    hinges = []
    for label, mult, kind in HINGES:
        dh = mult * prim["char"]["d_y"]
        hinges.append(dict(label=label, kind=kind, d=dh,
                           v=float(np.interp(dh, d, prim["v"]))))

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (0.90, 0.90, 4.90, 4.34))
    ax_drift = axes_inches(fig, (7.34, 3.42, 2.66, 1.94))
    ax_hinge = axes_inches(fig, (7.34, 0.90, 2.66, 1.94))

    main_panel(ax_main, d, runs, prim, hinges)
    profiles = drift_panel(ax_drift, prim)
    hinge_panel(ax_hinge, hinges)

    c = prim["char"]
    strong = runs[-1]["char"]
    panel_labels_inches(fig, [ax_main, ax_drift, ax_hinge], dx=-0.56, dy=0.30)
    savefig(fig, "855_str_pushover_curve.png")


if __name__ == "__main__":
    main()

"""Fibre stress-strain curves with break, yield and initial-modulus markers.

Each fibre is a single generated curve; every number on the page is measured
back off that curve - the initial modulus by a least-squares fit over the first
0.5 % of strain, the yield point by the first place the tangent modulus falls
below 35 % of that value, the work of rupture by integrating the curve, and the
scatter panel by reading the last point of each array.  Elastane is drawn in
the main panel too, where it is a flat line, and again in an inset on its own
strain scale, because forcing 520 % onto the same axis would flatten the four
technical fibres into the y axis.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
HAIR = "#cccccc"

FIBRES = [
    dict(name="Cotton", color="#c9a227", T=3.10, eb=0.075, n=2.6, h=0.15,
         m=3.0, kind="yield"),
    dict(name="Polyester PET", color="#3b5b92", T=5.60, eb=0.240, n=4.0,
         h=0.30, m=3.4, kind="yield"),
    dict(name="Nylon 6.6", color="#2e7d5b", T=6.80, eb=0.300, n=3.2, h=0.34,
         m=3.6, kind="yield"),
    dict(name="Aramid", color="#b23a48", T=20.50, eb=0.036, n=1.0, h=0.0,
         m=2.0, kind="linear"),
    dict(name="Elastane", color="#8e6bbf", T=0.85, eb=5.200, n=1.6, h=0.30,
         m=5.5, kind="j"),
]

XMAX = 33.0


def build(f, rng, npts=520):
    """Strain (%) and specific stress (cN/dtex) for one fibre."""
    e = np.linspace(0.0, f["eb"], npts)
    x = e / f["eb"]
    if f["kind"] == "j":
        shape = (1.0 - f["h"]) * x ** f["n"] + f["h"] * x ** f["m"]
    else:
        shape = ((1.0 - f["h"]) * (1.0 - (1.0 - x) ** f["n"]) +
                 f["h"] * x ** f["m"])
    s = f["T"] * shape
    ripple = np.convolve(rng.standard_normal(npts), np.ones(21) / 21.0,
                         mode="same")
    s = s + f["T"] * 0.0035 * ripple * x
    return e * 100.0, np.maximum(s, 0.0)


def initial_modulus(strain_pct, stress, upto=0.5):
    m = strain_pct <= upto
    if m.sum() < 3:
        m = np.arange(strain_pct.size) < 3
    return float(np.polyfit(strain_pct[m] / 100.0, stress[m], 1)[0])


def yield_point(strain_pct, stress, e0, frac=0.35):
    sm = np.convolve(stress, np.ones(15) / 15.0, mode="same")
    tan = np.gradient(sm, strain_pct / 100.0)
    ok = (strain_pct > 0.6) & (strain_pct < strain_pct[-1] * 0.9)
    hit = np.flatnonzero(ok & (tan < frac * e0))
    if hit.size == 0:
        return None
    i = int(hit[0])
    return strain_pct[i], stress[i]


def main():
    set_pub_style()
    rng = np.random.default_rng(835)

    data = {}
    for f in FIBRES:
        x, y = build(f, rng)
        e0 = initial_modulus(x, y)
        data[f["name"]] = dict(
            x=x, y=y, e0=e0, color=f["color"],
            yld=yield_point(x, y, e0) if f["kind"] == "yield" else None,
            tough=float(np.trapezoid(y, x / 100.0)),
            eb=float(x[-1]), tb=float(y[-1]))

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.80, 1.02, 5.58, 4.28))
    ax_sc = axes_inches(fig, (7.46, 3.30, 2.54, 1.88))
    ax_tg = axes_inches(fig, (7.46, 0.98, 2.54, 1.70))

    ymax = max(d["tb"] for d in data.values()) * 1.14
    for f in FIBRES:
        d = data[f["name"]]
        x, y = d["x"], d["y"]
        keep = x <= XMAX
        ax.plot(x[keep], y[keep], color=d["color"], lw=1.6, zorder=4,
                solid_capstyle="round",
                label="%s   %.1f cN/dtex at %.1f %%"
                      % (f["name"], d["tb"], d["eb"]))
        if d["eb"] <= XMAX:
            ax.plot(d["eb"], d["tb"], marker="x", ms=6.5, mew=1.7,
                    color=d["color"], zorder=6, clip_on=False)
        if d["yld"] is not None:
            ax.plot(d["yld"][0], d["yld"][1], marker="o", ms=4.4, mfc="white",
                    mec=d["color"], mew=1.2, zorder=6)
        xt = np.array([0.0, min(XMAX, 1.05 * d["tb"] / d["e0"] * 100.0)])
        ax.plot(xt, d["e0"] * xt / 100.0, color=d["color"], lw=0.8,
                ls=(0, (1.4, 1.6)), alpha=0.75, zorder=3)

    d_el = data["Elastane"]
    ax.annotate("Elastane keeps going to %.0f %% strain" % d_el["eb"],
                xy=(XMAX * 0.985, 0.35), xytext=(XMAX * 0.60, 3.1),
                fontsize=6.6, color=d_el["color"], ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-|>", color=d_el["color"],
                                lw=0.9, shrinkA=2, shrinkB=1))
    d_pet = data["Polyester PET"]
    if d_pet["yld"] is not None:
        ax.annotate("yield point", xy=d_pet["yld"],
                    xytext=(d_pet["yld"][0] + 4.2, d_pet["yld"][1] - 2.3),
                    fontsize=6.6, color=SUB,
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7))
    d_ar = data["Aramid"]
    ax.text(d_ar["eb"] + 0.9, d_ar["tb"] * 0.72,
            "initial modulus\n%.0f cN/dtex" % d_ar["e0"], fontsize=6.6,
            color=d_ar["color"], ha="left", va="center")
    ax.text(XMAX * 0.985, ymax * 0.985,
            "dotted lines: initial modulus tangent (0-0.5 % strain)\n"
            "crosses: break point      open circles: yield point",
            ha="right", va="top", fontsize=6.5, color=MUTED,
            linespacing=1.5)

    ax.set_xlim(0.0, XMAX)
    ax.set_ylim(0.0, ymax)
    ax.set_xlabel("strain (%)", fontsize=8.0)
    ax.set_ylabel("specific stress (cN/dtex)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.legend(fontsize=6.8, frameon=False, loc="upper left",
              bbox_to_anchor=(0.175, 0.985), handlelength=1.6,
              labelspacing=0.34)
    ax.set_title("Stress-strain response of five fibre classes",
                 fontsize=9.0, fontweight="bold", color=INK)

    axi = ax.inset_axes([0.455, 0.335, 0.30, 0.29])
    axi.plot(d_el["x"], d_el["y"], color=d_el["color"], lw=1.4)
    axi.plot(d_el["eb"], d_el["tb"], marker="x", ms=5.5, mew=1.6,
             color=d_el["color"], clip_on=False)
    axi.add_patch(Rectangle((0.0, 0.0), XMAX, d_el["tb"] * 1.1,
                            facecolor="#f0f0f0", edgecolor="none", zorder=1))
    axi.set_xlim(0.0, d_el["eb"] * 1.05)
    axi.set_ylim(0.0, d_el["tb"] * 1.15)
    axi.set_xlabel("strain (%)", fontsize=6.2, labelpad=1.5)
    axi.set_ylabel("cN/dtex", fontsize=6.2, labelpad=1.5)
    axi.tick_params(labelsize=6.0, length=2.0)
    axi.set_title("Elastane, own strain scale", fontsize=6.8, color=INK,
                  pad=2.5)
    for s in ("top", "right"):
        axi.spines[s].set_visible(False)
    axi.text(XMAX * 1.4, d_el["tb"] * 0.16, "grey band =\nmain panel range",
             fontsize=6.0, color=MUTED, va="center", linespacing=1.4)

    ebs = np.array([data[f["name"]]["eb"] for f in FIBRES])
    tbs = np.array([data[f["name"]]["tb"] for f in FIBRES])
    b, a = np.polyfit(np.log10(ebs), np.log10(tbs), 1)
    xg = np.logspace(np.log10(ebs.min() * 0.7), np.log10(ebs.max() * 1.4), 60)
    ax_sc.plot(xg, 10.0 ** (a + b * np.log10(xg)), color=GREY, lw=0.9,
               ls=(0, (3.0, 2.0)), zorder=2,
               label="power law, slope %.2f" % b)
    for f in FIBRES:
        d = data[f["name"]]
        ax_sc.plot(d["eb"], d["tb"], marker="o", ms=5.2, color=d["color"],
                   mec="white", mew=0.7, zorder=4)
        dy = 1.35 if f["name"] != "Aramid" else 0.72
        ax_sc.annotate(f["name"], (d["eb"], d["tb"]),
                       textcoords="offset points", xytext=(0.0, 6.0 * dy),
                       ha="center", va="bottom", fontsize=6.3,
                       color=d["color"])
    ax_sc.set_xscale("log")
    ax_sc.set_yscale("log")
    ax_sc.set_xlim(2.2, 900.0)
    ax_sc.set_ylim(0.45, 46.0)
    ax_sc.set_xlabel("breaking elongation (%)", fontsize=7.4)
    ax_sc.set_ylabel("tenacity (cN/dtex)", fontsize=7.4)
    ax_sc.tick_params(labelsize=6.5)
    ax_sc.legend(fontsize=6.2, frameon=False, loc="lower left",
                 handlelength=1.6)
    ax_sc.set_title("Strong fibres do not stretch", fontsize=8.5,
                    fontweight="bold", color=INK)

    order = sorted(FIBRES, key=lambda f: data[f["name"]]["tough"])
    pos = np.arange(len(order), dtype=float)
    vals = [data[f["name"]]["tough"] for f in order]
    ax_tg.barh(pos, vals, height=0.62,
               color=[data[f["name"]]["color"] for f in order],
               edgecolor="white", linewidth=0.5, zorder=3)
    for p, v in zip(pos, vals):
        ax_tg.text(v + max(vals) * 0.025, p, "%.2f" % v, va="center",
                   ha="left", fontsize=6.3, color=SUB)
    ax_tg.set_yticks(pos)
    ax_tg.set_yticklabels([f["name"] for f in order], fontsize=6.5)
    ax_tg.set_xlim(0.0, max(vals) * 1.24)
    ax_tg.set_xlabel("work of rupture (cN/dtex)", fontsize=7.4)
    ax_tg.tick_params(axis="x", labelsize=6.5)
    ax_tg.set_title("Toughness is area, not strength", fontsize=8.5,
                    fontweight="bold", color=INK)


    panel_labels_inches(fig, [ax], labels=["(a)"], dx=-0.58, dy=0.16)
    panel_labels_inches(fig, [ax_sc], labels=["(b)"], dx=-0.52, dy=0.14)
    panel_labels_inches(fig, [ax_tg], labels=["(c)"], dx=-0.52, dy=0.14)
    savefig(fig, "835_txt_fiber_tensile.png")


if __name__ == "__main__":
    main()

"""Seismic fragility of a code-designed RC frame, read at the design intensity.

The main panel is the usual family of lognormal fragility curves - one per
damage state, each the probability of exceeding that state given the spectral
acceleration on the horizontal axis.  A vertical rule marks the design-basis
intensity and the four exceedance probabilities are read off there, which is
the number an assessment actually reports.  The subordinate panels convert the
same four curves into the probability of *being in* each state, which must sum
to one at every intensity, and compare median capacities across structure
types.

The curves are Phi(ln(Sa / theta) / beta) evaluated on the plotted grid; the
band probabilities are differences of those curves, so their sum is one by
construction and the script checks it.  It also checks that the four curves
never cross, since a moderate state that is less likely than an extensive one
would be nonsense.  Illustrative parameters, not a real assessment.
"""
import textwrap

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, despine
from figkit import axes_inches, panel_labels_inches, halo_text

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
BASE = "#808080"

SA_MAX = 2.0            # g
DBE = 0.40              # g, design-basis earthquake
MCE = 0.60              # g, maximum considered earthquake

# Damage states, light to dark.  Dispersions grow slightly with the state,
# which is realistic and still leaves the curves far from crossing.
STATES = [
    dict(name="slight", theta=0.18, beta=0.42, color="#f4c088"),
    dict(name="moderate", theta=0.33, beta=0.45, color="#e69a54"),
    dict(name="extensive", theta=0.58, beta=0.48, color="#a04c18"),
    dict(name="complete", theta=0.95, beta=0.52, color="#66300c"),
]
NONE_COLOR = "#a4d7b8"

# Median "extensive" capacity of other typologies, relative to this frame.
TYPOLOGIES = [
    ("unreinforced masonry", 0.34),
    ("RC frame, pre-code", 0.62),
    ("RC frame, code (this study)", 1.00),
    ("steel moment frame", 1.24),
    ("RC shear wall", 1.42),
]


def fragility(sa, theta, beta):
    return norm.cdf(np.log(np.maximum(sa, 1e-9) / theta) / beta)


def band_probabilities(curves):
    """P(in state) from P(exceed state); rows sum to one by construction."""
    bands = [1.0 - curves[0]]
    for a, b in zip(curves[:-1], curves[1:]):
        bands.append(a - b)
    bands.append(curves[-1])
    return np.array(bands)


def main_panel(ax, sa, curves):
    ax.axvspan(DBE, MCE, color="#f5f5f5", zorder=0)
    for x, tag, side in ((DBE, "DBE", "right"), (MCE, "MCE", "left")):
        ax.axvline(x, color=RED if x == DBE else BASE, lw=1.1,
                   ls=(0, (4.5, 2.2)), zorder=4)
        off = -0.015 if side == "right" else 0.015
        ax.text(x + off, 1.012, f"{tag} {x:.2f} g", ha=side, va="bottom",
                transform=ax.get_xaxis_transform(), fontsize=6.6,
                color=RED if x == DBE else SUB)

    handles, labels, reads = [], [], []
    for s, cv in zip(STATES, curves):
        ln, = ax.plot(sa, cv, color=s["color"], lw=2.0, zorder=6)
        p = float(np.interp(DBE, sa, cv))
        reads.append(p)
        ax.plot([DBE], [p], marker="o", ms=4.6, mfc="white", mec=s["color"],
                mew=1.4, zorder=8)
        ax.plot([s["theta"]], [0.5], marker="|", ms=7.0, color=s["color"],
                mew=1.4, zorder=8)
        handles.append(ln)
        labels.append(f"{s['name']}: $\\theta$ = {s['theta']:.2f} g, "
                      f"$\\beta$ = {s['beta']:.2f}")

    ax.axhline(0.5, color="#cccccc", lw=0.7, zorder=1)
    ax.text(SA_MAX - 0.02, 0.512, "median capacity $\\theta$", ha="right",
            va="bottom", fontsize=6.4, color=MUTED)

    read_txt = "\n".join(
        f"{s['name']:<10s} {p:5.3f}" for s, p in zip(STATES, reads))
    ax.text(0.975, 0.235, f"exceedance at DBE = {DBE:.2f} g\n{read_txt}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            color=INK, linespacing=1.45, family="DejaVu Sans Mono", zorder=9,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor="#cccccc", lw=0.6, alpha=0.95))

    ax.set_xlim(0.0, SA_MAX)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel("spectral acceleration $S_a(T_1)$ (g)", fontsize=8.0)
    ax.set_ylabel("probability of exceedance (-)", fontsize=8.0)
    ax.tick_params(labelsize=7.2)
    despine(ax)
    ax.grid(axis="y", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_title("Moderate damage is more likely than not at the design "
                 "intensity", fontsize=9.0, fontweight="bold", color=INK,
                 pad=15.0)
    leg = ax.legend(handles, labels, fontsize=6.7, frameon=False,
                    loc="lower right", bbox_to_anchor=(1.0, 0.0),
                    handlelength=1.9, labelspacing=0.34)
    leg.set_zorder(10)
    return reads


def stack_panel(ax, sa, bands):
    names = ["none"] + [s["name"] for s in STATES]
    cols = [NONE_COLOR] + [s["color"] for s in STATES]
    ax.stackplot(sa, bands, colors=cols, edgecolor="white", lw=0.4, zorder=3)
    edges = np.cumsum(bands, axis=0)
    lower = np.vstack([np.zeros_like(sa), edges[:-1]])
    inner = slice(int(0.08 * sa.size), int(0.92 * sa.size))
    for k, (name, lo, hi) in enumerate(zip(names, lower, edges)):
        thick = hi - lo
        i = inner.start + int(np.argmax(thick[inner]))
        if thick[i] < 0.10:
            continue
        # Every band is a diagonal wedge, so their thickest points all sit
        # near mid-height; riding high and low alternately keeps neighbouring
        # labels from touching without pushing any of them out of its band.
        frac = 0.30 if k % 2 == 0 else 0.70
        halo_text(ax, sa[i], lo[i] + frac * thick[i], name, ha="center",
                  va="center", fontsize=6.4, color=INK, halo="white",
                  halo_lw=2.0, zorder=7)
    ax.axvline(DBE, color=RED, lw=1.0, ls=(0, (4, 2.4)), zorder=6)
    ax.set_xlim(0.0, SA_MAX)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.25))
    ax.set_xlabel("spectral acceleration $S_a(T_1)$ (g)", fontsize=7.6)
    ax.set_ylabel("probability of being in state (-)", fontsize=7.6)
    ax.tick_params(labelsize=6.8)
    despine(ax)
    ax.set_title("The five states always account for 100 %", fontsize=8.4,
                 fontweight="bold", color=INK)


def typology_panel(ax, ref_theta):
    names = [n for n, _ in TYPOLOGIES]
    vals = np.array([ref_theta * f for _, f in TYPOLOGIES])
    order = np.argsort(vals)
    names = [names[i] for i in order]
    vals = vals[order]
    y = np.arange(vals.size)
    cols = [RED if "this study" in n else BLUE for n in names]
    ax.barh(y, vals, height=0.62, color=cols, edgecolor="white", lw=0.7,
            zorder=3)
    # One aligned column of values, so none of them lands on the DBE rule.
    for yi, v in zip(y, vals):
        ax.text(vals.max() * 1.045, yi, f"{v:.2f}", ha="left", va="center",
                fontsize=6.4, color=INK)
    ax.axvline(DBE, color=RED, lw=1.0, ls=(0, (4, 2.4)), zorder=5)
    ax.text(DBE + 0.014, 0.028, f"DBE {DBE:.2f} g",
            transform=ax.get_xaxis_transform(), ha="left", va="bottom",
            fontsize=6.3, color=RED)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.4)
    ax.set_xlim(0.0, vals.max() * 1.20)
    ax.set_xlabel("median capacity for extensive damage (g)", fontsize=7.6)
    ax.tick_params(axis="x", labelsize=6.8)
    despine(ax)
    ax.grid(axis="x", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_title("Only masonry is fragile below the DBE", fontsize=8.4,
                 fontweight="bold", color=INK)
    return names, vals


def main():
    set_pub_style()
    rng = np.random.default_rng(85605)
    sa = np.linspace(0.0, SA_MAX, 1601)

    # A small perturbation on each median, so no printed number is hand-typed.
    for s in STATES:
        s["theta"] = s["theta"] * (1.0 + 0.02 * rng.normal())
    curves = [fragility(sa, s["theta"], s["beta"]) for s in STATES]
    stack = np.array(curves)

    gaps = np.diff(stack, axis=0)
    assert gaps.max() <= 1e-12, "fragility curves cross"
    bands = band_probabilities(curves)
    unity = float(np.abs(bands.sum(axis=0) - 1.0).max())
    assert unity < 1e-12, "damage-state probabilities do not sum to one"

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (0.90, 0.90, 4.94, 4.34))
    ax_stack = axes_inches(fig, (7.30, 3.42, 2.70, 1.94))
    ax_typ = axes_inches(fig, (7.30, 0.90, 2.70, 1.94))

    reads = main_panel(ax_main, sa, curves)
    stack_panel(ax_stack, sa, bands)
    names, vals = typology_panel(ax_typ, STATES[2]["theta"])

    i_dbe = int(np.argmin(np.abs(sa - DBE)))
    modal = ["none"] + [s["name"] for s in STATES]
    modal = modal[int(np.argmax(bands[:, i_dbe]))]
    panel_labels_inches(fig, [ax_main, ax_stack, ax_typ], dx=-0.56, dy=0.24)
    savefig(fig, "856_str_fragility_curves.png")


if __name__ == "__main__":
    main()

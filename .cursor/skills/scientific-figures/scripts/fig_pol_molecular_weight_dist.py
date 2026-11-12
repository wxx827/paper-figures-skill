"""Molecular weight distributions from GPC traces, with moments and PDI.

Four distributions are built on one log M grid, and every reported quantity is
a moment of the plotted array: Mn from the reciprocal-weighted sum, Mw from the
weight-weighted sum, Mz from the next moment up, and the polydispersity index
from their ratio.  The Cauchy-Schwarz inequality guarantees Mw >= Mn for any
non-negative distribution, and the script asserts it before drawing.  Curves
are scaled to unit peak height for display only; the moments always come from
the area-normalised weights.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
RED = "#b23a48"

LOGM = np.linspace(3.0, 7.4, 1100)

SAMPLES = [
    dict(name="Anionic PS standard", color="#3b5b92", lw=1.5,
         modes=[(5.00, 0.09, 1.0)]),
    dict(name="Polypropylene", color="#2e7d5b", lw=1.5,
         modes=[(4.90, 0.50, 1.0)]),
    dict(name="LDPE, autoclave", color="#c9a227", lw=1.5,
         modes=[(4.70, 0.62, 1.0)]),
    dict(name="Bimodal HDPE", color=RED, lw=2.2,
         modes=[(4.35, 0.26, 0.60), (5.60, 0.30, 0.40)]),
]


def distribution(modes, rng):
    w = np.zeros_like(LOGM)
    for mu, s, a in modes:
        w += a * np.exp(-0.5 * ((LOGM - mu) / s) ** 2) / s
    w *= 1.0 + 0.006 * np.convolve(rng.standard_normal(LOGM.size),
                                   np.ones(31) / 31.0, mode="same")
    w = np.maximum(w, 0.0)
    return w


def moments(w):
    """Mn, Mw, Mz and PDI from a dW/dlogM array on the LOGM grid."""
    m = 10.0 ** LOGM
    wi = w / np.trapezoid(w, LOGM)
    dx = LOGM
    s0 = np.trapezoid(wi, dx)
    mn = s0 / np.trapezoid(wi / m, dx)
    mw = np.trapezoid(wi * m, dx) / s0
    mz = np.trapezoid(wi * m * m, dx) / np.trapezoid(wi * m, dx)
    return mn, mw, mz, mw / mn


def main():
    set_pub_style()
    rng = np.random.default_rng(837)

    data = []
    for s in SAMPLES:
        w = distribution(s["modes"], rng)
        mn, mw, mz, pdi = moments(w)
        assert mw >= mn and pdi >= 1.0, "Mw must not fall below Mn"
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) *
                                               np.diff(LOGM))])
        cum = 100.0 * cum / cum[-1]
        data.append(dict(w=w / w.max(), cum=cum, mn=mn, mw=mw, mz=mz,
                         pdi=pdi, **s))

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.82, 1.02, 5.56, 4.28))
    ax_cu = axes_inches(fig, (7.50, 3.30, 2.50, 1.88))
    ax_pd = axes_inches(fig, (7.50, 0.98, 2.50, 1.70))

    m_axis = 10.0 ** LOGM
    hi = data[-1]
    ax.fill_between(m_axis, 0.0, hi["w"], color=RED, alpha=0.10, zorder=2)
    for d in data:
        ax.plot(m_axis, d["w"], color=d["color"], lw=d["lw"], zorder=4,
                label="%s   PDI = %.2f" % (d["name"], d["pdi"]))

    for key, lab in [("mn", "$M_n$"), ("mw", "$M_w$"), ("mz", "$M_z$")]:
        ax.axvline(hi[key], color=RED, lw=0.9, ls=(0, (3.0, 2.0)), alpha=0.85,
                   zorder=5, ymax=0.78)
        ax.text(hi[key] * 1.08, 0.30, "%s = %s" % (lab, fmt_m(hi[key])),
                rotation=90, ha="left", va="bottom", fontsize=6.6, color=RED,
                zorder=7, transform=ax.get_xaxis_transform(),
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82,
                          pad=1.0))
    ax.text(0.985, 0.985, "dashed: moments of the bimodal sample",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.6,
            color=RED)

    ax.set_xscale("log")
    ax.set_xlim(10.0 ** 3.15, 10.0 ** 7.35)
    ax.set_ylim(0.0, 1.16)
    ax.set_xlabel("molar mass $M$ (g/mol)", fontsize=8.0)
    ax.set_ylabel("normalised d$W$/dlog$M$ (-)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.legend(fontsize=7.0, frameon=False, loc="upper left",
              bbox_to_anchor=(0.012, 0.99), handlelength=1.7,
              labelspacing=0.34)
    ax.set_title("GPC molar mass distributions on a decadic log axis",
                 fontsize=9.0, fontweight="bold", color=INK)

    for d in data:
        ax_cu.plot(m_axis, d["cum"], color=d["color"], lw=d["lw"] * 0.8,
                   zorder=4)
    for q in (10.0, 50.0, 90.0):
        ax_cu.axhline(q, color=GREY, lw=0.7, ls=(0, (2.6, 2.0)), zorder=2)
        ax_cu.text(10.0 ** 3.22, q + 2.0, "D%.0f" % q, fontsize=6.2,
                   color=MUTED, va="bottom", ha="left")
        for d in data:
            mq = 10.0 ** float(np.interp(q, d["cum"], LOGM))
            ax_cu.plot(mq, q, marker="o", ms=3.2, color=d["color"],
                       mec="white", mew=0.5, zorder=5)
    d50 = 10.0 ** float(np.interp(50.0, hi["cum"], LOGM))
    ax_cu.annotate("D50 = %s g/mol" % fmt_m(d50), (d50, 50.0),
                   textcoords="offset points", xytext=(6.0, -13.0),
                   fontsize=6.3, color=RED)
    ax_cu.set_xscale("log")
    ax_cu.set_xlim(10.0 ** 3.15, 10.0 ** 7.35)
    ax_cu.set_ylim(-3.0, 108.0)
    ax_cu.set_xlabel("molar mass $M$ (g/mol)", fontsize=7.4)
    ax_cu.set_ylabel("cumulative weight (%)", fontsize=7.4)
    ax_cu.tick_params(labelsize=6.5)
    ax_cu.set_title("Cumulative distribution and decile points",
                    fontsize=8.5, fontweight="bold", color=INK)

    pos = np.arange(len(data), dtype=float)
    ax_pd.bar(pos, [d["pdi"] for d in data], width=0.62,
              color=[d["color"] for d in data], edgecolor="white",
              linewidth=0.5, zorder=3)
    for p, d in zip(pos, data):
        ax_pd.text(p, d["pdi"] + 0.22, "%.2f" % d["pdi"], ha="center",
                   va="bottom", fontsize=6.4, color=SUB)
    ax_pd.axhline(1.0, color=GREY, lw=0.8, ls=(0, (3.0, 2.0)), zorder=2)
    ax_pd.text(0.02, 0.97, "PDI = 1 is the monodisperse limit",
               transform=ax_pd.transAxes, ha="left", va="top", fontsize=6.2,
               color=MUTED)
    ax_pd.set_xticks(pos)
    ax_pd.set_xticklabels(["PS\nstandard", "PP", "LDPE", "Bimodal\nHDPE"],
                          fontsize=6.5)
    ax_pd.set_ylim(0.0, max(d["pdi"] for d in data) * 1.28)
    ax_pd.set_ylabel("PDI = $M_w/M_n$ (-)", fontsize=7.4)
    ax_pd.tick_params(axis="y", labelsize=6.5)
    ax_pd.set_title("Polydispersity across the four resins", fontsize=8.5,
                    fontweight="bold", color=INK)


    panel_labels_inches(fig, [ax], labels=["(a)"], dx=-0.60, dy=0.16)
    panel_labels_inches(fig, [ax_cu], labels=["(b)"], dx=-0.54, dy=0.14)
    panel_labels_inches(fig, [ax_pd], labels=["(c)"], dx=-0.54, dy=0.14)
    savefig(fig, "837_pol_molecular_weight_dist.png")


def fmt_m(v):
    if v >= 1.0e6:
        return "%.2f x 10$^6$" % (v / 1.0e6)
    if v >= 1.0e4:
        return "%.1f x 10$^4$" % (v / 1.0e4)
    return "%.0f" % v


if __name__ == "__main__":
    main()

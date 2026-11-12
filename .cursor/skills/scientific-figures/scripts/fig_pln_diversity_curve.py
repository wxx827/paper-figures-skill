"""Phanerozoic genus diversity, its sampling-standardised counterpart and rates.

Rates are the primitive here, not the curve.  Per-lineage origination and
extinction are simulated first, mass extinctions are added as pulses on the
extinction rate, and standing diversity is then integrated from them, so the
spikes in panel (b) and the drops in panel (a) cannot drift apart.  The
extinction magnitudes ranked in panel (c) are measured off the same integrated
curve rather than typed in.

The confidence band is the 2.5-97.5 percentile envelope of a 60-run ensemble
that differs only in its rate noise.  Time runs from old on the left to young
on the right in every panel that carries a time axis, matching the range chart
in this series, and the period band uses house neutrals instead of the
saturated chronostratigraphic standard colours.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
GREY = "#808080"
RED = "#b23a48"
BLUE = "#3b5b92"
GOLD = "#c9a227"

FIG_W, FIG_H = 10.4, 6.4
SEED = 41
N_ENS = 60
AGES = np.arange(541.0, -0.5, -1.0)       # Ma, descending: old -> young
CARRY = 1250.0                            # logistic ceiling, genera

PERIODS = [
    ("Cm", 541.0, 486.9), ("O", 486.9, 443.1), ("S", 443.1, 419.2),
    ("D", 419.2, 358.9), ("C", 358.9, 298.9), ("P", 298.9, 251.9),
    ("Tr", 251.9, 201.4), ("J", 201.4, 145.0), ("K", 145.0, 66.0),
    ("Pg", 66.0, 23.0), ("Ng", 23.0, 0.0),
]
BAND = ("#f0f0f0", "#e0e0e0")

# age (Ma), short boundary code, full name, pulse amplitude, pulse width (Myr)
EVENTS = [
    (443.8, "O-S", "End-Ordovician", 0.345, 2.2),
    (372.0, "F-F", "Late Devonian", 0.184, 3.0),
    (251.9, "P-Tr", "End-Permian", 0.414, 1.6),
    (201.4, "Tr-J", "End-Triassic", 0.276, 2.0),
    (66.0, "K-Pg", "End-Cretaceous", 0.316, 1.8),
]


def smooth_noise(rng, n, width=28):
    k = np.exp(-0.5 * (np.arange(-3 * width, 3 * width + 1) / width) ** 2)
    k /= k.sum()
    z = np.convolve(rng.normal(size=n + 6 * width), k, mode="same")
    z = z[3 * width:3 * width + n]
    return z / (z.std() + 1e-12)


def simulate(rng):
    """Diversity-dependent origination, pulsed extinction, logistic integration.

    Origination falls as standing diversity approaches the ceiling, which is
    what lets the curve rebound after each pulse instead of flat-lining; a
    constant-rate birth-death model would leave every crash permanent.
    """
    n = AGES.size
    p_free = 0.078 + 0.0045 * smooth_noise(rng, n)
    p_floor = 0.020
    q = 0.040 + 0.0035 * smooth_noise(rng, n)
    for age, _, _, amp, w in EVENTS:
        q = q + amp * np.exp(-0.5 * ((AGES - age) / w) ** 2)
    q = np.clip(q, 0.006, None)

    d = np.empty(n)
    p = np.empty(n)
    d[0] = 22.0
    p[0] = p_free[0] * (1.0 - d[0] / CARRY) + p_floor
    for i in range(1, n):
        p[i] = p_free[i] * max(1.0 - d[i - 1] / CARRY, 0.0) + p_floor
        d[i] = max(d[i - 1] * np.exp(p[i] - q[i]), 6.0)
    return p, q, d


def at_age(age, d):
    return float(np.interp(age, AGES[::-1], d[::-1]))


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)

    ens_p, ens_q, ens_d = [], [], []
    for _ in range(N_ENS):
        p, q, d = simulate(rng)
        ens_p.append(p)
        ens_q.append(q)
        ens_d.append(d)
    ens_d = np.asarray(ens_d)
    std_med = np.median(ens_d, axis=0)
    lo, hi = np.percentile(ens_d, [2.5, 97.5], axis=0)
    p_med = np.median(np.asarray(ens_p), axis=0)
    q_med = np.median(np.asarray(ens_q), axis=0)

    # Raw counts carry the pull of the Recent: sampling improves toward today.
    bias = 1.0 + 0.55 * np.exp(-AGES / 95.0)
    raw = std_med * bias * (1.0 + 0.035 * smooth_noise(rng, AGES.size, 8))

    losses = []
    for age, code, name, _, _ in EVENTS:
        pre, post = at_age(age + 7.0, std_med), at_age(age - 7.0, std_med)
        losses.append((name, code, age, 100.0 * (pre - post) / pre))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, (0.95, 1.44, 5.30, 4.03))
    ax_b = axes_inches(fig, (0.95, 1.12, 5.30, 0.28))
    ax_r = axes_inches(fig, (7.20, 3.62, 2.90, 1.85))
    ax_i = axes_inches(fig, (7.20, 0.92, 2.90, 1.85))

    # ---- (a) diversity ----------------------------------------------------
    top = raw.max() * 1.26
    ax.set_xlim(541.0, 0.0)
    ax.set_ylim(0.0, top)
    for k, (ab, base, tp) in enumerate(PERIODS):
        ax.axvspan(base, tp, facecolor=("#fafafa", "#f2f2f2")[k % 2],
                   zorder=0)
    for age, code, name, _, _ in EVENTS:
        ax.axvline(age, color=RED, lw=0.9, alpha=0.75, zorder=2)
        ax_b.axvline(age, color=RED, lw=0.9, alpha=0.75, zorder=5)
        ax_r.axvline(age, color=RED, lw=0.9, alpha=0.55, zorder=2)
        ax.text(age, top * 0.985, code, fontsize=6.5, color=RED, ha="center",
                va="top", rotation=90.0, zorder=6)

    ax.fill_between(AGES, lo, hi, color=BLUE, alpha=0.18, linewidth=0,
                    zorder=3, label="95 % ensemble band")
    ax.plot(AGES, raw, color=GREY, lw=1.1, zorder=4, label="raw genus count")
    ax.plot(AGES, std_med, color=BLUE, lw=1.6, zorder=5,
            label="subsampling-standardised")
    ax.set_ylabel("standing diversity (genera)", fontsize=8.0)
    ax.tick_params(axis="y", labelsize=7.0)
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.spines["bottom"].set_visible(False)
    ax.legend(fontsize=7.0, frameon=False, loc="lower right",
              bbox_to_anchor=(0.99, 0.02), handlelength=1.5,
              labelspacing=0.30)
    ax.set_title("Phanerozoic genus-level diversity, raw versus "
                 "sampling-standardised", fontsize=9.0, fontweight="bold",
                 color=INK)
    ax.text(0.015, 0.74, f"pull of the Recent: raw exceeds\nstandardised by "
            f"{100.0 * (raw[-1] / std_med[-1] - 1.0):.0f} % at 0 Ma",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.6,
            color=SUB, zorder=6)

    # ---- period band ------------------------------------------------------
    ax_b.set_xlim(541.0, 0.0)
    ax_b.set_ylim(0.0, 1.0)
    for k, (ab, base, tp) in enumerate(PERIODS):
        ax_b.axvspan(base, tp, facecolor=BAND[k % 2], edgecolor="white",
                     linewidth=0.6, zorder=1)
        ax_b.text(0.5 * (base + tp), 0.42, ab, fontsize=6.6, color=INK,
                  ha="center", va="center", zorder=6)
    ax_b.set_yticks([])
    ax_b.set_xlabel("geological time (Ma before present) -- older to the left",
                    fontsize=8.0)
    ax_b.tick_params(axis="x", labelsize=6.8)
    for s in ax_b.spines.values():
        s.set_visible(False)

    # ---- (b) origination and extinction rates ----------------------------
    ax_r.set_xlim(541.0, 0.0)
    ax_r.plot(AGES, p_med, color=BLUE, lw=1.0, zorder=3, label="origination")
    ax_r.plot(AGES, q_med, color=RED, lw=1.0, zorder=4, label="extinction")
    ax_r.set_ylim(0.0, q_med.max() * 1.42)
    ax_r.set_xlabel("geological time (Ma)", fontsize=7.4)
    ax_r.set_ylabel("per-lineage rate\n(events / lineage / Myr)",
                    fontsize=7.0, linespacing=1.25)
    ax_r.tick_params(labelsize=6.6)
    ax_r.legend(fontsize=6.6, frameon=False, loc="upper center",
                bbox_to_anchor=(0.52, 1.00), ncol=2, handlelength=1.3,
                columnspacing=1.2)
    ax_r.set_title("Extinction rate spikes; origination does not",
                   fontsize=8.5, fontweight="bold", color=INK)

    # ---- (c) ranked extinction intensity ---------------------------------
    ranked = sorted(losses, key=lambda r: r[3])
    yp = np.arange(len(ranked))
    vals = [r[3] for r in ranked]
    shade = [RED if v == max(vals) else "#c0603f" for v in vals]
    ax_i.barh(yp, vals, height=0.62, color=shade, edgecolor="white",
              linewidth=0.5, zorder=3)
    for y, v in zip(yp, vals):
        ax_i.text(v + 1.2, y, f"{v:.0f} %", fontsize=6.6, color=SUB,
                  ha="left", va="center", zorder=4)
    ax_i.set_yticks(yp)
    ax_i.set_yticklabels([f"{r[0]}  ({r[2]:.0f} Ma)" for r in ranked],
                         fontsize=6.6)
    ax_i.set_xlim(0.0, max(vals) * 1.24)
    ax_i.set_ylim(-0.65, len(ranked) - 0.35)
    ax_i.set_xlabel("standing genera lost across the event (%)", fontsize=7.4)
    ax_i.tick_params(labelsize=6.6, length=2.0)
    ax_i.set_title("Ranked severity of the big five", fontsize=8.5,
                   fontweight="bold", color=INK)


    panel_labels_inches(fig, [ax], labels=["(a)"], dx=-0.56, dy=0.10)
    panel_labels_inches(fig, [ax_r], labels=["(b)"], dx=-0.62, dy=0.10)
    panel_labels_inches(fig, [ax_i], labels=["(c)"], dx=-0.62, dy=0.10)
    savefig(fig, "826_pln_diversity_curve.png")


if __name__ == "__main__":
    main()

"""Zero-knowledge proofs: cheap to verify, expensive to produce, and the two
costs do not move together.

Main panel: prover wall clock against circuit size for four proof systems on
log-log axes, five decades of circuit size against six of time.  A one-minute
proving budget is drawn across the panel and the largest circuit each system
can finish inside it is interpolated from that system's own curve, so the
four quoted circuit sizes are readings rather than claims.

Auxiliary (both 58 % of the main panel width): proof size against circuit
size, where the constant-size pairing-based proofs separate from the
transparent ones that grow with the logarithm of the circuit; and
verification cost at one circuit size, on a log axis because a linear-time
verifier is three decades slower than a constant-time one, annotated with
the prove-to-verify asymmetry computed from the two panels.

Illustrative synthetic cost models; no real proving system or machine.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

DEEP = "#2b4470"
BLUE = "#3b5b92"
TEAL = "#2a7f7a"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
ACCENT = "#b23a48"
RUST = "#c0603f"
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRID = "#d5d5d5"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.98, 1.00, 5.50, 4.48)
AUX_T_RECT = (7.70, 3.48, 3.14, 2.00)
AUX_B_RECT = (7.70, 0.86, 3.14, 2.00)

BUDGET_S = 60.0        # proving-time budget drawn on the main panel
N_REF = 1e6            # circuit size used by the two auxiliary panels

SYSTEMS = [
    # name, colour, prover seconds, proof bytes, verify ms, setup note
    ("Groth16 (pairing SNARK)", DEEP,
     lambda n: 8.0e-7 * n * np.log2(n),
     lambda n: np.full_like(n, 192.0),
     lambda n: np.full_like(n, 1.8), "circuit-specific setup"),
    ("PLONK (universal SNARK)", TEAL,
     lambda n: 1.9e-6 * n * np.log2(n),
     lambda n: np.full_like(n, 480.0),
     lambda n: np.full_like(n, 3.5), "universal setup"),
    ("STARK (transparent)", GREEN,
     lambda n: 4.2e-6 * n * np.log2(n),
     lambda n: 120.0 * np.log2(n) ** 2,
     lambda n: 0.030 * np.log2(n) ** 2, "no setup"),
    ("Bulletproofs (transparent)", GOLD,
     lambda n: 2.0e-4 * n,
     lambda n: 64.0 * np.log2(n) + 448.0,
     lambda n: 9.0e-4 * n, "no setup"),
]


def main():
    set_pub_style(base_font=8)

    n = np.logspace(3.0, 8.0, 320)
    prove = np.array([s[2](n) for s in SYSTEMS])
    size_kb = np.array([s[3](n) for s in SYSTEMS]) / 1024.0
    verify = np.array([s[4](n) for s in SYSTEMS])
    slopes = [float(np.polyfit(np.log10(n), np.log10(y), 1)[0]) for y in prove]
    n_budget = [float(np.interp(BUDGET_S, y, n)) for y in prove]

    i_ref = int(np.argmin(np.abs(n - N_REF)))
    n_ref = n[i_ref]
    size_ref = size_kb[:, i_ref]
    ver_ref = verify[:, i_ref]
    prove_ref = prove[:, i_ref]
    asym = prove_ref * 1e3 / ver_ref
    size_spread = size_ref.max() / size_ref.min()

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main panel ---------------------------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(color=GRID, lw=0.5, alpha=0.55, which="major", zorder=0)
    ax.grid(color=GRID, lw=0.35, alpha=0.30, which="minor", zorder=0)
    y_lo, y_hi = 1.0e-3, 5.0e4
    ax.axhline(BUDGET_S, color=ACCENT, lw=1.2, ls=(0, (4.0, 2.2)), zorder=6)
    ax.text(1.15e3, BUDGET_S * 1.22,
            f"proving budget: {BUDGET_S:.0f} s", fontsize=6.8, color=ACCENT,
            ha="left", va="bottom")

    for (name, col, _, _, _, setup), y, nb, s in zip(SYSTEMS, prove,
                                                     n_budget, slopes):
        ax.plot(n, y, color=col, lw=1.9, zorder=5)
        ax.plot([nb, nb], [y_lo, BUDGET_S], color=col, lw=0.85,
                ls=(0, (2.2, 1.8)), zorder=4)
        ax.plot([nb], [BUDGET_S], marker="o", ms=4.8, mfc="white", mec=col,
                mew=1.4, zorder=9)
    for k, (nb, col) in enumerate(zip(n_budget,
                                      [s[1] for s in SYSTEMS])):
        ax.text(nb * 1.14, 10.0 ** (-2.55 + 0.42 * k),
                f"{nb / 1e6:.1f} M", fontsize=6.7, color=col, ha="left",
                va="center", fontweight="bold", zorder=8,
                bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                          edgecolor="none", alpha=0.94))
    ax.text(1.15e3, 1.4e-3,
            "largest circuit that finishes inside the budget\n"
            "(millions of constraints, read off each curve)",
            fontsize=6.4, color=MUTED, ha="left", va="bottom",
            linespacing=1.5)

    ax.axvline(n_ref, color=INK, lw=0.8, ls=(0, (2.4, 2.0)), zorder=3)
    ax.text(n_ref / 1.30, 6.0e-2,
            f"{n_ref / 1e6:.0f} M: panels (b), (c)",
            rotation=90.0, rotation_mode="anchor", fontsize=6.0, color=INK,
            ha="left", va="bottom", zorder=8,
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                      edgecolor="none", alpha=0.94))

    ax.set_xlim(1.0e3, 1.2e8)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("circuit size (R1CS constraints, log scale)", fontsize=8.5)
    ax.set_ylabel("prover wall clock (seconds, log scale)", fontsize=8.5)
    ax.set_title("Cost of producing one proof, single core",
                 fontsize=9.0, fontweight="bold", pad=5.0)
    hm = [Line2D([], [], color=p[1], lw=1.9,
                 label=f"{p[0]}  -  fitted exponent {s:.2f}, {p[5]}")
          for p, s in zip(SYSTEMS, slopes)]
    leg = ax.legend(handles=hm, loc="upper left", frameon=False, fontsize=6.8,
                    labelspacing=0.45, handlelength=2.0, borderpad=0.2,
                    bbox_to_anchor=(-0.010, 1.012))
    for t in leg.get_texts():
        t.set_color(SUB)

    # ---- aux (b): proof size ------------------------------------------------
    ax_t.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    for (name, col, _, _, _, _), y in zip(SYSTEMS, size_kb):
        ax_t.loglog(n, y, color=col, lw=1.7, zorder=5)
        ax_t.plot([n_ref], [y[i_ref]], marker="o", ms=3.4, mfc="white",
                  mec=col, mew=1.0, zorder=7)
    ax_t.axvline(n_ref, color=INK, lw=0.8, ls=(0, (2.4, 2.0)), zorder=3)
    ax_t.set_xlim(1.0e3, 1.2e8)
    ax_t.set_ylim(0.045, 400.0)
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("circuit size (constraints, log scale)", fontsize=7.6,
                    labelpad=1.5)
    ax_t.set_ylabel("proof size\n(KB, log scale)", fontsize=7.6)
    ax_t.set_title("Constant-size proofs against transparent ones",
                   fontsize=8.4, fontweight="bold", pad=4.0)
    for (name, col, _, _, _, _), y in zip(SYSTEMS, size_kb):
        ax_t.text(1.0e8, y[-1] * 1.22, name.split(" (")[0], fontsize=6.0,
                  color=col, ha="right", va="bottom")
    ax_t.text(1.15e3, 0.052,
              f"at {n_ref / 1e6:.0f} M constraints the four proofs span "
              f"{size_spread:.0f}x in size "
              f"({size_ref.min():.2f} KB to {size_ref.max():.0f} KB)",
              fontsize=6.1, color=SUB, ha="left", va="bottom")

    # ---- aux (c): verification cost -----------------------------------------
    ypos = np.arange(len(SYSTEMS), dtype=float)
    x_lo = 0.5
    ax_b.set_xscale("log")
    ax_b.grid(color=GRID, lw=0.5, alpha=0.7, axis="x", which="major",
              zorder=0)
    for i, (p, v, a) in enumerate(zip(SYSTEMS, ver_ref, asym)):
        ax_b.barh(ypos[i], v - x_lo, left=x_lo, height=0.58,
                  color=lighten(p[1], 0.25), edgecolor=p[1], lw=0.7, zorder=4)
        lab = (f"{v:.1f} ms" if v < 100.0 else f"{v / 1e3:.2f} s")
        ax_b.text(v * 1.30, ypos[i],
                  f"{lab}    prove/verify {a:,.0f}x", fontsize=6.1,
                  color=p[1], ha="left", va="center", zorder=6)
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels([p[0].split(" (")[0] for p in SYSTEMS], fontsize=6.8)
    for tick, p in zip(ax_b.get_yticklabels(), SYSTEMS):
        tick.set_color(p[1])
    ax_b.invert_yaxis()
    ax_b.set_xlim(x_lo, 4.0e5)
    ax_b.set_ylim(len(SYSTEMS) - 0.42, -0.62)
    ax_b.set_xticks([1e0, 1e1, 1e2, 1e3])
    ax_b.set_xticklabels(["1 ms", "10 ms", "100 ms", "1 s"])
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel(f"verification time at {n_ref / 1e6:.0f} M constraints "
                    "(log scale)", fontsize=7.6, labelpad=1.5)
    ax_b.set_title("A linear-time verifier undoes the point of a proof",
                   fontsize=8.4, fontweight="bold", pad=4.0)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.70, dy=0.12)
    savefig(fig, "1018_ppc_zkp_cost.png")


if __name__ == "__main__":
    main()

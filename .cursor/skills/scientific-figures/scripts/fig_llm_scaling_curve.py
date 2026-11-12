"""Figure 314 - Neural scaling laws for language-model pre-training.

Panel (a) is the compute-optimal frontier for four data recipes: every run is
a dot, the line is the best achievable loss at that budget, and the fitted
power-law exponent is printed next to each family.  Panels (b) and (c) cut the
same parametric surface the other way, showing the iso-FLOP profiles whose
minima define the Chinchilla-optimal allocation of a fixed budget between
parameters and tokens.  All three panels share one loss axis.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, add_panel_label, style_axes

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
FAINT = "#8a8a8a"
GUIDE = "#cccccc"

# Chinchilla-style parametric loss  L(N, D) = E + A / N^a + B / D^b
FAMILIES = [
    ("Web crawl only", BLUE, dict(E=1.80, A=430.0, B=445.0)),
    ("Curated mix", GREEN, dict(E=1.73, A=412.0, B=418.0)),
    ("Multilingual", GOLD, dict(E=1.86, A=404.0, B=452.0)),
    ("Ours (dedup + curriculum)", ACCENT, dict(E=1.66, A=384.0, B=392.0)),
]
A_EXP, B_EXP = 0.34, 0.28
YLIM = (1.58, 4.7)
BUDGETS = [1e19, 1e20, 1e21, 1e22, 1e23]


def loss(N, D, E, A, B):
    return E + A / np.power(N, A_EXP) + B / np.power(D, B_EXP)


def frontier(budgets, p, grid=400):
    """Minimise the loss over the parameter split at each compute budget."""
    best_l, best_n = [], []
    for C in budgets:
        N = np.logspace(7.0, 12.4, grid)
        D = C / (6.0 * N)
        L = loss(N, D, **p)
        k = int(np.argmin(L))
        best_l.append(L[k])
        best_n.append(N[k])
    return np.array(best_l), np.array(best_n)


def main():
    set_pub_style(base_font=9)
    rng = np.random.default_rng(314)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.5))
    ax_a, ax_b, ax_c = axes

    # ---- (a) compute-optimal frontier -----------------------------------
    C = np.logspace(18.0, 24.0, 60)
    handles = []
    for name, col, p in FAMILIES:
        L, _ = frontier(C, p)
        lead = name.startswith("Ours")
        ax_a.plot(C, L, color=col, lw=2.0 if lead else 1.3,
                  zorder=4 if lead else 3)
        # individual training runs scatter slightly above the frontier
        Cs = np.logspace(18.2, 23.8, 9)
        Ls, _ = frontier(Cs, p)
        Ls = Ls * (1.0 + rng.uniform(0.008, 0.055, Ls.size))
        ax_a.plot(Cs, Ls, ls="none", marker="o", ms=2.6, mfc="white",
                  mec=col, mew=0.7, zorder=5 if lead else 2)
        slope = np.polyfit(np.log10(C), np.log10(L), 1)[0]
        handles.append(Line2D([], [], color=col, lw=2.0 if lead else 1.3,
                              marker="o", ms=3.0, mfc="white", mec=col,
                              mew=0.7,
                              label=f"{name}  $L\\propto C^{{{slope:.3f}}}$"))

    ax_a.axhline(FAMILIES[-1][2]["E"], color=SUB, ls=(0, (4, 2.5)), lw=0.9,
                 zorder=1)
    ax_a.text(1.4e18, FAMILIES[-1][2]["E"] * 1.02,
              "irreducible entropy of text", fontsize=6.6, color=SUB,
              ha="left", va="bottom")
    ax_a.set_xscale("log")
    ax_a.set_xlabel("training compute $C$ (FLOPs)")
    ax_a.set_ylabel("test loss (nats / token)")
    ax_a.set_title("Compute-optimal frontier", fontsize=8.5,
                   fontweight="bold")
    ax_a.set_xlim(1e18, 1.4e24)

    # ---- (b) iso-FLOP profiles over parameters --------------------------
    _, _, p_ours = FAMILIES[-1]
    N = np.logspace(7.6, 12.0, 400)
    opt_n, opt_l, opt_d = [], [], []
    for C0 in BUDGETS:
        D = C0 / (6.0 * N)
        L = loss(N, D, **p_ours)
        keep = (L < YLIM[1]) & (N / 1e9 < 620.0)
        ax_b.plot(N[keep] / 1e9, L[keep], color=GUIDE, lw=1.0, zorder=2)
        k = int(np.argmin(L))
        opt_n.append(N[k] / 1e9)
        opt_l.append(L[k])
        opt_d.append(C0 / (6.0 * N[k]) / 1e9)
        ax_b.plot(N[k] / 1e9, L[k], marker="o", ms=3.4, color=ACCENT,
                  zorder=5, clip_on=False)
        arm = np.flatnonzero(keep & (L < 4.25))[-1]
        ax_b.text(N[arm] / 1e9 * 0.90, L[arm] * 1.005,
                  f"$10^{{{int(np.log10(C0))}}}$", fontsize=6.2, color=FAINT,
                  ha="right", va="bottom")

    ax_b.plot(opt_n, opt_l, color=ACCENT, ls=(0, (4, 2)), lw=1.2, zorder=4)
    ax_b.set_xscale("log")
    ax_b.set_xlabel("model size $N$ (params, B)")
    ax_b.set_title("Parameter allocation", fontsize=8.5, fontweight="bold")
    ax_b.set_xlim(0.02, 700)
    ax_b.annotate("compute-optimal $N$", xy=(opt_n[2], opt_l[2]),
                  xytext=(0.030, 1.95), fontsize=6.8, color=ACCENT,
                  ha="left", va="center",
                  arrowprops=dict(arrowstyle="-", lw=0.7, color=ACCENT,
                                  shrinkA=1.0, shrinkB=2.0))

    # ---- (c) iso-FLOP profiles over tokens ------------------------------
    for C0 in BUDGETS:
        D = np.logspace(9.0, 13.6, 400)
        Nc = C0 / (6.0 * D)
        L = loss(Nc, D, **p_ours)
        keep = (L < YLIM[1]) & (D / 1e9 < 3.4e4)
        ax_c.plot(D[keep] / 1e9, L[keep], color=GUIDE, lw=1.0, zorder=2)
        arm = np.flatnonzero(keep & (L < 4.25))[-1]
        ax_c.text(D[arm] / 1e9 * 0.90, L[arm] * 1.005,
                  f"$10^{{{int(np.log10(C0))}}}$", fontsize=6.2, color=FAINT,
                  ha="right", va="bottom")
    ax_c.plot(opt_d, opt_l, color=ACCENT, ls=(0, (4, 2)), lw=1.2, zorder=4)
    ax_c.plot(opt_d, opt_l, ls="none", marker="o", ms=3.4, color=ACCENT,
              zorder=5, clip_on=False)
    ax_c.set_xscale("log")
    ax_c.set_xlabel("training data $D$ (tokens, B)")
    ax_c.set_title("Token allocation", fontsize=8.5, fontweight="bold")
    ax_c.set_xlim(2, 4e4)
    ratio = np.mean(np.array(opt_d) / np.array(opt_n))
    ax_c.text(0.03, 0.045, f"$D/N \\approx {ratio:.0f}$ tokens per parameter",
              transform=ax_c.transAxes, ha="left", va="bottom", fontsize=6.8,
              color=ACCENT,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                        edgecolor="#dcdcdc", linewidth=0.6, alpha=0.95))

    for k, ax in enumerate(axes):
        style_axes(ax)
        ax.set_yscale("log")
        ax.set_ylim(*YLIM)
        ax.set_yticks([1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
        ax.set_yticklabels(["1.6", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5"])
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.grid(True, which="major", axis="both", lw=0.5, color="#e6e6e6")
        add_panel_label(ax, f"({'abc'[k]})", x=-0.17, y=1.05)
        if k:
            ax.tick_params(labelleft=False)

    fig.tight_layout(rect=[0, 0.035, 1, 0.875], w_pad=2.0)
    savefig(fig, "314_llm_scaling_curve.png")


if __name__ == "__main__":
    main()

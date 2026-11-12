"""Virtual screening cascade: the attrition funnel from a two-million compound
library down to confirmed hits, beside the ROC-style enrichment curves that
justify each scoring stage and the early-enrichment factors they deliver."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
RED = "#b23a48"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
BASE = "#808080"

# stage label, compounds entering the stage, throughput note
STAGES = [
    ("Purchasable library", 2_100_000, "ZINC-like, 2D filtered"),
    ("Property + PAINS filter", 412_000, "Ro5, PAINS, reactive"),
    ("Pharmacophore match", 68_500, "3-point, 4 conformers"),
    ("Docking, fast mode", 9_400, "rigid receptor"),
    ("Docking, precision mode", 1_250, "flexible side chains"),
    ("MM-GBSA rescoring", 260, "implicit solvent"),
    ("Chemist triage", 72, "novelty and synthesis"),
    ("Confirmed hits", 14, "IC$_{50}$ < 10 $\\mu$M"),
]

# scoring function, active-score mean shift (sigma units), colour
METHODS = [
    ("MM-GBSA rescore", 2.05, RED),
    ("Docking, precision", 1.55, BLUE),
    ("Docking, fast", 1.05, GREEN),
    ("2D fingerprint", 0.62, GOLD),
]

N_ACTIVE, N_DECOY = 260, 24_000


def screen(rng, shift):
    """Rank a spiked library by one scoring function; return the ROC curve."""
    s_act = rng.normal(shift, 1.0, N_ACTIVE)
    s_dec = rng.normal(0.0, 1.0, N_DECOY)
    scores = np.concatenate([s_act, s_dec])
    is_act = np.concatenate([np.ones(N_ACTIVE, bool), np.zeros(N_DECOY, bool)])
    order = np.argsort(-scores)
    found = np.cumsum(is_act[order]) / N_ACTIVE
    frac = np.arange(1, scores.size + 1) / scores.size
    return frac, found


def ef_at(frac, found, top):
    """Enrichment factor at the top ``top`` fraction of the ranked list."""
    return float(np.interp(top, frac, found)) / top


def funnel_panel(ax, rng):
    counts = np.array([s[1] for s in STAGES], dtype=float)
    lw_ = np.log10(counts)
    half = 0.36 + 0.64 * (lw_ - lw_.min()) / (lw_.max() - lw_.min())
    ramp = [lighten(BLUE, 0.62 - 0.62 * i / (len(STAGES) - 1))
            for i in range(len(STAGES))]
    ramp[-1] = RED

    bh, gap = 0.72, 0.34
    tops = [-i * (bh + gap) for i in range(len(STAGES))]
    for i, ((name, cnt, note), hw, y, col) in enumerate(
            zip(STAGES, half, tops, ramp)):
        ax.add_patch(FancyBboxPatch(
            (-hw, y - bh), 2 * hw, bh,
            boxstyle="round,pad=0,rounding_size=0.14", facecolor=col,
            edgecolor="white", linewidth=0.9, mutation_aspect=0.42, zorder=3))
        light = i >= len(STAGES) - 5
        ax.text(0.0, y - bh * 0.44, f"{int(cnt):,}", ha="center", va="center",
                fontsize=7.0, color="white" if light else INK, zorder=5,
                fontweight="bold" if i == len(STAGES) - 1 else "normal")
        ax.text(-1.10, y - bh * 0.30, name, ha="right", va="center",
                fontsize=7.0, color=INK, zorder=5)
        ax.text(-1.10, y - bh * 0.74, note, ha="right", va="center",
                fontsize=6.0, color=MUTED, zorder=5)
        if i:
            keep = 100.0 * cnt / counts[i - 1]
            ax.annotate("", xy=(1.16, y - 0.04), xytext=(1.16, y + gap - 0.04),
                        arrowprops=dict(arrowstyle="-|>", color="#9a9a9a",
                                        lw=0.8, shrinkA=0, shrinkB=0))
            ax.text(1.24, y + gap / 2 - 0.04,
                    f"{keep:.1f} % kept", ha="left", va="center",
                    fontsize=6.1, color=SUB)

    total_keep = 100.0 * counts[-1] / counts[0]
    ax.text(1.16, tops[-1] - bh - 0.30,
            f"overall {total_keep * 1e4:.1f} per 10$^6$ compounds",
            ha="center", va="top", fontsize=6.3, color=RED)
    ax.set_xlim(-2.55, 2.15)
    ax.set_ylim(tops[-1] - bh - 0.62, 0.30)
    ax.axis("off")
    ax.set_title("Screening cascade, compounds surviving each stage",
                 fontsize=9.0, fontweight="bold", pad=6)


def enrich_panel(ax, curves):
    ax.plot([0, 100], [0, 100], color=BASE, lw=0.9, ls=(0, (4, 2.5)),
            zorder=2, label="Random ranking")
    for (name, _shift, col), (frac, found) in curves.items():
        ax.plot(100 * frac, 100 * found, color=col, lw=1.5, zorder=4,
                label=f"{name}   EF@1% = {ef_at(frac, found, 0.01):.1f}")
    ax.axvline(1.0, color="#9a9a9a", lw=0.8, ls=(0, (2, 2)), zorder=3)
    best = max(curves, key=lambda m: ef_at(*curves[m], 0.01))
    fb, hb = curves[best]
    ax.annotate(f"top 1 % holds\n{100 * np.interp(0.01, fb, hb):.0f} % of "
                f"the actives",
                xy=(1.0, 100 * np.interp(0.01, fb, hb)), xytext=(13.0, 14.0),
                fontsize=6.4, color=SUB, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=0.7,
                                shrinkA=1, shrinkB=2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Ranked library screened (%)", fontsize=8.0)
    ax.set_ylabel("Known actives recovered (%)", fontsize=8.0)
    ax.set_title("Enrichment of a spiked benchmark set", fontsize=9.0,
                 fontweight="bold", pad=4)
    ax.tick_params(labelsize=7.0)
    ax.grid(True, color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=6.8, frameon=False,
              handlelength=1.5, borderaxespad=0.3, labelspacing=0.30)
    style_axes(ax)


def ef_panel(ax, curves):
    names = [m[0] for m in METHODS]
    cols = [m[2] for m in METHODS]
    ef1 = [ef_at(*curves[m], 0.01) for m in METHODS]
    ef5 = [ef_at(*curves[m], 0.05) for m in METHODS]
    y = np.arange(len(names))[::-1]
    ax.barh(y + 0.19, ef1, height=0.34, color=cols, edgecolor="white",
            linewidth=0.6, zorder=3, label="EF@1%")
    ax.barh(y - 0.19, ef5, height=0.34, color=[lighten(c, 0.52) for c in cols],
            edgecolor="white", linewidth=0.6, zorder=3, label="EF@5%")
    for yi, a, b in zip(y, ef1, ef5):
        ax.text(a + 0.5, yi + 0.19, f"{a:.1f}", ha="left", va="center",
                fontsize=6.1, color=INK)
        ax.text(b + 0.5, yi - 0.19, f"{b:.1f}", ha="left", va="center",
                fontsize=6.1, color=SUB)
    ax.axvline(1.0, color=BASE, lw=0.8, ls=(0, (3, 2.2)), zorder=4)
    ax.text(1.35, y.min() - 0.52, "random = 1", fontsize=6.1, color=SUB,
            ha="left", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.5)
    ax.set_xlim(0, max(ef1) * 1.30)
    ax.set_ylim(y.min() - 0.75, y.max() + 0.60)
    ax.set_xlabel("Enrichment factor (fold over random)", fontsize=7.5)
    ax.set_title("Early enrichment by scoring stage", fontsize=8.5,
                 fontweight="bold", pad=4)
    ax.tick_params(labelsize=6.5)
    ax.xaxis.grid(True, color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=6.5, frameon=False,
              handlelength=1.3, borderaxespad=0.25)
    style_axes(ax)


def main():
    set_pub_style()
    rng = np.random.default_rng(20516)
    curves = {m: screen(rng, m[1]) for m in METHODS}

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.92],
                          height_ratios=[1.0, 0.58], wspace=0.30, hspace=0.60,
                          left=0.115, right=0.965, top=0.885, bottom=0.105)
    ax_f = fig.add_subplot(gs[:, 0])
    ax_e = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, 1])

    funnel_panel(ax_f, rng)
    enrich_panel(ax_e, curves)
    ef_panel(ax_b, curves)

    best = max(curves, key=lambda m: ef_at(*curves[m], 0.01))

    panel_labels_inches(fig, [ax_f, ax_e, ax_b], dx=-0.40, dy=0.08)
    savefig(fig, "516_drug_virtual_screening.png")


if __name__ == "__main__":
    main()

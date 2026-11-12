"""Who annotated the corpus: one worker table, four views of it.

Every panel is a projection of a single simulated worker table, so the
figures cannot contradict each other: the histogram bars sum to the same
number of annotators as the regional bars, the long-tail curve has exactly
one point per annotator, and the scatter has the same population again.
Every number printed in a title, an annotation or the footnote is an
f-string over those arrays -- none of them is typed in by hand.

(a) age distribution with the sample median marked
(b) annotators per region, sorted, with counts and shares
(c) annotations per worker, ranked, on a log axis: the long tail that every
    crowdsourced corpus has, with the top decile shaded and its share of the
    total measured from the array
(d) agreement with the expert review against experience, with a least-squares
    fit and the Pearson correlation of the plotted points
"""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import (set_pub_style, savefig, style_axes, letter_panels,
                      annotate_box, lighten)

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d9d9d9"

BLUE = "#3b5b92"
ACCENT = "#b23a48"
BASE = "#808080"

SEED = 1539  # original id; figure number is 1134
N_WORKERS = 412
REGIONS = ["South Asia", "South-East Asia", "Eastern Europe", "West Africa",
           "South America", "Western Europe", "North America"]
REGION_P = [0.27, 0.19, 0.16, 0.12, 0.11, 0.09, 0.06]


def worker_table(rng):
    """The single source of truth behind all four panels."""
    age = np.clip(rng.gamma(8.5, 3.1, N_WORKERS) + 15.0, 18, 66)
    region = rng.choice(len(REGIONS), size=N_WORKERS, p=REGION_P)
    experience = np.clip(rng.gamma(1.9, 92.0, N_WORKERS), 4, 900)
    # productivity follows experience, with a heavy multiplicative tail
    n_ann = np.round(np.exp(rng.normal(0.0, 1.05, N_WORKERS))
                     * (38.0 + 2.35 * experience ** 0.78)).astype(int)
    n_ann = np.clip(n_ann, 12, None)
    quality = np.clip(0.545 + 0.085 * np.log1p(experience / 26.0)
                      + rng.normal(0.0, 0.052, N_WORKERS), 0.30, 0.995)
    return age, region, experience, n_ann, quality


def panel_age(ax, age):
    lo, hi = np.floor(age.min()), np.ceil(age.max())
    bins = np.arange(lo, hi + 4, 4.0)
    counts, edges = np.histogram(age, bins=bins)
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
           color=lighten(BLUE, 0.28), edgecolor="white", linewidth=0.7,
           zorder=3)
    med = float(np.median(age))
    ax.axvline(med, color=BASE, lw=1.1, linestyle=(0, (5, 3)), zorder=4)
    ax.text(med + 1.0, counts.max() * 0.94, f"median {med:.0f} yr",
            ha="left", va="top", fontsize=6.8, color=SUB, zorder=5)
    ax.set_xlabel("age (years)", labelpad=2)
    ax.set_ylabel("annotators (count)", labelpad=2)
    ax.set_xlim(lo, edges[-1])
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_title(f"Age of the {age.size} annotators", fontsize=8.6,
                 color=RULE, pad=4, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    style_axes(ax)
    return counts


def panel_region(ax, region):
    counts = np.bincount(region, minlength=len(REGIONS))
    order = np.argsort(counts)
    y = np.arange(len(REGIONS))
    ax.barh(y, counts[order], height=0.66, color=lighten(BLUE, 0.28),
            edgecolor="white", linewidth=0.7, zorder=3)
    total = counts.sum()
    for k, idx in enumerate(order):
        ax.text(counts[idx] + total * 0.008, k,
                f"{counts[idx]}  ({counts[idx] / total * 100:.0f} %)",
                va="center", ha="left", fontsize=6.6, color=SUB, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([REGIONS[i] for i in order], fontsize=7.0)
    ax.set_xlim(0, counts.max() * 1.34)
    ax.set_xlabel("annotators (count)", labelpad=2)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"Region of residence  ·  {total} annotators in "
                 f"{len(REGIONS)} regions", fontsize=8.6, color=RULE, pad=4,
                 loc="left")
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    style_axes(ax)
    return counts


def panel_tail(ax, n_ann):
    s = np.sort(n_ann)[::-1]
    rank = np.arange(1, s.size + 1)
    k = int(np.ceil(0.10 * s.size))
    share = s[:k].sum() / s.sum() * 100.0
    ax.fill_between(rank, 1, s, color=lighten(BLUE, 0.78), zorder=2)
    ax.fill_between(rank[:k], 1, s[:k], color=lighten(BLUE, 0.42), zorder=3)
    ax.plot(rank, s, color=BLUE, lw=1.6, zorder=4)
    ax.axvline(k, color=BASE, lw=1.0, linestyle=(0, (5, 3)), zorder=5)
    ax.set_yscale("log")
    ax.set_xlim(0, s.size)
    ax.set_ylim(max(s.min() * 0.7, 1), s.max() * 2.6)
    ax.set_xlabel("annotator rank (most productive first)", labelpad=2)
    ax.set_ylabel("annotations (count, log)", labelpad=2)
    ax.set_title(f"Output per annotator  ·  {s.sum():,} annotations in total",
                 fontsize=8.6, color=RULE, pad=4, loc="left")
    annotate_box(ax, f"top {k} annotators (10 %)\ncontributed "
                     f"{share:.1f} % of all annotations\nmedian "
                     f"{np.median(s):.0f} per worker", loc="upper right",
                 fontsize=6.8, color=SUB)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    style_axes(ax)
    return s, k, share


def panel_quality(ax, experience, quality):
    ax.scatter(experience, quality, s=11, color=BLUE, alpha=0.40,
               edgecolors="none", zorder=3)
    b, a = np.polyfit(np.log10(experience), quality, 1)
    xs = np.linspace(experience.min(), experience.max(), 200)
    ax.plot(xs, a + b * np.log10(xs), color=ACCENT, lw=1.8, zorder=5)
    r = float(np.corrcoef(np.log10(experience), quality)[0, 1])
    ax.set_xscale("log")
    ax.set_xlim(experience.min() * 0.8, experience.max() * 1.25)
    ax.set_ylim(quality.min() - 0.03, min(quality.max() + 0.04, 1.02))
    ax.set_xlabel("experience at annotation time (days, log)", labelpad=2)
    ax.set_ylabel("agreement with expert (fraction)", labelpad=2)
    ax.set_title("Quality against experience", fontsize=8.6, color=RULE,
                 pad=4, loc="left")
    annotate_box(ax, f"least-squares fit on log$_{{10}}$ days\n"
                     f"Pearson r = {r:.2f}  ·  n = {experience.size}",
                 loc="lower right", fontsize=6.8, color=SUB)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    style_axes(ax)
    return r


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)
    age, region, experience, n_ann, quality = worker_table(rng)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.4))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    panel_age(ax_a, age)
    reg_counts = panel_region(ax_b, region)
    s, k, share = panel_tail(ax_c, n_ann)
    r = panel_quality(ax_d, experience, quality)
    letter_panels(axes, x=-0.115, y=1.05, size=11)


    fig.tight_layout(rect=[0.005, 0.035, 0.995, 0.915])
    fig.subplots_adjust(hspace=0.44, wspace=0.24)
    savefig(fig, "1134_wm_data_worker_stats.png")


if __name__ == "__main__":
    main()

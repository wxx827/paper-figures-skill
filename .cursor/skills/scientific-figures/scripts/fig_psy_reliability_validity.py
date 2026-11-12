"""Four evidence panels that jointly argue one scale is fit for use.

Internal consistency, split-half agreement, convergent / discriminant
validity and temporal stability are four different questions, and a scale
only passes if all four answer yes - so the four panels are deliberately the
same size: none of them is the conclusion on its own.  A single simulated
respondent sample feeds all of them, and every coefficient printed on the
figure (alpha, Spearman-Brown, the multitrait-multimethod means, ICC) is
computed from the data plotted in that panel.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, get_cmap, annotate_box, savefig
from figkit import panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
PALE = "#b3b3b3"
BLUE = "#3b5b92"
ACCENT = "#b23a48"

N_RESP = 220
N_ITEM = 20
ALPHA_GOOD = 0.80
ALPHA_MIN = 0.70
SB_REF = 5                 # anchor length for the Spearman-Brown projection
TRAITS = ["extraversion", "conscientiousness", "openness"]
METHODS = ["self", "peer"]
RETEST_WEEKS = 4


def scale_data(seed=599):
    """A 20-item single-factor scale plus a four-week retest."""
    rng = np.random.default_rng(seed)
    true = rng.normal(size=N_RESP)
    load = rng.uniform(0.52, 0.78, N_ITEM)
    items = (true[:, None] * load[None, :] +
             rng.normal(size=(N_RESP, N_ITEM)) * np.sqrt(1.0 - load ** 2))
    t1 = items.sum(axis=1)
    t2 = (true * load.sum() + rng.normal(0.0, 1.45, N_RESP) + 0.28)
    return items, t1, t2


def cronbach_alpha(x):
    k = x.shape[1]
    return (k / (k - 1.0)) * (1.0 - x.var(axis=0, ddof=1).sum() /
                              x.sum(axis=1).var(ddof=1))


def mtmm(seed=1599):
    """Correlations among 3 traits x 2 methods, from simulated scores."""
    rng = np.random.default_rng(seed)
    n_v = len(TRAITS) * len(METHODS)
    trait = rng.normal(size=(N_RESP, len(TRAITS)))
    method = rng.normal(size=(N_RESP, len(METHODS)))
    cols, names = [], []
    for ti, t in enumerate(TRAITS):
        for mi, m in enumerate(METHODS):
            cols.append(0.80 * trait[:, ti] + 0.42 * method[:, mi] +
                        rng.normal(0.0, 0.42, N_RESP))
            names.append(f"{t[:4].capitalize()}-{m}")
    r = np.corrcoef(np.column_stack(cols), rowvar=False)
    trait_of = np.repeat(np.arange(len(TRAITS)), len(METHODS))
    meth_of = np.tile(np.arange(len(METHODS)), len(TRAITS))
    conv = np.zeros((n_v, n_v), dtype=bool)
    for i in range(n_v):
        for j in range(n_v):
            if i != j and trait_of[i] == trait_of[j] and meth_of[i] != meth_of[j]:
                conv[i, j] = True
    return r, names, conv


def icc21(a, b):
    """ICC(2,1): two-way random effects, absolute agreement, single rating."""
    x = np.column_stack([a, b])
    n, k = x.shape
    grand = x.mean()
    ms_r = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1.0)
    ms_c = n * ((x.mean(axis=0) - grand) ** 2).sum() / (k - 1.0)
    resid = x - x.mean(axis=1, keepdims=True) - x.mean(axis=0) + grand
    ms_e = (resid ** 2).sum() / ((n - 1.0) * (k - 1.0))
    return (ms_r - ms_e) / (ms_r + (k - 1.0) * ms_e + k * (ms_c - ms_e) / n)


def main():
    set_pub_style()
    items, t1, t2 = scale_data()
    r_mtmm, names, conv = mtmm()

    ks = np.arange(3, N_ITEM + 1)
    alphas = np.array([cronbach_alpha(items[:, :k]) for k in ks])
    a_ref = cronbach_alpha(items[:, :SB_REF])
    sb = (ks / SB_REF * a_ref) / (1.0 + (ks / SB_REF - 1.0) * a_ref)
    hit = ks[alphas >= ALPHA_GOOD]
    k_good = int(hit[0]) if hit.size else -1

    odd = items[:, 0::2].sum(axis=1)
    even = items[:, 1::2].sum(axis=1)
    r_half = float(np.corrcoef(odd, even)[0, 1])
    r_sb = 2.0 * r_half / (1.0 + r_half)

    off = ~np.eye(len(names), dtype=bool)
    conv_mean = float(r_mtmm[conv].mean())
    disc_mean = float(r_mtmm[off & ~conv].mean())

    icc = float(icc21(t1, t2))
    r_retest = float(np.corrcoef(t1, t2)[0, 1])

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    fig.subplots_adjust(left=0.082, right=0.960, bottom=0.175, top=0.876,
                        wspace=0.38, hspace=0.60)

    # ---- (a) internal consistency vs test length -------------------------
    ax_a.plot(ks, sb, color=PALE, lw=1.3, ls=(0, (3.6, 2.0)), zorder=3,
              label=f"Spearman-Brown from $k$ = {SB_REF}")
    ax_a.plot(ks, alphas, color=BLUE, lw=1.8, marker="o", ms=3.0,
              mfc="white", mew=0.9, zorder=4, label="observed $\\alpha$")
    for lev, lab in [(ALPHA_MIN, "acceptable"), (ALPHA_GOOD, "good")]:
        ax_a.axhline(lev, color=GREY, lw=0.8, ls=(0, (1.8, 1.8)), zorder=2)
        ax_a.text(N_ITEM, lev + 0.008, f"{lab} {lev:.2f}", ha="right",
                  va="bottom", fontsize=6.3, color=SUB)
    if k_good > 0:
        ax_a.plot([k_good], [alphas[ks == k_good][0]], marker="o", ms=5.4,
                  mfc="none", mec=ACCENT, mew=1.3, zorder=6)
    annotate_box(ax_a,
                 f"$\\alpha \\geq$ {ALPHA_GOOD:.2f} from $k$ = {k_good} "
                 f"items\nfull length $\\alpha$ = {alphas[-1]:.3f}",
                 loc="upper left", fontsize=6.4)
    ax_a.set_xlim(2.4, N_ITEM + 0.6)
    ax_a.set_ylim(0.50, 1.04)
    ax_a.set_xlabel("number of items $k$ (items)", fontsize=7.8)
    ax_a.set_ylabel(r"Cronbach's $\alpha$ (-)", fontsize=7.8)
    ax_a.tick_params(labelsize=6.8)
    ax_a.grid(axis="y", lw=0.5, color="#e4e4e4")
    ax_a.set_title("Consistency grows with length", fontsize=8.6,
                   fontweight="bold", color=INK)
    ax_a.legend(loc="lower right", frameon=False, fontsize=6.4,
                handlelength=1.7, labelspacing=0.28, borderaxespad=0.4)

    # ---- (b) split-half agreement ----------------------------------------
    lo = min(odd.min(), even.min())
    hi = max(odd.max(), even.max())
    pad = 0.06 * (hi - lo)
    span = np.array([lo - pad, hi + pad])
    ax_b.plot(span, span, color=GREY, lw=0.9, ls=(0, (3.4, 2.0)), zorder=2,
              label="1 : 1")
    slope, icpt = np.polyfit(odd, even, 1)
    ax_b.plot(span, slope * span + icpt, color=ACCENT, lw=1.3, zorder=4,
              label="least squares")
    ax_b.scatter(odd, even, s=9, color=BLUE, alpha=0.55, linewidth=0.0,
                 zorder=3)
    ax_b.set_xlim(*span)
    ax_b.set_ylim(*span)
    ax_b.set_xlabel("odd-item half score (points)", fontsize=7.8)
    ax_b.set_ylabel("even-item half score (points)", fontsize=7.8)
    ax_b.tick_params(labelsize=6.8)
    ax_b.set_title("The two halves agree", fontsize=8.6, fontweight="bold",
                   color=INK)
    annotate_box(ax_b,
                 f"$r$ = {r_half:.3f}\n"
                 f"Spearman-Brown $r_{{sb}}$ = {r_sb:.3f}\n"
                 f"n = {N_RESP}", loc="upper left", fontsize=6.4)
    ax_b.legend(loc="lower right", frameon=False, fontsize=6.4,
                handlelength=1.6, labelspacing=0.26, borderaxespad=0.4)

    # ---- (c) multitrait-multimethod matrix -------------------------------
    cmap = get_cmap("cb_sequential_blue")
    im = ax_c.imshow(r_mtmm, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto",
                     interpolation="nearest")
    for i in range(len(names)):
        for j in range(len(names)):
            ax_c.text(j, i, f"{r_mtmm[i, j]:.2f}", ha="center", va="center",
                      fontsize=6.0,
                      color="white" if r_mtmm[i, j] > 0.55 else INK)
            if conv[i, j]:
                ax_c.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0,
                                         facecolor="none", edgecolor=ACCENT,
                                         linewidth=1.2, zorder=5))
    ax_c.set_xticks(np.arange(len(names)))
    ax_c.set_yticks(np.arange(len(names)))
    ax_c.set_xticklabels(names, fontsize=6.0, rotation=38.0, ha="right",
                         rotation_mode="anchor")
    ax_c.set_yticklabels(names, fontsize=6.0)
    ax_c.tick_params(length=2.0)
    for s in ax_c.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    ax_c.set_title("Convergent above discriminant", fontsize=8.6,
                   fontweight="bold", color=INK, pad=13.0)
    ax_c.text(0.0, 1.015,
              f"boxed convergent mean $r$ = {conv_mean:.2f}  vs  "
              f"discriminant mean $r$ = {disc_mean:.2f}",
              transform=ax_c.transAxes, ha="left", va="bottom",
              fontsize=6.3, color=ACCENT)
    safe_colorbar(fig, im, ax_c, label=r"correlation $r$ (-)", width=0.09,
                  pad=0.06, labelsize=6.3)

    # ---- (d) four-week retest --------------------------------------------
    lo2 = min(t1.min(), t2.min())
    hi2 = max(t1.max(), t2.max())
    pad2 = 0.06 * (hi2 - lo2)
    span2 = np.array([lo2 - pad2, hi2 + pad2])
    ax_d.plot(span2, span2, color=GREY, lw=0.9, ls=(0, (3.4, 2.0)), zorder=2,
              label="1 : 1")
    s2, i2 = np.polyfit(t1, t2, 1)
    ax_d.plot(span2, s2 * span2 + i2, color=ACCENT, lw=1.3, zorder=4,
              label="least squares")
    ax_d.scatter(t1, t2, s=9, color=BLUE, alpha=0.55, linewidth=0.0,
                 zorder=3)
    ax_d.set_xlim(*span2)
    ax_d.set_ylim(*span2)
    ax_d.set_xlabel("total score at T1 (points)", fontsize=7.8)
    ax_d.set_ylabel("total score at T2 (points)", fontsize=7.8)
    ax_d.tick_params(labelsize=6.8)
    ax_d.set_title(f"Stable across a {RETEST_WEEKS}-week retest",
                   fontsize=8.6, fontweight="bold", color=INK)
    annotate_box(ax_d,
                 f"ICC(2,1) = {icc:.3f}\n"
                 f"$r$ = {r_retest:.3f}\n"
                 f"mean drift = {np.mean(t2 - t1):+.2f} points",
                 loc="upper left", fontsize=6.4)
    ax_d.legend(loc="lower right", frameon=False, fontsize=6.4,
                handlelength=1.6, labelspacing=0.26, borderaxespad=0.4)


    note = (f"{N_RESP} simulated respondents, {N_ITEM} items. Full-length "
            f"$\\alpha$ = {alphas[-1]:.3f}; the dashed curve is the "
            f"Spearman-Brown projection from the first {SB_REF} items "
            f"($\\alpha$ = {a_ref:.3f}). Split-half uses odd against even "
            f"items. The multitrait-multimethod block crosses "
            f"{len(TRAITS)} traits with {len(METHODS)} methods; boxed cells "
            f"are same-trait / different-method. Retest is a "
            f"{RETEST_WEEKS}-week interval on the same sample.")

    panel_labels_inches(fig, [ax_a, ax_b, ax_c, ax_d],
                        labels=["(a)", "(b)", "(c)", "(d)"], dx=-0.46,
                        dy=0.10)
    savefig(fig, "599_psy_reliability_validity.png")


if __name__ == "__main__":
    main()

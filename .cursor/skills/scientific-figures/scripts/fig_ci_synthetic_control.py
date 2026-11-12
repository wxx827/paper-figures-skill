"""Figure 773 - Synthetic control for a single treated unit.

Main panel: the treated unit against its synthetic counterpart.  The two
paths are indistinguishable before the intervention - which is the only
evidence anyone has that the comparison is credible - and separate
afterwards, with the gap shaded so its size is readable without arithmetic.

Auxiliary (both at 58 % of the main panel width): the donor weights, which
are sparse because the fit is a non-negative least-squares problem with the
weights constrained to a simplex; and the placebo distribution, where every
donor is treated as if it had been the treated unit, giving the rank-based
p value quoted on the figure.

The weights come from an actual constrained fit to the pre-intervention
window, the gaps are differences between plotted series, and the p value is
the rank of the treated unit's post-to-pre RMSPE ratio among all units.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_label_inches, shared_limits

INK, SUB, MUTED, FAINT = "#1a1a1a", "#555555", "#777777", "#8a8a8a"
HAIR, GRID, BASE = "#cccccc", "#d9d9d9", "#808080"
CARD = "#f5f5f5"

TREAT = "#b23a48"        # the treated unit: the single emphasis
SYNTH = "#3b5b92"        # its synthetic counterpart
PLACEBO = "#b8b8b8"

YEARS = np.arange(1988, 2018)
T0 = 2008                # first treated period
N_DONOR = 27
TRUE_W = {4: 0.40, 11: 0.26, 17: 0.21, 23: 0.13}
EFFECT = -26.0           # saturated effect, packs per capita per year

CODES = ["AL", "AZ", "AR", "CO", "CT", "DE", "GA", "ID", "IL", "IN", "IA",
         "KS", "KY", "LA", "ME", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
         "NM", "NC", "ND", "OH", "OK"]


# ---------------------------------------------------------------------------
def make_data(rng):
    """Interactive fixed-effects panel: common factors, unit loadings."""
    n_t = YEARS.size
    k = np.arange(n_t)
    factors = np.column_stack([
        np.ones(n_t),
        -0.95 * k + 0.010 * k ** 2,
        7.0 * np.sin(2.0 * np.pi * k / 11.0) + 2.4 * np.cos(k / 3.1),
        np.cumsum(rng.normal(0.0, 1.15, n_t)),
    ])
    load = np.column_stack([
        rng.normal(128.0, 17.0, N_DONOR),
        rng.uniform(0.55, 1.45, N_DONOR),
        rng.uniform(0.30, 1.30, N_DONOR),
        rng.uniform(0.40, 1.40, N_DONOR),
    ])
    donors = load @ factors.T + rng.normal(0.0, 2.6, (N_DONOR, n_t))

    w = np.zeros(N_DONOR)
    for j, v in TRUE_W.items():
        w[j] = v
    l_t = load.T @ w + np.array([2.4, 0.02, -0.03, 0.01])
    post = np.clip((YEARS - T0 + 1) / 3.0, 0.0, 1.0) * (YEARS >= T0)
    treated = (l_t @ factors.T + EFFECT * post
               + rng.normal(0.0, 2.6, n_t))
    return treated, donors


def fit_weights(target_pre, donors_pre, penalty=1e3):
    """Non-negative weights on the simplex, by penalised NNLS.

    Appending one heavily weighted row of ones drives the weights to sum to
    one without leaving the non-negative least-squares solver, and NNLS is
    what makes the solution genuinely sparse - a ridge or an unconstrained
    fit would spread tiny weights across the whole donor pool and the weight
    panel would say nothing.
    """
    a = np.vstack([donors_pre.T, penalty * np.ones((1, donors_pre.shape[0]))])
    b = np.concatenate([target_pre, [penalty]])
    w, _ = nnls(a, b)
    s = w.sum()
    return w / s if s > 0 else w


def rmspe(gap, mask):
    return float(np.sqrt(np.mean(gap[mask] ** 2)))


# ---------------------------------------------------------------------------
def path_panel(ax, treated, synth, w, pre, post, pre_rm, att):
    ax.axvspan(T0 - 0.5, YEARS[-1] + 0.5, facecolor=CARD, edgecolor="none",
               zorder=0)
    ax.fill_between(YEARS[post], synth[post], treated[post],
                    facecolor=TREAT, alpha=0.20, edgecolor="none", zorder=2,
                    label="treatment gap")
    ax.plot(YEARS, synth, lw=1.7, color=SYNTH, ls=(0, (4.6, 2.4)), zorder=4,
            label="synthetic control")
    ax.plot(YEARS, treated, lw=2.0, color=TREAT, zorder=5,
            label="treated unit (CA)")
    ax.axvline(T0 - 0.5, color="#3a3a3a", lw=1.1, ls=(0, (3.4, 2.2)),
               zorder=6)

    xg = YEARS[-1]
    ax.annotate("", xy=(xg + 0.30, treated[-1]), xytext=(xg + 0.30, synth[-1]),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.1,
                                shrinkA=0, shrinkB=0, mutation_scale=8),
                annotation_clip=False, zorder=7)
    ax.text(xg + 0.48, 0.5 * (treated[-1] + synth[-1]),
            f"gap {xg}\n{treated[-1] - synth[-1]:+.1f}", ha="left",
            va="center", fontsize=6.8, color=INK, clip_on=False)

    ax.text(T0 - 0.9, 0.965, "pre-intervention",
            transform=ax.get_xaxis_transform(), ha="right", va="top",
            fontsize=6.6, color=MUTED)
    ax.text(T0 - 0.1, 0.965, "post-intervention",
            transform=ax.get_xaxis_transform(), ha="left", va="top",
            fontsize=6.6, color=SUB)

    ax.set_xlim(YEARS[0] - 0.5, YEARS[-1] + 1.6)
    ax.set_xticks(YEARS[::4])
    ax.set_xlabel(f"year (intervention in {T0})", fontsize=8.0)
    ax.set_ylabel("per-capita cigarette sales (packs/year)", fontsize=8.0)
    ax.tick_params(labelsize=7.0, length=2.5)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_title("Two paths that agree for twenty years, then do not",
                 fontsize=9.0, fontweight="bold", pad=3.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", fontsize=7.0, frameon=False, handlelength=1.6,
              borderaxespad=0.4, labelspacing=0.32)

    nz = int((w > 0.005).sum())
    ax.text(0.975, 0.885,
            f"pre-intervention RMSPE {pre_rm:.2f} packs "
            f"({(YEARS < T0).sum()} years)\n"
            f"average post-intervention gap {att:+.1f} packs/year\n"
            f"{nz} of {N_DONOR} donors carry positive weight",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
            color=SUB, linespacing=1.5)


def weight_panel(ax, w, top=11):
    order = np.argsort(-w)[:top]
    ys = np.arange(top)[::-1]
    cols = [TREAT if w[j] > 0.005 else "#d0d0d0" for j in order]
    ax.barh(ys, w[order], height=0.66, color=cols, edgecolor="white",
            linewidth=0.5, zorder=3)
    for y, j in zip(ys, order):
        if w[j] > 0.005:
            ax.text(w[j] + 0.012, y, f"{w[j]:.3f}", ha="left", va="center",
                    fontsize=6.2, color=SUB)

    ax.set_yticks(ys)
    ax.set_yticklabels([CODES[j] for j in order], fontsize=6.5)
    ax.set_ylim(-0.7, top - 0.3)
    ax.set_xlim(0.0, max(w.max() * 1.30, 0.1))
    ax.set_xlabel("donor weight (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.5, length=2.2)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_title("Sparse weights: a handful of donors", fontsize=8.5,
                 fontweight="bold", pad=3.0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    rest = N_DONOR - top
    ax.text(0.985, 0.045, f"{rest} further donors: weight 0",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3,
            color=MUTED)


def placebo_panel(ax, gap, placebos, keep, p_rank, ratio):
    ax.axvspan(T0 - 0.5, YEARS[-1] + 0.5, facecolor=CARD, edgecolor="none",
               zorder=0)
    ax.axhline(0.0, color=BASE, lw=0.8, ls=(0, (3.0, 2.2)), zorder=1)
    for j, g in enumerate(placebos):
        ax.plot(YEARS, g, lw=0.7, color=PLACEBO,
                alpha=0.85 if keep[j] else 0.35, zorder=2)
    ax.plot(YEARS, gap, lw=1.9, color=TREAT, zorder=4)
    ax.axvline(T0 - 0.5, color="#3a3a3a", lw=0.9, ls=(0, (2.8, 2.0)),
               zorder=3)

    lo, hi = shared_limits([placebos[keep], gap], pad=0.08)
    ax.set_ylim(lo, hi)
    ax.set_xlim(YEARS[0] - 0.5, YEARS[-1] + 0.5)
    ax.set_xticks(YEARS[::6])
    ax.set_xlabel("year", fontsize=7.4)
    ax.set_ylabel("gap vs synthetic (packs/year)", fontsize=7.4)
    ax.tick_params(labelsize=6.5, length=2.2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_title(f"Placebo in space: rank p = {p_rank:.3f}", fontsize=8.5,
                 fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.025, 0.045,
            f"grey: {int(keep.sum())} donors with comparable pre-fit\n"
            f"treated post/pre RMSPE ratio {ratio:.1f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.3,
            color=SUB, linespacing=1.4)


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(773)
    treated, donors = make_data(rng)

    pre, post = YEARS < T0, YEARS >= T0
    w = fit_weights(treated[pre], donors[:, pre])
    synth = donors.T @ w
    gap = treated - synth
    pre_rm = rmspe(gap, pre)
    att = float(gap[post].mean())
    ratio = rmspe(gap, post) / pre_rm

    # Placebo in space: every donor takes a turn as the treated unit, with
    # the remaining donors as its pool.
    placebos, ratios = [], []
    for j in range(N_DONOR):
        pool = np.delete(donors, j, axis=0)
        wj = fit_weights(donors[j, pre], pool[:, pre])
        gj = donors[j] - pool.T @ wj
        placebos.append(gj)
        ratios.append(rmspe(gj, post) / max(rmspe(gj, pre), 1e-9))
    placebos = np.array(placebos)
    ratios = np.array(ratios)

    # Units the synthetic method could not fit before the intervention are
    # not evidence about anything; they are shown faintly, not deleted.
    keep = np.array([rmspe(g, pre) <= 5.0 * pre_rm for g in placebos])
    all_ratios = np.concatenate([[ratio], ratios[keep]])
    p_rank = float((all_ratios >= ratio).sum() / all_ratios.size)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (0.86, 0.82, 5.24, 4.52))
    ax_w = axes_inches(fig, (7.05, 3.48, 3.05, 1.86))
    ax_p = axes_inches(fig, (7.05, 0.82, 3.05, 1.86))

    path_panel(ax_main, treated, synth, w, pre, post, pre_rm, att)
    weight_panel(ax_w, w)
    placebo_panel(ax_p, gap, placebos, keep, p_rank, ratio)

    n_more = int((all_ratios >= ratio).sum()) - 1

    for ax, lab in ((ax_main, "(a)"), (ax_w, "(b)"), (ax_p, "(c)")):
        panel_label_inches(fig, ax, lab, dx=-0.62, dy=0.10)
    savefig(fig, "773_ci_synthetic_control.png")


if __name__ == "__main__":
    main()

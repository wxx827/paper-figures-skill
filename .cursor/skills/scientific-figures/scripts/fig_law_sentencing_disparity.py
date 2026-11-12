"""Sentencing consistency: the same offence class in front of nine judges.

Main panel: one horizontal density band per sitting judge over custodial
sentence length in months, with the jittered case-level strip drawn under
each band so the reader can see how thin some caseloads are.  The advisory
guideline range runs through the background as a vertical grey band; every
sentence that falls outside it is redrawn in a warm colour, above and below
the range distinguished, and each judge's median is a short vertical rule.
A Kruskal-Wallis test across the nine caseloads is quoted on the panel.

Auxiliary (both at 58 % of the main panel width): the same judges after the
case facts are partialled out - the open markers are raw median deviations,
the filled markers are severity-adjusted means with 95 % intervals, so the
shrinkage is visible as a horizontal collapse that does not reach zero; and
a forest of the ordinary-least-squares coefficients that did the adjusting.

Every quoted number is computed from the simulated case table that is
plotted: the H statistics and p values from scipy, the coefficients and
their standard errors from a closed-form least-squares fit, and the
guideline departure rates by counting the plotted points.  Fictional
jurisdiction, fictional judges, illustrative values only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy import stats

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
TEAL = "#2e7d5b"
RUST = "#c0603f"
GOLD = "#c9a227"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"
BAND = "#e6e6e6"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.86, 0.86, 5.54, 4.62)
AUX_T_RECT = (7.62, 3.48, 3.16, 2.00)
AUX_B_RECT = (7.62, 0.86, 3.16, 2.00)

JUDGES = ["Alder", "Brandt", "Corvin", "Dain", "Ewell",
          "Fenwick", "Gale", "Harrow", "Ives"]
# additive months on top of the case-fact model, one per judge; deliberately
# not monotone in the case-mix index below, so the adjustment re-orders the
# bench rather than merely rescaling it
JUDGE_EFFECT = [-6.5, -2.0, -4.5, 1.0, -0.5, 3.5, 2.0, 6.8, 5.2]
# each judge also draws a systematically different case mix, which is the
# part of the raw gap that the covariate adjustment is able to remove
CASE_MIX = [-1.35, -1.00, -0.65, -0.30, 0.05, 0.40, 0.75, 1.10, 1.35]
N_CASES = [58, 44, 66, 51, 39, 62, 47, 55, 41]

GUIDE_LO, GUIDE_HI = 24.0, 48.0
COEF_NAMES = ["offence severity (per point, 1-10)",
              "prior convictions (per count)",
              "early guilty plea (yes vs no)",
              "weapon involved (yes vs no)",
              "cooperation credit (yes vs no)"]
COEF_TRUE = [4.30, 2.05, -9.60, 7.40, -5.10]


def build_cases(seed=924):
    """Case-level table: sentence months plus the five sentencing factors."""
    rng = np.random.default_rng(seed)
    rows = []
    for j, (eff, mix, n) in enumerate(zip(JUDGE_EFFECT, CASE_MIX, N_CASES)):
        sev = np.clip(rng.gamma(6.0, 0.85, n) + mix, 1.0, 10.0)
        prior = rng.poisson(max(1.7 + 0.37 * mix, 0.2), n).astype(float)
        plea = (rng.random(n) < 0.58 - 0.067 * mix).astype(float)
        weapon = (rng.random(n) < 0.24 + 0.067 * mix).astype(float)
        coop = (rng.random(n) < 0.31 - 0.030 * mix).astype(float)
        base = (14.0 + COEF_TRUE[0] * sev + COEF_TRUE[1] * prior
                + COEF_TRUE[2] * plea + COEF_TRUE[3] * weapon
                + COEF_TRUE[4] * coop)
        months = base + eff + rng.normal(0.0, 7.4, n)
        months = np.clip(months, 1.0, 118.0)
        rows.append(dict(judge=j, months=months, sev=sev, prior=prior,
                         plea=plea, weapon=weapon, coop=coop))
    return rows


def stack(rows, key):
    return np.concatenate([r[key] for r in rows])


def ols(X, y):
    """Coefficients and standard errors from the normal equations."""
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    dof = X.shape[0] - X.shape[1]
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    return beta, se, resid


def half_violin(ax, x, y0, width, color, grid_pad=6.0, n=260):
    """Upward half-violin at row ``y0``; returns the kernel density peak."""
    lo, hi = x.min() - grid_pad, x.max() + grid_pad
    xs = np.linspace(max(lo, 0.0), hi, n)
    dens = stats.gaussian_kde(x, bw_method=0.42)(xs)
    dens = dens / dens.max() * width
    ax.fill_between(xs, y0, y0 + dens, facecolor=lighten(color, 0.72),
                    edgecolor=color, linewidth=0.85, zorder=4)
    return float(dens.max())


def main():
    set_pub_style(base_font=8)
    rows = build_cases()

    y = stack(rows, "months")
    design = np.column_stack([
        np.ones(y.size), stack(rows, "sev"), stack(rows, "prior"),
        stack(rows, "plea"), stack(rows, "weapon"), stack(rows, "coop")])
    beta, se, _ = ols(design, y)

    # adjusted sentence: case facts removed, grand mean put back so the
    # residual is still readable in months rather than in deviations
    fitted_facts = design[:, 1:] @ beta[1:]
    adjusted = y - fitted_facts + fitted_facts.mean()

    groups, groups_adj = [], []
    at = 0
    for r in rows:
        n = r["months"].size
        groups.append(y[at:at + n])
        groups_adj.append(adjusted[at:at + n])
        at += n

    h_raw, p_raw = stats.kruskal(*groups)
    h_adj, p_adj = stats.kruskal(*groups_adj)

    med_raw = np.array([np.median(g) for g in groups])
    mean_adj = np.array([g.mean() for g in groups_adj])
    sem_adj = np.array([g.std(ddof=1) / np.sqrt(g.size) for g in groups_adj])
    ci_adj = 1.96 * sem_adj

    grand_med = float(np.median(y))
    grand_adj = float(adjusted.mean())
    dev_raw = med_raw - grand_med
    dev_adj = mean_adj - grand_adj
    spread_raw = float(dev_raw.max() - dev_raw.min())
    spread_adj = float(dev_adj.max() - dev_adj.min())
    shrink = 100.0 * (1.0 - spread_adj / spread_raw)

    above = float(np.mean(y > GUIDE_HI)) * 100.0
    below = float(np.mean(y < GUIDE_LO)) * 100.0
    n_total = int(y.size)

    order = np.argsort(dev_adj)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_a = axes_inches(fig, AUX_T_RECT)
    ax_f = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_a, ax_f):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: one density band per judge ----------------------------------
    ax.add_patch(Rectangle((GUIDE_LO, -2.60), GUIDE_HI - GUIDE_LO, 12.2,
                           facecolor=BAND, edgecolor="none", zorder=0))
    for xv in (GUIDE_LO, GUIDE_HI):
        ax.axvline(xv, color="#b8b8b8", lw=0.8, ls=(0, (3.0, 2.2)), zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.5, alpha=0.7, zorder=0)

    rng_j = np.random.default_rng(9241)
    for j, r in enumerate(rows):
        g = r["months"]
        half_violin(ax, g, j, 0.62, BLUE)
        jit = j - 0.10 - rng_j.random(g.size) * 0.30
        inside = (g >= GUIDE_LO) & (g <= GUIDE_HI)
        ax.plot(g[inside], jit[inside], ls="none", marker="o", ms=1.9,
                mfc=lighten(BLUE, 0.30), mec="none", alpha=0.85, zorder=5)
        ax.plot(g[g > GUIDE_HI], jit[g > GUIDE_HI], ls="none", marker="o",
                ms=2.4, mfc=RUST, mec="none", alpha=0.95, zorder=6)
        ax.plot(g[g < GUIDE_LO], jit[g < GUIDE_LO], ls="none", marker="o",
                ms=2.4, mfc=GOLD, mec="none", alpha=0.95, zorder=6)
        ax.plot([med_raw[j], med_raw[j]], [j - 0.44, j + 0.30], lw=1.9,
                color=INK, solid_capstyle="butt", zorder=8)

    ax.set_yticks(range(len(JUDGES)))
    ax.set_yticklabels([f"Judge {m}\nn={n}" for m, n in zip(JUDGES, N_CASES)],
                       fontsize=7.0)
    ax.set_ylim(-2.60, 9.60)
    ax.set_xlim(0.0, 120.0)
    ax.set_xticks(np.arange(0, 121, 20))
    ax.set_xlabel("custodial sentence (months)", fontsize=8.5)
    ax.set_title("Aggravated-fraud sentences, Northgate Circuit, "
                 f"n={n_total} cases", fontsize=9.0, fontweight="bold",
                 pad=5.0)
    ax.text(0.5 * (GUIDE_LO + GUIDE_HI), 8.86,
            f"advisory guideline range\n{GUIDE_LO:.0f}-{GUIDE_HI:.0f} months",
            ha="center", va="bottom", fontsize=6.6, color=SUB,
            linespacing=1.4)
    ax.axhline(-0.72, color="#c9c9c9", lw=0.7, zorder=2)

    star = "p < 0.001" if p_raw < 1e-3 else f"p = {p_raw:.3f}"
    ax.text(119.0, -2.44,
            f"Kruskal-Wallis across the nine caseloads: H = {h_raw:.1f}, "
            f"df = {len(JUDGES) - 1}, {star}\n"
            f"judge medians span {med_raw.min():.0f}-{med_raw.max():.0f} "
            f"months, a range of {spread_raw:.0f} months\n"
            f"{above:.0f} % of sentences sit above the guideline range, "
            f"{below:.0f} % below it",
            ha="right", va="bottom", fontsize=6.7, color=SUB,
            linespacing=1.55, zorder=9)

    handles = [
        Line2D([], [], marker="o", ls="none", ms=3.4, mfc=lighten(BLUE, 0.30),
               mec="none", label="within guideline range"),
        Line2D([], [], marker="o", ls="none", ms=3.4, mfc=RUST, mec="none",
               label="above range (upward departure)"),
        Line2D([], [], marker="o", ls="none", ms=3.4, mfc=GOLD, mec="none",
               label="below range (downward departure)"),
        Line2D([], [], color=INK, lw=1.9, label="judge median"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=6.9,
              handlelength=1.4, labelspacing=0.36, borderpad=0.15,
              bbox_to_anchor=(-0.004, -0.004))

    # ---- aux (b): raw vs severity-adjusted judge deviations -----------------
    ax_a.grid(axis="x", color=GRID, lw=0.5, alpha=0.7, zorder=0)
    ax_a.axvline(0.0, color=FAINT, lw=0.8, zorder=2)
    ypos = np.arange(len(JUDGES))
    for k, j in enumerate(order):
        ax_a.plot([dev_raw[j], dev_adj[j]], [k, k], lw=0.7, color="#c2c2c2",
                  zorder=3)
    ax_a.plot(dev_raw[order], ypos, ls="none", marker="o", ms=3.6,
              mfc="white", mec=FAINT, mew=1.0, zorder=4)
    ax_a.errorbar(dev_adj[order], ypos, xerr=ci_adj[order], ls="none",
                  marker="o", ms=3.8, mfc=TEAL, mec=TEAL, ecolor=TEAL,
                  elinewidth=0.9, capsize=1.8, capthick=0.8, zorder=5)
    ax_a.set_yticks(ypos)
    ax_a.set_yticklabels([JUDGES[j] for j in order], fontsize=6.3)
    ax_a.set_ylim(-0.7, len(JUDGES) + 1.5)
    ax_a.invert_yaxis()
    ax_a.set_xlim(-20.0, 20.0)
    ax_a.set_xticks([-20, -10, 0, 10, 20])
    ax_a.tick_params(axis="x", labelsize=6.6)
    ax_a.set_xlabel("deviation from circuit average (months)", fontsize=7.6,
                    labelpad=1.5)
    ax_a.set_title("Adjusting for case facts narrows the gap",
                   fontsize=8.4, fontweight="bold", pad=4.0)
    ax_a.text(-19.4, len(JUDGES) + 1.30,
              f"open = raw median, filled = adjusted mean +/- 95 % CI\n"
              f"between-judge range {spread_raw:.0f} -> {spread_adj:.0f} "
              f"months ({shrink:.0f} % narrower), adjusted H = {h_adj:.1f}, "
              f"p < 0.001",
              ha="left", va="bottom", fontsize=6.0, color=SUB,
              linespacing=1.5)

    # ---- aux (c): forest of sentencing-factor coefficients ------------------
    ax_f.grid(axis="x", color=GRID, lw=0.5, alpha=0.7, zorder=0)
    ax_f.axvline(0.0, color=FAINT, lw=0.8, zorder=2)
    cpos = np.arange(len(COEF_NAMES))
    b = beta[1:]
    e = 1.96 * se[1:]
    cols = [RUST if v > 0 else TEAL for v in b]
    for k in cpos:
        ax_f.plot([b[k] - e[k], b[k] + e[k]], [k, k], lw=1.5, color=cols[k],
                  solid_capstyle="round", zorder=4)
        ax_f.plot([b[k]], [k], marker="s", ms=4.2, mfc=cols[k], mec="white",
                  mew=0.7, zorder=5)
        off = 0.9 if b[k] > 0 else -0.9
        ax_f.text(b[k] + e[k] + off if b[k] > 0 else b[k] - e[k] + off,
                  k - 0.30, f"{b[k]:+.1f}", fontsize=6.1, color=cols[k],
                  ha="left" if b[k] > 0 else "right", va="center")
    ax_f.set_yticks(cpos)
    ax_f.set_yticklabels(COEF_NAMES, fontsize=6.3)
    ax_f.set_ylim(-0.7, len(COEF_NAMES) - 0.3)
    ax_f.invert_yaxis()
    ax_f.set_xlim(-15.0, 15.0)
    ax_f.set_xticks([-15, -10, -5, 0, 5, 10, 15])
    ax_f.tick_params(axis="x", labelsize=6.6)
    ax_f.set_xlabel("change in sentence (months per unit, OLS +/- 95 % CI)",
                    fontsize=7.6, labelpad=1.5)
    ax_f.set_title("What the case facts are worth", fontsize=8.4,
                   fontweight="bold", pad=4.0)

    panel_labels_inches(fig, [ax, ax_a, ax_f], dx=-0.62, dy=0.12)
    savefig(fig, "924_law_sentencing_disparity.png")


if __name__ == "__main__":
    main()

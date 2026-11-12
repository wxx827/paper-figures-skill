"""Figure 620 - Defect density against component complexity.

One synthetic component table feeds all three panels.  Panel (a) plots defect
density against cyclomatic complexity with bubble area set by component size,
a least-squares trend and its 95% confidence band, and the high-risk quadrant
(both axes above their 70th percentile) shaded behind everything.  The defect
counts in panels (b) and (c) are the same table integrated two different
ways: cumulative discovery and repair over the release train, and the
severity mix per component family.

Reference-gallery figure: the numbers are generated, the layout is the point.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy import stats

from pubstyle import set_pub_style, savefig, COLORS, lighten, darken
from figkit import panel_label_inches

ACCENT = "#b23a48"          # house red - risk quadrant and defect counts
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
FAMILIES = ["service", "interface", "data access", "library"]
FAM_COLORS = [BLUE, "#8e6bbf", GREEN, "#c9a227"]
SEVERITIES = ["blocker", "major", "minor"]
SEV_COLORS = [darken(ACCENT, 0.32), ACCENT, lighten(ACCENT, 0.52)]
N_SPRINT = 16


def build(rng, n=76):
    fam = rng.integers(0, 4, n)
    cx = np.exp(rng.normal(2.72, 0.46, n))                 # cyclomatic
    kloc = 0.4 + 2.6 * rng.gamma(2.0, 0.55, n)
    lift = np.array([1.00, 1.18, 0.86, 0.68])[fam]
    dens = lift * (0.30 + 0.128 * cx) + rng.normal(0.0, 0.85, n)
    dens = np.clip(dens, 0.08, None)
    defects = np.maximum(np.round(dens * kloc), 1.0)
    mix = rng.dirichlet([1.1, 3.0, 4.2], n)
    sev = np.round(defects[:, None] * mix)
    return fam, cx, kloc, dens, defects, sev


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(620)
    fam, cx, kloc, dens, defects, sev = build(rng)
    n = cx.size

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58],
                          left=0.062, right=0.965, top=0.885, bottom=0.125,
                          wspace=0.32, hspace=0.52)
    ax = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[1, 1])

    # ---- (a) density versus complexity ------------------------------------
    x_hi = float(np.quantile(cx, 0.70))
    y_hi = float(np.quantile(dens, 0.70))
    risky = (cx > x_hi) & (dens > y_hi)
    x_lim = (0.0, cx.max() * 1.10)
    y_lim = (0.0, dens.max() * 1.14)

    ax.add_patch(Rectangle((x_hi, y_hi), x_lim[1] - x_hi, y_lim[1] - y_hi,
                           facecolor=ACCENT, alpha=0.075, edgecolor="none",
                           zorder=1))
    ax.axvline(x_hi, color=COLORS["baseline"], lw=0.7, ls=(0, (3.5, 2.0)),
               zorder=2)
    ax.axhline(y_hi, color=COLORS["baseline"], lw=0.7, ls=(0, (3.5, 2.0)),
               zorder=2)
    ax.text(x_lim[1] * 0.985, y_lim[1] * 0.975,
            "high-risk quadrant\n%d of %d components" % (int(risky.sum()), n),
            ha="right", va="top", fontsize=6.8, color=ACCENT,
            linespacing=1.35, zorder=8)

    coef = np.polyfit(cx, dens, 1)
    xs = np.linspace(x_lim[0], x_lim[1], 200)
    yhat = np.polyval(coef, xs)
    resid = dens - np.polyval(coef, cx)
    s_err = np.sqrt((resid ** 2).sum() / (n - 2))
    sxx = ((cx - cx.mean()) ** 2).sum()
    band = stats.t.ppf(0.975, n - 2) * s_err * np.sqrt(
        1.0 / n + (xs - cx.mean()) ** 2 / sxx)
    r2 = np.corrcoef(cx, dens)[0, 1] ** 2

    ax.fill_between(xs, yhat - band, yhat + band, facecolor="#d5d5d5",
                    alpha=0.45, lw=0.0, zorder=3)
    ax.plot(xs, yhat, color="#1a1a1a", lw=1.5, zorder=6)

    for f, col in enumerate(FAM_COLORS):
        m = fam == f
        ax.scatter(cx[m], dens[m], s=7.0 * kloc[m] + 5.0, facecolor=col,
                   alpha=0.70, edgecolor="white", lw=0.5, zorder=5,
                   label=FAMILIES[f])

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_xlabel("cyclomatic complexity (-)", fontsize=8.0)
    ax.set_ylabel("defect density (defects/KLOC)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.grid(True, lw=0.45, alpha=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.025, 0.965,
            "density = %.3f x complexity + %.2f\n$R^2$ = %.2f, n = %d "
            "components" % (coef[0], coef[1], r2, n),
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
            color="#1a1a1a", linespacing=1.35, zorder=9,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#cccccc",
                      lw=0.6, alpha=0.92))
    ax.legend(loc="lower right", frameon=False, fontsize=6.8,
              handletextpad=0.35, borderpad=0.25, labelspacing=0.35,
              title="component family", title_fontsize=6.8)
    ax.set_title("Defect density versus component complexity", fontsize=8.8,
                 fontweight="bold", pad=4.0)

    # ---- (b) discovery and repair burn-up ---------------------------------
    total = float(defects.sum())
    sp = np.arange(N_SPRINT + 1)
    found = total * (1.0 - np.exp(-0.255 * sp))
    fixed = total * (1.0 - np.exp(-0.176 * np.maximum(sp - 1.4, 0.0)))
    found[0], fixed[0] = 0.0, 0.0
    found = found / found[-1] * total
    open_b = found - fixed

    ax_b.fill_between(sp, fixed, found, facecolor=ACCENT, alpha=0.18,
                      lw=0.0, zorder=2, label="open backlog")
    ax_b.plot(sp, found, color=BLUE, lw=1.6, zorder=4, label="found")
    ax_b.plot(sp, fixed, color=GREEN, lw=1.6, zorder=4, label="fixed")
    k = int(np.argmax(open_b))
    ax_b.annotate("peak backlog %d" % int(round(open_b[k])),
                  xy=(sp[k], 0.5 * (found[k] + fixed[k])),
                  xytext=(sp[k] + 2.4, 0.30 * total), fontsize=6.4,
                  color=ACCENT, ha="left", va="center",
                  arrowprops=dict(arrowstyle="-", lw=0.6, color=ACCENT))
    ax_b.set_xlim(0, N_SPRINT)
    ax_b.set_ylim(0, total * 1.08)
    ax_b.set_xlabel("sprint (-)", fontsize=7.4)
    ax_b.set_ylabel("cumulative defects (-)", fontsize=7.4)
    ax_b.tick_params(labelsize=6.6)
    ax_b.grid(True, lw=0.45, alpha=0.5)
    ax_b.set_axisbelow(True)
    for s in ("top", "right"):
        ax_b.spines[s].set_visible(False)
    ax_b.legend(loc="lower right", frameon=False, fontsize=6.4,
                handlelength=1.4, handletextpad=0.4, borderpad=0.2)
    ax_b.set_title("%d defects closed by sprint %d"
                   % (int(round(fixed[-1])), N_SPRINT), fontsize=8.5,
                   fontweight="bold", pad=3.5)

    # ---- (c) severity mix per family --------------------------------------
    counts = np.array([[sev[fam == f, s].sum() for s in range(3)]
                       for f in range(4)])
    yy = np.arange(4)[::-1]
    left = np.zeros(4)
    for s in range(3):
        ax_s.barh(yy, counts[:, s], height=0.56, left=left,
                  color=SEV_COLORS[s], edgecolor="white", lw=0.5,
                  label=SEVERITIES[s], zorder=3)
        left = left + counts[:, s]
    for y, tot in zip(yy, counts.sum(axis=1)):
        ax_s.text(tot + counts.sum(axis=1).max() * 0.015, y, "%d" % tot,
                  va="center", fontsize=6.2, color=COLORS["sub"])
    ax_s.set_yticks(yy)
    ax_s.set_yticklabels(FAMILIES, fontsize=6.6)
    ax_s.set_xlim(0, counts.sum(axis=1).max() * 1.16)
    ax_s.set_xlabel("defects (-)", fontsize=7.4)
    ax_s.tick_params(labelsize=6.6)
    ax_s.grid(axis="x", lw=0.45, alpha=0.5)
    ax_s.set_axisbelow(True)
    for s in ("top", "right"):
        ax_s.spines[s].set_visible(False)
    ax_s.legend(loc="lower right", frameon=False, fontsize=6.4, ncol=1,
                handlelength=1.1, handletextpad=0.35, labelspacing=0.32,
                borderpad=0.2)
    blocker = counts[:, 0].sum() / counts.sum()
    ax_s.set_title("Blockers are %.0f%% of all defects" % (100 * blocker),
                   fontsize=8.5, fontweight="bold", pad=3.5)

    for a, lab in ((ax, "(a)"), (ax_b, "(b)"), (ax_s, "(c)")):
        panel_label_inches(fig, a, lab, dx=-0.42, dy=0.09)


    savefig(fig, "620_swe_defect_density.png")


if __name__ == "__main__":
    main()

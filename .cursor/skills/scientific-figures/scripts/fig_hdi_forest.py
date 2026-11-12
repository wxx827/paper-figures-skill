"""Forest plot of posterior HDIs for a hierarchical treatment-effect model.

Every row carries a small ridgeline of the marginal posterior, a thick
50% HDI bar, a thin 95% HDI whisker and the posterior median.  Rows are
coloured by whether the 95% HDI excludes zero, and the numeric read-out
(median, 95% HDI, posterior probability of a positive effect) is set in a
matching column on the right.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from pubstyle import set_pub_style, savefig, style_axes

POS_C = "#2a7f7a"
NEG_C = "#b23a48"
NULL_C = "#8a8f96"
POP_C = "#3d5a8a"


def hdi(samples, prob):
    s = np.sort(samples)
    k = int(np.floor(prob * s.size))
    i = int(np.argmin(s[k:] - s[:s.size - k]))
    return s[i], s[i + k]


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(11235)

    sites = ["Aarhus", "Basel", "Cardiff", "Dresden", "Eindhoven",
             "Florence", "Ghent", "Helsinki", "Innsbruck", "Jena",
             "Krakow", "Lyon"]
    n_draw = 8000
    mu_true, tau_true = 0.34, 0.42
    theta = rng.normal(mu_true, tau_true, len(sites))
    se = rng.uniform(0.16, 0.46, len(sites))

    rows = []
    for name, th, s in zip(sites, theta, se):
        rows.append((name, rng.normal(th, s, n_draw), "site"))
    rows.append((r"$\mu$  (population mean)",
                 rng.normal(mu_true, 0.13, n_draw), "pop"))
    rows.append((r"$\tau$  (between-site SD)",
                 np.exp(rng.normal(np.log(tau_true), 0.22, n_draw)), "pop"))

    sites_sorted = sorted([r for r in rows if r[2] == "site"],
                          key=lambda r: np.median(r[1]))
    pops = [r for r in rows if r[2] == "pop"]
    ordered = sites_sorted + pops
    n_row = len(ordered)

    fig = plt.figure(figsize=(7.4, 6.4))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 0.46], wspace=0.02,
                  left=0.145, right=0.985, top=0.845, bottom=0.088)
    ax = fig.add_subplot(gs[0, 0])
    axt = fig.add_subplot(gs[0, 1], sharey=ax)
    style_axes(ax)
    axt.axis("off")

    ax.axvline(0, color="#666666", lw=0.9, ls=(0, (4, 3)), zorder=1)

    ypos, labels = [], []
    for i, (name, draws, kind) in enumerate(ordered):
        y = n_row - 1 - i
        ypos.append(y)
        labels.append(name)
        lo95, hi95 = hdi(draws, 0.95)
        lo50, hi50 = hdi(draws, 0.50)
        med = float(np.median(draws))
        if kind == "pop":
            col = POP_C
        elif lo95 > 0:
            col = POS_C
        elif hi95 < 0:
            col = NEG_C
        else:
            col = NULL_C

        if i % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#f4f5f7", lw=0, zorder=0)
            axt.axhspan(y - 0.5, y + 0.5, color="#f4f5f7", lw=0, zorder=0)

        # ridgeline of the marginal posterior
        gx = np.linspace(np.percentile(draws, 0.2),
                         np.percentile(draws, 99.8), 300)
        gd = gaussian_kde(draws)(gx)
        gd = 0.62 * gd / gd.max()
        ax.fill_between(gx, y, y + gd, color=col, alpha=0.16, lw=0, zorder=2)
        ax.plot(gx, y + gd, color=col, lw=0.7, alpha=0.75, zorder=3)

        ax.plot([lo95, hi95], [y, y], color=col, lw=1.0, solid_capstyle="butt",
                zorder=4)
        ax.plot([lo50, hi50], [y, y], color=col, lw=3.4,
                solid_capstyle="butt", zorder=5)
        ax.plot([med], [y], marker="o", ms=4.2, mfc="white", mec=col,
                mew=1.2, zorder=6)

        axt.text(0.07, y, f"{med:+.2f}", ha="right", va="center",
                 fontsize=7.4, color="#1a1a1a", fontweight="bold")
        axt.text(0.13, y, f"[{lo95:+.2f}, {hi95:+.2f}]", ha="left",
                 va="center", fontsize=7.2, color="#444444")
        # a positivity probability is only meaningful for effect parameters
        if name.startswith(r"$\tau$"):
            axt.text(0.99, y, "--", ha="right", va="center", fontsize=7.2,
                     color="#999999")
        else:
            axt.text(0.99, y, f"{float(np.mean(draws > 0)):.2f}", ha="right",
                     va="center", fontsize=7.2, color=col, fontweight="bold")

    # separator between site-level and population-level rows
    ysep = n_row - len(sites_sorted) - 0.5
    for a in (ax, axt):
        a.axhline(ysep, color="#bbbbbb", lw=0.8, zorder=7)

    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0 - 0.42, x1)
    for ytop, txt in ((n_row - 0.55, "SITE LEVEL"),
                      (ysep - 0.05, "POPULATION LEVEL")):
        ax.text(x0 - 0.36, ytop, txt, ha="left", va="top", fontsize=6.4,
                color="#8a8f96", fontweight="bold", zorder=8)

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(-0.7, n_row - 0.15)
    ax.set_xlabel("standardised treatment effect  " + r"$\theta_j$"
                  "   (posterior)", labelpad=2)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    axt.set_xlim(0, 1)

    head_y = n_row - 0.32
    axt.text(0.07, head_y, "median", ha="right", va="center", fontsize=7.0,
             color="#666666", fontweight="bold")
    axt.text(0.13, head_y, "95% HDI", ha="left", va="center", fontsize=7.0,
             color="#666666", fontweight="bold")
    axt.text(0.99, head_y, r"$\Pr(\theta > 0)$", ha="right", va="center",
             fontsize=7.0, color="#666666", fontweight="bold")

    handles = [
        plt.Line2D([], [], color="#555555", lw=3.4, label="50% HDI"),
        plt.Line2D([], [], color="#555555", lw=1.0, label="95% HDI"),
        plt.Line2D([], [], color="none", marker="o", ms=4.4, mfc="white",
                   mec="#555555", mew=1.2, label="posterior median"),
        plt.Line2D([], [], color=POS_C, lw=5, alpha=0.35,
                   label="95% HDI excludes 0"),
        plt.Line2D([], [], color=NULL_C, lw=5, alpha=0.35,
                   label="95% HDI includes 0"),
        plt.Line2D([], [], color=POP_C, lw=5, alpha=0.35,
                   label="population parameter"),
    ]

    savefig(fig, "131_hdi_forest.png")


if __name__ == "__main__":
    main()

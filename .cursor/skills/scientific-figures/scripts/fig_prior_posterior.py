"""Prior-to-posterior updating for six model parameters (2 x 3 grid).

Each panel overlays the prior (dashed outline, pale fill) and the posterior
(solid outline, saturated fill), reports both means with their central 95%
intervals, and quantifies the information gained by the data as the KL
divergence KL(posterior || prior) in bits plus the interval shrinkage.

All densities are analytic (scipy), so the reported numbers are exact.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy import stats
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

PRIOR_C = "#9a9a9a"
PRIOR_FILL = "#e8e8e8"
POST_C = "#1f5f8b"
POST_FILL = "#9cc3dd"
ACCENT = "#c0603f"


def summarise(dist, grid):
    pdf = dist.pdf(grid)
    lo, hi = dist.ppf(0.025), dist.ppf(0.975)
    return pdf, dist.mean(), lo, hi


def kl_bits(post, prior, lo, hi, n=4000):
    """KL(posterior || prior) evaluated numerically, in bits."""
    x = np.linspace(lo, hi, n)
    q = post.pdf(x)
    p = np.clip(prior.pdf(x), 1e-300, None)
    m = q > 1e-12
    return float(np.trapezoid(q[m] * np.log2(q[m] / p[m]), x[m]))


def main():
    set_pub_style(base_font=8)

    # (label, description, prior, posterior, display window)
    panels = [
        (r"$\alpha$", "intercept",
         stats.norm(0.0, 2.5), stats.norm(2.41, 0.20), (-6.4, 6.4)),
        (r"$\beta_{\mathrm{dose}}$", "dose slope",
         stats.norm(0.0, 1.5), stats.norm(-0.68, 0.14), (-4.0, 4.0)),
        (r"$\beta_{\mathrm{age}}$", "age slope",
         stats.norm(0.0, 1.5), stats.norm(0.42, 0.22), (-4.0, 4.0)),
        (r"$\sigma$", "residual scale",
         stats.halfnorm(0, 2.0), stats.lognorm(0.14, scale=0.83), (0, 4.0)),
        (r"$\tau$", "group-level SD",
         stats.gamma(1.4, scale=0.9), stats.lognorm(0.24, scale=0.46),
         (0, 3.6)),
        (r"$\theta$", "response rate",
         stats.beta(2, 2), stats.beta(48, 22), (0, 1)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.75))
    fig.subplots_adjust(left=0.045, right=0.988, top=0.845, bottom=0.135,
                        hspace=0.52, wspace=0.15)

    for k, (name, desc, prior, post, (x0, x1)) in enumerate(panels):
        ax = axes.ravel()[k]
        style_axes(ax)
        grid = np.linspace(x0, x1, 800)
        p_pdf, p_mu, p_lo, p_hi = summarise(prior, grid)
        q_pdf, q_mu, q_lo, q_hi = summarise(post, grid)

        ax.fill_between(grid, 0, p_pdf, color=PRIOR_FILL, lw=0, zorder=1)
        ax.plot(grid, p_pdf, color=PRIOR_C, lw=1.0, ls=(0, (4, 2)), zorder=3)
        ax.fill_between(grid, 0, q_pdf, color=POST_FILL, alpha=0.85, lw=0,
                        zorder=2)
        ax.plot(grid, q_pdf, color=POST_C, lw=1.5, zorder=4)

        ymax = float(q_pdf.max())
        ax.set_ylim(0, ymax * 1.55)
        ax.set_xlim(x0, x1)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(labelsize=6.8)

        # shift arrow: prior centre -> posterior centre
        yarrow = ymax * 1.06
        if abs(q_mu - p_mu) > 0.04 * (x1 - x0):
            ax.add_patch(FancyArrowPatch(
                (p_mu, yarrow), (q_mu, yarrow), transform=ax.transData,
                arrowstyle="-|>", mutation_scale=7, lw=0.9, color=ACCENT,
                shrinkA=0, shrinkB=1.5, zorder=6))
        ax.plot([p_mu], [yarrow], marker="o", ms=2.6, color=ACCENT, zorder=6)

        shrink = 1.0 - (q_hi - q_lo) / (p_hi - p_lo)
        kl = kl_bits(post, prior, q_lo - 6 * (q_hi - q_lo),
                     q_hi + 6 * (q_hi - q_lo))

        # place the read-out on the emptier side of the panel
        right_side = (q_mu - x0) < 0.5 * (x1 - x0)
        tx, ha = (0.985, "right") if right_side else (0.015, "left")
        ax.text(tx, 0.995, f"prior  {p_mu:.2f}  [{p_lo:.2f}, {p_hi:.2f}]",
                transform=ax.transAxes, ha=ha, va="top", fontsize=6.1,
                color="#7a7a7a")
        ax.text(tx, 0.885, f"post.  {q_mu:.2f}  [{q_lo:.2f}, {q_hi:.2f}]",
                transform=ax.transAxes, ha=ha, va="top", fontsize=6.1,
                color=POST_C, fontweight="bold")
        ax.text(tx, 0.775,
                f"shrinkage {shrink * 100:.0f}%  |  KL {kl:.1f} bits",
                transform=ax.transAxes, ha=ha, va="top", fontsize=6.1,
                color=ACCENT)

        ax.set_xlabel(f"{name}   ({desc})", fontsize=8.2, labelpad=2)
        add_panel_label(ax, f"({'abcdef'[k]})", x=-0.02, y=1.03, size=9)

    handles = [
        plt.Line2D([], [], color=PRIOR_C, lw=1.2, ls=(0, (4, 2)),
                   label="prior density"),
        plt.Line2D([], [], color=POST_C, lw=1.6, label="posterior density"),
        plt.Line2D([], [], color=ACCENT, lw=1.0, marker=">", ms=4,
                   label="shift of the central estimate"),
    ]

    savefig(fig, "129_prior_posterior.png")


if __name__ == "__main__":
    main()

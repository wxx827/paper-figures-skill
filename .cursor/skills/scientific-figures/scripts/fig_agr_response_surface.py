"""Figure 403 - Water x nitrogen response surface, and why the two optima differ.

A 5 x 5 factorial of nitrogen rate against irrigation depth is fitted with the
usual full quadratic; everything downstream - the surface, the floor contours,
both optima, the two marginal slices and every number in the annotations -
comes out of that one least-squares fit, so changing the seed moves all of them
together.

The point of the figure is the gap between the two stars.  The agronomic
optimum sits where the surface stops rising; the economic optimum sits where
the surface stops rising *faster than the inputs cost*, which is always at a
lower rate.  On this fit that is a large saving in nitrogen for a yield penalty
too small to see on the surface.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D             # noqa: F401  (projection)

from pubstyle import (set_pub_style, savefig, get_cmap, lighten, style_axes,
                      COLORS)
from figkit import safe_colorbar, panel_label_inches, apply_shared_limits

SEED = 31
N_MAX, W_MAX = 320.0, 400.0          # kg N/ha, mm of irrigation
N_LEV = np.array([0.0, 80.0, 160.0, 240.0, 320.0])
W_LEV = np.array([0.0, 100.0, 200.0, 300.0, 400.0])

# Prices used for the economic optimum (per hectare).
P_GRAIN, C_N, C_W = 230.0, 1.05, 0.22       # $/t, $/kg N, $/mm

# "True" agronomy, only ever used to synthesise the trial yields.
TRUE = dict(b0=4.20, bN=0.0340, bW=0.0200,
            bNN=-8.0e-5, bWW=-4.2e-5, bNW=2.0e-5)

OPT_C = "#3b5b92"                    # agronomic optimum
ECON_C = "#b23a48"                   # the single emphasis object


def design(seed=SEED):
    """Full 5 x 5 factorial with two centre replicates and plot-level noise."""
    rng = np.random.default_rng(seed)
    nn, ww = np.meshgrid(N_LEV, W_LEV, indexing="ij")
    n, w = nn.ravel(), ww.ravel()
    n = np.concatenate([n, [160.0, 160.0]])
    w = np.concatenate([w, [200.0, 200.0]])
    y = (TRUE["b0"] + TRUE["bN"] * n + TRUE["bW"] * w
         + TRUE["bNN"] * n ** 2 + TRUE["bWW"] * w ** 2 + TRUE["bNW"] * n * w)
    return n, w, y + rng.normal(0.0, 0.34, y.size)


def quad_fit(n, w, y):
    """Least squares on [1, N, W, N^2, W^2, NW]; returns beta and R^2."""
    X = np.c_[np.ones_like(n), n, w, n ** 2, w ** 2, n * w]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1.0 - resid.var() / y.var()
    return beta, r2, float(resid.std(ddof=X.shape[1]))


def predict(beta, n, w):
    return (beta[0] + beta[1] * n + beta[2] * w + beta[3] * n ** 2
            + beta[4] * w ** 2 + beta[5] * n * w)


def stationary(beta, gn=0.0, gw=0.0):
    """Solve grad(yield) = (gn, gw); gn/gw non-zero gives the economic point."""
    A = np.array([[2 * beta[3], beta[5]], [beta[5], 2 * beta[4]]])
    rhs = np.array([gn - beta[1], gw - beta[2]])
    return np.linalg.solve(A, rhs)


def main():
    set_pub_style(base_font=8)
    n_obs, w_obs, y_obs = design()
    beta, r2, s_res = quad_fit(n_obs, w_obs, y_obs)

    n_opt, w_opt = stationary(beta)
    n_eco, w_eco = stationary(beta, C_N / P_GRAIN, C_W / P_GRAIN)
    y_opt = predict(beta, n_opt, w_opt)
    y_eco = predict(beta, n_eco, w_eco)
    prof_opt = P_GRAIN * y_opt - C_N * n_opt - C_W * w_opt
    prof_eco = P_GRAIN * y_eco - C_N * n_eco - C_W * w_eco

    ng = np.linspace(0.0, N_MAX, 90)
    wg = np.linspace(0.0, W_MAX, 90)
    NN, WW = np.meshgrid(ng, wg, indexing="ij")
    YY = predict(beta, NN, WW)

    cmap = get_cmap("cb_sequential_green")
    z0 = float(YY.min()) - 1.6
    z1 = float(max(YY.max(), y_obs.max())) + 0.5

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.52],
                          height_ratios=[1.0, 1.0],
                          left=0.048, right=0.955, top=0.885, bottom=0.135,
                          wspace=0.10, hspace=0.42)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    # mplot3d sorts collections by depth and *ignores* zorder unless this is
    # switched off; with it on, both optimum markers sit exactly on the dome
    # and are painted over by it, which is the one thing the panel must show.
    ax3.computed_zorder = False
    ax_n = fig.add_subplot(gs[0, 1])
    ax_w = fig.add_subplot(gs[1, 1])

    # ---- (a) the fitted surface -------------------------------------------
    ax3.plot_surface(NN, WW, YY, cmap=cmap, vmin=YY.min(), vmax=YY.max(),
                     rstride=2, cstride=2, linewidth=0.15,
                     edgecolors=(1, 1, 1, 0.28), antialiased=True,
                     alpha=0.94, zorder=2)
    cs = ax3.contour(NN, WW, YY, levels=9, zdir="z", offset=z0, cmap=cmap,
                     linewidths=0.7, zorder=1)
    ax3.clabel(cs, cs.levels[::2], fmt="%.1f", fontsize=6.0,
               colors=COLORS["sub"])

    # 3-D scatter is *not* clipped by the axis limits, so filter first.
    keep = (y_obs >= z0) & (y_obs <= z1)
    ax3.scatter(n_obs[keep], w_obs[keep], y_obs[keep], s=9, marker="o",
                facecolor="white", edgecolor=COLORS["rule"], linewidth=0.5,
                depthshade=False, zorder=5)

    for (xn, xw, xy, col, mk, lab) in (
            (n_opt, w_opt, y_opt, OPT_C, "*",
             f"agronomic optimum  {n_opt:.0f} kg N/ha, {w_opt:.0f} mm"),
            (n_eco, w_eco, y_eco, ECON_C, "D",
             f"economic optimum  {n_eco:.0f} kg N/ha, {w_eco:.0f} mm")):
        # No drop line to the base plane: forcing it in front of the dome (the
        # only way to make it visible) puts a second marker in the middle of
        # the surface, which reads as a second optimum.
        ax3.scatter([xn], [xw], [xy], s=140 if mk == "*" else 44, marker=mk,
                    color=col, edgecolor="white", linewidth=0.6,
                    depthshade=False, zorder=7, label=lab)

    ax3.set_xlim(0, N_MAX)
    ax3.set_ylim(0, W_MAX)
    ax3.set_zlim(z0, z1)
    ax3.set_xlabel("nitrogen rate (kg N/ha)", fontsize=7.8, labelpad=-1)
    ax3.set_ylabel("irrigation depth (mm)", fontsize=7.8, labelpad=-1)
    ax3.set_zlabel("grain yield (t/ha)", fontsize=7.8, labelpad=-3)
    ax3.tick_params(labelsize=6.8, pad=0.6)
    ax3.view_init(elev=25, azim=-131)
    ax3.set_box_aspect((1.0, 1.0, 0.62), zoom=1.02)
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.set_facecolor("white")
        pane.set_edgecolor(COLORS["grid"])
        pane.set_alpha(1.0)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis._axinfo["grid"].update(color=COLORS["grid"], linewidth=0.5)
    ax3.set_title("Fitted water x nitrogen response surface, with its floor "
                  "contours", fontsize=8.8, fontweight="bold", pad=-2)
    ax3.legend(loc="upper left", bbox_to_anchor=(0.02, 0.94), frameon=False,
               fontsize=6.8, handletextpad=0.2, borderaxespad=0.0,
               labelspacing=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(YY.min(), YY.max()))
    safe_colorbar(fig, sm, ax3, side="bottom", width=0.10, pad=-0.28,
                  shrink=0.52, label="predicted grain yield (t/ha)")

    # ---- (b) / (c) the two marginal slices ---------------------------------
    y_n = predict(beta, ng, w_eco)
    y_w = predict(beta, n_eco, wg)
    near_n = np.abs(w_obs - w_eco) <= 55.0
    near_w = np.abs(n_obs - n_eco) <= 45.0

    for ax, xg, yg, obs_x, obs_y, xopt, xeco, xlab, unit, hold in (
            (ax_n, ng, y_n, n_obs[near_n], y_obs[near_n], n_opt, n_eco,
             "nitrogen rate (kg N/ha)", "kg N/ha",
             f"irrigation held at {w_eco:.0f} mm"),
            (ax_w, wg, y_w, w_obs[near_w], y_obs[near_w], w_opt, w_eco,
             "irrigation depth (mm)", "mm",
             f"nitrogen held at {n_eco:.0f} kg N/ha")):
        ax.fill_between(xg, yg - 1.96 * s_res, yg + 1.96 * s_res,
                        color=lighten("#2e7d5b", 0.86), lw=0.0, zorder=1)
        ax.plot(xg, yg, color="#2e7d5b", lw=1.5, zorder=4)
        ax.plot(obs_x, obs_y, marker="o", ls="none", ms=3.0, mfc="white",
                mec=COLORS["rule"], mew=0.6, zorder=5)
        ax.axvline(xopt, color=OPT_C, lw=1.0, ls=(0, (3, 2)), zorder=3)
        ax.axvline(xeco, color=ECON_C, lw=1.2, zorder=3)
        ax.annotate("", xy=(xeco, yg.min() + 0.35), xytext=(xopt,
                                                            yg.min() + 0.35),
                    arrowprops=dict(arrowstyle="<|-|>", color=ECON_C, lw=0.8,
                                    shrinkA=0, shrinkB=0, mutation_scale=7))
        ax.text(0.5 * (xopt + xeco), yg.min() + 0.52,
                f"{abs(xopt - xeco):.0f} {unit} saved", fontsize=6.4,
                color=ECON_C, ha="center", va="bottom")
        ax.set_xlabel(xlab, fontsize=8.0)
        ax.set_ylabel("grain yield (t/ha)", fontsize=8.0)
        ax.set_xlim(xg[0], xg[-1])
        ax.tick_params(labelsize=6.8)
        ax.grid(linewidth=0.5, alpha=0.65)
        ax.text(0.985, 0.045, hold, transform=ax.transAxes, fontsize=6.4,
                color=COLORS["muted"], ha="right", va="bottom")
        style_axes(ax)

    apply_shared_limits([ax_n, ax_w],
                        [y_n - 1.96 * s_res, y_n + 1.96 * s_res,
                         y_w - 1.96 * s_res, y_w + 1.96 * s_res,
                         y_obs[near_n], y_obs[near_w]], axis="y", pad=0.07)
    ax_n.set_title("Nitrogen slice through the economic optimum",
                   fontsize=8.6, fontweight="bold", pad=3.5)
    ax_w.set_title("Irrigation slice through the economic optimum",
                   fontsize=8.6, fontweight="bold", pad=3.5)

    ax_n.text(0.020, 0.955,
              f"quadratic fit R$^2$ = {r2:.3f}, s = {s_res:.2f} t/ha\n"
              f"n = {y_obs.size} plots, band = 95 % of residual spread",
              transform=ax_n.transAxes, fontsize=6.4, color=COLORS["sub"],
              ha="left", va="top", linespacing=1.35)

    panel_label_inches(fig, ax3, "(a)", dx=-0.32, dy=0.02)
    panel_label_inches(fig, ax_n, "(b)", dx=-0.40, dy=0.09)
    panel_label_inches(fig, ax_w, "(c)", dx=-0.40, dy=0.09)


    savefig(fig, "403_agr_response_surface.png")


if __name__ == "__main__":
    main()

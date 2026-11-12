"""Central composite design, its fitted response surface and the diagnostics.

A rotatable central composite design in two factors - four factorial corners,
four axial points at alpha = sqrt(2) and five replicated centre points - is
evaluated on a quadratic truth with normal error, and then everything else on
the page is read off one ordinary least-squares fit of that data.  The filled
contours are the fitted surface, the star is the stationary point obtained by
setting the gradient to zero and solving the 2x2 system (checked to be a
maximum through the Hessian), the Pareto bars are the regression coefficients
divided by their standard errors, and the parity panel compares the same fit
against the observations with a leave-one-out Q2 computed from the leverages.
Nothing is typed in twice, so the optimum quoted on the surface and the
predictions in the parity panel cannot drift apart.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from scipy import stats

from pubstyle import set_pub_style, savefig, lighten, get_cmap
from figkit import axes_inches, panel_labels_inches, halo_text

OCHRE = "#c0603f"        # product titer, the response
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

T_CENTRE, T_STEP = 35.0, 2.5          # deg C per coded unit
I_CENTRE, I_STEP = 0.50, 0.20         # mM per coded unit
ALPHA = np.sqrt(2.0)
R_OPER = 1.75            # coded radius of the plotted region of operability
N_CENTRE = 5
SIGMA = 0.22                          # g/L run-to-run error

TRUTH = np.array([8.40, 0.95, 1.35, -1.10, -0.85, -0.60])
TERMS = [r"$x_1$  temperature", r"$x_2$  inducer", r"$x_1^2$",
         r"$x_2^2$", r"$x_1x_2$"]

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 0.95, 4.62, 4.55)
TOP = (7.82, 3.61, 2.68, 1.89)
BOT = (7.82, 0.95, 2.68, 1.89)


def design():
    fac = [(a, b) for a in (-1.0, 1.0) for b in (-1.0, 1.0)]
    axi = [(-ALPHA, 0.0), (ALPHA, 0.0), (0.0, -ALPHA), (0.0, ALPHA)]
    cen = [(0.0, 0.0)] * N_CENTRE
    pts = np.array(fac + axi + cen)
    kind = np.array(["factorial"] * 4 + ["axial"] * 4 +
                    ["centre"] * N_CENTRE)
    return pts, kind


def model_matrix(x1, x2):
    return np.column_stack([np.ones_like(x1), x1, x2, x1 ** 2, x2 ** 2,
                            x1 * x2])


def fit(pts, y):
    xm = model_matrix(pts[:, 0], pts[:, 1])
    xtx_inv = np.linalg.inv(xm.T @ xm)
    beta = xtx_inv @ xm.T @ y
    resid = y - xm @ beta
    n, p = xm.shape
    s2 = float(resid @ resid) / (n - p)
    se = np.sqrt(np.diag(xtx_inv) * s2)
    lev = np.einsum("ij,jk,ik->i", xm, xtx_inv, xm)
    press = float(np.sum((resid / (1.0 - lev)) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return dict(beta=beta, se=se, tval=beta / se, df=n - p, fitted=xm @ beta,
                r2=1.0 - float(resid @ resid) / sst, q2=1.0 - press / sst,
                s=np.sqrt(s2))


def stationary_point(beta):
    """Gradient of the quadratic set to zero; returns the coded optimum."""
    _b0, b1, b2, b11, b22, b12 = beta
    hess = np.array([[2.0 * b11, b12], [b12, 2.0 * b22]])
    xs = np.linalg.solve(hess, np.array([-b1, -b2]))
    return xs, hess, np.linalg.eigvalsh(hess)


def to_natural(x1, x2):
    return T_CENTRE + T_STEP * x1, I_CENTRE + I_STEP * x2


def surface_panel(fig, ax, pts, kind, y, res, xs):
    g1 = np.linspace(-1.95, 1.95, 300)
    g2 = np.linspace(-1.95, 1.95, 300)
    a, b = np.meshgrid(g1, g2)
    zz = (model_matrix(a.ravel(), b.ravel()) @ res["beta"]).reshape(a.shape)
    # Outside the region of operability the quadratic runs away into values
    # the design never probed; keeping them would spend the whole colour ramp
    # on an extrapolated corner and flatten the surface that matters.
    zz = np.where(np.hypot(a, b) <= R_OPER, zz, np.nan)
    ta, ib = to_natural(a, b)

    levels = np.linspace(np.nanmin(zz), np.nanmax(zz), 13)
    cf = ax.contourf(ta, ib, zz, levels=levels,
                     cmap=get_cmap("cb_sequential_warm"), zorder=1)
    cs = ax.contour(ta, ib, zz, levels=levels, colors=[INK], linewidths=0.5,
                    alpha=0.55, zorder=2)
    labels = ax.clabel(cs, levels=levels[1::2], inline=True, fmt="%.1f",
                       fontsize=6.2, colors=[INK])
    # A single label colour cannot survive a sequential ramp: the same ink
    # that reads on the pale rim disappears in the dark core, so each label
    # is flipped according to the level it sits on.
    dark = levels[0] + 0.55 * (levels[-1] - levels[0])
    for t in labels:
        light = float(t.get_text()) > dark
        t.set_color("white" if light else INK)
        t.set_path_effects([pe.withStroke(
            linewidth=1.8, foreground=INK if light else "white")])

    ring = np.linspace(0.0, 2.0 * np.pi, 400)
    rt, ri = to_natural(R_OPER * np.cos(ring), R_OPER * np.sin(ring))
    ax.plot(rt, ri, color=SUB, lw=0.8, ls=(0, (3.0, 2.2)), zorder=4)

    styles = {"factorial": ("o", 5.4, BLUE), "axial": ("s", 5.0, GREEN),
              "centre": ("D", 5.2, INK)}
    for name, (mk, ms, col) in styles.items():
        sel = kind == name
        tn, inn = to_natural(pts[sel, 0], pts[sel, 1])
        ax.plot(tn, inn, ls="none", marker=mk, ms=ms, mfc="white", mec=col,
                mew=1.2, zorder=6)
    for (p1, p2), yi, kd in zip(pts, y, kind):
        if kd == "centre":
            continue
        tn, inn = to_natural(p1, p2)
        halo_text(ax, tn, inn + 0.021, "{:.2f}".format(yi), color=INK,
                  halo="white", halo_lw=1.9, fontsize=6.0, ha="center",
                  va="bottom", zorder=7)
    tc, ic = to_natural(0.0, 0.0)
    halo_text(ax, tc, ic - 0.026,
              "{:d} centre replicates\n".format(N_CENTRE) +
              "mean {:.2f} g/L, s = {:.2f}".format(
                  y[kind == "centre"].mean(),
                  y[kind == "centre"].std(ddof=1)),
              color="white", halo=INK, halo_lw=1.9, fontsize=6.2,
              ha="center", va="top", linespacing=1.28, zorder=7)

    t_opt, i_opt = to_natural(*xs)
    y_opt = float((model_matrix(np.array([xs[0]]), np.array([xs[1]])) @
                   res["beta"])[0])
    ax.plot([t_opt], [i_opt], marker="*", ms=16.0, color=OCHRE, mec="white",
            mew=1.8, zorder=9)
    ax.annotate(
        "predicted optimum\n{:.1f} ".format(t_opt) + r"$^{\circ}$C" +
        ",  {:.2f} mM\n".format(i_opt) + "titer {:.2f} g/L".format(y_opt),
        xy=(t_opt, i_opt), xycoords="data", xytext=(0.982, 0.980),
        textcoords="axes fraction", ha="right",
        va="top", fontsize=7.0, color=INK, linespacing=1.35, zorder=10,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                  edgecolor=HAIR, linewidth=0.6, alpha=0.94),
        arrowprops=dict(arrowstyle="-", color=OCHRE, lw=0.9))

    cax = ax.inset_axes([1.028, 0.0, 0.030, 1.0], transform=ax.transAxes)
    cb = fig.colorbar(cf, cax=cax, ticks=np.arange(
        np.ceil(levels[0]), np.floor(levels[-1]) + 0.5))
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.6, width=0.7, length=2.5, color="#333333")
    cb.set_label("predicted product titer (g/L)", fontsize=7.4, color=INK)

    ax.set_xlim(to_natural(g1[0], 0.0)[0], to_natural(g1[-1], 0.0)[0])
    ax.set_ylim(to_natural(0.0, g2[0])[1], to_natural(0.0, g2[-1])[1])
    ax.set_xlabel(r"induction temperature ($^{\circ}$C)")
    ax.set_ylabel("inducer concentration (mM)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [Line2D([], [], ls="none", marker="o", ms=5.4, mfc="white",
                      mec=BLUE, mew=1.2, label="factorial points"),
               Line2D([], [], ls="none", marker="s", ms=5.0, mfc="white",
                      mec=GREEN, mew=1.2,
                      label=r"axial points, $\alpha$ = " +
                      "{:.3f}".format(ALPHA)),
               Line2D([], [], ls="none", marker="D", ms=5.2, mfc="white",
                      mec=INK, mew=1.2, label="centre replicates"),
               Line2D([], [], ls="none", marker="*", ms=11.0, color=OCHRE,
                      label="fitted stationary point")]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=6.8,
                    frameon=True, facecolor="white", edgecolor=HAIR,
                    framealpha=0.92, handlelength=1.2,
                    borderaxespad=0.30, labelspacing=0.30)
    leg.get_frame().set_linewidth(0.6)
    ax.set_title("Fitted quadratic surface with the CCD runs that built it",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)
    return y_opt, t_opt, i_opt


def pareto_panel(ax, res):
    tval = np.abs(res["tval"][1:])
    order = np.argsort(tval)
    crit = float(stats.t.ppf(0.975, res["df"]))
    cols = [OCHRE if t >= crit else GREY for t in tval[order]]
    pos = np.arange(tval.size, dtype=float)

    ax.barh(pos, tval[order], height=0.62, color=cols, edgecolor="white",
            linewidth=0.7, zorder=3)
    for p, t, c in zip(pos, tval[order], cols):
        ax.text(t + crit * 0.05, p, "{:.1f}".format(t), ha="left",
                va="center", fontsize=6.3, color=c)
    ax.axvline(crit, color=INK, lw=1.0, ls=(0, (4, 2.4)), zorder=4)
    ax.text(crit + crit * 0.06, -0.58,
            "p = 0.05,  t = {:.2f}".format(crit), ha="left", va="bottom",
            fontsize=6.3, color=INK)

    ax.set_yticks(pos)
    ax.set_yticklabels([TERMS[i] for i in order], fontsize=6.8)
    ax.set_xlim(0.0, tval.max() * 1.30)
    ax.set_ylim(-0.62, tval.size - 0.20)
    ax.set_xlabel(r"standardised effect  |$t$| = |$b$| / SE($b$)",
                  fontsize=7.6)
    ax.tick_params(axis="x", labelsize=6.8)
    ax.grid(axis="x", lw=0.5, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Pareto of standardised effects, {:d} d.f.".format(
        res["df"]), fontsize=8.4, fontweight="bold", color=INK)
    return crit


def parity_panel(ax, y, res):
    pred = res["fitted"]
    lo = min(y.min(), pred.min()) - 0.35
    hi = max(y.max(), pred.max()) + 0.35
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1.0, ls=(0, (4, 2.4)), zorder=3)
    ax.fill_between([lo, hi], [lo - 2.0 * res["s"], hi - 2.0 * res["s"]],
                    [lo + 2.0 * res["s"], hi + 2.0 * res["s"]],
                    color=lighten(OCHRE, 0.88), lw=0.0, zorder=1)
    ax.plot(pred, y, ls="none", marker="o", ms=4.6, mfc="white", mec=OCHRE,
            mew=1.2, zorder=5)

    ax.text(0.035, 0.955,
            r"$R^2$ = " + "{:.4f}".format(res["r2"]) + "\n" +
            r"$Q^2$ = " + "{:.4f}".format(res["q2"]) +
            "  (leave-one-out)\n" + r"$s$ = " +
            "{:.3f} g/L,  n = {:d}".format(res["s"], y.size),
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
            color=INK, linespacing=1.42)
    ax.text(0.965, 0.06, r"shaded band $\pm 2s$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.2, color=SUB)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("predicted titer (g/L)", fontsize=7.6)
    ax.set_ylabel("observed titer (g/L)", fontsize=7.6)
    ax.tick_params(labelsize=6.8)
    ax.grid(lw=0.5, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Observed against predicted", fontsize=8.4,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    rng = np.random.default_rng(881)
    pts, kind = design()
    y = (model_matrix(pts[:, 0], pts[:, 1]) @ TRUTH +
         rng.normal(0.0, SIGMA, pts.shape[0]))
    res = fit(pts, y)
    xs, _hess, eig = stationary_point(res["beta"])

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_par = axes_inches(fig, TOP)
    ax_fit = axes_inches(fig, BOT)

    y_opt, t_opt, i_opt = surface_panel(fig, ax_main, pts, kind, y, res, xs)
    crit = pareto_panel(ax_par, res)
    parity_panel(ax_fit, y, res)

    n_sig = int(np.sum(np.abs(res["tval"][1:]) >= crit))
    panel_labels_inches(fig, [ax_main, ax_par, ax_fit], dx=-0.62, dy=0.14)
    savefig(fig, "881_bpr_doe_response_surface.png")


if __name__ == "__main__":
    main()

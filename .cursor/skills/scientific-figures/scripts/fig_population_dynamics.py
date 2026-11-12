"""Predator-prey population dynamics, four views of one simulation.

(a) multi-species density time series with the predator on a secondary axis;
(b) prey-predator phase plane with nullclines and a damped inward spiral;
(c) age-class composition through time as a stacked area with in-band labels;
(d) density dependence of the per-capita growth rate, with the fitted
    regression and the implied carrying capacity.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter1d

from pubstyle import set_pub_style, savefig, style_axes


PREY1 = "#2e7d5b"
PREY2 = "#c08f3e"
PRED = "#a2593c"
TEAL = "#3d8b93"
INK = "#1a1a1a"
AGE_COLS = ["#0d4232", "#2e7d5b", "#6fb392", "#a8d0bb", "#d8e8de"]
AGE_NAMES = ["Age 5+", "Age 4", "Age 3", "Age 2", "Age 1"]

# shared predator parameters
E_CONV, MORT = 0.30, 0.30


def panel_tag(fig, ax, label, dx=-0.052, dy=0.020, size=10):
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, label, fontsize=size,
             fontweight="bold", ha="left", va="bottom", color=INK)


def nice_top(v, step):
    return step * np.ceil(v / step)


def three_species(t, s):
    n1, n2, p = s
    r1, r2, k1, k2 = 0.90, 0.72, 6000.0, 4000.0
    a1, a2 = 0.0025, 0.0020
    dn1 = r1 * n1 * (1 - (n1 + 0.45 * n2) / k1) - a1 * n1 * p
    dn2 = r2 * n2 * (1 - (n2 + 0.35 * n1) / k2) - a2 * n2 * p
    dp = E_CONV * (a1 * n1 + a2 * n2) * p - MORT * p
    return [dn1, dn2, dp]


R2, K2, A2 = 0.90, 2500.0, 0.0025


def two_species(t, s):
    n, p = s
    return [R2 * n * (1 - n / K2) - A2 * n * p,
            E_CONV * A2 * n * p - MORT * p]


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(20260229)

    t_end = 60.0
    tt = np.linspace(0, t_end, 2400)
    sol = solve_ivp(three_species, (0, t_end), [900.0, 500.0, 190.0],
                    t_eval=tt, rtol=1e-9, atol=1e-9)
    n1, n2, pp = sol.y

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # ---- (a) time series --------------------------------------------------
    ax_a.set_facecolor("#fbfbf9")
    ax_a.plot(tt, n1, color=PREY1, lw=1.5, zorder=4)
    ax_a.plot(tt, n2, color=PREY2, lw=1.3, zorder=3)
    prey_top = nice_top(max(n1.max(), n2.max()) * 1.14, 200)
    ax_a.set_xlim(0, t_end)
    ax_a.set_ylim(0, prey_top)
    ax_a.set_yticks(np.linspace(0, prey_top, 4))
    ax_a.set_xlabel("Year")
    ax_a.set_ylabel("Prey density (ind. ha$^{-1}$)")
    ax_a.grid(axis="y", lw=0.5, color="#ebebe6")
    style_axes(ax_a)

    ax_a2 = ax_a.twinx()
    ax_a2.plot(tt, pp, color=PRED, lw=1.25, ls=(0, (5, 2.2)), zorder=5)
    pred_top = nice_top(pp.max() * 1.14, 100)
    ax_a2.set_ylim(0, pred_top)
    ax_a2.set_yticks(np.linspace(0, pred_top, 4))
    ax_a2.set_ylabel("Predator density (ind. ha$^{-1}$)", color=PRED)
    ax_a2.tick_params(axis="y", colors=PRED, length=3)
    ax_a2.spines["top"].set_visible(False)
    ax_a2.spines["right"].set_color(PRED)
    ax_a2.spines["left"].set_visible(False)

    ax_a.set_title("Density time series", fontsize=8.2, loc="left", pad=5)
    handles = [
        Line2D([], [], color=PREY1, lw=1.5, label="Prey A"),
        Line2D([], [], color=PREY2, lw=1.3, label="Prey B"),
        Line2D([], [], color=PRED, lw=1.25, ls=(0, (5, 2.2)),
               label="Predator"),
    ]
    la = ax_a.legend(handles=handles, loc="upper right", ncol=3,
                     fontsize=6.4, frameon=True, facecolor="white",
                     edgecolor="#dcdcd4", framealpha=0.92, handlelength=1.5,
                     columnspacing=0.9, handletextpad=0.4, borderpad=0.35,
                     borderaxespad=0.3)
    la.get_frame().set_linewidth(0.55)
    la.set_clip_on(False)

    # ---- (b) phase plane --------------------------------------------------
    ax_b.set_facecolor("#fbfbf9")
    n_star = MORT / (E_CONV * A2)
    p_star = R2 * (1 - n_star / K2) / A2

    sol2 = solve_ivp(two_species, (0, 62), [1250.0, 70.0],
                     t_eval=np.linspace(0, 62, 6000), rtol=1e-10, atol=1e-10)
    trx, tryy = sol2.y
    xmax = nice_top(max(trx.max() * 1.08, n_star * 1.6), 200)
    ymax = nice_top(max(tryy.max() * 1.12, p_star * 1.4), 100)

    ng = np.linspace(xmax * 0.03, xmax * 0.98, 17)
    pg = np.linspace(ymax * 0.04, ymax * 0.97, 13)
    NG, PG = np.meshgrid(ng, pg)
    dN, dP = two_species(0, [NG, PG])
    scale_n, scale_p = xmax, ymax
    un, up = dN / scale_n, dP / scale_p
    mag = np.hypot(un, up)
    ax_b.quiver(NG, PG, un / mag, up / mag, color="#dbe0dc", scale=34,
                width=0.0034, headwidth=3.6, headlength=4.2, zorder=1)

    ax_b.axvline(n_star, color=PRED, lw=0.95, ls=(0, (4, 2.2)), zorder=2)
    nline = np.linspace(0, xmax, 60)
    ax_b.plot(nline, R2 * (1 - nline / K2) / A2, color=PREY1, lw=0.95,
              ls=(0, (4, 2.2)), zorder=2)

    ax_b.plot(trx, tryy, color=TEAL, lw=1.15, zorder=4)
    ax_b.scatter([trx[0]], [tryy[0]], s=22, marker="o", facecolor="white",
                 edgecolor=TEAL, linewidth=1.0, zorder=6)
    ax_b.scatter([n_star], [p_star], s=40, marker="*", color="#33332f",
                 zorder=6)
    ax_b.text(trx[0] - 0.02 * xmax, tryy[0] + 0.03 * ymax, "$t_0$",
              fontsize=6.6, color=TEAL, ha="right", va="bottom")
    ax_b.annotate(f"equilibrium\n({n_star:.0f}, {p_star:.0f})",
                  xy=(n_star, p_star),
                  xytext=(n_star + 0.16 * xmax, p_star + 0.16 * ymax),
                  fontsize=6.3, color="#33332f", va="bottom",
                  linespacing=1.3,
                  arrowprops=dict(arrowstyle="-", color="#8d8d84", lw=0.6,
                                  shrinkA=1.0, shrinkB=4.0))
    ax_b.text(0.975, 0.955, "prey nullcline", transform=ax_b.transAxes,
              fontsize=6.2, color=PREY1, ha="right", va="top")
    ax_b.text(0.975, 0.882, "predator nullcline", transform=ax_b.transAxes,
              fontsize=6.2, color=PRED, ha="right", va="top")

    ax_b.set_xlim(0, xmax)
    ax_b.set_ylim(0, ymax)
    ax_b.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 4, 5, 10]))
    ax_b.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 4, 5, 10]))
    ax_b.set_xlabel("Prey density (ind. ha$^{-1}$)")
    ax_b.set_ylabel("Predator density (ind. ha$^{-1}$)")
    ax_b.set_title("Phase plane: inward spiral", fontsize=8.2, loc="left",
                   pad=5)
    style_axes(ax_b)

    # ---- (c) age structure ------------------------------------------------
    yrs = np.arange(0, 61)
    base = np.array([0.10, 0.15, 0.20, 0.24, 0.31])[:, None]
    wob = gaussian_filter1d(rng.normal(0, 1, (5, yrs.size)), 6.0, axis=1)
    wob = wob / np.abs(wob).max()
    comp = np.clip(base + 0.075 * wob, 0.02, None)
    comp = comp / comp.sum(axis=0)

    ax_c.stackplot(yrs, comp * 100.0, colors=AGE_COLS, edgecolor="white",
                   linewidth=0.35, zorder=3)
    cum = np.cumsum(comp * 100.0, axis=0)
    mid = cum - comp * 50.0
    for i, nm in enumerate(AGE_NAMES):
        ax_c.text(52.5, mid[i, 52], nm, fontsize=6.1, ha="center",
                  va="center", color="white" if i < 2 else "#22352c",
                  fontweight="bold", zorder=5)
    ax_c.set_xlim(0, t_end)
    ax_c.set_ylim(0, 100)
    ax_c.set_yticks([0, 25, 50, 75, 100])
    ax_c.set_xlabel("Year")
    ax_c.set_ylabel("Share of population (%)")
    ax_c.set_title("Age-class composition of Prey A", fontsize=8.2,
                   loc="left", pad=5)
    style_axes(ax_c)

    # ---- (d) density dependence -------------------------------------------
    ax_d.set_facecolor("#fbfbf9")
    # independent plots censused one year apart, spanning the density range
    dens = np.sort(rng.uniform(60.0, 1380.0, 72))
    growth = 0.56 * (1.0 - dens / 1060.0) + rng.normal(0, 0.085, dens.size)

    slope, icpt = np.polyfit(dens, growth, 1)
    xtop = nice_top(dens.max() * 1.08, 250)
    xs = np.linspace(0, xtop, 120)
    fit = icpt + slope * xs
    resid = growth - (icpt + slope * dens)
    se = resid.std() * np.sqrt(1.0 / dens.size +
                               (xs - dens.mean()) ** 2 /
                               ((dens - dens.mean()) ** 2).sum())
    ax_d.fill_between(xs, fit - 1.96 * se, fit + 1.96 * se, color=PREY1,
                      alpha=0.15, lw=0, zorder=2)
    ax_d.plot(xs, fit, color=PREY1, lw=1.4, zorder=4)
    ax_d.scatter(dens, growth, s=13, color=TEAL, alpha=0.75,
                 edgecolor="white", linewidth=0.4, zorder=5)
    ax_d.axhline(0, color="#8d8d84", lw=0.7, zorder=3)

    ytop = nice_top(max(np.abs(growth).max() * 1.30, 0.4), 0.2)
    k_hat = -icpt / slope
    ax_d.axvline(k_hat, color=PRED, lw=0.95, ls=(0, (3, 2.2)), zorder=3)
    right = k_hat < 0.55 * xtop
    ax_d.text(k_hat + (0.02 if right else -0.02) * xtop, 0.90 * ytop,
              f"$\\hat{{K}}$ = {k_hat:.0f} ind. ha$^{{-1}}$", fontsize=6.4,
              color=PRED, ha="left" if right else "right", va="top",
              fontweight="bold")

    ax_d.set_xlim(0, xtop)
    ax_d.set_ylim(-ytop, ytop)
    ax_d.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    ax_d.set_xlabel("Density $N_t$ (ind. ha$^{-1}$)")
    ax_d.set_ylabel("Growth rate ln($N_{t+1}/N_t$)")
    ax_d.set_title("Density dependence across 72 plots", fontsize=8.2,
                   loc="left", pad=5)
    ax_d.grid(axis="y", lw=0.5, color="#ebebe6")
    style_axes(ax_d)
    r2 = np.corrcoef(dens, growth)[0, 1] ** 2
    ax_d.text(0.028, 0.055,
              f"$\\beta$ = {slope * 1000:+.2f} per 10$^3$ ind.\n"
              f"$R^2$ = {r2:.2f},  $n$ = {dens.size}",
              transform=ax_d.transAxes, fontsize=6.3, va="bottom",
              linespacing=1.35, color="#4a4a44",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                        edgecolor="#dededa", linewidth=0.55, alpha=0.92))

    fig.subplots_adjust(left=0.078, right=0.918, top=0.865, bottom=0.085,
                        hspace=0.46, wspace=0.42)

    for ax, lab in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        panel_tag(fig, ax, f"({lab})")

    savefig(fig, "229_population_dynamics.png")


if __name__ == "__main__":
    main()

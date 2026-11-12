"""How a privacy budget is spent: three composition theorems, one query axis.

Main panel: cumulative privacy loss against the number of Gaussian-mechanism
releases, on log-log axes because the three accountants differ by two orders
of magnitude in how many queries they permit.  Basic composition adds
epsilon linearly, advanced composition pays a square-root term plus a
quadratic correction, and the Renyi / moments accountant is evaluated in
closed form at its optimal order.  The three exhaustion points are not
placed by hand: each is read back off the plotted curve by interpolating the
query axis at the budget line.

Auxiliary (both 58 % of the main panel width): the number of queries a fixed
budget buys as the Gaussian noise multiplier grows, with the scaling
exponents fitted in log space from the same arrays; and a fourteen-release
schedule under two spending policies, composed release by release with the
Renyi accountant, showing that a front-loaded policy runs out of budget
before the schedule ends.

Illustrative synthetic accounting, standard textbook formulas only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
DEEP = "#2b4470"
GREEN = "#2e7d5b"
TEAL = "#2a7f7a"
RUST = "#c0603f"
ACCENT = "#b23a48"
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRID = "#d5d5d5"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.92, 1.00, 5.50, 4.48)
AUX_T_RECT = (7.64, 3.48, 3.14, 2.00)
AUX_B_RECT = (7.64, 0.86, 3.14, 2.00)

EPS0 = 0.05        # per-release epsilon (dimensionless)
DELTA_Q = 1e-6     # per-release delta
DELTA_T = 1e-5     # target total delta
BUDGET = 4.0       # total epsilon the data custodian will release
# Gaussian mechanism calibration constant: sigma = SENS_CONST / epsilon_0
SENS_CONST = float(np.sqrt(2.0 * np.log(1.25 / DELTA_Q)))


def basic_eps(k, eps0):
    """Sequential composition: epsilon adds up release by release."""
    return np.asarray(k, dtype=float) * eps0


def advanced_eps(k, eps0, delta):
    """Dwork-Rothblum-Vadhan advanced composition at target delta."""
    k = np.asarray(k, dtype=float)
    return (np.sqrt(2.0 * k * np.log(1.0 / delta)) * eps0
            + k * eps0 * np.expm1(eps0))


def rdp_eps(k, sigma, delta):
    """Renyi accountant for k Gaussian releases, optimal order in closed form.

    The Gaussian mechanism is (alpha, alpha / (2 sigma^2))-RDP, so k releases
    give alpha k / (2 sigma^2); converting to (epsilon, delta) and minimising
    over alpha > 1 gives u + 2 sqrt(u ln(1/delta)) with u = k / (2 sigma^2).
    """
    u = np.asarray(k, dtype=float) / (2.0 * sigma ** 2)
    return u + 2.0 * np.sqrt(u * np.log(1.0 / delta))


def rdp_eps_from_u(u, delta):
    """Same conversion for a heterogeneous schedule, u = sum 1/(2 sigma_i^2)."""
    u = np.asarray(u, dtype=float)
    return u + 2.0 * np.sqrt(u * np.log(1.0 / delta))


def rdp_max_queries(sigma, delta, budget):
    """Exact inversion of rdp_eps for k."""
    c = np.log(1.0 / delta)
    return 2.0 * sigma ** 2 * (np.sqrt(c + budget) - np.sqrt(c)) ** 2


def queries_at_budget(kgrid, curve, budget):
    """Read the query count off a plotted, monotone epsilon curve."""
    return float(np.interp(budget, curve, kgrid))


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(1014)

    sigma0 = SENS_CONST / EPS0
    kgrid = np.logspace(0.0, np.log10(2.0e4), 900)

    curves = [
        ("basic composition", GREY, 1.5, basic_eps(kgrid, EPS0),
         r"$\varepsilon_{tot} = k\,\varepsilon_0$"),
        ("advanced composition", BLUE, 1.9,
         advanced_eps(kgrid, EPS0, DELTA_T),
         r"$\sqrt{2k\ln(1/\delta)}\,\varepsilon_0 + k\varepsilon_0"
         r"(e^{\varepsilon_0}-1)$"),
        ("Renyi / moments accountant", GREEN, 2.1,
         rdp_eps(kgrid, sigma0, DELTA_T),
         r"$u + 2\sqrt{u\ln(1/\delta)},\ u = k/2\sigma^2$"),
    ]
    k_out = [queries_at_budget(kgrid, c[3], BUDGET) for c in curves]
    gain = k_out[2] / k_out[0]

    # ---- aux (b): queries a fixed budget buys against the noise multiplier --
    sig = np.logspace(np.log10(3.0), np.log10(400.0), 160)
    kfine = np.logspace(-1.0, 8.0, 4000)
    k_basic = BUDGET / (SENS_CONST / sig)
    k_adv = np.array([queries_at_budget(
        kfine, advanced_eps(kfine, SENS_CONST / s, DELTA_T), BUDGET)
        for s in sig])
    k_rdp = rdp_max_queries(sig, DELTA_T, BUDGET)
    exps = [float(np.polyfit(np.log10(sig), np.log10(y), 1)[0])
            for y in (k_basic, k_adv, k_rdp)]

    # ---- aux (c): a fourteen-release schedule under two policies -----------
    n_rel = 14
    rel = np.arange(1, n_rel + 1)
    sig_flat = np.full(n_rel, 5.0)
    sig_front = np.where(rel <= 6, 3.0, 12.0)
    eps_flat = rdp_eps_from_u(np.cumsum(1.0 / (2.0 * sig_flat ** 2)), DELTA_T)
    eps_front = rdp_eps_from_u(np.cumsum(1.0 / (2.0 * sig_front ** 2)),
                               DELTA_T)
    over_front = np.flatnonzero(eps_front > BUDGET)
    stop_front = int(rel[over_front[0]]) if over_front.size else n_rel + 1
    over_flat = np.flatnonzero(eps_flat > BUDGET)
    stop_flat = int(rel[over_flat[0]]) if over_flat.size else n_rel + 1

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main panel ---------------------------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(color=GRID, lw=0.5, alpha=0.55, which="major", zorder=0)
    ax.grid(color=GRID, lw=0.35, alpha=0.30, which="minor", zorder=0)
    y_lo, y_hi = 0.02, 2.5e3
    ax.axhline(BUDGET, color=ACCENT, lw=1.2, ls=(0, (4.0, 2.2)), zorder=6)
    ax.text(1.15, BUDGET * 1.16,
            f"released budget: $\\varepsilon$ = {BUDGET:.0f}",
            fontsize=6.8, color=ACCENT, ha="left", va="bottom")

    for (name, col, lw, curve, _), kx in zip(curves, k_out):
        ax.plot(kgrid, curve, color=col, lw=lw, zorder=5)
        ax.plot([kx, kx], [y_lo, BUDGET], color=col, lw=0.85,
                ls=(0, (2.2, 1.8)), zorder=4)
        ax.plot([kx], [BUDGET], marker="o", ms=5.0, mfc="white", mec=col,
                mew=1.5, zorder=9)
        ax.text(kx * 1.14, BUDGET * 0.50, f"k = {kx:,.0f}", fontsize=6.8,
                color=col, ha="left", va="center", fontweight="bold")

    ax.annotate(f"the accountant alone buys\n{gain:,.0f}x more queries at the "
                f"same\nbudget and the same noise",
                xy=(k_out[2], BUDGET), xytext=(1.1e3, 0.20),
                fontsize=6.8, color=GREEN, ha="center", va="bottom",
                linespacing=1.5,
                arrowprops=dict(arrowstyle="-", lw=0.75, color=GREEN,
                                connectionstyle="arc3,rad=0.25"))
    ax.text(1.15, 0.024,
            f"Gaussian mechanism, per-release $\\varepsilon_0$ = {EPS0:g}\n"
            f"noise multiplier $\\sigma$ = {sigma0:.0f} (dimensionless)\n"
            f"per-release $\\delta$ = {DELTA_Q:.0e}",
            fontsize=6.4, color=MUTED, ha="left", va="bottom",
            linespacing=1.55, zorder=8,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.94))

    ax.set_xlim(1.0, 2.2e4)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("releases answered, $k$ (queries, log scale)", fontsize=8.5)
    ax.set_ylabel(r"cumulative privacy loss $\varepsilon$ "
                  "(dimensionless, log scale)", fontsize=8.5)
    ax.set_title("Cumulative privacy loss over a stream of queries, both axes "
                 "logarithmic", fontsize=9.0, fontweight="bold", pad=5.0)
    handles = [Line2D([], [], color=c[1], lw=c[2],
                      label=f"{c[0]}:  {c[4]}\n"
                            f"      budget gone after {kx:,.0f} queries")
               for c, kx in zip(curves, k_out)]
    leg = ax.legend(handles=handles, loc="upper left", frameon=False,
                    fontsize=6.8, labelspacing=0.75, handlelength=2.0,
                    borderpad=0.2, bbox_to_anchor=(-0.010, 1.012))
    for t in leg.get_texts():
        t.set_color(SUB)

    # ---- aux (b) ------------------------------------------------------------
    ax_t.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    for y, col, lw, name, e in zip((k_basic, k_adv, k_rdp),
                                   (GREY, BLUE, GREEN), (1.4, 1.6, 1.8),
                                   ("basic", "advanced", "Renyi"), exps):
        ax_t.loglog(sig, y, color=col, lw=lw, zorder=5,
                    label=f"{name}, slope {e:.2f}")
    ax_t.set_xlim(sig[0] * 0.85, sig[-1] * 1.2)
    ax_t.set_ylim(3.0, 4.0e5)
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel(r"noise multiplier $\sigma$ (dimensionless, log scale)",
                    fontsize=7.6, labelpad=1.5)
    ax_t.set_ylabel("queries affordable\n(queries, log scale)", fontsize=7.6)
    ax_t.set_title(f"What a fixed budget of $\\varepsilon$ = {BUDGET:.0f} buys",
                   fontsize=8.4, fontweight="bold", pad=4.0)
    lt = ax_t.legend(loc="upper left", frameon=False, fontsize=6.2,
                     handlelength=1.8, labelspacing=0.30, borderpad=0.15,
                     bbox_to_anchor=(-0.012, 1.035))
    for t in lt.get_texts():
        t.set_color(SUB)
    ax_t.text(0.975, 0.055, "exponents fitted in log space",
              transform=ax_t.transAxes, ha="right", va="bottom", fontsize=6.1,
              color=MUTED)

    # ---- aux (c) ------------------------------------------------------------
    ax_b.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    ax_b.axhspan(BUDGET, BUDGET * 1.9, color=ACCENT, alpha=0.07, lw=0,
                 zorder=1)
    ax_b.axhline(BUDGET, color=ACCENT, lw=1.1, ls=(0, (4.0, 2.2)), zorder=6)
    ax_b.step(np.r_[0, rel], np.r_[0.0, eps_flat], where="post", color=TEAL,
              lw=1.7, zorder=5)
    ax_b.step(np.r_[0, rel], np.r_[0.0, eps_front], where="post", color=RUST,
              lw=1.7, zorder=5)
    ok = rel < stop_front
    ax_b.plot(rel[ok], eps_front[ok], marker="o", ls="none", ms=3.2,
              mfc="white", mec=RUST, mew=1.0, zorder=7)
    ax_b.plot(rel[~ok], eps_front[~ok], marker="x", ls="none", ms=4.0,
              mec=ACCENT, mew=1.1, zorder=7)
    ax_b.plot(rel, eps_flat, marker="o", ls="none", ms=3.2, mfc="white",
              mec=TEAL, mew=1.0, zorder=7)
    ax_b.annotate(f"budget gone at release {stop_front}\n"
                  f"({n_rel - stop_front + 1} releases refused)",
                  xy=(stop_front, eps_front[stop_front - 1]),
                  xytext=(7.5, 0.55), fontsize=6.2, color=ACCENT, ha="left",
                  va="bottom", linespacing=1.5,
                  arrowprops=dict(arrowstyle="-", lw=0.7, color=ACCENT,
                                  connectionstyle="arc3,rad=-0.25"))
    ax_b.set_xlim(0.0, n_rel + 0.4)
    ax_b.set_ylim(0.0, BUDGET * 1.55)
    ax_b.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("scheduled release (index)", fontsize=7.6, labelpad=1.5)
    ax_b.set_ylabel(r"spent $\varepsilon$ (dimensionless)", fontsize=7.6)
    ax_b.set_title("One schedule, two spending policies", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    hb = [Line2D([], [], color=TEAL, lw=1.7,
                 label=r"flat, $\sigma$ = 5 every release "
                       f"(ends at {eps_flat[-1]:.2f})"),
          Line2D([], [], color=RUST, lw=1.7,
                 label=r"front-loaded, $\sigma$ = 3 for six releases")]
    lb = ax_b.legend(handles=hb, loc="upper left", frameon=False, fontsize=6.1,
                     handlelength=1.8, labelspacing=0.30, borderpad=0.15,
                     bbox_to_anchor=(-0.012, 1.035))
    for t in lb.get_texts():
        t.set_color(SUB)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.64, dy=0.12)
    savefig(fig, "1014_ppc_budget_composition.png")


if __name__ == "__main__":
    main()

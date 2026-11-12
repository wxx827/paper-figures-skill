"""Differential item functioning: two groups, one item, two different curves.

Uniform and non-uniform DIF are the two cases that have to be told apart, and
a signed index cannot do it: the crossing item in the small multiples has a
signed area near zero and an absolute area larger than one of the flagged
uniform items.  Both numbers are integrated from the plotted curves, so the
shaded region and the caption cannot drift apart.  The forest and funnel are
computed from simulated responses rather than from the generating parameters:
examinees are matched on total score, a 2 x 2 table is built inside each
stratum, and the Mantel-Haenszel odds ratio, its Robins-Breslow-Greenland
standard error and the ETS delta scale come out of those tables.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_label_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
HAIR = "#cccccc"
BLUE = "#3b5b92"          # favours the reference group
ORANGE = "#c0603f"        # favours the focal group
CURVE = "#2e74a8"
ACCENT = "#b23a48"

FIGW, FIGH = 10.4, 6.4
CELL_W, CELL_H = 2.42, 1.70
CELLS = [(0.78, 3.70), (3.81, 3.70), (0.78, 1.05), (3.81, 1.05)]
AUX_T = (7.05, 3.62, 3.00, 1.78)
AUX_B = (7.05, 1.05, 3.00, 1.78)

THETA_LO, THETA_HI = -4.0, 4.0
N_ITEM = 24
N_REF, N_FOC = 900, 600
N_STRATA = 6
ETS_B, ETS_C = 1.0, 1.5    # ETS delta cut points for the A / B / C bands


def icc(theta, a, b):
    """Two-parameter logistic response curve."""
    return 1.0 / (1.0 + np.exp(-a * (theta - b)))


def item_bank(seed=899):
    """Group-specific parameters: most items clean, a few deliberately not."""
    rng = np.random.default_rng(seed)
    b_ref = np.linspace(-1.75, 1.75, N_ITEM) + rng.normal(0.0, 0.12, N_ITEM)
    a_ref = np.clip(rng.normal(1.15, 0.22, N_ITEM), 0.55, 1.85)
    db = rng.normal(0.0, 0.10, N_ITEM)          # nuisance, ETS level A
    da = np.zeros(N_ITEM)
    db[3] = 0.80                                 # large uniform DIF
    db[17] = -0.72                               # large uniform, other way
    db[9] = 0.55                                 # moderate uniform
    db[21] = -0.50
    db[13] = 0.00                                # non-uniform: curves cross
    da[13] = 0.95
    db[6] = 0.02
    da[6] = -0.35                                # milder non-uniform
    return a_ref, b_ref, a_ref + da, b_ref + db


def responses(a_r, b_r, a_f, b_f, seed=8990):
    """Simulate the two cohorts and who saw which item."""
    rng = np.random.default_rng(seed)
    th_r = rng.normal(0.15, 1.00, N_REF)
    th_f = rng.normal(-0.25, 1.00, N_FOC)
    x_r = (rng.random((N_REF, N_ITEM)) < icc(th_r[:, None], a_r[None, :],
                                             b_r[None, :])).astype(int)
    x_f = (rng.random((N_FOC, N_ITEM)) < icc(th_f[:, None], a_f[None, :],
                                             b_f[None, :])).astype(int)
    # Items sat in different booklets, so precision varies across the bank.
    frac = rng.uniform(0.28, 1.00, N_ITEM)
    seen_r = rng.random((N_REF, N_ITEM)) < frac[None, :]
    seen_f = rng.random((N_FOC, N_ITEM)) < frac[None, :]
    return th_r, th_f, x_r, x_f, seen_r, seen_f


def mantel_haenszel(x_r, x_f, seen_r, seen_f, j, strata_r, strata_f):
    """MH odds ratio, ETS delta, RBG standard error and MH chi-square."""
    num = den = 0.0
    a_sum = e_sum = v_sum = 0.0
    p_sum = q_sum = pr_sum = ps_qr_sum = s_sum = r_sum = 0.0
    for k in range(N_STRATA):
        mr = seen_r[:, j] & (strata_r == k)
        mf = seen_f[:, j] & (strata_f == k)
        n_r, n_f = int(mr.sum()), int(mf.sum())
        if n_r < 2 or n_f < 2:
            continue
        a = float(x_r[mr, j].sum())          # reference correct
        b = n_r - a                          # reference wrong
        c = float(x_f[mf, j].sum())          # focal correct
        d = n_f - c
        n = n_r + n_f
        r_k, s_k = a * d / n, b * c / n
        num += r_k
        den += s_k
        a_sum += a
        e_sum += (a + b) * (a + c) / n
        v_sum += ((a + b) * (c + d) * (a + c) * (b + d) /
                  (n ** 2 * max(n - 1.0, 1.0)))
        p, q = (a + d) / n, (b + c) / n
        p_sum += p * r_k
        q_sum += q * s_k
        pr_sum += p * s_k + q * r_k
        r_sum += r_k
        s_sum += s_k
    alpha = max(num, 1e-6) / max(den, 1e-6)
    se_ln = np.sqrt(p_sum / (2 * max(r_sum, 1e-6) ** 2) +
                    pr_sum / (2 * max(r_sum, 1e-6) * max(s_sum, 1e-6)) +
                    q_sum / (2 * max(s_sum, 1e-6) ** 2))
    delta = -2.35 * np.log(alpha)
    chi2 = (abs(a_sum - e_sum) - 0.5) ** 2 / max(v_sum, 1e-9)
    return delta, 2.35 * se_ln, chi2


def ets_level(delta, chi2):
    """ETS A / B / C classification of DIF severity."""
    if abs(delta) < ETS_B or chi2 < 3.84:
        return "A"
    return "B" if abs(delta) < ETS_C else "C"


def main():
    set_pub_style()
    theta = np.linspace(THETA_LO, THETA_HI, 801)
    a_r, b_r, a_f, b_f = item_bank()
    th_r, th_f, x_r, x_f, seen_r, seen_f = responses(a_r, b_r, a_f, b_f)

    # Matching criterion: total score over the whole form, cut into strata.
    tot_r, tot_f = x_r.sum(axis=1), x_f.sum(axis=1)
    cuts = np.quantile(np.concatenate([tot_r, tot_f]),
                       np.linspace(0, 1, N_STRATA + 1)[1:-1])
    strata_r = np.digitize(tot_r, cuts)
    strata_f = np.digitize(tot_f, cuts)

    delta = np.zeros(N_ITEM)
    se = np.zeros(N_ITEM)
    chi2 = np.zeros(N_ITEM)
    n_seen = np.zeros(N_ITEM, dtype=int)
    for j in range(N_ITEM):
        delta[j], se[j], chi2[j] = mantel_haenszel(
            x_r, x_f, seen_r, seen_f, j, strata_r, strata_f)
        n_seen[j] = int(seen_r[:, j].sum() + seen_f[:, j].sum())
    level = np.array([ets_level(d, c) for d, c in zip(delta, chi2)])

    # The four small multiples, and the areas between their two curves.
    show = [3, 13, 21, 8]
    kind = ["uniform DIF", "non-uniform DIF", "uniform DIF, other direction",
            "negligible DIF"]
    p_ref = np.array([icc(theta, a_r[j], b_r[j]) for j in show])
    p_foc = np.array([icc(theta, a_f[j], b_f[j]) for j in show])
    signed = np.array([np.trapezoid(p_ref[i] - p_foc[i], theta)
                       for i in range(len(show))])
    absolute = np.array([np.trapezoid(np.abs(p_ref[i] - p_foc[i]), theta)
                         for i in range(len(show))])

    fig = plt.figure(figsize=(FIGW, FIGH))
    cells = [axes_inches(fig, (x, y, CELL_W, CELL_H)) for x, y in CELLS]
    ax_fo = axes_inches(fig, AUX_T)
    ax_fu = axes_inches(fig, AUX_B)

    # ---- main: four representative items ---------------------------------
    for i, (ax, j) in enumerate(zip(cells, show)):
        ax.fill_between(theta, p_ref[i], p_foc[i], color=CURVE, alpha=0.20,
                        lw=0.0, zorder=2)
        ax.plot(theta, p_ref[i], color=CURVE, lw=1.7, zorder=4)
        ax.plot(theta, p_foc[i], color=CURVE, lw=1.7, ls=(0, (4.4, 2.0)),
                zorder=4)
        cross = np.flatnonzero(np.diff(np.sign(p_ref[i] - p_foc[i])) != 0)
        if cross.size and abs(signed[i]) < 0.35:
            xc = float(theta[cross[0]])
            ax.axvline(xc, color=ACCENT, lw=0.7, ls=(0, (1.6, 1.8)), zorder=3)
            ax.text(xc - 0.14, 0.60, f"curves cross at theta = {xc:+.2f}",
                    ha="right", va="bottom", fontsize=6.0, color=ACCENT)
        ax.text(0.03, 0.965, f"item {j + 1:02d}   ETS level {level[j]}",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.6,
                color=INK, fontweight="bold")
        ax.text(0.03, 0.855, f"MH delta {delta[j]:+.2f} (ETS units)",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.2,
                color=SUB)
        ax.text(0.975, 0.045,
                f"shaded area between the curves\nsigned {signed[i]:+.2f}, "
                f"absolute {absolute[i]:.2f} (P x logits)",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.2, color=SUB, linespacing=1.35)
        ax.set_xlim(THETA_LO, THETA_HI)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(np.arange(-4, 5, 2))
        ax.set_yticks(np.arange(0.0, 1.01, 0.25))
        ax.tick_params(labelsize=6.6)
        ax.grid(axis="y", lw=0.5, color="#eeeeee")
        ax.set_title(kind[i], fontsize=7.6, fontweight="bold", color=SUB,
                     pad=3.5)
        if i >= 2:
            ax.set_xlabel("theta (logits, SD units)", fontsize=7.4)
        else:
            ax.set_xticklabels([])
        if i % 2 == 0:
            ax.set_ylabel("P(correct) (-)", fontsize=7.4)
        else:
            ax.set_yticklabels([])
    fig.legend(
        handles=[Line2D([], [], color=CURVE, lw=1.7,
                        label=f"reference group, solid (n = {N_REF} "
                              f"examinees)"),
                 Line2D([], [], color=CURVE, lw=1.7, ls=(0, (4.4, 2.0)),
                        label=f"focal group, dashed (n = {N_FOC} examinees)")],
        loc="center", bbox_to_anchor=(3.50 / FIGW, 3.24 / FIGH), ncol=2,
        frameon=False, fontsize=6.6, handlelength=2.4, columnspacing=2.4,
        borderaxespad=0.0)

    fig.text((0.78 + 3.81 + CELL_W) / 2.0 / FIGW, 5.62 / FIGH,
             "Matched on total score, the gap between the two curves is "
             "the DIF",
             ha="center", va="bottom", fontsize=9.0, fontweight="bold",
             color=INK)

    # ---- aux (b): forest of DIF effect sizes ------------------------------
    order = np.argsort(delta)
    ypos = np.arange(N_ITEM)
    for lo, hi, sh in ((ETS_B, ETS_C, 0.10), (ETS_C, 3.2, 0.18)):
        for s in (-1, 1):
            ax_fo.axvspan(s * lo, s * hi, color=GREY, alpha=sh, lw=0.0,
                          zorder=0)
    ax_fo.axvline(0.0, color=GREY, lw=0.8, zorder=2)
    cols = [ORANGE if delta[j] > 0 else BLUE for j in order]
    ax_fo.errorbar(delta[order], ypos, xerr=1.96 * se[order], fmt="none",
                   ecolor=HAIR, elinewidth=0.8, capsize=1.2, zorder=3)
    ax_fo.scatter(delta[order], ypos, s=13, facecolor=cols, edgecolor="white",
                  linewidth=0.4, zorder=4)
    for j in show:
        k = int(np.flatnonzero(order == j)[0])
        ax_fo.text(delta[j] + (0.16 if delta[j] >= 0 else -0.16), k,
                   f"item {j + 1:02d}", ha="left" if delta[j] >= 0 else "right",
                   va="center", fontsize=6, color=INK)
    for x_lab, txt in ((-2.35, "C"), (-1.25, "B"), (0.0, "A"), (1.25, "B"),
                       (2.35, "C")):
        ax_fo.text(x_lab, N_ITEM + 1.7, txt, ha="center", va="top",
                   fontsize=6.6, color=GREY, fontweight="bold")
    ax_fo.text(-2.92, N_ITEM + 3.4,
               f"ETS severity bands: {int((level == 'C').sum())} items reach "
               f"C, {int((level == 'B').sum())} reach B, "
               f"{int((level == 'A').sum())} stay at A",
               ha="left", va="top", fontsize=6.2, color=SUB)
    ax_fo.set_xlim(-3.0, 3.0)
    ax_fo.set_ylim(-1.0, N_ITEM + 3.6)
    ax_fo.set_yticks([])
    ax_fo.set_xticks(np.arange(-3, 4))
    ax_fo.set_xlabel("MH delta (ETS units; + favours the focal group)",
                     fontsize=7.4)
    ax_fo.tick_params(labelsize=6.6)
    ax_fo.spines["left"].set_visible(False)
    ax_fo.grid(axis="x", lw=0.5, color="#f2f2f2")
    ax_fo.set_title("Every item, ranked, with the ETS bands behind it",
                    fontsize=8.5, fontweight="bold", color=INK)

    # ---- aux (c): funnel of the same statistic against precision ----------
    scale = float(np.median(se * np.sqrt(n_seen)))
    ngrid = np.linspace(n_seen.min() * 0.85, n_seen.max() * 1.06, 120)
    for z, ls in ((1.96, (0, (4.0, 2.0))), (3.29, (0, (1.6, 1.8)))):
        ax_fu.plot(z * scale / np.sqrt(ngrid), ngrid, color=GREY, lw=0.8,
                   ls=ls, zorder=2)
        ax_fu.plot(-z * scale / np.sqrt(ngrid), ngrid, color=GREY, lw=0.8,
                   ls=ls, zorder=2)
    ax_fu.axvline(0.0, color=GREY, lw=0.8, zorder=2)
    out = np.abs(delta) > 1.96 * scale / np.sqrt(n_seen)
    ax_fu.scatter(delta[~out], n_seen[~out], s=15, facecolor="#c8d4e2",
                  edgecolor="#8fa4bd", linewidth=0.4, zorder=3,
                  label=f"inside the funnel ({int((~out).sum())} items)")
    ax_fu.scatter(delta[out], n_seen[out], s=22,
                  facecolor=[ORANGE if delta[j] > 0 else BLUE
                             for j in np.flatnonzero(out)],
                  edgecolor="white", linewidth=0.5, zorder=4,
                  label=f"outside ({int(out.sum())} items)")
    for j in show:
        if out[j]:
            ax_fu.text(delta[j], n_seen[j] + 28, f"item {j + 1:02d}",
                       ha="center", va="bottom", fontsize=6, color=INK)
    ax_fu.set_xlim(-3.0, 3.0)
    ax_fu.set_ylim(n_seen.min() * 0.80, n_seen.max() * 1.14)
    ax_fu.set_xticks(np.arange(-3, 4))
    ax_fu.set_xlabel("MH delta (ETS units)", fontsize=7.4)
    ax_fu.set_ylabel("examinees on the item (n)", fontsize=7.4)
    ax_fu.tick_params(labelsize=6.6)
    ax_fu.grid(axis="y", lw=0.5, color="#f2f2f2")
    ax_fu.set_title("Funnel: small samples scatter, bias does not",
                    fontsize=8.5, fontweight="bold", color=INK)
    ax_fu.legend(loc="lower left", frameon=False, fontsize=6.2,
                 handlelength=1.0, labelspacing=0.24, borderaxespad=0.35)
    ax_fu.text(0.025, 0.97,
               "funnel limits 95 % (dashed) and\n99.9 % (dotted), width "
               "1/sqrt(n)",
               transform=ax_fu.transAxes, ha="left", va="top",
               fontsize=6.1, color=MUTED, linespacing=1.4)


    note = (
        f"{N_REF} reference and {N_FOC} focal examinees answer a "
        f"{N_ITEM}-item form under a 2PL model in which most items share "
        f"their parameters across groups; five do not. Examinees are matched "
        f"on total score in {N_STRATA} strata, a 2 x 2 table is built per "
        f"stratum, and the Mantel-Haenszel odds ratio gives ETS delta = "
        f"-2.35 ln(alpha) with a Robins-Breslow-Greenland standard error and "
        f"a continuity-corrected chi-square ({int((level == 'C').sum())} "
        f"items reach level C, {int((level == 'B').sum())} level B). Areas "
        f"quoted in the small multiples are trapezoidal integrals of the two "
        f"plotted curves over theta {THETA_LO:+.0f} to {THETA_HI:+.0f}: the "
        f"crossing item carries a signed area of only "
        f"{signed[1]:+.2f} against an absolute area of {absolute[1]:.2f}, "
        f"which is why non-uniform DIF survives a signed index.")
    fig.text(0.5, 0.16 / FIGH, textwrap.fill(note, 195), ha="center",
             va="bottom", fontsize=6.5, color=MUTED, linespacing=1.5)

    panel_label_inches(fig, cells[0], "(a)", dx=-0.56, dy=0.42)
    panel_label_inches(fig, ax_fo, "(b)", dx=-0.56, dy=0.12)
    panel_label_inches(fig, ax_fu, "(c)", dx=-0.56, dy=0.12)
    savefig(fig, "899_edm_dif_analysis.png")


if __name__ == "__main__":
    main()

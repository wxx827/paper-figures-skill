"""Figure 1181 - Out-of-distribution detection from the world model's score.

Two score arrays drive the whole page.  ``scores`` draws n_id in-distribution
and n_ood out-of-distribution novelty scores; panel (a) is their histogram,
and panels (b) and (c) are ROC and precision-recall curves swept over exactly
those numbers - not a second sample.  AUROC is printed as the Mann-Whitney
count it is (``U / (n_id * n_ood)``), so the reader can see the division.

The four frames along the bottom are ordered by score and carry scores taken
from those same arrays at fixed ranks; the markers (i)-(iv) in panel (a) are
where they sit, and the divider between frames (ii) and (iii) is the threshold
line of panel (a).
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes
from figkit import image_axes
from _wm_scenes import build

SEED = 1586  # original id; figure number is 1181
N_ID, N_OOD = 1500, 600
TILE = (98, 152)

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#d5d5d5"
IDC, OODC, THR = "#5f9dc6", "#7b3294", "#1a1a1a"
CURVE, BASE, GRID, CARD = "#0b3d63", "#8a8a8a", "#ededed", "#e2e8ee"


def scores(rng):
    """Novelty score of the world model: low = familiar, high = novel."""
    sid = rng.beta(2.6, 7.0, N_ID)
    sood = rng.beta(5.2, 3.4, N_OOD)
    return sid, sood


def roc_pr(sid, sood):
    """ROC and PR curves swept over the pooled scores, OOD = positive."""
    s = np.concatenate([sid, sood])
    y = np.concatenate([np.zeros(sid.size), np.ones(sood.size)])
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = np.concatenate([[0.0], tp / sood.size])
    fpr = np.concatenate([[0.0], fp / sid.size])
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / sood.size
    auroc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") \
        else float(np.trapz(tpr, fpr))
    ap = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))
    return fpr, tpr, rec, prec, auroc, ap


def main():
    set_pub_style(base_font=8.5)
    rng = np.random.default_rng(SEED)
    sid, sood = scores(rng)
    fpr, tpr, rec, prec, auroc, ap = roc_pr(sid, sood)

    # AUROC as the rank statistic it is: how many (ID, OOD) pairs are ordered
    u = int((sood[:, None] > sid[None, :]).sum())
    pairs = N_ID * N_OOD
    assert abs(u / pairs - auroc) < 2e-3, "AUROC and the pair count disagree"

    tau = float(np.quantile(sid, 0.95))
    n_below = int((sid < tau).sum())
    tpr_tau = float((sood >= tau).mean())
    fpr_tau = float((sid >= tau).mean())
    prec_tau = float((sood >= tau).sum() /
                     max((sood >= tau).sum() + (sid >= tau).sum(), 1))
    base_rate = N_OOD / float(N_ID + N_OOD)

    # four frames, at fixed ranks of the two score arrays
    picks = [("(i)", float(np.quantile(sid, 0.20)), "in-distribution",
              dict(kind="road", seed=SEED, t=0.12, cam=None)),
             ("(ii)", float(np.quantile(sid, 0.90)), "in-distribution",
              dict(kind="road", seed=SEED + 41, t=0.55, cam=None)),
             ("(iii)", float(np.quantile(sood, 0.48)), "out-of-distribution",
              dict(kind="road", seed=SEED + 17, t=0.34,
                   cam=dict(exposure=0.68, contrast=0.86, grain=0.030,
                            dof=3.4, vignette=0.52, sharpen=0.35,
                            wb=(0.94, 1.00, 1.12)))),
             ("(iv)", float(np.quantile(sood, 0.94)), "out-of-distribution",
              dict(kind="aerial", seed=SEED + 5, t=0.40, cam=None))]
    picks.sort(key=lambda p: p[1])
    assert picks[0][1] < tau and picks[1][1] < tau < picks[2][1], \
        "the four frames no longer straddle the threshold"

    fig = plt.figure(figsize=(10.4, 6.0))
    gs = fig.add_gridspec(2, 12, height_ratios=[1.0, 1.06], left=0.058,
                          right=0.984, top=0.845, bottom=0.075, wspace=1.5,
                          hspace=0.42)
    axh = fig.add_subplot(gs[0, 0:5])
    axr = fig.add_subplot(gs[0, 5:8])
    axp = fig.add_subplot(gs[0, 8:12])
    for a in (axh, axr, axp):
        a.set_axisbelow(True)
        style_axes(a)
        a.grid(color=GRID, lw=0.7, zorder=0)

    # ---- (a) score distributions -------------------------------------------
    bins = np.linspace(0.0, 1.0, 46)
    axh.hist(sid, bins=bins, color=IDC, alpha=0.80, lw=0.4,
             edgecolor="white", zorder=3, label=f"in-distribution ($n$ = "
             f"{N_ID})")
    axh.hist(sood, bins=bins, color=OODC, alpha=0.62, lw=0.4,
             edgecolor="white", zorder=4, label=f"out-of-distribution ($n$ = "
             f"{N_OOD})")
    ytop = axh.get_ylim()[1] * 1.30
    axh.set_ylim(0, ytop)
    axh.plot([tau, tau], [0, ytop * 0.80], color=THR, lw=1.2,
             ls=(0, (3.2, 2.0)), zorder=6)
    axh.text(tau + 0.018, ytop * 0.80, f"$\\tau$ = {tau:.3f}", ha="left",
             va="top", fontsize=7.6, color=THR, fontweight="bold", zorder=7,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.3,
                       alpha=0.92))
    for lab, sc, _, _ in picks:
        axh.plot([sc], [ytop * 0.055], marker="^", ms=5.4, mfc="white",
                 mec=INK, mew=1.0, zorder=8)
        axh.text(sc, ytop * 0.095, lab, ha="center", va="bottom",
                 fontsize=6.8, color=INK, zorder=8,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.0,
                           alpha=0.92))
    axh.set_xlim(0.0, 1.0)
    axh.set_xlabel("Novelty score  $s$", fontsize=8.6)
    axh.set_ylabel("frames", fontsize=8.2)
    axh.set_title("(a)  Two score distributions and one threshold",
                  fontsize=9.0, fontweight="bold", pad=4.0)
    leg = axh.legend(loc="upper left", frameon=True, facecolor="white",
                     edgecolor=HAIR, framealpha=0.94, fontsize=7.2,
                     handlelength=1.3, labelspacing=0.35, borderpad=0.45)
    leg.get_frame().set_linewidth(0.5)
    axh.text(0.985, 0.60, f"$\\tau$ = 95th percentile of the ID scores\n"
             f"{n_below} / {N_ID} = {n_below / N_ID:.3f} below it",
             transform=axh.transAxes, ha="right", va="top", fontsize=7.0,
             color=SUB, linespacing=1.5, zorder=8,
             bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=HAIR,
                       lw=0.5, alpha=0.95))

    # ---- (b) ROC ------------------------------------------------------------
    axr.plot([0, 1], [0, 1], color=BASE, lw=1.0, ls=(0, (3.0, 2.2)), zorder=2)
    axr.plot(fpr, tpr, color=CURVE, lw=1.9, zorder=4)
    axr.plot([fpr_tau], [tpr_tau], marker="o", ms=6.0, mfc="white", mec=THR,
             mew=1.5, zorder=6)
    axr.annotate(f"at $\\tau$: TPR {tpr_tau:.3f}\nFPR {fpr_tau:.3f}",
                 xy=(fpr_tau, tpr_tau), xytext=(0.30, 0.30), ha="left",
                 va="top", fontsize=7.0, color=INK, linespacing=1.5,
                 zorder=7,
                 bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=HAIR,
                           lw=0.5, alpha=0.95),
                 arrowprops=dict(arrowstyle="-", lw=0.8, color="#333333",
                                 shrinkA=3, shrinkB=4))
    axr.set_xlim(0, 1)
    axr.set_ylim(0, 1.0)
    axr.set_aspect("equal")
    axr.set_xticks([0, 0.5, 1.0])
    axr.set_yticks([0, 0.5, 1.0])
    axr.set_xlabel("false-positive rate", fontsize=8.2)
    axr.set_ylabel("true-positive rate", fontsize=8.2)
    axr.tick_params(labelsize=7.2, length=2.4)
    axr.set_title(f"(b)  ROC, AUROC = {auroc:.3f}", fontsize=9.0,
                  fontweight="bold", pad=4.0)

    # ---- (c) precision-recall ----------------------------------------------
    axp.plot([0, 1], [base_rate, base_rate], color=BASE, lw=1.0,
             ls=(0, (3.0, 2.2)), zorder=2)
    axp.plot(rec, prec, color=OODC, lw=1.9, zorder=4)
    axp.plot([tpr_tau], [prec_tau], marker="o", ms=6.0, mfc="white", mec=THR,
             mew=1.5, zorder=6)
    axp.text(0.985, base_rate - 0.030, f"chance = {N_OOD} / {N_ID + N_OOD} = "
             f"{base_rate:.3f}", ha="right", va="top", fontsize=7.0,
             color=MUTED, zorder=7,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.3,
                       alpha=0.92))
    axp.annotate(f"at $\\tau$: recall {tpr_tau:.3f}\nprecision "
                 f"{prec_tau:.3f}", xy=(tpr_tau, prec_tau),
                 xytext=(0.035, 0.045), ha="left", va="bottom", fontsize=7.0,
                 color=INK, linespacing=1.5, zorder=7,
                 bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=HAIR,
                           lw=0.5, alpha=0.95),
                 arrowprops=dict(arrowstyle="-", lw=0.8, color="#333333",
                                 shrinkA=3, shrinkB=4))
    axp.set_xlim(0, 1)
    axp.set_ylim(0, 1.02)
    axp.set_aspect("equal")
    axp.set_xticks([0, 0.5, 1.0])
    axp.set_yticks([0, 0.5, 1.0])
    axp.set_xlabel("recall (OOD = positive)", fontsize=8.2)
    axp.set_ylabel("precision", fontsize=8.2)
    axp.tick_params(labelsize=7.2, length=2.4)
    axp.set_title(f"(c)  Precision-recall, AP = {ap:.3f}", fontsize=9.0,
                  fontweight="bold", pad=4.0)

    # ---- (d) four frames in score order ------------------------------------
    tiles = []
    for c, (lab, sc, verdict, spec) in enumerate(picks):
        ax = fig.add_subplot(gs[1, 3 * c:3 * c + 3])
        sceneshot = build(spec["kind"], seed=spec["seed"], shape=TILE,
                          t=spec["t"], cam=spec["cam"])
        ax.imshow(np.clip(sceneshot["rgb"], 0, 1), interpolation="bilinear")
        image_axes(ax, color="#9a9a9a", lw=0.8)
        col = OODC if sc > tau else IDC
        for s in ax.spines.values():
            s.set_color(col)
            s.set_linewidth(1.4)
        ax.set_title(f"{lab}   $s$ = {sc:.3f}", fontsize=8.2,
                     fontweight="bold", color=col, pad=3.0)
        ax.text(0.5, -0.075, f"{verdict}, {'flagged' if sc > tau else 'kept'}"
                f"  ($s$ {'>' if sc > tau else '<'} $\\tau$)",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.0,
                color=SUB)
        tiles.append(ax)

    fig.canvas.draw()
    b1 = tiles[1].get_position()
    b2 = tiles[2].get_position()
    xd = 0.5 * (b1.x1 + b2.x0)
    fig.add_artist(Line2D([xd, xd], [b1.y0 - 0.055, b1.y1 + 0.052],
                          color=THR, lw=1.2, ls=(0, (3.2, 2.0))))
    fig.text(xd, b1.y1 + 0.062, f"threshold $\\tau$ = {tau:.3f}", ha="center",
             va="bottom", fontsize=7.4, color=THR, fontweight="bold")

    # ---- headline + arithmetic ---------------------------------------------
    # wrapped: on one line this card is 10.3 in wide and the tight bbox took
    # the export out to 10.79 in, dropping the delivered h/w to 0.547
    savefig(fig, "1181_wm_unc_ood_detection.png")


if __name__ == "__main__":
    main()

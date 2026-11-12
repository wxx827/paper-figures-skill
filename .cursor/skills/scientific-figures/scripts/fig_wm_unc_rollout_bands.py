"""Figure 1177 - Predictive bands widen with the horizon, but not fast enough.

One data source drives both panels.  ``model_samples`` draws K rollouts from
the world model; the quantile bands are ``np.quantile`` of that array.
``truth_episodes`` draws M held-out ground-truth episodes from a slightly
different process; the coverage panel on the right counts how many of *those*
episodes fall inside *those* bands.  Change the sampler and the bands, the
in/out markers, the counts and every printed fraction move together.

Nothing is a hand-typed constant: every number on the page is an f-string over
an array, and the coverage panel prints the division itself (``112 / 120 =
0.933``) so the reader never has to trust a rounded ratio.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, style_axes

SEED = 1582  # original id; figure number is 1177
H = 40                     # prediction horizon, steps
K = 2000                   # rollouts sampled from the model
M = 400                    # held-out ground-truth episodes
CHECKS = (5, 10, 20, 30, 40)

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#d5d5d5"
BAND95, BAND80, BAND50 = "#dbe8f2", "#a9cade", "#6ea8cd"
MEDIAN, SAMPLE = "#0b3d63", "#2e74a8"
TRUTH, MISS = "#1a1a1a", "#7b3294"
GRID = "#ededed"


def _lane_change(h, amp, centre, width):
    """A sigmoidal lane change: the manoeuvre every sample disagrees about."""
    return amp[:, None] / (1.0 + np.exp(-(h[None, :] - centre[:, None]) /
                                        width[:, None]))


def model_samples(rng, n=K):
    """K rollouts of the predicted lateral offset, in metres."""
    h = np.arange(1, H + 1, dtype=float)
    amp = rng.normal(1.60, 0.115, n)
    centre = rng.normal(21.0, 3.00, n)
    width = np.clip(rng.normal(4.6, 0.70, n), 2.2, None)
    drift = np.cumsum(rng.normal(0.0, 0.0190, (n, H)), axis=1)
    return h, _lane_change(h, amp, centre, width) + drift


def truth_episodes(rng, n=M):
    """Held-out ground truth: the same manoeuvre, a wider real spread."""
    h = np.arange(1, H + 1, dtype=float)
    amp = rng.normal(1.60, 0.240, n)
    centre = rng.normal(21.2, 3.20, n)
    width = np.clip(rng.normal(4.7, 0.75, n), 2.2, None)
    drift = np.cumsum(rng.normal(0.0, 0.0170, (n, H)), axis=1)
    return h, _lane_change(h, amp, centre, width) + drift


def main():
    set_pub_style(base_font=8.5)
    rng = np.random.default_rng(SEED)
    h, samp = model_samples(rng)
    _, truth = truth_episodes(np.random.default_rng(SEED + 7))

    q = {p: np.quantile(samp, p, axis=0)
         for p in (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)}
    assert np.all(q[0.025] <= q[0.10]) and np.all(q[0.90] <= q[0.975]), \
        "quantile bands are not nested"

    inside = (truth >= q[0.025][None, :]) & (truth <= q[0.975][None, :])
    n_in = inside.sum(axis=0)                       # per horizon step
    cover = n_in / float(M)

    # The episode drawn in the main panel is a real member of ``truth``, not a
    # hand-drawn curve: of the episodes that start inside the band and are
    # outside at the final step, take the one that stays inside longest - the
    # honest illustration of "the band only fails late".
    tails = np.array([int(np.argmax(r[::-1])) if r.any() else H
                      for r in inside])          # trailing out-of-band run
    ok = np.nonzero(inside[:, :12].all(axis=1) & (tails >= 5) &
                    (tails <= 11))[0]
    j = int(ok[np.argmax(tails[ok])]) if ok.size else \
        int(np.argmin(inside.sum(axis=1)))
    gt, gt_in, tail = truth[j], inside[j], int(tails[j])
    steps_in = int(gt_in.sum())

    fig = plt.figure(figsize=(10.4, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.34], left=0.062,
                          right=0.978, top=0.845, bottom=0.125, wspace=0.20)
    ax = fig.add_subplot(gs[0, 0])
    axc = fig.add_subplot(gs[0, 1])
    for a in (ax, axc):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main panel: bands, samples, ground truth --------------------------
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.fill_between(h, q[0.025], q[0.975], color=BAND95, lw=0, zorder=1)
    ax.fill_between(h, q[0.10], q[0.90], color=BAND80, lw=0, zorder=2)
    ax.fill_between(h, q[0.25], q[0.75], color=BAND50, lw=0, zorder=3)
    for k in range(0, K, K // 13):
        ax.plot(h, samp[k], color=SAMPLE, lw=0.55, alpha=0.42, zorder=4)
    ax.plot(h, q[0.50], color=MEDIAN, lw=1.9, zorder=6)

    ax.plot(h, gt, color=TRUTH, lw=1.35, ls=(0, (3.4, 1.9)), zorder=7)
    ax.plot(h[gt_in], gt[gt_in], ls="none", marker="o", ms=3.4, mfc="white",
            mec=TRUTH, mew=0.95, zorder=8)
    ax.plot(h[~gt_in], gt[~gt_in], ls="none", marker="X", ms=5.6, mfc=MISS,
            mec="white", mew=0.7, zorder=9)

    ylo = float(min(q[0.025].min(), gt.min())) - 0.16
    yhi = float(max(q[0.975].max(), gt.max())) + 0.30
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(0.5, H + 0.5)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])

    # horizon check marks, drawn only inside the data area (never axvline:
    # that would run through the y tick labels in the left margin)
    for c in CHECKS:
        ax.plot([c, c], [ylo, yhi - 0.06], color="#c3cdd6", lw=0.7,
                ls=(0, (1.6, 2.2)), zorder=1)

    k_out = H - tail
    ax.annotate(f"outside the 95 % band for the\nlast {tail} steps "
                f"($h$ = {k_out + 1}...{H})",
                xy=(float(h[k_out]), float(gt[k_out])),
                xytext=(H - 0.3, ylo + 0.40 * (yhi - ylo)), ha="right",
                va="bottom",
                fontsize=7.2, color=MISS, linespacing=1.5, zorder=11,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cbb9d8",
                          lw=0.6, alpha=0.95),
                arrowprops=dict(arrowstyle="-", lw=0.8, color=MISS,
                                shrinkA=3, shrinkB=4))

    # how much the band actually widens - the claim in the headline, measured
    w95 = q[0.975] - q[0.025]
    w_near, w_far = f"{w95[CHECKS[0] - 1]:.2f}", f"{w95[CHECKS[-1] - 1]:.2f}"
    ax.text(0.985, 0.030, f"95 % band width  {w_near} m at $h$ = {CHECKS[0]}"
            f"   ->   {w_far} m at $h$ = {CHECKS[-1]}\n"
            f"{w_far} / {w_near} = "
            f"{float(w_far) / float(w_near):.1f}x wider, yet coverage falls\n"
            f"the episode drawn here is inside for {steps_in} / {H} steps "
            f"= {steps_in / H:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2,
            color=SUB, linespacing=1.6, zorder=12,
            bbox=dict(boxstyle="round,pad=0.42", fc="#e2e8ee", ec="#c3cdd6",
                      lw=0.6, alpha=1.0))

    ax.set_xlabel("Prediction horizon  $h$  (steps, open loop)", fontsize=9.0)
    ax.set_ylabel("Predicted lateral offset  (m)", fontsize=9.0)
    ax.set_title(f"{K} sampled rollouts, one held-out episode over them",
                 fontsize=9.5, fontweight="bold", pad=5.0)

    handles = [Patch(facecolor=BAND50, lw=0, label="50 % band"),
               Patch(facecolor=BAND80, lw=0, label="80 % band"),
               Patch(facecolor=BAND95, lw=0, label="95 % band"),
               Line2D([], [], color=MEDIAN, lw=1.9, label="predictive median"),
               Line2D([], [], color=SAMPLE, lw=0.7, alpha=0.6,
                      label="individual rollouts"),
               Line2D([], [], color=TRUTH, lw=1.3, ls=(0, (3.4, 1.9)),
                      marker="o", ms=3.4, mfc="white", mec=TRUTH,
                      label="ground truth (inside)"),
               Line2D([], [], color="none", marker="X", ms=5.6, mfc=MISS,
                      mec="white", mew=0.7, label="ground truth (outside)")]
    leg = ax.legend(handles=handles, loc="upper left", frameon=True,
                    facecolor="white", edgecolor=HAIR, framealpha=0.94,
                    fontsize=7.4, handlelength=1.7, labelspacing=0.42,
                    borderpad=0.5)
    leg.get_frame().set_linewidth(0.5)
    leg.set_zorder(12)

    # ---- right panel: empirical coverage of the 95 % band -------------------
    ypos = np.arange(len(CHECKS))[::-1].astype(float)
    axc.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    lo, hi = 0.78, 1.005
    axc.plot([0.95, 0.95], [ypos.min() - 0.62, ypos.max() + 0.62],
             color="#3b5b92", lw=1.0, ls=(0, (3.0, 2.0)), zorder=2)
    axc.text(0.95, ypos.max() + 0.72, "nominal 0.95", ha="center",
             va="bottom", fontsize=7.0, color="#3b5b92", fontweight="bold",
             zorder=6)

    shown = []
    for y, c in zip(ypos, CHECKS):
        i = c - 1
        v = float(cover[i])
        short = v < 0.95
        col = MISS if short else MEDIAN
        axc.plot([min(v, 0.95), max(v, 0.95)], [y, y], color=col, lw=1.6,
                 solid_capstyle="butt", zorder=3)
        axc.plot([v], [y], marker="o", ms=6.0, mfc=col, mec="white", mew=0.9,
                 zorder=4)
        txt = f"{n_in[i]} / {M} = {v:.3f}"
        shown.append((c, n_in[i], v))
        axc.text(lo + 0.012, y + 0.30, txt, ha="left", va="bottom",
                 fontsize=7.0, color=col, zorder=5,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.1,
                           alpha=0.92))

    for c, n, v in shown:
        assert abs(float(f"{v:.3f}") - n / M) < 5e-4, "coverage does not close"

    axc.set_ylim(ypos.min() - 0.72, ypos.max() + 1.15)
    axc.set_xlim(lo, hi)
    axc.set_yticks(ypos)
    axc.set_yticklabels([f"$h$ = {c}" for c in CHECKS], fontsize=8.0)
    axc.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00])
    axc.set_xticklabels(["0.80", "0.85", "0.90", "0.95", "1.00"],
                        fontsize=7.2)
    axc.set_xlabel("empirical coverage  (axis starts at 0.78)", fontsize=8.0)
    axc.set_title(f"95 % band vs {M} held-out episodes", fontsize=9.0,
                  fontweight="bold", pad=5.0)

    # ---- headline + closing arithmetic -------------------------------------
    c_near = f"{cover[CHECKS[0] - 1]:.3f}"
    c_far = f"{cover[CHECKS[-1] - 1]:.3f}"
    drop = float(c_near) - float(c_far)
    savefig(fig, "1177_wm_unc_rollout_bands.png")


if __name__ == "__main__":
    main()

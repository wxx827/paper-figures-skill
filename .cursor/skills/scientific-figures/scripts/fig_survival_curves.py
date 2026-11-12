"""Kaplan-Meier survival curves with confidence bands and risk tables.

Two independent cohorts, three risk strata each. Step curves carry Greenwood
log-log confidence bands and censoring ticks; a number-at-risk table sits
underneath each panel, and the log-rank P plus hazard ratio are boxed in the
corner. Estimator, variance and log-rank statistic are implemented directly
on numpy so no survival package is required.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

ARMS = ["Low risk", "Intermediate", "High risk"]
ARM_C = ["#2a7f7a", "#c9a227", "#b23a48"]
TICKS = [0, 12, 24, 36, 48, 60]
HORIZON = 60.0

COHORTS = [
    ("Discovery cohort", [62, 58, 60], [78.0, 42.0, 22.0], 5),
    ("Validation cohort", [48, 45, 47], [70.0, 38.0, 19.0], 17),
]


def simulate_arm(rng, n, median_months):
    """Weibull event times with administrative + random censoring."""
    shape = 1.35
    scale = median_months / np.log(2) ** (1 / shape)
    t_event = scale * rng.weibull(shape, n)
    t_cens = rng.uniform(8, 1.45 * HORIZON, n)
    time = np.minimum(t_event, np.minimum(t_cens, HORIZON))
    event = (t_event <= np.minimum(t_cens, HORIZON)).astype(int)
    return time, event


def km_estimate(time, event):
    """Kaplan-Meier survival with Greenwood log-log 95% bounds."""
    et = np.unique(time[event == 1])
    n_at = np.array([(time >= t).sum() for t in et], dtype=float)
    d = np.array([((time == t) & (event == 1)).sum() for t in et], dtype=float)
    surv = np.cumprod(1.0 - d / n_at)
    denom = n_at * (n_at - d)
    green = np.cumsum(np.divide(d, denom, out=np.zeros_like(d), where=denom > 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(green) / np.abs(np.log(np.maximum(surv, 1e-12)))
        lo = surv ** np.exp(1.96 * se)
        hi = surv ** np.exp(-1.96 * se)
    lo = np.clip(np.nan_to_num(lo, nan=0.0), 0, 1)
    hi = np.clip(np.nan_to_num(hi, nan=1.0), 0, 1)
    # pad with t=0 and carry the last step out to the follow-up horizon
    return (np.concatenate([[0.0], et, [HORIZON]]),
            np.concatenate([[1.0], surv, surv[-1:]]),
            np.concatenate([[1.0], lo, lo[-1:]]),
            np.concatenate([[1.0], hi, hi[-1:]]))


def logrank(times, events):
    """Multi-group log-rank: observed vs expected events per arm."""
    all_t = np.unique(np.concatenate([t[e == 1] for t, e in zip(times, events)]))
    k = len(times)
    obs, exp = np.zeros(k), np.zeros(k)
    for t in all_t:
        n_g = np.array([(tt >= t).sum() for tt in times], dtype=float)
        d_g = np.array([((tt == t) & (ee == 1)).sum()
                        for tt, ee in zip(times, events)], dtype=float)
        n, d = n_g.sum(), d_g.sum()
        if n <= 0:
            continue
        obs += d_g
        exp += d * n_g / n
    chi2 = np.sum((obs - exp) ** 2 / np.maximum(exp, 1e-9))
    p = stats.chi2.sf(chi2, k - 1)
    hr = (obs[-1] / exp[-1]) / (obs[0] / exp[0])
    se = np.sqrt(1 / max(exp[-1], 1e-9) + 1 / max(exp[0], 1e-9))
    return p, hr, np.exp(np.log(hr) - 1.96 * se), np.exp(np.log(hr) + 1.96 * se)


def fmt_p(p):
    return "P < 0.0001" if p < 1e-4 else f"P = {p:.4f}".rstrip("0")


def main():
    set_pub_style(base_font=8)
    fig = plt.figure(figsize=(9.2, 5.6))
    gs = fig.add_gridspec(2, 2, left=0.088, right=0.985, top=0.855,
                          bottom=0.085, height_ratios=[1.0, 0.30],
                          hspace=0.13, wspace=0.25)

    for ci, (cname, sizes, medians, seed) in enumerate(COHORTS):
        rng = np.random.default_rng(seed)
        times, events = [], []
        for n, m in zip(sizes, medians):
            t, e = simulate_arm(rng, n, m)
            times.append(t); events.append(e)

        ax = fig.add_subplot(gs[0, ci])
        for a in range(len(ARMS)):
            t, s, lo, hi = km_estimate(times[a], events[a])
            ax.fill_between(t, lo, hi, step="post", color=ARM_C[a], alpha=0.14,
                            lw=0)
            ax.step(t, s, where="post", color=ARM_C[a], lw=1.7,
                    label=f"{ARMS[a]}  (n={sizes[a]})")
            # censoring ticks sit on the step curve
            cens = np.sort(times[a][events[a] == 0])
            cens = cens[cens > 0]
            idx = np.searchsorted(t, cens, side="right") - 1
            ax.plot(cens, s[np.clip(idx, 0, s.size - 1)], ls="", marker="|",
                    ms=4.2, mew=0.9, color=ARM_C[a], alpha=0.85)

        p, hr, lo_hr, hi_hr = logrank(times, events)
        ax.text(0.965, 0.955,
                f"Log-rank {fmt_p(p)}\n"
                f"HR (high vs low) = {hr:.2f}\n[{lo_hr:.2f}-{hi_hr:.2f}]",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.2,
                linespacing=1.45, fontweight="bold", color="#1a1a1a",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#c8ccd2",
                          lw=0.6))
        ax.axhline(0.5, color="#aaaaaa", ls=(0, (3, 3)), lw=0.6, zorder=0)
        ax.set_xlim(0, HORIZON)
        ax.set_ylim(0, 1.045)
        ax.set_xticks(TICKS)
        ax.set_yticks(np.arange(0, 1.01, 0.25))
        ax.set_yticklabels([f"{v:.2f}" for v in np.arange(0, 1.01, 0.25)])
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_title(cname, fontsize=9.2, fontweight="bold", loc="left", pad=4)
        ax.legend(loc="lower left", fontsize=7.2, handlelength=1.6,
                  borderpad=0.2, labelspacing=0.35,
                  bbox_to_anchor=(0.005, 0.005))
        style_axes(ax)
        if ci == 0:
            ax.set_ylabel("Progression-free survival")
        add_panel_label(ax, f"({chr(97 + ci)})", x=-0.095, y=1.02)

        # ---- number-at-risk table -----------------------------------------
        axr = fig.add_subplot(gs[1, ci])
        axr.set_xlim(0, HORIZON)
        axr.set_ylim(-0.5, len(ARMS) - 0.4)
        axr.invert_yaxis()
        axr.set_xticks(TICKS)
        axr.set_yticks(range(len(ARMS)))
        axr.set_yticklabels(ARMS, fontsize=7.2)
        for lab, c in zip(axr.get_yticklabels(), ARM_C):
            lab.set_color(c)
        axr.tick_params(axis="both", length=0)
        axr.tick_params(axis="y", pad=7)
        for s in axr.spines.values():
            s.set_visible(False)
        axr.set_xlabel("Time since randomisation (months)")
        axr.set_title("Number at risk", fontsize=7.5, fontweight="bold",
                      loc="left", pad=3, color="#444444")
        for a in range(len(ARMS)):
            for ti, tk in enumerate(TICKS):
                # nudge the edge columns inwards so they clear the arm labels
                ha = "left" if ti == 0 else "right" if ti == len(TICKS) - 1 \
                    else "center"
                axr.text(tk, a, str(int((times[a] >= tk).sum())),
                         ha=ha, va="center", fontsize=7, color="#1a1a1a")

    savefig(fig, "74_survival_curves.png")


if __name__ == "__main__":
    main()

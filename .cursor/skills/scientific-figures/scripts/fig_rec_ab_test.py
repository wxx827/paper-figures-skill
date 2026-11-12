"""Figure 477 - Online A/B read-out for a ranking change.

Top row (auxiliary): the two headline metrics day by day, control against
treatment with 95 % bands and the ramp-up day marked, including the two-week
pre-period during which the arms are identical - without it a reader cannot
tell a real lift from a pre-existing imbalance.

Bottom (main): the per-segment relative lift as a forest plot.  Point plus
95 % interval, zero line dashed, intervals that exclude zero drawn solid and
bold, and the sample size and p value printed in aligned columns on the right.
Every interval, star and p value comes from the same simulated arm counts.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from pubstyle import set_pub_style, savefig, sig_stars, lighten
from figkit import axes_inches, panel_label_inches

INK, SUB, MUTED, FAINT = "#1a1a1a", "#555555", "#777777", "#8a8a8a"
HAIR, GRID = "#cccccc", "#d9d9d9"

CTRL = "#8a8a8a"
TREAT = "#3b5b92"
POOLED = "#b23a48"      # the single emphasis: the overall read-out

PRE, POST = 14, 21      # days before / after the ramp
# segment, exposed users per arm, true relative lift
SEGMENTS = [
    ("New users (< 7 d)",     128_000, 0.081),
    ("Casual (1-3 visits/wk)", 96_000, 0.032),
    ("Core (daily)",          384_000, 0.011),
    ("Mobile app",            516_000, 0.029),
    ("Desktop web",           186_000, 0.004),
    ("Cold-start items",      114_000, 0.104),
    ("Long-tail items",       162_000, 0.067),
    ("Head items",            435_000, -0.021),
]
BASE_CTR = 0.0432       # control click-through rate


def daily(rng, base, lift, noise, n_arm):
    """Control and treatment day series with their standard errors."""
    day = np.arange(-PRE, POST)
    season = 1.0 + 0.06 * np.sin(2 * np.pi * day / 7.0)
    ctrl = base * season * np.exp(rng.normal(0.0, noise, day.size))
    ramp = np.clip(day / 5.0, 0.0, 1.0) * (day >= 0)
    treat = ctrl * (1.0 + lift * ramp) * np.exp(rng.normal(0.0, noise * 0.6,
                                                           day.size))
    se_c = np.sqrt(ctrl * (1 - ctrl) / n_arm) if base < 1 else ctrl * 0.010
    se_t = np.sqrt(treat * (1 - treat) / n_arm) if base < 1 else treat * 0.010
    return day, ctrl, se_c, treat, se_t


def metric_panel(ax, day, ctrl, se_c, treat, se_t, ylabel, title, fmt):
    ax.axvspan(-PRE - 0.5, -0.5, facecolor="#f5f5f5", edgecolor="none",
               zorder=0)
    for y, se, col, lab in ((ctrl, se_c, CTRL, "control"),
                            (treat, se_t, TREAT, "treatment")):
        ax.fill_between(day, y - 1.96 * se, y + 1.96 * se,
                        facecolor=lighten(col, 0.66), edgecolor="none",
                        zorder=2)
        ax.plot(day, y, lw=1.5 if col == TREAT else 1.2, color=col, zorder=4,
                label=lab)
    ax.axvline(0.0, color="#3a3a3a", lw=0.9, ls=(0, (3.2, 2.2)), zorder=5)
    ax.text(0.5, 0.055, "ramp-up", transform=ax.get_xaxis_transform(),
            ha="left", va="bottom", fontsize=6.4, color=SUB)
    ax.text(-PRE + 0.5, 0.055, "pre-period",
            transform=ax.get_xaxis_transform(), ha="left", va="bottom",
            fontsize=6.4, color=MUTED)
    ax.set_xlim(-PRE - 0.5, POST - 0.5)
    ax.set_xlabel("day relative to ramp-up (d)", fontsize=8.0)
    ax.set_ylabel(ylabel, fontsize=8.0)
    ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(labelsize=7.0, length=2.5)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=3.0)
    ax.legend(loc="upper left", fontsize=7.0, frameon=False, ncol=2,
              handlelength=1.3, borderaxespad=0.3,
              bbox_to_anchor=(0.0, 1.0))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(477)

    fig = plt.figure(figsize=(10.4, 6.4))

    # ---- (a) / (b) daily metrics ------------------------------------------
    ax_a = axes_inches(fig, (0.78, 3.98, 4.05, 1.55))
    ax_b = axes_inches(fig, (5.92, 3.98, 4.05, 1.55))
    d, c, sc, t, st = daily(rng, BASE_CTR, 0.031, 0.028, 240_000)
    metric_panel(ax_a, d, 100 * c, 100 * sc, 100 * t, 100 * st,
                 "CTR (%)", "Click-through rate",
                 plt.FuncFormatter(lambda v, _p: f"{v:.1f}"))
    d2, c2, sc2, t2, st2 = daily(rng, 0.281, 0.024, 0.021, 240_000)
    metric_panel(ax_b, d2, c2, sc2, t2, st2, "NDCG@10 (-)",
                 "Ranking quality",
                 plt.FuncFormatter(lambda v, _p: f"{v:.3f}"))

    # ---- (c) forest of per-segment lift ------------------------------------
    ax_f = axes_inches(fig, (2.02, 0.62, 5.10, 2.62))
    rows, n_tot = [], 0
    for name, n, lift in SEGMENTS:
        p_c = BASE_CTR * float(np.exp(rng.normal(0.0, 0.16)))
        k_c = rng.binomial(n, p_c)
        k_t = rng.binomial(n, min(p_c * (1.0 + lift), 0.999))
        r_c, r_t = k_c / n, k_t / n
        # Delta method on the log ratio, the standard read-out for a rate lift.
        se = np.sqrt((1 - r_c) / k_c + (1 - r_t) / k_t)
        est = np.log(r_t / r_c)
        z = est / se
        p = 2.0 * stats.norm.sf(abs(z))
        rows.append((name, n, 100 * (np.exp(est) - 1.0),
                     100 * (np.exp(est - 1.96 * se) - 1.0),
                     100 * (np.exp(est + 1.96 * se) - 1.0), p))
        n_tot += 2 * n

    # Inverse-variance pooled estimate across the segments.
    ests = np.array([np.log1p(r[2] / 100) for r in rows])
    ses = np.array([(np.log1p(r[4] / 100) - np.log1p(r[3] / 100)) / (2 * 1.96)
                    for r in rows])
    w = 1.0 / ses ** 2
    pe = float((w * ests).sum() / w.sum())
    pse = float(np.sqrt(1.0 / w.sum()))
    pp = 2.0 * stats.norm.sf(abs(pe / pse))
    rows.append(("Overall (pooled)", n_tot // 2, 100 * (np.exp(pe) - 1.0),
                 100 * (np.exp(pe - 1.96 * pse) - 1.0),
                 100 * (np.exp(pe + 1.96 * pse) - 1.0), pp))

    ys = np.arange(len(rows))[::-1]
    ax_f.axvline(0.0, color="#3a3a3a", lw=0.9, ls=(0, (3.2, 2.2)), zorder=2)
    for y, (name, n, est, lo, hi, p) in zip(ys, rows):
        pooled = name.startswith("Overall")
        sig = (lo > 0) or (hi < 0)
        col = POOLED if pooled else (TREAT if sig else "#9a9a9a")
        lw = 2.0 if (sig or pooled) else 1.1
        ax_f.plot([lo, hi], [y, y], lw=lw, color=col, solid_capstyle="butt",
                  zorder=3)
        ax_f.plot([est], [y], marker="D" if pooled else "o",
                  ms=5.0 if pooled else 4.2, color=col,
                  mfc=col if (sig or pooled) else "white", mec=col, mew=1.1,
                  zorder=4)
        if pooled:
            ax_f.axhline(y + 0.5, color=HAIR, lw=0.7, zorder=1)

    ax_f.set_yticks(ys)
    ax_f.set_yticklabels([r[0] for r in rows], fontsize=7.0)
    ax_f.set_ylim(-0.8, len(rows) - 0.2)
    span = max(abs(min(r[3] for r in rows)), abs(max(r[4] for r in rows)))
    ax_f.set_xlim(-span * 1.12, span * 1.12)
    ax_f.set_xlabel("relative lift in CTR (%)", fontsize=8.0)
    ax_f.tick_params(labelsize=7.0, length=2.5)
    ax_f.grid(axis="x", color=GRID, linewidth=0.6)
    ax_f.set_title("Segment-level lift, 95 % confidence intervals",
                   fontsize=9.0, fontweight="bold")
    for s in ("top", "right", "left"):
        ax_f.spines[s].set_visible(False)
    ax_f.tick_params(axis="y", length=0)

    # Right-hand statistics columns, in axes fractions so they stay aligned.
    cols = ((1.045, "lift (95 % CI)"), (1.400, "n / arm"), (1.560, "p"))
    for x, head in cols:
        ax_f.text(x, len(rows) - 0.55, head, transform=ax_f.get_yaxis_transform(),
                  ha="left", va="center", fontsize=6.8, color=SUB,
                  fontweight="bold", clip_on=False)
    for y, (name, n, est, lo, hi, p) in zip(ys, rows):
        pooled = name.startswith("Overall")
        col = POOLED if pooled else INK
        wt = "bold" if pooled else "normal"
        ax_f.text(cols[0][0], y, f"{est:+.1f}  ({lo:+.1f}, {hi:+.1f})",
                  transform=ax_f.get_yaxis_transform(), ha="left",
                  va="center", fontsize=6.6, color=col, fontweight=wt,
                  clip_on=False)
        ax_f.text(cols[1][0], y, f"{n:,}", transform=ax_f.get_yaxis_transform(),
                  ha="left", va="center", fontsize=6.6, color=SUB,
                  clip_on=False)
        ptxt = "< 0.001" if p < 0.001 else f"{p:.3f}"
        ax_f.text(cols[2][0], y, f"{ptxt} {sig_stars(p)}",
                  transform=ax_f.get_yaxis_transform(), ha="left",
                  va="center", fontsize=6.6, color=col, fontweight=wt,
                  clip_on=False)

    gains = [r for r in rows[:-1] if r[3] > 0] or rows[:-1]
    best = max(gains, key=lambda r: r[2])
    panel_label_inches(fig, ax_a, "(a)", dx=-0.56, dy=0.10)
    panel_label_inches(fig, ax_b, "(b)", dx=-0.56, dy=0.10)
    panel_label_inches(fig, ax_f, "(c)", dx=-1.30, dy=0.10)
    savefig(fig, "477_rec_ab_test.png")


if __name__ == "__main__":
    main()

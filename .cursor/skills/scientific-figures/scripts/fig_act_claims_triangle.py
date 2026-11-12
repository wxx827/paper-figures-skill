"""Run-off triangle: what has been paid, and what the chain ladder adds.

Main panel: cumulative paid claims by accident year (rows) and development
year (columns).  The observed lower-left triangle is a warm sequential heat
map; the unobserved upper-right corner is left genuinely blank (NaN) and the
chain-ladder completion is laid over it as hatched blue cells with a dashed
edge, so a projection can never be mistaken for a payment.  A stepped rule
marks the latest diagonal.  The two narrow right-hand panels carry the
machinery and its consequence: the age-to-age development factors, and the
reserve waterfall that turns paid-to-date into ultimate loss.

Payments come from ``gen_triangle``; the factors, the reserve and every
annotated number are estimated from the plotted triangle at run time.
Values are illustrative.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from pubstyle import set_pub_style, savefig, style_axes, get_cmap, lighten
from figkit import axes_inches, panel_labels_inches, safe_colorbar

PAID = "#b23a48"
PROJ = "#3b5b92"
PROJ_FILL = "#eef3f8"
HATCH = lighten("#3b5b92", 0.66)
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"
DARK = "#3a3a3a"

FIG_W, FIG_H = 10.4, 6.4
MAIN_RECT = (0.86, 0.74, 4.90, 4.68)
AUX_T_RECT = (7.10, 3.47, 2.90, 1.95)
AUX_B_RECT = (7.10, 0.74, 2.90, 1.95)

N = 9
FIRST_AY = 2016
PATTERN = np.array([0.42, 0.235, 0.135, 0.082, 0.050, 0.031,
                    0.019, 0.017, 0.011])


def gen_triangle(seed=883):
    """Cumulative paid claims (k USD) and the observed-cell mask."""
    rng = np.random.default_rng(seed)
    p = PATTERN / PATTERN.sum()
    ult = 9800.0 * (1.052 ** np.arange(N)) * np.exp(rng.normal(0.0, 0.09, N))
    inc = ult[:, None] * p[None, :] * np.exp(rng.normal(0.0, 0.085, (N, N)))
    cum = np.cumsum(inc, axis=1)
    seen = np.zeros((N, N), dtype=bool)
    for i in range(N):
        seen[i, :N - i] = True
    return cum, seen


def chain_ladder(obs):
    """Volume-weighted and simple-average age-to-age factors, completed grid."""
    f_vw = np.empty(N - 1)
    f_sa = np.empty(N - 1)
    for j in range(N - 1):
        rows = [i for i in range(N) if i + j + 1 <= N - 1]
        num = sum(obs[i, j + 1] for i in rows)
        den = sum(obs[i, j] for i in rows)
        f_vw[j] = num / den
        f_sa[j] = float(np.mean([obs[i, j + 1] / obs[i, j] for i in rows]))
    full = obs.copy()
    for i in range(1, N):
        for j in range(N - i, N):
            full[i, j] = full[i, j - 1] * f_vw[j - 1]
    return f_vw, f_sa, full


def main():
    set_pub_style(base_font=8)
    cum, seen = gen_triangle()
    obs = np.where(seen, cum, np.nan)
    f_vw, f_sa, full = chain_ladder(obs)

    years = np.arange(FIRST_AY, FIRST_AY + N)
    paid_to_date = np.array([obs[i, N - 1 - i] for i in range(N)])
    ultimate = full[:, N - 1]
    reserve = ultimate - paid_to_date
    total_paid = paid_to_date.sum() / 1000.0            # M USD
    total_reserve = reserve.sum() / 1000.0
    total_ult = ultimate.sum() / 1000.0
    cdf = float(np.prod(f_vw))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_f = axes_inches(fig, AUX_T_RECT)
    ax_w = axes_inches(fig, AUX_B_RECT)
    for a in (ax_f, ax_w):
        a.set_axisbelow(True)
        a.grid(axis="y", color=GRID, lw=0.55, alpha=0.75, zorder=0)
        style_axes(a)

    # ---- main: the triangle -------------------------------------------------
    ax.set_axisbelow(False)
    cmap = get_cmap("cb_sequential_warm").with_extremes(bad=(0, 0, 0, 0))
    vmax = float(np.nanmax(obs))
    im = ax.imshow(obs, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto",
                   interpolation="nearest", zorder=2)

    for i in range(N):
        for j in range(N):
            if seen[i, j]:
                shade = obs[i, j] / vmax
                ax.text(j, i, f"{obs[i, j]:,.0f}", ha="center", va="center",
                        fontsize=6.2, color="white" if shade > 0.52 else INK,
                        zorder=5)
            else:
                # hatch colour follows the edge colour, so the tint and the
                # dashed border have to be two separate patches
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1.0, 1.0, facecolor=PROJ_FILL,
                    edgecolor=HATCH, linewidth=0.0, hatch="//", zorder=3))
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1.0, 1.0, facecolor="none",
                    edgecolor=PROJ, linewidth=0.7,
                    linestyle=(0, (2.2, 1.6)), zorder=4))
                t = ax.text(j, i, f"{full[i, j]:,.0f}", ha="center",
                            va="center", fontsize=6.2, color=PROJ,
                            style="italic", zorder=5)
                t.set_path_effects([pe.withStroke(linewidth=1.9,
                                                  foreground="white")])

    step_x, step_y = [], []
    for i in range(N):
        x = (N - 1 - i) + 0.5
        step_x += [x, x, x - 1.0]
        step_y += [i - 0.5, i + 0.5, i + 0.5]
    ax.plot(step_x, step_y, color=DARK, lw=1.5, zorder=6,
            solid_joinstyle="miter", clip_on=False)

    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels([str(j) for j in range(N)], fontsize=7.2)
    ax.set_yticklabels([str(y) for y in years], fontsize=7.2)
    ax.set_xticks(np.arange(-0.5, N, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1.0), minor=True)
    ax.grid(which="minor", color="white", lw=0.9, zorder=4)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("development year $j$ (years since accident year)",
                  fontsize=8.5)
    ax.set_ylabel("accident year $i$", fontsize=8.5)
    ax.set_title("Cumulative paid claims and the chain-ladder completion",
                 fontsize=9.0, fontweight="bold", pad=17.0)
    ax.xaxis.set_label_position("bottom")

    handles = [Patch(facecolor=cmap(0.62), edgecolor="none",
                     label="paid to date (observed)"),
               Patch(facecolor=PROJ_FILL, edgecolor=HATCH, hatch="//",
                     linewidth=0.0, label="chain-ladder projection"),
               Line2D([], [], color=DARK, lw=1.5, label="latest diagonal")]
    ax.legend(handles=handles, loc="lower center", frameon=False, ncol=3,
              fontsize=6.9, borderpad=0.25, columnspacing=1.6,
              handlelength=1.7, bbox_to_anchor=(0.5, 1.003))
    safe_colorbar(fig, im, ax, label="cumulative paid (k USD)", width=0.11,
                  pad=0.09)

    # ---- aux (b): age-to-age factors ---------------------------------------
    idx = np.arange(N - 1)
    ax_f.bar(idx, f_vw - 1.0, bottom=1.0, width=0.66, color=PROJ,
             edgecolor="white", linewidth=0.6, zorder=3,
             label="volume-weighted (selected)")
    ax_f.plot(idx, f_sa, ls="none", marker="D", ms=3.2, mfc="white",
              mec=PAID, mew=1.0, zorder=5, label="simple average")
    for k in idx:
        ax_f.text(k, f_vw[k] + 0.035 * (f_vw.max() - 1.0), f"{f_vw[k]:.3f}",
                  ha="center", va="bottom", fontsize=6.0, color=SUB,
                  rotation=90)
    ax_f.axhline(1.0, color=FAINT, lw=0.8, zorder=2)
    ax_f.set_xticks(idx)
    ax_f.set_xticklabels([f"{k}-{k + 1}" for k in idx], fontsize=6.4)
    ax_f.set_ylim(1.0, 1.0 + 1.42 * (f_vw.max() - 1.0))
    ax_f.tick_params(labelsize=6.8)
    ax_f.set_xlabel("development interval (years)", fontsize=7.8, labelpad=1.5)
    ax_f.set_ylabel("age-to-age factor (x)", fontsize=7.8)
    ax_f.set_title("Development factors run off to 1.0", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_f.legend(loc="upper right", frameon=False, fontsize=6.4,
                handlelength=1.4, labelspacing=0.35, borderpad=0.25)
    ax_f.text(0.985, 0.56, f"cumulative factor to ultimate = {cdf:.3f}x",
              transform=ax_f.transAxes, ha="right", va="top", fontsize=6.4,
              color=MUTED)

    # ---- aux (c): reserve waterfall ----------------------------------------
    recent = 4
    old_res = reserve[:N - recent].sum() / 1000.0
    steps = [("paid", total_paid, PAID, "total")]
    steps.append((f"<={years[N - recent - 1]}", old_res, PROJ, "delta"))
    for i in range(N - recent, N):
        steps.append((str(years[i]), reserve[i] / 1000.0, PROJ, "delta"))
    steps.append(("ult.", total_ult, DARK, "total"))

    run = 0.0
    for k, (name, val, col, kind) in enumerate(steps):
        if kind == "total":
            ax_w.bar(k, val, bottom=0.0, width=0.66, color=col,
                     edgecolor="white", linewidth=0.6, zorder=3)
            top = val
            run = val
        else:
            ax_w.bar(k, val, bottom=run, width=0.66, color=col,
                     edgecolor="white", linewidth=0.6, zorder=3)
            ax_w.plot([k - 0.33, k + 0.33], [run, run], color=FAINT, lw=0.6,
                      zorder=4)
            top = run + val
            run = top
        ax_w.text(k, top + 0.018 * total_ult, f"{val:,.1f}", ha="center",
                  va="bottom", fontsize=6.0, color=SUB, rotation=90)
    ax_w.set_xticks(np.arange(len(steps)))
    ax_w.set_xticklabels([s[0] for s in steps], fontsize=6.4)
    ax_w.set_ylim(0, total_ult * 1.22)
    ax_w.tick_params(labelsize=6.8)
    ax_w.set_xlabel("paid to date, then IBNR by accident year, to ultimate",
                    fontsize=7.5, labelpad=1.5)
    ax_w.set_ylabel("loss (M USD)", fontsize=7.8)
    ax_w.set_title("Reserve bridges paid to ultimate", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_w.text(0.03, 0.955,
              f"IBNR reserve = {total_reserve:,.1f} M USD\n"
              f"= {100.0 * total_reserve / total_ult:.1f}% of ultimate; "
              f"n = {N} accident years",
              transform=ax_w.transAxes, ha="left", va="top", fontsize=6.4,
              color=SUB, linespacing=1.45)

    panel_labels_inches(fig, [ax, ax_f, ax_w], dx=-0.56, dy=0.10)
    savefig(fig, "883_act_claims_triangle.png")


if __name__ == "__main__":
    main()

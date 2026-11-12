"""Wright map: examinee ability and item difficulty on one shared logit ruler.

The whole point of this display is that the two distributions are measured in
the same unit, so the three main sub-panels (person histogram, logit ruler,
item stack) are given the same y limits from one variable and never
autoscaled.  The targeting gap is not drawn by hand: it is the largest
interval between consecutive item difficulties, and the number of examinees
quoted inside it is counted from the ability sample.  The separation and
reliability card is computed the Rasch way - measurement error from the
information each side actually receives, true variance by subtraction - so a
card that says "reliable" cannot sit next to a map that shows a hole.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_label_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
HAIR = "#cccccc"
BLUE = "#3b5b92"          # examinees
ORANGE = "#c0603f"        # items
ACCENT = "#b23a48"

FIGW, FIGH = 10.4, 6.4
AX_P = (0.78, 1.05, 2.00, 4.35)      # person histogram
AX_S = (2.78, 1.05, 0.62, 4.35)      # shared logit ruler
AX_I = (3.46, 1.05, 2.77, 4.35)      # item difficulty stack
AUX_T = (7.05, 3.62, 3.00, 1.78)
AUX_B = (7.05, 1.05, 3.00, 1.78)

YLO, YHI = -4.6, 4.6
N_PERSON = 640
BINW = 0.25
REL_TARGET = 0.80


def cohort(seed=895):
    """Abilities in logits, and a blueprint that leaves a hole in the middle."""
    rng = np.random.default_rng(seed)
    theta = np.concatenate([
        rng.normal(0.55, 0.95, int(N_PERSON * 0.72)),
        rng.normal(-0.75, 1.15, N_PERSON - int(N_PERSON * 0.72))])
    d = np.concatenate([
        np.linspace(-2.70, -0.42, 19) + rng.normal(0.0, 0.055, 19),
        np.linspace(0.62, 2.60, 23) + rng.normal(0.0, 0.055, 23)])
    return np.clip(theta, YLO + 0.2, YHI - 0.2), np.sort(d)


def separation(x, se):
    """Rasch separation index and reliability from observed SD and error."""
    sd_obs = float(np.std(x, ddof=1))
    rmse = float(np.sqrt(np.mean(se ** 2)))
    sd_true = float(np.sqrt(max(sd_obs ** 2 - rmse ** 2, 1e-9)))
    g = sd_true / rmse
    return g, g ** 2 / (1.0 + g ** 2), sd_obs, rmse


def main():
    set_pub_style()
    theta, diff = cohort()

    # Rasch response probabilities: one matrix, both margins read off it.
    p = 1.0 / (1.0 + np.exp(-(theta[:, None] - diff[None, :])))
    w = p * (1.0 - p)
    se_person = 1.0 / np.sqrt(w.sum(axis=1))
    se_item = 1.0 / np.sqrt(w.sum(axis=0))
    g_p, rel_p, sd_p, rmse_p = separation(theta, se_person)
    g_i, rel_i, sd_i, rmse_i = separation(diff, se_item)
    targeting = float(theta.mean() - diff.mean())

    # The gap is measured, not drawn: widest gap between adjacent difficulties.
    k = int(np.argmax(np.diff(diff)))
    gap_lo, gap_hi = float(diff[k]), float(diff[k + 1])
    in_gap = ((theta > gap_lo) & (theta < gap_hi))
    n_gap = int(in_gap.sum())

    edges = np.arange(YLO, YHI + BINW, BINW)
    counts, _ = np.histogram(theta, bins=edges)
    centers = edges[:-1] + BINW / 2.0

    fig = plt.figure(figsize=(FIGW, FIGH))
    ax_p = axes_inches(fig, AX_P)
    ax_s = axes_inches(fig, AX_S)
    ax_i = axes_inches(fig, AX_I)
    ax_d = axes_inches(fig, AUX_T)
    ax_r = axes_inches(fig, AUX_B)
    for ax in (ax_p, ax_s, ax_i):
        ax.set_ylim(YLO, YHI)            # one ruler, set three times

    # ---- left: examinee ability ------------------------------------------
    ax_p.barh(centers, counts, height=BINW * 0.86, color=BLUE, alpha=0.85,
              edgecolor="white", linewidth=0.35, zorder=3)
    ax_p.axhline(theta.mean(), color=BLUE, lw=1.0, ls=(0, (4.0, 2.0)),
                 zorder=4)
    for s in (-1, 1):
        ax_p.axhline(theta.mean() + s * sd_p, color=BLUE, lw=0.7,
                     ls=(0, (1.6, 1.8)), alpha=0.8, zorder=4)
    ax_p.axhspan(gap_lo, gap_hi, color=ACCENT, alpha=0.07, lw=0.0, zorder=1)
    ax_p.set_xlim(0.0, counts.max() * 1.16)
    ax_p.invert_xaxis()
    ax_p.set_yticks([])
    ax_p.set_xlabel(f"examinees per {BINW:.2f}-logit bin (n = {N_PERSON})",
                    fontsize=7.6)
    ax_p.tick_params(labelsize=6.8)
    ax_p.spines["left"].set_visible(False)
    ax_p.spines["right"].set_visible(False)
    ax_p.set_title(f"examinee ability\nmean {theta.mean():+.2f}, "
                   f"SD {sd_p:.2f} logits",
                   fontsize=8.0, fontweight="bold", color=BLUE, pad=5.0)

    # ---- centre: the shared logit ruler ----------------------------------
    ax_s.set_xlim(0.0, 1.0)
    ax_s.axis("off")
    ax_s.plot([0.5, 0.5], [YLO, YHI], color=GREY, lw=0.9, zorder=2)
    for t in np.arange(-4, 5):
        ax_s.plot([0.40, 0.60], [t, t], color=GREY, lw=0.8, zorder=2)
        ax_s.text(0.5, t, f"{t:+d}", ha="center", va="center", fontsize=6.8,
                  color=INK, zorder=4,
                  bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                            edgecolor="none"))
    ax_s.plot([0.5], [theta.mean()], marker="<", ms=6.0, color=BLUE,
              zorder=5, clip_on=False)
    ax_s.plot([0.5], [diff.mean()], marker=">", ms=6.0, color=ORANGE,
              zorder=5, clip_on=False)
    ax_s.plot([0.30, 0.30], [theta.mean() - sd_p, theta.mean() + sd_p],
              color=BLUE, lw=2.4, alpha=0.55, solid_capstyle="butt", zorder=3)
    ax_s.plot([0.70, 0.70], [diff.mean() - sd_i, diff.mean() + sd_i],
              color=ORANGE, lw=2.4, alpha=0.55, solid_capstyle="butt",
              zorder=3)
    ax_s.text(0.5, YHI + 0.34, "logits", ha="center", va="bottom",
              fontsize=7.4, color=INK, fontweight="bold")
    ax_s.text(0.20, YLO + 0.10, "+/-1 SD", ha="center", va="bottom",
              fontsize=6.2, color=MUTED, rotation=90)

    # ---- right: item difficulty stack ------------------------------------
    # Items sharing a 0.16-logit band are placed side by side, so a column of
    # squares reads as a count the way a dot plot does.
    band = 0.16
    lane = np.floor((diff - YLO) / band).astype(int)
    slot = np.zeros(len(diff), dtype=int)
    seen = {}
    for j, b in enumerate(lane):
        slot[j] = seen.get(b, 0)
        seen[b] = slot[j] + 1
    xs = 0.42 + slot * 0.34
    ax_i.axhspan(gap_lo, gap_hi, color=ACCENT, alpha=0.07, lw=0.0, zorder=1)
    for t in np.arange(-4, 5):
        ax_i.axhline(t, color="#eeeeee", lw=0.6, zorder=0)
    ax_i.scatter(xs, diff, marker="s", s=30, facecolor=ORANGE,
                 edgecolor="white", linewidth=0.5, zorder=4)
    for x, y, n in zip(xs, diff, range(1, len(diff) + 1)):
        if n % 4 == 1:
            ax_i.text(x + 0.16, y, f"I{n:02d}", ha="left", va="center",
                      fontsize=6.0, color=MUTED, zorder=4)
    ax_i.axhline(diff.mean(), color=ORANGE, lw=1.0, ls=(0, (4.0, 2.0)),
                 zorder=5)
    for s in (-1, 1):
        ax_i.axhline(diff.mean() + s * sd_i, color=ORANGE, lw=0.7,
                     ls=(0, (1.6, 1.8)), alpha=0.8, zorder=5)
    ax_i.set_xlim(0.0, 2.55)
    ax_i.set_xticks([])
    ax_i.set_yticks([])
    for s in ax_i.spines.values():
        s.set_visible(False)
    ax_i.set_title(f"item difficulty, {len(diff)} items\n"
                   f"mean {diff.mean():+.2f}, SD {sd_i:.2f} logits",
                   fontsize=8.0, fontweight="bold", color=ORANGE, pad=5.0)
    ax_i.annotate(
        f"targeting gap {gap_hi - gap_lo:.2f} logits\n"
        f"({gap_lo:+.2f} to {gap_hi:+.2f}): no item here,\n"
        f"but {n_gap} examinees "
        f"({100.0 * n_gap / N_PERSON:.0f} % of the cohort) sit in it",
        xy=(0.60, (gap_lo + gap_hi) / 2.0),
        xytext=(1.22, (gap_lo + gap_hi) / 2.0 - 1.35),
        fontsize=6.7, color=ACCENT, ha="left", va="center", linespacing=1.4,
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8,
                        shrinkA=1.0, shrinkB=1.0))
    ax_i.text(2.50, YHI - 0.10,
              "one square per item; squares side by\nside share a difficulty "
              "to within 0.16 logits",
              ha="right", va="top", fontsize=6.2, color=MUTED,
              linespacing=1.35)

    # ---- aux (b): the same two distributions as densities ----------------
    grid = np.linspace(YLO, YHI, 401)
    band_p = 1.06 * sd_p * theta.size ** (-0.2)
    band_i = 1.06 * sd_i * diff.size ** (-0.2)
    dens_p = np.exp(-0.5 * ((grid[:, None] - theta[None, :]) / band_p) ** 2
                    ).sum(axis=1) / (theta.size * band_p * np.sqrt(2 * np.pi))
    dens_i = np.exp(-0.5 * ((grid[:, None] - diff[None, :]) / band_i) ** 2
                    ).sum(axis=1) / (diff.size * band_i * np.sqrt(2 * np.pi))
    ax_d.fill_between(grid, 0.0, dens_p, color=BLUE, alpha=0.22, lw=0.0)
    ax_d.plot(grid, dens_p, color=BLUE, lw=1.5,
              label=f"examinees  SD = {sd_p:.2f}")
    ax_d.fill_between(grid, 0.0, dens_i, color=ORANGE, alpha=0.22, lw=0.0)
    ax_d.plot(grid, dens_i, color=ORANGE, lw=1.5,
              label=f"items  SD = {sd_i:.2f}")
    ax_d.axvspan(gap_lo, gap_hi, color=ACCENT, alpha=0.09, lw=0.0)
    ax_d.set_xlim(YLO, YHI)
    ax_d.set_ylim(0.0, max(dens_p.max(), dens_i.max()) * 1.32)
    ax_d.set_xticks(np.arange(-4, 5, 2))
    ax_d.set_xlabel("logits", fontsize=7.4)
    ax_d.set_ylabel("density (1/logit)", fontsize=7.4)
    ax_d.tick_params(labelsize=6.6)
    ax_d.grid(axis="y", lw=0.5, color="#eeeeee")
    ax_d.set_title("Where the two distributions actually sit", fontsize=8.5,
                   fontweight="bold", color=INK)
    ax_d.legend(loc="upper left", frameon=False, fontsize=6.4,
                handlelength=1.4, labelspacing=0.28, borderaxespad=0.35)
    ax_d.text(YHI - 0.15, max(dens_p.max(), dens_i.max()) * 0.86,
              f"targeting =\nperson mean - item mean\n= {targeting:+.2f} logits",
              ha="right", va="top", fontsize=6.3, color=SUB, linespacing=1.4)

    # ---- aux (c): separation / reliability card --------------------------
    names = ["person reliability", "item reliability"]
    vals = [rel_p, rel_i]
    seps = [g_p, g_i]
    cols = [BLUE, ORANGE]
    ypos = np.array([1.0, 0.0])
    ax_r.barh(ypos, vals, height=0.42, color=cols, alpha=0.85,
              edgecolor="white", linewidth=0.5, zorder=3)
    ax_r.axvline(REL_TARGET, color=GREY, lw=0.9, ls=(0, (3.4, 2.0)), zorder=4)
    ax_r.text(REL_TARGET - 0.012, 1.72, f"target {REL_TARGET:.2f}",
              ha="right", va="top", fontsize=6.3, color=GREY)
    for y, v, g, c, nm in zip(ypos, vals, seps, cols, names):
        ax_r.text(v - 0.014, y, f"{v:.3f}", ha="right", va="center",
                  fontsize=7.0, color="white", fontweight="bold", zorder=5)
        ax_r.text(0.010, y + 0.26, nm, ha="left", va="bottom", fontsize=6.8,
                  color=c, zorder=5)
        ax_r.text(0.990, y + 0.26, f"separation G = {g:.2f}  ->  "
                                   f"{(4.0 * g + 1.0) / 3.0:.1f} strata",
                  ha="right", va="bottom", fontsize=6.4, color=SUB, zorder=5)
    ax_r.set_xlim(0.0, 1.0)
    ax_r.set_ylim(-0.62, 1.75)
    ax_r.set_yticks([])
    ax_r.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax_r.set_xlabel("Rasch reliability (-)", fontsize=7.4)
    ax_r.tick_params(labelsize=6.6)
    ax_r.spines["left"].set_visible(False)
    ax_r.set_title("Separation: how many distinct levels are resolved",
                   fontsize=8.5, fontweight="bold", color=INK)
    ax_r.text(0.012, -0.52,
              f"RMSE {rmse_p:.3f} logits (persons) / {rmse_i:.3f} (items)",
              ha="left", va="bottom", fontsize=6.3, color=MUTED)

    fig.text((AX_P[0] + AX_I[0] + AX_I[2]) / 2.0 / FIGW, 5.78 / FIGH,
             "One logit ruler, two populations: the cohort sits above the "
             "middle of the blueprint",
             ha="center", va="bottom", fontsize=9.0, fontweight="bold",
             color=INK)

    note = (
        f"Dichotomous Rasch model P = [1 + exp(-(theta - d))]^-1 for "
        f"{N_PERSON} examinees and {len(diff)} items on one logit scale, high "
        f"ability and high difficulty at the top of every panel. Person mean "
        f"{theta.mean():+.2f} logits (SD {sd_p:.2f}), item mean "
        f"{diff.mean():+.2f} logits (SD {sd_i:.2f}), so the blueprint is "
        f"{abs(targeting):.2f} logits "
        f"{'easier' if targeting > 0 else 'harder'} than the cohort. "
        "Standard errors come from the information each margin receives, "
        f"1/sqrt(sum P(1-P)); separation G = true SD / RMSE gives "
        f"G = {g_p:.2f} for persons and {g_i:.2f} for items. The shaded band "
        f"is the widest interval between adjacent item difficulties "
        f"({gap_hi - gap_lo:.2f} logits) and holds {n_gap} examinees.")
    fig.text(0.5, 0.16 / FIGH, textwrap.fill(note, 178), ha="center",
             va="bottom", fontsize=6.5, color=MUTED, linespacing=1.5)

    panel_label_inches(fig, ax_p, "(a)", dx=-0.56, dy=0.42)
    panel_label_inches(fig, ax_d, "(b)", dx=-0.56, dy=0.12)
    panel_label_inches(fig, ax_r, "(c)", dx=-0.56, dy=0.12)
    savefig(fig, "895_edm_wright_map.png")


if __name__ == "__main__":
    main()

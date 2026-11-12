"""Age-structured social contact matrix and its next-generation eigenvalue.

Main panel: mean daily contacts between 5-year age bands, built from four
additive components (household diagonal, school block, workplace block and
the parent-child off-diagonal wings). Auxiliary panels: marginal contact
intensity along the top and right edges, and the dominant eigenvalue of the
next-generation matrix under three contact-reduction scenarios.
"""
import numpy as np
import matplotlib.pyplot as plt
from figkit import halo_text, panel_label_inches, safe_colorbar
from pubstyle import (set_pub_style, savefig, style_axes, get_cmap,
                      annotate_box)

N_AGE = 16
LABELS = [f"{5 * i}-{5 * i + 4}" for i in range(N_AGE - 1)] + ["75+"]
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
INK = "#1a1a1a"
RULE = "#333333"
MUTED = "#777777"
DASH = (0, (4.0, 2.4))
R0_TARGET = 2.60


def components():
    """Four interpretable pieces of a synthetic POLYMOD-like matrix."""
    i, j = np.meshgrid(np.arange(N_AGE), np.arange(N_AGE), indexing="ij")
    household = 1.55 * np.exp(-((i - j) ** 2) / (2 * 1.25 ** 2))
    school = 4.30 * np.exp(-((i - 2.4) ** 2 + (j - 2.4) ** 2) / (2 * 1.45 ** 2))
    work = 1.30 * np.exp(-((i - 8.2) ** 2 + (j - 8.2) ** 2) / (2 * 3.10 ** 2))
    wings = (np.exp(-((i - j - 6) ** 2) / (2 * 1.30 ** 2)) +
             np.exp(-((j - i - 6) ** 2) / (2 * 1.30 ** 2)))
    wings *= 1.05 * np.exp(-((np.minimum(i, j) - 1.6) ** 2) / (2 * 2.4 ** 2))
    elderly = 0.50 * np.exp(-((i - 14.4) ** 2 + (j - 14.4) ** 2) / (2 * 1.9 ** 2))
    base = 0.12 * np.ones_like(household)
    return dict(household=household + wings + base + elderly,
                school=school, work=work)


def build(parts, school_scale=1.0, work_scale=1.0):
    m = parts["household"] + school_scale * parts["school"] \
        + work_scale * parts["work"]
    return 0.5 * (m + m.T)


def main():
    set_pub_style(base_font=9)
    parts = components()
    M = build(parts)
    q = R0_TARGET / float(np.linalg.eigvalsh(M).max())     # per-contact scaling
    lam = lambda m: q * float(np.linalg.eigvalsh(m).max())

    fig = plt.figure(figsize=(7.2, 4.5))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 0.24, 0.20, 0.66],
                          height_ratios=[0.26, 1.0], hspace=0.07, wspace=0.09,
                          left=0.085, right=0.985, top=0.845, bottom=0.235)

    ax_top = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    ax_rig = fig.add_subplot(gs[1, 1])
    ax_sc = fig.add_subplot(gs[1, 3])

    # ---- (a) the matrix ----------------------------------------------------
    im = ax.imshow(M, cmap=get_cmap("cb_sequential_blue"), origin="lower",
                   aspect="auto", vmin=0.0, vmax=float(M.max()))
    ticks = np.arange(0, N_AGE, 3)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([LABELS[t] for t in ticks], fontsize=6.6)
    ax.set_yticklabels([LABELS[t] for t in ticks], fontsize=6.6)
    ax.set_xlabel("Age of contact (years)", fontsize=8.0)
    ax.set_ylabel("Age of participant (years)", fontsize=8.0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    ax.tick_params(length=2.5)

    # block guides on the school and workplace clusters
    for (x0, x1, name) in ((0.5, 3.5, "school"), (4.5, 12.5, "workplace")):
        ax.plot([x0, x1, x1, x0, x0], [x0, x0, x1, x1, x0], color="white",
                lw=0.8, ls=DASH, alpha=0.85, zorder=4)
        halo_text(ax, (x0 + x1) / 2, x1 - 0.30, name, ha="center", va="top",
                  fontsize=6.2, color="white", halo=INK, zorder=5)

    annotate_box(ax, f"next-generation matrix\n"
                     f"dominant eigenvalue = {lam(M):.2f}",
                 loc="lower right", fontsize=6.6, linespacing=1.35)
    safe_colorbar(fig, im, ax, label="mean contacts (per person per day)",
                  side="bottom", width=0.10, pad=0.42, shrink=1.0)

    # ---- (b) top marginal --------------------------------------------------
    tot = M.sum(axis=0)
    ax_top.bar(np.arange(N_AGE), tot, width=0.82, color=BLUE, zorder=3)
    ax_top.set_xlim(-0.5, N_AGE - 0.5)
    ax_top.set_ylim(0, tot.max() * 1.18)
    ax_top.set_xticks([])
    ax_top.set_yticks([0, round(tot.max() / 2, 0) * 2])
    ax_top.tick_params(labelsize=6.2, length=2)
    ax_top.set_ylabel("total\ncontacts", fontsize=6.6, linespacing=1.2)
    j = int(np.argmax(tot))
    ax_top.text(j, tot[j] * 1.06, f"{LABELS[j]} y: {tot[j]:.0f} contacts/day",
                ha="center", va="bottom", fontsize=6.4, color=INK)
    style_axes(ax_top)

    # ---- (c) right marginal ------------------------------------------------
    ax_rig.barh(np.arange(N_AGE), tot, height=0.82, color=BLUE, zorder=3)
    ax_rig.set_ylim(-0.5, N_AGE - 0.5)
    ax_rig.set_xlim(0, tot.max() * 1.10)
    ax_rig.set_yticks([])
    ax_rig.set_xticks([0, round(tot.max() / 2, 0) * 2])
    ax_rig.tick_params(labelsize=6.2, length=2)
    ax_rig.set_xlabel("contacts\nper day", fontsize=6.6, linespacing=1.2)
    style_axes(ax_rig)

    # ---- (d) eigenvalue under contact reductions ---------------------------
    scen = [("Baseline", build(parts)),
            ("Schools closed", build(parts, school_scale=0.15)),
            ("Workplaces at 40%", build(parts, work_scale=0.40)),
            ("Both measures", build(parts, school_scale=0.15, work_scale=0.40))]
    vals = [lam(m) for _, m in scen]
    ypos = np.arange(len(scen))[::-1]
    cols = [GREEN if v < 1.0 else BLUE for v in vals]
    ax_sc.barh(ypos, vals, height=0.46, color=cols, zorder=3)
    for y, v, (name, _) in zip(ypos, vals, scen):
        ax_sc.text(v + 0.06, y, f"{v:.2f}", va="center", ha="left",
                   fontsize=6.8, color=INK)
        ax_sc.text(0.03, y + 0.34, name, va="bottom", ha="left",
                   fontsize=6.8, color=INK)
    ax_sc.axvline(1.0, color=RULE, ls=DASH, lw=0.9, zorder=4)
    ax_sc.text(1.06, -0.42, "epidemic threshold $R_t$ = 1",
               fontsize=6.4, color=RULE, ha="left", va="center")
    ax_sc.set_yticks([])
    ax_sc.set_ylim(-0.7, len(scen) - 0.30)
    ax_sc.set_xlim(0, max(vals) * 1.22)
    ax_sc.set_xlabel("Dominant eigenvalue of the NGM (-)", fontsize=7.6)
    ax_sc.set_title("Effect of contact reductions", fontsize=8.5,
                    fontweight="bold", loc="left", pad=4)
    ax_sc.tick_params(axis="x", labelsize=6.6)
    ax_sc.tick_params(axis="y", length=0)
    style_axes(ax_sc)

    panel_label_inches(fig, ax_top, "(a)", dx=-0.46, dy=0.06)
    panel_label_inches(fig, ax_sc, "(b)", dx=-0.30, dy=0.06)

    savefig(fig, "426_epi_contact_matrix.png")


if __name__ == "__main__":
    main()

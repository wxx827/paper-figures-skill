"""Petrophysical interpretation stack: porosity, saturation, permeability.

Main panel is a four-track depth display sharing one downward depth axis.
Each reservoir unit is generated with a fining-upward porosity trend and a
saturation-height profile, so water saturation rises towards the base of every
unit exactly as a transition zone does; permeability then follows a
flow-unit-specific log-linear porosity relation.

The companions (58 % of the main display's width) are the two crossplots that
justify the main panel: the porosity-permeability semilog, whose regression is
fitted at run time on the same points that are drawn, and the Buckles plot,
whose bulk-volume-water hyperbola is the mean of the plotted pay samples.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
GRID = "#d5d5d5"

HC = "#b23a48"
WATER = "#3b5b92"
CLAY = "#b3b3b3"
PHI_C = "#2e7d5b"
PERM_C = "#8e6bbf"
FU_COL = ["#b23a48", "#c9a227", "#3b5b92"]
FU_NAME = ["FU1 high quality", "FU2 moderate", "FU3 tight"]

TOP, BASE, STEP = 2500.0, 2900.0, 0.1524
PHI_LIM = (0.0, 0.35)
K_LIM = (0.01, 1.0e4)
# flow unit -> (intercept, slope on porosity units, Buckles constant)
FU_LAW = [(-1.05, 0.205, 0.035), (-1.70, 0.190, 0.055), (-2.45, 0.175, 0.080)]
K_CUT, PHI_CUT, SW_CUT = 1.0, 0.09, 0.50


def smoothed(a, sd, rng, res_m=0.45):
    a = gaussian_filter1d(a, max(res_m / STEP / 2.355, 0.6))
    n = gaussian_filter1d(rng.normal(size=a.size), 1.7)
    return a + n * (sd / (n.std() + 1e-12))


def build_units(rng):
    units, d = [], TOP
    while d < BASE:
        th = float(rng.uniform(4.0, 12.0))
        units.append({"top": d, "bot": min(d + th, BASE), "res": False})
        d += th
        if d >= BASE:
            break
        th = float(rng.uniform(9.0, 30.0))
        units.append({"top": d, "bot": min(d + th, BASE), "res": True,
                      "fu": int(rng.integers(0, 3)),
                      "phi0": float(rng.uniform(0.115, 0.255))})
        d += th
    return units


def runs(flag, md):
    out, start = [], None
    for i, f in enumerate(flag):
        if f and start is None:
            start = md[i]
        elif not f and start is not None:
            out.append((start, md[i]))
            start = None
    if start is not None:
        out.append((start, md[-1]))
    return out


def track(fig, rect, first=False, title=""):
    ax = axes_inches(fig, rect)
    ax.set_ylim(BASE, TOP)
    ax.tick_params(labelsize=6.6, length=2.4, width=0.7)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color(RULE)
    ax.yaxis.set_major_locator(plt.MultipleLocator(50.0))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(10.0))
    if first:
        ax.set_ylabel("Depth (m MD)", fontsize=8.0)
    else:
        ax.tick_params(labelleft=False)
    ax.grid(axis="y", color=GRID, lw=0.4, alpha=0.8)
    if title:
        ax.set_title(title, fontsize=8.2, fontweight="bold", pad=4.0)
    return ax


def main():
    set_pub_style()
    rng = np.random.default_rng(6104)

    md = np.arange(TOP, BASE + STEP, STEP)
    units = build_units(rng)

    phie = np.full_like(md, 0.022)
    sw = np.ones_like(md)
    fu_id = np.full(md.size, -1, dtype=int)
    for u in units:
        m = (md >= u["top"]) & (md < u["bot"])
        if not m.any():
            continue
        if not u["res"]:
            phie[m] = rng.uniform(0.015, 0.045)
            sw[m] = 1.0
            continue
        t = (md[m] - u["top"]) / max(u["bot"] - u["top"], 1e-9)
        # fining upward: best porosity in the upper third of the unit
        p = u["phi0"] * (1.0 - 0.42 * np.abs(t - 0.28) / 0.72)
        phie[m] = np.clip(p, 0.02, 0.32)
        c = FU_LAW[u["fu"]][2]
        swirr = np.clip(c / np.maximum(phie[m], 0.02), 0.06, 1.0)
        sw[m] = np.clip(swirr + (1.0 - swirr) * t ** 2.4, swirr, 1.0)
        fu_id[m] = u["fu"]

    phie = np.clip(smoothed(phie, 0.006, rng), 0.005, 0.33)
    sw = np.clip(smoothed(sw, 0.022, rng), 0.05, 1.0)
    phit = np.clip(phie + 0.020 + 0.05 * (1.0 - np.clip(phie / 0.25, 0, 1)),
                   phie + 0.004, 0.40)

    logk = np.empty_like(md)
    for i, f in enumerate(fu_id):
        a, b, _ = FU_LAW[max(f, 0)] if f >= 0 else (-3.6, 0.10, 0.0)
        logk[i] = a + b * (100.0 * phie[i])
    logk = smoothed(logk, 0.16, rng, res_m=0.30)
    k = np.clip(10.0 ** logk, *K_LIM)

    res = fu_id >= 0
    pay = res & (phie > PHI_CUT) & (sw < SW_CUT) & (k > K_CUT)
    wet = res & ~pay
    zones = [z for z in runs(pay, md) if z[1] - z[0] > 2.0]
    net = float(np.count_nonzero(pay) * STEP)
    phi_pay = float(phie[pay].mean()) if pay.any() else 0.0
    sw_pay = float(sw[pay].mean()) if pay.any() else 0.0
    bvw = float((phie[pay] * sw[pay]).mean()) if pay.any() else 0.0

    # regression on exactly the points the crossplot draws
    xs, ys = 100.0 * phie[res], np.log10(k[res])
    slope, icpt = np.polyfit(xs, ys, 1)
    pred = slope * xs + icpt
    r2 = 1.0 - np.sum((ys - pred) ** 2) / np.sum((ys - ys.mean()) ** 2)

    # ---- canvas -------------------------------------------------------------
    fig = plt.figure(figsize=(10.4, 6.4))
    widths, gap, x0 = [1.28, 1.28, 1.38, 0.34], 0.24, 0.88
    xs_pos, x = [], x0
    for w in widths:
        xs_pos.append(x)
        x += w + gap
    y_b, y_t = 1.02, 5.28
    titles = ["Porosity", "Saturation", "Permeability", "Flag"]
    ax_p, ax_s, ax_k, ax_f = [
        track(fig, (xx, y_b, w, y_t - y_b), first=(i == 0), title=t)
        for i, (xx, w, t) in enumerate(zip(xs_pos, widths, titles))]
    ax_pk = axes_inches(fig, (6.68, 3.58, 2.86, 1.70))
    ax_bk = axes_inches(fig, (6.68, 1.02, 2.86, 1.70))

    for ax in (ax_p, ax_s, ax_k):
        for z0, z1 in zones:
            ax.axhspan(z0, z1, facecolor=HC, alpha=0.09, zorder=0)

    # ---- porosity track ------------------------------------------------------
    ax_p.set_xlim(*PHI_LIM)
    ax_p.set_xticks([0.0, 0.1, 0.2, 0.3])
    ax_p.set_xlabel("PHIE / PHIT (v/v)", fontsize=7.4)
    ax_p.fill_betweenx(md, 0.0, phie, facecolor=PHI_C, alpha=0.30, lw=0)
    ax_p.fill_betweenx(md, phie, np.minimum(phit, PHI_LIM[1]),
                       facecolor=CLAY, alpha=0.55, lw=0)
    ax_p.plot(phie, md, color=PHI_C, lw=0.6)
    ax_p.plot(np.minimum(phit, PHI_LIM[1]), md, color=SUB, lw=0.45,
              ls=(0, (3, 1.6)))
    ax_p.axvline(PHI_CUT, color=RULE, lw=0.8)

    # ---- saturation track ----------------------------------------------------
    ax_s.set_xlim(0.0, 1.0)
    ax_s.set_xticks([0.0, 0.5, 1.0])
    ax_s.set_xlabel("Sw (v/v)", fontsize=7.4)
    ax_s.fill_betweenx(md, 0.0, sw, facecolor=WATER, alpha=0.45, lw=0)
    ax_s.fill_betweenx(md, sw, 1.0, facecolor=HC, alpha=0.55, lw=0)
    ax_s.plot(sw, md, color=WATER, lw=0.6)
    ax_s.axvline(SW_CUT, color=RULE, lw=0.8)

    # ---- permeability track --------------------------------------------------
    ax_k.set_xscale("log")
    ax_k.set_xlim(*K_LIM)
    ax_k.set_xticks([0.01, 1.0, 100.0, 1.0e4])
    ax_k.set_xticklabels(["0.01", "1", "100", "1e4"])
    ax_k.xaxis.set_minor_locator(plt.NullLocator())
    ax_k.set_xlabel("Permeability (mD)", fontsize=7.4)
    for j in range(3):
        m = fu_id == j
        kk = np.where(m, k, np.nan)
        ax_k.plot(kk, md, color=FU_COL[j], lw=0.85)
    ax_k.plot(np.where(res, np.nan, k), md, color=MUTED, lw=0.5)
    ax_k.axvline(K_CUT, color=RULE, lw=0.8)

    # ---- flag ribbon ----------------------------------------------------------
    ax_f.set_xlim(0.0, 1.0)
    ax_f.set_xticks([])
    ax_f.set_xlabel("Net pay", fontsize=7.4)
    ax_f.add_patch(Rectangle((0.0, TOP), 1.0, BASE - TOP, facecolor="#f0f0f0",
                             lw=0))
    for z0, z1 in runs(wet, md):
        ax_f.add_patch(Rectangle((0.0, z0), 1.0, z1 - z0, facecolor=WATER,
                                 alpha=0.8, lw=0))
    for z0, z1 in runs(pay, md):
        ax_f.add_patch(Rectangle((0.0, z0), 1.0, z1 - z0, facecolor=HC, lw=0))

    leg = [Rectangle((0, 0), 1, 1, facecolor=c, alpha=a, lw=0)
           for c, a in ((HC, 1.0), (WATER, 0.8), ("#f0f0f0", 1.0))]
    fig.legend(leg, ["Net pay", "Wet reservoir", "Non-reservoir"],
               frameon=False, fontsize=6.8, ncol=3, loc="lower center",
               bbox_to_anchor=(0.325, 0.052), handlelength=1.1,
               handleheight=0.9, columnspacing=1.5, handletextpad=0.5)

    # ---- porosity-permeability crossplot ---------------------------------------
    ax_pk.set_yscale("log")
    ax_pk.grid(color=GRID, lw=0.45, alpha=0.9)
    ax_pk.set_axisbelow(True)
    for j in range(3):
        m = fu_id == j
        ax_pk.scatter(100.0 * phie[m], k[m], s=5, color=FU_COL[j], alpha=0.55,
                      linewidths=0.0, label=FU_NAME[j])
    gx = np.linspace(xs.min(), xs.max(), 60)
    ax_pk.plot(gx, 10.0 ** (slope * gx + icpt), color=RULE, lw=1.2,
               ls=(0, (4, 2)), zorder=6)
    ax_pk.axhline(K_CUT, color=MUTED, lw=0.7, ls=(0, (2, 2)))
    ax_pk.set_xlabel("Effective porosity (p.u.)", fontsize=7.4)
    ax_pk.set_ylabel("Permeability (mD)", fontsize=7.4)
    ax_pk.tick_params(labelsize=6.6)
    ax_pk.set_ylim(K_LIM[0], K_LIM[1])
    t = ax_pk.text(0.035, 0.955,
                   r"$k = 10^{%.3f\,\phi %+.2f}$" % (slope, icpt)
                   + "\n" + r"$R^2$ = %.3f,  n = %d" % (r2, xs.size),
                   transform=ax_pk.transAxes, ha="left", va="top",
                   fontsize=6.6, color=INK, linespacing=1.5)
    t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])
    ax_pk.legend(frameon=False, fontsize=6.2, loc="lower right",
                 labelspacing=0.3, handletextpad=0.4, markerscale=1.6)
    ax_pk.set_title("Each flow unit keeps its own porosity-permeability law",
                    fontsize=8.5, fontweight="bold")

    # ---- Buckles plot ----------------------------------------------------------
    ax_bk.grid(color=GRID, lw=0.45, alpha=0.9)
    ax_bk.set_axisbelow(True)
    for j in range(3):
        m = (fu_id == j) & res
        ax_bk.scatter(100.0 * phie[m], sw[m], s=5, color=FU_COL[j], alpha=0.5,
                      linewidths=0.0)
    pg = np.linspace(3.0, 34.0, 200)
    for c in (0.02, 0.04, 0.08, 0.14):
        ax_bk.plot(pg, np.clip(c / (pg / 100.0), 0.0, 1.05), color=MUTED,
                   lw=0.7, ls=(0, (3, 2)))
        yq = c / 0.315
        if 0.03 < yq < 1.0:
            ax_bk.text(31.6, yq + 0.018, "BVW %.2f" % c, fontsize=6.0,
                       color=MUTED, ha="right", va="bottom")
    ax_bk.plot(pg, np.clip(bvw / (pg / 100.0), 0.0, 1.05), color=INK, lw=1.3)
    tb = ax_bk.text(0.035, 0.05, "mean BVW of pay = %.3f v/v" % bvw,
                    transform=ax_bk.transAxes, ha="left", va="bottom",
                    fontsize=6.6, color=INK)
    tb.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])
    ax_bk.set_xlim(3.0, 34.0)
    ax_bk.set_ylim(0.0, 1.05)
    ax_bk.set_xlabel("Effective porosity (p.u.)", fontsize=7.4)
    ax_bk.set_ylabel("Water saturation (v/v)", fontsize=7.4)
    ax_bk.tick_params(labelsize=6.6)
    ax_bk.set_title("Buckles hyperbolas: constant bulk volume water",
                    fontsize=8.5, fontweight="bold")

    panel_labels_inches(fig, [ax_p, ax_pk, ax_bk], dx=-0.60, dy=0.34)
    savefig(fig, "812_wlg_petrophysics_stack.png")


if __name__ == "__main__":
    main()

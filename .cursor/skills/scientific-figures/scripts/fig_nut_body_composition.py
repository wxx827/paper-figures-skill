"""Four-compartment body composition across a 28-week diet and training study.

Twenty-four participants are simulated once, individually, and every number in
the figure is then measured from that one cohort: the stacked bars are the
compartment means at each visit, the overlaid line is their sum (which is body
mass by construction), the regional map takes the week-28 fat fraction of each
segment, and the forest plot is a paired t-test of the week-28 change against
baseline with a Student 95 % interval.  The four compartments - fat, lean soft
tissue, total body water and bone mineral - partition body mass exactly, so
the bars can be stacked without double counting.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Patch
from matplotlib.lines import Line2D
from scipy import stats

from pubstyle import (get_cmap, savefig, set_pub_style, sig_stars)
from figkit import halo_text, panel_labels_inches, safe_colorbar, shared_norm

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
BLUE, GREEN, RED, GOLD = "#3b5b92", "#2e7d5b", "#b23a48", "#c9a227"
FAT, PROTEIN = "#c0603f", "#8e6bbf"       # macronutrient triad, batch-wide
BONE, SOLID = "#8a8a8a", "#3a3a3a"

N_SUB = 24
WEEKS = np.array([0, 4, 8, 12, 16, 20, 24, 28])
PHASES = [("Baseline", 0, 0, "#f5f5f5"),
          ("Deficit -2.1 MJ/d", 1, 3, "#f0f0f0"),
          ("Deficit + training", 4, 5, "#f5f5f5"),
          ("Maintenance", 6, 7, "#f0f0f0")]
COMPARTMENTS = [("Fat mass", FAT), ("Lean soft tissue", PROTEIN),
                ("Total body water", BLUE), ("Bone mineral", BONE)]
# region key, short caption, fat multiplier relative to the whole body
REGIONS = [("Head", "head", 0.55), ("Trunk", "trunk", 1.34),
           ("Left arm", "arm", 0.78), ("Right arm", "arm", 0.80),
           ("Left leg", "leg", 0.86), ("Right leg", "leg", 0.85)]


def simulate(rng):
    """Per-participant compartment trajectories in kg."""
    mass0 = rng.normal(84.0, 8.5, N_SUB)
    fatpc0 = np.clip(rng.normal(31.0, 4.6, N_SUB), 18.0, 44.0)
    fat0 = mass0 * fatpc0 / 100.0
    ffm0 = mass0 - fat0

    # phase-wise weekly rates (kg/week), fat loss slows as the deficit runs on
    fat_rate = np.array([0.0, -0.52, -0.44, -0.36, -0.30, -0.26, -0.05, 0.02])
    ffm_rate = np.array([0.0, -0.10, -0.08, -0.06, 0.10, 0.13, 0.05, 0.02])
    dw = np.diff(np.concatenate([[0], WEEKS]))

    fat = np.zeros((N_SUB, WEEKS.size))
    ffm = np.zeros((N_SUB, WEEKS.size))
    fat[:, 0], ffm[:, 0] = fat0, ffm0
    resp = rng.normal(1.0, 0.28, N_SUB)[:, None]      # responder spread
    for j in range(1, WEEKS.size):
        fat[:, j] = fat[:, j - 1] + fat_rate[j] * dw[j] * resp[:, 0] \
            + rng.normal(0.0, 0.30, N_SUB)
        ffm[:, j] = ffm[:, j - 1] + ffm_rate[j] * dw[j] \
            + rng.normal(0.0, 0.22, N_SUB)
    fat = np.clip(fat, 4.0, None)

    # Bone mineral is nearly inert over 28 weeks: it tracks its own baseline,
    # not the fat-free mass that is being reshaped around it.
    bone0 = (0.0425 + rng.normal(0.0, 0.0022, N_SUB)) * ffm0
    bone = bone0[:, None] * (1.0 + 0.00022 * WEEKS[None, :]) \
        + rng.normal(0.0, 0.012, (N_SUB, WEEKS.size))
    # Hydration of the fat-free mass creeps up once training starts.
    hyd = (rng.normal(0.735, 0.011, N_SUB)[:, None]
           + 0.0075 * np.clip((WEEKS[None, :] - 12.0) / 16.0, 0.0, 1.0)
           + rng.normal(0.0, 0.0035, (N_SUB, WEEKS.size)))
    water = hyd * (ffm - bone)
    lean_dry = ffm - bone - water
    mass = fat + ffm
    waist = 0.92 * fat + 46.0 + rng.normal(0.0, 1.2, (N_SUB, 1))
    return dict(fat=fat, lean=lean_dry, water=water, bone=bone, ffm=ffm,
                mass=mass, waist=waist)


GEOM = {
    "Head": ("ellipse", 5.00, 18.45, 2.30, 2.80),
    "Trunk": ("box", 3.28, 10.90, 3.44, 6.05),
    "Left arm": ("box", 1.38, 10.75, 1.62, 6.00),
    "Right arm": ("box", 7.00, 10.75, 1.62, 6.00),
    "Left leg": ("box", 3.34, 1.60, 1.52, 8.95),
    "Right leg": ("box", 5.14, 1.60, 1.52, 8.95),
}


def draw_body(ax, fig, frac):
    """DXA-style regional fat map on a schematic body outline."""
    norm = shared_norm([frac], vmin=float(min(frac) * 0.90),
                       vmax=float(max(frac) * 1.06))
    cmap = get_cmap("cb_sequential_warm")
    for (key, short, _), v in zip(REGIONS, frac):
        kind, x, y, w, h = GEOM[key]
        col = cmap(norm(v))
        if kind == "ellipse":
            ax.add_patch(Ellipse((x, y), w, h, facecolor=col,
                                 edgecolor="white", lw=1.0, zorder=3))
            cx, cy = x, y
        else:
            ax.add_patch(FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.42",
                facecolor=col, edgecolor="white", lw=1.0, zorder=3,
                mutation_aspect=1.0))
            cx, cy = x + w / 2.0, y + h / 2.0
        halo_text(ax, cx, cy, f"{short}\n{v:.0f}%", ha="center", va="center",
                  fontsize=6.0, color="white", halo="#3a3a3a", zorder=8,
                  linespacing=1.35)
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(0.0, 21.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Trunk holds\nthe most fat", fontsize=8.5,
                 fontweight="bold", pad=4, linespacing=1.35)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    safe_colorbar(fig, sm, ax, label="regional fat (%)", side="bottom",
                  width=0.10, pad=0.10, labelsize=6.2)


def main():
    set_pub_style()
    rng = np.random.default_rng(662)
    d = simulate(rng)

    stack = [d["fat"], d["lean"], d["water"], d["bone"]]
    means = np.array([s.mean(axis=0) for s in stack])       # 4 x 8, kg
    mass_mean = d["mass"].mean(axis=0)
    fatpc = 100.0 * d["fat"] / d["mass"]

    # ---- layout ------------------------------------------------------------
    # An outline of a human body is twice as tall as it is wide, so it gets a
    # narrow full-height column of its own; squeezed into a half-height cell
    # it would shrink until the segment labels collided.
    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.32, 0.52],
                          left=0.050, right=0.985, top=0.838, bottom=0.150,
                          wspace=0.32)
    ax_m = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_f = fig.add_subplot(gs[0, 2])

    # ---- main: stacked compartments ---------------------------------------
    xx = np.arange(WEEKS.size, dtype=float)
    for name, i0, i1, col in PHASES:
        ax_m.axvspan(xx[i0] - 0.5, xx[i1] + 0.5, color=col, zorder=0)
        ax_m.text((xx[i0] + xx[i1]) / 2.0, 0.985, name,
                  transform=ax_m.get_xaxis_transform(), ha="center",
                  va="top", fontsize=6.4, color=SUB, zorder=5)

    bottom = np.zeros(WEEKS.size)
    for row, (name, col) in zip(means, COMPARTMENTS):
        ax_m.bar(xx, row, bottom=bottom, width=0.62, color=col,
                 edgecolor="white", lw=0.7, zorder=3, label=name)
        for x, b, v in zip(xx, bottom, row):
            if v > 4.0:
                ax_m.text(x, b + v / 2.0, f"{v:.1f}", ha="center",
                          va="center", fontsize=6.0, color="white", zorder=5)
        bottom = bottom + row
    ax_m.plot(xx, mass_mean, color=SOLID, lw=1.5, marker="o", ms=4.2,
              mfc="white", mec=SOLID, mew=1.0, zorder=6, label="Body mass")
    for x, v in zip(xx, mass_mean):
        ax_m.text(x, v + 1.5, f"{v:.1f}", ha="center", va="bottom",
                  fontsize=6.2, color=SOLID, zorder=6)
    dm = mass_mean[-1] - mass_mean[0]
    ax_m.annotate(f"{dm:+.1f} kg body mass", xy=(xx[-1], mass_mean[-1]),
                  xytext=(xx[-1] + 0.05, mass_mean[-1] + 8.5),
                  fontsize=6.6, color=SOLID, ha="right", va="bottom",
                  arrowprops=dict(arrowstyle="->", lw=0.8, color=SOLID,
                                  shrinkA=2, shrinkB=6))

    ax_m.set_xticks(xx)
    ax_m.set_xticklabels([f"{w}" for w in WEEKS], fontsize=7.0)
    ax_m.set_xlim(-0.6, WEEKS.size - 0.4)
    ax_m.set_ylim(0.0, mass_mean.max() * 1.32)
    ax_m.set_xlabel("study week (wk)", fontsize=8.0)
    ax_m.set_ylabel("compartment mass (kg)", fontsize=8.0)
    ax_m.tick_params(labelsize=7.0)
    ax_m.grid(axis="y", lw=0.5, color="#ececec", zorder=1)
    ax_m.set_title(f"Four-compartment mass at every visit, n = {N_SUB}",
                   fontsize=9.0, fontweight="bold", pad=7)
    ax_m.legend(loc="upper left", bbox_to_anchor=(0.0, 0.930),
                frameon=False, fontsize=6.8, ncol=3, handlelength=1.4,
                columnspacing=1.3, borderaxespad=0.2)

    # ---- aux 1: regional map ----------------------------------------------
    base_fatpc = float(fatpc[:, -1].mean())
    frac = np.array([base_fatpc * m for _, _, m in REGIONS])
    draw_body(ax_b, fig, frac)

    # ---- aux 2: forest of week-28 change ----------------------------------
    metrics = [("Body mass", d["mass"], "kg", -1),
               ("Fat mass", d["fat"], "kg", -1),
               ("Fat-free mass", d["ffm"], "kg", +1),
               ("Lean soft tissue", d["lean"], "kg", +1),
               ("Total body water", d["water"], "kg", +1),
               ("Bone mineral", d["bone"], "kg", +1),
               ("Waist girth", d["waist"] + 0.0 * d["fat"], "cm", -1)]
    rows = []
    for name, arr, unit, good in metrics:
        a0, a1 = arr[:, 0], arr[:, -1]
        rel = 100.0 * (a1 - a0) / a0
        m = float(rel.mean())
        se = float(rel.std(ddof=1) / np.sqrt(N_SUB))
        ci = float(stats.t.ppf(0.975, N_SUB - 1)) * se
        p = float(stats.ttest_rel(a1, a0).pvalue)
        rows.append(dict(name=name, mean=m, ci=ci, p=p, unit=unit,
                         abs=float((a1 - a0).mean()), good=good))

    yy = np.arange(len(rows))[::-1]
    ax_f.axvline(0.0, color="#808080", lw=0.9, zorder=2)
    for y, r in zip(yy, rows):
        sig = (r["mean"] - r["ci"]) * (r["mean"] + r["ci"]) > 0
        col = (GREEN if r["mean"] * r["good"] > 0 else RED) if sig else MUTED
        ax_f.errorbar(r["mean"], y, xerr=r["ci"], fmt="o", ms=4.4,
                      color=col, ecolor=col, elinewidth=1.0, capsize=2.2,
                      mec="white", mew=0.6, zorder=4)
        ax_f.text(0.985, y, f"{r['abs']:+.1f} {r['unit']} "
                  f"{sig_stars(r['p'])}", transform=ax_f.get_yaxis_transform(),
                  ha="right", va="center", fontsize=6.0, color=SUB, zorder=5)
    lim = max(abs(r["mean"]) + r["ci"] for r in rows)
    ax_f.set_xlim(-lim * 1.28, lim * 1.62)
    ax_f.set_ylim(-0.8, len(rows) - 0.2)
    ax_f.set_yticks(yy)
    ax_f.set_yticklabels([r["name"] for r in rows], fontsize=6.6)
    ax_f.set_xlabel("change from baseline at week 28 (%)", fontsize=7.4)
    ax_f.tick_params(axis="x", labelsize=6.5)
    ax_f.tick_params(axis="y", length=0)
    ax_f.grid(axis="x", lw=0.5, color="#f2f2f2", zorder=0)
    ax_f.set_title("Fat falls,\nlean tissue holds", fontsize=8.5,
                   fontweight="bold", pad=4, linespacing=1.35)

    panel_labels_inches(fig, [ax_m, ax_b, ax_f], dx=-0.42, dy=0.10)
    d_fat = means[0, -1] - means[0, 0]
    d_ffm = float(d["ffm"][:, -1].mean() - d["ffm"][:, 0].mean())

    print(f"baseline mass {mass_mean[0]:.1f} kg -> week 28 "
          f"{mass_mean[-1]:.1f} kg ({dm:+.1f} kg)")
    print(f"fat {means[0, 0]:.1f} -> {means[0, -1]:.1f} kg, "
          f"FFM change {d_ffm:+.1f} kg, "
          f"body fat {fatpc[:, 0].mean():.1f}% -> {fatpc[:, -1].mean():.1f}%")
    for r in rows:
        print(f"  {r['name']:<18s}{r['mean']:+6.2f} % "
              f"[{r['mean'] - r['ci']:+.2f}, {r['mean'] + r['ci']:+.2f}] "
              f"p={r['p']:.2e}")
    savefig(fig, "662_nut_body_composition.png")


if __name__ == "__main__":
    main()

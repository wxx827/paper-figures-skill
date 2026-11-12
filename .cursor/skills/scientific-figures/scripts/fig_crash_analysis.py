"""Road-safety dashboard: where, when, what kind and how severe.

Panel (a) turns 520 synthetic crash records into a kernel-density hotspot
surface over a faint street grid; panel (b) is the hour-by-weekday count
matrix.  The two lower panels are deliberately shorter than the maps and
share one category order: the Pareto in (c) is indexed 1-8 and those same
indices label the severity composition in (d), so no legend has to repeat
the category names twice.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
from pubstyle import set_pub_style, savefig

KDE_RAMP = [(1, 1, 1, 0.0), (0.94, 0.86, 0.78, 0.55), (0.87, 0.66, 0.48, 0.80),
            (0.78, 0.42, 0.28, 0.92), (0.55, 0.20, 0.16, 0.98)]
HEAT = ["#f6f5f2", "#e3e6e5", "#bcd0d3", "#89aebb", "#57849c", "#31597a",
        "#1b3550"]
SEV = ["#8c3b2f", "#cf8a52", "#dfcc99"]
SEV_LAB = ["fatal", "serious", "slight"]
BAR = "#5a7f96"
INK = "#2b2b2b"
GREY = "#8c8c8c"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TYPES = ["Rear-end", "Right-angle", "Side-swipe", "Pedestrian",
         "Single-vehicle", "Cyclist", "Parked vehicle", "Head-on"]

FIGW, FIGH = 10.1, 6.75
EXT = 6.0


def R(x, y, w, h):
    return [x / FIGW, y / FIGH, w / FIGW, h / FIGH]


def crash_points(rng, n=520):
    hubs = np.array([[1.85, 4.25], [4.30, 4.05], [3.05, 1.95],
                     [4.85, 1.55], [2.35, 2.90]])
    hw = np.array([0.30, 0.26, 0.16, 0.13, 0.15])
    pts = []
    k_h = int(n * 0.62)
    idx = rng.choice(len(hubs), size=k_h, p=hw / hw.sum())
    pts.append(hubs[idx] + rng.normal(0, 0.34, (k_h, 2)))
    k_c = int(n * 0.24)                       # strung along two corridors
    s = rng.random(k_c)
    corr = np.where(rng.random(k_c) < 0.5,
                    1, 0)[:, None]
    a = np.column_stack([0.5 + 5.0 * s, 4.15 + 0.55 * np.sin(3 * s)])
    b = np.column_stack([3.35 + 0.45 * np.sin(3 * s), 0.5 + 5.0 * s])
    pts.append(np.where(corr == 1, a, b) + rng.normal(0, 0.16, (k_c, 2)))
    k_r = n - k_h - k_c
    pts.append(rng.uniform(0.3, 5.7, (k_r, 2)))
    p = np.vstack(pts)
    return p[(p[:, 0] > 0.15) & (p[:, 0] < 5.85)
             & (p[:, 1] > 0.15) & (p[:, 1] < 5.85)]


def main():
    set_pub_style()
    rng = np.random.default_rng(404)
    pts = crash_points(rng)

    fig = plt.figure(figsize=(FIGW, FIGH))
    axm = fig.add_axes(R(0.58, 3.35, 2.95, 2.95))
    axh = fig.add_axes(R(4.28, 3.90, 4.85, 2.40))
    cah = fig.add_axes(R(4.28, 3.42, 1.90, 0.10))
    axp = fig.add_axes(R(0.58, 0.70, 2.95, 1.85))
    axs = fig.add_axes(R(4.28, 0.70, 4.55, 1.85))

    # ------------------------------------------------------- (a) hotspots
    axm.set_facecolor("#fbfaf8")
    gx = np.linspace(0.35, EXT - 0.35, 8)
    for v in gx:
        axm.plot([v + 0.10 * np.sin(v)] * 2, [0, EXT], color="#e6e3dd",
                 lw=0.8, zorder=1)
        axm.plot([0, EXT], [v + 0.10 * np.cos(v)] * 2, color="#e6e3dd",
                 lw=0.8, zorder=1)
    for v in (1.85, 4.30):
        axm.plot([0, EXT], [v, v], color="#dad6ce", lw=2.0, zorder=1)
        axm.plot([v, v], [0, EXT], color="#dad6ce", lw=2.0, zorder=1)

    kde = gaussian_kde(pts.T, bw_method=0.20)
    g = np.linspace(0, EXT, 190)
    GX, GY = np.meshgrid(g, g)
    Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    Z = Z / Z.max()
    cmap = mpl.colors.LinearSegmentedColormap.from_list("kde", KDE_RAMP)
    im = axm.contourf(GX, GY, Z, levels=np.linspace(0.06, 1.0, 9), cmap=cmap,
                      zorder=2)
    axm.contour(GX, GY, Z, levels=[0.30, 0.60, 0.85], colors="white",
                linewidths=0.7, alpha=0.75, zorder=3)
    axm.plot(pts[:, 0], pts[:, 1], ls="none", marker="o", ms=1.5,
             mfc="#3a2a24", mec="none", alpha=0.45, zorder=4)

    for tag, (hx, hy) in zip("ABC", [(1.85, 4.25), (4.30, 4.05),
                                     (3.05, 1.95)]):
        axm.text(hx, hy, tag, ha="center", va="center", fontsize=6.6,
                 fontweight="bold", color="#4a1f18", zorder=7,
                 bbox=dict(boxstyle="circle,pad=0.22", fc="white",
                           ec="#4a1f18", lw=1.0, alpha=0.95))
    axm.text(0.035, 0.965, "A  Market St / 5th\nB  Ring Rd interchange\n"
             "C  Station approach", transform=axm.transAxes, fontsize=6.4,
             va="top", ha="left", color="#4a1f18", linespacing=1.45,
             zorder=8, bbox=dict(boxstyle="round,pad=0.26", fc="white",
                                 ec="#e2d8d4", lw=0.6, alpha=0.90))
    axm.plot([0.35, 1.35], [0.32, 0.32], color=INK, lw=2.0,
             solid_capstyle="butt", zorder=8)
    axm.text(0.85, 0.44, "1 km", fontsize=6.2, ha="center", va="bottom",
             color=INK, zorder=8)

    # inset density key, on its own white card so it stays legible
    axm.add_patch(Rectangle((2.82, 0.18), 3.02, 0.80, facecolor="white",
                            edgecolor="#e0ddd6", lw=0.6, alpha=0.90,
                            zorder=8))
    cam = fig.add_axes(R(2.06, 3.62, 1.28, 0.085))
    cb0 = fig.colorbar(im, cax=cam, orientation="horizontal",
                       ticks=[0.2, 0.6, 1.0])
    cb0.ax.set_xticklabels(["low", "med", "high"], fontsize=6.0)
    cb0.set_label("crash density", fontsize=6.2, labelpad=1.8)
    cb0.ax.tick_params(length=1.8, width=0.5, pad=1.2)
    cb0.outline.set_linewidth(0.5); cb0.outline.set_edgecolor("#777777")

    axm.set_xlim(0, EXT); axm.set_ylim(0, EXT)
    axm.set_aspect("equal")
    axm.set_xticks([]); axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.7)
        sp.set_edgecolor("#bdb9b1")

    # ------------------------------------------------- (b) hour x weekday
    hr = np.arange(24)
    base = (0.55 + 2.4 * np.exp(-0.5 * ((hr - 8.1) / 1.15) ** 2)
            + 3.1 * np.exp(-0.5 * ((hr - 17.6) / 1.55) ** 2)
            + 1.0 * np.exp(-0.5 * ((hr - 12.7) / 2.10) ** 2))
    night = 1.5 * np.exp(-0.5 * ((hr - 1.6) / 1.70) ** 2) + \
        1.1 * np.exp(-0.5 * ((hr - 23.4) / 1.40) ** 2)
    wk = np.array([1.00, 1.02, 1.03, 1.08, 1.22, 0.86, 0.70])
    nightw = np.array([0.30, 0.30, 0.35, 0.55, 1.35, 1.75, 0.95])
    M = (base[None, :] * wk[:, None] + night[None, :] * nightw[:, None]) * 6.0
    M = np.round(M * rng.lognormal(0, 0.16, M.shape))

    cmaph = mpl.colors.LinearSegmentedColormap.from_list("heat", HEAT)
    imh = axh.imshow(M, cmap=cmaph, aspect="auto", origin="upper",
                     extent=(0, 24, 7, 0), interpolation="nearest", zorder=2)
    for k in range(1, 7):
        axh.axhline(k, color="white", lw=0.5, zorder=3)
    i, j = np.unravel_index(np.argmax(M), M.shape)
    axh.add_patch(Rectangle((j, i), 1, 1, facecolor="none", edgecolor="#8c3b2f",
                            lw=1.3, zorder=5))
    axh.annotate(f"peak cell: {DAYS[i]} {j:02d}:00\n{M[i, j]:.0f} crashes/yr",
                 xy=(j, i + 0.5), xytext=(11.2, 1.35),
                 fontsize=6.6, color="#8c3b2f", fontweight="bold", ha="right",
                 va="center", linespacing=1.25, zorder=7,
                 bbox=dict(boxstyle="round,pad=0.22", fc="white",
                           ec="#e5d5d1", lw=0.6, alpha=0.93),
                 arrowprops=dict(arrowstyle="-|>", lw=0.75, color="#8c3b2f",
                                 shrinkA=3, shrinkB=3,
                                 connectionstyle="arc3,rad=0.25"))
    axh.set_yticks(np.arange(7) + 0.5)
    axh.set_yticklabels(DAYS, fontsize=6.8)
    axh.set_xticks(np.arange(0, 25, 3))
    axh.set_xticklabels([f"{h:02d}" for h in range(0, 25, 3)], fontsize=6.8)
    axh.set_xlabel("hour of day  (h)", labelpad=2, fontsize=7.8)
    axh.tick_params(length=2.2, pad=2.0)
    axh.tick_params(axis="y", length=0)
    for sp in axh.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.7)
        sp.set_edgecolor("#bdb9b1")

    cb = fig.colorbar(imh, cax=cah, orientation="horizontal")
    cb.set_label("crashes per year in cell", fontsize=6.8, labelpad=2.0)
    cb.ax.tick_params(labelsize=6.4, length=2.0, width=0.6, pad=1.5)
    cb.outline.set_linewidth(0.55); cb.outline.set_edgecolor("#666666")

    # ------------------------------------------------------- (c) Pareto
    cnt = np.array([1420, 980, 760, 505, 430, 315, 220, 140], dtype=float)
    cnt = np.round(cnt * rng.uniform(0.94, 1.06, cnt.size))
    o = np.argsort(-cnt)
    cnt = cnt[o]
    types = [TYPES[k] for k in o]
    cum = 100.0 * np.cumsum(cnt) / cnt.sum()
    xi = np.arange(len(cnt))

    axp.bar(xi, cnt, width=0.68, color=BAR, alpha=0.88, edgecolor="white",
            linewidth=0.4, zorder=3)
    axp.set_ylim(0, cnt.max() * 1.22)
    axp.set_xlim(-0.65, len(cnt) - 0.35)
    axp.set_xticks(xi)
    axp.set_xticklabels([str(k + 1) for k in xi], fontsize=7.0)
    axp.set_xlabel("crash type  (index, see panel d)", labelpad=2,
                   fontsize=7.8)
    axp.set_ylabel("crashes / yr", labelpad=3, fontsize=7.8)
    axp.tick_params(length=2.4, labelsize=6.8)
    axp.grid(axis="y", color="#eeeeee", lw=0.55, zorder=0)
    for sp in ("top", "right"):
        axp.spines[sp].set_visible(False)

    ax2 = axp.twinx()
    ax2.plot(xi, cum, color="#8c3b2f", lw=1.4, marker="o", ms=3.0,
             mfc="white", mec="#8c3b2f", mew=1.0, zorder=5)
    ax2.axhline(80, color=GREY, lw=0.7, ls=(0, (3.5, 2.2)), zorder=4)
    k80 = int(np.argmax(cum >= 80)) + 1
    ax2.text(len(cnt) - 0.45, 81.5, f"80 % from top {k80} types",
             fontsize=6.5, color="#6f6f6f", ha="right", va="bottom")
    ax2.set_ylim(0, 108)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_ylabel("cumulative  (%)", labelpad=3, fontsize=7.8,
                   color="#8c3b2f")
    ax2.tick_params(length=2.4, labelsize=6.8, colors="#8c3b2f")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#8c3b2f")
    ax2.spines["right"].set_linewidth(0.7)

    # -------------------------------------------- (d) severity composition
    fatal = np.array(
        [{"Rear-end": 0.7, "Right-angle": 1.6, "Side-swipe": 0.5,
          "Pedestrian": 6.2, "Single-vehicle": 4.1, "Cyclist": 3.2,
          "Parked vehicle": 0.3, "Head-on": 7.5}[t] for t in types])
    serious = np.array(
        [{"Rear-end": 9.0, "Right-angle": 15.5, "Side-swipe": 6.5,
          "Pedestrian": 31.0, "Single-vehicle": 21.0, "Cyclist": 26.0,
          "Parked vehicle": 3.5, "Head-on": 33.0}[t] for t in types])
    serious = serious * rng.uniform(0.95, 1.05, serious.size)
    slight = 100.0 - fatal - serious

    y = np.arange(len(types))[::-1]
    left = np.zeros(len(types))
    for vals, col, lab in zip([fatal, serious, slight], SEV, SEV_LAB):
        axs.barh(y, vals, left=left, height=0.66, color=col, alpha=0.92,
                 edgecolor="white", linewidth=0.5, zorder=3, label=lab)
        left = left + vals
    for k, t in enumerate(types):
        axs.text(-1.2, y[k], f"{k + 1}  {t}", ha="right", va="center",
                 fontsize=6.6, color="#3a3a3a", zorder=4)
        axs.text(101.5, y[k], f"{fatal[k] + serious[k]:.0f}% KSI",
                 ha="left", va="center", fontsize=6.4, color="#8c3b2f",
                 fontweight="bold", zorder=4)
    axs.set_xlim(-26, 118)
    axs.set_ylim(-0.7, len(types) - 0.3)
    axs.set_yticks([])
    axs.set_xticks([0, 25, 50, 75, 100])
    axs.set_xlabel("share of crashes of that type  (%)", labelpad=2,
                   fontsize=7.8)
    axs.spines["bottom"].set_bounds(0, 100)
    axs.tick_params(length=2.4, labelsize=6.8)
    for sp in ("top", "right", "left"):
        axs.spines[sp].set_visible(False)
    leg = axs.legend(loc="lower right", bbox_to_anchor=(1.0, 1.012), ncol=3,
                     frameon=False, fontsize=6.8, handlelength=1.1,
                     handleheight=0.8, columnspacing=1.3)
    leg.set_clip_on(False)

    # ---- panel tags on one shared inch offset ---------------------------
    for ax, tag, ttl in [(axm, "(a)", "Crash-density hotspots"),
                         (axh, "(b)", "Crashes by hour and weekday"),
                         (axp, "(c)", "Crash-type Pareto"),
                         (axs, "(d)", "Severity composition")]:
        pos = ax.get_position()
        fig.text(pos.x0 - 0.48 / FIGW, pos.y1 + 0.115 / FIGH, tag,
                 fontsize=10.5, fontweight="bold", ha="left", va="baseline",
                 color=INK)
        fig.text(pos.x0 - 0.12 / FIGW, pos.y1 + 0.115 / FIGH, ttl,
                 fontsize=8.8, fontweight="bold", ha="left", va="baseline",
                 color="#33474f")

    savefig(fig, "212_crash_analysis.png")


if __name__ == "__main__":
    main()

"""Bi-objective Pareto front of a heat-sink design study.

All candidate designs evaluated during the search are shown as grey dots;
the non-dominated set is highlighted, coloured by a third objective
(manufacturing cost) and joined into the Pareto front.  Three
representative designs - the lightest, the knee and the best-cooling one -
carry call-out boxes with their design variables, and the inset tracks the
hypervolume of the archive over the generations.

Method: an archive-based (mu + lambda) evolutionary search is actually run
here on analytic surrogate models of the mass, the thermal resistance and
the cost of a finned heat sink.  The highlighted set is filtered with an
exact non-dominance test, so no highlighted design is dominated by any
other point in the cloud, and the plotted hypervolume is the real
archive hypervolume, which is monotone non-decreasing by construction.
"""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, add_panel_label

TLIM = (0.8, 6.0)       # wall thickness [mm]
DLIM = (10.0, 70.0)     # fin density [fins per 100 mm]
REF = np.array([9.6, 0.82])   # hypervolume reference point
NGEN, NOFF, NINIT, NRND = 80, 28, 70, 5


def objectives(t, d):
    mass = 0.85 * t + 0.019 * d * t ** 0.6            # [kg]
    rth = 4.2 / (d ** 0.75 * t ** 0.15)               # [K/W]
    cost = 38.0 + 6.5 * t ** 1.3 + 0.55 * d ** 1.15   # [USD]
    return np.column_stack([mass, rth]), cost


def pareto_mask(f):
    n = f.shape[0]
    keep = np.ones(n, dtype=bool)
    order = np.argsort(f[:, 0], kind="stable")
    best = np.inf
    for i in order:
        if f[i, 1] < best - 1e-15:
            best = f[i, 1]
        else:
            keep[i] = False
    # strict check: drop points tied in f0 but worse in f1
    idx = np.where(keep)[0]
    for i in idx:
        if np.any((f[:, 0] <= f[i, 0]) & (f[:, 1] <= f[i, 1])
                  & ((f[:, 0] < f[i, 0]) | (f[:, 1] < f[i, 1]))):
            keep[i] = False
    return keep


def hypervolume(f):
    if f.size == 0:
        return 0.0
    s = f[np.argsort(f[:, 0])]
    hv, prev = 0.0, REF[1]
    for a, b in s:
        if a >= REF[0] or b >= prev:
            continue
        hv += (REF[0] - a) * (prev - b)
        prev = b
    return hv


def search(rng):
    # deliberately poor seed population: heavy walls, few fins
    x = np.column_stack([rng.uniform(3.2, 6.0, NINIT),
                         rng.uniform(10.0, 26.0, NINIT)])
    f, c = objectives(x[:, 0], x[:, 1])
    allx, allf, allc = [x], [f], [c]
    hv = []
    arch_x, arch_f = x.copy(), f.copy()
    for _ in range(NGEN):
        m = pareto_mask(arch_f)
        arch_x, arch_f = arch_x[m], arch_f[m]
        hv.append(hypervolume(arch_f))
        pick = rng.integers(0, arch_x.shape[0], NOFF)
        kid = arch_x[pick] + rng.normal(0, [0.30, 3.4], (NOFF, 2))
        kid = np.vstack([kid,
                         np.column_stack([rng.uniform(*TLIM, NRND),
                                          rng.uniform(*DLIM, NRND)])])
        kid[:, 0] = np.clip(kid[:, 0], *TLIM)
        kid[:, 1] = np.clip(kid[:, 1], *DLIM)
        kf, kc = objectives(kid[:, 0], kid[:, 1])
        allx.append(kid)
        allf.append(kf)
        allc.append(kc)
        arch_x = np.vstack([arch_x, kid])
        arch_f = np.vstack([arch_f, kf])
    m = pareto_mask(arch_f)
    hv.append(hypervolume(arch_f[m]))
    return (np.vstack(allx), np.vstack(allf), np.concatenate(allc),
            np.array(hv))


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(102)
    x, f, c, hv = search(rng)
    front = pareto_mask(f)
    fp, xp, cp = f[front], x[front], c[front]
    o = np.argsort(fp[:, 0])
    fp, xp, cp = fp[o], xp[o], cp[o]

    span = fp.max(axis=0) - fp.min(axis=0)
    dn = (fp - fp.min(axis=0)) / span
    knee = int(np.argmin(np.hypot(dn[:, 0], dn[:, 1])))

    fig = plt.figure(figsize=(7.2, 4.35))
    ax = fig.add_axes([0.082, 0.115, 0.775, 0.855])

    ax.scatter(f[~front, 0], f[~front, 1], s=6, c="#b9bfc5", alpha=0.45,
               lw=0, zorder=2, label="dominated designs (%d)"
               % int((~front).sum()), rasterized=True)
    ax.plot(fp[:, 0], fp[:, 1], color="#4a5259", lw=1.2, zorder=4,
            label="Pareto front")
    show = np.unique(np.linspace(0, len(fp) - 1, 44).astype(int))
    sc = ax.scatter(fp[show, 0], fp[show, 1], s=27, c=cp[show],
                    cmap="viridis", edgecolor="#12161b", linewidth=0.35,
                    zorder=6,
                    label="non-dominated archive (%d)" % int(front.sum()))

    ut = np.array([f[:, 0].min(), f[:, 1].min()])
    ax.plot([ut[0]], [ut[1]], marker="*", ms=10, mfc="none", mec="#8c2f2f",
            mew=1.0, zorder=7)
    ax.text(ut[0] + 0.28, ut[1] - 0.012, "utopia point (infeasible)",
            fontsize=6.3, color="#8c2f2f", ha="left", va="center")
    ax.annotate("", xy=(2.05, 0.395), xytext=(2.95, 0.512), zorder=7,
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#8c2f2f"))
    ax.text(3.02, 0.518, "improving", fontsize=6.4, color="#8c2f2f",
            ha="left", va="center", rotation=34)

    picks = [(0, (0.255, 0.72), "lightest design", "minimum mass"),
             (knee, (0.44, 0.315), "knee point", "best compromise"),
             (len(fp) - 1, (0.63, 0.125), "best cooling",
              "minimum resistance")]
    for j, (ix, xy, name, sub) in enumerate(picks):
        txt = (r"$\bf{%s}$" % name.replace(" ", r"\ ")
               + "\n" + sub
               + "\n" + r"$t$ = %.2f mm,  $\rho_f$ = %.0f /100 mm"
               % (xp[ix, 0], xp[ix, 1])
               + "\n" + r"$m$ = %.2f kg,  $R_{th}$ = %.3f K W$^{-1}$"
               % (fp[ix, 0], fp[ix, 1])
               + "\n" + r"cost = \$%.0f" % cp[ix])
        ax.annotate(txt, xy=(fp[ix, 0], fp[ix, 1]), xytext=xy,
                    textcoords="axes fraction", fontsize=6.2,
                    color="#111418", ha="left", va="center", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#b9b9b9", lw=0.5, alpha=0.95),
                    arrowprops=dict(arrowstyle="-", lw=0.7, color="#333333",
                                    shrinkA=3, shrinkB=4))
        ax.plot([fp[ix, 0]], [fp[ix, 1]],
                marker="D" if j == 1 else "o", ms=6.0 if j == 1 else 5.4,
                mfc="none", mec="#111418", mew=1.1, zorder=8)

    ax.set_xlabel(r"objective 1:  heat-sink mass  $m$  (kg)")
    ax.set_ylabel(r"objective 2:  thermal resistance  $R_{th}$  (K W$^{-1}$)")
    ax.set_xlim(0.4, 9.9)
    ax.set_ylim(0.10, 0.86)
    ax.grid(True, lw=0.45, color="#eeeeee")
    ax.tick_params(length=2.6)
    lg = ax.legend(loc="upper left", fontsize=6.4, handlelength=1.5,
                   borderpad=0.35, labelspacing=0.32, frameon=True,
                   framealpha=0.93, scatterpoints=1)
    lg.get_frame().set_linewidth(0.5)
    add_panel_label(ax, "(a)", x=-0.072, y=1.005, size=9.5)

    cb = fig.colorbar(sc, ax=ax, fraction=0.036, pad=0.02, shrink=0.98)
    cb.set_label(r"objective 3:  manufacturing cost  (USD)", fontsize=7.2)
    cb.ax.tick_params(labelsize=6.8, width=0.7, length=2.2)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    axi = ax.inset_axes([0.645, 0.585, 0.335, 0.365])
    g = np.arange(hv.size)
    axi.plot(g, hv, color="#2f6f9e", lw=1.4)
    axi.fill_between(g, 0, hv, color="#2f6f9e", alpha=0.12, lw=0)
    axi.axhline(hv[-1], color="#8c2f2f", lw=0.8, ls=(0, (3, 2)))
    j90 = int(np.argmax(hv >= 0.99 * hv[-1]))
    axi.plot([j90], [hv[j90]], marker="o", ms=3.6, mfc="#8c2f2f",
             mec="#12161b", mew=0.6)
    axi.text(j90 + 2, hv[j90] * 0.955, "99 %% of final HV\nat gen %d" % j90,
             fontsize=6.0, color="#8c2f2f", ha="left", va="top")
    axi.set_xlim(0, hv.size - 1)
    axi.set_ylim(hv.min() * 0.94, hv[-1] * 1.05)
    axi.set_xlabel("generation", fontsize=6.4, labelpad=1.5)
    axi.set_ylabel("hypervolume", fontsize=6.4)
    axi.yaxis.set_major_locator(plt.MaxNLocator(3))
    axi.tick_params(labelsize=6.0, length=1.8)
    axi.grid(True, lw=0.4, color="#efefef")
    axi.set_title("archive convergence", fontsize=6.6, color="#33404d",
                  pad=2.5)
    for s in axi.spines.values():
        s.set_linewidth(0.6)
    add_panel_label(axi, "(b)", x=-0.20, y=1.03, size=8.5)

    savefig(fig, "102_pareto_front.png")


if __name__ == "__main__":
    main()

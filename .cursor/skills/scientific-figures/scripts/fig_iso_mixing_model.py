"""Three-source isotope mixing: the triangle, the posterior and its weak spot.

The main panel is the mixing space itself.  Three end-members define a
triangle in delta-13C / delta-15N coordinates; the faint grid inside it is
the set of iso-contribution lines, drawn every 25 %, so a sample's position
can be read as a mixture without any arithmetic.  Sample points are the
measured sediments.

The supporting panels are where the honesty lives.  The posterior violins are
not decoration: they come from resampling every end-member within its stated
uncertainty and re-solving the mixing equations for every sample, so their
width is the real identifiability of each source.  The sensitivity panel then
shows how far those means travel when the least-constrained end-member is
shifted along the carbon axis.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

GREEN = "#2e7d5b"     # C3 terrestrial plants
BLUE = "#3b5b92"      # marine phytoplankton
OCHRE = "#c0603f"     # soil organic matter
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
SOLID = "#3a3a3a"

PERMIL = "\u2030"
N_SAMPLES = 96
N_DRAWS = 800
LEVELS = (0.25, 0.50, 0.75)

# name, colour, (d13C, d15N), (sd_d13C, sd_d15N)
ENDS = [("C3 plants", GREEN, (-28.6, 1.0), (0.55, 0.60)),
        ("marine POM", BLUE, (-20.0, 7.6), (0.50, 0.55)),
        ("soil OM", OCHRE, (-26.4, 9.1), (0.80, 0.70))]
SOFT_EM = 2           # the loosely constrained end-member, used in panel (c)


def vertices(shift_c=0.0, which=SOFT_EM):
    v = np.array([e[2] for e in ENDS], dtype=float)
    v[which, 0] += shift_c
    return v


def observations(seed=17):
    """Sediment samples drawn as Dirichlet mixtures of the three end-members."""
    rng = np.random.default_rng(seed)
    frac = rng.dirichlet([2.6, 1.9, 1.6], size=N_SAMPLES)
    xy = frac @ vertices()
    xy = xy + rng.normal(0.0, 0.26, xy.shape)
    return frac, xy


def solve_fractions(xy, verts):
    """Barycentric coordinates of ``xy`` in the triangle ``verts``."""
    a = np.array([[verts[0, 0] - verts[2, 0], verts[1, 0] - verts[2, 0]],
                  [verts[0, 1] - verts[2, 1], verts[1, 1] - verts[2, 1]]])
    rhs = (xy - verts[2]).T
    f01 = np.linalg.solve(a, rhs).T
    return np.column_stack([f01, 1.0 - f01.sum(axis=1)])


def posterior(xy, seed=5):
    """Resample the end-members within their errors, re-solve every sample.

    Each draw returns the *assemblage mean* contribution, which is the
    quantity a source-apportionment study actually reports.  The spread of
    those means is therefore end-member uncertainty made visible, not the
    sample-to-sample scatter that the triangle already shows.
    """
    rng = np.random.default_rng(seed)
    mu = np.array([e[2] for e in ENDS], dtype=float)
    sd = np.array([e[3] for e in ENDS], dtype=float)
    out = np.empty((N_DRAWS, 3))
    for k in range(N_DRAWS):
        f = np.clip(solve_fractions(xy, mu + rng.normal(0.0, 1.0, mu.shape)
                                    * sd), 0.0, 1.0)
        f = f / f.sum(axis=1, keepdims=True)
        out[k] = f.mean(axis=0) * 100.0
    return out


def panel_triangle(ax, xy, verts):
    cen = verts.mean(axis=0)
    tri = np.vstack([verts, verts[:1]])
    ax.plot(tri[:, 0], tri[:, 1], color=SOLID, lw=1.2, zorder=4)

    for i, (name, color, _, sd) in enumerate(ENDS):
        a, b = [k for k in range(3) if k != i]
        for c in LEVELS:
            p = (1.0 - c) * verts[a] + c * verts[i]
            q = (1.0 - c) * verts[b] + c * verts[i]
            ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=0.55,
                    alpha=0.45, zorder=2)
            # One edge per source, so the three label families do not pile up
            # on the same side of the triangle.
            e = (i + 1) % 3
            lp = (1.0 - c) * verts[e] + c * verts[i]
            # Push the tick outward along the edge normal; a radial offset
            # from the centroid barely clears the edge on a thin triangle.
            d = verts[i] - verts[e]
            nrm = np.array([-d[1], d[0]])
            nrm = nrm / (np.hypot(*nrm) + 1e-9)
            if np.dot(lp + nrm - cen, lp - cen) < \
                    np.dot(lp - nrm - cen, lp - cen):
                nrm = -nrm
            lab = lp + 0.62 * nrm
            ax.text(lab[0], lab[1], "%d" % round(c * 100), ha="center",
                    va="center", fontsize=6.0, color=color, zorder=6)
        ax.errorbar(verts[i, 0], verts[i, 1], xerr=sd[0], yerr=sd[1],
                    fmt="o", ms=7.0, color=color, mfc="white", mew=1.6,
                    ecolor=color, elinewidth=1.0, capsize=2.2, zorder=7)
        out = verts[i] + 0.30 * (verts[i] - cen)
        ax.text(out[0], out[1], name, ha="center", va="center", fontsize=7.6,
                fontweight="bold", color=color, zorder=7)

    ax.scatter(xy[:, 0], xy[:, 1], s=15, color=SUB, alpha=0.62,
               linewidths=0.0, zorder=5)
    ax.set_xlabel("$\\delta^{13}C$ (%s VPDB)" % PERMIL)
    ax.set_ylabel("$\\delta^{15}N$ (%s AIR)" % PERMIL)
    ax.set_xlim(verts[:, 0].min() - 2.6, verts[:, 0].max() + 2.2)
    ax.set_ylim(verts[:, 1].min() - 2.4, verts[:, 1].max() + 2.6)
    ax.grid(lw=0.5, alpha=0.6)
    ax.text(0.985, 0.03,
            "grid lines every 25 %% contribution,  n = %d samples"
            % xy.shape[0], transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.0, color=SUB)
    ax.set_title("Every sample resolves inside the three-source triangle",
                 fontsize=9.0, fontweight="bold", color=INK)


def sensitivity_curves(xy, lo=-2.0, hi=2.0, n=33):
    shifts = np.linspace(lo, hi, n)
    curves = np.array([solve_fractions(xy, vertices(s)).mean(axis=0) * 100.0
                       for s in shifts])
    return shifts, curves


def panel_posterior(ax, post, ytop):
    parts = ax.violinplot([post[:, i] for i in range(3)], positions=[0, 1, 2],
                          widths=0.72, showextrema=False, showmedians=False)
    for body, (_, color, _, _) in zip(parts["bodies"], ENDS):
        body.set_facecolor(color)
        body.set_alpha(0.45)
        body.set_edgecolor(color)
        body.set_linewidth(0.8)

    stats = []
    for i, (name, color, _, _) in enumerate(ENDS):
        lo, med, hi = np.percentile(post[:, i], [2.5, 50.0, 97.5])
        stats.append((name, med, lo, hi))
        ax.plot([i, i], [lo, hi], color=SOLID, lw=1.0, zorder=4)
        ax.plot([i], [med], marker="o", ms=4.2, color=color, mfc="white",
                mew=1.2, zorder=5)
        ax.text(i, hi + 2.0, "%.0f %%" % med, ha="center", va="bottom",
                fontsize=6.6, color=color)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([e[0] for e in ENDS], fontsize=6.8)
    ax.set_ylim(0.0, ytop)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_ylabel("mean contribution (%)", fontsize=7.6)
    ax.set_xlabel("end-member (-)", fontsize=7.6)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.65)
    ax.set_title("Posterior contributions", fontsize=8.5, fontweight="bold",
                 color=INK)
    return stats


def panel_sensitivity(ax, shifts, curves, base, ytop):
    swing = []
    for i, (name, color, _, _) in enumerate(ENDS):
        ax.plot(shifts, curves[:, i], color=color, lw=1.3, label=name)
        swing.append(curves[:, i].max() - curves[:, i].min())
    ax.axvline(0.0, color=GREY, lw=0.8, ls=(0, (2.6, 2.0)))
    for i, (_, color, _, _) in enumerate(ENDS):
        ax.plot([0.0], [base[i][1]], marker="o", ms=3.6, color=color,
                mfc="white", mew=1.0, zorder=5)

    ax.set_xlim(shifts[0], shifts[-1])
    ax.set_ylim(0.0, ytop)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel("shift applied to the %s $\\delta^{13}C$ (%s)"
                  % (ENDS[SOFT_EM][0], PERMIL), fontsize=7.6, labelpad=1.5)
    ax.set_ylabel("mean contribution (%)", fontsize=7.6)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.65)
    ax.legend(fontsize=6.3, frameon=False, loc="upper center", ncol=3,
              handlelength=1.1, columnspacing=0.8, borderaxespad=0.05)
    ax.set_title("End-member sensitivity", fontsize=8.5, fontweight="bold",
                 color=INK)
    return swing


def main():
    set_pub_style()
    _, xy = observations()
    verts = vertices()
    post = posterior(xy)
    shifts, curves = sensitivity_curves(xy)
    ytop = min(100.0, max(post.max(), curves.max()) + 14.0)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_tri = axes_inches(fig, (0.92, 0.82, 5.05, 4.62))
    ax_pos = axes_inches(fig, (7.10, 3.28, 2.85, 2.16))
    ax_sen = axes_inches(fig, (7.10, 0.86, 2.85, 1.72))

    panel_triangle(ax_tri, xy, verts)
    stats = panel_posterior(ax_pos, post, ytop)
    swing = panel_sensitivity(ax_sen, shifts, curves, stats, ytop)

    worst = int(np.argmax(swing))
    # Single-line footnotes this long push the tight bounding box past the
    # declared canvas width, which silently shrinks every panel (see
    # DESIGN_SYSTEM section 12.11).
    note = ("Posterior medians %s; the dashed rule in panel (c) is the "
            "published end-member value. A %s shift of the %s carbon "
            "end-member moves the %s share by up to %.0f percentage points, "
            "the widest swing of the three."
            % (", ".join("%s %.0f %% (%.0f-%.0f)" % (n, m, lo, hi)
                         for n, m, lo, hi in stats),
               "4 " + PERMIL, ENDS[SOFT_EM][0], ENDS[worst][0], swing[worst]))

    panel_labels_inches(fig, [ax_tri, ax_pos, ax_sen], dx=-0.62, dy=0.10)
    savefig(fig, "754_iso_mixing_model.png")


if __name__ == "__main__":
    main()

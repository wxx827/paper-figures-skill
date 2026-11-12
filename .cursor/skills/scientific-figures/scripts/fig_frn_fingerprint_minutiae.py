"""Latent fingerprint: synthetic ridge flow with a core, a delta and minutiae.

The ridge pattern is one scalar phase field.  Two square-root potentials give
the loop core and the delta; the field is then histogram-equalised so the ridge
spacing stays close to uniform instead of collapsing at the singular points,
and a signed phase dislocation is added at every marked minutia, which is what
actually produces the terminations and bifurcations the markers point at.
Ridge count, mean ridge period and minutia directions are all measured back off
that field rather than typed in.

Synthetic illustration for a visual reference library; not a case record.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

ORANGE = "#c0603f"          # questioned latent
BLUE = "#3b5b92"            # known exemplar / reference statistics
RIDGE = "#3a3a3a"
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

FW, FH = 10.4, 6.4
MAIN = (0.78, 0.88, 5.12, 4.46)
AUX_T = (6.88, 3.59, 2.01, 1.75)     # equal aspect, matches the print
AUX_B = (6.88, 0.88, 2.96, 1.75)

XMAX, YMAX = 23.0, 20.0              # print window, mm
ELL = (11.4, 10.0, 21.0, 18.6)       # cx, cy, w, h of the friction-ridge area
CORE = 12.0 + 13.2j                  # mm
DELTA = 7.2 + 5.4j                   # mm
A_CORE = 0.5 * np.pi
A_DELTA = -2.0 * np.pi / 3.0
LAM = 0.60
N_RIDGES = 19.0                      # ridge periods across the field

# x mm, y mm, dislocation sign; sign +1 is drawn as a ridge ending.
MINUTIAE = [(5.6, 13.4, +1), (9.1, 16.8, -1), (15.9, 15.9, +1),
            (17.8, 10.9, -1), (14.2, 6.3, +1), (9.6, 3.6, -1),
            (4.4, 7.9, +1), (16.4, 3.4, -1), (3.9, 11.0, -1)]


def base_phase(Z):
    w1 = (Z - CORE) * np.exp(-1j * A_CORE)
    w2 = (Z - DELTA) * np.exp(-1j * A_DELTA)
    return np.real(np.sqrt(w1)) + LAM * np.real(np.sqrt(w2))


def equalise(raw, inside, dx, dy, n_bin=320):
    """Remap the potential so ridge spacing stays close to uniform.

    The square-root potentials that place the core and the delta have a
    gradient that blows up near each singular point, so contours drawn at
    evenly spaced levels would crowd there and starve the middle of the print.
    Averaging the gradient magnitude over each level set gives a one
    dimensional correction ``du/dphi = C / <|grad phi|>`` that flattens the
    spacing without disturbing the topology.
    """
    gy, gx = np.gradient(raw, dy, dx)
    gm = np.hypot(gx, gy)
    v, g = raw[inside], gm[inside]
    lo, hi = float(v.min()), float(v.max())
    edges = np.linspace(lo, hi, n_bin + 1)
    k = np.clip(np.digitize(v, edges) - 1, 0, n_bin - 1)
    tot = np.bincount(k, weights=g, minlength=n_bin)
    cnt = np.bincount(k, minlength=n_bin).astype(float)
    mean_g = np.divide(tot, cnt, out=np.full(n_bin, np.nan), where=cnt > 0)
    ok = np.isfinite(mean_g)
    centres = 0.5 * (edges[:-1] + edges[1:])
    mean_g = np.interp(centres, centres[ok], mean_g[ok])
    step = np.cumsum(1.0 / np.maximum(mean_g, 1e-9))
    step = (step - step[0]) / (step[-1] - step[0])
    return np.interp(raw, centres, step) * N_RIDGES


def ridge_direction(u, X, Y, dx, dy, pt):
    gy, gx = np.gradient(u, dy, dx)
    j = int(np.argmin(np.abs(X[0, :] - pt.real)))
    i = int(np.argmin(np.abs(Y[:, 0] - pt.imag)))
    vx, vy = -gy[i, j], gx[i, j]
    n = float(np.hypot(vx, vy)) or 1.0
    return vx / n, vy / n


def draw_print(ax, X, Y, R, u, dx, dy):
    ell = Ellipse((ELL[0], ELL[1]), ELL[2], ELL[3], facecolor="white",
                  edgecolor=HAIR, linewidth=0.9, zorder=1)
    ax.add_patch(ell)
    cs = ax.contour(X, Y, R, levels=[0.0], colors=[RIDGE], linewidths=0.62,
                    zorder=2)
    cs.set_clip_path(ell)

    ax.plot([CORE.real], [CORE.imag], marker="o", ms=7.0, mfc="none",
            mec=ORANGE, mew=1.3, zorder=6)
    ax.plot([CORE.real], [CORE.imag], marker=".", ms=2.6, color=ORANGE,
            zorder=6)
    chip = dict(boxstyle="square,pad=0.16", facecolor="white",
                edgecolor="none", alpha=0.88)
    ax.text(CORE.real + 0.7, CORE.imag + 0.7, "core", fontsize=6.8,
            color=ORANGE, ha="left", va="bottom", zorder=8, bbox=chip)
    ax.plot([DELTA.real], [DELTA.imag], marker="^", ms=6.4, mfc="none",
            mec=ORANGE, mew=1.3, zorder=6)
    ax.text(DELTA.real - 0.7, DELTA.imag - 0.9, "delta", fontsize=6.8,
            color=ORANGE, ha="right", va="top", zorder=8, bbox=chip)

    for k, (mx, my, s) in enumerate(MINUTIAE, start=1):
        vx, vy = ridge_direction(u, X, Y, dx, dy, mx + 1j * my)
        ax.plot([mx, mx + 1.25 * vx], [my, my + 1.25 * vy], color=ORANGE,
                lw=1.0, zorder=6, solid_capstyle="round")
        ax.plot([mx], [my], marker="o" if s > 0 else "s", ms=5.2, mfc="none",
                mec=ORANGE, mew=1.1, zorder=7)
        ax.text(mx - 0.95 * vx - 0.20 * vy, my - 0.95 * vy + 0.20 * vx,
                str(k), fontsize=6.2, color=INK, ha="center", va="center",
                zorder=8,
                bbox=dict(boxstyle="circle,pad=0.10", facecolor="white",
                          edgecolor="none", alpha=0.88))

    hd = [plt.Line2D([], [], ls="none", marker="o", ms=5.0, mfc="none",
                     mec=ORANGE, mew=1.1),
          plt.Line2D([], [], ls="none", marker="s", ms=5.0, mfc="none",
                     mec=ORANGE, mew=1.1)]
    ax.legend(hd, ["ridge ending", "ridge bifurcation"], loc="lower right",
              fontsize=6.6, frameon=True, framealpha=0.92,
              facecolor="white", edgecolor=HAIR, handletextpad=0.5,
              borderpad=0.4)

    ax.set_xlim(0.0, XMAX)
    ax.set_ylim(0.0, YMAX)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel("x (mm)", fontsize=8.0)
    ax.set_ylabel("y (mm)", fontsize=8.0)
    ax.set_title("Questioned latent print with numbered minutiae",
                 fontsize=9.0, fontweight="bold", color=INK)


def draw_orientation(ax, X, Y, u, dx, dy):
    gy, gx = np.gradient(u, dy, dx)
    xs = np.linspace(1.8, XMAX - 1.8, 13)
    ys = np.linspace(1.6, YMAX - 1.6, 11)
    segs = []
    for yv in ys:
        i = int(np.argmin(np.abs(Y[:, 0] - yv)))
        for xv in xs:
            if (((xv - ELL[0]) / (0.5 * ELL[2])) ** 2 +
                    ((yv - ELL[1]) / (0.5 * ELL[3])) ** 2) > 0.94:
                continue
            j = int(np.argmin(np.abs(X[0, :] - xv)))
            vx, vy = -gy[i, j], gx[i, j]
            n = float(np.hypot(vx, vy)) or 1.0
            vx, vy = 0.72 * vx / n, 0.72 * vy / n
            segs.append([(xv - vx, yv - vy), (xv + vx, yv + vy)])
    ax.add_collection(LineCollection(segs, colors=BLUE, linewidths=0.85))
    ax.plot([CORE.real], [CORE.imag], marker="o", ms=5.0, mfc="none",
            mec=ORANGE, mew=1.1)
    ax.plot([DELTA.real], [DELTA.imag], marker="^", ms=5.0, mfc="none",
            mec=ORANGE, mew=1.1)
    ax.plot([CORE.real, DELTA.real], [CORE.imag, DELTA.imag], color=ORANGE,
            lw=0.8, ls=(0, (2.4, 1.6)))
    ax.set_xlim(0.0, XMAX)
    ax.set_ylim(0.0, YMAX)
    ax.set_aspect("equal")
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.tick_params(labelsize=6.4)
    ax.set_xlabel("x (mm)", fontsize=7.0)
    ax.set_ylabel("y (mm)", fontsize=7.0)
    ax.set_title("Ridge orientation field", fontsize=8.5, fontweight="bold",
                 color=INK)


def draw_scores(ax, rng, n_min):
    imp = rng.gamma(3.0, 5.0, 6000)
    gen = rng.normal(62.0, 11.0, 900)
    grid = np.linspace(0.0, 100.0, 500)

    def kde(sample, bw):
        d = (grid[:, None] - sample[None, :]) / bw
        return np.exp(-0.5 * d ** 2).sum(axis=1) / (sample.size * bw
                                                    * np.sqrt(2 * np.pi))

    d_imp, d_gen = kde(imp, 2.4), kde(gen, 3.2)
    ax.fill_between(grid, 0.0, d_imp, color=BLUE, alpha=0.22, lw=0.0)
    ax.fill_between(grid, 0.0, d_gen, color=ORANGE, alpha=0.22, lw=0.0)
    ax.plot(grid, d_imp, color=BLUE, lw=1.3, label=f"impostor (n = {imp.size})")
    ax.plot(grid, d_gen, color=ORANGE, lw=1.3,
            label=f"genuine (n = {gen.size})")

    score = 6.4 * n_min
    n_above = int(np.sum(imp >= score))
    ax.axvline(score, color=INK, lw=1.0)
    ax.text(0.97, 0.97, f"this comparison: {score:.0f}\n{n_above} of "
                        f"{imp.size} impostor scores reach it",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.2,
            color=INK, linespacing=1.35)
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, max(d_imp.max(), d_gen.max()) * 1.28)
    ax.tick_params(labelsize=6.4)
    ax.set_xlabel("minutiae match score (-)", fontsize=7.0)
    ax.set_ylabel("density (-)", fontsize=7.0)
    ax.set_title("Match score populations", fontsize=8.5, fontweight="bold",
                 color=INK)
    ax.legend(loc="upper left", fontsize=6.4, frameon=False, handlelength=1.4)
    return score, n_above, imp.size


def main():
    set_pub_style()
    rng = np.random.default_rng(2026)

    nx, ny = 660, 575
    xg = np.linspace(0.0, XMAX, nx)
    yg = np.linspace(0.0, YMAX, ny)
    X, Y = np.meshgrid(xg, yg)
    dx, dy = xg[1] - xg[0], yg[1] - yg[0]
    Z = X + 1j * Y

    inside = (((X - ELL[0]) / (0.5 * ELL[2])) ** 2 +
              ((Y - ELL[1]) / (0.5 * ELL[3])) ** 2) <= 1.0
    u_base = equalise(base_phase(Z), inside, dx, dy)

    u = u_base.copy()
    for mx, my, s in MINUTIAE:
        u = u + s * np.angle(Z - (mx + 1j * my)) / (2.0 * np.pi)
    R = np.cos(2.0 * np.pi * u)

    gy, gx = np.gradient(u_base, dy, dx)
    period = 1.0 / float(np.median(np.hypot(gx, gy)[inside]))

    # Ridge count is counted the way an examiner does it: crossings of the
    # drawn ridges along the straight core-to-delta line.
    t = np.linspace(0.0, 1.0, 1600)
    px = CORE.real + (DELTA.real - CORE.real) * t
    py = CORE.imag + (DELTA.imag - CORE.imag) * t
    jj = np.clip(np.searchsorted(xg, px), 0, nx - 1)
    ii = np.clip(np.searchsorted(yg, py), 0, ny - 1)
    sgn = np.sign(np.cos(2.0 * np.pi * u_base[ii, jj]))
    ridge_count = int(np.count_nonzero(np.diff(sgn) != 0) / 2)

    fig = plt.figure(figsize=(FW, FH))
    ax_main = axes_inches(fig, MAIN)
    ax_ofl = axes_inches(fig, AUX_T)
    ax_sc = axes_inches(fig, AUX_B)

    draw_print(ax_main, X, Y, R, u, dx, dy)
    draw_orientation(ax_ofl, X, Y, u_base, dx, dy)
    score, n_above, n_imp = draw_scores(ax_sc, rng, len(MINUTIAE))

    fig.text((AUX_T[0] + AUX_T[2] + 0.16) / FW, (AUX_T[1] + 1.42) / FH,
             f"pattern class\n  ulnar loop\n\nridge count\n  "
             f"{ridge_count} core to delta\n\nmean ridge period\n  "
             f"{period:.2f} mm\n\nmarked minutiae\n  {len(MINUTIAE)}",
             ha="left", va="top", fontsize=6.4, color=SUB, linespacing=1.55)


    panel_labels_inches(fig, [ax_main], labels=["(a)"], dx=-0.62, dy=0.14)
    panel_labels_inches(fig, [ax_ofl, ax_sc], labels=["(b)", "(c)"],
                        dx=-0.54, dy=0.14)
    savefig(fig, "794_frn_fingerprint_minutiae.png")


if __name__ == "__main__":
    main()

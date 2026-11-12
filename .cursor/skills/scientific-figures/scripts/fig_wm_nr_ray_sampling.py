"""Figure 1153 - Ray marching: coarse sampling, importance resampling, transmittance.

Everything in this figure comes out of ONE volume density field ``FIELD`` and
ONE ray.  The derivation chain is:

    FIELD (2-D grid)  --sample along the ray-->  sigma(t)
    sigma(t)          --exp(-cumulative integral)-->  T(t)
    sigma at the coarse strata --> alpha_i --> T_i --> w_i = T_i * alpha_i
    w_i               --inverse CDF-->  the importance (fine) samples

So panel (a)'s background image, the markers drawn on the ray, the sigma curve
in (b) and the transmittance curve in (c) are four views of the same array.
If you change FIELD or the ray, every number in every label follows, because
each one is an f-string over these arrays - nothing is hand-typed.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from scipy.ndimage import map_coordinates, gaussian_filter

from pubstyle import set_pub_style, savefig, get_cmap, add_panel_label
from figkit import background_card

INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"
# schematics like this one run sparse, so the backing cards use the deeper of
# the two house card greys (DESIGN_SYSTEM 2.1) to carry the empty corners
HAIR, CARD = "#d5d5d5", "#f0f0f0"
DENS = "#2e74a8"          # density / sigma channel
TRANS = "#134a76"         # transmittance channel
COARSE = "#3d6b8e"        # stratified samples
FINE = "#8e6bbf"          # importance-resampled samples
SEED = 1558  # original id; figure number is 1153

N = 360
CAM = np.array([-0.145, 0.610])
PIX = np.array([-0.058, 0.583])


def ray_dir():
    d = PIX - CAM
    return d / np.hypot(*d)


def density_field():
    """Two occluders and a back wall, all centred ON the ray by construction."""
    d = ray_dir()
    tx0 = (0.0 - CAM[0]) / d[0]
    tx1 = (1.0 - CAM[0]) / d[0]

    def at(frac):
        return CAM + (tx0 + frac * (tx1 - tx0)) * d

    y, x = np.mgrid[0:N, 0:N] / (N - 1.0)
    rng = np.random.default_rng(SEED)
    f = 0.30 * np.clip(gaussian_filter(rng.standard_normal((N, N)), 26.0) * 46.0
                       + 0.55, 0.0, None)

    def blob(c, rx, ry, amp, ang=0.0):
        ca, sa = np.cos(ang), np.sin(ang)
        dx, dy = x - c[0], y - c[1]
        u, v = dx * ca + dy * sa, -dx * sa + dy * ca
        return amp * np.exp(-((u / rx) ** 2 + (v / ry) ** 2) ** 1.35)

    f += blob(at(0.205), 0.030, 0.115, 3.4, ang=0.42)     # thin foliage sliver
    f += blob(at(0.560), 0.076, 0.058, 26.0, ang=-0.30)   # the opaque subject
    f += blob(at(0.545) + np.array([0.010, 0.050]), 0.036, 0.028, 11.0)
    f += 6.5 * np.exp(-((x - 0.912) / 0.042) ** 2)        # back wall
    return gaussian_filter(f, 1.6), tx0, tx1


FIELD, TX0, TX1 = density_field()
DVEC = ray_dir()
T_N = TX0 + 0.015 * (TX1 - TX0)
T_F = TX0 + 0.985 * (TX1 - TX0)


def sigma_at(t):
    """Bilinear lookup of FIELD at the ray positions r(t) = o + t d."""
    p = CAM[None, :] + np.atleast_1d(t)[:, None] * DVEC[None, :]
    rc = np.stack([np.clip(p[:, 1], 0, 1) * (N - 1),
                   np.clip(p[:, 0], 0, 1) * (N - 1)])
    return map_coordinates(FIELD, rc, order=1, mode="nearest")


def hierarchical_samples(n_coarse=24, n_fine=16):
    rng = np.random.default_rng(SEED)
    edges = np.linspace(T_N, T_F, n_coarse + 1)
    delta = np.diff(edges)
    tc = edges[:-1] + rng.random(n_coarse) * delta
    sc = sigma_at(tc)
    alpha = 1.0 - np.exp(-sc * delta)
    trans = np.concatenate([[1.0], np.cumprod(1.0 - alpha)[:-1]])
    w = trans * alpha

    cdf = np.concatenate([[0.0], np.cumsum(w) / w.sum()])
    u = (np.arange(n_fine) + rng.random(n_fine)) / n_fine
    k = np.clip(np.searchsorted(cdf, u, side="right") - 1, 0, n_coarse - 1)
    span = np.maximum(cdf[k + 1] - cdf[k], 1e-12)
    tf = edges[k] + (u - cdf[k]) / span * delta[k]
    return tc, sc, delta, alpha, trans, w, np.sort(tf), edges


def main():
    set_pub_style(base_font=8.5)
    tc, sc, delta, alpha, trans, w, tf, edges = hierarchical_samples()
    sf = sigma_at(tf)

    td = np.linspace(T_N, T_F, 1200)
    sd = sigma_at(td)
    tau = np.concatenate([[0.0], np.cumsum(0.5 * (sd[1:] + sd[:-1]) *
                                           np.diff(td))])
    Td = np.exp(-tau)

    k_peak = int(np.argmax(w))
    t_surf = float(tc[k_peak])
    T_surf = float(np.interp(t_surf, td, Td))
    near_surf = np.abs(tf - t_surf) < 0.09 * (T_F - T_N)
    peak_frac = 100.0 * w[k_peak] / w.sum()

    # shortest run of strata that already accounts for 90 % of the weight
    cw = np.cumsum(w) / w.sum()
    best = (np.inf, 0, w.size - 1)
    for i in range(w.size):
        lo = cw[i - 1] if i else 0.0
        j = int(np.searchsorted(cw, lo + 0.90))
        if j < w.size:
            span = edges[j + 1] - edges[i]
            best = min(best, (span, i, j))
    win_frac = 100.0 * best[0] / (T_F - T_N)

    fig = plt.figure(figsize=(10.4, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.42, 1.0],
                          left=0.052, right=0.975, top=0.845, bottom=0.245,
                          wspace=0.20, hspace=0.44)
    ax = fig.add_subplot(gs[:, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 1])

    # ---- (a) the scene, the ray and the two sample sets -------------------
    cmap = get_cmap("cb_sequential_blue")
    # a small negative vmin lifts empty space off pure white, so the volume
    # reads as a medium rather than as blank paper
    ax.imshow(FIELD, origin="lower", extent=(0, 1, 0, 1), cmap=cmap,
              vmin=-0.13 * float(FIELD.max()), vmax=float(FIELD.max()),
              interpolation="bilinear", zorder=1)
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#9aa7b2",
                           lw=0.7, zorder=6))
    ax.text(0.985, 0.975, "volume density $\\sigma(\\mathbf{x})$",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.0,
            color="#20374f", zorder=9,
            bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=HAIR, lw=0.5,
                      alpha=0.90))

    # camera body, image plane, the traced pixel
    ax.add_patch(Polygon([(CAM[0] - 0.052, CAM[1] + 0.040),
                          (CAM[0] - 0.052, CAM[1] - 0.040),
                          (CAM[0] + 0.004, CAM[1])], closed=True,
                         facecolor="#3a3a3a", edgecolor="none", zorder=7))
    ax.plot(*CAM, marker="o", ms=3.6, mfc="white", mec="#3a3a3a", mew=1.0,
            zorder=8)
    py = np.linspace(PIX[1] - 0.096, PIX[1] + 0.128, 7)
    for j in range(6):
        hit = py[j] <= PIX[1] < py[j + 1]
        ax.add_patch(Rectangle((PIX[0] - 0.013, py[j]), 0.026,
                               py[j + 1] - py[j],
                               facecolor="#cfe1ee" if hit else "white",
                               edgecolor="#5b7ba8" if hit else "#9aa7b2",
                               lw=1.0 if hit else 0.6, zorder=7))
    ax.text(PIX[0] - 0.030, py[-1] + 0.012, "image plane", ha="center",
            va="bottom", fontsize=7.0, color=SUB, zorder=9,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                      alpha=0.92))
    ax.text(CAM[0] - 0.058, CAM[1] - 0.058, "camera\ncentre $\\mathbf{o}$",
            ha="left", va="top", fontsize=7.0, color=SUB, linespacing=1.4,
            zorder=9, bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                                alpha=0.92))

    for tt, lab, va in ((T_N, "$t_n$", "top"), (T_F, "$t_f$", "top")):
        p = CAM + tt * DVEC
        n = np.array([-DVEC[1], DVEC[0]])
        ax.plot([p[0] - 0.022 * n[0], p[0] + 0.022 * n[0]],
                [p[1] - 0.022 * n[1], p[1] + 0.022 * n[1]], color="#20374f",
                lw=1.2, zorder=8)
        ax.text(p[0] + 0.006, p[1] - 0.030, lab, ha="center", va=va,
                fontsize=7.6, color="#20374f", zorder=9,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                          alpha=0.92))

    pa, pb = CAM + (T_N - 0.10) * DVEC, CAM + (T_F + 0.045) * DVEC
    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#20374f", lw=1.0,
            zorder=7)
    pc, pfn = CAM + tc[:, None] * DVEC[None, :], CAM + tf[:, None] * DVEC[None, :]
    ax.plot(pc[:, 0], pc[:, 1], ls="none", marker="o", ms=3.4, mfc="white",
            mec=COARSE, mew=0.9, zorder=9)
    ax.plot(pfn[:, 0], pfn[:, 1], ls="none", marker="D", ms=3.4, mfc=FINE,
            mec="white", mew=0.5, zorder=10)

    ps = CAM + t_surf * DVEC
    ax.annotate(f"weight peak $w_i$: $t$ = {t_surf:.2f}, $T$ = {T_surf:.2f}\n"
                f"this one interval carries {peak_frac:.0f} % of the pixel",
                xy=(ps[0], ps[1]), xytext=(0.255, 0.115),
                textcoords="axes fraction", fontsize=7.0, color="#20374f",
                ha="left", va="center", linespacing=1.45, zorder=11,
                bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#b9c9d8",
                          lw=0.6, alpha=0.94),
                arrowprops=dict(arrowstyle="-", lw=0.8, color="#20374f",
                                shrinkA=3, shrinkB=6))
    h_c = ax.plot([], [], ls="none", marker="o", ms=3.6, mfc="white",
                  mec=COARSE, mew=0.9,
                  label=f"{tc.size} stratified samples")[0]
    h_f = ax.plot([], [], ls="none", marker="D", ms=3.6, mfc=FINE, mec="white",
                  mew=0.5, label=f"{tf.size} importance samples")[0]
    leg = ax.legend(handles=[h_c, h_f], loc="upper left", fontsize=7.0,
                    frameon=True, facecolor="white", edgecolor=HAIR,
                    handletextpad=0.5, borderpad=0.42, labelspacing=0.42,
                    bbox_to_anchor=(0.005, 0.995))
    leg.get_frame().set_linewidth(0.5)
    leg.set_zorder(12)

    ax.set_xlim(-0.225, 1.030)
    ax.set_ylim(0.040, 1.030)
    ax.set_aspect("equal")
    ax.set_facecolor(CARD)
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                           edgecolor=HAIR, lw=0.7, zorder=0.5, clip_on=False))
    ax.axis("off")
    ax.set_title("Round two of sampling lands on the surface", fontsize=9.0,
                 fontweight="bold", pad=5.0)
    add_panel_label(ax, "(a)", x=-0.022, y=1.005)

    # ---- (b) sigma along the ray -----------------------------------------
    axb.fill_between(td, 0.0, sd, color="#cfe1ee", lw=0, zorder=1)
    axb.plot(td, sd, color=DENS, lw=1.5, zorder=4)
    axb.plot(tc, sc, ls="none", marker="o", ms=3.2, mfc="white", mec=COARSE,
             mew=0.9, zorder=6)
    axb.plot(tf, sf, ls="none", marker="D", ms=3.2, mfc=FINE, mec="white",
             mew=0.5, zorder=7)
    axb.axvline(t_surf, color="#20374f", lw=0.9, ls=(0, (4, 2.4)), zorder=3)
    axb.set_xlim(T_N, T_F)
    axb.set_ylim(0.0, 1.13 * float(sd.max()))
    axb.set_ylabel("density $\\sigma(t)$", fontsize=8.0)
    axb.set_title(f"Three occluders; the one at $t$ = {t_surf:.2f} peaks at "
                  f"$\\sigma$ = {sd.max():.0f}", fontsize=8.6,
                  fontweight="bold", pad=4.0)
    axb.tick_params(labelsize=7.0, length=2.4)
    axb.set_xticklabels([])
    axb.grid(True, axis="y", lw=0.45, color="#ececec")
    axb.set_axisbelow(True)
    axb.text(0.018, 0.95, f"{near_surf.sum()} of {tf.size} importance samples "
             f"land within\n9 % of the ray length of the peak",
             transform=axb.transAxes, ha="left", va="top", fontsize=6.8,
             color=SUB, linespacing=1.5, zorder=8,
             bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=HAIR,
                       lw=0.5, alpha=0.92))
    add_panel_label(axb, "(b)", x=-0.118, y=1.02)

    # ---- (c) transmittance ------------------------------------------------
    axc.fill_between(td, 0.0, Td, color="#e3edf5", lw=0, zorder=1)
    axc.plot(td, Td, color=TRANS, lw=1.7, zorder=4)
    axc.plot(edges[:-1], trans, ls="none", marker="o", ms=3.2, mfc="white",
             mec=COARSE, mew=0.9, zorder=6)
    axc.axvline(t_surf, color="#20374f", lw=0.9, ls=(0, (4, 2.4)), zorder=3)
    axc.plot([t_surf], [T_surf], marker="o", ms=5.0, mfc="white",
             mec="#20374f", mew=1.3, zorder=8)
    axc.annotate(f"same $t$ as the peak in (b):\n$T$ falls 1.00 "
                 f"$\\rightarrow$ {T_surf:.2f}", xy=(t_surf, T_surf),
                 xytext=(t_surf + 0.052 * (T_F - T_N), 0.60), fontsize=6.9,
                 color="#20374f", ha="left", va="center", linespacing=1.5,
                 zorder=9,
                 bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#b9c9d8",
                           lw=0.6, alpha=0.94),
                 arrowprops=dict(arrowstyle="-", lw=0.8, color="#20374f",
                                 shrinkA=2, shrinkB=5))
    axc.set_xlim(T_N, T_F)
    axc.set_ylim(0.0, 1.06)
    axc.set_xlabel("distance along the ray $t$", fontsize=8.0)
    axc.set_ylabel("transmittance $T(t)$", fontsize=8.0)
    axc.set_title(f"90 % of the pixel is decided in {win_frac:.0f} % of the "
                  f"ray", fontsize=8.6, fontweight="bold", pad=4.0)
    axc.tick_params(labelsize=7.0, length=2.4)
    axc.grid(True, axis="y", lw=0.45, color="#ececec")
    axc.set_axisbelow(True)
    add_panel_label(axc, "(c)", x=-0.118, y=1.02)

    # ---- headline + the volume rendering integral -------------------------

    background_card(fig, (0.052, 0.038, 0.923, 0.150), facecolor=CARD,
                    edgecolor=HAIR, zorder=-5, radius=0.010)
    fig.text(0.088, 0.140, r"$C(\mathbf{r})=\int_{t_n}^{t_f}$", fontsize=11.0,
             color=INK, ha="left", va="center")
    fig.text(0.183, 0.140, r"$T(t)$", fontsize=11.0, color=TRANS, ha="left",
             va="center")
    fig.text(0.220, 0.140, r"$\sigma(t)$", fontsize=11.0, color=DENS,
             ha="left", va="center")
    fig.text(0.266, 0.140, r"$\mathbf{c}(t,\mathbf{d})\,dt$", fontsize=11.0,
             color=INK, ha="left", va="center")
    fig.text(0.400, 0.140, r"$T(t)=\exp\!\left(-\int_{t_n}^{t}"
             r"\sigma(s)\,ds\right)$", fontsize=11.0, color=INK, ha="left",
             va="center")
    fig.text(0.680, 0.140, r"$\hat{C}=\sum_i T_i\left(1-e^{-\sigma_i"
             r"\delta_i}\right)\mathbf{c}_i$", fontsize=11.0, color=INK,
             ha="left", va="center")
    savefig(fig, "1153_wm_nr_ray_sampling.png")


if __name__ == "__main__":
    main()

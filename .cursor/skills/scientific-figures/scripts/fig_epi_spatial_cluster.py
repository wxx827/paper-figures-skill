"""Small-area incidence map with a scan-statistic cluster and a Moran plot.

Main panel: a synthetic set of administrative districts (bounded Voronoi cells)
shaded by notified incidence per 100k, with the most likely cluster from a
circular Poisson scan statistic ringed in black and its Monte-Carlo p-value
reported. Auxiliary panel: the Moran scatter of standardised incidence against
its spatial lag, with each district coloured by its LISA quadrant.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle
from scipy.spatial import Voronoi
from figkit import panel_label_inches, safe_colorbar
from pubstyle import set_pub_style, savefig, style_axes, get_cmap

NX, NY = 10, 9
INK = "#1a1a1a"
RULE = "#333333"
MUTED = "#777777"
DASH = (0, (4.0, 2.4))
QUAD_C = {"HH": "#b23a48", "LL": "#3b5b92", "HL": "#c9a227", "LH": "#8e6bbf"}
KMAX = 16
N_MC = 499


def districts(rng):
    """Bounded Voronoi cells: mirror the seeds so every cell is finite."""
    gx, gy = np.meshgrid((np.arange(NX) + 0.5) / NX, (np.arange(NY) + 0.5) / NY)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    pts += rng.uniform(-0.42, 0.42, pts.shape) * np.array([1 / NX, 1 / NY])
    mirrored = np.vstack([pts,
                          np.column_stack([-pts[:, 0], pts[:, 1]]),
                          np.column_stack([2 - pts[:, 0], pts[:, 1]]),
                          np.column_stack([pts[:, 0], -pts[:, 1]]),
                          np.column_stack([pts[:, 0], 2 - pts[:, 1]])])
    vor = Voronoi(mirrored)
    n = len(pts)
    polys = [np.array([vor.vertices[v]
                       for v in vor.regions[vor.point_region[i]]])
             for i in range(n)]
    nb = [[] for _ in range(n)]
    for a, b in vor.ridge_points:
        if a < n and b < n:
            nb[a].append(b)
            nb[b].append(a)
    return pts, polys, nb


def scan_max(cases, cum_pop, idx, tot_c, tot_p):
    """Largest Poisson log-likelihood ratio over all circular windows."""
    cum_c = np.cumsum(cases[idx], axis=1)
    exp = tot_c * cum_pop / tot_p
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = (cum_c * np.log(cum_c / exp)
              + (tot_c - cum_c) * np.log((tot_c - cum_c) / (tot_c - exp)))
    ll = np.where((cum_c > exp) & np.isfinite(ll) & (cum_pop < 0.5 * tot_p),
                  ll, 0.0)
    return ll


def main():
    set_pub_style(base_font=9)
    rng = np.random.default_rng(5)
    pts, polys, nb = districts(rng)
    n = len(pts)

    # ---- synthetic incidence with one true hotspot -------------------------
    pop = rng.integers(25000, 140000, n).astype(float)
    field = (255.0
             + 150.0 * np.exp(-((pts[:, 0] - 0.78) ** 2
                                + (pts[:, 1] - 0.24) ** 2) / 0.09)
             + 520.0 * np.exp(-((pts[:, 0] - 0.30) ** 2
                                + (pts[:, 1] - 0.70) ** 2) / 0.018))
    cases = rng.poisson(field / 1e5 * pop).astype(float)
    rate = cases / pop * 1e5

    # ---- circular scan statistic -------------------------------------------
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    idx = np.argsort(d, axis=1)[:, :KMAX]
    cum_pop = np.cumsum(pop[idx], axis=1)
    tot_c, tot_p = cases.sum(), pop.sum()
    ll = scan_max(cases, cum_pop, idx, tot_c, tot_p)
    i_best, k_best = np.unravel_index(int(np.argmax(ll)), ll.shape)
    ll_obs = ll[i_best, k_best]

    mc = np.empty(N_MC)
    for s in range(N_MC):
        sim = rng.multinomial(int(tot_c), pop / tot_p).astype(float)
        mc[s] = scan_max(sim, cum_pop, idx, tot_c, tot_p).max()
    p_scan = (1 + int((mc >= ll_obs).sum())) / (N_MC + 1)

    members = idx[i_best, :k_best + 1]
    centre = pts[i_best]
    radius = float(d[i_best, members].max()) * 1.06
    obs_c = cases[members].sum()
    exp_c = tot_c * pop[members].sum() / tot_p

    # ---- Moran's I ----------------------------------------------------------
    W = np.zeros((n, n))
    for i, js in enumerate(nb):
        for j in set(js):
            W[i, j] = 1.0
    W /= np.maximum(W.sum(axis=1, keepdims=True), 1.0)
    z = (rate - rate.mean()) / rate.std()
    lag = W @ z
    moran = float(z @ lag / (z @ z))
    perm = np.array([float(zp @ (W @ zp) / (zp @ zp))
                     for zp in (rng.permutation(z) for _ in range(999))])
    p_moran = (1 + int((perm >= moran).sum())) / 1000.0

    quad = np.where(z > 0, np.where(lag > 0, "HH", "HL"),
                    np.where(lag > 0, "LH", "LL"))

    # ---- figure -------------------------------------------------------------
    fig = plt.figure(figsize=(7.2, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.58], wspace=0.22,
                          left=0.055, right=0.975, top=0.845, bottom=0.155)
    ax = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])

    cmap = get_cmap("cb_sequential_blue")
    pc = PolyCollection(polys, array=rate, cmap=cmap, edgecolors="white",
                        linewidths=0.5, zorder=3)
    # clip the scale at the 98th percentile so the bulk of the map is readable
    pc.set_clim(float(rate.min()), float(np.percentile(rate, 98)))
    ax.add_collection(pc)
    ax.add_patch(Circle(centre, radius, facecolor="none", edgecolor=INK,
                        lw=2.0, zorder=6))
    ax.annotate(f"most likely cluster: {len(members)} districts\n"
                f"observed / expected = {obs_c / exp_c:.2f}, "
                f"Monte-Carlo P = {p_scan:.3f}",
                xy=(centre[0] + radius * 0.74, centre[1] + radius * 0.74),
                xytext=(0.975, 0.975), textcoords="axes fraction",
                fontsize=6.8, color=INK, linespacing=1.4, va="top",
                ha="right",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc",
                          lw=0.6, alpha=0.94),
                arrowprops=dict(arrowstyle="-", lw=0.8, color=INK))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Notified incidence by district", fontsize=9.0,
                 fontweight="bold", loc="left", pad=4)
    safe_colorbar(fig, pc, ax, label="incidence (cases per 100k)",
                  side="bottom", width=0.10, pad=0.10, shrink=0.86,
                  extend="max")

    # ---- Moran scatter ------------------------------------------------------
    for q, c in QUAD_C.items():
        m = quad == q
        axb.scatter(z[m], lag[m], s=15, color=c, edgecolor="white",
                    linewidth=0.35, zorder=4, label=f"{q} (n = {int(m.sum())})")
    lo, hi = z.min() - 0.35, z.max() + 0.35
    axb.plot([lo, hi], [moran * lo, moran * hi], color=RULE, lw=1.3, zorder=5)
    axb.axhline(0, color=MUTED, ls=DASH, lw=0.8, zorder=2)
    axb.axvline(0, color=MUTED, ls=DASH, lw=0.8, zorder=2)
    axb.set_xlim(lo, hi)
    axb.set_xlabel("Standardised incidence $z$ (-)", fontsize=8.0)
    axb.set_ylabel("Spatial lag $Wz$ (-)", fontsize=8.0)
    axb.set_title("Moran scatter by LISA quadrant", fontsize=8.5,
                  fontweight="bold", loc="left", pad=4)
    axb.text(0.030, 0.965,
             f"Moran's $I$ = {moran:.2f}\n999-permutation P = {p_moran:.3f}",
             transform=axb.transAxes, ha="left", va="top", fontsize=6.8,
             linespacing=1.4, color=INK,
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc",
                       lw=0.6))
    axb.legend(loc="lower right", fontsize=6.4, frameon=False, handlelength=0.9,
               handletextpad=0.4, labelspacing=0.25, borderaxespad=0.3)
    axb.tick_params(labelsize=6.8)
    style_axes(axb)

    panel_label_inches(fig, ax, "(a)", dx=-0.32, dy=0.06)
    panel_label_inches(fig, axb, "(b)", dx=-0.42, dy=0.06)

    savefig(fig, "432_epi_spatial_cluster.png")


if __name__ == "__main__":
    main()

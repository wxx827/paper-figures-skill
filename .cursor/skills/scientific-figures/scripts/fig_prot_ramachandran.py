"""Ramachandran plot of backbone dihedral angles for a synthetic protein.

Main panel: phi-psi scatter over favoured / allowed density contours, with the
three classical basins labelled.  Side panel: fraction of residues per basin.

Backbone angles are drawn from a mixture of von-Mises-like Gaussian basins on
the torus, so the scatter and the contours come from the same model and every
quoted percentage is computed from the sampled points.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import panel_labels_inches

SEED = 20250813

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
H_RED, H_BLUE, H_GREEN, H_GOLD, H_PURPLE = (
    "#b23a48", "#3b5b92", "#2e7d5b", "#c9a227", "#8e6bbf")

# (phi, psi, sigma_phi, sigma_psi, rho, weight) for the populated basins of a
# generic residue: beta / polyproline-II, right-handed alpha, left-handed alpha.
BASINS = [
    (-125.0, 140.0, 26.0, 22.0, 0.30, 0.34),   # beta sheet
    (-72.0, 148.0, 16.0, 16.0, 0.10, 0.14),    # polyproline II
    (-63.0, -43.0, 15.0, 14.0, 0.45, 0.44),    # alpha helix, right handed
    (57.0, 42.0, 13.0, 14.0, -0.25, 0.08),     # alpha helix, left handed
]

REGIONS = {
    "beta / PPII": dict(color=H_BLUE),
    r"$\alpha$-helix (R)": dict(color=H_RED),
    r"$\alpha$-helix (L)": dict(color=H_GREEN),
    "outlier": dict(color=H_GOLD),
}


def _mixture_density(phi, psi):
    """Sum of the four bivariate-normal basins, wrapped on the torus."""
    z = np.zeros_like(phi, dtype=float)
    for mu_p, mu_s, sp, ss, rho, w in BASINS:
        for shift_p in (-360.0, 0.0, 360.0):
            for shift_s in (-360.0, 0.0, 360.0):
                a = (phi - mu_p - shift_p) / sp
                b = (psi - mu_s - shift_s) / ss
                q = (a ** 2 - 2 * rho * a * b + b ** 2) / (1 - rho ** 2)
                z += w * np.exp(-0.5 * q)
    return z


def _sample(rng, n):
    w = np.array([b[-1] for b in BASINS])
    w = w / w.sum()
    pick = rng.choice(len(BASINS), size=n, p=w)
    phi = np.empty(n)
    psi = np.empty(n)
    for k, (mu_p, mu_s, sp, ss, rho, _) in enumerate(BASINS):
        m = pick == k
        cnt = int(m.sum())
        if not cnt:
            continue
        cov = [[sp ** 2, rho * sp * ss], [rho * sp * ss, ss ** 2]]
        draw = rng.multivariate_normal([mu_p, mu_s], cov, size=cnt)
        phi[m], psi[m] = draw[:, 0], draw[:, 1]
    # a handful of genuinely strained residues
    n_out = max(4, int(0.018 * n))
    idx = rng.choice(n, size=n_out, replace=False)
    phi[idx] = rng.uniform(-180, 180, n_out)
    psi[idx] = rng.uniform(-180, 180, n_out)
    return (phi + 180.0) % 360.0 - 180.0, (psi + 180.0) % 360.0 - 180.0


def _classify(phi, psi):
    lab = np.full(phi.shape, "outlier", dtype=object)
    beta = ((phi > -180) & (phi < -20) &
            ((psi > 62) | (psi < -160)))
    ar = (phi > -160) & (phi < -20) & (psi > -85) & (psi < 40)
    al = (phi > 18) & (phi < 105) & (psi > -25) & (psi < 95)
    lab[beta] = "beta / PPII"
    lab[ar] = r"$\alpha$-helix (R)"
    lab[al] = r"$\alpha$-helix (L)"
    return lab


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    n_res = 640
    phi, psi = _sample(rng, n_res)
    is_gly = rng.random(n_res) < 0.085
    # glycine is achiral and happily visits the left-handed basin
    g_idx = np.where(is_gly)[0]
    flip = g_idx[rng.random(g_idx.size) < 0.55]
    phi[flip] *= -1.0
    psi[flip] *= -1.0
    lab = _classify(phi, psi)

    grid = np.linspace(-180, 180, 361)
    PHI, PSI = np.meshgrid(grid, grid)
    dens = _mixture_density(PHI, PSI)
    dens /= dens.max()

    fig = plt.figure(figsize=(7.2, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.58],
                          left=0.085, right=0.975, bottom=0.185, top=0.885,
                          wspace=0.34)
    ax = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])

    # ---- density backdrop: favoured (98 %) and allowed (99.8 %) envelopes ---
    cmap = get_cmap("cb_sequential_blue")
    lv_allowed, lv_fav = 0.0022, 0.055
    ax.contourf(PHI, PSI, dens, levels=[lv_allowed, lv_fav, 1.0],
                colors=[cmap(0.16), cmap(0.36)], zorder=0)
    ax.contour(PHI, PSI, dens, levels=[lv_allowed, lv_fav],
               colors=[cmap(0.62), cmap(0.86)], linewidths=[0.7, 0.9],
               zorder=1)

    # ---- residues ---------------------------------------------------------
    for name, style in REGIONS.items():
        m = (lab == name) & ~is_gly
        ax.scatter(phi[m], psi[m], s=7.5, facecolor=style["color"],
                   edgecolor="white", linewidth=0.22, alpha=0.92,
                   zorder=4, label=None)
    ax.scatter(phi[is_gly], psi[is_gly], s=16.0, marker="^",
               facecolor="none", edgecolor=H_PURPLE, linewidth=0.75,
               zorder=5)

    for v in (-180, -90, 0, 90, 180):
        ax.axhline(v, color="#cccccc", lw=0.5, zorder=2)
        ax.axvline(v, color="#cccccc", lw=0.5, zorder=2)

    basin_notes = [
        (-148.0, 106.0, r"$\beta$ sheet", H_BLUE),
        (-30.0, 152.0, "PPII", H_BLUE),
        (-146.0, -52.0, r"$\alpha_{\mathrm{R}}$", H_RED),
        (63.0, 100.0, r"$\alpha_{\mathrm{L}}$", H_GREEN),
    ]
    for x, y, s, c in basin_notes:
        ax.text(x, y, s, fontsize=7.6, color=c, fontweight="bold",
                ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                          edgecolor="none", alpha=0.72))

    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-180, 181, 90))
    ax.set_yticks(np.arange(-180, 181, 90))
    ax.set_xlabel(r"$\phi$ (deg)", fontsize=8.0)
    ax.set_ylabel(r"$\psi$ (deg)", fontsize=8.0)
    ax.tick_params(labelsize=7.0, length=2.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.8)
        ax.spines[s].set_color(RULE)
    ax.set_title("Backbone dihedral distribution", fontsize=9.0,
                 fontweight="bold", pad=4.0)

    n_fav = int((dens[
        np.clip(((psi + 180) * (dens.shape[0] - 1) / 360).astype(int),
                0, dens.shape[0] - 1),
        np.clip(((phi + 180) * (dens.shape[1] - 1) / 360).astype(int),
                0, dens.shape[1] - 1)] >= lv_fav).sum())
    handles = [
        Line2D([], [], marker="o", ls="none", ms=3.2, mfc=H_BLUE,
               mec="white", mew=0.3, label="non-Gly residue"),
        Line2D([], [], marker="^", ls="none", ms=3.8, mfc="none",
               mec=H_PURPLE, mew=0.75, label="glycine"),
        Line2D([], [], marker="s", ls="none", ms=4.2, mfc=cmap(0.36),
               mec=cmap(0.86), mew=0.6, label="favoured"),
        Line2D([], [], marker="s", ls="none", ms=4.2, mfc=cmap(0.16),
               mec=cmap(0.62), mew=0.6, label="allowed"),
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.145), ncol=4, fontsize=6.8,
              frameon=False, handletextpad=0.4, columnspacing=1.1,
              borderaxespad=0.0)

    # ---- companion: occupancy per basin -----------------------------------
    names = list(REGIONS)
    frac = np.array([(lab == k).mean() for k in names]) * 100.0
    ypos = np.arange(len(names))[::-1]
    cols = [REGIONS[k]["color"] for k in names]
    axb.barh(ypos, frac, height=0.58, color=cols, edgecolor="white",
             linewidth=0.6, zorder=3)
    for y, v in zip(ypos, frac):
        axb.text(v + 1.6, y, f"{v:.1f}%", va="center", ha="left",
                 fontsize=6.6, color=SUB)
    axb.set_yticks(ypos)
    axb.set_yticklabels(names, fontsize=6.8)
    axb.set_xlim(0, max(frac) * 1.34)
    axb.set_xlabel("residues in basin (%)", fontsize=7.4)
    axb.tick_params(labelsize=6.5, length=2.2)
    axb.grid(axis="x", lw=0.45, color="#e6e6e6")
    axb.set_axisbelow(True)
    axb.set_title("Basin occupancy", fontsize=8.5, fontweight="bold", pad=4.0)

    gly_left = float(((lab == r"$\alpha$-helix (L)") & is_gly).sum())
    all_left = float((lab == r"$\alpha$-helix (L)").sum())

    panel_labels_inches(fig, [ax, axb], dx=-0.34, dy=0.09)
    savefig(fig, "522_prot_ramachandran.png")


if __name__ == "__main__":
    main()

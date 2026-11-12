"""Electronic band structure and density of states of wurtzite ZnO.

Left: bands along Gamma-M-K-Gamma-A, built from cosine hoppings so that every
branch is continuous and correctly degenerate at the high-symmetry points.
Right: the matching total and projected DOS, obtained by Gaussian broadening
of the same model evaluated on a 3D k-mesh.  Both panels share the energy
axis, the Fermi level is the zero of energy and the fundamental gap is shaded
across both.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig, add_panel_label

LABELS = ["$\\Gamma$", "M", "K", "$\\Gamma$", "A"]
KPTS = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1 / 3, 1 / 3, 0.0],
                 [0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])

# (e0, t_xy, t_xy2, t_z) -- O 2p valence manifold, max at Gamma
VAL = [(-1.05, 0.62, 0.28, 0.10), (-1.85, 0.44, 0.34, 0.16),
       (-2.55, 0.30, 0.22, 0.26), (-3.35, 0.22, 0.30, 0.18)]
# Zn 3d semicore: nearly dispersionless
SEMI = [(-4.95, 0.05, 0.02, 0.015), (-5.28, 0.04, 0.02, 0.010)]
# Zn 4s / 4p conduction manifold, min at Gamma
CON = [(5.27, -0.55, -0.20, -0.10), (6.75, -0.32, -0.14, -0.06),
       (7.55, -0.20, -0.16, 0.05)]

C_O = "#2f6f9e"
C_ZN = "#b5642f"
C_EF = "#b23a48"


def band_energy(k, e0, t1, t2, t3):
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    return (e0
            + t1 * (np.cos(2 * np.pi * kx) + np.cos(2 * np.pi * ky))
            + t2 * np.cos(2 * np.pi * kx) * np.cos(2 * np.pi * ky)
            + t3 * np.cos(2 * np.pi * kz))


def kpath(n=200):
    seg, dist, ticks = [], 0.0, [0.0]
    for i in range(len(KPTS) - 1):
        a, b = KPTS[i], KPTS[i + 1]
        length = float(np.linalg.norm(b - a))
        t = np.linspace(0.0, 1.0, n, endpoint=(i == len(KPTS) - 2))
        seg.append((a + (b - a) * t[:, None], dist + length * t))
        dist += length
        ticks.append(dist)
    return (np.vstack([s[0] for s in seg]),
            np.concatenate([s[1] for s in seg]), np.array(ticks))


def dos(energies, grid, sigma):
    out = np.zeros_like(grid)
    flat = np.asarray(energies).ravel()
    for e in flat:
        out += np.exp(-0.5 * ((grid - e) / sigma) ** 2)
    return out / (sigma * np.sqrt(2.0 * np.pi) * flat.size)


def main():
    set_pub_style(base_font=9)

    fig = plt.figure(figsize=(7.4, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.0], wspace=0.035,
                          left=0.075, right=0.975, top=0.885, bottom=0.10)
    axb = fig.add_subplot(gs[0])
    axd = fig.add_subplot(gs[1], sharey=axb)

    ks, xs, ticks = kpath()
    e_val = np.array([band_energy(ks, *p) for p in VAL])
    e_semi = np.array([band_energy(ks, *p) for p in SEMI])
    e_con = np.array([band_energy(ks, *p) for p in CON])

    vbm = e_val.max()
    e_val, e_semi, e_con = e_val - vbm, e_semi - vbm, e_con - vbm
    gap = e_con.min()
    e_lo, e_hi = -7.2, 8.0

    for ax in (axb, axd):
        ax.axhspan(0.0, gap, facecolor="#f1f1f1", lw=0, zorder=0)
        ax.axhline(0.0, color=C_EF, lw=0.9, ls=(0, (5, 2.5)), zorder=6)

    for e in e_val:
        axb.plot(xs, e, color=C_O, lw=1.15, zorder=4)
    for e in e_semi:
        axb.plot(xs, e, color=C_ZN, lw=1.15, zorder=4)
    for e in e_con:
        axb.plot(xs, e, color=C_ZN, lw=1.15, zorder=4)

    for t in ticks[1:-1]:
        axb.axvline(t, color="#9a9a9a", lw=0.6, zorder=2)
    axb.set_xticks(ticks)
    axb.set_xticklabels(LABELS)
    axb.set_xlim(ticks[0], ticks[-1])
    axb.set_ylim(e_lo, e_hi)
    axb.set_ylabel("E – E$_F$ (eV)")
    axb.set_xlabel("Wave vector")
    axb.tick_params(axis="x", length=0, pad=4, labelsize=9.5)
    axb.tick_params(axis="y", labelsize=8)
    for s in ("top", "right"):
        axb.spines[s].set_visible(True)
        axb.spines[s].set_linewidth(0.8)
        axb.spines[s].set_color("#333333")

    # --- direct gap call-out at the third high-symmetry point -------------
    xg = ticks[3]
    axb.plot([xg, xg], [0.0, gap], color="#333333", lw=0.0, zorder=6)
    axb.annotate("", xy=(xg, gap), xytext=(xg, 0.0),
                 arrowprops=dict(arrowstyle="<|-|>", color="#333333", lw=0.9,
                                 mutation_scale=7), zorder=7)
    axb.plot([xg], [0.0], marker="o", ms=4.2, mfc="white", mec=C_O, mew=1.0,
             zorder=8)
    axb.plot([xg], [gap], marker="o", ms=4.2, mfc="white", mec=C_ZN, mew=1.0,
             zorder=8)
    axb.text(xg + 0.035, 0.5 * gap,
             f"$E_g$ = {gap:.2f} eV\ndirect at $\\Gamma$", fontsize=7.2,
             color="#333333", ha="left", va="center", zorder=8,
             bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                       edgecolor="none", alpha=0.85))
    halo = dict(boxstyle="round,pad=0.18", facecolor="white",
                edgecolor="none", alpha=0.85)
    axb.text(ticks[1] * 0.42, e_semi.mean() + 0.55, "Zn 3d semicore",
             fontsize=6.8, color=C_ZN, ha="center", va="bottom", bbox=halo)
    axb.text(ticks[1] * 0.42, -2.15, "O 2p valence", fontsize=6.8, color=C_O,
             ha="center", va="center", bbox=halo)

    # --- density of states -------------------------------------------------
    n = 14
    g = (np.arange(n) + 0.5) / n
    kx, ky, kz = np.meshgrid(g, g, g, indexing="ij")
    mesh = np.stack([kx, ky, kz], axis=-1).reshape(-1, 3)
    mv = np.array([band_energy(mesh, *p) for p in VAL]) - vbm
    ms = np.array([band_energy(mesh, *p) for p in SEMI]) - vbm
    mc = np.array([band_energy(mesh, *p) for p in CON]) - vbm

    eg = np.linspace(e_lo, e_hi, 1200)
    d_o = 4.6 * dos(mv, eg, 0.16) + 0.55 * dos(mc, eg, 0.20)
    d_zn = 3.1 * dos(ms, eg, 0.11) + 0.9 * dos(mv, eg, 0.16) + \
        2.0 * dos(mc, eg, 0.20)
    total = d_o + d_zn

    axd.fill_betweenx(eg, 0.0, d_o, facecolor=C_O, alpha=0.28, lw=0, zorder=3)
    axd.plot(d_o, eg, color=C_O, lw=0.9, zorder=4)
    axd.fill_betweenx(eg, 0.0, d_zn, facecolor=C_ZN, alpha=0.28, lw=0,
                      zorder=3)
    axd.plot(d_zn, eg, color=C_ZN, lw=0.9, zorder=4)
    axd.plot(total, eg, color="#333333", lw=1.1, zorder=5)

    xmax = total.max() * 1.30
    axd.set_xlim(0.0, xmax)
    axd.set_xlabel("DOS (states eV$^{-1}$ f.u.$^{-1}$)")
    axd.tick_params(labelleft=False, labelsize=8)
    axd.set_xticks(np.arange(0, xmax, 2.0))
    for s in ("top", "right"):
        axd.spines[s].set_visible(True)
        axd.spines[s].set_linewidth(0.8)
        axd.spines[s].set_color("#333333")
    axd.text(xmax * 0.96, 0.32, "$E_F$", fontsize=7.4, color=C_EF,
             ha="right", va="bottom", fontweight="bold")
    axd.text(xmax * 0.96, 0.5 * gap, "band gap", fontsize=6.8,
             color="#7a7a7a", ha="right", va="center", rotation=90)

    handles = [Line2D([], [], color=C_O, lw=1.5, label="O 2p"),
               Line2D([], [], color=C_ZN, lw=1.5, label="Zn 3d / 4s"),
               Line2D([], [], color="#333333", lw=1.5, label="total DOS"),
               Line2D([], [], color=C_EF, lw=1.1, ls=(0, (5, 2.5)),
                      label="Fermi level")]

    add_panel_label(axb, "(a)", x=-0.082, y=1.006, size=10)
    add_panel_label(axd, "(b)", x=-0.10, y=1.006, size=10)
    savefig(fig, "126_dos_bandstructure.png")


if __name__ == "__main__":
    main()

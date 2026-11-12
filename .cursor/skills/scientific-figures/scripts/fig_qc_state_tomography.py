"""Two-qubit state tomography: real and imaginary parts of the density matrix.

The reconstructed density matrix is built once - a phased Bell state, a
depolarising channel, then a small Hermitian statistical residual - and every
panel and every number reads from that one array.  The two 3D panels are the
main result and share a single z range and a single diverging norm, because
letting matplotlib autoscale them separately would make a nearly real matrix
look as if it had a large imaginary part.  The two small panels on the right
carry the evidence: where the reconstruction differs from the target, and how
the four populations compare.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import panel_labels_inches, safe_colorbar, shared_norm

BLUE = "#3b5b92"
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

DIM = 4
KET = [r"$|00\rangle$", r"$|01\rangle$", r"$|10\rangle$", r"$|11\rangle$"]
SHOTS = 8192
DEPOL = 0.14
PHASE_DEG = 60.0
DIV = get_cmap("cb_diverging")


def reconstruct(seed=7):
    rng = np.random.default_rng(seed)
    psi = np.zeros(DIM, dtype=complex)
    psi[0] = 1.0 / np.sqrt(2.0)
    psi[3] = np.exp(1j * np.deg2rad(PHASE_DEG)) / np.sqrt(2.0)
    rho_ideal = np.outer(psi, psi.conj())

    rho = (1.0 - DEPOL) * rho_ideal + DEPOL * np.eye(DIM) / DIM
    res = (rng.normal(size=(DIM, DIM)) +
           1j * rng.normal(size=(DIM, DIM))) * 0.022
    res = 0.5 * (res + res.conj().T)
    res -= np.eye(DIM) * np.trace(res).real / DIM
    return psi, rho_ideal, rho + res


def style_3d(ax, zlo, zhi, title):
    ax.set_box_aspect((1.0, 1.0, 0.62))
    ax.view_init(elev=26.0, azim=-58.0)
    ax.set_zlim(zlo, zhi)
    ax.set_xticks(np.arange(DIM) + 0.5)
    ax.set_yticks(np.arange(DIM) + 0.5)
    ax.set_xticklabels(KET, fontsize=6.4, color=INK)
    ax.set_yticklabels(KET, fontsize=6.4, color=INK)
    ax.tick_params(axis="z", labelsize=6.4, colors=INK, pad=0.5)
    ax.set_zlabel("amplitude (-)", fontsize=7.4, labelpad=-4.0)
    ax.set_title(title, fontsize=8.8, fontweight="bold", color=INK, pad=-2.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#fcfcfc")
        axis.pane.set_edgecolor(HAIR)
        axis.pane.set_alpha(1.0)
        axis._axinfo["grid"].update(color="#e2e2e2", linewidth=0.5)


def bars(ax, mat, norm):
    i, j = np.meshgrid(np.arange(DIM), np.arange(DIM), indexing="ij")
    dz = mat.ravel()
    ax.bar3d(i.ravel() + 0.16, j.ravel() + 0.16, np.zeros(dz.size),
             0.68, 0.68, dz, color=DIV(norm(dz)), edgecolor="white",
             linewidth=0.35, shade=True, zsort="max")


def main():
    set_pub_style()
    psi, rho_ideal, rho = reconstruct()
    re, im = rho.real, rho.imag
    fidelity = float(np.real(psi.conj() @ rho @ psi))
    purity = float(np.real(np.trace(rho @ rho)))
    diff = re - rho_ideal.real

    norm = shared_norm([re, im], symmetric=True)
    zlo, zhi = norm.vmin * 1.08, norm.vmax * 1.08

    fig = plt.figure(figsize=(10.4, 5.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.62],
                          height_ratios=[1.0, 0.62], wspace=0.16,
                          hspace=0.55)
    ax_re = fig.add_subplot(gs[:, 0], projection="3d")
    ax_im = fig.add_subplot(gs[:, 1], projection="3d")
    ax_df = fig.add_subplot(gs[0, 2])
    ax_pp = fig.add_subplot(gs[1, 2])

    for ax, mat, title in [(ax_re, re, r"Re $\rho$  (reconstructed)"),
                           (ax_im, im, r"Im $\rho$  (reconstructed)")]:
        style_3d(ax, zlo, zhi, title)
        bars(ax, mat, norm)

    im_df = ax_df.imshow(diff, cmap=DIV, norm=shared_norm([diff],
                                                          symmetric=True))
    ax_df.set_aspect("equal")
    ax_df.set_xticks(np.arange(DIM))
    ax_df.set_yticks(np.arange(DIM))
    ax_df.set_xticklabels(KET, fontsize=6.2)
    ax_df.set_yticklabels(KET, fontsize=6.2)
    ax_df.tick_params(length=2.0)
    for s in ax_df.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    ax_df.set_title(r"Re $\rho_{\mathrm{meas}} - $ Re $\rho_{\mathrm{ideal}}$",
                    fontsize=8.5, fontweight="bold", color=INK)
    safe_colorbar(fig, im_df, ax_df, label="deviation (-)", width=0.09,
                  pad=0.06, labelsize=6.2)

    pos = np.arange(DIM)
    ideal_pop = np.real(np.diag(rho_ideal)) * 100.0
    meas_pop = np.real(np.diag(rho)) * 100.0
    ax_pp.bar(pos - 0.19, ideal_pop, width=0.36, color="none",
              edgecolor=GREY, linewidth=0.9, linestyle=(0, (2.4, 1.6)),
              label="ideal")
    ax_pp.bar(pos + 0.19, meas_pop, width=0.36, color=BLUE, edgecolor="white",
              linewidth=0.5, label="measured")
    ax_pp.set_xticks(pos)
    ax_pp.set_xticklabels(KET, fontsize=6.4)
    ax_pp.tick_params(axis="y", labelsize=6.4)
    ax_pp.set_ylabel("population (%)", fontsize=7.4)
    ax_pp.set_ylim(0.0, 60.0)
    ax_pp.legend(fontsize=6.4, frameon=False, loc="center", ncol=1,
                 bbox_to_anchor=(0.5, 0.62), handlelength=1.4)
    ax_pp.set_title("Diagonal populations", fontsize=8.5, fontweight="bold",
                    color=INK)

    fig.tight_layout(rect=[0, 0.155, 1, 0.945])

    bb_a, bb_b = ax_re.get_position(), ax_im.get_position()
    cax = fig.add_axes([bb_a.x0 + 0.05, 0.118,
                        bb_b.x1 - bb_a.x0 - 0.10, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIV), cax=cax,
                      orientation="horizontal")
    cb.set_label(r"matrix element $\rho_{ij}$ (-)", fontsize=6.8, color=INK)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.4, width=0.7, length=2.5, color="#333333")


    panel_labels_inches(fig, [ax_re, ax_im], labels=["(a)", "(b)"],
                        dx=-0.02, dy=-0.10)
    panel_labels_inches(fig, [ax_df], labels=["(c)"], dx=-0.46, dy=0.16)
    panel_labels_inches(fig, [ax_pp], labels=["(d)"], dx=-0.26, dy=0.16)
    savefig(fig, "494_qc_state_tomography.png")


if __name__ == "__main__":
    main()

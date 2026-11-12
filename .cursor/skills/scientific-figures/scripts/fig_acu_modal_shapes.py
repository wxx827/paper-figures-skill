"""First six bending modes of a simply supported steel panel.

Each tile shows the mode shape weighted by how strongly one fixed drive point
excites it, so the six panels really do differ in amplitude and a single
symmetric displacement scale carries information instead of being decorative.
Node lines are the zero contour of the shape; the natural frequency and modal
damping ratio on every tile come from the plate properties, not from typing.
"""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, get_cmap, letter_panels, COLORS
from figkit import shared_norm, halo_text

A, B, H = 1.20, 0.80, 0.004          # plate size (m) and thickness (m)
E, NU, RHO = 2.05e11, 0.30, 7850.0   # steel
MODES = [(1, 1), (2, 1), (1, 2), (3, 1), (2, 2), (3, 2)]
DRIVE = (0.27 * A, 0.31 * B)         # excitation point (m)
NG = 220


def natural_freq(m, n):
    """Simply supported plate natural frequency (Hz)."""
    d = E * H ** 3 / (12.0 * (1.0 - NU ** 2))
    return (np.pi / 2.0) * np.sqrt(d / (RHO * H)) * \
        ((m / A) ** 2 + (n / B) ** 2)


def damping_ratio(f):
    """Structural + air-radiation damping model, in fraction of critical."""
    return 0.0045 + 1.10 / f + 8.0e-6 * f


def main():
    set_pub_style()

    # interior grid so the clamped edges are not part of the node contour
    x = np.linspace(0.004 * A, 0.996 * A, NG)
    y = np.linspace(0.004 * B, 0.996 * B, NG)
    X, Y = np.meshgrid(x, y)

    f0 = natural_freq(*MODES[0])
    shapes, freqs, zetas = [], [], []
    for m, n in MODES:
        fn = natural_freq(m, n)
        phi = np.sin(m * np.pi * X / A) * np.sin(n * np.pi * Y / B)
        phi_d = (np.sin(m * np.pi * DRIVE[0] / A) *
                 np.sin(n * np.pi * DRIVE[1] / B))
        # drive-point participation with a gentle inertial roll-off
        shapes.append(phi * phi_d / (fn / f0) ** 0.35)
        freqs.append(fn)
        zetas.append(damping_ratio(fn))

    scale = 1.0 / max(abs(s).max() for s in shapes)       # -> mm, peak = 1 mm
    shapes = [s * scale for s in shapes]
    norm = shared_norm(shapes, symmetric=True)
    cmap = get_cmap("cb_diverging")
    ext = [0.0, A, 0.0, B]

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.4))
    for k, (ax, (m, n)) in enumerate(zip(axes.ravel(), MODES)):
        im = ax.imshow(shapes[k], origin="lower", extent=ext, cmap=cmap,
                       norm=norm, interpolation="bilinear", aspect="equal")
        ax.contour(X, Y, shapes[k], levels=[0.0], colors="white",
                   linewidths=1.0, linestyles=[(0, (4.0, 2.0))])
        ax.plot([DRIVE[0]], [DRIVE[1]], marker="*", ms=7.5, mfc="white",
                mec=COLORS["ink"], mew=0.7, zorder=5)
        peak = float(np.abs(shapes[k]).max())
        halo_text(ax, 0.028, 0.045, f"$\\zeta$ = {zetas[k] * 100:.2f} %   "
                  f"peak {peak:.2f} mm", transform=ax.transAxes, ha="left",
                  va="bottom", fontsize=6.6, color=COLORS["ink"],
                  halo="white", halo_lw=2.0)
        ax.set_title(f"Mode {k + 1}  ({m},{n})   $f_n$ = {freqs[k]:.0f} Hz",
                     fontsize=8.5, fontweight="bold")
        ax.set_xticks([0.0, 0.4, 0.8, 1.2])
        ax.set_yticks([0.0, 0.4, 0.8])
        ax.tick_params(labelsize=7.0, length=2.5)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.7)
            s.set_color(COLORS["rule"])
        if k // 3 == 1:
            ax.set_xlabel("Plate x (m)", fontsize=8.0)
        else:
            ax.tick_params(labelbottom=False)
        if k % 3 == 0:
            ax.set_ylabel("Plate y (m)", fontsize=8.0)
        else:
            ax.tick_params(labelleft=False)

    fig.tight_layout(rect=[0.0, 0.035, 0.885, 0.935], h_pad=1.2, w_pad=2.0)
    cax = fig.add_axes([0.900, 0.22, 0.014, 0.52])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("modal displacement (mm, shared scale)", fontsize=8.0,
                 color=COLORS["ink"])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(COLORS["rule"])
    cb.ax.tick_params(labelsize=7.0, width=0.7, length=2.5,
                      color=COLORS["rule"])
    letter_panels(axes, x=-0.10, y=1.05)

    peaks = [float(np.abs(s).max()) for s in shapes]
    loud, quiet = int(np.argmax(peaks)), int(np.argmin(peaks))
    savefig(fig, "416_acu_modal_shapes.png")


if __name__ == "__main__":
    main()

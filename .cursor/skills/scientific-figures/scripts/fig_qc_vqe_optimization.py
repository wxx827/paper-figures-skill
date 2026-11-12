"""VQE convergence: five restarts, the landscape they walk, optimiser cost.

Chemical accuracy is 1.6 mHa on an energy axis that spans half a Hartree, so
it cannot be drawn as a band on the main axes - the inset plots the same five
traces as an absolute error on a log axis, which is the only honest way to
show whether a run actually got there.  The restart that lands in the local
minimum is left in on purpose: it is what makes the initial-point scatter
worth plotting.  The landscape trajectories are real finite-difference
gradient descents on the surface being drawn, and the optimiser bar chart
counts iterations from the traces rather than quoting them.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, get_cmap, get_palette, savefig
from figkit import panel_labels_inches, safe_colorbar

RED = "#b23a48"         # the single emphasis: the best restart
BLUE = "#3b5b92"
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
SEQ = get_cmap("cb_sequential_blue", reverse=True)

E0 = -1.1373            # exact ground-state energy, Hartree
CHEM_ACC = 1.6e-3       # Hartree
N_ITER = 220
# (starting energy, time constant, offset of the attractor above E0)
RESTARTS = [(-0.62, 18.0, 0.0), (-0.78, 26.0, 0.0), (-0.55, 22.0, 0.0),
            (-0.71, 34.0, 0.052), (-0.66, 15.0, 0.0)]
OPTIMISERS = [("L-BFGS-B", 12.0, 2.0e-4), ("COBYLA", 26.0, 3.0e-4),
              ("SPSA", 19.0, 4.5e-4), ("Adam", 33.0, 2.5e-4)]


def trace(e_init, tau, offset, noise, rng):
    it = np.arange(N_ITER + 1, dtype=float)
    target = E0 + offset
    base = target + (e_init - target) * np.exp(-it / tau)
    jitter = (rng.normal(0.0, 6.0e-3, it.size) * np.exp(-it / 38.0) +
              rng.normal(0.0, noise, it.size))
    return it, base + jitter


def settle_step(err, tol=CHEM_ACC):
    """First iteration after which the error never leaves the tolerance."""
    bad = np.flatnonzero(err >= tol)
    if bad.size == 0:
        return 0
    step = int(bad.max()) + 1
    return step if step < err.size else None


def landscape(n=140):
    g = np.linspace(-np.pi, np.pi, n)
    T1, T2 = np.meshgrid(g, g, indexing="xy")
    return g, energy(T1, T2)


def energy(t1, t2):
    return (E0 + 0.62 * (1.0 - np.cos(t1 - 0.4) * np.cos(t2 - 0.9)) +
            0.12 * np.sin(2.0 * t1) * np.sin(2.0 * t2))


def descend(start, lr=0.22, steps=70, h=1e-3):
    p = np.array(start, dtype=float)
    path = [p.copy()]
    for _ in range(steps):
        gx = (energy(p[0] + h, p[1]) - energy(p[0] - h, p[1])) / (2.0 * h)
        gy = (energy(p[0], p[1] + h) - energy(p[0], p[1] - h)) / (2.0 * h)
        p = p - lr * np.array([gx, gy])
        path.append(p.copy())
    return np.asarray(path)


def main_panel(ax, it, curves):
    shades = get_palette("cb_sequential_blue", 7)[2:6]
    finals = np.array([c[-20:].mean() for c in curves])
    best = int(np.argmin(finals))

    order = [k for k in range(len(curves)) if k != best]
    for k, color in zip(order, shades):
        ax.plot(it, curves[k], color=color, lw=1.1, alpha=0.95,
                label=f"restart {k + 1}")
    ax.plot(it, curves[best], color=RED, lw=1.9, zorder=5,
            label=f"restart {best + 1} (best)")

    ax.axhline(E0, color=GREY, lw=1.0, ls=(0, (4, 2.6)), zorder=2)
    ax.text(N_ITER - 3.0, E0 - 0.028,
            f"exact ground state $E_0$ = {E0:.4f} Ha", ha="right", va="top",
            fontsize=7.0, color=SUB)

    step = settle_step(np.abs(curves[best] - E0))
    ax.plot([step], [curves[best][step]], marker="o", ms=4.0, color=RED,
            zorder=6)

    stuck = int(np.argmax(finals))
    ax.annotate(f"local minimum, {(finals[stuck] - E0) * 1e3:.0f} mHa above "
                f"$E_0$", xy=(N_ITER - 25.0, finals[stuck]),
                xytext=(N_ITER - 35.0, finals[stuck] + 0.055), ha="right",
                va="bottom", fontsize=6.8, color=SUB,
                arrowprops=dict(arrowstyle="-", color=SUB, lw=0.7))
    n_ok = sum(settle_step(np.abs(c - E0)) is not None for c in curves)

    ax.set_xlabel("optimisation iteration (-)")
    ax.set_ylabel("energy $E$ (Hartree)")
    ax.set_xlim(0.0, N_ITER)
    ax.set_ylim(-1.22, -0.48)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.8, frameon=False, loc="upper right", ncol=2,
              handlelength=1.4, labelspacing=0.3, columnspacing=1.0)
    ax.set_title(f"{n_ok} of {len(curves)} restarts reach chemical accuracy",
                 fontsize=9.0, fontweight="bold", color=INK)
    return best, step


def error_inset(ax, it, curves, best, step):
    axi = ax.inset_axes([0.48, 0.40, 0.48, 0.44])
    axi.set_zorder(7)
    axi.patch.set_visible(True)
    shades = get_palette("cb_sequential_blue", 7)[2:6]
    order = [k for k in range(len(curves)) if k != best]
    for k, color in zip(order, shades):
        axi.semilogy(it, np.abs(curves[k] - E0), color=color, lw=0.8)
    axi.semilogy(it, np.abs(curves[best] - E0), color=RED, lw=1.3)
    axi.axhline(CHEM_ACC, color=GREY, lw=0.9, ls=(0, (3, 2.2)))
    axi.text(4.0, CHEM_ACC * 1.55,
             f"chemical accuracy {CHEM_ACC * 1e3:.1f} mHa", ha="left",
             va="bottom", fontsize=6.2, color=SUB)
    if step is not None:
        axi.axvline(step, color=RED, lw=0.8, ls=(0, (2.2, 1.8)))
        axi.text(step + 5.0, 0.35, f"best restart\nsettles at {step}",
                 ha="left", va="top", fontsize=6.2, color=RED,
                 linespacing=1.3)
    axi.set_xlim(0.0, N_ITER)
    axi.set_ylim(1e-4, 1.0)
    axi.set_xlabel("iteration (-)", fontsize=6.4, labelpad=1.5)
    axi.set_ylabel(r"$|E - E_0|$ (Ha)", fontsize=6.4, labelpad=1.5)
    axi.tick_params(labelsize=6.0, length=2.0)
    for s in ("top", "right"):
        axi.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axi.spines[s].set_linewidth(0.8)
        axi.spines[s].set_color("#333333")
    return axi


def landscape_panel(fig, ax, starts):
    g, z = landscape()
    # Reversed ramp so the deepest basin is the darkest patch; with the
    # default direction the global minimum renders as the palest area and
    # every reader misreads it as a barrier.
    cf = ax.contourf(g, g, z, levels=14, cmap=SEQ)
    ax.contour(g, g, z, levels=14, colors=["#ffffff"], linewidths=0.35,
               alpha=0.5)

    iy, ix = np.unravel_index(np.argmin(z), z.shape)
    gmin = np.array([g[ix], g[iy]])
    paths = [descend(s) for s in starts]
    hit = int(np.argmin([np.hypot(*(p[-1] - gmin)) for p in paths]))
    for k, path in enumerate(paths):
        color = RED if k == hit else "#f2f2f2"
        ax.plot(path[:, 0], path[:, 1], color=color, lw=1.4, zorder=5)
        ax.plot([path[0, 0]], [path[0, 1]], marker="o", ms=3.6, color=color,
                mec="#333333", mew=0.5, zorder=6)
        ax.plot([path[-1, 0]], [path[-1, 1]], marker="s", ms=3.0,
                color=color, mec="#333333", mew=0.5, zorder=6)
    ax.plot([gmin[0]], [gmin[1]], marker="*", ms=12.0, color="#c9a227",
            mec="#333333", mew=0.7, zorder=7)
    ax.set_xlabel(r"$\theta_1$ (rad)", fontsize=7.4)
    ax.set_ylabel(r"$\theta_2$ (rad)", fontsize=7.4)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xticks([-3, -1.5, 0, 1.5, 3])
    ax.set_yticks([-3, -1.5, 0, 1.5, 3])
    ax.tick_params(labelsize=6.5)
    ax.set_title("Two-parameter landscape", fontsize=8.2, fontweight="bold",
                 color=INK)
    safe_colorbar(fig, cf, ax, label="$E$ (Ha)", width=0.09, pad=0.06,
                  labelsize=6.5)


def optimiser_panel(ax, steps):
    names = [n for n, _ in steps]
    vals = np.array([v for _, v in steps], dtype=float)
    order = np.argsort(vals)[::-1]
    best = int(np.argmin(vals))
    colors = [RED if i == best else BLUE for i in order]
    y = np.arange(len(names))
    ax.barh(y, vals[order], height=0.62, color=colors, edgecolor="white",
            linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=6.8)
    for yy, v in zip(y, vals[order]):
        ax.text(v + 3.0, yy, f"{v:.0f}", va="center", ha="left",
                fontsize=6.5, color=INK)
    ax.set_xlabel("iterations to chemical accuracy (-)", fontsize=7.4)
    ax.set_xlim(0.0, vals.max() * 1.28)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.7)
    ax.set_title("Optimiser cost", fontsize=8.2, fontweight="bold",
                 color=INK)


def main():
    set_pub_style()
    rng = np.random.default_rng(3)
    it, curves = None, []
    for e_init, tau, off in RESTARTS:
        it, y = trace(e_init, tau, off, 6.0e-4, rng)
        curves.append(y)

    steps = []
    for name, tau, noise in OPTIMISERS:
        _, y = trace(-0.68, tau, 0.0, noise, rng)
        s = settle_step(np.abs(y - E0))
        steps.append((name, N_ITER if s is None else s))

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.72], wspace=0.30, hspace=0.60)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_land = fig.add_subplot(gs[0, 1])
    ax_opt = fig.add_subplot(gs[1, 1])

    best, step = main_panel(ax_main, it, curves)
    error_inset(ax_main, it, curves, best, step)
    landscape_panel(fig, ax_land, [(-1.7, 2.5), (2.5, -0.6), (-2.6, -2.3)])
    optimiser_panel(ax_opt, steps)

    fig.tight_layout(rect=[0, 0.035, 1, 0.945], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_main, ax_land, ax_opt], dx=-0.42, dy=0.12)
    savefig(fig, "497_qc_vqe_optimization.png")


if __name__ == "__main__":
    main()

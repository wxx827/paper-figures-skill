"""Eight snapshots of an SIR cascade on a random geometric network.

The substrate is a random geometric graph plus a handful of long-range
shortcuts, so the contagion has a visible wavefront *and* the occasional jump
that seeds a second front - the thing that makes small-world contact networks
worth drawing at all.  Node positions are the generating points, so all eight
panels share one embedding for free.

Susceptible, infected and removed keep the same three colours everywhere;
edges that actually carried a transmission stay highlighted for the rest of
the run, which turns the last panel into the infection tree.  The full-width
strip underneath is the compartment curve, with the eight snapshot times
marked on it, so a reader can locate any panel on the epidemic.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from pubstyle import COLORS, savefig, set_pub_style
from figkit import panel_labels_inches

SEED = 7
N_NODES = 340
RADIUS = 0.088
N_SHORTCUT = 10
BETA, GAMMA = 0.26, 0.085
N_SNAPS = 8

C_SUS = "#cccccc"
C_INF = "#b23a48"          # house red, the single emphasis colour
C_REM = "#3b5b92"          # house blue


def geometric_graph(n, radius, n_shortcut, seed):
    """Random geometric graph on the unit square + long-range shortcuts."""
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 2))
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    A = ((d < radius) & (d > 0)).astype(float)
    for _ in range(n_shortcut):
        u, v = rng.integers(n, size=2)
        if u != v:
            A[u, v] = A[v, u] = 1.0
    return pos, A


def sir(A, seed_node, beta, gamma, seed, n_steps=90):
    """Discrete-time SIR; records state and the edge each infection used."""
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    state = np.zeros(n, dtype=int)                 # 0 S, 1 I, 2 R
    state[seed_node] = 1
    history, links = [state.copy()], [[]]
    for _ in range(n_steps):
        inf = np.flatnonzero(state == 1)
        new_links = []
        newly = set()
        for v in inf:
            for w in np.flatnonzero((A[v] > 0) & (state == 0)):
                if w not in newly and rng.random() < beta:
                    newly.add(int(w))
                    new_links.append((int(v), int(w)))
        recover = inf[rng.random(inf.size) < gamma]
        state = state.copy()
        state[recover] = 2
        state[list(newly)] = 1
        history.append(state.copy())
        links.append(new_links)
        if not (state == 1).any():
            break
    return history, links


def main():
    set_pub_style(base_font=8)

    pos, A = geometric_graph(N_NODES, RADIUS, N_SHORTCUT, SEED)
    # start from a well-connected node near the left edge so the wavefront
    # has room to sweep across the panel
    corner = np.argmin(pos[:, 0] + np.abs(pos[:, 1] - 0.5))
    history, links = sir(A, corner, BETA, GAMMA, SEED)
    T = len(history) - 1

    frac_s = np.array([(h == 0).mean() for h in history])
    frac_i = np.array([(h == 1).mean() for h in history])
    frac_r = np.array([(h == 2).mean() for h in history])
    peak_t = int(np.argmax(frac_i))

    # Snapshots span the *active* epidemic rather than the whole simulation:
    # an SIR run spends most of its wall clock in a finished state, and eight
    # frames evenly spaced over T would spend four of them on the same
    # all-removed picture.
    active = np.flatnonzero(frac_i > 0.02)
    t_end = int(active[-1]) if active.size else T
    snaps = np.round(np.linspace(0, t_end, N_SNAPS)).astype(int)

    iu, ju = np.triu_indices_from(A, k=1)
    edges = list(zip(iu[A[iu, ju] > 0], ju[A[iu, ju] > 0]))

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.0, 1.0, 0.62],
                  hspace=0.30, wspace=0.10)
    axes = [fig.add_subplot(gs[r, c]) for r in (0, 1) for c in range(4)]
    ax_curve = fig.add_subplot(gs[2, :])

    for ax, t in zip(axes, snaps):
        state = history[t]
        for i, j in edges:
            ax.plot(pos[[i, j], 0], pos[[i, j], 1], color="#e2e2e2", lw=0.35,
                    zorder=1, solid_capstyle="round")
        for i, j in [e for step in links[:t + 1] for e in step]:
            ax.plot(pos[[i, j], 0], pos[[i, j], 1], color=C_INF, lw=0.9,
                    alpha=0.75, zorder=2, solid_capstyle="round")
        for code, col, z in ((0, C_SUS, 3), (2, C_REM, 4), (1, C_INF, 5)):
            m = state == code
            ax.scatter(pos[m, 0], pos[m, 1], s=8.5, color=col,
                       edgecolor="white", linewidth=0.3, zorder=z)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"t = {t}", fontsize=8.2, pad=2.5, color=COLORS["ink"],
                     fontweight="bold")
        ax.text(0.5, -0.015, f"infected {100 * (state == 1).mean():.0f} %  ·  "
                f"removed {100 * (state == 2).mean():.0f} %",
                transform=ax.transAxes, ha="center", va="top", fontsize=6.8,
                color=COLORS["sub"])

    # ---- compartment curve, full width ----------------------------------
    tt = np.arange(T + 1)
    ax_curve.fill_between(tt, 0, frac_r, color=C_REM, alpha=0.20, lw=0)
    for y, col, lab in ((frac_s, C_SUS, "susceptible"),
                        (frac_i, C_INF, "infected"),
                        (frac_r, C_REM, "removed")):
        ax_curve.plot(tt, y, color=col, lw=1.7, label=lab)
    for t in snaps:
        ax_curve.axvline(t, color=COLORS["hairline"], lw=0.7, ls=":",
                         zorder=0)
        ax_curve.plot([t], [frac_i[t]], marker="o", ms=3.4, color=C_INF,
                      mec="white", mew=0.6, zorder=5)
    ax_curve.annotate(f"peak {100 * frac_i[peak_t]:.0f} % infected at t = "
                      f"{peak_t}", xy=(peak_t, frac_i[peak_t]),
                      xytext=(10, 14), textcoords="offset points",
                      fontsize=7.0, color=COLORS["ink"],
                      arrowprops=dict(arrowstyle="-", lw=0.7,
                                      color=COLORS["rule"]))
    ax_curve.set_xlim(0, T)
    ax_curve.set_ylim(0, 1.05)
    ax_curve.set_xlabel("time step (-)", fontsize=8.0)
    ax_curve.set_ylabel("fraction of nodes (-)", fontsize=8.0)
    ax_curve.tick_params(labelsize=7.0)
    ax_curve.set_title(f"Final attack rate {100 * frac_r[-1]:.0f} % of "
                       f"{N_NODES} nodes", fontsize=8.5, loc="left", pad=4)

    handles = [Line2D([], [], marker="o", ls="none", ms=4.4, mfc=c,
                      mec="white", mew=0.5, label=l)
               for c, l in ((C_SUS, "susceptible"), (C_INF, "infected"),
                            (C_REM, "removed"))]
    handles.append(Line2D([], [], color=C_INF, lw=1.2,
                          label="link that carried a transmission"))

    fig.tight_layout(rect=[0, 0.01, 1, 0.88], h_pad=2.2, w_pad=1.0)
    panel_labels_inches(fig, axes, dx=0.00, dy=0.14)
    panel_labels_inches(fig, [ax_curve], labels=["(i)"], dx=-0.42, dy=0.10)

    savefig(fig, "368_net_diffusion.png")


if __name__ == "__main__":
    main()

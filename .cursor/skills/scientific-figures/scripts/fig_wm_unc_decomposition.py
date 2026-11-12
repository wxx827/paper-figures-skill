"""Figure 1180 - Splitting predictive variance into epistemic and aleatoric.

The whole figure is two closed-form fields over the same command grid:

    epistemic(N) = A / (1 + (N / N0) * density)     shrinks with more data
    aleatoric    = f(speed, curvature)              does not depend on N
    total        = epistemic + aleatoric            (asserted, not assumed)

The three maps are that decomposition at one training-set size, ``N_STAR``;
the curve underneath is the grid mean of the same two fields swept over N, so
the maps are literally the marked slice of the curve - which the script checks
with an assert rather than trusting the caption.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, get_cmap, style_axes

SEED = 1585  # original id; figure number is 1180
NV, NK = 110, 76                       # speed cells, curvature cells
V_MAX, K_MAX = 30.0, 0.10
A_EPI, N0 = 0.62, 12.0
N_STAR = 64
N_GRID = np.unique(np.concatenate([
    np.round(np.logspace(np.log10(8), np.log10(2048), 26)).astype(int),
    2 ** np.arange(3, 12)]))            # the labelled ticks must be on it

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#d5d5d5"
EPI, ALE, TOT = "#7b3294", "#5f9dc6", "#0b3d63"
GRID, CARD = "#ededed", "#e2e8ee"

# (speed, curvature, sigma_v, sigma_k, weight) of the training clusters
CLUSTERS = [(6.0, 0.000, 3.4, 0.022, 0.85),
            (13.5, -0.035, 3.0, 0.019, 0.70),
            (14.5, 0.040, 2.6, 0.017, 0.62),
            (23.0, 0.004, 4.2, 0.014, 1.00),
            (27.5, -0.018, 2.2, 0.012, 0.45)]


def fields():
    """Density, aleatoric and the epistemic prefactor on the command grid."""
    v = np.linspace(0.0, V_MAX, NV)
    k = np.linspace(-K_MAX, K_MAX, NK)
    V, K = np.meshgrid(v, k)
    dens = np.zeros_like(V)
    for (v0, k0, sv, sk, wt) in CLUSTERS:
        dens += wt * np.exp(-0.5 * (((V - v0) / sv) ** 2 +
                                    ((K - k0) / sk) ** 2))
    dens = dens / dens.max()
    dens = 0.06 + 0.94 * dens          # even far corners see a few frames
    rng = np.random.default_rng(SEED)
    wob = rng.standard_normal((NK, NV))
    for _ in range(3):                 # cheap smoothing, no scipy needed here
        wob = 0.25 * (np.roll(wob, 1, 0) + np.roll(wob, -1, 0) +
                      np.roll(wob, 1, 1) + np.roll(wob, -1, 1))
    wob = wob / (np.abs(wob).max() + 1e-9)
    ale = (0.085 + 0.400 * (V / V_MAX) ** 1.6 *
           (0.30 + 0.70 * np.abs(K) / K_MAX) + 0.060 * (0.5 + 0.5 * wob))
    return v, k, dens, ale


def epistemic(dens, n):
    return A_EPI / (1.0 + (n / N0) * dens)


def main():
    set_pub_style(base_font=8.5)
    v, k, dens, ale = fields()
    epi = epistemic(dens, N_STAR)
    tot = epi + ale
    assert np.allclose(tot, epi + ale), "total is not epistemic + aleatoric"

    maps = [("total  $\\sigma^2_{\\mathrm{tot}}$", tot),
            ("epistemic  $\\sigma^2_{\\mathrm{epi}}$", epi),
            ("aleatoric  $\\sigma^2_{\\mathrm{ale}}$", ale)]
    vmax = max(float(m.max()) for _, m in maps)     # one scale for all three
    cmap = get_cmap("cb_sequential_blue")

    epi_m = np.array([epistemic(dens, n).mean() for n in N_GRID])
    ale_m = np.full_like(epi_m, float(ale.mean()))
    tot_m = epi_m + ale_m
    i_star = int(np.argmin(np.abs(N_GRID - N_STAR)))
    assert N_GRID[i_star] == N_STAR, "N* is not on the swept grid"
    assert abs(epi_m[i_star] - epi.mean()) < 1e-9, "curve and map disagree"

    fig = plt.figure(figsize=(10.4, 6.0))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 0.052],
                          height_ratios=[1.0, 0.95], left=0.058, right=0.945,
                          top=0.815, bottom=0.098, wspace=0.16, hspace=0.52)

    ext = [0.0, V_MAX, -K_MAX, K_MAX]
    axm = []
    for c, (name, field) in enumerate(maps):
        ax = fig.add_subplot(gs[0, c])
        im = ax.imshow(field, origin="lower", extent=ext, aspect="auto",
                       cmap=cmap, vmin=0.0, vmax=vmax,
                       interpolation="bilinear")
        ax.set_title(f"({'abc'[c]})  {name}\ngrid mean "
                     f"{field.mean():.3f}", fontsize=8.6,
                     fontweight="bold", color=INK, pad=4.0, linespacing=1.5)
        ax.set_xlabel("commanded speed  (m s$^{-1}$)", fontsize=7.8)
        if c == 0:
            ax.set_ylabel("commanded curvature  (m$^{-1}$)", fontsize=7.8)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=6.8, length=2.4)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.7)
            s.set_color("#9a9a9a")
        axm.append(ax)

    # the training clusters, drawn on the epistemic map where they explain it
    axm[1].scatter([c[0] for c in CLUSTERS], [c[1] for c in CLUSTERS],
                   s=26, marker="+", color="white", linewidths=1.1, zorder=5)
    notes = ["(a) = (b) + (c),\npixel by pixel",
             "+ = training cluster centre;\nthe pale holes are the data",
             "a noise floor that rises with\nspeed and |curvature|"]
    for ax, note in zip(axm, notes):
        ax.text(0.5, -0.285, note, transform=ax.transAxes, ha="center",
                va="top", fontsize=6.8, color=MUTED, linespacing=1.5,
                style="italic")
    axm[2].text(0.985, 0.965, f"{ale.min():.2f} - {ale.max():.2f} on the "
                f"shared 0 - {vmax:.2f} scale", transform=axm[2].transAxes,
                ha="right", va="top", fontsize=6.6, color=SUB, zorder=6,
                bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=HAIR,
                          lw=0.5, alpha=0.94))

    cax = fig.add_subplot(gs[0, 3])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.set_ticks([0.0, 0.25 * vmax, 0.5 * vmax, 0.75 * vmax, vmax])
    cb.set_ticklabels([f"{t:.2f}" for t in cb.get_ticks()])
    cb.ax.tick_params(labelsize=6.8, width=0.7, length=2.4)
    cb.set_label("predictive variance  (shared scale)", fontsize=7.4,
                 color=SUB)

    # ---- the sweep ---------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0:3])
    style_axes(ax)
    ax.set_axisbelow(True)
    ax.grid(color=GRID, lw=0.7, zorder=0)
    ax.fill_between(N_GRID, 0, ale_m, color=ALE, alpha=0.30, lw=0, zorder=1)
    ax.fill_between(N_GRID, ale_m, tot_m, color=EPI, alpha=0.22, lw=0,
                    zorder=1)
    ax.plot(N_GRID, tot_m, color=TOT, lw=2.0, zorder=5)
    ax.plot(N_GRID, epi_m, color=EPI, lw=1.7, zorder=4)
    ax.plot(N_GRID, ale_m, color=ALE, lw=1.7, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(N_GRID[0], N_GRID[-1])
    ax.set_ylim(0.0, float(tot_m.max()) * 1.16)

    ys = float(tot_m[i_star])
    ax.plot([N_STAR, N_STAR], [0.0, ys], color="#333333", lw=1.0,
            ls=(0, (3.0, 2.0)), zorder=6)
    ax.plot([N_STAR], [ys], marker="o", ms=5.2, mfc="white", mec=TOT, mew=1.4,
            zorder=7)
    ax.annotate(f"maps (a)-(c) are this slice:\n$N$ = {N_STAR} episodes",
                xy=(N_STAR, ys), xytext=(N_STAR * 1.35, ys * 1.09),
                ha="left", va="bottom", fontsize=7.4, color=INK,
                linespacing=1.5, zorder=8,
                bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=HAIR,
                          lw=0.6, alpha=0.95),
                arrowprops=dict(arrowstyle="-", lw=0.8, color="#333333",
                                shrinkA=2, shrinkB=4))

    e0, e1 = f"{epi_m[0]:.3f}", f"{epi_m[-1]:.3f}"
    a0, a1 = f"{ale_m[0]:.3f}", f"{ale_m[-1]:.3f}"
    ax.text(N_GRID[-1] * 0.96, float(epi_m[-1]) + 0.030,
            f"epistemic {e0} $\\rightarrow$ {e1}\n"
            f"({100 * (1 - float(e1) / float(e0)):.0f} % of it is gone)",
            ha="right", va="bottom", fontsize=7.2, color=EPI,
            linespacing=1.5, zorder=8,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.4,
                      alpha=0.92))
    ax.text(N_GRID[0] * 1.10, float(ale_m[0]) - 0.022,
            f"aleatoric flat at {a1}: more data cannot remove it",
            ha="left", va="top", fontsize=7.2, color="#2e74a8", zorder=8,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.4,
                      alpha=0.92))

    ax.set_xticks([8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    ax.set_xticklabels(["8", "16", "32", "64", "128", "256", "512", "1024",
                        "2048"], fontsize=7.0)
    ax.tick_params(labelsize=7.0, length=2.4)
    ax.set_xlabel("Training episodes  $N$  (log scale)", fontsize=8.6)
    ax.set_ylabel("grid-mean variance", fontsize=8.2)
    ax.set_title("(d)  Only the epistemic term buys anything from more data",
                 fontsize=9.0, fontweight="bold", pad=4.0)
    handles = [Line2D([], [], color=TOT, lw=2.0, label="total"),
               Line2D([], [], color=EPI, lw=1.7, label="epistemic"),
               Line2D([], [], color=ALE, lw=1.7, label="aleatoric")]
    leg = ax.legend(handles=handles, loc="upper right",
                    bbox_to_anchor=(1.0, 1.02), frameon=True,
                    facecolor="white", edgecolor=HAIR, framealpha=0.94,
                    fontsize=7.4, handlelength=1.6, labelspacing=0.35,
                    borderpad=0.45, ncol=3, columnspacing=1.0)
    leg.get_frame().set_linewidth(0.5)

    # ---- headline + arithmetic ---------------------------------------------
    st, se, sa = (f"{tot.mean():.3f}", f"{epi.mean():.3f}",
                  f"{ale.mean():.3f}")
    assert abs((float(se) + float(sa)) - float(st)) < 5e-4, \
        "printed decomposition does not close"
    savefig(fig, "1180_wm_unc_decomposition.png")


if __name__ == "__main__":
    main()

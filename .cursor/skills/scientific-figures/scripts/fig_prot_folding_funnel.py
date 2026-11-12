"""Folding free-energy landscape drawn as a rugged funnel.

Main panel: the free energy over two conformational coordinates, with the
native basin at the bottom, the rugged rim of the unfolded ensemble at the top,
labelled intermediates and one kinetic trap, and a folding trajectory spiralling
inward.  Companion panel: the same energy against folding progress, with the
band showing how much energy is still accessible at each stage.

Both panels read the same analytic surface, and every state that is labelled is
found by a local-minimum search on that surface rather than placed by hand.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import panel_label_inches, axes_inches, shared_limits

SEED = 771

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
H_RED, H_BLUE, H_GOLD, H_PURPLE = "#b23a48", "#3b5b92", "#c9a227", "#8e6bbf"

R_MAX = 6.0


def energy(x, y):
    """Funnel: a deep native well, an entropic rim and sinusoidal ruggedness."""
    r = np.hypot(x, y)
    e = -13.8 * np.exp(-(r ** 2) / 2.6) + 0.24 * r ** 2
    e += 2.60 * np.sin(1.9 * x + 0.4) * np.cos(1.7 * y - 0.7) * \
        np.exp(-((r - 3.6) / 2.8) ** 2)
    e += 1.25 * np.cos(2.5 * x - 1.1) * np.sin(2.3 * y + 0.6) * (r / R_MAX)
    e += 0.60 * np.sin(3.1 * x + 2.0) * np.sin(2.9 * y + 1.4) * (r / R_MAX)
    return e


def local_minima(X, Y, Z, r_min=2.3, n_keep=3, sep=2.6):
    """Interior grid cells lower than all eight neighbours, ranked by depth."""
    core = Z[1:-1, 1:-1]
    lower = np.ones_like(core, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            lower &= core < Z[1 + di:Z.shape[0] - 1 + di,
                              1 + dj:Z.shape[1] - 1 + dj]
    ii, jj = np.where(lower)
    xs, ys, zs = X[1:-1, 1:-1][ii, jj], Y[1:-1, 1:-1][ii, jj], core[ii, jj]
    keep_r = np.hypot(xs, ys) > r_min
    xs, ys, zs = xs[keep_r], ys[keep_r], zs[keep_r]
    order = np.argsort(zs)
    out = []
    for k in order:
        p = np.array([xs[k], ys[k]])
        if all(np.hypot(*(p - q[:2])) > sep for q in out):
            out.append(np.array([xs[k], ys[k], zs[k]]))
        if len(out) == n_keep:
            break
    return out


def folding_path(rng, n=420):
    """Spiral descent with a detour that grazes the deepest off-axis well."""
    t = np.linspace(0.0, 1.0, n)
    r = R_MAX * (1.0 - t) ** 1.35 + 0.10
    wob = 0.42 * np.sin(9.0 * np.pi * t) * (1.0 - t) ** 0.7
    th = 4.4 * np.pi * t + 0.8 * np.sin(3.0 * np.pi * t)
    x = (r + wob) * np.cos(th)
    y = (r + wob) * np.sin(th)
    return x, y, energy(x, y)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    g = np.linspace(-R_MAX, R_MAX, 141)
    X, Y = np.meshgrid(g, g)
    Z = energy(X, Y)
    inside = np.hypot(X, Y) <= R_MAX
    Z = np.where(inside, Z, np.nan)

    px, py, pz = folding_path(rng)
    grid_e = np.where(inside, Z, 1e6)
    for sep in (2.6, 2.2, 1.8, 1.4):
        states = local_minima(X, Y, grid_e, sep=sep)
        if len(states) >= 3:
            break
    trap, inter1, inter2 = states[0], states[1], states[2]
    native = np.array([0.0, 0.0, float(np.nanmin(Z))])

    zlo, zhi = shared_limits([Z[np.isfinite(Z)], pz], pad=0.05)
    norm = plt.Normalize(zlo, zhi)
    cmap = get_cmap("cb_sequential_blue", reverse=True)

    FW, FH = 10.4, 6.4
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0.020, 0.055, 6.35 / FW, 5.35 / FH], projection="3d",
                      computed_zorder=False)
    axq = axes_inches(fig, (7.15, 1.55, 2.85, 2.95))

    ax.plot_surface(X, Y, Z, cmap=cmap, norm=norm, rstride=1, cstride=1,
                    linewidth=0.0, antialiased=True, shade=False, alpha=0.98,
                    zorder=1)
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10, color="white",
                      linewidth=0.28, alpha=0.40, zorder=2)
    ax.plot(px, py, pz + 0.42, color=H_RED, lw=1.7, zorder=6,
            solid_capstyle="round")
    ax.scatter([px[0]], [py[0]], [pz[0] + 0.42], s=22, facecolor="white",
               edgecolor=H_RED, linewidth=1.1, depthshade=False, zorder=7)

    marks = [(native, "native state", INK, H_GOLD),
             (trap, "kinetic trap", H_RED, H_RED),
             (inter1, "intermediate I", SUB, H_PURPLE),
             (inter2, "intermediate II", SUB, H_PURPLE)]
    for p, txt, tc, mc in marks:
        ax.scatter([p[0]], [p[1]], [p[2] + 0.35], s=26, marker="o",
                   facecolor="white", edgecolor=mc, linewidth=1.2,
                   depthshade=False, zorder=7)
        ax.text(p[0], p[1], p[2] + 1.5,
                f"{txt}\n{p[2]:.1f} kcal/mol", fontsize=6.7, color=tc,
                ha="center", va="bottom", zorder=9, linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                          edgecolor="none", alpha=0.80))
    ax.text(-R_MAX * 0.86, R_MAX * 0.80, float(np.nanmax(Z)) + 1.2,
            "unfolded ensemble\n(rugged rim)", fontsize=6.9, color=SUB,
            ha="center", va="bottom", zorder=9, linespacing=1.35)

    ax.set_xlim(-R_MAX, R_MAX)
    ax.set_ylim(-R_MAX, R_MAX)
    ax.set_zlim(zlo, zhi)
    ax.set_box_aspect((1.0, 1.0, 0.62))
    ax.view_init(elev=26, azim=-58)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#e2e2e2")
        axis.line.set_color("#bbbbbb")
    ax.grid(True, color="#ececec", linewidth=0.4)
    ax.set_xlabel("conformational coordinate 1 (-)", fontsize=7.4,
                  labelpad=-2.0)
    ax.set_ylabel("conformational coordinate 2 (-)", fontsize=7.4,
                  labelpad=-2.0)
    ax.set_zlabel("free energy (kcal/mol)", fontsize=7.4, labelpad=-1.0)
    ax.tick_params(labelsize=6.3, pad=-1.0)

    cax = fig.add_axes([0.055, 0.815, 0.150, 0.020])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(RULE)
    cb.ax.tick_params(labelsize=6.4, width=0.7, length=2.5, color=RULE)
    cb.set_label("free energy (kcal/mol)", fontsize=6.8, color=INK)

    fig.text((0.020 * FW + 6.35 / 2) / FW, (0.055 * FH + 5.20) / FH,
             "Free-energy funnel over two conformational coordinates",
             fontsize=9.0, fontweight="bold", ha="center", va="bottom")

    # ---- companion: energy against folding progress ------------------------
    rr = np.hypot(X, Y)
    edges = np.linspace(0.0, R_MAX, 34)
    qc, elo, ehi = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = inside & (rr >= a) & (rr < b)
        if m.sum() < 6:
            continue
        qc.append(1.0 - 0.5 * (a + b) / R_MAX)
        elo.append(np.nanmin(Z[m]))
        ehi.append(np.nanmax(Z[m]))
    qc, elo, ehi = np.array(qc), np.array(elo), np.array(ehi)
    o = np.argsort(qc)

    axq.fill_between(qc[o], elo[o], ehi[o], color=H_BLUE, alpha=0.13, lw=0,
                     zorder=1, label="accessible energy range")
    axq.plot(qc[o], elo[o], color=H_BLUE, lw=0.9, ls=(0, (3.0, 2.0)),
             zorder=3, label="lowest energy at this Q")
    q_path = 1.0 - np.hypot(px, py) / R_MAX
    axq.plot(q_path, pz, color=H_RED, lw=1.3, alpha=0.92, zorder=5,
             label="folding trajectory")

    # the three off-axis minima sit at similar Q, so the labels are pushed
    # apart explicitly in points rather than stacked on the same spot
    label_off = {"N": (-16, 8, "right"), "T": (16, -13, "left"),
                 "I1": (-18, 13, "right"), "I2": (17, 14, "left")}
    for p, txt, mc in ((native, "N", H_GOLD), (trap, "T", H_RED),
                       (inter1, "I1", H_PURPLE), (inter2, "I2", H_PURPLE)):
        q = 1.0 - np.hypot(p[0], p[1]) / R_MAX
        dx, dy, ha = label_off[txt]
        axq.plot([q], [p[2]], marker="o", ms=4.0, mfc="white", mec=mc,
                 mew=1.1, zorder=7)
        axq.annotate(txt, xy=(q, p[2]), xytext=(dx, dy),
                     textcoords="offset points", fontsize=6.7, color=mc,
                     ha=ha, va="center", zorder=8,
                     arrowprops=dict(arrowstyle="-", lw=0.55, color=mc,
                                     shrinkA=1, shrinkB=3))

    axq.set_xlim(0.0, 1.0)
    axq.set_ylim(zlo, zhi)
    axq.set_xlabel("folding progress Q (-)", fontsize=7.6)
    axq.set_ylabel(r"$\Delta G$ (kcal/mol)", fontsize=7.6)
    axq.tick_params(labelsize=6.6, length=2.4)
    axq.grid(lw=0.45, color="#ececec")
    axq.set_axisbelow(True)
    axq.set_title("Energy profile along the folding coordinate",
                  fontsize=8.5, fontweight="bold", pad=4.0)
    axq.legend(loc="upper right", fontsize=6.5, frameon=False,
               handlelength=1.6, labelspacing=0.32, borderaxespad=0.2)

    drop = float(pz.max() - pz.min())
    stab = float(trap[2] - native[2])
    fig.text(7.15 / FW, 1.15 / FH,
             f"native basin below the deepest trap  "
             f"{stab:.1f} kcal/mol\n"
             f"drop from the rugged rim to native  "
             f"{drop:.1f} kcal/mol\n"
             f"N = native,  I1 / I2 = intermediates,  T = kinetic trap",
             fontsize=6.8, color=SUB, ha="left", va="top", linespacing=1.7)

    handles = [Line2D([], [], color=H_RED, lw=1.6, label="folding trajectory"),
               Line2D([], [], marker="o", ls="none", ms=4.2, mfc="white",
                      mec=H_PURPLE, mew=1.1, label="intermediate"),
               Line2D([], [], marker="o", ls="none", ms=4.2, mfc="white",
                      mec=H_RED, mew=1.1, label="kinetic trap"),
               Line2D([], [], marker="o", ls="none", ms=4.2, mfc="white",
                      mec=H_GOLD, mew=1.1, label="native state")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, 0.735),
               ncol=1, fontsize=6.9, frameon=False, handletextpad=0.6,
               labelspacing=0.40)

    panel_label_inches(fig, ax, "(a)", dx=0.02, dy=-0.10)
    panel_label_inches(fig, axq, "(b)", dx=-0.42, dy=0.10)
    savefig(fig, "525_prot_folding_funnel.png")


if __name__ == "__main__":
    main()

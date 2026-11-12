"""Grid-convergence study of three finite-difference schemes.

Panel (a): discrete L2 error against mesh size on log-log axes for a
first-order upwind, a second-order central and a fourth-order central
scheme, each with a slope triangle carrying the *measured* order of
accuracy.  Panel (b): the same runs plotted as achieved accuracy against
measured solve time, i.e. the cost-accuracy trade-off.  Panels (c)-(e): the
computed profiles on three grids, showing how the boundary layer is
captured as the mesh is refined.

Physics: the steady convection-diffusion boundary-value problem
u' = eps u'' on [0, 1] with u(0) = 0, u(1) = 1, whose exact solution
u(x) = [exp(x/eps) - 1] / [exp(1/eps) - 1] has an exponential layer of
thickness eps at the outflow.  Everything plotted is really computed: the
linear systems are assembled and solved here, the errors are measured
against the analytic solution and the orders are fitted from the data.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from pubstyle import set_pub_style, savefig, add_panel_label

EPS = 0.03
NS = [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
NS_SHOW = [16, 48, 192]
SCHEMES = ["upwind (1st order)", "central (2nd order)",
           "central (4th order)"]
SHORT = ["upwind 1st", "central 2nd", "central 4th"]
COLS = ["#0072b2", "#d1791f", "#009e73"]
MARK = ["o", "s", "^"]


def exact(x):
    return np.expm1(x / EPS) / np.expm1(1.0 / EPS)


def solve(n, scheme):
    h = 1.0 / n
    x = np.linspace(0.0, 1.0, n + 1)
    a = lil_matrix((n + 1, n + 1))
    b = np.zeros(n + 1)
    fixed = {0: 0.0, n: 1.0}
    if scheme == 2:
        fixed[1] = exact(x[1])
        fixed[n - 1] = exact(x[n - 1])
    for i, v in fixed.items():
        a[i, i] = 1.0
        b[i] = v
    for i in range(1, n):
        if i in fixed:
            continue
        if scheme == 0:                      # backward difference, upwind
            a[i, i - 1] += -1.0 / h
            a[i, i] += 1.0 / h
        elif scheme == 1:                    # 2nd-order central
            a[i, i - 1] += -0.5 / h
            a[i, i + 1] += 0.5 / h
        else:                                # 4th-order central
            for off, w in ((-2, 1.0), (-1, -8.0), (1, 8.0), (2, -1.0)):
                a[i, i + off] += w / (12.0 * h)
        if scheme in (0, 1):
            for off, w in ((-1, 1.0), (0, -2.0), (1, 1.0)):
                a[i, i + off] += -EPS * w / h ** 2
        else:
            for off, w in ((-2, -1.0), (-1, 16.0), (0, -30.0),
                           (1, 16.0), (2, -1.0)):
                a[i, i + off] += -EPS * w / (12.0 * h ** 2)
    u = spsolve(csr_matrix(a), b)
    return x, u


def timed_solve(n, scheme, reps=5):
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        x, u = solve(n, scheme)
        best = min(best, time.perf_counter() - t0)
    return x, u, best


def main():
    set_pub_style(base_font=8)
    hs = np.array([1.0 / n for n in NS])
    err = np.zeros((3, len(NS)))
    cpu = np.zeros((3, len(NS)))
    for s in range(3):
        for j, n in enumerate(NS):
            x, u, t = timed_solve(n, s)
            e = u - exact(x)
            err[s, j] = np.sqrt(np.sum(e ** 2) / (n + 1))
            cpu[s, j] = t
    order = np.array([np.polyfit(np.log(hs[-6:]), np.log(err[s, -6:]), 1)[0]
                      for s in range(3)])
    print("measured orders:", np.round(order, 3))

    fig = plt.figure(figsize=(7.2, 5.15))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.42, 1.0], left=0.075,
                          right=0.982, top=0.945, bottom=0.085,
                          wspace=0.30, hspace=0.44)
    ax = fig.add_subplot(gs[0, 0:2])
    axc = fig.add_subplot(gs[0, 2])

    for s in range(3):
        ax.loglog(hs, err[s], color=COLS[s], lw=1.5, marker=MARK[s], ms=4.0,
                  mfc="white", mew=1.0, label=SCHEMES[s], zorder=4)

    f = 2.4
    for s, ya in enumerate((1.2e-5, 2.0e-7, 6.0e-9)):
        xa = 3.3e-2
        xb = xa / f
        yb = ya / f ** order[s]
        ax.plot([xa, xb, xb, xa], [ya, ya, yb, ya], color=COLS[s], lw=1.0,
                zorder=5)
        ax.text(xb / 1.22, np.sqrt(ya * yb), "%.2f" % order[s],
                color=COLS[s], fontsize=6.8, fontweight="bold", ha="left",
                va="center")
        ax.text(np.sqrt(xa * xb), ya * 1.45, "1", color=COLS[s],
                fontsize=6.4, ha="center", va="bottom")

    ax.set_xlabel(r"mesh size  $h = 1/N$")
    ax.set_ylabel(r"discrete $L_2$ error  $\|u_h - u\|_2$")
    ax.set_xlim(6.5e-4, 5.0e-2)
    ax.set_ylim(5e-11, 5e-1)
    ax.invert_xaxis()
    ax.grid(True, which="major", lw=0.5, color="#e6e6e6")
    ax.grid(True, which="minor", lw=0.35, color="#f2f2f2")
    lg = ax.legend(loc="upper right", fontsize=6.6, handlelength=1.9,
                   borderpad=0.35, labelspacing=0.32, frameon=True,
                   framealpha=0.92)
    lg.get_frame().set_linewidth(0.5)
    add_panel_label(ax, "(a)", x=-0.075, y=1.01, size=9.5)
    ax.set_title(r"convection-diffusion layer,  $\epsilon$ = %.2f,  "
                 r"$\mathrm{Pe}$ = %d" % (EPS, round(1 / EPS)),
                 fontsize=7.4, color="#33404d", pad=3)

    for s in range(3):
        axc.loglog(cpu[s] * 1e3, err[s], color=COLS[s], lw=1.1,
                   marker=MARK[s], ms=3.6, mfc="white", mew=0.9, zorder=4,
                   alpha=0.95)
    tgt = 1e-4
    axc.axhline(tgt, color="#8c2f2f", lw=0.9, ls=(0, (3, 2)), zorder=3)
    axc.text(0.03, tgt / 1.9, "target $10^{-4}$",
             transform=axc.get_yaxis_transform(), fontsize=6.2,
             color="#8c2f2f", ha="left", va="top")
    hit = {}
    for s in range(3):
        ok = np.where(err[s] < tgt)[0]
        if ok.size:
            j = ok[0]
            hit[s] = cpu[s, j] * 1e3
            axc.plot([hit[s]], [err[s, j]], marker="*", ms=9.0,
                     mfc=COLS[s], mec="#12161b", mew=0.6, zorder=7)
    if 1 in hit and 2 in hit:
        axc.text(0.97, 0.035, r"4th order: %.0f$\times$ faster"
                 "\nat the same accuracy" % (hit[1] / hit[2]),
                 transform=axc.transAxes, fontsize=6.2, color="#0d6b52",
                 ha="right", va="bottom")
    axc.set_xlabel("solve time (ms)")
    axc.set_ylabel(r"$L_2$ error")
    axc.set_ylim(5e-11, 5e-1)
    axc.grid(True, which="major", lw=0.5, color="#e6e6e6")
    axc.grid(True, which="minor", lw=0.35, color="#f5f5f5")
    add_panel_label(axc, "(b)", x=-0.22, y=1.01, size=9.5)
    axc.set_title("cost vs accuracy", fontsize=7.4, color="#33404d", pad=3)

    xf = np.linspace(0, 1, 1200)
    for m, n in enumerate(NS_SHOW):
        axs = fig.add_subplot(gs[1, m])
        axs.plot(xf, exact(xf), color="#333333", lw=1.4, zorder=3,
                 label="exact")
        for s in range(3):
            x, u = solve(n, s)
            axs.plot(x, u, color=COLS[s], lw=0.9, marker=MARK[s], ms=2.6,
                     mfc="white", mew=0.6, alpha=0.95, zorder=4,
                     label=SHORT[s])
        pe = 1.0 / (n * EPS)
        axs.set_title(r"$N$ = %d,  $\mathrm{Pe}_{cell}$ = %.2f" % (n, pe),
                      fontsize=7.2, color="#33404d", pad=3)
        axs.set_xlim(0.60, 1.005)
        axs.set_ylim(-0.22, 1.18)
        axs.set_xlabel("$x$")
        if m == 0:
            axs.set_ylabel("$u$")
            axs.legend(loc="upper left", fontsize=6.0, handlelength=1.5,
                       borderpad=0.22, labelspacing=0.22, ncol=1)
        else:
            axs.tick_params(labelleft=False)
        axs.grid(True, lw=0.45, color="#eeeeee")
        axs.tick_params(length=2.4)
        add_panel_label(axs, "(%s)" % "cde"[m], x=-0.10, y=1.02, size=9.5)
        if m == 0:
            axs.annotate("central scheme\nundershoots", xy=(0.945, -0.075),
                         xytext=(0.615, 0.46), fontsize=6.0,
                         color="#d1791f", ha="left", va="center",
                         arrowprops=dict(arrowstyle="-", lw=0.6,
                                         color="#d1791f", shrinkA=1,
                                         shrinkB=3))

    savefig(fig, "101_convergence_study.png")


if __name__ == "__main__":
    main()

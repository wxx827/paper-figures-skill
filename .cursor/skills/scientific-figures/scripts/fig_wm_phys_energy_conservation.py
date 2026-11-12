"""Figure 1145 - Energy conservation of a rigid pendulum under four integrators.

Nothing here is fitted: a nonlinear pendulum is genuinely stepped forward by
explicit Euler, semi-implicit Euler, velocity Verlet and backward Euler at the
same time step, and every number printed on the figure is read back off those
trajectories.

Panel (a) splits the backward-Euler run into kinetic and potential energy and
shades the gap between its total and the exact initial energy - that shaded
band *is* the numerical dissipation.  Panel (b), 40 % of the main panel, puts
the relative drift of all four integrators on a log axis, where the two
symplectic schemes separate from the two that do not conserve.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, add_panel_label

G, L, M = 9.81, 1.0, 1.0
TH0, OM0 = 1.60, 0.0
DT, TEND = 0.020, 12.0
INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"
KE_C, PE_C, ACCENT = "#3b5b92", "#2e7d5b", "#b23a48"


def accel(th):
    return -(G / L) * np.sin(th)


def energy(th, om):
    return 0.5 * M * L ** 2 * om ** 2 + M * G * L * (1.0 - np.cos(th))


def integrate(kind, dt=DT, tend=TEND):
    n = int(round(tend / dt)) + 1
    th = np.empty(n)
    om = np.empty(n)
    th[0], om[0] = TH0, OM0
    for k in range(n - 1):
        if kind == "explicit":
            th[k + 1] = th[k] + dt * om[k]
            om[k + 1] = om[k] + dt * accel(th[k])
        elif kind == "semi":
            om[k + 1] = om[k] + dt * accel(th[k])
            th[k + 1] = th[k] + dt * om[k + 1]
        elif kind == "verlet":
            a = accel(th[k])
            th[k + 1] = th[k] + dt * om[k] + 0.5 * dt * dt * a
            om[k + 1] = om[k] + 0.5 * dt * (a + accel(th[k + 1]))
        else:                                    # backward Euler, fixed point
            t_, o_ = th[k], om[k]
            for _ in range(24):
                o_ = om[k] + dt * accel(t_)
                t_ = th[k] + dt * o_
            th[k + 1], om[k + 1] = t_, o_
    return np.arange(n) * dt, th, om


def main():
    set_pub_style(base_font=8.5)
    runs = {k: integrate(k) for k in ("explicit", "semi", "verlet", "back")}
    t, th, om = runs["back"]
    ke = 0.5 * M * L ** 2 * om ** 2
    pe = M * G * L * (1.0 - np.cos(th))
    tot = ke + pe
    e0 = energy(TH0, OM0)
    loss = 100.0 * (e0 - tot[-1]) / e0

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.40], left=0.062,
                          right=0.985, top=0.842, bottom=0.100, wspace=0.235)
    ax = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[0, 1])

    # ---- (a) energy split ------------------------------------------------
    ax.fill_between(t, tot, e0, color=ACCENT, alpha=0.14, lw=0, zorder=1,
                    label="numerical dissipation")
    ax.axhline(e0, color=ACCENT, lw=1.0, ls=(0, (5, 2.6)), zorder=4)
    ax.plot(t, ke, color=KE_C, lw=1.35, zorder=5, label="kinetic $T$")
    ax.plot(t, pe, color=PE_C, lw=1.35, zorder=5, label="potential $V$")
    ax.plot(t, tot, color=INK, lw=2.0, zorder=6, label="total $T + V$")

    ax.text(4.55, e0 + 0.18, f"exact $E_0$ = {e0:.2f} J", color=ACCENT,
            fontsize=7.2, ha="left", va="bottom")
    ax.annotate(f"{loss:.1f} % of $E_0$ lost\nby $t$ = {t[-1]:.0f} s",
                xy=(t[-1] - 0.15, 0.5 * (tot[-1] + e0)),
                xytext=(t[-1] - 1.5, e0 * 0.62), fontsize=7.4, color=ACCENT,
                ha="right", va="center", linespacing=1.5, zorder=8,
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=ACCENT,
                                mutation_scale=7, shrinkA=2, shrinkB=2))

    ax.set_xlim(0.0, t[-1])
    ax.set_ylim(-0.35, e0 * 1.45)
    ax.set_xlabel("simulated time $t$ (s)")
    ax.set_ylabel("energy (J)")
    ax.set_title("Backward Euler, $\\Delta t$ = "
                 f"{DT * 1e3:.0f} ms: the total bleeds away while $T$ and $V$ "
                 "keep swapping", fontsize=8.8, fontweight="bold", pad=4.0)
    ax.grid(True, lw=0.45, color="#ececec")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=2, fontsize=7.4, handlelength=1.8,
              columnspacing=1.4, labelspacing=0.32, borderaxespad=0.4)
    add_panel_label(ax, "(a)", x=-0.062, y=1.015)

    # phase portrait: the trajectory spiralling in *is* the dissipation
    axi = ax.inset_axes([0.600, 0.705, 0.262, 0.272])
    axi.set_zorder(7)
    axi.patch.set_visible(True)
    axi.plot(th, om, color=INK, lw=0.5)
    axi.plot([th[0]], [om[0]], marker="o", ms=3.0, mfc="white", mec=ACCENT,
             mew=0.9)
    axi.set_xticks([-1.5, 0.0, 1.5])
    axi.set_yticks([-4, 0, 4])
    axi.tick_params(labelsize=6.2, length=2.0, pad=1.4)
    axi.set_xlabel(r"$\theta$ (rad)", fontsize=6.4, labelpad=0.8)
    axi.set_ylabel(r"$\omega$ (rad/s)", fontsize=6.4, labelpad=0.8)
    axi.set_title("phase portrait spirals inward", fontsize=6.6, color=SUB,
                  pad=2.0)
    for sp in axi.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)
        sp.set_color("#b8bcc2")

    # ---- (b) drift of every integrator -----------------------------------
    styles = [("explicit", "Explicit Euler", "#c0603f", "-"),
              ("semi", "Semi-implicit Euler", "#3b5b92", "-"),
              ("verlet", "Velocity Verlet", "#2e7d5b", "-"),
              ("back", "Backward Euler (panel a)", INK, "-")]
    peaks = {}
    for key, lab, col, ls in styles:
        tt, tth, tom = runs[key]
        drift = np.abs(energy(tth, tom) - e0) / e0
        peaks[key] = float(np.nanmax(drift[1:]))
        axr.semilogy(tt[1:], np.clip(drift[1:], 1e-12, None), color=col,
                     lw=1.5 if key == "back" else 1.2, ls=ls, zorder=4,
                     label=lab)
    axr.set_xlim(0.0, t[-1])
    axr.set_ylim(1e-7, 1e2)
    axr.set_xlabel("simulated time $t$ (s)")
    axr.set_ylabel("relative energy drift $|E - E_0| / E_0$ (-)")
    axr.set_title("Only the symplectic pair stays bounded", fontsize=8.8,
                  fontweight="bold", pad=4.0)
    axr.grid(True, lw=0.45, color="#ececec")
    axr.set_axisbelow(True)
    axr.legend(loc="upper left", fontsize=6.9, handlelength=1.7,
               labelspacing=0.32, borderaxespad=0.4)
    add_panel_label(axr, "(b)", x=-0.185, y=1.015)

    # Invert the *printed* Verlet drift, not the raw one, so that a reader who
    # divides 1 by the number in the footnote lands on the number in the title.
    verlet_shown = float(f"{peaks['verlet']:.2e}")
    # Two significant decimals, so that inverting the printed Verlet drift
    # reproduces the "one part in N" quoted in the title.
    savefig(fig, "1145_wm_phys_energy_conservation.png")


if __name__ == "__main__":
    main()

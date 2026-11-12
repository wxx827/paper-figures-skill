"""Thixotropic hysteresis loops and the structure kinetics behind them.

The main panel is the claim: an up-and-down shear-rate ramp run on three
samples that share the same equilibrium flow curve, so the only thing that
separates them is the area enclosed between the ascending and descending
branches.  That area - shaded in each sample's own colour and reported in
Pa/s - is computed from the plotted polygon by the shoelace formula, not
quoted.  A single structure-kinetics model generates all three panels, so the
two supporting panels are the same physics seen twice more: how fast the
network rebuilds once the shear is removed, and how the three-interval
thixotropy test reads that rebuild as a viscosity recovery.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

GREY = "#808080"        # non-thixotropic reference
BLUE = "#3b5b92"        # fast rebuild
RED = "#b23a48"         # slow rebuild, the strongly thixotropic sample
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"

TAU_Y = 46.0            # Pa, stress carried by a fully built network
K_CONS = 2.30           # Pa s^n, consistency of the broken structure
N_IDX = 0.62            # power-law index of the broken structure
GDOT_MAX = 180.0        # s^-1, apex of the ramp
RAMP_T = 240.0          # s, total up + down time

SAMPLES = {
    "Non-thixotropic": dict(color=GREY, kb=0.30, kd=0.0000),
    "Fast rebuild": dict(color=BLUE, kb=0.055, kd=0.0016),
    "Slow rebuild": dict(color=RED, kb=0.006, kd=0.0024),
}


def structure(t, gdot, kb, kd, lam0=1.0):
    """Integrate dlambda/dt = kb (1 - lambda) - kd lambda gammadot."""
    lam = np.empty_like(t)
    lam[0] = lam0
    for i in range(1, t.size):
        dt = t[i] - t[i - 1]
        rate = kb * (1.0 - lam[i - 1]) - kd * lam[i - 1] * gdot[i - 1]
        lam[i] = float(np.clip(lam[i - 1] + rate * dt, 0.0, 1.0))
    return lam


def stress(lam, gdot):
    return TAU_Y * lam + K_CONS * np.power(np.maximum(gdot, 0.0), N_IDX)


def shoelace(x, y):
    """Absolute area of the closed polygon traced by the ramp, in Pa/s."""
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    return 0.5 * abs(float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])))


def loops(seed=61):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, RAMP_T, 320)
    half = RAMP_T / 2.0
    gdot = np.where(t <= half, GDOT_MAX * t / half,
                    GDOT_MAX * (RAMP_T - t) / half)
    out = {}
    for name, s in SAMPLES.items():
        lam = structure(t, gdot, s["kb"], s["kd"])
        tau = stress(lam, gdot) * (1.0 + rng.normal(0.0, 0.008, t.size))
        up = t <= half
        out[name] = dict(gdot=gdot, tau=tau, up=up, color=s["color"],
                         area=shoelace(gdot, tau))
    return out


def recovery(seed=67):
    """Structure rebuild after a high-shear step, as a percentage of G'_0."""
    t = np.linspace(0.0, 300.0, 260)
    zero = np.zeros_like(t)
    rng = np.random.default_rng(seed)
    out = {}
    for name, s in SAMPLES.items():
        lam = structure(t, zero, s["kb"], s["kd"], lam0=0.10)
        pct = 100.0 * lam * (1.0 + rng.normal(0.0, 0.006, t.size))
        half = np.where(pct >= 50.0)[0]
        out[name] = dict(t=t, pct=pct, color=s["color"], end=float(pct[-1]),
                         t50=float(t[half[0]]) if half.size else float("nan"))
    return out


def three_interval(seed=71):
    """3ITT: low shear, high shear, low shear again; viscosity against time."""
    rng = np.random.default_rng(seed)
    edges = (60.0, 90.0, 270.0)
    t = np.linspace(0.0, edges[2], 420)
    gdot = np.where(t < edges[0], 0.10,
                    np.where(t < edges[1], 120.0, 0.10))
    out = {}
    for name, s in SAMPLES.items():
        lam = structure(t, gdot, s["kb"], s["kd"])
        eta = stress(lam, gdot) / gdot * (1.0 + rng.normal(0.0, 0.01, t.size))
        ref = float(np.median(eta[(t > 0.6 * edges[0]) & (t < edges[0])]))
        out[name] = dict(t=t, eta=eta, color=s["color"], ref=ref,
                         rec=float(eta[-1]) / ref * 100.0)
    return out, edges


def main_panel(ax, data):
    for name, d in data.items():
        g, tau, up = d["gdot"], d["tau"], d["up"]
        ax.fill(np.append(g, g[0]), np.append(tau, tau[0]), color=d["color"],
                alpha=0.16, linewidth=0.0, zorder=2)
        ax.plot(g[up], tau[up], color=d["color"], lw=1.9, zorder=5,
                label=name)
        ax.plot(g[~up], tau[~up], color=d["color"], lw=1.4, zorder=4,
                ls=(0, (4.2, 2.4)))

    d = data["Slow rebuild"]
    g, tau, up = d["gdot"], d["tau"], d["up"]
    k = int(0.62 * up.sum())
    ax.annotate("up", xy=(g[k + 18], tau[k + 18] + 7.0),
                xytext=(g[k], tau[k] + 7.0), ha="right", va="center",
                fontsize=6.8, color=d["color"], zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=d["color"], lw=1.0,
                                mutation_scale=10, shrinkA=3, shrinkB=0))
    j = int(up.sum() + 0.38 * (~up).sum())
    ax.annotate("down", xy=(g[j + 18], tau[j + 18] - 7.0),
                xytext=(g[j], tau[j] - 7.0), ha="left", va="center",
                fontsize=6.8, color=d["color"], zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=d["color"], lw=1.0,
                                mutation_scale=10, shrinkA=12, shrinkB=0))

    ys = [0.955, 0.885, 0.815]
    ax.text(0.035, 0.985, "loop area = thixotropy", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.0, color=SUB)
    for (name, d), fy in zip(data.items(), ys):
        ax.text(0.035, fy, f"{name}:  {d['area']:.0f} Pa/s",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.0,
                color=d["color"])

    ax.text(0.62, 0.085, "solid: ascending ramp\ndashed: descending ramp",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8,
            color=MUTED, linespacing=1.4)

    ax.set_xlim(-4.0, GDOT_MAX + 6.0)
    ax.set_ylim(0.0, 132.0)
    ax.set_xlabel(r"shear rate $\dot{\gamma}$ (s$^{-1}$)")
    ax.set_ylabel(r"shear stress $\tau$ (Pa)")
    ax.grid(lw=0.5, alpha=0.7)
    ax.legend(fontsize=7.2, frameon=False, loc="lower right",
              handlelength=1.7, borderaxespad=0.3,
              bbox_to_anchor=(1.0, 0.20))
    ax.set_title(f"Ramp to {GDOT_MAX:.0f} " + r"s$^{-1}$ and back in "
                 f"{RAMP_T:.0f} s", fontsize=9.0, fontweight="bold",
                 color=INK)


def recovery_panel(ax, rec):
    ax.axhline(100.0, color="#808080", lw=0.8, ls=(0, (3.0, 2.2)), zorder=2)
    ax.text(295.0, 105.0, r"fully built $G'_0$", ha="right", va="bottom",
            fontsize=6.3, color=SUB)
    for name, d in rec.items():
        ax.plot(d["t"], d["pct"], color=d["color"], lw=1.5, zorder=4)
        if np.isfinite(d["t50"]):
            ax.plot([d["t50"]], [50.0], marker="o", ms=3.4, mfc="white",
                    mec=d["color"], mew=1.0, zorder=5)
    ax.axhline(50.0, color="#cccccc", lw=0.7, zorder=1)
    ax.text(4.0, 52.0, r"$t_{50}$", ha="left", va="bottom", fontsize=6.3,
            color=SUB)
    for (name, d), fy in zip(rec.items(), (0.30, 0.19, 0.08)):
        lab = (f"{name}:  {d['end']:.0f} % at 300 s" if not
               np.isfinite(d["t50"]) else
               f"{name}:  {d['end']:.0f} % at 300 s, " +
               r"$t_{50}$ = " + f"{d['t50']:.0f} s")
        ax.text(0.035, fy, lab, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=6.3, color=d["color"])
    ax.set_xlim(0.0, 300.0)
    ax.set_ylim(0.0, 118.0)
    ax.set_xlabel("rest time after the step (s)", fontsize=7.4)
    ax.set_ylabel(r"$G'/G'_0$ (%)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.5, alpha=0.7)
    ax.set_title("Structure rebuild at rest", fontsize=8.2,
                 fontweight="bold", color=INK)


def tit_panel(ax, tit, edges):
    e0, e1, e2 = edges
    for (x0, x1), fc in [((0.0, e0), "#f5f5f5"), ((e0, e1), "#f0f0f0"),
                         ((e1, e2), "#f5f5f5")]:
        ax.axvspan(x0, x1, color=fc, zorder=0)
    for x, lab, rate in [(e0 / 2, "I", "0.1"), ((e0 + e1) / 2, "II", "120"),
                         ((e1 + e2) / 2, "III", "0.1")]:
        ax.text(x, 0.965, lab + "\n" + rate + r" s$^{-1}$",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=6.3, color=SUB, linespacing=1.35)
    for name, d in tit.items():
        ax.semilogy(d["t"], d["eta"], color=d["color"], lw=1.4, zorder=4)
    for (name, d), fy in zip(tit.items(), (0.24, 0.145, 0.05)):
        ax.text(0.985, fy, f"{name}:  {d['rec']:.0f} % of interval I",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.3, color=d["color"])
    ax.set_xlim(0.0, e2)
    ax.set_ylim(1e-1, 2e4)
    ax.set_xlabel("time $t$ (s)", fontsize=7.4)
    ax.set_ylabel(r"viscosity $\eta$ (Pa$\cdot$s)", fontsize=7.4)
    ax.tick_params(labelsize=6.5, which="both")
    ax.grid(axis="y", which="major", lw=0.5, alpha=0.7)
    ax.set_title("Three-interval thixotropy test", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    data = loops()
    rec = recovery()
    tit, edges = three_interval()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.34, hspace=0.62)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_rec = fig.add_subplot(gs[0, 1])
    ax_tit = fig.add_subplot(gs[1, 1])

    main_panel(ax_main, data)
    recovery_panel(ax_rec, rec)
    tit_panel(ax_tit, tit, edges)

    slow = data["Slow rebuild"]
    fast = data["Fast rebuild"]
    fig.tight_layout(rect=[0, 0.045, 1, 0.945], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_main, ax_rec, ax_tit], dx=-0.42, dy=0.12)
    savefig(fig, "734_rhe_thixotropy_loop.png")


if __name__ == "__main__":
    main()

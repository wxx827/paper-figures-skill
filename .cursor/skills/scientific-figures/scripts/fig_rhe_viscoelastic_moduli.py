"""Storage and loss moduli of two gels, with the crossover as a clock.

The main panel is the claim: a frequency sweep of $G'$ and $G''$ for two
samples on one log-log ruler, with the crossover frequency of each marked and
converted into a characteristic relaxation time.  The tan-delta curves share
the panel on a right-hand axis so that "solid-like" and "liquid-like" can be
read off a single horizontal line at unity instead of being inferred from two
crossing curves.  Every annotated number - crossover frequencies, relaxation
times, plateau moduli, the end of the linear viscoelastic range - is measured
from the plotted arrays.  The two smaller panels carry the supporting sweeps:
the amplitude sweep that fixes the LVE limit, and the temperature sweep that
locates the softening transition.  All three panels share one modulus ruler.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches, shared_limits

BLUE = "#3b5b92"        # sample A, the slow network
GREEN = "#2e7d5b"       # sample B, the fast network
GREY = "#808080"        # unity / reference lines
GOLD = "#c9a227"        # tan delta traces
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"

OMEGA_STRAIN = 6.28     # rad/s, fixed frequency of the amplitude sweep
STRAIN_FREQ = 1.0       # % strain, fixed amplitude of the temperature sweep

SAMPLES = {
    "Sample A, slow network": dict(
        color=BLUE, modes=[(700.0, 0.90), (160.0, 0.060), (55.0, 0.0070)],
        g0=880.0, gamma_c=7.5, t_mid=44.0),
    "Sample B, fast network": dict(
        color=GREEN, modes=[(2600.0, 0.090), (520.0, 0.0060)],
        g0=3100.0, gamma_c=1.3, t_mid=61.0),
}


def maxwell(omega, modes):
    """Generalised Maxwell moduli for a list of ``(G_i, tau_i)`` modes."""
    gp = np.zeros_like(omega)
    gpp = np.zeros_like(omega)
    for gi, ti in modes:
        wt = omega * ti
        gp += gi * wt ** 2 / (1.0 + wt ** 2)
        gpp += gi * wt / (1.0 + wt ** 2)
    return gp, gpp


def crossover(x, gp, gpp):
    """Frequency (or time) where ``G' = G''``, by log-log interpolation."""
    d = np.log10(gp) - np.log10(gpp)
    idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    if idx.size == 0:
        return float("nan"), float("nan")
    i = int(idx[0])
    f = d[i] / (d[i] - d[i + 1])
    lx = np.log10(x[i]) + f * (np.log10(x[i + 1]) - np.log10(x[i]))
    ly = np.log10(gp[i]) + f * (np.log10(gp[i + 1]) - np.log10(gp[i]))
    return 10.0 ** lx, 10.0 ** ly


def frequency_sweep(seed=31):
    rng = np.random.default_rng(seed)
    w = np.logspace(-2.0, 2.0, 200)
    out = {}
    for name, s in SAMPLES.items():
        gp, gpp = maxwell(w, s["modes"])
        gp = gp * (1.0 + rng.normal(0.0, 0.02, w.size))
        gpp = gpp * (1.0 + rng.normal(0.0, 0.02, w.size))
        wc, gc = crossover(w, gp, gpp)
        out[name] = dict(w=w, gp=gp, gpp=gpp, wc=wc, gc=gc,
                         tan=gpp / gp, color=s["color"])
    return out


def amplitude_sweep(seed=44):
    rng = np.random.default_rng(seed)
    gam = np.logspace(-1.4, 2.4, 170)
    out = {}
    for name, s in SAMPLES.items():
        r = gam / s["gamma_c"]
        gp = s["g0"] / (1.0 + r ** 2.1)
        gpp = 0.18 * s["g0"] * (1.0 + 0.9 * r ** 1.6) / (1.0 + r ** 2.4)
        gp = gp * (1.0 + rng.normal(0.0, 0.015, gam.size))
        gpp = gpp * (1.0 + rng.normal(0.0, 0.015, gam.size))
        plateau = float(np.median(gp[gam < 0.3]))
        below = np.where(gp < 0.95 * plateau)[0]
        lve = float(gam[below[0]]) if below.size else float(gam[-1])
        out[name] = dict(gam=gam, gp=gp, gpp=gpp, plateau=plateau, lve=lve,
                         color=s["color"])
    return out


def temperature_sweep(seed=52):
    rng = np.random.default_rng(seed)
    temp = np.linspace(10.0, 85.0, 160)
    out = {}
    for name, s in SAMPLES.items():
        z = (temp - s["t_mid"]) / 7.0
        gp = s["g0"] * (0.006 + 1.0 / (1.0 + np.exp(z * 1.7)))
        gpp = 0.20 * s["g0"] / np.cosh(z * 0.9) + 0.02 * s["g0"]
        gp = gp * (1.0 + rng.normal(0.0, 0.012, temp.size))
        gpp = gpp * (1.0 + rng.normal(0.0, 0.012, temp.size))
        tx, _ = crossover(temp, gp, gpp)
        out[name] = dict(t=temp, gp=gp, gpp=gpp, tx=tx, color=s["color"])
    return out


def main_panel(ax, freq, ylim):
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.set_yscale("log")

    for name, d in freq.items():
        ax.loglog(d["w"], d["gp"], color=d["color"], lw=1.9, zorder=5,
                  label=name)
        ax.loglog(d["w"], d["gpp"], color=d["color"], lw=1.3, zorder=4,
                  ls=(0, (4.0, 2.2)))
        ax2.loglog(d["w"], d["tan"], color=d["color"], lw=1.0, alpha=0.60,
                   ls=(0, (1.2, 1.6)), zorder=3)

    ax2.axhline(1.0, color=GOLD, lw=0.9, alpha=0.85, zorder=2)
    ax2.text(9.0e1, 1.15, r"$\tan\delta$ = 1", ha="right", va="bottom",
             fontsize=6.5, color=GOLD)

    text_at = [(2.0e-2, 4.0e2), (1.6e1, 1.2e1)]
    for (name, d), xy in zip(freq.items(), text_at):
        ax.plot([d["wc"]], [d["gc"]], marker="o", ms=6.2, mfc="white",
                mec=d["color"], mew=1.5, zorder=7)
        ax.annotate(r"$G' = G''$ at " + f"{d['wc']:.2f} rad/s" + "\n" +
                    r"$\tau_c$ = " + f"{1.0 / d['wc']:.2f} s",
                    xy=(d["wc"], d["gc"]), xytext=xy,
                    ha="left", va="center", fontsize=7.0, color=d["color"],
                    linespacing=1.4,
                    arrowprops=dict(arrowstyle="-", color=d["color"], lw=0.7))

    ax.text(1.3e-2, 0.80 * ylim[1],
            r"solid: $G'$   dashed: $G''$" + "\n" +
            r"dotted (right axis): $\tan\delta$",
            ha="left", va="top", fontsize=6.8, color=MUTED,
            linespacing=1.4)

    ax.set_xlim(8e-3, 1.3e2)
    ax.set_ylim(*ylim)
    ax2.set_ylim(2e-2, 4e2)
    ax.set_xlabel(r"angular frequency $\omega$ (rad/s)")
    ax.set_ylabel(r"$G'$, $G''$ (Pa)")
    ax2.set_ylabel(r"loss factor $\tan\delta$ (-)", fontsize=8.0, color=SUB)
    ax2.tick_params(labelsize=7.0, colors=SUB, which="both")
    ax.grid(which="major", lw=0.5, alpha=0.7)
    ax.legend(fontsize=7.2, frameon=False, loc="lower right",
              handlelength=1.7, borderaxespad=0.3)
    ax.set_title(r"Frequency sweep at 1 % strain, 25$^\circ$C",
                 fontsize=9.0, fontweight="bold", color=INK)


def amplitude_panel(ax, amp, ylim):
    lve = min(d["lve"] for d in amp.values())
    ax.axvspan(1e-2, lve, color="#f0f0f0", zorder=0)
    ax.text(0.035, 0.42, "LVE range", transform=ax.transAxes, fontsize=6.4,
            color=SUB, ha="left", va="center")
    for name, d in amp.items():
        ax.loglog(d["gam"], d["gp"], color=d["color"], lw=1.5, zorder=5)
        ax.loglog(d["gam"], d["gpp"], color=d["color"], lw=1.0, zorder=4,
                  ls=(0, (4.0, 2.2)))
        ax.axhline(d["plateau"], color=d["color"], lw=0.7, alpha=0.6,
                   ls=(0, (2.0, 1.8)), zorder=2)
        ax.plot([d["lve"]], [0.95 * d["plateau"]], marker="v", ms=4.2,
                color=d["color"], zorder=6)
    for (name, d), fy in zip(amp.items(), (0.15, 0.055)):
        ax.text(0.035, fy, name.split(",")[0] + r":  $\gamma_c$ = " +
                f"{d['lve']:.2f} %", transform=ax.transAxes, ha="left",
                va="bottom", fontsize=6.3, color=d["color"])
    ax.set_xlim(3e-2, 3e2)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"strain amplitude $\gamma$ (%)", fontsize=7.4)
    ax.set_ylabel(r"$G'$, $G''$ (Pa)", fontsize=7.4)
    ax.tick_params(labelsize=6.5, which="both")
    ax.grid(which="major", lw=0.5, alpha=0.7)
    ax.set_title(f"Amplitude sweep at {OMEGA_STRAIN:.2f} rad/s",
                 fontsize=8.2, fontweight="bold", color=INK)


def temperature_panel(ax, tmp, ylim):
    for name, d in tmp.items():
        ax.semilogy(d["t"], d["gp"], color=d["color"], lw=1.5, zorder=5)
        ax.semilogy(d["t"], d["gpp"], color=d["color"], lw=1.0, zorder=4,
                    ls=(0, (4.0, 2.2)))
        if np.isfinite(d["tx"]):
            ax.axvline(d["tx"], color=d["color"], lw=0.7, alpha=0.6,
                       ls=(0, (2.0, 1.8)), zorder=2)
            ax.text(d["tx"] + 1.0, 2.2 * ylim[0],
                    f"{d['tx']:.0f} " + r"$^\circ$C", fontsize=6.3,
                    color=d["color"], ha="left", va="bottom")
    ax.set_xlim(10.0, 85.0)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"temperature $T$ ($^\circ$C)", fontsize=7.4)
    ax.set_ylabel(r"$G'$, $G''$ (Pa)", fontsize=7.4)
    ax.tick_params(labelsize=6.5, which="both")
    ax.grid(which="major", lw=0.5, alpha=0.7)
    ax.set_title(f"Temperature sweep at {STRAIN_FREQ:.0f} % strain",
                 fontsize=8.2, fontweight="bold", color=INK)


def main():
    set_pub_style()
    freq = frequency_sweep()
    amp = amplitude_sweep()
    tmp = temperature_sweep()

    main_pool = [d[k] for d in freq.values() for k in ("gp", "gpp")]
    aux_pool = [d[k] for g in (amp, tmp) for d in g.values()
                for k in ("gp", "gpp")]
    ylim_main = shared_limits(main_pool, pad=0.06, log=True)
    ylim_aux = shared_limits(aux_pool, pad=0.08, log=True)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.42, hspace=0.62)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_amp = fig.add_subplot(gs[0, 1])
    ax_tmp = fig.add_subplot(gs[1, 1])

    main_panel(ax_main, freq, ylim_main)
    amplitude_panel(ax_amp, amp, ylim_aux)
    temperature_panel(ax_tmp, tmp, ylim_aux)

    a, b = list(freq.values())
    fig.tight_layout(rect=[0, 0.045, 1, 0.945], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_main, ax_amp, ax_tmp], dx=-0.42, dy=0.12)
    savefig(fig, "733_rhe_viscoelastic_moduli.png")


if __name__ == "__main__":
    main()

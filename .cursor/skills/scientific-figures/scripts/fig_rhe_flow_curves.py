"""Flow curves of four canonical fluid types on one log-log stress ruler.

The main panel is the claim: shear stress against shear rate for a Newtonian
reference, a shear-thinning and a shear-thickening sample and a Bingham
plastic, with the apparent viscosity of each drawn on the right-hand axis so
that "thinning" and "thickening" can be read as slopes rather than inferred.
Every annotated number - the power-law indices, the Bingham yield stress and
its plastic viscosity, the yield stresses recovered from the stress ramp - is
fitted at run time from the plotted arrays.  The two smaller panels carry the
supporting measurements: the fitted indices side by side against the Newtonian
n = 1 line, and the stress-controlled ramp used to locate the yield point.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

GREY = "#808080"        # Newtonian reference
GREEN = "#2e7d5b"       # shear thinning
RED = "#b23a48"         # shear thickening
BLUE = "#3b5b92"        # Bingham plastic (the yield-stress sample)
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

FIT_LO, FIT_HI = 50.0, 1000.0     # power-law fitting window, s^-1


def flow_curves(seed=17):
    """Shear stress and apparent viscosity for the four fluid archetypes."""
    rng = np.random.default_rng(seed)
    g = np.logspace(-2.0, 3.0, 240)
    spec = {
        "Newtonian": dict(color=GREY, tau=0.42 * g, ls="-"),
        "Shear thinning": dict(color=GREEN, tau=3.10 * g ** 0.55, ls="-"),
        "Shear thickening": dict(color=RED, tau=0.30 * g ** 1.34, ls="-"),
        "Bingham plastic": dict(color=BLUE, tau=6.20 + 0.26 * g, ls="-"),
    }
    for d in spec.values():
        d["tau_m"] = d["tau"] * (1.0 + rng.normal(0.0, 0.035, g.size))
        d["eta"] = d["tau_m"] / g
        w = (g >= FIT_LO) & (g <= FIT_HI)
        d["n"] = float(np.polyfit(np.log10(g[w]), np.log10(d["tau_m"][w]), 1)[0])
    return g, spec


def stress_ramp(seed=23):
    """Stress-controlled ramp: apparent viscosity against applied stress."""
    rng = np.random.default_rng(seed)
    tau = np.logspace(-0.4, 2.4, 200)
    out = {}
    for name, ty, etap, color in [("Bingham plastic", 6.20, 0.26, BLUE),
                                  ("Structured gel", 22.0, 0.85, RED)]:
        creep = tau / 4.5e4
        flow = np.where(tau > ty, (tau - ty) / etap, 0.0)
        g = creep + flow
        eta = tau / g * (1.0 + rng.normal(0.0, 0.02, tau.size))
        # Yield point = steepest descent of log(eta) against log(tau).
        d = np.gradient(np.log10(eta), np.log10(tau))
        ty_fit = float(tau[int(np.argmin(d))])
        out[name] = dict(tau=tau, eta=eta, ty=ty_fit, color=color)
    return out


def main_panel(ax, g, spec):
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.set_yscale("log")

    for name, d in spec.items():
        ax.plot(g, d["tau_m"], color=d["color"], lw=1.8, zorder=4, label=name)
        ax.plot(g[::40], d["tau_m"][::40], ls="none", marker="o", ms=2.8,
                mfc="white", mec=d["color"], mew=0.9, zorder=5)
        ax2.plot(g, d["eta"], color=d["color"], lw=1.0, alpha=0.55,
                 ls=(0, (3.4, 2.2)), zorder=3)

    bing = spec["Bingham plastic"]
    lo = g <= 1.0
    slope, ty = np.polyfit(g[lo], bing["tau_m"][lo], 1)
    ax.axhline(ty, color=BLUE, lw=0.9, ls=(0, (2.4, 2.0)), zorder=2)
    ax.annotate(r"yield stress $\tau_y$ = " + f"{ty:.1f} Pa" + "\n" +
                r"plastic viscosity $\eta_{pl}$ = " + f"{slope:.2f} " +
                r"Pa$\cdot$s",
                xy=(1.6e-2, ty), xytext=(1.6e-2, ty * 3.4),
                ha="left", va="bottom", fontsize=7.0, color=BLUE,
                linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7))

    ax.text(1.6e-2, 4.0e2,
            "solid + markers: shear stress (left)\n"
            r"dashed: apparent viscosity $\eta$ (right)",
            ha="left", va="top", fontsize=6.8, color=MUTED, linespacing=1.4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(8e-3, 1.4e3)
    ax.set_ylim(1.5e-2, 1.2e3)
    ax2.set_ylim(6e-3, 3e3)
    ax.set_xlabel(r"shear rate $\dot{\gamma}$ (s$^{-1}$)")
    ax.set_ylabel(r"shear stress $\tau$ (Pa)")
    ax2.set_ylabel(r"apparent viscosity $\eta$ (Pa$\cdot$s)", fontsize=8.0,
                   color=SUB)
    ax2.tick_params(labelsize=7.0, colors=SUB, which="both")
    ax.tick_params(which="both")
    ax.grid(which="major", lw=0.5, alpha=0.7)
    ax.legend(fontsize=7.2, frameon=False, loc="lower right", ncol=2,
              handlelength=1.7, columnspacing=1.0, borderaxespad=0.3)
    ax.set_title("Stress and apparent viscosity over five decades of rate",
                 fontsize=9.0, fontweight="bold", color=INK)
    return ty, slope


def index_panel(ax, spec):
    names = list(spec)
    vals = [spec[k]["n"] for k in names]
    cols = [spec[k]["color"] for k in names]
    ypos = np.arange(len(names))[::-1]
    ax.barh(ypos, vals, height=0.62, color=cols, alpha=0.88,
            edgecolor="white", linewidth=0.7, zorder=3)
    ax.axvline(1.0, color="#808080", lw=0.9, ls=(0, (3.0, 2.2)), zorder=4)
    ax.text(1.03, ypos[0] + 0.55, "Newtonian, n = 1", fontsize=6.4,
            color=SUB, ha="left", va="center")
    for y, v, c in zip(ypos, vals, cols):
        ax.text(v + 0.035, y, f"{v:.2f}", va="center", ha="left",
                fontsize=6.6, color=c)
    ax.set_yticks(ypos)
    ax.set_yticklabels([n.replace(" ", "\n", 0) for n in names], fontsize=6.5)
    ax.set_xlim(0.0, 1.62)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel("power-law index $n$ (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.7)
    ax.set_title(f"Index fitted over {FIT_LO:.0f}-{FIT_HI:.0f} " +
                 r"s$^{-1}$", fontsize=8.2, fontweight="bold", color=INK)


def yield_panel(ax, ramps):
    for name, d in ramps.items():
        ax.loglog(d["tau"], d["eta"], color=d["color"], lw=1.5, zorder=4,
                  label=name)
        ax.axvline(d["ty"], color=d["color"], lw=0.8, ls=(0, (2.2, 1.8)),
                   zorder=2)
        ax.text(d["ty"] * 1.12, 6e4, f"{d['ty']:.1f} Pa", rotation=90,
                ha="left", va="top", fontsize=6.3, color=d["color"])
    ax.set_xlim(0.4, 2.6e2)
    ax.set_ylim(1e-1, 2e5)
    ax.set_xlabel(r"applied shear stress $\tau$ (Pa)", fontsize=7.4)
    ax.set_ylabel(r"$\eta$ (Pa$\cdot$s)", fontsize=7.4)
    ax.tick_params(labelsize=6.5, which="both")
    ax.grid(which="major", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.4, frameon=False, loc="lower left",
              handlelength=1.4, borderaxespad=0.2)
    ax.set_title("Yield point from the stress ramp", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    g, spec = flow_curves()
    ramps = stress_ramp()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.42, hspace=0.62)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_idx = fig.add_subplot(gs[0, 1])
    ax_yld = fig.add_subplot(gs[1, 1])

    ty, etap = main_panel(ax_main, g, spec)
    index_panel(ax_idx, spec)
    yield_panel(ax_yld, ramps)

    fig.tight_layout(rect=[0, 0.045, 1, 0.945], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_main, ax_idx, ax_yld], dx=-0.42, dy=0.12)
    savefig(fig, "732_rhe_flow_curves.png")


if __name__ == "__main__":
    main()

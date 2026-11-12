"""Run-to-run variability of three training recipes over ten random seeds.

One panel per method: hairlines are the individual seeds, the heavy line is
the seed mean and the ribbon is +-1 sigma.  The three panels share one x and
y range - computed from every seed of every method before anything is drawn
- so the width of the ribbons can be compared directly.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

OURS = "#3b5b92"
GREY = "#8c8c8c"
ORANGE = "#c9a227"
GRID = "#eeeeee"

EP = np.arange(10, 151)
NSEED = 10

# method -> (colour, final mean, seed spread, time constant, jitter)
METHODS = [
    ("SGD baseline", GREY, 91.15, 0.95, 26.0, 0.62),
    ("SAM baseline", ORANGE, 92.60, 0.48, 23.0, 0.42),
    ("Ours", OURS, 94.30, 0.16, 19.0, 0.26),
]


def runs():
    rng = np.random.default_rng(311)
    t = EP.astype(float)
    out = {}
    for name, _, af, sd, tau, jit in METHODS:
        tr = np.empty((NSEED, t.size))
        for k in range(NSEED):
            a_inf = af + rng.normal(0, sd)
            tk = tau * (1.0 + 0.14 * rng.normal())
            base = a_inf - (a_inf - 42.0) * np.exp(-t / tk)
            wob = np.cumsum(rng.normal(0, 0.10 * jit / 0.3, t.size))
            wob -= wob * np.linspace(0, 1, t.size)
            tr[k] = base + wob + rng.normal(0, jit, t.size) * \
                np.exp(-t / 90.0)
        out[name] = tr
    return out


def main():
    set_pub_style(base_font=8)
    data = runs()

    lo, hi = np.inf, -np.inf
    for name, *_ in METHODS:
        tr = data[name]
        lo = min(lo, tr.min(), (tr.mean(0) - tr.std(0)).min())
        hi = max(hi, tr.max(), (tr.mean(0) + tr.std(0)).max())
    ylo = 5.0 * np.floor(lo / 5.0)
    yhi = np.ceil(hi) + 0.8
    yticks = np.arange(ylo + 5.0, yhi, 10.0)
    ours_final = data["Ours"].mean(0)[-1]
    print(f"  shared y range [{ylo:.0f}, {yhi:.0f}], x range "
          f"[{EP[0]}, {EP[-1]}]")

    fig = plt.figure(figsize=(7.4, 2.70))
    gs = fig.add_gridspec(1, 3, left=0.068, right=0.988, top=0.88,
                          bottom=0.18, wspace=0.075)

    for k, (name, col, *_rest) in enumerate(METHODS):
        ax = fig.add_subplot(gs[0, k])
        tr = data[name]
        mu, sd = tr.mean(0), tr.std(0)
        hot = name == "Ours"

        ax.set_axisbelow(True)
        ax.grid(color=GRID, lw=0.7, zorder=0)
        ax.axhline(ours_final, color=OURS, lw=0.7, ls=(0, (1.4, 1.8)),
                   alpha=0.55, zorder=2)

        for s in range(NSEED):
            ax.plot(EP, tr[s], color=col, lw=0.75, alpha=0.38, zorder=3)
        ax.fill_between(EP, mu - sd, mu + sd, color=col, alpha=0.22, lw=0,
                        zorder=4)
        ax.plot(EP, mu, color=col, lw=1.7 if hot else 1.5, zorder=6)

        fin = tr[:, -1]
        cv = 100.0 * fin.std() / fin.mean()
        ax.text(0.965, 0.075,
                f"${fin.mean():.2f} \\pm {fin.std():.2f}$ %\n"
                f"CV $=$ {cv:.2f} %   ($n{{=}}${NSEED})",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.0, color=col, linespacing=1.5,
                fontweight="bold" if hot else "normal", zorder=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#dcdcdc", lw=0.6, alpha=0.95))

        ax.set_xlim(EP[0], EP[-1])
        ax.set_ylim(ylo, yhi)
        ax.set_yticks(yticks)
        ax.set_xticks([25, 50, 75, 100, 125, 150])
        ax.set_title(name, fontsize=8.6, pad=3.5,
                     fontweight="bold" if hot else "normal",
                     color=OURS if hot else "#1a1a1a")
        ax.set_xlabel("Epoch", fontsize=8)
        if k:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Validation accuracy (%)", fontsize=8)
        add_panel_label(ax, f"({'abc'[k]})", x=-0.045 if k else -0.135,
                        y=1.02, size=9.5)
        style_axes(ax)
        print(f"  {name:14s} final {fin.mean():6.2f} +- {fin.std():.2f} "
              f"(CV {cv:.2f} %)  worst seed {fin.min():.2f}")

    savefig(fig, "311_convergence_seeds.png")


if __name__ == "__main__":
    main()

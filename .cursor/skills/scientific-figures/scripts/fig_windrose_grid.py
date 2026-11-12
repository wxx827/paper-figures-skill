"""2x3 grid of stacked wind roses for two stations and three periods.

Hourly wind is synthesised with a seasonally rotating prevailing direction
(von Mises spread) and a Weibull speed distribution whose scale grows with
alignment to that direction. Each polar panel stacks 16 direction sectors by
speed class; the donut hole reports the calm frequency. All panels share the
radial scale and the speed-class legend so the roses can be compared directly.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pubstyle import set_pub_style, savefig

# light-to-heavy speed classes: cool for gentle air, warm for gales
BINS = np.array([0.5, 2, 4, 6, 8, 10, 14, np.inf])
BIN_LABELS = ["0.5-2", "2-4", "4-6", "6-8", "8-10", "10-14", "$\\geq$14"]
BIN_COLORS = ["#dde8f0", "#b2cfe1", "#7fb0cb", "#4f88ac", "#33628c",
              "#e0a03f", "#b0432b"]
NSECT = 16
SECT = 2 * np.pi / NSECT
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

STATIONS = [
    dict(name="Coastal buoy, 52 m a.s.l.", short="Coastal buoy",
         mu_w=315.0, mu_s=200.0, kap_w=3.0, kap_s=1.6,
         sc_w=9.8, sc_s=6.2, calm=0.020),
    dict(name="Inland plateau, 1840 m a.s.l.", short="Inland plateau",
         mu_w=60.0, mu_s=240.0, kap_w=2.1, kap_s=2.7,
         sc_w=5.4, sc_s=7.6, calm=0.075),
]
PERIODS = ["DJF", "JJA", "Annual"]
NICE = [1, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]


def simulate(rng, st, nyear=8):
    """Hourly direction (rad, compass) and speed (m/s) for several years."""
    n = 8760 * nyear
    doy = (np.arange(n) / 24.0) % 365.0
    w = 0.5 + 0.5 * np.cos(2 * np.pi * (doy - 15.0) / 365.0)   # 1 = midwinter

    zw, zs = np.exp(1j * np.deg2rad(st["mu_w"])), np.exp(1j * np.deg2rad(st["mu_s"]))
    mu = np.angle(w * zw + (1 - w) * zs)
    kappa = st["kap_w"] * w + st["kap_s"] * (1 - w)
    scale = st["sc_w"] * w + st["sc_s"] * (1 - w)

    direction = np.mod(rng.vonmises(mu, kappa), 2 * np.pi)
    align = np.cos(direction - mu)
    speed = scale * (0.58 + 0.42 * align) * rng.weibull(2.1, n)
    speed = np.clip(speed, 0.0, None)

    calm_p = st["calm"] * (1.35 - 0.7 * w)          # calmer in summer
    calm = rng.random(n) < calm_p
    speed[calm] = rng.uniform(0.0, 0.5, calm.sum())
    return direction, speed, doy


def rose_table(direction, speed):
    """Percent of all observations per (sector, speed class), plus calm %."""
    total = speed.size
    calm_pct = 100.0 * np.mean(speed < BINS[0])
    live = speed >= BINS[0]
    sec = np.floor((direction[live] + SECT / 2) / SECT).astype(int) % NSECT
    cls = np.digitize(speed[live], BINS) - 1
    tab = np.zeros((NSECT, len(BIN_LABELS)))
    np.add.at(tab, (sec, cls), 1.0)
    return 100.0 * tab / total, calm_pct


def main():
    set_pub_style()
    rng = np.random.default_rng(83)

    data = {}
    for si, st in enumerate(STATIONS):
        d, s, doy = simulate(rng, st)
        masks = {
            "DJF": (doy < 59) | (doy >= 334),
            "JJA": (doy >= 151) & (doy < 243),
            "Annual": np.ones(d.size, bool),
        }
        for key, m in masks.items():
            data[(si, key)] = rose_table(d[m], s[m]) + (s[m].mean(),)

    peak = max(t.sum(1).max() for t, _, _ in data.values())
    step = next(v for v in NICE if v >= peak / 3.0)
    rings = np.arange(step, 3 * step + 0.1, step)
    rmax = 3 * step

    fig = plt.figure(figsize=(10.4, 7.3))
    gs = fig.add_gridspec(2, 3, left=0.035, right=0.985, top=0.875,
                          bottom=0.125, wspace=0.24, hspace=0.36)
    theta = np.arange(NSECT) * SECT

    for si, st in enumerate(STATIONS):
        for pi, key in enumerate(PERIODS):
            tab, calm_pct, vmean = data[(si, key)]
            ax = fig.add_subplot(gs[si, pi], projection="polar")
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_rorigin(-rmax * 0.22)

            bottom = np.zeros(NSECT)
            for ci in range(len(BIN_LABELS)):
                ax.bar(theta, tab[:, ci], width=SECT * 0.88, bottom=bottom,
                       color=BIN_COLORS[ci], edgecolor="white", linewidth=0.35,
                       zorder=3)
                bottom += tab[:, ci]

            ax.set_ylim(0, rmax)
            # anchor the radial labels on the emptiest sector of this panel
            quiet = float(np.degrees(theta[int(tab.sum(1).argmin())]))
            ax.set_rgrids(rings, labels=[f"{r:g}%" for r in rings],
                          angle=quiet, fontsize=6.0, color="#4a4a4a")
            for lbl in ax.get_yticklabels():
                lbl.set_bbox(dict(boxstyle="round,pad=0.10", fc="white",
                                  ec="none", alpha=0.75))
            ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 4))
            ax.set_xticklabels(COMPASS, fontsize=7.2)
            ax.tick_params(axis="x", pad=0.5)
            ax.grid(color="#cccccc", lw=0.5, ls=(0, (2, 2)))
            ax.set_axisbelow(True)
            ax.spines["polar"].set_color("#bbbbbb")
            ax.spines["polar"].set_linewidth(0.7)

            ax.text(0.5, 0.5, f"calm\n{calm_pct:.1f}%",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=6.2, color="#555555", linespacing=1.15)
            ax.set_title(f"{st['short']} - {key}", fontsize=8.8,
                         fontweight="bold", pad=11)
            ax.text(1.03, -0.02, f"mean {vmean:.1f} m s$^{{-1}}$",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=6.8, color="#555555")
            ax.text(-0.10, 1.05, f"({chr(97 + si * 3 + pi)})",
                    transform=ax.transAxes, fontsize=11, fontweight="bold",
                    va="bottom", ha="left")

    handles = [Patch(facecolor=c, edgecolor="white", linewidth=0.35, label=l)
               for c, l in zip(BIN_COLORS, BIN_LABELS)]
    leg = fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.058),
                     ncol=7, frameon=False, fontsize=7.6, handlelength=1.5,
                     handleheight=0.95, columnspacing=1.2, handletextpad=0.5,
                     title="Wind speed class (m s$^{-1}$)")
    leg.get_title().set_fontsize(8)
    leg.get_title().set_fontweight("bold")

    savefig(fig, "83_windrose_grid.png")


if __name__ == "__main__":
    main()

"""Radar / spider chart: multi-series comparison across service axes.

Filled semi-transparent polygons with vertex markers over shared axes.  Panel
(a) tracks one landscape through three census years, panel (b) contrasts the
three counties with the reserve average.  Legends sit below each plot so they
never overlap the polygons, and the radial tick labels are pushed into the
gap between two spokes.
"""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, add_panel_label

AXES = ["Water\nyield", "Carbon", "Habitat", "Soil\nretention",
        "NPP", "Recreation", "Air\nquality"]
NA = len(AXES)
YEAR_COLORS = ["#8fbfa8", "#2e7d5b", "#1d4f3c"]          # ordered ramp
COUNTY_COLORS = ["#3d6b8e", "#c9772f", "#8c2f2f", "#4a4a4a"]


def radar_axes(fig, pos):
    ax = fig.add_subplot(pos, projection="polar")
    ang = np.linspace(0, 2 * np.pi, NA, endpoint=False)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(ang)
    ax.set_xticklabels(AXES, fontsize=8)
    ax.tick_params(axis="x", pad=7)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6.2,
                       color="#8a8a8a")
    # place the radial labels in the gap between two spokes
    ax.set_rlabel_position(np.degrees(np.pi / NA) + 1)
    for lab in ax.get_yticklabels():
        lab.set_bbox(dict(boxstyle="square,pad=0.12", fc="white", ec="none",
                          alpha=0.8))
    ax.spines["polar"].set_color("#cccccc")
    ax.spines["polar"].set_linewidth(0.8)
    ax.grid(color="#dcdcdc", lw=0.7)
    ax.set_axisbelow(True)
    return ax, ang


def plot_series(ax, ang, vals, color, label, lw=1.8, fill=0.12):
    v = np.concatenate([vals, vals[:1]])
    a = np.concatenate([ang, ang[:1]])
    ax.plot(a, v, color=color, lw=lw, label=label, zorder=5)
    if fill:
        ax.fill(a, v, color=color, alpha=fill, zorder=2, linewidth=0)
    ax.scatter(ang, vals, color=color, s=18, zorder=6, edgecolors="white",
               linewidths=0.7)


def main():
    set_pub_style()
    rng = np.random.default_rng(61)
    fig = plt.figure(figsize=(10.4, 5.6))

    # (a) one landscape, three census years: services shift, they do not jump
    base = rng.uniform(0.34, 0.62, NA)
    delta = rng.uniform(-0.06, 0.14, NA)
    ax1, ang = radar_axes(fig, 121)
    for k, (name, col) in enumerate(zip(["2000", "2010", "2020"],
                                        YEAR_COLORS)):
        vals = np.clip(base + k * delta + rng.normal(0, 0.02, NA), 0.05, 0.98)
        plot_series(ax1, ang, vals, col, name)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3,
               fontsize=8, frameon=False, columnspacing=2.0)
    ax1.set_title("Ecosystem-service profile by census year", fontsize=9,
                  fontweight="bold", pad=24)

    # (b) counties deviate around the reserve mean
    ax2, ang = radar_axes(fig, 122)
    reserve = np.clip(base + 2 * delta, 0.08, 0.95)
    names = ["Antu", "Fusong", "Changbai", "Reserve mean"]
    for name, col in zip(names, COUNTY_COLORS):
        if name == "Reserve mean":
            plot_series(ax2, ang, reserve, col, name, lw=2.2, fill=0.10)
        else:
            vals = np.clip(reserve + rng.normal(0, 0.11, NA), 0.05, 0.98)
            plot_series(ax2, ang, vals, col, name, lw=1.4, fill=0.0)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4,
               fontsize=8, frameon=False, columnspacing=1.6)
    ax2.set_title("Service profile by county, 2020", fontsize=9,
                  fontweight="bold", pad=24)

    for ax, lab in zip([ax1, ax2], "ab"):
        add_panel_label(ax, f"({lab})", x=-0.12, y=1.06, size=10)
    fig.tight_layout(rect=(0, 0.03, 1, 1), w_pad=5.0)
    savefig(fig, "15_radar.png")


if __name__ == "__main__":
    main()

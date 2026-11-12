"""A gridded emission inventory and the two tables that always sit beside it.

The main panel is the map an inventory is delivered as: nitrogen-oxide
emissions on a one-kilometre grid over an eighty by sixty kilometre domain,
drawn on an equal-aspect axes and on a logarithmic colour scale because a
power station in a single cell and the rural background differ by three
orders of magnitude and a linear scale would show only the station.

The map is the sum of five sector fields, and the companion panels are the
two projections of those same fields.  The sector totals are the grid sums, so
their percentages add to one hundred by construction, and the monthly profiles
are the factors that redistribute each sector's annual total through the year -
each normalised to a mean of one, so the year they describe is the year the
map shows.

Illustrative synthetic inventory, not a reported national submission.
"""
import textwrap

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, style_axes, get_cmap, add_footer
from figkit import (axes_inches, panel_labels_inches, safe_colorbar,
                    halo_text, add_scale_bar)

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.66, 1.06, 5.35, 4.01)
AUX_T_RECT = (7.44, 3.40, 3.10, 1.80)
AUX_B_RECT = (7.44, 1.06, 3.10, 1.80)

NX, NY = 80, 60             # 1 km cells
CITY = (34.0, 31.0)
TOWN = (62.0, 45.0)
CORE_R = 9.0                # km, radius of the urban core used in the note

SECTORS = ["power", "industry", "road transport", "residential",
           "agriculture"]
SEC_COL = {"power": "#b23a48", "industry": "#8c5a3c",
           "road transport": "#c9a227", "residential": "#3b5b92",
           "agriculture": "#2e7d5b"}

# Monthly redistribution factors, later renormalised to a mean of one.
PROFILE = {
    "power": [1.22, 1.16, 1.02, 0.88, 0.84, 0.94, 1.10, 1.08, 0.90, 0.88,
              1.02, 1.18],
    "industry": [0.94, 0.90, 1.04, 1.06, 1.08, 1.06, 0.88, 0.82, 1.08, 1.10,
                 1.06, 0.98],
    "road transport": [0.95, 0.96, 1.02, 1.03, 1.04, 1.05, 0.98, 0.94, 1.05,
                       1.04, 1.01, 0.96],
    "residential": [1.94, 1.78, 1.32, 0.82, 0.44, 0.26, 0.22, 0.24, 0.44,
                    0.92, 1.48, 1.86],
    "agriculture": [0.52, 0.62, 1.34, 1.74, 1.42, 1.06, 0.88, 0.82, 1.06,
                    1.24, 0.86, 0.58],
}
MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def blob(X, Y, x0, y0, r):
    """Isotropic Gaussian blob on the grid, peak one."""
    return np.exp(-0.5 * (((X - x0) ** 2 + (Y - y0) ** 2) / r ** 2))


def corridor(X, Y, p0, p1, width):
    """Distance-decayed ribbon along a segment, peak one on the centreline."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L2 = float(d @ d)
    t = np.clip(((X - p0[0]) * d[0] + (Y - p0[1]) * d[1]) / L2, 0.0, 1.0)
    dx = X - (p0[0] + t * d[0])
    dy = Y - (p0[1] + t * d[1])
    return np.exp(-0.5 * (dx ** 2 + dy ** 2) / width ** 2)


def build_fields(rng):
    """Five sector fields in t/km2/yr, and the point sources that drive one."""
    x = np.arange(NX) + 0.5
    y = np.arange(NY) + 0.5
    X, Y = np.meshgrid(x, y)

    urban = blob(X, Y, *CITY, CORE_R) + 0.55 * blob(X, Y, *TOWN, 5.0)
    roads = (corridor(X, Y, (2.0, 20.0), (78.0, 40.0), 1.6)
             + 0.8 * corridor(X, Y, (34.0, 2.0), (34.0, 58.0), 1.4)
             + 0.6 * corridor(X, Y, CITY, TOWN, 1.2))

    fields = {}
    fields["road transport"] = 46.0 * roads * (0.30 + 0.95 * urban) + 1.1
    fields["residential"] = 62.0 * urban + 0.9
    fields["agriculture"] = (3.1 * (1.0 - np.clip(urban, 0.0, 1.0))
                             * (0.6 + 0.5 * np.sin(X / 11.0)
                                * np.cos(Y / 9.0) + 0.5))
    ind = np.zeros_like(X)
    for _ in range(9):
        ind += rng.uniform(0.3, 1.0) * blob(X, Y, rng.uniform(8, 74),
                                            rng.uniform(6, 54), 2.6)
    fields["industry"] = 34.0 * ind + 0.6

    stacks, power = [], np.zeros_like(X)
    for _ in range(4):
        px, py = rng.uniform(6, 74), rng.uniform(5, 55)
        q = rng.uniform(1.6e3, 5.2e3)
        i, j = int(py), int(px)
        power[i, j] += q
        stacks.append((px, py, q))
    fields["power"] = power + 0.25
    return X, Y, fields, stacks


def check_layout(fig, foot, axes):
    """Warn if the footnote box touches an axis label, tick row or title."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for a in axes:
        ab = a.get_window_extent()
        arts = [a.xaxis.label, a.yaxis.label, a.title]
        arts += [t for t in a.get_xticklabels()
                 if ab.x0 - 3.0 <= t.get_window_extent().x0 <= ab.x1 + 3.0]
        for art in arts:
            if art.get_text() and fb.overlaps(art.get_window_extent()):
                print("WARNING: footnote overlaps:", art.get_text()[:44])


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(1012)
    X, Y, fields, stacks = build_fields(rng)
    total = sum(fields.values())
    tot_kt = {s: fields[s].sum() / 1.0e3 for s in SECTORS}
    grand = sum(tot_kt.values())
    share = {s: 100.0 * tot_kt[s] / grand for s in SECTORS}

    core = ((X - CITY[0]) ** 2 + (Y - CITY[1]) ** 2) <= CORE_R ** 2
    core_share = 100.0 * total[core].sum() / total.sum()
    core_area = 100.0 * core.sum() / total.size

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_c = axes_inches(fig, AUX_T_RECT)
    ax_d = axes_inches(fig, AUX_B_RECT)
    for a in (ax_c, ax_d):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: the gridded map ----------------------------------------------
    ax.set_aspect("equal")
    v_hi = float(np.percentile(total, 99.5))
    norm = mpl.colors.LogNorm(vmin=1.5, vmax=v_hi)
    im = ax.pcolormesh(np.arange(NX + 1), np.arange(NY + 1), total,
                       cmap=get_cmap("cb_sequential_warm"), norm=norm,
                       shading="flat", zorder=2)
    ax.contour(X, Y, total, levels=[30.0, 90.0], colors="#5a4a40",
               linewidths=0.55, alpha=0.75, zorder=4)
    for px, py, q in stacks:
        ax.plot([px], [py], marker="^", ms=4.6, mfc="white", mec=INK, mew=0.9,
                zorder=7)
    big = max(stacks, key=lambda s: s[2])
    halo_text(ax, big[0], big[1] + 1.6,
              f"largest point source, {big[2] / 1e3:.1f} kt/yr in one cell",
              ha="center", va="bottom", fontsize=6.4, color=INK, halo="white",
              halo_lw=2.2, zorder=8)
    circ = plt.Circle(CITY, CORE_R, facecolor="none", edgecolor="#2e7d5b",
                      lw=1.2, ls=(0, (4.0, 2.4)), zorder=6)
    ax.add_patch(circ)
    halo_text(ax, CITY[0], CITY[1] - CORE_R - 1.4,
              f"urban core, r = {CORE_R:.0f} km", ha="center", va="top",
              fontsize=6.4, color="#2e7d5b", halo="white", halo_lw=2.2,
              zorder=8)
    add_scale_bar(ax, 10.0, "10 km", loc=(0.035, 0.055), height=0.017,
                  fontsize=6.2)

    ax.set_xlim(0.0, NX)
    ax.set_ylim(0.0, NY)
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel("easting (km)", fontsize=8.4, labelpad=1.5)
    ax.set_ylabel("northing (km)", fontsize=8.4)
    ax.set_title(f"Gridded $NO_x$ emissions, 1 km cells, "
                 f"{grand:.1f} kt/yr over the domain", fontsize=9.0,
                 fontweight="bold", pad=5.0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.012, 0.978,
            f"the urban core is {core_area:.0f} % of the area and "
            f"{core_share:.0f} % of the emissions\ntriangles are the four "
            "power stations; contours at 30 and 90 $t/km^2/yr$\ncolour "
            f"saturates above {v_hi:.0f} $t/km^2/yr$ so the city stays "
            "readable",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
            color=INK, linespacing=1.50, zorder=12,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.93))
    safe_colorbar(fig, im, ax, label="$t/km^2/yr$", width=0.13, pad=0.09,
                  shrink=0.90, labelsize=7.0, extend="max")

    # ---- aux (b): sector totals ---------------------------------------------
    order = sorted(SECTORS, key=lambda s: tot_kt[s])
    ypos = np.arange(len(SECTORS))
    ax_c.barh(ypos, [tot_kt[s] for s in order], height=0.62,
              color=[SEC_COL[s] for s in order], edgecolor="white",
              linewidth=0.6, zorder=4)
    for y, s in zip(ypos, order):
        ax_c.text(tot_kt[s] + grand * 0.012, y,
                  f"{tot_kt[s]:.1f} kt/yr  ({share[s]:.1f} %)", ha="left",
                  va="center", fontsize=6.3, color=INK)
    ax_c.set_yticks(ypos)
    ax_c.set_yticklabels(order, fontsize=6.8)
    ax_c.set_xlim(0.0, max(tot_kt.values()) * 1.62)
    ax_c.set_ylim(-1.42, len(SECTORS) - 0.38)
    ax_c.tick_params(labelsize=6.8)
    ax_c.set_xlabel("annual emission (kt/yr)", fontsize=7.5, labelpad=1.5)
    ax_c.set_title("The map, summed by sector", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_c.grid(axis="x", color=GRID, lw=0.55, alpha=0.75, zorder=0)
    ax_c.text(0.985, 0.025,
              f"bars are grid sums of the five sector fields;\nthe shares add "
              f"to {sum(share.values()):.0f} % and the total to "
              f"{grand:.1f} kt/yr",
              transform=ax_c.transAxes, ha="right", va="bottom", fontsize=6.3,
              color=SUB, linespacing=1.45)

    # ---- aux (c): monthly profiles ------------------------------------------
    mon = np.arange(12)
    prof = {s: np.array(PROFILE[s]) / np.mean(PROFILE[s]) for s in SECTORS}
    dom = sum(prof[s] * tot_kt[s] for s in SECTORS) / grand
    for s in SECTORS:
        ax_d.plot(mon, prof[s], color=SEC_COL[s], lw=1.35, marker="o",
                  ms=2.2, zorder=5, label=s)
    ax_d.plot(mon, dom, color=INK, lw=1.9, zorder=7, label="domain total")
    ax_d.axhline(1.0, color=FAINT, lw=0.8, ls=(0, (3.0, 2.2)), zorder=2)
    j_hi = int(np.argmax(dom))
    ax_d.plot([mon[j_hi]], [dom[j_hi]], marker="o", ms=4.4, mfc="white",
              mec=INK, mew=1.3, zorder=8)
    ax_d.text(0.985, 0.965,
              "black curve is the emission-weighted\ntotal; it peaks at "
              f"{dom[j_hi]:.2f} in month {MONTHS[j_hi]}",
              transform=ax_d.transAxes, ha="right", va="top", fontsize=6.3,
              color=SUB, linespacing=1.45)
    ax_d.set_xlim(-0.5, 11.5)
    ax_d.set_ylim(0.0, 2.95)
    ax_d.set_xticks(mon)
    ax_d.set_xticklabels(MONTHS, fontsize=6.6)
    ax_d.tick_params(labelsize=6.8)
    ax_d.set_xlabel("month of the inventory year", fontsize=7.5, labelpad=1.5)
    ax_d.set_ylabel("monthly factor (mean 1.00)", fontsize=7.5)
    ax_d.set_title("How the year is redistributed", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_d.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)
    ax_d.legend(loc="upper left", frameon=False, fontsize=6, ncol=2,
                handlelength=1.2, columnspacing=0.9, labelspacing=0.20,
                borderaxespad=0.20)

    panel_labels_inches(fig, [ax, ax_c, ax_d], dx=-0.56, dy=0.10)
    foot = add_footer(fig, "Synthetic data, seed=1012")
    check_layout(fig, foot, [ax, ax_c, ax_d])
    savefig(fig, "1012_aqm_emission_inventory.png")


if __name__ == "__main__":
    main()

"""Space-time congestion heat map for an arterial monitoring set.

Eighteen directional corridors (rows) are profiled over a full weekday in
15-minute bins (columns).  Rows are ordered by the clock time of their
own daily maximum, which turns the morning- and evening-dominated groups
into two clean blocks and makes the tidal pattern of the network obvious.
The marginal panels are deliberately subordinate: the network-average
trace on top is 19 % of the map's height and the daily congested-hours
bars on the right are 26 % of its width.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pubstyle import set_pub_style, savefig

RAMP = ["#f5f7f4", "#d5e2cf", "#9dbb92", "#dcc077", "#d1834b", "#a8433a",
        "#6e2a25"]
INK = "#2b2b2b"
GREY = "#8c8c8c"
THRESH = 4.5                      # congestion index counted as "congested"

CORRIDORS = [
    ("Airport Expy", 0.95), ("North Ring", 0.80), ("East Ring", 0.72),
    ("South Ring", 0.68), ("West Ring", 0.62), ("Central Ave", 1.00),
    ("Riverside Rd", 0.58), ("University Rd", 0.66), ("Harbour Link", 0.74),
]


def profile(t, am_w, pm_w, mid_w, am_t, pm_t):
    return (am_w * np.exp(-0.5 * ((t - am_t) / 0.95) ** 2)
            + pm_w * np.exp(-0.5 * ((t - pm_t) / 1.25) ** 2)
            + mid_w * np.exp(-0.5 * ((t - 13.0) / 2.40) ** 2)
            + 0.25 * np.exp(-0.5 * ((t - 21.0) / 1.60) ** 2))


def main():
    set_pub_style()
    rng = np.random.default_rng(31)

    nb = 96
    t = (np.arange(nb) + 0.5) * 24.0 / nb
    rows, names = [], []
    for name, wgt in CORRIDORS:
        for d, tag in [(0, "IB"), (1, "OB")]:
            am = (2.6 if d == 0 else 0.9) * wgt
            pm = (1.1 if d == 0 else 2.9) * wgt
            am_t = rng.normal(8.05, 0.30)
            pm_t = rng.normal(17.85, 0.40)
            y = profile(t, am, pm, 0.55 * wgt, am_t, pm_t)
            y = 9.4 * y / 3.05 + 0.55
            y = y * np.exp(rng.normal(0, 0.055, nb))
            y += rng.normal(0, 0.10, nb)
            rows.append(np.clip(y, 0.15, 10.0))
            names.append(f"{name} {tag}")
    ci = np.array(rows)

    # a single incident: sustained breakdown on one outbound corridor
    inc_r, inc_a, inc_b = 11, 58, 70
    ci[inc_r, inc_a:inc_b] = np.clip(
        ci[inc_r, inc_a:inc_b] + 4.6 * np.hanning(inc_b - inc_a) + 1.4,
        0.15, 10.0)

    order = np.argsort([t[np.argmax(r)] for r in ci])
    ci = ci[order]
    names = [names[i] for i in order]
    inc_row = int(np.where(order == inc_r)[0][0])

    cmap = mpl.colors.LinearSegmentedColormap.from_list("cong", RAMP)
    norm = mpl.colors.Normalize(0, 10)
    n = len(names)

    fig = plt.figure(figsize=(10.2, 6.45))
    axm = fig.add_axes([0.140, 0.098, 0.585, 0.637])
    axt = fig.add_axes([0.140, 0.800, 0.585, 0.140])
    axb = fig.add_axes([0.742, 0.098, 0.152, 0.637])
    cax = fig.add_axes([0.930, 0.098, 0.012, 0.637])

    # ------------------------------------------------------- (b) heat map
    im = axm.imshow(ci, cmap=cmap, norm=norm, aspect="auto", origin="upper",
                    extent=(0, 24, n, 0), interpolation="nearest", zorder=2)
    for k in range(1, n):
        axm.axhline(k, color="white", lw=0.35, zorder=3)
    axm.set_yticks(np.arange(n) + 0.5)
    axm.set_yticklabels(names, fontsize=6.5)
    axm.set_xticks(np.arange(0, 25, 3))
    axm.set_xticklabels([f"{h:02d}" for h in range(0, 25, 3)], fontsize=7.0)
    axm.set_xlabel("time of day  (h)", labelpad=2)
    axm.tick_params(length=2.4, pad=2.0)
    axm.tick_params(axis="y", length=0)
    for sp in axm.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.7)
        sp.set_edgecolor("#b9b5ac")

    # peak-time group brackets
    pk = np.array([t[np.argmax(r)] for r in ci])
    split = int(np.searchsorted(pk, 12.0))
    for y0, y1, lab, col in [(0, split, "morning-dominant", "#3f6d8a"),
                             (split, n, "evening-dominant", "#a8624a")]:
        axm.plot([-0.42, -0.42], [y0 + 0.18, y1 - 0.18], color=col, lw=2.2,
                 clip_on=False, solid_capstyle="butt", zorder=6)
        axm.text(-0.75, (y0 + y1) / 2, lab, rotation=90, ha="center",
                 va="center", fontsize=6.8, fontweight="bold", color=col,
                 clip_on=False, zorder=6)

    # incident call-out
    axm.add_patch(Rectangle((inc_a * 24 / nb, inc_row + 0.04),
                            (inc_b - inc_a) * 24 / nb, 0.92,
                            facecolor="none", edgecolor="#3a3a3a", lw=1.0,
                            zorder=5))
    axm.annotate("lane-blocking incident\n15:00-18:00",
                 xy=((inc_b - 1) * 24 / nb, inc_row + 0.5),
                 xytext=(21.4, inc_row + 3.6), fontsize=6.8, color=INK,
                 fontweight="bold", ha="right", va="center", linespacing=1.25,
                 zorder=7,
                 bbox=dict(boxstyle="round,pad=0.22", fc="white",
                           ec="#d8d5cf", lw=0.6, alpha=0.93),
                 arrowprops=dict(arrowstyle="-|>", lw=0.75, color="#3a3a3a",
                                 shrinkA=3, shrinkB=3,
                                 connectionstyle="arc3,rad=0.25"))

    # ------------------------------------------- (a) network-average trace
    mean_ci = ci.mean(axis=0)
    axt.fill_between(t, 0, mean_ci, color="#c9736a", alpha=0.16, lw=0,
                     zorder=2)
    axt.plot(t, mean_ci, color="#a8433a", lw=1.5, zorder=3)
    axt.axhline(THRESH, color=GREY, lw=0.7, ls=(0, (3.5, 2.2)), zorder=2)
    axt.text(0.22, THRESH + 0.25, "congested threshold", fontsize=6.3,
             color="#7b7b7b", va="bottom")
    for lab, lo, hi in [("AM peak", 5.0, 11.0), ("PM peak", 14.0, 21.0)]:
        seg = (t >= lo) & (t <= hi)
        k = np.argmax(np.where(seg, mean_ci, -1))
        axt.plot([t[k]], [mean_ci[k]], marker="o", ms=3.6, mfc="white",
                 mec="#a8433a", mew=1.1, zorder=5)
        hh = int(t[k]); mm = int(round((t[k] - hh) * 60))
        axt.annotate(f"{lab}  {hh:02d}:{mm:02d}\nCI {mean_ci[k]:.1f}",
                     xy=(t[k], mean_ci[k]), xytext=(t[k] - 1.5, 8.6),
                     fontsize=6.6, color="#8a3a30", fontweight="bold",
                     ha="right" if lab == "AM peak" else "left",
                     va="top", linespacing=1.25, zorder=6,
                     arrowprops=dict(arrowstyle="-", lw=0.7, color="#c9a49e",
                                     shrinkA=2, shrinkB=3))
    axt.set_xlim(0, 24)
    axt.set_ylim(0, 10)
    axt.set_xticks(np.arange(0, 25, 3))
    axt.set_xticklabels([])
    axt.set_yticks([0, 5, 10])
    axt.set_ylabel("network\nmean CI", fontsize=7.4, labelpad=3,
                   linespacing=1.25)
    axt.tick_params(length=2.4, labelsize=6.8)
    axt.grid(axis="x", color="#f0f0f0", lw=0.5, zorder=0)
    for sp in ("top", "right"):
        axt.spines[sp].set_visible(False)

    # ------------------------------------------ (c) daily congested hours
    hours = (ci > THRESH).sum(axis=1) * 24.0 / nb
    ypos = np.arange(n) + 0.5
    bcol = cmap(norm(np.clip(hours / hours.max() * 9.2, 0, 10)))
    axb.barh(ypos, hours, height=0.74, color=bcol, edgecolor="white",
             linewidth=0.4, zorder=3)
    for y, h in zip(ypos, hours):
        axb.text(h + 0.10, y, f"{h:.1f}", va="center", ha="left",
                 fontsize=6.2, color="#5c5c5c", zorder=4)
    axb.axvline(hours.mean(), color=INK, lw=0.9, ls=(0, (3.5, 2.2)), zorder=5)
    axb.set_ylim(n, 0)
    axb.set_xlim(0, hours.max() * 1.42)
    axb.text(hours.max() * 1.40, 0.55, f"mean {hours.mean():.1f} h",
             fontsize=6.4, color=INK, fontweight="bold", ha="right",
             va="center", zorder=6,
             bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                       alpha=0.88))
    axb.set_yticks([])
    axb.set_xticks(np.arange(0, hours.max() * 1.42, 2))
    axb.set_xlabel("congested\nhours per day", fontsize=7.4, labelpad=2,
                   linespacing=1.25)
    axb.tick_params(length=2.4, labelsize=6.8)
    axb.grid(axis="x", color="#efefef", lw=0.5, zorder=0)
    for sp in ("top", "right", "left"):
        axb.spines[sp].set_visible(False)

    cb = fig.colorbar(im, cax=cax, ticks=[0, 2, 4, 6, 8, 10])
    cb.set_label("congestion index  (0 = free flow, 10 = gridlock)",
                 fontsize=7.2, labelpad=3)
    cb.ax.tick_params(labelsize=6.8, length=2.2, width=0.7)
    cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor("#555555")

    # ---- panel tags on one shared inch offset ---------------------------
    for ax, tag, ttl in [(axt, "(a)", "Network-average congestion"),
                         (axm, "(b)", "Corridor x time congestion index"),
                         (axb, "(c)", "Daily exposure")]:
        pos = ax.get_position()
        fig.text(pos.x0 - 0.52 / fig.get_figwidth(),
                 pos.y1 + 0.115 / fig.get_figheight(), tag, fontsize=10.5,
                 fontweight="bold", ha="left", va="baseline", color=INK)
        fig.text(pos.x0 - 0.16 / fig.get_figwidth(),
                 pos.y1 + 0.115 / fig.get_figheight(), ttl, fontsize=8.8,
                 fontweight="bold", ha="left", va="baseline", color="#33474f")

    savefig(fig, "208_congestion_heatmap.png")


if __name__ == "__main__":
    main()

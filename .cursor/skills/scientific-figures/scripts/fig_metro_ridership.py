"""Metro network schematic with station loading and a daily gate profile.

The octilinear network diagram carries five lines drawn with white
casings so crossings layer correctly; plain stations are line-coloured
dots sized by daily boardings, interchanges are white rings.  The two
smaller panels on the right report the mirrored entry / exit profile at
the busiest interchange and the ten heaviest stations, each bar tinted
with its own line colour so the ranking ties back to the map without a
separate legend.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
from pubstyle import set_pub_style, savefig

# ---- series-wide colour key ----------------------------------------------
LCOL = ["#b06a52", "#40708f", "#6f9464", "#d3a04c", "#8a7aa3"]
LNAME = ["L1", "L2", "L3", "L4", "L5"]
INK = "#2b2b2b"
GREY = "#8c8c8c"
ENTRY, EXIT = "#40708f", "#b06a52"

LINES = [
    [(0.8, 4.6), (2.2, 4.6), (3.4, 5.8), (6.6, 5.8), (7.8, 4.6), (9.2, 4.6)],
    [(5.0, 9.2), (5.0, 0.8)],
    [(1.2, 1.2), (8.8, 8.8)],
    [(0.8, 8.4), (3.0, 8.4), (4.2, 7.2), (8.0, 7.2), (8.0, 1.2)],
    [(0.9, 2.0), (3.0, 2.0), (4.2, 3.2), (6.4, 3.2), (7.6, 2.0), (9.1, 2.0)],
]
NAMES = ["Riverside", "Old Town", "Cathedral", "Market Gate", "Harbour",
         "Airport Link", "University", "Stadium", "Grand Plaza", "North Gate",
         "Exchange", "Museum", "South Quay", "Garden City", "West End",
         "Bridgehead", "Civic Centre", "East Quay", "Fairground", "Arsenal",
         "Kings Wharf", "Belmont", "Highfield", "Clockwork", "Saltmarsh",
         "Ironworks", "Lakeside", "Foundry", "Windmill", "Orchard",
         "Beacon Hill", "Silver Row", "Tannery", "Elmwood", "Portside",
         "Granary", "Copperfield", "Marlow", "Redbrick", "Sandhill"]


def resample(pts, spacing=0.92):
    pts = np.asarray(pts, float)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.arange(0.0, cum[-1] + 1e-9, spacing)
    if cum[-1] - s[-1] > 0.35 * spacing:
        s = np.append(s, cum[-1])
    else:
        s[-1] = cum[-1]
    return np.column_stack([np.interp(s, cum, pts[:, 0]),
                            np.interp(s, cum, pts[:, 1])])


def build_stations(rng):
    raw = []
    for li, pts in enumerate(LINES):
        for p in resample(pts):
            raw.append([p[0], p[1], li])
    clusters = []
    for x, y, li in raw:
        for c in clusters:
            if np.hypot(c["x"] - x, c["y"] - y) < 0.45:
                c["lines"].add(li)
                break
        else:
            clusters.append({"x": x, "y": y, "lines": {li}})
    for c in clusters:
        r = np.hypot(c["x"] - 5.0, c["y"] - 5.0)
        base = 34.0 * np.exp(-(r / 3.4) ** 1.7) + 3.0
        base *= 1.0 + 0.55 * (len(c["lines"]) - 1)
        c["rid"] = base * rng.lognormal(0.0, 0.24)
        c["hub"] = len(c["lines"]) > 1
    order = np.argsort([-c["rid"] for c in clusters])
    for rank, i in enumerate(order):
        clusters[i]["name"] = NAMES[rank % len(NAMES)]
    return clusters


def gate_profile(t, rng, peak_in, peak_out):
    e = (0.35 + 3.10 * np.exp(-0.5 * ((t - peak_in) / 0.80) ** 2)
         + 1.05 * np.exp(-0.5 * ((t - 12.6) / 1.90) ** 2)
         + 2.35 * np.exp(-0.5 * ((t - peak_out) / 1.05) ** 2)
         + 0.55 * np.exp(-0.5 * ((t - 21.4) / 1.40) ** 2))
    return np.clip(e * (1 + rng.normal(0, 0.035, t.size)), 0.05, None)


def main():
    set_pub_style()
    rng = np.random.default_rng(17)
    st = build_stations(rng)

    fig = plt.figure(figsize=(10.6, 6.35))
    axn = fig.add_axes([0.012, 0.050, 0.560, 0.855])
    axg = fig.add_axes([0.640, 0.578, 0.325, 0.310])
    axr = fig.add_axes([0.640, 0.078, 0.325, 0.330])

    # ------------------------------------------------------ (a) network map
    axn.set_facecolor("white")
    for li, pts in enumerate(LINES):
        p = np.asarray(pts, float)
        axn.plot(p[:, 0], p[:, 1], color="white", lw=5.6, zorder=2 + li * 0.1,
                 solid_capstyle="round", solid_joinstyle="round")
    for li, pts in enumerate(LINES):
        p = np.asarray(pts, float)
        axn.plot(p[:, 0], p[:, 1], color=LCOL[li], lw=3.1,
                 zorder=2.5 + li * 0.1, solid_capstyle="round",
                 solid_joinstyle="round")
        for end, off in [(p[0], p[1] - p[0]), (p[-1], p[-2] - p[-1])]:
            d = off / (np.hypot(*off) + 1e-9)
            axn.text(end[0] - 0.42 * d[0], end[1] - 0.42 * d[1], LNAME[li],
                     ha="center", va="center", fontsize=6.8,
                     fontweight="bold", color="white", zorder=9,
                     bbox=dict(boxstyle="round,pad=0.24", fc=LCOL[li],
                               ec="white", lw=0.8))

    rid = np.array([c["rid"] for c in st])
    for c in st:
        s = 14 + 132 * (c["rid"] / rid.max()) ** 1.25
        if c["hub"]:
            axn.plot([c["x"]], [c["y"]], marker="o", ms=np.sqrt(s) * 0.72,
                     mfc="white", mec="#3a3a3a", mew=1.25, zorder=7)
        else:
            li = list(c["lines"])[0]
            axn.plot([c["x"]], [c["y"]], marker="o", ms=np.sqrt(s) * 0.56,
                     mfc=LCOL[li], mec="white", mew=0.9, zorder=6)

    # greedy label placement: pick the candidate corner that sits farthest
    # from every station and from the labels already committed
    top = sorted(st, key=lambda c: -c["rid"])[:6]
    placed = []
    allxy = np.array([[c["x"], c["y"]] for c in st])
    for c in top:
        best, best_score = None, -1e9
        for dx, dy, ha, va in [(0.36, 0.30, "left", "bottom"),
                               (0.36, -0.30, "left", "top"),
                               (-0.36, 0.30, "right", "bottom"),
                               (-0.36, -0.30, "right", "top")]:
            ax_, ay_ = c["x"] + dx, c["y"] + dy
            if not (0.2 < ax_ < 9.9 and 0.2 < ay_ < 9.9):
                continue
            d_st = np.hypot(allxy[:, 0] - ax_, allxy[:, 1] - ay_)
            d_st = np.sort(d_st)[1]
            d_lab = min([np.hypot(px - ax_, py - ay_) for px, py in placed],
                        default=9.0)
            score = min(d_st, 1.4) + 1.6 * min(d_lab, 2.2)
            if score > best_score:
                best_score, best = score, (ax_, ay_, ha, va)
        ax_, ay_, ha, va = best
        placed.append((ax_, ay_))
        axn.text(ax_, ay_, c["name"], ha=ha, va=va, fontsize=6.6,
                 fontweight="bold", color="#2f3a42", zorder=10,
                 bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none",
                           alpha=0.88))

    axn.set_xlim(-0.15, 10.15)
    axn.set_ylim(-1.95, 10.15)
    axn.set_aspect("equal")
    axn.set_xticks([]); axn.set_yticks([])
    for sp in axn.spines.values():
        sp.set_visible(False)

    hs = [Line2D([], [], ls="none", marker="o", ms=m, mfc="#7d7d7d",
                 mec="white", mew=0.8,
                 label=f"{v:.0f}k boardings/day")
          for m, v in [(3.2, rid.min()), (5.4, np.median(rid)),
                       (8.0, rid.max())]]
    hs.append(Line2D([], [], ls="none", marker="o", ms=7.0, mfc="white",
                     mec="#3a3a3a", mew=1.25, label="interchange"))
    leg = axn.legend(handles=hs, loc="lower center",
                     bbox_to_anchor=(0.5, -0.008), ncol=4, fontsize=6.6,
                     columnspacing=1.5, handletextpad=0.6, borderpad=0.5,
                     frameon=False)
    leg.set_clip_on(False)

    # --------------------------------------------- (b) gate entry / exit
    t = np.linspace(0, 24, 289)
    hub = max((c for c in st if c["hub"]), key=lambda c: c["rid"])
    # scale so the daily integral matches the station's boardings
    pe = gate_profile(t, rng, 7.9, 17.8)
    px = gate_profile(t, rng, 9.1, 18.6)
    ent = pe * (3.0 * hub["rid"] / pe.sum())
    ext = px * (3.0 * hub["rid"] * 0.96 / px.sum())
    axg.fill_between(t, 0, ent, color=ENTRY, alpha=0.28, lw=0, zorder=2)
    axg.plot(t, ent, color=ENTRY, lw=1.4, zorder=3)
    axg.fill_between(t, 0, -ext, color=EXIT, alpha=0.28, lw=0, zorder=2)
    axg.plot(t, -ext, color=EXIT, lw=1.4, zorder=3)
    axg.axhline(0, color="#4a4a4a", lw=0.8, zorder=4)
    lim = max(ent.max(), ext.max()) * 1.30
    axg.set_ylim(-lim, lim)
    axg.set_xlim(0, 24)
    axg.set_xticks(np.arange(0, 25, 4))
    axg.set_xticklabels([f"{h:02d}" for h in range(0, 25, 4)], fontsize=6.8)
    axg.yaxis.set_major_locator(MaxNLocator(nbins=6, symmetric=True))
    axg.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{abs(v):g}"))
    axg.set_xlabel("time of day  (h)", labelpad=2, fontsize=7.6)
    axg.set_ylabel("passengers  (10$^3$ / 15 min)", labelpad=2, fontsize=7.6)
    axg.text(0.30, lim * 0.90, "entries", fontsize=7.0, fontweight="bold",
             color=ENTRY, va="top")
    axg.text(0.30, -lim * 0.90, "exits", fontsize=7.0, fontweight="bold",
             color=EXIT, va="bottom")
    axg.annotate("AM commute\nasymmetry", xy=(8.2, ent.max() * 0.72),
                 xytext=(11.4, lim * 0.86), fontsize=6.6, color="#5a6570",
                 ha="left", va="top", linespacing=1.25,
                 arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#b3bcc2",
                                 shrinkA=2, shrinkB=3,
                                 connectionstyle="arc3,rad=0.25"))
    axg.text(23.7, -lim * 0.92, hub["name"] + " interchange", fontsize=6.6,
             color=GREY, ha="right", va="bottom", fontstyle="italic")
    axg.grid(axis="x", color="#f0f0f0", lw=0.5, zorder=0)
    axg.tick_params(length=2.4)
    for sp in ("top", "right"):
        axg.spines[sp].set_visible(False)

    # ------------------------------------------------- (c) top-10 stations
    tops = sorted(st, key=lambda c: -c["rid"])[:10][::-1]
    y = np.arange(len(tops))
    vals = [c["rid"] for c in tops]
    cols = [("#6f6f6f" if c["hub"] else LCOL[list(c["lines"])[0]])
            for c in tops]
    vmax = max(vals)
    axr.barh(y, vals, height=0.70, color=cols, alpha=0.90, edgecolor="white",
             linewidth=0.4, zorder=3)
    for yi, c, col in zip(y, tops, cols):
        axr.text(c["rid"] + vmax * 0.012, yi, f"{c['rid']:.0f}", va="center",
                 ha="left", fontsize=6.5, color="#5c5c5c", zorder=4)
        tag = "+".join(LNAME[i] for i in sorted(c["lines"]))
        axr.text(-vmax * 0.020, yi, f"{c['name']}  {tag}", va="center",
                 ha="right", fontsize=6.5, color=col, fontweight="bold",
                 zorder=5)
    axr.set_ylim(-0.65, len(tops) - 0.35)
    axr.set_xlim(-vmax * 0.60, vmax * 1.12)
    axr.set_yticks([])
    axr.set_xticks(np.arange(0, vmax * 1.12, 20))
    axr.spines["bottom"].set_bounds(0, vmax * 1.12)
    axr.set_xlabel("daily boardings  (10$^3$ passengers)", labelpad=2,
                   fontsize=7.6)
    axr.tick_params(length=2.4, labelsize=6.8)
    axr.grid(axis="x", color="#efefef", lw=0.5, zorder=0)
    for sp in ("top", "right", "left"):
        axr.spines[sp].set_visible(False)

    # ---- panel tags, one shared inch offset ------------------------------
    for ax, tag, ttl, dx in [(axn, "(a)", "Metro network and station "
                              "loading", 0.30),
                             (axg, "(b)", "Gate flow at the busiest "
                              "interchange", -0.44),
                             (axr, "(c)", "Ten heaviest stations", -0.44)]:
        pos = ax.get_position()
        fig.text(pos.x0 + dx / fig.get_figwidth(),
                 pos.y1 + 0.115 / fig.get_figheight(), tag, fontsize=10.5,
                 fontweight="bold", ha="left", va="baseline", color=INK)
        fig.text(pos.x0 + (dx + 0.36) / fig.get_figwidth(),
                 pos.y1 + 0.115 / fig.get_figheight(), ttl, fontsize=8.8,
                 fontweight="bold", ha="left", va="baseline", color="#33474f")

    savefig(fig, "209_metro_ridership.png")


if __name__ == "__main__":
    main()

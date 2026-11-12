"""Stay-point extraction: where people stop, for how long, and when.

Stops detected from a synthetic trajectory set are drawn over the street
network with marker area encoding dwell time and colour encoding the
inferred activity type.  Because dwell times span two orders of magnitude -
a two-minute transfer and a nine-hour work stay are the same variable - the
duration histogram uses a log axis rather than a linear one that would pile
everything into the first bin.  The polar rose repeats the same stops by
start hour, so the map's spatial story gets its temporal counterpart.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

from pubstyle import set_pub_style, savefig, get_cmap, CATEGORICAL, COLORS
from figkit import add_scale_bar, panel_labels_inches, unclip_legend

W, H = 6.6, 6.0                      # city extent in km
INK = COLORS["ink"]
SUB = COLORS["sub"]
MUTED = COLORS["muted"]
ROAD_MAJOR = "#b4b0a8"
ROAD_MINOR = "#cbc8c1"
LAND = "#fbfaf8"
WATER = "#ccd8de"
PARK = "#e7ede2"
FRAME = "#c9c6bf"

# name, colour, centres (x, y, spread), median dwell (min), sigma, peak hour
ACTIVITY = [
    ("work", CATEGORICAL[1],
     [(3.20, 3.30, 0.70), (2.30, 4.35, 0.45), (4.55, 2.35, 0.50)],
     250.0, 0.42, 9.0, 76),
    ("shopping", CATEGORICAL[3],
     [(4.15, 4.35, 0.42), (1.55, 2.10, 0.38), (5.35, 1.65, 0.34)],
     62.0, 0.55, 15.5, 58),
    ("dining", CATEGORICAL[0],
     [(3.55, 2.35, 0.36), (2.05, 3.55, 0.30), (4.85, 3.60, 0.30)],
     46.0, 0.48, 19.0, 52),
    ("transfer", CATEGORICAL[2],
     [(2.60, 1.35, 0.26), (4.40, 4.85, 0.24), (1.20, 4.20, 0.22)],
     9.5, 0.60, 8.0, 84),
]


def build_network(rng):
    """Jittered street grid plus a ring road; returns polylines and widths."""
    nx, ny = 16, 15
    gx = np.linspace(0.20, W - 0.20, nx)
    gy = np.linspace(0.22, H - 0.22, ny)
    X, Y = np.meshgrid(gx, gy)
    r = np.hypot(X - W / 2, Y - H / 2)
    jit = 0.020 + 0.048 * np.clip(r / 3.0, 0, 1) ** 1.5
    X = X + rng.normal(0, 1, X.shape) * jit
    Y = Y + rng.normal(0, 1, Y.shape) * jit

    def width(i, j):
        if i % 5 == 0 or j % 5 == 0:
            return 1.25
        if i % 2 == 0 or j % 2 == 0:
            return 0.70
        return 0.46

    segs, lws = [], []
    for i in range(ny):
        for j in range(nx - 1):
            w = width(i, j)
            if w < 0.5 and rng.random() < 0.24:
                continue
            segs.append(np.array([[X[i, j], Y[i, j]],
                                  [X[i, j + 1], Y[i, j + 1]]]))
            lws.append(w)
    for i in range(ny - 1):
        for j in range(nx):
            w = width(i, j)
            if w < 0.5 and rng.random() < 0.24:
                continue
            segs.append(np.array([[X[i, j], Y[i, j]],
                                  [X[i + 1, j], Y[i + 1, j]]]))
            lws.append(w)

    th = np.linspace(0, 2 * np.pi, 240)
    rr = 1.88 + 0.10 * np.cos(3 * th + 0.5) + 0.05 * np.cos(5 * th)
    ring = np.c_[W / 2 + rr * np.cos(th), H / 2 + rr * np.sin(th)]
    for a in range(0, 232, 8):
        segs.append(ring[a:a + 9])
        lws.append(1.25)
    return segs, np.array(lws)


def build_stays(rng):
    """One record per detected stop: position, duration, type, start hour."""
    xs, ys, dur, kind, hour = [], [], [], [], []
    for k, (_n, _c, centres, med, sig, peak, count) in enumerate(ACTIVITY):
        for i in range(count):
            cx, cy, sp = centres[i % len(centres)]
            xs.append(np.clip(cx + rng.normal(0, sp), 0.15, W - 0.15))
            ys.append(np.clip(cy + rng.normal(0, sp), 0.15, H - 0.15))
            dur.append(med * np.exp(rng.normal(0.0, sig)))
            kind.append(k)
            h = peak + rng.normal(0.0, 2.4) + (3.0 if rng.random() < 0.22
                                               else 0.0)
            hour.append(h % 24.0)
    return (np.array(xs), np.array(ys), np.array(dur), np.array(kind),
            np.array(hour))


def main():
    set_pub_style()
    rng = np.random.default_rng(484)
    segs, lws = build_network(rng)
    xs, ys, dur, kind, hour = build_stays(rng)

    fig = plt.figure(figsize=(10.4, 6.4))
    axm = fig.add_axes([0.032, 0.062, 0.578, 0.818])
    axh = fig.add_axes([0.672, 0.585, 0.300, 0.295])
    axp = fig.add_axes([0.678, 0.088, 0.288, 0.330], projection="polar")

    # ------------------------------------------------------------- base map
    axm.add_patch(Rectangle((0, 0), W, H, facecolor=LAND, edgecolor="none",
                            zorder=0))
    ry = np.linspace(-0.2, H + 0.2, 220)
    rx = 5.15 + 0.48 * np.sin(ry / 1.55) + 0.14 * np.sin(ry / 0.55)
    axm.plot(rx, ry, color=WATER, lw=7.0, solid_capstyle="round", zorder=1)
    for cx, cy, r0 in [(1.25, 1.40, 0.50), (2.55, 4.85, 0.40)]:
        th = np.linspace(0, 2 * np.pi, 60)
        rad = r0 * (1 + 0.16 * np.cos(3 * th + cx) + 0.08 * np.cos(5 * th))
        axm.add_patch(Polygon(np.c_[cx + rad * np.cos(th),
                                    cy + rad * np.sin(th)],
                              facecolor=PARK, edgecolor="#d5ddcd", lw=0.6,
                              zorder=1))
    axm.add_collection(LineCollection(
        segs, colors=np.where(lws > 0.5, ROAD_MAJOR, ROAD_MINOR),
        linewidths=lws, capstyle="round", zorder=2))

    # ---------------------------------------------------------- stay points
    size = 4.0 + 116.0 * (dur / dur.max()) ** 0.62
    order = np.argsort(-size)                    # big stays underneath
    axm.scatter(xs[order], ys[order], s=size[order],
                c=[ACTIVITY[k][1] for k in kind[order]], alpha=0.72,
                edgecolors="white", linewidths=0.35, zorder=4)

    axm.set_xlim(0, W)
    axm.set_ylim(0, H)
    axm.set_aspect("equal")
    axm.set_xticks([])
    axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
        sp.set_edgecolor(FRAME)
    add_scale_bar(axm, 1.0, "1 km", loc=(0.032, 0.042), height=0.014,
                  color="#333333", text_color="#1a1a1a", halo="white",
                  fontsize=6.2)

    hk = [Line2D([], [], marker="o", ls="none", ms=4.6, mfc=ACTIVITY[k][1],
                 mec="white", mew=0.4, label=ACTIVITY[k][0])
          for k in range(4)]
    leg1 = axm.legend(handles=hk, loc="upper left",
                      bbox_to_anchor=(0.012, 0.988), fontsize=6.7,
                      title="activity type", labelspacing=0.42,
                      handletextpad=0.5, borderpad=0.45, frameon=True,
                      framealpha=0.92, edgecolor="#dcd9d2", facecolor="white")
    leg1.get_title().set_fontsize(6.9)
    leg1.get_title().set_fontweight("bold")
    leg1.get_frame().set_linewidth(0.6)
    leg1.set_zorder(10)

    refs = [15.0, 60.0, 240.0]
    hs = [Line2D([], [], marker="o", ls="none",
                 ms=np.sqrt(4.0 + 116.0 * (r / dur.max()) ** 0.62),
                 mfc="#9a9a9a", mec="white", mew=0.4,
                 label=f"{r:.0f} min") for r in refs]
    leg2 = axm.legend(handles=hs, loc="lower right",
                      bbox_to_anchor=(0.988, 0.012), fontsize=6.7,
                      title="dwell time", labelspacing=0.75,
                      handletextpad=0.6, borderpad=0.5, frameon=True,
                      framealpha=0.92, edgecolor="#dcd9d2", facecolor="white")
    leg2.get_title().set_fontsize(6.9)
    leg2.get_title().set_fontweight("bold")
    leg2.get_frame().set_linewidth(0.6)
    axm.add_artist(leg1)          # the second legend() call evicted the first
    unclip_legend(leg1)           # ... and a re-added legend gets clipped
    leg2.set_zorder(10)

    axm.text(0.988, 0.965, f"{len(dur)} stay points, one weekday",
             transform=axm.transAxes, ha="right", va="top", fontsize=6.4,
             color=SUB, zorder=9)

    # ------------------------------------------------- (b) dwell histogram
    bins = np.logspace(np.log10(2.0), np.log10(dur.max() * 1.08), 26)
    bottom = np.zeros(bins.size - 1)
    ctr = np.sqrt(bins[:-1] * bins[1:])
    widths = np.diff(bins) * 0.92
    for k in range(4):
        cnt, _ = np.histogram(dur[kind == k], bins=bins)
        axh.bar(ctr, cnt, width=widths, bottom=bottom, color=ACTIVITY[k][1],
                edgecolor="white", linewidth=0.3, zorder=3)
        bottom = bottom + cnt
    med = float(np.median(dur))
    axh.axvline(med, color="#333333", lw=1.0, zorder=5)
    axh.text(0.030, 0.965, f"vertical rule: median {med:.0f} min",
             transform=axh.transAxes, fontsize=6.5, color=INK, ha="left",
             va="top", zorder=6)
    axh.set_xscale("log")
    axh.set_xlim(bins[0], bins[-1])
    axh.set_ylim(0, bottom.max() * 1.20)
    axh.set_xlabel("dwell time (min, log scale)", fontsize=7.4, labelpad=2)
    axh.set_ylabel("stay points (count)", fontsize=7.4, labelpad=2)
    axh.set_title("Dwell-time distribution", fontsize=8.6, fontweight="bold",
                  color=INK, pad=4)
    axh.tick_params(labelsize=6.6, length=2.4)
    axh.grid(axis="y", color="#ececec", lw=0.55, zorder=0)

    # ------------------------------------------------- (c) start-hour rose
    edges = np.arange(25)
    cnt, _ = np.histogram(hour, bins=edges)
    theta = np.deg2rad(edges[:-1] * 15.0)
    cmap = get_cmap("cb_sequential_blue")
    cols = cmap(0.22 + 0.72 * cnt / cnt.max())
    axp.bar(theta, cnt, width=np.deg2rad(13.4), bottom=0.0, color=cols,
            edgecolor="white", linewidth=0.4, align="edge", zorder=3)
    axp.set_theta_zero_location("N")
    axp.set_theta_direction(-1)
    axp.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    axp.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)],
                        fontsize=6.4)
    axp.set_rlabel_position(322)
    axp.tick_params(axis="y", labelsize=6.2, pad=0.5)
    axp.set_ylim(0, cnt.max() * 1.10)
    axp.grid(color="#e2dfd9", lw=0.55)
    axp.spines["polar"].set_color("#cfccc5")
    axp.spines["polar"].set_linewidth(0.7)
    peak_h = int(np.argmax(cnt))
    axp.set_title("Stay starts by hour", fontsize=8.6, fontweight="bold",
                  color=INK, pad=7)
    axp.text(0.5, -0.155, f"radius: stays per hour (count), peak "
             f"{cnt.max()} at {peak_h:02d}:00", transform=axp.transAxes,
             ha="center", va="top", fontsize=6.3, color=MUTED)

    panel_labels_inches(fig, [axm, axh, axp], dx=-0.20, dy=0.30)

    savefig(fig, "484_stm_dwell_stay.png")


if __name__ == "__main__":
    main()

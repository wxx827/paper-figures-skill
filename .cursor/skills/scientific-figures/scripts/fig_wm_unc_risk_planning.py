"""Figure 1182 - Risk-aware planning: three candidates over one risk field.

The only figure of this set allowed to spend warm colour: high collision risk
has to read as red.

One risk field and three polylines drive everything.  ``risk_field`` builds a
potential from the predicted obstacles and the road edges; each candidate is
sampled against that field, and the three cost terms are measurements of the
polyline itself:

    safety     = mean risk along the path      (the field in panel (a))
    efficiency = extra path length + how far it ends from the route line
    comfort    = mean |curvature| of the path

so the stacked bars in (c) cannot drift away from the picture in (a), and the
curves in (b) are the integrand of the safety term.  The selected candidate is
``argmin`` of the totals, not a hand-picked winner.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

from pubstyle import set_pub_style, savefig, style_axes

SEED = 1587  # original id; figure number is 1182
X0, X1, Y0, Y1 = 0.0, 46.0, -8.0, 8.0
NX, NY = 300, 110
LANE = 3.5
EDGE = 5.25

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#d5d5d5"
LOW, MED, HIGH = "#3b5b92", "#c9a227", "#b23a48"
C_SAFE, C_EFF, C_COMF = "#b23a48", "#3b5b92", "#2e7d5b"
GRID, CARD = "#ededed", "#e2e8ee"
# Warm is the point of this figure - red has to mean "do not go here" - but a
# fully warm ramp turns the whole board sepia and blows the batch budget, so
# the empty two thirds of the board stay cool and only the cores go red.
RISK_CMAP = LinearSegmentedColormap.from_list(
    "wm_risk", ["#eef4f9", "#d5e5f0", "#eadfd0", "#eec089", "#dd8f4a",
                "#b23a48"])

# (x, y, along sigma, cross sigma, weight, label, half length, half width)
OBSTACLES = [(26.0, 0.15, 5.2, 1.55, 1.00, "lead vehicle, braking", 2.1, 0.85),
             (17.0, -3.40, 4.0, 1.40, 0.72, "right lane, steady", 2.1, 0.85),
             (36.0, 2.15, 3.4, 1.20, 0.80, "motorcycle between lanes",
              1.1, 0.38)]
# weights of the three cost terms - printed in the figure, so they are data
W_SAFE, W_EFF, W_COMF = 1.00, 0.55, 0.45


def risk_field():
    x = np.linspace(X0, X1, NX)
    y = np.linspace(Y0, Y1, NY)
    X, Y = np.meshgrid(x, y)
    R = np.zeros_like(X)
    for (ox, oy, sx, sy, wt, _, _, _) in OBSTACLES:
        R += wt * np.exp(-0.5 * (((X - ox) / sx) ** 2 + ((Y - oy) / sy) ** 2))
    R += 0.30 * np.exp(-0.5 * ((np.abs(Y) - (EDGE + 2.15)) / 1.10) ** 2)
    return x, y, R


def sample(R, x, y, px, py):
    """Bilinear-free nearest lookup of the field along a polyline."""
    ix = np.clip(np.searchsorted(x, px) - 1, 0, NX - 1)
    iy = np.clip(np.searchsorted(y, py) - 1, 0, NY - 1)
    return R[iy, ix]


def candidates():
    """Three lateral profiles over the same longitudinal sweep."""
    s = np.linspace(X0 + 1.5, X1 - 1.0, 260)

    def sig(a, c, w):
        return a / (1.0 + np.exp(-(s - c) / w))

    return [("A", 0.0 * s, "hold lane"),
            ("B", sig(6.8, 15.0, 3.0) - sig(1.4, 34.0, 3.0),
             "shoulder swing"),
            ("C", sig(3.5, 17.5, 3.6), "lane change")], s


def costs(R, x, y, s, lat):
    """Three measurements of one polyline against one field.

    The safety term is deliberately superlinear in the *peak* risk: one
    near-contact is not traded away by a long stretch of empty road, which is
    what makes a straight line through the lead vehicle unacceptable rather
    than merely mediocre.
    """
    r = sample(R, x, y, s, lat)
    peak, avg = float(r.max()), float(r.mean())
    safety = 1.60 * peak ** 1.6 + 0.50 * avg
    ds = np.hypot(np.diff(s), np.diff(lat))
    extra = float(ds.sum() / (s[-1] - s[0]) - 1.0)
    efficiency = 26.0 * extra + 0.055 * abs(float(lat[-1]))
    kappa = np.abs(np.gradient(np.gradient(lat, s), s))
    comfort = 62.0 * float(kappa.mean())
    return r, peak, safety, efficiency, comfort


def main():
    set_pub_style(base_font=8.5)
    x, y, R = risk_field()
    cands, s = candidates()

    rows = []
    for (name, lat, note) in cands:
        r, pk, sa, ef, co = costs(R, x, y, s, lat)
        rows.append(dict(name=name, lat=lat, note=note, r=r, peak=pk,
                         safe=W_SAFE * sa, eff=W_EFF * ef,
                         comf=W_COMF * co))
    for d in rows:
        d["total"] = d["safe"] + d["eff"] + d["comf"]
        # Every total printed on the page is the sum of the three *printed*
        # terms, not a separately rounded number - otherwise a reader adding
        # up the segment labels lands 0.01 away from the bar label.
        d["s2"], d["e2"], d["c2"] = (round(d["safe"], 2), round(d["eff"], 2),
                                     round(d["comf"], 2))
        d["t2"] = d["s2"] + d["e2"] + d["c2"]
        # colour is a statement about peak risk, not about the decision
        d["col"] = HIGH if d["peak"] > 0.60 else (MED if d["peak"] > 0.35
                                                  else LOW)
        d["band"] = ("high" if d["peak"] > 0.60 else
                     ("moderate" if d["peak"] > 0.35 else "low"))
    best = int(np.argmin([d["total"] for d in rows]))
    assert best == int(np.argmin([d["t2"] for d in rows])), \
        "rounding changes which candidate wins"
    assert rows[best]["col"] != HIGH, "the plan picked a high-risk path"
    assert len({d["col"] for d in rows}) == 3, "the three risk bands collapsed"

    fig = plt.figure(figsize=(10.4, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.40],
                          height_ratios=[1.0, 0.46], left=0.055, right=0.985,
                          top=0.845, bottom=0.115, wspace=0.20, hspace=0.42)
    axb = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[1, 0])
    axc = fig.add_subplot(gs[:, 1])

    # ---- (a) bird's-eye view ------------------------------------------------
    im = axb.imshow(R, origin="lower", extent=[X0, X1, Y0, Y1], aspect="auto",
                    cmap=RISK_CMAP, vmin=0.0,
                    vmax=float(R.max()), interpolation="bilinear", zorder=1)
    for ly in (-EDGE, EDGE):
        axb.plot([X0, X1], [ly, ly], color="#4d4d4d", lw=1.4, zorder=3)
    for ly in (-LANE / 2 - LANE / 2, LANE / 2 + LANE / 2):
        axb.plot([X0, X1], [ly, ly], color="#6f6f6f", lw=0.9,
                 ls=(0, (7.0, 6.0)), zorder=3)

    for (ox, oy, _, _, _, _, hw, hh) in OBSTACLES:
        axb.add_patch(FancyBboxPatch((ox - hw, oy - hh), 2 * hw, 2 * hh,
                                     boxstyle="round,pad=0.06,rounding_size=0.3",
                                     facecolor="#3a3a3a", edgecolor="white",
                                     lw=0.8, zorder=6))
    axb.add_patch(FancyBboxPatch((0.6, -0.85), 4.2, 1.7,
                                 boxstyle="round,pad=0.06,rounding_size=0.4",
                                 facecolor="#0b3d63", edgecolor="white",
                                 lw=0.9, zorder=7))
    axb.text(2.7, -1.55, "ego", ha="center", va="top", fontsize=7.2,
             color="#0b3d63", fontweight="bold", zorder=8,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                       alpha=0.92))
    for (ox, oy, _, _, _, lab, _, hh), dy in zip(OBSTACLES,
                                                 (-1.35, -1.35, -1.05)):
        axb.text(ox, oy + dy, lab, ha="center", va="top", fontsize=6.8,
                 color=INK, zorder=8,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                           alpha=0.92))

    for d in rows:
        lw = 2.6 if d is rows[best] else 1.7
        axb.plot(s, d["lat"], color=d["col"], lw=lw, zorder=8,
                 solid_capstyle="round")
        axb.plot(s, d["lat"], color="white", lw=lw + 1.6, zorder=7,
                 solid_capstyle="round", alpha=0.65)
        axb.text(s[-1] + 0.6, d["lat"][-1], d["name"], ha="left",
                 va="center", fontsize=8.6, color=d["col"],
                 fontweight="bold", zorder=9)
    axb.set_xlim(X0, X1 + 3.0)
    axb.set_ylim(Y0, Y1)
    axb.set_yticks([-EDGE, -LANE, 0, LANE, EDGE])
    axb.set_xlabel("distance ahead of the ego  (m)", fontsize=8.4)
    axb.set_ylabel("lateral offset  (m)", fontsize=8.4)
    axb.tick_params(labelsize=7.2, length=2.4)
    axb.set_title("(a)  Predicted risk field and three candidate "
                  "trajectories", fontsize=9.2, fontweight="bold", pad=4.0)
    cb = fig.colorbar(im, ax=axb, fraction=0.030, pad=0.012)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.set_ticks([0.0, 0.5 * R.max(), R.max()])
    cb.set_ticklabels(["0", f"{0.5 * R.max():.2f}", f"{R.max():.2f}"])
    cb.ax.tick_params(labelsize=6.8, width=0.7, length=2.2)
    cb.set_label("collision risk", fontsize=7.4, color=SUB)

    handles = [Line2D([], [], color=HIGH, lw=2.0, label="high risk"),
               Line2D([], [], color=MED, lw=2.0, label="moderate risk"),
               Line2D([], [], color=LOW, lw=2.0, label="low risk")]
    leg = axb.legend(handles=handles, loc="lower left",
                     bbox_to_anchor=(0.006, 0.02), frameon=True,
                     facecolor="white", edgecolor=HAIR, framealpha=0.94,
                     fontsize=7.2, handlelength=1.6, labelspacing=0.32,
                     borderpad=0.45, ncol=3, columnspacing=0.9)
    leg.get_frame().set_linewidth(0.5)
    leg.set_zorder(10)

    # ---- (b) the integrand of the safety term -------------------------------
    style_axes(axr)
    axr.set_axisbelow(True)
    axr.grid(color=GRID, lw=0.7, zorder=0)
    for d in rows:
        axr.plot(s, d["r"], color=d["col"], lw=1.9 if d is rows[best] else 1.3,
                 zorder=4)
    axr.set_xlim(X0, X1 + 2.2)
    axr.set_ylim(0, float(max(d["r"].max() for d in rows)) * 1.22)
    axr.set_xlabel("distance ahead of the ego  (m)", fontsize=8.4)
    axr.set_ylabel("risk along\nthe path", fontsize=8.0)
    axr.tick_params(labelsize=7.2, length=2.4)
    axr.set_title("(b)  The same field, read off each trajectory",
                  fontsize=8.8, fontweight="bold", pad=4.0)
    axr.text(0.012, 0.94, "safety cost = 1.60 $\\cdot$ peak$^{1.6}$ + 0.50 "
             "$\\cdot$ mean", transform=axr.transAxes, ha="left", va="top",
             fontsize=7.2, color=INK, zorder=7,
             bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=HAIR,
                       lw=0.5, alpha=0.95))
    for d in rows:
        k = int(np.argmax(d["r"]))
        axr.plot([s[k], s[k]], [d["r"][k], d["r"][k]], marker="o", ms=3.6,
                 mfc="white", mec=d["col"], mew=1.1, zorder=6)
    axr.text(0.985, 0.92, "peaks line up with the obstacles in (a)",
             transform=axr.transAxes, ha="right", va="top", fontsize=7.0,
             color=MUTED, zorder=7,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.3,
                       alpha=0.92))

    # ---- (c) cost breakdown -------------------------------------------------
    style_axes(axc)
    axc.set_axisbelow(True)
    axc.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    xs = np.arange(len(rows))
    bot = np.zeros(len(rows))
    for key, col, lab in ((("safe"), C_SAFE, "safety"),
                          (("eff"), C_EFF, "efficiency"),
                          (("comf"), C_COMF, "comfort")):
        vals = np.array([d[key] for d in rows])
        axc.bar(xs, vals, bottom=bot, width=0.60, color=col,
                edgecolor="white", linewidth=0.7, zorder=3, label=lab)
        for xi, (v, b) in enumerate(zip(vals, bot)):
            if v > 0.09:
                axc.text(xi, b + v / 2, f"{v:.2f}", ha="center", va="center",
                         fontsize=7.0, color="white", fontweight="bold",
                         zorder=5)
        bot = bot + vals
    for xi, d in enumerate(rows):
        axc.text(xi, d["total"] + 0.028, f"{d['t2']:.2f}", ha="center",
                 va="bottom", fontsize=8.0, color=INK,
                 fontweight="bold" if xi == best else "normal", zorder=5)
    axc.plot([best - 0.36, best + 0.36],
             [rows[best]["total"] + 0.115, rows[best]["total"] + 0.115],
             color="#0b3d63", lw=1.3, zorder=6)
    axc.text(best, rows[best]["total"] + 0.145, "selected", ha="center",
             va="bottom", fontsize=7.6, color="#0b3d63", fontweight="bold",
             zorder=6)
    axc.set_xlim(-0.62, len(rows) - 0.38)
    axc.set_ylim(0, max(d["total"] for d in rows) * 1.30)
    axc.set_xticks(xs)
    axc.set_xticklabels([f"{d['name']}\n{d['note']}" for d in rows],
                        fontsize=7.4, linespacing=1.6)
    for lab, d in zip(axc.get_xticklabels(), rows):
        lab.set_color(d["col"])
    axc.get_xticklabels()[best].set_fontweight("bold")
    axc.set_ylabel("planning cost  (lower is better)", fontsize=8.4)
    axc.tick_params(labelsize=7.2, length=2.4)
    axc.set_title("(c)  Cost breakdown", fontsize=9.2, fontweight="bold",
                  pad=4.0)
    pat = [Patch(facecolor=C_SAFE, edgecolor="white",
                 label=f"safety   $w$ = {W_SAFE:.2f}"),
           Patch(facecolor=C_EFF, edgecolor="white",
                 label=f"efficiency   $w$ = {W_EFF:.2f}"),
           Patch(facecolor=C_COMF, edgecolor="white",
                 label=f"comfort   $w$ = {W_COMF:.2f}")]
    leg2 = axc.legend(handles=pat, loc="upper center",
                      bbox_to_anchor=(0.5, -0.115), frameon=False,
                      fontsize=7.2, handlelength=1.3, labelspacing=0.35,
                      ncol=1)
    leg2.set_zorder(6)

    # ---- headline + arithmetic ---------------------------------------------
    w = rows[best]
    ss, se, sc = f"{w['s2']:.2f}", f"{w['e2']:.2f}", f"{w['c2']:.2f}"
    tot = float(ss) + float(se) + float(sc)
    assert abs(tot - w["t2"]) < 1e-9, \
        "the printed cost terms do not add up to the printed total"
    safest = int(np.argmin([d["safe"] for d in rows]))
    savefig(fig, "1182_wm_unc_risk_planning.png")


if __name__ == "__main__":
    main()

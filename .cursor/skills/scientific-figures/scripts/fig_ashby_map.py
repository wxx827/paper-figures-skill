"""Ashby material-selection chart: strength versus density (log-log).

Each material family is a cloud of members enclosed by a smoothed convex hull
("bubble").  Dashed guide lines carry the material indices sigma/rho,
sigma^(2/3)/rho and sigma^(1/2)/rho, and a handful of engineering alloys and
ceramics are called out by name.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
from pubstyle import set_pub_style, savefig

# label, log10 rho centre/sd, log10 sigma centre/sd, correlation, n, colour
FAMILIES = [
    ("Foams", 2.00, 0.30, -0.70, 0.50, 0.88, 26, "#8fb0c4"),
    ("Elastomers", 3.01, 0.055, 0.95, 0.32, 0.30, 20, "#b58fb0"),
    ("Polymers", 3.05, 0.075, 1.72, 0.22, 0.35, 30, "#6b9ec4"),
    ("Natural\nmaterials", 2.62, 0.25, 1.35, 0.38, 0.90, 26, "#7f9e6b"),
    ("Composites", 3.24, 0.050, 2.58, 0.24, 0.25, 22, "#c48a4a"),
    ("Technical\nceramics", 3.52, 0.105, 3.02, 0.28, 0.15, 24, "#b0604a"),
    ("Metals &\nalloys", 3.76, 0.175, 2.48, 0.36, 0.10, 34, "#5f6f80"),
]

# family label anchor (rho, sigma) and text alignment
FAM_LABEL = {
    "Foams": (115, 0.42, "center"),
    "Elastomers": (1010, 1.9, "center"),
    "Polymers": (1150, 235, "center"),
    "Natural\nmaterials": (430, 13, "center"),
    "Composites": (1520, 2300, "center"),
    "Technical\nceramics": (1900, 8200, "right"),
    "Metals &\nalloys": (10500, 25, "center"),
}

# named members are forced inside their family hull so the two agree
ANCHORS = {
    "Natural\nmaterials": [(160, 22), (900, 130)],
    "Elastomers": [(950, 12)],
    "Polymers": [(1310, 105)],
    "Composites": [(1550, 900)],
    "Technical\nceramics": [(3900, 2600)],
    "Metals &\nalloys": [(2700, 310), (4430, 950), (7850, 1620)],
}

NAMED = [
    ("Balsa", 160, 22, 6, -10),
    ("Nat. rubber", 950, 12, -14, -12),
    ("PEEK", 1310, 105, 12, -12),
    ("CFRP", 1550, 900, -16, -14),
    ("Al 6061-T6", 2700, 310, 5, 9),
    ("Ti–6Al–4V", 4430, 950, 12, 8),
    ("4340 steel", 7850, 1620, 10, 12),
    ("Al$_2$O$_3$", 3900, 2600, -13, 9),
]


def cloud(rng, lr, sr, ls, ss, rho_c, n):
    cov = np.array([[sr ** 2, rho_c * sr * ss], [rho_c * sr * ss, ss ** 2]])
    return rng.multivariate_normal([lr, ls], cov, size=n)


def bubble(pts, expand=1.16, n=260):
    hull = ConvexHull(pts)
    v = pts[hull.vertices]
    cen = v.mean(axis=0)
    v = cen + (v - cen) * expand
    v = np.vstack([v, v[:1]])
    tck, _ = splprep([v[:, 0], v[:, 1]], s=0, per=True,
                     k=3 if len(v) > 3 else 2)
    u = np.linspace(0.0, 1.0, n)
    return splev(u, tck)


def main():
    set_pub_style(base_font=9)
    rng = np.random.default_rng(23)

    fig, ax = plt.subplots(figsize=(7.5, 5.7))

    for name, lr, sr, ls, ss, corr, n, col in FAMILIES:
        pts = cloud(rng, lr, sr, ls, ss, corr, n)
        anchors = ANCHORS.get(name, [])
        hull_pts = np.vstack([pts, np.log10(anchors)]) if anchors else pts
        bx, by = bubble(hull_pts)
        ax.fill(10 ** np.asarray(bx), 10 ** np.asarray(by), facecolor=col,
                alpha=0.17, lw=0, zorder=2)
        ax.plot(10 ** np.asarray(bx), 10 ** np.asarray(by), color=col,
                lw=1.0, zorder=3)
        ax.plot(10 ** pts[:, 0], 10 ** pts[:, 1], ls="none", marker="o",
                ms=2.1, mfc=col, mec="white", mew=0.3, alpha=0.85, zorder=4)
        rx, ry, ha = FAM_LABEL[name]
        ax.text(rx, ry, name, color=col, fontsize=7.8, fontweight="bold",
                ha=ha, va="center", zorder=8, linespacing=1.15,
                path_effects=[pe.withStroke(linewidth=2.2,
                                            foreground="white")])

    for lab, rho, sig, dx, dy in NAMED:
        ax.plot(rho, sig, marker="o", ms=3.6, mfc="white", mec="#2b2b2b",
                mew=0.8, ls="none", zorder=9)
        ax.annotate(lab, xy=(rho, sig), xytext=(dx, dy),
                    textcoords="offset points", fontsize=6.4,
                    color="#1a1a1a", zorder=10,
                    ha="left" if dx > 0 else "right",
                    va="bottom" if dy > 0 else "top",
                    path_effects=[pe.withStroke(linewidth=2.0,
                                                foreground="white")])

    # --- material-index guide lines ---------------------------------------
    rho_g = np.array([12.0, 340.0])
    for expo, lab, col in ((1.0, "$\\sigma_f/\\rho$", "#4d4d4d"),
                           (1.5, "$\\sigma_f^{2/3}/\\rho$", "#4d4d4d"),
                           (2.0, "$\\sigma_f^{1/2}/\\rho$", "#4d4d4d")):
        c = 26.0 / 14.0 ** expo
        y = c * rho_g ** expo
        ax.plot(rho_g, y, color=col, lw=0.8, ls=(0, (5, 2.5)), zorder=5)
        ax.text(rho_g[1] * 1.08, y[1], lab, fontsize=7, color=col,
                ha="left", va="center")
    ax.annotate("", xy=(52, 1500), xytext=(140, 190),
                arrowprops=dict(arrowstyle="-|>", color="#8a3f2f", lw=1.0,
                                mutation_scale=8))
    ax.text(46, 1900, "increasing\nindex", fontsize=6.6, color="#8a3f2f",
            ha="center", va="bottom", linespacing=1.15, fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, 3.0e4)
    ax.set_ylim(1e-2, 2.0e4)
    ax.set_xlabel("Density, $\\rho$ (kg m$^{-3}$)")
    ax.set_ylabel("Strength, $\\sigma_f$ (MPa)")
    ax.grid(which="major", color="#dcdcdc", lw=0.5, zorder=0)
    ax.grid(which="minor", color="#efefef", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.8)
        ax.spines[s].set_color("#333333")
    ax.tick_params(which="both", direction="in", top=True, right=True)

    ax.set_title("Ashby chart: strength against density for engineering "
                 "materials", fontsize=10.5, pad=7)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.925, bottom=0.10)
    savefig(fig, "124_ashby_map.png")


if __name__ == "__main__":
    main()

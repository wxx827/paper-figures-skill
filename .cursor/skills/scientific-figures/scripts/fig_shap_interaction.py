"""SHAP interaction dependence grid (2x3).

Each panel is a SHAP dependence scatter: x = feature value, y = SHAP value,
point colour = value of the strongest interacting feature. A LOWESS trend
line with a bootstrap-free local band summarises the main effect, and the
annotated interaction strength is the |correlation| between the interacting
feature and the residual left after removing that main effect.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

CMAP = LinearSegmentedColormap.from_list(
    "indigo_amber",
    ["#33447a", "#7b93bd", "#cfd3d6", "#e3b268", "#b0592c"])

# (main feature, interacting feature, unit, main-effect shape, interaction gain)
SPECS = [                              # ordered by descending interaction
    ("Age", "BMI", "years", "sigmoid", 0.95),
    ("HbA1c", "Waist circ.", "%", "sigmoid", 0.72),
    ("Serum creatinine", "Age", "mg/dL", "rise", 0.55),
    ("Physical activity", "Age", "MET-h/wk", "decay", 0.42),
    ("Systolic BP", "Sodium intake", "mmHg", "vshape", 0.30),
    ("LDL cholesterol", "Statin dose", "mmol/L", "rise", 0.18),
]


def lowess(x, y, frac=0.40, n_out=140):
    """Tricube-weighted local linear smoother (no statsmodels dependency)."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    xout = np.linspace(xs[0], xs[-1], n_out)
    r = max(int(np.ceil(frac * xs.size)), 3)
    yout = np.empty(n_out)
    sout = np.empty(n_out)
    for i, x0 in enumerate(xout):
        d = np.abs(xs - x0)
        h = max(np.partition(d, r - 1)[r - 1], 1e-9)
        w = np.clip(d / h, 0, 1)
        w = (1 - w ** 3) ** 3
        sw = w.sum()
        mx = (w * xs).sum() / sw
        my = (w * ys).sum() / sw
        vx = (w * (xs - mx) ** 2).sum()
        b = (w * (xs - mx) * (ys - my)).sum() / vx if vx > 1e-12 else 0.0
        yout[i] = my + b * (x0 - mx)
        resid = ys - (my + b * (xs - mx))
        neff = sw ** 2 / (w ** 2).sum()
        sout[i] = np.sqrt((w * resid ** 2).sum() / sw) / np.sqrt(neff)
    return xout, yout, sout


def main_effect(kind, u):
    """u in [0,1] -> main-effect SHAP contribution (arbitrary log-odds units)."""
    if kind == "sigmoid":
        return 1.0 / (1 + np.exp(-9 * (u - 0.55))) - 0.5
    if kind == "rise":
        return 0.85 * (u ** 1.7) - 0.28
    if kind == "vshape":
        return 1.5 * (u - 0.5) ** 2 - 0.16
    if kind == "decay":
        return 0.55 * np.exp(-3.1 * u) - 0.20
    raise ValueError(kind)


def freest_corner(xf, yf, w=0.36, h=0.30):
    """Pick the axes corner holding the fewest points (fractional coords)."""
    boxes = {
        "tl": (0.0, w, 1 - h, 1.0), "tr": (1 - w, 1.0, 1 - h, 1.0),
        "bl": (0.0, w, 0.0, h), "br": (1 - w, 1.0, 0.0, h),
    }
    counts = {k: int(np.sum((xf >= x0) & (xf <= x1) & (yf >= y0) & (yf <= y1)))
              for k, (x0, x1, y0, y1) in boxes.items()}
    return min(counts, key=counts.get)


def panel(ax, rng, spec, label, show_ylabel):
    feat, inter, unit, kind, gain = spec
    n = 420
    u = np.clip(rng.beta(2.4, 2.2, n), 0.005, 0.995)          # latent 0-1
    # interacting feature is only weakly correlated with the main feature, so
    # the colour gradient stratifies vertically rather than along x
    v = np.clip(0.22 * u + 0.78 * rng.beta(2.1, 2.1, n)
                + rng.normal(0, 0.04, n), 0.005, 0.995)
    shap = (main_effect(kind, u)
            + gain * 0.60 * (v - 0.5) * (0.30 + u)
            + rng.normal(0, 0.060, n))

    lo, hi = {"Age": (38, 82), "Serum creatinine": (0.5, 2.4),
              "Systolic BP": (98, 178), "HbA1c": (4.6, 10.5),
              "Physical activity": (0, 42),
              "LDL cholesterol": (1.6, 5.8)}[feat]
    x = lo + u * (hi - lo)

    sc = ax.scatter(x, shap, c=v, cmap=CMAP, norm=Normalize(0, 1),
                    s=9, alpha=0.80, linewidths=0.15, edgecolors="white",
                    rasterized=True)

    xs, ys, se = lowess(x, shap)
    ax.fill_between(xs, ys - 1.96 * se, ys + 1.96 * se,
                    color="#4a4a4a", alpha=0.16, lw=0)
    ax.plot(xs, ys, color="#1a1a1a", lw=1.7,
            path_effects=[pe.Stroke(linewidth=3.0, foreground="white"),
                          pe.Normal()])
    ax.axhline(0, color="#9a9a9a", lw=0.7, ls=(0, (4, 3)), zorder=0)

    resid = shap - np.interp(x, xs, ys)
    strength = abs(np.corrcoef(resid, v)[0, 1])

    ax.set_xlabel(f"{feat} ({unit})" if unit else feat, fontsize=7.8)
    if show_ylabel:
        ax.set_ylabel("SHAP value (log-odds)", fontsize=7.8)
    span = shap.max() - shap.min()
    ax.set_ylim(shap.min() - 0.09 * span, shap.max() + 0.26 * span)
    ax.margins(x=0.03)
    ax.tick_params(labelsize=6.8, length=2.6)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    corner = freest_corner((x - x0) / (x1 - x0), (shap - y0) / (y1 - y0))
    ha = "left" if corner[1] == "l" else "right"
    va = "top" if corner[0] == "t" else "bottom"
    tx = 0.035 if ha == "left" else 0.965
    ty = 0.965 if va == "top" else 0.035
    ax.text(tx, ty, f"$\\rho_{{int}}$ = {strength:.2f}\ncolour: {inter}",
            transform=ax.transAxes, ha=ha, va=va, linespacing=1.45,
            fontsize=7.0, fontweight="bold", color="#1a1a1a",
            bbox=dict(boxstyle="round,pad=0.30", fc="white",
                      ec="#c8c8c8", lw=0.6, alpha=0.92))
    style_axes(ax)
    add_panel_label(ax, label, x=-0.155 if show_ylabel else -0.075,
                    y=1.02, size=9.5)
    return sc


def main():
    set_pub_style()
    rng = np.random.default_rng(2024)
    fig = plt.figure(figsize=(10.6, 6.3), layout="constrained")
    fig.get_layout_engine().set(rect=(0.004, 0.045, 0.992, 0.945),
                                h_pad=0.06, w_pad=0.05, hspace=0.03)
    axes = fig.subplots(2, 3)
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    sc = None
    for k, (ax, spec) in enumerate(zip(axes.ravel(), SPECS)):
        sc = panel(ax, rng, spec, labels[k], show_ylabel=(k % 3 == 0))

    cb = fig.colorbar(sc, ax=axes, fraction=0.019, pad=0.012, aspect=32)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Low", "High"])
    cb.ax.tick_params(labelsize=7, length=0)
    cb.set_label("Interacting feature value", fontsize=7.6, labelpad=3)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    savefig(fig, "63_shap_interaction.png")


if __name__ == "__main__":
    main()

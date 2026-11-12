"""How forecast error grows with the horizon, for four competing models.

(a) MAE (solid) and RMSE (dashed) against horizon, one colour per model, with
the horizon at which each model's MAE doubles marked on the curve; (b) the
absolute-error distribution of the proposed model at five horizons, which is
the part a mean curve hides.  Two legends live on panel (a), so the second one
is un-clipped explicitly (DESIGN_SYSTEM 12.1).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import panel_labels_inches, unclip_legend

MODELS = [("Proposed", "#3b5b92", 0.82, 0.40),
          ("Gradient boosting", "#2e7d5b", 0.94, 0.49),
          ("SARIMA", "#c9a227", 1.06, 0.55),
          ("Seasonal naive", "#8e6bbf", 1.28, 0.62)]
ACCENT = "#b23a48"
GREY = "#808080"
H_MAX = 24
VIOLIN_H = [1, 3, 6, 12, 24]
N_SAMP = 400


def curves(rng):
    h = np.arange(1, H_MAX + 1, dtype=float)
    out = {}
    for name, col, a, b in MODELS:
        mae = a * h ** b * (1.0 + 0.012 * rng.normal(0, 1, h.size))
        rmse = mae * (1.24 + 0.004 * h)
        out[name] = (col, mae, rmse)
    return h, out


def doubling_horizon(h, mae):
    """Interpolated horizon where MAE reaches twice its one-step value."""
    target = 2.0 * mae[0]
    if mae[-1] < target:
        return None
    k = int(np.argmax(mae >= target))
    if k == 0:
        return float(h[0])
    x0, x1, y0, y1 = h[k - 1], h[k], mae[k - 1], mae[k]
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


def panel_curves(ax, h, data):
    for name, (col, mae, rmse) in data.items():
        ax.plot(h, mae, color=col, lw=1.6, zorder=4)
        ax.plot(h, rmse, color=col, lw=1.2, ls=(0, (3.2, 2.0)), alpha=0.9,
                zorder=3)
        ax.text(h[-1] + 0.45, mae[-1], name, ha="left", va="center",
                fontsize=6.8, color=col)
    ax.set_xlabel("Forecast horizon $h$ (steps)")
    ax.set_ylabel("Error (MW)")
    ax.set_xlim(0.4, H_MAX + 0.6)
    ax.set_xticks([1, 6, 12, 18, 24])


def panel_violin(ax, samples, base_col):
    parts = ax.violinplot(samples, positions=range(len(samples)),
                          widths=0.78, showextrema=False, showmedians=False)
    for i, body in enumerate(parts["bodies"]):
        shade = lighten(base_col, 0.55 - 0.11 * i)
        body.set_facecolor(shade)
        body.set_edgecolor(base_col)
        body.set_linewidth(0.7)
        body.set_alpha(1.0)
    med = [np.median(s) for s in samples]
    q1 = [np.percentile(s, 25) for s in samples]
    q3 = [np.percentile(s, 75) for s in samples]
    x = np.arange(len(samples))
    ax.vlines(x, q1, q3, color=base_col, lw=3.0, zorder=4)
    ax.plot(x, med, ls="none", marker="o", ms=3.2, mfc="white",
            mec=base_col, mew=0.9, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in VIOLIN_H], fontsize=6.5)
    ax.set_xlabel("Horizon $h$ (steps)", fontsize=7.5)
    ax.set_ylabel("Absolute error $|e_{t+h}|$ (MW)", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.set_ylim(0, max(np.percentile(s, 99) for s in samples) * 1.08)
    ax.set_title("Spread grows faster than the mean", fontsize=8.5,
                 fontweight="bold")


def main():
    set_pub_style()
    rng = np.random.default_rng(17)
    h, data = curves(rng)

    fig = plt.figure(figsize=(7.8, 4.4))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 0.46], wspace=0.255,
                  left=0.070, right=0.855, top=0.885, bottom=0.145)
    ax = fig.add_subplot(gs[0, 0])
    bx = fig.add_subplot(gs[0, 1])

    panel_curves(ax, h, data)

    # Doubling horizons, computed from the curves that were just drawn.
    doubles = {}
    for name, (col, mae, _) in data.items():
        hd = doubling_horizon(h, mae)
        doubles[name] = hd
        if hd is not None:
            ax.plot([hd], [2.0 * mae[0]], marker="v", ms=4.6, color=ACCENT,
                    mec="white", mew=0.6, zorder=7)
    slow = max((v, k) for k, v in doubles.items() if v is not None)
    ax.annotate(f"error doubles at $h$ = {slow[0]:.1f} for {slow[1]},\n"
                f"the latest of the four (markers)",
                xy=(slow[0], 2.0 * data[slow[1]][1][0]),
                xytext=(slow[0] + 1.6, 0.30),
                textcoords=("data", "axes fraction"),
                fontsize=6.8, color=ACCENT, ha="left", va="top",
                linespacing=1.35,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=ACCENT,
                                connectionstyle="arc3,rad=0.22"))

    leg_models = ax.legend(
        handles=[Line2D([], [], color=c, lw=1.7, label=n)
                 for n, c, _, _ in MODELS],
        loc="upper left", fontsize=7, handlelength=1.6, labelspacing=0.30,
        borderaxespad=0.35)
    ax.legend(handles=[Line2D([], [], color=GREY, lw=1.6, label="MAE"),
                       Line2D([], [], color=GREY, lw=1.2,
                              ls=(0, (3.2, 2.0)), label="RMSE")],
              loc="lower right", fontsize=7, handlelength=1.8,
              labelspacing=0.28, borderaxespad=0.35)
    ax.add_artist(leg_models)
    unclip_legend(leg_models)

    base_col = MODELS[0][1]
    mae0 = data[MODELS[0][0]][1]
    samples = [np.abs(rng.normal(0.0, mae0[k - 1] * 1.25, N_SAMP))
               for k in VIOLIN_H]
    panel_violin(bx, samples, base_col)

    for a in (ax, bx):
        a.grid(axis="y", lw=0.5, color="#f0f0f0")
        a.set_axisbelow(True)
        style_axes(a)

    panel_labels_inches(fig, [ax, bx], dx=-0.54, dy=0.10)
    savefig(fig, "362_ts_horizon_error.png")


if __name__ == "__main__":
    main()

"""Rolling-origin backtest: five forecast folds laid over the realised series.

(a) each fold's 12-month path is drawn from its own origin on top of the
actual curve, the folds shaded from light to dark so the reader can see the
evaluation marching forward in time; (b) per-fold MAE bars with RMSE markers
against the pooled mean; (c) a card stack with the headline metrics.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, get_palette
from figkit import panel_labels_inches, text_card, use_inch_units

ACT_C = "#3a3a3a"      # realised series
GREY = "#808080"
N = 132
H = 12                 # forecast horizon per fold
ORIGINS = [72, 84, 96, 108, 120]
FOLD_C = get_palette("cb_sequential_blue", 7)[2:]   # light -> dark by fold


def synth_actual(rng, n):
    t = np.arange(n, dtype=float)
    trend = 240.0 + 0.85 * t - 0.0032 * t ** 2
    season = 14.0 * np.sin(2 * np.pi * t / 12.0 + 0.4)
    cycle = 9.0 * np.sin(2 * np.pi * t / 41.0)
    return trend + season + cycle + rng.normal(0.0, 4.2, n)


def fold_forecast(actual, origin, h, rng, bias):
    """A plausible fold: right at the origin, drifting away with horizon."""
    idx = np.arange(origin, min(origin + h, actual.size))
    steps = np.arange(1, idx.size + 1, dtype=float)
    drift = bias * steps + rng.normal(0.0, 1.6, idx.size) * np.sqrt(steps)
    padded = np.pad(actual, 1, mode="edge")
    smooth = np.convolve(padded, np.ones(3) / 3.0, mode="valid")[idx]
    return idx, smooth + drift


def panel_paths(ax, actual, folds):
    ax.plot(np.arange(actual.size), actual, color=ACT_C, lw=1.15, zorder=3,
            label="Realised")
    for k, ((idx, yhat), col) in enumerate(zip(folds, FOLD_C), start=1):
        ax.axvline(idx[0], color=col, lw=0.7, ls=(0, (1.5, 2.0)), alpha=0.8,
                   zorder=2)
        ax.plot(idx, yhat, color=col, lw=1.5, zorder=4 + k)
        ax.plot(idx[0], yhat[0], marker="o", ms=3.4, color=col, zorder=9,
                mec="white", mew=0.6)
    ax.set_ylabel("Demand (GWh)")
    ax.set_xlim(-2, actual.size + 1)
    stack = np.concatenate([actual] + [y for _, y in folds])
    lo, hi = stack.min(), stack.max()
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.16 * (hi - lo))
    ax.set_xticks(np.arange(0, actual.size + 1, 12))
    handles = [Line2D([], [], color=ACT_C, lw=1.3, label="Realised")]
    handles += [Line2D([], [], color=c, lw=1.6, label=f"fold {k}")
                for k, c in enumerate(FOLD_C, start=1)]
    ax.legend(handles=handles, loc="upper left", ncol=6, fontsize=7,
              handlelength=1.5, columnspacing=1.1, borderaxespad=0.3)
    ax.set_title("Each fold re-forecasts 12 months from a later origin",
                 fontsize=9, fontweight="bold")


def panel_errors(ax, mae, rmse):
    x = np.arange(1, mae.size + 1)
    ax.bar(x, mae, width=0.56, color=FOLD_C, edgecolor="none", zorder=3)
    ax.plot(x, rmse, ls="none", marker="D", ms=4.0, mfc="white",
            mec="#1a1a1a", mew=0.9, zorder=5)
    pooled = mae.mean()
    ax.axhline(pooled, color=GREY, lw=0.9, ls=(0, (3.0, 2.2)), zorder=4)
    ax.text(mae.size + 0.44, pooled, f" mean\n {pooled:.1f}", ha="left",
            va="center", fontsize=6.5, color="#555555", linespacing=1.25)
    for xi, m, r in zip(x, mae, rmse):
        ax.text(xi, max(m, r) + 0.55, f"{m:.1f}", ha="center", va="bottom",
                fontsize=6.5, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{o}" for o in ORIGINS], fontsize=7)
    ax.set_xlim(0.4, mae.size + 1.35)
    ax.set_ylim(0, max(rmse.max(), mae.max()) * 1.46)
    ax.set_xlabel("Fold origin (month index)")
    ax.set_ylabel("Error (GWh)", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.legend(handles=[Line2D([], [], color=FOLD_C[2], lw=5, label="MAE"),
                       Line2D([], [], ls="none", marker="D", ms=3.6,
                              mfc="white", mec="#1a1a1a", mew=0.9,
                              label="RMSE")],
              loc="upper left", fontsize=6.5, ncol=2, handlelength=1.2,
              columnspacing=1.0, borderaxespad=0.25)


def panel_card(ax, stats):
    w, h = use_inch_units(ax)
    ch = (h - 0.16 * (len(stats) - 1)) / len(stats)
    for i, (name, value) in enumerate(stats):
        y = h - (i + 1) * ch - i * 0.16
        text_card(ax, 0.02, y, w - 0.04, ch, value, title=name,
                  fs=8.6, title_fs=6.6, title_color="#777777", color="#1a1a1a",
                  ha="left", pad=0.10, rounding=0.045, fc="#f5f5f5")


def main():
    set_pub_style()
    rng = np.random.default_rng(23)
    actual = synth_actual(rng, N)
    biases = [0.62, -0.44, 1.05, -0.32, 0.58]
    folds = [fold_forecast(actual, o, H, rng, b)
             for o, b in zip(ORIGINS, biases)]

    errs = [yhat - actual[idx] for idx, yhat in folds]
    mae = np.array([np.mean(np.abs(e)) for e in errs])
    rmse = np.array([np.sqrt(np.mean(e ** 2)) for e in errs])
    allerr = np.concatenate(errs)
    denom = np.concatenate([actual[idx] for idx, _ in folds])
    mape = 100.0 * np.mean(np.abs(allerr / denom))

    fig = plt.figure(figsize=(7.8, 4.7))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 0.235],
                  height_ratios=[1.0, 0.58], hspace=0.42, wspace=0.10,
                  left=0.072, right=0.985, top=0.885, bottom=0.115)
    ax = fig.add_subplot(gs[0, 0])
    bx = fig.add_subplot(gs[1, 0])
    cx = fig.add_subplot(gs[:, 1])

    panel_paths(ax, actual, folds)
    panel_errors(bx, mae, rmse)
    for a in (ax, bx):
        a.grid(axis="y", lw=0.5, color="#f0f0f0")
        a.set_axisbelow(True)
        style_axes(a)

    cx.set_axis_off()
    panel_card(cx, [
        ("Pooled MAE", f"{np.mean(np.abs(allerr)):.1f} GWh"),
        ("Pooled RMSE", f"{np.sqrt(np.mean(allerr ** 2)):.1f} GWh"),
        ("MAPE", f"{mape:.2f} %"),
        ("Worst fold", f"{int(np.argmax(mae)) + 1}  ({mae.max():.1f} GWh)"),
        ("Design", f"{len(ORIGINS)} folds x {H} mo"),
    ])

    panel_labels_inches(fig, [ax, bx], dx=-0.56, dy=0.12)
    panel_labels_inches(fig, [cx], labels=["(c)"], dx=-0.10, dy=0.12)
    savefig(fig, "357_ts_rolling_backtest.png")


if __name__ == "__main__":
    main()

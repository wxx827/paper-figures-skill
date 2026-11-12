"""Forecast combination: six base learners, one inverse-error ensemble.

The main panel shows the observed series, the six base forecasts as thin
semi-transparent lines over the holdout horizon, and the weighted combination
as one heavy line against the realised values in black.  The two subordinate
panels on the right report where the weights came from (validation error) and
what the combination actually bought over the single model that looked best on
that same validation window.

Every number printed on the figure - weights, errors, the gain in percent - is
recomputed from the arrays that are plotted, so the caption cannot drift away
from the curves.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from pubstyle import (CATEGORICAL, CATEGORICAL_EXTRA, COLORS, lighten,
                      savefig, set_pub_style)
from figkit import panel_labels_inches

SEED = 7
N_HIST = 108          # observed points before the forecast origin
H = 24                # forecast horizon
N_VAL = 24            # validation window (last part of the history)
PERIOD = 12

# name, trend damping, seasonal gain, seasonal phase shift, level bias,
# wander amplitude - each base learner gets its own systematic failure mode
MODELS = [
    ("ARIMA(2,1,1)",     0.97, 0.95,  0.00,  0.15, 0.45),
    ("ETS damped trend", 0.86, 1.04,  0.10, -0.60, 0.35),
    ("Theta method",     1.03, 0.86, -0.12,  0.35, 0.40),
    ("Gradient boosting", 0.93, 1.12,  0.22,  0.55, 0.70),
    ("Seasonal naive",   0.78, 1.00,  0.00, -1.30, 0.25),
    ("LSTM (64 units)",  1.06, 0.93, -0.26, -0.35, 0.85),
]


def ar1(n, rho, sigma, rng):
    """AR(1) path - forecast errors are serially correlated, not white."""
    e = rng.normal(0.0, sigma, n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1.0 - rho ** 2)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def components(t):
    trend = 0.052 * t
    seasonal = 3.1 * np.sin(2 * np.pi * t / PERIOD) + \
        1.15 * np.cos(4 * np.pi * t / PERIOD)
    return trend, seasonal


def base_forecast(t, spec, rng):
    """Deterministic signal distorted by the learner's own biases."""
    _, damp, gain, phase, bias, wander = spec
    trend = 0.052 * t * damp
    seasonal = gain * (3.1 * np.sin(2 * np.pi * t / PERIOD + phase) +
                       1.15 * np.cos(4 * np.pi * t / PERIOD + phase))
    return 20.0 + trend + seasonal + bias + ar1(t.size, 0.72, wander, rng)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    n_tot = N_HIST + H
    t = np.arange(n_tot, dtype=float)
    trend, seasonal = components(t)
    y = 20.0 + trend + seasonal + ar1(n_tot, 0.55, 0.62, rng)

    hist = np.arange(N_HIST)
    fut = np.arange(N_HIST, n_tot)
    val = np.arange(N_HIST - N_VAL, N_HIST)

    preds = np.stack([base_forecast(t, m, rng) for m in MODELS])
    names = [m[0] for m in MODELS]
    n_mod = len(MODELS)

    # ---- weights from validation error only (no peeking at the holdout) ----
    mse_val = ((preds[:, val] - y[val]) ** 2).mean(axis=1)
    w = (1.0 / mse_val)
    w = w / w.sum()
    best = int(np.argmin(mse_val))                 # best single model a priori

    ens = w @ preds                                # weighted combination

    # ---- holdout accuracy --------------------------------------------------
    ae_ens = np.abs(ens[fut] - y[fut])
    ae_best = np.abs(preds[best, fut] - y[fut])
    mae_ens, mae_best = ae_ens.mean(), ae_best.mean()
    gain_pct = 100.0 * (mae_best - mae_ens) / mae_best
    mae_each = np.abs(preds[:, fut] - y[fut][None, :]).mean(axis=1)

    cols = (CATEGORICAL + CATEGORICAL_EXTRA)[:n_mod]

    fig = plt.figure(figsize=(10.4, 6.2))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 0.55],
                  height_ratios=[1.0, 0.86], hspace=0.55, wspace=0.24)
    ax = fig.add_subplot(gs[:, 0])
    ax_w = fig.add_subplot(gs[0, 1])
    ax_g = fig.add_subplot(gs[1, 1])

    # ---- (a) history, base forecasts, ensemble ----------------------------
    show = np.arange(N_HIST - 54, n_tot)
    ax.axvspan(N_HIST - 0.5, n_tot - 1, color=COLORS["card"], zorder=0)
    ax.axvspan(val[0] - 0.5, N_HIST - 0.5, color=lighten(CATEGORICAL[1], 0.90),
               zorder=0)

    ax.plot(hist[hist >= show[0]], y[hist[hist >= show[0]]], color="#1a1a1a",
            lw=1.25, zorder=5, label="observed")
    ax.plot(fut, y[fut], color="#1a1a1a", lw=1.25, ls=(0, (4.5, 1.8)),
            zorder=5, label="realised (holdout)")

    for k in range(n_mod):
        ax.plot(fut, preds[k, fut], color=cols[k], lw=0.85, alpha=0.55,
                zorder=3)

    ax.plot(fut, ens[fut], color=COLORS["accent"], lw=2.4, zorder=6,
            solid_capstyle="round")
    ax.scatter([N_HIST - 1], [y[N_HIST - 1]], s=16, color="#1a1a1a",
               zorder=7, clip_on=False)

    ax.axvline(N_HIST - 0.5, color=COLORS["rule"], lw=0.8, ls=":", zorder=4)
    ax.text(N_HIST - 1.5, ax.get_ylim()[1], "forecast origin  ", ha="right",
            va="top", fontsize=6.8, color=COLORS["sub"], rotation=90)
    ax.text((val[0] + N_HIST) / 2.0, y[show].min(), "weights fitted here",
            ha="center", va="bottom", fontsize=6.6, color=CATEGORICAL[1])

    ax.set_xlim(show[0], n_tot - 1 + 0.5)
    ax.set_xlabel("time index (months)", fontsize=8.0)
    ax.set_ylabel("demand index (-)", fontsize=8.0)
    ax.set_title("Observed series and the 24-step holdout forecast",
                 fontsize=9.0, loc="left", pad=4)
    ax.tick_params(labelsize=7.0)
    ax.grid(axis="y", lw=0.5, color=COLORS["grid"], alpha=0.55)

    handles = [Line2D([], [], color="#1a1a1a", lw=1.25, label="observed"),
               Line2D([], [], color="#1a1a1a", lw=1.25, ls=(0, (4.5, 1.8)),
                      label="realised (holdout)"),
               Line2D([], [], color=COLORS["accent"], lw=2.4,
                      label="weighted ensemble"),
               Line2D([], [], color=COLORS["baseline"], lw=0.85, alpha=0.7,
                      label=f"{n_mod} base learners")]
    ax.legend(handles=handles, loc="upper left", fontsize=7.0, ncol=2,
              handlelength=1.9, columnspacing=1.1, borderaxespad=0.3)

    ax.text(0.0, -0.115, f"ensemble MAE {mae_ens:.2f} vs best single "
            f"({names[best]}) {mae_best:.2f} over h = 1..{H}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.0,
            color=COLORS["sub"])

    # ---- (b) weights -------------------------------------------------------
    order = np.argsort(w)
    ypos = np.arange(n_mod)
    ax_w.barh(ypos, w[order] * 100.0, height=0.64,
              color=[cols[i] for i in order], edgecolor="white", linewidth=0.7)
    for yy, i in zip(ypos, order):
        ax_w.text(w[i] * 100.0 + 0.8, yy, f"{w[i] * 100:.1f}%", va="center",
                  ha="left", fontsize=6.6, color=COLORS["sub"])
    ax_w.set_yticks(ypos)
    ax_w.set_yticklabels([names[i] for i in order], fontsize=6.8)
    ax_w.set_xlim(0, w.max() * 100.0 * 1.32)
    ax_w.set_xlabel("combination weight (%)", fontsize=7.5)
    ax_w.tick_params(labelsize=6.8)
    ax_w.set_title("Weights $\\propto$ 1 / validation MSE", fontsize=8.5,
                   loc="left", pad=4)
    ax_w.grid(axis="x", lw=0.5, color=COLORS["grid"], alpha=0.55)

    # ---- (c) gain over the best single model ------------------------------
    steps = np.arange(1, H + 1)
    run_ens = np.cumsum(ae_ens) / steps
    run_best = np.cumsum(ae_best) / steps
    ax_g.fill_between(steps, run_ens, run_best, where=run_best >= run_ens,
                      color=lighten(COLORS["accent"], 0.75), lw=0, zorder=1,
                      label="ensemble advantage")
    ax_g.fill_between(steps, run_ens, run_best, where=run_best < run_ens,
                      color=COLORS["hairline"], lw=0, zorder=1)
    ax_g.plot(steps, run_best, color=cols[best], lw=1.5, zorder=3,
              label=f"best single ({names[best].split('(')[0].strip()})")
    ax_g.plot(steps, run_ens, color=COLORS["accent"], lw=2.0, zorder=4,
              label="ensemble")
    ax_g.set_xlim(1, H)
    ax_g.set_ylim(0, max(run_best.max(), run_ens.max()) * 1.42)
    ax_g.set_xlabel("forecast step h", fontsize=7.5)
    ax_g.set_ylabel("running MAE (-)", fontsize=7.5)
    ax_g.tick_params(labelsize=6.8)
    ax_g.set_title(f"Gain from combining: {gain_pct:+.1f}% MAE",
                   fontsize=8.5, loc="left", pad=4)
    ax_g.legend(loc="upper left", fontsize=6.6, handlelength=1.6,
                borderaxespad=0.3)
    ax_g.grid(axis="y", lw=0.5, color=COLORS["grid"], alpha=0.55)
    beat = int((mae_each > mae_ens).sum())
    ax_g.text(0.98, 0.06, f"ensemble beats {beat} of {n_mod} learners",
              transform=ax_g.transAxes, ha="right", va="bottom",
              fontsize=6.6, color=COLORS["sub"])

    fig.tight_layout(rect=[0, 0.015, 1, 0.925], h_pad=2.2, w_pad=2.6)
    panel_labels_inches(fig, [ax, ax_w, ax_g], dx=-0.42, dy=0.12)

    savefig(fig, "363_ts_ensemble_forecast.png")


if __name__ == "__main__":
    main()

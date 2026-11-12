"""Observed, predicted and residual origin-destination matrices.

A fourteen-zone gravity-model OD table is compared with the output of a
synthetic spatio-temporal predictor.  The observed and predicted panels are
drawn through one ``shared_norm`` computed over both tables, so the eye can
compare them cell by cell; the residual panel keeps its own diverging map,
symmetric about zero, because a signed quantity cannot honestly share a
sequential scale.  Every number in the figure - RMSE, MAPE, the per-stratum
bars - is computed from the arrays being displayed.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, get_cmap, COLORS
from figkit import panel_labels_inches, safe_colorbar, shared_norm

NZ = 14
DISTRICTS = [("core", 5), ("north", 5), ("south", 4)]
INK = COLORS["ink"]
SUB = COLORS["sub"]
MUTED = COLORS["muted"]
FRAME = "#c9c6bf"
MAPE_FLOOR = 20.0                     # trips; MAPE is undefined on tiny flows


def build_od(rng):
    """Gravity-model observed table plus a plausibly biased prediction."""
    xy = np.c_[rng.uniform(0.4, 5.6, NZ), rng.uniform(0.4, 5.6, NZ)]
    xy[:5] = np.c_[rng.uniform(2.0, 4.0, 5), rng.uniform(2.0, 4.0, 5)]
    prod = np.exp(rng.normal(0.0, 0.55, NZ)) * (1.9 - 0.1 * np.arange(NZ))
    attr = np.exp(rng.normal(0.0, 0.50, NZ)) * (1.0 + 0.06 * np.arange(NZ))
    d = np.hypot(xy[:, 0][:, None] - xy[:, 0][None, :],
                 xy[:, 1][:, None] - xy[:, 1][None, :])
    true = np.outer(prod, attr) * np.exp(-d / 2.1)
    true = true / true.max() * 880.0
    true = true * np.exp(rng.normal(0.0, 0.18, true.shape))
    np.fill_diagonal(true, np.nan)

    # A believable model: shrinks the biggest flows, inflates the smallest,
    # and is noisier where the flow is thin.
    rel = np.log10(np.fmax(true, 1.0) / 120.0)
    bias = -0.085 * rel
    noise = rng.normal(0.0, 1.0, true.shape) * (0.06 + 0.16
                                                / np.sqrt(np.fmax(true, 4.0)))
    pred = np.clip(true * (1.0 + bias + noise * 2.6), 0.0, None)
    np.fill_diagonal(pred, np.nan)
    return true, pred


def main():
    set_pub_style()
    rng = np.random.default_rng(482)
    true, pred = build_od(rng)
    err = pred - true

    ok = np.isfinite(err)
    rmse = float(np.sqrt(np.nanmean(err[ok] ** 2)))
    big = ok & (true > MAPE_FLOOR)
    mape = float(np.nanmean(np.abs(err[big] / true[big])) * 100.0)
    bias_tot = float(np.nanmean(err[ok]))

    norm_t = shared_norm([true, pred], robust=(0, 99))
    norm_e = shared_norm([err], robust=(1, 99), symmetric=True)
    cmap_t = get_cmap("cb_sequential_blue")
    cmap_e = get_cmap("cb_diverging")
    for cm in (cmap_t, cmap_e):
        cm.set_bad("#ededed")

    fig = plt.figure(figsize=(10.4, 6.4))
    w = 0.2450
    axa = fig.add_axes([0.0400, 0.455, w, 0.4094])
    axb = fig.add_axes([0.3120, 0.455, w, 0.4094])
    axc = fig.add_axes([0.6520, 0.455, w, 0.4094])
    axh = fig.add_axes([0.0400, 0.088, 0.5170, 0.2375])
    axs = fig.add_axes([0.6520, 0.088, w, 0.2375])

    edges = np.cumsum([n for _, n in DISTRICTS])[:-1] - 0.5
    labels = [f"Z{i + 1:02d}" for i in range(NZ)]
    panels = [(axa, true, cmap_t, norm_t, "Observed OD"),
              (axb, pred, cmap_t, norm_t, "Predicted OD"),
              (axc, err, cmap_e, norm_e, "Prediction error")]
    ims = []
    for ax, m, cm, nm, ttl in panels:
        ims.append(ax.imshow(m, cmap=cm, norm=nm, origin="upper",
                             interpolation="nearest"))
        for e in edges:
            ax.axhline(e, color="white", lw=1.1)
            ax.axvline(e, color="white", lw=1.1)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(NZ))
        ax.set_yticks(np.arange(NZ))
        ax.set_xticklabels(labels, fontsize=6.0, rotation=90)
        ax.set_yticklabels(labels, fontsize=6.0)
        ax.tick_params(length=1.8, width=0.6, pad=1.4)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.7)
            sp.set_edgecolor(FRAME)
        ax.set_title(ttl, fontsize=8.8, fontweight="bold", color=INK, pad=4)
        ax.set_xlabel("destination zone", fontsize=7.4, labelpad=2)
    axa.set_ylabel("origin zone", fontsize=7.4, labelpad=2)
    for ax in (axb, axc):
        ax.tick_params(labelleft=False)

    safe_colorbar(fig, ims[1], axb, label="trips (count), shared by (a), (b)",
                  width=0.10, pad=0.06, labelsize=6.6)
    safe_colorbar(fig, ims[2], axc, label="predicted - observed (trips)",
                  width=0.10, pad=0.06, labelsize=6.6)

    # ------------------------------------------------ (d) error distribution
    e = err[ok]
    bins = np.linspace(norm_e.vmin, norm_e.vmax, 41)
    cnt, _ = np.histogram(e, bins=bins)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    axh.bar(ctr, cnt, width=(bins[1] - bins[0]) * 0.92,
            color=cmap_e(norm_e(ctr)), edgecolor="white", linewidth=0.35,
            zorder=3)
    axh.axvline(0.0, color="#808080", lw=0.9, zorder=4)
    axh.axvline(bias_tot, color="#b23a48", lw=1.2, zorder=5)
    axh.annotate(f"mean bias {bias_tot:+.1f} trips",
                 xy=(bias_tot, cnt.max() * 0.86),
                 xytext=(norm_e.vmin * 0.42, cnt.max() * 0.99),
                 fontsize=6.6, color="#b23a48", ha="left", va="top",
                 arrowprops=dict(arrowstyle="-", lw=0.7, color="#b23a48"))
    axh.set_xlim(norm_e.vmin, norm_e.vmax)
    axh.set_xlabel("prediction error (trips)", fontsize=7.4, labelpad=2)
    axh.set_ylabel("OD pairs (count)", fontsize=7.4, labelpad=2)
    axh.set_title("Residual distribution", fontsize=8.8, fontweight="bold",
                  color=INK, pad=4)
    axh.tick_params(labelsize=6.6, length=2.4)
    axh.grid(axis="y", color="#ececec", lw=0.55, zorder=0)
    axh.text(0.985, 0.94, f"RMSE = {rmse:.1f} trips\nMAPE = {mape:.1f} %\n"
             f"n = {int(ok.sum())} OD pairs", transform=axh.transAxes,
             ha="right", va="top", fontsize=6.8, color=INK, linespacing=1.35,
             bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                       edgecolor="#cccccc", linewidth=0.6, alpha=0.92))

    # ---------------------------------------------- (e) error by flow stratum
    q = np.nanquantile(true[ok], [0.25, 0.50, 0.75])
    idx = np.digitize(true[ok], q)
    names = [f"< {q[0]:.0f}", f"{q[0]:.0f}-{q[1]:.0f}",
             f"{q[1]:.0f}-{q[2]:.0f}", f"> {q[2]:.0f}"]
    strat_rmse = np.array([np.sqrt(np.mean(e[idx == k] ** 2))
                           for k in range(4)])
    strat_bias = np.array([e[idx == k].mean() for k in range(4)])
    ypos = np.arange(4)[::-1]
    cols = [cmap_t(v) for v in np.linspace(0.35, 0.92, 4)]
    axs.barh(ypos, strat_rmse, height=0.60, color=cols, edgecolor="white",
             linewidth=0.5, zorder=3)
    for y, r, b in zip(ypos, strat_rmse, strat_bias):
        axs.text(r + strat_rmse.max() * 0.03, y, f"{b:+.1f}", va="center",
                 ha="left", fontsize=6.4, color=SUB)
    axs.set_yticks(ypos)
    axs.set_yticklabels(names, fontsize=6.6)
    axs.set_xlim(0, strat_rmse.max() * 1.28)
    axs.set_xlabel("RMSE (trips)", fontsize=7.4, labelpad=2)
    axs.set_ylabel("observed flow (trips)", fontsize=7.4, labelpad=2)
    axs.set_title("Error by flow stratum", fontsize=8.8, fontweight="bold",
                  color=INK, pad=4)
    axs.tick_params(labelsize=6.6, length=2.4)
    axs.grid(axis="x", color="#ececec", lw=0.55, zorder=0)
    axs.text(0.0, -0.30, "labels: mean signed bias (trips)",
             transform=axs.transAxes, fontsize=6.3, color=MUTED, ha="left",
             va="top")

    panel_labels_inches(fig, [axa, axb, axc, axh, axs], dx=-0.34, dy=0.24)

    savefig(fig, "482_stm_od_prediction.png")


if __name__ == "__main__":
    main()

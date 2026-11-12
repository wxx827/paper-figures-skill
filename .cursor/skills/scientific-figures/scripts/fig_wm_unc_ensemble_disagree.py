"""Figure 1179 - Where a four-member ensemble disagrees is where it is wrong.

One conveyor scene, one predicted instant.  Each ensemble member rolls the
same dynamics forward by a slightly different amount, so the four predictions
differ exactly where something moves.  Everything downstream is derived from
those four rasters:

    pred_k      -> mean_pred = mean_k pred_k
                -> disagree  = var_k pred_k            (tile 5)
                -> error     = |mean_pred - truth|     (tile 6)

The two boxes are placed on the scene's own ``anchors`` (the pusher head and a
belt part), picked by mean disagreement - never on hand-typed pixels - and the
same two boxes are redrawn on every tile.  The scatter underneath is the
patch-wise version of tiles 5 and 6, so the correlation it reports is between
the two maps actually printed above it.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, get_cmap, style_axes
from figkit import image_axes
from _wm_scenes import build, grey

SEED = 1584  # original id; figure number is 1179
SHAPE = (152, 190)
T_TRUE = 0.62
DT = (-0.058, -0.021, 0.026, 0.069)          # member-specific time offsets
GAIN = (0.985, 1.008, 0.994, 1.013)          # member-specific exposure drift
PATCH = 11                                   # patch side for the scatter

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#d5d5d5"
BOXC, SCAT, HL = "#1a1a1a", "#2e74a8", "#7b3294"
GRID, CARD = "#ededed", "#e2e8ee"


def frames():
    """Ground truth plus four member predictions, as luminance maps."""
    truth = grey(build("conveyor", seed=SEED, shape=SHAPE, t=T_TRUE)["rgb"])
    sc = build("conveyor", seed=SEED, shape=SHAPE, t=T_TRUE)
    preds = []
    for dt, g in zip(DT, GAIN):
        s = build("conveyor", seed=SEED, shape=SHAPE, t=T_TRUE + dt)
        preds.append(np.clip(grey(s["rgb"]) * g, 0, 1))
    return truth, np.stack(preds), sc["anchors"]


def patch_means(a, p=PATCH):
    h, w = a.shape
    hh, ww = h // p, w // p
    return a[:hh * p, :ww * p].reshape(hh, p, ww, p).mean(axis=(1, 3))


def main():
    set_pub_style(base_font=8.5)
    truth, preds, anchors = frames()
    mean_pred = preds.mean(axis=0)
    disagree = preds.var(axis=0)
    error = np.abs(mean_pred - truth)

    # one shared grey scale for the four member tiles, found before drawing
    vlo, vhi = float(preds.min()), float(preds.max())
    dmax = float(np.quantile(disagree, 0.995))
    emax = float(np.quantile(error, 0.995))

    # Boxes come from the scene's own anchors, ranked by mean disagreement.
    # One box per object family (the sweeping pusher, the travelling parts) so
    # the pair spans the two independent motion sources instead of landing on
    # two neighbouring belt parts.
    h, w = SHAPE
    cands = []
    for a in anchors:
        bw = int(np.clip(1.60 * a["rx"], 22, 0.32 * w))
        bh = int(np.clip(1.85 * a["ry"], 20, 0.44 * h))
        x0 = int(np.clip(a["cx"] - bw / 2, 0, w - bw))
        y0 = int(np.clip(a["cy"] - bh / 2, 0, h - bh))
        cands.append((float(disagree[y0:y0 + bh, x0:x0 + bw].mean()),
                      a["name"], x0, y0, bw, bh))
    cands.sort(reverse=True)
    boxes = []
    for fam in ("pusher", "part"):
        boxes.append(next(c for c in cands if c[1] == fam))
    boxes.sort(key=lambda c: c[2])

    inside = np.zeros(SHAPE, dtype=bool)
    for _, _, x0, y0, bw, bh in boxes:
        inside[y0:y0 + bh, x0:x0 + bw] = True
    px_share = inside.mean()
    d_share = disagree[inside].sum() / disagree.sum()
    e_share = error[inside].sum() / error.sum()

    pd_, pe = patch_means(disagree).ravel(), patch_means(error).ravel()
    pin = patch_means(inside.astype(float)).ravel() > 0.5
    r = float(np.corrcoef(pd_, pe)[0, 1])

    rmse = np.array([float(np.sqrt(((p - truth) ** 2).mean()))
                     for p in preds])
    rmse_ens = float(np.sqrt(((mean_pred - truth) ** 2).mean()))
    assert rmse_ens < rmse.min(), "the ensemble mean is not the best predictor"

    fig = plt.figure(figsize=(11.0, 4.4))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.10], left=0.032,
                          right=0.986, top=0.815, bottom=0.155, wspace=0.10,
                          hspace=0.72)

    titles = [f"member {k + 1}" for k in range(4)] + \
             ["disagreement  $\\mathrm{Var}_k$", "true error  |mean $-$ GT|"]
    ims, axes = [], []
    for c in range(6):
        ax = fig.add_subplot(gs[0, c])
        if c < 4:
            im = ax.imshow(preds[c], cmap="gray", vmin=vlo, vmax=vhi,
                           interpolation="bilinear")
        elif c == 4:
            im = ax.imshow(disagree, cmap=get_cmap("cb_sequential_blue"),
                           norm=PowerNorm(0.62, vmin=0.0, vmax=dmax),
                           interpolation="bilinear")
        else:
            im = ax.imshow(error, cmap=get_cmap("cb_sequential_purple"),
                           norm=PowerNorm(0.62, vmin=0.0, vmax=emax),
                           interpolation="bilinear")
        image_axes(ax, color="#9a9a9a", lw=0.7)
        ax.set_title(titles[c], fontsize=8.0,
                     fontweight="bold" if c >= 4 else "normal",
                     color=INK if c >= 4 else SUB, pad=3.5)
        for _, _, x0, y0, bw, bh in boxes:
            # white under-stroke so the box reads on grey, blue and purple
            for lw, col in ((2.1, "white"), (0.9, BOXC)):
                ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), bw, bh,
                                       facecolor="none", edgecolor=col,
                                       lw=lw, zorder=5 if lw > 1 else 6,
                                       alpha=1.0 if c >= 4 else 0.75))
        ims.append(im)
        axes.append(ax)

    for c, (im, lab) in enumerate(zip(ims[4:], ["variance (grey$^2$)",
                                                "|error| (grey)"])):
        cax = axes[4 + c].inset_axes([0.0, -0.135, 1.0, 0.062])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.outline.set_linewidth(0.6)
        cb.outline.set_edgecolor("#333333")
        hi = im.get_clim()[1]
        cb.set_ticks([0.0, 0.25 * hi, hi])
        cb.set_ticklabels(["0", f"{0.25 * hi:.3f}", f"{hi:.3f}"])
        cb.ax.tick_params(labelsize=6.4, width=0.6, length=2.0, pad=1.5)
        cb.set_label(lab, fontsize=6.6, color=SUB, labelpad=1.5)

    axes[0].text(0.0, -0.075, f"the four members roll the same dynamics to "
                 f"$t$ = {T_TRUE:.2f} with offsets {DT[0]:+.3f} ... "
                 f"{DT[-1]:+.3f}", transform=axes[0].transAxes, ha="left",
                 va="top", fontsize=6.8, color=MUTED)

    # ---- patch-wise scatter -------------------------------------------------
    axs = fig.add_subplot(gs[1, 0:3])
    style_axes(axs)
    axs.grid(color=GRID, lw=0.7, zorder=0)
    axs.scatter(pd_[~pin], pe[~pin], s=9, facecolor=SCAT, alpha=0.42,
                edgecolor="none", zorder=3, label="patch outside the boxes")
    axs.scatter(pd_[pin], pe[pin], s=22, facecolor=HL, edgecolor="white",
                linewidth=0.5, zorder=4, label="patch inside a box")
    k = np.polyfit(pd_, pe, 1)
    xs = np.linspace(0, pd_.max() * 1.02, 2)
    axs.plot(xs, np.polyval(k, xs), color="#0b3d63", lw=1.2,
             ls=(0, (3.2, 2.0)), zorder=5)
    axs.set_xlim(0, pd_.max() * 1.05)
    axs.set_ylim(0, pe.max() * 1.10)
    axs.set_xlabel(f"patch disagreement  ({PATCH}$\\times${PATCH} px mean of "
                   f"tile 5)", fontsize=7.6)
    axs.set_ylabel("patch |error|\n(tile 6)", fontsize=7.6)
    axs.tick_params(labelsize=6.8, length=2.4)
    axs.set_title(f"Disagreement predicts error: $r$ = {r:.2f} over "
                  f"{pd_.size} patches", fontsize=8.4, fontweight="bold",
                  pad=4.0)
    leg = axs.legend(loc="upper left", frameon=True, facecolor="white",
                     edgecolor=HAIR, framealpha=0.94, fontsize=6.8,
                     handlelength=1.2, labelspacing=0.35, borderpad=0.4)
    leg.get_frame().set_linewidth(0.5)

    # ---- member vs ensemble RMSE -------------------------------------------
    axb = fig.add_subplot(gs[1, 3:6])
    style_axes(axb)
    axb.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    xs = np.arange(5)
    vals = np.append(rmse, rmse_ens)
    cols = ["#9cc3dd"] * 4 + ["#0b3d63"]
    axb.bar(xs, vals, width=0.62, color=cols, edgecolor="white",
            linewidth=0.6, zorder=3)
    for x, v in zip(xs, vals):
        axb.text(x, v + 0.0012, f"{v:.4f}", ha="center", va="bottom",
                 fontsize=6.8, color=INK if x == 4 else SUB, zorder=5)
    best = int(np.argmin(rmse))
    axb.plot([-0.42, 4.52], [rmse[best], rmse[best]], color="#8a8a8a",
             lw=0.9, ls=(0, (2.6, 2.0)), zorder=4)
    axb.text(4.66, rmse[best], f"best single\nmember\n({rmse[best]:.4f})",
             ha="left", va="center", fontsize=6.8, color=MUTED,
             linespacing=1.5, zorder=6)
    axb.set_xlim(-0.6, 6.0)
    axb.set_ylim(0, vals.max() * 1.30)
    axb.set_xticks(xs)
    axb.set_xticklabels([f"m{k + 1}" for k in range(4)] + ["ensemble\nmean"],
                        fontsize=7.0)
    axb.set_ylabel("RMSE vs GT", fontsize=7.6, labelpad=2.0)
    axb.tick_params(labelsize=6.8, length=2.4)
    axb.set_title("Averaging the members removes part of that error "
                  "(grey levels)", fontsize=8.4, fontweight="bold", pad=4.0)

    # ---- headline + arithmetic ---------------------------------------------
    names = " and ".join(b[1] for b in boxes)
    shown_best, shown_ens = f"{rmse[best]:.4f}", f"{rmse_ens:.4f}"
    savefig(fig, "1179_wm_unc_ensemble_disagree.png")


if __name__ == "__main__":
    main()

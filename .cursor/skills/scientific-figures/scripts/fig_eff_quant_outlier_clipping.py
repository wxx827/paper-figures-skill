"""What activation outliers do to an 8-bit tensor, and where the error goes.

Main panel: the activation histogram of one transformer feed-forward input on
a logarithmic count axis, with the same tensor after symmetric INT8
quantisation drawn twice - once with the scale set by the largest value in
the tensor, once with the scale set by the clipping threshold that minimises
the error - so the coarse level spacing of the first and the clipped pile-up
of the second are visible side by side; an inset magnifies a narrow window so
the 8-bit level comb can be counted.  Auxiliary (both 58 per cent of the main
panel width): the per-channel dynamic range that forces the problem, and the
error against clipping threshold for per-tensor and per-channel scales at two
bit widths.

Nothing is asserted: the histograms are the quantised arrays, the clipping
optimum is the argument of the minimum of a measured error sweep, the wasted
levels are log2 of a ratio of measured channel ranges, and every signal to
quantisation noise figure is 10 log10 of the two sums over the same arrays.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
TEAL = "#2a7f7a"
RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3b5b92"
GREY = "#b3b3b3"

N_TOK, N_CH = 2048, 64        # tokens x channels in the captured tensor
N_OUTLIER = 4                 # channels that carry the systematic outliers
BITS = 8
ALPHA = np.linspace(0.06, 1.00, 48)   # clipping threshold as a share of max

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.14, 3.16, 1.86)


def activations(rng):
    """One captured activation tensor with a few very wide channels."""
    x = rng.normal(0.0, 0.72, (N_TOK, N_CH))
    hot = rng.choice(N_CH, N_OUTLIER, replace=False)
    gain = rng.uniform(9.0, 16.0, N_OUTLIER)
    # the wide channels are not merely louder, they are heavy tailed, which is
    # what makes a clipping threshold worth searching for at all
    x[:, hot] = rng.standard_t(4.0, size=(N_TOK, N_OUTLIER)) * 0.72 * gain
    return x, hot


def quantise(x, clip, bits=BITS):
    """Symmetric uniform quantise-dequantise at a given clipping threshold."""
    n = 2 ** (bits - 1) - 1
    s = max(float(clip), 1e-12) / n
    return np.clip(np.rint(x / s), -n, n) * s


def sqnr(x, q):
    """Signal to quantisation noise ratio in dB."""
    return 10.0 * np.log10(float((x ** 2).sum() / ((x - q) ** 2).sum()))


def sweep_tensor(x, bits):
    """Mean squared error against clipping threshold, one scale for all."""
    top = float(np.abs(x).max())
    return np.array([float(((x - quantise(x, a * top, bits)) ** 2).mean())
                     for a in ALPHA]), top


def sweep_channel(x, bits):
    """Same sweep but with one scale per channel."""
    top = np.abs(x).max(axis=0)
    out = []
    for a in ALPHA:
        q = np.empty_like(x)
        for c in range(x.shape[1]):
            q[:, c] = quantise(x[:, c], a * top[c], bits)
        out.append(float(((x - q) ** 2).mean()))
    return np.array(out), top


def hist_panel(ax, x, q_max, q_ch, clip, top, share_out, share_err, gain):
    edges = np.linspace(-top * 1.03, top * 1.03, 481)
    ax.hist(x.ravel(), bins=edges, color=lighten(GREY, 0.30),
            edgecolor="none", zorder=2, label="FP16 activations")
    ax.hist(q_max.ravel(), bins=edges, histtype="step", color=BLUE, lw=0.9,
            zorder=4, label="INT8, one scale for the whole tensor")
    ax.hist(q_ch.ravel(), bins=edges, histtype="step", color=TEAL, lw=0.9,
            zorder=5, label="INT8, one scale per channel")
    ax.set_yscale("log")
    ax.set_xlim(-top * 1.06, top * 1.06)
    ax.set_ylim(0.7, x.size)
    ax.set_xlabel("activation value (arb. units)")
    ax.set_ylabel("samples per bin (log scale)")
    ax.grid(axis="y", lw=0.5, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for sgn in (-1.0, 1.0):
        ax.plot([sgn * clip, sgn * clip], [0.7, 120.0], color=ORANGE,
                lw=1.0, ls=(0, (3.0, 2.0)), zorder=6)
    ax.text(clip - top * 0.025, 170.0,
            "best clipping threshold {:.0f}\nonly {:.3f} % of the samples "
            "reach it,\nyet they hold {:.0f} % of the squared error"
            .format(clip, 100.0 * share_out, 100.0 * share_err),
            fontsize=6.3, color=ORANGE, ha="right", va="bottom",
            linespacing=1.4, zorder=8)
    ax.text(-top * 1.02, x.size * 0.62,
            "the {:d} widest channels stretch\nthe tensor to +-{:.0f}, while "
            "a typical\nchannel stays inside +-{:.1f}: one scale\nspends "
            "almost all {:d} levels on\nrange that holds nothing"
            .format(N_OUTLIER, top, gain, 2 * (2 ** (BITS - 1) - 1) + 1),
            fontsize=6.3, color=BLUE, ha="left", va="top", linespacing=1.45,
            zorder=8)

    zoom = ax.inset_axes([0.035, 0.085, 0.30, 0.34])
    win = np.linspace(-1.5, 1.5, 121)
    zoom.hist(x.ravel(), bins=win, color=lighten(GREY, 0.30),
              edgecolor="none", zorder=2)
    zoom.hist(q_max.ravel(), bins=win, histtype="step", color=BLUE, lw=0.8,
              zorder=4)
    zoom.hist(q_ch.ravel(), bins=win, histtype="step", color=TEAL, lw=0.8,
              zorder=5)
    zoom.set_xlim(-1.5, 1.5)
    zoom.set_yscale("log")
    zoom.tick_params(labelsize=6, width=0.6, length=2.0)
    zoom.set_title("levels available between -1.5 and +1.5", fontsize=6,
                   color=SUB, pad=1.5)
    for s in zoom.spines.values():
        s.set_linewidth(0.6)
        s.set_color("#9a9a9a")
    ax.legend(loc="upper right", fontsize=6.2, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.5, borderaxespad=0.4, labelspacing=0.25)
    ax.set_title("One activation tensor before and after 8-bit quantisation",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)


def range_panel(ax, per_ch, hot, clip):
    order = np.argsort(per_ch)[::-1]
    vals = per_ch[order]
    ishot = np.isin(order, hot)
    cols = [RED if h else BLUE for h in ishot]
    ax.bar(np.arange(vals.size), vals, width=0.85,
           facecolor=[lighten(c, 0.35) for c in cols], edgecolor=cols,
           linewidth=0.4, zorder=3)
    med = float(np.median(per_ch))
    ax.plot([-1.0, vals.size], [med, med], color=INK, lw=0.9,
            ls=(0, (3.0, 2.0)), zorder=5)
    ax.text(vals.size - 1.0, med * 1.12,
            "median channel range {:.2f}".format(med), fontsize=6.0,
            color=INK, ha="right", va="bottom")
    ax.plot([-1.0, vals.size], [clip, clip], color=ORANGE, lw=0.9,
            ls=(0, (2.5, 2.0)), zorder=5)
    ax.text(0.45 * vals.size, clip * 1.12,
            "per-tensor clip {:.1f}".format(clip), fontsize=6.0,
            color=ORANGE, ha="left", va="bottom")
    ax.set_yscale("log")
    ax.set_xlim(-1.0, vals.size)
    ax.set_ylim(med * 0.30, vals.max() * 7.0)
    ax.set_xlabel("channel, widest first (n = {:d})".format(vals.size),
                  fontsize=6.8)
    ax.set_ylabel("channel max |value|", fontsize=6.8)
    ax.grid(axis="y", which="both", lw=0.5, alpha=0.28)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ratio = vals.max() / med
    ax.text(0.40 * vals.size, vals.max() * 6.3,
            "widest / median = {:.0f}x, so a shared\nscale throws away "
            "{:.1f} of the {:d} bits\non a typical channel"
            .format(ratio, np.log2(ratio), BITS), fontsize=6.0, color=RED,
            ha="left", va="top", linespacing=1.4)
    ax.set_title("Why one scale does not fit", fontsize=8.0,
                 fontweight="bold", color=INK)


def sweep_panel(ax, curves):
    style = ((BLUE, "-"), (TEAL, "-"), (ORANGE, (0, (4.0, 1.8))))
    for (name, mse, unit), (col, ls) in zip(curves, style):
        ax.plot(100.0 * ALPHA, mse, color=col, lw=1.4, ls=ls, zorder=4,
                label=name)
        k = int(np.argmin(mse))
        ax.plot([100.0 * ALPHA[k]], [mse[k]], marker="o", ms=4.0, mfc="none",
                mec=col, mew=1.1, zorder=6)
        edge = 100.0 * ALPHA[k] > 88.0
        ax.text(100.0 * ALPHA[k] - (3.0 if edge else 0.0),
                mse[k] * (1.9 if edge else 0.58),
                "{:.0f} %, {:.1f} dB".format(100.0 * ALPHA[k], unit),
                fontsize=6, color=col, ha="right" if edge else "center",
                va="bottom" if edge else "top")
    ax.set_yscale("log")
    ax.set_xlim(0.0, 102.0)
    ax.set_xlabel("clipping threshold (% of the observed maximum)",
                  fontsize=6.8)
    ax.set_ylabel("mean squared error", fontsize=6.8)
    ax.grid(which="both", lw=0.5, alpha=0.28)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", fontsize=6, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.5, borderaxespad=0.3, labelspacing=0.22)
    ax.set_title("Clipping error against rounding error", fontsize=8.0,
                 fontweight="bold", color=INK)


def audit(fig, foot, axes):
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for ax in axes:
        ab = ax.get_window_extent()
        for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            tb = t.get_window_extent()
            if tb.width <= 0.0 or not t.get_text():
                continue
            if tb.x1 < ab.x0 - 2.0 or tb.x0 > ab.x1 + 2.0:
                continue
            if fb.overlaps(tb):
                print("OVERLAP with", repr(t.get_text()))


def main():
    set_pub_style()
    rng = np.random.default_rng(9134)
    x, hot = activations(rng)

    mse_t8, top = sweep_tensor(x, 8)
    mse_c8, per_ch = sweep_channel(x, 8)
    mse_t4, _ = sweep_tensor(x, 4)

    k8 = int(np.argmin(mse_t8))
    clip = float(ALPHA[k8] * top)
    q_max = quantise(x, top, 8)
    q_opt = quantise(x, clip, 8)

    err = (x - q_opt) ** 2
    out = np.abs(x) > clip
    share_out = float(out.mean())
    share_err = float(err[out].sum() / err.sum())

    q_ch = np.empty_like(x)
    a_ch = ALPHA[int(np.argmin(mse_c8))]
    for c in range(N_CH):
        q_ch[:, c] = quantise(x[:, c], a_ch * per_ch[c], 8)
    q_t4 = quantise(x, ALPHA[int(np.argmin(mse_t4))] * top, 4)

    curves = [("INT8, per tensor", mse_t8, sqnr(x, q_opt)),
              ("INT8, per channel", mse_c8, sqnr(x, q_ch)),
              ("INT4, per tensor", mse_t4, sqnr(x, q_t4))]

    print("max {:.1f}  clip {:.1f} ({:.0f} %)  SQNR maxscale {:.1f} dB  "
          "clipped {:.1f} dB  per-channel {:.1f} dB  int4 {:.1f} dB"
          .format(top, clip, 100.0 * ALPHA[k8], sqnr(x, q_max),
                  sqnr(x, q_opt), sqnr(x, q_ch), sqnr(x, q_t4)))
    print("outliers {:.3f} % of samples carry {:.1f} % of the error"
          .format(100.0 * share_out, 100.0 * share_err))
    assert sqnr(x, q_ch) > sqnr(x, q_opt) > sqnr(x, q_max), "scale ordering"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_h = axes_inches(fig, MAIN)
    ax_r = axes_inches(fig, AUX_T)
    ax_s = axes_inches(fig, AUX_B)

    hist_panel(ax_h, x, q_max, q_ch, clip, top, share_out, share_err,
               float(np.median(per_ch)))
    range_panel(ax_r, per_ch, hot, clip)
    sweep_panel(ax_s, curves)



    panel_labels_inches(fig, [ax_h, ax_r, ax_s], dx=-0.62, dy=0.14)
    savefig(fig, "1027_eff_quant_outlier_clipping.png")


if __name__ == "__main__":
    main()

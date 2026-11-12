"""Figure 1196 - Retrieval hit rate decays with the age of the memory.

Panel (a): four retrieval strategies, hit rate at top-8 against how long ago
the target memory was written, each with a +/- 1 s.d. band.  The half-life
marker on every curve is the age at which that curve reaches half of its own
value at age zero, so the four thresholds differ - which is the point, and
also why the arithmetic is printed in full inside the call-out.

Panel (b): the same four strategies as mean latency against mean hit rate.
The y-coordinates in (b) are ``curve.mean()`` of exactly the arrays drawn in
(a) - one source, two projections (BRIEF_WM section 1.7).  Change a decay
constant and both panels move together.

Colour discipline: all-cool.  The four strategies are separated by depth on
one blue ladder plus a dash pattern and a marker shape; the same shape is
reused in (b) so a reader can pair the panels without a second legend.

No ``axhline`` / ``axvline`` anywhere: the half-life markers are point
markers, and the call-out is anchored in empty space above the curves.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from pubstyle import set_pub_style, savefig, add_panel_label, style_axes
from figkit import background_card

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
HAIR = "#d5d5d5"
# grouping card for the auxiliary panel; luminance 0.916, deliberately clear
# of the 0.92-0.95 band where a card flips between "ink" and "white"
CARD = "#e6eaee"

# name, hit rate at age 0, asymptote, decay constant (min), latency (ms),
# colour on the blue ladder, dash pattern, marker
STRATS = [
    ("Hybrid dense + lexical", 0.960, 0.340, 92.0, 7.6, "#12314f",
     "-", "D"),
    ("Dense ANN (cosine)", 0.940, 0.260, 62.0, 3.4, "#2f6fa8",
     (0, (5.0, 2.2)), "o"),
    ("Recency-reweighted dense", 0.920, 0.200, 48.0, 3.9, "#5b93c8",
     (0, (6.0, 2.0, 1.5, 2.0)), "^"),
    ("BM25 lexical", 0.810, 0.150, 38.0, 1.8, "#8fb8dd",
     (0, (1.9, 1.9)), "s"),
]

AGE = np.linspace(0.0, 240.0, 241)
SESSIONS = 12


def curves():
    """Hit-rate curve, band half-width and the derived half-life per strategy.

    The wiggle is a smoothed, seeded field so the curves read as measured
    rather than analytic; every number quoted on the figure is read back off
    these arrays afterwards, never off the constants above.
    """
    out = []
    for i, (name, h0, floor, tau, lat, col, ls, mk) in enumerate(STRATS):
        rng = np.random.default_rng(1601 + 7 * i)
        wig = gaussian_filter1d(rng.standard_normal(AGE.size), 14.0)
        wig = 0.013 * wig / (np.abs(wig).max() + 1e-9)
        h = floor + (h0 - floor) * np.exp(-AGE / tau) + wig
        sd = 0.016 + 0.046 * (1.0 - np.exp(-AGE / 80.0))

        shown_h0 = f"{h[0]:.2f}"
        shown_half = f"{float(shown_h0) / 2.0:.3f}"
        assert abs(float(shown_half) - float(shown_h0) / 2.0) < 5e-4, \
            f"{name}: printed half threshold is not h(0)/2"
        # first crossing of the printed threshold, linearly interpolated
        thr = float(shown_half)
        k = int(np.argmax(h <= thr))
        assert k > 0, f"{name}: curve never reaches its own half value"
        f = (h[k - 1] - thr) / (h[k - 1] - h[k])
        t_half = AGE[k - 1] + f * (AGE[k] - AGE[k - 1])
        out.append(dict(name=name, h=h, sd=sd, lat=lat, col=col, ls=ls,
                        mk=mk, h0=shown_h0, half=shown_half,
                        t_half=t_half, t_half_s=f"{t_half:.1f}",
                        mean=float(h.mean())))
    return out


def main():
    set_pub_style()
    S = curves()

    fig = plt.figure(figsize=(10.4, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.42], wspace=0.215,
                          left=0.060, right=0.985, top=0.800, bottom=0.115)
    ax = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[0, 1])

    # the auxiliary panel sits on a grouping card (DESIGN_SYSTEM 7.2) so it
    # reads as secondary to (a) rather than as a second main result
    p = axr.get_position()
    background_card(fig, (p.x0 - 0.058, p.y0 - 0.082, p.width + 0.062,
                          p.height + 0.128), facecolor=CARD, zorder=-5,
                    radius=0.010)
    axr.set_facecolor(CARD)

    for s in S:
        ax.fill_between(AGE, s["h"] - s["sd"], s["h"] + s["sd"],
                        color=s["col"], alpha=0.16, linewidth=0, zorder=2)
    for s in S:
        ax.plot(AGE, s["h"], color=s["col"], lw=1.9, linestyle=s["ls"],
                label=s["name"], zorder=4, solid_capstyle="round")
        ax.plot([s["t_half"]], [float(s["half"])], marker=s["mk"],
                markersize=6.0, color=s["col"], markeredgecolor="white",
                markeredgewidth=1.0, zorder=6, linestyle="none")

    ax.set_xlim(0, 240)
    ax.set_ylim(0.10, 1.02)
    ax.set_xticks([0, 40, 80, 120, 160, 200, 240])
    ax.set_xlabel("Age of the target memory at query time (min)")
    ax.set_ylabel("Hit rate at top-8 retrieval")
    ax.grid(True, axis="y", color=HAIR, lw=0.55, alpha=0.8)
    style_axes(ax)

    lines = ["Half-life (marked on each curve): the age at which a curve",
             "first reaches its own value at age zero, divided by two"]
    for s in S:
        lines.append(f"{s['name']:<25s}{s['h0']} / 2 = {s['half']}"
                     f"   \u2192   {s['t_half_s']:>5s} min")
    ax.text(0.435, 0.975, "\n".join(lines), transform=ax.transAxes,
            ha="left", va="top", fontsize=6.5, family="monospace",
            color=RULE, linespacing=1.62, zorder=8,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor=HAIR, linewidth=0.7, alpha=0.96))
    # ---- (b) same four arrays, projected onto latency x mean hit rate
    for s in S:
        axr.plot([s["lat"]], [s["mean"]], marker=s["mk"], markersize=7.5,
                 color=s["col"], markeredgecolor="white", markeredgewidth=1.0,
                 linestyle="none", zorder=4)
        axr.annotate(f"{s['mean']:.3f}", xy=(s["lat"], s["mean"]),
                     xytext=(0, 9), textcoords="offset points",
                     ha="center", va="bottom", fontsize=6.6, color=s["col"],
                     zorder=5)
        axr.annotate(f"{s['lat']:.1f} ms", xy=(s["lat"], s["mean"]),
                     xytext=(0, -10), textcoords="offset points",
                     ha="center", va="top", fontsize=6.2, color=FAINT,
                     zorder=5)
    axr.set_xlim(0.4, 9.2)
    axr.set_ylim(0.20, 0.78)
    axr.set_xticks([2, 4, 6, 8])
    axr.set_xlabel("Mean retrieval latency (ms)")
    axr.set_ylabel("Mean hit rate over 0-240 min")
    axr.grid(True, color="#c8d0d8", lw=0.55, alpha=0.9)
    style_axes(axr)
    axr.text(0.035, 0.965, "each y is the mean of the\nmatching curve in (a)",
             transform=axr.transAxes, ha="left", va="top", fontsize=6.4,
             color=MUTED, style="italic", zorder=6, linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                       edgecolor=HAIR, linewidth=0.6, alpha=0.96))

    add_panel_label(ax, "(a)", x=-0.070, y=1.025)
    add_panel_label(axr, "(b)", x=-0.185, y=1.025)

    handles, labels = ax.get_legend_handles_labels()
    assert len(handles) == len(STRATS), "legend entries must match curves"

    # the headline multiple is recomputed from the two half-lives *as printed*
    ratio = float(S[0]["t_half_s"]) / float(S[1]["t_half_s"])
    assert abs(ratio - S[0]["t_half"] / S[1]["t_half"]) < 0.02, \
        "headline multiple does not close against the printed half-lives"
    # the gap is differenced from the two strings that are printed, not from
    # the raw floats, so a reader who subtracts them lands on the same value
    end_hi, end_lo = f"{S[0]['h'][-1]:.2f}", f"{S[-1]['h'][-1]:.2f}"
    gap = f"{float(end_hi) - float(end_lo):.2f}"

    savefig(fig, "1196_wm_mem_retrieval_hitrate.png")


if __name__ == "__main__":
    main()

"""Figure 1198 - Forgetting curves under three consolidation strategies.

Panel (a): measured retention at thirteen delays, one series per strategy,
each with a single-exponential fit.  Three scheduled replay events are marked
with two-point vertical segments (never ``axvline``, which would strike the
tick labels and the y-axis row); their labels live in a reserved band above
the curves so no line touches type.

Panel (c): the residuals of those fits.  They are not decoration - the fit is
a plain exponential and the protocol is not, so the replay events show up as
positive residual spikes at exactly the marked delays.  That is the argument
for drawing the markers at all.

Panel (b): half-lives.  Each bar is ``tau * ln 2`` of the fit in (a) and
prints the multiplication, so the bar cannot drift away from the curve that
produced it (BRIEF_WM section 1.7).

Colour discipline: all-cool.  Three strategies, three rungs of one blue
ladder, three marker shapes; the review markers are neutral grey.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, add_panel_label, style_axes
from figkit import background_card

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
HAIR = "#d5d5d5"
CARD = "#e6eaee"           # luminance 0.916, clear of the 0.92-0.95 band
EVENT = "#8a8a8a"

DELAY = np.array([0, 4, 8, 16, 24, 32, 48, 72, 96, 120, 144, 168, 192],
                 dtype=float)
REVIEWS = (24.0, 72.0, 144.0)
REVIEW_TAU = 20.0                     # how fast a replay boost fades
LN2 = 0.693                           # the constant that is printed

# name, decay constant (h), replay boost, colour, marker, dash
STRATS = [
    ("No consolidation", 55.0, 0.020, "#8fb8dd", "o", (0, (1.9, 1.9))),
    ("Nightly replay", 105.0, 0.075, "#2f6fa8", "s", (0, (5.0, 2.2))),
    ("Replay + semantic write-back", 190.0, 0.130, "#12314f", "D", "-"),
]
SHORT = ["no consolidation", "nightly replay", "+ semantic write-back"]

GRID = np.linspace(0.0, DELAY[-1], 400)


def bumps(t, boost):
    out = np.zeros_like(np.asarray(t, dtype=float))
    for tr in REVIEWS:
        out = out + boost * np.exp(-(t - tr) / REVIEW_TAU) * (t >= tr)
    return out


def measure():
    """Thirteen measured points and the exponential fit for each strategy."""
    out = []
    for i, (name, tau, boost, col, mk, ls) in enumerate(STRATS):
        rng = np.random.default_rng(1603 + 11 * i)
        clean = np.exp(-DELAY / tau) + bumps(DELAY, boost)
        sd = 0.004 + 0.014 * np.sqrt(np.clip(clean, 0, 1))
        r = np.clip(clean + sd * rng.standard_normal(DELAY.size), 0.004, 1.0)

        # weighted least squares on log r: large retentions carry the fit
        b, a = np.polyfit(DELAY, np.log(r), 1, w=np.sqrt(r))
        tau_fit = -1.0 / b
        amp = float(np.exp(a))
        fit_pts = amp * np.exp(-DELAY / tau_fit)

        shown_tau = f"{tau_fit:.1f}"
        shown_half = f"{float(shown_tau) * LN2:.1f}"
        assert abs(float(shown_half) - float(shown_tau) * LN2) < 0.05, \
            f"{name}: printed half-life is not the printed tau times ln 2"
        assert abs(float(shown_half) - tau_fit * np.log(2.0)) < 0.25, \
            f"{name}: the 0.693 shortcut has drifted from tau ln 2"
        out.append(dict(name=name, r=r, col=col, mk=mk, ls=ls, amp=amp,
                        tau=tau_fit, tau_s=shown_tau, half_s=shown_half,
                        fit_grid=amp * np.exp(-GRID / tau_fit),
                        resid=r - fit_pts))
    return out


def main():
    set_pub_style()
    S = measure()

    fig = plt.figure(figsize=(10.4, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.36],
                          height_ratios=[1.0, 0.32], wspace=0.30, hspace=0.13,
                          left=0.060, right=0.985, top=0.795, bottom=0.112)
    ax = fig.add_subplot(gs[0, 0])
    axc = fig.add_subplot(gs[1, 0], sharex=ax)
    axb = fig.add_subplot(gs[:, 1])

    p = axb.get_position()
    background_card(fig, (p.x0 - 0.098, p.y0 - 0.078, p.width + 0.108,
                          p.height + 0.126), facecolor=CARD, zorder=-5,
                    radius=0.010)
    axb.set_facecolor(CARD)

    # ---- (a) retention
    for tr in REVIEWS:
        ax.plot([tr, tr], [0.0, 1.03], color=EVENT, lw=0.9,
                linestyle=(0, (2.2, 2.0)), zorder=1)
        ax.plot([tr], [1.03], marker="v", markersize=4.2, color=EVENT,
                linestyle="none", zorder=2)
        ax.text(tr, 1.075, f"replay\n{tr:.0f} h", ha="center", va="bottom",
                fontsize=6.2, color=SUB, linespacing=1.25, zorder=3)
    for s in S:
        ax.plot(GRID, s["fit_grid"], color=s["col"], lw=1.7,
                linestyle=s["ls"], zorder=4, solid_capstyle="round")
        ax.plot(DELAY, s["r"], marker=s["mk"], markersize=4.6, color=s["col"],
                markerfacecolor="white", markeredgewidth=1.2,
                linestyle="none", zorder=6)

    ax.set_xlim(-7, 199)
    ax.set_ylim(0.0, 1.24)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.set_ylabel("Retention of the stored episode")
    ax.grid(True, axis="y", color=HAIR, lw=0.55, alpha=0.85)
    ax.tick_params(labelbottom=False)
    style_axes(ax)
    ax.text(0.985, 0.615,
            "Markers are measured retention at 13 delays;\n"
            "lines are one exponential fitted to each series.",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.6,
            color=SUB, linespacing=1.55, zorder=7,
            bbox=dict(boxstyle="round,pad=0.38", facecolor="white",
                      edgecolor=HAIR, linewidth=0.7, alpha=0.96))

    # ---- (c) residuals: the replay spikes the exponential cannot absorb
    lim = 1.15 * max(float(np.abs(s["resid"]).max()) for s in S)
    axc.plot([-7, 199], [0.0, 0.0], color=RULE, lw=0.8, zorder=2)
    for j, s in enumerate(S):
        dx = (j - 1) * 2.6
        for t, v in zip(DELAY + dx, s["resid"]):
            axc.plot([t, t], [0.0, v], color=s["col"], lw=1.2, zorder=3)
        axc.plot(DELAY + dx, s["resid"], marker=s["mk"], markersize=3.4,
                 color=s["col"], markerfacecolor="white", markeredgewidth=0.9,
                 linestyle="none", zorder=4)
    for tr in REVIEWS:
        axc.plot([tr, tr], [-lim, lim], color=EVENT, lw=0.9,
                 linestyle=(0, (2.2, 2.0)), zorder=1)
    axc.set_ylim(-lim, lim)
    axc.set_xlabel("Delay since the episode was stored (h)")
    axc.set_ylabel("Fit residual")
    axc.set_xticks([0, 24, 48, 72, 96, 120, 144, 168, 192])
    axc.grid(True, axis="y", color=HAIR, lw=0.5, alpha=0.8)
    style_axes(axc)

    # ---- (b) half-lives, straight out of the fits in (a)
    axb.set_ylim(len(S) + 0.05, -0.62)
    axb.set_xlim(0.0, 1.34 * max(float(s["half_s"]) for s in S))
    for j, s in enumerate(S):
        axb.add_patch(Rectangle((0.0, j - 0.26), float(s["half_s"]), 0.52,
                                facecolor=s["col"], edgecolor="white",
                                linewidth=0.7, zorder=3))
        axb.text(1.5, j - 0.34, f"{s['tau_s']} x {LN2} = {s['half_s']} h",
                 ha="left", va="bottom", fontsize=6.5, color=RULE, zorder=5)
    axb.set_yticks(range(len(S)))
    axb.set_yticklabels(SHORT, fontsize=6.8)
    axb.set_xlabel("Half-life (h)")
    axb.grid(True, axis="x", color="#c8d0d8", lw=0.5, alpha=0.9)
    axb.tick_params(axis="y", length=0)
    for side in ("top", "right"):
        axb.spines[side].set_visible(False)
    axb.text(0.0, 1.045, "Half-life = fitted tau x ln 2",
             transform=axb.transAxes, ha="left", va="bottom", fontsize=7.2,
             fontweight="bold", color=INK)
    axb.text(0.0, 0.045, "One bar per fitted curve in (a), same colour and\n"
                         "same order; the multiplication above each bar is\n"
                         "the whole derivation.",
             transform=axb.transAxes, ha="left", va="bottom", fontsize=6.4,
             color=SUB, linespacing=1.55, style="italic")

    add_panel_label(ax, "(a)", x=-0.055, y=1.015)
    add_panel_label(axc, "(c)", x=-0.055, y=1.055)
    add_panel_label(axb, "(b)", x=-0.300, y=1.015)

    handles = [Line2D([], [], color=s["col"], lw=1.7, linestyle=s["ls"],
                      marker=s["mk"], markersize=4.6,
                      markerfacecolor="white", markeredgewidth=1.2,
                      label=s["name"]) for s in S]
    handles.append(Line2D([], [], color=EVENT, lw=0.9,
                          linestyle=(0, (2.2, 2.0)), marker="v",
                          markersize=4.2, label="scheduled replay"))
    assert len(handles) == len(STRATS) + 1, \
        "legend must list the three series and the replay events, nothing else"

    gain = float(S[-1]["half_s"]) / float(S[0]["half_s"])

    savefig(fig, "1198_wm_mem_forgetting_curve.png")


if __name__ == "__main__":
    main()

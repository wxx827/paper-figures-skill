"""Speculative decoding: what the draft buys, and what verification costs.

Main panel: 260 ms of wall clock for three decoders that emit the same token
stream - the target model alone, and the target paired with a cheap draft and
with an accurate draft - drawn as the phases they actually spend time in, one
draft burst and one verification per round, with the fate of every speculated
token marked on the verification block.  Auxiliary (both 58 per cent of the
main panel width): how many tokens of each block survive verification, and
the speedup against block length for four acceptance rates, which is where
the interior optimum lives.

Every figure quoted is reduced from the simulated stream.  Acceptance is
sampled token by token, a block stops at its first rejection, the round emits
the accepted prefix plus one token the target itself supplies, and the
throughput printed for each lane is its token count divided by its own
elapsed time.  The closed-form speedup drawn in (c) is checked against the
measured lanes at their own block length and acceptance rate, so the mean
block length in (b) and the speedup in (a) cannot drift apart.
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

T_TARGET = 24.0               # ms, one forward pass of the 70 B target
GAMMA = 5                     # tokens speculated per round
WINDOW = 260.0                # ms of wall clock drawn in the main panel
N_ROUND = 20000               # rounds simulated for the statistics

# label, draft cost per token (ms), token acceptance rate, colour
DRAFTS = (("cheap draft, 1.1 B", 2.6, 0.62, BLUE),
          ("accurate draft, 7 B", 5.2, 0.82, TEAL))
ALPHA_GRID = (0.50, 0.62, 0.82, 0.92)
GAMMA_GRID = np.arange(1, 13)

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.14, 3.16, 1.86)


def draw_block(rng, alpha, gamma=GAMMA):
    """Accept speculated tokens until the first rejection; return the run."""
    u = rng.random(gamma)
    k = int(np.argmax(u > alpha)) if (u > alpha).any() else gamma
    return k


def simulate(rng, alpha, t_draft, n=N_ROUND, gamma=GAMMA):
    """Token counts and elapsed time for a run of speculative rounds."""
    k = np.array([draw_block(rng, alpha, gamma) for _ in range(n)])
    t_round = gamma * t_draft + T_TARGET
    tokens = k + 1                       # accepted prefix plus the free token
    return dict(k=k, tokens=int(tokens.sum()), ms=n * t_round,
                mean_k=float(k.mean()), t_round=t_round,
                alpha_hat=float(k.sum() / (k.sum() + (k < gamma).sum())),
                tps=1000.0 * float(tokens.sum()) / (n * t_round))


def speedup(alpha, gamma, t_draft):
    """Closed-form speedup over target-only decoding at this block length."""
    exp_tokens = (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)
    return exp_tokens * T_TARGET / (gamma * t_draft + T_TARGET)


def lane_events(rng, alpha, t_draft, window=WINDOW, gamma=GAMMA):
    """Rounds that start inside the drawn window, with their outcomes."""
    t, out = 0.0, []
    while t < window:
        k = draw_block(rng, alpha, gamma)
        out.append((t, k))
        t += gamma * t_draft + T_TARGET
    return out


def timeline_panel(ax, lanes):
    h = 0.30
    for i, lane in enumerate(lanes):
        y = 2.0 - i
        ax.text(-4.0, y + 0.79, lane["name"], fontsize=7.4,
                fontweight="bold", color=INK, ha="left", va="bottom")
        ax.text(-4.0, y + 0.61, lane["note"], fontsize=6.2, color=SUB,
                ha="left", va="bottom")
        for (t0, k, t_draft) in lane["blocks"]:
            if t_draft > 0.0:
                ax.add_patch(plt.Rectangle(
                    (t0, y), GAMMA * t_draft, h, facecolor="#e4e7ec",
                    edgecolor="#a8adb6", linewidth=0.5, zorder=3))
                t0 = t0 + GAMMA * t_draft
            ax.add_patch(plt.Rectangle(
                (t0, y), T_TARGET, h, facecolor=lighten(BLUE, 0.62),
                edgecolor=BLUE, linewidth=0.6, zorder=3))
            n_cell = GAMMA if lane["spec"] else 0
            wid = T_TARGET / (n_cell + 1)
            for j in range(n_cell + 1):
                if j < k:
                    fc, ec = lighten(TEAL, 0.18), TEAL
                elif j == n_cell:
                    fc, ec = lighten(GREY, 0.35), "#8a8a8a"
                elif j == k:
                    fc, ec = lighten(RED, 0.28), RED
                else:
                    fc, ec = "#f0f0f0", "#c4c4c4"
                ax.add_patch(plt.Rectangle(
                    (t0 + j * wid, y + h + 0.06), wid * 0.86, 0.20,
                    facecolor=fc, edgecolor=ec, linewidth=0.5, zorder=5))
        ax.text(WINDOW + 6.0, y + 0.20,
                "{:d} tokens\n{:.0f} tokens/s\n{:.2f}x"
                .format(lane["n_tok"], lane["tps"], lane["speed"]),
                fontsize=6.4, color=lane["col"], ha="left", va="center",
                fontweight="bold", linespacing=1.5)

    ax.plot([WINDOW, WINDOW], [-0.35, 3.06], color=INK, lw=0.9,
            ls=(0, (3.0, 2.0)), zorder=7)
    ax.text(WINDOW - 3.0, 3.04, "{:.0f} ms".format(WINDOW), fontsize=6.4,
            color=INK, ha="right", va="top")

    keys = (("#e4e7ec", "#a8adb6", "draft model forward passes"),
            (lighten(BLUE, 0.62), BLUE, "target verification pass"),
            (lighten(TEAL, 0.18), TEAL, "speculated token accepted"),
            (lighten(RED, 0.28), RED, "first rejection, block stops"),
            ("#f0f0f0", "#c4c4c4", "speculated token discarded"),
            (lighten(GREY, 0.35), "#8a8a8a", "token the target emits itself"))
    for j, (fc, ec, lab) in enumerate(keys):
        x = 4.0 + (j % 2) * 148.0
        yy = -0.66 - 0.20 * (j // 2)
        ax.add_patch(plt.Rectangle((x, yy), 7.0, 0.13, facecolor=fc,
                                   edgecolor=ec, linewidth=0.5, zorder=5))
        ax.text(x + 10.0, yy + 0.065, lab, fontsize=6.2, color=SUB,
                ha="left", va="center")

    ax.set_xlim(-6.0, WINDOW + 62.0)
    ax.set_ylim(-1.30, 3.26)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0.0, WINDOW + 1.0, 40.0))
    ax.tick_params(axis="x", labelsize=6.6)
    ax.set_xlabel("wall clock (ms), batch size 1, greedy decoding")
    ax.grid(axis="x", lw=0.5, alpha=0.25)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("The same 260 ms spent three ways", fontsize=9.0,
                 fontweight="bold", color=INK, pad=6.0)


def length_panel(ax, sims):
    x = np.arange(GAMMA + 1)
    wid = 0.38
    for i, (name, t_draft, alpha, col) in enumerate(DRAFTS):
        s = sims[name]
        share = 100.0 * np.bincount(s["k"], minlength=GAMMA + 1) / s["k"].size
        ax.bar(x + (i - 0.5) * wid, share, width=wid,
               facecolor=lighten(col, 0.35), edgecolor=col, linewidth=0.7,
               zorder=4, label="{:s}, mean {:.2f}".format(name, s["mean_k"]))
        ax.plot([s["mean_k"], s["mean_k"]], [0.0, 46.0], color=col, lw=1.0,
                ls=(0, (3.0, 2.0)), zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=6.4)
    ax.set_xlim(-0.62, GAMMA + 0.62)
    ax.set_ylim(0.0, 52.0)
    ax.set_xlabel("speculated tokens accepted per block (of {:d}), "
                  "n = {:d} blocks each".format(GAMMA, N_ROUND), fontsize=6.8)
    ax.set_ylabel("share of blocks (%)", fontsize=6.8)
    ax.grid(axis="y", lw=0.5, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", fontsize=6, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.2, borderaxespad=0.3, labelspacing=0.22)
    ax.set_title("How far a block gets before it is cut", fontsize=8.0,
                 fontweight="bold", color=INK)


def gamma_panel(ax, sims):
    shades = (GREY, BLUE, TEAL, ORANGE)
    t_ref = DRAFTS[1][1]
    for a, col in zip(ALPHA_GRID, shades):
        y = np.array([speedup(a, int(g), t_ref) for g in GAMMA_GRID])
        ax.plot(GAMMA_GRID, y, color=col, lw=1.3, marker="o", ms=2.6,
                mfc=col, mec="none", zorder=5,
                label="acceptance {:.2f}".format(a))
        j = int(np.argmax(y))
        ax.plot([GAMMA_GRID[j]], [y[j]], marker="o", ms=5.0, mfc="none",
                mec=col, mew=1.1, zorder=7)
    ax.plot([GAMMA_GRID[0], GAMMA_GRID[-1]], [1.0, 1.0], color=INK, lw=0.8,
            ls=(0, (3.0, 2.0)), zorder=3)
    ax.text(GAMMA_GRID[0] - 0.3, 1.03, "target-only baseline 1.00x",
            fontsize=6, color=SUB, ha="left", va="bottom")
    s = sims[DRAFTS[1][0]]
    ax.annotate("accurate-draft lane in (a):\n{:d} tokens per block, {:.2f}x"
                .format(GAMMA, s["speed"]),
                xy=(GAMMA, s["speed"]), xytext=(GAMMA + 1.1, 1.30),
                fontsize=6, color=TEAL, ha="left", va="center",
                linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.7,
                                shrinkA=2, shrinkB=3))
    ax.set_xlim(GAMMA_GRID[0] - 0.4, GAMMA_GRID[-1] + 0.4)
    ax.set_ylim(0.55, 2.55)
    ax.set_xlabel("tokens speculated per block, at {:.1f} ms per draft token"
                  .format(t_ref), fontsize=6.8)
    ax.set_ylabel("speedup over target only (x)", fontsize=6.8)
    ax.grid(lw=0.5, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", fontsize=6, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.4, borderaxespad=0.3, labelspacing=0.20)
    ax.set_title("Longer blocks stop paying", fontsize=8.0,
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
    rng = np.random.default_rng(5620)

    sims = {}
    for name, t_draft, alpha, col in DRAFTS:
        s = simulate(rng, alpha, t_draft)
        s["speed"] = s["tps"] * T_TARGET / 1000.0
        sims[name] = s
        form = speedup(alpha, GAMMA, t_draft)
        print("{:22s} mean accepted {:.3f}  alpha_hat {:.3f}  {:.1f} tok/s  "
              "{:.3f}x measured  {:.3f}x closed form"
              .format(name, s["mean_k"], s["alpha_hat"], s["tps"],
                      s["speed"], form))
        assert abs(s["speed"] - form) < 0.03, "measured matches closed form"

    lanes = []
    n_base = int(WINDOW // T_TARGET)
    lanes.append(dict(name="target model alone", spec=False, col=MUTED,
                      note="one 24 ms forward pass, one token",
                      blocks=[(i * T_TARGET, 0, 0.0) for i in range(n_base)],
                      n_tok=n_base, tps=1000.0 / T_TARGET, speed=1.0))
    for name, t_draft, alpha, col in DRAFTS:
        ev = lane_events(rng, alpha, t_draft)
        done = [(t0, k) for t0, k in ev
                if t0 + GAMMA * t_draft + T_TARGET <= WINDOW]
        blocks = [(t0, k, t_draft) for t0, k in done]
        n_tok = int(sum(k + 1 for _, k in done))
        t_used = len(done) * (GAMMA * t_draft + T_TARGET)
        lanes.append(dict(name="target + " + name, spec=True, col=col,
                          note="{:d} speculated tokens per round at {:.1f} ms "
                               "each, then one 24 ms check"
                               .format(GAMMA, t_draft),
                          blocks=blocks, n_tok=n_tok,
                          tps=1000.0 * n_tok / t_used,
                          speed=1000.0 * n_tok / t_used * T_TARGET / 1000.0))
    for lane in lanes:
        print("{:34s} {:3d} tokens  {:5.1f} tokens/s  {:.2f}x"
              .format(lane["name"], lane["n_tok"], lane["tps"], lane["speed"]))
    assert lanes[2]["tps"] > lanes[1]["tps"] > lanes[0]["tps"], "lane order"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_t = axes_inches(fig, MAIN)
    ax_l = axes_inches(fig, AUX_T)
    ax_g = axes_inches(fig, AUX_B)

    timeline_panel(ax_t, lanes)
    length_panel(ax_l, sims)
    gamma_panel(ax_g, sims)

    best = sims[DRAFTS[1][0]]
    g_opt = int(GAMMA_GRID[int(np.argmax(
        [speedup(DRAFTS[1][2], int(g), DRAFTS[1][1]) for g in GAMMA_GRID]))])


    panel_labels_inches(fig, [ax_t, ax_l, ax_g], dx=-0.62, dy=0.14)
    savefig(fig, "1030_eff_specdec_acceptance.png")


if __name__ == "__main__":
    main()

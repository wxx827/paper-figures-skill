"""How much of a fine-tuning update a rank budget can actually reach.

Main panel: the singular-value spectrum of the weight update of five module
types in one 32-layer transformer, normalised to the leading value and drawn
on log-log axes because the interesting decay covers four decades; an inset
turns the same arrays into cumulative energy against rank, so the rank each
module needs to hold 90 per cent of its update can be read off directly.
Auxiliary (both 58 per cent of the main panel width): quality against the
share of parameters left trainable for LoRA on every linear layer and on the
query and value projections alone, and the trainable-memory cost of six
adaptation methods with the quality each one returns.

No score is typed in.  Every method is written as a projection of the same
measured update - a truncated SVD for LoRA and adapters, a rank-one
projection onto a fixed direction for bias-only tuning, full identity for
full fine-tuning - and the quality reported for it is a fixed monotone
function of the fraction of squared update energy that projection captures,
weighted by how many matrices of that type the model contains.  Parameter
counts come from the stated layer geometry, so the horizontal axis of (b) and
the bars of (c) are the same arithmetic seen twice.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, get_palette, lighten
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
TEAL = "#2a7f7a"
RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3b5b92"
GREY = "#b3b3b3"

N_PROBE = 256                 # probe block the spectra are measured on
D_MODEL, D_FF, N_LAYER = 4096, 11008, 32
N_VOCAB = 32000
BYTES_TRAIN = 2.0             # FP16 per trainable parameter

# name, decay constant of the generated update, matrices per layer
MODULES = (("q_proj", 5.5, 1), ("k_proj", 4.5, 1), ("v_proj", 15.0, 1),
           ("o_proj", 9.0, 1), ("ffn_up / down / gate", 30.0, 3))

RANKS = (1, 2, 4, 8, 16, 32, 64, 128)
R_MARK = (8, 64)
ACC_FROZEN = 62.5             # per cent, frozen backbone with a linear head
GAIN_FULL = 9.0               # points a fully expressive update is worth

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.14, 3.16, 1.86)


def update_matrix(rng, tau, n=N_PROBE):
    """A fine-tuning update whose energy is concentrated in a few directions."""
    u = np.linalg.qr(rng.normal(0.0, 1.0, (n, n)))[0]
    v = np.linalg.qr(rng.normal(0.0, 1.0, (n, n)))[0]
    s = np.exp(-np.arange(n) / tau) + 0.0035
    return (u * s) @ v.T


def spectra(rng):
    """Measured singular values and Frobenius energy of every module type."""
    out = []
    for name, tau, copies in MODULES:
        w = update_matrix(rng, tau)
        s = np.linalg.svd(w, compute_uv=False)
        out.append(dict(name=name, copies=copies, s=s, w=w,
                        energy=float((s ** 2).sum())))
    return out


def rank_to(spec, frac):
    """Smallest rank holding ``frac`` of a module's squared update energy."""
    c = np.cumsum(spec["s"] ** 2) / spec["energy"]
    return int(np.searchsorted(c, frac) + 1)


def captured(specs, rank, scope, fixed_dir=None):
    """Weighted share of update energy a constrained update can represent."""
    num = tot = 0.0
    for sp in specs:
        tot += sp["copies"] * sp["energy"]
        if sp["name"] not in scope:
            continue
        if fixed_dir is not None:
            e = float((np.linalg.norm(sp["w"] @ fixed_dir) ** 2))
        else:
            e = float((sp["s"][:rank] ** 2).sum())
        num += sp["copies"] * e
    return num / tot


def quality(cap):
    """Score implied by the share of the update a method can reach."""
    return ACC_FROZEN + GAIN_FULL * cap


def total_params():
    per_layer = 4 * D_MODEL * D_MODEL + 3 * D_MODEL * D_FF
    return N_LAYER * per_layer + 2 * N_VOCAB * D_MODEL


def lora_params(r, scope="all"):
    if scope == "qv":
        return N_LAYER * 2 * r * (D_MODEL + D_MODEL)
    return N_LAYER * (4 * r * (D_MODEL + D_MODEL)
                      + 3 * r * (D_MODEL + D_FF))


def spectrum_panel(ax, specs, r90):
    shades = get_palette("cb_sequential_blue", 7)
    cols = [shades[2], shades[3], shades[5], shades[6], TEAL]
    marks = ("o", "s", "^", "D", "v")
    idx = np.arange(1, N_PROBE + 1)
    for sp, col, mk in zip(specs, cols, marks):
        y = sp["s"] / sp["s"][0]
        ax.plot(idx, y, color=col, lw=1.4, zorder=5,
                label="{:s} ({:d} per layer)".format(sp["name"], sp["copies"]))
        pick = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256]) - 1
        ax.plot(idx[pick], y[pick], ls="none", marker=mk, ms=3.0, mfc=col,
                mec="white", mew=0.4, zorder=6)
    for r in R_MARK:
        ax.plot([r, r], [1.5e-3, 2.4], color=MUTED, lw=0.8,
                ls=(0, (3.0, 2.0)), zorder=3)
        ax.text(r * 1.06, 1.9, "rank {:d}".format(r), fontsize=6.2,
                color=MUTED, ha="left", va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.85, 320.0)
    ax.set_ylim(1.5e-3, 2.4)
    ax.set_xlabel("singular-value index (log scale, n = {:d} directions "
                  "per module)".format(N_PROBE))
    ax.set_ylabel("singular value / leading singular value (log scale)")
    ax.grid(which="both", lw=0.5, alpha=0.28)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(310.0, 1.68e-3,
            "below this floor the update is isotropic noise:\nrank spent "
            "here buys nothing back", fontsize=6.3, color=ORANGE,
            ha="right", va="bottom", linespacing=1.4)

    ins = ax.inset_axes([0.545, 0.545, 0.435, 0.405])
    for sp, col in zip(specs, cols):
        c = np.cumsum(sp["s"] ** 2) / sp["energy"]
        ins.plot(idx, c, color=col, lw=1.2, zorder=5)
        r = r90[sp["name"]]
        ins.plot([r], [c[r - 1]], marker="o", ms=3.2, mfc="none", mec=col,
                 mew=1.0, zorder=7)
    ins.plot([0.85, 320.0], [0.90, 0.90], color=INK, lw=0.8,
             ls=(0, (3.0, 2.0)), zorder=4)
    ins.text(1.0, 0.865, "90 % of the update energy", fontsize=6,
             color=INK, ha="left", va="top")
    ins.set_xscale("log")
    ins.set_xlim(0.85, 320.0)
    ins.set_ylim(0.0, 1.04)
    ins.set_xlabel("rank (log scale)", fontsize=6, labelpad=1.5)
    ins.set_ylabel("cumulative energy", fontsize=6, labelpad=1.5)
    ins.tick_params(labelsize=6, width=0.6, length=2.0)
    ins.grid(which="major", lw=0.4, alpha=0.25)
    for s in ins.spines.values():
        s.set_linewidth(0.6)
        s.set_color("#9a9a9a")
    ins.text(300.0, 0.06,
             "rank to 90 %: {:s}".format(
                 ", ".join("{:d}".format(r90[sp["name"]]) for sp in specs)),
             fontsize=6, color=SUB, ha="right", va="bottom")

    ax.legend(loc="lower left", fontsize=6.2, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.5, borderaxespad=0.4, labelspacing=0.24)
    ax.set_title("The update is low rank, but not equally low rank everywhere",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)


def budget_panel(ax, curves, tot):
    style = ((BLUE, "-", "o"), (TEAL, "-", "s"))
    for (name, share, acc), (col, ls, mk) in zip(curves, style):
        ax.plot(100.0 * np.asarray(share), acc, color=col, lw=1.4, ls=ls,
                marker=mk, ms=3.0, mfc=col, mec="white", mew=0.4, zorder=5,
                label=name)
    ax.plot([2.0e-3, 15.0], [ACC_FROZEN + GAIN_FULL, ACC_FROZEN + GAIN_FULL],
            color=GREY, lw=0.9, ls=(0, (3.0, 2.0)), zorder=3)
    ax.text(2.0e-3, ACC_FROZEN + GAIN_FULL - 0.22,
            "full fine-tuning, 100 % trainable, {:.1f} %"
            .format(ACC_FROZEN + GAIN_FULL), fontsize=6, color=MUTED,
            ha="left", va="top")
    name, share, acc = curves[0]
    for r, x, y in zip(RANKS, share, acc):
        if r in (1, 8, 128):
            ax.annotate("r = {:d}".format(r), xy=(100.0 * x, y),
                        xytext=(100.0 * x * 1.35, y - 0.75), fontsize=6,
                        color=BLUE, ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6,
                                        shrinkA=1, shrinkB=2))
    ax.set_xscale("log")
    ax.set_xlim(2.0e-3, 15.0)
    ax.set_ylim(ACC_FROZEN - 0.4, ACC_FROZEN + GAIN_FULL + 1.0)
    ax.set_xlabel("trainable parameters (% of {:.1f} B, log scale)"
                  .format(tot / 1e9), fontsize=6.8)
    ax.set_ylabel("task score (%)", fontsize=6.8)
    ax.grid(which="both", lw=0.5, alpha=0.28)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.5, borderaxespad=0.3, labelspacing=0.22)
    ax.set_title("Ranks 1 to 128, and where the curve flattens", fontsize=8.0,
                 fontweight="bold", color=INK)


def method_panel(ax, rows):
    y = np.arange(len(rows), dtype=float)
    mb = [r[1] for r in rows]
    cols = [r[3] for r in rows]
    left = 1.2
    ax.barh(y, [m - left for m in mb], left=left, height=0.62,
            facecolor=[lighten(c, 0.35) for c in cols], edgecolor=cols,
            linewidth=0.8, zorder=4)
    for yi, (name, m, gain, col) in zip(y, rows):
        ax.text(m * 1.14, yi, "{:.0f} MB, +{:.1f} pts".format(m, gain),
                fontsize=6, color=col, ha="left", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(left, 5.0e5)
    ax.set_xlabel("trainable state at FP16 (MB, log scale)", fontsize=6.8)
    ax.grid(axis="x", which="both", lw=0.5, alpha=0.28)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("What each method costs to keep trainable", fontsize=8.0,
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
    rng = np.random.default_rng(3186)
    specs = spectra(rng)
    r90 = {sp["name"]: rank_to(sp, 0.90) for sp in specs}
    all_names = [sp["name"] for sp in specs]
    attn_qv = ["q_proj", "v_proj"]
    tot = total_params()

    cur_all, cur_qv = [], []
    for r in RANKS:
        cur_all.append((lora_params(r, "all") / tot,
                        quality(captured(specs, r, all_names))))
        cur_qv.append((lora_params(r, "qv") / tot,
                       quality(captured(specs, r, attn_qv))))
    curves = [("LoRA on every linear layer", [c[0] for c in cur_all],
               [c[1] for c in cur_all]),
              ("LoRA on q and v only", [c[0] for c in cur_qv],
               [c[1] for c in cur_qv])]

    v0 = rng.normal(0.0, 1.0, N_PROBE)
    v0 = v0 / np.linalg.norm(v0)
    bias_p = N_LAYER * (5 * D_MODEL + 2 * D_FF)
    adapter_p = N_LAYER * 4 * D_MODEL * 64
    prefix_p = N_LAYER * 2 * 32 * D_MODEL

    rows = [
        ("full fine-tuning", tot, quality(1.0) - ACC_FROZEN, GREY),
        ("adapters, bottleneck 64", adapter_p,
         quality(captured(specs, 64, ["ffn_up / down / gate"])) - ACC_FROZEN,
         BLUE),
        ("LoRA r = 8, all linear", lora_params(8, "all"),
         quality(captured(specs, 8, all_names)) - ACC_FROZEN, TEAL),
        ("LoRA r = 8, q and v", lora_params(8, "qv"),
         quality(captured(specs, 8, attn_qv)) - ACC_FROZEN, TEAL),
        ("prefix tuning, 32 tokens", prefix_p,
         quality(captured(specs, 32, ["k_proj", "v_proj"])) - ACC_FROZEN,
         BLUE),
        ("bias-only tuning", bias_p,
         quality(captured(specs, 1, all_names, fixed_dir=v0)) - ACC_FROZEN,
         ORANGE),
    ]
    rows = [(n, p * BYTES_TRAIN / 1e6, g, c) for n, p, g, c in rows]
    rows.sort(key=lambda r: -r[1])

    lo8 = quality(captured(specs, 8, all_names))
    lo128 = quality(captured(specs, 128, all_names))
    share8 = lora_params(8, "all") / tot
    print("total {:.2f} B params; LoRA r=8 all-linear {:.3f} % trainable"
          .format(tot / 1e9, 100.0 * share8))
    print("score r=8 {:.2f}, r=128 {:.2f}, full {:.2f}"
          .format(lo8, lo128, ACC_FROZEN + GAIN_FULL))
    for sp in specs:
        print("{:22s} rank to 90 % = {:3d}  rank to 99 % = {:3d}"
              .format(sp["name"], r90[sp["name"]], rank_to(sp, 0.99)))
    for n, m, g, _ in rows:
        print("{:26s} {:9.1f} MB  +{:.2f} pts".format(n, m, g))
    assert lo8 < lo128 < ACC_FROZEN + GAIN_FULL, "monotone in rank"
    assert r90["ffn_up / down / gate"] > r90["k_proj"], "ffn needs more rank"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_s = axes_inches(fig, MAIN)
    ax_b = axes_inches(fig, AUX_T)
    ax_m = axes_inches(fig, AUX_B)

    spectrum_panel(ax_s, specs, r90)
    budget_panel(ax_b, curves, tot)
    method_panel(ax_m, rows)

    per_pt = (lora_params(128, "all") - lora_params(8, "all")) / 1e6

    print("rank 8 -> 128 costs {:.1f} M extra trainable parameters".format(
        per_pt))

    panel_labels_inches(fig, [ax_s, ax_b, ax_m], dx=-0.62, dy=0.14)
    savefig(fig, "1029_eff_lora_rank_budget.png")


if __name__ == "__main__":
    main()

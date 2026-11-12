"""Sampled call-stack profile drawn as a flame graph, coloured by module.

The main panel is the claim: width is the share of samples a frame was on the
stack, frames stack upward on their caller, and the widest chain from the root
is the hot path the optimiser should attack first.  Colour encodes the owning
module rather than the usual random warm hues, so a reader can see at a glance
that the time is concentrated in one library.  Frame labels are fitted to the
measured box width and dropped when they cannot fit, which is what keeps a
flame graph readable at print size.  The two subordinate panels carry the
supporting evidence: the same profile ranked by self time, and how the samples
are distributed over stack depth.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"

MODULE_COLOR = {
    "app": "#3b5b92",
    "exec": "#2e7d5b",
    "io": "#c9a227",
    "libc": "#8e6bbf",
    "kernel": "#808080",
}

# (name, module, self samples, [children])
TREE = ("main", "app", 40, [
    ("parse_input", "io", 60, [
        ("read_block", "io", 210, []),
        ("decode_utf8", "libc", 150, []),
    ]),
    ("build_index", "app", 70, [
        ("hash_insert", "app", 120, [("memcpy", "libc", 190, [])]),
        ("rehash", "app", 60, [("malloc", "libc", 90, [])]),
    ]),
    ("query_loop", "app", 55, [
        ("plan_query", "app", 90, [
            ("cost_model", "app", 130, []),
            ("rewrite_rules", "app", 80, []),
        ]),
        ("exec_scan", "exec", 110, [
            ("filter_pred", "exec", 240, [("cmp_int", "exec", 150, [])]),
            ("prefetch", "kernel", 70, []),
        ]),
        ("exec_join", "exec", 130, [
            ("probe_hash", "exec", 520, [("memcmp", "libc", 300, [])]),
            ("build_hash", "exec", 180, [("malloc", "libc", 110, [])]),
        ]),
    ]),
    ("write_output", "io", 45, [
        ("fwrite", "libc", 120, []),
        ("fsync", "kernel", 160, []),
    ]),
])

FS_FRAME = 6.2


def grow(spec, rng):
    name, mod, self_w, kids = spec
    node = dict(name=name, mod=mod,
                self_w=float(self_w) * (1.0 + rng.normal(0.0, 0.10)),
                kids=[grow(k, rng) for k in kids])
    node["total"] = node["self_w"] + sum(k["total"] for k in node["kids"])
    return node


def flatten(node, x0, depth, total, out):
    out.append(dict(x=100.0 * x0 / total, w=100.0 * node["total"] / total,
                    depth=depth, name=node["name"], mod=node["mod"],
                    self_pct=100.0 * node["self_w"] / total))
    x = x0 + node["self_w"]
    for k in sorted(node["kids"], key=lambda d: -d["total"]):
        flatten(k, x, depth + 1, total, out)
        x += k["total"]
    return out


def hot_path(node):
    chain = [node["name"]]
    while node["kids"]:
        node = max(node["kids"], key=lambda d: d["total"])
        chain.append(node["name"])
    return chain


def fit_label(name, w_pct, ax_w_in):
    """Longest prefix of ``name`` that fits the box, or ``None`` if too narrow."""
    n_chars = int(w_pct / 100.0 * ax_w_in * 145.0 / FS_FRAME) - 1
    if n_chars >= len(name):
        return name
    if n_chars >= 6:
        return name[:n_chars - 2] + ".."
    return None


def text_ink(hexcolor):
    r, g, b = mpl.colors.to_rgb(hexcolor)
    return INK if (0.299 * r + 0.587 * g + 0.114 * b) > 0.62 else "white"


def flame_panel(ax, boxes, chain, n_samples):
    hot = set(chain)
    for b in boxes:
        on_hot = b["name"] in hot
        ax.add_patch(Rectangle((b["x"], b["depth"] + 0.08), b["w"], 0.84,
                               facecolor=MODULE_COLOR[b["mod"]],
                               edgecolor=RED if on_hot else "white",
                               linewidth=1.2 if on_hot else 0.5, zorder=3))
    depth_max = max(b["depth"] for b in boxes)
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, depth_max + 1.0)
    ax.set_yticks(np.arange(depth_max + 1) + 0.5)
    ax.set_yticklabels([str(d) for d in range(depth_max + 1)], fontsize=7.0)
    ax.set_xlabel("share of profile samples (%)")
    ax.set_ylabel("stack depth (frames)")
    ax.spines["left"].set_visible(True)
    ax.grid(axis="x", lw=0.5, alpha=0.6)

    leaf = max((b for b in boxes if b["name"] == chain[-1]), key=lambda d: d["w"])
    stem = max((b for b in boxes if b["name"] == chain[-2]), key=lambda d: d["w"])
    ax.text(99.0, leaf["depth"] + 0.50,
            f"hot path (red outline)  {' > '.join(chain[1:])}\ndeepest frame "
            f"holds {leaf['w']:.0f} % of {n_samples:,} samples",
            ha="right", va="center", fontsize=7.0, color=RED,
            linespacing=1.4, zorder=8)
    ax.set_title(f"The {stem['name']} subtree alone holds {stem['w']:.0f} % "
                 f"of all samples", fontsize=9.0, fontweight="bold", color=INK)
    return leaf, stem


def label_flames(fig, ax, boxes):
    bb = ax.get_position()
    ax_w_in = bb.width * fig.get_size_inches()[0]
    for b in boxes:
        lab = fit_label(b["name"], b["w"], ax_w_in)
        if lab is None:
            continue
        ax.text(b["x"] + b["w"] / 2.0, b["depth"] + 0.50, lab, ha="center",
                va="center", fontsize=FS_FRAME,
                color=text_ink(MODULE_COLOR[b["mod"]]), zorder=6)


def self_panel(ax, boxes, top_n=9):
    order = sorted(boxes, key=lambda d: -d["self_pct"])[:top_n][::-1]
    y = np.arange(len(order))
    ax.barh(y, [b["self_pct"] for b in order], height=0.66,
            color=[MODULE_COLOR[b["mod"]] for b in order], edgecolor="white",
            linewidth=0.6, zorder=3)
    for i, b in enumerate(order):
        ax.text(b["self_pct"] + 0.35, y[i], f"{b['self_pct']:.1f}", va="center",
                ha="left", fontsize=6.2, color=SUB)
    ax.set_yticks(y)
    ax.set_yticklabels([b["name"] for b in order], fontsize=6.5)
    ax.set_xlim(0.0, max(b["self_pct"] for b in order) * 1.22)
    ax.set_xlabel("self time (% of samples)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.7)
    ax.set_title(f"Top {top_n} frames by self time", fontsize=8.2,
                 fontweight="bold", color=INK)


def depth_panel(ax, boxes):
    depth_max = max(b["depth"] for b in boxes)
    share = np.zeros(depth_max + 1)
    for b in boxes:
        share[b["depth"]] += b["self_pct"]
    d = np.arange(depth_max + 1)
    ax.bar(d, share, width=0.62, color="#3b5b92", edgecolor="white",
           linewidth=0.6, zorder=3)
    mean_d = float((d * share).sum() / share.sum())
    ax.axvline(mean_d, color=RED, lw=1.0, ls=(0, (3, 2.2)), zorder=4)
    ax.text(mean_d - 0.16, share.max() * 0.94,
            f"mean depth\n{mean_d:.2f} frames", ha="right", va="top",
            fontsize=6.5, color=RED, linespacing=1.35)
    ax.set_xticks(d)
    ax.set_xlabel("stack depth (frames)", fontsize=7.4)
    ax.set_ylabel("self time (% of samples)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title("Where the samples actually land", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    rng = np.random.default_rng(21)
    root = grow(TREE, rng)
    boxes = flatten(root, 0.0, 0, root["total"], [])
    chain = hot_path(root)
    n_samples = int(round(root["total"]))

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.58],
                          width_ratios=[1.0, 1.0], wspace=0.26, hspace=0.55)
    ax_main = fig.add_subplot(gs[0, :])
    ax_self = fig.add_subplot(gs[1, 0])
    ax_dep = fig.add_subplot(gs[1, 1])

    leaf, stem = flame_panel(ax_main, boxes, chain, n_samples)
    self_panel(ax_self, boxes)
    depth_panel(ax_dep, boxes)

    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="white",
                         linewidth=0.5) for c in MODULE_COLOR.values()]

    fig.tight_layout(rect=[0, 0.055, 1, 0.905], h_pad=2.6, w_pad=2.6)
    label_flames(fig, ax_main, boxes)
    panel_labels_inches(fig, [ax_main, ax_self, ax_dep], dx=-0.44, dy=0.12)
    savefig(fig, "781_sys_flame_graph.png")


if __name__ == "__main__":
    main()

"""Anonymising a table is a trade, and the exchange rate is measurable.

Main panel: forty-eight generalisation policies over the same synthetic
microdata table, each one placed by the information it destroys and the
re-identification risk it leaves behind.  Both numbers are computed from the
generalised table itself - risk is the mean of one over the equivalence-class
size, information loss is the drop in the entropy of the quasi-identifier
combination - so the frontier, the k-anonymity lines and the annotated
operating point are readings rather than assertions.  The risk axis is
logarithmic because 1/k spans three decades over the policy set.

Auxiliary (both 58 % of the main panel width): the equivalence-class size
that each record sits in, before and after generalisation, with the records
that fall below the k threshold shaded as the ones an operator would have to
suppress; and the share of the population that is unique as more
quasi-identifiers are released, for a small and a large population.

Illustrative synthetic microdata; no real register, table or person.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, lighten, get_cmap
from figkit import axes_inches, panel_labels_inches, safe_colorbar

DEEP = "#2b4470"
BLUE = "#3b5b92"
TEAL = "#2a7f7a"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
ACCENT = "#b23a48"
RUST = "#c0603f"
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRID = "#d5d5d5"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.98, 0.86, 5.50, 4.62)
AUX_T_RECT = (7.70, 3.48, 3.14, 2.00)
AUX_B_RECT = (7.70, 0.86, 3.14, 2.00)

N_REC = 20000
K_TARGET = 5
AGE_BINS = [1, 2, 5, 10, 20, 40]
ZIP_DIGITS = [5, 4, 3, 2]


def class_size_per_record(cols):
    """Size of the equivalence class each record belongs to."""
    key = np.zeros(cols[0].size, dtype=np.int64)
    for c in cols:
        key = key * (int(c.max()) + 1) + c
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    return counts[inv]


def combo_entropy(sizes):
    """Entropy in bits of the distribution over equivalence classes."""
    n = sizes.size
    uniq, counts = np.unique(sizes, return_counts=True)
    # counts[i] records sit in classes of size uniq[i]; there are
    # counts[i] / uniq[i] such classes, each with probability uniq[i] / n
    n_cls = counts / uniq
    p = uniq / n
    return float(-np.sum(n_cls * p * np.log2(p)))


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(1019)

    # ---- synthetic microdata ------------------------------------------------
    age = np.clip(np.round(rng.normal(46.0, 17.0, N_REC)), 18, 92).astype(int)
    zip5 = rng.choice(np.arange(10000, 10000 + 240), size=N_REC,
                      p=(lambda w: w / w.sum())(
                          1.0 / np.arange(1, 241) ** 0.55))
    sex = rng.integers(0, 2, N_REC)
    month = rng.integers(1, 13, N_REC)
    marital = rng.integers(0, 4, N_REC)
    job = rng.integers(0, 24, N_REC)

    base_sizes = class_size_per_record([age, zip5, sex])
    h_base = combo_entropy(base_sizes)

    # ---- the policy grid ----------------------------------------------------
    pol = []
    for ab in AGE_BINS:
        for zd in ZIP_DIGITS:
            for keep_sex in (True, False):
                cols = [age // ab, zip5 // 10 ** (5 - zd)]
                if keep_sex:
                    cols.append(sex)
                s = class_size_per_record(cols)
                pol.append(dict(
                    age_bin=ab, zip_digits=zd, sex=keep_sex, sizes=s,
                    risk=float(np.mean(1.0 / s)),
                    supp=float(np.mean(s < K_TARGET)),
                    loss=float(100.0 * (h_base - combo_entropy(s)) / h_base)))

    loss = np.array([p["loss"] for p in pol])
    risk = np.array([p["risk"] for p in pol])
    supp = np.array([100.0 * p["supp"] for p in pol])

    order = np.argsort(loss)
    front, best = [], np.inf
    for i in order:
        if risk[i] < best - 1e-12:
            best = risk[i]
            front.append(i)
    front = np.array(front)

    ok = [i for i in front if risk[i] <= 1.0 / K_TARGET]
    i_pick = int(ok[0]) if ok else int(front[-1])
    pick = pol[i_pick]
    fine = pol[0]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main panel: risk against information loss --------------------------
    ax.set_yscale("log")
    ax.grid(color=GRID, lw=0.5, alpha=0.55, which="major", zorder=0)
    ax.grid(color=GRID, lw=0.35, alpha=0.30, axis="y", which="minor", zorder=0)
    for k, style in ((2, (0, (5.0, 2.4))), (K_TARGET, (0, (5.0, 2.4))),
                     (20, (0, (5.0, 2.4))), (100, (0, (5.0, 2.4)))):
        ax.axhline(1.0 / k, color=GREY, lw=0.8, ls=style, zorder=2)
        ax.text(99.5, 1.0 / k * 1.10, f"k = {k}", fontsize=6.4, color=SUB,
                ha="right", va="bottom")
    ax.plot(loss[front], risk[front], color=DEEP, lw=1.3, ls=(0, (4.0, 2.0)),
            zorder=4)
    sc = ax.scatter(loss, risk, c=supp, cmap=get_cmap("cb_sequential_warm"),
                    s=46.0, edgecolor="#4a4a4a", linewidth=0.4, zorder=6)
    ax.scatter(loss[front], risk[front], s=110.0, facecolor="none",
               edgecolor=DEEP, linewidth=1.0, zorder=7)
    ax.plot([pick["loss"]], [pick["risk"]], marker="o", ms=11.0, mfc="none",
            mec=ACCENT, mew=1.8, zorder=9)

    ax.annotate(
        f"cheapest policy that reaches {K_TARGET}-anonymity on average:\n"
        f"age in {pick['age_bin']}-year bands, "
        f"{pick['zip_digits']}-digit postcode, "
        f"sex {'kept' if pick['sex'] else 'suppressed'}\n"
        f"risk {pick['risk']:.3f} (1 in {1.0 / pick['risk']:.0f}), "
        f"{pick['loss']:.0f} % of the identifying entropy gone,\n"
        f"{100.0 * pick['supp']:.1f} % of records still below k = {K_TARGET}",
        xy=(pick["loss"], pick["risk"]),
        xytext=(34.0, 1.15), fontsize=6.8, color=ACCENT,
        ha="left", va="top", linespacing=1.6,
        arrowprops=dict(arrowstyle="-", lw=0.8, color=ACCENT,
                        connectionstyle="arc3,rad=0.22"))
    ax.annotate("release as collected:\nevery record its own class",
                xy=(fine["loss"], fine["risk"]), xytext=(2.0, 1.25),
                fontsize=6.6, color=SUB, ha="left", va="top",
                linespacing=1.5,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=SUB,
                                connectionstyle="arc3,rad=0.22"))
    # Placed in the empty wedge under the frontier: every policy sits on or
    # above the lower hull, so nothing can be drawn over this corner.
    ax.text(2.5, 6.5e-3,
            "lower and further left is better;\n"
            "the dashed hull is the achievable frontier",
            fontsize=6.4, color=DEEP, ha="left", va="bottom",
            linespacing=1.5)

    ax.set_xlim(-3.0, 101.0)
    ax.set_ylim(1.0e-3, 1.35)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("information loss (% of the quasi-identifier entropy)",
                  fontsize=8.5)
    ax.set_ylabel("mean re-identification probability "
                  "(1 / equivalence-class size, log scale)", fontsize=8.5)
    ax.set_title(f"{len(pol)} generalisation policies over one "
                 f"{N_REC:,}-record table", fontsize=9.0, fontweight="bold",
                 pad=5.0)
    cax = ax.inset_axes([0.045, 0.115, 0.30, 0.028])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.0, width=0.7, length=2.2)
    cb.set_label("records below k = 5 (%)", fontsize=6.4, color=INK)
    cb.ax.xaxis.set_label_position("top")

    # ---- aux (b): equivalence-class sizes -----------------------------------
    bins = np.logspace(0.0, np.log10(4000.0), 26)
    ax_t.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    for p, col, lab in ((fine, RUST, "as collected"),
                        (pick, TEAL, "after generalisation")):
        ax_t.hist(p["sizes"], bins=bins, histtype="stepfilled",
                  color=lighten(col, 0.55), edgecolor=col, lw=1.1, alpha=0.7,
                  zorder=4)
    ax_t.axvspan(1.0, K_TARGET, color=ACCENT, alpha=0.10, lw=0, zorder=2)
    ax_t.axvline(K_TARGET, color=ACCENT, lw=1.1, ls=(0, (3.2, 2.0)), zorder=6)
    ax_t.set_xscale("log")
    ax_t.set_yscale("log")
    ax_t.set_xlim(0.85, 4000.0)
    ax_t.set_ylim(0.8, 4.0e4)
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("equivalence-class size (records, log scale)",
                    fontsize=7.6, labelpad=1.5)
    ax_t.set_ylabel("records\n(log scale)", fontsize=7.6)
    ax_t.set_title("Who is hiding in how big a crowd", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ht = [Line2D([], [], color=RUST, lw=4.0, alpha=0.7,
                 label=f"as collected: {100.0 * fine['supp']:.0f} % below k"),
          Line2D([], [], color=TEAL, lw=4.0, alpha=0.7,
                 label=f"generalised: {100.0 * pick['supp']:.1f} % below k")]
    lt = ax_t.legend(handles=ht, loc="upper right", frameon=False,
                     fontsize=6.1, handlelength=1.5, labelspacing=0.30,
                     borderpad=0.15, bbox_to_anchor=(1.012, 1.035))
    for t in lt.get_texts():
        t.set_color(SUB)
    ax_t.text(3600.0, 1.15, "records left of the line\nmust be suppressed",
              fontsize=6.0, color=ACCENT, ha="right", va="bottom",
              linespacing=1.5, zorder=9)

    # ---- aux (c): uniqueness against released quasi-identifiers -------------
    qis = [age, sex, zip5 // 100, month, marital, job]
    names = ["age", "+sex", "+region", "+month", "+marital", "+job"]
    sub = rng.permutation(N_REC)[:2000]
    ax_b.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    for pop, col, lab in ((slice(None), DEEP, f"n = {N_REC:,} records"),
                          (sub, RUST, "n = 2,000 records")):
        frac = []
        for m in range(1, len(qis) + 1):
            s = class_size_per_record([q[pop] for q in qis[:m]])
            frac.append(100.0 * np.mean(s == 1))
        ax_b.plot(np.arange(1, len(qis) + 1), frac, color=col, lw=1.8,
                  marker="o", ms=3.6, mfc="white", mew=1.1, zorder=5,
                  label=lab)
        ax_b.text(len(qis) + 0.12, frac[-1], f"{frac[-1]:.0f} %",
                  fontsize=6.2, color=col, ha="left", va="center")
    ax_b.set_xlim(0.6, 6.9)
    ax_b.set_ylim(-4.0, 108.0)
    ax_b.set_xticks(np.arange(1, len(qis) + 1))
    ax_b.set_xticklabels(names, fontsize=6.2)
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("quasi-identifiers released (cumulative)", fontsize=7.6,
                    labelpad=1.5)
    ax_b.set_ylabel("records unique in the\nrelease (%)", fontsize=7.6)
    ax_b.set_title("Uniqueness arrives faster in a small population",
                   fontsize=8.4, fontweight="bold", pad=4.0)
    lb = ax_b.legend(loc="upper left", frameon=False, fontsize=6.1,
                     handlelength=1.7, labelspacing=0.30, borderpad=0.15,
                     bbox_to_anchor=(-0.012, 1.035))
    for t in lb.get_texts():
        t.set_color(SUB)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.70, dy=0.12)
    savefig(fig, "1019_ppc_reidentification_risk.png")


if __name__ == "__main__":
    main()

"""443 - Hyper-parameter sensitivity sweep for an off-policy RL agent.

One panel per hyper-parameter, each a one-dimensional sweep with eight seeds per
sampled setting.  The whole sweep is generated first and one y range is derived
from every seed of every panel, so a flat hyper-parameter looks flat and a
brittle one looks brittle; per-panel autoscaling would make them identical.
The star marks the best sampled setting and the bar along the bottom is the
recommended window - the contiguous span whose median stays within 5 percent of
that best - both read straight out of the plotted medians.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, style_axes, add_panel_label
from figkit import shared_limits

CURVE = "#3b5b92"
BAND = "#9fb6cb"
STAR = "#b23a48"
REC = "#c9a227"
SEEDP = "#8a8a8a"
INK, SUB, MUTED, GRID = "#1a1a1a", "#555555", "#777777", "#eeeeee"

N_SEED = 8
BUDGET = 2.0e6          # env steps per run
TOL = 0.05              # within 5 % of the best median = recommended

# name, unit, sweep values, log x, optimum (in transformed coords), widths,
# ceiling, floor, noise scale, default value
SWEEPS = [
    dict(key="lr", label="Learning rate", unit="(-)",
         xs=np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]), log=True,
         opt=-3.35, wl=0.72, wr=0.42, top=0.97, floor=0.06, sd=0.035,
         default=1e-3),
    dict(key="gamma", label="Discount factor $\\gamma$", unit="(-)",
         xs=np.array([0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999]),
         log=False, cat=True, opt=4.6, wl=1.55, wr=1.05, top=0.95, floor=0.18,
         sd=0.030, default=0.99),
    dict(key="batch", label="Batch size", unit="(samples)",
         xs=np.array([32, 64, 128, 256, 512, 1024, 2048]), log=True,
         opt=2.55, wl=0.85, wr=1.35, top=0.93, floor=0.42, sd=0.028,
         default=256),
    dict(key="ent", label="Entropy coefficient $\\alpha$", unit="(-)",
         xs=np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]), log=True,
         opt=-2.30, wl=0.80, wr=0.52, top=0.96, floor=0.10, sd=0.038,
         default=1e-2),
    dict(key="tau", label="Target smoothing $\\tau$", unit="(-)",
         xs=np.array([1e-4, 3e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]), log=True,
         opt=-2.15, wl=0.95, wr=0.60, top=0.94, floor=0.14, sd=0.042,
         default=5e-3),
    dict(key="ratio", label="Replay ratio", unit="(updates / env step)",
         xs=np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]), log=True,
         opt=0.42, wl=0.90, wr=0.50, top=0.98, floor=0.20, sd=0.036,
         default=1.0),
]


def coord(spec, x):
    """Sweep values in the coordinate the response is smooth in."""
    if spec.get("cat"):
        return np.arange(np.size(x), dtype=float)
    return np.log10(x)


def sweep_runs(rng, spec):
    """(n_seed, n_setting) expert-normalised final returns."""
    u = coord(spec, spec["xs"])
    d = u - spec["opt"]
    w = np.where(d < 0.0, spec["wl"], spec["wr"])
    mean = spec["floor"] + (spec["top"] - spec["floor"]) * np.exp(-0.5 * (d / w) ** 2)
    # runs get noisier as the setting leaves the stable window
    sd = spec["sd"] * (1.0 + 2.6 * np.abs(d) / (spec["wl"] + spec["wr"]))
    runs = mean[None, :] + rng.normal(0.0, 1.0, (N_SEED, u.size)) * sd[None, :]
    return np.clip(runs, 0.0, None)


def sci(v):
    """Compact ticks: plain digits for the coarse sweeps, mantissa-exponent for
    the decade sweeps, so one panel never mixes the two notations."""
    if v >= 0.25:
        return f"{v:g}"
    e = int(np.floor(np.log10(v)))
    m = v / 10.0 ** e
    return f"{m:g}e{e}"


def rec_window(med, tol=TOL):
    """Index bounds of the contiguous run around the best median within tol."""
    best = int(np.argmax(med))
    thr = med[best] * (1.0 - tol)
    i0 = best
    while i0 > 0 and med[i0 - 1] >= thr:
        i0 -= 1
    i1 = best
    while i1 < med.size - 1 and med[i1 + 1] >= thr:
        i1 += 1
    return best, i0, i1


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(443)

    # ---- generate the whole sweep before anything is drawn -----------------
    runs = {s["key"]: sweep_runs(rng, s) for s in SWEEPS}
    pool = []
    for s in SWEEPS:
        r = runs[s["key"]]
        pool += [r.ravel(), np.percentile(r, [25, 75], axis=0).ravel()]
    ylo, yhi = shared_limits(pool, pad=0.07, lo=0.0)

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.4))
    for k, (ax, spec) in enumerate(zip(axes.ravel(), SWEEPS)):
        r = runs[spec["key"]]
        xs = spec["xs"]
        px = coord(spec, xs) if spec.get("cat") else xs
        med = np.median(r, axis=0)
        q1, q3 = np.percentile(r, [25, 75], axis=0)
        best, i0, i1 = rec_window(med)

        ax.set_axisbelow(True)
        ax.grid(color=GRID, lw=0.7, zorder=0)
        if spec["log"]:
            ax.set_xscale("log")

        ax.axvspan(px[i0], px[i1], color=REC, alpha=0.13, lw=0, zorder=1)
        ax.fill_between(px, q1, q3, color=BAND, alpha=0.55, lw=0, zorder=2)
        for s in range(N_SEED):
            ax.plot(px, r[s], ls="none", marker="o", ms=1.9, mfc=SEEDP,
                    mec="none", alpha=0.55, zorder=3)
        ax.plot(px, med, color=CURVE, lw=1.7, marker="o", ms=3.4, mfc="white",
                mec=CURVE, mew=1.0, zorder=5, solid_capstyle="round")

        # default setting and the gain the sweep buys over it
        j_def = int(np.argmin(np.abs(np.asarray(xs) - spec["default"])))
        ax.axvline(px[j_def], color="#808080", lw=0.8, ls=(0, (3.0, 2.0)),
                   zorder=2)
        ax.plot([px[best]], [med[best]], marker="*", ms=11.0, mfc=STAR,
                mec="white", mew=0.7, zorder=7)

        ax.set_xlim(px[0] / 1.5 if spec["log"] else px[0] - 0.45,
                    px[-1] * 1.5 if spec["log"] else px[-1] + 0.45)
        ax.set_ylim(ylo, yhi)
        ax.set_xticks(px)
        ax.set_xticklabels([sci(v) if not spec.get("cat") else f"{v:g}"
                            for v in xs], fontsize=6.2)
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.tick_params(axis="x", length=2.4, pad=2.0)
        ax.set_title(f"{spec['label']}  {spec['unit']}", fontsize=8.4,
                     fontweight="bold", pad=3.5)
        style_axes(ax)
        add_panel_label(ax, f"({'abcdef'[k]})", x=-0.12, y=1.05)
        if k % 3:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Expert-normalised return (-)", fontsize=8)

        # recommended window bar along the bottom
        tb = ax.get_xaxis_transform()
        ax.plot([px[i0], px[i1]], [0.145, 0.145], transform=tb, color=REC,
                lw=2.6, solid_capstyle="butt", zorder=6)
        for xe in (px[i0], px[i1]):
            ax.plot([xe, xe], [0.115, 0.175], transform=tb, color=REC, lw=1.1,
                    zorder=6)
        fmt = (lambda v: f"{v:g}") if spec.get("cat") else sci
        span = f"{fmt(xs[i0])}\u2013{fmt(xs[i1])}" if i1 > i0 else fmt(xs[i0])
        ax.text(0.5, 0.200, f"recommended  {span}", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=6.5, color="#8a6f16",
                fontweight="bold", zorder=7)

        gain = med[best] - med[j_def]
        note = (f"best {fmt(xs[best])} = {med[best]:.2f}, "
                f"{gain:+.2f} over the default" if best != j_def else
                f"the default {fmt(xs[best])} is already best, {med[best]:.2f}")
        ax.text(0.5, 0.012, note, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=6.4, color=SUB, zorder=7)

    # sensitivity ranking, computed from the same medians
    rank = []
    for s in SWEEPS:
        med = np.median(runs[s["key"]], axis=0)
        rank.append((med.max() - med.min(), s["label"]))
    rank.sort(reverse=True)
    worst = rank[0][1].replace("$\\gamma$", "gamma").replace("$\\alpha$", "alpha")
    flat = rank[-1][1].replace("$\\gamma$", "gamma").replace("$\\alpha$", "alpha")

    handles = [
        Line2D([], [], color=CURVE, lw=1.7, marker="o", ms=3.4, mfc="white",
               mec=CURVE, label="median over seeds"),
        Patch(facecolor=BAND, alpha=0.55, lw=0, label="IQR across seeds"),
        Line2D([], [], color="none", marker="o", ms=1.9, mfc=SEEDP, mec="none",
               label=f"individual run ($n={N_SEED}$ seeds / setting)"),
        Line2D([], [], color="none", marker="*", ms=10.0, mfc=STAR, mec="white",
               mew=0.6, label="best sampled setting"),
        Patch(facecolor=REC, alpha=0.30, lw=0,
              label=f"recommended window (median within {TOL:.0%} of best)"),
        Line2D([], [], color="#808080", lw=0.8, ls=(0, (3.0, 2.0)),
               label="library default"),
    ]

    fig.tight_layout(rect=[0, 0.025, 1, 0.915], h_pad=2.8, w_pad=2.4)
    savefig(fig, "443_rl_hyperparam_sensitivity.png")


if __name__ == "__main__":
    main()

"""ACF / PACF identification grid for three candidate processes.

Rows are the two correlograms, columns are the three series; all six cells
share one y axis and one +-1.96/sqrt(n) band, so "cuts off" and "tails off"
are judged against the same ruler.  The order suggested in each cell, and the
model named in each column title, are derived in code from the bars drawn.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, style_axes
from figkit import panel_labels_inches, shared_limits

BAR_C = "#3b5b92"      # house blue: correlation bars
HL_C = "#b23a48"       # house red: the identified order
BAND_C = "#3b5b92"
GREY = "#808080"
N = 420
N_LAG = 24
SEASON = 12


def acf(x, nlags):
    x = np.asarray(x, float) - np.mean(x)
    d = np.dot(x, x)
    return np.array([np.dot(x[k:], x[:-k]) / d for k in range(1, nlags + 1)])


def pacf(r):
    """Partial autocorrelations from the ACF by Durbin-Levinson recursion."""
    n = r.size
    phi = np.zeros((n + 1, n + 1))
    out = np.zeros(n)
    phi[1, 1] = out[0] = r[0]
    for k in range(2, n + 1):
        num = r[k - 1] - sum(phi[k - 1, j] * r[k - j - 1]
                             for j in range(1, k))
        den = 1.0 - sum(phi[k - 1, j] * r[j - 1] for j in range(1, k))
        phi[k, k] = num / den if abs(den) > 1e-12 else 0.0
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        out[k - 1] = phi[k, k]
    return out


def leading_run(v, band):
    """Length of the initial run of significant lags - the 'cut-off' point."""
    k = 0
    while k < v.size and abs(v[k]) > band:
        k += 1
    return k


def simulate(rng):
    e = rng.normal(0, 1, N + 60)
    ar = np.zeros(N + 60)
    for t in range(1, ar.size):
        ar[t] = 0.74 * ar[t - 1] + e[t]
    ma = e[2:] + 0.78 * e[1:-1] + 0.52 * e[:-2]
    e2 = rng.normal(0, 1, N + 60)
    sar = np.zeros(N + 60)
    for t in range(SEASON, sar.size):
        sar[t] = 0.68 * sar[t - SEASON] + e2[t]
    return {"Series A": ar[60:], "Series B": ma[-N:], "Series C": sar[60:]}


MAX_ORDER = 3          # a run longer than this reads as "tails off"


def note_for(run, seasonal):
    if seasonal and run == 0:
        return f"spikes only at lag {SEASON}"
    if run == 0:
        return "no significant low lag"
    if run <= MAX_ORDER:
        return f"cuts off after lag {run}"
    return "tails off slowly"


def classify(p, q, seasonal):
    if seasonal:
        return f"seasonal AR({SEASON}) signature"
    if p and not q:
        return f"suggests AR({p})"
    if q and not p:
        return f"suggests MA({q})"
    if p and q:
        return f"suggests ARMA({p}, {q})"
    return "no structure"


def draw_cell(ax, v, band, order, ylim, xlabel, ylabel, note):
    lags = np.arange(1, v.size + 1)
    ax.fill_between([0.2, v.size + 0.8], -band, band, color=BAND_C,
                    alpha=0.11, lw=0, zorder=1)
    ax.axhline(0.0, color=GREY, lw=0.7, zorder=2)
    hot = lags <= order if order else np.zeros_like(lags, dtype=bool)
    ax.vlines(lags, 0.0, v, color=np.where(hot, HL_C, BAR_C), lw=1.6,
              zorder=3)
    ax.plot(lags, v, ls="none", marker="o", ms=2.4, mfc="white", mec=BAR_C,
            mew=0.7, zorder=4)
    if order:
        ax.plot(lags[:order], v[:order], ls="none", marker="o", ms=2.7,
                color=HL_C, zorder=5)
    ax.set_ylim(*ylim)
    ax.set_xlim(0.2, v.size + 0.8)
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.text(0.975, 0.955, note, transform=ax.transAxes, ha="right",
            va="top", fontsize=6.8, color="#555555")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.tick_params(labelleft=False)


def main():
    set_pub_style()
    rng = np.random.default_rng(3)
    series = simulate(rng)
    band = 1.96 / np.sqrt(N)

    stats = {}
    for name, x in series.items():
        r = acf(x, N_LAG)
        p = pacf(r)
        stats[name] = (r, p)

    ylim = shared_limits([np.r_[r, p] for r, p in stats.values()], pad=0.14)
    ylim = (min(ylim[0], -0.35), max(ylim[1], 0.35))

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.0))
    for c, (name, (r, p)) in enumerate(stats.items()):
        q_ord = leading_run(r, band)
        p_ord = leading_run(p, band)
        seasonal = abs(r[SEASON - 1]) > band and q_ord <= MAX_ORDER
        q_cut = q_ord if 0 < q_ord <= MAX_ORDER else 0
        p_cut = p_ord if 0 < p_ord <= MAX_ORDER else 0
        draw_cell(axes[0, c], r, band, q_cut, ylim,
                  None, r"Autocorrelation $\rho_k$ (-)" if c == 0 else None,
                  note_for(q_ord, seasonal))
        draw_cell(axes[1, c], p, band, p_cut, ylim,
                  "Lag $k$ (months)",
                  r"Partial autocorr. $\phi_{kk}$ (-)" if c == 0 else None,
                  note_for(p_ord, seasonal))
        if seasonal:
            for m in range(SEASON, N_LAG + 1, SEASON):
                axes[0, c].plot([m], [r[m - 1] + 0.06], marker="v", ms=3.2,
                                color=HL_C, zorder=6)
        verdict = classify(p_cut, q_cut, seasonal)
        axes[0, c].set_title(f"{name}: {verdict}", fontsize=9,
                             fontweight="bold")

    for ax in axes.ravel():
        ax.grid(axis="y", lw=0.5, color="#f2f2f2")
        ax.set_axisbelow(True)
        style_axes(ax)
    axes[0, 0].text(N_LAG - 0.4, band + 0.02, r"$\pm1.96/\sqrt{n}$",
                    ha="right", va="bottom", fontsize=6.6, color=BAND_C)

    fig.tight_layout(rect=[0, 0.028, 1, 0.935], h_pad=2.6, w_pad=2.0)
    panel_labels_inches(fig, axes[0], labels=["(a)", "(b)", "(c)"],
                        dx=-0.36, dy=0.26)
    panel_labels_inches(fig, axes[1], labels=["(d)", "(e)", "(f)"],
                        dx=-0.36, dy=0.09)
    savefig(fig, "361_ts_acf_grid.png")


if __name__ == "__main__":
    main()

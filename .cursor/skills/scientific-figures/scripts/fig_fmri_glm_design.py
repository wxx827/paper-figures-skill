"""Figure 535 - First-level fMRI GLM: design matrix, HRF convolution, collinearity.

Main panel (a) is the design matrix itself; (b) shows where one of its columns
comes from - stimulus train, canonical HRF, predicted BOLD, all three drawn on
the same second axis so the reader can see the lag rather than be told about
it; (c) and (d) are the subordinate diagnostics that decide whether the matrix
is estimable at all.  Panel (b) plots the *actual* design column, so the
triptych cannot drift out of sync with the matrix beside it.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

from pubstyle import set_pub_style, savefig, get_cmap, COLORS, CATEGORICAL
from figkit import axes_inches, panel_labels_inches, safe_colorbar, shared_norm

TR = 2.0                      # repetition time (s)
N_SCANS = 180                 # volumes in the run
DT = 0.05                     # convolution grid (s)
ZOOM_S = 150.0                # window shown in the convolution triptych (s)

# (name, group, onsets s, duration s) - the experimental paradigm
CONDITIONS = [
    ("Cue", 0, np.arange(14.0, 356.0, 30.0), 0.5),
    ("Visual", 0, np.arange(20.0, 356.0, 60.0), 16.0),
    ("Auditory", 0, np.arange(44.0, 356.0, 60.0), 16.0),
    ("Motor", 0, np.arange(36.0, 356.0, 60.0), 4.0),
    ("Error", 0, np.array([68.0, 152.0, 233.0, 311.0]), 1.0),
]
MOTION = ["tx", "ty", "tz", "rx", "ry", "rz"]
GROUPS = [("Task", "#3b5b92"), ("Head motion", "#2e7d5b"),
          ("Drift + constant", "#8e6bbf")]
HILITE = 1                    # design column expanded in panel (b): "Visual"


def canonical_hrf(t, peak=6.0, under=16.0, ratio=1.0 / 6.0):
    """Double-gamma HRF, scaled to unit peak (arbitrary units)."""
    h = gamma.pdf(t, peak) - ratio * gamma.pdf(t, under)
    return h / np.abs(h).max()


def build_design(rng):
    """Return (X, names, group_id) for the first-level design."""
    t_fine = np.arange(0.0, N_SCANS * TR, DT)
    t_scan = np.arange(N_SCANS) * TR
    hrf = canonical_hrf(np.arange(0.0, 32.0, DT))
    cols, names, gid = [], [], []

    for name, _, onsets, dur in CONDITIONS:
        box = np.zeros_like(t_fine)
        for o in onsets:
            box[(t_fine >= o) & (t_fine < o + dur)] = 1.0
        conv = np.convolve(box, hrf)[:t_fine.size] * DT
        cols.append(np.interp(t_scan, t_fine, conv))
        names.append(name)
        gid.append(0)

    walk = np.cumsum(rng.normal(0, 1.0, (6, N_SCANS)), axis=1)
    walk -= walk.mean(axis=1, keepdims=True)
    scale = np.array([0.14, 0.11, 0.19, 0.09, 0.07, 0.12])   # mm and degrees
    for i, m in enumerate(MOTION):
        r = scale[i] * walk[i] / (np.abs(walk[i]).max() + 1e-9) * 6.0
        r[112:114] += 0.55 if i == 2 else 0.12                # a single spike
        cols.append(r)
        names.append(m)
        gid.append(1)

    for k in range(2):
        cols.append(np.cos(np.pi * (k + 1) * (2 * t_scan + TR) /
                           (2 * N_SCANS * TR)))
        names.append(f"DCT{k + 1}")
        gid.append(2)
    cols.append(np.ones(N_SCANS))
    names.append("Const")
    gid.append(2)
    return np.column_stack(cols), names, np.array(gid), hrf, t_fine, t_scan


def standardise(X):
    """Columns in comparable units so one grey scale serves the whole matrix."""
    Z = X - X.mean(axis=0, keepdims=True)
    s = Z.std(axis=0, ddof=1)
    out = np.where(s > 1e-9, Z / np.where(s > 1e-9, s, 1.0), 1.0)
    return out


def cosine_collinearity(X):
    """|cos theta| between design columns - SPM's design-orthogonality view."""
    n = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-12)
    return np.abs(n.T @ n)


def variance_inflation(X):
    """VIF from the correlation matrix of the non-constant regressors."""
    Z = standardise(X[:, :-1])
    R = np.corrcoef(Z, rowvar=False)
    return np.diag(np.linalg.inv(R + 1e-8 * np.eye(R.shape[0])))


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(29)
    X, names, gid, hrf, t_fine, t_scan = build_design(rng)
    Z = standardise(X)
    norm = shared_norm([Z], symmetric=True, robust=(1, 99))

    FW, FH = 10.4, 6.4
    y0, y1 = 0.78, 5.52                       # shared vertical band (inches)
    fig = plt.figure(figsize=(FW, FH))

    # ---- (a) design matrix -------------------------------------------------
    ax = axes_inches(fig, (0.62, y0, 2.26, y1 - y0))
    ax.imshow(Z, cmap="gray_r", norm=norm, aspect="auto",
                   interpolation="nearest", origin="upper",
                   extent=(-0.5, len(names) - 0.5, N_SCANS - 0.5, -0.5))
    for b in (4.5, 10.5):
        ax.axvline(b, color="white", lw=1.6)
    ax.add_patch(plt.Rectangle((HILITE - 0.5, -0.5), 1.0, N_SCANS,
                               fill=False, edgecolor=CATEGORICAL[0], lw=1.1,
                               zorder=5))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6.4)
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.tick_params(labelsize=7.0, length=2.5)
    ax.set_ylabel(f"Scan (TR = {TR:.1f} s)", fontsize=8.0)
    ax.set_title("Design matrix, standardised columns", fontsize=8.8,
                 fontweight="bold", pad=22)
    trans = ax.get_xaxis_transform()
    spans = [(-0.5, 4.5), (4.5, 10.5), (10.5, len(names) - 0.5)]
    for (lo, hi), (lab, col) in zip(spans, GROUPS):
        ax.plot([lo + 0.12, hi - 0.12], [1.012, 1.012], transform=trans,
                color=col, lw=2.0, clip_on=False, solid_capstyle="butt")
        ax.text((lo + hi) / 2, 1.026, lab, transform=trans, ha="center",
                va="bottom", fontsize=6.6, color=col, clip_on=False)

    # ---- (b) stimulus -> HRF -> predicted BOLD ----------------------------
    xb, wb = 3.52, 3.30
    hb = (y1 - y0 - 2 * 0.46) / 3.0
    box = np.zeros_like(t_fine)
    for o in CONDITIONS[HILITE][2]:
        box[(t_fine >= o) & (t_fine < o + CONDITIONS[HILITE][3])] = 1.0
    t_hrf = np.arange(0.0, 32.0, DT)
    series = [
        ("Stimulus train, %s blocks (on/off)" % CONDITIONS[HILITE][0].lower(),
         t_fine, box, "#3b5b92"),
        ("Canonical HRF kernel (a.u.)", t_hrf, hrf, "#c9a227"),
        ("Predicted BOLD = design column %d (a.u.)" % (HILITE + 1),
         t_scan, X[:, HILITE], CATEGORICAL[0]),
    ]
    axb = []
    for i, (lab, tt, yy, col) in enumerate(series):
        a = axes_inches(fig, (xb, y1 - (i + 1) * hb - i * 0.46, wb, hb))
        if i == 0:
            a.fill_between(tt, 0, yy, color=col, alpha=0.30, lw=0, step="mid")
        a.plot(tt, yy, color=col, lw=1.3)
        a.axhline(0.0, color=COLORS["baseline"], lw=0.6, zorder=0)
        a.set_xlim(0, ZOOM_S)
        a.set_title(lab, fontsize=7.6, color=COLORS["ink"], pad=2.5,
                    loc="left")
        a.tick_params(labelsize=7.0, length=2.5,
                      labelbottom=(i == len(series) - 1))
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        axb.append(a)
    axb[0].set_yticks([0, 1])
    axb[1].annotate("peak ~ 5 s", xy=(t_hrf[np.argmax(hrf)], 1.0),
                    xytext=(26, -2), textcoords="offset points", fontsize=6.6,
                    color=COLORS["sub"], va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.55,
                                    color=COLORS["grid"]))
    axb[1].annotate("post-stimulus undershoot",
                    xy=(t_hrf[np.argmin(hrf)], hrf.min()), xytext=(20, -9),
                    textcoords="offset points", fontsize=6.6,
                    color=COLORS["sub"], va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.55,
                                    color=COLORS["grid"]))
    axb[2].set_xlabel("Time from run onset (s)", fontsize=8.0)
    for i, sym in enumerate(("\u2193  convolved with", "\u2193  gives")):
        fig.text((xb + 0.10) / FW, (y1 - (i + 1) * hb - i * 0.46 - 0.23) / FH,
                 sym, fontsize=6.8, color=COLORS["muted"], ha="left",
                 va="center")
    fig.text((xb + wb) / FW, (y1 + 0.30) / FH,
             f"first {ZOOM_S:.0f} s of the {N_SCANS * TR:.0f} s run",
             fontsize=6.6, color=COLORS["muted"], ha="right", va="center")

    # ---- (c) collinearity, (d) variance inflation --------------------------
    xc, wc = 7.52, 2.10
    C = cosine_collinearity(X)
    axc = axes_inches(fig, (xc, y1 - wc, wc, wc))
    imc = axc.imshow(C, cmap=get_cmap("cb_sequential_blue"), vmin=0.0,
                     vmax=1.0, interpolation="nearest")
    axc.set_aspect("equal")
    for b in (4.5, 10.5):
        axc.axvline(b, color="white", lw=1.0)
        axc.axhline(b, color="white", lw=1.0)
    axc.set_xticks(range(len(names)))
    axc.set_yticks(range(len(names)))
    axc.set_xticklabels(names, rotation=90, fontsize=6.0)
    axc.set_yticklabels(names, fontsize=6.0)
    axc.tick_params(length=1.8, pad=1.5)
    for s in axc.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    axc.set_title("Regressor collinearity", fontsize=8.8, fontweight="bold",
                  pad=4)
    off = C - np.eye(len(names))
    i, j = np.unravel_index(np.argmax(off), off.shape)
    safe_colorbar(fig, imc, axc, label="|cos $\\theta$| (-)", side="right",
                  width=0.10, pad=0.09, shrink=1.0, labelsize=6.6,
                  ticks=[0, 0.25, 0.5, 0.75, 1.0])

    vif = variance_inflation(X)
    axd = axes_inches(fig, (xc + 0.30, y0, wc - 0.20, 1.62))
    cols = [GROUPS[g][1] for g in gid[:-1]]
    ypos = np.arange(len(vif))
    axd.barh(ypos, vif, color=cols, height=0.72, lw=0)
    axd.axvline(5.0, color=CATEGORICAL[0], lw=0.8, ls="--", dashes=(3, 2))
    axd.text(5.18, 0.97, "VIF = 5", fontsize=6.2, color=CATEGORICAL[0],
             va="top", ha="left", transform=axd.get_xaxis_transform())
    axd.set_yticks(ypos)
    axd.set_yticklabels(names[:-1], fontsize=6.0)
    axd.invert_yaxis()
    axd.set_xlim(0, max(6.2, vif.max() * 1.18))
    axd.set_xlabel("Variance inflation factor (-)", fontsize=7.2)
    axd.tick_params(labelsize=6.5, length=2.0, pad=1.5)
    for s in ("top", "right"):
        axd.spines[s].set_visible(False)

    panel_labels_inches(fig, [ax, axb[0], axc, axd], dx=-0.42, dy=0.34)

    savefig(fig, "535_fmri_glm_design.png")


if __name__ == "__main__":
    main()

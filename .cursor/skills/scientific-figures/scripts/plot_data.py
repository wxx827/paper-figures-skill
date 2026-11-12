"""Turn a client's data file into a publication-quality figure, from one command.

The 1219 generators in this library are worth reading, but almost all of them
compute their own data -- a Kaimal spectrum, a Kalman filter, a synthetic
microscopy scene -- and there is nothing in them to swap a CSV into. This is
the front door for the other case: someone has a table and wants it to look
like it belongs in a journal.

    python plot_data.py --data sales.csv --describe
    python plot_data.py --list-kinds
    python plot_data.py --data sales.csv --kind scatter_fit --x spend --y revenue --group region
    python plot_data.py --data trial.xlsx --kind forest --y estimate --lo ci_lo --hi ci_hi --labels arm

Columns are inferred when not given, so the first command that produces a
figure is usually just ``--data FILE --kind KIND``. Styling is not
configurable on purpose: it comes from ``pubstyle``, which is the part of this
library that carries the value.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import (set_pub_style, set_cn_font, savefig, style_axes,
                      get_palette, add_figure_header, header_height,
                      annotate_box, smooth_line,
                      CATEGORICAL, COLORS, FIGSIZES, TYPE_SCALE)
import dataio
from dataio import DataError

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

KINDS = {
    "scatter_fit": "scatter with a fitted trend and marginal distributions",
    "bar_group":   "grouped or stacked bars across categories",
    "line_ci":     "line / time series with a confidence band",
    "dist":        "distribution by group (box, violin or raincloud)",
    "corr":        "correlation matrix heat map of the numeric columns",
    "hist":        "histogram with a kernel density overlay",
    "ecdf":        "empirical cumulative distribution by group",
    "area":        "stacked area over an ordered axis",
    "donut":       "composition as a donut, with the share table",
    "forest":      "point estimates with confidence intervals",
    "lollipop":    "ranked categories as a lollipop chart",
    "errorbar":    "group means with error bars and the raw points",
    "survival":    "step / survival curves by group",
    "pairs":       "scatter-plot matrix of the numeric columns",
}


# ---- helpers ---------------------------------------------------------------
def _needs_cjk(df, *extra):
    """True when anything that will be drawn contains non-Latin text.

    A client's column headers and category levels go straight onto the axes.
    If they are Chinese and nobody enabled a CJK face, matplotlib draws a row
    of tofu boxes -- a silent failure that looks like the tool is broken.
    """
    def has(text):
        return any(ord(ch) > 0x2E7F for ch in str(text))

    for col in df.columns:
        if has(col):
            return True
        s = df[col]
        if s.dtype == object or str(s.dtype).startswith(("str", "string")):
            for v in s.dropna().unique()[:200]:
                if has(v):
                    return True
    return any(has(t) for t in extra if t)


def _finite(*arrays):
    """Row-wise drop of any position that is NaN in any of the inputs."""
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    ok = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        ok &= np.isfinite(a)
    return [a[ok] for a in arrays]


def _groups(df, group, limit=8):
    """Ordered group keys, largest first, with a hard cap.

    More than about eight series stops being a figure and starts being a
    lookup table, and the house palette only carries ten distinguishable hues.
    """
    if not group:
        return [None]
    counts = df[group].value_counts(dropna=True)
    keys = list(counts.index[:limit])
    if len(counts) > limit:
        print(f"note: {group!r} has {len(counts)} levels; showing the "
              f"{limit} largest")
    return keys


def _mask(df, group, key):
    return slice(None) if key is None else (df[group] == key).to_numpy()


def _colors(n):
    return get_palette("cb_safe", n) if n > 5 else CATEGORICAL[:n]


def _label(key, group):
    return str(group) if key is None else str(key)


def _fmt_axis(ax, xlabel=None, ylabel=None):
    if xlabel:
        ax.set_xlabel(str(xlabel), fontsize=TYPE_SCALE["axis_label"])
    if ylabel:
        ax.set_ylabel(str(ylabel), fontsize=TYPE_SCALE["axis_label"])
    style_axes(ax)


def _legend(ax, **kw):
    kw.setdefault("frameon", False)
    kw.setdefault("fontsize", TYPE_SCALE["legend"])
    return ax.legend(**kw)


def _datetime_axis(ax, series):
    import pandas as pd
    if pd.api.types.is_datetime64_any_dtype(series):
        fig = ax.get_figure()
        fig.autofmt_xdate(rotation=0, ha="center")


# ---- chart kinds -----------------------------------------------------------
# Each builder takes (df, args, roles) and returns (fig, subtitle).
def k_scatter_fit(df, a, roles):
    x = dataio.pick(df, a.x, "numeric", roles)
    y = dataio.pick(df, a.y, "numeric", roles, exclude=(x,))
    if not x or not y:
        raise DataError("scatter_fit needs two numeric columns; pass --x and --y")

    fig = plt.figure(figsize=FIGSIZES["double"])
    gs = fig.add_gridspec(2, 2, width_ratios=(1, 0.24),
                          height_ratios=(0.24, 1), wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    axx = fig.add_subplot(gs[0, 0], sharex=ax)
    axy = fig.add_subplot(gs[1, 1], sharey=ax)
    for spine in ("top", "right", "left"):
        axx.spines[spine].set_visible(False)
    for spine in ("top", "right", "bottom"):
        axy.spines[spine].set_visible(False)
    axx.tick_params(labelbottom=False, left=False, labelleft=False)
    axy.tick_params(labelleft=False, bottom=False, labelbottom=False)

    keys = _groups(df, a.group)
    cols = _colors(len(keys))
    # One group gets a filled marginal; several get outlines. Stacking four
    # translucent fills turns the margin into an unreadable brown wash.
    solo = len(keys) == 1
    n_total = 0
    for key, col in zip(keys, cols):
        m = _mask(df, a.group, key)
        xs, ys = _finite(df[x].to_numpy()[m], df[y].to_numpy()[m])
        if xs.size == 0:
            continue
        n_total += xs.size
        ax.scatter(xs, ys, s=16, color=col, alpha=0.72, linewidths=0.4,
                   edgecolors="white", zorder=3, label=_label(key, a.group))
        if xs.size >= 3:
            style = (dict(histtype="stepfilled", alpha=0.5, edgecolor=col,
                          linewidth=0.8) if solo
                     else dict(histtype="step", linewidth=1.1, alpha=0.9))
            axx.hist(xs, bins=26, color=col, density=True, **style)
            axy.hist(ys, bins=26, color=col, density=True,
                     orientation="horizontal", **style)

    xs, ys = _finite(df[x].to_numpy(), df[y].to_numpy())
    stats = ""
    if xs.size >= 3:
        if a.smooth:
            gx, gy = smooth_line(xs, ys, frac=0.35)
            ax.plot(gx, gy, color=COLORS["rule"], lw=1.5, zorder=5,
                    label="LOWESS")
        else:
            b, c = np.polyfit(xs, ys, 1)
            gx = np.linspace(xs.min(), xs.max(), 200)
            ax.plot(gx, b * gx + c, color=COLORS["rule"], lw=1.4, ls="--",
                    zorder=5, label="OLS fit")
            r = float(np.corrcoef(xs, ys)[0, 1])
            stats = f"$y = {b:.3g}x {c:+.3g}$\n$r = {r:.3f}$   $n = {xs.size}$"
    if stats:
        annotate_box(ax, stats, loc="upper left",
                     fontsize=TYPE_SCALE["annotation"])
    _fmt_axis(ax, x, y)
    if a.group or a.smooth:
        _legend(ax, loc="lower right")
    return fig, f"n = {n_total or xs.size} observations"


def k_bar_group(df, a, roles):
    cat = dataio.pick(df, a.x, "categorical", roles)
    val = dataio.pick(df, a.y, "numeric", roles, exclude=(cat,))
    if not cat:
        raise DataError("bar_group needs a categorical column; pass --x")

    if val:
        piv = df.pivot_table(index=cat, columns=a.group, values=val,
                             aggfunc=a.agg) if a.group else \
            df.groupby(cat, dropna=True)[val].agg(a.agg).to_frame(val)
        ylab = f"{a.agg} of {val}"
    else:
        piv = (df.groupby([cat, a.group]).size().unstack() if a.group
               else df[cat].value_counts().to_frame("count"))
        ylab = "count"
    piv = piv.sort_values(piv.columns[0], ascending=False).head(a.top)

    fig, ax = plt.subplots(figsize=FIGSIZES["double"])
    labels = [str(i) for i in piv.index]
    cols = _colors(len(piv.columns))
    idx = np.arange(len(piv))
    if a.stacked:
        bottom = np.zeros(len(piv))
        for col, colour in zip(piv.columns, cols):
            v = piv[col].fillna(0).to_numpy(dtype=float)
            ax.bar(idx, v, bottom=bottom, width=0.74, color=colour,
                   edgecolor="white", linewidth=0.5, label=str(col), zorder=3)
            bottom += v
    else:
        w = 0.8 / len(piv.columns)
        for j, (col, colour) in enumerate(zip(piv.columns, cols)):
            v = piv[col].fillna(0).to_numpy(dtype=float)
            ax.bar(idx + (j - (len(piv.columns) - 1) / 2) * w, v, width=w,
                   color=colour, edgecolor="white", linewidth=0.5,
                   label=str(col), zorder=3)
    ax.set_xticks(idx)
    ax.set_xticklabels([textwrap.fill(t, 14) for t in labels],
                       fontsize=TYPE_SCALE["tick"])
    ax.yaxis.grid(True, color="#d9d9d9", lw=0.6)
    ax.set_axisbelow(True)
    _fmt_axis(ax, None, ylab)
    if len(piv.columns) > 1:
        _legend(ax, loc="upper right", ncol=min(len(piv.columns), 4))
    return fig, f"{len(piv)} categories" + (f", {len(piv.columns)} series"
                                            if len(piv.columns) > 1 else "")


def k_line_ci(df, a, roles):
    x = (dataio.pick(df, a.x, "datetime", roles)
         or dataio.pick(df, a.x, "numeric", roles))
    y = dataio.pick(df, a.y, "numeric", roles, exclude=(x,))
    if not x or not y:
        raise DataError("line_ci needs an ordered column and a numeric column; "
                        "pass --x and --y")

    fig, ax = plt.subplots(figsize=FIGSIZES["double"])
    keys = _groups(df, a.group)
    cols = _colors(len(keys))
    for key, col in zip(keys, cols):
        m = _mask(df, a.group, key)
        sub = df.loc[m, [x, y]].dropna().sort_values(x)
        if sub.empty:
            continue
        # Repeated x values become a mean with a 95% band; a single
        # observation per x just draws the line.
        grp = sub.groupby(x)[y]
        mu, cnt, sd = grp.mean(), grp.size(), grp.std()
        ax.plot(mu.index, mu.to_numpy(), color=col, lw=1.6, zorder=4,
                label=_label(key, a.group))
        if (cnt > 1).any():
            se = (sd / np.sqrt(cnt)).fillna(0).to_numpy()
            ax.fill_between(mu.index, mu.to_numpy() - 1.96 * se,
                            mu.to_numpy() + 1.96 * se, color=col, alpha=0.18,
                            linewidth=0, zorder=2)
    ax.grid(True, color="#d9d9d9", lw=0.6)
    ax.set_axisbelow(True)
    _fmt_axis(ax, x, y)
    _datetime_axis(ax, df[x])
    if a.group:
        _legend(ax, loc="best")
    return fig, f"{len(df)} rows" + (", shaded band is the 95% CI of the mean"
                                     if a.group else "")


def k_dist(df, a, roles):
    val = dataio.pick(df, a.y, "numeric", roles)
    cat = dataio.pick(df, a.group or a.x, "categorical", roles, exclude=(val,))
    if not val:
        raise DataError("dist needs a numeric column; pass --y")

    keys = _groups(df, cat) if cat else [None]
    cols = _colors(len(keys))
    fig, ax = plt.subplots(figsize=FIGSIZES["double"])
    rng = np.random.default_rng(0)
    for i, (key, col) in enumerate(zip(keys, cols)):
        m = _mask(df, cat, key)
        v = df[val].to_numpy()[m]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        parts = ax.violinplot([v], positions=[i], widths=0.72,
                              showextrema=False, showmedians=False)
        for body in parts["bodies"]:
            body.set_facecolor(col)
            body.set_alpha(0.28)
            body.set_edgecolor(col)
            body.set_linewidth(0.7)
        bp = ax.boxplot([v], positions=[i], widths=0.16, patch_artist=True,
                        showfliers=False, zorder=4,
                        medianprops=dict(color="white", lw=1.2),
                        whiskerprops=dict(color=col, lw=0.9),
                        capprops=dict(color=col, lw=0.9))
        bp["boxes"][0].set(facecolor=col, edgecolor=col, alpha=0.95, lw=0.6)
        jitter = rng.normal(0, 0.045, v.size)
        ax.scatter(i + 0.30 + jitter, v, s=6, color=col, alpha=0.5,
                   linewidths=0, zorder=3)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([textwrap.fill(_label(k, cat or val), 14) for k in keys],
                       fontsize=TYPE_SCALE["tick"])
    ax.yaxis.grid(True, color="#d9d9d9", lw=0.6)
    ax.set_axisbelow(True)
    _fmt_axis(ax, None, val)
    return fig, (f"{len(keys)} groups, box on violin with the raw points"
                 if cat else f"n = {int(df[val].notna().sum())}")


def k_corr(df, a, roles):
    num = [c for c in roles["numeric"] if df[c].notna().sum() > 2][:a.top]
    if len(num) < 2:
        raise DataError("corr needs at least two numeric columns")
    corr = df[num].corr(method=a.method).to_numpy()

    size = min(11.0, 3.0 + 0.42 * len(num))
    fig, ax = plt.subplots(figsize=(size, size * 0.86))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num)))
    ax.set_yticks(range(len(num)))
    ax.set_xticklabels(num, rotation=45, ha="right",
                       fontsize=TYPE_SCALE["dense"])
    ax.set_yticklabels(num, fontsize=TYPE_SCALE["dense"])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    if len(num) <= 14:
        for i in range(len(num)):
            for j in range(len(num)):
                v = corr[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=TYPE_SCALE["dense"],
                        color="white" if abs(v) > 0.55 else COLORS["ink"])
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, shrink=0.86)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(COLORS["rule"])
    cb.ax.tick_params(labelsize=TYPE_SCALE["tick"], width=0.7, length=2.5)
    cb.set_label(f"{a.method.title()} correlation",
                 fontsize=TYPE_SCALE["axis_label"])
    return fig, f"{len(num)} numeric columns, {len(df)} rows"


def k_hist(df, a, roles):
    val = dataio.pick(df, a.x or a.y, "numeric", roles)
    if not val:
        raise DataError("hist needs a numeric column; pass --x")
    fig, ax = plt.subplots(figsize=FIGSIZES["double"])
    keys = _groups(df, a.group)
    cols = _colors(len(keys))
    for key, col in zip(keys, cols):
        m = _mask(df, a.group, key)
        v = df[val].to_numpy()[m]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        ax.hist(v, bins=a.bins, density=True, color=col, alpha=0.42,
                edgecolor="white", linewidth=0.4, zorder=3,
                label=_label(key, a.group))
        if v.size > 4 and np.ptp(v) > 0:
            grid = np.linspace(v.min(), v.max(), 256)
            bw = 1.06 * v.std() * v.size ** (-0.2) or 1.0
            dens = np.exp(-0.5 * ((grid[:, None] - v[None, :]) / bw) ** 2)
            ax.plot(grid, dens.sum(axis=1) / (v.size * bw * np.sqrt(2 * np.pi)),
                    color=col, lw=1.4, zorder=4)
    ax.yaxis.grid(True, color="#d9d9d9", lw=0.6)
    ax.set_axisbelow(True)
    _fmt_axis(ax, val, "density")
    if a.group:
        _legend(ax, loc="upper right")
    return fig, f"n = {int(df[val].notna().sum())}, {a.bins} bins"


def k_ecdf(df, a, roles):
    val = dataio.pick(df, a.x or a.y, "numeric", roles)
    if not val:
        raise DataError("ecdf needs a numeric column; pass --x")
    fig, ax = plt.subplots(figsize=FIGSIZES["double"])
    for key, col in zip(_groups(df, a.group), _colors(len(_groups(df, a.group)))):
        m = _mask(df, a.group, key)
        v = np.sort(df[val].to_numpy()[m])
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        ax.step(v, np.arange(1, v.size + 1) / v.size, where="post", color=col,
                lw=1.5, zorder=3, label=f"{_label(key, a.group)} (n={v.size})")
    ax.set_ylim(0, 1.02)
    ax.grid(True, color="#d9d9d9", lw=0.6)
    ax.set_axisbelow(True)
    _fmt_axis(ax, val, "cumulative proportion")
    _legend(ax, loc="lower right")
    return fig, "empirical cumulative distribution"


def k_area(df, a, roles):
    x = (dataio.pick(df, a.x, "datetime", roles)
         or dataio.pick(df, a.x, "numeric", roles))
    val = dataio.pick(df, a.y, "numeric", roles, exclude=(x,))
    if not x or not a.group:
        raise DataError("area needs --x (ordered) and --group (the series)")
    piv = (df.pivot_table(index=x, columns=a.group, values=val, aggfunc=a.agg)
           if val else df.groupby([x, a.group]).size().unstack())
    piv = piv.sort_index().fillna(0)

    fig, ax = plt.subplots(figsize=FIGSIZES["double"])
    cols = _colors(len(piv.columns))
    ax.stackplot(piv.index, *[piv[c].to_numpy() for c in piv.columns],
                 colors=cols, labels=[str(c) for c in piv.columns],
                 edgecolor="white", linewidth=0.4)
    ax.set_xlim(piv.index.min(), piv.index.max())
    _fmt_axis(ax, x, f"{a.agg} of {val}" if val else "count")
    _datetime_axis(ax, piv.index.to_series())
    _legend(ax, loc="upper left", ncol=min(len(piv.columns), 4))
    return fig, f"{len(piv.columns)} series over {len(piv)} steps"


def k_donut(df, a, roles):
    cat = dataio.pick(df, a.x or a.group, "categorical", roles)
    val = dataio.pick(df, a.y, "numeric", roles, exclude=(cat,))
    if not cat:
        raise DataError("donut needs a categorical column; pass --x")
    s = (df.groupby(cat)[val].agg(a.agg) if val else df[cat].value_counts())
    s = s.sort_values(ascending=False)
    if len(s) > a.top:
        other = s.iloc[a.top:].sum()
        s = s.iloc[:a.top]
        s.loc["Other"] = other

    fig, (ax, axt) = plt.subplots(1, 2, figsize=(8.2, 4.4),
                                  gridspec_kw={"width_ratios": (1, 0.85)})
    cols = _colors(len(s))
    total = float(s.sum())
    wedges, _ = ax.pie(s.to_numpy(), colors=cols, startangle=90,
                       counterclock=False,
                       wedgeprops=dict(width=0.42, edgecolor="white", lw=1.0))
    ax.text(0, 0, f"{total:,.0f}\ntotal", ha="center", va="center",
            fontsize=TYPE_SCALE["panel_title"], fontweight="bold",
            color=COLORS["ink"], linespacing=1.5)
    ax.set(aspect="equal")

    axt.axis("off")
    # Fixed row pitch, block centred. Spreading four rows over the full height
    # leaves gaps wide enough that the swatch stops reading as a legend.
    pitch = min(0.085, 0.9 / max(len(s), 1))
    top = 0.5 + pitch * (len(s) - 1) / 2.0
    for i, (name, v) in enumerate(s.items()):
        y = top - i * pitch
        axt.plot([0.02], [y], marker="s", ms=5.5, color=cols[i],
                 transform=axt.transAxes, clip_on=False)
        axt.text(0.08, y, str(name)[:30], transform=axt.transAxes,
                 va="center", fontsize=TYPE_SCALE["annotation"],
                 color=COLORS["ink"])
        axt.text(0.99, y, f"{v:,.0f}   {100 * v / total:4.1f} %",
                 transform=axt.transAxes, va="center", ha="right",
                 fontsize=TYPE_SCALE["annotation"], color=COLORS["sub"])
    return fig, f"{len(s)} categories, {total:,.0f} total"


def k_forest(df, a, roles):
    est = dataio.pick(df, a.y, "numeric", roles)
    lo = dataio.pick(df, a.lo, "numeric", roles, exclude=(est,))
    hi = dataio.pick(df, a.hi, "numeric", roles, exclude=(est, lo))
    lab = dataio.pick(df, a.labels or a.x, "categorical", roles)
    if not est or not lo or not hi:
        raise DataError("forest needs --y (estimate), --lo and --hi "
                        "(interval bounds)")
    sub = df[[c for c in (lab, est, lo, hi) if c]].dropna().head(a.top)
    names = ([str(v) for v in sub[lab]] if lab
             else [f"row {i + 1}" for i in range(len(sub))])

    fig, ax = plt.subplots(figsize=(7.2, max(2.6, 0.32 * len(sub) + 1.4)))
    y = np.arange(len(sub))[::-1]
    e = sub[est].to_numpy(float)
    ax.hlines(y, sub[lo].to_numpy(float), sub[hi].to_numpy(float),
              color=COLORS["rule"], lw=1.2, zorder=3)
    ax.scatter(e, y, s=34, color=CATEGORICAL[0], zorder=4, edgecolors="white",
               linewidths=0.6)
    ref = a.ref if a.ref is not None else (1.0 if (e > 0).all()
                                           and 0.2 < np.median(e) < 5 else 0.0)
    ax.axvline(ref, color=COLORS["baseline"], lw=0.9, ls="--", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=TYPE_SCALE["tick"])
    ax.set_ylim(-0.8, len(sub) - 0.2)
    for i, (v, l, h) in enumerate(zip(e, sub[lo], sub[hi])):
        ax.text(1.01, y[i], f"{v:.2f} [{l:.2f}, {h:.2f}]",
                transform=ax.get_yaxis_transform(), va="center",
                fontsize=TYPE_SCALE["annotation"], color=COLORS["sub"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _fmt_axis(ax, est, None)
    return fig, f"{len(sub)} estimates, reference at {ref:g}"


def k_lollipop(df, a, roles):
    cat = dataio.pick(df, a.x, "categorical", roles)
    val = dataio.pick(df, a.y, "numeric", roles, exclude=(cat,))
    if not cat:
        raise DataError("lollipop needs a categorical column; pass --x")
    s = (df.groupby(cat)[val].agg(a.agg) if val else df[cat].value_counts())
    s = s.sort_values(ascending=True).tail(a.top)

    fig, ax = plt.subplots(figsize=(7.2, max(2.6, 0.28 * len(s) + 1.3)))
    y = np.arange(len(s))
    ax.hlines(y, 0, s.to_numpy(float), color="#c8c8c8", lw=1.0, zorder=2)
    ax.scatter(s.to_numpy(float), y, s=42, color=CATEGORICAL[1], zorder=3,
               edgecolors="white", linewidths=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([str(i)[:34] for i in s.index],
                       fontsize=TYPE_SCALE["tick"])
    for yi, v in zip(y, s.to_numpy(float)):
        ax.text(v, yi + 0.34, f"{v:,.3g}", ha="center", va="bottom",
                fontsize=TYPE_SCALE["annotation"], color=COLORS["sub"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color="#d9d9d9", lw=0.6)
    ax.set_axisbelow(True)
    _fmt_axis(ax, f"{a.agg} of {val}" if val else "count", None)
    return fig, f"top {len(s)} of {df[cat].nunique()} categories"


def k_errorbar(df, a, roles):
    val = dataio.pick(df, a.y, "numeric", roles)
    cat = dataio.pick(df, a.x or a.group, "categorical", roles, exclude=(val,))
    if not val or not cat:
        raise DataError("errorbar needs --y (numeric) and --x (categorical)")
    grp = df.groupby(cat)[val]
    mu, sd, n = grp.mean(), grp.std(), grp.size()
    se = (sd / np.sqrt(n)).fillna(0)
    order = mu.sort_values(ascending=False).index[:a.top]

    fig, ax = plt.subplots(figsize=FIGSIZES["double"])
    rng = np.random.default_rng(0)
    for i, key in enumerate(order):
        v = df.loc[df[cat] == key, val].to_numpy(float)
        v = v[np.isfinite(v)]
        ax.scatter(i + rng.normal(0, 0.055, v.size), v, s=7, color="#b8c4d4",
                   alpha=0.75, linewidths=0, zorder=2)
        ax.errorbar(i, mu[key], yerr=1.96 * se[key], fmt="o", ms=5.5,
                    color=CATEGORICAL[1], ecolor=COLORS["rule"], elinewidth=1.1,
                    capsize=3, zorder=4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([textwrap.fill(str(k), 14) for k in order],
                       fontsize=TYPE_SCALE["tick"])
    ax.yaxis.grid(True, color="#d9d9d9", lw=0.6)
    ax.set_axisbelow(True)
    _fmt_axis(ax, None, val)
    return fig, "group mean with a 95% CI, raw observations behind"


def k_survival(df, a, roles):
    t = dataio.pick(df, a.x, "numeric", roles)
    if not t:
        raise DataError("survival needs a numeric time column; pass --x")
    event = a.event if (a.event and a.event in df.columns) else None

    fig, ax = plt.subplots(figsize=FIGSIZES["double"])
    keys = _groups(df, a.group)
    for key, col in zip(keys, _colors(len(keys))):
        m = _mask(df, a.group, key)
        sub = df.loc[m, [c for c in (t, event) if c]].dropna().sort_values(t)
        if sub.empty:
            continue
        times = sub[t].to_numpy(float)
        obs = (sub[event].to_numpy().astype(bool) if event
               else np.ones(len(sub), bool))
        # Kaplan-Meier: survival falls only at an observed event, and the
        # denominator is whoever is still at risk.
        surv, s, at_risk = [], 1.0, len(times)
        for i, ti in enumerate(times):
            if obs[i]:
                s *= 1.0 - 1.0 / max(at_risk, 1)
            surv.append(s)
            at_risk -= 1
        ax.step(np.r_[0, times], np.r_[1.0, surv], where="post", color=col,
                lw=1.6, zorder=3, label=f"{_label(key, a.group)} (n={len(times)})")
        if event is not None:
            cens = times[~obs]
            ax.scatter(cens, np.array(surv)[~obs], marker="|", s=26, color=col,
                       zorder=4)
    ax.set_ylim(0, 1.02)
    ax.grid(True, color="#d9d9d9", lw=0.6)
    ax.set_axisbelow(True)
    _fmt_axis(ax, t, "survival probability")
    _legend(ax, loc="upper right")
    return fig, ("Kaplan-Meier, ticks mark censoring" if event
                 else "all observations treated as events (pass --event)")


def k_pairs(df, a, roles):
    num = [c for c in roles["numeric"] if df[c].notna().sum() > 2][:min(a.top, 6)]
    if len(num) < 2:
        raise DataError("pairs needs at least two numeric columns")
    k = len(num)
    fig, axes = plt.subplots(k, k, figsize=(1.55 * k + 1.2, 1.45 * k + 1.2))
    keys = _groups(df, a.group, limit=6)
    cols = _colors(len(keys))
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            for key, col in zip(keys, cols):
                m = _mask(df, a.group, key)
                if i == j:
                    v = df[num[i]].to_numpy()[m]
                    v = v[np.isfinite(v)]
                    if v.size:
                        ax.hist(v, bins=20, color=col, alpha=0.5,
                                density=True, edgecolor="white", linewidth=0.3)
                else:
                    xs, ys = _finite(df[num[j]].to_numpy()[m],
                                     df[num[i]].to_numpy()[m])
                    ax.scatter(xs, ys, s=5, color=col, alpha=0.6,
                               linewidths=0)
            style_axes(ax)
            ax.tick_params(labelsize=TYPE_SCALE["dense"], length=2)
            if i < k - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(num[j][:14], fontsize=TYPE_SCALE["dense"])
            if j > 0:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(num[i][:14], fontsize=TYPE_SCALE["dense"])
    if a.group:
        fig.legend(handles=[Line2D([], [], marker="o", ls="", color=c,
                                   label=_label(kk, a.group), ms=4)
                            for kk, c in zip(keys, cols)],
                   loc="upper right", frameon=False,
                   fontsize=TYPE_SCALE["legend"],
                   bbox_to_anchor=(0.995, 0.995))
    return fig, f"{k} numeric columns, {len(df)} rows"


BUILDERS = {
    "scatter_fit": k_scatter_fit, "bar_group": k_bar_group,
    "line_ci": k_line_ci, "dist": k_dist, "corr": k_corr, "hist": k_hist,
    "ecdf": k_ecdf, "area": k_area, "donut": k_donut, "forest": k_forest,
    "lollipop": k_lollipop, "errorbar": k_errorbar, "survival": k_survival,
    "pairs": k_pairs,
}


# ---- suggestions -----------------------------------------------------------
def suggest(roles):
    """Chart kinds this table can actually support, best first."""
    n, c, d = (len(roles["numeric"]), len(roles["categorical"]),
               len(roles["datetime"]))
    out = []
    if n >= 2:
        out += ["scatter_fit", "corr", "pairs"]
    if (d or n) and n >= 1:
        out += ["line_ci"]
    if c and n:
        out += ["dist", "bar_group", "errorbar", "lollipop"]
    elif c:
        out += ["bar_group", "donut", "lollipop"]
    if n:
        out += ["hist", "ecdf"]
    if c and (d or n):
        out += ["area"]
    if c:
        out += ["donut"]
    seen, ordered = set(), []
    for k in out:
        if k not in seen:
            seen.add(k)
            ordered.append(k)
    return ordered


# ---- Where does a generator's data come from? ------------------------------
def contract(script):
    """Report where a ``fig_*.py`` builds its data, and whether you can swap it.

    The catalogue documentation used to promise that every generator isolates
    its synthetic data in ``gen_data()``. Eighteen of them do. The rest either
    build it inline in ``main()`` or have no random data at all because the
    figure *is* a model. Rather than leave someone to discover that per script,
    this locates the data for the one they are looking at.
    """
    import ast

    path = script if os.path.isabs(script) else os.path.join(HERE, script)
    if not os.path.exists(path) and not script.startswith("fig_"):
        path = os.path.join(HERE, f"fig_{script}.py")
    if not os.path.exists(path):
        raise DataError(f"no such script: {script}")

    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src)
    lines = src.splitlines()

    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    rng_sites = []
    for name, node in funcs.items():
        for n in ast.walk(node):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in ("default_rng", "RandomState", "seed")):
                rng_sites.append((name, n.lineno))
    # Module constants, minus the palette. A hex colour is styling, and the
    # styling is the part nobody should be editing.
    const_lines = []
    for i, ln in enumerate(lines):
        if not ln or not ln[0].isupper() or "=" not in ln.split("#")[0]:
            continue
        if ln.startswith(("def ", "class ", "import ", "from ")):
            continue
        rhs = ln.split("=", 1)[1]
        if re.search(r'["\']#[0-9a-fA-F]{3,8}["\']', rhs) or "PALETTES[" in rhs:
            continue
        const_lines.append((i + 1, ln))

    print(f"script : {os.path.relpath(path, ROOT)}")
    doc = ast.get_docstring(tree) or ""
    if doc:
        print(f"purpose: {doc.strip().splitlines()[0]}")
    print()

    if "gen_data" in funcs:
        node = funcs["gen_data"]
        print(f"data    : gen_data() at line {node.lineno} -- the documented "
              f"pattern.\n"
              f"          Replace the body, keep the return signature, and "
              f"nothing else changes.")
    elif rng_sites:
        where = sorted({n for n, _ in rng_sites})
        first = min(ln for _, ln in rng_sites)
        if where == ["main"]:
            print(f"data    : built inline inside main(), from line {first}.\n"
                  f"          Not separable without reading the function: the "
                  f"data and the plotting are interleaved.\n"
                  f"          Substitute your arrays at the point each one is "
                  f"first used.")
        else:
            print(f"data    : built in {', '.join(where)}() "
                  f"(first RNG at line {first}).\n"
                  f"          Replace those functions, keeping what they "
                  f"return.")
    else:
        print("data    : no random generator -- this figure computes its own\n"
              "          values from the module constants below, or draws a\n"
              "          diagram. There is probably nothing to substitute a\n"
              "          data file into.")

    if const_lines:
        print(f"\nmodule constants ({len(const_lines)}, colours excluded) -- "
              f"usually the knobs worth changing first:")
        for ln, text in const_lines[:14]:
            print(f"  {ln:5d}| {text.strip()[:88]}")
        if len(const_lines) > 14:
            print(f"  ... and {len(const_lines) - 14} more")

    m = re.search(r"""savefig\(\s*\w+\s*,\s*f?["']([^"']+)["']""", src)
    if m:
        print(f"\noutput  : output/{m.group(1)}")
    print(f"\nIf your data is a table rather than a model, the front door is\n"
          f"  python plot_data.py --data YOURFILE --describe")
    return 0


# ---- CLI -------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Plot a data file in the house publication style.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="chart kinds:\n" + "\n".join(
            f"  {k:<12} {v}" for k, v in KINDS.items()))
    p.add_argument("--data", metavar="FILE",
                   help="csv / tsv / xlsx / xls / parquet / json")
    p.add_argument("--kind", choices=sorted(KINDS),
                   help="chart kind; omit with --describe")
    p.add_argument("--x", help="x column (inferred when omitted)")
    p.add_argument("--y", help="y / value column (inferred when omitted)")
    p.add_argument("--group", help="column that splits the data into series")
    p.add_argument("--lo", help="lower interval bound (forest)")
    p.add_argument("--hi", help="upper interval bound (forest)")
    p.add_argument("--labels", help="row label column (forest)")
    p.add_argument("--event", help="1/0 event column (survival)")

    p.add_argument("--title", help="figure title; state the conclusion")
    p.add_argument("--subtitle", help="override the generated subtitle")
    p.add_argument("--footer", help="provenance line; defaults to the filename")
    p.add_argument("--corner", help="small top-right stamp")

    p.add_argument("--agg", default="mean",
                   choices=["mean", "median", "sum", "count", "min", "max"],
                   help="aggregation for grouped kinds (default mean)")
    p.add_argument("--bins", type=int, default=30, help="histogram bins")
    p.add_argument("--top", type=int, default=20,
                   help="cap on categories / columns shown (default 20)")
    p.add_argument("--stacked", action="store_true", help="stack bar_group")
    p.add_argument("--smooth", action="store_true",
                   help="LOWESS instead of a straight fit (scatter_fit)")
    p.add_argument("--ref", type=float, help="reference line (forest)")
    p.add_argument("--method", default="pearson",
                   choices=["pearson", "spearman", "kendall"],
                   help="correlation method (default pearson)")

    p.add_argument("--out", help="output filename (default from data + kind)")
    p.add_argument("--outdir", help="output directory (default output/)")
    p.add_argument("--formats", default="png,pdf",
                   help="comma-separated export formats (default png,pdf)")
    p.add_argument("--theme", default="default",
                   choices=["default", "nature", "science", "ieee",
                            "presentation"])
    p.add_argument("--base-font", type=float, default=9.0)
    p.add_argument("--cn-font", dest="cn_font", action="store_true",
                   default=None,
                   help="force a Chinese font (auto-detected by default)")
    p.add_argument("--no-cn-font", dest="cn_font", action="store_false",
                   help="never switch fonts, even for CJK labels")
    p.add_argument("--sheet", help="Excel worksheet name or index")
    p.add_argument("--encoding", help="override the encoding sniffer")
    p.add_argument("--delimiter", help="override the delimiter sniffer")
    p.add_argument("--nrows", type=int, help="read only the first N rows")

    p.add_argument("--describe", action="store_true",
                   help="profile the file and suggest chart kinds, plot nothing")
    p.add_argument("--list-kinds", action="store_true",
                   help="list the chart kinds and exit")
    p.add_argument("--contract", metavar="SCRIPT",
                   help="report where a fig_*.py gets its data, and whether "
                        "you can substitute your own")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)

    if a.list_kinds:
        print("chart kinds:\n")
        for k, v in KINDS.items():
            print(f"  {k:<12} {v}")
        return 0
    if a.contract:
        try:
            return contract(a.contract)
        except DataError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    if not a.data:
        print("error: --data is required (or use --list-kinds)", file=sys.stderr)
        return 2

    try:
        df = dataio.load_table(a.data, sheet=a.sheet, encoding=a.encoding,
                               delimiter=a.delimiter, nrows=a.nrows)
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    roles = dataio.column_roles(df)

    if a.describe or not a.kind:
        print(f"file: {a.data}\n")
        print(dataio.profile(df))
        ideas = suggest(roles)
        print("chart kinds this table supports:\n")
        for k in ideas:
            print(f"  {k:<12} {KINDS[k]}")
        if ideas:
            print(f"\ntry:\n  python plot_data.py --data {a.data} "
                  f"--kind {ideas[0]}")
        if not a.kind:
            return 0
        return 0

    set_pub_style(base_font=a.base_font, theme=a.theme)

    want_cjk = (a.cn_font if a.cn_font is not None
                else _needs_cjk(df, a.title, a.subtitle, a.footer, a.corner))
    if want_cjk:
        if set_cn_font():
            print("note: CJK text detected, enabled a Chinese font "
                  "(Latin text stays in Times)")
        else:
            print("warning: CJK text detected but no Chinese font is "
                  "installed; those labels will render as empty boxes.\n"
                  "         Install one of: Microsoft YaHei, SimSun, "
                  "Noto Serif CJK SC, Source Han Serif SC", file=sys.stderr)

    try:
        fig, auto_sub = BUILDERS[a.kind](df, a, roles)
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    title = a.title or f"{a.kind.replace('_', ' ').capitalize()} of " \
                       f"{os.path.basename(a.data)}"
    footer = a.footer or (f"Source: {os.path.basename(a.data)}, "
                          f"{len(df)} rows. Generated with scientific-figures.")
    multi = len(fig.axes) > 1 and a.kind in ("pairs",)
    head = header_height(fig, subtitle=True, panel_labels=multi)
    fig.tight_layout(rect=[0, 0.045, 1, 1 - head / fig.get_size_inches()[1]])
    add_figure_header(fig, title, subtitle=a.subtitle or auto_sub,
                      footer=footer, corner=a.corner)

    stem = a.out or f"{os.path.splitext(os.path.basename(a.data))[0]}_{a.kind}"
    fmts = tuple(f.strip().lstrip(".") for f in a.formats.split(",") if f.strip())
    path = savefig(fig, stem if "." in stem else f"{stem}.{fmts[0]}",
                   formats=fmts, outdir=a.outdir)
    print(f"\n{a.kind}: {os.path.basename(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

---
name: scientific-figures
description: >-
  1219 ready-to-run matplotlib generators for publication-quality scientific
  figures, on an enforced Times New Roman design system with a bundled
  metric-compatible fallback font and a typography linter that reads the fonts
  actually embedded in the rendered PDF. Covers core statistical charts
  (threshold regression, raincloud, ridgeline, clustermap, radar, Sankey,
  treemap, UpSet, Taylor, volcano, Manhattan, forest, 3x2 six-panel grids),
  geospatial mapping, ML and SHAP interpretation, bio-omics, climate,
  graphical abstracts, physics and engineering, economics, clinical trials,
  chemistry and materials, Bayesian/MCMC diagnostics, neuroscience and
  signals, optimisation and metaheuristics, UAV and robot path planning,
  remote sensing, power systems, transport, hydrology and ocean, ecology,
  astronomy, manufacturing quality, computer vision, network architecture
  diagrams, attention and feature visualisation, 3D vision and point clouds,
  generative models, training and benchmark panels, LLM analysis, pie/donut
  composition, vision-language multimodal figures, and world models.
license: MIT
---

# Scientific Figures (publication-quality)

**1219 ready-to-run matplotlib generators**, numbered `0-1218`, that reproduce
the expensive-journal figure aesthetic: Times New Roman throughout, thin dark
spines, muted-saturated palettes, tight multi-panel grids, threshold and metric
annotations, marginal densities, 300-dpi export.

Every script is self-contained. It runs with no arguments, invents its own
synthetic data, and writes a PNG into `output/` named after its number.

```bash
cd scripts
python fig_threshold_regression.py     # -> output/01_threshold_regression.png
```

## Finding the right figure

Do **not** read this file looking for a chart name; the catalogue is split so
you only load what you need.

1. Read [`catalog/INDEX.md`](catalog/INDEX.md) — 135 groups, each a discipline
   or chart family, with its number range.
2. Read the one group file that matches. Each lists every figure in its range
   with what it draws and when to use it.
3. Read that script before adapting it.

When the chart name is already known, skip the index and search all groups:

```bash
grep -ril "sankey" catalog/
grep -ril "生存曲线\|kaplan" catalog/
```

To see what a figure looks like before running it, open
`reference/thumbs/<number>_<name>.jpg` — all 1219 are there, downscaled.
`reference/curated/` holds 33 of them at full resolution.

If nothing in the catalogue fits, start from the general-purpose group
(`0-54`): distributions, correlation, regression, ranking, composition and time
series, in a form any discipline can borrow.

## Fonts: read this before your first run

The house face is **Times New Roman**. It is proprietary and is *not* bundled;
`fonts/` carries **Tinos** (SIL OFL) instead, which has byte-identical advance
widths, so a figure lays out the same whether or not the machine has the real
thing. Confirm the stack resolves before trusting any output:

```bash
cd scripts
python check_typography.py --metrics
#   vs Tinos              identical    (Tinos-Regular.ttf)
```

If matplotlib ever falls through to DejaVu, `set_pub_style()` raises rather
than degrading quietly — a wrong typeface is invisible to every PNG-based
check. Details in [`fonts/README.md`](fonts/README.md).

## Requirements

`matplotlib>=3.6` and `numpy` are needed by every figure; `scipy`, `pandas`,
`seaborn`, `scikit-learn`, `openpyxl` and `Pillow` by subsets of the library
and the tooling. The 3.6 floor matters: glyph-by-glyph font fallback is what
lets a Chinese label borrow a Song face while the numerals stay in Times.

```bash
pip install -r requirements.txt
```

## The shared style: `scripts/pubstyle.py`

Import this in any new figure to match the aesthetic:

```python
from pubstyle import (set_pub_style, PALETTES, savefig, style_axes,
                      panel_title, add_figure_header, header_height,
                      panel_labels_inches)

set_pub_style()                       # apply rcParams once; never touch them yourself
fig, axes = ...                       # build your figure
panel_title(axes[0], "Drift magnitude over time")

head = header_height(fig, subtitle=True)
fig.tight_layout(rect=[0, 0.03, 1, 1 - head / fig.get_size_inches()[1]])
add_figure_header(fig,
                  "Loop closure removes unbounded drift",   # the conclusion
                  subtitle="n = 240 trajectories, 8 sequences")
panel_labels_inches(fig, axes)        # (a) (b) (c) at equal inch offsets

savefig(fig, "my_fig.png")            # 300 dpi into output/
```

Sizes come from `pubstyle.TYPE_SCALE`, so do not pass `fontsize=` when one of
these helpers will do it for you. Key palettes in `PALETTES`: `teal_rust`,
`blue_red`, `greens`, `earth`, `div_rdbu`, `pastel4`, `zones`, plus the
colour-blind-safe `cb_*` ramps. Reuse these instead of matplotlib defaults —
they are what make the figures look expensive.

`scripts/figkit.py` is the companion construction layer: zoom insets, text
cards, inch-based axes, anti-overlap label placement.

## Adapting to real data

**If the data is a table, do not start from these scripts.** Use the front door
instead, which applies the whole design system to a file:

```bash
cd scripts
python plot_data.py --data yours.csv --describe    # what can I draw?
python plot_data.py --data yours.csv --kind scatter_fit --x a --y b --group g
```

Fourteen chart kinds, encoding and delimiter sniffing, `"1,234"` and `"12.5%"`
coerced back to numbers, automatic CJK font selection for Chinese column names,
and `.xlsx` as well as `.csv`.

**To adapt one of the 1219 generators**, first find out where its data lives:

```bash
python plot_data.py --contract fig_threshold_regression.py
```

The scripts are not uniform about this. **19** of the 1219 isolate their data in
a `gen_data()` function, **299** keep it in some other named function that has
to be read first, **656** build it inline inside `main()`, and **245** have no
substitutable data at all because the figure *is* a model — a Kaimal spectrum,
a Kalman filter, a synthetic microscopy scene. The `--contract` report says
which case applies and names the constants worth changing.

Whichever gets edited, keep the plotting and styling code. The styling is the
valuable part; do not rewrite it with matplotlib defaults.

## Aesthetic rules (what makes these look expensive)

1. Hide top and right spines; keep spines thin (`0.8`) and dark grey (`#333`), not black.
2. Times New Roman throughout, small (7-9 pt body, 6.0 pt hard floor), bold only
   for the figure title, panel letters and panel titles.
3. Muted-saturated palette pairs (teal/rust, blue/red) — never primary RGB.
4. Add marginal densities or histograms to scatter plots whenever space allows.
5. Annotate statistics in a rounded white box with a thin border; put threshold
   text in the accent colour.
6. Dense multi-panel grids with `tight_layout(h_pad, w_pad)`; share the legend.
7. Export at 300 dpi with `bbox_inches="tight"` (handled by `savefig`).

[`DESIGN_SYSTEM.md`](DESIGN_SYSTEM.md) has the enforced numbers behind each rule.

## Verification

Open the PNG in `output/` and compare it against the request. If a panel is
cramped, raise `figsize` or `h_pad`/`w_pad`; if colours clash, switch to a
single `PALETTES` entry.

Reading the PNG cannot tell you which typeface was used, because a PNG stores
no font metadata. Run the typography audit as well — it renders to PDF and
reads the fonts the renderer embedded:

```bash
cd scripts
python check_typography.py              # errors must be zero
python check_figures.py --jobs 8        # run everything, audit every PNG
python run_all.py --parallel 8          # rebuild all 1219 (about 15 minutes)
python run_all.py --range 173-192       # just the remote-sensing block
python run_all.py --parallel 8 --vector # PDF beside every PNG, for submission
```

## Related documents

- [`catalog/INDEX.md`](catalog/INDEX.md) — the 135 groups and their number ranges.
- [`DESIGN_SYSTEM.md`](DESIGN_SYSTEM.md) — the cross-figure visual standard
  (canvas sizes, palettes, type scale, spacing, banned colormaps). Read before
  writing a new figure or editing an existing one.
- [`WORKFLOW.md`](WORKFLOW.md) — the operating manual: picking a generator,
  swapping in real data, scaffolding a new figure with `new_figure.py`, running
  the quality gates. `SKILL.md` answers *which figure*; `WORKFLOW.md` answers
  *how to use it*.
- [`scripts/reference.md`](scripts/reference.md) — the full `pubstyle` API.
- [`fonts/README.md`](fonts/README.md) — why a font ships here, which one, and
  under what licence.
- [`NUMBER_REGISTRY.md`](NUMBER_REGISTRY.md) — which number ranges are taken.
  New figures start at 1219.

"""Scaffold a new figure generator that already follows the house style.

    cd scripts
    python new_figure.py 55 choropleth_grid "County-level choropleth panels"

Writes ``scripts/fig_choropleth_grid.py`` wired to ``pubstyle`` and saving
``output/55_choropleth_grid.png``.  Refuses to overwrite an existing file and
warns when the figure number is already taken by another script.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

NAME_RE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
SAVEFIG_RE = re.compile(r"""savefig\(\s*\w+\s*,\s*["'](\d+)_""")

TEMPLATE = '''"""{title}: {description}

{description}

Synthetic data lives in ``gen_data`` only -- swap that one function for a
``pandas.read_csv`` loader to plot real data and leave the styling untouched.
"""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import (set_pub_style, PALETTES, savefig, style_axes,
                      add_figure_header, header_height)


def gen_data(seed={seed}):
    """Synthetic stand-in data. REPLACE THIS FUNCTION with your real loader.

    Example::

        import pandas as pd
        df = pd.read_csv("data/{name}.csv")
        return df["x"].to_numpy(), df["y"].to_numpy(), df["group"].to_numpy()
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, 160)
    y = 2.2 * np.sin(0.9 * x) + 0.28 * x + rng.normal(0, 0.42, x.size)
    group = rng.choice(["A", "B"], x.size)
    return x, y, group


def main():
    set_pub_style()
    x, y, group = gen_data()
    cols = PALETTES["teal_rust"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for col, key in zip(cols, ["A", "B"]):
        m = group == key
        ax.scatter(x[m], y[m], s=18, color=col, alpha=0.75,
                   edgecolors="white", linewidths=0.4, label=f"Group {{key}}",
                   zorder=3)

    coef = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, np.polyval(coef, xs), color="#333333", lw=1.4, ls="--",
            zorder=4, label="OLS fit")

    ax.set_xlabel("Predictor (unit)")
    ax.set_ylabel("Response (unit)")
    ax.legend(loc="upper left", ncol=2)
    style_axes(ax)

    # Reserve the header row first, then stamp it. Sizes and colours come from
    # pubstyle.TYPE_SCALE -- do not pass fontsize= here.
    head = header_height(fig, subtitle=True, panel_labels=False)
    fig.tight_layout(rect=[0, 0.03, 1, 1 - head / fig.get_size_inches()[1]])
    add_figure_header(
        fig,
        # TODO: state the conclusion, not the chart type.
        # "Response rises with predictor above the threshold", not "Scatter plot".
        "{title}",
        subtitle="n = 160 synthetic observations",
        footer="Synthetic data, seed={seed}. Replace gen_data() to plot real data.")

    savefig(fig, "{png}")


if __name__ == "__main__":
    main()
'''


def titleize(name):
    return name.replace("_", " ").capitalize()


def used_numbers(scripts_dir):
    """Map figure number -> script filename, read from each savefig() call."""
    taken = {}
    for fname in sorted(os.listdir(scripts_dir)):
        if not (fname.startswith("fig_") and fname.endswith(".py")):
            continue
        try:
            with open(os.path.join(scripts_dir, fname), encoding="utf-8") as fh:
                src = fh.read()
        except OSError:
            continue
        for num in SAVEFIG_RE.findall(src):
            taken.setdefault(int(num), fname)
    return taken


def scaffold(index, name, description, scripts_dir=HERE, force=False):
    if not NAME_RE.match(name):
        sys.exit(f"invalid name {name!r}: use lower snake_case, "
                 "e.g. choropleth_grid")

    num = f"{int(index):02d}"
    target = os.path.join(scripts_dir, f"fig_{name}.py")
    png = f"{num}_{name}.png"

    if os.path.exists(target) and not force:
        sys.exit(f"refusing to overwrite existing file:\n  {target}\n"
                 "pick another name, or delete it first.")

    taken = used_numbers(scripts_dir)
    owner = taken.get(int(index))
    if owner and owner != f"fig_{name}.py":
        print(f"warning: figure number {num} is already used by {owner}; "
              "two scripts sharing a number will confuse the catalog.")

    body = TEMPLATE.format(title=titleize(name), description=description,
                           name=name, png=png, seed=1000 + int(index))
    with open(target, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(body)

    rel = os.path.relpath(target, ROOT).replace("\\", "/")
    print(f"created: {rel}")
    print("\nnext steps:")
    print(f"  1. edit gen_data() in {rel} to load your real data")
    print("  2. replace the demo scatter in main() with the chart you want "
          "(copy a nearby fig_*.py for a closer starting point)")
    print(f"  3. cd scripts && python fig_{name}.py     "
          f"# -> output/{png}")
    print(f"  4. python build_reference.py --range {int(index)}-{int(index)}   "
          f"# refresh its thumbnail in reference/thumbs/")
    print(f"  5. add a row for #{num} to its group file in catalog/ "
          "(and to catalog/INDEX.md if it starts a new group)")
    return target


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Scaffold a new fig_*.py generator from the house template.")
    p.add_argument("index", type=int, help="figure number, e.g. 55")
    p.add_argument("name", help="snake_case figure name, e.g. choropleth_grid")
    p.add_argument("description", help="one-line description for the docstring")
    p.add_argument("--force", action="store_true",
                   help="allow overwriting an existing script")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    scaffold(a.index, a.name, a.description, force=a.force)


if __name__ == "__main__":
    main()

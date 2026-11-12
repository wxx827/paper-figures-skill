"""Exercise every chart kind in plot_data.py against the sample data.

Run from scripts/. Writes into examples/_smoke/ so nothing lands in output/.
Exits non-zero if any kind fails, so CI can use it as a gate.
"""
from __future__ import annotations

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
EX = os.path.join(ROOT, "examples")
OUT = os.path.join(EX, "_smoke")

SALES = os.path.join(EX, "sales.csv")
CLIN = os.path.join(EX, "clinical_gbk.csv")
TRIAL = os.path.join(EX, "trial_effects.csv")

# (label, argv). Chosen so every kind is hit at least once, and so the
# Chinese/GBK file drives the kinds a clinical client would actually want.
CASES = [
    ("scatter_fit", ["--data", SALES, "--kind", "scatter_fit",
                     "--x", "spend", "--y", "revenue", "--group", "region",
                     "--title", "Revenue scales with spend in every region"]),
    ("scatter_fit_lowess", ["--data", SALES, "--kind", "scatter_fit",
                            "--x", "units", "--y", "revenue", "--smooth"]),
    ("bar_group", ["--data", SALES, "--kind", "bar_group", "--x", "region",
                   "--y", "revenue", "--group", "channel", "--agg", "sum"]),
    ("bar_stacked", ["--data", SALES, "--kind", "bar_group", "--x", "region",
                     "--group", "channel", "--stacked"]),
    ("line_ci", ["--data", SALES, "--kind", "line_ci", "--x", "date",
                 "--y", "revenue", "--group", "region"]),
    ("dist", ["--data", SALES, "--kind", "dist", "--y", "satisfaction",
              "--group", "channel"]),
    ("corr", ["--data", SALES, "--kind", "corr"]),
    ("hist", ["--data", SALES, "--kind", "hist", "--x", "revenue",
              "--group", "channel"]),
    ("ecdf", ["--data", SALES, "--kind", "ecdf", "--x", "revenue",
              "--group", "region"]),
    ("area", ["--data", SALES, "--kind", "area", "--x", "date",
              "--y", "revenue", "--group", "channel", "--agg", "sum"]),
    ("donut", ["--data", SALES, "--kind", "donut", "--x", "region",
               "--y", "revenue", "--agg", "sum"]),
    # The title states only what the intervals support: two of the eight cross
    # the reference line, so "every arm beats it" would be false.
    ("forest", ["--data", TRIAL, "--kind", "forest", "--y", "estimate",
                "--lo", "ci_lo", "--hi", "ci_hi", "--labels", "arm",
                "--title", "Six of eight arms separate from the comparator"]),
    ("lollipop", ["--data", SALES, "--kind", "lollipop", "--x", "region",
                  "--y", "revenue", "--agg", "sum"]),
    ("errorbar", ["--data", SALES, "--kind", "errorbar", "--x", "channel",
                  "--y", "satisfaction"]),
    ("pairs", ["--data", SALES, "--kind", "pairs", "--group", "region"]),
    # The Chinese / GBK path.
    ("cn_survival", ["--data", CLIN, "--kind", "survival", "--x", "生存天数",
                     "--event", "是否发生事件", "--group", "组别"]),
    ("cn_dist", ["--data", CLIN, "--kind", "dist", "--y", "疗效评分",
                 "--group", "组别"]),
    ("cn_scatter", ["--data", CLIN, "--kind", "scatter_fit", "--x", "基线评分",
                    "--y", "疗效评分", "--group", "组别"]),
]


def main():
    sys.path.insert(0, HERE)
    os.makedirs(OUT, exist_ok=True)
    import plot_data

    failed = []
    for label, argv in CASES:
        argv = list(argv) + ["--out", label, "--outdir", OUT,
                             "--formats", "png"]
        try:
            code = plot_data.main(argv)
        except Exception:
            traceback.print_exc()
            code = 1
        status = "ok  " if code == 0 else f"FAIL({code})"
        print(f"  {status} {label}")
        if code != 0:
            failed.append(label)

    print(f"\n{len(CASES)} case(s) | {len(failed)} failed")
    if failed:
        print("failed:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

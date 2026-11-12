"""Detect figures whose saved PNG is much wider than its declared figsize.

`bbox_inches="tight"` expands the canvas rather than clipping, so a long
single-line footer silently inflates the export and shrinks every panel.
Source-level figsize audits cannot see this; only the rendered pixels can.
"""
from __future__ import annotations

import pathlib
import re

from PIL import Image

SCRIPTS = pathlib.Path(__file__).resolve().parent
OUTPUT = SCRIPTS.parent / "output"
DPI = 300.0
THRESHOLD = 1.12

RE_FIGSIZE = re.compile(r"figsize\s*=\s*\(\s*([\d.]+)\s*,\s*([\d.]+)")
RE_SAVEFIG = re.compile(r"savefig\(\s*fig\s*,\s*[\"'](\d+)_")


def main() -> None:
    declared: dict[int, tuple[float, float, str]] = {}
    for path in SCRIPTS.glob("fig_*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        num = RE_SAVEFIG.search(text)
        size = RE_FIGSIZE.search(text)
        if num and size:
            declared[int(num.group(1))] = (
                float(size.group(1)),
                float(size.group(2)),
                path.name,
            )

    inflated = []
    for png in OUTPUT.glob("*.png"):
        head = png.name.split("_", 1)[0]
        if not head.isdigit():
            continue
        num = int(head)
        if num not in declared:
            continue
        want_w, _, name = declared[num]
        px_w, _ = Image.open(png).size
        got_w = px_w / DPI
        ratio = got_w / want_w
        if ratio > THRESHOLD:
            inflated.append((ratio, num, name, want_w, got_w))

    inflated.sort(reverse=True)
    print(f"compared      : {len(declared)}")
    print(f"inflated >12% : {len(inflated)}")
    for ratio, num, name, want, got in inflated:
        print(f"  x{ratio:.2f}  #{num:>3}  {name:<40} declared={want:.1f}  actual={got:.1f}")


if __name__ == "__main__":
    main()

"""Rebuild reference/thumbs/ from the figures rendered in output/.

    cd scripts
    python build_reference.py                 # every PNG in output/
    python build_reference.py --range 0-54    # just the figures you changed
    python build_reference.py --out some/dir  # write elsewhere, to compare

reference/thumbs/ is the committed, downscaled copy of all 1219 figures: what a
person skims to pick a figure on a clone that has never rendered anything. It
goes stale whenever a figure changes, so run this after re-rendering and commit
the result.

reference/curated/ is hand-picked and is not touched here; to refresh one of
those, copy the PNG over from output/.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.abspath(os.path.join(HERE, ".."))
OUTPUT = os.path.join(SKILL, "output")
THUMBS = os.path.join(SKILL, "reference", "thumbs")

THUMB_WIDTH = 520
JPEG_QUALITY = 82
NUMBERED = re.compile(r"^(\d+)_.+\.png$")


def parse_range(text):
    lo, _, hi = text.partition("-")
    return int(lo), int(hi or lo)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--range", metavar="LO-HI",
                   help="only figures whose number falls in this range")
    p.add_argument("--out", default=THUMBS,
                   help="destination folder (default: reference/thumbs)")
    a = p.parse_args(argv)

    if not os.path.isdir(OUTPUT):
        print(f"nothing to do: {OUTPUT} does not exist. Render first with "
              f"`python run_all.py --parallel 8`.")
        return 1

    lo, hi = parse_range(a.range) if a.range else (0, 10 ** 9)
    pngs = []
    for name in sorted(os.listdir(OUTPUT)):
        m = NUMBERED.match(name)
        if m and lo <= int(m.group(1)) <= hi:
            pngs.append(name)
    if not pngs:
        print("no matching PNGs in output/")
        return 1

    os.makedirs(a.out, exist_ok=True)
    made, failed = 0, []
    for i, name in enumerate(pngs, 1):
        dst = os.path.join(a.out, os.path.splitext(name)[0] + ".jpg")
        try:
            with Image.open(os.path.join(OUTPUT, name)) as im:
                im = im.convert("RGB")
                height = max(1, round(im.height * THUMB_WIDTH / im.width))
                im = im.resize((THUMB_WIDTH, height), Image.LANCZOS)
                im.save(dst, "JPEG", quality=JPEG_QUALITY, optimize=True,
                        progressive=True)
            made += 1
        except Exception as exc:                              # noqa: BLE001
            failed.append((name, str(exc)))
        if i % 200 == 0:
            print(f"  {i}/{len(pngs)}", flush=True)

    total = sum(os.path.getsize(os.path.join(a.out, f))
                for f in os.listdir(a.out))
    print(f"{made} thumbnail(s) written to {os.path.relpath(a.out)}; "
          f"folder is now {total / 1024 / 1024:.1f} MB")
    for name, why in failed[:10]:
        print(f"  FAILED {name}: {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

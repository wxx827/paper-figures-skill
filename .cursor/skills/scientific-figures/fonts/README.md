# Bundled fonts

The house face of this library is **Times New Roman**. That font is
proprietary to Monotype and ships with Microsoft Office, so it cannot be
redistributed here. Bundling nothing was not an option either: on a machine
without it, matplotlib silently falls back to DejaVu Serif, which is about 8%
wider than Times. Every label reflows, and the figure you generate stops
matching the one in the gallery.

The fix is a *metrically compatible* clone. **Tinos** was designed by Steve
Matteson (Ascender) to have byte-identical advance widths to Times New Roman,
and it is released under the SIL Open Font License, which explicitly permits
redistribution inside a larger work.

The claim is verified rather than assumed:

```console
$ python -m check_typography --metrics
Times New Roman vs Tinos: 0.0 max advance-width delta
```

So a figure lays out identically whether it renders against the real Times New
Roman (Windows, or any machine with the MS core fonts) or against the bundled
Tinos (Linux, Docker, CI).

## Resolution order

Set in `SERIF_STACK` in [`../scripts/pubstyle.py`](../scripts/pubstyle.py):

| Order | Family | Where it comes from | Times-metric |
|---|---|---|---|
| 1 | Times New Roman | Windows / MS core fonts | reference |
| 2 | **Tinos** | **bundled here** | yes (measured, 0.0 delta) |
| 3 | Nimbus Roman | URW clone, default on most Linux distros | yes |
| 4 | Liberation Serif | default on most Linux distros | yes |
| 5 | TeX Gyre Termes | ships with TeX Live | yes |
| 6 | Times | macOS | yes |
| 7 | STIXGeneral | bundled with matplotlib itself | **no** |

Entries 1-6 share advance widths, so swapping between them never reflows a
label. `STIXGeneral` is the deliberate exception: it *looks* like Times, but
`check_typography.py --metrics` measures it 121 units/1000em away. It sits
last because a slightly-reflowed serif still reads as a journal figure, which
a DejaVu fallback does not. DejaVu is absent from the list entirely, and
`pubstyle.assert_font_ok()` raises if matplotlib reaches it anyway.

## Files

| File | License |
|---|---|
| `Tinos-Regular.ttf` | SIL OFL 1.1 |
| `Tinos-Bold.ttf` | SIL OFL 1.1 |
| `Tinos-Italic.ttf` | SIL OFL 1.1 |
| `Tinos-BoldItalic.ttf` | SIL OFL 1.1 |

Full licence text: [`LICENSE-Tinos.txt`](LICENSE-Tinos.txt).
Upstream: <https://github.com/googlefonts/tinos>.

The SIL OFL is independent of the licence covering this library's own code;
see the top-level `LICENSE` file for that.

## Chinese labels

`pubstyle.set_cn_font()` puts a Song/Ming face (SimSun, Songti SC, Noto Serif
CJK SC, ...) at the end of a concrete `font.family` list, so Latin text and
numerals stay in Times while only CJK characters come from the CJK face.

The concrete-list part matters more than it looks. Matplotlib walks a family
list glyph by glyph only when the list names real families; setting
`font.family="serif"` and extending `font.serif` instead resolves cleanly,
raises nothing, warns nothing, and renders every Chinese character as an empty
box. If you touch this code, test it by rendering a CJK label and looking at
the PNG — no automated check can distinguish a correct glyph from a tofu box.

`plot_data.py` detects non-Latin text in the column names and category levels
and calls this automatically, so a client can hand over a Chinese spreadsheet
without knowing any of the above.

Journal submissions should still use English labels.

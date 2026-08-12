# Reinforcement Learning for Robotics — print edition

The LaTeX build of the book, in the **Shannon Robotics** design language. Chapters
are generated from the web edition's MDX, so there is one source of truth for the
content and two presentations of it.

Current output: **`main.pdf`, 252 pages, 7 × 10 in.**

## Building

```bash
make            # convert MDX → LaTeX, then build the PDF
make convert    # regenerate chapters/*.tex only
make pdf        # build the PDF from the current chapters/
make clean      # remove build artefacts, keep the PDF
make dist       # remove everything generated
```

The build runs pdfLaTeX three times (contents, cross-references, final page
numbers) and **fails loudly** if LaTeX reports any error — logs are read with
`grep -a`, because TeX logs are binary and a plain `grep` silently skips them.

### Requirements

A TeX distribution with `tcolorbox`, `titlesec`, `fontawesome5`, `newpx`,
`listings`, `tikz`, `colortbl` and `collection-latexrecommended`.

This machine has **TinyTeX in `~/.TinyTeX`** (a userspace install — no root was
needed). If `pdflatex` is not on your `PATH`:

```bash
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
```

To install missing packages: `tlmgr install <package>`.

## Layout

```
rl4r-latex/
├── main.tex           # book assembly: title, colophon, contents, parts
├── shannon.sty        # the design system — all colour, type and block styling
├── build/
│   └── mdx2tex.py     # MDX → LaTeX converter
├── chapters/          # generated; do not edit by hand
│   └── ch01.tex … ch22.tex
└── main.pdf
```

## The design system

`shannon.sty` translates shannon.id into print:

| Token | Value | Role |
|---|---|---|
| `shTeal` | `#14B8A6` | primary accent — rules, badges, chapter labels |
| `shTealDark` | `#0F766E` | accent on white, where teal needs more contrast |
| `shSky` / `shSkyDark` | `#0EA5E9` / `#0369A1` | secondary accent, theorem blocks |
| `shSlate900 … shSlate50` | slate scale | ink, rules, block fills |
| `shFoundation` / `shConceptual` / `shPractical` | sky / teal / amber | the three FCP layers |

Headings and structural furniture use the display face — **Space Grotesk** when
compiled with XeLaTeX or LuaLaTeX on a system that has it, and a metric-compatible
sans otherwise. Body text is a Palatino-class serif chosen for long-form reading;
the brand's grotesk is deliberately reserved for structure, because a geometric
sans is the wrong tool for 250 pages of derivations.

Blocks are flat cards with a coloured leading rule — no gradients, no shadows,
matching the site's aesthetic.

### Environments

| Environment | Purpose |
|---|---|
| `shfoundation`, `shconceptual`, `shpractical` | the FCP layer callouts |
| `shinsight`, `shwarning`, `shnote`, `shrobot` | key ideas, caveats, asides, robot-thread notes |
| `shtheorem`, `shdefinition`, `shlemma`, `shproposition`, `shcorollary` | boxed statements, numbered per chapter |
| `proof` | restyled; the QED marker uses amsthm's `\pushQED` so it lands on the last line |
| `shrust` | a Rust listing with a file-path bar and caption |
| `shexercises` + `\shexercise` | exercises tagged by layer with a difficulty rating |
| `shcodingtask` | build-this blocks with deliverables |
| `shreferences` + `\shref` | bibliography, with `\shbaseline` marking the four baseline works |
| `\shwidget` | an interactive's print fallback |
| `\shepigraph`, `\shoverview`, `shoutcomes` | chapter opener furniture |

## How the conversion works

`build/mdx2tex.py` parses the MDX directly rather than going through pandoc,
because the book's content lives in custom JSX components that a generic
converter would drop. It handles:

- **Frontmatter** → chapter title and epigraph
- **Every custom component** → its matching LaTeX environment
- **Markdown** → headings, lists, tables, emphasis, links, rules
- **Math** → passed through, since it is already LaTeX; only stray Unicode
  symbols are respelled as commands (`τ` → `\tau`), with separate maps for text
  and math mode so `$…$` never gets escaped
- **Rust fences** → `shrust` + `lstlisting`, with source transliterated to ASCII
  because pdfLaTeX's verbatim path is fragile with multi-byte input
- **Long display equations** → scaled to the measure when they cannot break
- **Interactive simulations** → described figures carrying the widget id, so a
  print reader knows exactly what to open in the web edition

One diagram — the see–think–act loop that opens Chapter 1 — is redrawn in TikZ
rather than described, since it is the book's opening illustration. Add more by
extending `TIKZ_FIGURES`, keyed on the figure label.

The generated `.tex` files are **pure ASCII**, so they build the same way
anywhere.

## Editing

Edit the MDX in `../rl4r-web/content/chapters/` — never `chapters/*.tex`, which
is overwritten on every `make convert`. Fixes to prose or math benefit both
editions at once.

To change presentation, edit `shannon.sty`. To change how a component maps into
print, edit `build/mdx2tex.py`.

## Known limitations

- Ten overfull boxes remain (worst 48 pt), all in wide tables or long formulas.
  They are visible only as slight margin encroachment.
- Interactive simulations are described, not rendered. Producing static plots
  from the TypeScript engine and embedding them would be the next real
  improvement.
- The index and appendices in TOC.md are not yet built.

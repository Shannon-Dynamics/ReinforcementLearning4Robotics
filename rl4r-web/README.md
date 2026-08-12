# Reinforcement Learning for Robotics — The FCP Way

The interactive web book. Twenty-two chapters teaching reinforcement learning for robotics through three interleaved layers: **Foundation** (complete mathematical formalism), **Conceptual** (an interactive simulation for every hard idea), and **Practical** (Rust implementations).

## Running it

```bash
npm install
npm run dev      # http://localhost:3000
```

Requires Node 18.18+ (Next.js 15). Other scripts:

```bash
npm run build      # production build; also the correctness gate
npm run typecheck  # tsc --noEmit
npm run lint
```

## Layout

```
rl4r-web/
├── app/                      # Next.js App Router
│   ├── page.tsx              # landing
│   ├── about/                # the method, palette, toolchain
│   └── chapters/[slug]/      # chapter renderer (SSG, one route per chapter)
├── content/chapters/         # the book itself — 22 MDX files
├── components/
│   ├── layout/               # header, theme provider, chapter navigation
│   ├── book/                 # MDX component map, callouts, quotes, exercises
│   ├── viz/                  # Nivo chart wrappers + chart chrome
│   └── sim/                  # 19 interactive simulations
└── lib/
    ├── chapters.ts           # the TOC as typed data (mirrors ../TOC.md)
    ├── theme.ts              # the validated palette and Nivo theme
    ├── mdx.ts                # MDX loading and heading extraction
    └── rl/                   # the simulation engine (see below)
```

### The simulation engine

Every interactive runs real algorithms, not canned animations. [lib/rl/](lib/rl/) contains:

| Module | Contents |
|---|---|
| `random.ts` | Seeded mulberry32 RNG, Gaussian/categorical/Beta sampling |
| `gridworld.ts` | Rusty's warehouse MDP — both the white-box `transitions` and black-box `step` views |
| `dp.ts` | Policy evaluation, policy iteration, value iteration — as generators yielding one snapshot per sweep, so the UI can animate them |
| `bandits.ts` | ε-greedy, UCB1, gradient bandit, Thompson sampling, and the 10-armed testbed harness |
| `td.ts` | MC prediction, TD(0), SARSA, Expected SARSA, Q-learning, Double Q, Sarsa(λ) |
| `pendulum.ts` | Pendle's dynamics with Euler / semi-implicit / RK4 integrators and an energy-shaping controller |

Constants match the chapter designs exactly — the warehouse is 12×9 with `p_slip = 0.2`, `γ = 0.95`, rewards +25 / −1 / −10, as fixed in Chapter 4 and reused through Chapter 7.

## Writing a chapter

Chapters are MDX in `content/chapters/chNN-slug.mdx`. Frontmatter carries the title and the researcher epigraph:

```mdx
---
title: Chapter Title
chapter: 5
quote:
  text: The quotation.
  author: Who said it
  affiliation: Where they work
  source: Which work it comes from
---
```

Components available in MDX without importing (see [components/book/MdxContent.tsx](components/book/MdxContent.tsx)):

- **Structure** — `ChapterOverview`, `Callout`, `Theorem`, `Proof`, `Figure`
- **Code** — `RustSnippet` wrapping a fenced ` ```rust ` block
- **Ending** — `Exercises`, `CodingTask`, `References`
- **Charts** — `LineChart`, `BarChart`, `StatRow`, `StatTile`
- **Simulations** — `RustyDrive`, `SuccessLevels`, `PendleSim`, `ContractionDemo`, `BanditTestbed`, `MdpExplorer`, `GpiDashboard`, `TdDashboard`, `LambdaDial`, `DeadlyTriad`, `ReplayBuffer`, `PolicyGradientLab`, `EntropyDial`, `ModelBiasFan`, `CurseOfDimensionality`, `DomainRandomization`, `CovariateShift`, `DmpSculptor`, `RewardMixer`, `PipelineSwitcher`, `GraspWrench`

Math is KaTeX: `$inline$` and `$$display$$`. Code blocks are highlighted at build time by Shiki, so no highlighting JS ships to the reader.

Adding a chapter means writing the MDX file and adding its entry to `lib/chapters.ts`. The route, navigation, and static generation follow automatically.

## Design rules

**The palette is validated, not chosen.** Both light and dark sets pass the lightness band, chroma floor, colorblind separation (Machado 2009 at severity 1.0), normal-vision floor, and contrast checks. Do not add colors — use the eight categorical slots in order, the sequential ramp for magnitude, and the diverging pair for polarity. Scatter-type charts cap at three series.

**Encoding is consistent book-wide.** Value functions get the sequential ramp; policies get directional arrows in slot 1; TD error, advantage and signed reward get the diverging pair; uncertainty gets a fan or band. A color means the same thing in Chapter 20 as in Chapter 4.

**Accessibility is structural.** Every chart carries a table view. Every callout pairs an icon with a text label, so meaning never rests on color alone. Every simulation has a caption describing what it shows. Wide content scrolls inside its own container; the page body never scrolls horizontally.

**Both themes are selected, not flipped.** The dark palette is its own set of steps chosen against the dark surface. Tokens live in [app/globals.css](app/globals.css); nothing else defines a color.

## Relationship to the design docs

The parent directory holds the book's design: `TOC.md` is the authoritative structure, and `Chapter-1.md` … `Chapter-22.md` are the per-chapter design documents specifying sections, widgets, Rust plans and exercises. `CLAUDE.md` holds the development conventions.

This web app implements those designs. When they disagree, the design docs are the source of truth for *what* a chapter covers; this app is the source of truth for *how* it is presented.

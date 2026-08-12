# CLAUDE.md — Development Guide for *Reinforcement Learning for Robotics — The FCP Way*

This repository holds the **design and implementation** of an interactive web book. Read `TOC.md` first — it is the single source of truth for book structure, chapter scope, and the recurring cast of robots. Each `Chapter-N.md` (N = 1…22) is the design/storyline document for one chapter.

## 1. What this project is

An interactive web book teaching reinforcement learning for robotics with the **FCP method** — every chapter delivers three interleaved layers:

- **F — Foundation**: complete mathematical formalism (definitions, theorems, full derivations, convergence conditions, and the assumptions robots violate).
- **C — Conceptual**: every hard concept gets an interactive visual — a web widget, animation, or live dashboard the reader manipulates.
- **P — Practical**: every algorithm implemented in Rust with best-in-class crates; the same code trains natively and demos in-browser via WASM.

Baseline references (PDFs in `RLBooks/Reinforcement Learning for Robotics/`):

| File | Work | Used for |
|---|---|---|
| `SuttonBartoIPRLBook2ndEd.pdf` | Sutton & Barto, *RL: An Introduction* 2nd ed. — **note: the on-disk PDF is the "in progress" draft**, whose chapter/section numbering diverges from the published 2018 edition (bandit sections shift by one from §2.4 on; eligibility traces are draft ch. 7; the draft has no ch. 12–13 and lacks the deadly-triad and policy-gradient-theorem material entirely) | Parts I–II mathematical spine. **Citation convention: all design docs cite published-2018 numbering as primary**, flagging draft numbering in parentheses where it differs; material absent from the draft is verifiable only against the published edition |
| `Kober_IJRR_2013.pdf` | Kober, Bagnell & Peters, IJRR 2013 survey | Part III (four curses, representations, priors, models, ball-in-a-cup) |
| `2408.03539v3.pdf` | Tang et al. 2024, *DRL for Robotics: Real-World Successes* | Part IV taxonomy, L0–L5 success levels, modern trends |
| `ireti_rl_learning.pdf` | Akinola, Columbia lecture slides | Pedagogical framing, LfD material (Ch 16) |

**Modernization contract**: where the references stop, the book continues (PPO/SAC workhorses, world models, offline RL, teacher–student sim-to-real, foundation models) — always grounded in the same mathematical spine and cited to the baseline texts where they lead.

## 2. Repository layout

```
RLforRobotics/
├── TOC.md            # Book structure — SINGLE SOURCE OF TRUTH
├── CLAUDE.md         # This file
├── Chapter-1.md … Chapter-22.md   # Per-chapter design/storyline docs
├── RLBooks/          # Baseline reference PDFs (read-only)
└── rl4r-web/         # THE BOOK — Next.js implementation (see rl4r-web/README.md)
```

**Two artifacts, two roles.** The `Chapter-N.md` design docs specify *what* each chapter covers — sections, F/C/P cells, widget inventory, Rust plan, exercises. `rl4r-web/` is the published book that implements them. When they disagree, the design docs are authoritative for scope; the web app is authoritative for presentation.

Inside `rl4r-web/`:

```
rl4r-web/
├── app/                      # Next.js App Router: landing, /about, /chapters/[slug]
├── content/chapters/         # The book text — 22 MDX files, chNN-<slug>.mdx
├── components/
│   ├── layout/               # header, ThemeProvider, chapter navigation
│   ├── book/                 # MdxContent (the component map), callouts, quotes, exercises
│   ├── viz/                  # Nivo wrappers + ChartFrame (title/legend/table-view chrome)
│   └── sim/                  # interactive simulations, one file per widget
└── lib/
    ├── chapters.ts           # TOC as typed data — mirrors TOC.md
    ├── theme.ts              # validated palette, colormaps, Nivo theme
    ├── mdx.ts                # MDX loading, heading extraction
    └── rl/                   # TypeScript simulation engine (gridworld, dp, bandits, td, pendulum)
```

**Rust reference crates** (`rl-core`, `rl-tabular`, `rl-deep`, `rl-envs`, `rl-sim`) remain the book's **P-layer curriculum** — the code readers build and the snippets chapters display. They are specified in each design doc's §5 and are a separate deliverable from the website; the site presents them, it does not compile them.

## 3. Toolchain decisions (and why)

### Web platform (revised 2026-08 — supersedes the original mdBook plan)

**Next.js 15 (App Router) + React 19 + TypeScript + Tailwind 3**, with content as **MDX** (`next-mdx-remote/rsc`), math by **KaTeX** (`remark-math` + `rehype-katex`), and code highlighted at build time by **Shiki** (`rehype-pretty-code`), so no highlighting JavaScript reaches the reader.

Charts use **Nivo** (`@nivo/line`, `bar`, `heatmap`, `radar`, `scatterplot`); bespoke simulations are hand-written SVG/canvas driven by a **TypeScript RL engine** in `lib/rl/`, so every interactive runs the real algorithm rather than a canned animation.

*Why this replaced mdBook + Rust/WASM widgets:* the original plan compiled each widget to a separate WASM bundle embedded by iframe. React composition gives dramatically tighter interaction between prose and simulation (a slider in one panel updating three charts and a diagram), far faster iteration, and no per-widget build step. The cost is that the in-page simulations are TypeScript rather than the book's own Rust — accepted deliberately: the Rust crates remain the taught curriculum and the source of every code listing, while the browser simulations exist to make the *math* manipulable.

**Constraint on file:** Node 18.19 is the available runtime, which caps Next at 15.x (`^18.18.0`) and Tailwind at 3.x. Revisit if Node 20+ becomes available.

### Rust crates for the P layer

These are the crates the book teaches and whose APIs appear in chapter code listings (verify latest on crates.io before pinning):

| Domain | Primary | Why | Alternatives (documented in Appendix B) |
|---|---|---|---|
| Deep learning | `burn` | Pure Rust, multi-backend (WGPU/CUDA/Metal/ndarray), trains natively AND runs in-browser via WGPU/WebGL — one codebase for the whole F→P pipeline | `candle` (lean inference, HF ecosystem), `tch-rs` (libtorch escape hatch) |
| Tensors/linear algebra | `ndarray`, `nalgebra` | ndarray for ML-shaped data; nalgebra for geometry/SE(3)/Riccati | `faer` for heavy linear algebra |
| Physics | `rapier2d` / `rapier3d` | Pure Rust, deterministic option, WASM-compatible — sim runs in the reader's browser | `avian` (bevy-native), MuJoCo bindings (native-only) |
| Robot models | `urdf-rs`, `k` | URDF parsing + serial-chain FK/IK/Jacobians | `rs-opw-kinematics` (analytic 6-DoF IK) |
| 3D scenes | `bevy` + `bevy_rapier` (+ `bevy_urdf` where it fits) | WASM-capable 3D for locomotion/manipulation demos | `three-d`, raw `wgpu` |
| Dashboards/plots | `egui`/`eframe` + `egui_plot` | Immediate-mode, trivially WASM, perfect for live training dashboards | `plotters` (static/canvas), `rerun` (native dev telemetry — not embedded in book pages) |
| Parallelism | `rayon` | Vectorized env rollout farms | — |
| RNG/stats | `rand`, `rand_distr`, `statrs` | Seeded, reproducible experiments | — |
| Optimization | `cmaes` | Policy search on DMP weights (Ch 17), sysid (Ch 15) | `argmin` |
| Infra | `serde`, `clap`, `tracing`, `criterion`, `proptest` | Checkpoints/configs, CLIs, logs, benchmarks, property tests | — |

**Rule**: the book's teaching code lives in the book's own crates (`rl-core` etc.) — we *build* the gym-style abstractions rather than depending on immature third-party RL crates; that construction is itself curriculum (Ch 4). External RL crates may be *mentioned* (survey sidebar), never depended on.

## 4. Chapter design documents (`Chapter-N.md`) — required template

Every chapter design doc uses **exactly** this skeleton (agents and future sessions: do not improvise headings):

```markdown
# Chapter N — <Title>

> **Part** <roman> — <part name> · **Builds on:** Ch … · **Feeds:** Ch …
> **Modernizes:** <specific sections of the baseline references>

## 1. Purpose & Learning Outcomes
<1 short paragraph + 4–7 outcome bullets ("The reader can …")>

## 2. Storyline
<The narrative arc in 3 acts: the hook/problem, the development, the payoff.
Name the running robots used (Rusty/Pendle/Reacher/Ferris) and how they advance.>

## 3. Section-by-Section Design
### N.1 <Section title>
- **F:** <math delivered: definitions, theorems, derivations — be specific, name the equations>
- **C:** <the visual/interactive that teaches it — what the reader sees and manipulates>
- **P:** <what gets built in Rust — crates, functions, experiment>
<repeat for 5–9 sections; final section is always "N.x Chapter Bridge" (recap + what's next)>

## 4. Interactive Widgets & Dashboards
| ID | Type | What it shows | Reader interaction | Tech |
<table of every widget; IDs like `chNN-<slug>`; Type ∈ {widget, animation, dashboard, sandbox, gallery}>

## 5. Rust Implementation Plan
<crates touched, modules/files added, one representative code sketch (10–40 lines, realistic API),
experiments/benchmarks run, and what artifacts the reader can run natively vs in-browser>

## 6. Robot Thread
<which cast robots appear, their state before/after this chapter>

## 7. Exercises & Explorations
<4–8: mix of math derivations (F), widget experiments (C), code extensions (P)>

## 8. Notation Introduced
<new symbols with definitions, consistent with Appendix C conventions (S&B-compatible)>

## 9. References & Further Reading
<baseline-book sections first, then modern papers with year>
```

Length target per design doc: **150–300 lines**. No placeholders like "TBD" — every F/C/P cell must be concrete enough that a writer could produce the chapter from it.

## 5. Style guide

**Math (F layer)**
- Notation follows Sutton & Barto 2nd ed. (states `s`, actions `a`, `v_\pi(s)`, `q_\pi(s,a)`, returns `G_t`, TD error `\delta_t`); robotics symbols follow convention: configurations `q`, torques `\tau`, `M(q)\ddot q + C(q,\dot q)\dot q + g(q) = \tau`. Appendix C is the arbiter; new symbols must be registered in each chapter's §8.
- Derivations are **complete** — no "it can be shown". If a proof is genuinely out of scope, state the theorem precisely and cite where the proof lives.
- KaTeX-compatible LaTeX only (mdbook-katex); display math in `$$…$$`.

**Prose (C layer)**
- Second person, active voice, robot-first examples. Every abstraction lands on one of the cast robots within a paragraph.
- Each hard concept: intuition paragraph → visual → formalism → back-reference to the visual ("drag λ to 1 and you have just rediscovered Monte Carlo").
- Widgets must degrade gracefully: every interactive has a static figure + caption fallback (accessibility + print).

**Rust (P layer)**
- Edition 2021+, `cargo clippy -- -D warnings` clean, `rustfmt` default.
- Teaching code optimizes for readability: no `unsafe`, minimal generics in chapter code (traits live in `rl-core`), explicit types at API boundaries.
- Every algorithm: seeded RNG, deterministic tests (`proptest` for laws like Bellman-consistency), a `criterion` benchmark when performance is the point.
- Every chapter's code compiles to **both** native and `wasm32-unknown-unknown` unless physically impossible (note it when so).

**Visualization**
- One visual language book-wide: value = sequential ramp, policy = directional arrows in slot 1, reward/δ/advantage = diverging pair, uncertainty = fan or band. A color means the same thing in Ch 20 as in Ch 4.
- The palette is **validated, not chosen**: both modes pass the lightness band, chroma floor, CVD separation (Machado 2009, severity 1.0), normal-vision floor and contrast checks. Do not add colors — use the eight categorical slots in fixed order. Scatter-type charts cap at three series. All tokens live in `app/globals.css`; nothing else defines a color.
- Dashboards show *live quantities the math named* (δ, Δ, KL, clip fraction, entropy) — widgets are the equations made tangible, not decorations.
- Accessibility is structural: every chart carries a table view, every callout pairs an icon with a text label, every simulation has a standalone caption, and wide content scrolls inside its own container.

## 6. Build & development workflow

```bash
cd rl4r-web
npm install
npm run dev        # http://localhost:3000, live reload
npm run build      # production build — the correctness gate; run before declaring done
npm run typecheck  # tsc --noEmit
```

`npm run build` catches MDX syntax errors, broken component references, and type errors across the whole book, so it is the standing check after any content or component change.

Chapter development order: read the `Chapter-N.md` design doc, build any simulation the chapter needs (in `components/sim/`, backed by real algorithms in `lib/rl/`), then write the MDX against the working widget. **Simulation before prose** — writing "drag λ to 1 and you have rediscovered Monte Carlo" is only honest once the dial exists and does that.

Definition of done (per chapter):
1. All design-doc §3 sections present with F/C/P layers; derivations verified symbol-by-symbol.
2. Every widget the chapter references is implemented, works in both themes, and carries a caption that stands alone.
3. Rust listings are realistic against the crate APIs and carry a caption saying what they demonstrate.
4. Citations checked against the PDFs in `RLBooks/`; baseline references flagged as such in the `References` block.
5. `npm run build` clean; chapter registered in `lib/chapters.ts`.
6. Reviewed against TOC.md scope — no drift.

## 7. Working conventions for Claude sessions

- **TOC.md is authoritative** for scope/order/naming. If a design decision conflicts with it, update TOC.md deliberately (and say so) rather than silently diverging.
- The four cast robots (Rusty, Pendle, Reacher, Ferris) are fixed vocabulary — do not invent new mascots.
- Widget IDs are stable API: `chNN-<slug>` referenced from prose; never rename without a grep.
- When citing baselines, cite precisely (e.g., "Kober §3.2 curse of real-world samples", "Tang §4.1.1", "S&B §6.5") — the PDFs are in `RLBooks/` for verification; extracted text may be regenerated with `pdftotext`.
- Verify crate versions on crates.io before pinning; record any substitution rationale in Appendix B and in §3 above.
- Keep chapter design docs and any future book prose in sync — a change to one requires reviewing the other.

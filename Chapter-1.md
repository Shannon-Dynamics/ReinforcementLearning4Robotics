# Chapter 1 — Why Reinforcement Learning for Robotics?

> **Part** I — Foundations of Sequential Decision-Making · **Builds on:** — (book entry point) · **Feeds:** Ch 2 (math toolkit), Ch 3 (first learning algorithms), Ch 4 (MDP formalism), Ch 14 (four curses, previewed here), Ch 16 (LfD in depth), Ch 18–20 (competency deep dives)
> **Modernizes:** Kober §1.1–1.3 (RL in the context of ML, optimal control, robotics); Tang §1 and §3.1–3.4 (competency taxonomy, problem formulation, solution approach, levels L0–L5); Akinola motivation slides (see–think–act, why direct programming breaks, LfD vs RL); S&B Ch 1 read informally.

## 1. Purpose & Learning Outcomes

This chapter sells the *why* before any theorem: robots that must perceive, decide, and act in unstructured worlds outgrow hand-written controllers, and reinforcement learning is the discipline of replacing hand-written decision rules with optimized ones. The reader leaves with the book's vocabulary (informal), the cast of robots, a working toolchain, and — crucially — the *felt experience* of a hand-coded controller failing in their own browser.

The reader can:
- trace the see–think–act loop on a concrete robot and name which stage each robotics subfield automates;
- explain, using Akinola's four axes (structure/consistency, perception, manipulation, deformation), why direct programming stops scaling from vacuum robots to laundry robots;
- position learning from demonstration, reinforcement learning, and hybrid pipelines relative to each other, and state each one's preconditions;
- describe the four informal ingredients of an RL problem (agent, environment, reward, policy) on three different robots;
- place any published robot-RL system in Tang's taxonomy: competency, problem formulation, solution approach, and level of real-world success L0–L5;
- build the book's Rust workspace, run the first WASM demo, and reproduce the hand-coded controller's failure modes deliberately.

## 2. Storyline

**Act I — the hook.** The reader opens `ch01-drive-rusty` and meets **Rusty**, a differential-drive robot in a tidy warehouse pen, driven by a 15-line hand-coded wall-follower that looks flawless. Then the reader presses *Randomize*: shelf layout reshuffles, lidar noise rises, a wheel starts slipping. The controller was correct; the world was not. Akinola's four axes give the failure a name — the pen had structure, the real warehouse does not.

**Act II — the development.** Three roads out: program harder (works exactly when the task has vacuum-robot structure), demonstrate (LfD: a teacher shows state–action pairs), or let the robot practice against a scalar judgment of success (RL). Kober's framing anchors the chapter: if a task can be phrased as an optimization problem and exhibits temporal structure, RL applies (Kober §1.3). We dissect the anatomy of an RL problem informally — agent, environment, reward, policy, exploration — and make one precise promise: every arrow in the see–think–act loop will become a measurable mathematical object (Ch 2 builds the probability; Ch 4 names the loop an MDP).

**Act III — the payoff.** A guided tour of what actually worked, organized by Tang's taxonomy: champion-level drone racing (L5), ANYmal-class rough-terrain locomotion (L4), locomotion controllers shipped in production quadrupeds (L5), in-hand cube reorientation (L2). The L0–L5 ladder is presented as an honesty instrument, not a leaderboard — most of the literature lives at L0–L1. The chapter closes on the book's contract (F/C/P), the cast (Rusty now; Pendle joins in Ch 2, Reacher in Ch 3, Ferris in Ch 15), and the toolchain the reader has just verified end to end.

## 3. Section-by-Section Design

### 1.1 See, Think, Act
- **F:** The perception–decision–actuation loop as a discrete-time dynamical system, stated informally but with exact shape: at step $t$ the robot receives observation $o_t$, chooses action $a_t = \pi(o_t)$, and the world advances to a new configuration influenced by $a_t$ plus disturbance. Explicit promissory notes: "receives" and "advances" become random variables on a probability space (Ch 2.1) and a conditional distribution $p(s' \mid s, a)$ (Ch 4.1). The observation-vs-state distinction is planted here, quoting Kober §1.3 (robots never see true state, noise-free), and harvested in Ch 4.6 (POMDPs).
- **C:** `ch01-see-think-act` — Rusty's loop drawn as an animated ring: lidar frame at *see*, a decision box at *think*, wheel commands at *act*, world state flowing back. The reader plays/pauses, steps one stage at a time, and turns a sensor-noise dial to watch a fixed *think* box make progressively worse choices — the loop's fragility is felt before it is formalized.
- **P:** Workspace bootstrap: `cargo xtask serve` builds demos + book with live reload; guided tour of `crates/` — `rl-core` (traits, metrics), `rl-tabular`, `rl-deep`, `rl-envs`, `rl-sim`, `rl-viz` — with a one-screen map of which chapter fills which crate.

### 1.2 Why Direct Programming Breaks
- **F:** A controller is a function $\pi : \mathcal{O} \to \mathcal{A}$; programming is specifying $\pi$ by hand, case by case. The informal scaling argument: specification effort grows with the diversity of inputs actually encountered, and Akinola's four axes name the diversity sources — (i) degree of structure and consistency, (ii) perception, (iii) manipulation contact, (iv) deformation — illustrated by the slides' contrasts (vacuum robot vs home-cleaning robot, manufacturing arm vs cooking robot, pool cleaner vs laundry robot). Kober §1.3's quantitative note: 10–30 dimensional continuous state–action spaces are *normal* in robotics and already count as large.
- **C:** `ch01-drive-rusty` — sandbox: hand-coded wall-follower vs the reader driving with WASD. Buttons randomize layout, lidar noise $\sigma$, and wheel-slip probability; a failure counter accumulates, and each crash expands into a "why" trace showing the exact lidar frame that fooled the controller.
- **P:** `rl-envs::rusty`: differential-drive kinematics, a `Controller` trait, and `WallFollower` (the §5 code sketch). Experiment: `rayon`-parallel sweep over noise $\sigma \in [0, 0.5]$ m × 100 layout seeds; plot the success-rate cliff with `egui_plot`. The reader's first reproducible experiment — seeded, config-driven.

### 1.3 Three Roads to a Policy: Demonstrate, Reinforce, Hybridize
- **F:** LfD stated as supervised learning on a demonstration set $D = \{(s, a)\}$ with its preconditions made explicit (a teacher exists; demonstration is possible — Akinola's LfD slides); RL stated as optimization of cumulative reward with no teacher, only evaluative feedback (Kober §1.1: reward specifies *what*, never *how*); hybrid pipelines (demonstrations initialize, RL refines) previewed with pointers to Kober §5 (prior knowledge), Ch 16 (imitation & offline RL), and Ch 17 (ball-in-a-cup). A three-column cost table: teacher availability vs sample cost vs reward-design burden.
- **C:** `ch01-three-roads` — the LfD / RL / hybrid triangle with ~15 real systems positioned inside it; clicking a system reveals which preconditions held (was a teacher available? was reward easy to write?) and links to its L0–L5 card in `ch01-success-levels`.
- **P:** Three scripted vignettes in the Rusty sandbox: replay of a recorded human demo (LfD teaser), a reward ticker driving visibly random exploration (RL teaser), demo-then-improve (hybrid teaser). No learning code yet — staged trajectories; Ch 3 onward builds the machinery honestly.

### 1.4 The Anatomy of an RL Problem
- **F:** Informal but disciplined definitions: agent, environment, reward $r_{t+1} \in \mathbb{R}$ (indexed per S&B convention: the reward *follows* the action), policy $\pi$, return as accumulated reward (discounting teased, defined in Ch 4.2), and the exploration–exploitation dilemma stated as the tension Ch 3 isolates in its purest form. S&B's four elements (policy, reward signal, value function, model — S&B §1.3) each land on a robot: Rusty's route policy, a docking reward, "how promising is this corridor" as value, a floor map as model.
- **C:** Overlay lenses on `ch01-see-think-act`: a *reward lens* (scalar ticker over the animation), a *policy lens* (arrow field over the pen), a *value lens* (heatmap rendered from a precomputed table — a deliberate glimpse of the Ch 4–5 payoff).
- **P:** Instrument the sandbox with `rl_core::Metrics` (per-step reward, episode return, collision count). The reader writes their first reward function in a `serde` config — `+10` dock, `−1` per step, `−5` collision — and watches the ticker judge *their own* driving before any agent is trained on it.

### 1.5 What Actually Worked: Competencies and Levels
- **F:** Tang's four taxonomy axes stated precisely (§3.1–3.4): *competency* (mobility → locomotion/navigation; manipulation → stationary/mobile; multi-agent → HRI/multi-robot), *problem formulation* (action-space level low/mid/high, observation space, sparse vs dense reward — formalized in Ch 4.7), *solution approach* (simulator usage, model learning, expert usage, policy optimization, representation), and *level of real-world success*: L0 validated only in simulation; L1 limited lab; L2 diverse lab; L3 confined real-world operational conditions; L4 diverse, representative real-world operational conditions; L5 deployed in commercial products. Framed as the evaluation rubric the book applies to itself (Ch 14, Ch 22).
- **C:** `ch01-competency-explorer` — the taxonomy as an expandable tree with example systems at each leaf; `ch01-success-levels` — every surveyed system on the L0–L5 ladder, filterable by competency and year, showing the empirical pattern Tang reports: locomotion mature (L4–L5), manipulation stuck below L3 by object diversity, HRI simulation-starved.
- **P:** Both dashboards are data-driven from `book/data/systems.json` (`serde` schema = the taxonomy itself): ~40 systems tagged with competency, formulation, approach, level, year, citation. Readers extend it by PR — tagging a paper correctly *is* the exercise.

### 1.6 The FCP Method, the Cast, and the Toolchain
- **F:** The book's contract stated as three invariants: derivations complete (F), every hard concept manipulable (C), every algorithm compiling in Rust for native + WASM (P); "grounded modernization" defined — S&B's spine, Kober's bridge, Tang's taxonomy, then PPO/SAC/world-models where the references stop.
- **C:** `ch01-crate-map` — the workspace dependency graph with hover cards (crate → role → chapters that fill it); cast cards for Rusty (Ch 4–7, 9, 19), Pendle (Ch 2, 8, 10–13), Reacher (Ch 3, 11, 13, 16–17, 20), Ferris (Ch 15, 18, 22).
- **P:** Toolchain smoke test: build `ch01-drive-rusty` for native and `wasm32-unknown-unknown`, run `cargo xtask test-all`; a green run certifies the reader's environment for the entire book.

### 1.7 Chapter Bridge
- **F:** Recap of the claims made and the IOUs issued: the loop is not yet math, "reward" is not yet a random variable, "eventually" is not yet a limit theorem. Forward pointer: Ch 2 supplies probability spaces, Markov chains, contractions, ODE discretization, and Robbins–Monro — the toolkit every later proof draws from — and introduces Pendle, the robot simple enough to do all of it exactly.
- **C:** A single mermaid roadmap diagram: Part I's chapters as stations, each annotated with the tool it adds and the robot that carries it.
- **P:** `cargo xtask test-all` as the standing definition of "your setup still works"; pointer to Appendix A for readers new to Rust.

## 4. Interactive Widgets & Dashboards

| ID | Type | What it shows | Reader interaction | Tech |
|---|---|---|---|---|
| `ch01-see-think-act` | animation | Rusty's perception–decision–actuation ring with live data flowing; reward/policy/value overlay lenses | play/pause, single-step stages, sensor-noise dial, toggle lenses | `egui` + `rl-viz` embed shell |
| `ch01-drive-rusty` | sandbox | Rusty in a warehouse pen: hand-coded wall-follower vs manual driving; failure counter with per-crash lidar trace | WASD driving, toggle controller, randomize layout/noise/slip, inspect failure traces | `egui` + `rapier2d` (WASM) |
| `ch01-three-roads` | widget | LfD / RL / hybrid triangle with real systems positioned by preconditions | click systems for precondition cards; filter by robot type | `egui` |
| `ch01-competency-explorer` | widget | Tang's competency taxonomy as an expandable tree with exemplar systems | expand/collapse nodes, filter by morphology and year | `egui`, data from `systems.json` |
| `ch01-success-levels` | dashboard | Surveyed systems on the L0–L5 ladder, colored by competency | hover for system cards, filter by competency/year/formulation | `egui_plot`, data from `systems.json` |
| `ch01-crate-map` | widget | Workspace crate graph and robot-cast cards with chapter threads | hover crates/robots for role + chapter links | `mermaid` + `egui` hover layer |

Every widget ships the style guide's static fallback: a captioned figure of its most instructive state (e.g., the wall-follower's failure frame for `ch01-drive-rusty`).

## 5. Rust Implementation Plan

Crates touched: `rl-envs` (new module `rusty/`), `rl-core` (new `metrics.rs`), `rl-viz` (first use of the WASM embed shell), `demos/ch01-drive-rusty`, `demos/ch01-see-think-act`. Files: `rl-envs/src/rusty/diff_drive.rs`, `rl-envs/src/rusty/lidar.rs`, `rl-envs/src/rusty/warehouse_pen.rs`, `rl-core/src/metrics.rs`.

```rust
// rl-envs/src/rusty/diff_drive.rs
pub struct Pose { pub x: f64, pub y: f64, pub theta: f64 }
pub struct Twist { pub v: f64, pub omega: f64 }   // forward m/s, yaw rad/s

/// Unicycle kinematics, one control period `dt` (zero-order hold).
pub fn step(p: &Pose, u: &Twist, dt: f64) -> Pose {
    Pose {
        x: p.x + u.v * p.theta.cos() * dt,
        y: p.y + u.v * p.theta.sin() * dt,
        theta: p.theta + u.omega * dt,
    }
}

pub trait Controller {
    fn act(&mut self, scan: &LidarScan) -> Twist;
}

/// The hand-coded baseline this chapter exists to break.
pub struct WallFollower { pub target_gap: f64, pub k_p: f64 }

impl Controller for WallFollower {
    fn act(&mut self, scan: &LidarScan) -> Twist {
        let gap = scan.min_range_deg(-95.0..=-85.0); // right-side beams
        Twist { v: 0.4, omega: self.k_p * (gap - self.target_gap) }
    }
}
```

Experiments: (1) noise × layout sweep (`rayon`, 100 seeds × 11 noise levels, seeded `StdRng`), success-cliff plot; (2) slip-probability ablation. Artifacts: `ch01-drive-rusty` runs natively (`cargo run -p ch01-drive-rusty`) and in-browser; the sweep runs natively and exports the plot the book page embeds as static fallback.

Definition of done for this chapter's code: workspace compiles native + `wasm32-unknown-unknown`, `clippy -D warnings` clean, both experiments reproduce byte-identical plots from pinned seeds.

## 6. Robot Thread

- **Rusty** — introduced. Before: does not exist. After: has a body (unicycle kinematics), a sensor (2-D lidar with noise model), a pen (`rapier2d` warehouse), a hand-coded controller with documented failure modes, and a reward config — everything except the ability to learn. Ch 4 formalizes his world; Ch 5–6 teach him.
- **Pendle, Reacher, Ferris** — trailed on cast cards only (Ch 2, Ch 3, Ch 15 respectively); no math or code yet.
- **Toolchain state** — after this chapter the reader's machine builds the entire workspace and one WASM demo; every later chapter assumes exactly this baseline and nothing more.

## 7. Exercises & Explorations

1. **(F)** For each of Akinola's four axes, give one easy/hard task pair *not* used in the chapter and justify the placement in two sentences each.
2. **(F)** Kober §1.3: RL applies when a task "can be phrased as an optimization problem and exhibits temporal structure." Give a robot task that is an optimization problem with *no* temporal structure, and name the problem class it belongs to (you have just previewed Ch 3).
3. **(F)** Write the see–think–act loop for a thermostat: identify $o_t$, $a_t$, $r_{t+1}$. Is exploration present? Should it be?
4. **(C)** In `ch01-drive-rusty`, find the smallest noise $\sigma$ that halves the wall-follower's success rate over 20 seeds; describe the recurring failure trace.
5. **(C)** In `ch01-success-levels`, find a competency with no entry above L3 and, using the formulation axis of `ch01-competency-explorer`, propose one reason grounded in Tang's analysis.
6. **(P)** Implement a second hand-coded controller (bang-bang gap keeper) and add it to the `rayon` sweep. Does it dominate the P-controller anywhere in the noise–slip plane?
7. **(P)** Add one 2024–2025 paper to `systems.json` with full taxonomy tags; defend the L-level assignment in a paragraph, citing the paper's experimental evidence as Tang §3.4 prescribes.
8. **(C/P)** Record a 60-second manual drive in the sandbox, then replay it under three reward configs (dock-only sparse; step-penalty dense; collision-heavy). Rank your own driving under each and explain why the ranking changes — you have just previewed Kober's curse of goal specification (Ch 14).

## 8. Notation Introduced

| Symbol | Meaning | Status |
|---|---|---|
| $t$ | discrete time step | informal here; formal clock fixed in Ch 4.1 |
| $o_t$, $s_t$ | observation / (unobserved) state at $t$ | distinction planted; formalized Ch 4.1/4.6 |
| $a_t$ | action at $t$ | formal spaces in Ch 4.1 |
| $r_{t+1}$ | reward following action $a_t$ (S&B indexing) | random-variable status deferred to Ch 2/4 |
| $\pi$ | policy, $\pi : \mathcal{O} \to \mathcal{A}$ (deterministic, informal) | stochastic $\pi(a\mid s)$ arrives Ch 3.5/4.3 |
| $D$ | demonstration set $\{(s,a)\}$ | reused in Ch 16 |
| return (unnamed) | accumulated reward over an episode | named $G_t$ and discounted in Ch 4.2 |
| L0–L5 | Tang's levels of real-world success | evaluation rubric, reused Ch 14/22 |

## 9. References & Further Reading

- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. (published-2018 numbering, per CLAUDE.md; the in-repo PDF is the in-progress draft, which agrees for Ch 1): Ch 1, esp. §1.1–1.3 (elements of RL), §1.5 (tic-tac-toe as first complete example).
- Kober, Bagnell & Peters, IJRR 2013: §1.1 (RL in the context of ML), §1.2 (RL vs optimal control), §1.3 (RL in the context of robotics — dimensionality, partial observability, cost of experience).
- Tang et al. 2024: §1 (motivation, real-world focus), §3.1 (competencies), §3.2 (problem formulation), §3.3 (solution approach), §3.4 (levels of real-world success L0–L5); §4.1–4.3 for tour exemplars.
- Akinola, *RL for Robotics* (Columbia lecture slides): see–think–act; direct-programming difficulty axes; LfD/RL/hybrid framing.
- Kaufmann, Bauersfeld, Loquercio, Müller, Koltun & Scaramuzza, "Champion-level drone racing using deep reinforcement learning," *Nature*, 2023.
- Lee, Hwangbo, Wellhausen, Koltun & Hutter, "Learning quadrupedal locomotion over challenging terrain," *Science Robotics*, 2020; Miki et al., "Learning robust perceptive locomotion for quadrupedal robots in the wild," *Science Robotics*, 2022.
- OpenAI (Akkaya et al.), "Solving Rubik's Cube with a robot hand," arXiv, 2019 — the in-hand line traced in Ch 20.
- Hwangbo et al., "Learning agile and dynamic motor skills for legged robots," *Science Robotics*, 2019 — the ANYmal recipe's origin, revisited in Ch 15/18.
- Sünderhauf et al., "The limits and potentials of deep learning for robotics," *IJRR*, 2018 — the pre-success-era assessment Tang §1 positions itself against.

# Chapter 22 — Capstone: An End-to-End Learned Robot in Rust

> **Part** V — Frontiers & Capstone · **Builds on:** Ch 10, 13, 14, 15, 16, 17, 18, 19, 20 (and every chapter's notation) · **Feeds:** the reader's own fork (book finale)
> **Modernizes:** Kober §7's case-study discipline (task & reward → representation → prior knowledge → learning → simulation → results) executed with Tang-era methods (§4.1 recipe, §3.4 L0–L5, §5 evaluation challenge).

## 1. Purpose & Learning Outcomes

One project, the whole book. Kober §7 taught that a robot-RL success is a *chain of defensible design decisions*, not an algorithm; this chapter makes that chain executable. Ferris must patrol a cluttered course: reach a sequence of goals over rough terrain, survive pushes, and recover from falls. Every decision — observation, action, reward, randomization, curriculum, distillation, evaluation — is made in writing, justified against the evidence chapters, implemented in the `capstone` crate, and shipped as a native trainer plus an in-browser demo with live telemetry.

The reader can:
- Write a complete formal problem statement for a real robot task, assembling notation from every prior chapter without contradiction.
- Justify each design decision against survey evidence (Ch 14–18) rather than habit, and record it in a machine-readable config.
- Build a randomized `rapier3d` training environment with curriculum, pushes, and reset logic, instrumented for telemetry.
- Run teacher–student PPO training at `rayon` scale, distill to a deployable recurrent student, and export it to WASM.
- Evaluate with the L0–L5 rubric and honest statistics, and state exactly which level the result earns (and why it is L0 by construction).
- Reproduce every number in the chapter from pinned seeds and configs in CI — and fork the pipeline for their own robot.

## 2. Storyline

**Act I — The specification is the project.** Hook: the reader has every tool; the capstone's first artifact is not code but a *specification* — the patrol task written as Kober §7.1 wrote ball-in-a-cup: task, reward, and what "success" will mean, before a line of training runs. The formal problem statement (22.2) is presented as the book's final exam in notation: one POMDP tuple in which $q$, $\tau$, $\hat v$, $\xi$, $e_t$, $b_t$, $w_k$ and friends all appear with their Ch 2–21 meanings intact.

**Act II — The chain of decisions, executed.** Each pipeline stage is built as a decision with named alternatives and cited evidence: the design ledger (22.3) chooses PD-position actions (Ch 17/18) over torques, a Ch 14-rubric reward over naive distance, randomization ranges from Ch 15's sysid experiments, teacher–student from Ch 18.6 over end-to-end exteroception; the environment (22.4) and training run (22.5) then execute the ledger at `rayon` scale with the mission-control dashboard as the reader's window. Failures are not edited out: the failure-mode gallery (22.7) documents the runs that collapsed and traces each to the curse (Ch 14) that caused it.

**Act III — Ship it, then hand over the keys.** Evaluation applies the book's own L0–L5 rubric to the book's own robot with Wilson intervals from Ch 20's harness — earning an honest L0 ("validated only in simulation, under diverse simulated conditions") and stating exactly what an L1/L2 claim would additionally require. The pipeline ships: native binary, WASM patrol demo with live telemetry, CI-reproducible seeds — and the epilogue reframes the whole repository as a template: swap the URDF, rewrite the spec, keep the discipline. The reader leaves with a working robot-learning pipeline and, more importantly, the habit of writing the spec first.

## 3. Section-by-Section Design

### 22.1 The Task: Ferris on Patrol (Kober §7.1 Discipline)
- **F:** Task specification in prose-then-math: a $20\,\mathrm m \times 20\,\mathrm m$ course of Ch 18 terrain tiles plus clutter; goal sequence $g_{1:K}$ ($K{=}4$ waypoints); external pushes (impulse perturbations) at random intervals; falls allowed but must be recovered (Ch 18.7 skill). Success criteria defined *before* training: all goals reached within $T_{\max}$, graded metrics (goals reached, time, falls, recovery time) declared as the evaluation contract.
- **C:** `ch22-course-tour` — visualization: a `bevy` flythrough of the course with goal markers, terrain-tile difficulty shading, and push-zone annotations; the spec rendered as a place.
- **P:** `capstone::spec`: the specification as code — `TaskSpec` struct (`serde`), course description, goal sampler, success predicate; the single source every later module imports (spec drift becomes a compile error).

### 22.2 The Formal Problem Statement — the Whole Book in One POMDP
- **F:** The assembled formalism, written in full as a worked example of specifying a real system: POMDP $\langle\mathcal S, \mathcal A, P, R, \Omega, O, \gamma\rangle$ with concrete spaces — $o_t \in \mathbb R^{48}$ itemized in a units table (gravity direction 3, base angular velocity 3, joint positions 12, joint velocities 12, previous action 12, goal vector in base frame 3, phase/timer 3); privileged $e_t$ (heightscan $11{\times}11$, friction, contact flags, push force, terrain type); $a_t \in \mathbb R^{12}$ PD targets ($q^\* = q_{\text{def}} + k_a a$, Ch 18.3); $R$ as the full Ch 18.4 term table with every $w_k, c_k$ printed; $\gamma$, horizon, termination and reset conditions; randomization $\xi \sim p(\xi)$ with ranges tabulated (mass $\pm 20\%$, friction $[0.3, 1.25]$, gains $\pm 10\%$, latency $0$–$20\,\mathrm{ms}$, push impulses); curriculum over course difficulty (Ch 18.5 machinery). Explicit cross-reference for each symbol to the chapter that introduced it — the notation audit as content.
- **C:** `ch22-problem-statement` — widget: the POMDP rendered as an interactive document; clicking any symbol opens its definition, source chapter, and the config line that implements it — spec, book, and code linked live.
- **P:** `capstone::pomdp`: observation builder, reward assembly (Ch 18's `CompositeReward` reused), randomization hooks (Ch 15's `rl-sim` API); `proptest` checks that the implemented spaces match the spec's declared dimensions and bounds.

### 22.3 The Design-Decision Ledger
- **F:** Every axis of Tang §3.2–3.3 decided with alternatives and evidence: action level (low, via PD — Ch 17 evidence table; alternatives costed), observation space (proprioception + goal for the student; exteroception rejected on scope, with the Miki-style upgrade path noted), reward (dense shaped, Ch 14 rubric-audited; sparse-goal alternative and its exploration cost), simulator usage (zero-shot from randomized sim — Tang §5's dominant recipe for locomotion), expert usage (none; teacher is an RL policy, not a human), algorithm (PPO — Ch 10/11's comparison protocol applied), architecture (teacher MLP + student GRU — Ch 19.2). Each row: decision, alternatives, evidence citation, config key, risk.
- **C:** `ch22-decision-ledger` — dashboard: the ledger as an interactive table; each row expands to the evidence (chapter back-links, survey citations) and — after training — the ablation result that tests it, closing the loop between claim and measurement.
- **P:** `capstone::config`: the ledger *is* the config schema — every decision a typed field with the rejected alternatives as enum variants, so ablations are config edits, not code forks (sketch in §5).

### 22.4 The Environment: a Randomized Course in `rapier3d`
- **F:** Environment-construction math consolidated: terrain tiling and difficulty parameterization (Ch 18.5), clutter placement as Poisson sampling with reachability check (A* over the heightfield from Ch 19.4's stack), push-impulse scheduling as a marked point process, reset logic (fall detection via base contact/orientation, recovery-or-teleport policy with reset-cost accounting per Ch 14).
- **C:** `ch22-randomization-draws` — gallery: a grid of sampled $(\xi, \text{course})$ draws rendered as thumbnails with parameter readouts; the reader resamples and filters by difficulty — the training distribution made inspectable.
- **P:** `capstone::env`: the full environment over `rl-envs::ferris` + `rl-envs::terrain`; deterministic per-seed generation (seed → course, verified by hash in tests); `rerun` recording hooks on every rollout worker.

### 22.5 Training: Teacher–Student PPO at `rayon` Scale
- **F:** The training plan as math: teacher PPO objective with privileged $e_t$ (all Ch 10 quantities — clipped surrogate, GAE($\lambda$), entropy bonus — instantiated with the run's actual hyperparameters, printed in a table, not an appendix); student distillation loss on on-policy student rollouts (DAgger discipline, Ch 16; encoder $z_t = \phi(o_{t-H:t})$, Ch 18.6); curriculum and randomization schedules as declared distributions; compute accounting (envs × steps × epochs) with the reproducibility target stated (single desktop GPU, ~1 hour teacher + ~20 min distillation).
- **C:** `ch22-mission-control` — the book's visual finale, part 1: live dashboard of returns, per-term rewards, KL, clip fraction, entropy (Ch 10's named quantities), curriculum histogram (Ch 18.5), randomization draws, gait telemetry (Ch 18.2 footfall strip), student-vs-teacher gap — every panel a quantity the math named.
- **P:** `capstone::train`: `rayon` rollout farm (4096 envs) driving `burn` PPO (WGPU); distillation stage; checkpointing with full config+seed embedding; `rerun` dev telemetry; `tracing` structured logs consumed by the dashboard.

### 22.6 Evaluation: the L0–L5 Rubric Applied to Ourselves
- **F:** The evaluation protocol as a pre-registered design: three condition tiers (nominal / randomized-in-range / out-of-range stress), 500 seeded episodes per tier, metrics from the 22.1 contract, Wilson CIs via Ch 20's harness; per-tier stratification and the test-leakage audit (evaluation seeds and courses disjoint from training by construction). The verdict argued honestly: the system earns **L0** — validated only in simulation — however diverse the simulated conditions; the section derives what evidence L1 and L2 would each require (hardware trials under limited/diverse lab conditions) and maps the survey's L3–L4 locomotion systems' extra ingredients (sysid on real logs, exteroception, on-robot safety layers) onto the pipeline's extension points. Baselines: flat-terrain-only policy and no-randomization policy as the ablation floor.
- **C:** `ch22-eval-heatmap` — mission control, part 2: success-rate heatmaps over (terrain difficulty × push magnitude) with CI-width overlays; tier comparison bars; every number clickable through to its episode replays.
- **P:** `capstone::eval`: batch evaluator importing `rl-core::eval` (Ch 20); JSON report artifact; the ablation matrix (ledger rows × metrics) that feeds `ch22-decision-ledger`.

### 22.7 Failure Modes and Post-Mortems
- **F:** Failure taxonomy mapped to Ch 14's curses: reward hacking instances (vibrating stance farming the air-time term — with the term-algebra fix), curriculum collapse (all envs demoted; scheduler bug post-mortem), sim-only artifacts (contact chatter exploited for propulsion; solver-parameter sensitivity from Ch 15), distillation gaps (student blind to a terrain class the heightscan saw), and honest unsolved cases (recovery on steep slopes). Each: symptom, telemetry signature, diagnosis, fix or open status.
- **C:** `ch22-failure-gallery` — gallery: curated failure clips with synchronized telemetry strips and the post-mortem text; a "spot the signature" mode shows telemetry first and lets the reader predict the failure before the clip plays.
- **P:** Automatic failure-clip capture: episodes violating invariants (torque saturation streaks, reward-term outliers, termination cascades) are flagged in the rollout farm and archived as `rerun` recordings with configs attached.

### 22.8 Shipping It: Native, WASM, Telemetry, Seeds
- **F:** Reproducibility contract stated formally: every artifact = function(config, seed, crate versions); seed-forking discipline (one master seed → named RNG streams for env/init/eval); determinism boundaries documented (`rapier` determinism settings, `burn` backend caveats — where bitwise reproducibility ends and statistical reproducibility begins, per the Ch 20.7 statistics).
- **C:** `ch22-patrol-demo` — the shipped artifact: in-browser WASM demo of the distilled student patrolling a reader-configurable course, with a live mini-telemetry panel (velocity tracking, footfalls, per-term rewards) — the book's opening promise ("the same code compiles natively for training and to WASM") kept at full scale.
- **P:** `capstone` crate finalized: `cargo run --release -- train|distill|eval|export` CLI (`clap`), config-driven runner (sketch in §5), CI workflow (seeded smoke-train, evaluation-report diff against pinned numbers with statistical tolerance), WASM export via the `xtask` pipeline.

### 22.9 Chapter Bridge (Epilogue)
- Recap of the chain: spec → ledger → environment → training → evaluation → post-mortem → shipping, each stage owned by a chapter of the book. What the reader does next: the Ch 21 research compass for directions, this repository as the vehicle — swap the URDF (Appendix A/B tooling), rewrite `TaskSpec`, keep the discipline. Final scene: all four robots on screen — Pendle's swing-up, Rusty's patrol, Reacher's insertion, Ferris walking — each running the reader's own compiled artifacts.

## 4. Interactive Widgets & Dashboards

| ID | Type | What it shows | Reader interaction | Tech |
|---|---|---|---|---|
| `ch22-course-tour` | visualization | Flythrough of the patrol course; difficulty and push-zone annotations | Orbit/fly camera; toggle annotation layers | bevy, WASM |
| `ch22-problem-statement` | widget | The full POMDP with every symbol linked to definition, chapter, config line | Click symbols; jump to source chapters | egui, WASM |
| `ch22-decision-ledger` | dashboard | Design decisions with alternatives, evidence, and post-hoc ablation results | Expand rows; sort by risk; open ablation plots | egui + serde data |
| `ch22-randomization-draws` | gallery | Sampled courses and $\xi$ draws as inspectable thumbnails | Resample; filter by difficulty/parameter | egui + bevy renders |
| `ch22-mission-control` | dashboard | Training finale: returns, per-term rewards, KL/clip/entropy, curriculum, gait telemetry, student gap | Scrub runs; compare checkpoints; live-attach to native training | egui_plot (+ tracing feed) |
| `ch22-eval-heatmap` | dashboard | Success heatmaps over difficulty × push with CI overlays; tier bars | Click cells for episode replays; toggle baselines | egui_plot |
| `ch22-failure-gallery` | gallery | Failure clips + telemetry signatures + post-mortems | Browse; "spot the signature" prediction mode | bevy replays + egui |
| `ch22-patrol-demo` | sandbox | The shipped robot: distilled student patrolling reader-configured courses | Edit course/goals/pushes; drive difficulty; watch live telemetry | bevy + rapier3d + burn, WASM |

## 5. Rust Implementation Plan

**Crates touched:** new top-level `capstone` crate (also the template repository); imports `rl-core` (reward, gait, eval), `rl-envs` (ferris, terrain), `rl-sim` (randomization), `rl-deep` (PPO, distill); `xtask` grows `build-capstone` and `ci-repro` commands.

**Modules/files:** `capstone/src/{spec.rs, config.rs, pomdp.rs, env.rs, train.rs, distill.rs, eval.rs, export.rs, main.rs}`, `capstone/configs/{patrol-v1.toml, ablations/*.toml}`, `.github/workflows/capstone-repro.yml`.

Representative sketch — the config-driven runner, the ledger made executable (`capstone/src/main.rs`):

```rust
#[derive(Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub seed: u64,                      // master seed; all streams fork from it
    pub spec: TaskSpec,                 // Sec. 22.1: course, goals, success contract
    pub env: EnvConfig,                 // Sec. 22.4: terrain, clutter, pushes, resets
    pub randomization: RandomizationRanges, // Ch 15 hooks; printed in Sec. 22.2 table
    pub reward: Vec<RewardTermConfig>,  // Ch 18 CompositeReward, weights explicit
    pub teacher: PpoConfig,             // Ch 10 hyperparameters, named and tabled
    pub student: DistillConfig,         // Ch 18.6/16: encoder, history H, DAgger iters
    pub eval: EvalConfig,               // Sec. 22.6: tiers, trials, CI level
}

#[derive(clap::Parser)]
struct Cli {
    #[arg(long)] config: PathBuf,
    #[command(subcommand)] cmd: Cmd,
}
#[derive(clap::Subcommand)]
enum Cmd { Train, Distill, Eval, Export }

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let cfg: ExperimentConfig = toml::from_str(&fs::read_to_string(&cli.config)?)?;
    let streams = RngStreams::fork(cfg.seed); // named streams: env, init, eval
    match cli.cmd {
        Cmd::Train => train_teacher(&cfg, &streams)?,   // rayon farm + burn PPO
        Cmd::Distill => distill_student(&cfg, &streams)?,
        Cmd::Eval => evaluate(&cfg, &streams)?,          // writes eval/report.json
        Cmd::Export => export_wasm_policy(&cfg)?,        // student -> ch22-patrol-demo
    }
    Ok(())
}
```

**Experiments/benchmarks:** (1) the headline run (teacher + distillation + three-tier evaluation, pinned seeds); (2) the ledger ablation matrix (no-randomization, no-curriculum, torque actions, MLP student, sparse reward) — each a one-line config change; (3) throughput benchmarks (`criterion`: envs/sec vs threads, policy inference latency native vs WASM); (4) CI reproduction: smoke-scale run whose evaluation report must match pinned numbers within declared statistical tolerance.

**Native vs browser:** training, evaluation, `rerun` telemetry, and CI native; `ch22-patrol-demo`, `ch22-problem-statement`, `ch22-course-tour`, and all report dashboards in-browser; `ch22-mission-control` runs in-browser on logged data and can live-attach to a native training run via a local socket feed.

## 6. Robot Thread

- **Ferris** (capstone): from Ch 15 randomization testbed and Ch 18 walker to a specified, trained, distilled, evaluated, and shipped patrol robot — the thread's terminus.
- **Rusty, Reacher, Pendle** (curtain call): appear in the epilogue montage running their own chapters' artifacts; Rusty's Ch 19 navigation stack additionally donates the course reachability check in 22.4.

## 7. Exercises & Explorations

1. **(F)** Audit the 22.2 problem statement against Appendix C: list every symbol whose meaning changed since its introducing chapter (there should be none) and every deliberate reuse (there are several) — then do the same audit on a published quadruped paper of your choice.
2. **(F)** The spec fixes $\gamma = 0.99$ at 50 Hz. Derive the effective horizon in seconds, argue whether it suffices for the $K{=}4$ patrol, and compute what changing the control rate to 100 Hz would require of $\gamma$ to preserve it.
3. **(F)** Write the pre-registered evaluation protocol for an imagined L1 claim (real Ferris hardware, one lab room): conditions, trial counts from Ch 20.7's formulas for ±5% at 95%, and stopping rules.
4. **(C)** In `ch22-decision-ledger`, pick the highest-risk decision, open its ablation, and write the paragraph the chapter would need if the ablation had gone the other way.
5. **(C)** Use `ch22-failure-gallery`'s prediction mode on all clips; for the one you diagnose wrong, trace the telemetry signature you missed back to the dashboard panel that shows it.
6. **(P)** Run the full pinned-seed pipeline; then change only the master seed and quantify which reported numbers move outside their CIs (none should) — the reproducibility contract, tested.
7. **(P)** Add a fifth waypoint on a stepping-stone tile (the known Ch 18 failure); extend the curriculum to rescue it, or document with telemetry why it stays unsolved.
8. **(P)** The fork: replace Ferris's URDF with another quadruped (or your own robot), rewrite `TaskSpec` for a task you care about, keep the ledger discipline, and produce your own `eval/report.json` — the book's actual final exam.

## 8. Notation Introduced

| Symbol | Meaning |
|---|---|
| $g_{1:K}, T_{\max}$ | patrol goal sequence; time budget (the success contract) |
| $o_t \in \mathbb R^{48}$ | capstone observation vector (itemized with units in 22.2) |
| — | *No other new symbols by design*: the chapter is an assembly of Ch 2–21 notation; the 22.2 table cross-references each symbol's introducing chapter, and the audit exercise (7.1) enforces it |

## 9. References & Further Reading

- **Baseline:** Kober §7.1–7.6 (the case-study discipline this chapter transplants: task & reward, representation, prior knowledge, policy search, simulation use, real results); Tang §3.4 (L0–L5 as the claim's scope), §4.1.1 (the recipe evidence), §5 (benchmarking real-world success — the gap this chapter's protocol answers at L0 scale).
- Rudin et al. 2021, massively parallel learning-to-walk (the compute template for 22.5).
- Lee et al. 2020 / Miki et al. 2022, teacher–student ANYmal systems (the L3–L4 systems whose extra ingredients 22.6 maps as upgrade paths).
- Kumar et al. 2021, *RMA* (the student-adaptation alternative registered in the ledger).
- Agarwal et al. 2021, *Deep RL at the edge of the statistical precipice* (statistical evaluation practice behind 22.6).
- Henderson et al. 2018, *Deep RL that matters* (reproducibility failures the seed discipline of 22.8 answers).
- Ma et al. 2024, *DrEureka* (what automating this chapter's ledger might look like — the Ch 21 frontier pointed back at the capstone).
- Kober & Peters 2011, policy search for motor primitives (the ball-in-a-cup result whose write-up discipline this chapter imitates at modern scale).

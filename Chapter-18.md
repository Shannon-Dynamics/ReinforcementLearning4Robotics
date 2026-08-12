# Chapter 18 — Learning Locomotion

> **Part** IV — Competencies: RL on Real Robots · **Builds on:** Ch 10, 13, 14, 15, 17 · **Feeds:** Ch 19, 22
> **Modernizes:** Tang §4.1 (4.1.1 quadruped, 4.1.2 biped, 4.1.3 quadrotor flight, 4.1.4 trends); Kober's pre-deep locomotion lineage (early quadruped gait RL, helicopter control) as historical contrast.

## 1. Purpose & Learning Outcomes

Locomotion is where deep RL for robotics grew up: quadruped controllers trained in simulation now ship on commercial machines (Tang's L4–L5 evidence: ANYbotics, Swiss-Mile, Boston Dynamics integrations). This chapter dissects *why* this competency matured first and turns the modern quadruped recipe into working Rust code: Ferris, introduced in Ch 15, learns to walk end-to-end.

The reader can:
- Explain, with Tang §4.1's evidence, why locomotion reached L4–L5 before manipulation (stable dynamics, simulable contact, dense rewardable gaits).
- Formalize gaits as phase variables on a torus: duty factors, phase offsets, footfall diagrams, and the Froude-number scaling that predicts gait transitions.
- Write and dimensionally analyze a composite locomotion reward, predicting the qualitative gait each term produces.
- Formalize a terrain curriculum as scheduled task distributions $p_k(\xi)$ and implement the adaptive promotion rule.
- Reproduce the ANYmal-style teacher–student pipeline and the RMA adaptation-module variant, and say when each applies.
- Train a 12-DoF quadruped with PPO over `rayon`-parallel `rapier3d` environments and evaluate the learned gait in a browser demo.

## 2. Storyline

**Act I — Why did the legs win?** Hook: the same PPO the reader built in Ch 10, with the sim-to-real kit of Ch 15, produces controllers that walk on ice, gravel, and stairs — while pick-and-place is still stuck below L3 (Ch 20 will ask why). Tang §4.1's answer, examined critically: quadruped dynamics are inherently stable, contact with terrain is simulable, and "walk at commanded velocity" admits a dense reward. Ferris steps forward as the chapter's protagonist: he stood passively in Ch 15's randomization experiments; here he must *move*.

**Act II — The recipe, one ingredient at a time.** The modern quadruped recipe is assembled ingredient by ingredient, each with its formalism and its widget: gait/phase language for reading behavior (18.2), observation/action design on the PD substrate (18.3), reward anatomy with dimensional analysis (18.4), terrain curricula as distribution scheduling (18.5), and privileged teacher → deployable student (18.6). Each ingredient is ablated on Ferris so the reader sees what breaks without it.

**Act III — Payoff and the dynamics-difficulty axis.** Ferris walks: the full training run, gait emergence filmed in `bevy`, and the WASM demo where the reader draws terrain and watches him cross it. Then the honest widening of the lens: bipeds (underactuation, falling is expensive, reference-guided rewards) and quadrotors (agile flight, drone racing at L5 — Kaufmann 2023) show the recipe straining as dynamics get less forgiving, setting up navigation (Ch 19) where these skills become building blocks.

## 3. Section-by-Section Design

### 18.1 Locomotion as Deep RL's Flagship — the Evidence
- **F:** Locomotion as an MDP family: state (base pose/twist + joint state + terrain), velocity-command-conditioned policies $\pi(a\mid o, \hat v)$; instantiation of Tang's formulation axes (low-level actions, proprioceptive + optional exteroceptive observations, dense reward). Precise restatement of the L0–L5 rubric from Ch 1/14 as an evaluation function over test-condition sets, applied to the surveyed systems.
- **C:** `ch18-l-level-tracker` — dashboard of ~20 published locomotion systems (quadruped/biped/quadrotor) plotted on L0–L5 vs year, filterable by technique (privileged learning, curriculum, real-world RL); hovering shows the recipe delta each system introduced.
- **P:** `rl-envs::ferris` skeleton: 12-DoF quadruped assembled in `rapier3d` (multibody joints from a small URDF via `urdf-rs`), standing under gravity with default PD holds; smoke tests for determinism with fixed seeds.

### 18.2 Gaits: Phase, Duty Factor, and Footfall Formalism
- **F:** Per-leg phase $\phi_i \in [0,1) \cong S^1$, stride frequency $\dot\phi_i = f$; a gait = phase-offset vector $\Delta = (\Delta_1,\dots,\Delta_4)$ plus duty factor $\beta$ (stance iff $\mathrm{frac}(\phi_i) < \beta$): walk $(0,\tfrac12,\tfrac14,\tfrac34)$, trot $(0,\tfrac12,\tfrac12,0)$, pace, bound, pronk as points on the torus $T^4$. Support polygon and static-stability margin; dynamic gaits as its deliberate violation; Froude number $Fr = v^2/(gL)$ derived by nondimensionalizing the pendular leg — predicting walk→trot transitions and previewing 18.4's dimensional analysis. Link back to Ch 17's CPGs: a CPG is a prior over this torus.
- **C:** `ch18-gait-phases` — animation: a footfall (Hildebrand) diagram synchronized with a trotting quadruped silhouette; the reader drags $\Delta$ and $\beta$ sliders and watches the gait morph between walk/trot/bound, with the stability margin plotted live.
- **P:** `rl-core::gait`: phase-vector type, footfall extraction from contact histories (`Vec<ContactEvent> -> GaitDiagram`), gait-classification utility used by every later experiment's telemetry.

### 18.3 The Recipe I — Observations, Actions, and the PD Substrate
- **F:** The canonical observation vector: gravity direction in base frame, base angular velocity, joint positions/velocities, previous action, command $\hat v = (\hat v_x, \hat v_y, \hat\omega_z)$ — and why base *linear* velocity is excluded on real hardware (unobservable without slippage-prone estimation; POMDP honesty from Ch 4). Action = PD position targets: $q^\* = q_{\text{default}} + k_a a$, $\tau = K_p(q^\* - q) - K_d \dot q$, policy at 50 Hz over PD at 200 Hz; formal argument (Ch 17 payoff) why position+PD beats torque actions: the PD loop is a stabilizing prior that shapes the exploration distribution and bounds torque under randomized dynamics.
- **C:** `ch18-action-substrate` — widget: same pretrained gait replayed through three action spaces (torque, position+PD, CPG-parameter) with learning curves and a "perturb mid-swing" button showing PD's disturbance rejection.
- **P:** `rl-envs::ferris`: PD inner loop (`nalgebra`), configurable $K_p, K_d$, action scaling $k_a$, control-rate decimation; observation builder with explicit unit documentation per entry.

### 18.4 The Recipe II — Reward Anatomy with Dimensional Analysis
- **F:** Reward-term algebra: $r_t = \sum_k w_k\, r_k / c_k$ with raw terms in physical units and scales $c_k$ chosen so each summand is dimensionless and $O(1)$: velocity tracking $\exp(-\lVert \hat v_{xy} - v_{xy}\rVert^2/\sigma_v^2)$ (dimensionless by construction, $[\sigma_v] = \mathrm{m/s}$); effort $-\lVert\tau\rVert^2$ ($[c] = \mathrm{N^2 m^2}$); action rate $-\lVert a_t - a_{t-1}\rVert^2$; feet air-time $\sum_i (T_{\text{air},i} - T^\*)$ (the Rudin trick, $[c] = \mathrm{s}$); orientation and height penalties. Theorem-level point: rescaling units silently rescales weights, so publishing $w_k$ without $c_k$ is meaningless — dimensional analysis as reproducibility discipline (Ch 14 rubric extended). Each term mapped to the failure mode it suppresses and the gait caricature it produces when overweighted.
- **C:** `ch18-reward-mixer` — the chapter's signature widget: one slider per reward term; a bank of pre-trained policies on a coarse weight grid lets the mixer show the *nearest trained gait caricature* (foot-dragging shuffler, pronking energy-waster, statue) plus its footfall diagram from 18.2, within a second of any slider move.
- **P:** `rl-core::reward`: `RewardTerm` trait with `raw()` and `scale()`, `CompositeReward` with per-term telemetry (code sketch in §5); the weight-grid training script that generates the mixer's policy bank.

### 18.5 The Recipe III — Terrain Curricula as Distribution Scheduling
- **F:** Task parameter $\xi \in \Xi$ (slope, step height, roughness amplitude, friction); a curriculum is a schedule of distributions $p_k(\xi)$ over training phase $k$. The adaptive (game-inspired) rule formalized as a per-environment Markov chain on difficulty levels: promote when episode tracking reward exceeds $\rho_{\uparrow}$, demote below $\rho_{\downarrow}$; contrast with static randomization (Ch 15's minimax view) and derive why curricula fix the flat-gradient cold-start: $\nabla J$ under hard $p(\xi)$ is near zero when success probability is near zero.
- **C:** `ch18-terrain-curriculum` — dashboard: live histogram of environment instances over difficulty levels during training, terrain-tile mosaic colored by current success rate, replayable "curriculum movie" comparing adaptive vs fixed-hard vs fixed-easy schedules.
- **P:** `rl-envs::terrain`: procedural heightfield generator (pyramid slopes, stairs, steps, Perlin roughness) parameterized by $\xi$; curriculum state machine; `rapier3d` heightfield colliders regenerated per reset.

### 18.6 The Recipe IV — Privileged Teachers, Adapting Students
- **F:** The ANYmal recipe formalized: teacher $\pi_T(a \mid o_t, e_t)$ with privileged extrinsics $e_t$ (terrain heightscan, friction, contact states, applied pushes) trained by PPO; student $\pi_S(a \mid o_{t-H:t})$ with encoder $z_t = \phi(o_{t-H:t})$ trained by distillation: $\min_\phi \mathbb{E}[\lVert z_t - \psi(e_t)\rVert^2 + \lVert \pi_S - \pi_T\rVert^2]$ on on-policy student rollouts (DAgger discipline, Ch 16 crossref). RMA as the same diagram with the adaptation module regressing the extrinsics latent online; asymmetric actor–critic (Ch 15) as the degenerate case where only the critic is privileged. When each variant applies, per Tang §4.1.1's evidence.
- **C:** `ch18-teacher-student-flow` — animation extending Ch 15's information-flow diagram: privileged bits visibly "squeezed" through the history encoder; a toggle lesions the encoder and shows the student stumbling on the terrain the teacher handles.
- **P:** `rl-deep::distill`: teacher PPO with privileged observations, student GRU/TCN encoder in `burn`, on-policy distillation loop; ablation experiment (teacher-only vs student vs blind student) on the 18.5 terrain suite.

### 18.7 Harder Dynamics: Bipeds, Quadrotors, and Recovery
- **F:** The dynamics-difficulty axis (Tang §4.1.4): degree of underactuation and cost of failure. Bipeds: non-static stability, reference-guided reward (imitation or periodic-composition rewards à la Siekmann) as priors replacing the stability the morphology lacks. Quadrotors: rigid-body flight dynamics, the collective-thrust-and-body-rate action space and why it transfers better than motor commands; drone racing as L5 evidence that RL beats optimal control by optimizing the *long-horizon task objective* directly (Kaufmann 2023). Recovery behaviors as a separate skill and the auto-reset prerequisite for real-world RL (Ch 14's reset curse).
- **C:** `ch18-fragility-axis` — gallery: quadruped/biped/quadrotor failure compilations arranged along the underactuation axis, each with the recipe modification that compensates; includes Ferris fall-recovery clips.
- **P:** Fall-recovery task variant for Ferris (initialize supine, reward uprightness then stand); trained with the same PPO stack, composed with walking via a hand-coded switcher — the naive skill composition Ch 19 will formalize and improve.

### 18.8 Case Study: Ferris Learns to Walk
- **F:** The assembled MDP written in full (the Ch 22 problem statement's dress rehearsal): observation table with units, action space, all reward terms with $w_k, c_k$, curriculum schedule, randomization ranges (Ch 15), episode/termination conditions. Training-run analysis: which metrics (tracking error, per-term rewards, gait diagram entropy) predict the gait transitions observed.
- **C:** `ch18-walk-demo` — sandbox (WASM): the reader draws a terrain profile, the trained student policy walks Ferris across it in-browser (`rapier3d` + `bevy` compiled to WASM), with live footfall diagram and per-term reward strip charts.
- **P:** The full experiment: PPO (`burn`, WGPU backend) over 4096 `rayon`-parallel Ferris instances, ~30 min native training on a desktop GPU as the reproducibility target; `bevy` gait visualization; checkpoint export to the WASM demo.

### 18.9 Chapter Bridge
- Recap: the recipe is five decisions — substrate, observations, reward algebra, curriculum, distillation — and every one traces to a curse from Ch 14. Locomotion solved "move the body"; Ch 19 asks "move the body *somewhere*": navigation, hierarchy, and Rusty's graduation; the locomotion policy built here returns as a low-level skill (legged navigation) and as the capstone's engine (Ch 22).

## 4. Interactive Widgets & Dashboards

| ID | Type | What it shows | Reader interaction | Tech |
|---|---|---|---|---|
| `ch18-l-level-tracker` | dashboard | Published locomotion systems on L0–L5 vs year | Filter by morphology/technique; hover for recipe deltas | egui + egui_plot |
| `ch18-gait-phases` | animation | Footfall (Hildebrand) diagram synced to walking quadruped | Drag phase offsets $\Delta$ and duty factor $\beta$; watch gait morph, stability margin live | egui, WASM |
| `ch18-action-substrate` | widget | Torque vs position+PD vs CPG action spaces | Toggle substrate; perturb mid-swing; compare learning curves | egui + rapier2d slice |
| `ch18-reward-mixer` | widget | Reward-term sliders → nearest trained gait caricature + footfall diagram | Drag per-term weights; inspect per-term reward traces | egui + policy bank (burn WASM inference) |
| `ch18-terrain-curriculum` | dashboard | Difficulty-level histogram, success mosaic, schedule comparison | Scrub training time; switch adaptive/fixed schedules | egui_plot |
| `ch18-teacher-student-flow` | animation | Privileged information squeezed into student's history encoder | Lesion encoder; toggle RMA vs distillation variant | egui/SVG animation |
| `ch18-fragility-axis` | gallery | Failure modes along the underactuation axis; recovery clips | Browse clips; reveal per-system recipe fix | bevy replays |
| `ch18-walk-demo` | sandbox | Trained Ferris crossing reader-drawn terrain | Draw height profile; drive velocity commands; watch live gait telemetry | bevy + rapier3d + burn, WASM |

## 5. Rust Implementation Plan

**Crates touched:** `rl-envs` (new `ferris`, `terrain` modules), `rl-core` (`gait`, `reward` modules), `rl-deep` (`distill` module; PPO from Ch 10 reused unchanged), `rl-sim` (randomization hooks from Ch 15), demos `ch18-*`.

**Modules/files:** `rl-envs/src/ferris/{model.rs, pd.rs, obs.rs, env.rs}`, `rl-envs/src/terrain.rs`, `rl-core/src/gait.rs`, `rl-core/src/reward.rs`, `rl-deep/src/distill.rs`, `demos/ch18-walk-demo/`.

Representative sketch — composable reward with per-term telemetry (`rl-core/src/reward.rs`):

```rust
/// One named term of a composite locomotion reward (Ch 14 rubric discipline).
pub trait RewardTerm: Send + Sync {
    fn name(&self) -> &'static str;
    /// Raw value in the term's natural physical units (documented per impl).
    fn raw(&self, s: &FerrisState, a: &Action, prev_a: &Action) -> f64;
    /// Scale c_k that non-dimensionalizes `raw`, so weights are unitless.
    fn scale(&self) -> f64;
}

pub struct CompositeReward {
    terms: Vec<(Box<dyn RewardTerm>, f64)>, // (term, dimensionless weight w_k)
}

impl CompositeReward {
    /// Total reward plus per-term breakdown, consumed by training telemetry
    /// and the ch18-reward-mixer widget.
    pub fn eval(&self, s: &FerrisState, a: &Action, prev_a: &Action) -> (f64, Vec<TermLog>) {
        let mut total = 0.0;
        let mut logs = Vec::with_capacity(self.terms.len());
        for (term, w) in &self.terms {
            let value = w * term.raw(s, a, prev_a) / term.scale();
            total += value;
            logs.push(TermLog { name: term.name(), value });
        }
        (total, logs)
    }
}
```

**Experiments/benchmarks:** (1) action-substrate ablation (torque vs PD vs CPG) on flat ground; (2) reward-term ablations generating the mixer's policy bank; (3) curriculum vs fixed randomization; (4) teacher vs student vs blind student on the terrain suite; (5) `criterion` benchmark of `rayon` rollout throughput (envs/sec vs thread count).

**Native vs browser:** full training native (WGPU); `ch18-walk-demo`, `ch18-gait-phases`, `ch18-reward-mixer` run in-browser (inference-only `burn` + `rapier3d` WASM). Training in-browser is documented as possible-but-slow; not shipped.

## 6. Robot Thread

- **Ferris** (flagship): entered Ch 15 as a standing randomization testbed; leaves Ch 18 walking over procedural terrain at commanded velocities, with fall recovery and a distilled deployable student policy — the skill Ch 19 composes and Ch 22 hardens.
- **Pendle** (cameo): the Froude-number derivation reuses Pendle's pendulum scaling from Ch 2/13.
- Rusty and Reacher rest this chapter; Rusty returns in Ch 19.

## 7. Exercises & Explorations

1. **(F)** Derive the Froude number by nondimensionalizing a pendular leg of length $L$; predict Ferris's walk→trot transition speed and verify against the trained policy's gait diagram.
2. **(F)** Show that scaling torque units from N·m to mN·m rescales the effort weight by $10^6$; restate the 18.4 reward table so all published weights are dimensionless, and check two published quadruped papers for this discipline.
3. **(F)** Prove that potential-based terms (Ch 14) added to the 18.4 reward leave the optimal policy unchanged, and exhibit one *non*-potential term from the mixer that does change it.
4. **(C)** Using `ch18-reward-mixer`, find two distinct weight settings that both produce trotting; compare their per-term traces and explain which is preferable for hardware and why.
5. **(C)** In `ch18-terrain-curriculum`, find a fixed randomization range where training never takes off, then show the adaptive schedule solving the same final distribution; relate to the flat-gradient argument of 18.5.
6. **(P)** Add a "lateral-slip penalty" `RewardTerm` with correct units and scale; retrain and document the gait change with a footfall diagram.
7. **(P)** Implement a pronk-only curriculum via phase-offset reward shaping and measure its robustness to pushes vs the trot policy, using the 18.7 recovery task as tiebreaker.
8. **(P)** Extend the terrain generator with stepping stones (sparse contacts) and reproduce the failure Tang §4.1.1 attributes to end-to-end policies; document what curriculum change (if any) rescues it.

## 8. Notation Introduced

| Symbol | Meaning |
|---|---|
| $\phi_i \in [0,1)$ | phase of leg $i$; gait = point on torus $T^4$ |
| $\Delta_i$ | phase offset of leg $i$ relative to leg 1 |
| $\beta$ | duty factor (stance fraction); *clash note:* $\beta_\omega$ remains option termination (Ch 17), context disambiguates |
| $Fr = v^2/(gL)$ | Froude number; gait-transition scaling |
| $\hat v = (\hat v_x, \hat v_y, \hat\omega_z)$ | commanded base velocity |
| $w_k, c_k$ | dimensionless reward weight and dimensional scale of term $k$ |
| $\sigma_v$ | tracking-kernel width (m/s) |
| $q^\*, K_p, K_d, k_a$ | PD target, gains, action scale |
| $\xi \in \Xi,\ p_k(\xi)$ | terrain/task parameter and curriculum distribution at phase $k$ |
| $e_t, z_t$ | privileged extrinsics; student's inferred extrinsics latent |
| $T_{\text{air},i}$ | swing (air) time of foot $i$ |

## 9. References & Further Reading

- **Baseline:** Tang §4.1.1–4.1.4 (the chapter's spine and evidence tables); Tang §3.2/3.4 (formulation axes, L0–L5); Kober §3.4 (reward shaping), §4 (representations — CPG/DMP lineage), early gait RL cited therein (Kimura et al. 2001).
- Hwangbo et al. 2019, *Learning agile and dynamic motor skills for legged robots* (actuator nets, first ANYmal recipe).
- Lee et al. 2020, *Learning quadrupedal locomotion over challenging terrain* (privileged teacher–student).
- Miki et al. 2022, *Learning robust perceptive locomotion* (exteroception, the wild-ANYmal L4 system).
- Kumar et al. 2021, *RMA: Rapid Motor Adaptation* (adaptation module); Zhang et al. 2023 (RMA for quadrotors, per Tang §4.1.3).
- Rudin et al. 2021, *Learning to walk in minutes using massively parallel deep RL* (parallel-sim recipe, air-time reward, game-inspired curriculum).
- Margolis & Agrawal 2022, *Walk These Ways* (gait-conditioned MoB policies); Fu et al. 2021 (energy minimization → gait emergence).
- Siekmann et al. 2021, *Sim-to-real learning of all common bipedal gaits via periodic reward composition*; Radosavovic et al. 2024 (humanoid locomotion with transformers).
- Kaufmann et al. 2023, *Champion-level drone racing using deep RL* (L5 flight); Eschmann et al. 2024 (18-second off-policy quadrotor training).
- Cheng et al. 2024 / Zhuang et al. 2023 (parkour: unified rewards, multi-skill distillation).

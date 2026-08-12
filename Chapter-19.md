# Chapter 19 — Learning Navigation & Mobile Manipulation

> **Part** IV — Competencies: RL on Real Robots · **Builds on:** Ch 4, 9, 10, 13, 15, 17, 18 · **Feeds:** Ch 20, 21, 22
> **Modernizes:** Tang §4.2 (4.2.1 wheeled, 4.2.2 legged, 4.2.3 aerial navigation, 4.2.4 trends) and §4.4 (4.4.1 whole-body control, 4.4.2 short-horizon, 4.4.3 long-horizon MoMa, 4.4.4 trends); Kober's perception-integration open question (§8.1) made concrete.

## 1. Purpose & Learning Outcomes

Locomotion moved the body; navigation decides *where* the body goes — the perception-heavy competency where partial observability is the whole problem, and where Tang's evidence says neither end-to-end learning nor the classical modular stack wins universally. This chapter graduates Rusty from the Ch 4 gridworld to lidar-based continuous navigation, formalizes recurrent policies as belief trackers, composes skills hierarchically, and ends with mobile manipulation: a Reacher arm bolted onto Rusty's base.

The reader can:
- Write the belief-MDP reduction of a navigation POMDP and identify what a Bayes filter, an occupancy map, and a GRU hidden state have in common.
- State precisely why a recurrent policy trained on returns approximates a belief-state tracker, and probe a trained GRU to verify it.
- Argue the end-to-end vs hybrid-modular trade-off from Tang §4.2's evidence, and reproduce both architectures on the same scenes.
- Formalize skill composition with semi-MDP options (Ch 17) and train a two-level policy for a long-horizon fetch task.
- Derive weighted-pseudoinverse redundancy resolution for a base+arm system and contrast it with learned whole-body control.
- Train recurrent PPO with a GRU policy in `burn` and evaluate it against a classical planner baseline.

## 2. Storyline

**Act I — Rusty can't see.** Hook: drop the Ch 4 warehouse's God's-eye grid. Rusty now has a 2D lidar (64 beams, noisy, limited range), no map, and a goal vector. The Ch 9 DQN that mastered the visual gridworld thrashes: identical scans at different corridors alias state. The POMDP honesty promised in Ch 4 is now unavoidable — and the belief-MDP formalism turns "Rusty is confused" into mathematics.

**Act II — Memory, modules, and the architecture war.** Three responses to partial observability, each built and compared: carry a belief (recurrent policies as learned filters, 19.2–19.3), keep the classical stack and learn only inside it (modular hybrids, 19.4), or restructure the problem hierarchically (skills and semi-MDPs, 19.6). Tang §4.2's verdict — end-to-end shines in simulation and dense clutter, modular ships in the real world — is reproduced live in the pipeline-switcher widget, then stress-tested on legged/aerial embodiments where locomotion (Ch 18) refuses to stay abstracted away (19.5).

**Act III — Put an arm on it.** Mobile manipulation as the synthesis competency: whole-body control couples base and arm through shared kinematic redundancy (Ch 13 payoff), short-horizon skills chain into a long-horizon fetch, and the "what skills should a robot learn?" open question (Tang §5) is posed honestly. Rusty + Reacher fetch an object behind a door: navigation, docking, reaching, and retreat composed by a learned selector — the book's first robot that both moves and manipulates, rehearsing Ch 22's integration discipline.

## 3. Section-by-Section Design

### 19.1 Navigation Is a POMDP: the Belief-MDP Formalism
- **F:** Navigation POMDP instantiated: state = pose + map, observation = lidar scan + odometry, goal-conditioned reward. Belief update derived in full: $b_{t+1}(s') = \eta\, O(o_{t+1}\mid s', a_t) \sum_s P(s'\mid s, a_t)\, b_t(s)$; the belief MDP $(\mathcal B, \mathcal A, \tau, \rho)$ with $\rho(b,a) = \sum_s b(s) R(s,a)$; restatement (from Ch 4) that finite-horizon $v^\*(b)$ is piecewise-linear convex, and why exact belief planning dies at lidar dimensionality — motivating everything after. Occupancy-grid mapping cast as a factored belief over map cells.
- **C:** `ch19-belief-cloud` — Rusty's Ch 4 belief-cloud widget upgraded to continuous rooms: particle cloud over poses shrinking/splitting as scans arrive; the reader teleoperates and watches corridor aliasing bifurcate the belief.
- **P:** `rl-envs::lidar_rusty`: differential-drive Rusty in `rapier2d` rooms; raycast lidar (`parry` ray queries) with range noise and dropout; obstacle-layout randomization hooks (Ch 15 style); goal-conditioned `Env` implementation.

### 19.2 Recurrent Policies as Belief Trackers
- **F:** Claim, stated carefully: a sufficient statistic of history is exactly what value prediction requires; training $h_t = g_\theta(h_{t-1}, o_t, a_{t-1})$ (GRU update equations written out) to maximize return drives $h_t$ toward an approximate belief statistic — the filter recursion and the RNN recursion have the same signature, one derived from Bayes, one learned from reward. Truncated BPTT as the bias this introduces; observation/action history stacks (Ch 18's student encoder) as the finite-window special case.
- **C:** `ch19-belief-probe` — widget: a linear probe decodes Rusty's true pose and nearest-obstacle distance from a trained GRU's $h_t$ live; decoding error spikes exactly where the particle cloud of 19.1 is multimodal — the two visualizations run side by side.
- **P:** `rl-deep::recurrent`: GRU policy/value module in `burn`, sequence rollout storage (episodes, not transitions), truncated-BPTT PPO variant; the probe as a ridge regression over logged hidden states.

### 19.3 Rusty Graduates: Lidar Navigation in Continuous Rooms
- **F:** Reward design for point-goal navigation with the Ch 14 rubric: progress term $\Delta\lVert x - g\rVert$ (potential-based, invariance argument restated), collision penalty, heading bonus, time cost — dimensional analysis discipline from Ch 18.4 reused. Success metrics: success rate, SPL (success weighted by path length) defined.
- **C:** `ch19-lidar-pov` — visualization: split screen of the room (bird's eye) and what the policy "sees" (the 64-beam scan as a polar plot with saliency weights from the policy's first layer); the reader drags obstacles and watches the scan and the action respond.
- **P:** Full training run: recurrent PPO on randomized rooms; curriculum over obstacle density; evaluation harness reporting success rate and SPL with seeds; `bevy` scene replays of best/worst episodes.

### 19.4 End-to-End vs Hybrid-Modular: the Evidence
- **F:** The classical stack as a factored policy: $\pi = \text{plan}_{\text{local}} \circ \text{plan}_{\text{global}} \circ \text{localize} \circ \text{map}$; enumeration of substitution points where RL has real-world wins per Tang §4.2.1 (learned local planner, learned exploration policy) vs full end-to-end; error analysis: modular errors are diagnosable but compound at interfaces, end-to-end errors are holistic but unattributable; why commercial deployments remain classical (safety, interpretability — Tang's L-level evidence).
- **C:** `ch19-pipeline-switcher` — the chapter's signature widget: same randomized scene, toggle between (a) classical (occupancy map + A* + DWA-style local), (b) classical-with-RL-local-planner, (c) end-to-end recurrent policy; failure counters and trajectory overlays accumulate per architecture as the reader spawns scenarios (narrow gap, dynamic obstacle, dead end).
- **P:** `rl-envs::nav_stack`: minimal occupancy-grid mapper, A* global planner, arc-sampling local planner (all `ndarray`/`nalgebra`, ~300 lines total — a teaching stack, not ROS); the three architectures benchmarked on a fixed 200-scene suite.

### 19.5 Embodiment Matters: Legged and Aerial Navigation
- **F:** The abstraction ladder: navigation policy commands $(\hat v_x, \hat v_y, \hat\omega_z)$ consumed by Ch 18's locomotion policy — formal composition and its validity condition (time-scale separation; command tracked faster than commands change). Tang §4.2.2's kinematic-abstraction finding (low-fidelity sim transfers better when dynamics are delegated) vs joint loco-navigation training for agile terrain where the abstraction breaks (gaps, stairs); aerial navigation: drone racing (Kaufmann 2023) as the L5 case where planning and control *must* be optimized jointly — RL's structural advantage over layered optimal control restated from Ch 18.7.
- **C:** `ch19-abstraction-ladder` — animation: the same cluttered course attempted by (a) velocity-abstracted Ferris (Ch 18 policy underneath) and (b) a hypothetical joint policy; overlaid annotations show where abstraction is safe (flat) and where it fails (gap requiring gait change); includes a social-navigation trajectory gallery panel (deferred formally to Ch 21).
- **P:** `rl-envs::ferris_nav`: waypoint navigation over Ch 18's terrain with the frozen locomotion policy as actuator; measures composition overhead vs flat-ground oracle. (Joint training is discussed, costed, and deliberately not run — the capstone revisits it.)

### 19.6 Hierarchy: Skills, Semi-MDPs, and Long-Horizon Fetch
- **F:** Options recap from Ch 17 with navigation instantiation: option $\omega = (I_\omega, \pi_\omega, \beta_\omega)$; semi-MDP Bellman equation $q(s,\omega) = \mathbb E[r_{1:k} + \gamma^k \max_{\omega'} q(s', \omega')]$ derived with random duration $k$; chaining condition (termination set of $\omega_i$ inside initiation set of $\omega_{i+1}$); why sparse long-horizon rewards defeat flat RL (exploration horizon scaling argument) and how temporal abstraction shortens the effective horizon; Tang §5's twin open questions — *what* skills, and *how combined* — as the section's honest frame.
- **C:** `ch19-skill-graph` — animation: the fetch task's skill graph (navigate → dock → open → reach → grasp-lite → retreat) lighting up as the high-level policy executes; edge thickness = learned transition preference; failure replays show mis-chaining (docked too far, reach out of workspace).
- **P:** `rl-deep::hier`: waypoint-conditioned navigation skill + docking skill + reach skill (each small PPO/SAC policies), high-level selector trained with semi-MDP Q-learning over skill outcomes; comparison against flat recurrent PPO on the fetch task (expected result per Tang: flat fails to leave L0).

### 19.7 Mobile Manipulation: Whole-Body Control and Redundancy
- **F:** Kinematic redundancy for base+arm: task $x = f(q)$, $q = (q_{\text{base}}, q_{\text{arm}}) \in \mathbb R^n$, $\dot x = J\dot q$, $J \in \mathbb R^{m\times n}, n > m$; weighted least-norm resolution derived: $\dot q = W^{-1}J^\top (J W^{-1} J^\top)^{-1} \dot x$, null-space projector $N = I - J^{\dagger}_W J$ for secondary posture objectives; where the analytic answer fails (contact, slip, actuator limits) and DRL's role per Tang §4.4.1; the action-space zoo of MoMa (joint, task-space, factored base/arm) and Tang §4.4.4's "no principled selection method" — connecting back to Ch 17's action-space evidence.
- **C:** `ch19-nullspace` — sandbox: drag mobile-Reacher's end-effector; sliders trade base motion vs arm motion via $W$; toggle an RL whole-body policy trained on the same task and compare its implicit weighting, including behaviors the pseudoinverse cannot express (backing up before reaching).
- **P:** `rl-envs::mobile_reacher`: Rusty base + Reacher arm as one `rapier2d` articulated body; whole-body reach task (SAC, mid-level twist+joint-velocity actions); then the full fetch of 19.6 rerun with learned WBC as the reach skill — the chapter's integration experiment.

### 19.8 Chapter Bridge
- Recap: partial observability handled three ways (belief tracking, modular structure, hierarchy), and the honest scorecard of each against Tang's L-levels. Rusty now navigates; mobile-Reacher fetches. Ch 20 zooms into the hand: contact-rich manipulation, where the hard part is not *where to be* but *what touching does*. Ch 21 picks up social navigation (humans in the scene) and Ch 22 inherits the evaluation harness and hierarchy lessons.

## 4. Interactive Widgets & Dashboards

| ID | Type | What it shows | Reader interaction | Tech |
|---|---|---|---|---|
| `ch19-belief-cloud` | widget | Particle belief over Rusty's pose in continuous rooms | Teleoperate; watch aliasing split/merge the belief | egui + rapier2d, WASM |
| `ch19-belief-probe` | widget | Pose/obstacle decoding from GRU hidden state vs particle filter | Scrub a trajectory; compare probe error to belief multimodality | egui_plot + burn WASM inference |
| `ch19-lidar-pov` | visualization | Bird's-eye room + policy's polar-scan view with input saliency | Drag obstacles/goal; inspect per-beam saliency | egui, WASM |
| `ch19-pipeline-switcher` | widget | Classical vs hybrid vs end-to-end on identical scenes | Toggle architecture; spawn stress scenarios; failure counters accumulate | egui + rapier2d, WASM |
| `ch19-abstraction-ladder` | animation | Velocity-abstracted vs joint loco-navigation on one course; social-trajectory gallery panel | Scrub course; reveal where the abstraction breaks | bevy replays |
| `ch19-skill-graph` | animation | Fetch skill graph executing; learned transition preferences | Replay successes/failures; inspect chaining violations | egui/SVG + logged rollouts |
| `ch19-nullspace` | sandbox | Analytic redundancy resolution vs learned whole-body policy | Drag end-effector; tune weight matrix $W$; toggle RL policy | egui + rapier2d, WASM |

## 5. Rust Implementation Plan

**Crates touched:** `rl-envs` (`lidar_rusty`, `nav_stack`, `ferris_nav`, `mobile_reacher`), `rl-deep` (`recurrent`, `hier`), `rl-core` (episode-major rollout storage), `rl-viz` (polar-scan plot component), demos `ch19-*`.

**Modules/files:** `rl-envs/src/lidar_rusty/{env.rs, lidar.rs, rooms.rs}`, `rl-envs/src/nav_stack/{grid.rs, astar.rs, local.rs}`, `rl-envs/src/mobile_reacher.rs`, `rl-deep/src/recurrent.rs`, `rl-deep/src/hier.rs`.

Representative sketch — GRU policy step, the belief tracker made concrete (`rl-deep/src/recurrent.rs`):

```rust
pub struct GruPolicy<B: Backend> {
    encoder: Linear<B>,
    gru: GruCell<B>,
    actor_head: Linear<B>,  // Gaussian mean over (v, omega)
    critic_head: Linear<B>,
}

pub struct StepOut<B: Backend> {
    pub mean: Tensor<B, 2>,
    pub value: Tensor<B, 2>,
    pub h: Tensor<B, 2>, // carried belief surrogate; probed by ch19-belief-probe
}

impl<B: Backend> GruPolicy<B> {
    /// One control step: h_t = GRU(h_{t-1}, [o_t, a_{t-1}]) — the learned
    /// counterpart of the Bayes-filter recursion in Sec. 19.1.
    pub fn step(&self, obs_and_prev_a: Tensor<B, 2>, h: Tensor<B, 2>) -> StepOut<B> {
        let x = relu(self.encoder.forward(obs_and_prev_a));
        let h_next = self.gru.forward(x, h);
        StepOut {
            mean: self.actor_head.forward(h_next.clone()),
            value: self.critic_head.forward(h_next.clone()),
            h: h_next,
        }
    }
}
```

**Experiments/benchmarks:** (1) feedforward vs frame-stack vs GRU on aliased corridors (the memory ablation); (2) the 200-scene architecture benchmark (classical/hybrid/end-to-end) with success rate + SPL + failure taxonomies; (3) flat vs hierarchical on long-horizon fetch; (4) analytic WBC vs learned WBC reach accuracy under randomized dynamics; (5) `criterion` on lidar raycasting throughput.

**Native vs browser:** all four `rapier2d` environments compile to WASM; training native; every widget above ships in-browser with pretrained checkpoints; `bevy` replays pre-rendered for `ch19-abstraction-ladder`.

## 6. Robot Thread

- **Rusty** (graduation): from Ch 4–7 tabular gridworld and Ch 9 visual gridworld to continuous lidar navigation with memory; ends the chapter also serving as the mobile base.
- **Reacher** (promotion): mounted on Rusty as mobile-Reacher; its Ch 11/13 reaching skills now execute from a moving, redundant base.
- **Ferris** (cameo): Ch 18's locomotion policy consumed as the actuator for legged waypoint navigation in 19.5 — deliberately frozen, not retrained.
- **Pendle**: rests.

## 7. Exercises & Explorations

1. **(F)** Derive the belief update for a two-corridor aliasing world with three states and show the belief remains bimodal under any open-loop action sequence until a disambiguating observation arrives.
2. **(F)** Prove the progress reward $r_t = \lVert x_{t-1} - g\rVert - \lVert x_t - g\rVert$ is potential-based shaping, and exhibit the policy-distorting variant (per-step distance penalty) the Ch 14 theorem warns about.
3. **(F)** Derive the weighted pseudoinverse of 19.7 from the constrained least-norm program, and show how $W \to \mathrm{diag}(\infty \cdot I_{\text{base}}, I_{\text{arm}})$ recovers arm-only manipulation.
4. **(C)** Using `ch19-pipeline-switcher`, find a scene where the classical stack beats end-to-end and one where the reverse holds; write a paragraph mapping each to Tang §4.2.4's takeaways.
5. **(C)** With `ch19-belief-probe`, locate a trajectory point where probe error is high but the policy still acts well; explain what statistic the GRU kept and what it discarded.
6. **(P)** Add a second lidar dropout mode (sector blackout) to `lidar_rusty`; measure which of the three architectures degrades least and relate to the modular error-attribution argument of 19.4.
7. **(P)** Replace the semi-MDP selector with a flat PPO given an extended timeout; reproduce the exploration-horizon failure and report sample counts at matched success rates.
8. **(P)** Train mobile-Reacher's WBC with the base motors disabled at random episodes; verify the policy learns a null-space-like preference and compare against the analytic $N$-projector behavior in `ch19-nullspace`.

## 8. Notation Introduced

| Symbol | Meaning |
|---|---|
| $b_t(s), \eta$ | belief state; Bayes normalizer |
| $\mathcal B, \tau, \rho$ | belief space, belief transition kernel, belief reward |
| $h_t$ | recurrent hidden state (learned belief surrogate) |
| $\omega = (I_\omega, \pi_\omega, \beta_\omega)$ | option (skill); $\omega$ chosen to avoid clash with observation $o_t$ |
| $g$ | navigation/fetch goal |
| SPL | success weighted by (normalized inverse) path length |
| $J, J^{\dagger}_W, W, N$ | task Jacobian, weighted pseudoinverse, weight matrix, null-space projector |
| $q = (q_{\text{base}}, q_{\text{arm}})$ | whole-body configuration |

## 9. References & Further Reading

- **Baseline:** Tang §4.2.1–4.2.4 (wheeled/legged/aerial evidence and takeaways), §4.4.1–4.4.4 (WBC, short/long-horizon MoMa), §5 (long-horizon open questions); Kober §8.1 (perception integration); S&B ch. 17 context for options (with Ch 17's semi-MDP treatment).
- Wijmans et al. 2020, *DD-PPO* (near-perfect simulated point-goal — and why that is not L3).
- Chaplot et al. 2020, *Active Neural SLAM* (modular RL exploration beating end-to-end).
- Truong et al. 2023, *Rethinking sim-to-real: lower fidelity simulation leads to higher sim-to-real transfer* (kinematic abstraction for legged navigation).
- Lee et al. 2024, wheeled-legged kilometer-scale urban navigation (Tang's L4 legged-navigation exemplar).
- Kaufmann et al. 2023, champion-level drone racing (joint planning+control, restated from Ch 18).
- Kendall et al. 2019, *Learning to drive in a day*; Jang et al. 2024 fleet-scale cruise control (Tang §4.2.1 autonomous-driving caveats).
- Fu et al. 2023, *Deep whole-body control* (base+arm coupling learned end-to-end); Herzog et al. 2023 (waste-sorting long-horizon MoMa).
- Sutton, Precup & Singh 1999, *Between MDPs and semi-MDPs* (options; proof source for 19.6).
- Bakker 2001 / Ni et al. 2022, recurrent RL as POMDP solvers (evidence for 19.2's claim).

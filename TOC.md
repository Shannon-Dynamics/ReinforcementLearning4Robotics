# Reinforcement Learning for Robotics — The FCP Way

**An interactive web book: Foundation · Conceptual · Practical**

*Mathematical formalism in full detail · Interactive web simulations for every hard idea · Production-quality Rust implementations*

---

## 0. Vision & Method

This book modernizes and unifies four baseline references into a single, coherent, interactive learning path:

| Baseline reference | Role in this book |
|---|---|
| **Sutton & Barto**, *Reinforcement Learning: An Introduction* (2nd ed.) | Mathematical spine of Parts I–II (bandits → MDPs → DP/MC/TD → function approximation → policy gradients) |
| **Kober, Bagnell & Peters**, *RL in Robotics: A Survey* (IJRR 2013) | The robotics bridge of Part III: the four curses, representation/prior-knowledge/model tractability, ball-in-a-cup case study |
| **Tang et al.**, *Deep RL for Robotics: A Survey of Real-World Successes* (2024) | Modern taxonomy of Part IV: competencies (locomotion, navigation, manipulation, mobile manipulation, HRI, multi-robot), levels of real-world success, problem-formulation/solution-approach axes |
| **Akinola**, *RL for Robotics* (Columbia lecture) | Pedagogical framing: see–think–act, learning-from-demonstration vs RL vs hybrid, simulators & sim-to-real |

**Modernization contract.** Where the references stop (2013 survey methods, tabular-era pedagogy), this book continues: PPO/SAC as the workhorses, world models, offline RL, teacher–student sim-to-real, massively parallel simulation, and foundation-model frontiers — each grounded in the same mathematical spine.

### The FCP method

Every chapter delivers three interleaved layers:

- **F — Foundation.** Full mathematical formalism: definitions, theorems, derivations (not proof sketches — the actual algebra), convergence conditions, and the assumptions robots violate.
- **C — Conceptual.** Every hard concept gets a **visual**: an interactive web widget, animation, or live dashboard embedded in the page. The reader *manipulates* the math (drag a policy, watch value functions ripple, randomize physics parameters and watch transfer break).
- **P — Practical.** Every algorithm is implemented in **Rust** with the best current crates, from tabular Q-learning to PPO on a simulated quadruped — the same code compiles natively for training and to WASM for in-browser demos.

### Toolchain (detailed in CLAUDE.md and Appendix B)

- **Web book**: Next.js 15 + React 19 + TypeScript + Tailwind, content in MDX, math via KaTeX, code highlighted at build time by Shiki, charts by Nivo, and interactive simulations driven by a TypeScript RL engine. *(Revised 2026-08 from the original mdBook + Rust/WASM-widget plan — React composition binds prose and simulation far more tightly; see CLAUDE.md §3 for the rationale.)*
- **Numerics** (the taught Rust curriculum): `ndarray`, `nalgebra`, `rand`/`rand_distr`, `statrs`
- **Deep learning**: `burn` (primary; WGPU backend, trains natively, runs in-browser), `candle` (lean inference alternative)
- **Physics & robots**: `rapier2d`/`rapier3d`, `parry`, `urdf-rs`, `k` (kinematics), `bevy` + `bevy_rapier` + `bevy_urdf`
- **Visualization**: `egui`/`eframe` + `egui_plot` (WASM dashboards), `plotters`, `bevy` (3D scenes), `rerun` (native dev-time robot telemetry)
- **Infrastructure**: `rayon`, `serde`, `tracing`, `criterion`; book-owned crates `rl-core`, `rl-tabular`, `rl-deep`, `rl-envs`, `rl-sim`, `rl-viz`

### Running robots (the book's recurring cast)

| Robot | Introduced | Main thread (starring chapters; cameos elsewhere) |
|---|---|---|
| **Rusty** — differential-drive mobile robot | Ch 1 | hello-robot → gridworld → visual gridworld → lidar navigation → multi-agent (Ch 1, 4–7, 9, 16, 19, 21) |
| **Pendle** — pendulum / cart-pole | Ch 2 | classical control ↔ RL bridge (Ch 2, 8, 10–13, 15) |
| **Reacher** — 2-link planar arm | Ch 3 | manipulation thread (Ch 3, 11, 13–17, 19–21) |
| **Ferris** — quadruped | Ch 15 | locomotion & sim-to-real thread (Ch 15, 18–19, 22) |

---

## Part I — Foundations of Sequential Decision-Making

*Sutton & Barto's spine, retold with robots and interactive math.*

### Chapter 1 — Why Reinforcement Learning for Robotics?
*Sources: Kober §1, Tang §1–3, Akinola motivation slides.*
The see–think–act loop; why direct programming breaks (structure, perception, manipulation, deformation); learning from demonstration vs reinforcement learning vs hybrid; a guided tour of real-world successes (drone racing, ANYmal, in-hand cube) organized by Tang's competency taxonomy and **levels of real-world success (L0–L5)**; the anatomy of an RL problem (agent, environment, reward, policy) informally; the book's method, cast of robots, and toolchain.
**F**: agent–environment loop as a measurable dynamical system (informal → formal preview). **C**: interactive competency-taxonomy explorer; L0–L5 success-level dashboard of surveyed systems; see–think–act animation. **P**: Rust/WASM "hello robot": drive Rusty in the browser with a hand-coded controller, feel why hand-coding fails; tour of the workspace crates.

### Chapter 2 — The Mathematical Toolkit
*Sources: S&B notation + prerequisites modernized; robotics math from Kober §2 context.*
Probability spaces, random variables, expectation, conditional expectation, Markov chains; vectors, matrices, eigen-structure; gradients, Jacobians, Hessians; convexity and gradient descent; ODEs and discretization (how a robot's continuous dynamics become a discrete-time decision process); stochastic approximation (Robbins–Monro) — the theorem that makes every RL algorithm tick.
**F**: full formal statements; contraction mappings and fixed points (Banach) proved. **C**: interactive gradient-descent landscape; Markov-chain mixing animation; discretization-error explorer (Euler vs RK4 on Pendle). **P**: `ndarray`/`nalgebra` crash projects; implement Robbins–Monro; simulate and plot Pendle's ODE with `egui_plot`.

### Chapter 3 — Multi-Armed Bandits: Exploration & Exploitation
*Sources: S&B ch. 2.*
The n-armed bandit as the atom of RL; action-value estimation, incremental updates, nonstationarity; ε-greedy, optimistic initialization, UCB, gradient bandits (softmax preferences — the seed of policy gradients), Thompson sampling (modernization); contextual bandits as the bridge to full RL.
Robot framing: **Reacher** choosing among grasp primitives; auto-tuning a controller gain online.
**F**: regret, concentration inequalities (Hoeffding), UCB regret bound derived. **C**: live bandit testbed dashboard (10-armed, distribution violins, regret curves racing per algorithm); UCB confidence-interval animation. **P**: `rl-core` `Bandit` trait; all algorithms in Rust; parameter-study harness with `rayon`; WASM playground where the reader designs arm distributions.

### Chapter 4 — Markov Decision Processes: The Formalism
*Sources: S&B ch. 3; Tang §3.2 problem-formulation axis.*
The MDP tuple (S, A, P, R, γ); returns, episodic vs continuing tasks; policies; state-value and action-value functions; **Bellman expectation and optimality equations derived in full**; existence/uniqueness via contraction; partial observability (POMDP) introduced honestly — because robots never see state — with belief states; Tang's formulation axes: action-space level (low/mid/high), observation space, sparse vs dense reward.
Robot framing: **Rusty**'s warehouse world as MDP; the same world with noisy odometry as POMDP.
**F**: full Bellman derivations; γ-contraction proof; MDP↔POMDP relationship. **C**: interactive MDP graph editor (edit P/R, watch v_π ripple); backup-diagram animator; POMDP belief-cloud visualization over Rusty's map. **P**: `rl-core`: `Mdp`, `Env`, `Space` traits (the book's gym); exact v_π solving via `nalgebra` linear algebra; property tests with `proptest`.

### Chapter 5 — Dynamic Programming: Planning with a Known Model
*Sources: S&B ch. 4.*
Policy evaluation (iterative), policy improvement theorem (proved), policy iteration, value iteration, asynchronous DP, generalized policy iteration (GPI) as the book's master pattern; computational reality: curse of dimensionality quantified (preview of Kober's curses).
**F**: policy improvement theorem; convergence of value iteration; complexity analysis. **C**: the signature widget — **live GPI dashboard**: heatmap of V, arrows of π, sweep-by-sweep animation with speed control on Rusty's warehouse; asynchronous-sweep comparator. **P**: `rl-tabular`: policy/value iteration over `rl-core` MDPs; benchmark sweeps with `criterion`; state-space-size explosion experiment (why tabular DP dies at robot scale).

### Chapter 6 — Learning from Experience: Monte Carlo & Temporal-Difference
*Sources: S&B ch. 5–6.*
Learning without a model: MC prediction/control, exploring starts, ε-soft policies; importance sampling (ordinary vs weighted, derived); TD(0), the TD error δ; SARSA vs Q-learning (on- vs off-policy — the distinction that rules deep RL); expected SARSA; convergence conditions; maximization bias and Double Q-learning.
Robot framing: Rusty learns the warehouse from rollouts alone; sensor-noise ablation.
**F**: first-visit MC unbiasedness; importance-sampling variance analysis; Q-learning convergence (Watkins) statement + proof sketch; Double-Q estimator math. **C**: MC vs TD "credit propagation" side-by-side animation; TD-error river (δ flowing backward along a trajectory); maximization-bias casino widget. **P**: `rl-tabular`: MC, SARSA, Q-learning, Double-Q; live-training WASM dashboard (reward curve, ε schedule, Q-heatmap updating in real time).

### Chapter 7 — Unifying Learning & Planning: n-step, Traces, Dyna & MCTS
*Sources: S&B ch. 7, 8 (+ ch. 12 concepts).*
n-step TD and the bias–variance dial; forward vs backward view; eligibility traces, TD(λ), Sarsa(λ); Dyna-Q: learning + planning from an imagined model; prioritized sweeping; when the model is wrong (Dyna-Q+); trajectory sampling; MCTS (the AlphaGo connection) as decision-time planning.
Robot framing: Rusty replans when the warehouse layout changes overnight (model-wrong experiments).
**F**: λ-return equivalence theorem (forward=backward); n-step error bounds. **C**: **λ-dial widget** (drag λ∈[0,1], watch trace decay and credit assignment morph between TD(0) and MC); Dyna imagination visualizer (real vs imagined transitions); MCTS tree growth animation. **P**: `rl-tabular`: Sarsa(λ), Dyna-Q(+), prioritized sweeping with a binary heap; MCTS on a maze; planning-budget vs performance study.

---

## Part II — Scaling Up: Function Approximation & Deep RL

*From tables to tensors — the leap robots require.*

### Chapter 8 — Function Approximation & the Deadly Triad
*Sources: S&B ch. 9–11; Kober §2.4, §4.2.*
Why robots outgrow tables (continuous state); prediction as supervised projection: the VE objective, semi-gradient TD; linear methods and their geometry (projection operator, fixed point); feature construction: polynomials, RBFs, **tile coding** (implemented for real), Fourier basis; neural networks as features that learn themselves; **the deadly triad** (function approximation + bootstrapping + off-policy) with Baird's counterexample run live; gradient-TD fixes in brief.
**F**: semi-gradient derivation; linear TD fixed-point theorem; Baird divergence analysis. **C**: feature-space visualizer (tilings over Pendle's state space); **Baird's counterexample live-diverging dashboard**; projection-geometry diagram (animated). **P**: `rl-deep` begins: linear semi-gradient SARSA on cart-pole with tile coding (pure `ndarray`); first `burn` MLP: nonlinear value function on Pendle; side-by-side stability experiments.

### Chapter 9 — Deep Value-Based Methods: DQN & Descendants
*Sources: modernization beyond S&B; Tang solution-approach axis (off-policy value-based).*
From Q-table to Q-network; why naive deep Q-learning explodes; the DQN recipe (replay buffer, target network) as variance/correlation surgery; Double DQN, Dueling, Prioritized Replay, n-step returns, Rainbow synthesis; distributional RL (C51) as the modern frontier of value learning; where value-based methods sit in robotics (discrete/mid-level action spaces, e.g., grasp selection — Tang Table 5 context).
**F**: replay as breaking Markov correlation; target-network fixed-point analysis; distributional Bellman operator. **C**: replay-buffer flow animation; target-network "two clocks" widget; C51 distribution-morphing visualization; loss-landscape stability comparison dashboard. **P**: `rl-deep`: full DQN in `burn` (WGPU backend) on visual-gridworld Rusty; prioritized replay with a sum-tree; training telemetry streamed to an `egui` dashboard; WASM inference demo of the trained agent.

### Chapter 10 — Policy Gradients: REINFORCE → PPO
*Sources: S&B ch. 13-equivalents; Kober §2.2.2 policy search; modern PPO practice (Tang: on-policy workhorse).*
Why robots prefer policies over values (continuous actions, smooth updates, prior structure — Kober's argument formalized); **the policy gradient theorem derived step by step**; REINFORCE and its variance problem; baselines and advantage; actor–critic; GAE(λ); natural gradient and trust regions (TRPO sketch); **PPO in full** (clipped surrogate, implementation details that actually matter: advantage normalization, entropy bonus, minibatching, KL watchdogs) — the algorithm behind most real-world robot RL successes in Tang's survey.
**F**: PG theorem (both episodic and continuing derivations); variance of score-function estimators; GAE bias–variance telescoping; PPO surrogate-objective analysis. **C**: **policy-space vs parameter-space widget** (drag θ, watch π morph and J(θ) respond); variance-of-gradient-estimate live histograms (REINFORCE vs +baseline vs GAE); PPO clipping-function interactive plot; trust-region ellipse animation. **P**: `rl-deep`: REINFORCE → A2C → PPO in `burn`, vectorized environments via `rayon`; continuous-action Gaussian policies; Pendle swing-up solved; full training dashboard (returns, KL, clip fraction, entropy).

### Chapter 11 — Off-Policy Continuous Control: DDPG, TD3 & SAC
*Sources: modernization; Tang solution-approach axis (off-policy for sample efficiency); Kober's sample-cost arguments.*
Sample efficiency as a robotics imperative (every real rollout wears bearings); deterministic policy gradients (DPG theorem derived); DDPG and its fragility; TD3's three fixes (clipped double-Q, delayed policy updates, target smoothing); **maximum-entropy RL**: soft Bellman operator, SAC derivation (temperature, reparameterization trick), automatic entropy tuning; practical comparison protocol: when PPO vs SAC in robotics.
Robot framing: **Reacher** learns to reach targets — the book's first true continuous-control robot task.
**F**: DPG theorem; soft policy iteration convergence; reparameterization-gradient derivation; overestimation-bias analysis. **C**: overestimation-bias accumulation animation; entropy-temperature dial (watch exploration breathe on Reacher); Q-landscape over action space (live surface while training); PPO-vs-SAC sample-efficiency race dashboard. **P**: `rl-deep`: DDPG, TD3, SAC in `burn`; `rl-envs`: Reacher (2-link arm, `rapier2d`); replay + target-network infrastructure shared with Ch 9; WASM demo: trained Reacher tracking the reader's cursor.

### Chapter 12 — Model-Based RL & World Models
*Sources: Kober §6 (models & mental rehearsal) modernized; Tang model-learning axis; S&B ch. 8 lineage.*
Why models matter for robots (Kober: mental rehearsal, sample reuse; Tang: real-world learning challenge); learning dynamics: deterministic nets, Gaussian ensembles, quantifying epistemic vs aleatoric uncertainty; planning with learned models: random shooting, CEM, MPC; PETS; Dyna-style hybrids (MBPO idea); **latent world models** (Dreamer lineage): learn in imagination; model-bias horror stories and how ensembles tame them; connection to classical system identification.
**F**: model-bias compounding bound (error growth with horizon); CEM as importance-sampled optimization; ELBO derivation for latent dynamics. **C**: **"imagination rollout" widget**: real trajectory vs model rollouts diverging with horizon, ensemble fan visualization; MPC receding-horizon animation on Pendle; latent-space trajectory projector. **P**: `rl-deep`: ensemble dynamics models in `burn`; CEM-MPC controller for Pendle swing-up with learned model; imagination-vs-reality evaluation harness; uncertainty-calibration plots.

---

## Part III — The Robotics Side

*Kober's bridge, rebuilt with modern materials.*

### Chapter 13 — The Robot as an Environment: Kinematics, Dynamics & Control
*Sources: Kober §1.3, §2 context; robotics fundamentals the surveys assume.*
What's inside the "env" black box when it's a robot: rigid-body kinematics (SE(2)/SE(3), forward/inverse kinematics), differential kinematics (Jacobians), Lagrangian dynamics (manipulator equation M(q)q̈ + C + g = τ derived for Reacher); underactuation; classical control as baselines RL must beat: PID, LQR (derived), gravity compensation, operational-space control; MPC preview; where learning enters: as policy, as model, as residual, as tuner; observability and state estimation (EKF sketch) — why POMDPs are the honest formalism.
**F**: full FK/IK math for Reacher; Lagrangian derivation; LQR Riccati derivation; linearization validity. **C**: interactive FK/IK sandbox (drag Reacher's end-effector, watch joint space vs task space); Jacobian ellipsoid animation; LQR-vs-PID disturbance-rejection dashboard on Pendle. **P**: `rl-sim`: `urdf-rs` + `k` for kinematic chains; Reacher and Pendle as `rapier2d` articulated bodies; PID and LQR controllers in `nalgebra`; controller benchmark suite — the baselines later chapters must beat.

### Chapter 14 — The Four Curses of Robot RL
*Sources: Kober §3 (the heart of the 2013 survey) + modern updates; Tang §5 challenges.*
Kober's four curses, each quantified and demonstrated: **dimensionality** (continuous high-D state–action), **real-world samples** (wear, time, resets, safety), **under-modeling** (sim≠real), **goal specification** (reward design is the interface between human intent and optimization); modern additions: reward hacking galleries, safe exploration, automatic resets (Tang's real-world-learning challenge); reward-shaping theory (potential-based shaping theorem proved); curriculum design; evaluation methodology for robot RL claims (Tang's L0–L5 revisited as an evaluation rubric).
**F**: sample-complexity lower-bound intuition; potential-based shaping invariance theorem; constrained MDPs (CMDP formalism) for safety. **C**: **curse-of-dimensionality exponential-wall visualizer**; reward-hacking zoo (interactive examples: the reader designs a reward, a trained agent exploits it); shaping-invariance widget; safety-constraint boundary animation. **P**: reward-design lab: same Reacher task, six reward functions, six trained SAC behaviors compared in a WASM gallery; CMDP Lagrangian-SAC prototype; reset-cost accounting in `rl-core` metrics.

### Chapter 15 — Simulation & the Sim-to-Real Bridge
*Sources: Kober §6.1 mental-rehearsal issues; Tang §3.3 simulator-usage axis + zero-shot/few-shot findings; Akinola simulator slides.*
Physics engines from the inside: integrators, contact models, solver stiffness — why contact is where sim and real diverge; the sim-to-real gap taxonomy; **domain randomization** (dynamics + visual) and why it works (robustness as distributional coverage); system identification (fit rapier parameters to logged real data); teacher–student / privileged learning (the ANYmal recipe); massively parallel simulation (GPU farms → `rayon` farms); zero-shot vs few-shot transfer decision tree from Tang's evidence; introducing **Ferris** the quadruped.
**F**: contact LCP formulation sketch; domain randomization as minimax/distributionally-robust objective; asymmetric actor-critic formalism. **C**: **integrator-stability playground** (crank Δt until Pendle explodes); contact-parameter sensitivity heatmap; domain-randomization slider wall (mass/friction/latency distributions → policy robustness meter); teacher–student information-flow diagram (animated). **P**: `rl-sim`: parameterized-physics `rapier` environments with randomization hooks; `rayon` vectorized rollout farm (thousands of Ferris instances); sysid: fit sim parameters to trajectory data via CMA-ES (`cmaes` crate); train randomized vs non-randomized Reacher, compare transfer to a perturbed "real" sim.

### Chapter 16 — Demonstrations, Imitation & Offline RL
*Sources: Kober §5 (prior knowledge); Akinola LfD slides (modes, BC, covariate shift); Tang expert-usage axis + real-world-learning trend; modern offline RL.*
Where data can come from besides exploration: demonstration modes (teleoperation, kinesthetic teaching, motion capture, video); behavior cloning and the **covariate-shift theorem** (compounding errors, DAgger fix); inverse RL (max-margin, MaxEnt IRL derived); demo-accelerated RL (demos in replay, BC-regularized objectives); **offline RL**: distributional shift, pessimism, CQL and IQL derived; offline-to-online fine-tuning (the modern deployment recipe per Tang); GAIL sketch.
**F**: BC error compounding bound (T² vs DAgger's T); MaxEnt IRL partition-function derivation; CQL lower-bound property; IQL expectile regression math. **C**: **covariate-shift drift animation** (cloned Rusty drifting off the demonstrated lane, error cone growing); demo-mode comparison cards; pessimism-value-landscape widget (CQL pushing down out-of-distribution actions, live surface); offline→online fine-tuning timeline dashboard. **P**: teleoperate Reacher in the browser (mouse = demos recorded to `serde` datasets); BC and DAgger in `burn`; IQL on logged Reacher data; fine-tune with SAC and measure the offline-to-online jump.

### Chapter 17 — Motor-Skill Policy Representations
*Sources: Kober §4 (tractability through representation) + §7 ball-in-a-cup case study; Tang §3.2 action-space axis.*
Kober's insight, still true in the deep era: *what* the policy outputs matters as much as *how* it learns; action-space levels (joint torque / joint position+PD / task-space / primitives — Tang's low/mid/high taxonomy with empirical trade-offs); **dynamic movement primitives** (DMPs) derived in full (canonical system, forcing term, learning from a demo, goal generalization); central pattern generators (CPGs) for rhythmic locomotion; residual RL (learned Δ on a classical controller); options & skills (semi-MDP formalism) for hierarchy; **the ball-in-a-cup case study recreated**: Kober's 2013 pipeline (DMP + imitation init + policy search) vs a 2020s pipeline (SAC from pixels + demos), same task, compared honestly.
**F**: DMP stability proof (spring–damper convergence); semi-MDP Bellman equations; residual-policy stability argument. **C**: **DMP forcing-function sculptor** (draw a trajectory, watch basis functions fit it, drag the goal — generalization live); CPG phase-oscillator network animation; action-space level comparison dashboard (same Reacher task, three action spaces, learning curves racing); ball-in-a-cup then-vs-now split-screen. **P**: `rl-core`: DMP module (pure `nalgebra`); CPG oscillator bank; residual-RL wrapper env; ball-in-a-cup in `rapier2d` with both pipelines; policy search on DMP weights in the spirit of Kober's pipeline (PoWER derived in the chapter; CMA-ES deployed as the practical optimizer, substitution flagged).

---

## Part IV — Competencies: RL on Real Robots

*Tang's taxonomy as three deep dives: what actually worked, why, and rebuilt in Rust.*

### Chapter 18 — Learning Locomotion
*Sources: Tang §4.1 (quadruped, biped, quadrotor) + trends; Kober lineage examples.*
Why locomotion became DRL's flagship (stable dynamics, simulable contact, dense rewardable gaits — Tang's analysis); the modern quadruped recipe end-to-end: observation/action design, reward anatomy (velocity tracking + effort + air-time + …), terrain curricula, teacher–student with privileged terrain info, domain randomization for zero-shot transfer (L4 systems: ANYmal, production quadrupeds); biped/humanoid differences (underactuation, falling is expensive); quadrotor agile flight (drone racing at L5); recovery behaviors; central case study: **Ferris learns to walk**.
**F**: gait phase formalism; reward-term algebra with dimensional analysis; curriculum as distribution scheduling. **C**: **reward-anatomy mixer** (sliders for each reward term, retrain preview of resulting gait caricature); gait phase-diagram animation (footfall patterns); terrain-curriculum progression dashboard; sim-to-real level tracker per published system. **P**: `rl-envs`: Ferris quadruped in `rapier3d` (12-DoF, PD low-level); PPO with `rayon`-parallel envs in `burn`; terrain generator with difficulty curriculum; `bevy` 3D visualization of learned gaits; WASM demo: walk trained Ferris across reader-drawn terrain.

### Chapter 19 — Learning Navigation & Mobile Manipulation
*Sources: Tang §4.2 (wheeled/legged/aerial navigation) + §4.4 (whole-body control, short/long-horizon MoMa).*
Navigation as the perception-heavy competency: local planning vs global planning, end-to-end vs hybrid-modular (Tang's evidence that neither wins universally); point-goal, social, and off-road navigation; legged navigation (locomotion + navigation composed hierarchically); aerial navigation; **mobile manipulation**: whole-body control (base + arm coordination), short-horizon skills vs long-horizon tasks; skill composition, hierarchical RL, and the "what skills should a robot learn?" open question (Tang §5 long-horizon challenge); Rusty's graduation: from Ch 4 gridworld to lidar-based continuous navigation.
**F**: POMDP navigation formalism (belief MDPs, recurrent policies as belief trackers); hierarchical/semi-MDP composition theory; whole-body kinematic redundancy resolution. **C**: **end-to-end vs modular pipeline switcher** (same scene, toggle architecture, compare failure modes); lidar-beam POV visualization of what the policy "sees"; hierarchical skill-graph animation for a long-horizon fetch task; social-navigation trajectory gallery. **P**: `rl-envs`: lidar Rusty in `rapier2d` rooms with obstacle randomization; recurrent PPO (GRU policy in `burn`); waypoint-conditioned skills + high-level skill selector (hierarchical); mobile-Reacher (Rusty base + Reacher arm) whole-body fetch task; `bevy` scene replays.

### Chapter 20 — Learning Manipulation
*Sources: Tang §4.3 (grasping, pick-and-place, contact-rich, in-hand, non-prehensile) + trends; Kober case-study lineage.*
Why manipulation is the hard one (contact richness, object diversity, perception coupling — Tang's maturity analysis); grasping: from analytic grasp metrics to learned grasp policies; end-to-end pick-and-place and why diversity keeps it below L3; contact-rich skills: assembly/insertion (impedance action spaces, force observations), articulated and deformable objects; **in-hand manipulation** (the OpenAI cube lineage: massive randomization + recurrent policies); non-prehensile manipulation (pushing, tossing); demos + offline RL as manipulation's preferred data diet (Tang Table 3/4 patterns); evaluation honesty: success-rate methodology.
**F**: grasp wrench space & force closure math; impedance-control formalism as action space; contact-mode combinatorics (why planning through contact explodes). **C**: **grasp wrench-space visualizer** (drag contact points on an object, watch force-closure region form); insertion funnel animation (impedance vs position control under uncertainty); randomization-breadth vs in-hand-success dashboard; contact-mode explosion counter. **P**: `rl-envs`: parallel-jaw gripper + objects in `rapier3d`; grasp-selection DQN (Ch 9 payoff) + continuous insertion with SAC + impedance action space (Ch 11/13 payoff); demo-boosted training from browser-teleop data (Ch 16 payoff); push-to-goal non-prehensile task; success-rate evaluation harness with confidence intervals.

---

## Part V — Frontiers & Capstone

### Chapter 21 — Frontiers: HRI, Multi-Robot & Foundation Models
*Sources: Tang §4.5 (HRI), §4.6 (multi-robot), §5 (open challenges), §6; Kober §8 open questions, cross-checked against the modern era.*
Human–robot interaction: collaborative & non-collaborative pHRI, shared autonomy, why HRI is simulation-starved (humans are the unmodelable part) and what that implies (real-world learning, offline data); multi-robot systems: Dec-POMDP formalism, CTDE, MARL instability, collision avoidance, robot soccer (L4); **foundation models meet robot RL**: LLM/VLM reward and goal specification, vision-language-action models, RL fine-tuning of large policies, generalist-robot debates; open-challenge synthesis merging Kober's 2013 questions with Tang's 2024 list — what got solved, what didn't, what's next; research compass for the reader.
**F**: Dec-POMDP formalism and complexity results; CTDE policy-gradient math (MAPPO sketch); RLHF objective (reward model + KL-regularized fine-tuning) derived. **C**: Kober-2013 vs Tang-2024 open-problem **diff dashboard** (solved / partially / open); Dec-POMDP coordination-failure animation (lazy-agent problem); shared-autonomy blending slider; VLA architecture explorer. **P**: multi-agent gridworld with CTDE PPO in `burn` (two Rustys coordinating a delivery); shared-autonomy Reacher (blend reader teleop with trained SAC assistance, live in WASM); candle-based small-VLM reward-labeling experiment (feasibility sketch).

### Chapter 22 — Capstone: An End-to-End Learned Robot in Rust
*Sources: everything; structured after Kober §7's case-study discipline, executed with Tang-era methods.*
One project, the whole book: specify a task (Ferris patrols a cluttered course, reaches goals, recovers from pushes), formalize the MDP (observation/action/reward choices justified against Ch 14–18 evidence), build the randomized `rapier3d` environment, train teacher–student PPO with terrain + dynamics randomization, distill to a deployable policy, evaluate with the L0–L5 rubric and honest statistics, and ship: **native binary + in-browser WASM demo with live telemetry dashboard** — the reader walks away with a complete, reproducible robot-learning pipeline in pure Rust.
**F**: the full formal problem statement, assembled from every prior chapter's notation — a worked example of writing the math of a real system. **C**: mission-control dashboard (training curves, randomization draws, evaluation heatmaps, gait telemetry — the book's visual finale); failure-mode gallery with post-mortems. **P**: the `capstone` crate: config-driven experiment runner (`serde` + `clap`), `rayon` rollout farm, `burn` training, `rerun` dev telemetry, `bevy`/WASM deployment, CI-reproducible seeds — a template repository readers fork for their own robots.

---

## Appendices

- **Appendix A — Rust for Scientists**: ownership for people who think in math; ndarray/nalgebra idioms; error handling in experiment code; WASM build pipeline.
- **Appendix B — The Crate Compendium**: every crate used, why it was chosen, alternatives (candle/tch-rs vs burn; avian vs rapier; egui vs bevy UI), version policy.
- **Appendix C — Notation & Math Reference**: unified symbol table (S&B-compatible), identity cheat-sheets (matrix calculus, expectations, KL/entropy), theorem index.

---

## Chapter file map

Design/storyline documents for each chapter live beside this file: `Chapter-1.md` … `Chapter-22.md`. Development conventions, toolchain details, and the chapter template live in `CLAUDE.md`.

| Part | Chapters | Theme |
|---|---|---|
| I | 1–7 | Foundations of sequential decision-making (S&B spine) |
| II | 8–12 | Function approximation & deep RL |
| III | 13–17 | The robotics bridge (Kober spine, modernized) |
| IV | 18–20 | Competencies: locomotion, navigation/MoMa, manipulation (Tang spine) |
| V | 21–22 | Frontiers & capstone |

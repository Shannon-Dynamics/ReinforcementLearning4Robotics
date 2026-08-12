# Chapter 17 — Motor-Skill Policy Representations

> **Part** III — The Robotics Side · **Builds on:** Ch 10 (PPO), Ch 11 (SAC & Reacher), Ch 13 (PD/LQR layers, manipulator dynamics), Ch 14 (dimensionality & goal-specification curses), Ch 15 (randomization, farm), Ch 16 (teleop demos, imitation initialization) · **Feeds:** Ch 18 (locomotion action spaces, CPG-informed gaits), Ch 19 (hierarchical skills via options), Ch 20 (impedance/task-space actions, primitives), Ch 22 (capstone action-space justification)
> **Modernizes:** Kober §4 (tractability through representation, esp. §4.3 pre-structured policies) and §7 (ball-in-a-cup case study, recreated end-to-end); Tang §3.2 (low/mid/high action-space axis) and §5 (principled action-space design as open challenge); Ijspeert, Nakanishi & Schaal 2003 / Schaal et al. 2007 (DMPs); Sutton, Precup & Singh 1999 (options); Johannink et al. 2019 / Silver et al. 2018 (residual RL)

## 1. Purpose & Learning Outcomes

Kober's deepest lesson survives the deep-learning era intact: *what the policy outputs matters as much as how it learns*. This chapter is the representation shop. We formalize Tang's action-space levels and race them on the same task; derive dynamic movement primitives in full and prove their stability; build central pattern generators for rhythm, residual policies for safety, and options for hierarchy; then close Part III by recreating Kober §7's ball-in-a-cup twice — with the 2013 pipeline (kinesthetic demo → DMP → policy search) and a 2020s pipeline (SAC from pixels + demos) — and scoring both honestly.

The reader can:
- Place any robot controller on Tang's low/mid/high action-space axis and predict the exploration, smoothness, and sim-to-real consequences of the choice.
- Derive the DMP equations (canonical system, transformed system, forcing term) and prove convergence to the goal via the vanishing-perturbation argument.
- Fit DMP weights to a single demonstration by locally weighted regression and generalize the movement to new goals and durations without retraining.
- Build a coupled phase-oscillator CPG and read a quadruped gait as a phase-offset matrix.
- State the residual-RL decomposition and its boundedness/stability argument, and the semi-MDP Bellman equations for options.
- Reproduce Kober's ball-in-a-cup result with ~100 episodes of policy search on DMP weights, run the modern pixels+demos pipeline on the same task, and defend a representation choice with measurements rather than fashion.

## 2. Storyline

**Act I — Three arms, one task, three fates.** The hook is a race the reader can't unsee: the identical SAC learner (Ch 11) trains Reacher to reach three times — once outputting joint torques, once PD joint targets, once task-space deltas through Ch 13's IK. Same algorithm, same reward, same seeds; wildly different curves, energies, and jerk profiles. The action space was the decision that mattered. Tang's low/mid/high taxonomy names what we saw, and Kober §4 explains *why*: representation is where robot priors enter the optimization.

**Act II — The representation shop.** Four structured outputs, each derived, visualized, built: DMPs (a demo becomes an attractor landscape — sculptor widget; stability proved, goal-dragging generalization live), CPGs (rhythm as coupled phase oscillators; a gait is a phase matrix — Ferris's future gaits previewed), residual RL (a learned Δ on Ch 13's controllers: keep the guarantees, learn the last 20 %), and options (temporal abstraction with real Bellman equations, feeding Ch 19's hierarchies). Running thread: each representation is a different answer to "how much do we trust our priors?"

**Act III — Ball-in-a-cup, then and now.** The book's inherited case study, rebuilt for real. 2013 lane: teleop demo (Ch 16 rig standing in for kinesthetic teach-in), DMP fit, episodic policy search on weights (PoWER derived; CMA-ES deployed — substitution noted honestly), Kober's rim-crossing reward, success in ~10² episodes. 2020s lane: SAC from 64×64 pixels + demos in the buffer (Ch 16), domain-randomized string physics (Ch 15), success in ~10⁵–10⁶ sim steps but with pixel inputs, goal-conditioning, and robustness the 2013 lane never had. The split-screen scoreboard renders the chapter's thesis: *structure buys sample efficiency; learning buys generality; modern systems are engineered mixtures* — exactly the mixture Part IV deploys per competency.

Robots: **Reacher** (action-space race; cup mounted on its wrist for ball-in-a-cup), **Ferris** (CPG preview of Ch 18 gaits), **Pendle** (residual-RL demo on the LQR baseline), **Rusty** (options cameo: warehouse skills as temporally extended actions).

## 3. Section-by-Section Design

### 17.1 Action Spaces: Tang's Low/Mid/High Axis
- **F:** Formal ladder with control-loop diagrams: low = joint torques $\tau$ at 100 Hz–1 kHz (policy inside the innermost loop); mid = setpoints for an inner controller — PD joint targets $q^d$ (locomotion's workhorse), task-space/impedance commands (manipulation's favorite, formalism deferred to Ch 20); high = temporally extended primitives/subroutines (grasp selection, DMP parameter vectors — the bandit limit of Ch 3). Consequences derived, not asserted: exploration noise filtered through an inner loop has bounded state-space effect (variance propagation through the PD transfer function, computed for a 1-DoF link); torque policies inherit the full sim-to-real actuation gap (Ch 15); high-level actions shorten horizons (Ch 14's $T^2$ factors shrink) at the price of expressiveness. Tang's empirical evidence table: quadrupeds → PD targets; drones → collective thrust + body rates; manipulation → task-space/impedance; and §5's verdict that principled selection remains open.
- **C:** `ch17-action-space-race`: the Act-I dashboard — three learning curves racing on identical seeds, with live strips of torque, jerk, and energy per candidate; the reader toggles PD gains and inner-loop rate to watch the mid-level lane's advantage grow or shrink.
- **P:** `rl-envs::reacher` gains `ActionSpace::{Torque, JointPd, TaskSpaceIk}` behind one enum; the race harness (pinned seeds, `SampleLedger`, `SuccessReport`) produces the dashboard data; results table quoted by Ch 18/20 when they pick their action spaces.

### 17.2 Dynamic Movement Primitives, Derived
- **F:** Full derivation in Kober §7.2's notation. Canonical system: $\dot z = -\tau\,\alpha_z z$, $z(0)=1$, $\tau = 1/T$ — time made a decaying resource. Transformed system per DoF: $\dot x_2 = \tau\big(\alpha_x(\beta_x(g - x_1) - x_2)\big) + \tau A f(z)$, $\dot x_1 = \tau x_2$, amplitude matrix $A=\mathrm{diag}(g - x_1^0)$ for spatial scaling. Forcing term: $f(z) = \sum_{i=1}^N \psi_i(z)\, w_i\, z$ with normalized Gaussian kernels $\psi_i(z) = \exp(-h_i(z-c_i)^2)/\sum_j \exp(-h_j(z-c_j)^2)$ — every symbol defined, kernel placement in log-time derived. **Learning from one demo by LWR:** invert the transformed system on demo data $(y,\dot y,\ddot y)$ to get $f^{\text{target}}(z_t)$, then per-kernel weighted least squares $w_i = \frac{\sum_t \psi_i(z_t)\, z_t f^{\text{target}}_t}{\sum_t \psi_i(z_t)\, z_t^2}$ — displayed and implemented. Meta-parameters ($g$, $\tau$, $A$) as the generalization interface; linearity in $w$ as the property that makes policy search cheap (§17.7).
- **C:** `ch17-dmp-sculptor`: the reader draws a trajectory with the mouse; basis activations $\psi_i(z)$ and fitted weights $w_i$ appear as live bar strips; then the payoff gestures — drag the goal $g$, stretch duration $\tau$, and watch the movement re-aim while keeping its shape; a perturbation button shoves the state mid-rollout and the attractor pulls it home.
- **P:** `rl-core::dmp` in pure `nalgebra` (sketch in §5): `Dmp::from_demo` (LWR), `Dmp::step` (semi-implicit Euler, consistent with Ch 15 §15.1), multi-DoF with shared canonical phase; `proptest` law: for zero forcing, convergence to $g$ from random initial conditions.

### 17.3 Why DMPs Cannot Miss: The Stability Proof
- **F:** **Theorem: for bounded weights, every DMP trajectory converges to $(g, 0)$.** Proof in three displayed stages. (1) Canonical subsystem: $z(t) = e^{-\tau\alpha_z t} \to 0$ exponentially. (2) Unforced transformed system is LTI with matrix $\begin{pmatrix}0 & \tau\\ -\tau\alpha_x\beta_x & -\tau\alpha_x\end{pmatrix}$; characteristic polynomial $s^2 + \tau\alpha_x s + \tau^2\alpha_x\beta_x$; with $\beta_x = \alpha_x/4$ the eigenvalue $-\tau\alpha_x/2$ is repeated and negative — critical damping derived, not decreed. (3) Forcing as vanishing perturbation: $|A f(z)| \le \|A\| \max_i |w_i|\, z \to 0$ since normalized kernels sum to one; the cascade (GAS driving system + exponentially stable driven LTI + vanishing input) converges by the ISS/comparison argument, written out with the explicit bound. Boundary of validity stated per Kober: stability of movement *generation*, not of the physical *execution* under contact. Rhythmic DMP variant in one paragraph: phase on a circle, forcing periodic in $\phi$ — the bridge to CPGs.
- **C:** The sculptor's "proof mode": overlays the three proof stages on a live rollout — $z$ decaying, the eigen-ellipse of the LTI part, the forcing envelope shrinking — so each displayed inequality has a moving picture.
- **P:** `proptest` upgrade: random weights within bound, random perturbation times → terminal error < tolerance in all cases; a deliberate violation (unnormalized kernels) fails the test — the proof's hypotheses made executable.

### 17.4 Central Pattern Generators: Rhythm as a Network
- **F:** Coupled phase oscillators: $\dot\phi_i = \omega_i + \sum_j K_{ij}\sin(\phi_j - \phi_i - \bar\psi_{ij})$; phase-locking condition for two oscillators derived (existence of fixed point in phase difference when $|\Delta\omega| \le$ coupling); amplitude dynamics via Hopf normal form $\dot r = a(\mu_r - r^2)r$ (limit-cycle convergence shown). A quadruped gait *is* the offset matrix $\bar\psi$: walk/trot/pace/bound written as four matrices; foot targets $p_i(t) = r_i(\cos\phi_i, \varepsilon\sin\phi_i)$ through the Ch 13 leg IK. Where learning enters (Tang mid/high axis): RL modulates $(\omega, \mu_r, \bar\psi)$ or adds a learned forcing on each oscillator — the action space Ch 18 will consider against plain PD targets.
- **C:** `ch17-cpg-network`: four oscillators as phase dots on circles, coupling springs drawn between them; a gait selector morphs the offset matrix and the footfall diagram scrolls underneath; the reader detunes one leg's $\omega$ and watches the network re-entrain — or fail, past the derived locking bound (shown live).
- **P:** `rl-core::cpg`: oscillator bank with configurable coupling graph (`nalgebra`), gait presets; demo binary: Ferris marionette — CPG drives leg targets through Ch 13 PD in the `bevy` viewer (open-loop, no learning yet; honest label: choreography, not locomotion — Ch 18 closes the loop).

### 17.5 Residual RL: Learn Only the Correction
- **F:** Decomposition $a = \pi_0(s) + \Delta_\theta(s)$ with $\pi_0$ a Ch 13 classical controller (Johannink et al. 2019; Silver et al. 2018). Stability argument, stated and proved at the ISS level: if the closed loop under $\pi_0$ admits a Lyapunov function with $\dot V \le -c_3\|x\|^2 + c_4\|x\|\,\|\Delta\|$ and the residual is bounded $\|\Delta_\theta\| \le \delta$ (tanh-squashed output scaled by $\delta$), then trajectories are ultimately bounded within radius $O(\delta)$ of the origin — the safety dial is the *architecture*, not a hope. Learning benefits derived: initialization at $\Delta \equiv 0$ starts at baseline performance (exploration around competence, not from scratch); the residual sees only what the model missed — Ch 15's under-modeling gap becomes the *learning target*. Failure mode: a fighting base controller (integrator wind-up against the residual) — when to residual, when to replace.
- **C:** `ch17-residual-dial`: Pendle under Ch 13 LQR near-upright with a deliberately wrong mass model; a $\delta$ slider grows the residual authority: at $\delta=0$ the steady-state error persists, small $\delta$ removes it, large $\delta$ visibly destabilizes learning transients — the ultimate-bound radius drawn as a shrinking/growing disk in phase space.
- **P:** `rl-core::wrappers::ResidualEnv` (wraps any `Env` + base controller, exposes bounded $\Delta$ as the action); experiment: SAC on residual vs SAC from scratch on perturbed-mass Pendle stabilization — samples-to-criterion and worst transient compared on the `SampleLedger`.

### 17.6 Options & the Semi-MDP View
- **F:** Options formalism (Sutton, Precup & Singh 1999): $o = (\mathcal I_o, \pi_o, \beta_o)$ — initiation set, intra-option policy, termination probability. The induced process over decision points is a semi-MDP; **Bellman equations displayed and derived:** $q_\Omega(s,o) = \mathbb E\big[R(s,o) + \gamma^k \max_{o'\in\mathcal O(s')} q_\Omega(s',o')\big]$ with multi-time reward model $R(s,o) = \mathbb E[r_{t+1} + \gamma r_{t+2} + \dots + \gamma^{k-1} r_{t+k}]$ and transition model $P(s',k\,|\,s,o)$ — the $\gamma^k$ making temporal abstraction mathematically first-class; optimality over $\Pi(\mathcal O)$ vs primitive optimality (what hierarchy can cost). Intra-option Q-learning update stated. DMPs/CPGs/residuals recognized as *options with hand-designed $\pi_o$* — the unifying sentence of the chapter. Option discovery flagged as open (Tang §5's "what skills should a robot learn?"), deferred to Ch 19/21.
- **C:** `ch17-options-timeline`: a Rusty warehouse episode rendered twice — as primitive steps (dense tick marks, $\gamma$ per tick) and as three options ("leave dock", "traverse aisle", "dock") with $\gamma^k$ discount arcs spanning each; hovering an option reveals its $(\mathcal I_o, \pi_o, \beta_o)$ triple and its multi-time model numbers.
- **P:** `rl-tabular::smdp`: options over the Ch 4 warehouse MDP, SMDP value iteration + intra-option Q-learning; test: hierarchy with good options beats primitive Q-learning in decision count, and a designed *bad* option set provably caps performance — both bounds from **F** reproduced numerically.

### 17.7 Ball-in-a-Cup, 2013: Demo → DMP → Policy Search
- **F:** The full Kober §7 pipeline, formalized. Task model in `rapier2d`: cup on Reacher's wrist, ball on a 40 cm string modeled per Kober §7.5 (pendulum + elastic string, switching to ballistic point mass when slack). Reward exactly Kober's: $r(t_c) = \exp(-\alpha(x_c - x_b)^2 - \alpha(y_c - y_b)^2)$ at the rim-crossing instant $t_c$ with downward ball velocity, zero otherwise, $\alpha = 100$ — with the §14.5 story of *why* (the hit-from-below exploit this reward was engineered to kill). Policy: one DMP per joint, weights $\theta \in \mathbb R^{2 \times N}$ ($N \approx 30$/DoF, Kober used 31 on 7 DoF = 217 params). **PoWER derived** as EM on the return-weighted exploration lower bound: $\theta' = \theta + \frac{\mathbb E[\sum_t \varepsilon_t\, Q^\pi(s_t,a_t,t)]_{\omega}}{\mathbb E[\sum_t Q^\pi(s_t,a_t,t)]_{\omega}}$ with parameter-space exploration $a = (\theta+\varepsilon_t)^\top\mu(s,t)$ and importance sampling over the 10 best episodes — each piece from Kober §7.2/7.4. Honest note: our implementation optimizes the same objective with CMA-ES (per TOC; black-box, no learning rate, same episodic regime); PoWER stated as the historical algorithm, one exercise implements it.
- **C:** `ch17-biac-2013` (left half of the split screen): demo replay → DMP fit → policy-search episodes ticking up with the swing evolving; the reward's rim-crossing instant flagged each episode; expected success onset around episode 40–80, matching Kober §7.6's real-robot numbers displayed alongside.
- **P:** `rl-envs::ball_in_cup` (string model, rim-crossing detector); `rl-deep::search::cmaes_policy` driving `rl-core::dmp` weights; the pipeline binary: record teleop demo (Ch 16 rig) → `Dmp::from_demo` → CMA-ES (population 16, ~100 episodes) → success; `SampleLedger` total published for §17.8's scoreboard.

### 17.8 Ball-in-a-Cup, 2020s: Pixels + Demos + Randomization
- **F:** The modern stack assembled from prior chapters, with its formulation choices justified on Tang's axes: observation = 64×64 grayscale stack (high-dim, no ball-tracker engineering — trades §14.5's reward instrumentation for representation learning), action = PD joint targets at 20 Hz (mid-level, §17.1's evidence), reward = same rim-crossing function (computed from sim state during training — privileged reward, Ch 15's asymmetric trick), demos in the replay buffer + BC-regularized SAC (Ch 16 §16.5), string-parameter randomization (Ch 15). Expected budget derived from published analogues (Schwab et al. 2019 ball-in-a-cup from pixels): order $10^5$–$10^6$ steps on the farm. The comparison protocol fixed *before* results: samples, wall-clock on the `rayon` farm, success CI, robustness to string-length shift, goal-shift generalization (move the cup), engineering hours logged.
- **C:** `ch17-biac-split`: the chapter's signature widget — 2013 lane and 2020s lane replayed side by side with live counters (episodes vs env-steps, both in `SampleLedger` units); then the two stress buttons: *shift the goal* (DMP re-aims via $g$; SAC needs the goal-conditioned variant or fails — shown), *lengthen the string* (randomized SAC shrugs; DMP policy misses until re-searched — shown). No winner banner; a trade-off table.
- **P:** Pixels+demos SAC config on the Ch 15 farm (CNN encoder in `burn`, demo buffer from the same teleop set as §17.7); both pipelines emit identical `SuccessReport`s; the honest scoreboard lands in the book as a table the reader can regenerate with two commands.

### 17.9 Chapter Bridge
- **F:** Part III closes. The representation shop's inventory mapped to Tang's axis and to each curse it mitigates (DMP/options → dimensionality & horizon; residual → under-modeling & safety; demo-init → samples; all → the priors-vs-generality dial). One paragraph states the Part IV contract: Ch 18 picks PD targets (+CPG comparison), Ch 19 composes options, Ch 20 chooses impedance/task-space and primitives — each citing §17.1's race and §17.8's scoreboard rather than fashion.
- **C:** Static map (mermaid): representation → chapters that deploy it; the ball-in-a-cup scoreboard reproduced as the closing figure of Part III.
- **P:** Handoff manifest: `dmp`, `cpg`, `ResidualEnv`, `smdp` land in `rl-core`/`rl-tabular` with doc examples; `ball_in_cup` joins `rl-envs`' permanent collection (Ch 22 uses it as a regression benchmark for the capstone toolchain).

## 4. Interactive Widgets & Dashboards

| ID | Type | What it shows | Reader interaction | Tech |
|---|---|---|---|---|
| `ch17-action-space-race` | dashboard | same SAC learner × three action spaces: learning curves + torque/jerk/energy strips | switch PD gains & inner-loop rate; replay any checkpoint's behavior | egui_plot + recorded runs + rapier2d (WASM) |
| `ch17-dmp-sculptor` | sandbox | drawn trajectory → live LWR fit ($\psi_i$, $w_i$ bars) → attractor rollout; goal/duration dragging; proof-mode overlays | draw, drag $g$ and $\tau$, perturb mid-rollout, toggle proof mode | rl-core::dmp (WASM) + egui_plot |
| `ch17-cpg-network` | animation | four coupled phase oscillators, gait offset matrices, scrolling footfall diagram, entrainment limits | select gait, detune $\omega_i$, edit coupling $K$, watch re-locking or failure | egui + custom canvas (WASM) |
| `ch17-residual-dial` | widget | LQR-with-wrong-mass Pendle + bounded residual; ultimate-bound disk growing with $\delta$ | drag residual bound $\delta$, re-run learning transient, compare phase portraits | rapier2d + burn inference (WASM) |
| `ch17-options-timeline` | animation | one episode as primitive ticks vs three options with $\gamma^k$ arcs and multi-time models | hover options for $(\mathcal I_o,\pi_o,\beta_o)$; scrub the episode | egui + rl-tabular (WASM) |
| `ch17-biac-2013` | animation | demo → DMP fit → CMA-ES episodes to success, Kober's real-robot numbers alongside | step episodes, inspect reward instant $t_c$, reseed search | rapier2d (WASM) + egui |
| `ch17-biac-split` | widget | then-vs-now split screen: budgets in shared units, goal-shift and string-shift stress tests | play both lanes, press stress buttons, sort the trade-off table | rapier2d + burn inference (WASM) + egui |

## 5. Rust Implementation Plan

**Crates touched:** `rl-core` (`dmp`, `cpg`, `wrappers::ResidualEnv`), `rl-tabular` (`smdp`: SMDP VI + intra-option Q), `rl-envs` (`reacher::ActionSpace`, `ball_in_cup`), `rl-deep` (`search::cmaes_policy` bridging the `cmaes` crate to policy weights; CNN-SAC config; BC-regularizer reuse from Ch 16), `rl-sim` (farm + randomization reused unchanged), `demos/ch17-*` (seven WASM crates).

**New modules/files:** `rl-core/src/dmp.rs`, `rl-core/src/cpg.rs`, `rl-core/src/wrappers/residual.rs`, `rl-tabular/src/smdp.rs`, `rl-envs/src/ball_in_cup/{mod.rs,string.rs,reward.rs}`, `rl-deep/src/search/cmaes_policy.rs`.

Representative sketch — the DMP core (pure `nalgebra`, shared canonical phase across DoFs):

```rust
/// rl-core/src/dmp.rs — Ijspeert/Schaal discrete DMP, Kober §7.2 notation.
pub struct Dmp {
    pub alpha_z: f64,                     // canonical decay: ż = −τ α_z z
    pub alpha_x: f64, pub beta_x: f64,    // spring–damper; β_x = α_x/4 ⇒ critical damping
    pub tau: f64,                         // temporal scaling τ = 1/T
    pub y0: DVector<f64>, pub goal: DVector<f64>,
    pub centers: DVector<f64>, pub widths: DVector<f64>,  // kernel c_i, h_i in phase space
    pub weights: DMatrix<f64>,            // N_basis × n_dof, from LWR or CMA-ES
}

impl Dmp {
    /// f(z) = Σ_i ψ_i(z) w_i z with normalized Gaussian kernels ψ_i.
    fn forcing(&self, z: f64) -> DVector<f64> {
        let psi = self.centers.zip_map(&self.widths, |c, h| (-h * (z - c).powi(2)).exp());
        (self.weights.tr_mul(&psi)) * (z / psi.sum())
    }
    /// One semi-implicit Euler step; returns desired q̈ for the PD layer beneath.
    pub fn step(&self, st: &mut DmpState, dt: f64) -> DVector<f64> {
        st.z += -self.tau * self.alpha_z * st.z * dt;
        let amp = &self.goal - &self.y0;                        // A = diag(g − y0)
        let acc = ( (&self.goal - &st.y) * self.beta_x - &st.dy / self.tau ) * self.alpha_x
                    * self.tau + amp.component_mul(&self.forcing(st.z)) * self.tau;
        st.dy += &acc * dt;  st.y += &st.dy * dt;
        acc
    }
    pub fn from_demo(demo: &Trajectory, n_basis: usize) -> Self { /* LWR fit, §17.2 */ }
}
```

**Experiments/benchmarks:** (1) the three-way action-space race (pinned seeds); (2) DMP `proptest` stability laws + LWR reconstruction error vs $N$; (3) CPG entrainment boundary vs derived locking condition; (4) residual-vs-scratch on perturbed Pendle; (5) SMDP option ablation on the warehouse; (6) ball-in-a-cup 2013 lane (CMA-ES, ~100 episodes, 5 seeds); (7) 2020s lane on the farm; (8) the joint scoreboard with CIs. `criterion` on `Dmp::step` (must be sub-microsecond: it runs inside 1 kHz loops in Ch 18).

**Native vs browser:** both ball-in-a-cup training lanes native (2013 lane is light enough for a laptop CPU; 2020s lane needs the farm + WGPU); every widget in-browser — the sculptor, CPG net, and 2013 pipeline run *live* in WASM (rapier2d + `rl-core::dmp` are tiny), the split screen replays recorded checkpoints for the SAC lane.

## 6. Robot Thread

- **Reacher** (from Ch 16: teachable, with teleop pipeline) — the chapter's protagonist: races three action spaces, then wears the cup. After: Reacher has performed the book's flagship case study under two paradigms and carries a permanent `ball_in_cup` benchmark.
- **Pendle** (from Ch 13/15: LQR baseline, engine-understood) — residual-RL testbed; after: the wrong-model steady-state error it carried since Ch 13 is finally learned away, safely.
- **Ferris** (from Ch 15: stands, farmed) — CPG marionette in the `bevy` viewer: choreographed footfall patterns, no closed-loop balance yet; the explicit cliffhanger for Ch 18.
- **Rusty** (from Ch 5 warehouse) — options cameo: its gridworld skills become the SMDP worked example that Ch 19 scales up.

## 7. Exercises & Explorations

1. **(F)** Show that $\beta_x = \alpha_x/4$ is exactly critical damping for the unforced transformed system, and derive the overshoot that appears when $\beta_x > \alpha_x/4$ and the sluggish tail when $\beta_x < \alpha_x/4$; verify both regimes in the sculptor.
2. **(F)** Prove spatial-scaling invariance: with $A = \mathrm{diag}(g - y_0)$, show a weight vector fit on demo $(y_0, g)$ reproduces the same *shape* (time-normalized trajectory, affinely mapped) for any new $(y_0', g')$; find the degenerate case $g' \to y_0'$ and its standard fix.
3. **(F)** Derive the two-oscillator phase-locking condition: for $\dot\phi_{1,2}$ with coupling $K\sin(\phi_2-\phi_1\mp\bar\psi)$, reduce to a 1-D ODE in $\Delta\phi$ and give the existence/stability condition of its fixed point in terms of $\Delta\omega/K$.
4. **(F)** From the option models $R(s,o)$ and $P(s',k|s,o)$, derive the SMDP Bellman *optimality* equation from first principles (mirror the Ch 4 derivation, tracking $\gamma^k$), and verify your equation against `rl-tabular::smdp` on a 3-state chain.
5. **(C)** In the sculptor, fit the same drawn trajectory with $N \in \{5, 15, 50\}$ kernels; relate the reconstruction error curve to Kober's report that ~31 parameters per DoF sufficed and that 3× more only slowed convergence slightly.
6. **(C)** Use the race dashboard to make the torque lane *win*: find the (gain, control-rate) setting where the inner PD loop hurts more than it helps, and explain the result via the variance-propagation formula of §17.1.
7. **(P)** Add Pastor et al.'s obstacle-avoidance coupling term $\varphi(x) $ to `Dmp::step` (repulsive rotation of $\dot y$ around a point obstacle) and show goal convergence is preserved (the term vanishes at zero velocity) both by argument and by `proptest`.
8. **(P)** Implement PoWER exactly as derived in §17.7 (state-dependent exploration, 10-best importance sampler) and race it against CMA-ES on the 2013 lane at equal episode budget; reproduce Kober's observation that the importance sampler's greediness is the sensitive knob.
9. **(P)** Goal-shift study on the split screen's data: move the cup 5 cm and measure episodes (DMP lane: re-search from re-aimed $g$) vs env-steps (SAC lane: fine-tune) to recover 80 % success; add the two numbers to the scoreboard table.

## 8. Notation Introduced

| Symbol | Meaning |
|---|---|
| $z$, $\alpha_z$ | DMP canonical phase and its decay rate |
| $\tau$ (DMP) | temporal scaling $\tau = 1/T$ — Kober §7.2's usage; the joint-torque vector is always bold $\boldsymbol\tau$ (disambiguation registered for Appendix C) |
| $\alpha_x, \beta_x$ | transformed-system spring–damper gains ($\beta_x = \alpha_x/4$ critical) |
| $\psi_i(z), c_i, h_i, w_i$ | normalized Gaussian kernels, centers, widths, forcing weights |
| $g$, $A$, $f(z)$ | goal, amplitude matrix $\mathrm{diag}(g-y_0)$, forcing term |
| $\phi_i, \omega_i, K_{ij}, \bar\psi_{ij}$ | oscillator phase, intrinsic frequency, coupling gain, gait phase offsets |
| $\pi_0$, $\Delta_\theta$, $\delta$ | base controller, learned residual, residual bound |
| $o = (\mathcal I_o, \pi_o, \beta_o)$ | option: initiation set, intra-option policy, termination function |
| $q_\Omega(s,o)$, $R(s,o)$, $P(s',k\|s,o)$ | option-value function and multi-time option models |
| $t_c$ | rim-crossing instant in the ball-in-a-cup reward |

## 9. References & Further Reading

- **Kober, Bagnell & Peters (IJRR 2013)** — §4.1–4.3 (discretization, value-function approximation, pre-structured policies: via-points, motor primitives, locally linear controllers), §7.1–7.7 (ball-in-a-cup end to end: task/reward, DMP representation, teach-in, PoWER, simulation use, real-robot results, Nemec's value-based alternative).
- **Tang et al. (2024)** — §3.2 (action-space axis), §4.1/4.3 (PD targets in locomotion; thrust-and-body-rates for drones; task-space/impedance in manipulation), §5 (principled action-space design as open challenge; benchmark studies cited therein).
- **Sutton & Barto** — §3–4 (Bellman machinery the SMDP equations extend), §13 (policy-gradient view of parameter-space search).
- Ijspeert, Nakanishi & Schaal (2003); Schaal et al. (2007); Ijspeert et al. (2013) — DMPs. Pastor et al. (2009) — DMP obstacle coupling. Kober & Peters (2009, 2010) — PoWER and ball-in-a-cup.
- Ijspeert (2008) — CPG review (salamander lineage). Kohl & Stone (2004) — Aibo gait search. Righetti & Ijspeert (2008) — pattern generators with feedback.
- Sutton, Precup & Singh (1999) — options/semi-MDPs. Bacon, Harb & Precup (2017) — option-critic (learned options, Ch 19/21 pointer).
- Johannink et al. (2019) — residual RL. Silver et al. (2018) — residual policy learning.
- Peng & van de Panne (2017) — action parameterization matters. Aljalbout et al. (2024) — action-space study for robot RL. Kaufmann et al. (2023) — champion-level drone racing (action-space choice at L5). Schwab et al. (2019) — ball-in-a-cup from pixels on a real robot (the 2020s lane's closest published relative). Hansen (2016) — CMA-ES tutorial.

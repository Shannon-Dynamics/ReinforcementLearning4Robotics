# Chapter 14 — The Four Curses of Robot RL

> **Part** III — The Robotics Side · **Builds on:** Ch 5 (GPI & the dimensionality preview), Ch 10–11 (PPO/SAC), Ch 13 (`rl-sim`, Reacher, classical baselines) · **Feeds:** Ch 15 (under-modeling → sim-to-real), Ch 16 (demonstrations vs the sample curse), Ch 17 (representations vs dimensionality), Ch 18–20 (reward anatomy per competency), Ch 22 (L0–L5 evaluation rubric)
> **Modernizes:** Kober §3.1–3.4 (the four curses — the heart of the 2013 survey); Tang §3.4 (levels of real-world success) and §5 (real-world learning, benchmarking real-world success); Ng, Harada & Russell 1999 (shaping invariance); Altman 1999 (CMDPs)

## 1. Purpose & Learning Outcomes

Parts I–II built algorithms that assume cheap samples, a faithful simulator, modest state spaces, and a given reward. This chapter names the four ways robots violate those assumptions — Kober's curses of **dimensionality**, **real-world samples**, **under-modeling**, and **goal specification** — quantifies each with math and a live experiment, and adds the modern extensions the 2013 survey could not have: reward-hacking galleries, constrained MDPs for safe exploration, automatic-reset economics, and Tang's L0–L5 levels recast as an evaluation rubric the rest of the book will apply to every claim (including its own).

The reader can:
- State all four curses precisely, with the Kober §3 examples (ball-paddling's 20+7 dimensions, pole-reset labor, the elastic-string sim that lied) and map each to the chapter that counters it (Ch 15/16/17).
- Derive the exponential covering-number bound $N(\varepsilon) = \lceil 1/(2\varepsilon)\rceil^d$ and quote the tabular sample-complexity lower bound $\tilde\Omega\!\big(|\mathcal S||\mathcal A|\,(1-\gamma)^{-3}\varepsilon^{-2}\big)$, explaining what each factor costs a robot.
- Prove the potential-based shaping invariance theorem and use it to accelerate a sparse-reward task without moving the optimum.
- Diagnose reward hacking from behavior, and repair a hacked reward with shaping, constraints, or re-specification.
- Formulate a safety requirement as a CMDP, write its Lagrangian, and explain how dual ascent turns SAC into Lagrangian-SAC.
- Grade any robot-RL paper or demo on the L0–L5 rubric and audit its reset/supervision costs.

## 2. Storyline

**Act I — The invoice.** The hook is an itemized bill. You take the Ch 11 SAC agent that mastered simulated Reacher in 400k steps and price the same run on a physical arm: at 20 Hz with 8-second episodes, 400k steps is ~5.5 hours of *motion* — plus 2,500 manual resets, bearing wear, one melted gearbox, and a graduate student. Kober's four curses are introduced as the four line items on this invoice, each cross-referenced to where the book already brushed against it (Ch 5's state-space explosion experiment; Ch 12's model-bias horror stories).

**Act II — Four audits.** Each curse gets a quantified audit. Dimensionality: the Ch 5 preview becomes a theorem-grade wall — covering numbers, the lower bound, and the ladder Reacher (8 state-action dims) → Kober's ball-paddler (27) → Ferris-to-come (49). Real-world samples: wear, resets, delays, and real-time budgets are formalized (delay-augmented MDPs, reset-cost ledgers). Under-modeling: the compounding-error bound from Ch 12 is re-read as a *transfer* bound, with Kober's self-stabilizing vs unstable task dichotomy demonstrated on Pendle. Goal specification: the reader designs rewards for Reacher in the reward-hacking zoo and watches trained SAC agents exploit every loophole — orbiting, hovering, slamming.

**Act III — The toolkit and the rubric.** The chapter turns from diagnosis to instruments: the shaping theorem (proved in full) gives a *license* to add dense guidance without corrupting the optimum; CMDPs give safety a formalism instead of a hope; curricula schedule task distributions; and Tang's L0–L5 levels become the book's standing evaluation rubric — applied first to the six zoo agents, later to Ferris in Ch 22. Closing beat: each curse now has a named antidote chapter, and Part III's arc is set.

Robots: **Reacher** (reward lab, CMDP prototype), **Pendle** (delay and stability demos), **Rusty** (shaping widget on the warehouse gridworld — a deliberate callback to Ch 4–5). Ferris is *named* in the dimensionality ladder but does not appear until Ch 15.

## 3. Section-by-Section Design

### 14.1 The Invoice: Why Naive RL Dies on Contact with Hardware
- **F:** The robot-RL problem restated as a POMDP with continuous $\mathcal S,\mathcal A$, fixed control rate, sensing/actuation delays, and non-resettable dynamics; formal definitions of *sample cost* (wall-clock + labor + wear per transition) and *reset cost*. Kober's §3 inventory mapped onto this formalism, one clause per curse.
- **C:** The "invoice" widget prelude: an animated cost ledger that replays the Ch 11 SAC Reacher learning curve with a second y-axis in hours, resets, and euros; the reader toggles control rate and episode length and watches the bill change.
- **P:** `rl-core::metrics` gains `SampleLedger` (per-episode: env steps, wall-clock, resets, interventions, cumulative torque²·dt as wear proxy); every later experiment in the book reports it. Retro-instrument the Ch 11 SAC run.

### 14.2 The Curse of Dimensionality, Quantified
- **F:** Bellman's curse made precise. (i) Discretization: $d$ dimensions at resolution $k$ gives $k^d$ cells — Kober's ball-paddling example computed ($2\times(7+3)=20$ state dims, 7 action dims). (ii) Covering numbers: an $\varepsilon$-cover of $[0,1]^d$ in $\ell_\infty$ needs $N(\varepsilon)=\lceil 1/(2\varepsilon)\rceil^d$ points — derived. (iii) Statement of the generative-model lower bound $\tilde\Theta\big(\tfrac{|\mathcal S||\mathcal A|}{(1-\gamma)^3\varepsilon^2}\big)$ (Azar et al. 2013; proof cited, not reproduced) and what each factor means physically. (iv) The escape routes named — function approximation (Ch 8), structure in policies (Ch 17), demonstrations that localize search (Ch 16, Kakade–Langford argument previewed from Kober §5.1).
- **C:** `ch14-exponential-wall`: drag sliders for dimension $d$ and bins-per-dimension $k$; log-scale bars show table cells, bytes, and years-to-visit-once at 20 Hz; overlaid markers for Rusty's gridworld (Ch 4), Pendle (4-D), Reacher (8-D), Kober's paddler (27-D), Ferris (49-D). The wall goes vertical between the markers the reader has already trained and the ones they haven't.
- **P:** Extend Ch 5's state-space-explosion experiment: tabular Q-learning on Reacher discretized at $k\in\{3,5,9,17\}$ bins; `criterion` memory/time table plus final-policy quality vs the Ch 11 SAC baseline — the numbers behind the wall widget are the reader's own.

### 14.3 The Curse of Real-World Samples
- **F:** Wear, non-stationarity ("tracking solutions"), and supervision from Kober §3.2 formalized. Delay-augmented MDP: with $k$-step actuation delay the Markov state must include $(a_{t-k},\dots,a_{t-1})$, multiplying state dimension — the delay/Markov trade-off derived. Real-time constraint: decision latency must fit the control period $1/f_c$; episodic vs continuing settings for when learning can pause. Automatic resets as a *second* MDP (Tang §5 real-world learning): reset policy, reset detection, and the human-in-the-loop fallback.
- **C:** `ch14-reset-ledger`: a dashboard replaying training runs as a strip chart of episodes with color-coded reset events (auto vs human), cumulative intervention count, and a "graduate-student-hours" odometer; reader compares a self-resetting Pendle to a Reacher that entangles.
- **P:** Delay experiment: wrap Reacher in `DelayedEnv{k}` (action queue in `rl-envs`), train SAC for $k\in\{0,1,2,4\}$, plot degradation; then recover performance by state augmentation and report the dimensionality price on the `SampleLedger`.

### 14.4 The Curse of Under-Modeling & Model Uncertainty
- **F:** Kober §3.3 in Ch 12's language: if the one-step model error is $\varepsilon_m$ in total variation, the $T$-step state-distribution error grows as $O(T\varepsilon_m)$ and the policy-value transfer gap as $O(T^2\varepsilon_m)$ (bound restated from Ch 12 and re-derived in two displayed lines). The self-stabilizing/unstable dichotomy: for exponentially stable closed loops, perturbation bounds keep sim and real trajectories close (ISS-style statement); for unstable equilibria (pole balancing) no such bound exists — Kober's ball-paddling elastic string vs pole-balancing contrast retold.
- **C:** `ch14-transfer-gap`: animation of the same policy rolled out in "sim" Pendle and three "real" Pendles (mass ±5 %, friction ×2, delay +1 step); trajectories fan apart; a toggle switches between the stabilizing task (damped pendulum settling) and the unstable one (inverted balance) so the reader *sees* the dichotomy.
- **P:** Quantify: evaluate the Ch 11 SAC Reacher policy across a grid of perturbed `rl-sim` physics params; heatmap of return vs (mass scale, friction); this artifact becomes the "before" picture that Ch 15's domain randomization must fix.

### 14.5 The Curse of Goal Specification: Reward Design & the Hacking Zoo
- **F:** Reward as the interface between intent and optimization (Kober §3.4). Formal statement of reward misspecification: designer's true objective $J^\star$ vs proxy $J_R$; hacking = $\arg\max_\pi J_R$ far from $\arg\max_\pi J^\star$. Taxonomy with equations for each zoo exhibit: sparse success, dense distance, distance+effort, velocity-toward-goal (the orbit exploit computed: circular motion keeps $\dot x^\top(x^\star\!-\!x)/\|x^\star\!-\!x\|$ positive forever), shaped sparse, minimum-time. Kober's ball-in-a-cup reward trick (nonzero only at rim-crossing with downward velocity) previewed for Ch 17.
- **C:** `ch14-reward-hacking-zoo`: gallery of six trained SAC Reacher agents, one per reward; the reader picks a reward, watches the behavior, then sees two curves — proxy reward (climbing) vs true success rate (flat or falling). A "design your own" panel lets the reader compose reward terms and submits them to a pre-trained bank of nearest-neighbor behaviors.
- **P:** The reward-design lab: `rl-envs::reacher::RewardSpec` enum with the six functions; train six SAC agents (seeds pinned, configs in `serde` TOML); export WASM inference bundles for the zoo gallery; a `criterion`-timed harness reports proxy return *and* ground-truth success for each.

### 14.6 Shaping Without Regret: The Potential-Based Invariance Theorem
- **F:** **Theorem (Ng–Harada–Russell 1999), proved in full.** Let $M=(\mathcal S,\mathcal A,P,R,\gamma)$ and $M'$ identical except $R'(s,a,s') = R(s,a,s') + F(s,a,s')$ with $F(s,a,s') = \gamma\Phi(s') - \Phi(s)$ for any $\Phi:\mathcal S\to\mathbb R$ (episodic case: $\Phi(\text{terminal})=0$). Proof: the shaped return telescopes, $G'_t = \sum_k \gamma^k\big[R_{t+k+1} + \gamma\Phi(S_{t+k+1}) - \Phi(S_{t+k})\big] = G_t - \Phi(S_t)$, hence $q'_\pi(s,a) = q_\pi(s,a) - \Phi(s)$ and $v'_\pi(s) = v_\pi(s) - \Phi(s)$ for every $\pi$; the subtraction is action-independent, so greedy sets, policy orderings, and optimal policies coincide. Converse stated and proved by the original counterexample construction: any non-potential $F$ can change the optimal policy for some $P,R$. Corollary (Wiewiora 2003): Q-learning with shaping $\Phi$ ≡ Q-learning with table initialized at $\Phi$. Choice of $\Phi \approx v^\star$ as the ideal; $\Phi = -c\,\|x - x^\star\|$ as the practical default.
- **C:** `ch14-shaping-invariance`: Rusty's Ch 4 warehouse; the reader *paints* a potential $\Phi$ heatmap with the mouse; the widget overlays the induced $F$ arrows, re-solves the MDP live (Ch 5 machinery), and shows $\pi^\star$ unchanged — then a "break it" switch adds a non-potential bonus (e.g., +0.1 for entering a corridor) and the optimal policy visibly deforms.
- **P:** Add `ShapedReward<Φ>` wrapper to `rl-core`; property test with `proptest`: for random small MDPs and random $\Phi$, exact $\pi^\star$ (via Ch 5 value iteration) is invariant. Then the payoff experiment: sparse-reward Reacher (zoo exhibit 1) + distance potential trains in a fraction of the steps and reaches the *same* behavior — measured on the `SampleLedger`.

### 14.7 Safety as a Constraint: CMDPs & Lagrangian-SAC
- **F:** CMDP formalism (Altman 1999): tuple $(\mathcal S,\mathcal A,P,R,\{c_i\},\{d_i\},\gamma)$; maximize $J_R(\pi)$ subject to $J_{c_i}(\pi) = \mathbb E_\pi\big[\sum_t \gamma^t c_i\big] \le d_i$. Lagrangian $\mathcal L(\pi,\lambda) = J_R(\pi) - \sum_i \lambda_i\big(J_{c_i}(\pi) - d_i\big)$; occupancy-measure LP view shows the constrained problem stays linear ⇒ zero duality gap for stochastic policies (stated with citation). Dual ascent: $\lambda \leftarrow \big[\lambda + \eta_\lambda(\hat J_c - d)\big]_+$, interpretation of $\lambda^\star$ as the shadow price of safety. Lagrangian-SAC: safety critic $Q_c$, actor loss $\mathbb E[\alpha\log\pi - Q_R + \lambda Q_c]/(1+\lambda)$. Contrast with shaping: a cost penalty folded into reward has a *fixed* exchange rate; the multiplier *finds* the exchange rate that meets the budget.
- **C:** `ch14-cmdp-frontier`: animation of Lagrangian-SAC training Reacher under a joint-velocity constraint: left, behavior with a translucent constraint boundary in task space; right, live traces of $J_R$, $J_c$ vs budget line $d$, and $\lambda$ breathing — overshoot, correction, convergence onto the boundary. A slider moves $d$ and replays from checkpoints, tracing the Pareto front.
- **P:** `rl-deep::cmdp`: `CostCritic` (twin of Ch 11's Q nets), `LagrangianDual` (§5 sketch); prototype on Reacher with cost $c=\mathbf 1[\|\dot q\|>\dot q_{\max}]$; deliverable table: unconstrained SAC vs fixed-penalty vs Lagrangian at three budgets — violations/episode and return.

### 14.8 Curricula, Resets & the L0–L5 Evaluation Rubric
- **F:** Curriculum as distribution scheduling: task distribution $p_k(\xi)$ indexed by difficulty, with the scheduling objective (train where success is neither 0 nor 1 — learnability band) formalized; forward reference to Ch 18's terrain curricula. Evaluation methodology: Tang's six levels defined verbatim (L0 sim-only → L5 commercial deployment) and operationalized as a rubric: conditions tested, number/diversity of trials, confidence intervals on success rates (Wilson interval given), reset and supervision disclosure. The rubric is the book's standing contract — Ch 22 grades Ferris with it.
- **C:** `ch14-l0l5-rubric`: interactive rubric dashboard seeded with the Ch 1 survey systems (drone racing L5, ANYmal L4, in-hand cube L2…); the reader grades the six zoo agents and two published papers from their claimed evidence; hovering a level shows Tang's definition and a canonical example.
- **P:** `rl-core::eval`: `SuccessReport` (n trials, successes, Wilson 95 % CI, conditions descriptor, `SampleLedger` totals) with `serde` export; wired into the zoo harness so every gallery card carries its CI — no bare success percentages anywhere in the book from here on.

### 14.9 Chapter Bridge
- **F:** One-table recap: curse → formal core → antidote chapter (dimensionality → §14.2 → Ch 17 representations; samples → §14.3 → Ch 16 demonstrations & offline data, Ch 15 parallel sim; under-modeling → §14.4 → Ch 15 sim-to-real; goal specification → §14.5–14.7 → shaping + CMDP + Ch 16 IRL).
- **C:** Static curse-antidote map (mermaid) reused as Part III's navigation graphic.
- **P:** Checklist artifact: `curse_audit.toml` template the reader fills for their own task; Ch 22 opens by filling one for Ferris. Next stop: inside the simulator (Ch 15).

## 4. Interactive Widgets & Dashboards

| ID | Type | What it shows | Reader interaction | Tech |
|---|---|---|---|---|
| `ch14-exponential-wall` | widget | $k^d$ growth: cells, bytes, years-to-visit at 20 Hz; robot markers on the ladder | drag $d$ and $k$ sliders; toggle log/linear; hover markers for each robot's dims | egui + egui_plot |
| `ch14-reset-ledger` | dashboard | episode strip chart with auto/human reset events, intervention odometer, wear proxy | scrub runs, compare two tasks, toggle cost model (time/labor/wear) | egui + serde-loaded run logs |
| `ch14-transfer-gap` | animation | sim vs perturbed-real trajectory fans on Pendle; stable vs unstable task dichotomy | pick perturbation (mass/friction/delay); switch task; replay | rapier2d (WASM) + egui |
| `ch14-reward-hacking-zoo` | gallery | six SAC Reacher behaviors, proxy-reward vs true-success curves per reward | select reward, watch live policy, compose custom reward terms | rapier2d + burn (WASM inference) + egui |
| `ch14-shaping-invariance` | widget | painted potential $\Phi$, induced shaping arrows, live re-solved $\pi^\star$ invariant; non-potential bonus breaks it | paint $\Phi$ on Rusty's warehouse; toggle "break it" switch | egui + rl-tabular (WASM) |
| `ch14-cmdp-frontier` | animation | Lagrangian-SAC traces: $J_R$, $J_c$ vs budget, $\lambda$ dynamics, constraint boundary in task space | move budget $d$ slider, replay from checkpoints, trace Pareto front | egui_plot + recorded checkpoints |
| `ch14-l0l5-rubric` | dashboard | L0–L5 definitions, graded systems, CI-carrying success reports | grade zoo agents and papers; filter by competency | egui + serde |

## 5. Rust Implementation Plan

**Crates touched:** `rl-core` (metrics: `SampleLedger`, `eval::SuccessReport`, `ShapedReward` wrapper), `rl-envs` (`reacher::RewardSpec`, `DelayedEnv`), `rl-deep` (`cmdp` module: `CostCritic`, `LagrangianDual`, Lagrangian-SAC trainer reusing Ch 11 SAC), `rl-sim` (perturbed-physics evaluation grid), `demos/ch14-*` (seven WASM crates).

**New modules/files:** `rl-core/src/metrics/ledger.rs`, `rl-core/src/eval.rs`, `rl-core/src/shaping.rs`, `rl-envs/src/reacher/reward.rs`, `rl-envs/src/wrappers/delay.rs`, `rl-deep/src/cmdp/{mod,dual,trainer}.rs`.

Representative sketch — the dual controller that turns Ch 11's SAC into Lagrangian-SAC:

```rust
/// rl-deep/src/cmdp/dual.rs — one multiplier per constraint, λ ≥ 0 by log-parameterization.
pub struct LagrangianDual {
    log_lambda: f64,
    budget: f64,   // d: allowed (undiscounted-average) episode cost
    lr: f64,       // η_λ
}

impl LagrangianDual {
    /// Dual-ascent step after each evaluation batch: λ ← [λ + η_λ (Ĵ_c − d)]₊ .
    pub fn update(&mut self, mean_episode_cost: f64) -> f64 {
        let violation = mean_episode_cost - self.budget;
        self.log_lambda += self.lr * violation * (-self.log_lambda).exp().min(1.0);
        self.lambda()
    }
    pub fn lambda(&self) -> f64 { self.log_lambda.exp() }
}

// In the SAC actor step (burn tensors; q_r, q_c from reward/cost critics):
// L_actor = E[ α·log π(a|s) − Q_r(s,a) + λ·Q_c(s,a) ] / (1 + λ)
let lam = dual.lambda() as f32;
let actor_loss = (alpha * log_pi - q_r + q_c * lam).mean() / (1.0 + lam);
```

**Experiments/benchmarks:** (1) tabular-Reacher discretization sweep (`criterion`, memory + return table); (2) delay sweep $k\in\{0,1,2,4\}$ with/without state augmentation; (3) perturbed-physics evaluation heatmap of the Ch 11 policy; (4) six-reward SAC zoo with pinned seeds; (5) shaped-vs-sparse sample-count comparison; (6) CMDP budget sweep. All emit `SampleLedger` + `SuccessReport` JSON consumed by the widgets.

**Native vs browser:** all training native (`burn` WGPU); zoo/CMDP/transfer widgets ship recorded checkpoints + WASM inference; shaping and wall widgets run fully in-browser (tabular solves are milliseconds).

## 6. Robot Thread

- **Reacher** (from Ch 11: trained SAC policy exists) — becomes the chapter's lab animal: six reward variants, delay wrappers, CMDP constraints. After: Reacher has a *calibrated* cost-aware training pipeline and a perturbation heatmap exposing its sim-fragility (handed to Ch 15).
- **Pendle** (from Ch 13: rapier articulated body + LQR baseline) — demonstrates delays and the stable/unstable transfer dichotomy. Unchanged after, but its transfer fan is the motivating exhibit for Ch 15.
- **Rusty** (from Ch 5: warehouse MDP solved exactly) — cameo in the shaping widget, where exact re-solves make invariance visibly checkable.
- **Ferris** — named on the dimensionality ladder (49-D) as a promise; first appearance Ch 15.

## 7. Exercises & Explorations

1. **(F)** Episodic shaping: show that if $\Phi(\text{terminal})\neq 0$, the invariance theorem fails, and exhibit a 3-state counterexample where shaping changes $\pi^\star$. Then prove the Wiewiora 2003 equivalence between shaping and Q-table initialization.
2. **(F)** Derive the orbit exploit: for reward $r = \dot x^\top(x^\star - x)/\|x^\star - x\|$ on a point mass, construct a circular trajectory whose return exceeds that of any trajectory that stops at $x^\star$, for suitable $\gamma$.
3. **(F)** In the CMDP Lagrangian, prove that at an optimal saddle point either $J_c(\pi^\star)=d$ or $\lambda^\star=0$ (complementary slackness), and interpret $\lambda^\star$ as marginal return per unit of relaxed budget.
4. **(C)** Use `ch14-exponential-wall` to find the largest $d$ for which a 10-bin table of `f32` values fits in 16 GB; check it against the displayed formula, then find where *your* laptop dies in the §14.2 experiment.
5. **(C)** In the zoo, design a reward that makes Reacher spin permanently while proxy reward climbs; then repair it *only* by adding a potential-based term and verify in the widget that the repaired optimum matches the sparse-reward optimum.
6. **(P)** Add a jerk constraint ($c = \|\dddot q\|^2$ above a threshold) as a second CMDP constraint; extend `LagrangianDual` to a vector of multipliers and plot the 2-constraint Pareto surface.
7. **(P)** Implement an automatic-reset detector for Reacher (entanglement = end-effector below the base for >1 s) and report how the `SampleLedger` human-intervention count changes; compare to Tang §5's real-world-learning discussion in one paragraph.

## 8. Notation Introduced

| Symbol | Meaning |
|---|---|
| $N(\varepsilon)$ | $\varepsilon$-covering number of the state (or state-action) space |
| $\Phi(s)$ | shaping potential; $F(s,a,s') = \gamma\Phi(s') - \Phi(s)$ the shaping function |
| $R'$, $q'_\pi$, $v'_\pi$ | reward / value functions of the shaped MDP $M'$ |
| $c_i(s,a)$, $d_i$ | $i$-th constraint cost and budget in a CMDP |
| $J_{c}(\pi)$ | expected discounted constraint cost $\mathbb E_\pi[\sum_t \gamma^t c(S_t,A_t)]$ |
| $\lambda_i$, $\mathcal L(\pi,\lambda)$ | Lagrange multiplier(s); CMDP Lagrangian |
| $\varepsilon_m$ | one-step model error (total variation), from Ch 12, reused for transfer bounds |
| L0–L5 | Tang levels of real-world success, used as evaluation rubric |

All S&B-compatible; registered for Appendix C. $\Phi$ is reserved for potentials (never CDFs) from here on.

## 9. References & Further Reading

- **Kober, Bagnell & Peters (IJRR 2013)** — §3.1 curse of dimensionality (ball-paddling dims), §3.2 real-world samples (wear, resets, delays, real-time), §3.3 under-modeling (elastic string vs pole balance), §3.4 goal specification (ball-in-a-cup reward, IRL pointer); §5.1 (demonstrations remove global exploration — bridge to Ch 16).
- **Tang et al. (2024)** — §3.4 levels of real-world success (L0–L5 definitions); §5 real-world learning (resets, safe exploration), benchmarking real-world success (evaluation protocols).
- **Sutton & Barto** — §4 (DP complexity, GPI), §13 (policy gradient objectives referenced in CMDP form).
- Bellman (1957) — the original curse. Azar, Munos & Kappen (2013) — minimax sample-complexity bounds. Kakade & Langford (2002) — state-distribution knowledge makes RL tractable.
- Ng, Harada & Russell (1999) — potential-based shaping invariance (+ necessity). Wiewiora (2003) — shaping ≡ initialization.
- Altman (1999) — *Constrained Markov Decision Processes*. Achiam et al. (2017) — CPO. Ray, Achiam & Amodei (2019) — Safety Gym & Lagrangian baselines. Paternain et al. (2019) — zero duality gap for CMDPs.
- Amodei et al. (2016) — concrete AI-safety problems (reward hacking). Skalse et al. (2022) — defining reward hacking. Ibarz et al. (2021) — lessons from real-robot deep RL (resets, wear, evaluation honesty).

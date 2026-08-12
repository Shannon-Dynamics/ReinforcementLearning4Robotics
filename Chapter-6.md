# Chapter 6 — Learning from Experience: Monte Carlo & Temporal-Difference

> **Part** I — Foundations of Sequential Decision-Making · **Builds on:** Ch 3 (ε-greedy, incremental estimates), Ch 4 (MDP formalism, `rl-core` `Env` trait), Ch 5 (GPI, greedy improvement) · **Feeds:** Ch 7 (n-step, traces, Dyna extend TD), Ch 8 (semi-gradient TD), Ch 9 (Q-learning → DQN, Double Q → Double DQN), Ch 11 (overestimation → TD3's clipped double-Q), Ch 16 (importance sampling → offline RL)
> **Modernizes:** S&B 2nd ed. ch. 5–6 unified into one experience-learning arc; + Double Q-learning (van Hasselt 2010) taught tabularly before its deep reprise; + robot-flavored sensor-noise ablations Kober's sample-cost arguments demand

## 1. Purpose & Learning Outcomes

The blueprint is gone: warehouse maintenance rearranged the shelving and nobody updated the model. Rusty keeps only the ability to *act* and *observe* — the `Env` trait instead of `Mdp`. This chapter builds the entire model-free toolkit: Monte Carlo methods that learn from complete returns, TD methods that learn from single transitions via the TD error $\delta_t$, and the on-/off-policy distinction that will govern every deep-RL design choice in Part II.

- The reader can prove first-visit MC is unbiased and explain why every-visit MC is biased yet consistent.
- The reader can derive the importance-sampling identity, contrast ordinary vs weighted estimators, and exhibit a case of infinite variance.
- The reader can define $\delta_t$, derive the MC-error-as-sum-of-TD-errors identity, and articulate the bias–variance trade between MC and TD(0).
- The reader can implement Sarsa, Expected Sarsa, Q-learning, and Double Q-learning over any `rl-core` `Env` and state their convergence conditions precisely (GLIE; Watkins/Robbins–Monro).
- The reader can explain maximization bias via $\mathbb{E}[\max_a Q(a)] \ge \max_a \mathbb{E}[Q(a)]$ and show how the double estimator removes it.
- The reader can run a sensor-noise ablation and report how observation corruption degrades MC vs TD differently.

## 2. Storyline

**Act 1 — The blueprint is revoked (hook).** Rusty's Ch 5 policy assumed the model; the rearranged warehouse silently invalidates it. Rusty can still drive and still gets rewards — experience is available, the model is not. First idea: estimate values the way Ch 3 estimated arm values — average what you observe. Averaging *returns* from full episodes is Monte Carlo, and GPI (Ch 5's master pattern) still turns those estimates into control. But MC must wait for episode end, and exploration must be engineered (exploring starts, then ε-soft).

**Act 2 — Learning during the drive (development).** The pivotal move: bootstrap. TD(0) updates from a single transition using its own estimate — the TD error $\delta_t$ is born, and the TD-error-river widget shows an MC return is nothing but TD errors summed along the trajectory. Control follows in two flavors that look almost identical in code and behave completely differently: Sarsa evaluates the policy Rusty *actually follows* (noise, ε-blunders and all); Q-learning evaluates the greedy policy while behaving otherwise. The loading-dock experiment (S&B's cliff, warehouse edition) makes the distinction physical: Q-learning hugs the dock edge and falls during exploration; Sarsa learns the safe aisle.

**Act 3 — The max that lies (payoff + modern touch).** Off-policy learning has a subtle pathology: taking a max over noisy estimates inflates them. The casino widget shows Q-learning repeatedly choosing a provably worse aisle because *some* noisy sample looked good. Double Q-learning — two estimators, one to choose, one to judge — fixes it, and the reader is told plainly: this exact figure, redrawn with neural networks, is the Double DQN paper (Ch 9), and the same bias resurfaces in continuous control as TD3's clipped double-Q (Ch 11). Closing ablation: corrupt Rusty's odometry with noise and watch MC and TD degrade differently — the first honest contact with Kober's real-world-samples curse (Ch 14).

Running robot: **Rusty** throughout; sensor-noise ablation foreshadows the POMDP honesty of Ch 4 becoming operational in Ch 19.

## 3. Section-by-Section Design

### 6.1 When the Blueprint Is Gone
- **F:** Formal shift from planning to learning: the agent interacts via $S_t, A_t, R_{t+1}$ samples drawn from an unknown $p$; definition of *prediction* (estimate $v_\pi$) vs *control* (find $\pi_*$) from experience; GPI restated with sampled evaluation. Why $q_\pi$ replaces $v_\pi$ as the object of interest: greedification w.r.t. $v$ needs the model, greedification w.r.t. $q$ does not.
- **C:** Split-screen animation: Ch 5's dashboard (model queries, sweeping all states) vs the new regime (one Rusty, one trajectory, only visited states can learn) — the "fog of experience" rendered literally as unvisited cells staying dark.
- **P:** Recap of `rl_core::Env` (`reset() -> Obs`, `step(action) -> (Obs, f64, bool)`); the warehouse re-exposed model-free as `rl_envs::warehouse::WarehouseEnv` with seeded RNG and an `obs_noise: f64` corruption knob (used in §6.8's ablation); `rl-tabular/src/mc.rs` opened.

### 6.2 Monte Carlo Prediction
- **F:** First-visit MC: $V(s) = \frac{1}{N(s)}\sum_{i} G^{(i)}(s)$ over first visits; proof of unbiasedness (each first-visit return is an i.i.d. draw with mean $v_\pi(s)$) and consistency by SLLN; every-visit MC: returns within an episode are correlated → biased for finite $N$, still consistent (statement + citation). Incremental form $V(s) \leftarrow V(s) + \frac{1}{N(s)}[G_t - V(s)]$ and constant-$\alpha$ variant for nonstationarity (Ch 3 machinery reused). No bootstrapping: variance of $G_t$ grows with episode length and reward stochasticity — derived for an i.i.d.-reward chain: $\mathrm{Var}[G_t] = \sum_{k\ge 0}\gamma^{2k}\sigma_r^2$.
- **C:** `ch06-mc-vs-td-credit` (left panel active): a full Rusty episode plays; nothing updates until the terminal flag, then every visited cell flashes and jumps toward its return simultaneously — credit arrives late but whole.
- **P:** `mc::FirstVisitMc` and `mc::EveryVisitMc` (prediction over a fixed policy); test: on the warehouse with the Ch 5 exact $v_\pi$ as oracle, RMS error decays as $O(1/\sqrt{\text{episodes}})$.

### 6.3 Monte Carlo Control: Exploring Starts & ε-Soft Policies
- **F:** MC-ES (evaluate $q$, greedify, exploring starts assumption stated honestly: unavailable to a physical robot — you cannot teleport Rusty into every state–action). ε-soft fix: full derivation of the ε-greedy policy improvement theorem — for ε-greedy $\pi'$ w.r.t. $q_\pi$ of an ε-soft $\pi$: $\sum_a \pi'(a|s)q_\pi(s,a) = \frac{\varepsilon}{|\mathcal{A}|}\sum_a q_\pi(s,a) + (1-\varepsilon)\max_a q_\pi(s,a) \ge v_\pi(s)$, using the weighted-average inequality $\max_a q \ge \sum_a \frac{\pi(a|s)-\varepsilon/|\mathcal{A}|}{1-\varepsilon} q_\pi(s,a)$; conclusion via Ch 5 Ex. 3's stochastic improvement theorem: best-in-class among ε-soft policies.
- **C:** Exploring-starts absurdity card: Rusty teleported mid-air into "crashing into shelf 7 at full speed" as a start state — a one-panel cartoon that makes the assumption's physical impossibility stick before the math replaces it.
- **P:** `mc::McEsControl` (for the widget's idealized mode) and `mc::EpsilonSoftMc`; experiment: ε ∈ {0.01, 0.1, 0.3} vs episodes-to-90%-optimal on the warehouse, 50 seeds via `rayon`.

### 6.4 Off-Policy Learning & Importance Sampling
- **F:** Behavior policy $b$, target policy $\pi$, coverage assumption $\pi(a|s)>0 \Rightarrow b(a|s)>0$; the ratio $\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}$; derivation of $\mathbb{E}_b[\rho_{t:T-1} G_t \mid S_t{=}s] = v_\pi(s)$ (trajectory-probability ratio: transition terms cancel). Ordinary IS (unbiased, variance can be unbounded) vs weighted IS (biased, consistent, bounded weights); variance analysis: $\mathrm{Var}_b[\rho G] = \mathbb{E}_b[\rho^2 G^2] - v_\pi^2$, with S&B's one-state infinite-variance example (Example 5.5) worked symbol-by-symbol; why products of ratios explode with horizon — the robotics moral: long-horizon off-policy MC is statistically radioactive (Ch 16's offline-RL problem in embryo).
- **C:** `ch06-is-lab` widget: sliders shape $\pi$ (near-greedy) and $b$ (uniform-ish) over the warehouse's actions; 1 000 replicate estimates histogrammed for ordinary vs weighted IS side by side; running max-$\rho$ ticker; a preset button reproduces the infinite-variance example and the ordinary-IS histogram visibly refuses to settle.
- **P:** `mc::OffPolicyMc { kind: IsKind::Ordinary | Weighted }` with incremental weighted-IS update $V \leftarrow V + \frac{W}{C}(G - V)$; experiment logs estimator variance vs horizon length.

### 6.5 TD(0) and the TD Error
- **F:** The update $V(S_t) \leftarrow V(S_t) + \alpha\,\delta_t$ with $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$; identity (derived, and load-bearing for Ch 7): for $V$ held fixed, $G_t - V(S_t) = \sum_{k=t}^{T-1}\gamma^{k-t}\delta_k$ — MC error is the discounted sum of TD errors. Bias–variance: TD target biased by bootstrap, variance of a single transition; convergence of tabular TD(0) to $v_\pi$ under Robbins–Monro conditions $\sum\alpha_t = \infty, \sum \alpha_t^2 < \infty$ (Ch 2's stochastic approximation theorem instantiated; full proof cited). Batch TD(0) computes the certainty-equivalence MDP estimate while batch MC minimizes training-set squared error — S&B §6.3's you-are-the-predictor argument reproduced.
- **C:** `ch06-td-river` animation: a trajectory drawn as a river; at each transition a droplet of $\delta_t$ (diverging colormap: red positive, blue negative, sized by $|\delta_t|$) appears, and with learning frozen the droplets flow backward and stack into exactly $G_t - V(S_t)$ — the identity animated. Toggle learning on: droplets shrink episode by episode as $V \to v_\pi$. Right panel of `ch06-mc-vs-td-credit` activates: cells update *during* the drive, one step behind Rusty.
- **P:** `rl-tabular/src/td.rs`: `td::Td0` prediction; every `update` returns $\delta_t$ (the dashboard subscribes to it — the style guide's "live quantities the math named" made concrete).

### 6.6 Sarsa & Expected Sarsa: On-Policy Control
- **F:** Sarsa update $Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]$; convergence theorem statement (Singh, Jaakkola, Littman & Szepesvári 2000): $Q \to q_*$ w.p. 1 under GLIE policies (greedy in the limit, infinite exploration) + Robbins–Monro steps. Expected Sarsa: target $R_{t+1} + \gamma\sum_a \pi(a|S_{t+1})Q(S_{t+1},a)$; derivation that it has the same fixed point with strictly lower target variance (the $A_{t+1}$ sampling noise integrated out), at $O(|\mathcal{A}|)$ extra cost.
- **C:** `ch06-dock-edge` widget (the loading-dock/cliff experiment, staged early here, revisited in §6.7): warehouse aisle along an open loading dock; falling costs $-100$ and a reset. Sarsa's learned path arcs safely one aisle in; per-episode online reward plotted live. ε slider; toggle to Expected Sarsa shows the same route with a visibly smoother learning curve.
- **P:** `td::Sarsa`, `td::ExpectedSarsa` over `Env`; shared `EpsGreedy` action-selection helper (imported from Ch 3's `rl-core::policy`), ε-decay schedules as `enum Schedule { Const(f64), LinearTo(f64, usize), InverseVisit }`.

### 6.7 Q-Learning: Off-Policy TD Control
- **F:** Update $Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[R_{t+1} + \gamma\max_a Q(S_{t+1},a) - Q(S_t,A_t)]$; why it is off-policy (target = greedy, behavior = anything with coverage) *without* importance ratios (one-step bootstrapping never multiplies ratios — contrast with §6.4). **Watkins Q-learning convergence theorem (stated precisely):** for a finite MDP with bounded rewards, if every state–action pair is updated infinitely often and step sizes $\alpha_t(s,a)$ satisfy $\sum_t \alpha_t(s,a) = \infty$, $\sum_t \alpha_t(s,a)^2 < \infty$, then $Q_t \to q_*$ with probability 1 (Watkins & Dayan 1992). Proof sketch: write the update as a stochastic approximation of the contraction $T^*$ ($\max$-operator from Ch 5); the noise term $R_{t+1} + \gamma\max_a Q(S_{t+1},a) - (T^*Q)(S_t,A_t)$ is zero-mean and bounded-variance; invoke Jaakkola–Jordan–Singh (1994). On-policy vs off-policy consequence demonstrated on the dock: Q-learning's greedy target values the edge-hugging path even though its ε-greedy *behavior* keeps falling off — online return lower, learned policy shorter.
- **C:** `ch06-dock-edge` completes: both learned paths overlaid (Sarsa's safe arc vs Q-learning's edge hug); online-reward curves cross exactly as S&B's cliff figure predicts; setting ε → 0 slowly makes both converge to the edge path — GLIE made tangible.
- **P:** `td::QLearning` (code sketch §5); `criterion` bench: updates/sec for Sarsa vs Q-learning vs Expected Sarsa (the $\max$ vs expectation vs sample cost, measured).

### 6.8 Maximization Bias & Double Q-Learning
- **F:** The bias: for noisy estimates, $\mathbb{E}[\max_a Q(a)] \ge \max_a \mathbb{E}[Q(a)]$ (Jensen applied to the convex max); worked two-state example (S&B Example 6.7 numbers): actions from the "casino aisle" have true mean $-0.1$ but $\max$ over noisy samples looks positive, so Q-learning detours. **Double estimator math (van Hasselt 2010):** maintain independent $Q_1, Q_2$; estimate $\max_a \mathbb{E}[Q(a)]$ by $Q_2(\text{argmax}_a Q_1(a))$; proof that this is unbiased-or-underestimating: $\mathbb{E}[Q_2(a^*_1)] = \sum_a \Pr[a^*_1{=}a]\,\mathbb{E}[Q_2(a)] \le \max_a \mathbb{E}[Q(a)]$ by independence. Double Q-learning update (coin-flip which table updates, the other supplies the evaluation); convergence to $q_*$ retained (van Hasselt's theorem, statement + citation). Forward pointers made explicit: Double DQN (Ch 9), clipped double-Q in TD3 (Ch 11). Closing ablation: `obs_noise` sweep — TD's bootstrap partially launders per-step observation noise while MC integrates it over whole returns; measured, plotted, and connected to Kober §3.2's real-world-samples curse.
- **C:** `ch06-max-bias-casino` widget: the casino aisle drawn as a row of slot machines (arms ~ $\mathcal{N}(-0.1, 1)$); live plot of "% episodes entering the casino" for Q-learning vs Double Q vs the 5% ε-floor optimum; sliders for arm count and reward σ show bias growing with both — more options to be wrong about, more noise to be fooled by.
- **P:** `td::DoubleQLearning` (two `Array2<f64>` tables, seeded coin); ablation runner `examples/sensor_noise_ablation.rs`: MC vs TD(0) prediction error vs `obs_noise` ∈ {0, 0.05, 0.1, 0.2}, 100 seeds, `rayon`; results feed the chapter's closing figure.

### 6.9 Chapter Bridge
- **F:** Recap table: {MC, TD(0), Sarsa, Expected Sarsa, Q-learning, Double Q} × {target, bias, variance, on/off-policy, convergence conditions}. The gap named: MC and TD are two ends of a spectrum — what lives between?
- **C:** Teaser frame of Ch 7's λ-dial widget at its two endpoints, labeled "you already know both ends."
- **P:** `ch06-train-live` dashboard shipped as the chapter's capstone artifact; `rl-tabular::td` API frozen — Ch 7 subclasses nothing, it *composes* (n-step and traces wrap the same $\delta_t$).

## 4. Interactive Widgets & Dashboards

| ID | Type | What it shows | Reader interaction | Tech |
|---|---|---|---|---|
| `ch06-mc-vs-td-credit` | animation | Same recorded episode on two synced maps: MC updates all-at-once at episode end vs TD updating one step behind Rusty; RMS-error-vs-episodes race below | Step/auto-play; episode scrubber; α slider per side | egui, WASM |
| `ch06-td-river` | animation | Trajectory as a river; $\delta_t$ droplets (diverging colormap, sized by magnitude) flowing backward and stacking into $G_t - V(S_t)$ | Freeze/unfreeze learning; hover a droplet for its arithmetic; episode counter | egui, WASM |
| `ch06-is-lab` | widget | Ordinary vs weighted IS estimate histograms over 1 000 replicates; running variance; max-$\rho$ ticker | Shape $\pi$ and $b$; horizon slider; infinite-variance preset | egui_plot, WASM |
| `ch06-dock-edge` | widget | Loading-dock gridworld: Sarsa's safe arc vs Q-learning's edge path; live online-reward curves | ε slider + decay toggle; algorithm toggles (Sarsa / Expected Sarsa / Q); fall counter | egui, WASM |
| `ch06-max-bias-casino` | widget | % of episodes detouring into the negative-mean casino aisle, per algorithm, vs episodes | Arm count and σ sliders; Q vs Double-Q toggle; seed field | egui_plot, WASM |
| `ch06-train-live` | dashboard | **Capstone.** Live Q-heatmap + greedy arrows (Ch 5's `rl-viz` components reused), per-episode return, ε schedule, $\delta$ magnitude histogram, fall/collision counters | Choose algorithm; α, γ, ε-schedule controls; `obs_noise` slider (the ablation, live); pause/inspect any cell's $q$ values | egui + egui_plot, WASM |

Static fallbacks: captioned PNGs per widget; `ch06-dock-edge` fallback reproduces the S&B cliff figure with warehouse art.

## 5. Rust Implementation Plan

**Crates touched:** `rl-tabular` (modules `mc.rs`, `td.rs` added), `rl-core` (uses `Env`, `policy::EpsGreedy`, `Schedule`), `rl-envs` (`WarehouseEnv` gains `obs_noise` and the dock-edge map variant `warehouse::dock_edge()`), `rl-viz` (reuses `ValueHeatmap`/`PolicyArrows`; adds `DeltaHistogram`).

**Modules/files:** `rl-tabular/src/mc.rs` (`FirstVisitMc`, `EveryVisitMc`, `McEsControl`, `EpsilonSoftMc`, `OffPolicyMc`, `IsKind`), `rl-tabular/src/td.rs` (`Td0`, `Sarsa`, `ExpectedSarsa`, `QLearning`, `DoubleQLearning`, shared `TdConfig { alpha, gamma, schedule }`), `rl-tabular/examples/sensor_noise_ablation.rs`, `rl-tabular/benches/td_updates.rs`, `demos/ch06-*/` (six demo crates).

Representative code sketch (Q-learning; `update` returns $\delta_t$ for the dashboards):

```rust
// rl-tabular/src/td.rs
use ndarray::Array2;
use rand::Rng;
use rl_core::policy::eps_greedy;

pub struct QLearning {
    pub q: Array2<f64>, // [n_states, n_actions]
    pub alpha: f64,
    pub gamma: f64,
    pub epsilon: f64,
}

impl QLearning {
    pub fn act<R: Rng>(&self, s: usize, rng: &mut R) -> usize {
        eps_greedy(self.q.row(s), self.epsilon, rng)
    }

    /// One transition. Returns the TD error δ_t — dashboards subscribe to it.
    pub fn update(&mut self, s: usize, a: usize, r: f64, s2: usize, done: bool) -> f64 {
        let bootstrap = if done {
            0.0
        } else {
            self.q.row(s2).iter().copied().fold(f64::NEG_INFINITY, f64::max)
        };
        let delta = r + self.gamma * bootstrap - self.q[[s, a]];
        self.q[[s, a]] += self.alpha * delta;
        delta
    }
}
```

**Experiments/benchmarks:** (1) MC RMS error $O(1/\sqrt{n})$ verification; (2) ε-schedule study (§6.3); (3) IS variance vs horizon; (4) dock-edge online-return comparison, 100 seeds; (5) casino bias curves; (6) sensor-noise ablation; (7) `criterion` update-cost bench. **Native:** all experiments (seeded, `rayon`-parallel replicates). **In-browser:** all six widgets; `ch06-train-live` trains in real time in WASM (~10⁵ updates/s is ample for an 80-state warehouse).

## 6. Robot Thread

- **Rusty** (primary): enters optimal-under-a-stale-model, exits having *learned* the rearranged warehouse from rollouts alone — including a safe dock route (Sarsa) and an aggressive one (Q-learning), and knowing which of its own estimators to distrust ($\max$ over noise). The `obs_noise` knob is Rusty's first honest sensor; it stays in the codebase for Ch 19's lidar upgrade.
- **Pendle:** absent (returns for function approximation in Ch 8, where its continuous state defeats these tables). **Reacher, Ferris:** absent.

## 7. Exercises & Explorations

1. **(F)** Prove that every-visit MC is biased for finite $N$ using a two-visit episode in a one-state MDP with $\gamma=1$, then show consistency as $N \to \infty$.
2. **(F)** Complete the trajectory-probability derivation of $\mathbb{E}_b[\rho_{t:T-1}G_t|S_t{=}s] = v_\pi(s)$, marking exactly where the unknown transition probabilities cancel — the reason IS needs no model.
3. **(F)** Show that the one-step Q-learning target needs no importance ratio even under an arbitrary behavior policy with coverage, and identify the precise step where the *n*-step version would break (you will fix it in Ch 7).
4. **(F)** For two actions with values $\mathcal{N}(0, \sigma^2)$-estimated, compute $\mathbb{E}[\max(\hat Q_1, \hat Q_2)] = \sigma/\sqrt{\pi}$ and check it against `ch06-max-bias-casino` with matching parameters.
5. **(C)** In `ch06-is-lab`, find behavior/target pairs where *weighted* IS visibly beats ordinary IS, then a pair where they are indistinguishable. State the rule you infer about $\max\rho$.
6. **(C)** Using `ch06-dock-edge`, find the largest constant ε for which Q-learning's *online* return still beats Sarsa's. Explain the answer via the fall counter.
7. **(P)** Implement `td::NStepPeek`: a Sarsa variant whose target uses two real steps $R_{t+1} + \gamma R_{t+2} + \gamma^2 Q(S_{t+2}, A_{t+2})$. Compare learning curves on the warehouse — you have just built the first rung of Ch 7's ladder.
8. **(P)** Extend `sensor_noise_ablation.rs` with a *reward*-noise sweep and show (plot + one paragraph) why reward noise hurts MC and TD symmetrically while observation noise does not.

## 8. Notation Introduced

| Symbol | Meaning |
|---|---|
| $\delta_t$ | TD error $R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ (action-value form analogous) |
| $N(s)$, $N(s,a)$ | visit counts |
| $b(a\vert s)$ | behavior policy (data-generating); $\pi$ reserved for the target policy |
| $\rho_{t:T-1}$ | importance-sampling ratio $\prod_{k=t}^{T-1}\pi(A_k\vert S_k)/b(A_k\vert S_k)$ |
| $W, C$ | incremental weighted-IS weight and cumulative weight |
| $Q_1, Q_2$ | the two tables of the double estimator |
| GLIE | greedy in the limit with infinite exploration (convergence condition) |

$G_t$, $v_\pi$, $q_\pi$, $\alpha$, $\varepsilon$, $\gamma$ carry over from Ch 3–5 unchanged (Appendix C is arbiter).

## 9. References & Further Reading

- **S&B 2nd ed. ch. 5**: §5.1 MC prediction (first-visit); §5.3–5.4 MC control, exploring starts, ε-soft improvement derivation; §5.5 importance sampling incl. the infinite-variance Example 5.5; §5.6 incremental weighted IS; §5.7 off-policy MC control.
- **S&B 2nd ed. ch. 6**: §6.1–6.2 TD(0) and its advantages; §6.3 batch TD / certainty equivalence; §6.4 Sarsa; §6.5 Q-learning and the cliff example; §6.6 Expected Sarsa; §6.7 maximization bias and double learning. *(Numbering note: the repo's in-progress-draft PDF matches §5.1–5.7 and §6.1–6.5, but Expected Sarsa and §6.7's double learning appear only in the published 2nd edition — cite the published numbering.)*
- Sutton, R. (1988). "Learning to predict by the methods of temporal differences." *Machine Learning* 3 — TD(0)'s origin.
- Watkins, C. (1989). *Learning from Delayed Rewards*. PhD thesis, Cambridge — Q-learning.
- Watkins, C. & Dayan, P. (1992). "Q-learning." *Machine Learning* 8 — the convergence theorem of §6.7.
- Jaakkola, T., Jordan, M. & Singh, S. (1994). "On the convergence of stochastic iterative dynamic programming algorithms." *Neural Computation* 6 — the stochastic-approximation scaffold of the proof sketch.
- Tsitsiklis, J. (1994). "Asynchronous stochastic approximation and Q-learning." *Machine Learning* 16 — alternative proof route.
- Rummery, G. & Niranjan, M. (1994). "On-line Q-learning using connectionist systems." Cambridge TR 166 — Sarsa's origin.
- Singh, S., Jaakkola, T., Littman, M. & Szepesvári, C. (2000). "Convergence results for single-step on-policy reinforcement-learning algorithms." *Machine Learning* 38 — GLIE conditions (§6.6).
- van Seijen, H., van Hasselt, H., Whiteson, S. & Wiering, M. (2009). "A theoretical and empirical analysis of Expected Sarsa." IEEE ADPRL — variance result of §6.6.
- van Hasselt, H. (2010). "Double Q-learning." *NeurIPS* 23 — §6.8's estimator and convergence theorem; deep reprise in Ch 9 (Double DQN, van Hasselt et al. 2016).
- Kober, J., Bagnell, J. A. & Peters, J. (2013). IJRR survey, §3.2 "Curse of Real-World Samples" — why rollout-hungry MC is expensive on hardware; frames the sensor-noise ablation.

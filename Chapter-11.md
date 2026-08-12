# Chapter 11 — Off-Policy Continuous Control: DDPG, TD3 & SAC

> **Part** II — Scaling Up: Function Approximation & Deep RL · **Builds on:** Ch 3, 6, 8, 9, 10 · **Feeds:** Ch 12, 14, 16, 17, 20, 21
> **Modernizes:** Kober §3.2 (curse of real-world samples — the economic argument for off-policy learning); Tang §3.3 solution-approach axis (off-policy policy optimization) and Tang's open-challenge finding that on-policy sample efficiency is the bottleneck for real-world RL; extends S&B ch. 6 off-policy TD and ch. 13 policy gradients into the deep continuous-control era.

## 1. Purpose & Learning Outcomes

PPO (Ch 10) is a data glutton: it uses each rollout once and throws it away. Real robots pay for every rollout in time, wear, and risk — Kober §3.2's curse of real-world samples, restated by Tang 2024 as the central obstacle to real-world learning. This chapter builds the off-policy continuous-control family that reuses every transition: DDPG, its stabilized successor TD3, and the maximum-entropy workhorse SAC. Reacher — glimpsed in Ch 3 choosing grasp primitives as a bandit — makes its true debut as a continuous-control robot.

The reader can:
- Derive the deterministic policy gradient theorem and explain why it needs off-policy exploration.
- Reproduce the overestimation-bias analysis and justify each of TD3's three fixes from it.
- State the maximum-entropy RL objective, derive the soft Bellman operator, and prove soft policy iteration converges in the tabular setting.
- Derive the reparameterization-trick gradient, including the tanh-squashing log-density correction, and contrast its variance with the score-function estimator.
- Derive the automatic temperature-tuning objective from the entropy-constrained formulation.
- Implement DDPG, TD3, and SAC in `burn` on shared replay/target infrastructure, and train Reacher to track targets.
- Choose between PPO and SAC for a given robot problem using an explicit sample-cost/stability protocol.

## 2. Storyline

**Act 1 — The bill for on-policy learning.** You reprint Ch 10's PPO swing-up bill: ~1M environment steps. At Pendle's 50 Hz that is fictional-cheap in simulation and 5.5 hours of continuous hardware time in reality — before a single hyperparameter retry. Kober's argument lands: every real rollout wears bearings, needs resets, risks damage. The obvious fix is to reuse old data — but Ch 10 proved the policy gradient is an expectation *under the current policy*. The chapter's driving question: how do you learn a continuous-action policy from a replay buffer full of stale actions?

**Act 2 — Determinism and its discontents.** If the policy is deterministic, a = μ_θ(s), the expectation over actions disappears and the gradient can ride on Q_φ learned off-policy — the DPG theorem. Bolting on Ch 9's DQN machinery gives DDPG: it solves Reacher... on seed 3 of 5. The `ch11-q-landscape` widget shows why: Q_φ grows a spurious mountain, μ_θ climbs it, the critic bootstraps from the inflated peak — Ch 6's maximization bias reborn in continuous action space. TD3's three fixes are surgical responses to that diagnosed failure.

**Act 3 — Exploration as an objective, not a hack.** DDPG/TD3 inject exploration noise by hand. SAC instead *optimizes* for stochasticity: maximize return plus entropy. The soft Bellman operator, soft policy iteration, the reparameterization trick, and automatic temperature tuning fall out of one principle. Payoff: SAC trains Reacher to track a moving target in ~40k steps where PPO needs ~400k (the `ch11-ppo-sac-race` dashboard), and the trained policy ships as the chapter's WASM demo — Reacher chasing the reader's cursor. A closing protocol says when PPO still wins (cheap massively parallel sim, Ch 15/18 foreshadowed).

## 3. Section-by-Section Design

### 11.1 The Price of a Sample
- **F:** Sample complexity vs wall-clock vs hardware-cost axes; definition of *update-to-data ratio* (UTD); replay buffer 𝒟 as an off-policy estimator's dataset; restate importance-sampling correction (Ch 6) and show its variance explodes for continuous-action Gaussian ratios over long horizons — motivating estimators that avoid ratios entirely.
- **C:** Cost-of-a-rollout calculator: reader sets robot type (Pendle rig / Reacher arm / "cloud sim"), sees dollars+hours per 1M steps; PPO's Ch 10 learning curve replotted against these axes.
- **P:** `rl-core::replay` refactor: the Ch 9 `ReplayBuffer` generalized to continuous actions (`Box` action `Space` from Ch 4 traits); `rl-envs::reacher` introduced — 2-link arm in `rapier2d`, torque actions in [−1,1]², reward = −‖p_ee − p_goal‖² − 0.001‖τ‖².

### 11.2 Deterministic Policy Gradients
- **F:** The DPG theorem (Silver 2014) derived: for μ_θ with performance J(θ) = E_{s∼ρ^μ}[Q^μ(s, μ_θ(s))],
$$\nabla_\theta J(\theta) = \mathbb{E}_{s\sim\rho^\mu}\!\left[\nabla_\theta \mu_\theta(s)\, \nabla_a Q^\mu(s,a)\big|_{a=\mu_\theta(s)}\right],$$
derivation via the same recursive expansion as Ch 10's stochastic PG theorem, exchanging ∇ and ∫ under stated regularity conditions; DPG as the limit of the stochastic PG as policy variance → 0 (stated with citation, limit argument sketched honestly); why deterministic policies force *external* exploration noise and hence an off-policy formulation.
- **C:** Side-by-side gradient-flow diagram: stochastic PG (gradient through log π weighted by returns) vs DPG (gradient chained through the critic's action derivative); reader drags μ_θ(s) on a frozen Q-surface and watches ∇_a Q push it uphill.
- **P:** `rl-deep::dpg`: gradient-check experiment — finite-difference ∇_θ J vs DPG estimator on a 1-D LQR-like toy where Q^μ is computable in closed form; `proptest` tolerance test.

### 11.3 DDPG: DQN Machinery Meets DPG
- **F:** The DDPG recipe as four components: replay 𝒟, target networks (Polyak averaging θ′ ← ηθ + (1−η)θ′ — we reserve τ for torque book-wide), critic regression to y = r + γ Q_{φ′}(s′, μ_{θ′}(s′)), actor ascent on ∇_a Q_φ; exploration noise (Gaussian vs OU, evidence they're equivalent in practice); fragility analysis: actor–critic coupling means critic error is *actively sought out* by the actor — the deadly triad (Ch 8) with an adversarial search direction added.
- **C:** DDPG anatomy diagram with live data flowing (replay → critic → actor → env); a "fragility gallery": five seeds of DDPG-on-Reacher learning curves, two of which collapse, with the collapse moment linked to Q-value explosion in a synchronized plot.
- **P:** `rl-deep::ddpg` in `burn`: `DdpgAgent { actor, critic, targets, noise }` reusing Ch 9's `ReplayBuffer` and Polyak-update helper verbatim; 5-seed Reacher study logged for the fragility gallery.

### 11.4 Overestimation Bias & TD3's Three Fixes
- **F:** Overestimation analysis: for approximation errors ε_a with E[ε_a]=0, Jensen/max inequality gives E[max_a(Q(s,a)+ε_a)] ≥ max_a Q(s,a); in actor–critic form, the actor's ascent on an erroneous critic satisfies E[Q_φ(s, μ(s))] ≥ E[Q^μ(s, μ(s))] locally (Fujimoto 2018's argument reproduced step by step); bias propagates through bootstrapping — accumulation recursion derived. TD3's fixes mapped to terms in the analysis: (1) clipped double-Q y = r + γ min_{i=1,2} Q_{φ′_i}(s′, ã) attacks the max-bias (Ch 6's Double-Q ancestor cited); (2) delayed policy updates decouple actor from transient critic error; (3) target-policy smoothing ã = μ_{θ′}(s′)+clip(ε,−c,c) as regularization enforcing similar values for similar actions.
- **C:** `ch11-overestimation-cascade`: animated bootstrap chain where a bump of critic noise inflates targets update-by-update; reader toggles single-Q / double-Q / clipped-double-Q and watches the cascade tamed; `ch11-q-landscape` shows the true vs learned Q-surface over Reacher's 2-D torque space live during training.
- **P:** `rl-deep::td3` as a ~60-line diff on DDPG (twin critics, update delay d=2, smoothing noise); ablation harness runs the 2³ fix combinations on Reacher with `rayon`, producing the chapter's ablation table.

### 11.5 Maximum-Entropy RL & the Soft Bellman Operator
- **F:** MaxEnt objective J(π) = Σ_t E[γ^t (r_t + α H(π(·|s_t)))]; soft value V(s) = E_{a∼π}[Q(s,a) − α log π(a|s)]; **soft Bellman operator** 𝒯^π_soft Q(s,a) = r(s,a) + γ E_{s′}[V(s′)] proved a γ-contraction in ‖·‖_∞ (Banach from Ch 2 reused); **soft policy improvement**: π_new = argmin_{π′} D_KL(π′(·|s) ‖ exp(Q^{π_old}(s,·)/α)/Z(s)) satisfies Q^{π_new} ≥ Q^{π_old} — proved via the KL objective's optimality condition; **soft policy iteration convergence** theorem (Haarnoja 2018) assembled from the two lemmas, tabular case complete.
- **C:** Boltzmann-policy morph widget: a fixed Q-row rendered as bars, π ∝ exp(Q/α) morphing from uniform (α→∞) to greedy (α→0) as the reader drags α; `ch11-entropy-dial` applies the same dial to live Reacher training — exploration visibly "breathes."
- **P:** `rl-tabular::soft_q`: soft value iteration on the Ch 4 warehouse MDP; numerically verify contraction rate γ and the policy-improvement inequality per sweep (deterministic seeded test).

### 11.6 SAC: Reparameterization, Squashing & Automatic Temperature
- **F:** **Reparameterization derivation**: a = f_θ(s, ε) = tanh(m_θ(s) + σ_θ(s) ⊙ ε), ε∼𝒩(0,I); actor objective ∇_θ E_{ε}[α log π_θ(f_θ(s,ε)|s) − min_i Q_{φ_i}(s, f_θ(s,ε))] with gradients flowing *through* the critic — contrasted with the score-function estimator and its variance (Ch 10 callback); change-of-variables correction log π(a|s) = log 𝒩(u; m, σ) − Σ_j log(1 − tanh²(u_j)) derived from the Jacobian of tanh; **automatic temperature**: constrained program max_π E[return] s.t. E[H(π(·|s_t))] ≥ H̄, Lagrangian dual gives J(α) = E_{a∼π}[−α log π(a|s) − α H̄] with heuristic H̄ = −dim(𝒜) (= −2 for Reacher); dual-descent interpretation stated with its approximations flagged honestly.
- **C:** `ch11-reparam-path` animation: score-function vs reparameterized gradient estimates as scatter clouds shrinking with sample count; temperature-autopilot trace: α, policy entropy, and return co-plotted so the reader sees α fall as competence rises.
- **P:** `rl-deep::sac` (code sketch in §5): squashed Gaussian head, twin critics, learned log α; Reacher solved with UTD=1; checkpoint exported for the WASM cursor demo.

### 11.7 PPO vs SAC: A Robotics Decision Protocol
- **F:** Formal comparison table: estimator (ratio-free reparameterized vs clipped likelihood-ratio), data regime (replay vs on-policy batch), stability guarantees (soft policy iteration in tabular limit vs trust-region surrogate), failure modes (Q-bias vs premature entropy collapse); decision criteria formalized: sample cost c_env vs update cost c_grad, parallelism width N — when N c_env ≪ c_grad, on-policy wins (foreshadows Ch 15's rayon farms and Ch 18's PPO locomotion recipe, matching Tang's observed PPO-for-zero-shot / off-policy-for-real-world split).
- **C:** `ch11-ppo-sac-race` dashboard: same Reacher task, PPO (Ch 10 code) vs SAC vs TD3, return curves vs env-steps *and* vs wall-clock, x-axis toggle flips the winner — the chapter's thesis in one interaction.
- **P:** Benchmark harness `rl-deep::bench::reacher_race`: pinned seeds, 5 runs each, `criterion` for per-update timing; results table checked into the book.

### 11.8 Chapter Bridge
- **F:** Recap ladder: DPG theorem → DDPG → bias analysis → TD3 → soft Bellman → SAC; open thread — all of it still learns from *real* transitions only; what if the buffer could be filled by a learned model? (Ch 12). Reacher's dynamics were a black box throughout; Ch 13 opens it.
- **C:** Family-tree diagram DQN→DDPG→TD3, PG→SAC with the two lineages converging on twin critics.
- **P:** Pointer: replay + SAC infrastructure is reused verbatim by Ch 12 (MBPO-style hybrids), Ch 14 (reward-design lab), Ch 16 (offline-to-online), Ch 20 (impedance-action SAC).

## 4. Interactive Widgets & Dashboards

| ID | Type | What it shows | Reader interaction | Tech |
|---|---|---|---|---|
| `ch11-overestimation-cascade` | animation | Critic noise inflating bootstrapped targets update-by-update | Toggle single/double/clipped-double Q; noise-scale slider; step/play | egui + egui_plot |
| `ch11-q-landscape` | dashboard | Live Q_φ(s,a) surface over Reacher's 2-D torque space during training, μ_θ(s) marker riding it | Rotate surface; scrub training checkpoints; select state; overlay true-return estimate | wgpu surface plot + egui, burn checkpoints |
| `ch11-entropy-dial` | widget | Reacher's π_θ(·|s) as an action-space heat blob; exploration breathing with temperature | Drag α (log scale); toggle auto-tuning; watch entropy, return, and blob width co-evolve | WASM egui + rapier2d + burn inference |
| `ch11-reparam-path` | animation | Score-function vs reparameterized gradient estimates converging | Sample-count slider; variance readout; toggle baseline | egui_plot |
| `ch11-ppo-sac-race` | dashboard | PPO vs TD3 vs SAC on identical Reacher task | X-axis toggle env-steps/wall-clock; seed picker; UTD readout | egui_plot over logged runs |
| `ch11-reacher-cursor` | sandbox | Trained SAC Reacher tracking the reader's cursor — the chapter's WASM finale | Cursor = goal; drag/perturb links; switch DDPG/TD3/SAC checkpoints; entropy overlay | WASM: burn (WGPU) inference + rapier2d + egui |

## 5. Rust Implementation Plan

Crates touched: `rl-deep` (new modules `ddpg`, `td3`, `sac`, `bench`), `rl-envs` (new `reacher` env on `rapier2d`), `rl-core` (replay generalized to continuous `Box` spaces), demo crates `demos/ch11-*`. Shared infrastructure: Ch 9's `ReplayBuffer`, Polyak helper, and telemetry channel are imported, not rewritten — the diff between DQN and DDPG is itself pedagogy.

Representative sketch — the SAC update in `burn`:

```rust
// crates/rl-deep/src/sac/agent.rs
pub struct SacAgent<B: AutodiffBackend> {
    actor: SquashedGaussianPolicy<B>, // heads: mean m_theta, log_std
    critics: [QNet<B>; 2],            // Q_{phi_1}, Q_{phi_2}
    targets: [QNet<B>; 2],            // Polyak copies (eta = 0.005)
    log_alpha: Param<Tensor<B, 1>>,   // learned temperature, log-space
    target_entropy: f64,              // -|A| = -2.0 for Reacher torques
}

impl<B: AutodiffBackend> SacAgent<B> {
    pub fn update(&mut self, batch: &Batch<B>, cfg: &SacConfig) -> SacMetrics {
        let alpha = self.log_alpha.val().exp();
        // -- Critic: soft Bellman target with clipped double-Q --------------
        let (a2, logp2) = self.actor.rsample(&batch.next_obs); // reparameterized
        let q2 = Tensor::min_pair(
            self.targets[0].forward(&batch.next_obs, &a2),
            self.targets[1].forward(&batch.next_obs, &a2),
        );
        let y = batch.reward.clone()
            + batch.not_done.clone() * cfg.gamma * (q2 - alpha.clone() * logp2);
        let critic_loss = self.critics.iter()
            .map(|q| mse(q.forward(&batch.obs, &batch.act), y.detach()))
            .sum::<Tensor<B, 1>>();
        // -- Actor: minimize E[ alpha * log pi - min_i Q_i ] ----------------
        let (a, logp) = self.actor.rsample(&batch.obs);
        let q_min = Tensor::min_pair(
            self.critics[0].forward(&batch.obs, &a),
            self.critics[1].forward(&batch.obs, &a),
        );
        let actor_loss = (alpha.clone() * logp.clone() - q_min).mean();
        // -- Temperature: dual descent on the entropy constraint ------------
        let alpha_loss =
            (-self.log_alpha.val().exp() * (logp.detach() + self.target_entropy)).mean();
        self.step_optimizers(critic_loss, actor_loss, alpha_loss, cfg)
    }
}
```

Experiments/benchmarks: 5-seed DDPG fragility study; TD3 2³ fix ablation (rayon-parallel); PPO/TD3/SAC Reacher race with pinned seeds; `criterion` per-update timing feeding the wall-clock axis. Native artifacts: all training binaries (`cargo run -p rl-deep --bin train_sac_reacher`). In-browser: `ch11-entropy-dial`, `ch11-reacher-cursor`, and all plots (inference-only WASM; training stays native).

## 6. Robot Thread

- **Reacher** (true debut as a continuous-control task; informally seen in Ch 3 choosing grasp primitives as a bandit): before — no continuous-control policy exists for it; after — SAC tracks moving targets at >95% success within 2 cm, checkpoint shipped in the cursor demo. Carried to Ch 13 (its dynamics derived), Ch 14 (reward-design lab), Ch 16–17, Ch 20.
- **Pendle**: cameo — PPO's Ch 10 swing-up re-solved by SAC in the race dashboard as a controlled comparison; returns as model-based testbed in Ch 12 and control baseline plant in Ch 13.

## 7. Exercises & Explorations

1. **(F)** Complete the DPG derivation for the continuing (average-reward-free, discounted-state-distribution) case, identifying exactly where ∇_θ ρ^μ is dropped and citing the condition under which that is exact.
2. **(F)** Let ε_a ∼ Uniform[−c, c] i.i.d. over m candidate actions with equal true values. Compute E[max_a(Q+ε_a)] − max_a Q in closed form; verify against `ch11-overestimation-cascade` with matching noise.
3. **(F)** Derive the tanh log-density correction from the change-of-variables formula, and show entropy of the squashed policy has no closed form — hence why SAC estimates it by sampling.
4. **(F)** Prove soft policy improvement for the tabular case (the KL-projection lemma) without consulting §11.5, then check each step against the text.
5. **(C)** Using `ch11-entropy-dial`, find the fixed α that matches auto-tuned final performance on Reacher, and report how many seeds it costs you. Explain what the dual-descent objective bought.
6. **(C)** In `ch11-q-landscape`, capture a frame where Q_φ shows a spurious peak and the actor marker sits on it; then enable clipped double-Q and capture the same state 10k steps later.
7. **(P)** Implement DDPG's OU noise, run the Gaussian-vs-OU ablation on Reacher (5 seeds each), and reproduce the "it doesn't matter" finding — or refute it, with curves.
8. **(P)** Add n-step returns (Ch 7) to SAC's critic target and measure the off-policy bias/variance trade-off on Reacher for n ∈ {1, 3, 5}.

## 8. Notation Introduced

| Symbol | Meaning |
|---|---|
| μ_θ(s) | deterministic policy, parameters θ |
| ρ^μ | discounted state distribution under μ |
| Q_{φ_1}, Q_{φ_2} | twin critics; φ′_i target parameters |
| η | Polyak averaging coefficient (τ stays reserved for torque, per Appendix C) |
| α | entropy temperature; H̄ target entropy |
| H(π(·|s)) | policy entropy at state s |
| 𝒯^π_soft | soft Bellman operator |
| f_θ(s, ε) | reparameterized action sampler; u pre-squash action, a = tanh(u) |
| 𝒟 | replay buffer; UTD update-to-data ratio |

## 9. References & Further Reading

- **Kober, Bagnell & Peters, IJRR 2013** — §3.2 (curse of real-world samples: roll-outs are costly, hard to reproduce — this chapter's economic premise); §1.3 (policy-search preference in robotics, context for why continuous-action methods matter).
- **Tang et al. 2024** — §3.3 solution-approach axis (policy optimization: off-policy vs on-policy classification used throughout Part IV); §7 open challenges (sample efficiency of on-policy RL as the barrier to real-world and few-shot learning; off-policy/offline data as the promising direction) — the modern evidence for this chapter's thesis.
- **S&B** — §6.5 Q-learning and §6.7/§6.8 (maximization bias, Double Q-learning: the tabular ancestors of §11.4); ch. 13 (policy-gradient foundations extended here).
- Silver et al. 2014, *Deterministic Policy Gradient Algorithms* (DPG theorem, §11.2).
- Lillicrap et al. 2016, *Continuous Control with Deep Reinforcement Learning* (DDPG).
- Thrun & Schwartz 1993, *Issues in Using Function Approximation for Reinforcement Learning* (original overestimation analysis).
- Fujimoto, van Hoof & Meger 2018, *Addressing Function Approximation Error in Actor-Critic Methods* (TD3; §11.4's analysis).
- Haarnoja et al. 2018, *Soft Actor-Critic* and Haarnoja et al. 2019, *Soft Actor-Critic Algorithms and Applications* (soft policy iteration; automatic temperature tuning).
- Ziebart 2010 (maximum-entropy control lineage, background for §11.5).

# Chapter 12 — Model-Based RL & World Models

> **Part** II — Scaling Up: Function Approximation & Deep RL · **Builds on:** Ch 2, 5, 7, 9, 11 · **Feeds:** Ch 13, 15, 22
> **Modernizes:** Kober §6 (models & mental rehearsal: simulation bias, distributions over models, sampling with reused random numbers) — the direct ancestor of this chapter; Tang §3.3 model-learning axis (whether transition dynamics are learned from robot data) and the sample-efficiency trend behind it; S&B ch. 8 (Dyna, planning-learning integration) carried from tables to neural ensembles and latent imagination.

## 1. Purpose & Learning Outcomes

Ch 11 made every real transition reusable; this chapter makes transitions *manufacturable*. Kober called it mental rehearsal: learn a forward model from real experience, then train against the model. The catch — Kober §6.1's simulation bias — is that optimizers exploit model errors, and errors compound with horizon. This chapter delivers the modern toolkit: probabilistic ensembles that know what they don't know (uncertainty math from Ch 2 cashed in), CEM-MPC planners (PETS), Dyna-style hybrids (MBPO), and latent world models that learn in imagination (Dreamer). Pendle is the testbed throughout: swing-up from a learned model with a fraction of Ch 10's data.

The reader can:
- Formulate dynamics learning as probabilistic regression and decompose predictive uncertainty into epistemic and aleatoric parts via the ensemble variance identity.
- Derive the model-bias compounding bound: one-step error ε growing to O(ε L^H) in state and the discounted return-gap bound it implies.
- Derive the cross-entropy method as iterated importance-sampled distribution fitting, and assemble it into a receding-horizon MPC loop.
- Explain MBPO's branched-rollout argument: why short model rollouts from real states dodge the compounding bound, in S&B's Dyna lineage.
- Derive the ELBO for a latent state-space model and identify what each term trains in a Dreamer-style world model.
- Implement a probabilistic ensemble and CEM-MPC in `burn` and solve Pendle swing-up with ~10× less real data than PPO.
- Diagnose model exploitation using calibration plots and ensemble-fan visualizations.

## 2. Storyline

**Act 1 — The robot that dreams wrong.** Hook: train a single deterministic neural dynamics model on 2k Pendle transitions, plan a swing-up against it, execute — the model predicts a graceful swing; the real pole flails. The `ch12-imagination-fan` widget replays the divergence: prediction and reality agree for 15 steps, then part ways exponentially. This is Kober's simulation bias made visceral, and Ch 7's Dyna warning ("when the model is wrong") returning at neural scale. Quantifying *how* wrong, and *knowing when* you're wrong, is the chapter's spine.

**Act 2 — Uncertainty as the antidote.** Ch 2's probability toolkit pays off: Gaussian NLL heads capture aleatoric noise, ensembles capture epistemic ignorance, and the variance-decomposition identity separates them. The compounding-error bound shows horizon is the enemy; three responses follow, in increasing sophistication: replan constantly (CEM-MPC/PETS — short effective horizon), rollout short branches from real states (MBPO — Dyna reborn, feeding Ch 11's SAC with imagined transitions), or learn a latent space where long imagination is cheap and trained end-to-end (Dreamer's ELBO).

**Act 3 — The dream that holds.** Payoff: CEM-MPC with a 5-member ensemble swings Pendle up after ~3 minutes of simulated "real" experience; the `ch12-mpc-horizon` animation shows the planner re-dreaming 25 steps ahead, 50 times a second. Honest coda: model-based wins the sample-efficiency race but adds compute, tuning surface, and new failure modes — and every learned model here begs the question Ch 13 answers: robots come with centuries of *analytic* model knowledge. Why learn what Lagrange already derived?

## 3. Section-by-Section Design

### 12.1 Why Models: Mental Rehearsal, Then and Now
- **F:** Formal definitions: model-free vs model-based on the (S, A, P, R, γ) tuple — learning P̂ ≈ P (and optionally R̂); taxonomy axes (what is learned: forward/inverse/latent; how used: planning vs Dyna-style data generation vs analytic gradients); Kober's three core issues of mental rehearsal restated as formal desiderata (bias control, stochasticity handling, efficient simulation-based optimization); Tang's model-learning axis as the modern survey lens.
- **C:** Interactive lineage map: Dyna (1990) → PILCO (2011) → PETS (2018) → MBPO (2019) → Dreamer (2020–23), each node expandable with its one-line trick; sample-efficiency bar chart (env-steps to solve Pendle) foreshadowing the chapter's experiments.
- **P:** `rl-deep::mbrl` module scaffold; experiment harness `real_step_budget` metering: every algorithm in this chapter reports real-vs-imagined step counts through `rl-core` metrics (reset/wear accounting foreshadows Ch 14).

### 12.2 Learning Dynamics: Ensembles & Two Kinds of Uncertainty
- **F:** Dynamics learning as regression on Δs: model p_ψ(s′ | s, a) = 𝒩(s + f_ψ^μ(s,a), diag(f_ψ^σ(s,a)²)); Gaussian NLL loss derived; **variance decomposition** for an ensemble {p_{ψ_i}}_{i=1}^N:
$$\operatorname{Var}[s'] \;=\; \underbrace{\tfrac{1}{N}\sum_i \sigma_{\psi_i}^2(s,a)}_{\text{aleatoric}} \;+\; \underbrace{\tfrac{1}{N}\sum_i \big(\mu_{\psi_i}(s,a) - \bar\mu(s,a)\big)^2}_{\text{epistemic}},$$
derived from the law of total variance (Ch 2); calibration defined (predicted quantile coverage vs empirical); connection to Kober's "distributions over models" paragraph — ensembles as the deep-era Gaussian process posterior, PILCO cited as the GP original.
- **C:** `ch12-calibration-dash`: reliability diagram + fan chart; reader drags the training-set-size slider and watches epistemic variance shrink where data lives while aleatoric stays put (Pendle with injected actuator noise).
- **P:** `rl-deep::mbrl::ensemble`: N=5 bootstrapped Gaussian-head MLPs in `burn`, trained with `rayon` across members; unit test: on a synthetic heteroscedastic function, recovered aleatoric σ matches ground truth within tolerance.

### 12.3 The Compounding-Error Bound
- **F:** Theorem (compounding state error): if the true dynamics f is L-Lipschitz and the model satisfies one-step error sup‖f̂(s,a) − f(s,a)‖ ≤ ε, then open-loop rollouts obey
$$\|\hat s_H - s_H\| \;\le\; \varepsilon \sum_{i=0}^{H-1} L^i \;=\; \varepsilon\,\frac{L^H - 1}{L - 1},$$
proved by induction (triangle inequality + Lipschitz step — full algebra); return-gap corollary in the MBPO style: for one-step model error ε_m in total variation and policy shift ε_π,
$$\big|J^\pi_{\hat M} - J^\pi_M\big| \;\le\; \frac{2\, r_{\max}\,\gamma\,(\varepsilon_m + 2\varepsilon_\pi)}{(1-\gamma)^2} + \frac{4\, r_{\max}\, \varepsilon_\pi}{1-\gamma},$$
stated with the simulation-lemma proof for the ε_π = 0 case done fully and the general case cited (Janner 2019); reading: the (1−γ)^{-2} is the price of trusting a model forever — motivating short horizons.
- **C:** `ch12-imagination-fan`: real Pendle trajectory vs 5-member ensemble rollouts; fan width vs horizon plotted live against the bound's L^H envelope; reader sets data budget and watches the divergence point move.
- **P:** Empirical bound check: measure one-step ε on held-out data, fit local L by finite differences, overlay predicted vs measured rollout error on Pendle — the book's "theory meets telemetry" pattern.

### 12.4 Planning with a Learned Model: Shooting, CEM & MPC (PETS)
- **F:** Open-loop planning problem max_{a_{0:H−1}} E[Σ γ^t r̂(s_t, a_t)]; random shooting as Monte Carlo argmax; **CEM derived as importance-sampled optimization**: minimizing D_KL between the elite-conditioned level-set distribution and the Gaussian sampling family q_{m,σ} yields the elite-refit update
$$q^{(k+1)} = \arg\min_{q\in\mathcal{Q}} \; -\mathbb{E}_{x\sim q^{(k)}}\!\left[\mathbf{1}\{R(x) \ge \kappa_k\}\, \log q(x)\right],$$
whose solution for diagonal Gaussians is exactly elite mean/variance (derivation via the cross-entropy/MLE identity, Rubinstein lineage); MPC receding horizon: plan H steps, execute one, replan — closed-loop feedback recovered from open-loop plans; PETS assembled: trajectory sampling (TS-1/TS-∞) propagating particles through ensemble members so plans respect epistemic uncertainty.
- **C:** `ch12-cem-funnel`: sampling distribution over action sequences narrowing onto high-return elites, iteration by iteration, against random shooting's flat scatter; `ch12-mpc-horizon`: Pendle swing-up with the planned 25-step tail ghosted ahead of the executing pole, replanned every step; disturbance injection shows MPC absorbing pushes.
- **P:** `rl-deep::mbrl::cem` planner (code sketch in §5) + `MpcController` wrapper implementing `rl-core::Agent`; Pendle swing-up experiment: ensemble + CEM-MPC vs Ch 10 PPO vs Ch 11 SAC, real-step budget on the x-axis.

### 12.5 Dyna Reborn: Branched Rollouts & MBPO
- **F:** Dyna (S&B ch. 8, Ch 7) restated for the deep setting: imagined transitions fill Ch 11's replay buffer; **branched-rollout bound**: k-step model rollouts launched from real off-policy states trade the (1−γ)^{-2} horizon penalty for a k-controlled term — MBPO's return bound stated, the k-dependence analyzed, and the corollary derived that a k exists where model data helps iff one-step model error is small relative to policy shift; monotone-improvement reading and its practical caveats.
- **C:** `ch12-mbpo-dial`: drag rollout length k, see (a) the bound's terms re-plotted, (b) empirical SAC-on-Pendle return after a fixed real budget — theory curve and practice curve side by side, disagreeing instructively at large k.
- **P:** `mbpo_lite` experiment: Ch 11's SAC unchanged, fed by a 20:1 imagined:real mixture from k-step ensemble branches; ablation over k ∈ {1, 5, 20}; demonstrates ~5× real-sample reduction on Pendle swing-up at k=5, collapse at k=20.

### 12.6 Latent World Models: the ELBO & Learning in Imagination
- **F:** Latent state-space model p_ψ(z_t | z_{t−1}, a_{t−1}), decoder p_ψ(o_t | z_t), variational posterior q_ξ(z_t | z_{t−1}, a_{t−1}, o_t); **ELBO derived in full** from Jensen's inequality on log p(o_{1:T} | a_{1:T}):
$$\log p_\psi(o_{1:T}\,|\,a_{1:T}) \;\ge\; \sum_{t=1}^{T} \mathbb{E}_{q_\xi}\big[\log p_\psi(o_t|z_t)\big] - \mathbb{E}_{q_\xi}\big[D_{\mathrm{KL}}\big(q_\xi(z_t|\cdot)\,\|\,p_\psi(z_t|z_{t-1},a_{t-1})\big)\big],$$
each term named (reconstruction trains the decoder; KL trains the prior to *be* the dynamics — imagination is rolling the prior); actor-critic in latent space (Dreamer): λ-returns (Ch 7 callback) over imagined latent trajectories, gradients through the reparameterized rollout; what is honestly out of scope (discrete latents, straight-through — cited to DreamerV2/V3).
- **C:** `ch12-latent-projector`: 2-D PCA projection of Pendle's learned latent space with imagined vs encoded-real trajectories as threads; hover decodes z_t to a rendered pendulum frame — the reader literally watches the model dream.
- **P:** `rl-deep::mbrl::rssm_lite`: compact GRU-based latent model on Pendle from 3-frame pixel stacks (32×32, `burn` WGPU); train the world model only (policy learning in latent space is an exercise); reconstruction-quality vs KL-weight study.

### 12.7 When Models Fail: Exploitation, Sysid & the Analytic Alternative
- **F:** Model exploitation formalized: the planner solves max E_{p̂}[R], so return gaps concentrate exactly where the optimizer steers into high-epistemic-variance regions — Kober's "RL exploits model inaccuracies" as an optimization statement; mitigations tabulated with the mechanism each attacks (ensembles/TS, uncertainty penalties r̂ − β·Var, short horizons, noise injection à la Kober); classical system identification as the parametric limiting case: known model structure M(q)q̈ + C q̇ + g = τ with unknown (m, l, c) — regression on physically-structured features, previewing Ch 13's derivation and Ch 15's sysid pipeline.
- **C:** Model-exploitation horror gallery: three replayable failures (Pendle planner "teleporting" through an unmodeled torque limit; reward achieved in imagination only; ensemble-disagreement alarm firing before each crash) with the uncertainty trace that would have caught each.
- **P:** Uncertainty-penalized CEM (r̂ − β·epistemic-std) ablation over β; parametric sysid teaser: fit (m, l) of Pendle by least squares on the manipulator-equation features using `nalgebra`, compare 200-sample parametric fit vs 2k-sample neural ensemble accuracy — the punchline that opens Part III.

### 12.8 Chapter Bridge
- **F:** Recap: know your model's ignorance (ensembles), respect the horizon (bound), replan or branch (MPC/MBPO), or dream in latent space (ELBO); the ledger: model-based buys real-sample efficiency with compute and bias risk. Open thread: everything here *learned* P̂ from scratch, yet a robot's P is largely written down already — kinematics, Lagrangian dynamics, motor models. Ch 13 opens the black box; Ch 15 fuses the two (sim + sysid + randomization).
- **C:** Decision flowchart (mermaid): data cost / dynamics smoothness / observation type → PPO vs SAC vs PETS vs MBPO vs Dreamer vs "use the analytic model."
- **P:** Pointer table: `ensemble` reused by Ch 15 (randomization-aware models) and Ch 22; `real_step_budget` metering reused by Ch 14's curse-of-samples accounting.

## 4. Interactive Widgets & Dashboards

| ID | Type | What it shows | Reader interaction | Tech |
|---|---|---|---|---|
| `ch12-imagination-fan` | widget | Real Pendle trajectory vs ensemble rollouts; fan widening with horizon against the L^H envelope | Horizon slider; ensemble size; data-budget slider; replay divergence moment | WASM egui + rapier2d + burn inference |
| `ch12-calibration-dash` | dashboard | Reliability diagram + epistemic/aleatoric split across state space | Training-set-size slider; toggle ensemble vs single Gaussian head; probe states | egui_plot |
| `ch12-cem-funnel` | animation | CEM's sampling distribution narrowing onto elites over iterations vs random shooting | Step iterations; elite-fraction and population sliders; return-landscape backdrop | egui_plot |
| `ch12-mpc-horizon` | animation | Receding-horizon CEM-MPC swing-up on Pendle: planned tail ghosted, executed head solid | Play/step; horizon H and replan-rate controls; push-disturbance button | WASM rapier2d + egui |
| `ch12-mbpo-dial` | widget | Branched-rollout length k vs bound terms and empirical return | Drag k; toggle bound/empirical overlay; real-budget selector | egui_plot over logged runs |
| `ch12-latent-projector` | widget | 2-D projection of learned latent space; imagined vs real trajectory threads; hover-decode to frames | Scrub trajectories; select seed states; imagination-length slider | egui + burn (WGPU) decoder |

## 5. Rust Implementation Plan

Crates touched: `rl-deep` (new `mbrl` module tree: `ensemble`, `cem`, `mpc`, `mbpo_lite`, `rssm_lite`), `rl-envs` (Pendle gains pixel-observation mode for §12.6), `rl-core` (metrics: real-vs-imagined step ledger), demos `demos/ch12-*`. Ch 11's SAC and replay are dependencies, unmodified.

Representative sketch — ensemble CEM-MPC planning step:

```rust
// crates/rl-deep/src/mbrl/cem.rs
pub struct CemPlanner {
    pub horizon: usize,   // H = 25 for Pendle
    pub pop: usize,       // 500 candidate sequences
    pub elite_frac: f64,  // 0.1
    pub iters: usize,     // 5 CE iterations
}

impl CemPlanner {
    /// One receding-horizon action choice (executed head; tail discarded).
    pub fn plan<B: Backend>(
        &self,
        ensemble: &DynamicsEnsemble<B>,   // N Gaussian-head members
        reward: &dyn RewardModel<B>,
        s0: Tensor<B, 1>,
        rng: &mut StdRng,
    ) -> Action {
        let (mut mean, mut std) = self.warm_start();       // [H, A], shifted prev plan
        for _ in 0..self.iters {
            let cands = gaussian_sequences(&mean, &std, self.pop, rng); // [P, H, A]
            // TS-inf: each particle commits to one ensemble member for its
            // whole rollout, so plans price in epistemic disagreement.
            let returns = ensemble.propagate_ts_inf(&s0, &cands, reward); // [P]
            let elites = top_frac(&cands, &returns, self.elite_frac);
            (mean, std) = elites.fit_diag_gaussian();      // the CE/MLE update
        }
        Action::from_tensor(mean.clone().slice([0..1]))    // MPC: first action only
    }
}
```

Experiments/benchmarks: bound-vs-measured rollout error (§12.3); Pendle swing-up race PETS vs MBPO-lite vs SAC vs PPO with real-step budgets and pinned seeds; k-ablation for MBPO-lite; β-ablation for uncertainty-penalized CEM; `criterion` benchmark of `propagate_ts_inf` (rayon across particles) since planner latency is the point for real-time MPC. Native: all training/planning binaries; `rerun` telemetry of imagined vs real rollouts during development. In-browser: `ch12-imagination-fan`, `ch12-mpc-horizon`, `ch12-latent-projector` (inference-only WASM).

## 6. Robot Thread

- **Pendle** (central): before — solved by PPO (Ch 10) and SAC (Ch 11) model-free; after — swung up from ~10× less real data via ensemble CEM-MPC, owns a learned pixel-latent world model, and has served as the compounding-error measurement rig. Hands off to Ch 13 as the plant whose *analytic* model gets derived, and to Ch 15's integrator/sysid playground.
- **Reacher**: cameo — the sysid teaser (§12.7) prefigures fitting its dynamic parameters after Ch 13 derives them; MBPO-lite is flagged as directly applicable to Ch 11's Reacher SAC (exercise 7).

## 7. Exercises & Explorations

1. **(F)** Redo the compounding-error induction for L < 1 (contractive dynamics) and show the error saturates at ε/(1−L). Which of the cast robots plausibly has contractive regions, and where does Pendle violate the assumption?
2. **(F)** Derive the diagonal-Gaussian CEM refit update from the cross-entropy program in §12.4 — show elite mean/variance is the exact argmin, not a heuristic.
3. **(F)** Complete the ELBO derivation for T=2 explicitly (no telescoping shortcuts), labeling which network each term differentiates through.
4. **(F)** Prove the ensemble variance decomposition from the law of total variance, and explain why bootstrapped data (not just random init) matters for the epistemic term.
5. **(C)** Using `ch12-imagination-fan`, find the smallest data budget where the 25-step fan still contains the real trajectory; cross-check the divergence horizon against your fitted L from exercise 1.
6. **(C)** In `ch12-mpc-horizon`, find the minimal horizon H that still swings Pendle up, then repeat with the disturbance button held. Explain the gap using the receding-horizon feedback argument.
7. **(P)** Port `mbpo_lite` to Ch 11's Reacher SAC and report real-step savings at k=5; instrument ensemble disagreement to auto-truncate branches (truncate when epistemic std > threshold).
8. **(P)** Implement TS-1 (particles reshuffle members each step) alongside TS-∞ and compare plan quality and diversity on Pendle; explain the difference via the epistemic term of exercise 4.

## 8. Notation Introduced

| Symbol | Meaning |
|---|---|
| p_ψ(s′\|s,a) | learned dynamics model, parameters ψ; f̂ its mean function |
| {p_{ψ_i}}_{i=1}^N | ensemble of N members; μ_{ψ_i}, σ_{ψ_i} member mean/std heads |
| ε, ε_m, ε_π | one-step model error (state / total-variation) and policy-shift error |
| L | Lipschitz constant of dynamics; H planning/rollout horizon |
| κ_k | CEM elite threshold at iteration k; elite fraction its quantile |
| k | branched-rollout length (MBPO) |
| z_t | latent state; o_t observation; q_ξ variational posterior, ξ its parameters |
| ELBO terms | reconstruction E_q[log p_ψ(o_t\|z_t)], consistency KL(q_ξ ‖ p_ψ) |

## 9. References & Further Reading

- **Kober, Bagnell & Peters, IJRR 2013** — §6 (models & mental rehearsal, incl. Dyna lineage credit to Sutton 1990); §6.1 (simulation biases = "over-fitting to the model," noise injection as smoothing, distributions over models, PEGASUS random-number reuse); §6.2 (forward-model learning approaches, incl. locally linear LQR solutions); §3.3 (curse of under-modeling: small model errors accumulate to substantially different behavior) — this chapter is that section, modernized.
- **Tang et al. 2024** — §3.3 solution-approach axis, model-learning dimension (whether transition dynamics are learned from robot data); §7 (combining model-free and model-based approaches flagged as the promising sample-efficiency direction; world-model successes such as DayDreamer cited there).
- **S&B** — ch. 8 (Dyna architecture, planning as learning from simulated experience, when-the-model-is-wrong §8.3) — the tabular spine under §12.5; §7 lineage for the λ-returns Dreamer uses in imagination.
- Deisenroth & Rasmussen 2011, *PILCO* (GP models, uncertainty propagation — the "distributions over models" exemplar Kober highlights).
- Chua et al. 2018, *Deep RL in a Handful of Trials* (PETS: probabilistic ensembles + trajectory sampling + CEM-MPC).
- Janner et al. 2019, *When to Trust Your Model* (MBPO: branched rollouts, return-gap bounds of §12.3/§12.5).
- Hafner et al. 2020 (*Dream to Control*), 2021 (*DreamerV2*), 2023 (*DreamerV3*) — latent world models, learning in imagination.
- Rubinstein & Kroese 2004, *The Cross-Entropy Method* (CEM's importance-sampling derivation).
- Sutton 1990, *Integrated Architectures for Learning, Planning and Reacting* (Dyna — cited via both S&B ch. 8 and Kober §6).

# Chapter 9 — Deep Value-Based Methods: DQN & Descendants

> **Part** II — Scaling Up: Function Approximation & Deep RL · **Builds on:** Ch 6 (Q-learning, maximization bias, Double Q), Ch 8 (semi-gradient methods, deadly triad, first `burn` network) · **Feeds:** Ch 11 (shared replay/target infrastructure), Ch 20 (grasp-selection DQN)
> **Modernizes:** the post-tabular gap the baseline texts leave open — S&B 2nd ed. ch. 11's stability warnings carried into the deep era (published ed. §16.5 covers DQN as a case study; the repo's draft PDF predates it); Kober §4.2's neural value-function divergence history, resolved by the DQN recipe; Tang solution-approach axis (off-policy value-based methods) and §4.3.1.1 grasping-as-discrete-selection context

## 1. Purpose & Learning Outcomes

Chapter 8 ended with a warning: deep + bootstrapped + off-policy = the deadly triad. This chapter walks straight into it, because that corner is where sample-efficient discrete-action control lives. The reader learns why naive deep Q-learning explodes, then rebuilds stability as two pieces of *statistical surgery* — experience replay against correlation, target networks against moving bootstrap targets — and climbs the descendant tree (Double, Dueling, PER, n-step, C51, Rainbow) knowing exactly which pathology each rung treats. The payoff robot: Rusty learns a visual gridworld from pixels, the trained network later reused as Ch 20's grasp-selection pattern.

The reader can:
- Write the naive deep Q-learning loss and name its three failure sources (correlated samples, moving targets, off-policy max).
- Explain replay as restoring the sampling assumptions of Robbins–Monro, and target networks as fitted value iteration; state the resulting error-propagation bound.
- Derive the Double DQN target from the maximization-bias analysis of Ch 6 and the Dueling identifiability constraint.
- Implement prioritized replay with a sum-tree, including the importance-sampling correction $(N\,P(i))^{-\beta}$.
- State the distributional Bellman operator, prove its $\gamma$-contraction in maximal Wasserstein distance for policy evaluation, and explain C51's projection + cross-entropy step.
- Place value-based methods correctly in robotics: discrete/mid-level action spaces, grasp selection, sample-efficiency arguments (Tang).

## 2. Storyline

**Act 1 — Rusty gets eyes and loses its mind.** Rusty returns from Part I's warehouse, but the clean $(row,col)$ state is gone: the observation is now a 64×64 pixel top-down view (`rl-envs::VisualGridworld`). Tile-code pixels? Hopeless. So we do the obvious thing — bolt Ch 6's Q-learning update onto a `burn` convnet — and watch the loss curve leave the chart. The reader is invited to diagnose it with Ch 8's tools: every leg of the deadly triad is present, plus two new sins the tabular world hid (consecutive frames are nearly identical, and the regression target moves every step).

**Act 2 — Surgery.** Two interventions, each derived before it is coded. A replay buffer turns the correlated trajectory stream into approximately independent draws from a mixture distribution — Robbins–Monro's conditions patched back in, at the price of learning off-policy from stale data (which Q-learning, alone among Ch 6's algorithms, tolerates by design). A target network freezes the bootstrap target for $C$ steps, turning the chase-your-own-tail regression into a sequence of quasi-stationary supervised problems — fitted value iteration with a contraction argument. The `ch09-stability-dash` ablation (naive / +replay / +target / both) is the act's centerpiece: the reader watches stability get engineered back in, one leg at a time. DQN trains; pixel-Rusty navigates.

**Act 3 — The descendants, and where robots actually use this.** Each extension is introduced as treatment for a measured disease: Double DQN for the overestimation Rusty's Q-values visibly show, Dueling for states where action choice barely matters, PER for the rare-success sparse-reward variant of the gridworld, n-step for slow credit propagation, C51 because a *distribution* over returns is strictly more informative supervision. Rainbow assembles them and the ablation histogram says which mattered. The chapter closes by placing the whole family in the robotics landscape via Tang: value-based off-policy methods dominate where actions are discrete or mid-level — grasp candidates, primitive selection — exactly the Ch 20 payoff, with Rusty's trained convnet as the template.

**Robots:** Rusty is the protagonist (pixels for the first time). Pendle sits this one out (continuous actions must wait for Ch 10–11); a foreshadowing sidebar shows why argmax over a continuum is the blocker.

## 3. Section-by-Section Design

### 9.1 From Q-Table to Q-Network
- **F:** The lookup table as the one-hot special case of $\hat q(s,a,\mathbf{w})$; the naive deep Q-learning loss $L(\mathbf{w})=\mathbb{E}_{(s,a,r,s')}\big[(r+\gamma\max_{a'}\hat q(s',a',\mathbf{w})-\hat q(s,a,\mathbf{w}))^2\big]$ with the semi-gradient convention (no gradient through the target) inherited from Ch 8; architecture as generalization prior — convnet for pixels, one output head per action so a single forward pass yields all Q-values (the Mnih 2015 trick that makes $\max_{a'}$ cheap).
- **C:** Table-to-network morph animation: Rusty's Ch 6 Q-heatmap dissolves into a convnet diagram; hovering an output neuron highlights which pixels feed it (receptive-field overlay), making "features that learn themselves" (Ch 8.6) concrete for images.
- **P:** `rl-envs::VisualGridworld` (64×64×3 render of the warehouse, `ndarray` frames, seeded layout randomization); `rl-deep::dqn::QNet` in `burn`: Conv(16,8,4)-Conv(32,4,2)-FC(256)-FC(|A|), WGPU backend; the *naive* trainer, run and logged — its divergence is data for §9.2, not an embarrassment to hide.

### 9.2 Diagnosing the Explosion
- **F:** Three sins, formalized. (i) *Correlation*: successive gradient samples $g_t,g_{t+1}$ along a trajectory have $\text{Cov}(g_t,g_{t+1})\neq 0$; SGD's variance analysis (Ch 2) shows the effective sample size of a window shrinks by the autocorrelation factor, and worst-case feedback: the network's own errors steer the policy into the states that reinforce them. (ii) *Moving target*: the regression target $r+\gamma\max_{a'}\hat q(s',a',\mathbf{w})$ changes with every update — no fixed objective is being descended (Ch 8's "semi-gradient is not a gradient", now with the target moving too). (iii) *Deadly triad*: $\max$ makes it off-policy by construction; Baird's lesson applies with a nonlinear twist — Kober §4.2's catalogue (one overestimated state propagating globally, Boyan & Moore 1995) is exactly what the logs from §9.1 show.
- **C:** `ch09-replay-flow` part 1: a correlation meter over the naive trainer's sample stream — autocorrelation of consecutive gradient minibatches plotted live; the reader sees ρ≈0.9 for trajectory-ordered samples.
- **P:** Instrumentation module `rl-deep::diag`: gradient-autocorrelation and Q-value-inflation probes (mean $\max_a\hat q$ vs. ground-truth $v^*$ from Ch 5's DP solver on the equivalent tabular gridworld) — the two dashboards every later run displays.

### 9.3 The DQN Recipe: Replay Buffer & Target Network
- **F:** *Replay*: uniform sampling from buffer $\mathcal{D}$ of the last $N$ transitions ≈ i.i.d. draws from the mixture $\bar\mu=\frac1N\sum_k\mu_{\pi_k}$; decorrelation restores the Robbins–Monro sampling regime; the price is off-policyness w.r.t. $\bar\mu$ — tolerable because Q-learning's target is policy-independent (Ch 6). *Target network*: define $\mathbf{w}^*(\mathbf{w}^-)=\arg\min_{\mathbf{w}}\mathbb{E}_{\mathcal{D}}[(r+\gamma\max_{a'}\hat q(s',a',\mathbf{w}^-)-\hat q(s,a,\mathbf{w}))^2]$; the outer iteration $\mathbf{w}^-_{k+1}=\mathbf{w}^*(\mathbf{w}^-_k)$ is fitted Q-iteration — with exact regression it *is* the Bellman optimality operator, a $\gamma$-contraction (Ch 4), giving $\|q_k-q_*\|_\infty\le\gamma^k\|q_0-q_*\|_\infty+\frac{\varepsilon_{\max}}{1-\gamma}$ when each regression incurs error $\le\varepsilon_{\max}$ — the two-timescale reading: a fast SGD clock inside a slow Bellman clock. Polyak averaging $\mathbf{w}^-\leftarrow\tau\mathbf{w}+(1-\tau)\mathbf{w}^-$ as the continuous variant (bridge to Ch 11).
- **C:** `ch09-replay-flow` part 2: transitions flow env → ring buffer → uniform minibatch; toggling "sample sequentially" re-breaks training before the reader's eyes. `ch09-two-clocks` (signature widget): two clock faces — the fast SGD hand and the slow target hand that only ticks every $C$ steps; a slider for $C$ (and a $\tau$ mode) with a live loss surface that visibly re-anchors at each slow tick; $C=1$ reproduces §9.1's explosion.
- **P:** `rl-core::replay::RingBuffer` (shared with Ch 11 by design — `push`, `sample_uniform`, seeded); `rl-deep::dqn::Trainer` with target sync period $C$ and $\epsilon$-greedy schedule; full DQN on visual-gridworld Rusty: seeded run reaches ≥ 95% of the DP-optimal return; `egui` telemetry dashboard streams loss, return, mean-max-Q, and the diag probes of §9.2.

### 9.4 Descendants I: Double DQN & Dueling
- **F:** Overestimation: $\mathbb{E}[\max_{a'}\hat q]\ge\max_{a'}\mathbb{E}[\hat q]$ (Jensen applied to the max — Ch 6's maximization-bias math resurfacing at scale); Double DQN target $Y_t=r+\gamma\,\hat q\big(s',\arg\max_{a'}\hat q(s',a',\mathbf{w}),\mathbf{w}^-\big)$ — selection by the online net, evaluation by the target net, reusing the frozen copy as the second estimator (van Hasselt 2016, descending from Double Q-learning, van Hasselt 2010). Dueling decomposition $\hat q(s,a)=V(s)+A(s,a)-\frac{1}{|\mathcal{A}|}\sum_{a'}A(s,a')$; why the mean-subtraction is needed: $(V,A)$ is unidentifiable up to a per-state constant without it (Wang 2016).
- **C:** Overestimation race: mean $\max_a\hat q$ vs. DP ground truth for DQN vs. Double DQN on the same seeds — the inflation gap visibly closes. Dueling anatomy widget: hover any gridworld cell to see $V(s)$ and the advantage bars; in corridor cells the bars flatten (actions don't matter), at junctions they spike.
- **P:** Both descendants as `Trainer` config flags (< 40 changed lines each — the reader diffs them); the overestimation experiment scripted and seeded.

### 9.5 Descendants II: Prioritized Replay & n-Step Returns
- **F:** PER: priorities $p_i=|\delta_i|+\varepsilon_p$, sampling probability $P(i)=p_i^{\alpha_{\text{PER}}}/\sum_k p_k^{\alpha_{\text{PER}}}$; the induced bias and its importance-sampling correction $w_i=(N\,P(i))^{-\beta}/\max_j w_j$ with $\beta$ annealed to 1 (Schaul 2016) — an importance-ratio argument the reader knows from Ch 6; sum-tree: internal nodes store subtree priority sums, sample by drawing $u\sim U[0,p_{\text{total}})$ and descending, $O(\log N)$ insert/update/sample. n-step targets in replay $Y_t=\sum_{k=0}^{n-1}\gamma^k r_{t+k}+\gamma^n\max_{a'}\hat q(s_{t+n},a',\mathbf{w}^-)$; the honest caveat: stored $n$-step segments are off-policy w.r.t. the current greedy policy and uncorrected in practice — small $n$ (Rainbow's $n=3$) as the bias–variance-staleness compromise (Ch 7's dial, third appearance).
- **C:** `ch09-sumtree`: the tree drawn live with priority-proportional node widths; click "sample" to watch the descent path; sliders for $\alpha_{\text{PER}}$ (uniform ↔ greedy prioritization) and $\beta$, with the IS-weight histogram responding.
- **P:** `rl-core::replay::SumTree` + `PrioritizedBuffer` (unit-tested: sampling frequencies match priorities within tolerance, `proptest` for tree-sum invariants); sparse-reward gridworld variant where uniform replay stalls and PER doesn't — the motivating experiment, curves exported.

### 9.6 Distributional RL: C51
- **F:** The return as a random variable $Z^\pi(s,a)$ with $q_\pi=\mathbb{E}[Z^\pi]$; the distributional Bellman operator $(\mathcal{T}^\pi Z)(s,a)\overset{D}{=}R(s,a)+\gamma Z(S',A')$, $S'\sim p(\cdot|s,a),A'\sim\pi$; **theorem** (Bellemare 2017): $\mathcal{T}^\pi$ is a $\gamma$-contraction in the maximal Wasserstein metric $\bar d_p(Z_1,Z_2)=\sup_{s,a}W_p(Z_1(s,a),Z_2(s,a))$ — proof via the shift/scale invariances of $W_p$ ($W_p(A+X, A+Y) \le W_p(X,Y)$, $W_p(\gamma X,\gamma Y)=\gamma W_p(X,Y)$); the control operator is *not* a contraction in any distributional metric (stated with the counterexample's shape), yet expectations still converge to $q_*$. C51 parametrization: fixed atoms $z_i=V_{\min}+i\Delta z$, $i=0..50$, learned probabilities via softmax; the projection $\Phi$ distributing each target atom $r+\gamma z_j$ (clipped to $[V_{\min},V_{\max}]$) onto its two neighboring atoms proportionally; loss = cross-entropy $-\sum_i[\Phi\hat{\mathcal{T}}Z]_i\log p_i(s,a)$, whose sample gradient is unbiased (unlike a sampled Wasserstein loss — the theory/practice gap flagged honestly).
- **C:** `ch09-c51-morph` (signature widget): pick any gridworld cell-action; watch its 51-atom distribution get shifted by $r$, shrunk by $\gamma$, and re-projected onto the fixed support — the operator as visible mass transport. During training, distributions near hazards grow a visible low-return mode that the scalar $\hat q$ averaged away.
- **P:** `rl-deep::c51`: categorical head, projection as a vectorized `burn` op (no per-atom loop), same `Trainer` harness; risk-reading experiment: compare learned distributions at safe vs. hazardous cells of the cliff-variant gridworld.

### 9.7 Rainbow & Where Value-Based Methods Sit in Robotics
- **F:** Rainbow (Hessel 2018) as the six-way composition (Double + PER + Dueling + n-step + distributional + noisy nets — the last summarized, not implemented); reading its ablations: PER, n-step, and distributional carry most of the weight. Placement in robotics per Tang: mature real-world value-based successes concentrate where the action space is discrete or mid-level — grasping framed as bandit/classification over discrete grasp candidates executed open-loop (Tang §4.3.1.1), primitive/skill selection; on-policy PG dominates zero-shot sim-to-real for its hyperparameter robustness, while off-policy value methods are the sample-efficiency lever (Tang §6, Table 5); the argmax-over-continuum blocker that hands continuous torque control to Ch 10–11.
- **C:** Ablation bar-chart dashboard reproducing the book's own five-component ablation (noisy nets excluded) on visual gridworld; a Tang-derived map: competencies × solution-approach axis with value-based successes highlighted (grasping, discrete navigation) — the reader sees the territory Ch 20 will claim.
- **P:** `rl-deep::rainbow`: config-composed trainer (each extension a flag — the ablation study is a `clap` loop over flags, seeded ×5); WASM inference demo: trained Rusty runs in-browser on reader-edited maps with a Q-value (or C51 distribution) overlay.

### 9.8 Chapter Bridge
Recap: the triad was tamed, not refuted — replay restored the sampling assumptions, target networks restored a quasi-stationary contraction, and each descendant patched a named, measured pathology; Rusty navigates from pixels and its convnet + trainer become the Ch 20 grasp-selection template. What's next: none of this produces continuous torques — $\max_{a'}$ over a continuum is the blocker. Ch 10 abandons the value-first doctrine and optimizes the policy directly (on-policy, triad-free); Ch 11 then re-imports this chapter's replay + target machinery to make continuous control off-policy and sample-efficient. The `RingBuffer`/`SumTree`/target-sync code written here is deliberately the same code Ch 11 imports.

## 4. Interactive Widgets & Dashboards

| ID | Type | What it shows | Reader interaction | Tech |
|---|---|---|---|---|
| `ch09-table-to-net` | animation | Q-table dissolving into a convnet; receptive-field overlay on Rusty's pixels | Hover neurons/actions; scrub the morph | egui custom canvas |
| `ch09-replay-flow` | animation | Transitions env → ring buffer → minibatch; live gradient-autocorrelation meter | Toggle uniform vs. sequential sampling; buffer-size slider | egui |
| `ch09-two-clocks` | widget | Fast SGD clock vs. slow target clock; loss surface re-anchoring at each sync | Sync-period C slider; Polyak-τ mode; C=1 explosion | egui_plot |
| `ch09-stability-dash` | dashboard | Ablation curves: naive / +replay / +target / full DQN (loss, return, mean-max-Q vs. DP truth) | Ablation checkboxes; seed selector | egui + burn WASM |
| `ch09-overestimation` | widget | Mean maxₐ q̂ vs. ground truth, DQN vs. Double DQN | Seed selector; toggle Double | egui_plot |
| `ch09-dueling-anatomy` | widget | V(s) scalar + advantage bars per gridworld cell | Hover cells; corridor vs. junction presets | egui |
| `ch09-sumtree` | widget | Sum-tree with priority-proportional widths; animated sampling descent; IS-weight histogram | Click-sample; α_PER and β sliders | egui |
| `ch09-c51-morph` | animation | 51-atom distribution under shift → scale → projection Φ; hazard cells growing low-return modes | Pick (s,a); r and γ sliders; play training evolution | egui_plot |
| `ch09-rusty-plays` | sandbox | Trained agent inference on reader-edited maps, Q/distribution overlay | Paint walls/goals/hazards; step or run policy | burn WASM inference + egui |

## 5. Rust Implementation Plan

Crates: `rl-deep` gains `dqn`, `c51`, `rainbow`, `diag`; **`rl-core` gains `replay::{RingBuffer, SumTree, PrioritizedBuffer}`** (deliberately in `rl-core`: Ch 11 imports them unchanged); `rl-envs` gains `VisualGridworld` (+ cliff/sparse variants); `rl-viz` gains the telemetry-streaming dashboard shell. Training natively on WGPU; inference demo compiled to WASM (WebGL backend).

Representative sketch — the Double-DQN training step in `burn`:

```rust
/// rl-deep/src/dqn/step.rs — one minibatch update (Double-DQN target).
pub fn train_step<B: AutodiffBackend>(
    qnet: QNet<B>,
    target: &QNet<B>,            // periodically synced copy, never backprops
    batch: &Batch<B>,            // obs [N,3,64,64], action [N,1], reward [N,1], not_done [N,1]
    optim: &mut OptimizerAdaptor<Adam, QNet<B>, B>,
    cfg: &DqnConfig,
) -> (QNet<B>, f32) {
    // a* = argmax_a q̂(s',a,w): SELECT with the online net…
    let a_star = qnet.forward(batch.next_obs.clone()).argmax(1);         // [N,1]
    // …EVALUATE with the frozen target net (semi-gradient: detach).
    let q_next = target.forward(batch.next_obs.clone())
        .gather(1, a_star).detach();                                     // [N,1]
    let y = batch.reward.clone()
        + batch.not_done.clone() * q_next * cfg.gamma;                   // [N,1]
    let q_sa = qnet.forward(batch.obs.clone()).gather(1, batch.action.clone());
    let loss = HuberLossConfig::new(1.0)
        .init()
        .forward(q_sa, y, Reduction::Mean);
    let scalar = loss.clone().into_scalar().elem::<f32>();
    let grads = GradientsParams::from_grads(loss.backward(), &qnet);
    (optim.step(cfg.lr, qnet, grads), scalar)                            // for telemetry
}
```

Experiments/benchmarks: the four-way stability ablation (§9.3, seeded ×5); overestimation study vs. DP ground truth (§9.4); PER-vs-uniform on sparse rewards (§9.5); C51 hazard-distribution probe (§9.6); five-flag Rainbow ablation (§9.7); `criterion` bench on `SumTree` ops; `proptest` on buffer invariants. Native: all training runs + `egui` telemetry. In-browser: every widget above; `ch09-rusty-plays` runs full inference in WASM; `ch09-stability-dash` replays logged curves (training the convnet in-page is out of budget — noted per the WASM-feasibility rule).

## 6. Robot Thread

**Rusty** (protagonist): graduates from tabular warehouse coordinates (Ch 4–7) to raw pixels; leaves with a trained deep Q-network, a diagnosis toolkit, and a trainer that Ch 20 re-targets at discrete grasp candidates; Ch 19 later upgrades it again to lidar + continuous control. **Pendle** (sidebar only): the argmax-over-continuum blocker names why Pendle waits for Ch 10. Reacher and Ferris do not appear, but Ch 11's note that Reacher inherits this chapter's replay/target code is planted in §9.8.

## 7. Exercises & Explorations

1. **(F)** Prove $\mathbb{E}[\max_{a'}\hat q(s',a',\mathbf{w})]\ge\max_{a'}\mathbb{E}[\hat q(s',a',\mathbf{w})]$ for any random $\hat q$, and construct a two-action example where Double DQN's target is *under*-biased — the correction is not free.
2. **(F)** Complete the fitted-Q-iteration error bound of §9.3: from $\|\hat q_{k+1}-\mathcal{T}\hat q_k\|_\infty\le\varepsilon_k$, derive $\limsup_k\|\hat q_k-q_*\|_\infty\le\frac{\sup_k \varepsilon_k}{1-\gamma}$, and identify which step fails when the regression distribution $\bar\mu$ misses states that $\mathcal{T}$'s max cares about.
3. **(F)** Prove the two $W_p$ lemmas used in §9.6 ($W_p(A+X,A+Y)\le W_p(X,Y)$ for $A$ independent of $X,Y$; $W_p(\gamma X,\gamma Y)=\gamma W_p(X,Y)$) and assemble the $\bar d_p$ contraction of $\mathcal{T}^\pi$.
4. **(C)** In `ch09-two-clocks`, find for one fixed seed the smallest sync period $C$ that trains stably, then the largest $C$ that still reaches 95% of optimal within budget; explain both edges in terms of the two clocks.
5. **(C)** In `ch09-sumtree`, set $\beta=0$ (no IS correction) and $\alpha_{\text{PER}}=1$ on the sparse map; the return curve improves early and plateaus below the corrected run — connect what you see to the biased expectation PER induces.
6. **(P)** Add Polyak averaging (`τ` mode) to the trainer and replicate the §9.3 ablation; compare hard-sync vs. soft-sync stability at matched effective timescales ($\tau\approx 1/C$).
7. **(P)** Extend `VisualGridworld` with frame-stacking ($k=4$) and measure the effect on a variant where the goal blinks (partially observed without memory) — a POMDP teaser pointing at Ch 19's recurrent policies.
8. **(P, stretch)** Implement quantile regression DQN (QR-DQN) atop the C51 module (swap fixed-support + projection for learned quantiles + pinball loss) and compare hazard-cell distributions; note which of C51's two approximation steps disappears.

## 8. Notation Introduced

| Symbol | Meaning |
|---|---|
| $\mathcal{D}$, $N$ | replay buffer and its capacity (script D; roman $\mathbf{D}$ of Ch 8 remains the μ-diagonal matrix) |
| $\mathbf{w}^-$ | target-network weights; sync period $C$, Polyak rate $\tau$ |
| $Y_t$ | regression target (DQN / Double / n-step variants) |
| $\bar\mu$ | replay mixture distribution $\frac1N\sum_k \mu_{\pi_k}$ |
| $p_i,\ \alpha_{\text{PER}},\ \beta$ | PER priority and its exponents (subscripted to avoid clashing with step size $\alpha$) |
| $Z^\pi(s,a)$ | random return; $q_\pi=\mathbb{E}[Z^\pi]$ |
| $\mathcal{T}^\pi,\ \mathcal{T}$ | distributional Bellman expectation / optimality operators |
| $\bar d_p$, $W_p$ | maximal / plain $p$-Wasserstein metrics |
| $z_i,\ \Delta z,\ \Phi$ | C51 support atoms, spacing, projection onto the support |

## 9. References & Further Reading

- **S&B 2nd ed.**: §6.5 (Q-learning), §6.7 (maximization bias, Double Q-learning), §11.2–11.3 (Baird, deadly triad — the theory this chapter engineers around); published-edition §16.5 (DQN Atari case study). *The repository draft PDF predates the DQN era and contains none of the deep material; its ch. 10 (off-policy approximation) is a stub — cite the published edition.*
- **Kober, Bagnell & Peters (IJRR 2013)**: §4.2 (neural value-function approximation in robotics: global-generalization divergence, Boyan & Moore 1995; Riedmiller's Brainstormers RoboCup MLPs — the NFQ lineage DQN industrialized).
- **Tang et al. (2024)**: solution-approach axis (off-policy value-based methods; Table 5 usage patterns), §4.3.1.1 (grasping as bandit/classification over discrete grasp candidates — the Ch 20 payoff), §6 (sample-efficiency: off-policy as the lever beyond on-policy robustness).
- Mnih et al. (2015) — DQN: replay + target networks + convnet, Nature.
- Lin (1992) — experience replay, the original.
- Riedmiller (2005) — Neural Fitted Q-iteration, the fitted-value-iteration bridge §9.3 formalizes.
- van Hasselt (2010); van Hasselt, Guez & Silver (2016) — Double Q-learning; Double DQN.
- Wang et al. (2016) — Dueling networks.
- Schaul et al. (2016) — Prioritized experience replay.
- Bellemare, Dabney & Munos (2017) — C51 and the distributional Bellman contraction.
- Dabney et al. (2018) — QR-DQN (exercise 8).
- Hessel et al. (2018) — Rainbow and its ablations.

# Chapter 10 — Policy Gradients: REINFORCE → PPO

> **Part** II — Scaling Up: Function Approximation & Deep RL · **Builds on:** Ch 2 (gradients, Robbins–Monro), Ch 3 (gradient bandits — softmax preferences), Ch 8 (function approximation, on-policy stability) · **Feeds:** Ch 11 (off-policy actor–critic), Ch 15 (parallel-sim PPO), Ch 18 (locomotion PPO), Ch 19 (recurrent PPO), Ch 22 (capstone teacher–student PPO)
> **Modernizes:** S&B 2nd ed. ch. 13 (§13.1–13.7; the repo's draft PDF carries this material as ch. 11 "Policy Approximation" without the PG theorem — cite the published edition); Kober §2.2.2 (policy search: the robotics case for the primal problem); Schulman 2015/2016/2017 line (TRPO, GAE, PPO); Tang §4.1.1 remark + Table 5 (on-policy PPO as the real-world workhorse)

## 1. Purpose & Learning Outcomes

Everything so far routed control through a value function and an argmax. This chapter optimizes the policy itself: $\pi(a|s,\boldsymbol\theta)$, updated along $\nabla J(\boldsymbol\theta)$. That single move buys what robots need most — continuous actions, smooth policy change, and a place to install priors (Kober §2.2.2's argument, made formal) — at the price of gradient variance, which the rest of the chapter systematically drains: baselines, critics, GAE, trust regions, and finally PPO, the algorithm behind most of Tang's surveyed real-world successes. Pendle's swing-up — a task no amount of Ch 8 linear Sarsa solves elegantly — is the running payoff.

The reader can:
- Argue, with Kober's LQR parameter-count example, why robots often prefer policy search to value fitting.
- Derive the policy gradient theorem step by step for both episodic and continuing (average-reward) cases, and explain why $\nabla\mu$ never needs computing.
- Quantify REINFORCE's variance (the $O(T^3)$ scaling argument) and prove baseline unbiasedness.
- Derive GAE by telescoping $k$-step advantage estimators and place $\lambda$ on the bias–variance dial from Ch 7.
- Sketch TRPO: performance-difference lemma, surrogate objective, KL trust region, natural gradient.
- Analyze PPO's clipped surrogate piecewise (both signs of $\hat A$), and list the implementation details that decide whether it works: advantage normalization, entropy bonus, minibatch epochs, KL watchdog.
- Train a Gaussian-policy PPO in `burn` over `rayon`-vectorized environments and read its diagnostics (KL, clip fraction, entropy) like an engineer.

## 2. Storyline

**Act 1 — The argmax hits a wall.** Pendle hangs down. The task: swing up and balance, with one continuous force. Ch 9's recipe stalls at the first line — $\max_a$ over a continuum is an optimization problem *per timestep* — and coarse discretization visibly chatters. Kober §2.2.2 reframes the whole enterprise: solve the primal problem. Parametrize the controller and climb $J(\boldsymbol\theta)$: fewer parameters (LQR's policy is linear in state where its value is quadratic), room for structure, updates that move the robot's behavior smoothly ($\boldsymbol\theta_{i+1}=\boldsymbol\theta_i+\alpha\nabla_{\boldsymbol\theta}J$). The reader meets `ch10-theta-drag` and *feels* $J(\boldsymbol\theta)$ as terrain before any theorem: drag $\boldsymbol\theta$, watch $\pi$ morph and the return respond.

**Act 2 — The theorem and the variance war.** The policy gradient theorem is derived in full — the recursive unrolling of $\nabla v_\pi$, the emergence of the visitation weighting $\mu$, the miracle that the gradient never touches $\nabla\mu$ — then again for the continuing case, where stationarity makes the cancellation exact. REINFORCE falls out as the sampled form, works on a toy, and drowns on Pendle: the gradient-variance histograms go wide-screen. Each fix is derived, implemented, and measured on the same histograms: a baseline (unbiased, proved), a learned critic (biased, bounded, worth it), GAE's $\lambda$-dial (Ch 7's forward view reborn in advantage space). An A2C agent almost swings up — and then a too-large step collapses the policy to a deterministic wrong answer. Cue Act 3.

**Act 3 — Don't trust a step you can't measure.** The performance-difference lemma says exactly what a new policy gains; the TRPO bound says when the old policy's data can be trusted about it; the natural gradient reshapes steps by the Fisher metric — beautiful, and heavy. PPO is presented as the engineering distillation: clip the ratio, take the min, iterate minibatches, watch the KL. The clipped surrogate is dissected piecewise in `ch10-ppo-clip`, the "details that actually matter" get a section of their own, and vectorized PPO swings Pendle up in minutes on the reader's own machine — with the dashboard showing KL, clip fraction, and entropy behaving exactly as the math said they should. Closing frame: Tang's finding that this loop, robust to hyperparameters, is the algorithm real robots ship with (Ch 15/18/19/22 all reuse it).

**Robots:** Pendle is the protagonist (swing-up solved end-to-end). Reacher cameos in §10.1's action-space discussion (its continuous joint torques await Ch 11); Ferris is name-dropped as the Ch 18 customer of this exact PPO code.

## 3. Section-by-Section Design

### 10.1 Why Robots Prefer Policies
- **F:** Parametrized policies: softmax-over-preferences $\pi(a|s,\boldsymbol\theta)\propto e^{h(s,a,\boldsymbol\theta)}$ (Ch 3's gradient bandit, now with states) and the Gaussian policy $\pi(a|s,\boldsymbol\theta)=\mathcal{N}(a;\,\mu_{\boldsymbol\theta}(s),\,\sigma_{\boldsymbol\theta}^2)$ for continuous actions (S&B §13.7). Kober §2.2.2 formalized: (i) parameter economy — for LQR the optimal value is quadratic in state ($O(n^2)$ parameters) while the optimal policy is linear ($O(n)$); (ii) structure and priors enter as policy architecture without changing the problem; (iii) iterative local updates $\boldsymbol\theta_{i+1}=\boldsymbol\theta_i+\Delta\boldsymbol\theta_i$ keep behavior change bounded — a safety property values can't offer; (iv) exploration becomes differentiable (learned $\sigma$). Objectives: episodic $J(\boldsymbol\theta)=v_{\pi_{\boldsymbol\theta}}(s_0)$; the black-box alternative (evaluate $J$ only — finite differences, CMA-ES) parked for Ch 17.
- **C:** `ch10-theta-drag` (signature widget): left pane a 2-D $\boldsymbol\theta$ slice with $J(\boldsymbol\theta)$ contours (a 2-parameter linear-Gaussian controller on Pendle); right pane the induced policy — Gaussian action distribution over three probe states — morphing as the reader drags $\boldsymbol\theta$; a "follow $\nabla J$" button animates ascent; a ridge in the contour plot foreshadows why step size will be the chapter's villain.
- **P:** `rl-deep::policy`: `GaussianPolicy` (state-dependent mean via `burn` MLP, state-independent learnable $\log\sigma$) and `SoftmaxPolicy`; `log_prob`, `sample`, `entropy` methods unit-tested against `statrs` closed forms; `rl-envs::PendleSwingUp` (torque-limited, reward $\cos\theta - c_u u^2$, from the Ch 2 ODE).

### 10.2 The Policy Gradient Theorem
- **F:** Episodic derivation in full (S&B §13.2): $\nabla v_\pi(s)=\sum_a\big[\nabla\pi(a|s,\boldsymbol\theta)\,q_\pi(s,a)+\pi(a|s,\boldsymbol\theta)\sum_{s'}p(s'|s,a)\,\gamma\nabla v_\pi(s')\big]$, unrolled recursively to $\nabla v_\pi(s_0)=\sum_x\sum_{k=0}^\infty \gamma^k\Pr(s_0\!\to\!x,k,\pi)\sum_a\nabla\pi(a|x,\boldsymbol\theta)\,q_\pi(x,a)$; defining $\eta(s)=\sum_k\gamma^k\Pr(s_0\!\to\!s,k,\pi)$ and $\mu(s)=\eta(s)/\sum_{s'}\eta(s')$ gives $\nabla J(\boldsymbol\theta)\propto\sum_s\mu(s)\sum_a q_\pi(s,a)\,\nabla\pi(a|s,\boldsymbol\theta)$ — every algebraic step shown, and the punchline stressed: the environment's $p$ appears only through sampling; $\nabla\mu$ is never needed. Continuing case (S&B §13.6): $J(\boldsymbol\theta)=r(\pi)$, differential values $\tilde v,\tilde q$ with $r(\pi)$ subtracted in the Bellman equation; solving the $\nabla v$ recursion for $\nabla r(\pi)$, weighting by the stationary $\mu$, and using $\mu^\top\mathbf{P}_\pi=\mu^\top$ to cancel the $\sum_s\mu(s)\nabla\tilde v(s)$ terms on both sides — here the theorem holds with equality. Score-function form: $\nabla J=\mathbb{E}_\pi[q_\pi(S,A)\,\nabla\ln\pi(A|S,\boldsymbol\theta)]$ via $\nabla\pi=\pi\nabla\ln\pi$.
- **C:** Derivation stepper: the unrolling rendered as an expanding tree over a 3-state MDP — each application of the recursion visibly pushes $\nabla$ one step deeper while $q_\pi\nabla\pi$ terms accumulate at visited nodes with $\gamma^k$-fading weights; the reader steps until the tree's leaf contribution vanishes and the $\mu$-weighted sum remains.
- **P:** Finite-MDP verification harness: exact $\nabla J$ by autodiff through the (differentiable) linear-system solve for $v_\pi$ vs. the theorem's $\mu$-weighted sum vs. a score-function Monte Carlo estimate — three numbers, one `proptest` over random 4-state MDPs, agreement to tolerance. The theorem as a unit test.

### 10.3 REINFORCE and the Variance Problem
- **F:** From all-actions to sampled: $\nabla J\propto\mathbb{E}_\pi[G_t\nabla\ln\pi(A_t|S_t,\boldsymbol\theta)]$; REINFORCE update $\boldsymbol\theta\leftarrow\boldsymbol\theta+\alpha\,\gamma^t G_t\nabla\ln\pi(A_t|S_t,\boldsymbol\theta)$ (S&B §13.3); convergence under Robbins–Monro (unbiased gradient + decaying $\alpha$). Variance analysis done concretely: for a length-$T$ trajectory the whole-trajectory estimator is $\big(\sum_t\nabla\ln\pi_t\big)\,R(\tau)$ — with per-step rewards of size $r_{\max}$, $R(\tau)=O(T\,r_{\max})$ multiplies a random walk of $T$ score terms with variance $O(T)$, giving estimator variance $O(T^3 r_{\max}^2)$; worked Gaussian case: $\nabla_{\mu}\ln\pi=(a-\mu)/\sigma^2$, so variance also scales as $1/\sigma^2$ — precision *raises* gradient noise, the exploration–estimation tension robots live in. Causality refinement (rewards before $t$ don't depend on $A_t$; proof via iterated expectations) → reward-to-go $G_t$, the first free variance cut.
- **C:** `ch10-variance-hist` (first appearance): live histogram of 1000 single-trajectory gradient estimates (one shared coordinate) on Pendle swing-up under REINFORCE — the reader watches the spread dwarf the mean, toggles reward-to-go, and sees the first visible shrink; a running "trajectories needed for a reliable sign" counter makes variance operational.
- **P:** `rl-deep::pg::Reinforce`; on a short-horizon Pendle ($T{=}100$) it crawls to swing-up in ~hours-equivalent samples — logged honestly as the baseline the next three sections beat; the variance-probe instrument (per-coordinate estimator statistics) that every later run reuses.

### 10.4 Baselines, Advantage & Actor–Critic
- **F:** Baseline invariance: $\sum_a b(s)\nabla\pi(a|s,\boldsymbol\theta)=b(s)\nabla\sum_a\pi(a|s,\boldsymbol\theta)=b(s)\nabla 1=0$ — full proof, so $\nabla J\propto\mathbb{E}[(G_t-b(S_t))\nabla\ln\pi]$ for any state-dependent $b$ (S&B §13.4); variance-optimal per-coordinate baseline $b^*_i=\mathbb{E}[G\,(\partial_i\ln\pi)^2]/\mathbb{E}[(\partial_i\ln\pi)^2]$ stated, with $b=v_\pi$ as the practical near-optimum. Advantage $A^\pi(s,a)=q_\pi(s,a)-v_\pi(s)$; the one-step TD error as advantage estimator: $\mathbb{E}[\delta_t|S_t,A_t]=A^\pi(S_t,A_t)$ when $\hat v=v_\pi$ — and the bias when it isn't (bootstrapping enters, Ch 8's trade re-made deliberately, this time on-policy so the triad stays broken). One-step actor–critic: $\boldsymbol\theta\leftarrow\boldsymbol\theta+\alpha_\theta\,\delta_t\nabla\ln\pi$, $\mathbf{w}\leftarrow\mathbf{w}+\alpha_w\,\delta_t\nabla\hat v$ (S&B §13.5) — two learners, one TD error.
- **C:** `ch10-variance-hist` (second appearance): +baseline and +critic-$\delta$ histograms stack under REINFORCE's — spread shrinking, mean drifting slightly off-center for the critic (bias made visible); a bias/variance crosshair plot tracks each estimator as a point.
- **P:** `rl-deep::pg::A2c` (synchronous advantage actor–critic; shared trunk, separate heads); first `rayon` vectorization: $K{=}16$ Pendle instances stepped in parallel per update — the batch that makes histograms tight enough to train on.

### 10.5 GAE: The Bias–Variance Dial
- **F:** $k$-step advantage estimators via telescoping: $\hat A_t^{(k)}=\sum_{l=0}^{k-1}\gamma^l\delta_{t+l}=-\hat v(s_t)+r_t+\gamma r_{t+1}+\dots+\gamma^k\hat v(s_{t+k})$ — the telescope shown term by term; exponential averaging $\hat A_t^{\text{GAE}(\gamma,\lambda)}=(1-\lambda)\sum_{k\ge 1}\lambda^{k-1}\hat A_t^{(k)}=\sum_{l\ge 0}(\gamma\lambda)^l\delta_{t+l}$ (Schulman 2016), the geometric-series algebra done in full; endpoints: $\lambda{=}0\Rightarrow\delta_t$ (max bias, min variance), $\lambda{=}1\Rightarrow G_t-\hat v(s_t)$ (unbiased given only the baseline, max variance) — Ch 7's $\lambda$-return, reborn in advantage space, with the same forward-view semantics; $\gamma$ vs. $\lambda$ roles separated ($\gamma$ changes the objective, $\lambda$ only the estimator).
- **C:** `ch10-gae-dial`: a real logged Pendle trajectory with its $\delta_l$ spikes; dragging $\lambda$ morphs the weight envelope $(\gamma\lambda)^l$ over them and re-composes $\hat A_t$ live; the bias/variance crosshair from §10.4 traces a curve as $\lambda$ sweeps — the dial made visible, with a back-reference: "drag to 1 and you have rediscovered Monte Carlo advantages."
- **P:** `gae()` as a single reverse-scan (`iter().rev().scan`) over rollout buffers — 8 lines, property-tested against the $O(T^2)$ definitional double sum; $\lambda$-sweep experiment on Pendle: final return and gradient-noise scale vs. $\lambda\in\{0,0.9,0.95,0.99,1\}$, reproducing the classic interior optimum.

### 10.6 Natural Gradient & Trust Regions: TRPO in Sketch
- **F:** Performance-difference lemma: $J(\pi')-J(\pi)=\mathbb{E}_{\tau\sim\pi'}\big[\sum_t\gamma^t A^\pi(s_t,a_t)\big]$, proved by telescoping $v_\pi$ along $\pi'$'s trajectories; the surrogate $L_\pi(\pi')=J(\pi)+\mathbb{E}_{s\sim d^\pi,a\sim\pi'}[A^\pi(s,a)]$ replaces $d^{\pi'}$ by $d^\pi$ — usable from on-policy data via the ratio $\tfrac{\pi'(a|s)}{\pi(a|s)}$; TRPO's guarantee (Schulman 2015, stated with constants, monotonic-improvement proof sketched): $J(\pi')\ge L_\pi(\pi')-C\max_s D_{\mathrm{KL}}(\pi\|\pi')[s]$, $C=\tfrac{4\epsilon\gamma}{(1-\gamma)^2}$ — improve the surrogate, pay a KL penalty, never move further than the data can testify. Practical TRPO: maximize $L$ s.t. $\bar D_{\mathrm{KL}}\le\delta_{\mathrm{TR}}$; second-order expansion makes the constraint $\tfrac12\Delta\boldsymbol\theta^\top\mathbf{F}\Delta\boldsymbol\theta\le\delta_{\mathrm{TR}}$ with Fisher $\mathbf{F}=\mathbb{E}[\nabla\ln\pi\,\nabla\ln\pi^\top]$, giving the natural-gradient direction $\mathbf{F}^{-1}\mathbf{g}$ (Kakade 2002) — steepest ascent in *policy* space, invariant to reparametrization (the ridge from `ch10-theta-drag` explained); cost: conjugate-gradient solves + line search. Kober's closing echo: robot policy-search methods succeed when constrained to modest path-distribution change — the trust region is that instinct, proved.
- **C:** `ch10-trust-region`: on the §10.1 $J(\boldsymbol\theta)$ contours, the KL ball drawn as an ellipse (its shape from $\mathbf{F}$, changing as $\boldsymbol\theta$ moves); vanilla-gradient vs. natural-gradient steps compared from the same start — vanilla falls off the ridge at large $\alpha$, natural walks it; a $\delta_{\mathrm{TR}}$ slider grows the ellipse until the monotonic-improvement bound visibly breaks.
- **P:** No full TRPO implementation (deliberate — stated in the text): a 2-parameter didactic natural-gradient ascent with exact $\mathbf{F}$ powers the widget; the engineering budget goes to PPO, with a sidebar note on why the field did the same.

### 10.7 PPO in Full
- **F:** Ratio $r_t(\boldsymbol\theta)=\tfrac{\pi_{\boldsymbol\theta}(a_t|s_t)}{\pi_{\boldsymbol\theta_{\text{old}}}(a_t|s_t)}$; clipped surrogate $L^{\text{CLIP}}(\boldsymbol\theta)=\mathbb{E}\big[\min\big(r_t\hat A_t,\ \text{clip}(r_t,1{-}\epsilon,1{+}\epsilon)\hat A_t\big)\big]$ (Schulman 2017), analyzed piecewise: for $\hat A_t>0$ the objective is flat for $r_t>1{+}\epsilon$ (no incentive to overshoot); for $\hat A_t<0$, flat for $r_t<1{-}\epsilon$; the $\min$ makes $L^{\text{CLIP}}$ a pointwise *lower bound* (pessimistic estimate) of the unclipped surrogate that only removes incentive, never adds it — and gradients vanish exactly where a datum has already "spent" its trust budget. Full objective $L=L^{\text{CLIP}}-c_v\,L^{\text{VF}}+c_H\,\mathcal{H}[\pi_{\boldsymbol\theta}]$; Gaussian entropy $\mathcal{H}=\sum_i(\log\sigma_i+\tfrac12\log 2\pi e)$. **The details that actually matter**, each justified from this chapter's math: per-batch advantage normalization (harmless per §10.4's baseline invariance — an affine reweighting — and it standardizes the clip scale); multiple epochs of minibatch SGD over each rollout (why clipping exists at all: the data goes off-policy within the update); a KL watchdog (early-stop the epoch loop when $\bar D_{\mathrm{KL}}$ exceeds a target — the trust region PPO doesn't formally have, reinstalled empirically); entropy floor vs. premature $\sigma$-collapse (Act 2's failure, named); observation/return normalization; gradient-norm clipping; clip fraction as the health metric the theory predicts should sit in a band. Tang grounding: PPO's hyperparameter robustness is *why* it is the predominant algorithm in mature zero-shot sim-to-real systems (Tang §4.1.1 remark, Table 5) — the surveyed reality this section trains the reader to reproduce.
- **C:** `ch10-ppo-clip` (signature widget): $L^{\text{CLIP}}$ vs. $r$ for draggable $\hat A$ (both signs) and $\epsilon$ — gradient-active and gradient-dead zones shaded; a second tab scatter-plots a real minibatch's $(r_t,\hat A_t)$ pairs on the same axes with clipped points dimmed, tying the cartoon to live training. `ch10-ppo-dashboard`: return, $\bar D_{\mathrm{KL}}$, clip fraction, entropy, explained-variance of $\hat v$ — the reader watches entropy anneal and clip fraction stabilize near 0.1–0.2 as swing-up is solved.
- **P:** `rl-deep::pg::Ppo`: rollout buffer over $K{=}32$ `rayon`-parallel `PendleSwingUp` instances (steps: collect $K{\times}T$ → GAE → 4 epochs × minibatches → KL watchdog); solves swing-up+balance in < 5 min native WGPU, seeded; checkpoint exported for the WASM demo `ch10-swingup-demo` (reader shoves the pole with the mouse; the trained Gaussian policy recovers — deterministic-mean vs. stochastic action toggle).

### 10.8 Chapter Bridge
Recap: the policy gradient theorem gave an estimable gradient with no $\nabla\mu$; variance was drained in four derived steps (reward-to-go, baseline, critic-$\delta$, GAE); trust regions turned Kober's "modest changes" instinct into a bound, and PPO shipped it as a loop robots actually run. Pendle swings up; the PPO module written here is imported *unchanged* by Ch 15 (randomized-physics farms), Ch 18 (Ferris walks with it), Ch 19 (a GRU goes in the trunk), and Ch 22 (teacher–student capstone). What's next: PPO discards every rollout after one update — Ch 11 asks what it costs to keep them (off-policy actor–critic: DDPG/TD3/SAC, importing Ch 9's replay machinery), because on a real robot every sample wears the bearings (Tang §6's sample-efficiency challenge).

## 4. Interactive Widgets & Dashboards

| ID | Type | What it shows | Reader interaction | Tech |
|---|---|---|---|---|
| `ch10-theta-drag` | widget | 2-D θ slice with J(θ) contours; induced Gaussian policy at probe states morphing | Drag θ; "follow ∇J"; ridge preset | egui_plot |
| `ch10-pg-derivation` | animation | The ∇v_π recursion unrolling as a tree; γᵏ-weighted q∇π terms accumulating into the μ-sum | Step/auto-play recursion depth; episodic↔continuing toggle | egui custom canvas |
| `ch10-variance-hist` | widget | Histograms of per-coordinate gradient estimates: REINFORCE / +reward-to-go / +baseline / critic-δ / GAE; bias–variance crosshair | Toggle estimators; sample-count slider; σ slider (1/σ² scaling) | egui_plot |
| `ch10-gae-dial` | widget | δ-spikes of a real Pendle trajectory under the (γλ)ˡ envelope; recomposed Â_t | Drag λ∈[0,1]; γ slider; crosshair trace | egui_plot |
| `ch10-trust-region` | animation | KL ellipse from F on J(θ) contours; vanilla vs. natural steps; bound breaking as δ_TR grows | Drag δ_TR and start point; step both methods | egui_plot |
| `ch10-ppo-clip` | widget | L^CLIP vs. r piecewise for both signs of Â; live minibatch (r, Â) scatter with clipped points dimmed | Drag ε and Â; switch cartoon/live tabs | egui_plot |
| `ch10-ppo-dashboard` | dashboard | Live PPO on swing-up: return, KL, clip fraction, entropy, explained variance | Pause; lr/ε/c_H sliders; seed reset; KL-watchdog toggle | egui + burn WASM |
| `ch10-swingup-demo` | sandbox | Trained swing-up policy running; reader perturbs the pole | Mouse-shove cart/pole; deterministic↔stochastic toggle | burn WASM inference + egui |

## 5. Rust Implementation Plan

Crates: `rl-deep` gains `policy` (`GaussianPolicy`, `SoftmaxPolicy`), `pg` (`Reinforce`, `A2c`, `Ppo`, `gae`, rollout buffer), `variance probe` in `diag`; `rl-envs` gains `PendleSwingUp` and the `VecEnv` `rayon` wrapper (reused by Ch 15/18); `rl-viz` gains the multi-metric PPO dashboard panel. Training native (WGPU); `ch10-ppo-dashboard` retrains the small MLP live in-browser (WebGL — feasible at Pendle scale, unlike Ch 9's convnet), `ch10-swingup-demo` is inference-only.

Representative sketch — the PPO minibatch loss with a Gaussian policy in `burn`:

```rust
/// rl-deep/src/pg/ppo.rs — clipped-surrogate loss for one minibatch.
/// obs [N,d_s], act [N,d_a], logp_old/adv/ret [N]; adv pre-normalized per batch.
pub fn ppo_loss<B: AutodiffBackend>(
    pi: &GaussianPolicy<B>,
    vf: &ValueNet<B>,
    mb: &Minibatch<B>,
    cfg: &PpoConfig,          // eps, c_v, c_h
) -> PpoLossOut<B> {
    let (mean, log_std) = pi.forward(mb.obs.clone());
    let logp = gaussian_log_prob(mean, log_std.clone(), mb.act.clone()); // [N]
    let ratio = (logp.clone() - mb.logp_old.clone()).exp();             // r_t(θ)
    let surr = ratio.clone() * mb.adv.clone();
    let surr_clip = ratio.clone()
        .clamp(1.0 - cfg.eps, 1.0 + cfg.eps) * mb.adv.clone();
    let l_clip = surr.min_pair(surr_clip).mean();                        // E[min(·,·)]
    let l_vf = (vf.forward(mb.obs.clone()).squeeze(1) - mb.ret.clone())
        .powf_scalar(2.0).mean();
    let entropy = (log_std + 0.5 * (2.0 * PI * E).ln()).sum_dim(1).mean();
    let loss = -l_clip + l_vf * cfg.c_v - entropy.clone() * cfg.c_h;
    // Diagnostics the dashboard streams: approx-KL and clip fraction.
    let approx_kl = (mb.logp_old.clone() - logp).mean();                 // E[-log r]
    let clipped = ratio.sub_scalar(1.0).abs().greater_elem(cfg.eps);
    PpoLossOut { loss, approx_kl, clip_frac: clipped.float().mean(), entropy }
}
```

Experiments/benchmarks: the PG-theorem `proptest` verification (§10.2); estimator-variance study across the five estimators at matched sample budgets (§10.3–10.5, seeded ×5); $\lambda$-sweep (§10.5); REINFORCE vs. A2C vs. PPO learning curves on swing-up (the chapter's summary figure); PPO ablation grid {advantage-norm, entropy bonus, KL watchdog, clipping} on/off — reproducing "the details are the algorithm"; `criterion` bench on `VecEnv` rollout throughput (steps/s vs. thread count). Native: all trainers and studies. In-browser: all eight widgets; live PPO training at Pendle scale; swing-up inference demo.

## 6. Robot Thread

**Pendle** (protagonist): arrives balanced-only (Ch 8's linear Sarsa, small-angle regime); leaves swung-up and robust to shoves, controlled by a Gaussian PPO policy — its Ch 12 role (model-based MPC on the same task) sets up the model-free vs. model-based comparison. **Reacher** (cameo): named in §10.1 as the continuous-torque customer whose task needs Ch 11's sample-efficient off-policy methods. **Ferris** (foreshadowed): §10.8 states that Ch 18's locomotion pipeline is this chapter's `Ppo` + `VecEnv` with a bigger environment. **Rusty**: rests; returns for recurrent PPO navigation in Ch 19.

## 7. Exercises & Explorations

1. **(F)** Re-derive the policy gradient theorem for the episodic *undiscounted* case ($\gamma=1$, proper episodes) and identify where the average-episode-length proportionality constant enters — then show why it is absorbed into $\alpha$ without harm.
2. **(F)** Prove the causality step of §10.3: $\mathbb{E}[R_k\,\nabla\ln\pi(A_t|S_t,\boldsymbol\theta)]=0$ for $k\le t$, via iterated expectations conditioning on $(S_t)$ — hence reward-to-go is unbiased.
3. **(F)** Derive the variance-optimal constant baseline $b^*=\mathbb{E}[G\|\nabla\ln\pi\|^2]/\mathbb{E}[\|\nabla\ln\pi\|^2]$ by differentiating the estimator variance, and show $b=\mathbb{E}[G]$ is suboptimal whenever $G$ and $\|\nabla\ln\pi\|^2$ are correlated.
4. **(F)** Verify the GAE geometric-series algebra: expand $(1-\lambda)\sum_{k\ge1}\lambda^{k-1}\hat A^{(k)}_t$ and regroup by $\delta_{t+l}$ to obtain $\sum_l(\gamma\lambda)^l\delta_{t+l}$; state exactly which exchange of summations requires $\lambda<1$ or truncation.
5. **(C)** In `ch10-variance-hist`, shrink $\sigma$ toward 0.05 and watch the REINFORCE histogram widen as $1/\sigma^2$ predicts; find the $\sigma$ below which even GAE's sign is unreliable at 64 samples — the exploration floor, measured.
6. **(C)** In `ch10-ppo-clip`'s live tab, raise the learning rate until the KL watchdog fires every epoch; correlate the clip-fraction spike with the return dip on `ch10-ppo-dashboard`, then restore health using only $\epsilon$.
7. **(P)** Add a state-*dependent* $\log\sigma(s)$ head to `GaussianPolicy` and rerun swing-up: measure entropy trajectories and final robustness to shoves vs. the state-independent default; explain the failure mode that motivates the default.
8. **(P, stretch)** Implement the adaptive-KL-penalty PPO variant ($L^{\text{KLPEN}}$ with $\beta_{\mathrm{KL}}$ doubling/halving) and race it against clipping on the ablation grid; reproduce the finding that clipping's advantage is robustness, not asymptotic performance.

## 8. Notation Introduced

| Symbol | Meaning |
|---|---|
| $\boldsymbol\theta$, $\pi(a|s,\boldsymbol\theta)$ | policy parameters and parametrized policy ($\mathbf{w}$ stays the critic's) |
| $J(\boldsymbol\theta)$ | scalar objective: $v_{\pi_{\boldsymbol\theta}}(s_0)$ episodic; $r(\pi)$ continuing |
| $\eta(s)$, $\mu(s)$ | (discounted) visitation measure and its normalization; stationary dist. in continuing case |
| $r(\pi)$, $\tilde v,\tilde q$ | average reward and differential value functions |
| $\nabla\ln\pi(a|s,\boldsymbol\theta)$ | score function |
| $b(s)$, $A^\pi(s,a)$ | baseline; advantage $q_\pi-v_\pi$ |
| $\hat A_t^{(k)}$, $\hat A_t^{\text{GAE}(\gamma,\lambda)}$ | $k$-step and GAE advantage estimators |
| $d^\pi$, $L_\pi(\pi')$ | discounted state distribution; surrogate objective |
| $\mathbf{F}$, $\delta_{\mathrm{TR}}$ | Fisher information matrix; trust-region radius |
| $r_t(\boldsymbol\theta)$, $\epsilon$, $L^{\text{CLIP}}$ | probability ratio, clip range, clipped surrogate |
| $c_v, c_H$, $\mathcal{H}[\pi]$ | value/entropy loss coefficients; policy entropy |

## 9. References & Further Reading

- **S&B 2nd ed.**: ch. 13 — §13.1 (policy approximation & its advantages), §13.2 (policy gradient theorem, episodic proof), §13.3 (REINFORCE), §13.4 (baseline), §13.5 (actor–critic), §13.6 (continuing case / average reward), §13.7 (Gaussian policies for continuous actions). *The repository draft PDF carries only a precursor as its ch. 11 "Policy Approximation" (actor–critic + R-learning, no PG theorem) — cite the published edition.*
- **Kober, Bagnell & Peters (IJRR 2013)**: §2.2.2 (policy search: primal formulation, LQR parameter-count argument, prestructuring, $\theta_{i+1}=\theta_i+\alpha\nabla_\theta J$, black-box vs. white-box); §4.2–§4.3 context (methods succeed when constrained to modest policy/path-distribution changes — the trust-region instinct).
- **Tang et al. (2024)**: §4.1.1 remark on RL algorithms (PPO predominant in zero-shot sim-to-real for hyperparameter robustness), Table 5 (algorithm-usage patterns), §6 (on-policy sample-cost as the open challenge Ch 11 takes up).
- Williams (1992) — REINFORCE.
- Sutton, McAllester, Singh & Mansour (2000) — the policy gradient theorem with function approximation.
- Kakade (2002) — natural policy gradient; Peters & Schaal (2008) — natural actor–critic in robotics (the Kober-era bridge).
- Schulman et al. (2015) — TRPO (monotonic-improvement bound).
- Schulman et al. (2016) — GAE.
- Schulman et al. (2017) — PPO (clipped surrogate; KL-penalty variant of exercise 8).
- Andrychowicz et al. (2021) — *What Matters in On-Policy RL* (the ablation culture behind §10.7's "details").

# Chapter 8 — Function Approximation & the Deadly Triad

> **Part** II — Scaling Up: Function Approximation & Deep RL · **Builds on:** Ch 2 (gradients, Robbins–Monro, contractions), Ch 6 (TD(0), Sarsa, Q-learning, on/off-policy), Ch 7 (n-step returns, traces) · **Feeds:** Ch 9 (DQN), Ch 10 (policy gradients), Ch 11 (off-policy continuous control), Ch 12 (learned models)
> **Modernizes:** S&B 2nd ed. ch. 9 (§9.1–9.5, 9.7), ch. 10 (§10.1–10.2), ch. 11 (§11.2–11.4, 11.7); Kober §2.4 (function approximation), §4.2 (value-function approximation in robotics)

## 1. Purpose & Learning Outcomes

Tabular RL dies the moment Pendle's cart position becomes a real number. This chapter makes the leap from tables to parameterized functions $\hat v(s,\mathbf{w})$ honestly: what objective we are actually optimizing, why TD with approximation is *not* gradient descent, when linear methods provably converge, how to build features that work on a real robot state space, and exactly how the combination of approximation + bootstrapping + off-policy learning destroys convergence — demonstrated live, not asserted.

The reader can:
- State the $\overline{\text{VE}}$ objective and explain why the on-policy distribution $\mu$ weights it.
- Derive semi-gradient TD(0) from true SGD and identify the dropped term that makes it "semi".
- Derive the linear TD fixed point $\mathbf{w}_{TD}=\mathbf{A}^{-1}\mathbf{b}$ and prove why on-policy sampling makes $\mathbf{A}$ positive definite.
- Implement tile coding from scratch and choose tilings/offsets for a 4-D robot state space.
- Reproduce Baird's counterexample and explain the divergence via the eigenvalues of the expected-update matrix.
- Name the deadly triad's three ingredients, say which one each practical fix removes, and sketch gradient-TD (TDC).
- Train a first `burn` MLP value function and compare its stability against tile coding empirically.

## 2. Storyline

**Act 1 — The wall.** Pendle returns from Chapter 6's world with an insult: discretize its 4-D cart-pole state $(x,\dot x,\theta,\dot\theta)$ at a modest 100 bins per dimension and the Q-table needs $10^8$ rows, almost all forever unvisited. Kober §2.4 names the diagnosis: in continuous state spaces tabular representation is intractable and *generalization* — not memorization — is the actual job. The reader watches Rusty's Chapter 5 state-explosion experiment replayed with Pendle and loses.

**Act 2 — The projection.** We reframe prediction as supervised learning onto a subspace: pick features, accept that $v_\pi$ is not representable, and aim for the best projection under the distribution the robot actually visits. Semi-gradient TD is derived, linear methods get their complete geometry (projection operator, fixed point, $\frac{1}{1-\gamma}$ error bound), and feature construction becomes an engineering craft the reader practices on Pendle: polynomials, Fourier basis, RBFs, and tile coding built for real. A first neural network enters as "features that learn themselves." Semi-gradient Sarsa with tile coding balances the pole — the chapter's working victory.

**Act 3 — The betrayal.** Then we do everything "right" off-policy and it explodes. Baird's counterexample diverges live on screen while the reader adjusts $\alpha$ in vain; the analysis shows divergence is structural (an eigenvalue with negative real part in $-\mathbf{A}$), not a tuning problem. The deadly triad is named, each escape route is mapped to a later chapter (target networks → Ch 9, on-policy PG → Ch 10, gradient-TD here in brief), and Pendle ends the chapter balanced by a linear learner but haunted: every deep method to come lives inside the triad.

**Robots:** Pendle is the protagonist (continuous state, first approximate value functions, first `burn` network). Rusty cameos in Act 1 (tabular nostalgia) and returns in Ch 9 with pixels.

## 3. Section-by-Section Design

### 8.1 Why Robots Outgrow Tables
- **F:** Memory and sample-complexity cost of discretization: $\prod_i n_i$ cells for resolution $n_i$ per dimension; generalization defined as sharing updates across states; the supervised-learning framing and its violated assumption — RL data is neither i.i.d. nor stationary, and the approximator shapes its own future data because it defines the policy that collects it (Kober §2.4). Formal statement of the prediction task: find $\mathbf{w}\in\mathbb{R}^d$, $d\ll|\mathcal{S}|$, with $\hat v(s,\mathbf{w})\approx v_\pi(s)$.
- **C:** Reprise of the Ch 5 exponential-wall visualizer, now with Pendle's 4-D state: a bin-resolution slider shows table size racing past RAM while a coverage meter shows the fraction of cells ever visited by 10k episodes collapsing toward zero.
- **P:** `rl-deep` crate is created (workspace member, `burn` + `ndarray` deps). Experiment: tabular Q-learning from `rl-tabular` on discretized cart-pole at 3 resolutions — the coarse one underfits, the fine one never converges in budget; numbers logged for the chapter's motivating table.

### 8.2 Prediction as Projection: the VE Objective and Semi-Gradient TD
- **F:** The objective $\overline{\text{VE}}(\mathbf{w})=\sum_s \mu(s)\,[v_\pi(s)-\hat v(s,\mathbf{w})]^2$ (S&B eq. 9.1) with $\mu$ the on-policy distribution; why uniform weighting is the wrong contract for a robot that lives on a low-dimensional manifold of its state space. True SGD: $\mathbf{w}_{t+1}=\mathbf{w}_t+\alpha[v_\pi(S_t)-\hat v(S_t,\mathbf{w}_t)]\nabla\hat v(S_t,\mathbf{w}_t)$; substitute target $U_t$: Monte Carlo ($U_t=G_t$) is unbiased → true SGD, converges to a local optimum under the Robbins–Monro conditions of Ch 2. TD target $U_t=R_{t+1}+\gamma\hat v(S_{t+1},\mathbf{w})$ depends on $\mathbf{w}$: the full gradient of $\tfrac12\delta_t^2$ is $\delta_t[\nabla\hat v(S_t,\mathbf{w})-\gamma\nabla\hat v(S_{t+1},\mathbf{w})]$ (residual gradient, Baird 1995); semi-gradient TD(0) keeps only the first term. Derivation of why the residual-gradient objective (Bellman error) needs double sampling: $\mathbb{E}[\delta]^2\neq\mathbb{E}[\delta^2]$, two independent successor samples required — impossible from a single robot trajectory (S&B §11.5).
- **C:** "Gradient or not?" widget: a 2-weight toy chain shows the semi-gradient vector field vs. the residual-gradient field vs. the true $\overline{\text{VE}}$ gradient field; the reader toggles fields and traces trajectories, seeing that semi-gradient TD follows no scalar potential yet still spirals into a sensible point on-policy.
- **P:** `rl-deep::linear`: `GradientMc` and `SemiGradientTd0` for arbitrary `Features` trait objects; validation on the 1000-state random walk (S&B §9.1's testbed re-created in `rl-envs`) with state aggregation, reproducing the characteristic staircase value plot.

### 8.3 Linear Methods and Their Geometry
- **F:** $\hat v(s,\mathbf{w})=\mathbf{w}^\top\mathbf{x}(s)$, $\nabla\hat v=\mathbf{x}(s)$. Expected update $\mathbb{E}[\Delta\mathbf{w}]=\alpha(\mathbf{b}-\mathbf{A}\mathbf{w})$ with $\mathbf{A}=\mathbb{E}[\mathbf{x}_t(\mathbf{x}_t-\gamma\mathbf{x}_{t+1})^\top]=\mathbf{X}^\top\mathbf{D}(\mathbf{I}-\gamma\mathbf{P})\mathbf{X}$, $\mathbf{b}=\mathbb{E}[R_{t+1}\mathbf{x}_t]$; fixed point $\mathbf{w}_{TD}=\mathbf{A}^{-1}\mathbf{b}$. Full proof that $\mathbf{D}(\mathbf{I}-\gamma\mathbf{P})$ is positive definite when $\mu$ is the stationary distribution of $\mathbf{P}$ (positive diagonal, negative off-diagonals, row sums positive, column sums $(1-\gamma)\mu^\top>0$ by stationarity $\mu^\top\mathbf{P}=\mu^\top$) — hence convergence (S&B §9.4). The quality bound $\overline{\text{VE}}(\mathbf{w}_{TD})\le\frac{1}{1-\gamma}\min_{\mathbf{w}}\overline{\text{VE}}(\mathbf{w})$. Geometry: $\|v\|_\mu^2=\sum_s\mu(s)v(s)^2$, projection $\Pi=\mathbf{X}(\mathbf{X}^\top\mathbf{D}\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{D}$; MC converges to $\Pi v_\pi$; TD converges to the fixed point of $\Pi B_\pi$, which is *not* $\Pi v_\pi$ in general; definitions of $\overline{\text{BE}}$ and $\overline{\text{PBE}}$ (S&B §11.4).
- **C:** `ch08-projection-geometry`: the value-function space drawn as a plane (3-state MDP → 3-D space, feature subspace a 2-D sheet). Animated arrows show a Bellman backup leaving the sheet and $\Pi$ pulling it back; the reader steps the composite map and watches it spiral into $\Pi B_\pi$'s fixed point, then drags $\gamma\to 1$ and watches the $\frac{1}{1-\gamma}$ bound balloon.
- **P:** Closed-form $\mathbf{w}_{TD}$ solver: estimate $\mathbf{A},\mathbf{b}$ from batch data with `ndarray`, solve via `ndarray-linalg`/`nalgebra`; compare against the incremental learner's trajectory; `proptest` law: on-policy $\hat{\mathbf{A}}$ estimates are positive definite (all eigenvalues' real parts > 0) across random 5-state MDPs.

### 8.4 Feature Construction: Polynomials, Fourier, RBFs, Tile Coding
- **F:** Polynomial features and their coupling explosion; Fourier basis $x_i(s)=\cos(\pi\,\mathbf{c}^i\!\cdot\mathbf{s})$ with per-feature step-size scaling $\alpha_i=\alpha/\|\mathbf{c}^i\|$ (Konidaris et al. 2011; S&B §9.5.2); RBFs $x_i(s)=\exp(-\|s-c_i\|^2/2\sigma_i^2)$ — Kober §2.4's workhorse for robot value functions; tile coding (S&B §9.5.4): $m$ tilings, asymmetric offsets by odd multiples $(1,3,5,\dots)$ of $w/m$ per dimension, exactly $m$ active binary features, one-trial learning with $\alpha=1/m$, hashing for memory; Kober §4.2's physics-inspired features — Schaal's 1996 result that a quadratic Taylor expansion suffices to balance a pole within ±15–30° of upright, and that dropping the cross-terms makes the task unlearnable (representation is destiny).
- **C:** `ch08-tile-visualizer` (signature widget): Pendle's $(\theta,\dot\theta)$ plane covered by draggable tilings — sliders for tile width, number of tilings, offset scheme (uniform vs. asymmetric); hovering a state lights its $m$ active tiles and shows the learned $\hat v$ surface assembling from staircase contributions. `ch08-basis-gallery`: fit $v_\pi$ of the balance policy with polynomial/Fourier/RBF/quadratic-Taylor bases; toggle Schaal's cross-terms off and watch the fit fail.
- **P:** `rl-deep::features`: `Poly`, `Fourier`, `Rbf`, `TileCoding` (pure `ndarray`, hashing via `FxHash`), unit-tested for the exactly-$m$-active invariant; `criterion` bench: feature evaluation ≥ 10⁶ states/s for the 4-D cart-pole coding (it will be the inner loop of §8.5).

### 8.5 Control with Approximation: Semi-Gradient Sarsa Balances Pendle
- **F:** $\hat q(s,a,\mathbf{w})$ with per-action weight slices; episodic semi-gradient Sarsa update $\mathbf{w}\leftarrow\mathbf{w}+\alpha[R_{t+1}+\gamma\hat q(S_{t+1},A_{t+1},\mathbf{w})-\hat q(S_t,A_t,\mathbf{w})]\nabla\hat q(S_t,A_t,\mathbf{w})$ (S&B §10.1); chattering vs. convergence for control (no fixed-point theorem survives the policy-improvement loop — stated precisely, counter-intuition flagged); why on-policy control with linear FA is nonetheless the empirically stable corner of the design space (Kober §4.2: linear approximators stable for on-policy evaluation, trouble starts off-policy).
- **C:** Live training dashboard: episode return, $\epsilon$ schedule, and the $\hat q$ surface over $(\theta,\dot\theta)$ sculpting itself in real time; a "one-trial learning" moment — reader pauses after the first balancing success and sees exactly $m$ tiles per visited state changed.
- **P:** `rl-deep::sarsa_linear`: semi-gradient Sarsa over `TileCoding` on `rl-envs::CartPole` (Pendle balance, from Ch 2's ODE via RK4); seeded runs solve balance (return ≥ 475/500) in < 200 episodes; WASM build of the trainer for the in-page dashboard.

### 8.6 Neural Networks: Features That Learn Themselves
- **F:** MLP $\hat v(s,\mathbf{w})$ as composed nonlinear features; backprop as the chain rule the reader already owns from Ch 2; what is lost: convexity of the linear case, the fixed-point theorem, the $\frac{1}{1-\gamma}$ bound — semi-gradient TD with nonlinear $\hat v$ has no general convergence guarantee, and Kober §4.2 documents the historical price (neural value functions diverging even on-policy; global generalization propagating one state's overestimate everywhere — Boyan & Moore 1995) alongside the historical prize (Riedmiller's Brainstormers winning RoboCup with MLP value functions).
- **C:** "Global vs. local generalization" widget: perturb one state's value target and watch the ripple — tile coding changes a neighborhood, the MLP changes the whole surface; a smoothness/interference dial makes the trade-off tactile.
- **P:** First `burn` usage of the book: `ValueMlp` (2×64 ReLU) on the WGPU backend; semi-gradient TD(0) implemented with `.detach()` on the bootstrap target — the semi-gradient made syntactically explicit; evaluation-only comparison vs. tile coding on fixed-policy Pendle data.

### 8.7 The Deadly Triad, Baird's Counterexample & Gradient-TD
- **F:** Baird's counterexample in full (S&B §11.2): 7 states, 8 weights, $\hat v(s_i)=2w_i+w_8$ for the six upper states, $\hat v(s_7)=w_7+2w_8$; behavior policy takes the dashed action w.p. 6/7, target policy always solid; all rewards 0 so $v_\pi\equiv 0$ is exactly representable at $\mathbf{w}=\mathbf{0}$ — yet semi-gradient off-policy TD diverges for $\gamma=0.99$, even as an *expected* (DP-style) update, so sampling noise is not the culprit. Analysis: off-policy $\mathbf{A}$ loses positive definiteness; an eigenvalue of $-\mathbf{A}$ has positive real part, so $\mathbf{I}-\alpha\mathbf{A}$ has spectral radius > 1 for every $\alpha>0$. The **deadly triad** (S&B §11.3): function approximation + bootstrapping + off-policy training; remove any one leg and stability returns. Gradient-TD in brief: the $\overline{\text{PBE}}=\|\Pi\bar\delta_{\mathbf{w}}\|_\mu^2$ objective and the TDC update with secondary weights, $\mathbf{w}_{t+1}=\mathbf{w}_t+\alpha\delta_t\mathbf{x}_t-\alpha\gamma\mathbf{x}_{t+1}(\mathbf{x}_t^\top\mathbf{v}_t)$, $\mathbf{v}_{t+1}=\mathbf{v}_t+\beta(\delta_t-\mathbf{x}_t^\top\mathbf{v}_t)\mathbf{x}_t$, two-timescale $\beta\gg\alpha$, $O(d)$, provably convergent off-policy (Sutton et al. 2009; S&B §11.7).
- **C:** `ch08-baird-divergence` (signature widget): the 7-state star MDP drawn live with per-weight traces climbing exponentially; reader sliders: $\alpha$, $\gamma$, behavior-policy mix (drag toward on-policy and watch divergence heal), algorithm switch TD ↔ TDC (TDC's traces bend back to zero); a spectral-radius readout ties the picture to the analysis.
- **P:** `rl-deep::baird`: exact and sampled Baird runs; eigenvalue computation of $\mathbf{A}$ via `nalgebra` printed alongside; TDC implementation; side-by-side stability experiment matrix {linear tiles, MLP} × {on-policy, off-policy} × {TD, TDC} on Pendle data — the empirical face of the triad, exported as the chapter's summary figure.

### 8.8 Chapter Bridge
Recap: prediction became projection under $\mu$; semi-gradient TD works because $\mathbf{A}$ is positive definite on-policy, and Baird showed exactly how off-policy breaks it; features are a robotics craft (tile coding shipped, Schaal's warning absorbed); the first `burn` network ran. What's next: Ch 9 lives deliberately inside the deadly triad — deep, bootstrapped, off-policy — and engineers stability back in with replay and target networks; Ch 10 escapes the triad entirely by never bootstrapping a value estimate into its objective. Pendle keeps its balance; Rusty is about to get eyes.

## 4. Interactive Widgets & Dashboards

| ID | Type | What it shows | Reader interaction | Tech |
|---|---|---|---|---|
| `ch08-state-explosion` | widget | Table size vs. bin resolution for Pendle's 4-D state; visit-coverage collapse | Resolution slider, episode-budget slider | egui + egui_plot |
| `ch08-gradient-fields` | widget | Semi-gradient vs. residual-gradient vs. true-VE vector fields on a 2-weight chain | Toggle fields, drop a particle, trace its path | egui_plot |
| `ch08-projection-geometry` | animation | Value space, feature subspace, Bellman backup vs. projection Π, spiral to the ΠB_π fixed point | Step/play the composite map, drag γ, switch MC↔TD targets | egui custom canvas |
| `ch08-tile-visualizer` | widget | Tilings over Pendle's (θ, θ̇) plane; active tiles; assembled v̂ surface | Drag tile width, #tilings, offsets; hover states | egui + WASM |
| `ch08-basis-gallery` | gallery | Polynomial/Fourier/RBF/quadratic-Taylor fits to v_π on the balance task | Select basis & order; toggle Schaal cross-terms | egui_plot |
| `ch08-sarsa-dashboard` | dashboard | Live semi-gradient Sarsa training: returns, ε, q̂ surface sculpting | Pause, α/ε sliders, seed reset | egui + WASM trainer |
| `ch08-generalization` | widget | Local (tiles) vs. global (MLP) response to a single target perturbation | Click a state to perturb; interference dial | egui + burn WASM |
| `ch08-baird-divergence` | dashboard | Baird's star MDP with live weight traces and spectral-radius readout | α, γ, behavior-mix sliders; TD↔TDC switch | egui_plot |

## 5. Rust Implementation Plan

Crates: **`rl-deep` created** (modules `features`, `linear`, `sarsa_linear`, `value_mlp`, `baird`); `rl-envs` gains `CartPole` (RK4 integration of Pendle's ODE from Ch 2) and `RandomWalk1000`; `rl-viz` gains the surface-plot component reused by Ch 9–12. First `burn` dependency enters the workspace (WGPU backend natively, WebGL for WASM).

Representative sketch — tile coding in pure `ndarray` with the sparse semi-gradient Sarsa update:

```rust
/// rl-deep/src/features/tiles.rs — m tilings with asymmetric odd-multiple offsets.
pub struct TileCoding {
    lo: Array1<f64>, inv_width: Array1<f64>,   // per-dim 1/tile_width
    tiles_per_dim: usize, num_tilings: usize,
}

impl TileCoding {
    /// Exactly `num_tilings` active indices for state `s` (S&B §9.5.4).
    pub fn active(&self, s: ArrayView1<f64>) -> SmallVec<[usize; 16]> {
        let per_tiling = self.tiles_per_dim.pow(s.len() as u32);
        (0..self.num_tilings).map(|t| {
            let mut idx = 0usize;
            for (i, &si) in s.iter().enumerate() {
                let offset = (t * (2 * i + 1)) as f64
                    / (self.num_tilings * self.tiles_per_dim) as f64;
                let u = ((si - self.lo[i]) * self.inv_width[i] + offset)
                    .clamp(0.0, (self.tiles_per_dim - 1) as f64) as usize;
                idx = idx * self.tiles_per_dim + u;
            }
            t * per_tiling + idx
        }).collect()
    }
}

/// Sparse semi-gradient Sarsa update: only m weights touched per step.
pub fn sarsa_update(w: &mut Array2<f64>, alpha: f64, delta: f64,
                    a: usize, active: &[usize]) {
    for &i in active { w[[a, i]] += alpha * delta; }  // ∇q̂ is a 0/1 mask
}
```

Experiments/benchmarks: 1000-state random-walk validation of `GradientMc`/`SemiGradientTd0`; closed-form vs. incremental $\mathbf{w}_{TD}$ agreement test; `criterion` bench of `TileCoding::active`; the {representation}×{policy-ness}×{algorithm} stability matrix of §8.7 with pinned seeds; Baird eigenvalue printout. Native artifacts: all trainers + stability matrix runner. In-browser: `ch08-sarsa-dashboard` live trainer, all widgets (the MLP comparison runs `burn` on WebGL).

## 6. Robot Thread

**Pendle** (protagonist): arrives as an ODE from Ch 2 and a discretization casualty from Ch 6; leaves with a tile-coded semi-gradient Sarsa balance controller and a first neural value function — the platform Ch 10 will swing up and Ch 12 will model. **Rusty** (cameo): its tabular warehouse triumphs of Ch 5–7 are the "before" picture; returns in Ch 9 with pixel observations. Reacher and Ferris do not appear.

## 7. Exercises & Explorations

1. **(F)** Show that on-policy semi-gradient TD(0) with state aggregation is exact tabular TD(0) on the aggregated MDP. What does $\mu$-weighting do to states sharing a group?
2. **(F)** Complete the positive-definiteness proof of §8.3 for the *episodic* case, where $\mu$ includes restart mass: show column sums of $\mathbf{D}(\mathbf{I}-\gamma\mathbf{P})$ remain positive when $\gamma<1$, and find the failure mode at $\gamma=1$ with zero restart probability.
3. **(F)** Derive the residual-gradient update from $\tfrac12\mathbb{E}[\delta^2]$ and exhibit a 2-state MDP where its fixed point differs from $\mathbf{w}_{TD}$ — then explain which one a robot with a single trajectory stream can actually estimate.
4. **(C)** In `ch08-tile-visualizer`, set uniform offsets and find the diagonal artifact in $\hat v$; switch to asymmetric odd-multiple offsets and measure (with the widget's RMS readout) how much the artifact shrinks at equal parameter count.
5. **(C)** In `ch08-baird-divergence`, find the *largest* behavior-policy mix (fraction of dashed) at which TD still converges with $\gamma=0.99$. Relate what you found to the spectral-radius readout crossing 1.
6. **(P)** Add a hashed tile coder (`TileCoding::hashed(n_bins)`) and measure on Pendle balance how small the hash table can get before return degrades; plot collisions vs. return.
7. **(P)** Implement Fourier features with the $\alpha_i=\alpha/\|\mathbf{c}^i\|$ step-size scaling, then disable the scaling and reproduce the instability at high order — a Robbins–Monro condition violated per-coordinate.
8. **(P, stretch)** Implement TDC for the linear Pendle setting and add it to the §8.7 stability matrix; verify it converges in the off-policy column where TD diverges, and measure the price in asymptotic $\overline{\text{VE}}$.

## 8. Notation Introduced

| Symbol | Meaning |
|---|---|
| $\mathbf{w}\in\mathbb{R}^d$ | value-function weight vector |
| $\hat v(s,\mathbf{w})$, $\hat q(s,a,\mathbf{w})$ | approximate state / action value |
| $\mathbf{x}(s)$ | feature vector; $\hat v=\mathbf{w}^\top\mathbf{x}(s)$ in linear case |
| $\mu(s)$ | on-policy (stationary/visitation) distribution |
| $\overline{\text{VE}},\overline{\text{BE}},\overline{\text{PBE}}$ | mean-squared value / Bellman / projected Bellman error |
| $\|v\|_\mu^2$ | $\mu$-weighted squared norm |
| $\Pi$ | projection onto the feature subspace under $\|\cdot\|_\mu$ |
| $B_\pi$ | Bellman expectation operator (registered Ch 4; composed here as $\Pi B_\pi$) |
| $\mathbf{A},\mathbf{b},\mathbf{w}_{TD}$ | expected-update matrix/vector, linear TD fixed point $\mathbf{A}^{-1}\mathbf{b}$ |
| $\mathbf{v}_t,\beta$ | gradient-TD secondary weights and their (faster) step size |

## 9. References & Further Reading

- **S&B 2nd ed.**: §9.1–9.4 (VE, SGD/semi-gradient, linear fixed point and bound), §9.5 (feature construction; §9.5.2 Fourier, §9.5.4 tile coding), §9.7 (nonlinear FA/ANNs), §10.1–10.2 (episodic semi-gradient Sarsa), §11.2 (Baird's counterexample), §11.3 (the deadly triad), §11.4 (linear value-function geometry), §11.5 (learnability of the Bellman error), §11.7 (gradient-TD). *Note: the repository PDF (`SuttonBartoIPRLBook2ndEd.pdf`) is an in-progress draft whose ch. 9–11 numbering and coverage differ from the published 2nd edition cited here (its ch. 10 is a stub and it lacks Baird/deadly-triad sections); verify against the published edition.*
- **Kober, Bagnell & Peters (IJRR 2013)**: §2.4 (function approximation; non-i.i.d. data, RBF networks, tile coding as coarse discretization), §4.2 (value-function approximation in robotics: physics-inspired features, Schaal 1996 pole balancing, neural-network divergence history, Brainstormers RoboCup).
- Tsitsiklis & Van Roy (1997) — convergence of linear on-policy TD; the off-policy caveat both S&B and Kober cite.
- Baird (1995) — residual algorithms and the counterexample.
- Sutton, Maei et al. (2009) — GTD/TDC fast gradient-TD family.
- Konidaris, Osentoski & Thomas (2011) — Fourier basis for value-function approximation.
- Boyan & Moore (1995) — early divergence catalogue for nonlinear FA.

# Chapter 5 — Dynamic Programming: Planning with a Known Model

> **Part** I — Foundations of Sequential Decision-Making · **Builds on:** Ch 2 (contraction mappings, Banach fixed-point), Ch 4 (MDP formalism, Bellman equations, `rl-core` traits) · **Feeds:** Ch 6 (GPI without a model), Ch 7 (planning returns as Dyna & MCTS), Ch 8 (why function approximation), Ch 12 (planning with *learned* models), Ch 14 (the four curses)
> **Modernizes:** S&B 2nd ed. ch. 4 in full, retold GPI-first with live visual sweeps; Kober, Bagnell & Peters (2013) §3.1 pulled forward as a quantified curse-of-dimensionality preview

## 1. Purpose & Learning Outcomes

Chapter 4 gave Rusty a formal map of the warehouse — the full MDP tuple $(\mathcal{S}, \mathcal{A}, p, r, \gamma)$. This chapter cashes it in: when the model is *known*, optimal behavior is a computation, not a learning problem. The reader derives and implements the complete DP toolkit, proves why it works, meets **generalized policy iteration (GPI)** — the master pattern every later algorithm instantiates — and then runs head-first into the computational wall that makes pure DP impossible for real robots.

- The reader can derive iterative policy evaluation from the Bellman expectation equation and bound its convergence rate by $\gamma^k$.
- The reader can prove the policy improvement theorem line by line and explain where each inequality comes from.
- The reader can implement policy iteration and value iteration over any `rl-core` `Mdp` and state their convergence guarantees precisely.
- The reader can explain asynchronous DP, when in-place sweeps converge, and why sweep *order* changes wall-clock time but not the fixed point.
- The reader can describe GPI as two interacting processes (evaluation pulls toward $v_\pi$, improvement pulls toward greedy) and recognize it inside every algorithm in Chapters 6–11.
- The reader can quantify the curse of dimensionality — count states, memory, and backups for Rusty, Pendle, Reacher, and Kober's 20-D ball-paddling robot — and say exactly why tabular DP dies at robot scale.

## 2. Storyline

**Act 1 — The blueprint (hook).** Overnight, Rusty receives the warehouse blueprint: the exact Ch 4 MDP — the $12\times 9$ shelving grid, slip probability $p_{\text{slip}}=0.2$, step reward $-1$, delivery $+25$, shelf bump $-10$, $\gamma = 0.95$. No sensors needed, no trial and error: with $p$ and $r$ in hand, "what should Rusty do?" is pure computation. But the naive computation — enumerate all deterministic policies, evaluate each — costs $|\mathcal{A}|^{|\mathcal{S}|} = 4^{80}$ evaluations. Act 1 ends with the question DP answers: how do we exploit Bellman structure to replace enumeration with iteration?

**Act 2 — The engine (development).** Iterative policy evaluation turns the Bellman expectation equation into an update rule; the Ch 2 contraction machinery proves it converges geometrically. The policy improvement theorem — proved in full — shows greedification never hurts. Alternating the two gives policy iteration; collapsing evaluation to a single sweep gives value iteration; freeing the sweep order gives asynchronous DP. All of it happens live on the **GPI dashboard**: the reader watches value ripple outward from the delivery bay and arrows snap toward it, sweep by sweep.

**Act 3 — The wall (payoff with a twist).** Rusty is optimal — provably. Then we ask the robotics question: what if Rusty's state were not "which cell" but pose, velocity, and joint state, like a real robot? The dimensionality-wall widget multiplies it out: Kober's ball-paddling robot has a 20-D state; at 10 levels per dimension that is $10^{20}$ states — more backups per sweep than seconds in the universe's history. Tabular DP is the right *theory* and an impossible *practice*. The chapter closes by naming the two escapes the book will take: learn from samples instead of sweeping everything (Ch 6–7), and approximate instead of tabulate (Ch 8+).

Running robot: **Rusty** (sole star). Pendle and Reacher appear only as data points on the dimensionality wall.

## 3. Section-by-Section Design

### 5.1 Planning When the Map Is Known
- **F:** Definition of *planning* vs *learning*: access to $p(s',r\,|\,s,a)$ as a queryable function. Restatement of the Bellman expectation equation $v_\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)\,[r + \gamma v_\pi(s')]$ and optimality equation from Ch 4 as *systems of equations* to be solved; policy-enumeration lower bound $|\mathcal{A}|^{|\mathcal{S}|}$ motivating iteration.
- **C:** Warm-up animation on the Ch 4 MDP-graph editor: the warehouse as a graph, one Bellman backup rendered as the Ch 4 backup diagram overlaid on real cells — expectation over the slip distribution shown as three weighted ghost-Rustys.
- **P:** Recap of the `rl_core::Mdp` trait (Ch 4): `n_states()`, `n_actions()`, `transitions(s, a) -> &[Transition]` with `Transition { prob, next_state, reward }`. New crate `rl-tabular` scaffolded; `dp.rs` module opened.

### 5.2 Iterative Policy Evaluation
- **F:** The Bellman operator $(T^\pi v)(s) = \sum_a \pi(a|s)\sum_{s',r} p(s',r|s,a)[r + \gamma v(s')]$; full proof that $T^\pi$ is a $\gamma$-contraction in $\|\cdot\|_\infty$ (Ch 2 Banach applied, not re-proved); consequence $\|v_k - v_\pi\|_\infty \le \gamma^k \|v_0 - v_\pi\|_\infty$; expected-update pseudocode with stopping threshold $\theta$ on $\Delta = \max_s |v_{k+1}(s) - v_k(s)|$; Jacobi (two-array) vs Gauss–Seidel (in-place) variants.
- **C:** `ch05-eval-ripple` animation: value heatmap starts flat at 0; each sweep, value "ripples" outward from the delivery bay one cell-distance further; a log-scale strip chart of $\Delta$ per sweep draws a straight line of slope $\log_{10}\gamma$ — the contraction made visible. Reader sets $\gamma \in [0.5, 0.99]$ and watches the slope tilt.
- **P:** `rl-tabular::dp::policy_evaluation(mdp, &policy, gamma, theta) -> Array1<f64>`; `proptest` law: output satisfies Bellman expectation residual $< \theta \gamma/(1-\gamma)$ against Ch 4's exact `nalgebra` linear solve.

### 5.3 The Policy Improvement Theorem
- **F:** Full proof. Given $q_\pi(s, \pi'(s)) \ge v_\pi(s)\ \forall s$, show $v_{\pi'}(s) \ge v_\pi(s)$: $v_\pi(s) \le q_\pi(s,\pi'(s)) = \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t{=}s, A_t{=}\pi'(s)] \le \mathbb{E}_{\pi'}[R_{t+1} + \gamma q_\pi(S_{t+1}, \pi'(S_{t+1})) \mid S_t{=}s] \le \cdots \le \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \mid S_t{=}s] = v_{\pi'}(s)$ — every expansion step justified (tower property + the premise applied at $S_{t+1}$). Corollary: strict inequality somewhere unless $\pi$ already satisfies the Bellman optimality equation, i.e. greedification stalls only at optimality.
- **C:** `ch05-improvement-ladder` animation: the telescoping chain rendered as a ladder. Each rung is one inequality; clicking a rung expands it on Rusty's map — "follow $\pi'$ for $k$ steps, then $\pi$ forever" shown as a path whose first $k$ segments are recolored. The scrubber walks $k \to \infty$ and the ladder converges onto $v_{\pi'}$.
- **P:** `dp::greedy_policy(mdp, &v, gamma) -> Vec<usize>` (argmax over one-step lookahead); unit test: greedification w.r.t. an exact $v_\pi$ never lowers any state's exact value (checked with the Ch 4 linear solver).

### 5.4 Policy Iteration
- **F:** The algorithm: evaluate $\pi_k$ to (near-)convergence, greedify, repeat. Finite convergence proof: finitely many deterministic policies ($|\mathcal{A}|^{|\mathcal{S}|}$), each iteration strictly improves some state's value or terminates, values are monotone so no policy repeats — termination at $\pi_*$ in finitely many iterations. Discussion: truncated evaluation still converges (bridge to value iteration and GPI); Howard (1960) attribution.
- **C:** Part of `ch05-gpi-dashboard` (§4): "Policy Iteration mode" runs the two-phase rhythm visibly — heatmap settles (evaluation), then all arrows snap at once (improvement) — with an iteration counter showing the warehouse needs only 4–6 improvement rounds despite $4^{80}$ possible policies.
- **P:** `dp::policy_iteration(mdp, gamma, theta) -> DpResult { v, policy, eval_sweeps, improve_rounds }`; experiment: total sweeps vs evaluation-truncation depth (evaluate-to-$\theta$ vs 5 sweeps vs 1 sweep), plotted with `egui_plot`.

### 5.5 Value Iteration
- **F:** Bellman optimality operator $(T^* v)(s) = \max_a \sum_{s',r} p(s',r|s,a)[r + \gamma v(s')]$; contraction proof including the max-inequality lemma $|\max_a f(a) - \max_a g(a)| \le \max_a |f(a) - g(a)|$; value iteration as policy iteration with one-sweep evaluation; stopping-rule theorem: $\|v_{k+1} - v_k\|_\infty < \theta \Rightarrow \|v_{k+1} - v_*\|_\infty < \theta\gamma/(1-\gamma)$ (derived via triangle inequality on the fixed point); greedy-policy loss bound (Singh & Yee 1994): $\|v - v_*\|_\infty \le \varepsilon \Rightarrow v_{\pi_{\text{greedy}(v)}} \ge v_* - \frac{2\gamma\varepsilon}{1-\gamma}$, so near-correct values give near-optimal *behavior*.
- **C:** `ch05-gpi-dashboard` "Value Iteration mode": each cell simultaneously shows its max-backup; a side gauge tracks the certified suboptimality bound $2\gamma\|v_{k+1}-v_k\|_\infty/(1-\gamma)$ shrinking — the reader watches the *guarantee* tighten, not just the numbers.
- **P:** `dp::value_iteration` (code sketch in §5); `criterion` benchmark `dp_sweeps`: sweeps-to-$\theta$ and ns/backup for policy iteration vs value iteration vs truncated hybrids on warehouses from $5{\times}5$ to $100{\times}100$.

### 5.6 Asynchronous DP & Generalized Policy Iteration
- **F:** Asynchronous DP: update any states in any order; convergence theorem *statement* — in-place value iteration converges to $v_*$ provided every state continues to be updated (all states updated infinitely often in the limit); proof deferred with citation (Bertsekas & Tsitsiklis 1989). GPI formalized as two interacting processes: evaluation drags $V$ toward $v_\pi$, improvement drags $\pi$ toward $\text{greedy}(V)$; both stabilize $\iff$ Bellman optimality holds. Declared the book's **master pattern**: a table mapping GPI's two arrows onto MC control (Ch 6), Sarsa/Q-learning (Ch 6), Dyna (Ch 7), DQN (Ch 9), actor–critic (Ch 10).
- **C:** `ch05-async-sweeper` comparator: two warehouse panels race to $\Delta < \theta$ under selectable sweep orders — row-major, reverse, random state sampling, "goal-outward" (breadth-first from the delivery bay) — with live sweep/backup counters. Goal-outward wins decisively, planting the seed for prioritized sweeping (Ch 7). Plus the GPI two-arrows diagram animated: $V$ and $\pi$ as two dots spiraling into the joint fixed point.
- **P:** `dp::async_value_iteration(mdp, gamma, order: SweepOrder, budget)`, `enum SweepOrder { RowMajor, Reverse, RandomSeeded(u64), GoalOutward }`; `rayon` used to parallelize *independent replicates* of the race (not within-sweep — in-place updates are order-dependent, and saying why is part of the lesson).

### 5.7 The Computational Wall: the Curse of Dimensionality
- **F:** Complexity accounting: one sweep costs $O(|\mathcal{S}|\,|\mathcal{A}|\,b)$ backups ($b$ = branching factor of $p$); DP is polynomial in $|\mathcal{S}|, |\mathcal{A}|$ — exponentially better than policy enumeration — but $|\mathcal{S}|$ itself is exponential in state *dimension*: discretizing $n$ dimensions at $\ell$ levels gives $\ell^n$ states (Bellman 1957). Worked table: Rusty grid ($n{=}2$, 80 states), Pendle ($n{=}4$: $\theta,\dot\theta,x,\dot x$ at 100 levels $\to 10^8$), Reacher ($n{=}4$ joints+velocities $\to 10^8$), Kober's ball-paddling arm ($n{=}20$, 7-D actions $\to 10^{20}$ states; Kober §3.1's own arithmetic reproduced). Memory and time: one `f64` table for $10^{20}$ states = 800 exabytes; at $10^9$ backups/s one sweep of $q$ exceeds $10^{13}$ years. Honest preview of the escapes: sampling (Ch 6–7), function approximation (Ch 8), and Kober's representational tricks (Ch 17).
- **C:** `ch05-dimensionality-wall` widget: sliders for state dimension $n$ (1–20), levels per dimension $\ell$ (2–100), action count; log-scale bar chart of $|\mathcal{S}|$, table memory, and single-sweep wall-clock shooting past annotated reference lines (RAM of a laptop, age of the universe). Markers pin the cast robots and the ball-paddling robot onto the curve.
- **P:** `rl-tabular` experiment `examples/state_explosion.rs`: programmatically generated $n$-D gridworlds; measure actual sweep time with `criterion` up to the machine's memory limit, then extrapolate on the fitted $\ell^n$ line — the plot where measurement ends and extrapolation begins *is* the wall.

### 5.8 Chapter Bridge
- **F:** One-paragraph recap: DP = Bellman equations + contraction, GPI = the pattern; boxed statement of the three assumptions DP made (known $p$, tabular $\mathcal{S}$, sweepable size).
- **C:** Recap card linking each assumption to the chapter that removes it: known model → Ch 6, sweepable size → Ch 7 (sample where it matters) and Ch 8 (approximate).
- **P:** `rl-tabular::dp` API frozen and re-exported; Ch 6 will import `greedy_policy` unchanged — GPI's improvement half survives the loss of the model, and the code makes that literal.

## 4. Interactive Widgets & Dashboards

| ID | Type | What it shows | Reader interaction | Tech |
|---|---|---|---|---|
| `ch05-gpi-dashboard` | dashboard | **Signature widget.** Rusty's warehouse: $V$ heatmap (sequential colormap) + greedy-policy arrows, sweep-by-sweep; log-$\Delta$ convergence strip; suboptimality-bound gauge | Mode switch (policy eval / policy iteration / value iteration); speed slider incl. single-step; edit $\gamma$, $p_{\text{slip}}$; click a cell to perturb $V$, drag arrows to hand-set $\pi$ and watch evaluation respond | egui + egui_plot, WASM |
| `ch05-eval-ripple` | animation | Value rippling outward from the delivery bay under $T^\pi$; straight-line $\log \Delta$ decay with slope $\log\gamma$ | $\gamma$ slider; pause/scrub sweeps | egui, WASM |
| `ch05-improvement-ladder` | animation | Policy improvement theorem as a ladder of inequalities; each rung = "follow $\pi'$ for $k$ steps then $\pi$" path on the map | Scrub $k$; click any rung to expand its justification | egui, WASM |
| `ch05-async-sweeper` | widget | Two panels racing to convergence under different sweep orders; backup counters | Choose orders (row-major / reverse / random / goal-outward); seed field; reset | egui, WASM |
| `ch05-dimensionality-wall` | widget | $\ell^n$ state count, memory, sweep time on log axes vs reference lines; cast robots pinned | Sliders $n$, $\ell$, $|\mathcal{A}|$; hover for exact numbers | egui_plot, WASM |

Static fallbacks: each widget ships a captioned PNG of a representative configuration (per CLAUDE.md §5).

## 5. Rust Implementation Plan

**Crates touched:** `rl-tabular` (new), `rl-core` (read-only dependency), `rl-envs` (warehouse reused from Ch 4), `rl-viz` (heatmap + arrow-field components added: `ValueHeatmap`, `PolicyArrows` — reused by Ch 6/7 dashboards).

**Modules/files:** `rl-tabular/src/lib.rs`, `rl-tabular/src/dp.rs` (`policy_evaluation`, `greedy_policy`, `policy_iteration`, `value_iteration`, `async_value_iteration`, `DpResult`, `SweepOrder`), `rl-tabular/benches/dp_sweeps.rs`, `rl-tabular/examples/state_explosion.rs`, `demos/ch05-gpi-dashboard/` (+ one demo crate per widget above).

Representative code sketch (in-place value iteration over the Ch 4 trait):

```rust
// rl-tabular/src/dp.rs
use ndarray::Array1;
use rl_core::mdp::Mdp;

pub struct DpResult {
    pub v: Array1<f64>,
    pub policy: Vec<usize>,
    pub sweeps: usize,
}

/// In-place (Gauss–Seidel) value iteration to accuracy `theta`.
/// Returns v with ‖v − v*‖∞ ≤ θ·γ/(1 − γ) and the greedy policy.
pub fn value_iteration<M: Mdp>(mdp: &M, gamma: f64, theta: f64) -> DpResult {
    let mut v = Array1::<f64>::zeros(mdp.n_states());
    let mut sweeps = 0;
    loop {
        let mut delta = 0.0_f64;
        for s in 0..mdp.n_states() {
            let best = (0..mdp.n_actions())
                .map(|a| {
                    mdp.transitions(s, a)
                        .iter()
                        .map(|t| t.prob * (t.reward + gamma * v[t.next_state]))
                        .sum::<f64>()
                })
                .fold(f64::NEG_INFINITY, f64::max);
            delta = delta.max((best - v[s]).abs());
            v[s] = best; // in-place: later states in this sweep see the new value
        }
        sweeps += 1;
        if delta < theta {
            break;
        }
    }
    let policy = greedy_policy(mdp, &v, gamma);
    DpResult { v, policy, sweeps }
}
```

**Experiments/benchmarks:** (1) `criterion` sweep-cost scaling over warehouse sizes; (2) evaluation-truncation study (§5.4); (3) sweep-order race statistics over 100 seeds via `rayon`; (4) `state_explosion` measurement + extrapolation. **Native:** all experiments + benches. **In-browser:** all five widgets (WASM); the $100{\times}100$ dashboard runs at interactive rates because a full sweep is ~$4\times10^4$ backups.

## 6. Robot Thread

- **Rusty** (primary): enters with a formal MDP of the warehouse (Ch 4), exits with a *provably optimal* policy and a live dashboard of how it was computed. Rusty's Ch 6 predicament — the blueprint revoked — is staged in the bridge.
- **Pendle, Reacher:** non-speaking cameos as data points on `ch05-dimensionality-wall`, quantifying why their continuous states will need Ch 8. **Ferris** (not yet introduced) is deliberately absent.

## 7. Exercises & Explorations

1. **(F)** Prove the max-inequality lemma $|\max_a f(a) - \max_a g(a)| \le \max_a |f(a)-g(a)|$ and use it to complete the contraction proof for $T^*$ without consulting §5.5.
2. **(F)** Derive the stopping-rule bound $\|v_{k+1}-v_*\|_\infty < \theta\gamma/(1-\gamma)$ from the triangle inequality and contraction; then show by a 2-state counterexample that $\Delta < \theta$ alone does *not* imply $\|v_{k+1}-v_\pi\|_\infty < \theta$.
3. **(F)** Extend the policy improvement theorem to stochastic $\pi'$: show the premise $\sum_a \pi'(a|s) q_\pi(s,a) \ge v_\pi(s)$ suffices. (This exact form is reused for $\varepsilon$-soft policies in Ch 6.)
4. **(C)** On `ch05-gpi-dashboard`, hand-set a deliberately bad policy (all arrows pointing away from the delivery bay), run evaluation only, and explain from the heatmap why $v_\pi$ is still finite. Then flip exactly one arrow and predict — before pressing evaluate — which cells change value.
5. **(C)** Using `ch05-async-sweeper`, find a sweep order that beats goal-outward on a warehouse with *two* delivery bays. Report backup counts over 10 seeds.
6. **(P)** Implement `SweepOrder::Prioritized` that re-sweeps states in descending order of last-seen $|{\Delta v(s)}|$; benchmark against goal-outward with `criterion`. (You have just pre-invented half of prioritized sweeping — Ch 7 names it.)
7. **(P)** Add `dp::q_value_iteration` producing $q_*$ as an `Array2<f64>`, and verify with `proptest` that $\max_a q_*(s,a) = v_*(s)$ within tolerance on random MDPs.

## 8. Notation Introduced

| Symbol | Meaning |
|---|---|
| $T^\pi$, $T^*$ | Bellman expectation / optimality operators (Ch 2 contraction machinery applied) |
| $v_k$ | value estimate after sweep $k$ (subscript = sweep index, not time step) |
| $\Delta$ | max absolute value change in a sweep, $\max_s \lvert v_{k+1}(s)-v_k(s)\rvert$ |
| $\theta$ | stopping threshold on $\Delta$ (S&B's accuracy parameter) |
| $\pi'$ | improved (greedified) policy |
| $b$ | branching factor of the transition kernel |
| $\ell, n$ | discretization levels per dimension, state dimension ($\lvert\mathcal{S}\rvert = \ell^n$) |

Consistent with Appendix C; $v_\pi, q_\pi, G_t, \gamma, p(s',r|s,a)$ carry over from Ch 4 unchanged.

## 9. References & Further Reading

- **S&B 2nd ed. ch. 4**: §4.1 iterative policy evaluation; §4.2 policy improvement (theorem and telescoping proof); §4.3 policy iteration; §4.4 value iteration; §4.5 asynchronous DP; §4.6 GPI; §4.7 efficiency of DP. (The repo's in-progress-draft PDF matches this numbering for ch. 4.)
- **Kober, Bagnell & Peters (2013)**, IJRR, §3.1 "Curse of Dimensionality" — the ball-paddling $2\times(7+3)=20$-D accounting reproduced in §5.7; also §3.1's note on hierarchical decomposition (foreshadows Ch 17).
- Bellman, R. (1957). *Dynamic Programming*. Princeton University Press — origin of the term and the curse.
- Howard, R. (1960). *Dynamic Programming and Markov Processes* — policy iteration.
- Bertsekas, D. & Tsitsiklis, J. (1989). *Parallel and Distributed Computation* — asynchronous DP convergence proof deferred to here.
- Puterman, M. (1994). *Markov Decision Processes* — value-iteration rate results and stopping rules in full generality.
- Singh, S. & Yee, R. (1994). "An upper bound on the loss from approximate optimal-value functions." *Machine Learning* 16 — the greedy-policy loss bound of §5.5.
- Littman, M., Dean, T. & Kaelbling, L. (1995). "On the complexity of solving Markov decision problems." UAI — DP's polynomial complexity in $|\mathcal{S}|,|\mathcal{A}|$.
- Rust, J. (1997). "Using randomization to break the curse of dimensionality." *Econometrica* 65 — the sampling escape route, cited via Kober §2.1; segue to Ch 6.

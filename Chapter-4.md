# Chapter 4 — Markov Decision Processes: The Formalism

> **Part** I — Foundations of Sequential Decision-Making · **Builds on:** Ch 2 (Markov chains, Neumann series, Banach), Ch 3 (contextual bandits at the cliff edge) · **Feeds:** Ch 5 (DP over the `Mdp` trait, contraction convergence), Ch 6–7 (sampling over the `Env` trait), Ch 13 (state estimation, POMDP honesty), Ch 14 (formulation axes as reward/curse analysis), Ch 19 (belief MDPs, recurrent policies)
> **Modernizes:** S&B Ch 3 (in-repo draft §3.1–3.10) with the contraction machinery made explicit (draft defers it), POMDPs treated honestly because robots never see state (Kober §1.3 and §2's belief-state footnote), and Tang §3.2's problem-formulation axes folded in as the *engineering* face of the same formalism.

## 1. Purpose & Learning Outcomes

This is the book's constitutional chapter: the MDP tuple, returns, value functions, and the Bellman equations derived line by line, existence and uniqueness proved by the contraction machinery Ch 2 built. Then two honesty passes that pure-RL texts postpone: partial observability (**Rusty**'s odometry is noisy — belief states, derived), and Tang's formulation axes (action level, observation space, reward density) — because for a robot, *writing down the MDP is a design act*. The chapter ships the book's gym: the `rl-core` `Env`/`Space`/`Mdp` traits every later chapter trains against.

The reader can:
- specify a finite MDP via the four-argument dynamics $p(s', r \mid s, a)$ and derive $p(s' \mid s,a)$, $r(s,a)$, $r(s,a,s')$ from it;
- prove convergence of the discounted return and manipulate the recursion $G_t = R_{t+1} + \gamma G_{t+1}$;
- derive the Bellman expectation and optimality equations from first principles (tower property + Markov property), and read/draw them as backup diagrams;
- solve $v_\pi$ exactly via $(I - \gamma P_\pi)^{-1} r_\pi$ and justify invertibility;
- prove that $T_\pi$ and $T_*$ are $\gamma$-contractions in $\|\cdot\|_\infty$ and conclude existence, uniqueness, and iterative convergence via Banach;
- derive the Bayes belief update and explain in what sense a POMDP is an MDP over beliefs — and what that equivalence costs;
- place any robot task on Tang's formulation axes and predict the engineering consequences of each placement;
- implement an environment against the `rl-core` traits and validate it with property-based conformance tests.

## 2. Storyline

**Act I — the hook.** Rusty's warehouse, drawn as a grid: shelves, a charging dock, slippery patches near the loading bay. The reader paints a route policy by hand in `ch04-mdp-editor` and asks the obvious question — *how good is it?* — and discovers the question is circular: the value of a cell depends on the value of its neighbors. That circularity, embraced instead of avoided, is the Bellman equation; the chapter's arc is making it rigorous, then solvable, then honest.

**Act II — the development.** The formal ladder, each rung on Rusty: the tuple and $p(s', r \mid s, a)$ (wheel slip makes $p$ genuinely stochastic); returns and $\gamma$ (the discount-lens widget makes $1/(1-\gamma)$ tactile); $v_\pi$ and $q_\pi$ with the Bellman expectation equations **derived step by step** — every conditioning move named (tower property from Ch 2.1, Markov property from Ch 2.2); the matrix form solved exactly by Ch 2.3's Neumann series; then optimality — $v_*$, $q_*$, the max enters, linearity dies, and the fixed-point machinery of Ch 2.5 takes over: $T_\pi$ and $T_*$ proved $\gamma$-contractions, Banach delivers existence, uniqueness, and geometric convergence. The Ch 2 "coincidence" — contraction modulus and discount sharing the letter $\gamma$ — is now a theorem.

**Act III — the payoff, then the confession.** Payoff: the `rl-core` gym — `Space`, `Env` (black-box sampling), `Mdp` (white-box probabilities) — with Rusty's warehouse implementing both, exact $v_\pi$ solving in `nalgebra`, and `proptest` conformance suites; Ch 5 will plan against `Mdp`, Ch 6 will learn against `Env`. Confession: Rusty *never knows which cell he is in*. Noisy odometry turns the MDP into a POMDP; the belief update is derived, the belief-cloud widget shows a kidnapped Rusty recovering his identity from door-frame sightings, and Tang's axes close the chapter by reframing everything just built as *choices* an engineer makes — the same warehouse is a different problem at torque level than at grid level.

## 3. Section-by-Section Design

### 4.1 Rusty's Warehouse Becomes an MDP
- **F:** The finite MDP: $(\mathcal{S}, \mathcal{A}, p, \gamma)$ with four-argument dynamics $p(s', r \mid s, a) = \mathbb{P}(S_{t+1} = s', R_{t+1} = r \mid S_t = s, A_t = a)$; derived quantities $p(s' \mid s,a)$, $r(s,a)$, $r(s,a,s')$ (each derivation shown); the Markov property as the defining assumption, quoting Kober §2: "the Markov property recapitulates the idea of state — a sufficient statistic for predicting the future." The agent–environment boundary discussion (S&B draft §3.1): the boundary sits at the limit of arbitrary control, so Rusty's motors are environment. Rusty's warehouse instantiated exactly (constants fixed here, reused verbatim by Ch 5–7): $12\times 9$ grid minus shelf cells (~80 free states), $\mathcal{A} = \{N, E, S, W\}$, slip probability $p_{\text{slip}} = 0.2$ (lateral slip), $R$: $+25$ delivery at the dock, $-1$ per step, $-10$ shelf bump, $\gamma = 0.95$.
- **C:** `ch04-mdp-editor`, first contact: the warehouse as grid + transition graph; hovering $(s,a)$ fans out the $p(\cdot \mid s,a)$ arrows with thickness = probability; clicking an arrow opens the $(s', r)$ table.
- **P:** `rl-envs::gridworld::WarehouseMdp` built from a declarative map string via `serde`; dynamics enumerated as sparse $(s, a, s', r, p)$ triples; unit test: probabilities sum to 1 for every $(s,a)$.

### 4.2 Returns, Episodes, and the Discount
- **F:** $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$; absolute convergence for bounded rewards with the bound $|G_t| \le R_{\max}/(1-\gamma)$ derived (geometric series — Ch 2.3's algebra); the recursion $G_t = R_{t+1} + \gamma G_{t+1}$ derived and flagged as the engine of everything that follows; episodic vs continuing tasks; unified notation via absorbing terminal states (S&B draft §3.4) with the convention proved consistent; effective horizon $1/(1-\gamma)$; $\gamma$ as a modeling knob for robots — battery horizons, myopic vs far-sighted docking — not a nuisance constant.
- **C:** `ch04-discount-lens` — a fixed logged Rusty trajectory; a $\gamma$ slider reweights each future reward bar by $\gamma^k$ in place, the effective-horizon marker $1/(1-\gamma)$ sliding along the trajectory; at $\gamma = 0$ only the next reward survives, near $1$ the dock's $+25$ dominates from anywhere.
- **P:** `rl-core::ret` utilities: discounted return over reward slices (exact and incremental); computed over Ch 1 sandbox logs so the reader's own driving gets a $G_0$.

### 4.3 Policies and Value Functions: the Bellman Expectation Equations
- **F:** Stochastic policy $\pi(a \mid s)$; $v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$, $q_\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$. **Full derivation** of $v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma\, v_\pi(s')\right]$: expand via the recursion, condition on $(A_t, S_{t+1}, R_{t+1})$ using the tower property, discharge history-dependence by the Markov property and time-invariance — every equality justified by name. The interlocking pair $v_\pi(s) = \sum_a \pi(a|s)\, q_\pi(s,a)$ and $q_\pi(s,a) = \sum_{s',r} p(s',r|s,a)[r + \gamma v_\pi(s')]$; backup-diagram semantics defined (open circle = state/expectation over $\pi$, solid dot = action/expectation over $p$). Matrix form $v_\pi = r_\pi + \gamma P_\pi v_\pi$ with $(P_\pi)_{ss'} = \sum_a \pi(a|s)\, p(s'|s,a)$; unique solution $v_\pi = (I - \gamma P_\pi)^{-1} r_\pi$ by Ch 2.3 ($\rho(\gamma P_\pi) \le \gamma < 1$).
- **C:** `ch04-mdp-editor` live: the reader paints $\pi$ as arrows; $v_\pi$ re-solves on every edit and the heatmap *ripples* outward from the edit — value as a global consequence of local choices. `ch04-backup-animator` steps through the equation itself: nodes light up in derivation order, each term traced to its diagram edge.
- **P:** `rl_tabular::exact::solve_v_pi` via `nalgebra` LU; `criterion` bench exact-solve vs naive iteration across grid sizes (foreshadows Ch 5's trade); `proptest`: solved $v_\pi$ satisfies the Bellman equation pointwise to $10^{-10}$ on random MDPs.

### 4.4 Optimal Policies: the Bellman Optimality Equations
- **F:** Partial order $\pi \ge \pi'$ iff $v_\pi(s) \ge v_{\pi'}(s)\ \forall s$; $v_*(s) = \max_\pi v_\pi(s)$, $q_*(s,a)$; **Bellman optimality equations derived**: $v_*(s) = \max_a \sum_{s',r} p(s',r|s,a)[r + \gamma v_*(s')]$ and its $q_*$ twin — the derivation follows S&B draft §3.8's argument ($v_*$ must equal the value of the best action under itself) made rigorous via §4.5's fixed-point theorem, and the chapter is explicit about which step needs it. Greedy-from-$v_*$ is optimal: the one-step-lookahead argument written out. Existence of a deterministic optimal policy for finite MDPs (proved from $q_*$: pick any argmax). Nonlinearity: the max destroys superposition, so no closed-form solve — iteration is not a concession but the method (Ch 5).
- **C:** `ch04-mdp-editor` *optimal lens*: solve $v_*$ (value iteration under the hood) and overlay greedy arrows; the reader edits $R$ (make collisions cheap, steps expensive) and watches $\pi_*$ reroute through shelf gaps — reward design changing optimal behavior, Ch 14's theme in miniature.
- **P:** Value iteration expressed literally as `rl_core::math::fix_point(T_star, v0, tol)` — deliberate reuse of Ch 2.5's iterator; stopping tolerance from the a-posteriori Banach bound with the $\frac{\gamma}{1-\gamma}$ factor in the doc comment.

### 4.5 Existence and Uniqueness: Bellman Operators as Contractions
- **F:** Operators on $(\mathbb{R}^{|\mathcal{S}|}, \|\cdot\|_\infty)$: $(T_\pi v)(s)$ and $(T_* v)(s)$ as the right-hand sides of §4.3/§4.4. **Both proved $\gamma$-contractions in full**: the $T_\pi$ case by direct expansion; the $T_*$ case via the max-difference lemma $|\max_a f(a) - \max_a g(a)| \le \max_a |f(a) - g(a)|$, itself proved (two symmetric inequalities). Banach (Ch 2.5) then yields: $v_\pi$ and $v_*$ exist, are unique, and iteration converges geometrically with the error bounds already in hand. Monotonicity of both operators proved as a lemma (needed by Ch 5's policy-improvement theorem). Honest scope note: this machinery is finite-$\mathcal{S}$; with function approximation the projection composes with $T$ and the contraction can fail — the deadly-triad forecast (Ch 8), stated now so the guarantee's boundary is visible from day one.
- **C:** `ch02-contraction-cobweb` reprised inside this chapter's page with a real 2-state MDP: $T_*$ drawn as a 2-D map, iterates spiraling into $v_*$; the $\gamma$ slider is now simultaneously the discount and the contraction modulus — the widget caption says exactly that.
- **P:** `proptest` laws on random MDPs: (i) $T_\pi$ monotone; (ii) $\|T_\pi u - T_\pi v\|_\infty \le \gamma \|u - v\|_\infty$ on random pairs; (iii) `fix_point(T_pi, …)` agrees with the LU solve. The book's second machine-checked theorem suite.

### 4.6 Robots Never See State: POMDPs and Belief
- **F:** The POMDP: $(\mathcal{S}, \mathcal{A}, p, \mathcal{O}, Z, \gamma)$ with observation set $\mathcal{O}$ and observation model $Z(o \mid s', a)$; why robots live here (Kober §1.3: noise, aliasing, filters; his footnote that the filter's sufficient statistic can *serve as* state is this section's thesis). Belief $b \in \Delta(\mathcal{S})$; **Bayes update derived in full**: $b'(s') = \frac{Z(o \mid s', a) \sum_s p(s' \mid s,a)\, b(s)}{\mathbb{P}(o \mid b, a)}$ with the normalizer expanded. The belief-MDP construction: beliefs form a continuous-state MDP whose optimal value function suffices for optimal behavior — sufficiency argument shown, full treatment cited (Åström 1965; Kaelbling, Littman & Cassandra 1998). The cost stated without flinching: exact POMDP solving is PSPACE-hard (Papadimitriou & Tsitsiklis 1987); the practical outs the book actually uses later — filters (Ch 13's EKF), recurrent policies as learned belief trackers (Ch 19).
- **C:** `ch04-belief-cloud` — Rusty drives his warehouse with noisy odometry; his belief over cells renders as a translucent cloud that diffuses with motion and snaps sharper at door-frame observations; a corridor of identical shelving shows aliasing (multi-modal belief); a *kidnap* button teleports Rusty and the reader watches the cloud go bimodal, then resolve.
- **P:** `rl-envs::gridworld::NoisyOdometryEnv` — a wrapper that hides state and emits observations (`Env` in the §4.8 sense, provably not `Mdp` from the agent's side); discrete Bayes filter over cells in `ndarray`; belief-entropy metric streamed to the widget's side panel.

### 4.7 Formulation Is a Design Decision: Tang's Axes
- **F:** Tang §3.2's axes given formal content: **action-space level** — low (joint/motor commands), mid (task-space commands), high (temporally extended commands/subroutines); **observation space** — estimated low-dimensional state vs high-dimensional sensing; **reward** — sparse vs dense. Each axis is a *different MDP/POMDP for the same robot and task*, with the difficulty moved, not removed: Rusty at grid level (this chapter, high), velocity level (mid — Ch 19's lidar Rusty), torque level (low — Ch 13's territory); dock-only reward (sparse) vs $-1$-per-step shaping (dense, with the shaping-invariance question deferred to Ch 14's potential-based theorem). The semi-formal claim connecting levels: a high-level MDP is the low-level one under a fixed low-level controller — hierarchy previewed (Ch 17/19).
- **C:** `ch04-formulation-axes` — three toggle rows (action level / observation / reward); each combination renders the induced `Env` type signature, an honest difficulty forecast (which Part of the book handles it), and the Tang-survey systems that made that choice.
- **P:** The same warehouse task behind two `Env` implementations — grid actions vs continuous velocity commands (a `rapier2d` teaser build) — sharing one evaluation harness: the abstraction proves itself by making the axis switch a one-line change in the experiment config.

### 4.8 The Book's Gym: `Env`, `Space`, and `Mdp` in `rl-core`
- **F:** The environment contract written as invariants (doc-tested, `proptest`-checked): a reset distribution $\rho_0$; `step` samples $p(s', r \mid s, a)$; declared observation/action `Space`s that every emitted/accepted value inhabits; bounded rewards; seed-determinism (same seed ⇒ same trajectory). The white-box/black-box split as the chapter's formal moral: `Mdp` exposes $p$ (planning, Ch 5), `Env` only samples it (learning, Ch 6+) — the difference between *knowing* the equations and *living* them.
- **C:** A mermaid architecture diagram: `Space` / `Env` / `Mdp` / `Policy` trait relationships, with a per-Part usage map (Ch 5 ↔ `Mdp`, Ch 6–12 ↔ `Env`, Ch 13+ ↔ `Env` over `rapier`); static figure with hover cards in the web build.
- **P:** `rl-core::{space, env, mdp}` (the §5 sketch); `WarehouseMdp` implements **both** traits; conformance suite: per-$(s,a)$ probability simplex checks, space-membership of every step, seeded-determinism replay, and an `Env`-from-`Mdp` adapter (`SampledEnv<M: Mdp>`) proving the traits compose.

### 4.9 Chapter Bridge
- **F:** Recap ledger: what is now proved (Bellman equations, existence/uniqueness, exact solve, belief update) and what is deliberately open — *how to find $\pi_*$ efficiently*. Forecast: Ch 5 plans with the model (`Mdp` + GPI), Ch 6 learns without it (`Env` + samples), and the POMDP thread sleeps until Ch 13/19 while everything in between assumes full observability *on purpose, and now with a clear conscience*.
- **C:** The roadmap diagram gains its "formalism complete" badge; the belief cloud is frozen as the chapter's parting image — the honest gap between the math and the robot.
- **P:** `cargo xtask test-all` now exercises the full conformance suite; the `Env`/`Mdp` traits are frozen as stable API (widget IDs and trait names are book-wide vocabulary from here on).

## 4. Interactive Widgets & Dashboards

| ID | Type | What it shows | Reader interaction | Tech |
|---|---|---|---|---|
| `ch04-mdp-editor` | sandbox | Rusty's warehouse as editable MDP: transition fans, reward table, $v_\pi$ heatmap, greedy overlay | paint $\pi$ arrows; edit $p$/$R$/$\gamma$; watch $v_\pi$ ripple; toggle optimal lens ($v_*$, $\pi_*$) | `egui` + `nalgebra` solve in WASM |
| `ch04-backup-animator` | animation | Backup diagrams for $v_\pi$, $q_\pi$, $v_*$, $q_*$ synchronized with each equation term | step through derivation order; expectation vs max nodes highlighted; follow one $(s',r)$ branch | `egui` |
| `ch04-discount-lens` | widget | $\gamma^k$ reweighting of future rewards along a fixed trajectory; effective horizon $1/(1-\gamma)$ marker | drag $\gamma \in [0, 0.999]$; pin two $\gamma$ values to compare $G_0$ | `egui_plot` |
| `ch04-belief-cloud` | animation | Rusty's belief $b_t$ over warehouse cells under noisy odometry; entropy trace | drive Rusty; toggle observation quality; kidnap button; watch aliasing corridors | `egui` + `ndarray` filter (WASM) |
| `ch04-formulation-axes` | widget | Tang's action/observation/reward axes; induced `Env` signature and difficulty forecast per combination | toggle each axis; see example systems from `systems.json`; jump to handling chapter | `egui` |

## 5. Rust Implementation Plan

Crates touched: `rl-core` (new `space.rs`, `env.rs`, `mdp.rs`, `policy.rs`; `ret.rs` return utilities), `rl-tabular` (new `exact.rs`: LU policy evaluation, fixed-point value iteration), `rl-envs` (`gridworld/` module: `warehouse.rs`, `noisy_odometry.rs`), `demos/ch04-*` (five widget crates). Tests: `proptest` conformance + Bellman-law suites; benches: `criterion` exact-vs-iterative across grid sizes.

```rust
// rl-core/src/env.rs
use rand::rngs::StdRng;

pub trait Space {
    type Elem;
    fn contains(&self, x: &Self::Elem) -> bool;
    fn sample(&self, rng: &mut StdRng) -> Self::Elem;
}

pub struct Step<O> {
    pub obs: O,
    pub reward: f64,
    pub terminated: bool, // absorbing state reached (§4.2 unified notation)
    pub truncated: bool,  // time limit — not part of the MDP, and says so
}

/// Black-box interface: the agent may only sample p(s', r | s, a).
pub trait Env {
    type Obs;
    type Act;
    fn reset(&mut self, seed: u64) -> Self::Obs;          // draws s0 ~ rho_0
    fn step(&mut self, a: &Self::Act) -> Step<Self::Obs>;
}

// rl-core/src/mdp.rs
/// White-box interface: full access to the four-argument dynamics.
pub trait Mdp {
    fn n_states(&self) -> usize;
    fn n_actions(&self) -> usize;
    /// All (s', r, p) triples with p > 0 for the pair (s, a).
    fn transitions(&self, s: usize, a: usize) -> &[(usize, f64, f64)];
    fn gamma(&self) -> f64;
}
```

Experiments: (1) exact $v_\pi$ vs `fix_point` value iteration: wall-clock and iterations-to-tolerance vs grid size (`criterion`); (2) slip-probability sweep showing $v_\pi$ degradation near the loading bay; (3) belief-filter study: localization entropy vs observation-noise level, kidnap-recovery time distribution over 100 seeds. Artifacts: all five widgets in-browser; the conformance suite and benches run natively; the editor's exact solver is the same `rl-tabular` code compiled to WASM.

## 6. Robot Thread

- **Rusty** — the chapter's protagonist. Before: a body with a hand-coded controller and a reward config (Ch 1). After: his world is a fully specified MDP (`WarehouseMdp`), his route quality is a solvable quantity ($v_\pi$, $v_*$), and his sensing honesty is on record (the POMDP variant with a working Bayes filter). He still cannot *learn* — Ch 5 plans for him, Ch 6 finally lets him learn from experience.
- **Pendle** — cameo in §4.7: his Ch 2 discretized dynamics cited as a continuous-state MDP awaiting Ch 8's function approximation. **Reacher** — cameo in §4.7's axis table (grasp primitives = high-level actions; Ch 3's bandit was the one-state MDP, now provable as claimed in Ch 3.1).

## 7. Exercises & Explorations

1. **(F)** From $p(s', r \mid s, a)$, derive $p(s' \mid s, a)$, $r(s,a)$, and $r(s,a,s')$, and compute all three for a slip cell of the warehouse.
2. **(F)** Prove the interlocking identities of §4.3 and combine them into the one-equation form for $q_\pi$ alone: $q_\pi(s,a) = \sum_{s',r} p(s',r|s,a)\big[r + \gamma \sum_{a'} \pi(a'|s')\, q_\pi(s',a')\big]$.
3. **(F)** Prove the max-difference lemma for finite $\mathcal{A}$, then write out the $T_*$ contraction proof with every inequality justified. Where does the argument need bounded rewards?
4. **(F)** Show that adding a constant $c$ to every reward shifts $v_\pi$ by $c/(1-\gamma)$ uniformly in continuing tasks, but can change optimal behavior in episodic tasks — and explain the connection to Ch 14's shaping theorem.
5. **(C)** In `ch04-mdp-editor`, find a single reward edit that reroutes $\pi_*$ through the narrow shelf gap while leaving $v_*$ at the dock unchanged to two decimals; explain via the optimality equation.
6. **(C)** In `ch04-discount-lens` and the editor, find the largest $\gamma$ at which Rusty's $\pi_*$ prefers a nearby $+3$ side-task reward over the far $+25$ dock delivery; verify against the effective-horizon estimate $1/(1-\gamma)$.
7. **(P)** Implement an absorbing-state wrapper converting any episodic `Env` into the continuing convention of §4.2; `proptest` that discounted returns agree on both representations.
8. **(P)** Extend the Bayes filter with a second door-frame sensor on the opposite wall; measure entropy-reduction per observation and kidnap-recovery time vs the single-sensor baseline over 100 seeds.

## 8. Notation Introduced

| Symbol | Meaning | Notes |
|---|---|---|
| $\mathcal{S}, \mathcal{A}, \mathcal{R}$ | state, action, reward sets | finite in this chapter |
| $p(s', r \mid s, a)$ | four-argument dynamics | S&B 2nd-ed convention; derived forms $p(s'|s,a)$, $r(s,a)$ |
| $\gamma$, $G_t$ | discount, return | $\gamma$ now provably both discount and contraction modulus |
| $\pi(a \mid s)$ | stochastic policy | supersedes Ch 1's informal $\pi$ |
| $v_\pi, q_\pi, v_*, q_*$ | state/action value functions, optimal counterparts | S&B notation throughout |
| $P_\pi, r_\pi$ | policy-induced kernel and reward vector | links to Ch 2.2–2.3 objects |
| $T_\pi, T_*$ | Bellman expectation / optimality operators | contraction modulus $\gamma$ |
| $\mathcal{O}$, $Z(o \mid s', a)$ | observation set and model | $\Omega$ stays reserved for Ch 2's sample space |
| $b_t \in \Delta(\mathcal{S})$, $\rho_0$ | belief state, initial-state distribution | $\Delta(\cdot)$ = probability simplex |

## 9. References & Further Reading

- Sutton & Barto, 2nd ed. (published-2018 numbering, per CLAUDE.md): §3.1 (agent–environment interface, incl. the Markov property — the in-repo draft treats it separately as its §3.5), §3.2 (goals and rewards), §3.3 (returns and episodes), §3.4 (unified notation), §3.5 (policies and value functions — the derivations §4.3 completes), §3.6 (optimal policies and optimal value functions), §3.7 (optimality and approximation).
- Kober, Bagnell & Peters, IJRR 2013: §1.3 (partial observability as robotics' default), §2 (MDP formulation for robots; footnote 1 — the belief/information state as a substitute state — is §4.6's thesis), §2.1 (goals and return models).
- Tang et al. 2024: §3.2 (problem-formulation axes: action level, observation space, reward density — §4.7's source), §3.1 (competency context for the axis examples).
- Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic Programming*, 1994 — Ch 5–6 for the linear-algebraic and contraction treatments this chapter follows.
- Bellman, *Dynamic Programming*, 1957 — the optimality principle's origin.
- Åström, "Optimal control of Markov processes with incomplete state information," *J. Math. Anal. Appl.*, 1965 — belief-MDP sufficiency.
- Kaelbling, Littman & Cassandra, "Planning and acting in partially observable stochastic domains," *Artificial Intelligence*, 1998 — the POMDP planning survey behind §4.6.
- Papadimitriou & Tsitsiklis, "The complexity of Markov decision processes," *Mathematics of Operations Research*, 1987 — PSPACE-hardness cited in §4.6.
- Bertsekas & Tsitsiklis, *Neuro-Dynamic Programming*, 1996 — operator-theoretic DP foundations (monotonicity + contraction lemmas).

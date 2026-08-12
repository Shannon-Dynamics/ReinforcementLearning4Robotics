# Chapter 7 — Unifying Learning & Planning: n-step, Traces, Dyna & MCTS

> **Part** I — Foundations of Sequential Decision-Making · **Builds on:** Ch 3 (UCB → UCT), Ch 5 (DP backups, sweep-order insight), Ch 6 (TD error, Sarsa/Q-learning) · **Feeds:** Ch 8 (traces meet function approximation), Ch 9 (n-step returns in Rainbow), Ch 10 (GAE(λ) inherits the λ-return), Ch 12 (Dyna → model-based RL, decision-time planning → MPC)
> **Modernizes:** S&B 2nd ed. ch. 7 (n-step) + ch. 8 (planning, Dyna, MCTS) with the tabular core of ch. 12 (λ-return, traces) pulled forward; + the UCT→AlphaZero line drawn explicitly; model-wrong experiments staged as sim-to-real foreshadowing

## 1. Purpose & Learning Outcomes

Chapter 5 planned with a perfect model; Chapter 6 learned with none. This chapter dissolves the boundary three ways: **in time** (n-step returns and the λ-return interpolate TD(0)↔MC, with eligibility traces as the mechanical realization), **in data** (Dyna learns a model from experience and plans with it between real steps), and **in the moment** (MCTS plans forward from the current state at decision time). Rusty faces the book's first model-wrong experiments: the warehouse layout changes overnight and blind trust in the learned model becomes a measurable failure mode.

- The reader can define $G_{t:t+n}$ and prove the n-step error-reduction property.
- The reader can construct the λ-return, verify its weights sum to one, and recover TD(0) and MC as its endpoints.
- The reader can prove the offline forward↔backward equivalence of TD(λ) and state precisely when online equivalence fails (and how true-online TD(λ) restores it).
- The reader can implement Sarsa(λ), Dyna-Q, Dyna-Q+, and prioritized sweeping over `rl-core` traits, and explain each as a placement of the same Bellman backup.
- The reader can quantify the planning–performance trade: real steps saved per imagined update, and the price paid when the model is stale.
- The reader can implement UCT-based MCTS, connect its selection rule to Ch 3's UCB, and explain the two AlphaZero substitutions (prior policy in selection, learned value instead of rollouts).

## 2. Storyline

**Act 1 — The space between (hook).** Ch 6's bridge left a dial dangling: MC waits for whole returns (high variance, no bootstrap bias), TD(0) trusts one step (low variance, bootstrap bias). Rusty's deliveries make it concrete: a $+25$ reward at the bay should credit the *whole approach corridor*, not just the last cell. n-step returns give the dial discrete stops; the λ-return makes it continuous; eligibility traces make it *cheap* — one update per step touching every recently visited cell. The λ-dial widget is the act's set piece: drag λ from 0 to 1 and watch credit assignment morph between the two algorithms the reader already owns.

**Act 2 — The imagined warehouse (development).** A second dangling thread: Ch 5's planning machinery went unused since the blueprint was revoked. Dyna's move: rebuild the blueprint *from experience* — every real transition is stored, and between real steps Rusty "rehearses" hundreds of remembered transitions. Learning curves collapse from tens of episodes to a handful. Prioritized sweeping imports Ch 5's sweep-order lesson: plan where $|\delta|$ is largest, propagating backward from surprises like a wavefront. Then the trap springs: **overnight, maintenance closes Rusty's shortcut**. The learned model is now wrong, and Dyna-Q keeps rehearsing a corridor that no longer exists. Dyna-Q+'s exploration bonus $\kappa\sqrt{\tau}$ — curiosity about long-untested transitions — buys back adaptivity, and a second experiment (a *new* shortcut opening) shows why Dyna-Q never finds improvements it isn't curious about.

**Act 3 — Planning in the moment (payoff + modern touch).** Background planning improves a global table; decision-time planning asks only "what should I do *now*, from *this* state?" MCTS grows an asymmetric tree from Rusty's current junction, guided by Ch 3's UCB logic applied per node — UCT. The tree-growth widget shows budget buying depth exactly where it matters. Two substitutions — a learned prior in the selection rule, a learned value function instead of rollouts — turn UCT into AlphaZero's search, and the chapter closes Part I's tabular arc: every method in Chapters 5–7 was one Bellman backup, differing only in *where* and *when* it was applied. That table of backups is the launchpad for Part II.

Running robot: **Rusty** (final tabular chapter as protagonist; graduates to lidar navigation in Ch 19). Pendle waits in the wings for Ch 8.

## 3. Section-by-Section Design

### 7.1 The Space Between TD and MC: n-step Returns
- **F:** Definition $G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n V_{t+n-1}(S_{t+n})$; n-step TD update; **error-reduction property proved**: $\max_s \big|\mathbb{E}_\pi[G_{t:t+n} \mid S_t{=}s] - v_\pi(s)\big| \le \gamma^n \max_s |V(s) - v_\pi(s)|$ (expand the expectation, telescope the true Bellman equations, bound the surviving bootstrap term). Bias–variance reading: bias shrinks as $\gamma^n$, variance grows with the $n$ summed rewards; hence an interior optimum.
- **C:** `ch07-nstep-lab` widget: a 19-cell warehouse corridor (S&B's random-walk, aisle edition); RMS error after 10 episodes plotted over $\alpha$ for $n \in \{1,2,4,\dots,512\}$ — the family of U-curves with intermediate $n$ winning. Reader drags $n$ and $\alpha$; a corridor strip shows how far one reward's credit reaches per update.
- **P:** `rl-tabular/src/nstep.rs`: `NStepTd` (circular buffers for the last $n$ transitions), `NStepSarsa`; reproduction experiment for the U-curve figure (50 seeds, `rayon`).

### 7.2 The λ-Return: Forward View
- **F:** $G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_{t:t+n}$, episodic form $G_t^\lambda = (1-\lambda)\sum_{n=1}^{T-t-1}\lambda^{n-1}G_{t:t+n} + \lambda^{T-t-1}G_t$; proof the weights sum to 1 (geometric series); endpoint checks: $\lambda{=}0 \Rightarrow G_{t:t+1}$ (TD(0) target), $\lambda{=}1 \Rightarrow G_t$ (MC target); interpretation as a bias–variance *portfolio* over all n-step returns; why the forward view is unimplementable online (targets need the future).
- **C:** `ch07-lambda-dial` (signature widget, part 1): left panel shows the weight profile $(1-\lambda)\lambda^{n-1}$ as stacked bars over $n$, morphing from a spike at $n{=}1$ to a spike at the episode end as the reader drags the λ-dial. Captioned callback per the style guide: "drag λ to 1 and you have just rediscovered Monte Carlo."
- **P:** `rl-tabular/src/traces.rs` opened with an *offline* λ-return reference implementation `offline_lambda_return_v` (used as ground truth by the equivalence tests of §7.3).

### 7.3 Eligibility Traces: Backward View & the Equivalence Theorem
- **F:** Accumulating trace $z_t(s) = \gamma\lambda z_{t-1}(s) + \mathbb{1}\{S_t{=}s\}$; TD(λ) update: all states, every step, $V(s) \leftarrow V(s) + \alpha\,\delta_t\,z_t(s)$. **Offline equivalence theorem proved in full:** with $V$ held fixed over the episode, $G_t^\lambda - V(S_t) = \sum_{k=t}^{T-1}(\gamma\lambda)^{k-t}\delta_k$ (derived by unrolling $G_t^\lambda = R_{t+1} + \gamma[(1-\lambda)V(S_{t+1}) + \lambda G_{t+1}^\lambda]$, subtracting $V(S_t)$, and recursing — Ch 6's sum-of-TD-errors identity is the $\lambda{=}1$ case); then the double-sum exchange $\sum_t \alpha\big(G_t^\lambda - V(S_t)\big)\mathbb{1}\{S_t{=}s\} = \sum_k \alpha\,\delta_k z_k(s)$ shows total forward and backward updates coincide. Precise failure of *online* equivalence (V changes mid-episode) and its repair: true online TD(λ) with dutch traces (van Seijen & Sutton 2014) — statement + citation, implementation optional.
- **C:** `ch07-lambda-dial` part 2 + `ch07-forward-backward` split-screen: forward view (from a chosen state, fan of future n-step returns with their λ-weights) vs backward view (trace intensities glowing on visited cells, decaying by $\gamma\lambda$ per step; each $\delta_t$ broadcast backward proportionally). An "offline equivalence checker" runs one episode both ways and displays per-state total updates agreeing to machine epsilon; an "online" toggle shows the small mismatch appear, and "true-online" zeroes it again.
- **P:** `traces::TdLambda` with `z: Array1<f64>` and $O(|\text{active}|)$ sparse trace maintenance (`Vec<(usize, f64)>` with a decay-and-prune pass, threshold $10^{-6}$); `proptest` law: offline TD(λ) total update equals `offline_lambda_return_v` on random episodes.

### 7.4 Sarsa(λ): Control with Traces
- **F:** State–action traces $z_t(s,a)$; Sarsa(λ) update $Q(s,a) \leftarrow Q(s,a) + \alpha\,\delta_t\,z_t(s,a)$ with $\delta_t = R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)$; accumulating vs replacing traces defined, with the pathological revisit case showing why replacing traces exist; Watkins's Q(λ): traces cut to zero on non-greedy actions — the off-policy price, and the reason λ and off-policy learning coexist uneasily (full treatment deferred to Ch 8/12 pointers).
- **C:** `ch07-trace-heat` overlay mode on the warehouse: Rusty drives while every visited cell-action glows and fades ($\gamma\lambda$ decay rendered as afterglow); when the delivery reward lands, the whole glowing comet-tail brightens at once — one δ, many updates. Toggle accumulating/replacing shows double-visits saturating vs resetting.
- **P:** `traces::SarsaLambda`, `traces::WatkinsQLambda`; experiment: episodes-to-criterion vs λ ∈ {0, 0.4, 0.8, 0.9, 1.0} on the warehouse — the interior-λ optimum, mirroring §7.1's interior-n.

### 7.5 Dyna: Learning + Planning from an Imagined Model
- **F:** The Dyna architecture formalized: direct RL (Q-learning on real transitions) + model learning (tabular deterministic model $\hat{M}(s,a) = (s', r)$, last-observed) + planning ($n_{\text{plan}}$ Q-learning updates on uniformly sampled remembered pairs). Theorem-level statement: with a correct (deterministic, fully learned) model, planning updates are exact Q-learning updates on a re-weighted distribution, so convergence to $q_*$ is inherited from Watkins (Ch 6) as long as real exploration maintains coverage. Sample-complexity accounting: real steps vs total backups — planning trades computation for rollouts, Kober's mental-rehearsal argument (Ch 12 will replace the table with learned dynamics nets).
- **C:** `ch07-dyna-imagination` widget: warehouse map where real steps draw solid trails and each planning update flickers as a ghost-Rusty dashed hop sampled from the model; planning-steps selector $n_{\text{plan}} \in \{0, 5, 50\}$; learning-curve panel reproduces the S&B Fig-8.2-style separation (episode 2 already near-optimal at $n_{\text{plan}}{=}50$); counters for real steps vs imagined updates.
- **P:** `rl-tabular/src/dyna.rs`: `DynaQ` (code sketch in §5) with `model: HashMap<(usize, usize), (usize, f64)>`; planning-budget study: steps-to-optimal vs $n_{\text{plan}}$, and wall-clock vs $n_{\text{plan}}$ via `criterion` — the reader sees where imagination stops being free.

### 7.6 When the Model Is Wrong: Dyna-Q+ and the Overnight Layout Change
- **F:** Stale-model failure modes taxonomized: (i) world got *worse* where the model is optimistic → real surprises generate corrective $\delta$, recovery is slow but guaranteed under continued exploration; (ii) world got *better* where the model is pessimistic → no gradient of surprise ever arrives; greedy-on-model behavior never re-tests the closed belief. Dyna-Q+ exploration bonus: planning rewards augmented to $r + \kappa\sqrt{\tau(s,a)}$ with $\tau$ = time since $(s,a)$ was last tried for real; interpretation as an optimism-under-uncertainty prior on staleness (Ch 3's optimistic initialization, resurrected); the κ trade-off: adaptivity vs perpetual re-testing (wear on a real robot — Kober §3.2 flagged forward to Ch 14).
- **C:** `ch07-model-wrong-lab` widget: two scenario buttons — **Blocking** (shortcut closes at step 1 000) and **Shortcut** (new opening appears at step 3 000); cumulative-reward timelines for Dyna-Q vs Dyna-Q+ diverging exactly at the change points; map overlay renders the bonus field $\kappa\sqrt{\tau}$ as slowly brightening heat on untested transitions; κ slider from 0 (pure Dyna-Q) to visibly-thrashing.
- **P:** `dyna::DynaQPlus` (adds `last_tried: Array2<u64>`, global step clock); the two scenario maps added to `rl-envs` as `warehouse::blocking()` and `warehouse::shortcut()` (S&B Examples 8.2/8.3 with warehouse art); 100-seed replication with `rayon`, mean ± band plotted per CLAUDE.md's uncertainty conventions.

### 7.7 Prioritized Sweeping: Planning Where It Matters
- **F:** Uniform planning wastes backups on unsurprising pairs. Priority $p = |r + \gamma\max_a Q(s',a) - Q(s,a)|$; algorithm: maintain a max-heap keyed by $p$; pop, update, then push all *predecessors* $(\bar s, \bar a)$ predicted to lead to $s$ whose priorities exceed threshold $\theta_p$ — the backward wavefront formalized. Correctness argument: prioritized Dyna still performs valid Q-backups, so the Ch 6 convergence scaffold applies given coverage; efficiency evidence quoted (Moore & Atkeson 1993: order-of-magnitude fewer backups). Explicit callback to Ch 5's `ch05-async-sweeper` goal-outward result and Ch 5 Exercise 6 — the reader already invented this under a known model.
- **C:** `ch07-psweep-queue` widget: left, the warehouse with updates radiating backward from the delivery bay as a wavefront; right, the live binary heap drawn as sorted priority bars with the popped entry flying onto the map; counter comparing backups-to-solution vs uniform Dyna on the same seed.
- **P:** `rl-tabular/src/psweep.rs`: `PrioritizedSweeping` using `std::collections::BinaryHeap<PriorityEntry>` (with an `ordered-float` wrapper or a hand-rolled `Ord` impl over `f64` bits — noted as a teaching moment about `f64: !Ord`), predecessor index `HashMap<usize, SmallVec<(usize, usize)>>`; benchmark: backups-to-criterion, prioritized vs uniform, warehouse sizes $10^2$–$10^4$ states.

### 7.8 Decision-Time Planning: MCTS and the Road to AlphaZero
- **F:** Background vs decision-time planning distinguished (improve a global table vs plan from the *current* state, discard, repeat). MCTS's four phases formalized: selection by **UCT** $a = \arg\max_a \big[ \bar Q(s,a) + c\sqrt{\ln N(s) / N(s,a)}\big]$ — Ch 3's UCB applied at every node with the derivation of why per-node bandit logic is sound (each node faces a nonstationary bandit as the subtree improves; hence UCT's $\ln$ numerator over the parent count); expansion, simulation (random-rollout return as an unbiased-but-noisy leaf evaluation), backup (incremental means along the path). Consistency statement (Kocsis & Szepesvári 2006): with rewards bounded and the UCT constant sufficiently large, the probability UCT selects a suboptimal root action → 0 as simulations → ∞ (bias bound $O(\ln n / n)$ at the root; proof cited, not reproduced). The two AlphaZero substitutions stated as equations: PUCT selection $\bar Q(s,a) + c\,\pi_\theta(a|s)\frac{\sqrt{N(s)}}{1+N(s,a)}$ and $v_\theta(s_{\text{leaf}})$ replacing rollouts — with the honest robotics caveat: MCTS needs a simulator-quality model and discrete-ish actions, which is exactly Ch 12's problem statement.
- **C:** `ch07-mcts-tree` animation: Rusty paused at a junction in a maze-annex of the warehouse; the tree grows out of the current state — selection path highlighted edge by edge with UCT scores displayed, expansion node pops in, simulation streaks to a terminal as a dotted comet, backup re-colors $\bar Q$ along the path; budget slider 10–10 000 simulations shows the tree deepening asymmetrically toward promising corridors; "AlphaZero mode" toggle swaps rollouts for a value-heat leaf glow and adds prior-widened edges.
- **P:** `rl-tabular/src/mcts.rs`: `MctsNode { n: u32, q_bar: f64, children: Vec<Edge> }` in an arena `Vec<MctsNode>` (index-based tree — idiomatic Rust, no `Rc<RefCell>`), `Mcts::search(&env_model, root_state, budget) -> usize`, `UctConfig { c, rollout_depth, gamma }`; planning-budget-vs-performance study: delivery success rate vs simulations-per-move ∈ {1, 10, 100, 1 000}, against the Dyna-Q table policy as baseline.

### 7.9 Chapter Bridge
- **F:** The Part-I capstone table: {DP sweep, MC, TD(0), n-step, TD(λ), Dyna planning update, prioritized backup, MCTS backup} × {backup width, backup depth, model needed?, when applied} — every method one Bellman backup, placed differently (S&B's unified-view diagram, extended). The closing admission: everything so far indexed $Q$ by an integer. Pendle's $\theta \in \mathbb{R}$ does not fit in row $s$.
- **C:** Interactive version of the capstone table: hover any cell to see that method's backup diagram animate on a shared mini-warehouse.
- **P:** `rl-tabular` complete and frozen (its API is the reference semantics for Ch 8's approximate versions); the planning-budget study's artifacts archived as the baseline Ch 12's learned-model planners must beat.

## 4. Interactive Widgets & Dashboards

| ID | Type | What it shows | Reader interaction | Tech |
|---|---|---|---|---|
| `ch07-lambda-dial` | widget | **Signature widget.** λ-dial with weight profile $(1-\lambda)\lambda^{n-1}$ morphing TD(0)↔MC; trace afterglow on Rusty's trajectory; credit assignment on reward arrival | Drag λ ∈ [0,1]; step/replay episode; toggle weight-profile vs trace view | egui + egui_plot, WASM |
| `ch07-nstep-lab` | widget | U-curves of RMS error vs α per n on the 19-cell corridor | n and α sliders; seed batch re-run; hover exact values | egui_plot, WASM |
| `ch07-forward-backward` | widget | Forward λ-weighted return fan vs backward trace broadcast; per-state total-update equivalence table | Pick a state; offline/online/true-online mode switch; episode scrubber | egui, WASM |
| `ch07-trace-heat` | animation | Cell-action traces glowing and fading with $\gamma\lambda$ decay; comet-tail update on reward | λ dial (linked to `ch07-lambda-dial`); accumulating vs replacing toggle | egui, WASM |
| `ch07-dyna-imagination` | widget | Real (solid) vs imagined (ghost/dashed) transitions; learning-curve separation by $n_{\text{plan}}$ | $n_{\text{plan}}$ ∈ {0, 5, 50}; pause to inspect any imagined update's source memory | egui + egui_plot, WASM |
| `ch07-model-wrong-lab` | widget | Blocking/Shortcut scenarios; Dyna-Q vs Dyna-Q+ cumulative reward; $\kappa\sqrt{\tau}$ bonus heat on the map | Scenario buttons; κ slider; change-point markers; seed field | egui + egui_plot, WASM |
| `ch07-psweep-queue` | widget | Backward update wavefront + live max-heap of priorities; backups-to-solution counter vs uniform | Threshold $\theta_p$ slider; step/auto; heap entry hover shows its $(s,a,p)$ | egui, WASM |
| `ch07-mcts-tree` | animation | Tree growth from Rusty's current junction: selection/expansion/simulation/backup with UCT scores on edges | Budget slider 10–10 000; c slider; AlphaZero-mode toggle (prior + value leaf) | egui, WASM |

Static fallbacks: captioned PNGs; `ch07-lambda-dial` fallback is a three-frame strip (λ = 0, 0.7, 1).

## 5. Rust Implementation Plan

**Crates touched:** `rl-tabular` (modules `nstep.rs`, `traces.rs`, `dyna.rs`, `psweep.rs`, `mcts.rs`), `rl-envs` (`warehouse::blocking()`, `warehouse::shortcut()`, `corridor19()`, `maze_annex()`), `rl-viz` (adds `TraceGlow` overlay and `TreeView` components), `rl-core` (unchanged — everything composes over `Env`/`Mdp` and Ch 6's δ-returning update convention).

**Modules/files:** as listed per section; plus `rl-tabular/benches/planning_budget.rs`, `rl-tabular/examples/model_wrong.rs`, `rl-tabular/examples/lambda_sweep.rs`, `demos/ch07-*/` (eight demo crates).

Representative code sketch (Dyna-Q+ observe-and-plan; direct RL reuses Ch 6's `QLearning::update`):

```rust
// rl-tabular/src/dyna.rs
use rand::seq::IteratorRandom;
use rand::Rng;
use std::collections::HashMap;

pub struct DynaQPlus {
    pub learner: crate::td::QLearning,          // direct-RL core from Ch 6
    model: HashMap<(usize, usize), (usize, f64)>, // (s,a) -> (s', r), last observed
    last_tried: HashMap<(usize, usize), u64>,
    clock: u64,
    pub n_plan: usize,
    pub kappa: f64,
}

impl DynaQPlus {
    /// One real transition, then n_plan imagined updates. Returns real-step δ_t.
    pub fn observe<R: Rng>(
        &mut self, s: usize, a: usize, r: f64, s2: usize, done: bool, rng: &mut R,
    ) -> f64 {
        self.clock += 1;
        let delta = self.learner.update(s, a, r, s2, done); // (b) direct RL
        self.model.insert((s, a), (s2, r));                 // (c) model learning
        self.last_tried.insert((s, a), self.clock);
        for _ in 0..self.n_plan {                           // (d) planning
            let (&(ps, pa), &(ps2, pr)) =
                self.model.iter().choose(rng).expect("model non-empty");
            let tau = (self.clock - self.last_tried[&(ps, pa)]) as f64;
            let bonus = self.kappa * tau.sqrt();            // curiosity about staleness
            self.learner.update(ps, pa, pr + bonus, ps2, false);
        }
        delta
    }
}
```

**Experiments/benchmarks:** (1) n-step U-curve reproduction; (2) λ-sweep episodes-to-criterion; (3) offline forward/backward equivalence `proptest`; (4) Dyna planning-budget study (steps-to-optimal + wall-clock, `criterion`); (5) blocking/shortcut model-wrong replication, 100 seeds; (6) prioritized-vs-uniform backup counts across warehouse sizes; (7) MCTS budget-vs-success-rate curve. **Native:** all experiments. **In-browser:** all eight widgets; MCTS at 10 000 sims/move stays interactive in WASM because rollouts are table lookups on an 80-state maze.

## 6. Robot Thread

- **Rusty** (primary; final tabular bow): enters knowing only direct TD control; exits with trace-based credit assignment, an internal rehearsal model that survives (and detects) overnight layout changes, and a decision-time planner for novel junctions. Rusty's warehouse table is archived as the exact-semantics baseline that Ch 8's approximators and Ch 12's learned-dynamics planners are tested against. Rusty next appears in Ch 9 (visual gridworld DQN) and graduates to continuous lidar navigation in Ch 19.
- **Pendle:** staged in the bridge as the state that breaks the table — first speaking role in Ch 8. **Reacher, Ferris:** absent.

## 7. Exercises & Explorations

1. **(F)** Prove the λ-return weights sum to 1 in the episodic form (geometric series + the residual $\lambda^{T-t-1}$ term), and show $G_t^\lambda$ interpolates monotonically between the TD(0) target and $G_t$ when all $G_{t:t+n}$ are ordered.
2. **(F)** Derive the recursion $G_t^\lambda = R_{t+1} + \gamma\big[(1-\lambda)V(S_{t+1}) + \lambda G_{t+1}^\lambda\big]$ from the definition, then complete the offline-equivalence proof's double-sum exchange for a state visited twice in one episode.
3. **(F)** Show that a one-step Dyna planning update with a correct deterministic model is an ordinary Q-learning update under a specific sampling distribution, and identify exactly which convergence condition from Watkins's theorem (Ch 6) planning alone cannot maintain. (Answer shape: coverage — the model only contains visited pairs.)
4. **(F)** For a two-armed node with bounded rewards, adapt Ch 3's Hoeffding argument to justify UCT's exploration term, and explain why the parent's $\ln N(s)$ (not global $t$) appears.
5. **(C)** In `ch07-model-wrong-lab`, find a κ that recovers from the Blocking change within 500 steps yet loses less than 5% cumulative reward to re-testing in a static warehouse. Report the κ and both curves.
6. **(C)** Using `ch07-forward-backward` in online mode, construct (by driving Rusty) the shortest episode where forward and backward totals visibly disagree, then verify true-online mode zeroes the gap.
7. **(P)** Add replacing traces to `SarsaLambda` and reproduce the accumulating-vs-replacing comparison on a corridor with a forced revisit loop; explain the saturation failure in one paragraph.
8. **(P)** Give `Mcts` a *wrong* model (the pre-Blocking map after the change) and measure success rate vs budget. Show that more search makes a wrong model *more* confidently wrong — the Ch 12 model-bias theme, discovered empirically.

## 8. Notation Introduced

| Symbol | Meaning |
|---|---|
| $G_{t:t+n}$ | n-step return bootstrapped from $V_{t+n-1}(S_{t+n})$ |
| $G_t^\lambda$ | λ-return, $(1-\lambda)\sum_{n\ge1}\lambda^{n-1}G_{t:t+n}$ (+ episodic residual) |
| $z_t(s)$, $z_t(s,a)$ | eligibility traces (accumulating unless stated); decay $\gamma\lambda$ |
| $\hat{M}(s,a)$ | learned tabular model $\to (s', r)$ |
| $n_{\text{plan}}$ | planning updates per real step (Dyna) |
| $\tau(s,a)$, $\kappa$ | steps since $(s,a)$ last tried for real; Dyna-Q+ bonus scale in $\kappa\sqrt{\tau}$ |
| $\theta_p$ | prioritized-sweeping priority threshold |
| $N(s)$, $N(s,a)$, $\bar Q(s,a)$ | MCTS visit counts and mean value (tree statistics, not the global table) |
| $c$ | UCT/PUCT exploration constant |

$\delta_t$, $\alpha$, $\gamma$, $\lambda$ conventions per Appendix C; $\lambda$ is registered here as the trace-decay parameter (S&B ch. 12 usage).

## 9. References & Further Reading

- **S&B 2nd ed. ch. 7**: §7.1 n-step TD and the error-reduction property (Eq. 7.3); §7.2 n-step Sarsa; §7.3 n-step off-policy learning (the ratio problem flagged in Ch 6 Ex. 3); §7.5 tree-backup (mentioned as the IS-free route).
- **S&B 2nd ed. ch. 12**: §12.1 the λ-return; §12.2 TD(λ) and the backward view; §12.5 true online TD(λ); §12.7 Sarsa(λ); §12.10 Watkins's Q(λ). *(Numbering note: in the repo's in-progress-draft PDF this material lives in its ch. 7 "Eligibility Traces" — §7.2 forward view, §7.3 backward view, Sarsa(λ), Watkins's Q(λ) — with the offline forward↔backward equivalence stated there; cite the published ch. 12 numbering in prose.)*
- **S&B 2nd ed. ch. 8**: §8.1 models and planning; §8.2 Dyna; §8.3 when the model is wrong (Blocking/Shortcut examples, Dyna-Q+); §8.4 prioritized sweeping; §8.6 trajectory sampling; §8.10 rollout algorithms; §8.11 Monte Carlo tree search.
- Sutton, R. (1988). "Learning to predict by the methods of temporal differences." *Machine Learning* 3 — TD(λ) and traces.
- Sutton, R. (1990). "Integrated architectures for learning, planning, and reacting based on approximating dynamic programming." ICML — Dyna.
- Moore, A. & Atkeson, C. (1993). "Prioritized sweeping: reinforcement learning with less data and less time." *Machine Learning* 13 — §7.7's efficiency evidence.
- Peng, J. & Williams, R. (1993). "Efficient learning and planning within the Dyna framework." *Adaptive Behavior* 1 — concurrent prioritized-planning line.
- van Seijen, H. & Sutton, R. (2014). "True online TD(λ)." ICML — exact online forward↔backward equivalence via dutch traces.
- Kocsis, L. & Szepesvári, C. (2006). "Bandit based Monte-Carlo planning." ECML — UCT and its consistency result.
- Coulom, R. (2006). "Efficient selectivity and backup operators in Monte-Carlo tree search." *Computers and Games* — MCTS named and shaped.
- Browne, C. et al. (2012). "A survey of Monte Carlo tree search methods." *IEEE TCIAIG* 4 — breadth reference for §7.8.
- Silver, D. et al. (2016). "Mastering the game of Go with deep neural networks and tree search." *Nature* 529; Silver, D. et al. (2018). "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play." *Science* 362 — the PUCT + value-network substitutions of §7.8.
- Kober, J., Bagnell, J. A. & Peters, J. (2013). IJRR survey, §6 (models and mental rehearsal) — the robotics case for Dyna-style rehearsal, taken up in earnest in Ch 12.

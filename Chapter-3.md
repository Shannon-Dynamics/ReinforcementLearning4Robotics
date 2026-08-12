# Chapter 3 — Multi-Armed Bandits: Exploration & Exploitation

> **Part** I — Foundations of Sequential Decision-Making · **Builds on:** Ch 1 (informal anatomy), Ch 2 (Robbins–Monro, concentration tools) · **Feeds:** Ch 4 (contextual → full RL), Ch 6 (ε-greedy inside control loops), Ch 10 (gradient bandits → policy gradient theorem), Ch 15–16 (priors from simulation/demos as Bayesian head starts), Ch 20 (grasp selection)
> **Modernizes:** S&B Ch 2 (in-repo draft §2.1–2.9) with regret theory made explicit (Lai & Robbins 1985; Auer et al. 2002 UCB1 bound derived), Thompson sampling added as the modern Bayesian workhorse, and every algorithm re-grounded on a robot: **Reacher** choosing among grasp primitives, where every pull wears hardware (Kober §3.2).

## 1. Purpose & Learning Outcomes

The n-armed bandit is the atom of RL: one state, evaluative feedback, and the exploration–exploitation dilemma with nothing else in the way. This chapter builds every classic action-value method on that atom, proves the first real performance guarantee of the book (the UCB regret bound via Hoeffding), plants the seed of policy gradients (gradient bandits), and modernizes with Thompson sampling. **Reacher** enters the cast informally: a 2-link arm choosing among scripted grasp primitives, where regret is measured in worn bearings and dropped mugs, not abstract loss.

The reader can:
- formalize a K-armed bandit ($q_*(a)$, regret $L_n$, gaps $\Delta_a$) and explain why it is an MDP with one state — and why that makes it the right first problem;
- derive the incremental update $Q_{n+1} = Q_n + \frac{1}{n}(R_n - Q_n)$ and recognize it as Robbins–Monro (Ch 2.7), including when to abandon $1/n$ for constant $\alpha$;
- prove Hoeffding's inequality from the Chernoff method and turn it into UCB1's confidence radius;
- reproduce the gap-dependent $O(\ln n)$ UCB1 regret bound and contrast it with ε-greedy's linear regret;
- derive the gradient-bandit update as exact stochastic gradient ascent on $\mathbb{E}[R_t]$, baseline included — and say which Ch 10 theorem it previews;
- implement, test, and race all five algorithm families in Rust on the 10-armed testbed and on Reacher's grasp bandit, stationary and drifting.

## 2. Storyline

**Act I — the hook.** Reacher faces a mug and eight scripted grasp primitives (top pinch, side pinch, handle hook, …). Each attempt succeeds or fails stochastically — pose jitter, friction luck. Attempts are *expensive*: Kober §3.2's curse of real-world samples arrives on page one as a bandit with a hardware bill. The reader plays `ch03-grasp-bandit` by hand, feels themselves torn between the arm that worked twice and the arms never tried, and loses to ε-greedy over 100 pulls.

**Act II — the development.** Estimate, then act: sample averages, the incremental rewrite (Robbins–Monro pays its first dividend), nonstationarity (the mug's pose distribution drifts as the fixture loosens — constant $\alpha$ tracks, $1/n$ freezes). Then the exploration strategies parade through the live testbed, each with its mathematical character: ε-greedy (simple, provably linear regret), optimism (transient, fragile), UCB (Hoeffding forged into an algorithm — the book's first regret theorem, derived in full), gradient bandits (preferences and softmax — the first stochastic *policy* in the book), Thompson sampling (posterior sampling; priors as a place where simulation knowledge will later enter, foreshadowing Ch 15–16).
 
**Act III — the payoff.** The parameter-study: all algorithms, all hyperparameters, 2000 seeded runs each, racing on one dashboard — S&B's Figure-2 methodology upgraded with regret curves and `rayon`. Then the bridge: give the bandit *context* (the mug's observed pose) and policies become functions of observation; let actions *change* the context (grasping moves the mug) and you have fallen off the edge of bandit-land — Ch 4 is waiting there with states, returns, and Bellman.

## 3. Section-by-Section Design

### 3.1 One State, Many Arms: Reacher's Grasp Choice
- **F:** The K-armed bandit: arms $a \in \{1,\dots,K\}$ with reward distributions $\nu_a$, action values $q_*(a) = \mathbb{E}[R_t \mid A_t = a]$, optimal value $q_* = \max_a q_*(a)$, gaps $\Delta_a = q_* - q_*(a)$; horizon-$n$ **regret** $L_n = n\,q_* - \mathbb{E}\!\left[\sum_{t=1}^{n} R_t\right] = \sum_a \Delta_a\, \mathbb{E}[N_n(a)]$ (the counting identity derived — regret lives entirely in pulls of suboptimal arms). The one-state-MDP claim stated now, proved trivially in Ch 4 language later. Exploration–exploitation defined via this identity.
- **C:** `ch03-grasp-bandit` — Reacher above a mug; eight primitive buttons; success/failure animation, running success tally, and a hidden-$q_*$ reveal after 100 pulls comparing the reader's regret to ε-greedy's.
- **P:** `rl-core::bandit`: the `Bandit` trait (§5 sketch) plus `BernoulliBandit`, `GaussianBandit`, and `GraspBandit` — the latter backed by precomputed `rapier2d` grasp-outcome tables so the WASM widget stays light while the native version can re-simulate.

### 3.2 Estimating Action Values
- **F:** Sample average $Q_n(a)$; the incremental rewrite **derived in full algebra** (S&B draft §2.3): $Q_{n+1} = Q_n + \frac{1}{n}(R_n - Q_n)$, generalized to the master pattern $\text{New} \leftarrow \text{Old} + \alpha\,(\text{Target} - \text{Old})$; identification with Robbins–Monro (Ch 2.7): $\alpha_n = 1/n$ satisfies $\sum \alpha_n = \infty$, $\sum \alpha_n^2 < \infty$, so $Q_n \to q_*(a)$ a.s. Nonstationarity: constant $\alpha$ yields the exponential recency-weighted average $Q_{n+1} = (1-\alpha)^n Q_1 + \sum_{i=1}^{n} \alpha (1-\alpha)^{n-i} R_i$ (derived; weights shown to sum to 1), trading convergence for tracking — exactly the trade-off Ch 2.7's widget previewed.
- **C:** `ch03-bandit-testbed` in *drift mode*: $q_*(a)$ performs a random walk (the fixture loosens); sample-average vs constant-$\alpha$ estimates race, the frozen $1/n$ learner visibly stranded.
- **P:** `ValueEstimate` struct (count, mean, `StepSchedule` enum reused from Ch 2); `proptest` law: incremental mean equals batch mean to $10^{-12}$ over arbitrary reward sequences.

### 3.3 ε-Greedy and Optimistic Initialization
- **F:** ε-greedy defined; asymptotic probability of pulling the optimal arm $1 - \varepsilon + \varepsilon/K$ derived; **linear regret** for constant ε derived: $\mathbb{E}[L_n] \ge \frac{\varepsilon\, n}{K} \sum_a \Delta_a$ — exploration that never shrinks pays forever. Decaying schedules $\varepsilon_n \to 0$ with $\sum \varepsilon_n = \infty$ (GLIE teaser, harvested in Ch 6). Optimistic initial values (S&B draft §2.5): mechanism (systematic disappointment drives early sweeps), transience, and fragility under nonstationarity and unknown reward scales — stated as an engineering judgment, demonstrated in the testbed.
- **C:** `ch03-bandit-testbed` — the canonical race: % optimal action and cumulative regret per algorithm; a log-log regret view makes linear-vs-logarithmic growth visually undeniable.
- **P:** `EpsilonGreedy`, `OptimisticGreedy` policies over `ValueEstimate`; reproduce S&B's 10-armed testbed protocol (2000 runs × 1000 steps) in seconds via the `rayon` harness.

### 3.4 Concentration and Upper Confidence Bounds
- **F:** The concentration ladder, each rung proved: Markov → Chebyshev → the Chernoff method; **Hoeffding's lemma** stated (proof cited to Hoeffding 1963 / Lattimore & Szepesvári §5) and **Hoeffding's inequality** derived from it: for i.i.d. $X_i \in [0,1]$, $\mathbb{P}(\bar{X}_n - \mathbb{E}\bar{X}_n \ge t) \le e^{-2nt^2}$. Inverted into a confidence radius; UCB1 rule $A_t = \arg\max_a\, Q_t(a) + c\sqrt{\ln t / N_t(a)}$; **the regret theorem derived in full** (Auer et al. 2002): with $c = \sqrt{2}$, $\mathbb{E}[L_n] \le \sum_{a:\Delta_a > 0} \frac{8 \ln n}{\Delta_a} + \left(1 + \frac{\pi^2}{3}\right)\sum_a \Delta_a$, via the standard three-event decomposition (optimal arm underestimated / suboptimal arm overestimated / $N_t(a)$ still small) with union bounds shown line by line. Gap-independent corollary $O(\sqrt{Kn \ln n})$ set as Exercise 3. Lai & Robbins' $\Omega(\ln n)$ lower bound stated for perspective: logarithmic is not just good, it is essentially optimal.
- **C:** `ch03-ucb-intervals` — ten arms as vertical confidence intervals shrinking as pulls accumulate; the selected arm is always the highest *upper* end; "optimism events" (a currently-worse arm chosen for its wide interval) flash and are counted; a $c$ slider shows over/under-exploration.
- **P:** `Ucb1` policy; experiment: empirical regret overlaid on the theorem's bound across gap configurations $\Delta \in \{0.5, 0.1, 0.02\}$ — the reader sees the bound's looseness *and* its correct $\ln n$ shape and $1/\Delta_a$ scaling.

### 3.5 Gradient Bandits: the Seed of Policy Gradients
- **F:** Numerical preferences $H_t(a)$; softmax policy $\pi_t(a) = e^{H_t(a)} / \sum_b e^{H_t(b)}$ — the book's first stochastic policy. Update $H_{t+1}(a) = H_t(a) + \alpha (R_t - \bar{R}_t)\,(\mathbb{1}\{a = A_t\} - \pi_t(a))$. **The full S&B derivation reproduced** (draft §2.7): $\mathbb{E}[\text{update}] = \alpha\, \partial \mathbb{E}[R_t] / \partial H_t(a)$, including the baseline step — $\bar{R}_t$ subtracts out because $\sum_a \pi_t(a)\, \partial H$-terms vanish — so the baseline changes variance, never the expected direction. Explicit forward pointer: this *is* the score-function/REINFORCE trick on a one-state MDP; Ch 10 re-derives it with states and calls it the policy gradient theorem.
- **C:** `ch03-bandit-testbed` *preference mode*: preference bars $H_t(a)$ and the softmax distribution morphing after every pull; a baseline on/off toggle visibly changes update jitter (variance) while trajectories drift to the same place — Ch 10's baseline lesson, five chapters early.
- **P:** `GradientBandit` with optional baseline; ablation harness (baseline × step size), variance of $H$-updates logged and plotted.

### 3.6 Thompson Sampling
- **F:** The Bayesian bandit: prior over $q_*(a)$; Beta–Bernoulli conjugacy **derived** (posterior $\mathrm{Beta}(\alpha_0 + s_a, \beta_0 + f_a)$ after $s_a$ successes, $f_a$ failures); Thompson sampling as probability matching — sample $\tilde{q}_a \sim$ posterior for every arm, pull $\arg\max_a \tilde{q}_a$; why sampling explores exactly as much as the posterior is uncertain. Guarantees stated with citations: asymptotic optimality matching Lai–Robbins (Kaufmann, Korda & Munos 2012), finite-time bounds (Agrawal & Goyal 2012), empirical strength (Chapelle & Li 2011). The robotics reading: priors are the door through which simulation rollouts and demonstrations will later walk (Ch 15–16); a grasp-simulation prior gives Reacher a head start that UCB structurally cannot accept.
- **C:** `ch03-thompson-violins` — eight Beta posteriors drawn as violins that sharpen with every pull; sampled $\tilde{q}_a$ shown as darts each round; prior sliders let the reader inject a wrong prior and watch data overrule it.
- **P:** `ThompsonBernoulli` using `rand_distr::Beta`; prior-misspecification experiment: optimistic-wrong vs pessimistic-wrong vs flat priors, regret at $n \in \{10^2, 10^3, 10^4\}$.

### 3.7 Contextual Bandits: the Bridge to Full RL
- **F:** Associative search (S&B draft §2.8): observe context $x_t$ (the mug's estimated pose), choose $A_t \sim \pi(\cdot \mid x_t)$; regret redefined against the best *policy* in a class, not the best fixed arm. Tabular case: one independent bandit per discretized context, with the sample-fragmentation cost made explicit. LinUCB stated, not derived (Li et al. 2010). The two-sentence cliff the chapter ends on, stated as a formal observation: contextual bandit = the context process ignores your actions; the moment $x_{t+1}$ depends on $A_t$ (grasping moves the mug), values must account for the future — that requirement has a name, and Ch 4 derives its equations.
- **C:** `ch03-bandit-testbed` *context mode*: two mug poses alternate; a context-blind agent's regret grows linearly while the context-aware agent's flattens — the entire section in one picture.
- **P:** `ContextualBandit` trait extension (context type parameter on `select`); tabular per-context `ValueEstimate` bank; `ch03-arm-designer` gains a context switch.

### 3.8 Chapter Bridge
- **F:** Recap table: algorithm → regret behavior → hyperparameter → failure mode → where it reappears (ε-greedy → Ch 6 control; UCB → Ch 7 MCTS's UCT; gradient bandits → Ch 10; Thompson → Ch 12's posterior-over-models). The formal gap named: no state, no future — and Rusty, whose every motion changes what he sees next, is the counterexample the next chapter formalizes.
- **C:** Final testbed screenshot annotated as a "you are here" against the book roadmap; the grasp bandit replayed once more with the reader's improved intuition.
- **P:** The full parameter-study artifact (`bandit-study` binary): TOML-configured, seeded, `rayon`-parallel, emitting the book's summary figure — the harness template every later chapter's study reuses.

## 4. Interactive Widgets & Dashboards

| ID | Type | What it shows | Reader interaction | Tech |
|---|---|---|---|---|
| `ch03-bandit-testbed` | dashboard | 10-armed testbed: reward-distribution violins, % optimal action, cumulative-regret race; modes for drift, preferences, context | choose algorithms, sliders for $\varepsilon$/$c$/$\alpha$/$Q_1$, mode toggles, reseed, log-log regret view | `egui_plot` + `rl-core::bandit` (WASM) |
| `ch03-ucb-intervals` | animation | Per-arm confidence intervals shrinking; argmax-of-upper-bound selection; optimism events flashing | step/play, $c$ slider, event counter, pin an arm to inspect its radius formula | `egui` |
| `ch03-thompson-violins` | animation | Beta posteriors per arm sharpening; sampled $\tilde q_a$ darts; posterior vs true $q_*$ | manual or auto pulls, prior sliders, reveal-truth toggle | `egui_plot` + `rand_distr` |
| `ch03-grasp-bandit` | widget | Reacher attempting one of eight grasp primitives on a mug; success tally; regret vs algorithms after 100 pulls | click primitives, wear-meter showing pull cost, reveal $q_*$, race against ε-greedy/UCB | `egui` + precomputed `rapier2d` tables |
| `ch03-arm-designer` | sandbox | Reader-designed arm distributions (Bernoulli/Gaussian/bimodal), algorithms competing on the custom instance | drag distribution parameters, pick contenders, export scenario seed for sharing | `egui_plot` (WASM) |

## 5. Rust Implementation Plan

Crates touched: `rl-core` (new `bandit.rs`: `Bandit`, `BanditPolicy`, `ValueEstimate`; `metrics.rs` gains regret tracking), `rl-tabular` (new `bandits/` module: `epsilon_greedy.rs`, `optimistic.rs`, `ucb.rs`, `gradient.rs`, `thompson.rs`, `contextual.rs`), `rl-envs` (grasp-outcome tables under `reacher/grasp_bandit.rs`), `demos/ch03-*` (five widget crates), plus the `bandit-study` binary in `rl-tabular/examples/`.

```rust
// rl-core/src/bandit.rs
use rand::rngs::StdRng;

/// A K-armed bandit environment: one state, evaluative feedback only.
pub trait Bandit {
    fn n_arms(&self) -> usize;
    /// Sample a reward for `arm`; may mutate internal state (drift).
    fn pull(&mut self, arm: usize, rng: &mut StdRng) -> f64;
}

/// An action-selection rule with its estimator baked in.
pub trait BanditPolicy {
    fn select(&mut self, t: u64, rng: &mut StdRng) -> usize;
    fn update(&mut self, arm: usize, reward: f64);
}

// rl-tabular/src/bandits/ucb.rs
pub struct Ucb1 { q: Vec<f64>, n: Vec<u64>, c: f64 }

impl BanditPolicy for Ucb1 {
    fn select(&mut self, t: u64, _rng: &mut StdRng) -> usize {
        (0..self.q.len())
            .max_by(|&a, &b| {
                let bound = |i: usize| match self.n[i] {
                    0 => f64::INFINITY, // every arm once, first
                    n => self.q[i] + self.c * ((t as f64).ln() / n as f64).sqrt(),
                };
                bound(a).total_cmp(&bound(b))
            })
            .expect("bandit has at least one arm")
    }
    fn update(&mut self, arm: usize, r: f64) {
        self.n[arm] += 1;
        self.q[arm] += (r - self.q[arm]) / self.n[arm] as f64; // §3.2 pattern
    }
}
```

Experiments: (1) the 2000-run × 1000-step testbed reproduction with parameter sweeps ($\varepsilon$, $c$, $\alpha$, $Q_1$) via `rayon`, `criterion`-timed; (2) empirical-vs-bound UCB study across gaps; (3) Thompson prior-misspecification study; (4) drift study (sliding-window UCB and constant-$\alpha$ vs their stationary selves). Artifacts: every widget runs in-browser; `bandit-study` runs natively and regenerates all of this chapter's static figures from pinned seeds.

## 6. Robot Thread

- **Reacher** — introduced informally: a picture, eight scripted primitives, and a Bernoulli success model. No kinematics, no continuous control — that machinery arrives in Ch 11 (learning to reach) and Ch 13 (FK/IK, dynamics). After this chapter Reacher can *choose* well; he cannot yet *move* well. The grasp thread continues in Ch 16 (demos) and Ch 20 (grasp selection at scale).
- **Rusty** — cameo in prose only (his warehouse waits for Ch 4); **Pendle** — the gain-tuning bandit example reuses his Ch 2 P-controller as one discretized-gain arm set.

## 7. Exercises & Explorations

1. **(F)** Prove the regret counting identity $L_n = \sum_a \Delta_a\, \mathbb{E}[N_n(a)]$ from the definition of $L_n$.
2. **(F)** Show the constant-$\alpha$ recency weights $(1-\alpha)^n$ on $Q_1$ plus $\alpha(1-\alpha)^{n-i}$ on each $R_i$ sum to exactly 1, and compute the effective memory horizon $\sum_i i \cdot w_i$.
3. **(F)** Derive the gap-independent UCB1 corollary: split arms at $\Delta \lessgtr \sqrt{K \ln n / n}$ and conclude $\mathbb{E}[L_n] = O(\sqrt{K n \ln n})$.
4. **(F)** In the gradient-bandit derivation, verify the baseline-independence step explicitly: show $\sum_a \pi_t(a)\left(\mathbb{1}\{a = A_t\} - \pi_t(a)\right)$ has zero mean under $A_t \sim \pi_t$ for any baseline.
5. **(C)** In `ch03-arm-designer`, build a two-arm instance with heavily overlapping Gaussians where optimistic initialization beats UCB1 at $n = 500$; explain via the $1/\Delta_a$ term in the bound.
6. **(C)** In `ch03-ucb-intervals`, run $c = 0.5$ and $c = 4$; relate the optimism-event counts and final regret to the union-bound term the derivation attached to $c$.
7. **(P)** Implement sliding-window UCB and race it against constant-$\alpha$ ε-greedy on the drifting grasp bandit (mug fixture loosens over 5000 pulls); report which drift rates favor which.
8. **(P)** Add a `MedianOfMeans` estimator to `ValueEstimate` and show it rescues UCB on the heavy-tailed arm from `ch02-lln-sampler`'s warning.

## 8. Notation Introduced

| Symbol | Meaning | Notes |
|---|---|---|
| $K$ | number of arms | S&B's draft says "n-armed"; we reserve $n$ for time/pull counts |
| $q_*(a)$, $Q_t(a)$ | true / estimated action value | $q_*$ gains a state argument in Ch 4 |
| $A_t$, $R_t$, $N_t(a)$ | action, reward, pull count at time $t$ | S&B conventions |
| $\Delta_a$, $L_n$ | suboptimality gap, cumulative regret | regret framework beyond S&B; per Lai–Robbins/Auer |
| $\varepsilon$, $Q_1$, $c$ | exploration rate, optimistic initial value, UCB width | |
| $\alpha$ | step size (same object as Ch 2's $\alpha_n$) | |
| $H_t(a)$, $\pi_t(a)$, $\bar{R}_t$ | preference, softmax policy, baseline | $H$ freed from Ch 2 (Hessian is $\nabla^2 f$) |
| $\mathrm{Beta}(\alpha_0, \beta_0)$, $s_a, f_a$ | prior and success/failure counts | subscript-0 avoids clashing with step size $\alpha$ |
| $x_t$ | bandit context | becomes state $s_t$ the moment actions influence it (Ch 4) |

## 9. References & Further Reading

- Sutton & Barto, 2nd ed. (published-2018 numbering, per CLAUDE.md; in-repo draft numbering runs one lower from §2.4 on): §2.1 (k-armed bandit), §2.2 (action-value methods), §2.4 (incremental implementation), §2.5 (tracking nonstationarity), §2.6 (optimistic initial values), §2.7 (UCB), §2.8 (gradient bandits — the derivation §3.5 reproduces), §2.9 (associative search / contextual bandits).
- Kober, Bagnell & Peters, IJRR 2013: §3.2 (curse of real-world samples — the cost model behind Reacher's wear-meter); §2.1 (exploration–exploitation in the robotics setting).
- Tang et al. 2024: §4.3 (grasping as discrete decision-making — the modern payoff of the grasp-bandit framing, harvested in Ch 20).
- Lai & Robbins, "Asymptotically efficient adaptive allocation rules," *Advances in Applied Mathematics*, 1985 — the $\Omega(\ln n)$ lower bound.
- Hoeffding, "Probability inequalities for sums of bounded random variables," *JASA*, 1963.
- Auer, Cesa-Bianchi & Fischer, "Finite-time analysis of the multiarmed bandit problem," *Machine Learning*, 2002 — the UCB1 bound §3.4 derives.
- Thompson, "On the likelihood that one unknown probability exceeds another…," *Biometrika*, 1933; Chapelle & Li, "An empirical evaluation of Thompson sampling," NeurIPS 2011; Agrawal & Goyal, "Analysis of Thompson sampling," COLT 2012; Kaufmann, Korda & Munos, "Thompson sampling: an asymptotically optimal finite-time analysis," ALT 2012.
- Li, Chu, Langford & Schapire, "A contextual-bandit approach to personalized news article recommendation," WWW 2010 — LinUCB.
- Lattimore & Szepesvári, *Bandit Algorithms*, Cambridge, 2020 — proofs beyond this chapter's scope (Hoeffding's lemma, lower bounds).

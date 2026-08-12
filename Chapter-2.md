# Chapter 2 — The Mathematical Toolkit

> **Part** I — Foundations of Sequential Decision-Making · **Builds on:** Ch 1 · **Feeds:** Ch 3 (incremental estimates, concentration), Ch 4 (Markov chains → MDPs, Banach → Bellman), Ch 5 (contraction convergence), Ch 8 & 10 (gradients, stochastic approximation), Ch 13 (ODEs, dynamics), Ch 15 (integrator stability)
> **Modernizes:** the prerequisites S&B assume (probability, expectations) plus tools S&B defer or omit — contraction mappings proved in full, ODE discretization for robots (Kober §1.3/§2 continuous state–action context), and stochastic approximation (Robbins–Monro 1951) promoted from footnote to founding theorem.

## 1. Purpose & Learning Outcomes

Chapter 1 issued IOUs — "the loop becomes a measurable object", "eventually becomes a limit theorem". This chapter pays them. Six tools are built, each proved at full FCP depth and each stamped with *where it fires later*: probability and conditional expectation, Markov chains, the linear algebra of value functions, gradients and descent, the Banach fixed-point theorem, ODE discretization, and Robbins–Monro stochastic approximation. **Pendle** — the pendulum — joins the cast as the robot simple enough to do everything exactly.

The reader can:
- work with random variables, expectation, and conditional expectation (tower property) fluently enough to follow every derivation in Parts I–II;
- compute with Markov chains: Chapman–Kolmogorov, stationary distributions, and mixing, and say why the Markov property is the load-bearing assumption of Ch 4;
- bound $\|(I - \gamma P)^{-1}\|$ arguments via norms, spectral radius, and the Neumann series — the algebra behind exact value-function solving;
- derive the gradient-descent descent lemma and pick step sizes that provably do not diverge;
- prove the Banach fixed-point theorem from scratch and extract its a-priori/a-posteriori error bounds;
- discretize Pendle's ODE with Euler and RK4, quantify the discretization error, and explain how a continuous robot becomes a discrete-time decision process;
- state Robbins–Monro precisely, verify its step-size conditions ($\sum \alpha_n = \infty$, $\sum \alpha_n^2 < \infty$), and recognize the sample-average update as its special case.

## 2. Storyline

**Act I — the hook.** Pendle appears: a motor, a rod, a mass. The reader tries to simulate it naively (big Euler steps) in `ch02-integrator-explorer` and watches the pendulum gain energy from nothing and spin like a propeller. Simulation — the thing all of robot RL rests on — is itself a piece of math with failure conditions. If we cannot even *simulate* honestly, "learning" is hopeless. So we go build the toolkit.

**Act II — the development.** Five workshops in sequence, each opened by a robot question: "What is sensor noise?" (probability, expectation, LLN); "What does Rusty's random wander converge to?" (Markov chains, stationary distributions, mixing); "Can the equation $v = r + \gamma P v$ be solved?" (norms, spectral radius, Neumann series); "How does any learner improve?" (gradients, descent lemma); "Why does *iterating* anything ever settle?" (Banach, proved line by line, with the cobweb widget making the geometric envelope visible). A running gag with teeth: the contraction modulus is named $\gamma$ "by coincidence" — the coincidence becomes a theorem in Ch 4.5.

**Act III — the payoff.** Two syntheses. First, Pendle's ODE is discretized properly (RK4, zero-order hold) and the reader watches the continuous world become the discrete-time stochastic system $s_{t+1} = F(s_t, a_t, w_t)$ — the exact object Ch 4 will christen an MDP. Second, Robbins–Monro: estimating a mean from noisy samples, one sample at a time, with step sizes $1/n$ — stated as a theorem, dissected in the linear case, and revealed as *the* equation: Ch 3's bandit updates, Ch 6's TD, Ch 8's semi-gradients are all instances. The chapter ends with a tool→chapter firing map.

## 3. Section-by-Section Design

### 2.1 Probability for Decision-Makers
- **F:** Probability space $(\Omega, \mathcal{F}, \mathbb{P})$; random variables; expectation and its linearity; variance; independence; conditional expectation $\mathbb{E}[X \mid Y]$ defined for the discrete case with the **tower property** $\mathbb{E}[\mathbb{E}[X \mid Y]] = \mathbb{E}[X]$ proved in full (it powers every Bellman derivation in Ch 4.3). Markov's inequality and Chebyshev's inequality proved (two lines each); weak law of large numbers proved via Chebyshev, with the $O(1/\sqrt{n})$ error scale made explicit — the reason robot experiments need seeds *and* replicates.
- **C:** `ch02-lln-sampler` — pick a sensor-noise distribution (Gaussian, uniform, bimodal, heavy-tailed Cauchy-like), stream samples, watch the running mean converge inside a $1/\sqrt{n}$ band — and watch the heavy-tailed option refuse to, an honest warning about outlier-prone sensors.
- **P:** `rand`/`rand_distr`/`statrs` crash project: seeded `StdRng`, distribution sampling, Monte Carlo estimate of $\mathbb{E}[\cos\theta]$ under lidar noise; a reusable `SeededExperiment` helper in `rl-core` that every later chapter's experiment harness wraps.

### 2.2 Markov Chains: Memoryless Dynamics
- **F:** Markov property $\mathbb{P}(X_{t+1} \mid X_t, \dots, X_0) = \mathbb{P}(X_{t+1} \mid X_t)$; row-stochastic transition matrix $P$; Chapman–Kolmogorov derived, giving $n$-step transitions as $P^n$; stationary distribution $\mu P = \mu$; irreducibility and aperiodicity defined; the finite-chain convergence theorem stated precisely (via Perron–Frobenius; full proof cited to Levin & Peres), with total-variation distance and geometric mixing governed by the second-largest eigenvalue modulus.
- **C:** `ch02-markov-mixing` — a six-room version of Rusty's warehouse as a chain graph; the reader edits edge probabilities and watches the state-distribution bar chart flow to $\mu$; a TV-distance trace on a log axis shows a straight line whose slope the widget annotates with $|\lambda_2|$; a *spectrum view* toggle draws $P$'s eigenvalues in the complex unit disk and shrinks them by a $\gamma$ slider — pre-living §2.3.
- **P:** `ndarray`: $P$ as `Array2<f64>`, distribution evolution by matrix–vector products, power iteration for $\mu$, eigenvalue cross-check via `nalgebra`; property test: rows of a randomly generated `P` stay stochastic under multiplication.

### 2.3 The Linear Algebra of Value Functions
- **F:** Vector norms with emphasis on $\|v\|_\infty$; induced matrix norms, $\|P\|_\infty = 1$ for row-stochastic $P$; spectral radius $\rho(A)$ and $\rho(A) \le \|A\|$; the **Neumann series**: if $\rho(A) < 1$ then $(I - A)^{-1} = \sum_{k=0}^{\infty} A^k$, derived (partial-sum telescoping + convergence). Payoff pre-announced: for any discount $\gamma \in [0,1)$, $\rho(\gamma P) \le \gamma < 1$, so $v = r + \gamma P v$ has the unique solution $v = (I - \gamma P)^{-1} r$ — Ch 4.3 will only need to say "by Ch 2.3".
- **C:** The `ch02-markov-mixing` spectrum view carries this section: dragging $\gamma$ scales the spectrum into the disk, and a side panel accumulates Neumann partial sums $\sum_{k \le K} (\gamma P)^k r$ visibly converging to the exact solve as $K$ grows.
- **P:** `nalgebra` crash project: `DMatrix`, LU decomposition, `solve`; verify Neumann partial sums against `(I - γP).lu().solve(&r)` to $10^{-10}$; a `criterion` micro-benchmark of solve vs series (foreshadowing why Ch 5 iterates instead of inverting at scale).

### 2.4 Gradients, Descent, and Convexity
- **F:** Gradient, Jacobian $\partial f/\partial x$, Hessian; the chain rule in shape-checked matrix form; Taylor's theorem with remainder; convexity (zeroth/first/second-order characterizations, equivalences derived); $L$-smoothness; the **descent lemma** derived in full: $f(x_{k+1}) \le f(x_k) - \alpha\left(1 - \tfrac{L\alpha}{2}\right)\|\nabla f(x_k)\|^2$, hence any $\alpha < 2/L$ decreases $f$ and $\alpha = 1/L$ is the classical choice; the $O(1/k)$ convex convergence rate stated with proof; why robots care: every Part II update is (stochastic) descent, and every divergence in Ch 8–11 is a violated assumption from this section.
- **C:** `ch02-gd-landscape` — three 2-D surfaces (bowl; ill-conditioned ravine with condition-number dial $\kappa$; a nonconvex "pendulum energy" surface with saddle). The reader drags $x_0$, slides $\alpha$, and watches the trajectory: zig-zag at high $\kappa$, a divergence flash the moment $\alpha > 2/L$, saddle stalls on the nonconvex surface.
- **P:** Gradient descent in `ndarray` plus a finite-difference `grad_check` utility in `rl-core::math` — the same checker later validates every hand-derived gradient in Ch 8–11.

### 2.5 Contraction Mappings and the Banach Fixed-Point Theorem
- **F:** Metric spaces, Cauchy sequences, completeness; a map $T$ with modulus $\gamma < 1$: $d(Tx, Ty) \le \gamma\, d(x,y)$. **Banach's theorem proved completely**: iterates are Cauchy via the geometric sum $d(x_{n+m}, x_n) \le \frac{\gamma^n}{1-\gamma} d(x_1, x_0)$; the limit is a fixed point by continuity; uniqueness by contradiction ($d(x^*, y^*) \le \gamma\, d(x^*, y^*)$). Both error bounds extracted: a-priori $d(x_n, x^*) \le \frac{\gamma^n}{1-\gamma} d(x_1, x_0)$ and a-posteriori $d(x_n, x^*) \le \frac{\gamma}{1-\gamma} d(x_n, x_{n-1})$ — the latter becomes Ch 5's stopping rule. Closing line: in Ch 4.5 the Bellman operators are shown to be $\gamma$-contractions in $\|\cdot\|_\infty$; the letter collision is the point.
- **C:** `ch02-contraction-cobweb` — tab 1: 1-D cobweb plot of $x_{n+1} = T(x_n)$ with a slope slider through $\gamma = 1$ (watch convergence die exactly there) and the geometric error envelope overlaid; tab 2: a 2-D affine map $x \mapsto \gamma R x + b$ spiraling into its fixed point, iterate dots fading geometrically.
- **P:** Generic `fix_point(f, x0, tol)` in `rl-core::math` returning iterate history; `proptest` law: for random affine contractions, measured convergence rate matches $\gamma$ within tolerance — the book's first machine-checked theorem.

### 2.6 From Pendle's ODE to a Discrete-Time Decision Process
- **F:** Pendle defined: state $x = (\theta, \dot\theta)$, $\theta$ measured from upright, dynamics $m\ell^2 \ddot\theta = mg\ell \sin\theta - b\dot\theta + \tau$, torque limit $|\tau| \le \tau_{\max}$ chosen so swing-up requires pumping (the Ch 10 task, seeded here). Equilibria and linearization (unstable upright, $A$-matrix eigenvalues). **Explicit Euler**: $x_{k+1} = x_k + h f(x_k, \tau_k)$, local truncation error $O(h^2)$ derived by Taylor expansion, global error $O(h)$; **RK4** stated with its $O(h^5)$ local / $O(h^4)$ global orders (derivation cited); energy drift as the practical diagnostic; semi-implicit Euler mentioned as the physics-engine compromise (bridge to Ch 15). **Zero-order hold**: holding $\tau$ for control period $h$ and adding disturbance $w_k$ turns the ODE into $s_{k+1} = F(s_k, a_k, w_k)$ — a discrete-time stochastic dynamical system, i.e., precisely the transition kernel $p(s' \mid s, a)$ of Ch 4 before it has a name.
- **C:** `ch02-integrator-explorer` — Pendle animated beside its phase portrait; toggles for Euler/semi-implicit/RK4; step-size slider $h$; live energy-drift plot; a challenge banner: "find the largest $h$ where Euler survives 10 s" — the reader personally discovers stiffness.
- **P:** `rl-envs::pendle`: `PendleParams`, `dynamics`, `rk4_step` (the §5 sketch); phase-portrait and energy plots with `egui_plot`, identical code native and WASM.

### 2.7 Stochastic Approximation: Robbins–Monro
- **F:** Problem: find $x^*$ with $h(x^*) = 0$ from noisy evaluations $y_n = h(x_n) + w_n$. Iteration $x_{n+1} = x_n - \alpha_n y_n$. **Robbins–Monro conditions**: $\sum_n \alpha_n = \infty$ (enough total motion to reach anywhere), $\sum_n \alpha_n^2 < \infty$ (noise variance summable), $\mathbb{E}[w_n \mid \mathcal{F}_n] = 0$, bounded variance, monotone $h$. Theorem stated precisely (almost-sure convergence; full proof cited to Robbins & Monro 1951 and Bertsekas & Tsitsiklis 1996, Prop. 4.1). **Full derivation in the linear case** $h(x) = x - \bar{x}$: the mean-square-error recursion $e_{n+1} = (1-\alpha_n)^2 e_n + \alpha_n^2 \sigma^2$ is unrolled and both conditions are shown load-bearing — drop $\sum \alpha_n = \infty$ and bias survives; drop $\sum \alpha_n^2 < \infty$ and variance never dies. Corollary: the sample-average update $Q_{n+1} = Q_n + \frac{1}{n}(R_n - Q_n)$ is Robbins–Monro with $\alpha_n = 1/n$ — *the equation the rest of the book iterates* (Ch 3.2 bandits, Ch 6 TD(0), Ch 8 semi-gradient TD).
- **C:** `ch02-robbins-monro` — a strip chart of iterates hunting a noisy root; schedule picker $\alpha_n \in \{1/n,\ n^{-2/3},\ \text{const } \alpha\}$; noise dial $\sigma$; a *drift* toggle makes $\bar{x}$ wander — constant $\alpha$ tracks it, $1/n$ freezes — the reader pre-lives Ch 3.2's nonstationary bandit before meeting it.
- **P:** `rl-core::math::robbins_monro` with pluggable `StepSchedule` enum; experiment: auto-tune Rusty's straight-line P-gain from noisy overshoot measurements — stochastic approximation turning a real robot knob.

### 2.8 Chapter Bridge
- **F:** The firing map, as a table each row of which is a promise: probability → returns and Bellman derivations (Ch 4); Markov chains → MDP dynamics (Ch 4.1); Neumann series → exact $v_\pi$ (Ch 4.3); descent lemma → all of Part II; Banach → Bellman contraction (Ch 4.5) and value-iteration convergence (Ch 5); discretization → every simulated environment (Ch 15 stress-tests it); Robbins–Monro → every incremental update from the next chapter's first equation onward.
- **C:** The mermaid roadmap from Ch 1.7 revisited with tool badges now attached to stations.
- **P:** All Ch 2 utilities land in `rl-core::math` with doc-tests; `cargo xtask test-all` now runs the book's first property-based theorem checks.

## 4. Interactive Widgets & Dashboards

| ID | Type | What it shows | Reader interaction | Tech |
|---|---|---|---|---|
| `ch02-lln-sampler` | widget | Running mean of streamed sensor-noise samples inside a $1/\sqrt{n}$ band; heavy-tail failure case | pick distribution, stream/pause, reseed | `egui_plot`, `rand_distr` |
| `ch02-markov-mixing` | animation | Distribution over a six-room chain flowing to $\mu$; log-scale TV distance; eigen-spectrum view with $\gamma$ scaling and Neumann partial sums | edit edge probabilities, speed control, toggle spectrum view, drag $\gamma$ | `egui` + `ndarray`/`nalgebra` (WASM) |
| `ch02-gd-landscape` | sandbox | GD trajectories on bowl / ravine / nonconvex surfaces with contours | drag $x_0$, slide $\alpha$ and $\kappa$, watch $\alpha > 2/L$ divergence flash | `egui_plot` |
| `ch02-contraction-cobweb` | widget | Cobweb iteration with geometric error envelope; 2-D affine spiral tab | drag slope through $\gamma = 1$, drag start point, step/play | `egui_plot` |
| `ch02-integrator-explorer` | sandbox | Pendle under Euler / semi-implicit / RK4: animation, phase portrait, energy drift | slide $h$, toggle method, torque nudges, "largest stable $h$" challenge | `egui` + `rl-envs::pendle` (WASM) |
| `ch02-robbins-monro` | widget | Stochastic-approximation iterates under different step schedules, with target drift | schedule picker, noise dial, drift toggle, reseed | `egui_plot` |

## 5. Rust Implementation Plan

Crates touched: `rl-core` (new `math/` module: `fix_point.rs`, `robbins_monro.rs`, `grad_check.rs`, `seeded.rs`), `rl-envs` (new `pendle/` module: `params.rs`, `ode.rs`), `demos/ch02-*` (five widget crates). Property tests with `proptest`; benches with `criterion` (Neumann vs LU).

```rust
// rl-envs/src/pendle/ode.rs
use nalgebra::Vector2;

pub struct PendleParams { pub m: f64, pub l: f64, pub b: f64, pub g: f64 }

/// State x = [theta, theta_dot], theta measured from upright.
/// Torque tau is held constant across the step (zero-order hold).
pub fn dynamics(p: &PendleParams, x: Vector2<f64>, tau: f64) -> Vector2<f64> {
    let (theta, omega) = (x[0], x[1]);
    let inertia = p.m * p.l * p.l;
    let alpha = (p.g / p.l) * theta.sin() - p.b * omega / inertia + tau / inertia;
    Vector2::new(omega, alpha)
}

pub fn euler_step(p: &PendleParams, x: Vector2<f64>, tau: f64, h: f64) -> Vector2<f64> {
    x + h * dynamics(p, x, tau)
}

pub fn rk4_step(p: &PendleParams, x: Vector2<f64>, tau: f64, h: f64) -> Vector2<f64> {
    let k1 = dynamics(p, x, tau);
    let k2 = dynamics(p, x + 0.5 * h * k1, tau);
    let k3 = dynamics(p, x + 0.5 * h * k2, tau);
    let k4 = dynamics(p, x + h * k3, tau);
    x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
}
```

Experiments: (1) global-error study — Euler vs RK4 error at $t = 5$ s against a tiny-step reference, log–log slope confirming orders 1 and 4; (2) energy-drift comparison; (3) Robbins–Monro schedule shoot-out on the P-gain tuning task (20 seeds each). Artifacts: all five widgets in-browser; error-study and benchmark run natively and export the book's static figures.

## 6. Robot Thread

- **Pendle** — introduced. Before: a cast card. After: full parameterized ODE, verified RK4 simulation, phase-portrait visualization, and a documented torque limit that makes swing-up nontrivial — the body Ch 8 approximates over, Ch 10 swings up, and Ch 12–13 model and control.
- **Rusty** — cameo: his six-room layout is §2.2's Markov chain; his P-gain is §2.7's tuning knob. No new capability.

## 7. Exercises & Explorations

1. **(F)** Prove the tower property $\mathbb{E}[\mathbb{E}[X \mid Y]] = \mathbb{E}[X]$ for discrete $X, Y$, then use it to show $\mathbb{E}[X] = \sum_y \mathbb{P}(Y{=}y)\, \mathbb{E}[X \mid Y{=}y]$ on Rusty's noisy-lidar example.
2. **(F)** Show that for row-stochastic $P$ and $\gamma \in [0,1)$, $\|\gamma P\|_\infty = \gamma$, and conclude invertibility of $I - \gamma P$ via the Neumann series. Where exactly does $\gamma = 1$ break the argument?
3. **(F)** Derive the a-posteriori Banach bound $d(x_n, x^*) \le \frac{\gamma}{1-\gamma} d(x_n, x_{n-1})$ from the a-priori bound, and explain why it — not the a-priori bound — is a practical stopping rule.
4. **(F)** Unroll the linear-case Robbins–Monro recursion with constant $\alpha$: show the steady-state mean-square error is $\frac{\alpha \sigma^2}{2 - \alpha}$ and hence does not vanish — the price constant step sizes pay for tracking ability.
5. **(C)** In `ch02-markov-mixing`, construct a two-cluster chain with a single weak bridge edge; relate the observed TV-distance slope to $|\lambda_2|$ shown in the spectrum view.
6. **(C)** In `ch02-integrator-explorer`, find the largest $h$ at which (a) Euler and (b) RK4 keep Pendle's free swing bounded for 10 s; compare with the linearization's eigenvalues from §2.6.
7. **(P)** Implement semi-implicit (symplectic) Euler for Pendle and add it to the energy-drift experiment; explain its qualitatively different drift in one paragraph.
8. **(P)** Extend `robbins_monro` with Polyak–Ruppert iterate averaging and measure its variance reduction on the P-gain task across 20 seeds.

## 8. Notation Introduced

| Symbol | Meaning | Notes |
|---|---|---|
| $(\Omega, \mathcal{F}, \mathbb{P})$ | probability space | $\Omega$ reserved for sample space; POMDP observation set gets $\mathcal{O}$ (Ch 4.6) |
| $\mathbb{E}[X]$, $\mathrm{Var}[X]$, $\mathbb{E}[X \mid Y]$ | expectation, variance, conditional expectation | tower property is Ch 4's workhorse |
| $P$, $\mu$, $\lambda_2$ | transition matrix, stationary distribution, second eigenvalue | $P$ reused as MDP kernel matrix $P_\pi$ in Ch 4 |
| $\|v\|_\infty$, $\rho(A)$ | sup norm, spectral radius | the norm of Bellman analysis |
| $\nabla f$, $\partial f/\partial x$, $L$ | gradient, Jacobian, smoothness constant | Hessian written $\nabla^2 f$ to keep $H_t(a)$ free for Ch 3.5 |
| $\gamma$ | contraction modulus | deliberately shares the discount's letter; identified in Ch 4.5 |
| $x^*$, $\alpha_n$, $\mathcal{F}_n$ | fixed point / root, step size, filtration (informal) | $\alpha$ is also Ch 3's learning rate — same object |
| $\theta, \dot\theta, \tau, m, \ell, b$ | Pendle angle, velocity, torque, mass, length, damping | robotics convention per style guide |
| $h$, $w_k$ | integrator step, disturbance | $h$ freed after this chapter (preferences use $H_t(a)$) |

## 9. References & Further Reading

- Sutton & Barto, 2nd ed. (published-2018 numbering, per CLAUDE.md): §2.4 (incremental implementation — the Robbins–Monro special case; in-repo draft §2.3), §3.1 (the Markov property, folded there in the published edition; the in-repo draft treats it in full as its §3.5).
- Kober, Bagnell & Peters, IJRR 2013: §1.3 (continuous high-dimensional state–action spaces), §2 opening (the formal setting robots are forced into), §3.2 (cost of real-world samples — why sample-by-sample estimation matters).
- Robbins & Monro, "A stochastic approximation method," *Annals of Mathematical Statistics*, 1951.
- Banach, "Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales," *Fundamenta Mathematicae*, 1922.
- Bertsekas & Tsitsiklis, *Neuro-Dynamic Programming*, 1996 — Prop. 4.1 for the general stochastic-approximation proof; Borkar, *Stochastic Approximation: A Dynamical Systems Viewpoint*, 2008 for the ODE method.
- Levin, Peres & Wilmer, *Markov Chains and Mixing Times*, 2nd ed., 2017 — Perron–Frobenius and mixing proofs.
- Boyd & Vandenberghe, *Convex Optimization*, 2004; Nesterov, *Lectures on Convex Optimization*, 2018 — descent-lemma lineage and rates.
- Hairer, Nørsett & Wanner, *Solving Ordinary Differential Equations I*, 2nd ed., 1993 — RK order theory cited for §2.6.
- Puterman, *Markov Decision Processes*, 1994 — Neumann-series treatment of policy evaluation (Ch 4's exact solve).

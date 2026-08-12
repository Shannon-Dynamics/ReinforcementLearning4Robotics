# Chapter 15 — Simulation & the Sim-to-Real Bridge

> **Part** III — The Robotics Side · **Builds on:** Ch 10 (PPO), Ch 11 (SAC), Ch 12 (model bias & mental rehearsal), Ch 13 (`rl-sim`, rapier articulated bodies), Ch 14 (under-modeling curse, evaluation rubric) · **Feeds:** Ch 16 (teacher–student distillation reuses DAgger), Ch 18 (Ferris locomotion on the rollout farm), Ch 19 (randomized navigation scenes), Ch 22 (capstone training pipeline)
> **Modernizes:** Kober §6.1 (mental rehearsal: simulation bias, distributions over models, random-number reuse); Tang §3.3 (simulator-usage axis) and §5 (zero-shot vs few-shot evidence, real-world learning); Akinola simulator slides (mujoco/pybullet-era tooling, recast in Rust)

## 1. Purpose & Learning Outcomes

Ch 14 diagnosed under-modeling; this chapter builds the bridge. We open the physics engine the reader has trusted since Ch 13 — integrators, contact solvers, the parameters that don't exist in the real world — then assemble the modern transfer stack on top of it: domain randomization read as distributional robustness, system identification that fits the simulator to logged reality, teacher–student privileged learning, and massively parallel rollout collection. **Ferris the quadruped enters the book here**, because locomotion is where this recipe earned its reputation (Tang: most L3–L4 systems are zero-shot sim-to-real).

The reader can:
- Derive the stability limit of explicit Euler on $\dot y = \lambda y$ and explain why contact stiffness, not curiosity, sets the simulator timestep.
- Write down the velocity-level contact LCP (complementarity + friction cone) and name three solver-side parameters that shape "physics" yet have no physical counterpart.
- Express plain domain randomization as $\max_\theta \mathbb E_{\xi\sim p}[J(\theta;\xi)]$ and its robust variants (CVaR, KL-ball soft-min) as distributionally-robust objectives, and predict the robustness-vs-optimality trade-off from the width of $p$.
- Fit simulator parameters to logged real trajectories with CMA-ES and diagnose non-identifiability.
- State the asymmetric actor-critic estimator and the teacher–student distillation objective, and explain why privileged information may enter the critic/teacher but not the deployed policy.
- Choose zero-shot vs few-shot vs real-world learning for a given task using Tang's evidence, and defend the choice with the L0–L5 rubric from Ch 14.

## 2. Storyline

**Act I — The lie in the engine.** The hook is Ch 14's perturbation heatmap: the Ch 11 Reacher policy dies under a 5 % mass change. Why would sim and real ever agree? We open rapier: a semi-implicit integrator marching a linear test equation, then a contact solver trading physical truth for numerical survival. The reader cranks $\Delta t$ until Pendle explodes, then meets the parameters that were never physical: solver iterations, Baumgarte bias, friction-cone facets. Kober's verdict lands: his ball-in-a-cup simulator "matched recorded data very well" — and policies still failed to transfer in either direction.

**Act II — Engineering the bridge.** Four countermeasures, each with math and a build: (1) randomization — train on a *distribution* of worlds so reality is one more draw; formalized as DRO, with the slider wall showing robustness bought and optimality paid; (2) system identification — pull $p(\xi)$ toward reality by fitting rapier parameters to logged trajectories with CMA-ES; (3) teacher–student — let training-time policies and critics see privileged state the real robot never will, then distill; (4) scale — a `rayon` rollout farm making randomized experience cheap. Mid-act, **Ferris arrives**: a 12-DoF quadruped URDF that stands under Ch 13 PD control and becomes the farm's stress test — a thousand randomized Ferrises falling in parallel.

**Act III — Crossing, honestly.** The payoff experiment: two Reacher policies, one trained in a single sim, one under randomization + sysid, both evaluated on a held-out "real" sim (parameters shifted outside the training distribution, per Ch 14's rubric). The randomized policy survives; the gap is measured, not asserted. The chapter closes with Tang's decision tree — when zero-shot works (stable, simulable dynamics; a-priori dense rewards), when few-shot or real-world learning is forced (contact-rich manipulation, HRI) — and hands Ferris, the farm, and the randomization hooks to Ch 18.

Robots: **Pendle** (integrator playground), **Reacher** (sysid + transfer experiment), **Ferris** (first appearance: model, PD stand, farm), **Rusty** (cameo: perception-gap sidebar for Ch 19).

## 3. Section-by-Section Design

### 15.1 Inside the Engine I: Integrators & Stability
- **F:** From Ch 2's ODE discretization to engine reality. Test equation $\dot y=\lambda y$, $\mathrm{Re}\,\lambda<0$: explicit Euler $y_{k+1}=(1+h\lambda)y_k$ stable iff $|1+h\lambda|\le 1$, i.e. $h \le 2/|\lambda|$ — derived. Stiffness defined; contact/constraint forces make $|\lambda|$ huge. Semi-implicit (symplectic) Euler on the harmonic oscillator: update matrix, exact conservation of a modified energy (computed), stability for $h<2/\omega$ — why game-physics engines (rapier included) use it. RK4's accuracy-vs-cost trade-off; why *accuracy* is not the binding constraint in RL rollouts, *stability and bias consistency* are.
- **C:** `ch15-integrator-playground`: Pendle with selectable integrator (explicit Euler / semi-implicit / RK4) and $\Delta t$ slider; phase portrait + energy trace; the reader cranks $\Delta t$ until the explicit pendulum explodes and verifies the blow-up threshold against $2/|\lambda|$ printed live.
- **P:** `rl-sim::integrate` gets all three integrators for teaching (rapier keeps its own); `criterion` bench: steps/sec vs energy drift on Pendle; assert the measured explicit-Euler blow-up matches theory within 5 %.

### 15.2 Inside the Engine II: Contact, the LCP & the Parameters That Aren't Physical
- **F:** Velocity-impulse contact sketch: $v^+ = v^- + M^{-1}(J_n^\top \lambda_n + J_t^\top \lambda_t + h\,f_{ext})$ with complementarity $0 \le \lambda_n \perp (J_n v^+ + b) \ge 0$ (Baumgarte bias $b$ for penetration), Coulomb condition $\|\lambda_t\| \le \mu \lambda_n$ with maximum dissipation — the friction problem is a nonlinear complementarity problem; engines facet the cone → LCP, solved approximately by projected Gauss–Seidel/impulse iterations (Stewart–Trinkle, Anitescu–Potra cited for the exact formulations; full derivation out of scope, statement precise). Consequence, itemized: solver iteration count, bias coefficient, cone faceting, and $\Delta t$ all change effective contact behavior — *simulation parameters with no physical referent*, the formal root of Kober's under-modeling curse and of why "contact is where sim and real diverge."
- **C:** `ch15-contact-heatmap`: a box dropped onto a ramp (later: Ferris foot on ground); heatmap of final rest position / slip distance over a grid of (friction $\mu$, restitution) × (solver iterations, $\Delta t$); cliffs and plateaus make parameter sensitivity visceral.
- **P:** `rl-sim::params::PhysicsParams` — one serializable struct gathering every knob rapier exposes (gravity, per-body mass/inertia scales, friction, restitution, joint damping, motor strength, solver iters, $\Delta t$); `apply(&mut rapier_ctx)` implemented; this struct is the randomization and sysid substrate for the whole book.

### 15.3 The Gap Taxonomy — and Ferris
- **F:** The sim-to-real gap decomposed: dynamics gap (wrong $P$), actuation gap (motor/gearbox dynamics, latency — Tang's actuator-net evidence for quadrupeds), perception gap (visual/observation mismatch — flagged, deferred to Ch 19), reset & wear gap (Ch 14). Formalized as POMDP mismatch: real $(P^\ast,\Omega^\ast)$ vs sim family $(P_\xi,\Omega_\xi)$; transfer gap $J^\ast(\pi) - J_\xi(\pi)$ bounded by the Ch 14 $O(T^2\varepsilon_m)$ result, motivating both shrinking $\varepsilon_m$ (sysid) and widening training support (randomization). **Ferris introduced**: 12-DoF quadruped (3 per leg), URDF spec, mass/inertia table, joint-level PD (from Ch 13) as the low-level layer — and the honest note that its rapier contact model is exactly the kind of liar §15.2 described.
- **C:** `ch15-gap-anatomy`: animated diagram: one policy, four failure overlays (dynamics/actuation/perception/reset) on a Ferris step sequence; hovering each overlay shows the formal object that's wrong ($P$, delay kernel, $\Omega$, initial-state distribution).
- **P:** `rl-envs::ferris`: URDF via `urdf-rs` + `k`, rapier3d articulated body, PD stand-in-place controller, `bevy` viewer scene; smoke test: Ferris stands for 30 s under randomized ±20 % masses — the chapter's "hello quadruped."

### 15.4 Domain Randomization as Distributional Robustness
- **F:** Plain DR: $\max_\theta \; \mathbb E_{\xi\sim p(\xi)}\,\mathbb E_{\tau\sim P_\xi,\pi_\theta}[G_0]$ — reality is (hoped to be) in the support, policy must carry robustness because $\pi_\theta(a|o)$ cannot condition on $\xi$. Robust variants derived: CVaR$_\beta$ over $\xi$ (EPOpt — optimize the worst $\beta$-fraction of worlds); KL-ball DRO $\max_\theta \min_{q:\,\mathrm{KL}(q\|p)\le\rho} \mathbb E_q[J]$ with the closed-form inner solution $-\beta\log \mathbb E_p[e^{-J/\beta}]$ (soft-min; derivation via Donsker–Varadhan, three displayed lines). The price theorem-sketch: widening $p$ shrinks the feasible performance of any single ξ-blind policy — motivating either recurrent policies that *infer* $\xi$ (adaptation, →15.6) or narrowing $p$ with sysid (→15.5). Design table: what to randomize for dynamics (mass, friction, latency, motor strength) vs visual (deferred), and Kober §6.1's "adding noise smooths model errors" recognized as this idea's 1995-era ancestor (Jakobi).
- **C:** `ch15-randomization-wall`: a wall of distribution sliders (mass scale, friction, latency, motor strength: mean + width each); center panel shows sampled Reacher "worlds" as ghost overlays; right panel a robustness meter — success rate of two fixed policies (trained narrow vs trained wide) evaluated live on draws from the reader's distribution; the crossover as width grows *is* the lesson.
- **P:** `rl-sim::randomize::DomainDistribution` over `PhysicsParams` (per-param `Uniform`/`LogUniform` + latency categorical), `EnvFactory` trait: `make(params) -> Env`; hooks called at every reset. Train SAC Reacher under three widths (none / calibrated / extreme); evaluation deferred to §15.8's transfer table.

### 15.5 System Identification: Pulling the Simulator Toward Reality
- **F:** Sysid as inverse problem: $\hat\xi = \arg\min_\xi \sum_i \big\| \mathrm{sim}_\xi(s_0^i, a_{1:T}^i) - s_{1:T}^i \big\|^2_W$ over logged real trajectories (here: "real" = held-out perturbed sim, honestly labeled). Why gradient-free: rapier is non-differentiable through contacts; CMA-ES recap (covariance adaptation as sampled natural-gradient — connection to Ch 10's natural gradient stated). Identifiability: parameters only visible under exciting inputs (persistent excitation, stated); practical remedy — chirps and torque pulses as identification trajectories. Bayesian upgrade in one paragraph: keep a *posterior* $p(\xi\mid\mathcal D)$ and randomize from it (links Kober's distributions-over-models to modern real-to-sim pipelines).
- **C:** `ch15-sysid-lab`: side-by-side Reacher: logged "real" trajectory (ghost) vs current sim draw; the reader can hand-tune $\xi$ sliders first (feel the coupling), then press "CMA-ES" and watch the population ellipse contract on a 2-D projection (mass × friction) with fit error dropping.
- **P:** `rl-sim::sysid`: trajectory-discrepancy loss + `cmaes` crate driver; experiment: recover planted parameters from 10 logged trajectories, report error vs number/type of trajectories (random policy vs chirp excitation) — a table that teaches excitation better than any paragraph.

### 15.6 Teacher–Student & Privileged Learning
- **F:** Asymmetric actor-critic (Pinto et al. 2017): in sim, full state $s$ (and $\xi$) is available; critic $V_\psi(s,\xi)$, actor $\pi_\theta(a|o)$. The score-function gradient $\nabla_\theta J = \mathbb E[\nabla_\theta\log\pi_\theta(a_t|o_t)\,\hat A_t]$ remains unbiased for any return estimator — conditioning the *critic* on privileged information only reduces variance; the deployed policy never touches it (derivation from Ch 10's PG theorem, two lines). Teacher–student (the ANYmal recipe, Lee et al. 2020; RMA): teacher $\pi_T(a\,|\,s, e)$ trained with RL on privileged extrinsics $e(\xi)$ + terrain; student $\pi_S(a\,|\,o_{1:t})$ minimizes $\mathbb E_{d^{\pi_S}}\big[\mathrm{KL}\big(\pi_T(\cdot|s,e)\,\|\,\pi_S(\cdot|o_{1:t})\big)\big]$ with on-student state distribution — which is exactly DAgger, formally cross-referenced to Ch 16 where its regret guarantee is proved. Adaptation-module variant: student regresses $\hat z \approx \mu(e)$ from history, sharing the teacher's motor head.
- **C:** `ch15-teacher-student-flow`: animated information-flow diagram: privileged channels ($\xi$, contact states, true velocities) glow into the teacher/critic during phase 1; phase 2 shows distillation arrows from teacher to student under the *student's* trajectory distribution; a deployment toggle cuts the privileged wires and the student runs alone.
- **P:** Minimal but real instance on Reacher: teacher SAC conditioned on $(s,\xi)$ (oracle dynamics vector appended), student GRU policy in `burn` distilled with the Ch 16 DAgger loop (imported, pre-publication of that API); table: oracle teacher vs student vs ξ-blind baseline on randomized worlds. Full-scale quadruped version deferred to Ch 18.

### 15.7 Scale: The `rayon` Rollout Farm
- **F:** Throughput model: $\text{samples/s} = N_{\text{envs}} \cdot f_{\text{step}} \cdot \eta(N)$, with $\eta$ the parallel efficiency (Amdahl term for the learner's synchronous update); on-policy PPO's batch appetite (Ch 10: 2048–65536 transitions/update) vs off-policy replay reuse (Ch 11) — why massively parallel simulation revived on-policy methods for robots (Tang: PPO dominance in locomotion). Determinism discipline: per-env seeded RNG streams, Kober §6.1's random-number-reuse (PEGASUS) as variance reduction for policy comparison — same seeds, different policies.
- **C:** `ch15-rollout-farm`: dashboard over a live native run (recorded trace in the browser): envs/sec, per-core utilization bars, sample counter, and a tile wall of 64 miniature Ferrises stepping in lockstep; thread-count slider shows $\eta(N)$ bending.
- **P:** `rl-sim::farm`: `VecEnv` over `rayon` (`into_par_iter` per rollout worker, lock-free policy sharing via `Arc`), fixed base-seed + offset streams (`rand::SeedableRng`); bench: 4096 randomized Ferris instances, steps/sec native vs single-thread — the farm Ch 18 and Ch 22 train on.

### 15.8 Crossing the Bridge: Zero-Shot, Few-Shot & the Decision Tree
- **F:** The evaluation protocol: train on $p(\xi)$, test on held-out $\xi^\ast \notin \mathrm{supp}$-center (Ch 14 rubric applied; this is our honest stand-in for "real"). Definitions: zero-shot transfer (no target-domain updates), few-shot (limited target-domain adaptation: fine-tuning, residual models, online sysid), real-world learning (no sim). Tang's empirical synthesis as a decision tree with the evidence attached: stable simulable dynamics + a-priori dense reward → zero-shot (quadrupeds, drones — L3–L5); diverse objects / hard contact / perception-heavy → few-shot or demo-driven (manipulation — mostly ≤L2); unmodelable agents (HRI) → offline/real-world (Ch 21). Where residual few-shot adaptation fits (learn $\Delta$-dynamics from real data — Ch 12 models reused).
- **C:** `ch15-transfer-tree`: interactive decision tree; the reader answers questions about their task (Can you write the reward a priori? Is the closed loop self-stabilizing — Ch 14 §14.4's dichotomy? Object diversity?) and gets a recommended recipe with the Tang-surveyed exemplar systems and their L-levels at each leaf.
- **P:** The chapter's capstone experiment: Reacher transfer table — {no DR, DR, DR+sysid} × {in-distribution, held-out $\xi^\ast$} with Wilson CIs from Ch 14's `SuccessReport`; expected result reproduced: no-DR collapses on $\xi^\ast$, DR+sysid degrades gracefully; artifact shipped as the `ch15-randomization-wall` data source.

### 15.9 Chapter Bridge
- **F:** Recap ledger: the engine's honest limits (integrator, LCP), four bridge components (DR/DRO, sysid, privileged learning, scale), one protocol (held-out transfer + L-levels). What remains open: perception gap (Ch 19), demonstrations as the other data source (Ch 16, next), and representations that make transfer easier (Ch 17).
- **C:** Static bridge diagram (mermaid): sim side, real side, four spans, each labeled with its section and its Part IV consumer.
- **P:** Handoff manifest: `FerrisEnv` + `DomainDistribution` + `farm` + `sysid` land in `rl-sim`/`rl-envs` with doc examples; Ch 18 imports all four unchanged.

## 4. Interactive Widgets & Dashboards

| ID | Type | What it shows | Reader interaction | Tech |
|---|---|---|---|---|
| `ch15-integrator-playground` | sandbox | Pendle phase portrait + energy under three integrators; blow-up threshold vs theory | choose integrator, crank $\Delta t$, perturb state, read live $2/\|\lambda\|$ check | rapier2d-free custom ODE core + egui_plot (WASM) |
| `ch15-contact-heatmap` | widget | slip/rest-position sensitivity over (μ, restitution) × (solver iters, $\Delta t$) | pick axes pair, click a cell to replay that world | rapier2d (WASM) + egui |
| `ch15-gap-anatomy` | animation | four gap overlays (dynamics/actuation/perception/reset) on a Ferris step cycle | hover overlays for formal objects; step through the cycle | bevy (WASM) + egui |
| `ch15-randomization-wall` | widget | distribution sliders → sampled ghost worlds + robustness meter for narrow-vs-wide policies | shape $p(\xi)$; watch success crossover; load §15.8 table presets | rapier2d + burn inference (WASM) + egui |
| `ch15-sysid-lab` | sandbox | real-vs-sim trajectory overlay; CMA-ES population ellipse contracting on 2-D projection | hand-tune $\xi$; run/step CMA-ES; switch excitation dataset | rapier2d + cmaes (WASM) + egui_plot |
| `ch15-teacher-student-flow` | animation | privileged channels into teacher/critic; distillation arrows; deployment cut | phase scrubber; deployment toggle; inspect each channel | egui + custom canvas |
| `ch15-rollout-farm` | dashboard | envs/sec, core utilization, $\eta(N)$ curve, 64-Ferris tile wall | thread-count slider (recorded trace), zoom tiles | egui_plot + recorded native telemetry |
| `ch15-transfer-tree` | widget | Tang-evidence decision tree with exemplar systems and L-levels at leaves | answer task questions; expand leaves for citations | egui (WASM) |

## 5. Rust Implementation Plan

**Crates touched:** `rl-sim` (the chapter's center: `params`, `randomize`, `sysid`, `farm`, `integrate`), `rl-envs` (`ferris` module — new robot), `rl-deep` (GRU student policy, distillation trainer stub shared with Ch 16), `rl-viz` (farm telemetry panel), `demos/ch15-*` (eight WASM crates).

**New modules/files:** `rl-sim/src/{params.rs, randomize.rs, sysid.rs, farm.rs, integrate.rs}`, `rl-envs/src/ferris/{mod.rs, urdf/ferris.urdf, pd.rs}`, `rl-deep/src/distill.rs`.

Representative sketch — randomization hooks feeding the `rayon` farm:

```rust
/// rl-sim/src/randomize.rs — a reader-editable distribution over physics.
#[derive(Clone, Serialize, Deserialize)]
pub struct DomainDistribution {
    pub mass_scale: LogUniform,        // e.g. LU(0.8, 1.25) per link
    pub friction: Uniform,             // e.g. U(0.4, 1.0)
    pub motor_strength: LogUniform,    // actuation gap
    pub obs_latency: Categorical,      // {0,1,2} control steps
}

impl DomainDistribution {
    pub fn sample(&self, rng: &mut StdRng) -> PhysicsParams { /* one world ξ */ }
}

/// rl-sim/src/farm.rs — thousands of randomized envs, one seeded stream each.
pub fn collect<E, F>(factory: &F, dist: &DomainDistribution, policy: &(impl Policy + Sync),
                     n_envs: usize, horizon: usize, base_seed: u64) -> Vec<Rollout>
where F: EnvFactory<Env = E> + Sync, E: Env {
    (0..n_envs as u64).into_par_iter().map(|i| {
        let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(i)); // PEGASUS-style reuse
        let mut env = factory.make(dist.sample(&mut rng));  // fresh world ξ every episode
        rollout(&mut env, policy, horizon, &mut rng)        // lock-free: policy is Sync
    }).collect()
}
```

**Experiments/benchmarks:** (1) integrator blow-up vs theory (`criterion` + assert); (2) contact sensitivity grid; (3) sysid recovery vs excitation type; (4) three-width DR training sweep; (5) teacher/student/blind triple on randomized Reacher; (6) farm throughput scaling 1→32 cores; (7) the §15.8 transfer table with CIs. Seeds pinned; all outputs `serde` JSON consumed by widgets.

**Native vs browser:** training and the 4096-env farm are native (WGPU/`rayon`); every widget runs in-browser with WASM rapier or recorded traces (farm dashboard replays native telemetry — noted per CLAUDE.md as the one non-live widget).

## 6. Robot Thread

- **Ferris** — **first appearance.** Before: a name on Ch 14's dimensionality ladder. After: URDF-defined 12-DoF rapier3d quadruped with PD low-level control, standing under ±20 % mass randomization, instantiated 4096× on the farm. Cannot walk yet — that is Ch 18's payoff, using exactly this chapter's infrastructure.
- **Pendle** — the integrator playground's test mass; its Ch 13 dynamics now understood *as the engine computes them*.
- **Reacher** — the transfer guinea pig: sysid target, DR training subject, teacher–student demo; leaves with a policy that survives held-out physics (its Ch 14 fragility heatmap redrawn, green).
- **Rusty** — sidebar only: the perception gap it will face in Ch 19 is named but not solved.

## 7. Exercises & Explorations

1. **(F)** Derive the stability interval of semi-implicit Euler for the undamped oscillator $\ddot x=-\omega^2 x$ and show it conserves the modified energy $E_h = \tfrac12 v^2 + \tfrac12\omega^2 x^2 - \tfrac{h}{2}\omega^2 xv$ exactly; verify numerically in the playground.
2. **(F)** Complete the KL-ball DRO derivation: show $\min_{q:\mathrm{KL}(q\|p)\le\rho}\mathbb E_q[J] = \max_{\beta\ge 0}\big[-\beta\log\mathbb E_p[e^{-J/\beta}] - \beta\rho\big]$ and recover plain DR as $\rho\to 0$ and worst-case robustness as $\rho\to\infty$.
3. **(F)** Show the asymmetric-critic estimator is unbiased: conditioning $\hat A_t$ on $(s_t,\xi)$ leaves $\mathbb E[\nabla_\theta\log\pi_\theta(a_t|o_t)\hat A_t]$ equal to the PG-theorem gradient, but the *optimal policies* for the POMDP and the privileged MDP generally differ — construct a two-state aliasing example.
4. **(C)** In `ch15-contact-heatmap`, find a (μ, iterations) pair where halving $\Delta t$ changes the slip distance by >20 % — then explain, in LCP terms, which non-physical parameter you just exposed.
5. **(C)** Use `ch15-randomization-wall` to locate the training-distribution width at which the wide-trained policy's *in-nominal* success drops below the narrow policy's — the measured price of robustness from §15.4.
6. **(P)** Add per-episode observation-noise randomization to `DomainDistribution`; retrain and report whether it substitutes for latency randomization on the transfer table.
7. **(P)** Break sysid deliberately: generate "real" data with joint backlash (not in `PhysicsParams`), fit anyway, and document the biased compensation CMA-ES finds — Kober's under-modeling curse reproduced in 30 lines.
8. **(P)** Extend the farm with asynchronous evaluation workers (separate `rayon` pool) that grade checkpoints on held-out $\xi^\ast$ during training; plot transfer gap vs training time for the three DR widths.

## 8. Notation Introduced

| Symbol | Meaning |
|---|---|
| $h$ (also $\Delta t$) | integrator timestep |
| $\xi$, $p(\xi)$ | simulator/physics parameter vector; its randomization distribution |
| $P_\xi$, $\Omega_\xi$ | transition and observation kernels of the ξ-world (POMDP family) |
| $\lambda_n, \lambda_t$ | normal / tangential contact impulses (LCP variables) |
| $J_n, J_t$ | contact Jacobians (distinguished from objective $J(\theta)$ by subscript convention) |
| $\mu$ (contact) | Coulomb friction coefficient (context-separated from policy-mean $\mu$) |
| $\mathrm{CVaR}_\beta$, $\rho$ | conditional value-at-risk level; KL-ball radius in DRO |
| $e$, $\hat z$ | privileged extrinsics; student's estimate of the teacher's latent |
| $\pi_T, \pi_S$ | teacher and student policies |
| $\eta(N)$ | parallel efficiency of the rollout farm at $N$ workers |

Registered for Appendix C; the $J$-subscript and context rules for $\mu$ are recorded there as disambiguation conventions.

## 9. References & Further Reading

- **Kober, Bagnell & Peters (IJRR 2013)** — §3.3 (under-modeling), §6.1 (simulation bias; distributions over models; Jakobi-style noise; PEGASUS random-number reuse), §7.5 (ball-in-a-cup sim that matched data yet failed to transfer).
- **Tang et al. (2024)** — §3.3 (simulator-usage axis: zero-shot / few-shot / no-sim), §4.1.1 (actuator nets, dynamics randomization for quadrupeds), §5 (zero-shot dominance among L3–L4 systems; real-world learning as open challenge).
- **Akinola (Columbia slides)** — simulators as a key element of recent successes (mujoco/pybullet era) — recast here in Rust/rapier.
- Stewart & Trinkle (1996); Anitescu & Potra (1997) — time-stepping LCP contact. Hairer, Lubich & Wanner (2006) — geometric numerical integration (symplectic Euler).
- Jakobi, Husbands & Harvey (1995) — noise-based transfer. Tobin et al. (2017) — visual domain randomization. Peng et al. (2018) — dynamics randomization. Rajeswaran et al. (2017) — EPOpt (CVaR). Muratore et al. (2022) — sim-to-real survey.
- Pinto et al. (2017) — asymmetric actor-critic. Lee, Hwangbo et al. (2020, Science Robotics) — privileged terrain teacher–student. Kumar et al. (2021) — RMA. Hwangbo et al. (2019) — actuator networks.
- Hansen (2016) — CMA-ES tutorial. Ng et al. (2004) — PEGASUS. Makoviychuk et al. (2021) — Isaac Gym massively parallel RL (the GPU analogue of our `rayon` farm).

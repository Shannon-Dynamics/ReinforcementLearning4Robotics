# Chapter 20 — Learning Manipulation

> **Part** IV — Competencies: RL on Real Robots · **Builds on:** Ch 9, 11, 13, 14, 15, 16, 17 · **Feeds:** Ch 21, 22
> **Modernizes:** Tang §4.3 (4.3.1 pick-and-place & grasping, 4.3.2 contact-rich, 4.3.3 in-hand, 4.3.4 non-prehensile, 4.3.5 trends); Kober §7's case-study discipline applied to the deep era's manipulation record.

## 1. Purpose & Learning Outcomes

Manipulation is the competency Tang rates least mature — few deployments in the wild, many impressive demonstrations — and the reasons are structural: contact mechanics are combinatorial and hard to simulate, objects are endlessly diverse, and perception is coupled to control. This chapter builds the classical foundations RL quietly relies on (wrench spaces, force closure, impedance control), then pays off three earlier investments on a parallel-jaw gripper: the Ch 9 DQN becomes a grasp selector, Ch 11's SAC with Ch 13's impedance formalism learns insertion, and Ch 16's demonstration pipeline feeds a pushing task.

The reader can:
- Count contact modes and explain, with the $3^{n}$ argument, why planning through contact explodes and how RL's stochastic smoothing sidesteps it.
- Derive force closure from friction cones and the grasp map, and compute the Ferrari–Canny quality metric for a planar grasp.
- Formulate grasping as a one-step/bandit RL problem and explain why open-loop grasp selection reached real-world use before closed-loop control.
- Derive the impedance control law as an action space and explain, via the insertion funnel, why compliance beats position control under pose uncertainty.
- Explain the in-hand manipulation recipe (massive randomization + privileged distillation + recurrent adaptation) as the composition of Ch 15/18/19 machinery.
- Run success-rate evaluations with correct confidence intervals and state what "87% success" does and does not establish.

## 2. Storyline

**Act I — The hand is not a foot.** Hook: replay Ch 18's triumph, then swap terrain contact for object contact. Ferris's feet met the world four times per stride in ways a heightfield describes; a gripper meets a mug in a combinatorial explosion of sticking, sliding and separating contacts, on objects the robot has never seen. Tang's maturity gradient (grasping and assembly succeed, open-world pick-and-place stalls below L3) is the mystery the chapter explains: RL wins where the task family can be *enumerated at training time*.

**Act II — Foundations, then payoffs.** Classical grasp mechanics first, because the learned methods are unintelligible without it: wrench spaces and force closure (20.2) give the vocabulary for what a "good grasp" *is*. Then three payoff sections mirror the book's own toolset: discrete grasp selection with the Ch 9 DQN (20.3), contact-rich insertion with SAC acting in Ch 13's impedance space (20.4), and the in-hand lineage as the recipe Ch 15–19 already assembled, pushed to its limit (20.5). Reacher, wearing a parallel-jaw gripper for the first time, is the testbed throughout.

**Act III — Data diets and honest numbers.** Manipulation's preferred diet is demonstrations and offline data (Tang Tables 3–4 patterns): the Ch 16 browser-teleop pipeline collects pushes and insertions from the reader, and demo-boosted SAC measurably outruns exploration from scratch. The chapter closes on evaluation honesty — Bernoulli statistics, confidence intervals, and the trial counts real claims require — building the harness Ch 22 will inherit.

## 3. Section-by-Section Design

### 20.1 Why Manipulation Is the Hard One: Contact-Mode Combinatorics
- **F:** Hybrid dynamics of contact: each of $n$ potential contacts is in one of {separated, sticking, sliding} (sliding further directional), giving $\ge 3^n$ dynamics modes; piecewise-smooth vector fields, mode-dependent Jacobians; why gradients through mode boundaries are discontinuous and planners must search the mode lattice. RL's answer formalized: a stochastic policy and randomized dynamics smooth the discontinuity in expectation — $\nabla_\theta \mathbb E[G]$ exists even where $\nabla_a f$ does not. Tang §4.3's maturity analysis mapped onto this structure: enumerable task families (grasp, insert, in-hand) allow the smoothing to be *trained through*; open-world diversity does not.
- **C:** `ch20-contact-modes` — widget: a planar gripper + block scene where the reader toggles each fingertip/table contact through its modes; a live counter and mode-lattice graph show reachable modes exploding from 3 to $3^4$; a "planner view" tries to search the lattice while an "RL view" samples through it.
- **P:** `rl-envs::gripper`: parallel-jaw gripper + object set (boxes, cylinders, meshes) in `rapier3d`; contact-event instrumentation (mode classification per contact) exposed in `info` for every later experiment.

### 20.2 Grasping I: Wrench Space and Force Closure
- **F:** Contact model zoo (frictionless point, point-with-friction, soft-finger); friction cone $FC_i = \{f : \lVert f - (f^\top n_i) n_i \rVert \le \mu\, f^\top n_i\}$; contact wrench $w_i = (f_i,\; p_i \times f_i)$; grasp map $G = [w_1 \cdots w_m]$; grasp wrench space as the convex hull of cone-discretized unit wrenches; **force closure** $\iff 0 \in \mathrm{int}(\mathcal W)$ $\iff$ the contact wrenches positively span $\mathbb R^6$ ($\mathbb R^3$ planar), proved both directions for the planar case; Ferrari–Canny quality $\epsilon = \max\{r : B_r(0) \subseteq \mathcal W\}$. Why analytic metrics under-predict real grasp success (pose uncertainty, non-rigidity) — the opening RL exploits.
- **C:** `ch20-wrench-space` — the chapter's signature widget: drag two/three contact points around a 2D object outline, tune $\mu$; the wrench space (3D, projected) forms live, force closure flips a badge, and $\epsilon$ is plotted; preset "why antipodal works" and "why two frictionless points never close" tours.
- **P:** `rl-core::grasp`: planar wrench-space computation (`nalgebra` + a small convex-hull routine), force-closure test, $\epsilon$ metric; property tests (`proptest`): antipodal grasps with $\mu > 0$ achieve closure; closure is scale-invariant.

### 20.3 Grasping II: Learned Grasp Selection (Ch 9 Payoff)
- **F:** Grasping as a one-step MDP/contextual bandit (Ch 3 callback): action = discrete grasp candidate $(u, v, \theta)$ on a depth image, sparse lift-success reward, $Q(I, g)$ as a success classifier; why open-loop selection + scripted execution reached fulfillment deployments (Tang §4.3.1.1) while closed-loop visuomotor grasping lags: credit assignment over one step vs many, and self-supervised label collection at scale. Sim-to-real for perception: depth-image domain gap vs Ch 15's dynamics gap.
- **C:** `ch20-grasp-heatmap` — visualization: trained $Q$ painted over a scene's depth image as a grasp-quality heatmap per rotation channel; the reader rotates objects and drops distractors, and the heatmap re-ranks; a side panel compares against the 20.2 analytic $\epsilon$ ranking on the same candidates.
- **P:** `rl-deep`: grasp-selection DQN (Ch 9 architecture reused nearly verbatim — the payoff is the reuse) on synthetic depth renders of `gripper` scenes; rotation-equivariant candidate enumeration; sim evaluation on held-out object meshes.

### 20.4 Contact-Rich Skills: Insertion with Impedance Action Spaces
- **F:** Impedance control derived from Ch 13's operational-space formalism: command $F = K(x^\* - x) - D\dot x$, $\tau = J^\top F + g(q)$, critically damped $D = 2\sqrt{\Lambda K}$; the **action space** $a = (\Delta x^\*, \mathrm{diag}\,K)$ — the policy modulates both target and stiffness. The insertion funnel formalized: success basin in initial-pose-error space; under position control the basin is the clearance $\sim 0.1$ mm, under compliance contact forces generate corrective displacement $\delta x = K^{-1} F_{\text{contact}}$, dilating the basin by orders of magnitude — derivation for the planar peg. Force/torque observations as the contact-mode sensor. Residual-on-reference structure (Ch 17) as the industrial variant per Tang §4.3.2.1.
- **C:** `ch20-insertion-funnel` — animation: peg approaching hole under sampled initial pose errors; position control vs two stiffness settings shown as funnels in error space (success region rendered as a basin); sliders for $K$, clearance, and friction reshape the funnel live.
- **P:** `rl-envs::insertion` + `rl-deep`: peg-in-hole task; SAC (Ch 11 payoff) with impedance actions via a wrapper env (code sketch in §5); F/T-in-observation ablation; comparison against position-action SAC at matched budgets — the headline experiment reproducing the funnel argument empirically.

### 20.5 In-Hand Manipulation: the OpenAI Cube Lineage
- **F:** The lineage read as *composition of the book's machinery*: massive domain randomization (Ch 15's distributional-robustness objective, pushed to morphology/vision extremes), recurrent policies as implicit system identification (Ch 19.2's belief-tracker theorem applied to dynamics parameters rather than pose), privileged teacher → vision/touch student (Ch 18.6), automatic domain randomization as curriculum over $\xi$ (Ch 18.5). Honest accounting per Tang §4.3.3: known-object reorientation is L2-ish lab success; arbitrary-object, arbitrary-axis rotation is recent; integration with downstream tasks remains open.
- **C:** `ch20-randomization-dial` — dashboard: randomization breadth (mass/friction/visual/latency ranges) on one axis, in-hand success and adaptation speed on the other, built from published ablations plus our own insertion-task reproduction; the reader drags breadth and watches the train/test-gap bars invert.
- **P:** Deliberately scoped: no dexterous hand is built (documented as a physical-scope decision). Instead the *recipe* is validated on our tasks: randomization-breadth ablation on 20.4's insertion with a GRU policy, demonstrating the memory-equals-sysid effect measurably (adaptation within episodes to perturbed friction).

### 20.6 Non-Prehensile Manipulation and the Demonstration Diet
- **F:** Quasi-static pushing mechanics sketch (motion cone intuition, limit-surface statement with citation rather than proof — flagged per style guide); pushing as underactuated manipulation through a single patch contact. Manipulation's data diet formalized (Ch 16 payoff): demo-regularized SAC objective $J(\pi) = \mathbb E_{\text{RL}}[Q] + \lambda\, \mathbb E_{\mathcal D}[\log \pi(a\mid s)]$, demos-in-replay, and offline-to-online staging; Tang Tables 3–4 pattern stated: manipulation uses expert data far more than locomotion, because exploration near contact is expensive and demonstrations are cheap.
- **C:** `ch20-demo-diet` — dashboard: success-vs-samples curves for scratch RL / demos-in-replay / BC-regularized / offline-then-online on the push and insertion tasks; a data-composition pie per curve makes the "diet" literal; reader-collected demos (below) append live.
- **P:** Push-to-goal task in `gripper` scenes; browser teleop (Ch 16 pipeline reused: mouse → `serde` datasets) collecting reader demonstrations; demo-boosted SAC runs; the four-diet comparison experiment.

### 20.7 Evaluation Honesty: Success Rates with Statistics
- **F:** Success as Bernoulli($p$): $\hat p = k/n$; why Wald intervals lie at small $n$/extreme $\hat p$; Wilson interval derived: $\tilde p = \frac{\hat p + z^2/2n}{1 + z^2/n} \pm \frac{z}{1+z^2/n}\sqrt{\hat p(1-\hat p)/n + z^2/4n^2}$; sample-size planning: $n \approx z^2/(4\varepsilon^2) \Rightarrow n \approx 385$ for ±5% at 95% worst case — so "9/10 trials" spans roughly [60%, 98%]. Stratified evaluation (per-object, per-pose-error tier) and the temptation of test-set leakage via curriculum. Tang §5's benchmarking-real-world-success challenge and the L0–L5 rubric as the claim's *scope*, statistics as its *precision*.
- **C:** `ch20-success-stats` — widget: an interval explorer where the reader sets $k, n$ and sees Wald vs Wilson vs Clopper–Pearson; a "paper mode" pastes real reported trial counts from surveyed papers and renders what was actually established.
- **P:** `rl-core::eval`: evaluation harness — seeded trial batteries, stratification labels, Wilson CIs, JSON reports; run on every policy trained this chapter; this exact harness is imported by Ch 22.

### 20.8 Chapter Bridge
- Recap: manipulation succeeds where task families are enumerable, action spaces are compliant, and data diets include demonstrations — each a design decision, not an algorithm. The gripper stack, the demo pipeline, and the evaluation harness all feed Ch 22. Ch 21 widens the frame beyond a single robot's contact with objects: humans in the loop, teams of robots, and foundation models that may rewrite the data-diet economics just surveyed.

## 4. Interactive Widgets & Dashboards

| ID | Type | What it shows | Reader interaction | Tech |
|---|---|---|---|---|
| `ch20-contact-modes` | widget | Contact-mode lattice exploding as contacts are added | Toggle per-contact modes; race planner-search vs RL-sampling views | egui, WASM |
| `ch20-wrench-space` | widget | Grasp wrench space forming from dragged contacts; force-closure badge; $\epsilon$ metric | Drag contacts on 2D objects; tune friction $\mu$; guided tours | egui + nalgebra, WASM |
| `ch20-grasp-heatmap` | visualization | Learned $Q(I,g)$ as per-rotation grasp-quality heatmaps vs analytic ranking | Rotate objects, add distractors; compare learned vs $\epsilon$ ranking | egui + burn WASM inference |
| `ch20-insertion-funnel` | animation | Success basins in pose-error space: position vs impedance control | Sliders for stiffness $K$, clearance, friction; sample initial errors | egui_plot + rapier2d slice, WASM |
| `ch20-randomization-dial` | dashboard | Randomization breadth vs success and within-episode adaptation | Drag breadth; watch train/test gap invert; toggle GRU vs MLP | egui_plot |
| `ch20-demo-diet` | dashboard | Success-vs-samples under four data diets; live reader-demo appends | Collect demos via teleop; rerun comparison; inspect diet composition | egui + serde datasets |
| `ch20-success-stats` | widget | Wald/Wilson/Clopper–Pearson intervals; "what 9/10 means" | Set $k, n$, confidence; paste surveyed papers' trial counts | egui_plot, WASM |

## 5. Rust Implementation Plan

**Crates touched:** `rl-envs` (`gripper`, `insertion`, `push` modules), `rl-core` (`grasp`, `eval` modules), `rl-deep` (DQN and SAC reused from Ch 9/11; impedance wrapper new), `rl-sim` (contact instrumentation), demos `ch20-*`.

**Modules/files:** `rl-envs/src/gripper/{env.rs, objects.rs, contact_modes.rs}`, `rl-envs/src/insertion.rs`, `rl-envs/src/push.rs`, `rl-core/src/grasp.rs`, `rl-core/src/eval.rs`, `rl-deep/src/impedance.rs`, `demos/ch20-*/`.

Representative sketch — impedance action-space wrapper, the Ch 13 → Ch 11 payoff (`rl-deep/src/impedance.rs`):

```rust
/// Wraps a manipulation env so the agent acts in impedance space (Sec. 20.4):
/// action = [dx*, dy*, dtheta*, log k_xy, log k_th] — pose offset + stiffness.
pub struct ImpedanceEnv {
    inner: InsertionEnv,
    k_bounds: (f64, f64),   // stiffness clamp, N/m
    zeta: f64,              // damping ratio; D = 2*zeta*sqrt(Lambda*K)
    substeps: usize,        // inner control steps per RL step (e.g., 10)
}

impl Env for ImpedanceEnv {
    fn step(&mut self, a: &Action) -> Transition {
        let (dx_star, k) = self.decode(a); // clamp K to k_bounds, exp of log-k
        let x_star = self.inner.ee_pose().compose(&dx_star);
        for _ in 0..self.substeps {
            let err = pose_err(&x_star, &self.inner.ee_pose());
            let d = self.critically_damped(&k);
            let f = k.component_mul(&err) - d.component_mul(&self.inner.ee_vel());
            let tau = self.inner.jacobian().transpose() * f + self.inner.gravity_comp();
            self.inner.apply_torques(&tau);
        }
        // Observation includes the measured wrench: the contact-mode sensor.
        self.inner.observe_with_ft()
    }
}
```

**Experiments/benchmarks:** (1) analytic-$\epsilon$ vs learned-$Q$ grasp ranking correlation on held-out meshes; (2) impedance vs position action space on insertion at matched budgets (headline); (3) randomization-breadth × {GRU, MLP} ablation; (4) four-diet demo study; (5) every result reported through `rl-core::eval` with Wilson CIs; `criterion` on contact-instrumented stepping cost.

**Native vs browser:** training native; `ch20-wrench-space`, `ch20-contact-modes`, `ch20-insertion-funnel`, `ch20-success-stats`, teleop demo collection, and grasp-heatmap inference all in-browser; `rapier3d` gripper scenes run in WASM for replay and teleop, full-speed training documented native-only.

## 6. Robot Thread

- **Reacher** (transformation): gains a parallel-jaw gripper and 3D scenes — from reaching (Ch 11/13) through demo-collection (Ch 16) and action-space studies (Ch 17) to grasping, insertion, and pushing; ends the chapter as the book's manipulation platform.
- **Rusty/Ferris**: rest (mobile-Reacher's whole-body skills from Ch 19 are cited, not retrained).
- **Pendle**: cameo in 20.4 — the critically-damped-spring analysis is Pendle's Ch 13 math wearing a task-space coat.

## 7. Exercises & Explorations

1. **(F)** Prove that two frictionless point contacts can never achieve planar force closure, and compute the minimum $\mu$ for which an antipodal grasp on a square does.
2. **(F)** For the planar peg with clearance $c$ and stiffness $K$, derive the maximum initial lateral error the impedance controller corrects, and verify the prediction against `ch20-insertion-funnel`.
3. **(F)** Derive the Wilson interval from inverting the score test, and compute how many trials a claim of "≥90% with ±3% at 95%" requires.
4. **(C)** In `ch20-wrench-space`, try to find a three-contact grasp whose $\epsilon$ *decreases* as $\mu$ increases; explain why the widget will never show one (monotonicity of the friction cone in $\mu$), then find a contact-*placement* change that trades higher $\epsilon$ against higher sensitivity to placement error.
5. **(C)** Use `ch20-randomization-dial` to identify the breadth at which the MLP policy collapses but the GRU survives; connect to the Ch 19.2 belief-tracker argument in one paragraph.
6. **(P)** Add a soft-finger contact model option to `rl-core::grasp` and measure how the force-closure region of the 20.2 tours changes.
7. **(P)** Implement residual-on-reference insertion (Ch 17): scripted approach + learned residual impedance policy; compare sample efficiency against full SAC and relate to Tang §4.3.2.1's industrial pattern.
8. **(P)** Collect 30 of your own push demonstrations via the browser teleop and reproduce the four-diet comparison; report all results with Wilson CIs from `rl-core::eval`.

## 8. Notation Introduced

| Symbol | Meaning |
|---|---|
| $FC_i, \mu, n_i$ | friction cone at contact $i$, friction coefficient, contact normal |
| $w_i = (f_i, p_i \times f_i)$ | contact wrench; *clash note:* reward weights stay $w_k$ with index $k$ (Ch 18) |
| $G, \mathcal W$ | grasp map; grasp wrench space (convex hull of cone-edge wrenches) |
| $\epsilon$ | Ferrari–Canny grasp quality (largest origin-centered ball in $\mathcal W$) |
| $K, D, \Lambda$ | task-space stiffness, damping, inertia (impedance action space) |
| $x^\*, \Delta x^\*$ | commanded equilibrium pose and per-step offset |
| $\hat p, z, \varepsilon$ | success estimate, normal quantile, interval half-width |
| $\mathcal D, \lambda$ | demonstration dataset; demo-regularization weight (Ch 16 reuse) |

## 9. References & Further Reading

- **Baseline:** Tang §4.3.1–4.3.5 (evidence and maturity analysis), Tables 3–4 patterns (expert usage), §5 (benchmarking challenge); Kober §7 (case-study discipline; ball-in-a-cup as proto-contact-rich), §3.4 (reward shaping under contact).
- Mason 2018, *Toward Robotic Manipulation* (Annual Review; the taxonomy Tang follows); Murray, Li & Sastry 1994 (wrench space, force closure — proof sources).
- Ferrari & Canny 1992, grasp quality metrics.
- Kalashnikov et al. 2018, *QT-Opt* (closed-loop vision-based grasping at scale); Pinto & Gupta 2016 (self-supervised grasp labels); Mahler et al. 2017, *Dex-Net 2.0* (analytic-metric-supervised learning, the bridge from 20.2 to 20.3).
- Levine et al. 2016, deep visuomotor policies (end-to-end lineage).
- Tang et al. 2023, *IndustReal* (full sim-to-real assembly pipeline); Luo et al. 2024, *SERL* (real-world RL insertion, integrator comparison); Johannink et al. 2019 (residual RL for insertion).
- OpenAI et al. 2019, in-hand cube and Rubik's cube (massive randomization + LSTM adaptation); Handa et al. 2023, *DeXtreme*; Chen et al. 2023, *Visual Dexterity* (arbitrary-object reorientation); Nagabandi et al. 2019 (model-based Baoding balls).
- Zeng et al. 2020, *TossingBot* (dynamic non-prehensile); Zhou et al. 2023, *HACMan* (non-prehensile reorientation).
- Agarwal et al. 2021, *Deep RL at the edge of the statistical precipice* (evaluation statistics; companion to 20.7).

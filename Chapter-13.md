# Chapter 13 — The Robot as an Environment: Kinematics, Dynamics & Control

> **Part** III — The Robotics Side · **Builds on:** Ch 2, 4, 5, 11, 12 · **Feeds:** Ch 14, 15, 16, 17, 18, 20, 22
> **Modernizes:** Kober §1.3 (robotics as an RL domain: high-dimensional continuous state/action, partial observability, filters and information states) and §2 (RL formalism in robotics context); the classical results themselves (FK/IK, Jacobians, Lagrangian dynamics, PID/LQR, EKF) are standard textbook material cited to Spong et al. and Siciliano et al. — this chapter's modernization is presenting them *as the anatomy of the RL environment* and as the baselines RL must beat.

## 1. Purpose & Learning Outcomes

For twelve chapters `env.step(a)` has been a black box that Parts I–II poked with algorithms. Part III opens the box. When the environment is a robot, `step` is rigid-body kinematics, Lagrangian dynamics, motor torques, and noisy encoders — structure that classical robotics wrote down long before RL arrived, and that Kober §1.3 insists any serious robot-RL practitioner must know. This chapter derives that structure for the book's own cast (Reacher in full, Pendle as the underactuated foil), builds the `rl-sim` crate around it, and erects the classical controllers — PID and LQR — that every learned policy in Parts III–V must beat to justify its existence.

The reader can:
- Represent poses in SE(2)/SE(3), compose homogeneous transforms, and compute Reacher's forward kinematics by hand.
- Solve Reacher's inverse kinematics analytically (both elbow branches), and characterize workspace and singularities via the Jacobian determinant.
- Derive the manipulator equation M(q)q̈ + C(q,q̇)q̇ + g(q) = τ for Reacher from the Lagrangian, term by term.
- Derive the discrete-time LQR gain from the Riccati recursion — and recognize it as Ch 5's value iteration in closed form.
- Tune a PID loop and a gravity-compensated PD loop, and state when linearization-based control is valid.
- Sketch the EKF and explain why partial observability (Ch 4's POMDP honesty) is the rule on hardware.
- Build `rl-sim`: URDF-loaded kinematic chains (`urdf-rs` + `k`), rapier2d articulated bodies, and a benchmarked PID/LQR baseline suite.

## 2. Storyline

**Act 1 — Opening the black box.** Hook: Ch 11's SAC policy commands Reacher torques at 50 Hz and tracks the cursor beautifully — then you change one number, the second link's mass, and the policy face-plants. The policy never knew there *was* a mass; it memorized a dynamics function it never saw. Act 1's move: print the actual `step()` source of `rl-envs::reacher` — inside is rapier integrating equations this chapter will now derive by hand. The see–think–act loop from Ch 1 returns with the "act" box exploded into kinematics → dynamics → actuation → sensing.

**Act 2 — The machinery of motion.** Geometry first: where is the fingertip, given the joints (FK — three lines of trigonometry)? Which joints put the fingertip *there* (IK — two solutions, elbow-up and elbow-down, and the reader drags both in the `ch13-fk-ik-sandbox`)? How do joint velocities map to fingertip velocities (Jacobian — and its collapse at singularities, the manipulability ellipsoid flattening to a needle)? Then force: the Lagrangian derivation of M, C, g for Reacher, every partial derivative shown, landing on the equation the whole robotics literature abbreviates as M q̈ + C q̇ + g = τ. Pendle reappears as the underactuated counterexample: fewer motors than degrees of freedom, no IK shortcut to swing-up.

**Act 3 — The incumbents RL must beat.** Classical control enters as the reigning champion, not a straw man: PID holds Reacher on a setpoint with zero learning; gravity compensation plus PD gives global setpoint convergence; LQR — derived via Ch 5's dynamic programming, now with continuous states — rejects disturbances on linearized Pendle optimally. The `ch13-lqr-vs-pid` dashboard lets the reader poke the pole and watch each controller fight back. Honest scoreboard: where classical wins (known model, regulation, guarantees), where it strains (contact, model error, task-level goals) — precisely the seams where Ch 14–17 will insert learning as policy, model, residual, or tuner. Closing beat: the encoders were noisy all along — the EKF sketch and Ch 4's POMDP honesty, now with hardware teeth.

## 3. Section-by-Section Design

### 13.1 Inside `env.step`: the Robot as a Dynamical System
- **F:** The env decomposed: q ∈ 𝒬 configuration, state x = (q, q̇); continuous dynamics ẋ = h(x, τ) discretized at Δt (Ch 2's Euler/RK4 returns with a purpose); actuation chain (commanded action → motor torque → joint torque: gears, limits, delays); observation o = sensor(x) + noise ≠ x — Kober §1.3's partial-observability point stated at the top, not as a footnote; degrees of freedom, actuation matrix B_a, underactuation defined (rank B_a < dim q): Reacher fully actuated, Pendle's cart-pole underactuated.
- **C:** Exploded see–think–act diagram: the Ch 1 "act" box unfolds into kinematics/dynamics/actuator/sensor stages with signals labeled by symbol (q, q̇, τ, o); hovering any stage highlights the section that derives it.
- **P:** `rl-sim` crate is born: `RobotModel` trait (fk, jacobian, mass_matrix, coriolis, gravity), `SensorModel` trait (noise, quantization, delay); `rl-envs::reacher` refactored to sit on `rl-sim` so the same model serves analysis and simulation.

### 13.2 Pose & Forward Kinematics: SE(2), SE(3), and Reacher's Fingertip
- **F:** Rotations SO(2)/SO(3), homogeneous transforms as SE(2)/SE(3) group elements, composition and inverse; SE(3) given as the general statement (rotation matrix + translation, the form `k` computes with), SE(2) worked in full; Reacher FK derived by composing two SE(2) transforms:
$$x_{ee} = l_1\cos q_1 + l_2\cos(q_1{+}q_2), \qquad y_{ee} = l_1\sin q_1 + l_2\sin(q_1{+}q_2),$$
workspace characterized as the annulus |l_1 − l_2| ≤ ‖p_{ee}‖ ≤ l_1 + l_2; link frames and the URDF description of exactly this chain (the file the reader ships in `rl-envs`).
- **C:** `ch13-fk-ik-sandbox` (FK mode): joint sliders q_1, q_2 drive the arm; live trace of the fingertip; workspace annulus rendered; frame axes toggleable on each link.
- **P:** `rl-sim::chain`: load `reacher.urdf` via `urdf-rs`, build a `k::SerialChain`, verify `k`'s FK against the hand-derived formula to 1e-12 (`proptest` over random q) — trust-but-verify as pedagogy.

### 13.3 Inverse Kinematics: Two Answers, Zero Answers
- **F:** Analytic IK for the 2-link arm derived via law of cosines:
$$\cos q_2 = \frac{x^2 + y^2 - l_1^2 - l_2^2}{2 l_1 l_2}, \qquad q_1 = \operatorname{atan2}(y, x) - \operatorname{atan2}(l_2 \sin q_2,\; l_1 + l_2\cos q_2),$$
with q_2 = ±arccos(·) giving elbow-up/elbow-down; solvability iff the target lies in the workspace annulus; boundary (one solution) and unreachable (none) cases; numerical IK for chains without closed forms: damped least squares Δq = Jᵀ(JJᵀ + λ²I)^{-1} Δp, convergence behavior near singularities; why RL policies sidestep IK entirely (they output torques or joint targets) yet IK remains the language of task-space goals and of Ch 17's action-space levels.
- **C:** `ch13-fk-ik-sandbox` (IK mode): drag the fingertip target; both elbow branches ghost-rendered; dragging out of the annulus grays the arm; a "numerical IK" toggle animates damped-least-squares iterations crawling to the target.
- **P:** `rl-sim::ik`: analytic 2-link solver returning `SmallVec<[Solution; 2]>` + damped-least-squares fallback on `k` chains; property test: fk(ik(p)) = p for all reachable p, both branches.

### 13.4 Differential Kinematics: Jacobians & Manipulability
- **F:** J(q) = ∂p_{ee}/∂q derived for Reacher:
$$J(q) = \begin{bmatrix} -l_1 s_1 - l_2 s_{12} & -l_2 s_{12} \\ l_1 c_1 + l_2 c_{12} & l_2 c_{12} \end{bmatrix}, \qquad \det J = l_1 l_2 \sin q_2,$$
singularities exactly at q_2 ∈ {0, π} (arm stretched/folded); velocity map ṗ = J q̇ and the dual force map τ = Jᵀ F (derived from virtual work — the bridge Ch 20's impedance action spaces will stand on); **manipulability ellipsoid** {J q̇ : ‖q̇‖ ≤ 1} from the SVD of J, Yoshikawa measure w = |det J| = l_1 l_2 |sin q_2|; conditioning of J as "how well can this pose move/push in each direction."
- **C:** `ch13-jacobian-ellipsoid`: the ellipsoid rides the fingertip as the reader drags through configurations, flattening to a needle at singularities; toggle velocity vs force ellipsoid (reciprocal axes — one interaction that teaches duality); det J heatmap over joint space.
- **P:** `rl-sim::jacobian`: analytic J + SVD via `nalgebra`; ellipsoid axes exported to the widget; numeric-vs-analytic Jacobian check (finite differences) as a doctest.

### 13.5 Lagrangian Dynamics: Deriving M, C, g for Reacher
- **F:** The chapter's centerpiece, every step shown. Kinetic energy of both links via link-COM velocities (r_i = COM distance, I_i = link inertia); T = ½ q̇ᵀ M(q) q̇ with
$$M(q) = \begin{bmatrix} \alpha + 2\beta \cos q_2 & \delta + \beta\cos q_2 \\ \delta + \beta \cos q_2 & \delta \end{bmatrix},\quad \begin{aligned}\alpha &= I_1{+}I_2{+}m_1 r_1^2{+}m_2(l_1^2{+}r_2^2)\\ \beta &= m_2 l_1 r_2, \qquad \delta = I_2 + m_2 r_2^2;\end{aligned}$$
potential V(q) = (m_1 r_1 + m_2 l_1) g_0 sin q_1 + m_2 r_2 g_0 sin(q_1+q_2); Euler–Lagrange d/dt(∂L/∂q̇) − ∂L/∂q = τ expanded term by term to
$$M(q)\ddot q + C(q,\dot q)\dot q + g(q) = \tau,\qquad C(q,\dot q) = \begin{bmatrix} -\beta \sin q_2\, \dot q_2 & -\beta \sin q_2 (\dot q_1{+}\dot q_2) \\ \beta \sin q_2\, \dot q_1 & 0\end{bmatrix},$$
with Christoffel-symbol construction shown; structural properties stated and proved for the 2-link case: M ≻ 0 symmetric, Ṁ − 2C skew-symmetric (energy bookkeeping), linearity in dynamic parameters (the property Ch 12's sysid teaser used and Ch 15 will exploit); Pendle's cart-pole Lagrangian done as the underactuated companion (τ enters only the cart coordinate).
- **C:** `ch13-dynamics-terms`: scrub a recorded Reacher trajectory and watch stacked bars decompose τ(t) into Mq̈, Cq̇, and g contributions; toggle each term off and the ghost arm integrates the crippled dynamics — "what if physics forgot Coriolis?" made visible.
- **P:** `rl-sim::dynamics`: `mass_matrix`, `coriolis`, `gravity` in `nalgebra` straight from the derivation; cross-validated against rapier2d rollouts (energy drift and trajectory agreement tests); `criterion` benchmark: analytic dynamics vs rapier step cost.

### 13.6 Classical Control: PID, Gravity Compensation & LQR
- **F:** PID τ = K_p e + K_i ∫e + K_d ė with gain-tuning behavior on the derived Reacher model (steady-state error under gravity motivates the integral term — and its windup pathology); **gravity-compensated PD**: τ = K_p e − K_d q̇ + g(q), global asymptotic stability for setpoint regulation via the Lyapunov function V = ½q̇ᵀMq̇ + ½eᵀK_p e (proof uses the skew-symmetry property from §13.5 — cited as the standard textbook result, algebra shown); linearization ẋ = Ax + Bu around an equilibrium, validity radius discussed concretely for Pendle-upright; **LQR derived as Ch 5 value iteration**: assume quadratic value V_k(x) = xᵀP_k x, one Bellman backup gives
$$P_{k} = Q + A^\top P_{k+1} A - A^\top P_{k+1} B\,(R + B^\top P_{k+1} B)^{-1} B^\top P_{k+1} A,\qquad u = -Kx,\; K = (R + B^\top P B)^{-1} B^\top P A,$$
the discrete Riccati recursion — completing-the-square algebra in full, convergence to the stationary P stated with conditions (stabilizability/detectability, cited); operational-space control sketched (task-space dynamics via J, the Jacobian-transpose bridge from §13.4); MPC preview: constrained LQR re-solved each step = Ch 12's receding horizon with an analytic model — the two chapters shake hands.
- **C:** `ch13-lqr-vs-pid` dashboard: Pendle balancing under PID, gravity-comp PD, and LQR side by side; reader fires impulse disturbances, tunes K_p/K_d and Q/R live, watches a running quadratic-cost meter declare the winner; a "leave validity region" button shoves the pole past linearization's radius and LQR falls over — honesty by demonstration.
- **P:** `rl-sim::control`: `Pid`, `GravityCompPd`, `dlqr` (code sketch in §5) in `nalgebra`; the **controller benchmark suite**: settling time, quadratic cost, disturbance-rejection score on Reacher setpoints and Pendle balance, pinned seeds, `criterion`-timed — the baseline table Ch 14–17 and Ch 18/20 must beat.

### 13.7 Sensing, the EKF & POMDP Honesty
- **F:** Sensor models: encoder quantization, velocity from finite differencing (noise amplification computed), IMU bias, latency; the **EKF sketch**: predict x̂⁻ = f(x̂, u), P⁻ = F P Fᵀ + Q_n; update K = P⁻Hᵀ(HP⁻Hᵀ + R_n)^{-1}, x̂ = x̂⁻ + K(z − h(x̂⁻)), with F, H the dynamics/measurement Jacobians (of §13.5's f — the derivation reused), stated as the workhorse approximation with its caveats (linearization bias, tuning Q_n/R_n); observability rank condition sketched for Pendle-with-encoder-only; the punchline tying to Ch 4: on hardware the agent sees o, not x — the POMDP was always the honest formalism (Kober §1.3's filters and information states), and recurrent policies (Ch 19) vs filter-then-policy are the two escape routes.
- **C:** `ch13-ekf-belief`: EKF belief ellipse tracking Pendle's true state through noise bursts and sensor dropout; reader cranks encoder noise and watches the ellipse breathe; overlay of raw finite-difference velocity vs EKF velocity estimate.
- **P:** `rl-sim::estimation`: EKF over the analytic Pendle model; experiment: Ch 11's SAC evaluated with true state vs EKF estimate vs raw noisy observations — the performance gap that motivates Ch 14's curses and Ch 19's recurrent policies.

### 13.8 Where Learning Enters
- **F:** The chapter's synthesis, as a formal taxonomy on the block diagram: learning as **policy** (replace the controller: Ch 10–11), as **model** (feed MPC/planning: Ch 12, Ch 15 sysid), as **residual** (τ = τ_classical + π_θ(s), stability intuition for small residuals — formalized in Ch 17), as **tuner** (learn K_p, K_d, Q, R: the Ch 3 bandit gag now respectable); decision criteria per cell: model quality, contact richness, task-level vs joint-level goals — mapped onto Tang's action-space levels (joint torque / position+PD / task-space), previewing Ch 17's empirical trade-offs.
- **C:** Interactive block-diagram switcher: the classical control loop with four glowing insertion points; clicking each shows the corresponding chapter's approach wired in and a one-line contract of what must hold for it to win.
- **P:** Residual-control teaser: SAC learns a residual on top of gravity-comp PD for Reacher tracking under a deliberately wrong mass parameter — beats both pure-PD and pure-SAC on sample count, seeding Ch 17's full treatment.

### 13.9 Chapter Bridge
- **F:** Recap: the env is now glass — geometry (FK/IK/J), force (M, C, g), incumbents (PID/LQR), sensing (EKF, POMDP). The baselines are benchmarked and waiting. Open threads: what breaks these clean derivations on real hardware? Contact, model error, resets, reward design — Ch 14's four curses, each now nameable in this chapter's vocabulary; Ch 15 builds the simulator around `rl-sim` and confronts sim-vs-real.
- **C:** Part III roadmap graphic: this chapter's block diagram annotated with which curse (Ch 14) attacks which block and which chapter (15–17) defends it.
- **P:** Pointer table: `rl-sim` chain/dynamics reused by every subsequent env; controller suite is the standing baseline in Ch 14's reward lab, Ch 17's residual RL, Ch 18's PD-based low-level control, Ch 20's impedance control.

## 4. Interactive Widgets & Dashboards

| ID | Type | What it shows | Reader interaction | Tech |
|---|---|---|---|---|
| `ch13-fk-ik-sandbox` | sandbox | Reacher joint space ↔ task space: FK sliders, IK dragging, both elbow branches, workspace annulus | Drag fingertip (IK) or joint sliders (FK); branch toggle; numerical-IK iteration animation; out-of-workspace feedback | WASM egui + `k` + nalgebra |
| `ch13-jacobian-ellipsoid` | animation | Manipulability ellipsoid riding the fingertip; collapse at q_2 ∈ {0, π}; velocity/force duality | Drag configurations; ellipsoid-type toggle; det J joint-space heatmap; singularity alarm | egui + nalgebra SVD |
| `ch13-dynamics-terms` | widget | τ(t) decomposed into Mq̈, Cq̇, g along a trajectory; ghost arm integrates with terms disabled | Scrub trajectory; toggle each term; speed slider (Coriolis grows quadratically — visible) | egui_plot + rapier2d |
| `ch13-lqr-vs-pid` | dashboard | Pendle disturbance rejection: PID vs gravity-comp PD vs LQR with live cost meter | Impulse slider/button; tune K_p, K_d, Q, R; "leave validity region" shove; cost scoreboard | WASM rapier2d + egui |
| `ch13-ekf-belief` | widget | EKF belief ellipse vs true Pendle state under noise and dropout | Noise sliders; dropout bursts; raw vs filtered velocity overlay | egui_plot + nalgebra |
| `ch13-learning-entry` | widget | Classical control block diagram with four learning insertion points (policy/model/residual/tuner) | Click insertion points; see wiring + contract; links to Ch 10–17 | egui + mermaid-derived SVG |

## 5. Rust Implementation Plan

Crates touched: **`rl-sim` created** (modules `chain`, `ik`, `jacobian`, `dynamics`, `control`, `estimation`, `sensors`), `rl-envs` (Reacher/Pendle refactored onto `rl-sim` models; URDF files become the single source of robot parameters), `rl-core` (baseline-comparison metrics), demos `demos/ch13-*`. This crate is Part III's foundation — Ch 15 adds randomization hooks and sysid to it.

Representative sketch — the analytic model and the LQR baseline:

```rust
// crates/rl-sim/src/dynamics.rs
impl RobotModel for ReacherModel {
    /// M(q) — derived in Sec 13.5; params (m_i, l_i, r_i, I_i) come from the URDF.
    fn mass_matrix(&self, q: &Vector2<f64>) -> Matrix2<f64> {
        let p = &self.params;
        let (a, b, d) = (p.alpha(), p.beta(), p.delta());
        let c2 = q[1].cos();
        Matrix2::new(a + 2.0 * b * c2, d + b * c2,
                     d + b * c2,       d)
    }
    fn coriolis(&self, q: &Vector2<f64>, dq: &Vector2<f64>) -> Matrix2<f64> {
        let bs2 = self.params.beta() * q[1].sin();
        Matrix2::new(-bs2 * dq[1], -bs2 * (dq[0] + dq[1]),
                      bs2 * dq[0], 0.0)
    }
}

// crates/rl-sim/src/control/lqr.rs
/// Discrete-time LQR via the backward Riccati recursion (Sec 13.6):
/// value iteration with quadratic value functions, run to convergence.
pub fn dlqr(a: &DMatrix<f64>, b: &DMatrix<f64>,
            q: &DMatrix<f64>, r: &DMatrix<f64>, tol: f64) -> DMatrix<f64> {
    let mut p = q.clone();
    loop {
        let gain_denom = (r + b.transpose() * &p * b)
            .try_inverse().expect("R + B^T P B must be invertible (R > 0)");
        let p_next = q + a.transpose() * &p * a
            - a.transpose() * &p * b * &gain_denom * b.transpose() * &p * a;
        if (&p_next - &p).abs().max() < tol { p = p_next; break; }
        p = p_next;
    }
    (r + b.transpose() * &p * b).try_inverse().unwrap()
        * b.transpose() * &p * a                            // K:  u = -K x
}
```

Experiments/benchmarks: FK/IK round-trip property tests (`proptest`); analytic-vs-rapier dynamics agreement + energy-drift test; the controller benchmark suite (settling time, quadratic cost, disturbance rejection on Reacher setpoint + Pendle balance; `criterion` per-step timing — PID vs LQR vs SAC inference); EKF-vs-raw-observation SAC evaluation; residual-control teaser run. Native: all of the above (`cargo run -p rl-sim --example baseline_bench`). In-browser: all six widgets (analytic models are tiny — full dynamics run in WASM at 60 fps).

## 6. Robot Thread

- **Reacher**: before — a task solved blind by SAC (Ch 11); after — fully understood: URDF-described, FK/IK solved analytically, Jacobian and manipulability mapped, M/C/g derived and cross-validated, PID and gravity-comp baselines benchmarked against the Ch 11 policy. Ready for Ch 14's reward-design lab, Ch 16's teleop demos (IK powers mouse-to-joint mapping), Ch 17's action-space levels, Ch 20's impedance control.
- **Pendle**: before — model-free and model-based solutions exist (Ch 10–12); after — its Lagrangian and linearization are derived, LQR balances it optimally near upright, and the EKF tracks it through noise; becomes Ch 15's integrator-stability and sysid specimen.
- **Rusty/Ferris**: name-checked in §13.2 (SE(2) is Rusty's native pose space; Ferris's floating base awaits SE(3) in Ch 18) — no new capability this chapter.

## 7. Exercises & Explorations

1. **(F)** Derive the FK and analytic IK for a 2-link arm whose second link is prismatic (extends rather than rotates). How many IK branches now?
2. **(F)** Verify by direct computation that Ṁ − 2C for Reacher (§13.5's matrices) is skew-symmetric, and show where the Christoffel construction guarantees this in general.
3. **(F)** Complete the completing-the-square step of the Riccati derivation for the scalar case (A, B, Q, R ∈ ℝ), and show the fixed-point equation reduces to a quadratic in P. Solve it for Pendle's linearized upright parameters.
4. **(F)** Compute the observability matrix for linearized Pendle with (a) angle-only and (b) cart-position-only measurement. Which is observable? Reconcile with what `ch13-ekf-belief` shows.
5. **(C)** In `ch13-jacobian-ellipsoid`, find the configuration maximizing Yoshikawa's w and compare with the analytic argmax q_2 = ±π/2. Then find a pose where the *force* ellipsoid is best aligned for pushing straight up — why is it near-singular?
6. **(C)** Using `ch13-lqr-vs-pid`, find the largest impulse each controller survives. Then re-tune Q/R to sacrifice cost for robustness and report the trade-off curve you traced.
7. **(P)** Add viscous joint friction b q̇ to `rl-sim::dynamics`, re-derive gravity-comp PD stability (does the Lyapunov argument still close?), and measure how far the un-updated LQR degrades before re-linearizing fixes it.
8. **(P)** Wire the analytic Reacher model into Ch 12's CEM-MPC in place of the learned ensemble, and race it against PETS-with-learned-model and Ch 11's SAC on tracking cost per real step — three paradigms, one scoreboard.

## 8. Notation Introduced

| Symbol | Meaning |
|---|---|
| q, q̇, q̈ ∈ ℝⁿ | joint configuration, velocity, acceleration (n = 2 for Reacher) |
| τ ∈ ℝⁿ | joint torques (actions at the lowest level) |
| M(q), C(q,q̇), g(q) | mass matrix, Coriolis/centripetal matrix, gravity vector; α, β, δ Reacher's inertia constants |
| SE(2), SE(3), SO(n) | rigid-body pose/rotation groups; **T** homogeneous transform |
| J(q) | end-effector Jacobian; w = \|det J\| Yoshikawa manipulability |
| l_i, r_i, m_i, I_i | link length, COM offset, mass, inertia; s_1, c_{12} = sin q_1, cos(q_1+q_2) |
| K_p, K_i, K_d | PID/PD gain matrices; e = q_des − q setpoint error |
| A, B, Q, R, P, K | linearization, LQR cost weights, Riccati solution, LQR gain (context separates P from the transition kernel of Ch 4) |
| x̂, P⁻, F, H, Q_n, R_n | EKF estimate, covariances, Jacobians, noise covariances |
| B_a | actuation matrix; rank B_a < dim q ⇔ underactuated |

## 9. References & Further Reading

- **Kober, Bagnell & Peters, IJRR 2013** — §1.3 (robotics as an RL domain: 10–30-D continuous actions considered large; true state never observable noise-free; filters and information states — the Kalman-filter table-tennis example — as the honest interface; costly, hard-to-reproduce roll-outs); §2 (RL formalism in robotics context: state as prediction-sufficient information, torque-level actions in the navigation example) — the survey text this chapter equips the reader to *verify* rather than trust.
- **Tang et al. 2024** — §3.2 problem-formulation axes (action-space levels: joint torque / position / task-space — the taxonomy §13.8 maps learning-entry points onto).
- **S&B** — ch. 4 (dynamic programming: §13.6 derives LQR as exact GPI on a quadratic value function — the book's continuous-state payoff of Ch 5).
- Spong, Hutchinson & Vidyasagar 2006, *Robot Modeling and Control* (FK/IK, Jacobians, Euler–Lagrange dynamics, skew-symmetry and passivity properties, PD+gravity-compensation stability — the standard results §13.2–13.6 restate with full algebra).
- Siciliano, Sciavicco, Villani & Oriolo 2009, *Robotics: Modelling, Planning and Control* (manipulability ellipsoids, operational-space/task-space control, damped-least-squares IK).
- Yoshikawa 1985, *Manipulability of Robotic Mechanisms* (the manipulability measure of §13.4).
- Anderson & Moore 1990, *Optimal Control: Linear Quadratic Methods* (LQR/Riccati theory, stabilizability/detectability conditions cited in §13.6).
- Kalman 1960, *A New Approach to Linear Filtering and Prediction Problems*; Thrun, Burgard & Fox 2005, *Probabilistic Robotics* (EKF and its caveats, §13.7).
- Khatib 1987, *A Unified Approach for Motion and Force Control of Robot Manipulators* (operational-space formulation, sketched in §13.6, revisited in Ch 20).

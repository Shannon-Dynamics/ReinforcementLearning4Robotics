# Chapter 16 — Demonstrations, Imitation & Offline RL

> **Part** III — The Robotics Side · **Builds on:** Ch 10 (policy gradients), Ch 11 (SAC & Reacher), Ch 13 (LQR/IK experts, `rl-sim`), Ch 14 (sample & goal-specification curses), Ch 15 (teacher–student distillation = DAgger, promised proof delivered here) · **Feeds:** Ch 17 (imitation-initialized policy search, ball-in-a-cup), Ch 20 (demo-boosted manipulation), Ch 21 (offline data for HRI; shared autonomy)
> **Modernizes:** Kober §5 (tractability through prior knowledge, esp. §5.1 demonstrations) and §3.4 (inverse RL); Akinola LfD slides (demonstration modes, BC pros/cons, covariate shift); Tang §3.3 (expert-usage axis) and §5 (real-world learning via demonstrations, offline-to-online); Ross et al. 2011 (DAgger), Ziebart et al. 2008 (MaxEnt IRL), Kumar et al. 2020 (CQL), Kostrikov et al. 2021 (IQL)

## 1. Purpose & Learning Outcomes

Ch 14 priced real-world samples; this chapter is about not paying full price. Data can come from a teacher's hands instead of the robot's exploration: demonstrations, and — the modern extension — whatever logs already exist. We build the full ladder: behavior cloning and its compounding-error theorem, DAgger's interactive fix, inverse RL's answer to the goal-specification curse, demo-accelerated RL, and offline RL with pessimism (CQL) and expectile regression (IQL), ending at the deployment recipe Tang's survey identifies as the trend: *pretrain offline, fine-tune online*.

The reader can:
- Compare the four demonstration modes (teleoperation, kinesthetic teaching, motion capture, video) by embodiment gap, cost, and signal quality, following Akinola and Kober §5.1.
- Prove the $O(T^2\varepsilon)$ compounding-error bound for behavior cloning and the $O(uT\varepsilon)$ bound for DAgger, and explain *why* interaction converts quadratic to linear.
- Derive MaxEnt IRL from first principles — maximum entropy subject to feature matching — including the partition function and its gradient, and connect it to GAIL in one step.
- State the offline-RL failure mode (OOD action overestimation) and prove CQL's value lower-bound property.
- Derive IQL's expectile regression and explain how it avoids querying out-of-distribution actions entirely.
- Record their own teleop dataset in the browser, train BC/DAgger/IQL on it in `burn`, and measure the offline-to-online fine-tuning jump with Ch 14's evaluation rubric.

## 2. Storyline

**Act I — Your hands, its arm.** The reader opens the browser teleop rig and *drives Reacher with the mouse* — the Ch 11 robot, now a marionette. Twenty demonstrations of a reach-and-hold task are recorded to a `serde` dataset in two minutes; behavior cloning trains in seconds and… works, briefly. Then the camera pans: started from a pose just outside the demo corridor, the clone drifts, and each drifted step feeds it a state no teacher ever labeled. Akinola's covariate-shift warning becomes a felt experience, and Kober's framing sharpens it: demonstrations remove *global* exploration (the Fosbury-Flop argument, Kakade–Langford made rigorous) — but supervised imitation alone can't even stay *local*.

**Act II — Theory of the drift, and two cures.** The compounding-error theorem is proved: $\varepsilon$ per step under the expert's distribution can cost $T^2\varepsilon$ in return. Cure one: interaction — DAgger queries the expert on the *learner's* states; the no-regret argument buys the linear bound (and retroactively justifies Ch 15's teacher–student distillation, as promised). Cure two: stop cloning actions and infer *intent* — max-margin feature matching, then MaxEnt IRL derived in full, with the partition function tamed by soft value iteration; GAIL sketched as its adversarial, sampled descendant. Interlude: demos as a *seasoning* for RL rather than a replacement — demos in the replay buffer, BC-regularized actors, Kober's imitation-init (the Ch 17 bridge).

**Act III — Learning from the logs.** The robot has been running for chapters; its replay buffers are sitting on disk. Offline RL asks: how much policy can you squeeze from logs *without touching the robot*? The failure is demonstrated live — naive SAC on a logged dataset hallucinates Q-values for actions nobody took — then repaired twice: CQL pushes down where the data isn't (lower bound proved), IQL never asks (expectile regression derived). Finale: IQL-pretrained Reacher fine-tunes online with SAC and the dashboard shows the modern deployment curve — offline floor, small dip, fast climb — measured in `SampleLedger` currency. The chapter hands Ch 17 a teleop pipeline and an imitation toolbox, and hands Ch 20 its data diet.

Robots: **Reacher** (teleop, BC/DAgger/IQL, fine-tuning), **Rusty** (covariate-shift drift-cone widget: lane-following clone), **Pendle** (cameo: logged swing-up data for the OOD demonstration).

## 3. Section-by-Section Design

### 16.1 Where Data Comes From: Demonstration Modes
- **F:** LfD formalized (Akinola): dataset $\mathcal D = \{(s_i,a_i)\}$ drawn from expert policy $\pi^\ast$'s occupancy $d_{\pi^\ast}$; goal: $\hat\pi = \arg\min_\pi \mathbb E_{d_{\pi^\ast}}[\ell(\pi(s), \pi^\ast(s))]$. The correspondence/embodiment problem defined (Kober §5.1: human joints ≠ robot joints; task-space demos as the fix). Mode taxonomy with formal signal models: teleoperation (on-robot states, teacher-side latency), kinesthetic teaching (on-robot states, gravity-compensated, no latency — but contaminated wrenches), motion capture (off-robot, retargeting map required), video (off-robot, correspondence *and* perception). Suboptimality and inconsistency of human data as label noise vs distribution bias — which one BC can average away and which it cannot.
- **C:** `ch16-demo-modes`: four cards, each with a 10-second capture clip (Kober's kinesthetic ball-in-a-cup teach-in among them), an embodiment-gap meter, cost/quality bars, and the formal signal model; a "which mode for this task?" mini-quiz wired to Kober's Table 4 examples.
- **P:** The browser teleop rig: `rl-envs::reacher::Teleop` — mouse target → (optionally) Ch 13 IK → joint commands at 20 Hz; `rl-core::demo::DemoSet` (`serde`/bincode: obs, action, timestamp, episode id, metadata incl. control mode); record/replay/inspect CLI. The reader's own 20-episode dataset becomes this chapter's shared artifact.

### 16.2 Behavior Cloning & the Compounding-Error Theorem
- **F:** BC as supervised learning: $\hat\pi = \arg\min_\pi \mathbb E_{(s,a)\sim\mathcal D}[\|\pi(s)-a\|^2]$ (Gaussian MLE view for continuous actions; why multimodal demos break unimodal heads — flagged for Ch 20's diffusion-policy sidebar). **Theorem (Ross & Bagnell 2010), proved:** episodic task, horizon $T$, cost in $[0,1]$; if $\hat\pi$ has 0–1 imitation error $\mathbb E_{s\sim d_{\pi^\ast}}[\hat\pi(s)\neq\pi^\ast(s)] \le \varepsilon$, then $J(\hat\pi) \le J(\pi^\ast) + T^2\varepsilon$. Proof as in the reduction literature: the probability the clone has left the expert distribution by time $t$ is at most $t\varepsilon$; once off-distribution, per-step cost is bounded only by 1; summing $\sum_{t=1}^{T} t\varepsilon \le T^2\varepsilon$ — each line displayed, plus the matching lower-bound example (a cliff-edge MDP) showing $T^2$ is tight. Discounted analogue stated ($\varepsilon/(1-\gamma)^2$).
- **C:** `ch16-drift-cone`: Rusty clones a demonstrated warehouse lane; an error-rate slider $\varepsilon$ injects per-step mistakes; the widget draws the growing off-distribution cone and a live regret-vs-$t$ curve hugging the $t^2$ envelope; a log-log inset lets the reader *measure* the exponent.
- **P:** `rl-deep::imitation::bc`: MLP Gaussian policy in `burn`; train on the §16.1 teleop set; evaluation protocol: start-state perturbation sweep (in-corridor → out) with `SuccessReport` CIs — the quantitative version of Act I's drift.

### 16.3 DAgger: Interaction Buys the Linear Bound
- **F:** DAgger algorithm (Ross, Gordon & Bagnell 2011): iterate — roll out mixture $\pi_i = \beta_i \pi^\ast + (1-\beta_i)\hat\pi_i$, label *every visited state* with the expert, aggregate $\mathcal D \leftarrow \mathcal D \cup \mathcal D_i$, retrain. **Theorem (proved via online learning):** casting each iteration as an online loss $\ell_i(\pi) = \mathbb E_{d_{\pi_i}}[\ell(\pi)]$, a no-regret learner (Follow-the-Leader on the aggregated set, regret $\tilde O(1/N)$) yields a policy with $J(\hat\pi) \le J(\pi^\ast) + u T \varepsilon_N + \tilde O(uT/N)$, where $\varepsilon_N$ is the training error on the *aggregated* distribution and $u$ bounds the one-step cost increase an expert can recover from (recoverability). The quadratic→linear mechanism named precisely: training distribution now matches test distribution. Cost analysis: expert labeling effort, when $\beta$-mixing matters, scripted experts vs humans. Back-reference: Ch 15's teacher–student objective *is* $\ell_i$ with KL loss — the promised guarantee, delivered.
- **C:** `ch16-drift-cone` (mode 2): toggle "DAgger rounds" — after each round the cone visibly collapses toward a linear tube; the regret inset re-fits the exponent ≈ 1; a counter shows cumulative expert labels as the price paid.
- **P:** `rl-deep::imitation::dagger` (sketch in §5): scripted expert = Ch 13's IK+PD controller standing in for the human; experiment: BC-with-$N$-demos vs DAgger-with-equal-label-budget on perturbed starts — the fair comparison that makes the theorem operational.

### 16.4 Inverse RL: Inferring the Reward
- **F:** The goal-specification curse (Ch 14) inverted: recover $R$ from behavior. Feature-based rewards $R_\theta = \theta^\top f(s)$; Abbeel–Ng feature matching: any $\pi$ with $\mathbb E_\pi[f] = \mathbb E_{\pi^\ast}[f]$ earns expert return under the true linear reward — derived. Ambiguity → **MaxEnt IRL (Ziebart 2008) derived in full:** maximize trajectory-distribution entropy s.t. feature matching; Lagrangian yields the exponential family $P(\tau|\theta) = \exp\!\big(\sum_t \theta^\top f(s_t)\big)/Z(\theta)$ with partition function $Z(\theta) = \sum_\tau \exp(R_\theta(\tau))$; log-likelihood gradient $\nabla_\theta = \tilde f - \mathbb E_{P(\tau|\theta)}[f] = \tilde f - \sum_s D_s f(s)$ with state visitations $D_s$ computed by the soft value-iteration forward–backward pass (algorithm given for the tabular case; each recursion displayed). Sampling-based escape from $Z$: guided cost learning in one paragraph; **GAIL sketch**: occupancy matching with a discriminator as learned cost — the two-line reduction from MaxEnt IRL, per the TOC's promised sketch depth.
- **C:** `ch16-irl-lab`: Rusty's warehouse with hidden reward; the reader plays teacher (drives a few demo paths), then watches MaxEnt IRL's inferred reward heatmap sharpen with each gradient step, next to the ambiguity set (all rewards consistent with feature matching) shrinking.
- **P:** `rl-tabular::maxent_irl`: exact soft-VI forward–backward on gridworlds; test: plant a reward, demo from its optimal policy, recover to rank correlation >0.95; a deliberately under-featured run shows reward mis-attribution (goal-specification curse, mirrored).

### 16.5 Demos + RL: Acceleration Recipes
- **F:** Three composable recipes with objectives written out: (1) demos in the replay buffer with prioritized sampling (DDPGfD/SACfD) — why off-policy algorithms can digest them and on-policy can't; (2) BC-regularized policy improvement, $\max_\pi \mathbb E[Q(s,\pi(s))] - \alpha\,\mathbb E_{\mathcal D}\|\pi(s)-a\|^2$ (TD3+BC's one-line change); (3) imitation initialization + local search — Kober §5.1's argument formalized via Kakade–Langford (knowing a good policy's state distribution converts RL to a tractable local problem), the exact recipe Ch 17's ball-in-a-cup executes with DMPs. When each recipe fails: bad demos poison (1) mildly, (2) proportionally to $\alpha$, (3) fatally (local search can't leave the demo's basin — the Fosbury-Flop caveat).
- **C:** `ch16-recipe-race`: dashboard racing SAC, SAC+demos-in-buffer, SAC+BC-regularizer, and BC-init on sparse-reward Reacher; a "demo quality" selector (expert / mediocre / 10 %-corrupted) reorders the finishing line and makes each failure mode visible.
- **P:** `rl-deep::sac` gains `demo_buffer: Option<DemoSet>` + BC-regularizer term; experiment grid {recipe × demo quality} on sparse Reacher, `SampleLedger` cost to 90 % success reported.

### 16.6 Offline RL: Distributional Shift & CQL's Pessimism
- **F:** Setting: fixed $\mathcal D$ from behavior policy $\pi_\beta$; no interaction. The failure derived: policy improvement queries $Q(s,a)$ at $a\sim\pi$, but the Bellman backup only constrains $Q$ on $\mathrm{supp}(\pi_\beta)$ — maximization turns extrapolation error into overestimation (the Ch 6 maximization-bias casino, now with function approximation and no corrective data). **CQL (Kumar et al. 2020):** objective $\min_Q \alpha\big(\mathbb E_{s\sim\mathcal D, a\sim\mu}[Q] - \mathbb E_{(s,a)\sim\mathcal D}[Q]\big) + \tfrac12\mathbb E_{\mathcal D}\big[(Q - \hat{\mathcal B}^\pi \bar Q)^2\big]$. **Lower-bound property proved:** setting the functional derivative to zero gives $\hat Q = \mathcal B^\pi \bar Q - \alpha\big[\tfrac{\mu(a|s)}{\pi_\beta(a|s)} - 1\big]$; taking expectation under $\mu=\pi$ shows $\hat V^\pi(s) \le V^\pi(s)$ for all $s$ for sufficiently large $\alpha$ (finite-sample correction term stated with citation). Practical form with the log-sum-exp $\mu$ (entropy-regularized adversary) written out.
- **C:** `ch16-cql-landscape`: live Q-surface over Reacher's 2-D torque space at a fixed state; dataset actions as dots; an $\alpha$ slider inflates the pessimism — OOD regions sink while the data ridge holds; a second panel shows the resulting policy's chosen action migrating back into support.
- **P:** Dataset factory: log SAC replay from Ch 11 runs into D4RL-style mixtures (expert / medium / replay / random) with `serde`; naive offline SAC vs CQL in `burn` on each mixture — the overestimation blow-up and its repair, in one table.

### 16.7 IQL: Never Ask About Actions You Didn't Take
- **F:** Expectile regression: for random variable $U$, the $\kappa$-expectile minimizes $L_2^\kappa(u) = |\kappa - \mathbf 1[u<0]|\,u^2$; derivation of the first-order condition and the limit $\kappa\to 1$ → supported maximum. **IQL (Kostrikov et al. 2021):** $V_\psi = \arg\min \mathbb E_{(s,a)\sim\mathcal D}\,L_2^\kappa\big(Q_{\bar\theta}(s,a) - V_\psi(s)\big)$ — an in-support soft-max over *dataset* actions; $Q_\theta \leftarrow r + \gamma V_\psi(s')$ (SARSA-like, no $\max_a$, no policy in the backup); policy extraction by advantage-weighted regression $\max_\phi \mathbb E_{\mathcal D}\big[e^{\beta(Q-V)}\log\pi_\phi(a|s)\big]$, derived as a KL-constrained policy improvement (two lines from Ch 10's trust-region machinery). Why this sidesteps OOD queries entirely, and its cost: performance capped by dataset support.
- **C:** `ch16-expectile-dial`: a distribution of $Q$-values at a state (histogram from the real dataset); drag $\kappa$ from 0.5 → 0.99 and watch $V$ slide from mean toward supported max, never beyond the data — IQL's one idea, made draggable.
- **P:** `rl-deep::offline::iql` in `burn` (twin Q, value net, AWR head); train on the §16.6 mixtures; report versus CQL: return, training stability, wall-clock — IQL's simplicity dividend measured.

### 16.8 Offline-to-Online: The Modern Deployment Recipe
- **F:** The Tang §5 trend formalized: pretrain on logs ($\pi_0$ from IQL/CQL), then fine-tune online. The characteristic dip explained: entering online RL, the critic meets its own optimism on fresh OOD states; mitigations with objectives — keep the pessimism term and anneal $\alpha$, advantage-weighted online updates (AWAC), balanced replay of offline+online data. Protocol definition: report offline floor, dip depth, samples-to-recovery, final return — all in `SampleLedger` units; contrast with from-scratch SAC and with pure offline.
- **C:** `ch16-offline-online-timeline`: dashboard of the full pipeline: dataset bar → offline training curve → deployment marker → online fine-tuning curve with the dip annotated; the reader drags the switchover point and the α-anneal schedule and replays from checkpoints; a distribution-shift meter ($\mathrm{KL}$ of current policy vs $\pi_\beta$) runs underneath.
- **P:** The chapter capstone: IQL-pretrained Reacher (medium mixture) fine-tuned with SAC (Ch 11 trainer, warm-started critics); comparison table {scratch SAC, BC, IQL frozen, IQL→SAC} × {samples to 90 %, resets, final}; expected headline: IQL→SAC reaches criterion in a small fraction of scratch's online samples — the number Ch 20 will lean on.

### 16.9 Chapter Bridge
- **F:** The data-source ladder assembled: demonstrate → clone → interactively correct → infer intent → season RL → learn from logs → fine-tune online; each rung mapped to its bound or objective, and to the curse (Ch 14) it mitigates.
- **C:** Static ladder diagram (mermaid), reused by Ch 20 when choosing manipulation's data diet.
- **P:** Handoff: `DemoSet` + teleop rig → Ch 17 (kinesthetic-style demo for ball-in-a-cup) and Ch 21 (shared autonomy); imitation/offline trainers → Ch 20. Next: *what* the policy outputs — representations (Ch 17).

## 4. Interactive Widgets & Dashboards

| ID | Type | What it shows | Reader interaction | Tech |
|---|---|---|---|---|
| `ch16-teleop-rig` | sandbox | browser teleoperation of Reacher; live recording of $(o,a)$ pairs | drive with mouse, record/replay/scrub episodes, export dataset | rapier2d + egui (WASM), serde download |
| `ch16-demo-modes` | gallery | four demonstration modes with signal models, embodiment-gap meters, cost/quality bars | flip cards, run mode-choice quiz on Kober's examples | egui |
| `ch16-drift-cone` | animation | clone drifting off Rusty's demonstrated lane; error cone $\propto t^2$; DAgger mode collapses it to linear | set $\varepsilon$, measure exponent on log-log inset, toggle DAgger rounds, watch label counter | egui_plot + rl-tabular (WASM) |
| `ch16-irl-lab` | widget | MaxEnt IRL recovering a hidden warehouse reward from reader demos; ambiguity set shrinking | drive demo paths, step gradient, edit feature set to break recovery | egui + rl-tabular (WASM) |
| `ch16-recipe-race` | dashboard | SAC vs three demo-acceleration recipes on sparse Reacher under varying demo quality | select demo quality, race curves, inspect failure modes | egui_plot + recorded runs |
| `ch16-cql-landscape` | widget | live Q-surface over action space; pessimism pushing down OOD regions as $\alpha$ grows | drag $\alpha$, pick state, watch policy action re-enter support | burn (WASM) + egui_plot |
| `ch16-expectile-dial` | widget | $V$ as $\kappa$-expectile of dataset $Q$-values sliding from mean to supported max | drag $\kappa$, resample states, compare to naive max | egui_plot |
| `ch16-offline-online-timeline` | dashboard | full offline→online pipeline: floor, dip, recovery, shift meter | drag switchover & α-anneal, replay checkpoints | egui_plot + recorded runs |

## 5. Rust Implementation Plan

**Crates touched:** `rl-core` (`demo::DemoSet`, dataset mixtures), `rl-envs` (`reacher::Teleop`, perturbed-start evaluation), `rl-tabular` (`maxent_irl` with soft value iteration), `rl-deep` (`imitation::{bc,dagger}`, `offline::{cql,iql}`, SAC demo-buffer + BC-regularizer, AWAC-style fine-tune path), `demos/ch16-*` (eight WASM crates).

**New modules/files:** `rl-core/src/demo.rs`, `rl-envs/src/reacher/teleop.rs`, `rl-tabular/src/maxent_irl.rs`, `rl-deep/src/imitation/{bc.rs,dagger.rs}`, `rl-deep/src/offline/{cql.rs,iql.rs,finetune.rs}`.

Representative sketch — the DAgger loop against a scripted expert (the theorem, operationalized):

```rust
/// rl-deep/src/imitation/dagger.rs — aggregate, label everything, retrain.
pub fn dagger<B: Backend>(
    env: &mut ReacherEnv, expert: &dyn Expert,     // Ch 13 IK+PD controller
    seed_demos: DemoSet, rounds: usize, eps_per_round: usize,
) -> (BcPolicy<B>, DemoSet) {
    let mut data = seed_demos;
    let mut policy = BcPolicy::train(&data, &TrainCfg::default());
    for i in 0..rounds {
        let beta = 0.5_f64.powi(i as i32);          // expert-mixing schedule β_i
        for _ in 0..eps_per_round {
            let mut obs = env.reset();
            while !env.done() {
                let a_expert = expert.act(env.state());     // label EVERY visited state
                data.push(obs.clone(), a_expert.clone());
                let a = if env.rng().gen_bool(beta) { a_expert }
                        else { policy.act(&obs) };           // roll out the mixture π_i
                obs = env.step(&a).obs;
            }
        }
        policy = BcPolicy::train(&data, &TrainCfg::default()); // FTL on the aggregate
    }
    (policy, data)
}
```

**Experiments/benchmarks:** (1) BC start-perturbation sweep; (2) BC vs DAgger at equal label budget; (3) MaxEnt IRL recovery + under-featured failure; (4) recipe race × demo quality; (5) offline mixtures: naive SAC vs CQL vs IQL; (6) offline→online capstone table. All seeded, `SuccessReport`+`SampleLedger` JSON feeding the widgets; `criterion` bench on DemoSet serialization (teleop must never drop frames).

**Native vs browser:** teleop rig, drift cone, IRL lab, expectile dial fully in-browser; BC training is small enough to run in-browser on the reader's own recorded set (burn WGPU/WebGPU) — the chapter's showpiece; CQL/IQL training native with recorded curves for dashboards.

## 6. Robot Thread

- **Reacher** (from Ch 11/15: SAC-trained, transfer-hardened) — becomes *teachable*: browser teleop, BC/DAgger clones, offline datasets logged from its own history, and an offline→online pipeline. After: Reacher has a data diet and a pretrain-then-finetune recipe; Ch 17 borrows its teleop for ball-in-a-cup demos; Ch 20 inherits the recipe.
- **Rusty** (from Ch 5/14 warehouse) — the covariate-shift demonstrator: lane-following clone drifts in the cone widget; also the IRL lab's stage. Conceptual duty only; its own navigation upgrade waits for Ch 19.
- **Pendle** — cameo: its logged swing-up replay is the smallest OOD-overestimation example (one-line dataset, vivid Q blow-up).

## 7. Exercises & Explorations

1. **(F)** Extend the $T^2\varepsilon$ proof to the discounted continuing setting and show the bound becomes $O(\varepsilon/(1-\gamma)^2)$; identify which step of the episodic proof the discount replaces.
2. **(F)** Prove the cliff-edge lower bound: construct an MDP + expert where any policy with imitation error $\varepsilon$ under $d_{\pi^\ast}$ suffers $J(\hat\pi) - J(\pi^\ast) = \Omega(T^2\varepsilon)$.
3. **(F)** Complete the MaxEnt derivation: from $\max_P H(P)$ s.t. $\mathbb E_P[f]=\tilde f$ and normalization, derive the exponential family, then show $\partial \log Z/\partial\theta = \mathbb E_{P_\theta}[f]$ and hence the feature-matching stationarity condition.
4. **(F)** Show the $\kappa=1/2$ expectile is the mean, and prove monotonicity of expectiles in $\kappa$; conclude why IQL's $V$ interpolates between policy evaluation and in-support optimality.
5. **(C)** In `ch16-drift-cone`, fit the drift exponent for $\beta$-schedules $\beta_i \in \{0, 0.5^i, 1/i\}$ and rank them against the DAgger paper's recommendation.
6. **(C)** Use `ch16-cql-landscape` to find the smallest $\alpha$ that keeps the extracted action in-support across ten random states, and compare it with the $\alpha$ the theory demands for a guaranteed lower bound (widget displays the finite-sample term).
7. **(P)** Record 20 *deliberately sloppy* teleop demos; compare BC, DAgger (scripted expert), and IQL-on-sloppy-logs; write three sentences on which rung of the ladder tolerated your noise and why.
8. **(P)** Implement TD3+BC (one regularizer line on the Ch 11 TD3 trainer) and add it to the §16.6 mixture table; verify Fujimoto & Gu's claim that it matches CQL on expert-heavy data at a fraction of the complexity.

## 8. Notation Introduced

| Symbol | Meaning |
|---|---|
| $\mathcal D$, $\pi_\beta$ | offline dataset; behavior policy that generated it |
| $d_\pi$ | state (occupancy) distribution of policy $\pi$ (from Ch 10, now load-bearing) |
| $\varepsilon$, $u$ | per-step imitation error; recoverability constant in the DAgger bound |
| $\beta_i$ | DAgger expert-mixing coefficient at round $i$ |
| $f(s)$, $\tilde f$ | reward features; empirical expert feature expectation |
| $Z(\theta)$, $D_s$ | MaxEnt IRL partition function; expected state visitation counts |
| $\hat{\mathcal B}^\pi$ | empirical Bellman evaluation operator (offline setting) |
| $\mu(a|s)$ | CQL's adversarial action-proposal distribution |
| $\kappa$ | expectile level (literature writes $\tau$; reserved here for torque/DMP scaling — deviation registered for Appendix C) |
| $\beta$ (AWR) | inverse temperature of advantage-weighted regression (context-separated from $\beta_i$) |

## 9. References & Further Reading

- **Kober, Bagnell & Peters (IJRR 2013)** — §5.1 (demonstrations: kinesthetic vs mocap, hand-crafted policies, Kakade–Langford tractability argument, Fosbury Flop), §3.4 (inverse optimal control lineage: Abbeel–Ng, Ratliff, Ziebart), §7.3 (ball-in-a-cup teach-in — next chapter's seed).
- **Akinola (Columbia slides)** — LfD definition, demonstration modes, BC pros/cons (covariate shift, demo volume, suboptimal teachers), LfD/RL/hybrid triangle.
- **Tang et al. (2024)** — §3.3 (expert-usage axis), §5 (demonstrations for real-world learning; offline RL and fine-tuning as the emerging recipe; distributional-shift caveats).
- **Sutton & Barto** — §6.7 (maximization bias, the offline blow-up's tabular ancestor), §13 (policy gradient, AWR's trust-region root).
- Pomerleau (1989) — ALVINN. Argall et al. (2009) — LfD survey. Chernova & Thomaz (2014) — robot learning from human teachers.
- Ross & Bagnell (2010) — efficient reductions for imitation. Ross, Gordon & Bagnell (2011) — DAgger. Abbeel & Ng (2004); Ratliff et al. (2006) — apprenticeship & max-margin. Ziebart et al. (2008) — MaxEnt IRL. Finn et al. (2016) — guided cost learning. Ho & Ermon (2016) — GAIL.
- Vecerik et al. (2017) — DDPGfD. Rajeswaran et al. (2018) — DAPG. Fujimoto & Gu (2021) — TD3+BC.
- Levine et al. (2020) — offline RL tutorial. Kumar et al. (2020) — CQL. Kostrikov, Nair & Levine (2021) — IQL. Nair et al. (2020) — AWAC. Chi et al. (2023) — diffusion policy (multimodality sidebar, expanded in Ch 20).

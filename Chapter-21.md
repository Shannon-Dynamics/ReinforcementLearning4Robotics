# Chapter 21 — Frontiers: HRI, Multi-Robot & Foundation Models

> **Part** V — Frontiers & Capstone · **Builds on:** Ch 4, 10, 11, 14, 16, 19, 20 · **Feeds:** Ch 22
> **Modernizes:** Tang §4.5 (HRI), §4.6 (multi-robot), §5 (general trends & open challenges), §6 (conclusion); Kober §8.1–8.3 (open questions, practical challenges, lessons) — diffed against the 2024 record.

## 1. Purpose & Learning Outcomes

The book's last survey chapter covers the competencies where deep RL has succeeded *least* — humans in the loop and robots in teams — and the force most likely to change that: foundation models. The formal core is heavier than the empirical record: Dec-POMDPs and their complexity wall, CTDE policy gradients, and the KL-regularized RLHF objective. The chapter ends with the book's scholarly capstone: an item-by-item diff of Kober's 2013 open questions against Tang's 2024 open challenges — what a decade solved, moved, or left standing — and a research compass for the reader.

The reader can:
- Explain why HRI is simulation-starved (the human is the unmodelable part) and enumerate the three data strategies the literature actually uses.
- Formalize shared autonomy as a latent-goal POMDP and implement both Q-filter arbitration and linear policy blending.
- Write the Dec-POMDP tuple, state its NEXP-completeness, and explain what that implies for practical MARL.
- Derive the CTDE policy gradient with a centralized critic, explain the lazy-agent/credit-assignment failure, and sketch the counterfactual-baseline fix.
- Derive the KL-regularized RLHF objective and its closed-form optimal policy, and map VLA architectures onto the book's policy taxonomy.
- Defend, with citations, a verdict of solved / partially solved / open for each of Kober's 2013 questions.

## 2. Storyline

**Act I — Where the recipe stops working.** Hook: the zero-shot sim-to-real recipe that carried Ch 18–20 requires something HRI cannot provide — a simulator of the other party. Humans are non-Markovian, boundedly rational, and expensive to sample (Tang §4.5.4); multi-robot systems break the recipe differently, by making the environment nonstationary from every agent's perspective. The chapter frames both as *the same failure*: the world stops being a fixed MDP the moment it contains another learner.

**Act II — Two formalisms and their workarounds.** HRI first: the three data strategies (real-world RL, learned human models, hardcoded surrogates) as choices along a fidelity–cost frontier, with shared autonomy as the deployable bright spot — the reader flies a Reacher teleop task with a trained assistant blending in, feeling $\alpha$ move between autonomy and obedience. Then multi-robot: the Dec-POMDP wall (NEXP-complete), and CTDE as the engineering escape — two Rustys learn a cooperative delivery in a shared gridworld, and the lazy-agent pathology appears on schedule before a counterfactual baseline fixes it.

**Act III — The diff, and what comes next.** Foundation models enter as reward writers, goal translators, and policy priors, with the RLHF objective derived and a candle-based VLM reward-labeling experiment run honestly (it half-works, and the failure modes are instructive). Then the ledger: Kober 2013's questions vs Tang 2024's challenges, rendered as the diff dashboard — representation choice (moved, not solved), reward from data (transformed by RLHF/VLMs, open), model errors (largely tamed by randomization at scale), evaluation (still open; the book's own L0–L5 discipline as partial answer). The compass hands the reader their research direction; Ch 22 hands them the engineering.

## 3. Section-by-Section Design

### 21.1 The Frontier Map: What 2013 Asked, What 2024 Answered
- **F:** Framing formalism: the fixed-MDP assumption $P(s'\mid s,a)$ audited against each frontier — humans (unmodeled agent in the loop), teams (joint action space, nonstationarity), foundation models (reward/policy priors from outside the MDP). Kober §8's question list and Tang §5's challenge list stated precisely, as the chapter's rubric.
- **C:** `ch21-frontier-map` — visualization: the book's competency map (Ch 1 widget reprised) with L-level shading; HRI and multi-robot regions visibly sparse; clicking a region jumps to its section.
- **P:** None (survey section); the chapter's experiment configs are registered here for reproducibility.

### 21.2 HRI: the Simulation-Starved Competency
- **F:** The human-in-the-loop POMDP: human policy $\pi_H$ as latent, nonstationary, non-Markovian; why zero-shot sim-to-real needs $\hat\pi_H \approx \pi_H$ and why that is the hard part. The three data strategies formalized as simulator choices (Tang §4.5.4): (1) direct real-world RL — sample complexity vs human cost; (2) learned human models from motion capture/crowd-sourcing/RL self-play; (3) hardcoded surrogates for simple behaviors — with collaborative pHRI (handover), non-collaborative pHRI (social navigation, closing Ch 19's deferred thread), and offline RL (Ch 16 payoff: pre-collected interaction data, pessimism where humans cannot be re-queried) placed on this map.
- **C:** `ch21-human-model-fidelity` — widget: a social-navigation scene where the reader swaps the human model (constant-velocity, ORCA-style, recorded traces, learned policy) and watches the same trained robot policy's behavior and failure rate shift — the sim-to-real gap made visible with the *human* as the gap.
- **P:** `rl-envs::social_rusty`: Rusty navigating among scripted/recorded pedestrian agents in `rapier2d`; policy trained per human model; cross-evaluation matrix (train-model × test-model) quantifying the human-model gap.

### 21.3 Shared Autonomy: RL That Keeps the Human in Charge
- **F:** Formalization: user input $u_t$ as an observation of a latent goal $g$; belief over goals $b_t(g)$ (Ch 19.1 machinery reused on a discrete goal set); assistance as acting under goal uncertainty. Two arbitration schemes derived: (1) Q-filter (Reddy et al.): feasible set $\mathcal A_\delta = \{a : Q(s,a) \ge (1-\delta)\max_{a'}Q(s,a')\}$, execute $\arg\min_{a\in\mathcal A_\delta}\lVert a - u_t\rVert$ — optimality-constrained obedience; (2) linear blending $a = (1-\alpha)u_t + \alpha\, a_\pi$ (Dragan–Srinivasa arbitration) with $\alpha$ scheduled by goal-belief confidence. Guarantee discussion: what the $\delta$ knob does and does not promise.
- **C:** `ch21-blend-slider` — live WASM widget: the reader teleoperates Reacher toward one of three targets while a trained SAC assistant blends in; sliders for $\alpha$ and $\delta$; a belief bar shows $b_t(g)$ sharpening; deliberately ambiguous target layouts let the reader feel wrong-goal assistance.
- **P:** `rl-deep::shared`: SAC goal-conditioned assistant (Ch 11 reuse) + Q-filter and blending arbiters (code sketch in §5); pilot metrics logged (time-to-target, input effort, override rate) — the chapter's quantitative comparison of $\alpha$ policies.

### 21.4 Multi-Robot Systems: the Dec-POMDP Wall
- **F:** Dec-POMDP tuple $\langle \mathcal I, \mathcal S, \{\mathcal A^i\}, P, R, \{\Omega^i\}, \{O^i\}, \gamma\rangle$ defined; joint policies, local histories; **complexity theorem** (Bernstein et al. 2002): finite-horizon Dec-POMDP with $\ge 2$ agents is NEXP-complete — stated precisely with the POMDP (PSPACE) contrast, proof located, implications drawn: no poly-time general algorithm plausible, hence structure exploitation (cooperation, factored rewards, communication) is not optional. MARL nonstationarity: each agent's environment contains the others' changing policies, breaking single-agent convergence arguments (Ch 6's conditions revisited). Survey anchors: collision avoidance as Dec-MDP, loco-manipulation coordination, robot soccer at L4 (Tang §4.6).
- **C:** `ch21-lazy-agent` — animation: two Rustys pushing a shared block toward a slot; with a shared team reward one robot learns to idle (credit misassignment shown as per-agent advantage traces); switching on the counterfactual baseline revives it — the Dec-POMDP coordination failure and its fix in one loop.
- **P:** `rl-envs::warehouse_duo`: two-Rusty cooperative delivery gridworld (item handoff required); logging per-agent contribution metrics that the widget replays.

### 21.5 CTDE and MAPPO: Making MARL Trainable
- **F:** Centralized training, decentralized execution: critic sees global state (and teammates' actions), actors see local histories — the asymmetric actor–critic of Ch 15 reborn as a multi-agent principle. CTDE policy gradient derived: $\nabla_{\theta_i} J = \mathbb E[\nabla_{\theta_i}\log\pi^i(a^i\mid h^i)\, \hat A(s, \mathbf a)]$ with shared centralized advantage; the credit-assignment defect (every agent bathed in the same $\hat A$) formalized, and the counterfactual baseline $A^i = Q(s,\mathbf a) - \sum_{a'}\pi^i(a'\mid h^i)\, Q(s,(\mathbf a^{-i}, a'))$ (COMA) derived as its fix; MAPPO as clipped-surrogate CTDE with the Ch 10 implementation details that still matter (advantage normalization per agent, shared vs separate parameters).
- **C:** `ch21-ctde-dashboard` — dashboard: live MAPPO training on the warehouse duo; per-agent advantage histograms, policy-divergence trace between the two Rustys' policies (parameter sharing toggle), success/handoff metrics — the Ch 10 dashboard vocabulary extended to teams.
- **P:** `rl-deep::marl`: MAPPO in `burn` (Ch 10's PPO generalized: vectorized over agents, centralized critic); shared-vs-separate parameter experiment; the lazy-agent → counterfactual-baseline reproduction.

### 21.6 Foundation Models Meet Robot RL
- **F:** Three interfaces formalized. (1) *Reward writing*: RLHF pipeline — Bradley–Terry preference model $P(\sigma^1 \succ \sigma^2) = \exp(\sum r_\phi)/(\exp(\sum r_\phi)+\exp(\sum r_\phi))$ fit from comparisons; KL-regularized fine-tuning $\max_\pi \mathbb E_\pi[r_\phi] - \beta\, D_{\mathrm{KL}}(\pi \Vert \pi_{\text{ref}})$ with the closed form $\pi^\*(a\mid s) \propto \pi_{\text{ref}}(a\mid s)\exp(r_\phi(s,a)/\beta)$ derived in three lines (Ch 14's reward-design curse meeting Ch 16's offline machinery); LLM-written dense rewards and randomization configs (DrEureka) as reward search. (2) *Goal translation*: language-conditioned policies as goal-conditioned RL with a pretrained goal encoder. (3) *Policy priors*: VLA models as $\pi_{\text{ref}}$; RL fine-tuning of large policies and why PPO-style KL anchoring reappears. Sober assessment per Tang §5: opportunities, not yet L4 evidence.
- **C:** `ch21-vla-explorer` — widget: an interactive block diagram of a VLA (vision encoder → LLM backbone → action head/decoder); the reader routes the book's tasks through it, sees where RL enters (reward model, fine-tuning loss, action-head distillation), and compares parameter/latency budgets against the book's small policies.
- **P:** `rl-deep::vlm_reward`: candle-based small-VLM feasibility sketch — caption-and-score frames from Ch 20's push task as a reward labeler; agreement analysis against the true reward and against reader labels; documented failure cases (viewpoint sensitivity, object hallucination). Explicitly scoped as an experiment, not a dependency.

### 21.7 The Open-Problem Ledger: Kober §8 vs Tang §5
- **F:** The diff itself, item by item, each with verdict and evidence: representations (open → *moved*: networks learn features, but action-space choice is still expert-picked — Tang §5 echoes Kober §8.1 almost verbatim); reward from data (open → *transformed*: IRL begat RLHF/VLM labeling; general case open); prior knowledge (open → *rescaled*: demos at scale, foundation priors); perception integration (open → *largely closed at low L-levels*: end-to-end visuomotor exists; robustness open); parameter sensitivity (open → *partially closed*: PPO's robustness, but Tang's "principled design" challenge is its ghost); model errors/under-modeling (open → *substantially closed for locomotion*: randomization + parallel sim + residual real-data models; open for contact-rich); datasets (Kober §8.2 → *transformed*: OXE-scale corpora exist, exploitation open); evaluation (Kober §8.2 → *open*: Tang's benchmarking challenge; L0–L5 as the survey's own partial remedy). New-since-2013 items listed: safe real-world learning at scale, long-horizon skill discovery, MARL stability.
- **C:** `ch21-problem-diff` — the chapter's signature dashboard: two columns (Kober 2013, Tang 2024) with linked items colored solved/partial/open; clicking an item expands the evidence trail (papers with years and L-levels) and the book chapter where the reader met the machinery.
- **P:** The ledger as data: `book-data/open_problems.toml` (`serde`-loaded), so the dashboard, the prose, and future errata stay in sync — the book's own consistency discipline applied to its scholarship.

### 21.8 Chapter Bridge
- Recap: frontiers fail the fixed-MDP audit in three different ways, and each has a formalism (latent-goal POMDP, Dec-POMDP, KL-anchored fine-tuning) the reader now owns. The research compass points outward; Ch 22 points inward: everything mature enough to trust — PPO, randomization, teacher–student, reward rubric, evaluation statistics — gets assembled into one reproducible robot, end to end.

## 4. Interactive Widgets & Dashboards

| ID | Type | What it shows | Reader interaction | Tech |
|---|---|---|---|---|
| `ch21-frontier-map` | visualization | Competency map with L-level shading; sparse frontier regions | Click regions to navigate; filter by year | egui, WASM |
| `ch21-human-model-fidelity` | widget | Same robot policy under swapped human models; failure-rate shifts | Choose human model; run episodes; view cross-evaluation matrix | egui + rapier2d, WASM |
| `ch21-blend-slider` | widget | Shared-autonomy Reacher: teleop + SAC assistant | Drive with mouse; tune $\alpha, \delta$; watch goal belief sharpen | egui + rapier2d + burn, WASM |
| `ch21-lazy-agent` | animation | Cooperative push with credit misassignment; counterfactual fix | Toggle baseline; inspect per-agent advantage traces | egui_plot + logged rollouts |
| `ch21-ctde-dashboard` | dashboard | Live MAPPO on the two-Rusty delivery task | Toggle parameter sharing; watch per-agent metrics | egui_plot |
| `ch21-vla-explorer` | widget | VLA block diagram; where RL enters; budget comparison | Route tasks through; expand blocks; compare to book policies | egui/SVG, WASM |
| `ch21-problem-diff` | dashboard | Kober 2013 vs Tang 2024 open problems, linked and verdict-colored | Click items for evidence trails and chapter back-links | egui + serde data, WASM |

## 5. Rust Implementation Plan

**Crates touched:** `rl-envs` (`social_rusty`, `warehouse_duo`), `rl-deep` (`marl`, `shared`, `vlm_reward`), `rl-core` (multi-agent `Env` extension: per-agent observation/action maps), demos `ch21-*`; `candle` appears here only, as an isolated experiment dependency.

**Modules/files:** `rl-envs/src/social_rusty.rs`, `rl-envs/src/warehouse_duo.rs`, `rl-deep/src/marl.rs`, `rl-deep/src/shared.rs`, `rl-deep/src/vlm_reward.rs`, `book-data/open_problems.toml`.

Representative sketch — shared-autonomy arbitration (`rl-deep/src/shared.rs`):

```rust
/// Q-filter + blending arbitration (Sec. 21.3).
/// Candidates come from the SAC actor plus perturbations of the user input.
pub fn assist(
    q: &QNet,
    s: &Obs,
    user: &ActionVec,
    candidates: &[ActionVec],
    delta: f64, // optimality tolerance: keep a with Q >= (1-delta) * Q*
    alpha: f64, // blending: 0 = pure teleop, 1 = full autonomy
) -> ActionVec {
    let q_star = candidates
        .iter()
        .map(|a| q.value(s, a))
        .fold(f64::NEG_INFINITY, f64::max);
    let a_robot = candidates
        .iter()
        .filter(|a| q.value(s, a) >= (1.0 - delta) * q_star)
        .min_by(|a, b| dist(a, user).total_cmp(&dist(b, user)))
        .cloned()
        .unwrap_or_else(|| user.clone()); // empty filter: obey the human
    user.lerp(&a_robot, alpha)
}
```

**Experiments/benchmarks:** (1) human-model cross-evaluation matrix (train × test model); (2) shared-autonomy user-metric study over $\alpha$ grid (time-to-target, input effort, override rate); (3) lazy-agent reproduction and counterfactual-baseline fix; (4) MAPPO shared-vs-separate parameters; (5) VLM-reward agreement rates with confusion analysis; all success claims through Ch 20's `rl-core::eval` harness.

**Native vs browser:** `ch21-blend-slider` and `ch21-human-model-fidelity` fully in-browser (the human input *is* the experiment); MAPPO training native with the dashboard attachable; the candle VLM experiment native-only (model size), with logged outputs shipped to the browser dashboard. Static fallbacks per style guide.

## 6. Robot Thread

- **Rusty** (multiplied): two Rustys learn cooperative delivery — the gridworld of Ch 4 returns as a Dec-POMDP; Rusty also navigates among humans in `social_rusty`, closing Ch 19's deferred social-navigation thread.
- **Reacher**: becomes the shared-autonomy platform — the reader's hand and the Ch 11 SAC policy share one arm.
- **Ferris**: rests (cited in loco-manipulation and soccer discussions); returns as the star of Ch 22.
- **Pendle**: retired from active duty; acknowledged in the compass as where the reader's control intuition began.

## 7. Exercises & Explorations

1. **(F)** Write the two-Rusty delivery task first as a full Dec-POMDP, then find the assumptions (shared reward, full communication) under which it collapses to a single-agent MDP over the joint space; measure the joint action-space growth for $n$ robots.
2. **(F)** Derive the closed-form RLHF policy $\pi^\* \propto \pi_{\text{ref}}\exp(r_\phi/\beta)$ from the KL-regularized objective, and show what happens as $\beta \to 0$ and $\beta \to \infty$; relate both limits to reward hacking (Ch 14).
3. **(F)** Show that the COMA counterfactual baseline leaves the CTDE policy gradient unbiased (the baseline argument of Ch 10, redone per-agent).
4. **(C)** In `ch21-blend-slider`, find an $\alpha$ where assistance *hurts* on ambiguous target layouts; explain via the goal belief $b_t(g)$ and propose (then test) a confidence-scheduled $\alpha(b_t)$.
5. **(C)** Using `ch21-problem-diff`, pick the verdict you most disagree with and write a one-page rebuttal citing at least three papers from the evidence trails.
6. **(P)** Add a third Rusty to `warehouse_duo` and measure MAPPO's degradation with team size; relate to the Dec-POMDP complexity discussion and Tang §4.6.4's scalability challenge.
7. **(P)** Replace the hand-designed push-task reward with your best VLM-labeled reward from `vlm_reward` and train; document the reward-hacking behaviors that emerge with a Ch 14 reward-anatomy autopsy.
8. **(P)** Implement Boltzmann-rational human simulation ($\pi_H \propto \exp(Q_H/\tau_H)$) for `social_rusty`; find the $\tau_H$ at which policies trained on it transfer best to recorded traces.

## 8. Notation Introduced

| Symbol | Meaning |
|---|---|
| $\mathcal I,\ \mathbf a = (a^1,\dots,a^n)$ | agent set; joint action |
| $\Omega^i, O^i, h^i$ | agent $i$'s observation set, observation function, local history |
| $\pi_H, \hat\pi_H$ | human policy and its model/surrogate |
| $u_t,\ b_t(g),\ \alpha,\ \delta$ | user input; goal belief; blending weight; Q-filter tolerance |
| $\hat A(s,\mathbf a),\ A^i$ | centralized advantage; counterfactual per-agent advantage |
| $r_\phi,\ \sigma^1 \succ \sigma^2$ | learned reward model; trajectory preference |
| $\pi_{\text{ref}},\ \beta$ | reference policy; KL-regularization coefficient (*clash note:* duty factor $\beta$ (Ch 18) and option termination $\beta_\omega$ (Ch 17) remain distinct by context) |

## 9. References & Further Reading

- **Baseline:** Tang §4.5.1–4.5.4, §4.6.1–4.6.4, §5, §6; Kober §8.1 (open questions — the diff's left column), §8.2 (practical challenges), §8.3 (lessons for RL).
- Bernstein, Givan, Immerman & Zilberstein 2002, Dec-POMDP NEXP-completeness (proof source for 21.4).
- Yu et al. 2022, *The surprising effectiveness of PPO in cooperative multi-agent games* (MAPPO); Foerster et al. 2018, *COMA* (counterfactual baseline); Lowe et al. 2017, *MADDPG* (CTDE lineage).
- Chen et al. 2017 / Everett et al. 2018, decentralized social/multi-agent collision avoidance; Long et al. 2018 (lidar-based, curriculum); Sartoretti et al. 2019, *PRIMAL* (blocking penalty).
- Nachum et al. 2019, multi-agent manipulation via locomotion; Haarnoja et al. 2024, learning agile soccer skills (1v1, self-play; Tang's robot-soccer L4 anchor).
- Reddy, Dragan & Levine 2018, shared autonomy via deep RL (Q-filter); Dragan & Srinivasa 2013, policy-blending formalism; Schaff & Walter 2020, residual shared autonomy; Nair et al. 2021, language-conditioned offline RL (LOReL).
- Christen et al. 2023, learning human-to-robot handovers; Hirose et al. 2023, *SACSoN* (real-world social navigation with residual Q).
- Christiano et al. 2017, deep RL from human preferences; Ouyang et al. 2022, InstructGPT (the KL-anchored pipeline of 21.6).
- Brohan et al. 2023, *RT-2*; Kim et al. 2024, *OpenVLA*; Ma et al. 2024, *DrEureka* (LLM-designed rewards and randomization); Open X-Embodiment collaboration 2024 (dataset scale).

#!/usr/bin/env python3
"""
mdx2tex.py — convert the book's MDX chapters into LaTeX for the print edition.

The web edition is the source of truth for content; this script projects it onto
the Shannon Robotics LaTeX design system. It handles:

  · YAML frontmatter (title, chapter number, epigraph)
  · every custom MDX component used by the book
  · GitHub-flavoured markdown: headings, lists, tables, emphasis, links, rules
  · math, which passes through untouched because it is already LaTeX
  · Rust code fences, routed into the shrust environment

Interactive simulations cannot exist on paper, so each becomes a described
figure carrying its widget id — the static fallback the style guide requires.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# A tolerant parser for the JS object/array literals used in JSX props.
# ---------------------------------------------------------------------------


class JsLiteral:
    """Parses the subset of JS literal syntax that appears in this book's MDX."""

    def __init__(self, text: str):
        self.s = text
        self.i = 0

    def parse(self):
        self._ws()
        return self._value()

    def _ws(self):
        while self.i < len(self.s):
            c = self.s[self.i]
            if c in " \t\r\n":
                self.i += 1
            elif self.s.startswith("//", self.i):
                nl = self.s.find("\n", self.i)
                self.i = len(self.s) if nl < 0 else nl
            else:
                break

    def _value(self):
        self._ws()
        if self.i >= len(self.s):
            return None
        c = self.s[self.i]
        if c == "[":
            return self._array()
        if c == "{":
            return self._object()
        if c in "\"'`":
            return self._string(c)
        return self._bare()

    def _array(self):
        out = []
        self.i += 1  # [
        while True:
            self._ws()
            if self.i >= len(self.s):
                break
            if self.s[self.i] == "]":
                self.i += 1
                break
            out.append(self._value())
            self._ws()
            if self.i < len(self.s) and self.s[self.i] == ",":
                self.i += 1
        return out

    def _object(self):
        out = {}
        self.i += 1  # {
        while True:
            self._ws()
            if self.i >= len(self.s):
                break
            if self.s[self.i] == "}":
                self.i += 1
                break
            # key
            if self.s[self.i] in "\"'":
                key = self._string(self.s[self.i])
            else:
                m = re.match(r"[A-Za-z_$][\w$]*", self.s[self.i :])
                if not m:
                    self.i += 1
                    continue
                key = m.group(0)
                self.i += len(key)
            self._ws()
            if self.i < len(self.s) and self.s[self.i] == ":":
                self.i += 1
            out[key] = self._value()
            self._ws()
            if self.i < len(self.s) and self.s[self.i] == ",":
                self.i += 1
        return out

    def _string(self, quote: str) -> str:
        self.i += 1
        buf = []
        while self.i < len(self.s):
            c = self.s[self.i]
            if c == "\\":
                nxt = self.s[self.i + 1] if self.i + 1 < len(self.s) else ""
                buf.append({"n": "\n", "t": "\t", "r": "\r"}.get(nxt, nxt))
                self.i += 2
                continue
            if c == quote:
                self.i += 1
                break
            buf.append(c)
            self.i += 1
        return "".join(buf)

    def _bare(self):
        m = re.match(r"[^,\]\}\s]+", self.s[self.i :])
        if not m:
            self.i += 1
            return None
        tok = m.group(0)
        self.i += len(tok)
        if tok == "true":
            return True
        if tok == "false":
            return False
        if tok == "null":
            return None
        try:
            return float(tok) if "." in tok else int(tok)
        except ValueError:
            return tok


def parse_js(text: str):
    return JsLiteral(text).parse()


def match_braces(text: str, start: int) -> int:
    """Index just past the brace group opening at `start`, respecting strings."""
    depth = 0
    i = start
    quote = None
    while i < len(text):
        c = text[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "\"'`":
            quote = c
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


# ---------------------------------------------------------------------------
# Inline conversion
# ---------------------------------------------------------------------------

ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

# Symbols that mean the same thing in either mode need two spellings: one that
# is safe in running prose, one that is safe inside $...$.
MATH_SYMBOL = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\varepsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
    "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu", "ξ": r"\xi",
    "π": r"\pi", "ρ": r"\rho", "σ": r"\sigma", "τ": r"\tau",
    "φ": r"\phi", "ψ": r"\psi", "ω": r"\omega", "Δ": r"\Delta",
    "Σ": r"\Sigma", "Φ": r"\Phi", "Ω": r"\Omega", "Ψ": r"\Psi",
    "Γ": r"\Gamma", "Λ": r"\Lambda",
    "×": r"\times", "≈": r"\approx", "≤": r"\leq", "≥": r"\geq",
    "≠": r"\neq", "→": r"\rightarrow", "←": r"\leftarrow",
    "↔": r"\leftrightarrow", "⇒": r"\Rightarrow", "±": r"\pm",
    "∞": r"\infty", "∈": r"\in", "∀": r"\forall", "∃": r"\exists",
    "∇": r"\nabla", "∂": r"\partial", "≫": r"\gg", "≪": r"\ll",
    "⊤": r"\top", "⊙": r"\odot", "∝": r"\propto", "†": r"\dagger",
    "∑": r"\sum", "∫": r"\int", "≡": r"\equiv", "⟨": r"\langle",
    "⟩": r"\rangle", "•": r"\bullet", "ℓ": r"\ell", "√": r"\surd",
    "∥": r"\|", "‖": r"\|", "−": "-", "·": r"\cdot",
    "°": r"^\circ", "′": "'", "″": "''",
    "¹": "^1", "²": "^2", "³": "^3", "⁻": "^-",
    "ᵀ": r"^\top", "ᴴ": "^H", "ᵗ": "^t", "ⁿ": "^n",
    "ₜ": "_t", "ₖ": "_k", "₀": "_0", "₁": "_1", "₂": "_2",
    "ẋ": r"\dot{x}", "ẏ": r"\dot{y}", "ż": r"\dot{z}",
    "Ṁ": r"\dot{M}", "ÿ": r"\ddot{y}", "ẍ": r"\ddot{x}",
}

# Text-mode spellings. Anything not listed and not ASCII is dropped.
UNICODE = {
    "—": "---", "–": "--", "’": "'", "‘": "`",
    "“": "``", "”": "''", "…": r"\dots{}", "§": r"\S",
    "‑": "-", " ": "~", " ": r"\,", "​": "", " ": r"\,",
    "á": r"\'a", "é": r"\'e", "í": r"\'i", "ó": r"\'o",
    "ú": r"\'u", "à": r"\`a", "è": r"\`e", "ä": r'\"a',
    "ë": r'\"e', "ö": r'\"o', "ü": r'\"u', "ñ": r"\~n",
    "ç": r"\c{c}", "ø": r"\o{}", "å": r"\aa{}",
    "Á": r"\'A", "É": r"\'E", "Í": r"\'I", "Ó": r"\'O",
    "Ü": r'\"U', "Ö": r'\"O', "Ä": r'\"A', "Ø": r"\O{}",
    "ß": r"\ss{}",
}
# Every math symbol also gets a text-mode form by wrapping it in $...$.
for _k, _v in MATH_SYMBOL.items():
    UNICODE.setdefault(_k, "$" + _v + "$")

COMBINING = {"̇": r"\dot", "̈": r"\ddot", "̄": r"\bar",
             "̂": r"\hat", "̃": r"\tilde", "⃗": r"\vec"}

# Code listings run through pdfLaTeX's verbatim path, where multi-byte input is
# fragile. Source comments are transliterated to ASCII, which is how Rust
# comments are normally written anyway.
CODE_ASCII = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "κ": "kappa", "λ": "lambda",
    "μ": "mu", "ξ": "xi", "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau",
    "φ": "phi", "ψ": "psi", "ω": "omega", "Δ": "Delta", "Σ": "Sigma",
    "Φ": "Phi", "Ω": "Omega", "Γ": "Gamma", "Λ": "Lambda",
    "≤": "<=", "≥": ">=", "≠": "!=", "≈": "~=", "→": "->", "←": "<-",
    "⇒": "=>", "×": "*", "·": ".", "±": "+/-", "∞": "inf", "∈": "in",
    "⊙": "*", "∥": "||", "‖": "||", "−": "-", "√": "sqrt", "∝": "prop",
    "¹": "1", "²": "2", "³": "3", "⁻": "-", "ᵀ": "^T", "ᴴ": "^H",
    "ᵗ": "^t", "ⁿ": "^n", "ₜ": "_t", "ₖ": "_k", "₀": "_0", "₁": "_1",
    "₂": "_2", "ℓ": "l", "′": "'", "″": "''",
    "ẋ": "x_dot", "ẏ": "y_dot", "ż": "z_dot", "Ṁ": "M_dot",
    "ÿ": "y_ddot", "ẍ": "x_ddot", "q̇": "q_dot", "q̈": "q_ddot",
    "—": "--", "–": "-", "’": "'", "‘": "'", "“": '"', "”": '"',
    "…": "...", "§": "sec.", "é": "e", "í": "i", "ü": "u", "á": "a",
    "ø": "o", "ñ": "n", "ö": "o", "ä": "a", "ç": "c",
}


def asciify_code(s: str) -> str:
    """Make a code block safe for the verbatim path, preserving meaning."""
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        nxt = s[i + 1] if i + 1 < len(s) else ""
        pair = ch + nxt
        if pair in CODE_ASCII:            # e.g. q + combining dot
            out.append(CODE_ASCII[pair])
            i += 2
            continue
        if nxt in COMBINING:              # base letter + unmapped mark
            out.append(ch + "_dot" if nxt == "̇" else ch)
            i += 2
            continue
        if ch in CODE_ASCII:
            out.append(CODE_ASCII[ch])
        elif ord(ch) > 127:
            out.append("?")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _walk(s, symbol_map, wrap, escape):
    """Shared scanner for text and math mode, with combining-mark lookahead."""
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        nxt = s[i + 1] if i + 1 < len(s) else ""
        if nxt in COMBINING and (ch.isalpha() or ch in symbol_map):
            base = symbol_map.get(ch, ch)
            if base.startswith("\\") and escape:
                base = base.strip("$")
            out.append(wrap(COMBINING[nxt] + "{" + base.strip("$") + "}"))
            i += 2
            continue
        if ch in COMBINING:          # orphaned mark -- drop it
            i += 1
            continue
        if escape and ch in ESCAPES:
            out.append(ESCAPES[ch])
        elif ch in symbol_map:
            out.append(symbol_map[ch])
        elif ord(ch) > 127:
            out.append("")           # unknown glyph: drop rather than break
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def escape_text(s):
    """Prose: escape LaTeX specials, spell symbols so they work in text mode."""
    return _walk(s, UNICODE, lambda cmd: "$" + cmd + "$", True)


def fix_math(s):
    """Math span: no escaping; symbols take their bare math-mode command."""
    return _walk(s, MATH_SYMBOL, lambda cmd: cmd, False)


def inline(text: str) -> str:
    """Markdown inline → LaTeX, protecting math and code from escaping."""
    protected: list[str] = []

    def stash(payload: str) -> str:
        protected.append(payload)
        return f"\x00{len(protected) - 1}\x00"

    # Math is already LaTeX; it only needs stray Unicode symbols spelled out.
    text = re.sub(
        r"\$\$(.+?)\$\$",
        lambda m: stash("$$" + fix_math(m.group(1)) + "$$"),
        text,
        flags=re.S,
    )
    text = re.sub(
        r"(?<!\$)\$([^\$\n]+?)\$(?!\$)",
        lambda m: stash("$" + fix_math(m.group(1)) + "$"),
        text,
    )
    # Inline code
    text = re.sub(
        r"`([^`]+)`",
        lambda m: stash(r"\texttt{" + escape_text(m.group(1)).replace("-", r"-\/") + "}"),
        text,
    )
    # Links
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: stash(r"\href{" + m.group(2).replace("%", r"\%").replace("#", r"\#")
                        + "}{" + escape_text(m.group(1)) + "}"),
        text,
    )

    text = escape_text(text)

    # Emphasis, after escaping so the markers survive intact.
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\\textbf{\\emph{\1}}", text, flags=re.S)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text, flags=re.S)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"\\emph{\1}", text)

    for i, payload in enumerate(protected):
        text = text.replace(f"\x00{i}\x00", payload)
    return text


# ---------------------------------------------------------------------------
# Block conversion
# ---------------------------------------------------------------------------

CALLOUT_ENV = {
    "foundation": "shfoundation",
    "conceptual": "shconceptual",
    "practical": "shpractical",
    "insight": "shinsight",
    "warning": "shwarning",
    "note": "shnote",
    "robot": "shrobot",
}

THEOREM_ENV = {
    "Theorem": "shtheorem",
    "Definition": "shdefinition",
    "Lemma": "shlemma",
    "Proposition": "shproposition",
    "Corollary": "shcorollary",
}

WIDGET_TITLES = {
    "RustyDrive": ("ch01-drive-rusty", "Drive Rusty --- the case for learning",
        "A hand-written potential-field controller steers Rusty toward a goal and away from obstacles. Choose the room (empty, cluttered, concave trap) and the driver (hand-coded or your own arrow keys). In the empty room the controller is optimal; in the concave trap it parks in a local minimum and never escapes, while a human drives out immediately."),
    "SuccessLevels": ("ch01-success-levels", "Levels of real-world success",
        "Tang et al.'s L0--L5 maturity ladder with surveyed systems placed on it, filterable by competency. Locomotion reaches L5; human--robot interaction sits at L1."),
    "ContractionDemo": ("ch02-contraction-map", "A contraction, iterated",
        "Iterates of $T(x) = \\gamma x + 2$ converge to the same fixed point from any start. Drag $\\gamma$ toward 1 and convergence slows; the a-priori bound is plotted beside the actual error and never dips below it."),
    "PendleSim": ("ch02-integrator-playground", "Pendle: continuous dynamics, discrete steps",
        "The pendulum integrated by explicit Euler, semi-implicit Euler and RK4, with total energy plotted. At $\\tau = 0$ energy should be flat: explicit Euler's climbs without bound, RK4's holds."),
    "BanditTestbed": ("ch03-bandit-testbed", "The 10-armed testbed",
        "Average reward, percentage of optimal actions, and cumulative regret for $\\varepsilon$-greedy, optimistic initialization, UCB, gradient bandits and Thompson sampling, averaged over independent runs."),
    "MdpExplorer": ("ch04-mdp-editor", "The warehouse as an MDP",
        "Click any floor cell to read its full transition distribution $p(s', r \\mid s, a)$ and the four action values. Changing $\\gamma$ or $p_{\\text{slip}}$ re-solves the MDP live."),
    "GpiDashboard": ("ch05-gpi-dashboard", "Generalized policy iteration, sweep by sweep",
        "Value heatmap and greedy-policy arrows on Rusty's warehouse, advancing one sweep at a time, with $\\Delta$ and the implied suboptimality bound streaming beside them."),
    "CurseOfDimensionality": ("ch14-dimensionality-wall", "The exponential wall",
        "Set the degrees of freedom and bins per dimension, and read how long one exhaustive tabular sweep would take. Seven degrees of freedom at ten bins already outlives the solar system."),
    "TdDashboard": ("ch06-train-live", "Learning without a model",
        "Rusty learns the warehouse from experience alone. Value bleeds backwards from the dock as the TD error propagates; episode return and mean $|\\delta|$ stream alongside."),
    "LambdaDial": ("ch07-lambda-dial", "The $\\lambda$ dial: from TD(0) to Monte Carlo",
        "Two views of $\\lambda$: the weights $(1-\\lambda)\\lambda^{n-1}$ placed on each $n$-step return, and the eligibility trace decaying along a corridor Rusty has just driven."),
    "DeadlyTriad": ("ch08-deadly-triad", "The deadly triad, one ingredient at a time",
        "Baird's counterexample running live. All rewards are zero and the true value function is exactly representable, yet the parameters diverge. Switch off function approximation, bootstrapping or off-policy training and stability returns."),
    "ReplayBuffer": ("ch09-replay-target", "Replay and target networks as variance surgery",
        "Disable each and watch its characteristic failure appear: correlated batches add noise, a moving target adds oscillation."),
    "PolicyGradientLab": ("ch10-gradient-variance", "Three unbiased estimators, three variances",
        "The same policy gradient estimated hundreds of times under REINFORCE, REINFORCE with a baseline, and GAE. All share a mean; the spread differs by decades."),
    "EntropyDial": ("ch11-entropy-dial", "The entropy temperature",
        "The maximum-entropy optimal policy $\\pi^*(a\\mid s) \\propto \\exp(Q(s,a)/\\alpha)$ over a fixed $Q$-landscape. As $\\alpha \\to 0$ it collapses to a deterministic spike; raised, it hedges across every action worth considering."),
    "ModelBiasFan": ("ch12-imagination-fan", "Imagination diverges from reality",
        "An ensemble of learned models rolled forward from one state, fanning out as errors compound. Measured disagreement is plotted against the geometric bound $\\varepsilon(L^H-1)/(L-1)$."),
    "ReacherKinematics": ("ch13-kinematics-sandbox", "Reacher: joint space and task space",
        "Drag the end-effector to solve inverse kinematics, or move the joints directly. The manipulability ellipsoid rounds out where the arm is dexterous and flattens to a line as $\\det J = \\ell_1\\ell_2\\sin q_2$ approaches zero."),
    "DomainRandomization": ("ch15-randomization-wall", "Trading peak for robustness",
        "Success against the real robot's friction, for policies trained on progressively wider parameter distributions. The narrow policy peaks at nominal and fails elsewhere; the randomized one is merely good everywhere."),
    "CovariateShift": ("ch16-covariate-drift", "The cloned robot drifts",
        "A cloned policy drifting off the demonstrated lane inside a growing error cone, with the $O(\\varepsilon T^2)$ and $O(\\varepsilon T)$ bounds plotted. Switching on DAgger collapses the cone to a linear band."),
    "DmpSculptor": ("ch17-dmp-sculptor", "A movement primitive you can reshape",
        "A dynamic movement primitive: a spring toward the goal plus a learned forcing term. Move the goal and the demonstrated shape survives the deformation while the trajectory still terminates exactly at $g$."),
    "RewardMixer": ("ch18-reward-mixer", "Reward anatomy: the weights decide the gait",
        "Slide the weight on each locomotion reward term --- velocity tracking, torque penalty, foot air time, orientation, foot slip --- and see the gait that would emerge, with each term's physical units shown."),
    "PipelineSwitcher": ("ch19-pipeline-switcher", "End-to-end or modular?",
        "The same navigation task under three architectures as the environment becomes less structured, with success, interpretability, tuning burden and data requirement tracked for each."),
    "GraspWrench": ("ch20-grasp-wrench", "Force closure: when a grasp actually holds",
        "Drag contact points around an object and watch their friction cones rotate. When the cones' span collapses into a half-plane, force closure fails and a direction of push exists that the grasp cannot resist."),
    "SharedAutonomy": ("ch21-shared-autonomy", "Assistance that helps, and assistance that annoys",
        "The robot infers which target you want from noisy input and blends its action with yours. Raising the blend improves task success and lowers your sense of control --- the trade no reward function here captures."),
    "MissionControl": ("ch22-mission-control", "Mission control",
        "The capstone's telemetry: training curves across five seeds, reward-term composition, evaluation with Wilson intervals, and the failure post-mortems."),
}


# Diagrams that exist as inline SVG in the web edition, redrawn for print.
TIKZ_FIGURES = {
    "Figure 1.1": r"""\begin{tikzpicture}[
    font=\sffamily\scriptsize,
    box/.style={rounded corners=3pt, draw, line width=0.7pt, minimum width=2.1cm,
                minimum height=1.05cm, align=center},
    flow/.style={-{Latex[length=1.6mm]}, draw=shSlate500, line width=0.55pt}]
  \node[box, draw=shConceptual, fill=shConceptual!8]  (see)   at (0,0)
        {\textbf{SEE}\\[1pt]\textcolor{shInkMuted}{lidar, camera}};
  \node[box, draw=shFoundation, fill=shFoundation!10, line width=1.1pt] (think) at (4,0)
        {\textbf{THINK}\\[1pt]\textcolor{shInkMuted}{policy $\pi$}};
  \node[box, draw=shPractical, fill=shPractical!8]    (act)   at (8,0)
        {\textbf{ACT}\\[1pt]\textcolor{shInkMuted}{motors}};

  \draw[flow] (see)   -- node[above, font=\sffamily\tiny] {$o_t$} (think);
  \draw[flow] (think) -- node[above, font=\sffamily\tiny] {$a_t$} (act);

  \draw[flow, dashed] (act.south) -- ++(0,-0.72)
        -| node[pos=0.26, below, font=\sffamily\tiny, text=shInkMuted]
        {the world advances --- and not always the same way} (see.south);
\end{tikzpicture}""",
}


def convert_table(lines: list[str]) -> str:
    rows = []
    for ln in lines:
        ln = ln.strip()
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue
        rows.append(cells)
    if not rows:
        return ""
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]

    # Narrow first column, remaining share the rest — keeps wide tables on-page.
    colspec = ">{\\raggedright\\arraybackslash}p{0.20\\linewidth}" + "".join(
        [f">{{\\raggedright\\arraybackslash}}p{{{0.76 / max(ncols - 1, 1):.3f}\\linewidth}}"
         for _ in range(ncols - 1)]
    )

    # Wide tables carry formulas in narrow columns; step the size down so they fit.
    size = "\\scriptsize" if ncols >= 4 else "\\footnotesize"
    out = [f"\\begingroup{size}\\setlength{{\\tabcolsep}}{{4pt}}"
           "\\setlength{\\emergencystretch}{2em}",
           f"\\begin{{longtable}}{{{colspec}}}", "\\toprule"]
    out.append(" & ".join(f"\\textbf{{{inline(c)}}}" for c in rows[0]) + " \\\\")
    out.append("\\midrule\\endhead")
    for r in rows[1:]:
        out.append(" & ".join(inline(c) for c in r) + " \\\\")
    out += ["\\bottomrule", "\\end{longtable}", "\\endgroup", ""]
    return "\n".join(out)


def convert_markdown(md: str) -> str:
    """Convert a markdown block (no custom components) to LaTeX."""
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    para: list[str] = []
    list_stack: list[str] = []

    def flush_para():
        nonlocal para
        if para:
            out.append(inline(" ".join(para).strip()))
            out.append("")
            para = []

    def close_lists():
        while list_stack:
            out.append(f"\\end{{{list_stack.pop()}}}")
        out.append("")

    while i < len(lines):
        ln = lines[i]
        stripped = ln.strip()

        # Display math block
        if stripped.startswith("$$"):
            flush_para()
            close_lists()
            block = [ln]
            if stripped.count("$$") < 2:
                i += 1
                while i < len(lines):
                    block.append(lines[i])
                    if "$$" in lines[i]:
                        break
                    i += 1
            body = "\n".join(block).strip()
            body = body[2:-2] if body.startswith("$$") and body.endswith("$$") else body
            eq = fix_math(body.strip())
            # A single-line display that cannot be broken is scaled to the
            # measure rather than allowed to run into the margin. Aligned or
            # multi-line environments already handle their own breaking.
            structured = any(
                tok in eq for tok in (r"\begin{aligned}", r"\begin{split}",
                                      r"\begin{array}", r"\\")
            )
            if len(eq) > 118 and not structured:
                out += [r"\begin{equation*}",
                        r"\resizebox{\linewidth}{!}{$\displaystyle " + eq + "$}",
                        r"\end{equation*}", ""]
            else:
                out += [r"\begin{equation*}", eq, r"\end{equation*}", ""]
            i += 1
            continue

        # Headings
        m = re.match(r"^(#{2,4})\s+(.*)$", stripped)
        if m:
            flush_para()
            close_lists()
            level, title = len(m.group(1)), m.group(2).strip()
            # Strip the leading "N.N " numbering — LaTeX numbers sections itself.
            title = re.sub(r"^\d+\.\d+\s+", "", title)
            cmd = {2: "section", 3: "subsection", 4: "subsubsection"}[level]
            out += [f"\\{cmd}{{{inline(title)}}}", ""]
            i += 1
            continue

        # Horizontal rule
        if re.fullmatch(r"-{3,}|\*{3,}", stripped):
            flush_para()
            close_lists()
            out += ["\\vspace{6pt}\\noindent{\\color{shRule}\\rule{\\linewidth}{0.4pt}}\\vspace{6pt}", ""]
            i += 1
            continue

        # Table
        if stripped.startswith("|"):
            flush_para()
            close_lists()
            tbl = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl.append(lines[i])
                i += 1
            out.append(convert_table(tbl))
            continue

        # Lists
        m_ul = re.match(r"^(\s*)[-*+]\s+(.*)$", ln)
        m_ol = re.match(r"^(\s*)\d+\.\s+(.*)$", ln)
        if m_ul or m_ol:
            flush_para()
            env = "itemize" if m_ul else "enumerate"
            body = (m_ul or m_ol).group(2)
            if not list_stack:
                out.append(f"\\begin{{{env}}}")
                list_stack.append(env)
            elif list_stack[-1] != env:
                out.append(f"\\end{{{list_stack.pop()}}}")
                out.append(f"\\begin{{{env}}}")
                list_stack.append(env)
            # Gather continuation lines
            item = [body]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(
                r"^\s*([-*+]|\d+\.)\s+|^#{2,4}\s|^\||^\$\$", lines[i]
            ):
                item.append(lines[i].strip())
                i += 1
            out.append(f"\\item {inline(' '.join(item))}")
            continue

        # Blank line
        if not stripped:
            flush_para()
            if list_stack:
                close_lists()
            i += 1
            continue

        para.append(stripped)
        i += 1

    flush_para()
    close_lists()
    return "\n".join(out)


def find_component(text: str, start: int):
    """Locate the next JSX component at or after `start`. Returns a dict or None."""
    m = re.compile(r"<([A-Z][A-Za-z0-9]*)").search(text, start)
    if not m:
        return None
    name = m.group(1)
    # Walk the opening tag, respecting braces and strings.
    i = m.end()
    depth = 0
    quote = None
    while i < len(text):
        c = text[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == ">" and depth == 0:
            break
        i += 1
    tag_end = i
    open_tag = text[m.start(): tag_end + 1]
    self_closing = open_tag.rstrip().endswith("/>")

    if self_closing:
        return {"name": name, "attrs_src": open_tag, "body": "",
                "start": m.start(), "end": tag_end + 1}

    # Find the matching close tag, allowing nesting of the same component.
    depth = 1
    j = tag_end + 1
    pat = re.compile(rf"<(/?){name}[\s>/]")
    while j < len(text) and depth > 0:
        mm = pat.search(text, j)
        if not mm:
            break
        depth += -1 if mm.group(1) else 1
        j = mm.end()
    close = text.rfind(f"</{name}>", 0, j + len(name) + 4)
    if close < tag_end:
        close = text.find(f"</{name}>", tag_end)
    body_end = close if close > 0 else len(text)
    return {
        "name": name,
        "attrs_src": open_tag,
        "body": text[tag_end + 1: body_end],
        "start": m.start(),
        "end": (body_end + len(name) + 3) if close > 0 else len(text),
    }


def parse_attrs(src: str) -> dict:
    """Extract JSX attributes: string literals and {…} expressions."""
    attrs = {}
    body = src
    # Strip the tag name.
    body = re.sub(r"^<[A-Za-z0-9]+", "", body).rstrip("/>").rstrip(">")
    i = 0
    while i < len(body):
        m = re.compile(r"([A-Za-z_][\w]*)\s*=\s*").search(body, i)
        if not m:
            break
        key = m.group(1)
        j = m.end()
        if j >= len(body):
            break
        if body[j] in "\"'":
            q = body[j]
            k = j + 1
            buf = []
            while k < len(body):
                if body[k] == "\\":
                    buf.append(body[k + 1])
                    k += 2
                    continue
                if body[k] == q:
                    break
                buf.append(body[k])
                k += 1
            attrs[key] = "".join(buf)
            i = k + 1
        elif body[j] == "{":
            end = match_braces(body, j)
            attrs[key] = parse_js(body[j + 1: end - 1])
            i = end
        else:
            i = j + 1
    return attrs


def render_component(comp: dict) -> str:
    name = comp["name"]
    attrs = parse_attrs(comp["attrs_src"])
    body = comp["body"]

    if name in WIDGET_TITLES:
        wid, title, desc = WIDGET_TITLES[name]
        return f"\\shwidget{{{wid}}}{{{title}}}{{{desc}}}\n"

    if name == "ChapterOverview":
        summary = inline(attrs.get("summary", ""))
        f = inline(attrs.get("foundation", ""))
        c = inline(attrs.get("conceptual", ""))
        p = inline(attrs.get("practical", ""))
        out = [f"\\shoverview{{{summary}}}{{{f}}}{{{c}}}{{{p}}}"]
        outcomes = attrs.get("outcomes") or []
        if outcomes:
            out.append("\\begin{shoutcomes}")
            for o in outcomes:
                out.append(f"  \\item {inline(str(o))}")
            out.append("\\end{shoutcomes}")
        return "\n".join(out) + "\n"

    if name == "Callout":
        env = CALLOUT_ENV.get(attrs.get("variant", "note"), "shnote")
        title = attrs.get("title")
        opt = f"[{inline(title)}]" if title else ""
        return f"\\begin{{{env}}}{opt}\n{convert_body(body)}\n\\end{{{env}}}\n"

    if name == "Theorem":
        env = THEOREM_ENV.get(attrs.get("kind", "Theorem"), "shtheorem")
        title = attrs.get("title")
        opt = f"[{inline(title)}]" if title else ""
        return f"\\begin{{{env}}}{opt}\n{convert_body(body)}\n\\end{{{env}}}\n"

    if name == "Proof":
        title = attrs.get("title", "Proof")
        # The proof environment supplies its own end marker, so drop any
        # \blacksquare the prose already carried.
        cleaned = re.sub(r"(\\qquad\s*)?\\blacksquare\s*", "", body)
        tex = convert_body(cleaned).rstrip()
        # When a proof closes on a display equation, \qedhere puts the marker on
        # that line instead of stranding it in a paragraph below.
        if tex.endswith(r"\end{equation*}"):
            tex = tex[: -len(r"\end{equation*}")] + "\\qedhere\n\\end{equation*}"
        return f"\\begin{{proof}}[{inline(title)}]\n{tex}\n\\end{{proof}}\n"

    if name == "RustSnippet":
        title = attrs.get("title", "")
        caption = inline(attrs.get("caption", ""))
        m = re.search(r"```rust\n(.*?)```", body, re.S)
        code = asciify_code(m.group(1).rstrip()) if m else ""
        safe_title = title.replace("_", r"\_").replace("#", r"\#")
        return (f"\\begin{{shrust}}{{{safe_title}}}{{{caption}}}\n"
                f"\\begin{{lstlisting}}\n{code}\n\\end{{lstlisting}}\n"
                f"\\end{{shrust}}\n")

    if name == "Exercises":
        items = attrs.get("items") or []
        out = ["\\subsection*{Exercises}", "\\begin{shexercises}"]
        for it in items:
            layer = str(it.get("layer", "F"))
            diff = int(it.get("difficulty", 2) or 2)
            out.append(
                f"\\shexercise{{{layer}}}{{{diff}}}"
                f"{{{inline(str(it.get('title', '')))}}}"
                f"{{{inline(str(it.get('body', '')))}}}"
            )
        out.append("\\end{shexercises}")
        return "\n".join(out) + "\n"

    if name == "CodingTask":
        title = inline(attrs.get("title", ""))
        crate = attrs.get("crate", "")
        crate_tex = crate.replace("_", r"\_") if crate else ""
        out = [f"\\begin{{shcodingtask}}{{{title}}}{{{crate_tex}}}", convert_body(body)]
        deliverables = attrs.get("deliverables") or []
        if deliverables:
            out.append("\\vspace{3pt}\\noindent{\\displayfont\\bfseries\\fontsize{8}{10}"
                       "\\selectfont\\color{shInkMuted}DELIVERABLES}")
            out.append("\\begin{itemize}[leftmargin=13pt, topsep=2pt, itemsep=1.5pt]")
            for d in deliverables:
                out.append(f"  \\item {inline(str(d))}")
            out.append("\\end{itemize}")
        out.append("\\end{shcodingtask}")
        return "\n".join(out) + "\n"

    if name == "References":
        items = attrs.get("items") or []
        baseline = [r for r in items if r.get("baseline")]
        modern = [r for r in items if not r.get("baseline")]
        out = ["\\begin{shreferences}"]
        for group, label in ((baseline, "Baseline references"),
                             (modern, "Further reading and modern sources")):
            if not group:
                continue
            out.append("\\item[]\\hspace{-11pt}{\\displayfont\\bfseries\\fontsize{8}{10}"
                       f"\\selectfont\\color{{shInkMuted}}\\MakeUppercase{{{label}}}}}")
            for r in group:
                note = inline(str(r.get("note", ""))) if r.get("note") else ""
                venue = inline(str(r.get("venue", ""))) if r.get("venue") else ""
                prefix = "\\shbaseline " if r.get("baseline") else ""
                out.append(
                    f"\\shref{{{prefix}{inline(str(r.get('authors', '')))}}}"
                    f"{{{inline(str(r.get('year', '')))}}}"
                    f"{{{inline(str(r.get('title', '')))}}}"
                    f"{{{venue}}}{{{note}}}"
                )
        out.append("\\end{shreferences}")
        return "\n".join(out) + "\n"

    if name == "Figure":
        caption = inline(attrs.get("caption", ""))
        label = inline(attrs.get("label", "Figure"))
        # Diagrams authored as inline SVG are redrawn in TikZ for print; the
        # lookup is keyed on the figure label.
        drawing = TIKZ_FIGURES.get(attrs.get("label", "").strip(), "")
        body_tex = (f"\\begin{{center}}\n{drawing}\n\\end{{center}}\n\\vspace{{2pt}}\n"
                    if drawing else "")
        return ("\\begin{tcolorbox}[shblock, colframe=shSlate300, colback=shSlate50]\n"
                f"{{\\displayfont\\bfseries\\fontsize{{8}}{{10}}\\selectfont"
                f"\\color{{shInkMuted}}\\MakeUppercase{{{label}}}}}\\par\\vspace{{4pt}}\n"
                f"{body_tex}"
                f"{{\\fontsize{{9.5}}{{12.5}}\\selectfont\\color{{shInkSecondary}} {caption}}}\n"
                "\\end{tcolorbox}\n")

    # Unknown component — render its body rather than dropping content.
    return convert_body(body)


def convert_body(text: str) -> str:
    """Alternate between markdown runs and JSX components."""
    out = []
    pos = 0
    while True:
        comp = find_component(text, pos)
        if not comp:
            out.append(convert_markdown(text[pos:]))
            break
        out.append(convert_markdown(text[pos: comp["start"]]))
        out.append(render_component(comp))
        pos = comp["end"]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Frontmatter and driver
# ---------------------------------------------------------------------------


def parse_frontmatter(src: str):
    m = re.match(r"^---\n(.*?)\n---\n", src, re.S)
    if not m:
        return {}, src
    raw, rest = m.group(1), src[m.end():]
    meta, quote = {}, {}
    in_quote = False
    for line in raw.split("\n"):
        if not line.strip():
            continue
        if re.match(r"^quote:\s*$", line):
            in_quote = True
            continue
        m2 = re.match(r"^(\s*)([A-Za-z_]+):\s*(.*)$", line)
        if not m2:
            continue
        indent, key, val = m2.group(1), m2.group(2), m2.group(3).strip()
        val = val.strip("'\"")
        if in_quote and indent:
            quote[key] = val
        else:
            in_quote = False
            meta[key] = val
    if quote:
        meta["quote"] = quote
    return meta, rest


def convert_chapter(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(src)

    slug = re.sub(r"^ch\d+-", "", path.stem)
    title = meta.get("title", path.stem)

    out = [
        f"% Generated from content/chapters/{path.name} -- do not edit by hand.",
        f"\\chapter{{{inline(title)}}}",
        f"\\setchapterslug{{{slug}}}",
        "",
    ]

    q = meta.get("quote")
    if q:
        out.append(
            "\\shepigraph{%s}{%s}{%s}{%s}" % (
                inline(q.get("text", "")),
                inline(q.get("author", "")),
                inline(q.get("affiliation", "")),
                inline(q.get("source", "")),
            )
        )
        out.append("")

    out.append(convert_body(body))
    return "\n".join(out) + "\n"


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    src_dir = root / "rl4r-web" / "content" / "chapters"
    out_dir = Path(__file__).resolve().parents[1] / "chapters"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("ch*.mdx"))
    if not files:
        print(f"no chapters found in {src_dir}", file=sys.stderr)
        return 1

    for f in files:
        tex = convert_chapter(f)
        target = out_dir / f"{f.stem.split('-')[0]}.tex"
        target.write_text(tex, encoding="utf-8")
        print(f"{f.name:52s} → {target.name}  ({len(tex.splitlines()):5d} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

'use client';

import { useMemo, useState } from 'react';
import { SimPanel, Slider } from './SimControls';
import { StatTile } from '@/components/viz/StatTile';
import { LineChart } from '@/components/viz/LineChart';
import { useTheme } from '@/components/layout/ThemeProvider';
import { seriesColor } from '@/lib/theme';

/**
 * `ch17-dmp-sculptor` — dynamic movement primitives, and what they generalize.
 *
 * A DMP is a spring–damper pulled toward a goal, plus a learned forcing term
 * that shapes the path taken to get there. The spring guarantees convergence
 * no matter what the forcing term does; the forcing term carries the style of
 * the demonstrated motion. Move the goal and the shape survives — that is the
 * generalization property that made DMPs the standard motor representation.
 */
export function DmpSculptor() {
  const { mode } = useTheme();
  const [goal, setGoal] = useState(1.0);
  const [tau, setTau] = useState(1.0);
  const [nBasis, setNBasis] = useState(10);
  const [forcingScale, setForcingScale] = useState(1.0);

  const { trajectory, basisCurves, forcing, converged } = useMemo(() => {
    // Canonical system: ẋ = −α_x x / τ, decaying from 1 to 0. It is the DMP's
    // clock, and because it decays the forcing term must vanish — which is why
    // convergence to the goal is guaranteed regardless of what was learned.
    const alphaX = 4.0;
    const alphaY = 25.0;
    const betaY = alphaY / 4; // critical damping
    const dt = 0.004;
    const steps = 900;
    const y0 = 0;

    // Gaussian basis functions spaced along the canonical variable.
    const centers = Array.from({ length: nBasis }, (_, i) =>
      Math.exp((-alphaX * i) / (nBasis - 1)),
    );
    const widths = centers.map((_, i) =>
      i < nBasis - 1 ? 1.2 / Math.pow(centers[i + 1] - centers[i], 2) : 1.2 / Math.pow(0.05, 2),
    );

    // Weights that would be fitted from a demonstration by locally weighted
    // regression. Here: a fixed "reach over an obstacle then descend" shape.
    const weights = Array.from({ length: nBasis }, (_, i) => {
      const t = i / (nBasis - 1);
      return forcingScale * (140 * Math.sin(Math.PI * t) - 60 * Math.sin(2 * Math.PI * t));
    });

    let x = 1;
    let y = y0;
    let dy = 0;
    const traj: Array<{ x: number; y: number }> = [];
    const forcingTrace: Array<{ x: number; y: number }> = [];

    for (let k = 0; k < steps; k++) {
      // Weighted sum of basis activations, normalized.
      let num = 0;
      let den = 0;
      for (let i = 0; i < nBasis; i++) {
        const psi = Math.exp(-widths[i] * Math.pow(x - centers[i], 2));
        num += psi * weights[i];
        den += psi;
      }
      // The (g − y₀)x factor makes the forcing scale with the movement extent
      // and vanish as x → 0.
      const f = den > 1e-9 ? (num / den) * x * (goal - y0) : 0;

      // Transformation system: a critically damped spring toward the goal,
      // perturbed by the learned forcing term.
      const ddy = (alphaY * (betaY * (goal - y) - dy) + f) / (tau * tau);
      dy += ddy * dt;
      y += dy * dt;
      x += (-alphaX * x * dt) / tau;

      if (k % 4 === 0) {
        traj.push({ x: k * dt, y });
        forcingTrace.push({ x: k * dt, y: f / 100 });
      }
    }

    // Basis activations over time, for the lower panel.
    const basis = Array.from({ length: Math.min(nBasis, 6) }, (_, i) => {
      const idx = Math.floor((i * (nBasis - 1)) / Math.max(1, Math.min(nBasis, 6) - 1));
      let xx = 1;
      const pts: Array<{ x: number; y: number }> = [];
      for (let k = 0; k < steps; k += 8) {
        const psi = Math.exp(-widths[idx] * Math.pow(xx - centers[idx], 2));
        pts.push({ x: k * dt, y: psi });
        xx += (-alphaX * xx * dt * 8) / tau;
      }
      return { id: `ψ${idx + 1}`, data: pts };
    });

    return {
      trajectory: traj,
      basisCurves: basis,
      forcing: forcingTrace,
      converged: Math.abs(y - goal),
    };
  }, [goal, tau, nBasis, forcingScale]);

  const W = 400;
  const H = 175;
  const yVals = trajectory.map((p) => p.y);
  const lo = Math.min(...yVals, 0, goal) - 0.15;
  const hi = Math.max(...yVals, goal) + 0.15;
  const sx = (t: number) => 34 + (t / 3.6) * (W - 50);
  const sy = (v: number) => H - 24 - ((v - lo) / (hi - lo)) * (H - 44);

  return (
    <SimPanel
      title="A movement primitive you can reshape"
      id="ch17-dmp-sculptor"
      subtitle="τ²ÿ = α(β(g − y) − τẏ) + f(x) — a spring toward the goal, plus a learned forcing term carrying the demonstrated style."
      controls={
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Slider
            label="Goal g"
            value={goal}
            min={-0.6}
            max={2.2}
            step={0.05}
            onChange={setGoal}
            hint="move it — the shape survives"
          />
          <Slider
            label="Time scaling τ"
            value={tau}
            min={0.4}
            max={2.5}
            step={0.05}
            onChange={setTau}
            hint="slow down or speed up, same path"
          />
          <Slider
            label="Basis functions"
            value={nBasis}
            min={3}
            max={30}
            step={1}
            onChange={setNBasis}
            format={(v) => v.toFixed(0)}
            hint="capacity of the forcing term"
          />
          <Slider
            label="Forcing amplitude"
            value={forcingScale}
            min={0}
            max={2}
            step={0.05}
            onChange={setForcingScale}
            hint="0 = pure spring, no style"
          />
        </div>
      }
      caption="Set the forcing amplitude to zero and the DMP becomes a plain critically-damped spring: it reaches the goal by the most boring path possible. Turn it up and the demonstrated shape reappears — the arc over the obstacle, the descent onto the target. Now drag the goal: the shape deforms smoothly rather than breaking, and the trajectory still ends exactly at g. That guarantee comes from the canonical system decaying to zero, which forces f to vanish and leaves the spring in charge at the end."
    >
      <div className="grid gap-3 lg:grid-cols-[1fr,185px]">
        <div>
          <svg
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            className="max-w-full rounded-lg"
            style={{ background: 'var(--surface-sunken)' }}
            role="img"
            aria-label="DMP trajectory from start to goal, showing the shaped path"
          >
            <line
              x1={30}
              y1={sy(goal)}
              x2={W - 12}
              y2={sy(goal)}
              stroke="var(--status-good)"
              strokeWidth={1.5}
              strokeDasharray="5 4"
            />
            <text x={W - 46} y={sy(goal) - 5} fontSize={9.5} fill="var(--status-good)">
              goal g
            </text>

            <path
              d={`M${trajectory.map((p) => `${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' L')}`}
              fill="none"
              stroke={seriesColor(0, mode)}
              strokeWidth={2.5}
            />
            <circle cx={sx(0)} cy={sy(0)} r={5} fill={seriesColor(1, mode)} />
            <text x={sx(0) - 4} y={sy(0) + 18} fontSize={9.5} fill="var(--text-muted)">
              start
            </text>
            <text x={34} y={14} fontSize={9.5} fill="var(--text-muted)">
              position y(t)
            </text>
          </svg>

          <LineChart
            data={[...basisCurves, { id: 'forcing f(x)/100', data: forcing }]}
            height={155}
            xLegend="time (s)"
            yLegend="activation"
            caption="Basis activations sweep across the movement in order, gated by the canonical clock; their weighted sum is the forcing term."
          />
        </div>

        <div className="space-y-2">
          <StatTile
            label="Final error |y − g|"
            value={converged}
            status={converged < 0.02 ? 'good' : 'warning'}
            hint="the spring guarantees this → 0"
          />
          <StatTile
            label="Movement duration"
            value={3.6 * tau}
            unit="s"
            hint="τ rescales time, not shape"
          />
          <StatTile
            label="Learned parameters"
            value={nBasis}
            hint="one weight per basis function"
          />
          <p className="rounded-lg border border-hairline px-2.5 py-2 text-[11.5px] leading-snug text-ink-muted">
            A whole reaching motion in ten numbers — which is why policy search
            over DMP weights was tractable on real robots long before deep RL.
          </p>
        </div>
      </div>
    </SimPanel>
  );
}

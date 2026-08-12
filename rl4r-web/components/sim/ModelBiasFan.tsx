'use client';

import { useMemo, useState } from 'react';
import { LineChart } from '@/components/viz/LineChart';
import { StatTile } from '@/components/viz/StatTile';
import { SimPanel, Slider } from './SimControls';
import { gaussian, mulberry32 } from '@/lib/rl/random';
import { useTheme } from '@/components/layout/ThemeProvider';
import { seriesColor } from '@/lib/theme';

/**
 * `ch12-imagination-fan` — why model-based rollouts must be short.
 *
 * A learned dynamics model has per-step error ε. Roll it forward and the error
 * compounds: each prediction is fed back in as the next input, so the model is
 * soon extrapolating on states it never saw. The ensemble fan shows the
 * disagreement growing with horizon — which is exactly the quantity MBPO uses
 * to decide how far to imagine.
 */
export function ModelBiasFan() {
  const { mode } = useTheme();
  const [epsilon, setEpsilon] = useState(0.03);
  const [horizon, setHorizon] = useState(25);
  const [ensemble, setEnsemble] = useState(5);

  const { members, truth, spread, boundSeries } = useMemo(() => {
    const rng = mulberry32(29);
    // Ground truth: a damped oscillation — Pendle relaxing toward upright.
    const trueTraj = Array.from({ length: horizon + 1 }, (_, t) => ({
      x: t,
      y: 2.2 * Math.exp(-0.06 * t) * Math.cos(0.42 * t),
    }));

    // Each ensemble member has its own slightly wrong dynamics; errors feed
    // back through the rollout, so deviation grows super-linearly.
    const memberTrajs = Array.from({ length: ensemble }, () => {
      const damp = 0.06 + gaussian(rng, 0, epsilon * 0.9);
      const freq = 0.42 + gaussian(rng, 0, epsilon * 1.1);
      let drift = 0;
      return Array.from({ length: horizon + 1 }, (_, t) => {
        drift += gaussian(rng, 0, epsilon * 0.5);
        return { x: t, y: 2.2 * Math.exp(-damp * t) * Math.cos(freq * t) + drift };
      });
    });

    // Empirical spread at each horizon.
    const spreadAt = Array.from({ length: horizon + 1 }, (_, t) => {
      const ys = memberTrajs.map((m) => m[t].y);
      const mean = ys.reduce((a, b) => a + b, 0) / ys.length;
      const sd = Math.sqrt(ys.reduce((a, b) => a + (b - mean) ** 2, 0) / ys.length);
      return { x: t, y: sd };
    });

    // The theoretical bound: with Lipschitz constant L > 1, error after H steps
    // is bounded by ε(L^H − 1)/(L − 1) — geometric, not linear.
    const L = 1.08;
    const bound = Array.from({ length: horizon + 1 }, (_, H) => ({
      x: H,
      y: Math.min(8, (epsilon * (Math.pow(L, H) - 1)) / (L - 1)),
    }));

    return {
      members: memberTrajs,
      truth: trueTraj,
      spread: spreadAt,
      boundSeries: bound,
    };
  }, [epsilon, horizon, ensemble]);

  const finalSpread = spread[spread.length - 1]?.y ?? 0;
  // The horizon at which disagreement first exceeds a usable threshold.
  const trustHorizon = spread.findIndex((p) => p.y > 0.5);

  const W = 420;
  const H = 190;
  const xScale = (x: number) => 30 + (x / horizon) * (W - 45);
  const yScale = (y: number) => H / 2 - y * 28;

  return (
    <SimPanel
      title="Imagination diverges from reality"
      id="ch12-imagination-fan"
      subtitle="An ensemble of learned models rolled forward from the same state. The fan is the model's own estimate of how much it should be trusted."
      controls={
        <div className="grid gap-3 sm:grid-cols-3">
          <Slider
            label="Per-step model error ε"
            value={epsilon}
            min={0.005}
            max={0.12}
            step={0.005}
            onChange={setEpsilon}
            format={(v) => v.toFixed(3)}
            hint="how wrong one prediction is"
          />
          <Slider
            label="Rollout horizon H"
            value={horizon}
            min={5}
            max={60}
            step={1}
            onChange={setHorizon}
            format={(v) => `${v.toFixed(0)} steps`}
          />
          <Slider
            label="Ensemble members"
            value={ensemble}
            min={2}
            max={10}
            step={1}
            onChange={setEnsemble}
            format={(v) => v.toFixed(0)}
          />
        </div>
      }
      caption="Nothing here is a bad model — every member has small per-step error. The problem is feedback: each prediction becomes the next input, so errors compound geometrically rather than adding. This is why MBPO branches short rollouts off real states instead of imagining whole episodes, and why the ensemble's own disagreement is the right signal for when to stop trusting it."
    >
      <div className="grid gap-3 lg:grid-cols-[1fr,200px]">
        <div>
          <svg
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            className="max-w-full rounded-lg"
            style={{ background: 'var(--surface-sunken)' }}
            role="img"
            aria-label="Ensemble of model rollouts fanning out from a shared initial state, with the true trajectory overlaid"
          >
            <line x1={30} y1={H / 2} x2={W - 15} y2={H / 2} stroke="var(--gridline)" />
            {members.map((m, i) => (
              <path
                key={i}
                d={`M${m.map((p) => `${xScale(p.x)},${yScale(p.y)}`).join(' L')}`}
                fill="none"
                stroke={seriesColor(0, mode)}
                strokeWidth={1.5}
                opacity={0.42}
              />
            ))}
            <path
              d={`M${truth.map((p) => `${xScale(p.x)},${yScale(p.y)}`).join(' L')}`}
              fill="none"
              stroke={seriesColor(1, mode)}
              strokeWidth={2.5}
            />
            {trustHorizon > 0 && (
              <>
                <line
                  x1={xScale(trustHorizon)}
                  y1={10}
                  x2={xScale(trustHorizon)}
                  y2={H - 20}
                  stroke="var(--status-warning)"
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                />
                <text
                  x={xScale(trustHorizon) + 4}
                  y={20}
                  fontSize={9.5}
                  fill="var(--status-warning)"
                >
                  trust ends
                </text>
              </>
            )}
            <text x={30} y={H - 5} fontSize={9.5} fill="var(--text-muted)">
              step 0
            </text>
            <text x={W - 55} y={H - 5} fontSize={9.5} fill="var(--text-muted)">
              step {horizon}
            </text>
          </svg>
          <p className="mt-1.5 text-[11.5px] text-ink-muted">
            Thin lines: ensemble members. Thick line: the real trajectory.
          </p>
        </div>

        <div className="space-y-2">
          <StatTile
            label="Spread at horizon"
            value={finalSpread}
            status={finalSpread > 1 ? 'critical' : finalSpread > 0.4 ? 'warning' : 'good'}
            hint="ensemble standard deviation"
          />
          <StatTile
            label="Usable horizon"
            value={trustHorizon > 0 ? `${trustHorizon} steps` : `> ${horizon} steps`}
            mono={false}
            hint="before disagreement exceeds 0.5"
          />
          <StatTile
            label="Compounding bound"
            value={boundSeries[boundSeries.length - 1]?.y ?? 0}
            hint="ε(Lᴴ−1)/(L−1)"
          />
        </div>
      </div>

      <LineChart
        data={[
          { id: 'ensemble disagreement', data: spread },
          { id: 'theoretical bound ε(Lᴴ−1)/(L−1)', data: boundSeries },
        ]}
        height={195}
        xLegend="rollout horizon H"
        yLegend="error"
        dashed={['theoretical bound ε(Lᴴ−1)/(L−1)']}
        caption="Measured disagreement tracks the geometric bound — which is why doubling the horizon does far more than double the error."
      />
    </SimPanel>
  );
}

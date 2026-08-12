'use client';

import { useMemo, useState } from 'react';
import { LineChart } from '@/components/viz/LineChart';
import { StatTile } from '@/components/viz/StatTile';
import { SimPanel, Slider } from './SimControls';
import { mulberry32, gaussian } from '@/lib/rl/random';

/**
 * `ch15-randomization-wall` — why training on a distribution beats training on
 * a point estimate.
 *
 * A policy trained at one friction value is brittle: it peaks on the nominal
 * parameter and falls off a cliff either side. Widen the randomization range
 * and the peak drops but the curve flattens — the policy becomes robust
 * *because* it was never allowed to overfit one world. That trade is the
 * distributionally-robust objective the chapter derives, plotted.
 */
export function DomainRandomization() {
  const [range, setRange] = useState(0.25);
  const [realFriction, setRealFriction] = useState(1.35);

  const NOMINAL = 1.0;

  /**
   * Performance model: a policy trained over friction ~ U(1−r, 1+r) behaves
   * like a kernel-smoothed version of the narrow-training response. Wider
   * training ⇒ wider, lower competence bump.
   */
  const curves = useMemo(() => {
    const rng = mulberry32(11);
    const width = 0.06 + range * 0.9;
    const peak = 1 / (1 + range * 1.6);
    const xs = Array.from({ length: 121 }, (_, i) => 0.4 + i * 0.015);

    const perf = (x: number) => {
      const z = (x - NOMINAL) / width;
      const base = peak * Math.exp(-0.5 * z * z);
      return Math.max(0, Math.min(1, base + gaussian(rng, 0, 0.004)));
    };

    return [
      {
        id: `randomized ±${(range * 100).toFixed(0)}%`,
        data: xs.map((x) => ({ x, y: perf(x) })),
      },
      {
        id: 'trained on nominal only',
        data: xs.map((x) => {
          const z = (x - NOMINAL) / 0.06;
          return { x, y: Math.max(0, Math.exp(-0.5 * z * z)) };
        }),
      },
    ];
  }, [range]);

  const scoreAtReal = useMemo(() => {
    const width = 0.06 + range * 0.9;
    const peak = 1 / (1 + range * 1.6);
    const z = (realFriction - NOMINAL) / width;
    const randomized = peak * Math.exp(-0.5 * z * z);
    const zn = (realFriction - NOMINAL) / 0.06;
    const narrow = Math.exp(-0.5 * zn * zn);
    return { randomized, narrow };
  }, [range, realFriction]);

  const worstCase = useMemo(() => {
    const width = 0.06 + range * 0.9;
    const peak = 1 / (1 + range * 1.6);
    // Worst case over the plausible reality band [0.7, 1.6].
    const zs = [0.7, 1.6].map((x) => Math.abs((x - NOMINAL) / width));
    return peak * Math.exp(-0.5 * Math.max(...zs) ** 2);
  }, [range]);

  return (
    <SimPanel
      title="Domain randomization: trading peak for robustness"
      id="ch15-randomization-wall"
      subtitle="Success rate as a function of the real robot's friction, for policies trained on progressively wider parameter distributions."
      controls={
        <div className="grid gap-3 sm:grid-cols-2">
          <Slider
            label="Randomization range (± fraction)"
            value={range}
            min={0}
            max={0.6}
            step={0.01}
            onChange={setRange}
            format={(v) => `±${(v * 100).toFixed(0)}%`}
            hint="how much friction varies during training"
          />
          <Slider
            label="The real robot's friction"
            value={realFriction}
            min={0.5}
            max={1.7}
            step={0.01}
            onChange={setRealFriction}
            hint="unknown at training time — that is the whole problem"
          />
        </div>
      }
      caption="At ±0% the narrow policy is perfect at exactly 1.0 and useless at 1.35 — the classic sim-to-real failure, where a controller that looked flawless in simulation falls over on hardware. Widen the range and peak performance drops on purpose: you are buying insurance against a parameter you cannot measure. The worst-case number is the one that ships."
    >
      <div className="grid gap-3 lg:grid-cols-[1fr,215px]">
        <LineChart
          data={curves}
          height={260}
          xLegend="real friction coefficient"
          yLegend="success rate"
          yMin={0}
          yMax={1.05}
          dashed={['trained on nominal only']}
        />
        <div className="space-y-2">
          <StatTile
            label="Randomized policy"
            value={scoreAtReal.randomized}
            hint={`on the real robot (μ = ${realFriction.toFixed(2)})`}
            status={scoreAtReal.randomized > 0.5 ? 'good' : 'warning'}
          />
          <StatTile
            label="Nominal-only policy"
            value={scoreAtReal.narrow}
            hint="trained at μ = 1.00 exactly"
            status={scoreAtReal.narrow > 0.5 ? 'good' : 'critical'}
          />
          <StatTile
            label="Worst case over band"
            value={worstCase}
            hint="min over μ ∈ [0.7, 1.6] — the shippable number"
          />
        </div>
      </div>
    </SimPanel>
  );
}

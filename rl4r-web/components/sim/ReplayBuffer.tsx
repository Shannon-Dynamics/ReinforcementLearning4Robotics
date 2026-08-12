'use client';

import { useMemo, useState } from 'react';
import { LineChart } from '@/components/viz/LineChart';
import { StatTile } from '@/components/viz/StatTile';
import { SimPanel, Slider } from './SimControls';
import { useTheme } from '@/components/layout/ThemeProvider';
import { seriesColor } from '@/lib/theme';
import { mulberry32 } from '@/lib/rl/random';

/**
 * `ch09-replay-target` — the two pieces of surgery that made deep Q-learning work.
 *
 * Consecutive transitions from a robot are strongly correlated, which violates
 * the i.i.d. assumption every gradient method relies on; and a target computed
 * from the network being trained is a target that runs away as you chase it.
 * The replay buffer fixes the first, the target network the second. The reader
 * disables each and watches the training curve degrade in its characteristic way.
 */
export function ReplayBuffer() {
  const { mode } = useTheme();
  const [useReplay, setUseReplay] = useState(true);
  const [useTarget, setUseTarget] = useState(true);
  const [bufferSize, setBufferSize] = useState(2000);
  const [syncEvery, setSyncEvery] = useState(100);

  const { curves, stability, correlation } = useMemo(() => {
    const rng = mulberry32(17);
    const STEPS = 400;

    // A stylized learning-progress model: correlated batches slow convergence
    // and inflate variance; a moving target adds a divergence-prone oscillation.
    const batchCorrelation = useReplay ? Math.max(0.05, 32 / Math.sqrt(bufferSize)) : 0.92;
    const targetLag = useTarget ? syncEvery : 1;

    const loss: Array<{ x: number; y: number }> = [];
    const value: Array<{ x: number; y: number }> = [];

    let q = 0;
    let target = 0;
    let osc = 0;

    for (let t = 0; t < STEPS; t++) {
      if (t % targetLag === 0) target = q;

      // Chasing a target that moves with you produces growing oscillation.
      const chase = useTarget ? 0 : 0.045 * (q - target + 1);
      osc = 0.9 * osc + chase;

      const noise = (rng() - 0.5) * batchCorrelation * 2.2;
      const signal = (10 - q) * 0.045;
      q = q + signal + noise + osc;

      const err = Math.abs(10 - q) + Math.abs(osc) * 5;
      loss.push({ x: t, y: Math.min(err, 40) });
      value.push({ x: t, y: Math.max(-15, Math.min(30, q)) });
    }

    const tail = loss.slice(-80).map((p) => p.y);
    const mean = tail.reduce((a, b) => a + b, 0) / tail.length;
    const variance = tail.reduce((a, b) => a + (b - mean) ** 2, 0) / tail.length;

    return {
      curves: [
        { id: 'TD error magnitude', data: loss },
        { id: 'Q estimate (true value = 10)', data: value },
      ],
      stability: variance,
      correlation: batchCorrelation,
    };
  }, [useReplay, useTarget, bufferSize, syncEvery]);

  return (
    <SimPanel
      title="Replay and target networks as variance surgery"
      id="ch09-replay-target"
      subtitle="Switch each off and watch the characteristic failure appear: correlated batches add noise, a moving target adds oscillation."
      controls={
        <div className="space-y-2.5">
          <div className="flex flex-wrap gap-3">
            <label className="flex items-center gap-2 text-[12px] text-ink-secondary">
              <input
                type="checkbox"
                checked={useReplay}
                onChange={(e) => setUseReplay(e.target.checked)}
                className="accent-[var(--series-1)]"
              />
              Experience replay
            </label>
            <label className="flex items-center gap-2 text-[12px] text-ink-secondary">
              <input
                type="checkbox"
                checked={useTarget}
                onChange={(e) => setUseTarget(e.target.checked)}
                className="accent-[var(--series-1)]"
              />
              Target network
            </label>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <Slider
              label="Replay buffer size"
              value={bufferSize}
              min={100}
              max={20000}
              step={100}
              onChange={setBufferSize}
              format={(v) => (v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(0))}
              hint="larger buffer ⇒ less correlated batches"
            />
            <Slider
              label="Target sync interval"
              value={syncEvery}
              min={1}
              max={500}
              step={1}
              onChange={setSyncEvery}
              format={(v) => `${v.toFixed(0)} steps`}
              hint="longer ⇒ more stable, slower to track"
            />
          </div>
        </div>
      }
      caption="Turn off replay and the estimate rattles: consecutive robot transitions are nearly identical, so each gradient step sees almost the same data and the batch carries far less information than its size suggests. Turn off the target network and the estimate oscillates and can run away: you are regressing toward a quantity that moves every time you update."
    >
      <div className="grid gap-3 lg:grid-cols-[1fr,205px]">
        <LineChart
          data={curves}
          height={255}
          xLegend="gradient step"
          yLegend="magnitude"
        />
        <div className="space-y-2">
          <StatTile
            label="Batch correlation"
            value={correlation}
            status={correlation > 0.5 ? 'critical' : correlation > 0.2 ? 'warning' : 'good'}
            hint={useReplay ? 'sampled uniformly from buffer' : 'consecutive transitions'}
          />
          <StatTile
            label="Tail variance"
            value={stability}
            status={stability > 8 ? 'critical' : stability > 2 ? 'warning' : 'good'}
            hint="spread over the last 80 steps"
          />
          <div className="rounded-lg border border-hairline p-2.5">
            <p className="mb-1.5 text-[10.5px] font-semibold uppercase tracking-[0.06em] text-ink-muted">
              Buffer occupancy
            </p>
            <svg width="100%" height={40} viewBox="0 0 180 40" role="img" aria-label="Replay buffer sampling illustration">
              {Array.from({ length: 24 }, (_, i) => {
                const recent = i > 19;
                return (
                  <rect
                    key={i}
                    x={2 + i * 7.4}
                    y={useReplay ? 6 : recent ? 6 : 20}
                    width={6}
                    height={useReplay ? 28 : recent ? 28 : 14}
                    rx={2}
                    fill={
                      useReplay || recent ? seriesColor(0, mode) : 'var(--baseline)'
                    }
                    opacity={useReplay ? 0.55 + (i % 5) * 0.09 : recent ? 1 : 0.25}
                  />
                );
              })}
            </svg>
            <p className="mt-1 text-[10.5px] leading-snug text-ink-muted">
              {useReplay
                ? 'Batches drawn from across the whole history.'
                : 'Only the newest transitions are seen — and they all look alike.'}
            </p>
          </div>
        </div>
      </div>
    </SimPanel>
  );
}

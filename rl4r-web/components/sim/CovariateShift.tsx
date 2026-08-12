'use client';

import { useMemo, useState } from 'react';
import { SimPanel, Slider } from './SimControls';
import { StatTile } from '@/components/viz/StatTile';
import { LineChart } from '@/components/viz/LineChart';
import { mulberry32, gaussian } from '@/lib/rl/random';
import { useTheme } from '@/components/layout/ThemeProvider';
import { seriesColor } from '@/lib/theme';

/**
 * `ch16-covariate-drift` — why behaviour cloning fails in a way supervised
 * learning does not.
 *
 * The cloned policy makes a small mistake, which moves it slightly off the
 * demonstrated distribution, where it was never trained — so it makes a bigger
 * mistake. Errors compound quadratically in the horizon (Ross & Bagnell's
 * T²ε bound) rather than linearly. The widget draws the demonstrated lane, the
 * cloned rollout drifting off it, and the growing error cone.
 */
export function CovariateShift() {
  const { mode } = useTheme();
  const [epsilon, setEpsilon] = useState(0.04);
  const [horizon, setHorizon] = useState(120);
  const [dagger, setDagger] = useState(false);

  const W = 520;
  const H = 210;

  const { path, lane, cone, finalError } = useMemo(() => {
    const rng = mulberry32(3);
    const laneY = (x: number) => H / 2 + 34 * Math.sin((x / W) * Math.PI * 2.1);

    const lanePts = Array.from({ length: 90 }, (_, i) => {
      const x = (i / 89) * W;
      return { x, y: laneY(x) };
    });

    // Cloned rollout: per-step error ε, and — without DAgger — an extra
    // penalty that grows with how far off-distribution the state already is.
    let y = laneY(0);
    const pts: Array<{ x: number; y: number }> = [{ x: 0, y }];
    for (let i = 1; i <= horizon; i++) {
      const x = (i / horizon) * W;
      const target = laneY(x);
      const deviation = Math.abs(y - target);
      // DAgger re-queries the expert off-distribution, so the compounding term
      // vanishes and the bound becomes linear in T.
      const compounding = dagger ? 0 : (deviation / 30) * epsilon * 26;
      const noise = gaussian(rng, 0, epsilon * 12) + compounding * (y > target ? 1 : -1);
      y = y + (target - y) * (dagger ? 0.45 : 0.16) + noise;
      y = Math.max(6, Math.min(H - 6, y));
      pts.push({ x, y });
    }

    const conePts = Array.from({ length: 60 }, (_, i) => {
      const t = i / 59;
      const x = t * W;
      const growth = dagger ? epsilon * horizon * t * 1.2 : epsilon * Math.pow(horizon * t, 1.55) * 0.09;
      return { x, y: laneY(x), spread: Math.min(H / 2 - 4, growth) };
    });

    return {
      path: pts,
      lane: lanePts,
      cone: conePts,
      finalError: Math.abs(pts[pts.length - 1].y - laneY(W)),
    };
  }, [epsilon, horizon, dagger]);

  const bounds = useMemo(() => {
    const ts = Array.from({ length: 40 }, (_, i) => (i + 1) * 8);
    return [
      { id: 'behaviour cloning: O(εT²)', data: ts.map((t) => ({ x: t, y: epsilon * t * t })) },
      { id: 'DAgger: O(εT)', data: ts.map((t) => ({ x: t, y: epsilon * t })) },
    ];
  }, [epsilon]);

  const pathD = `M${path.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' L')}`;
  const laneD = `M${lane.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' L')}`;
  const coneD =
    `M${cone.map((p) => `${p.x.toFixed(1)},${(p.y - p.spread).toFixed(1)}`).join(' L')}` +
    ` L${cone
      .slice()
      .reverse()
      .map((p) => `${p.x.toFixed(1)},${(p.y + p.spread).toFixed(1)}`)
      .join(' L')} Z`;

  return (
    <SimPanel
      title="Covariate shift: the cloned robot drifts"
      id="ch16-covariate-drift"
      subtitle="Rusty clones a demonstrated lane. Each small error moves him somewhere the demonstrations never went."
      controls={
        <div className="flex flex-wrap items-end gap-4">
          <Slider
            className="w-48"
            label="Per-step error ε"
            value={epsilon}
            min={0.005}
            max={0.12}
            step={0.005}
            onChange={setEpsilon}
            format={(v) => v.toFixed(3)}
            hint="supervised accuracy on the demo distribution"
          />
          <Slider
            className="w-44"
            label="Horizon T"
            value={horizon}
            min={20}
            max={300}
            step={10}
            onChange={setHorizon}
            format={(v) => v.toFixed(0)}
          />
          <label className="flex items-center gap-2 text-[12px] text-ink-secondary">
            <input
              type="checkbox"
              checked={dagger}
              onChange={(e) => setDagger(e.target.checked)}
              className="accent-[var(--series-1)]"
            />
            Use DAgger (re-query the expert off-distribution)
          </label>
        </div>
      }
      caption="Behaviour cloning's error is not ε — it is ε multiplied by the horizon, twice over, because each mistake changes the distribution of states the policy will face next. Tick DAgger and the cone collapses to a linear band: asking the expert what to do in the states the *learner* actually visits is what breaks the feedback loop."
    >
      <div className="grid gap-3 lg:grid-cols-[1fr,320px]">
        <div>
          <svg
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            className="max-w-full rounded-lg"
            style={{ background: 'var(--surface-sunken)' }}
            role="img"
            aria-label="Demonstrated lane with a cloned rollout drifting away inside a growing error cone"
          >
            <path d={coneD} fill={seriesColor(0, mode)} opacity={0.13} />
            <path
              d={laneD}
              fill="none"
              stroke={seriesColor(0, mode)}
              strokeWidth={2}
              strokeDasharray="5 4"
            />
            <path d={pathD} fill="none" stroke={seriesColor(1, mode)} strokeWidth={2.5} />
            <circle
              cx={path[path.length - 1].x}
              cy={path[path.length - 1].y}
              r={6}
              fill={seriesColor(1, mode)}
              stroke="var(--surface-1)"
              strokeWidth={2}
            />
            <text x={8} y={16} fontSize={10} fill="var(--text-muted)">
              — — demonstrated lane · —— cloned rollout · shaded = error cone
            </text>
          </svg>
          <div className="mt-2 grid grid-cols-2 gap-2">
            <StatTile
              label="Final deviation"
              value={finalError}
              unit="px"
              status={finalError > 40 ? 'critical' : finalError > 15 ? 'warning' : 'good'}
            />
            <StatTile
              label="Error bound"
              value={dagger ? epsilon * horizon : epsilon * horizon * horizon}
              hint={dagger ? 'O(εT) — linear' : 'O(εT²) — quadratic'}
            />
          </div>
        </div>

        <LineChart
          data={bounds}
          height={250}
          xLegend="horizon T"
          yLegend="worst-case regret"
          caption="Ross & Bagnell (2011): cloning's quadratic bound vs DAgger's linear one."
        />
      </div>
    </SimPanel>
  );
}

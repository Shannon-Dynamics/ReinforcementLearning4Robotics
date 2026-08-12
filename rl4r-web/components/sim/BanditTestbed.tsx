'use client';

import { useMemo, useState } from 'react';
import {
  EpsilonGreedy,
  GradientBandit,
  ThompsonSampling,
  Ucb1,
  runBanditExperiment,
  type BanditPolicy,
} from '@/lib/rl/bandits';
import { mulberry32 } from '@/lib/rl/random';
import { LineChart } from '@/components/viz/LineChart';
import { SimPanel, Slider } from './SimControls';
import { seriesColor } from '@/lib/theme';
import { useTheme } from '@/components/layout/ThemeProvider';

interface Algo {
  key: string;
  label: string;
  make: (k: number, eps: number) => BanditPolicy;
}

const ALGOS: Algo[] = [
  { key: 'greedy', label: 'greedy (ε=0)', make: (k) => new EpsilonGreedy(k, 0) },
  { key: 'eps', label: 'ε-greedy', make: (k, eps) => new EpsilonGreedy(k, eps) },
  {
    key: 'optimistic',
    label: 'optimistic greedy (Q₁=5)',
    make: (k) => new EpsilonGreedy(k, 0, 0.1, 5),
  },
  { key: 'ucb', label: 'UCB (c=2)', make: (k) => new Ucb1(k, 2) },
  { key: 'gradient', label: 'gradient bandit', make: (k) => new GradientBandit(k, 0.1) },
  { key: 'thompson', label: 'Thompson sampling', make: (k) => new ThompsonSampling(k) },
];

/**
 * `ch03-bandit-testbed` — the 10-armed testbed as a live race.
 *
 * The reader picks algorithms and ε, and the three panels answer the three
 * questions the chapter's math poses: how much reward (average reward), how
 * often right (% optimal action), and how much was lost to exploration
 * (cumulative regret — the quantity the UCB bound actually bounds).
 */
export function BanditTestbed() {
  const { mode } = useTheme();
  const [epsilon, setEpsilon] = useState(0.1);
  const [runs, setRuns] = useState(120);
  const [selected, setSelected] = useState<string[]>(['greedy', 'eps', 'ucb']);

  const results = useMemo(() => {
    return ALGOS.filter((a) => selected.includes(a.key)).map((a) =>
      runBanditExperiment((k) => a.make(k, epsilon), {
        runs,
        steps: 1000,
        k: 10,
        seed: 42,
        rngFactory: mulberry32,
      }),
    );
  }, [selected, epsilon, runs]);

  const labels = useMemo(
    () => ALGOS.filter((a) => selected.includes(a.key)).map((a) => a.label),
    [selected],
  );

  const rewardSeries = results.map((r, i) => ({
    id: labels[i],
    data: r.avgReward.map((y, x) => ({ x: x + 1, y })).filter((_, x) => x % 5 === 0),
  }));
  const optimalSeries = results.map((r, i) => ({
    id: labels[i],
    data: r.optimalPct.map((y, x) => ({ x: x + 1, y })).filter((_, x) => x % 5 === 0),
  }));
  const regretSeries = results.map((r, i) => ({
    id: labels[i],
    data: r.regret.map((y, x) => ({ x: x + 1, y })).filter((_, x) => x % 5 === 0),
  }));

  return (
    <SimPanel
      title="The 10-armed testbed"
      id="ch03-bandit-testbed"
      subtitle="Each curve averages independent runs on fresh bandit problems with q*(a) ~ N(0,1), rewards ~ N(q*(a), 1)."
      controls={
        <div className="space-y-2.5">
          <div className="flex flex-wrap gap-1.5">
            {ALGOS.map((a, i) => {
              const on = selected.includes(a.key);
              return (
                <button
                  key={a.key}
                  type="button"
                  onClick={() =>
                    setSelected((s) =>
                      s.includes(a.key) ? s.filter((x) => x !== a.key) : [...s, a.key],
                    )
                  }
                  aria-pressed={on}
                  className="flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11.5px] font-medium transition-colors"
                  style={{
                    borderColor: on ? seriesColor(selected.indexOf(a.key), mode) : 'var(--border-hairline)',
                    background: on
                      ? `color-mix(in srgb, ${seriesColor(selected.indexOf(a.key), mode)} 12%, transparent)`
                      : 'transparent',
                    color: on ? 'var(--text-primary)' : 'var(--text-secondary)',
                  }}
                >
                  <span
                    aria-hidden
                    className="inline-block h-2 w-2 rounded-full"
                    style={{
                      background: on
                        ? seriesColor(selected.indexOf(a.key), mode)
                        : 'var(--text-muted)',
                    }}
                  />
                  {a.label}
                </button>
              );
            })}
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <Slider
              label="ε (for ε-greedy)"
              value={epsilon}
              min={0}
              max={0.5}
              step={0.01}
              onChange={setEpsilon}
              hint="constant ε ⇒ linear regret forever"
            />
            <Slider
              label="Independent runs"
              value={runs}
              min={20}
              max={400}
              step={20}
              onChange={setRuns}
              format={(v) => v.toFixed(0)}
              hint="more runs ⇒ smoother averages"
            />
          </div>
        </div>
      }
      caption="Watch the greedy curve plateau below the others: with ε = 0 it locks onto whichever arm looked good first. Raise ε and early reward drops but the % optimal keeps climbing — exploration paid for in the short run, recovered in the long one. UCB's regret curve bends toward logarithmic; ε-greedy's stays a straight line."
    >
      <div className="space-y-3">
        <LineChart
          data={rewardSeries}
          height={230}
          xLegend="steps"
          yLegend="average reward"
          caption="Average reward per step."
        />
        <div className="grid gap-3 lg:grid-cols-2">
          <LineChart
            data={optimalSeries}
            height={210}
            xLegend="steps"
            yLegend="% optimal action"
            yMin={0}
            yMax={100}
            caption="How often the best arm was chosen."
          />
          <LineChart
            data={regretSeries}
            height={210}
            xLegend="steps"
            yLegend="cumulative regret"
            caption="Cumulative regret — the quantity UCB's bound controls."
          />
        </div>
      </div>
    </SimPanel>
  );
}

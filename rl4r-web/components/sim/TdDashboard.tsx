'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { GridWorld } from '@/lib/rl/gridworld';
import { TabularLearner, smooth, type EpisodeStats, type LearnerKind } from '@/lib/rl/td';
import { mulberry32 } from '@/lib/rl/random';
import { GridWorldCanvas, ValueLegend } from './GridWorldCanvas';
import { Segmented, SimControls, SimPanel, Slider } from './SimControls';
import { StatTile } from '@/components/viz/StatTile';
import { LineChart } from '@/components/viz/LineChart';

/**
 * `ch06-train-live` — model-free control, learning in front of you.
 *
 * Rusty starts knowing nothing: no map, no transition probabilities, only the
 * rewards that arrive after he acts. The heatmap fills in from the dock
 * backwards as the TD error propagates value one step at a time — the single
 * most important animation in Part I, because it shows *bootstrapping* doing
 * its work rather than describing it.
 */
export function TdDashboard() {
  const [kind, setKind] = useState<LearnerKind>('qlearning');
  const [alpha, setAlpha] = useState(0.15);
  const [epsilon, setEpsilon] = useState(0.2);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(12);
  const [history, setHistory] = useState<EpisodeStats[]>([]);
  const [tick, setTick] = useState(0);

  const env = useMemo(() => new GridWorld(), []);
  const rng = useRef(mulberry32(7));
  const learner = useRef<TabularLearner | null>(null);
  const raf = useRef<number | null>(null);
  const acc = useRef(0);

  const rebuild = useCallback(() => {
    rng.current = mulberry32(7);
    learner.current = new TabularLearner(env, kind, {
      alpha,
      epsilon,
      epsilonDecay: 0.995,
      epsilonMin: 0.02,
      lambda: 0.9,
    });
    setHistory([]);
    setTick((t) => t + 1);
  }, [env, kind, alpha, epsilon]);

  useEffect(() => {
    rebuild();
    setPlaying(false);
  }, [rebuild]);

  const runEpisode = useCallback(() => {
    if (!learner.current) return;
    const stats = learner.current.stepEpisode(rng.current);
    setHistory((h) => [...h, stats]);
    setTick((t) => t + 1);
  }, []);

  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const loop = (now: number) => {
      acc.current += (now - last) * (speed / 1000);
      last = now;
      let guard = 0;
      while (acc.current >= 1 && guard < 12) {
        acc.current -= 1;
        guard += 1;
        runEpisode();
      }
      raf.current = requestAnimationFrame(loop);
    };
    raf.current = requestAnimationFrame(loop);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [playing, speed, runEpisode]);

  const V = learner.current?.valueFunction();
  const policy = learner.current?.greedyPolicy();
  const last = history[history.length - 1];

  const vRange = useMemo(() => {
    if (!V) return { lo: 0, hi: 1 };
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of env.states) {
      if (env.isTerminal(s)) continue;
      lo = Math.min(lo, V[s]);
      hi = Math.max(hi, V[s]);
    }
    return { lo: Number.isFinite(lo) ? lo : 0, hi: Number.isFinite(hi) ? hi : 1 };
    // `tick` forces a recompute as the tables change in place.
  }, [V, env, tick]);

  const curves = useMemo(() => {
    const stride = Math.max(1, Math.floor(history.length / 240));
    const returns = smooth(history.map((h) => h.totalReward), 20);
    const deltas = smooth(history.map((h) => h.meanAbsDelta), 20);
    return {
      returns: [
        {
          id: 'episode return (smoothed)',
          data: returns.map((y, x) => ({ x: x + 1, y })).filter((_, i) => i % stride === 0),
        },
      ],
      deltas: [
        {
          id: 'mean |δ|',
          data: deltas.map((y, x) => ({ x: x + 1, y })).filter((_, i) => i % stride === 0),
        },
      ],
    };
  }, [history]);

  return (
    <SimPanel
      title="Learning without a model"
      id="ch06-train-live"
      subtitle="Rusty has no map. Every value on the heatmap was bootstrapped from experience alone."
      controls={
        <div className="space-y-2.5">
          <div className="flex flex-wrap items-end gap-4">
            <Segmented
              label="Algorithm"
              value={kind}
              onChange={(v) => setKind(v)}
              options={[
                { value: 'qlearning', label: 'Q-learning' },
                { value: 'sarsa', label: 'SARSA' },
                { value: 'expected-sarsa', label: 'Expected SARSA' },
                { value: 'double-q', label: 'Double Q' },
              ]}
            />
            <SimControls
              playing={playing}
              onPlayPause={() => setPlaying((p) => !p)}
              onStep={runEpisode}
              onReset={() => {
                setPlaying(false);
                rebuild();
              }}
              speed={speed}
              onSpeedChange={setSpeed}
            />
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <Slider
              label="Learning rate α"
              value={alpha}
              min={0.01}
              max={0.6}
              step={0.01}
              onChange={setAlpha}
              hint="too large and the estimates rattle; too small and they crawl"
            />
            <Slider
              label="Initial exploration ε"
              value={epsilon}
              min={0}
              max={0.6}
              step={0.02}
              onChange={setEpsilon}
              hint="decays ×0.995 per episode toward 0.02"
            />
          </div>
        </div>
      }
      caption="Press play and watch value bleed backwards from the dock. Early on the arrows are nonsense because every Q is zero and ties break at random; the policy sharpens only where the TD error has actually reached. Set ε = 0 and the run usually stalls — with no exploration Rusty commits to the first corridor that ever paid off."
    >
      <div className="grid gap-4 lg:grid-cols-[auto,1fr]">
        <div>
          {V && policy ? (
            <GridWorldCanvas
              env={env}
              V={V}
              policy={policy}
              path={last?.path}
              cellSize={32}
            />
          ) : null}
          <div className="mt-2">
            <ValueLegend min={vRange.lo} max={vRange.hi} />
          </div>
        </div>

        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <StatTile label="Episodes" value={history.length} />
            <StatTile
              label="Last return"
              value={last?.totalReward ?? 0}
              status={last && last.totalReward > 0 ? 'good' : undefined}
              hint={last ? `${last.steps} steps` : 'not started'}
            />
            <StatTile
              label="Mean |δ| (last ep.)"
              value={last?.meanAbsDelta ?? 0}
              hint="magnitude of the learning signal"
            />
            <StatTile
              label="Current ε"
              value={last?.epsilon ?? epsilon}
              hint="exploration remaining"
            />
          </div>
          <LineChart
            data={curves.returns}
            height={175}
            xLegend="episode"
            yLegend="return"
            caption="Smoothed episode return."
          />
          <LineChart
            data={curves.deltas}
            height={165}
            xLegend="episode"
            yLegend="mean |δ|"
            caption="The TD error decays as the value function becomes self-consistent."
          />
        </div>
      </div>
    </SimPanel>
  );
}

'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { GridWorld } from '@/lib/rl/gridworld';
import { collect, policyIteration, valueIteration, type DpSnapshot } from '@/lib/rl/dp';
import { policyEvaluation } from '@/lib/rl/dp';
import { GridWorldCanvas, ValueLegend } from './GridWorldCanvas';
import { Segmented, SimControls, SimPanel, Slider } from './SimControls';
import { StatTile } from '@/components/viz/StatTile';
import { LineChart } from '@/components/viz/LineChart';

type Mode = 'policy-eval' | 'policy-iteration' | 'value-iteration';

/**
 * `ch05-gpi-dashboard` — the book's signature widget.
 *
 * Generalized policy iteration made watchable: the value heatmap and the greedy
 * policy arrows update sweep by sweep on Rusty's warehouse, with the
 * convergence measure Δ and the implied suboptimality bound streaming beside
 * them. Every quantity on screen is one the math named.
 */
export function GpiDashboard() {
  const [mode, setMode] = useState<Mode>('value-iteration');
  const [gamma, setGamma] = useState(0.95);
  const [pSlip, setPSlip] = useState(0.2);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(8);
  const [index, setIndex] = useState(0);
  const raf = useRef<number | null>(null);
  const acc = useRef(0);

  const env = useMemo(() => new GridWorld(undefined, { gamma, pSlip }), [gamma, pSlip]);

  const snapshots: DpSnapshot[] = useMemo(() => {
    if (mode === 'value-iteration') return collect(valueIteration(env, 1e-4, 200));
    if (mode === 'policy-iteration') return collect(policyIteration(env, 1e-4, 30));
    // Policy evaluation of a fixed "always east" policy — the widget's warm-up.
    const fixed = new Int8Array(env.nStates).fill(-1);
    for (const s of env.states) if (!env.isTerminal(s)) fixed[s] = 1;
    return collect(policyEvaluation(env, fixed, 1e-4, 200));
  }, [env, mode]);

  useEffect(() => setIndex(0), [mode, gamma, pSlip]);

  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const tick = (now: number) => {
      acc.current += (now - last) * (speed / 1000);
      last = now;
      if (acc.current >= 1) {
        const advance = Math.floor(acc.current);
        acc.current -= advance;
        setIndex((i) => {
          const next = i + advance;
          if (next >= snapshots.length - 1) {
            setPlaying(false);
            return snapshots.length - 1;
          }
          return next;
        });
      }
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [playing, speed, snapshots.length]);

  const snap = snapshots[Math.min(index, snapshots.length - 1)];
  const vRange = useMemo(() => {
    if (!snap) return { lo: 0, hi: 1 };
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of env.states) {
      if (env.isTerminal(s)) continue;
      lo = Math.min(lo, snap.V[s]);
      hi = Math.max(hi, snap.V[s]);
    }
    return { lo: Number.isFinite(lo) ? lo : 0, hi: Number.isFinite(hi) ? hi : 1 };
  }, [snap, env]);

  const deltaSeries = useMemo(
    () => [
      {
        id: 'Δ (max value change)',
        data: snapshots
          .slice(0, index + 1)
          .map((s, i) => ({ x: i + 1, y: Math.max(s.delta, 1e-6) })),
      },
    ],
    [snapshots, index],
  );

  if (!snap) return null;

  return (
    <SimPanel
      title="Generalized policy iteration, sweep by sweep"
      id="ch05-gpi-dashboard"
      subtitle="Rusty's warehouse: value heatmap + greedy policy arrows, with the convergence measure the theory names."
      controls={
        <div className="space-y-2.5">
          <div className="flex flex-wrap items-end gap-4">
            <Segmented
              label="Algorithm"
              value={mode}
              onChange={(v) => setMode(v)}
              options={[
                { value: 'policy-eval', label: 'Policy evaluation' },
                { value: 'policy-iteration', label: 'Policy iteration' },
                { value: 'value-iteration', label: 'Value iteration' },
              ]}
            />
            <SimControls
              playing={playing}
              onPlayPause={() => setPlaying((p) => !p)}
              onStep={() => setIndex((i) => Math.min(i + 1, snapshots.length - 1))}
              onReset={() => {
                setPlaying(false);
                setIndex(0);
              }}
              speed={speed}
              onSpeedChange={setSpeed}
            />
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <Slider
              label="Discount γ"
              value={gamma}
              min={0.5}
              max={0.995}
              step={0.005}
              onChange={setGamma}
              hint={`effective horizon ≈ ${(1 / (1 - gamma)).toFixed(0)} steps`}
            />
            <Slider
              label="Slip probability p_slip"
              value={pSlip}
              min={0}
              max={0.6}
              step={0.05}
              onChange={setPSlip}
              hint="lateral slip splits evenly"
            />
            <Slider
              label="Sweep"
              value={index}
              min={0}
              max={Math.max(snapshots.length - 1, 1)}
              step={1}
              onChange={(v) => {
                setPlaying(false);
                setIndex(v);
              }}
              format={(v) => `${v + 1} / ${snapshots.length}`}
            />
          </div>
        </div>
      }
      caption="Drag γ toward 1 and watch value spread further from the dock before decaying — the effective horizon 1/(1−γ) made visible. Raise p_slip and the optimal policy stops hugging the shelves, because a slip into a shelf costs −10."
    >
      <div className="grid gap-4 lg:grid-cols-[auto,1fr]">
        <div>
          <GridWorldCanvas
            env={env}
            V={snap.V}
            policy={snap.policy}
            showPolicy
            cellSize={34}
          />
          <div className="mt-2">
            <ValueLegend min={vRange.lo} max={vRange.hi} />
          </div>
        </div>

        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <StatTile
              label="Sweep"
              value={`${snap.sweep}`}
              hint={snap.phase === 'improvement' ? 'policy improvement' : 'policy evaluation'}
            />
            <StatTile
              label="Δ = max|Vₖ₊₁ − Vₖ|"
              value={snap.delta}
              status={snap.delta < 1e-4 ? 'good' : undefined}
              hint={snap.delta < 1e-4 ? 'converged' : 'still changing'}
            />
            <StatTile
              label="Suboptimality bound"
              value={snap.suboptimalityBound}
              hint="2γΔ/(1−γ) ≥ ‖v_π − v*‖∞"
            />
            <StatTile
              label="States swept"
              value={env.states.length}
              hint={`${env.rows}×${env.cols} grid minus shelves`}
            />
          </div>

          <LineChart
            data={deltaSeries}
            height={190}
            xLegend="sweep"
            yLegend="Δ (log-ish)"
            caption="Δ contracts geometrically — the γ-contraction of the Bellman operator, measured."
          />
        </div>
      </div>
    </SimPanel>
  );
}

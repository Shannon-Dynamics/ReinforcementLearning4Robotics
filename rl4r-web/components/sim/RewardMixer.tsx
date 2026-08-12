'use client';

import { useMemo, useState } from 'react';
import { SimPanel, Slider } from './SimControls';
import { StatTile } from '@/components/viz/StatTile';
import { BarChart } from '@/components/viz/BarChart';
import { useTheme } from '@/components/layout/ThemeProvider';
import { seriesColor } from '@/lib/theme';

interface Term {
  key: string;
  label: string;
  formula: string;
  units: string;
  description: string;
  /** How this term shapes the resulting gait, used to caricature the outcome. */
  effect: (w: number) => { stride: number; height: number; effort: number; stability: number };
}

/**
 * `ch18-reward-mixer` — the reward function as an engineering artifact.
 *
 * Locomotion rewards are sums of five or six terms whose weights decide the
 * gait far more than the algorithm does. The reader slides the weights and
 * sees both the dimensional bookkeeping (every term must be commensurate) and
 * a caricature of the gait that would emerge.
 */
const TERMS: Term[] = [
  {
    key: 'velocity',
    label: 'Velocity tracking',
    formula: 'exp(−‖v_xy − v*_xy‖² / σ)',
    units: 'dimensionless',
    description: 'The task itself: follow the commanded velocity.',
    effect: (w) => ({ stride: 0.4 + w * 0.6, height: 0, effort: w * 0.5, stability: 0 }),
  },
  {
    key: 'effort',
    label: 'Torque penalty',
    formula: '−‖τ‖²',
    units: 'N²·m²',
    description: 'Discourages slamming the actuators; too high and the robot refuses to move.',
    effect: (w) => ({ stride: -w * 0.5, height: -w * 0.3, effort: -w, stability: w * 0.2 }),
  },
  {
    key: 'airtime',
    label: 'Foot air time',
    formula: 'Σ_f (t_air,f − 0.5)',
    units: 's',
    description: 'Rewards deliberate swing phases — the classic cure for shuffling.',
    effect: (w) => ({ stride: w * 0.7, height: w * 0.9, effort: w * 0.4, stability: -w * 0.2 }),
  },
  {
    key: 'orientation',
    label: 'Base orientation',
    formula: '−‖g_proj,xy‖²',
    units: 'dimensionless',
    description: 'Keeps the body level; the main thing standing between you and a faceplant.',
    effect: (w) => ({ stride: -w * 0.1, height: 0, effort: w * 0.1, stability: w }),
  },
  {
    key: 'slip',
    label: 'Foot slip',
    formula: '−Σ_f ‖v_f‖² · 1[contact]',
    units: 'm²/s²',
    description: 'Penalizes feet sliding while loaded — the difference between walking and skating.',
    effect: (w) => ({ stride: -w * 0.2, height: w * 0.2, effort: w * 0.2, stability: w * 0.6 }),
  },
];

export function RewardMixer() {
  const { mode } = useTheme();
  const [weights, setWeights] = useState<Record<string, number>>({
    velocity: 1.0,
    effort: 0.15,
    airtime: 0.3,
    orientation: 0.5,
    slip: 0.2,
  });

  const gait = useMemo(() => {
    let stride = 0;
    let height = 0;
    let effort = 0;
    let stability = 0;
    for (const t of TERMS) {
      const e = t.effect(weights[t.key]);
      stride += e.stride;
      height += e.height;
      effort += e.effort;
      stability += e.stability;
    }
    return {
      stride: Math.max(0.05, Math.min(1, stride)),
      height: Math.max(0.02, Math.min(1, height)),
      effort: Math.max(0, Math.min(1, effort + 0.4)),
      stability: Math.max(0, Math.min(1, stability + 0.35)),
    };
  }, [weights]);

  const verdict = useMemo(() => {
    if (weights.velocity < 0.3) return { text: 'Stands still — no incentive to move', status: 'critical' as const };
    if (weights.effort > 0.7) return { text: 'Creeps — torque penalty dominates the task', status: 'critical' as const };
    if (weights.orientation < 0.15) return { text: 'Unstable — body pitches and falls', status: 'critical' as const };
    if (weights.airtime > 0.8) return { text: 'Prancing — exaggerated swing, wasted energy', status: 'warning' as const };
    if (weights.slip < 0.05) return { text: 'Skating — feet slide under load', status: 'warning' as const };
    return { text: 'Plausible trotting gait', status: 'good' as const };
  }, [weights]);

  const contributions = TERMS.map((t) => ({
    id: t.label,
    value: Math.abs(weights[t.key]),
  }));

  return (
    <SimPanel
      title="Reward anatomy: the weights decide the gait"
      id="ch18-reward-mixer"
      subtitle="A locomotion reward is a weighted sum of terms with different physical units. The weights are the design."
      controls={
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {TERMS.map((t) => (
            <Slider
              key={t.key}
              label={t.label}
              value={weights[t.key]}
              min={0}
              max={1}
              step={0.05}
              onChange={(v) => setWeights((w) => ({ ...w, [t.key]: v }))}
              hint={t.units}
            />
          ))}
        </div>
      }
      caption="Every term carries different units — seconds, N²m², dimensionless — so the weights are not preferences but unit conversions that also express preferences. That is why locomotion reward tuning is the reproducibility problem of the field: publish the weights or publish nothing."
    >
      <div className="grid gap-4 lg:grid-cols-[300px,1fr]">
        <div>
          {/* Gait caricature */}
          <svg
            width={300}
            height={170}
            viewBox="0 0 300 170"
            className="max-w-full rounded-lg"
            style={{ background: 'var(--surface-sunken)' }}
            role="img"
            aria-label={`Caricature of the resulting gait: ${verdict.text}`}
          >
            <line x1={20} y1={130} x2={280} y2={130} stroke="var(--baseline)" strokeWidth={2} />
            {/* Body — height and pitch respond to the weights */}
            <rect
              x={110}
              y={70 - gait.stability * 8}
              width={80}
              height={26}
              rx={6}
              fill={seriesColor(0, mode)}
              transform={`rotate(${(1 - gait.stability) * 14} 150 83)`}
            />
            {/* Legs: stride length and foot clearance */}
            {[0, 1, 2, 3].map((i) => {
              const phase = (i % 2) * Math.PI;
              const x = 122 + (i % 2) * 52 + Math.floor(i / 2) * 4;
              const swing = Math.sin(phase) * gait.stride * 22;
              const lift = Math.abs(Math.sin(phase)) * gait.height * 22;
              return (
                <line
                  key={i}
                  x1={x}
                  y1={96 - gait.stability * 8}
                  x2={x + swing}
                  y2={130 - lift}
                  stroke={seriesColor(0, mode)}
                  strokeWidth={4}
                  strokeLinecap="round"
                  opacity={i < 2 ? 1 : 0.55}
                />
              );
            })}
            <text x={150} y={158} textAnchor="middle" fontSize={11} fill="var(--text-muted)">
              stride {(gait.stride * 100).toFixed(0)}% · clearance {(gait.height * 100).toFixed(0)}%
            </text>
          </svg>

          <div className="mt-2 space-y-2">
            <StatTile
              label="Predicted outcome"
              value={verdict.text}
              mono={false}
              status={verdict.status}
            />
            <StatTile
              label="Energetic cost"
              value={gait.effort}
              hint="normalized torque expenditure"
            />
          </div>
        </div>

        <div className="space-y-3">
          <BarChart
            data={contributions}
            layout="horizontal"
            height={195}
            xLegend="weight magnitude"
            title="Reward-term weights"
          />
          <div className="space-y-1.5">
            {TERMS.map((t) => (
              <div
                key={t.key}
                className="rounded-lg border border-hairline bg-surface px-3 py-2 text-[12px]"
              >
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <span className="font-semibold text-ink">{t.label}</span>
                  <code className="font-mono text-[11px] text-ink-secondary">{t.formula}</code>
                </div>
                <p className="mt-0.5 text-[11.5px] leading-snug text-ink-muted">{t.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </SimPanel>
  );
}

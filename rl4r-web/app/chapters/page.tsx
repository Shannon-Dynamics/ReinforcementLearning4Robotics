import Link from 'next/link';
import type { Metadata } from 'next';
import { CHAPTERS, PARTS } from '@/lib/chapters';

export const metadata: Metadata = {
  title: 'Contents',
  description: 'The full table of contents for Reinforcement Learning for Robotics — The FCP Way.',
};

export default function ChaptersPage() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-12">
      <header className="mb-10">
        <h1 className="text-[clamp(1.75rem,4vw,2.4rem)] font-bold tracking-[-0.02em] text-ink">
          Contents
        </h1>
        <p className="mt-2 max-w-2xl text-[15.5px] leading-relaxed text-ink-secondary">
          Twenty-two chapters in five parts, from the definition of a Markov decision process to a
          quadruped trained end to end in Rust. Every chapter carries the same three layers:
          complete mathematics, an interactive visual for each hard idea, and working code.
        </p>
      </header>

      <div className="space-y-10">
        {PARTS.map((part) => (
          <section key={part.id}>
            <div className="mb-4 border-b border-hairline pb-2">
              <h2 className="text-[13px] font-semibold uppercase tracking-[0.08em] text-series-1">
                Part {part.id}
              </h2>
              <p className="mt-0.5 text-[19px] font-semibold tracking-tight text-ink">
                {part.title}
              </p>
              <p className="mt-0.5 text-[13.5px] italic text-ink-muted">{part.tagline}</p>
            </div>

            <ul className="space-y-2">
              {part.chapters.map((n) => {
                const ch = CHAPTERS.find((c) => c.n === n);
                if (!ch) return null;
                return (
                  <li key={ch.n}>
                    <Link
                      href={`/chapters/${ch.slug}`}
                      className="group flex gap-4 rounded-xl border border-hairline bg-surface px-4 py-3.5 no-underline transition-colors hover:bg-surface-sunken"
                    >
                      <span className="tabular mt-0.5 text-[15px] font-semibold text-ink-muted transition-colors group-hover:text-series-1">
                        {String(ch.n).padStart(2, '0')}
                      </span>
                      <span className="min-w-0">
                        <span className="block text-[15.5px] font-semibold leading-snug tracking-tight text-ink">
                          {ch.title}
                        </span>
                        <span className="mt-0.5 block text-[13.5px] leading-relaxed text-ink-secondary">
                          {ch.blurb}
                        </span>
                        <span className="mt-1.5 flex flex-wrap gap-1">
                          {ch.sources.map((s) => (
                            <span
                              key={s}
                              className="rounded border border-hairline px-1.5 py-0.5 text-[10.5px] text-ink-muted"
                            >
                              {s}
                            </span>
                          ))}
                        </span>
                      </span>
                    </Link>
                  </li>
                );
              })}
            </ul>
          </section>
        ))}
      </div>
    </div>
  );
}

'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';
import { Menu, Moon, Sun, X } from 'lucide-react';
import { useTheme } from './ThemeProvider';
import { ChapterList } from './ChapterList';
import { cn } from '@/lib/utils';

export function SiteHeader() {
  const { mode, toggle, ready } = useTheme();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [progress, setProgress] = useState(0);
  const pathname = usePathname();
  const onChapter = pathname?.startsWith('/chapters/') && pathname !== '/chapters';

  useEffect(() => setMobileOpen(false), [pathname]);

  useEffect(() => {
    if (!onChapter) {
      setProgress(0);
      return;
    }
    const onScroll = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      setProgress(max > 0 ? Math.min(100, (window.scrollY / max) * 100) : 0);
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, [onChapter, pathname]);

  return (
    <>
      <header className="sticky top-0 z-40 border-b border-hairline bg-surface/85 backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-[1600px] items-center gap-3 px-4">
          <button
            type="button"
            onClick={() => setMobileOpen((v) => !v)}
            className="rounded-md p-1.5 text-ink-secondary hover:bg-surface-sunken lg:hidden"
            aria-label={mobileOpen ? 'Close chapter list' : 'Open chapter list'}
            aria-expanded={mobileOpen}
          >
            {mobileOpen ? <X size={18} /> : <Menu size={18} />}
          </button>

          <Link href="/" className="flex items-center gap-2.5 no-underline">
            <span
              aria-hidden
              className="grid h-7 w-7 place-items-center rounded-md bg-series-1 text-[13px] font-bold text-white"
            >
              R
            </span>
            <span className="hidden text-[13.5px] font-semibold tracking-tight text-ink sm:block">
              RL for Robotics
              <span className="ml-1.5 font-normal text-ink-muted">· The FCP Way</span>
            </span>
          </Link>

          <nav className="ml-auto flex items-center gap-1">
            <Link
              href="/chapters"
              className={cn(
                'rounded-md px-2.5 py-1.5 text-[13px] font-medium no-underline transition-colors',
                pathname === '/chapters'
                  ? 'bg-surface-sunken text-ink'
                  : 'text-ink-secondary hover:bg-surface-sunken hover:text-ink',
              )}
            >
              Contents
            </Link>
            <Link
              href="/about"
              className={cn(
                'rounded-md px-2.5 py-1.5 text-[13px] font-medium no-underline transition-colors',
                pathname === '/about'
                  ? 'bg-surface-sunken text-ink'
                  : 'text-ink-secondary hover:bg-surface-sunken hover:text-ink',
              )}
            >
              Method
            </Link>
            <button
              type="button"
              onClick={toggle}
              className="ml-1 rounded-md p-1.5 text-ink-secondary transition-colors hover:bg-surface-sunken hover:text-ink"
              aria-label={`Switch to ${mode === 'light' ? 'dark' : 'light'} theme`}
            >
              {ready && mode === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </nav>
        </div>

        {onChapter && (
          <div
            className="h-0.5 bg-series-1 transition-[width] duration-150"
            style={{ width: `${progress}%` }}
            role="progressbar"
            aria-label="Reading progress"
            aria-valuenow={Math.round(progress)}
            aria-valuemin={0}
            aria-valuemax={100}
          />
        )}
      </header>

      {mobileOpen && (
        <div className="fixed inset-0 top-14 z-30 overflow-y-auto bg-surface px-4 py-4 lg:hidden">
          <ChapterList />
        </div>
      )}
    </>
  );
}

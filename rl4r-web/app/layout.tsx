import type { Metadata } from 'next';
import './globals.css';
import { ThemeProvider, themeScript } from '@/components/layout/ThemeProvider';
import { SiteHeader } from '@/components/layout/SiteHeader';

export const metadata: Metadata = {
  title: {
    default: 'Reinforcement Learning for Robotics — The FCP Way',
    template: '%s · RL for Robotics',
  },
  description:
    'An interactive web book teaching reinforcement learning for robotics through Foundation (full mathematical formalism), Conceptual (interactive simulations) and Practical (Rust implementations) layers.',
  keywords: [
    'reinforcement learning',
    'robotics',
    'deep RL',
    'Rust',
    'sim-to-real',
    'PPO',
    'SAC',
    'interactive book',
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className="min-h-screen antialiased">
        <ThemeProvider>
          <SiteHeader />
          <main id="main">{children}</main>
        </ThemeProvider>
      </body>
    </html>
  );
}

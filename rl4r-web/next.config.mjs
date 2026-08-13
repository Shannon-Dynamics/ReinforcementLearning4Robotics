/** @type {import('next').NextConfig} */
const nextConfig = {
  /**
   * Fully static export.
   *
   * `npm run build` emits a plain directory of HTML, JS and CSS in `out/` that
   * any static host will serve — no Node process, no serverless functions, no
   * runtime compute of any kind. Everything the book computes, it computes in
   * the reader's browser: the simulations run in a Web Worker, the MDX and the
   * syntax highlighting are resolved at build time.
   *
   * The constraint this imposes: no API routes, no middleware, no server
   * actions, no on-demand revalidation, and every dynamic route must enumerate
   * its paths through generateStaticParams (chapters do).
   */
  output: 'export',

  // Static hosts serve /chapters/slug/index.html; the trailing slash keeps
  // relative asset paths resolving correctly on hosts without rewrite rules.
  trailingSlash: true,

  // There is no image optimizer without a server, so images pass through as-is.
  images: { unoptimized: true },

  reactStrictMode: true,

  // Nivo ships ESM packages that benefit from transpilation in the server bundle.
  transpilePackages: [
    '@nivo/core',
    '@nivo/bar',
    '@nivo/line',
    '@nivo/heatmap',
    '@nivo/radar',
    '@nivo/scatterplot',
  ],

  experimental: {
    optimizePackageImports: ['lucide-react'],
  },
};

export default nextConfig;

/** @type {import('next').NextConfig} */
const nextConfig = {
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

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  devIndicators: false,
  env: {
    NEXT_PUBLIC_BACKEND_URL: process.env.NEXT_PUBLIC_BACKEND_URL,
  },
  // Add this for better production performance
  experimental: {
    optimizeCss: true,
  },
  // Optimize images
  images: {
    domains: ['your-company-domain.com'], // Add your company domain if needed
  },
};

export default nextConfig;
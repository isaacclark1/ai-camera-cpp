import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // URL re-writes
  async rewrites() {
    return [
      {
        source: "/cpp/:path*",
        destination: "http://localhost:9898/:path*",
      },
    ];
  },
};

export default nextConfig;

"use client"

import dynamic from "next/dynamic"

const GammaDashboard = dynamic(
  () => import("@/components/gamma-dashboard").then(m => m.GammaDashboard),
  { ssr: false }
)

export default function EmbedPage() {
  return (
    <div className="embed-wrapper">
      <style>{`
        /* Override dashboard styles for iframe embedding */
        .embed-wrapper .min-h-screen { min-height: unset; }
        .embed-wrapper { padding: 0; }
      `}</style>
      <GammaDashboard />
    </div>
  )
}

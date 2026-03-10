"use client"

import dynamic from "next/dynamic"

const GammaDashboard = dynamic(
  () => import("@/components/gamma-dashboard").then(m => m.GammaDashboard),
  { ssr: false }
)

export default function Home() {
  return <GammaDashboard />
}

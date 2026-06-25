"use client"

import dynamic from "next/dynamic"

const LeakageBoundDashboard = dynamic(
  () => import("@/components/leakage-bound-dashboard").then(m => m.LeakageBoundDashboard),
  { ssr: false }
)

export default function LeakageBoundPage() {
  return <LeakageBoundDashboard />
}

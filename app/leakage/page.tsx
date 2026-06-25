"use client"

import dynamic from "next/dynamic"

const LeakageDashboard = dynamic(
  () => import("@/components/leakage-dashboard").then(m => m.LeakageDashboard),
  { ssr: false }
)

export default function LeakagePage() {
  return <LeakageDashboard />
}

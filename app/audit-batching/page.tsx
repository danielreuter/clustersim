"use client"

import dynamic from "next/dynamic"

const AuditBatchingDashboard = dynamic(
  () => import("@/components/audit-batching-dashboard").then(module => module.AuditBatchingDashboard),
  { ssr: false },
)

export default function AuditBatchingPage() {
  return <AuditBatchingDashboard />
}

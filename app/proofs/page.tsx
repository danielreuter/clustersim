"use client"

import dynamic from "next/dynamic"

const ProofsDashboard = dynamic(
  () => import("@/components/proofs-dashboard").then(m => m.ProofsDashboard),
  { ssr: false }
)

export default function ProofsPage() {
  return <ProofsDashboard />
}

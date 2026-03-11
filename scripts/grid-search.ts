#!/usr/bin/env npx tsx
/**
 * Grid search across hardware regimes × constraint types.
 * Outputs structured findings for each cell.
 */

import { simulateDirect, computeHardwarePreset } from "@/lib/sim"
import type { DirectScenario, GammaResult } from "@/lib/sim"

type Params = {
  gpu: string
  count: number
  computeUtil?: number
  memoryUtil?: number
  alpha?: number
  bOut?: number
  bIn?: number
  sanitization?: boolean
  epochS?: number
  downtimeS?: number
  survivingBytes?: number
  stateBytes?: number
  flopPerUnit?: number
  ingressBytesPerUnit?: number
  egressBytesPerUnit?: number
}

function sim(p: Params): GammaResult & { _params: Params } {
  const hw = computeHardwarePreset(p.gpu, p.count)
  const cu = p.computeUtil ?? 0.5
  const mu = p.memoryUtil ?? 0.5
  const scenario: DirectScenario = {
    hardware: hw,
    honest: {
      claimedComputeFlops: cu * hw.computeFlops,
      claimedMemoryBytes: mu * hw.hbmBytes,
    },
    verifier: {
      alpha: p.alpha ?? 0.83,
      covertEgressBps: p.bOut ?? 1e6,
      covertIngressBps: p.bIn ?? 100e6,
      sanitizationEnabled: p.sanitization ?? true,
      epochSeconds: p.epochS ?? 3981,
      downtimeSeconds: p.downtimeS ?? 10,
      survivingStateBytes: p.survivingBytes ?? 100e9,
    },
    covert: {
      label: "grid",
      kind: "inference",
      unit: "token",
      stateBytes: p.stateBytes ?? 140e9,
      flopPerUnit: p.flopPerUnit ?? 580e6,
      ingressBytesPerUnit: p.ingressBytesPerUnit ?? 0,
      egressBytesPerUnit: p.egressBytesPerUnit ?? 200,
    },
  }
  return { ...simulateDirect(scenario), _params: p }
}

function fmtG(g: number | null): string {
  if (g === null || !Number.isFinite(g)) return "INF"
  if (g >= 1e6) return `${(g / 1e6).toFixed(1)}M`
  if (g >= 1e3) return `${(g / 1e3).toFixed(1)}K`
  return g.toFixed(2)
}

// ============================================================
// GRID DEFINITION
// ============================================================

const SMALL_HW: Params[] = [
  { gpu: "A100", count: 8 },
  { gpu: "H200", count: 8 },
  { gpu: "B200", count: 8 },
  { gpu: "B200", count: 72 },
]

const LARGE_HW: Params[] = [
  { gpu: "B200", count: 72 },
  { gpu: "Rubin", count: 512 },
  { gpu: "Rubin", count: 1024 },
]

const logSteps = (min: number, max: number, n: number) =>
  Array.from({ length: n }, (_, i) => 10 ** (Math.log10(min) + (Math.log10(max) - Math.log10(min)) * i / (n - 1)))

// ============================================================
// EGRESS SWEEP
// ============================================================

function sweepEgress(hwList: Params[], label: string) {
  console.log(`\n${"=".repeat(70)}`)
  console.log(`EGRESS SWEEP — ${label}`)
  console.log("=".repeat(70))

  const bOutValues = logSteps(100, 1e9, 10)

  for (const hw of hwList) {
    console.log(`\n--- ${hw.count}x ${hw.gpu} ---`)
    console.log(`${"bOut".padEnd(12)} ${"gamma".padEnd(12)} ${"dominant".padEnd(15)} finite`)
    for (const bOut of bOutValues) {
      const r = sim({ ...hw, bOut, sanitization: false })
      console.log(`${bOut.toFixed(0).padEnd(12)} ${fmtG(r.gamma).padEnd(12)} ${r.dominant.padEnd(15)} ${r.finite}`)
    }

    // Now with sanitization
    console.log(`\n  + With sanitization (epoch=5s, downtime=0.25s):`)
    for (const bOut of [1e3, 1e5, 1e7, 1e9]) {
      const r = sim({ ...hw, bOut, sanitization: true })
      console.log(`  bOut=${bOut.toFixed(0).padEnd(12)} gamma=${fmtG(r.gamma).padEnd(12)} dom=${r.dominant.padEnd(15)} finite=${r.finite}`)
    }

    // Vary egressBytesPerUnit
    console.log(`\n  + Varying egressBytesPerUnit (bOut=20000, no sanit):`)
    for (const epu of [10, 50, 200, 1000, 10000]) {
      const r = sim({ ...hw, bOut: 20000, egressBytesPerUnit: epu, sanitization: false })
      console.log(`  egress/unit=${String(epu).padEnd(8)} gamma=${fmtG(r.gamma).padEnd(12)} dom=${r.dominant}`)
    }

    // Vary alpha with fixed bOut
    console.log(`\n  + Varying alpha (bOut=20000, no sanit):`)
    for (const alpha of [0, 0.25, 0.5, 0.75, 1.0]) {
      const r = sim({ ...hw, bOut: 20000, alpha, sanitization: false })
      console.log(`  alpha=${alpha.toFixed(2).padEnd(6)} gamma=${fmtG(r.gamma).padEnd(12)} dom=${r.dominant}`)
    }
  }
}

// ============================================================
// INGRESS SWEEP
// ============================================================

function sweepIngress(hwList: Params[], label: string) {
  console.log(`\n${"=".repeat(70)}`)
  console.log(`INGRESS SWEEP — ${label}`)
  console.log("=".repeat(70))

  const bInValues = logSteps(100, 1e9, 10)

  // Ingress only matters when ingressBytesPerUnit > 0
  const ingressPerUnit = [1000, 10000, 100000]

  for (const hw of hwList) {
    for (const ipu of ingressPerUnit) {
      console.log(`\n--- ${hw.count}x ${hw.gpu}, ingressBytesPerUnit=${ipu} ---`)
      console.log(`${"bIn".padEnd(12)} ${"gamma".padEnd(12)} ${"dominant".padEnd(15)} finite`)
      for (const bIn of bInValues) {
        const r = sim({ ...hw, bIn, ingressBytesPerUnit: ipu, sanitization: false })
        console.log(`${bIn.toFixed(0).padEnd(12)} ${fmtG(r.gamma).padEnd(12)} ${r.dominant.padEnd(15)} ${r.finite}`)
      }
    }

    // Alpha interaction
    console.log(`\n  + Alpha interaction (bIn=10000, ingressPerUnit=10000, no sanit):`)
    for (const alpha of [0, 0.25, 0.5, 0.75, 1.0]) {
      const r = sim({ ...hw, bIn: 10000, ingressBytesPerUnit: 10000, alpha, sanitization: false })
      console.log(`  alpha=${alpha.toFixed(2).padEnd(6)} gamma=${fmtG(r.gamma).padEnd(12)} dom=${r.dominant}`)
    }
  }
}

// ============================================================
// SANITIZATION SWEEP
// ============================================================

function sweepSanitization(hwList: Params[], label: string) {
  console.log(`\n${"=".repeat(70)}`)
  console.log(`SANITIZATION SWEEP — ${label}`)
  console.log("=".repeat(70))

  for (const hw of hwList) {
    console.log(`\n--- ${hw.count}x ${hw.gpu} ---`)

    // Sweep epochS
    console.log(`\n  Sweep epochS (downtime=0.25s, surviving=17GB):`)
    console.log(`  ${"epochS".padEnd(10)} ${"gamma".padEnd(12)} ${"dominant".padEnd(15)} finite`)
    for (const epochS of [0.5, 1, 2, 5, 10, 20, 50, 100]) {
      const r = sim({ ...hw, sanitization: true, epochS, downtimeS: 0.25, survivingBytes: 17e9 })
      console.log(`  ${String(epochS).padEnd(10)} ${fmtG(r.gamma).padEnd(12)} ${r.dominant.padEnd(15)} ${r.finite}`)
    }

    // Sweep downtimeS
    console.log(`\n  Sweep downtimeS (epoch=5s, surviving=17GB):`)
    for (const downtimeS of [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5]) {
      const r = sim({ ...hw, sanitization: true, epochS: 5, downtimeS, survivingBytes: 17e9 })
      console.log(`  dt=${String(downtimeS).padEnd(6)} gamma=${fmtG(r.gamma).padEnd(12)} dom=${r.dominant.padEnd(15)} finite=${r.finite}`)
    }

    // Sweep survivingBytes
    console.log(`\n  Sweep survivingBytes (epoch=5s, downtime=0.25s):`)
    for (const sb of [0, 1e9, 5e9, 17e9, 50e9, 100e9, 140e9]) {
      const r = sim({ ...hw, sanitization: true, epochS: 5, downtimeS: 0.25, survivingBytes: sb })
      console.log(`  surviving=${(sb/1e9).toFixed(0).padEnd(6)}GB gamma=${fmtG(r.gamma).padEnd(12)} dom=${r.dominant.padEnd(15)} finite=${r.finite}`)
    }

    // Small vs large covert workloads under aggressive sanitization
    console.log(`\n  Model size under aggressive sanit (epoch=2s, downtime=0.5s, surviving=1GB):`)
    const models = [
      { name: "8B", stateBytes: 16e9, flopPerUnit: 16e6 },
      { name: "70B", stateBytes: 140e9, flopPerUnit: 580e6 },
      { name: "405B", stateBytes: 810e9, flopPerUnit: 8e9 },
    ]
    for (const m of models) {
      const r = sim({ ...hw, sanitization: true, epochS: 2, downtimeS: 0.5, survivingBytes: 1e9, ...m })
      console.log(`  ${m.name.padEnd(6)} gamma=${fmtG(r.gamma).padEnd(12)} dom=${r.dominant.padEnd(15)} finite=${r.finite}`)
    }

    // Alpha + sanitization combined
    console.log(`\n  Alpha + sanitization combined (epoch=5s, downtime=0.25s, surviving=17GB):`)
    for (const alpha of [0, 0.5, 0.8, 1.0]) {
      const r = sim({ ...hw, sanitization: true, alpha, epochS: 5, downtimeS: 0.25, survivingBytes: 17e9 })
      console.log(`  alpha=${alpha.toFixed(1).padEnd(4)} gamma=${fmtG(r.gamma).padEnd(12)} dom=${r.dominant.padEnd(15)} finite=${r.finite}`)
    }
  }
}

// ============================================================
// RUN ALL
// ============================================================

sweepEgress(SMALL_HW, "Small hardware")
sweepEgress(LARGE_HW, "Large hardware")
sweepIngress(SMALL_HW, "Small hardware")
sweepIngress(LARGE_HW, "Large hardware")
sweepSanitization(SMALL_HW, "Small hardware")
sweepSanitization(LARGE_HW, "Large hardware")

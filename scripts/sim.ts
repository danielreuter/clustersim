#!/usr/bin/env npx tsx
/**
 * CLI wrapper around simulateDirect.
 *
 * Usage:
 *   npx tsx scripts/sim.ts                        # run with defaults
 *   npx tsx scripts/sim.ts '{"gpu":"H100","count":8,"alpha":1,...}'
 *
 * Input JSON fields (all optional, sensible defaults):
 *   gpu          — GPU key: H100, H200, A100, H20, B200, Rubin  (default: H100)
 *   count        — number of GPUs                                 (default: 8)
 *   computeUtil  — honest compute utilization 0-1                 (default: 0.96)
 *   memoryUtil   — honest memory utilization 0-1                  (default: 0.83)
 *   alpha        — proved compute fraction 0-1                    (default: 0.83)
 *   bOut         — covert egress bandwidth bytes/s                (default: 1e6)
 *   bIn          — covert ingress bandwidth bytes/s               (default: 100e6)
 *   sanitization — enable memory sanitization                     (default: true)
 *   epochS       — sanitization epoch length seconds              (default: 3981)
 *   downtimeS    — sanitization downtime seconds                  (default: 10)
 *   survivingBytes — covert bytes surviving sanitization          (default: 100e9)
 *   stateBytes   — covert state size bytes                        (default: 140e9)
 *   flopPerUnit  — FLOP per unit of covert output                 (default: 580e6)
 *   ingressBytesPerUnit — covert ingress per input                (default: 0)
 *   egressBytesPerUnit  — covert egress per output                (default: 200)
 *
 * Output: JSON with input params + full GammaResult.
 */

import { simulateDirect, computeHardwarePreset } from "@/lib/sim"
import type { DirectScenario, GammaResult } from "@/lib/sim"

type Input = {
  gpu?: string
  count?: number
  computeUtil?: number
  memoryUtil?: number
  alpha?: number
  alphaMemory?: number
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

function run(input: Input = {}) {
  const gpu = input.gpu ?? "H100"
  const count = input.count ?? 8
  const hw = computeHardwarePreset(gpu, count)

  const computeUtil = input.computeUtil ?? 0.96
  const memoryUtil = input.memoryUtil ?? 0.83
  const alpha = input.alpha ?? 0.83

  const scenario: DirectScenario = {
    hardware: hw,
    honest: {
      claimedComputeFlops: computeUtil * hw.computeFlops,
      claimedMemoryBytes: memoryUtil * hw.hbmBytes,
    },
    verifier: {
      alpha,
      alphaMemory: input.alphaMemory ?? 1.0,
      covertEgressBps: input.bOut ?? 1e6,
      covertIngressBps: input.bIn ?? 100e6,
      sanitizationEnabled: input.sanitization ?? true,
      epochSeconds: input.epochS ?? 3981,
      downtimeSeconds: input.downtimeS ?? 10,
      survivingStateBytes: input.survivingBytes ?? 100e9,
    },
    covert: {
      label: "cli",
      kind: "inference",
      unit: "token",
      stateBytes: input.stateBytes ?? 140e9,
      flopPerUnit: input.flopPerUnit ?? 580e6,
      ingressBytesPerUnit: input.ingressBytesPerUnit ?? 0,
      egressBytesPerUnit: input.egressBytesPerUnit ?? 200,
    },
  }

  const result: GammaResult = simulateDirect(scenario)

  console.log(JSON.stringify({
    input: { gpu, count, computeUtil, memoryUtil, ...scenario.verifier, ...scenario.covert },
    gamma: result.gamma,
    finite: result.finite,
    dominant: result.dominant,
    theta0: result.theta0,
    thetaVerified: result.thetaVerified,
    gammaCompute: result.gammaCompute,
    gammaIngress: result.gammaIngress,
    gammaEgress: result.gammaEgress,
    gammaDuty: result.gammaDuty,
    fitMarginBytes: result.fitMarginBytes,
    reason: result.reason,
  }, null, 2))
}

// Parse CLI arg as JSON, or use defaults
const arg = process.argv[2]
const input: Input = arg ? JSON.parse(arg) : {}
run(input)

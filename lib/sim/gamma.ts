import type {
  Scenario,
  ThroughputEstimate,
  GammaResult,
  CovertWorkloadV1,
  CovertWorkloadInference,
  CovertWorkloadTraining,
} from "./types"
import { isV2Workload } from "./types"
import {
  rooflineLite,
  rooflineLiteTraining,
  deriveSyncIO,
  resolveModel,
  resolveGpu,
} from "./roofline"

// ---------------------------------------------------------------------------
// Closed-form throughput backends
// ---------------------------------------------------------------------------

/** Dedicated throughput: θ₀ = Ĝ / g (no verification, compute-limited) */
export function dedicatedThroughput(s: Scenario): ThroughputEstimate {
  const covert = s.covert as CovertWorkloadV1
  return {
    unitsPerSecond: s.hardware.computeFlops / covert.flopPerUnit,
    regime: "compute",
  }
}

/** Verified compute throughput: (Ĝ - αf*) / g */
export function verifiedComputeThroughput(s: Scenario): ThroughputEstimate {
  const covert = s.covert as CovertWorkloadV1
  const avail = Math.max(
    0,
    s.hardware.computeFlops - s.verifier.alpha * s.honest.claimedComputeFlops,
  )
  return {
    unitsPerSecond: avail / covert.flopPerUnit,
    regime: "compute",
  }
}

// ---------------------------------------------------------------------------
// Core Γ combiner
// ---------------------------------------------------------------------------

export function composeGamma(
  scenario: Scenario,
  dedicated: ThroughputEstimate,
  verifiedCompute: ThroughputEstimate,
): GammaResult {
  const { hardware, honest, verifier } = scenario
  const covert = scenario.covert as CovertWorkloadV1
  const theta0 = dedicated.unitsPerSecond

  // --- Memory fit check ---
  const fitMarginBytes =
    hardware.hbmBytes - honest.claimedMemoryBytes - covert.stateBytes

  if (fitMarginBytes < 0) {
    return {
      gamma: Infinity,
      dominant: "memory-fit",
      theta0,
      thetaVerified: 0,
      gammaDuty: 1,
      gammaCompute: Infinity,
      gammaIngress: 1,
      gammaEgress: 1,
      tReload: 0,
      tCovert: 0,
      fitMarginBytes,
      finite: false,
      reason: "Covert state does not fit alongside honest memory",
    }
  }

  // --- Duty cycle (sanitization) ---
  let gammaDuty = 1
  let tReload = 0
  let tCovert = verifier.epochSeconds

  if (verifier.sanitizationEnabled) {
    const reloadable = covert.persistBytes ?? covert.stateBytes
    tReload =
      Math.max(0, reloadable - verifier.survivingStateBytes) /
      verifier.covertIngressBps

    tCovert = Math.max(
      0,
      verifier.epochSeconds - verifier.downtimeSeconds - tReload,
    )

    gammaDuty = tCovert > 0 ? verifier.epochSeconds / tCovert : Infinity
  }

  // --- Operational throughput caps ---
  const thetaIngressCap =
    covert.ingressBytesPerUnit === 0
      ? Infinity
      : verifier.covertIngressBps / covert.ingressBytesPerUnit

  const thetaEgressCap =
    covert.egressBytesPerUnit === 0
      ? Infinity
      : verifier.covertEgressBps / covert.egressBytesPerUnit

  const thetaOp = Math.min(
    verifiedCompute.unitsPerSecond,
    thetaIngressCap,
    thetaEgressCap,
  )

  const duty = tCovert / verifier.epochSeconds
  const thetaVerified = duty * thetaOp

  // --- Individual Γ factors ---
  const gammaCompute =
    verifiedCompute.unitsPerSecond > 0
      ? theta0 / verifiedCompute.unitsPerSecond
      : Infinity

  const gammaIngress =
    thetaIngressCap < Infinity
      ? Math.max(1, theta0 / thetaIngressCap)
      : 1

  const gammaEgress =
    thetaEgressCap < Infinity
      ? Math.max(1, theta0 / thetaEgressCap)
      : 1

  const gamma = thetaVerified > 0 ? theta0 / thetaVerified : Infinity

  // --- Dominant factor ---
  const factors = [
    ["duty", gammaDuty],
    ["compute", gammaCompute],
    ["ingress", gammaIngress],
    ["egress", gammaEgress],
  ] as const

  const dominant: GammaResult["dominant"] = !Number.isFinite(gamma)
    ? tCovert === 0
      ? "duty"
      : fitMarginBytes < 0
        ? "memory-fit"
        : "compute"
    : (factors.slice().sort((a, b) => (b[1] as number) - (a[1] as number))[0][0] as GammaResult["dominant"])

  return {
    gamma,
    dominant,
    theta0,
    thetaVerified,
    gammaDuty,
    gammaCompute,
    gammaIngress,
    gammaEgress,
    tReload,
    tCovert,
    fitMarginBytes,
    finite: Number.isFinite(gamma),
    reason: !Number.isFinite(gamma)
      ? tCovert === 0
        ? `Reload time (${(tReload / 86400).toFixed(1)} days) exceeds epoch`
        : `No available compute after verification`
      : undefined,
  }
}

// ---------------------------------------------------------------------------
// Convenience: run the full pipeline with closed-form backend
// ---------------------------------------------------------------------------

export function simulate(scenario: Scenario): GammaResult {
  return composeGamma(
    scenario,
    dedicatedThroughput(scenario),
    verifiedComputeThroughput(scenario),
  )
}

// ---------------------------------------------------------------------------
// Sweep: vary one parameter and return results
// ---------------------------------------------------------------------------

export type SweepPoint = { value: number; result: GammaResult }

export function sweep(
  base: Scenario,
  mutate: (scenario: Scenario, value: number) => Scenario,
  values: number[],
): SweepPoint[] {
  return values.map((value) => ({
    value,
    result: simulate(mutate(base, value)),
  }))
}

/** Generate log-spaced values from 10^start to 10^stop */
export function logRange(start: number, stop: number, steps: number): number[] {
  const result: number[] = []
  const step = (stop - start) / (steps - 1)
  for (let i = 0; i < steps; i++) {
    result.push(10 ** (start + i * step))
  }
  return result
}

/** Generate linearly-spaced values */
export function linRange(start: number, stop: number, steps: number): number[] {
  const result: number[] = []
  const step = (stop - start) / (steps - 1)
  for (let i = 0; i < steps; i++) {
    result.push(start + i * step)
  }
  return result
}

// ---------------------------------------------------------------------------
// V2: batch-aware inference + sync-aware training
// ---------------------------------------------------------------------------

function simulateV2Inference(scenario: Scenario, wl: CovertWorkloadInference): GammaResult {
  const model = resolveModel(wl.modelKey)
  const gpu = resolveGpu(wl.gpuKey)
  const { hardware, honest, verifier } = scenario

  // Full-budget roofline (dedicated throughput)
  const fullBudget = rooflineLite(
    model, gpu, wl.nGpu, wl.contextLength,
    hardware.computeFlops, hardware.hbmBytes,
  )

  // Verified-budget roofline (after α*f* consumed)
  const availFlops = Math.max(0, hardware.computeFlops - verifier.alpha * honest.claimedComputeFlops)
  const availHbm = Math.max(0, hardware.hbmBytes - honest.claimedMemoryBytes)
  const verifiedBudget = rooflineLite(
    model, gpu, wl.nGpu, wl.contextLength,
    availFlops, availHbm,
  )

  // Build a v1-equivalent scenario for composeGamma
  const stateBytes = fullBudget.nPersist + fullBudget.workspace
  const g = fullBudget.throughput.unitsPerSecond > 0
    ? hardware.computeFlops / fullBudget.throughput.unitsPerSecond
    : Infinity
  const v1Covert: CovertWorkloadV1 = {
    label: wl.label,
    kind: "inference",
    unit: "token",
    stateBytes,
    persistBytes: fullBudget.nPersist,
    flopPerUnit: g,
    ingressBytesPerUnit: 4, // token embedding
    egressBytesPerUnit: 4,
  }

  const v1Scenario: Scenario = { ...scenario, covert: v1Covert }
  const dedicated: ThroughputEstimate = fullBudget.throughput
  const verified: ThroughputEstimate = verifiedBudget.throughput

  const result = composeGamma(v1Scenario, dedicated, verified)

  return {
    ...result,
    v2: {
      regime: verifiedBudget.throughput.regime ?? "compute",
      optimalBatchSize: verifiedBudget.optBatch,
      nPersistBytes: fullBudget.nPersist,
      workspaceBytes: fullBudget.workspace,
    },
  }
}

function simulateV2Training(scenario: Scenario, wl: CovertWorkloadTraining): GammaResult {
  const model = resolveModel(wl.modelKey)
  const gpu = resolveGpu(wl.gpuKey)
  const { hardware, honest, verifier } = scenario

  // Full-budget roofline
  const fullBudget = rooflineLiteTraining(
    model, gpu, wl.nGpu,
    hardware.computeFlops, hardware.hbmBytes,
  )

  // Verified-budget roofline
  const availFlops = Math.max(0, hardware.computeFlops - verifier.alpha * honest.claimedComputeFlops)
  const availHbm = Math.max(0, hardware.hbmBytes - honest.claimedMemoryBytes)
  const verifiedBudget = rooflineLiteTraining(
    model, gpu, wl.nGpu,
    availFlops, availHbm,
  )

  // Derive sync I/O
  const { dIn, dOut } = deriveSyncIO(wl.syncPolicy)

  const stateBytes = fullBudget.nPersist + fullBudget.workspace
  const g = fullBudget.throughput.unitsPerSecond > 0
    ? hardware.computeFlops / fullBudget.throughput.unitsPerSecond
    : Infinity

  const v1Covert: CovertWorkloadV1 = {
    label: wl.label,
    kind: "training",
    unit: "train-token",
    stateBytes,
    persistBytes: fullBudget.nPersist,
    flopPerUnit: g,
    ingressBytesPerUnit: dIn,
    egressBytesPerUnit: dOut,
  }

  const v1Scenario: Scenario = { ...scenario, covert: v1Covert }
  const dedicated: ThroughputEstimate = fullBudget.throughput
  const verified: ThroughputEstimate = verifiedBudget.throughput

  const result = composeGamma(v1Scenario, dedicated, verified)

  return {
    ...result,
    v2: {
      regime: verifiedBudget.throughput.regime ?? "compute",
      optimalBatchSize: verifiedBudget.optBatch,
      nPersistBytes: fullBudget.nPersist,
      workspaceBytes: fullBudget.workspace,
    },
  }
}

/**
 * V2 simulation: dispatches on workload type.
 * - v1 (no backend field): falls through to existing simulate()
 * - v2 inference: roofline-lite batch search
 * - v2 training: roofline-lite + sync-aware d_in/d_out
 */
export function simulateV2(scenario: Scenario): GammaResult {
  const wl = scenario.covert
  if (!isV2Workload(wl)) {
    return simulate(scenario)
  }
  if (wl.kind === "inference") {
    return simulateV2Inference(scenario, wl as CovertWorkloadInference)
  }
  return simulateV2Training(scenario, wl as CovertWorkloadTraining)
}

/**
 * Run both v1 and v2 paths for comparison.
 * For v2 workloads, constructs an equivalent v1 workload from resolved parameters.
 */
export function simulateComparison(scenario: Scenario): { v1: GammaResult; v2: GammaResult } {
  const v2Result = simulateV2(scenario)

  // For v1 comparison, if it's a v2 workload, use the resolved parameters
  const wl = scenario.covert
  if (!isV2Workload(wl)) {
    return { v1: v2Result, v2: v2Result }
  }

  // Construct v1 equivalent from v2 resolved values
  const model = resolveModel(wl.modelKey)
  const nPersist = model.totalParams * model.weightPrecisionBytes
  const g = 2 * model.totalActiveParams // simple 2N FLOP/token approximation
  let dIn = 0
  let dOut = 4 // default egress for inference

  if (wl.kind === "training") {
    const twl = wl as CovertWorkloadTraining
    const sync = deriveSyncIO(twl.syncPolicy)
    dIn = sync.dIn
    dOut = sync.dOut
  }

  const v1Covert: CovertWorkloadV1 = {
    label: wl.label,
    kind: wl.kind,
    unit: wl.unit,
    stateBytes: nPersist,
    flopPerUnit: g,
    ingressBytesPerUnit: dIn,
    egressBytesPerUnit: dOut,
  }

  const v1Scenario: Scenario = { ...scenario, covert: v1Covert }
  const v1Result = simulate(v1Scenario)
  return { v1: { ...v1Result }, v2: { ...v2Result, gammaV1: v1Result.gamma } }
}

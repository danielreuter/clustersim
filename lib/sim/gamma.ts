import type {
  Scenario,
  ResolvedHardware,
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
  resolveHardware,
} from "./roofline"

// ---------------------------------------------------------------------------
// Closed-form throughput backends (used by composeGamma internally)
// ---------------------------------------------------------------------------

/** Dedicated throughput: θ₀ = Ĝ / g (no verification, compute-limited) */
export function dedicatedThroughput(computeFlops: number, covert: CovertWorkloadV1): ThroughputEstimate {
  return {
    unitsPerSecond: computeFlops / covert.flopPerUnit,
    regime: "compute",
  }
}

/** Verified compute throughput: (Ĝ - αf*) / g */
export function verifiedComputeThroughput(
  computeFlops: number, alpha: number, claimedComputeFlops: number, covert: CovertWorkloadV1,
): ThroughputEstimate {
  const avail = Math.max(0, computeFlops - alpha * claimedComputeFlops)
  return {
    unitsPerSecond: avail / covert.flopPerUnit,
    regime: "compute",
  }
}

// ---------------------------------------------------------------------------
// Core Γ combiner
// ---------------------------------------------------------------------------

export function composeGamma(
  hbmBytes: number,
  honest: { claimedMemoryBytes: number },
  verifier: Scenario["verifier"],
  covert: CovertWorkloadV1,
  dedicated: ThroughputEstimate,
  verifiedCompute: ThroughputEstimate,
): GammaResult {
  const theta0 = dedicated.unitsPerSecond

  // --- Memory fit check ---
  const fitMarginBytes = hbmBytes - honest.claimedMemoryBytes - covert.stateBytes

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
// Sweep + range utilities
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
// Model-based simulation (roofline backends)
// ---------------------------------------------------------------------------

function simulateWithRoofline(
  scenario: Scenario,
  rh: ResolvedHardware,
  wl: CovertWorkloadInference | CovertWorkloadTraining,
): GammaResult {
  const model = resolveModel(wl.modelKey)
  const { honest, verifier } = scenario

  const isInference = wl.kind === "inference"
  const ctx = isInference ? (wl as CovertWorkloadInference).contextLength : 1

  // Roofline function
  const roofline = isInference ? rooflineLite : rooflineLiteTraining

  // Total rack HBM bandwidth
  const totalHbmBw = rh.nGpu * rh.gpu.hbmBandwidthBps

  // Full-budget roofline (dedicated throughput)
  const fullBudget = isInference
    ? rooflineLite(model, rh.gpu, rh.nGpu, ctx, rh.computeFlops, rh.hbmBytes, totalHbmBw)
    : rooflineLiteTraining(model, rh.gpu, rh.nGpu, rh.computeFlops, rh.hbmBytes, totalHbmBw)

  // Verified-budget roofline (after α*f* consumed)
  const availFlops = Math.max(0, rh.computeFlops - verifier.alpha * honest.claimedComputeFlops)
  const availHbm = Math.max(0, rh.hbmBytes - honest.claimedMemoryBytes)
  const availHbmBw = Math.max(0, totalHbmBw * (1 - honest.claimedMemoryBytes / rh.hbmBytes))
  const verifiedBudget = isInference
    ? rooflineLite(model, rh.gpu, rh.nGpu, ctx, availFlops, availHbm, availHbmBw)
    : rooflineLiteTraining(model, rh.gpu, rh.nGpu, availFlops, availHbm, availHbmBw)

  // Derive I/O requirements
  let dIn = 0
  let dOut = 4 // token embedding for inference
  if (!isInference) {
    const sync = deriveSyncIO((wl as CovertWorkloadTraining).syncPolicy)
    dIn = sync.dIn
    dOut = sync.dOut
  }

  // Build v1-equivalent for composeGamma
  const stateBytes = fullBudget.nPersist + fullBudget.workspace
  const g = fullBudget.throughput.unitsPerSecond > 0
    ? rh.computeFlops / fullBudget.throughput.unitsPerSecond
    : Infinity

  const v1Covert: CovertWorkloadV1 = {
    label: wl.label,
    kind: wl.kind,
    unit: wl.unit,
    stateBytes,
    persistBytes: fullBudget.nPersist,
    flopPerUnit: g,
    ingressBytesPerUnit: dIn,
    egressBytesPerUnit: dOut,
  }

  const result = composeGamma(
    rh.hbmBytes, honest, verifier, v1Covert,
    fullBudget.throughput, verifiedBudget.throughput,
  )

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
 * Main simulation entry point. Resolves hardware from GPU specs,
 * then dispatches to roofline backend.
 */
export function simulate(scenario: Scenario): GammaResult {
  const wl = scenario.covert
  if (!isV2Workload(wl)) {
    throw new Error("V1 workloads are no longer supported. Use model-based workloads.")
  }

  const model = resolveModel(wl.modelKey)
  const rh = resolveHardware(scenario.hardware, model.weightPrecisionBytes)
  return simulateWithRoofline(scenario, rh, wl)
}

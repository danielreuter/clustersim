import type {
  Scenario,
  ThroughputEstimate,
  GammaResult,
} from "./types"

// ---------------------------------------------------------------------------
// Closed-form throughput backends
// ---------------------------------------------------------------------------

/** Dedicated throughput: θ₀ = Ĝ / g (no verification, compute-limited) */
export function dedicatedThroughput(s: Scenario): ThroughputEstimate {
  return {
    unitsPerSecond: s.hardware.computeFlops / s.covert.flopPerUnit,
    regime: "compute",
  }
}

/** Verified compute throughput: (Ĝ - αf*) / g */
export function verifiedComputeThroughput(s: Scenario): ThroughputEstimate {
  const avail = Math.max(
    0,
    s.hardware.computeFlops - s.verifier.alpha * s.honest.claimedComputeFlops,
  )
  return {
    unitsPerSecond: avail / s.covert.flopPerUnit,
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
  const { hardware, honest, verifier, covert } = scenario
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
    tReload =
      Math.max(0, covert.stateBytes - verifier.survivingStateBytes) /
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
